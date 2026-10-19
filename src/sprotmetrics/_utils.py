"""Helpers shared by every metric family.

Each of the seven Viash components these metrics come from carried its own copy of
``_dense``, ``_get_matrix``, ``_qc_mask`` and friends, and the copies had quietly
drifted apart: three different layer-fallback rules and two Otsu implementations
that disagreed on which scale they returned. The versions here are parameterised
rather than flattened, so every caller keeps the behaviour it had.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import entropy

from sprotmetrics._typing import Mask, MatrixLike, Sequence, VectorLike

#: ``.obs`` column the QC components write. Annotation, not filtering: cells that
#: fail are flagged and kept, which is what makes the ``_qc_pass`` metric variants
#: the only place the choice of QC method becomes visible.
DEFAULT_QC_COLUMN = "passed_QC"

__all__ = [
    "DEFAULT_QC_COLUMN",
    "aggregate",
    "coefficient_of_variation",
    "evenness",
    "gini",
    "marker_names",
    "otsu_threshold",
    "pearson_rows",
    "positive_mask",
    "r2_against",
    "rank_rows",
    "resolve_subset",
    "resolve_vector",
    "safe_float",
    "subsample_rows",
    "to_dense",
    "to_matrix",
]


def _is_anndata(data: object) -> bool:
    # Duck-typed so that importing anndata is not required to pass an ndarray.
    return hasattr(data, "obs") and hasattr(data, "var_names") and hasattr(data, "layers")


def to_dense(x: object) -> np.ndarray:
    """Dense ndarray from a sparse matrix, a DataFrame or an array, preserving dtype."""
    if sparse.issparse(x):
        return x.toarray()
    if isinstance(x, pd.DataFrame):
        return x.to_numpy()
    return np.asarray(x)


def to_matrix(
    data: MatrixLike,
    layer: str | None = None,
    *,
    fallback_layer: str | None = None,
    dtype: np.dtype | type | None = None,
    nan_to_num: bool = False,
) -> np.ndarray:
    """Dense cell-by-marker matrix.

    Parameters
    ----------
    data
        An AnnData, or anything array-like, in which case `layer` is ignored.
    layer
        Layer to read. Falls back to `fallback_layer` if that layer exists, then
        to ``.X``. A falsy `layer` skips straight to the fallback chain.
    fallback_layer
        Layer to try before ``.X``. ``None`` disables the intermediate step.
    dtype
        Cast the result. ``None`` keeps the stored dtype, which matters when a
        caller reproduces float32 arithmetic.
    nan_to_num
        Replace NaN and infinities with zero.
    """
    if _is_anndata(data):
        if layer and layer in data.layers:
            out = to_dense(data.layers[layer])
        elif fallback_layer and fallback_layer in data.layers:
            out = to_dense(data.layers[fallback_layer])
        else:
            out = to_dense(data.X)
    else:
        out = to_dense(data)
    if dtype is not None:
        out = out.astype(dtype, copy=False)
    if nan_to_num:
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def marker_names(data: MatrixLike, markers: Sequence[str] | None = None, n_markers: int | None = None) -> list[str]:
    """Marker names from `markers`, else ``var_names``, else ``marker_0 ...``."""
    if markers is not None:
        return list(markers)
    if _is_anndata(data):
        return [str(v) for v in data.var_names]
    if n_markers is None:
        n_markers = np.asarray(data).shape[1]
    return [f"marker_{i}" for i in range(n_markers)]


def resolve_vector(
    data: MatrixLike,
    value: VectorLike | None,
    *,
    name: str,
    dtype: type = float,
    required: bool = False,
) -> np.ndarray | None:
    """Resolve a per-cell vector, taking a ``str`` to mean a column of ``.obs``.

    Returns ``None`` when the value is absent and not `required`, so that a metric
    can report NaN rather than guess.
    """
    if value is None:
        if required:
            raise ValueError(f"{name} is required but was not given")
        return None
    if isinstance(value, str):
        if not _is_anndata(data):
            raise TypeError(
                f"{name}={value!r} names an .obs column, but the data is not an AnnData; pass an array instead"
            )
        if value not in data.obs.columns:
            if required:
                raise KeyError(f"{name}={value!r} is not in .obs (have: {list(data.obs.columns)})")
            return None
        out = np.asarray(data.obs[value].to_numpy(), dtype=dtype)
    else:
        out = np.asarray(value, dtype=dtype)
    n_obs = data.shape[0] if hasattr(data, "shape") else len(out)
    if out.shape[0] != n_obs:
        raise ValueError(f"{name} has length {out.shape[0]}, expected {n_obs}")
    return out


def resolve_subset(data: MatrixLike, subset: Mask, *, name: str = "subset") -> np.ndarray | None:
    """Boolean row mask from an ``.obs`` column name, a boolean array or an index array.

    ``None`` means every cell, and is returned unchanged so callers can tell
    "no subset asked for" from "a subset that selected nothing".
    """
    if subset is None:
        return None
    n_obs = data.shape[0]
    if isinstance(subset, str):
        if not _is_anndata(data):
            raise TypeError(f"{name}={subset!r} names an .obs column, but the data is not an AnnData")
        if subset not in data.obs.columns:
            return None
        return np.asarray(data.obs[subset].to_numpy()).astype(bool)
    arr = np.asarray(subset)
    if arr.dtype == bool:
        if arr.shape[0] != n_obs:
            raise ValueError(f"{name} has length {arr.shape[0]}, expected {n_obs}")
        return arr
    mask = np.zeros(n_obs, dtype=bool)
    mask[arr.astype(int)] = True
    return mask


def safe_float(x: object) -> float:
    """Float, or NaN where the value cannot be one."""
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def otsu_threshold(values: np.ndarray, *, log1p: bool = True) -> float:
    """Otsu threshold, returned on the same scale as `values`.

    The two implementations this replaces both thresholded ``log1p(x)`` but
    disagreed on what they handed back: one returned the log-scale cut, the other
    ``expm1`` of it. Monotone, so the partitions matched, but only one of them
    could be compared against a raw intensity. This returns the original scale.
    """
    from skimage.filters import threshold_otsu

    finite = np.asarray(values, dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or np.all(finite == finite[0]):
        return 0.0
    work = np.log1p(np.clip(finite, 0, None)) if log1p else finite
    try:
        cut = float(threshold_otsu(work))
    except ValueError:
        # Raised when the histogram cannot be split; a blanket `except Exception`
        # here would also swallow MemoryError on a full-resolution mosaic.
        return 0.0
    return float(np.expm1(cut)) if log1p else cut


def positive_mask(values: np.ndarray, *, threshold: float | None = None, log1p: bool = True) -> np.ndarray:
    """Boolean positivity, Otsu-derived unless `threshold` is given.

    The comparison happens in log space when `log1p` is set, rather than against
    :func:`otsu_threshold`'s original-scale value. The two agree in exact
    arithmetic, but a ``log1p``/``expm1`` round trip can move a value sitting
    exactly on the cut, and marker positivity feeds discrete counts where a
    single flipped cell is visible in the result. Callers that threshold raw
    pixels, and report the cut they used, take the original-scale value instead.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0 or np.allclose(finite, finite[0]):
        return np.zeros(arr.shape, dtype=bool)
    if threshold is not None:
        return arr > threshold
    if not log1p:
        return arr > otsu_threshold(arr, log1p=False)

    from skimage.filters import threshold_otsu

    logged = np.log1p(np.clip(arr, 0, None))
    try:
        cut = float(threshold_otsu(logged[np.isfinite(logged)]))
    except ValueError:
        return np.zeros(arr.shape, dtype=bool)
    return logged > cut


def coefficient_of_variation(x: np.ndarray, *, axis: int | None = None) -> float | np.ndarray:
    """Standard deviation over mean, guarded against a zero mean."""
    arr = np.asarray(x, dtype=float)
    mean = arr.mean(axis=axis)
    return arr.std(axis=axis) / np.maximum(mean, 1e-12)


def gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector; NaN when it is empty or all zero."""
    arr = np.sort(np.asarray(x, dtype=float))
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return float("nan")
    arr = np.clip(arr, 0, None)
    total = arr.sum()
    if total <= 0:
        return float("nan")
    return float((n + 1 - 2 * np.sum(np.cumsum(arr)) / total) / n)


def evenness(x: np.ndarray) -> float:
    """Shannon entropy of the shares of a non-negative vector, normalised to [0, 1]."""
    arr = np.asarray(x, dtype=float)
    positive = arr[arr > 0]
    if positive.size < 2:
        return float("nan")
    shares = positive / positive.sum()
    return float(entropy(shares) / np.log(positive.size))


def r2_against(y: np.ndarray, design: np.ndarray, *, nonnegative: bool = False, default: float = float("nan")) -> float:
    """R^2 of `y` regressed on `design` by least squares.

    `default` is returned when `y` is constant, so the caller chooses between
    "undefined" (NaN) and "explains nothing" (0.0); `nonnegative` clamps the
    result at zero, which a covariate-dependence score wants and a general
    goodness-of-fit does not.
    """
    y = np.asarray(y, dtype=float)
    if np.std(y) == 0:
        return default
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return default
    try:
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return default
    ss_res = float(np.sum((y - design @ coef) ** 2))
    value = 1.0 - ss_res / ss_tot
    return float(max(0.0, value) if nonnegative else value)


def rank_rows(x: np.ndarray) -> np.ndarray:
    """Within-row ranks by a stable double argsort (ties broken by position)."""
    return np.argsort(np.argsort(x, axis=1, kind="stable"), axis=1, kind="stable")


def pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation between corresponding rows of two equal-shaped arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (a * b).sum(axis=1) / denom, np.nan)


def subsample_rows(n_obs: int, max_cells: int, random_state: int) -> np.ndarray | None:
    """Row index for a deterministic subsample, or ``None`` when none is needed."""
    if max_cells <= 0 or n_obs <= max_cells:
        return None
    return np.random.RandomState(random_state).choice(n_obs, size=max_cells, replace=False)


def aggregate(values: np.ndarray | Sequence[float], how: str | None) -> float:
    """NaN-safe panel aggregation; the one place the registry's `aggregation` happens."""
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if how is None:
        return safe_float(arr[0]) if arr.size == 1 else float("nan")
    if arr.size == 0:
        return float("nan")
    return float({"median": np.median, "mean": np.mean, "min": np.min, "max": np.max}[how](arr))
