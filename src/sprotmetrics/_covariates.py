"""Residual dependence on cell area, and whether the transform stabilised variance.

Cell area is the imaging analogue of library size. Scoring both the counts and
the normalized layer is deliberate and load-bearing: per-cell total normalization
divides out exactly the area scaling that `sum` aggregation creates, so judging
the normalized layer alone makes `mean` and `sum` aggregation indistinguishable.
The counts variant is the only metric in the benchmark that can see the
aggregation axis at all.
"""

from __future__ import annotations

import numpy as np

from sprotmetrics._registry import Direction, Inputs, Reference, Task, metric
from sprotmetrics._typing import Mask, MatrixLike, VectorLike
from sprotmetrics._utils import (
    DEFAULT_QC_COLUMN,
    evenness,
    r2_against,
    resolve_subset,
    resolve_vector,
    to_matrix,
)

__all__ = [
    "covariate_dependence",
    "covariate_pc_correlation",
    "marker_variance_evenness",
    "mean_variance_correlation",
    "variance_explained_by_covariate",
]

FAMILY = "technical_covariates"


def _design(c: np.ndarray) -> np.ndarray:
    """Intercept, area, and log area.

    Area enters both linearly and logged: a sum-aggregated marker is close to
    linear in it, while other artefacts scale sublinearly.
    """
    return np.column_stack([np.ones_like(c), c, np.log1p(np.clip(c, 0, None))])


def _prepare(data, layer, covariate, subset):
    X = to_matrix(data, layer, fallback_layer="counts", dtype=np.float64)
    cov = resolve_vector(data, covariate, name="covariate")
    mask = resolve_subset(data, subset)
    if mask is not None:
        X = X[mask]
        cov = None if cov is None else cov[mask]
    return X, cov


@metric(
    name="variance_explained_by_covariate",
    label="Expression variance explained by cell area",
    family=FAMILY,
    task=(Task.AGGREGATION, Task.NORMALIZATION),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX_OBS,
    min=0,
    max=1,
    direction=Direction.LOWER,
    aggregations=("mean", "max"),
    suffixes=("counts", "normalized"),
)
def variance_explained_by_covariate(
    data: MatrixLike,
    covariate: VectorLike = "area",
    *,
    layer: str | None = "normalized",
    subset: Mask = None,
) -> np.ndarray:
    """R^2 of each marker on cell area, fitted on ``[1, area, log1p(area)]``.

    Cell area is the imaging analogue of library size, and expression that
    tracks it is usually an artefact of how pixels were summarised rather than
    biology. Area enters both linearly and logged because a sum-aggregated
    marker is close to linear in it while other artefacts scale sublinearly.

    Which layer you score matters. On the counts layer this sees the aggregation
    choice -- `sum` makes every channel scale with area and `mean` does not. On
    the normalized layer it asks whether normalization removed that dependence,
    and per-cell total normalization divides out exactly the area scaling `sum`
    creates, so judging the normalized layer alone makes the two aggregations
    indistinguishable.
    """
    X, cov = _prepare(data, layer, covariate, subset)
    if X.shape[0] < 3 or X.shape[1] == 0 or cov is None or np.std(cov) == 0:
        return np.array([], dtype=float)
    design = _design(cov.astype(float))
    return np.array([r2_against(X[:, j], design, nonnegative=True, default=0.0) for j in range(X.shape[1])])


@metric(
    name="covariate_pc_correlation",
    label="Correlation of leading PCs with cell area",
    family=FAMILY,
    task=(Task.AGGREGATION, Task.NORMALIZATION),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX_OBS,
    min=0,
    max=1,
    direction=Direction.LOWER,
    aggregations=("mean",),
    suffixes=("counts", "normalized"),
)
def covariate_pc_correlation(
    data: MatrixLike,
    covariate: VectorLike = "area",
    *,
    layer: str | None = "normalized",
    n_pcs: int = 10,
    subset: Mask = None,
) -> np.ndarray:
    """Absolute correlation of each leading principal component with cell area.

    The same question as :func:`variance_explained_by_covariate`, asked of the
    panel's dominant axes of variation rather than of each marker separately. If
    the first principal component is largely cell size, every downstream
    embedding and clustering inherits that.
    """
    X, cov = _prepare(data, layer, covariate, subset)
    if X.shape[0] < 3 or X.shape[1] == 0 or cov is None or np.std(cov) == 0:
        return np.array([], dtype=float)
    c = cov.astype(float)
    k = int(min(n_pcs, X.shape[1], X.shape[0] - 1))
    if k < 1:
        return np.array([], dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD rather than sklearn PCA: one fewer dependency, same components.
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Xc @ vt[:k].T
    return np.array([abs(float(np.corrcoef(pcs[:, i], c)[0, 1])) for i in range(k) if np.std(pcs[:, i]) > 0])


@metric(
    name="mean_variance_correlation",
    label="Dependence of marker variance on marker mean",
    family=FAMILY,
    task=(Task.NORMALIZATION,),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX,
    min=-1,
    max=1,
    direction=Direction.LOWER,
)
def mean_variance_correlation(data: MatrixLike, *, layer: str | None = "normalized", subset: Mask = None) -> float:
    """Correlation between per-marker mean and per-marker variance.

    Tests whether the transform stabilised variance: a strong positive
    correlation means bright markers are still the variable ones.
    """
    X, _ = _prepare(data, layer, None, subset)
    if X.shape[1] < 3:
        return float("nan")
    means, variances = X.mean(axis=0), X.var(axis=0)
    if np.std(means) == 0 or np.std(variances) == 0:
        return float("nan")
    return float(np.corrcoef(means, variances)[0, 1])


@metric(
    name="marker_variance_evenness",
    label="Evenness of variance across the panel",
    family=FAMILY,
    task=(Task.NORMALIZATION,),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX,
    min=0,
    max=1,
    direction=Direction.HIGHER,
)
def marker_variance_evenness(data: MatrixLike, *, layer: str | None = "normalized", subset: Mask = None) -> float:
    """Shannon entropy of each marker's share of total variance, normalised to [0, 1].

    Low means a few bright markers dominate the variance and the rest contribute
    nothing to any downstream embedding.
    """
    X, _ = _prepare(data, layer, None, subset)
    return evenness(X.var(axis=0)) if X.size else float("nan")


def _compute(X, X_counts, cov, n_pcs, max_cells, random_state) -> dict[str, float]:
    """Every covariate metric over one set of cells, in the benchmark's emission order."""
    out = {
        "mean_variance_correlation": float("nan"),
        "marker_variance_evenness": float("nan"),
    }
    n_cells, n_markers = X.shape
    if n_cells < 3 or n_markers == 0:
        return out

    if max_cells > 0 and n_cells > max_cells:
        idx = np.sort(np.random.default_rng(random_state).choice(n_cells, size=max_cells, replace=False))
        X = X[idx]
        X_counts = None if X_counts is None else X_counts[idx]
        cov = None if cov is None else cov[idx]

    for matrix, suffix in ((X, "normalized"), (X if X_counts is None else X_counts, "counts")):
        keys = (
            f"variance_explained_by_covariate_{suffix}",
            f"max_marker_variance_explained_by_covariate_{suffix}",
            f"covariate_pc_correlation_{suffix}",
        )
        out.update(dict.fromkeys(keys, float("nan")))
        if matrix.shape[0] < 3 or matrix.shape[1] == 0 or cov is None or np.std(cov) == 0:
            continue
        c = cov.astype(float)
        design = _design(c)
        r2 = [r2_against(matrix[:, j], design, nonnegative=True, default=0.0) for j in range(matrix.shape[1])]
        out[keys[0]] = float(np.mean(r2))
        out[keys[1]] = float(np.max(r2))

        k = int(min(n_pcs, matrix.shape[1], matrix.shape[0] - 1))
        if k >= 1:
            Xc = matrix - matrix.mean(axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(Xc, full_matrices=False)
            pcs = Xc @ vt[:k].T
            corrs = [abs(float(np.corrcoef(pcs[:, i], c)[0, 1])) for i in range(k) if np.std(pcs[:, i]) > 0]
            if corrs:
                out[keys[2]] = float(np.mean(corrs))

    means, variances = X.mean(axis=0), X.var(axis=0)
    if n_markers >= 3 and np.std(means) > 0 and np.std(variances) > 0:
        out["mean_variance_correlation"] = float(np.corrcoef(means, variances)[0, 1])
    out["marker_variance_evenness"] = evenness(variances)
    return out


def covariate_dependence(
    data: MatrixLike,
    covariate: VectorLike = "area",
    *,
    layer: str | None = "normalized",
    counts_layer: str | None = "counts",
    n_pcs: int = 10,
    max_cells: int = 5000,
    random_state: int = 0,
    subset: Mask = None,
    qc_variants: bool | str = False,
    qc_column: str = DEFAULT_QC_COLUMN,
) -> dict[str, float]:
    """Technical-covariate dependence on both the normalized and the counts layer."""
    X = to_matrix(data, layer, fallback_layer="counts", dtype=np.float64)
    has_counts = hasattr(data, "layers") and counts_layer in getattr(data, "layers", {})
    X_counts = to_matrix(data, counts_layer, dtype=np.float64) if has_counts else None
    cov = resolve_vector(data, covariate, name="covariate")

    rows = resolve_subset(data, subset)
    if rows is not None:
        X = X[rows]
        X_counts = None if X_counts is None else X_counts[rows]
        cov = None if cov is None else cov[rows]

    out = _compute(X, X_counts, cov, n_pcs, max_cells, random_state)
    if not qc_variants:
        return out

    column = qc_variants if isinstance(qc_variants, str) else qc_column
    passed = resolve_subset(data, column)
    if passed is not None and rows is not None:
        passed = passed[rows]
    if passed is not None and passed.any():
        over_qc = _compute(
            X[passed],
            None if X_counts is None else X_counts[passed],
            None if cov is None else cov[passed],
            n_pcs,
            max_cells,
            random_state,
        )
    else:
        import warnings

        warnings.warn(
            f"{column!r} is absent or selects no cells; the _qc_pass variants are NaN.",
            UserWarning,
            stacklevel=2,
        )
        over_qc = dict.fromkeys(out, float("nan"))
    out.update({f"{k}_qc_pass": v for k, v in over_qc.items()})
    return out
