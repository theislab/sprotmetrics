"""Raw-versus-normalized diagnostics: what did the normalization actually do?

``total_cv_ratio_norm_over_raw`` was, for a long time, one of only two metrics in
the whole benchmark with a direction that worked. The rank-correlation pair in
this family is the cautionary opposite: it is exactly 1.0 in every run, because
every normalization implemented is monotone per cell and so cannot reorder a
cell's markers. It is kept for compatibility, registered as descriptive, and
excluded from the OpenProblems export.
"""

from __future__ import annotations

import numpy as np

from sprotmetrics._registry import Direction, Inputs, Reference, Task, metric
from sprotmetrics._typing import Mask, MatrixLike
from sprotmetrics._utils import (
    DEFAULT_QC_COLUMN,
    coefficient_of_variation,
    pearson_rows,
    resolve_subset,
    to_matrix,
)

__all__ = [
    "marker_variance_ratio",
    "normalization_diagnostics",
    "total_cv",
    "total_cv_ratio",
]

FAMILY = "normalization_diagnostics"

_IDS = (
    "raw_total_cv",
    "normalized_total_cv",
    "total_cv_ratio_norm_over_raw",
    "median_raw_total_intensity",
    "median_normalized_total_intensity",
    "median_marker_variance_ratio_norm_over_raw",
    "median_cell_marker_rank_correlation",
    "mean_cell_marker_rank_correlation",
)


def _rank_rows(x: np.ndarray) -> np.ndarray:
    # Average ranks are unnecessary for a marker panel this small; a stable
    # double argsort gives a deterministic per-cell rank vector.
    return np.argsort(np.argsort(x, axis=1, kind="mergesort"), axis=1, kind="mergesort").astype(float)


def _layers(data: MatrixLike, raw_layer: str, normalized_layer: str, subset: Mask):
    raw = to_matrix(data, raw_layer, dtype=np.float64)
    norm = to_matrix(data, normalized_layer, dtype=np.float64)
    mask = resolve_subset(data, subset)
    return (raw, norm) if mask is None else (raw[mask], norm[mask])


@metric(
    name="total_cv",
    label="Spread of per-cell total intensity",
    family=FAMILY,
    task=(Task.NORMALIZATION,),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX,
    min=0,
    max=np.inf,
    direction=Direction.LOWER,
    suffixes=("raw", "normalized"),
)
def total_cv(data: MatrixLike, *, layer: str | None = "counts", subset: Mask = None) -> float:
    """Coefficient of variation of per-cell summed intensity.

    Measured on one layer. Its value before and after normalization is what
    :func:`total_cv_ratio` compares; on its own it describes how uneven the
    per-cell totals are, which depends on the tissue as much as the processing.
    """
    X = to_matrix(data, layer, dtype=np.float64)
    mask = resolve_subset(data, subset)
    if mask is not None:
        X = X[mask]
    if X.shape[0] == 0:
        return float("nan")
    return float(coefficient_of_variation(X.sum(axis=1)))


@metric(
    name="total_cv_ratio_norm_over_raw",
    label="Compression of total-intensity spread",
    family=FAMILY,
    task=(Task.NORMALIZATION,),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX_TWO_LAYERS,
    min=0,
    max=np.inf,
    direction=Direction.LOWER,
)
def total_cv_ratio(
    data: MatrixLike,
    *,
    raw_layer: str = "counts",
    normalized_layer: str = "normalized",
    subset: Mask = None,
) -> float:
    """Normalized over raw coefficient of variation of per-cell totals.

    Below 1 means the normalization compressed the spread in per-cell total
    intensity, which is what it is for.
    """
    raw, norm = _layers(data, raw_layer, normalized_layer, subset)
    if raw.shape[0] == 0:
        return float("nan")
    raw_cv = float(coefficient_of_variation(raw.sum(axis=1)))
    norm_cv = float(coefficient_of_variation(norm.sum(axis=1)))
    if not np.isfinite(raw_cv) or raw_cv == 0:
        return float("nan")
    return float(norm_cv / raw_cv)


@metric(
    name="marker_variance_ratio_norm_over_raw",
    label="Per-marker variance change",
    family=FAMILY,
    task=(Task.NORMALIZATION,),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX_TWO_LAYERS,
    min=0,
    max=np.inf,
    direction=Direction.NONE,
    aggregations=("median",),
)
def marker_variance_ratio(
    data: MatrixLike,
    *,
    raw_layer: str = "counts",
    normalized_layer: str = "normalized",
    subset: Mask = None,
) -> np.ndarray:
    """Normalized over raw variance, per marker.

    Records what the transform did to each marker's spread. Descriptive: a
    normalization is supposed to change variance, and neither increasing nor
    decreasing it is right in general. Useful for seeing whether one marker was
    affected far more than the rest.
    """
    raw, norm = _layers(data, raw_layer, normalized_layer, subset)
    if raw.shape[0] == 0:
        return np.array([], dtype=float)
    return np.var(norm, axis=0) / np.maximum(np.var(raw, axis=0), 1e-12)


def _compute(X_raw, X_norm, max_cells: int, random_state: int) -> dict[str, float]:
    """Every diagnostic over one set of cells, in the benchmark's emission order."""
    if X_raw.shape[0] == 0:
        return dict.fromkeys(_IDS, float("nan"))

    raw_totals = X_raw.sum(axis=1)
    norm_totals = X_norm.sum(axis=1)
    raw_cv = float(coefficient_of_variation(raw_totals))
    norm_cv = float(coefficient_of_variation(norm_totals))
    ratio = float(norm_cv / raw_cv) if np.isfinite(raw_cv) and raw_cv != 0 else float("nan")
    var_ratio = np.var(X_norm, axis=0) / np.maximum(np.var(X_raw, axis=0), 1e-12)

    # Rank correlation is the only part that subsamples, and it uses
    # default_rng with a sorted index; keeping that exact is what makes the
    # value reproducible against the benchmark.
    n = X_raw.shape[0]
    if max_cells > 0 and n > max_cells:
        idx = np.sort(np.random.default_rng(random_state).choice(n, size=max_cells, replace=False))
        Xr, Xn = X_raw[idx], X_norm[idx]
    else:
        Xr, Xn = X_raw, X_norm

    if Xr.shape[1] >= 2 and Xr.shape[0] > 0:
        corr = pearson_rows(_rank_rows(Xr), _rank_rows(Xn))
        median_corr, mean_corr = float(np.nanmedian(corr)), float(np.nanmean(corr))
    else:
        median_corr = mean_corr = float("nan")

    return {
        "raw_total_cv": raw_cv,
        "normalized_total_cv": norm_cv,
        "total_cv_ratio_norm_over_raw": ratio,
        "median_raw_total_intensity": float(np.median(raw_totals)),
        "median_normalized_total_intensity": float(np.median(norm_totals)),
        "median_marker_variance_ratio_norm_over_raw": float(np.median(var_ratio)),
        "median_cell_marker_rank_correlation": median_corr,
        "mean_cell_marker_rank_correlation": mean_corr,
    }


def normalization_diagnostics(
    data: MatrixLike,
    *,
    raw_layer: str = "counts",
    normalized_layer: str = "normalized",
    max_cells: int = 5000,
    random_state: int = 0,
    subset: Mask = None,
    qc_variants: bool | str = False,
    qc_column: str = DEFAULT_QC_COLUMN,
) -> dict[str, float]:
    """Compare the normalized layer against the raw one."""
    raw, norm = _layers(data, raw_layer, normalized_layer, subset)
    out = _compute(raw, norm, max_cells, random_state)
    if not qc_variants:
        return out

    column = qc_variants if isinstance(qc_variants, str) else qc_column
    passed = resolve_subset(data, column)
    rows = resolve_subset(data, subset)
    if passed is not None and rows is not None:
        passed = passed[rows]
    if passed is not None and passed.any():
        over_qc = _compute(raw[passed], norm[passed], max_cells, random_state)
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
