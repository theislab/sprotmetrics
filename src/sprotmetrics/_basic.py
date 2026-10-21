"""Dataset and object statistics: how many cells, how big, how bright.

Mostly descriptive. ``median_area`` of 76 against 255 does not say which
segmentation was better, so these are registered with ``Direction.NONE`` and are
deliberately excluded from the OpenProblems export: they describe a run, they do
not rank one.
"""

from __future__ import annotations

import numpy as np

from sprotmetrics._registry import Direction, Inputs, Reference, Status, Strength, Task, metric
from sprotmetrics._typing import Mask, MatrixLike, VectorLike
from sprotmetrics._utils import (
    DEFAULT_QC_COLUMN,
    resolve_subset,
    resolve_vector,
    safe_float,
    to_matrix,
)

__all__ = [
    "basic_statistics",
    "cell_area",
    "marker_coverage",
    "n_cells",
    "n_markers",
    "qc_pass_rate",
    "total_intensity",
]

FAMILY = "basic"

#: These two cannot differ between all cells and the QC-passing subset: the panel
#: size is fixed, and the pass rate within the passing cells is 1 by definition.
NO_QC_VARIANT = frozenset({"n_markers", "qc_pass_rate"})


def _matrix(data: MatrixLike, layer: str | None, subset: Mask) -> np.ndarray:
    X = to_matrix(data, layer, dtype=np.float64)
    mask = resolve_subset(data, subset)
    return X if mask is None else X[mask]


@metric(
    name="n_cells",
    label="Number of segmented objects",
    family=FAMILY,
    task=(Task.SEGMENTATION,),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX,
    min=0,
    max=np.inf,
    direction=Direction.NONE,
    notes="More is not better: over-segmentation inflates it.",
)
def n_cells(data: MatrixLike, *, layer: str | None = "counts", subset: Mask = None) -> float:
    """Number of segmented objects in the table."""
    return float(_matrix(data, layer, subset).shape[0])


@metric(
    name="n_markers",
    label="Panel size",
    family=FAMILY,
    task=(Task.DATASET_STATISTICS,),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX,
    min=0,
    max=np.inf,
    direction=Direction.NONE,
    qc_variant=False,
    notes="Invariant by construction; carries no benchmark signal.",
)
def n_markers(data: MatrixLike, *, layer: str | None = "counts", subset: Mask = None) -> float:
    """Number of markers in the panel."""
    return float(_matrix(data, layer, subset).shape[1])


@metric(
    name="qc_pass_rate",
    label="Fraction of cells passing QC",
    family=FAMILY,
    task=(Task.QC,),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX_OBS,
    min=0,
    max=1,
    direction=Direction.NONE,
    qc_variant=False,
    notes="Has no direction and cannot: a no-op QC method scores 1.0 by construction.",
)
def qc_pass_rate(data: MatrixLike, *, qc_column: str = DEFAULT_QC_COLUMN, subset: Mask = None) -> float:
    """Fraction of segmented objects flagged as passing QC."""
    passed = resolve_subset(data, qc_column)
    if passed is None:
        return float("nan")
    mask = resolve_subset(data, subset)
    if mask is not None:
        passed = passed[mask]
    return safe_float(np.mean(passed)) if passed.size else float("nan")


@metric(
    name="area",
    label="Object area",
    family=FAMILY,
    task=(Task.SEGMENTATION,),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX_OBS,
    min=0,
    max=np.inf,
    direction=Direction.NONE,
    aggregations=("median", "mean"),
    notes="No biological target is encoded, so neither larger nor smaller is better.",
)
def cell_area(data: MatrixLike, area: VectorLike = "area", *, subset: Mask = None) -> np.ndarray:
    """Per-cell mask area, as written by the aggregation step.

    Descriptive, and deliberately so: a median area of 76 against 255 says the
    segmentations differ, not which one is right. What a plausible area is
    depends on the tissue and the magnification. Area matters here mostly as the
    technical covariate other metrics regress against.
    """
    values = resolve_vector(data, area, name="area")
    if values is None:
        return np.array([], dtype=float)
    mask = resolve_subset(data, subset)
    return values if mask is None else values[mask]


@metric(
    name="total_intensity",
    label="Per-cell summed intensity",
    family=FAMILY,
    task=(Task.AGGREGATION,),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX,
    min=0,
    max=np.inf,
    direction=Direction.NONE,
    aggregations=("median", "mean"),
)
def total_intensity(data: MatrixLike, *, layer: str | None = "counts", subset: Mask = None) -> np.ndarray:
    """Per-cell intensity summed across the panel.

    The imaging analogue of library size. Descriptive on its own, and the
    quantity a per-cell normalization is usually trying to equalise -- see
    :func:`total_cv_ratio`.
    """
    return _matrix(data, layer, subset).sum(axis=1)


@metric(
    name="marker_coverage",
    label="Per-marker detection rate",
    family=FAMILY,
    task=(Task.IMAGE_CORRECTION,),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX,
    min=0,
    max=1,
    direction=Direction.HIGHER,
    strength=Strength.WEAK,
    aggregations=("median", "min"),
    status=Status.STABLE,
    notes="Saturates near 0.99 on exemplar-001, hence Strength.WEAK.",
)
def marker_coverage(
    data: MatrixLike,
    *,
    layer: str | None = "counts",
    threshold: float = 0.1,
    subset: Mask = None,
) -> np.ndarray:
    """Fraction of cells whose intensity exceeds `threshold`, per marker.

    Reads as a detection rate: what share of cells carry this marker at all. It
    saturates near 1 on a well-stained panel, which is why it is weak evidence
    -- once every marker is detected in 99% of cells, the metric has nothing
    left to separate methods with.
    """
    X = _matrix(data, layer, subset)
    if X.shape[0] == 0:
        return np.array([], dtype=float)
    return (X > threshold).mean(axis=0)


def _statistics(
    X: np.ndarray, area: np.ndarray | None, passed: np.ndarray | None, threshold: float
) -> dict[str, float]:
    """Every basic metric over one set of cells, in the benchmark's emission order."""
    n = float(X.shape[0])
    has_area = area is not None and area.size
    totals = X.sum(axis=1) if n else None
    coverage = (X > threshold).mean(axis=0) if n else None
    return {
        "n_cells": n,
        "n_markers": float(X.shape[1]),
        "qc_pass_rate": safe_float(np.mean(passed)) if passed is not None and n else float("nan"),
        "median_area": safe_float(np.median(area)) if has_area else float("nan"),
        "mean_area": safe_float(np.mean(area)) if has_area else float("nan"),
        "median_total_intensity": safe_float(np.median(totals)) if n else float("nan"),
        "mean_total_intensity": safe_float(np.mean(totals)) if n else float("nan"),
        "median_marker_coverage": safe_float(np.median(coverage)) if n else float("nan"),
        "min_marker_coverage": safe_float(np.min(coverage)) if n else float("nan"),
    }


def basic_statistics(
    data: MatrixLike,
    *,
    layer: str | None = "counts",
    area: VectorLike = "area",
    threshold: float = 0.1,
    subset: Mask = None,
    qc_variants: bool | str = False,
    qc_column: str = DEFAULT_QC_COLUMN,
) -> dict[str, float]:
    """Dataset and object statistics for one run.

    Parameters
    ----------
    qc_variants
        Also report every metric over the QC-passing cells, under a ``_qc_pass``
        suffix. ``True`` uses `qc_column`; a string names another column. Without
        it a library user gets nine numbers rather than sixteen.
    """
    X = to_matrix(data, layer, dtype=np.float64)
    area_values = resolve_vector(data, area, name="area")
    column = qc_variants if isinstance(qc_variants, str) else qc_column
    passed = resolve_subset(data, column) if qc_variants else None
    rows = resolve_subset(data, subset)
    if rows is not None:
        X = X[rows]
        area_values = None if area_values is None else area_values[rows]
        passed = None if passed is None else passed[rows]

    # The QC column is needed for qc_pass_rate even when the twins are not wanted.
    rate_mask = passed if passed is not None else resolve_subset(data, qc_column)
    if rows is not None and passed is None and rate_mask is not None:
        rate_mask = rate_mask[rows]
    out = _statistics(X, area_values, rate_mask, threshold)

    if not qc_variants:
        return out
    if passed is not None and passed.any():
        over_qc = _statistics(
            X[passed],
            None if area_values is None else area_values[passed],
            passed[passed],
            threshold,
        )
    else:
        import warnings

        warnings.warn(
            f"{column!r} is absent or selects no cells; the _qc_pass variants are NaN. "
            "Without QC annotation the QC axis of a benchmark carries no signal.",
            UserWarning,
            stacklevel=2,
        )
        over_qc = dict.fromkeys(out, float("nan"))
    for key, value in over_qc.items():
        if key not in NO_QC_VARIANT:
            out[f"{key}_qc_pass"] = value
    return out
