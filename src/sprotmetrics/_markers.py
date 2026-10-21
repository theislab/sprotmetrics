"""Per-marker distribution: brightness, spread, detection rate, dynamic range.

The benchmark reported only panel-level summaries and buried the per-marker table
in ``uns``. Here :func:`marker_table` is public, because the useful question is
usually not "what is the median coverage" but "which marker is dragging it down".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sprotmetrics._registry import Direction, Inputs, Reference, Status, Strength, Task, metric
from sprotmetrics._typing import Mask, MatrixLike, Sequence
from sprotmetrics._utils import (
    DEFAULT_QC_COLUMN,
    marker_names,
    otsu_threshold,
    resolve_subset,
    to_matrix,
)

__all__ = [
    "marker_bimodality",
    "marker_cv",
    "marker_distribution",
    "marker_dynamic_range",
    "marker_mean",
    "marker_table",
    "marker_zero_fraction",
]

FAMILY = "marker_distribution"


def _matrix(data: MatrixLike, layer: str | None, subset: Mask) -> np.ndarray:
    X = to_matrix(data, layer, dtype=np.float64)
    mask = resolve_subset(data, subset)
    return X if mask is None else X[mask]


def _safe_nanmedian(x) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.nanmedian(arr)) if arr.size else float("nan")


def _safe_nanmean(x) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def marker_table(
    data: MatrixLike,
    markers: Sequence[str] | None = None,
    *,
    layer: str | None = "counts",
    zero_threshold: float = 0.0,
    coverage_threshold: float = 0.1,
    quantiles: tuple[float, float] = (0.01, 0.99),
    subset: Mask = None,
) -> pd.DataFrame:
    """Per-marker mean, std, cv, median, zero fraction, coverage, quantiles, dynamic range.

    The dynamic range is a high-minus-low quantile difference rather than
    max-minus-min, so a single hot pixel cannot define it.
    """
    X = _matrix(data, layer, subset)
    names = marker_names(data, markers, n_markers=X.shape[1] if X.ndim == 2 else None)
    if X.size == 0:
        return pd.DataFrame(index=names)
    low_q, high_q = quantiles
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    low = np.quantile(X, low_q, axis=0)
    high = np.quantile(X, high_q, axis=0)
    return pd.DataFrame(
        {
            "marker": names,
            "mean": mean,
            "std": std,
            "cv": std / np.maximum(mean, 1e-12),
            "median": np.median(X, axis=0),
            "zero_fraction": (X <= zero_threshold).mean(axis=0),
            "coverage": (X > coverage_threshold).mean(axis=0),
            f"q{int(low_q * 100):02d}": low,
            f"q{int(high_q * 100):02d}": high,
            "dynamic_range": high - low,
        },
        index=names,
    )


@metric(
    name="marker_mean",
    label="Per-marker mean intensity",
    family=FAMILY,
    task=(Task.IMAGE_CORRECTION,),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX,
    min=0,
    max=np.inf,
    direction=Direction.NONE,
    aggregations=("median",),
)
def marker_mean(data: MatrixLike, *, layer: str | None = "counts", subset: Mask = None) -> np.ndarray:
    """Mean intensity across cells, per marker.

    Descriptive. Panel-wide brightness depends on antibody concentration,
    exposure and detector gain as much as on biology, so a higher value is not a
    better one.
    """
    X = _matrix(data, layer, subset)
    return X.mean(axis=0) if X.size else np.array([], dtype=float)


@metric(
    name="marker_cv",
    label="Per-marker coefficient of variation",
    family=FAMILY,
    task=(Task.NORMALIZATION,),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX,
    min=0,
    max=np.inf,
    direction=Direction.AMBIGUOUS,
    aggregations=("median",),
    notes="Spread across cells may be biology or noise, so it cannot rank a method alone.",
)
def marker_cv(data: MatrixLike, *, layer: str | None = "counts", subset: Mask = None) -> np.ndarray:
    """Standard deviation over mean across cells, per marker.

    Deliberately ambiguous. A high coefficient of variation can mean a marker
    cleanly separates two populations, or that it is noisy; nothing in the
    number distinguishes those. Read it beside :func:`marker_bimodality`, which
    asks whether the spread resolves into two populations or one smear.
    """
    X = _matrix(data, layer, subset)
    if X.size == 0:
        return np.array([], dtype=float)
    return X.std(axis=0) / np.maximum(X.mean(axis=0), 1e-12)


@metric(
    name="marker_zero_fraction",
    label="Per-marker fraction at or below zero",
    family=FAMILY,
    task=(Task.IMAGE_CORRECTION,),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX,
    min=0,
    max=1,
    direction=Direction.LOWER,
    strength=Strength.WEAK,
    aggregations=("median", "mean"),
)
def marker_zero_fraction(
    data: MatrixLike,
    *,
    layer: str | None = "counts",
    threshold: float = 0.0,
    subset: Mask = None,
) -> np.ndarray:
    """Fraction of cells at or below `threshold`, per marker.

    The complement of :func:`marker_coverage`, kept separate because a threshold
    of exactly zero answers a different question: how much of the panel was
    never detected at all. Fluorescence intensities are rarely exactly zero, so
    a threshold above zero is usually what you want.
    """
    X = _matrix(data, layer, subset)
    return (X <= threshold).mean(axis=0) if X.size else np.array([], dtype=float)


@metric(
    name="marker_dynamic_range",
    label="Per-marker dynamic range",
    family=FAMILY,
    task=(Task.IMAGE_CORRECTION,),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX,
    min=0,
    max=np.inf,
    direction=Direction.HIGHER,
    strength=Strength.WEAK,
    aggregations=("median",),
)
def marker_dynamic_range(
    data: MatrixLike,
    *,
    layer: str | None = "counts",
    quantiles: tuple[float, float] = (0.01, 0.99),
    subset: Mask = None,
) -> np.ndarray:
    """High minus low quantile of intensity, per marker.

    A quantile spread rather than max minus min, which two pixels would
    otherwise decide. Image correction that clips bright signal compresses this;
    so does a marker that failed to stain, and the two look alike here, which is
    why it is weak evidence on its own.
    """
    X = _matrix(data, layer, subset)
    if X.size == 0:
        return np.array([], dtype=float)
    low, high = quantiles
    return np.quantile(X, high, axis=0) - np.quantile(X, low, axis=0)


@metric(
    name="marker_bimodality",
    label="Marker positive-population separability",
    family=FAMILY,
    task=(Task.IMAGE_CORRECTION, Task.NORMALIZATION),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX,
    min=0,
    max=np.inf,
    direction=Direction.HIGHER,
    aggregations=("median", "min"),
    status=Status.EXPERIMENTAL,
    references=("10.4137/CIN.S2846",),
    notes="Asks whether a marker can still be called positive at all after preprocessing.",
)
def marker_bimodality(
    data: MatrixLike,
    *,
    layer: str | None = "counts",
    method: str = "otsu_variance_ratio",
    log1p: bool = True,
    subset: Mask = None,
) -> np.ndarray:
    """Separability of the two populations an Otsu split produces, per marker.

    ``otsu_variance_ratio`` is between-class over within-class variance.
    ``bimodality_index`` is Wang et al. (2009), ``sqrt(p(1-p)) * |m1-m2| / sigma``,
    with both components taken from the Otsu split rather than an EM fit; it is
    the version with a conventional cutoff (roughly BI > 1.1).

    This is what image correction and normalization are *for*, and nothing in the
    benchmark measured it.
    """
    X = _matrix(data, layer, subset)
    if X.size == 0:
        return np.array([], dtype=float)
    out = np.full(X.shape[1], np.nan)
    for j in range(X.shape[1]):
        column = X[:, j]
        column = column[np.isfinite(column)]
        if column.size < 3:
            continue
        work = np.log1p(np.clip(column, 0, None)) if log1p else column
        thr = otsu_threshold(column, log1p=log1p)
        cut = np.log1p(max(thr, 0.0)) if log1p else thr
        hi, lo = work[work > cut], work[work <= cut]
        if hi.size < 2 or lo.size < 2:
            continue
        p = hi.size / work.size
        delta = abs(hi.mean() - lo.mean())
        within = (lo.size * lo.var() + hi.size * hi.var()) / work.size
        if within <= 0:
            continue
        if method == "bimodality_index":
            out[j] = float(np.sqrt(p * (1 - p)) * delta / np.sqrt(within))
        elif method == "otsu_variance_ratio":
            between = p * (1 - p) * delta**2
            out[j] = float(between / within)
        else:
            raise ValueError(f"unknown method: {method!r}")
    return out


def _summarise(table: pd.DataFrame) -> dict[str, float]:
    """Panel summaries, in the benchmark's emission order."""
    return {
        "median_marker_mean": _safe_nanmedian(table.get("mean", [])),
        "median_marker_cv": _safe_nanmedian(table.get("cv", [])),
        "median_marker_zero_fraction": _safe_nanmedian(table.get("zero_fraction", [])),
        "median_marker_coverage": _safe_nanmedian(table.get("coverage", [])),
        "median_marker_dynamic_range": _safe_nanmedian(table.get("dynamic_range", [])),
        "mean_marker_zero_fraction": _safe_nanmean(table.get("zero_fraction", [])),
        "mean_marker_coverage": _safe_nanmean(table.get("coverage", [])),
    }


def marker_distribution(
    data: MatrixLike,
    markers: Sequence[str] | None = None,
    *,
    layer: str | None = "counts",
    zero_threshold: float = 0.0,
    coverage_threshold: float = 0.1,
    quantiles: tuple[float, float] = (0.01, 0.99),
    subset: Mask = None,
    qc_variants: bool | str = False,
    qc_column: str = DEFAULT_QC_COLUMN,
) -> dict[str, float]:
    """Panel-level summaries of the per-marker distribution."""
    kwargs = {
        "layer": layer,
        "zero_threshold": zero_threshold,
        "coverage_threshold": coverage_threshold,
        "quantiles": quantiles,
    }
    out = _summarise(marker_table(data, markers, subset=subset, **kwargs))
    if not qc_variants:
        return out

    column = qc_variants if isinstance(qc_variants, str) else qc_column
    passed = resolve_subset(data, column)
    rows = resolve_subset(data, subset)
    if passed is not None and rows is not None:
        passed = passed & rows
    if passed is not None and passed.any():
        over_qc = _summarise(marker_table(data, markers, subset=passed, **kwargs))
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
