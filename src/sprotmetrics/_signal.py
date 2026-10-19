"""Pixel-level metrics: what the cell table cannot see.

Portions adapted from theislab/sprot_best_practices (BSD-3-Clause,
Copyright (c) 2023 Theis Lab); see NOTICE and
LICENSES/BSD-3-Clause-theislab-sprot_best_practices.txt.

These are the only metrics computed from pixels rather than from the aggregated
matrix, and so the only ones that can see signal which never made it inside a
mask, or background that correction failed to remove. Every one is a reduction
over a :class:`~sprotmetrics.CellPixelStats` built once per image.
"""

from __future__ import annotations

import numpy as np

from sprotmetrics._pixelstats import CellPixelStats, cell_pixel_stats
from sprotmetrics._registry import (
    Direction,
    Inputs,
    Reference,
    Status,
    Strength,
    Task,
    metric,
)
from sprotmetrics._typing import Mask, MatrixLike, Sequence
from sprotmetrics._utils import DEFAULT_QC_COLUMN

__all__ = [
    "cellular_specificity",
    "intracell_coverage",
    "marker_gini",
    "nuclear_signal_containment",
    "rel_sni",
    "signal_background",
    "signal_background_from_stats",
    "signal_drop_on_aggregation",
    "signal_to_background_ratio",
]

FAMILY = "signal_background"

_trapezoid = getattr(np, "trapezoid", None) or np.trapz

_IDS = (
    "median_rel_sni",
    "min_rel_sni",
    "median_intracell_coverage",
    "median_marker_gini",
    "median_signal_to_background_ratio",
    "median_cellular_specificity",
)


def _gini(x: np.ndarray) -> float:
    arr = np.sort(np.asarray(x, dtype=float))
    n = arr.size
    total = arr.sum()
    if n == 0 or total <= 0:
        return float("nan")
    return float((n + 1 - 2 * np.sum(np.cumsum(arr)) / total) / n)


@metric(
    name="rel_sni",
    label="Signal integral against a rank-matched background",
    family=FAMILY,
    task=(Task.IMAGE_CORRECTION, Task.SEGMENTATION),
    reference=Reference.NONE,
    inputs=Inputs.IMAGE_LABELS,
    min=-np.inf,
    max=np.inf,
    direction=Direction.HIGHER,
    aggregations=("median", "min"),
    origin="sprot protein_quality_metrics.metric_rel_sni (adapted)",
    notes="Negative means cells are dimmer than the ring immediately around them.",
)
def rel_sni(stats: CellPixelStats, *, norm_percentile: float = 95.0) -> np.ndarray:
    """Area between the sorted per-cell intensity curve and a rank-matched background curve.

    Normalised by a high percentile of cell intensity so markers of different
    brightness are comparable. The headline signal-to-background measure.
    """
    sel = stats.selection
    out = np.full(len(stats.markers), np.nan)
    for c in range(len(stats.markers)):
        cell_means = stats.means[sel, c]
        cell_means = cell_means[np.isfinite(cell_means)]
        background = stats.background_curves[c]
        if cell_means.size == 0 or background is None or background.size == 0:
            continue
        y_cell = np.sort(cell_means)
        y_bg = np.interp(np.linspace(0, 100, y_cell.size), np.linspace(0, 100, background.size), background)
        norm = float(np.percentile(y_cell, norm_percentile))
        if norm > 0:
            out[c] = float(_trapezoid(y_cell - y_bg) / y_cell.size / norm)
    return out


@metric(
    name="intracell_coverage",
    label="Extent of staining within the brightest cells",
    family=FAMILY,
    task=(Task.IMAGE_CORRECTION,),
    reference=Reference.NONE,
    inputs=Inputs.IMAGE_LABELS,
    min=0,
    max=1,
    direction=Direction.HIGHER,
    aggregations=("median",),
    status=Status.EXPERIMENTAL,
    origin="sprot protein_quality_metrics.metric_intracell_coverage (adapted)",
    notes=(
        "Reads exactly 1.0 on exemplar-001. Averaging the top 5% of cells by "
        "coverage selects cells that are above their own Otsu threshold "
        "everywhere, so the statistic saturates by construction. Marked "
        "experimental until the selection is redesigned."
    ),
)
def intracell_coverage(stats: CellPixelStats, *, top_fraction: float = 0.05) -> np.ndarray:
    """Fraction of its own pixels above threshold, in the most-covered cells, per marker.

    Separates real staining, which is bright *and* extensive, from punctate
    artefact, which is bright and sparse.
    """
    sel = stats.selection
    out = np.full(len(stats.markers), np.nan)
    for c in range(len(stats.markers)):
        cov = stats.prop_positive[sel, c]
        cov = cov[np.isfinite(cov)]
        if cov.size:
            top = np.sort(cov)[-max(1, int(cov.size * top_fraction)) :]
            out[c] = float(top.mean())
    return out


@metric(
    name="signal_to_background_ratio",
    label="Mean inside masks over mean in the ring",
    family=FAMILY,
    task=(Task.IMAGE_CORRECTION,),
    reference=Reference.NONE,
    inputs=Inputs.IMAGE_LABELS,
    min=0,
    max=np.inf,
    direction=Direction.HIGHER,
    aggregations=("median",),
    origin="Metrics-2.xlsx row 4 (no prior code)",
    notes=(
        "Reads only 1.02-1.04 on exemplar-001: a 3px ring is mostly cell-adjacent "
        "membrane. Raise dilation_iterations to push the ring into real background."
    ),
)
def signal_to_background_ratio(stats: CellPixelStats) -> np.ndarray:
    """Mean intensity inside the masks over mean intensity in the surrounding ring.

    Needs no pseudo-cell sampling, so it survives the cases where the
    rank-matched curve of :func:`rel_sni` cannot be built.

    The division is carried out in the image's own dtype. Multiplexed images are
    stored float32, and under NumPy's weak scalar promotion the benchmark's
    ``np.mean(image[c][labels > 0]) / ring_mean`` stays float32; doing it in
    float64 is marginally more accurate but shifts the result in the eighth
    significant figure, which would silently move published numbers.
    """
    dtype = stats.intensity_dtype
    out = np.full(len(stats.markers), np.nan)
    for c in range(len(stats.markers)):
        ring = stats.ring_mean[c]
        if np.isfinite(ring) and ring > 0 and np.isfinite(stats.inside_mean[c]):
            # Scalar division, so NumPy's weak promotion keeps a float32
            # numerator in float32 -- array/array division would silently widen
            # it to float64 and move the result.
            out[c] = float(dtype.type(stats.inside_mean[c]) / float(ring))
    return out


@metric(
    name="cellular_specificity",
    label="Share of positive pixels falling inside masks",
    family=FAMILY,
    task=(Task.IMAGE_CORRECTION,),
    reference=Reference.NONE,
    inputs=Inputs.IMAGE_LABELS,
    min=0,
    max=1,
    direction=Direction.HIGHER,
    aggregations=("median",),
    origin="Metrics-2.xlsx row 14 (no prior code)",
)
def cellular_specificity(stats: CellPixelStats) -> np.ndarray:
    """Positive pixels inside the masks, over positive pixels inside plus in the ring."""
    inside, ring = stats.inside_positive_fraction, stats.ring_positive_fraction
    total = inside + ring
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(total) & (total > 0), inside / total, np.nan)


@metric(
    name="marker_gini",
    label="Concentration of intensity across cells",
    family=FAMILY,
    task=(Task.MARKER_CHARACTERISATION,),
    reference=Reference.NONE,
    inputs=Inputs.IMAGE_LABELS,
    min=0,
    max=1,
    direction=Direction.NONE,
    aggregations=("median",),
    origin="sprot protein_quality_metrics.metric_sparsity_gini (adapted)",
    notes="Descriptive: high suits a lineage marker and is wrong for a housekeeper.",
)
def marker_gini(stats: CellPixelStats) -> np.ndarray:
    """Gini coefficient of per-cell mean intensity, per marker."""
    sel = stats.selection
    out = np.full(len(stats.markers), np.nan)
    for c in range(len(stats.markers)):
        values = stats.means[sel, c]
        values = values[np.isfinite(values)]
        if values.size:
            out[c] = _gini(values)
    return out


@metric(
    name="nuclear_signal_containment",
    label="Nuclear signal falling inside any mask",
    family=FAMILY,
    task=(Task.SEGMENTATION,),
    reference=Reference.PANEL_KNOWLEDGE,
    inputs=Inputs.IMAGE_LABELS,
    min=0,
    max=1,
    direction=Direction.HIGHER,
    strength=Strength.WEAK,
    status=Status.EXPERIMENTAL,
    origin="proposed in ISP_BENCHMARK_ROADMAP.md",
    notes=(
        "Must be read against spillover_ratio, never alone. Growing every mask "
        "raises it, so on exemplar-001 it prefers watershed_expand (0.67) to "
        "threshold (0.44) while spillover_ratio says watershed over-expands "
        "(1.21 against 1.09). The pair brackets segmentation; either one by "
        "itself is gameable."
    ),
)
def nuclear_signal_containment(stats: CellPixelStats, nuclear_markers: Sequence[str] | str | None = None) -> float:
    """Fraction of total nuclear-channel signal that falls inside a mask.

    The only way to see cells the segmentation missed entirely: every metric
    computed from the cell table is blind to them, because a missed cell is
    simply absent from the table.
    """
    from sprotmetrics._panel import resolve_nuclear_markers

    idx = resolve_nuclear_markers(stats.markers, nuclear_markers)
    if not idx:
        return float("nan")
    total = float(stats.total_sum[idx].sum())
    if total <= 0:
        return float("nan")
    return float(stats.inside_sum[idx].sum() / total)


@metric(
    name="signal_drop_on_aggregation",
    label="Positive cells aggregated away to zero",
    family=FAMILY,
    task=(Task.AGGREGATION,),
    reference=Reference.NONE,
    inputs=Inputs.IMAGE_LABELS,
    min=0,
    max=1,
    direction=Direction.LOWER,
    aggregations=("median", "max"),
    status=Status.EXPERIMENTAL,
    origin="Metrics-2.xlsx row 13",
)
def signal_drop_on_aggregation(
    stats: CellPixelStats,
    table: MatrixLike,
    *,
    layer: str | None = "counts",
    cell_id_key: str | None = None,
    min_positive_pixels: int = 1,
) -> np.ndarray:
    """Cells holding positive pixels whose aggregated value is nonetheless zero, per marker.

    Aggregation is the axis the benchmark covers worst, and this asks the
    question directly: did summarising the pixels throw the signal away?

    Rows of `table` are aligned to label ids through ``obs[cell_id_key]`` when
    given, else through integer-castable ``obs_names``, else positionally -- and
    a positional fallback whose length disagrees raises rather than scoring a
    silently mismatched join.
    """
    from sprotmetrics._utils import to_matrix

    X = to_matrix(table, layer, dtype=np.float64)
    n_labels = stats.means.shape[0]

    labels_for_rows = None
    if cell_id_key is not None and hasattr(table, "obs"):
        labels_for_rows = np.asarray(table.obs[cell_id_key], dtype=np.int64)
    elif hasattr(table, "obs_names"):
        try:
            labels_for_rows = np.asarray([int(n) for n in table.obs_names], dtype=np.int64)
        except (TypeError, ValueError):
            labels_for_rows = None
    if labels_for_rows is None:
        if X.shape[0] != n_labels:
            raise ValueError(
                f"cannot align {X.shape[0]} table rows to {n_labels} labels: obs_names are not "
                "integer label ids and the lengths differ. Pass cell_id_key."
            )
        labels_for_rows = np.arange(1, n_labels + 1)

    aggregated = np.full((n_labels, X.shape[1]), np.nan)
    valid = (labels_for_rows >= 1) & (labels_for_rows <= n_labels)
    aggregated[labels_for_rows[valid] - 1] = X[valid]

    sel = stats.selection
    positive_pixels = stats.prop_positive * stats.areas[:, None]
    out = np.full(len(stats.markers), np.nan)
    for c in range(min(len(stats.markers), X.shape[1])):
        had_signal = sel & (positive_pixels[:, c] >= min_positive_pixels)
        if not had_signal.any():
            continue
        values = aggregated[had_signal, c]
        out[c] = float(np.mean(np.nan_to_num(values, nan=0.0) <= 0))
    return out


def _summarise(stats: CellPixelStats, norm_percentile: float, top_fraction: float) -> dict:
    """The benchmark's six ids over one selection of cells, in its emission order."""
    if stats.n_objects == 0:
        return dict.fromkeys(_IDS, float("nan"))

    def med(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.median(finite)) if finite.size else float("nan")

    sni = rel_sni(stats, norm_percentile=norm_percentile)
    finite_sni = sni[np.isfinite(sni)]
    return {
        "median_rel_sni": med(sni),
        "min_rel_sni": float(np.min(finite_sni)) if finite_sni.size else float("nan"),
        "median_intracell_coverage": med(intracell_coverage(stats, top_fraction=top_fraction)),
        "median_marker_gini": med(marker_gini(stats)),
        "median_signal_to_background_ratio": med(signal_to_background_ratio(stats)),
        "median_cellular_specificity": med(cellular_specificity(stats)),
    }


def qc_mask_from_table(table: MatrixLike, n_labels: int, qc_column: str = DEFAULT_QC_COLUMN):
    """Boolean per-label mask from a cell table's QC column.

    QC lives in the cell table, not in the pixels, so it has to be carried
    across by label id.
    """
    if not hasattr(table, "obs") or qc_column not in table.obs.columns:
        return None
    passed = np.asarray(table.obs[qc_column].to_numpy()).astype(bool)
    keep = np.zeros(n_labels, dtype=bool)
    for cid, ok in zip(np.asarray(table.obs_names, dtype=str), passed, strict=False):
        try:
            label = int(cid)
        except ValueError:
            continue
        if ok and 1 <= label <= n_labels:
            keep[label - 1] = True
    return keep


def signal_background_from_stats(
    stats: CellPixelStats,
    *,
    norm_percentile: float = 95.0,
    top_fraction: float = 0.05,
    cell_subset: Mask = None,
    qc_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Every pixel-level metric for a :class:`CellPixelStats` already built."""
    view = stats if cell_subset is None else stats.subset(cell_subset)
    out = _summarise(view, norm_percentile, top_fraction)
    if qc_mask is None:
        return out
    selected = qc_mask & view.present
    if selected.any():
        over_qc = _summarise(view.subset(selected), norm_percentile, top_fraction)
    else:
        import warnings

        warnings.warn(
            "the QC mask selects no labelled cells; the _qc_pass variants are NaN.",
            UserWarning,
            stacklevel=2,
        )
        over_qc = dict.fromkeys(out, float("nan"))
    out.update({f"{k}_qc_pass": v for k, v in over_qc.items()})
    return out


def signal_background(
    image: np.ndarray,
    labels: np.ndarray,
    markers: Sequence[str] | None = None,
    *,
    table: MatrixLike | None = None,
    dilation_iterations: int = 3,
    norm_percentile: float = 95.0,
    top_fraction: float = 0.05,
    random_state: int = 0,
    cell_subset: Mask = None,
    qc_variants: bool | str = False,
    qc_column: str = DEFAULT_QC_COLUMN,
) -> dict[str, float]:
    """Every pixel-level metric, from a single pass over the image.

    Parameters
    ----------
    table
        Cell table supplying the QC column for the ``_qc_pass`` variants, matched
        to label ids through its ``obs_names``.
    qc_variants
        Also report each metric over the QC-passing cells. Requires `table`.
    """
    stats = cell_pixel_stats(
        image,
        labels,
        markers,
        dilation_iterations=dilation_iterations,
        random_state=random_state,
    )
    mask = None
    if qc_variants:
        column = qc_variants if isinstance(qc_variants, str) else qc_column
        mask = qc_mask_from_table(table, stats.means.shape[0], column) if table is not None else None
        if mask is None:
            import warnings

            warnings.warn(
                f"no table with a {column!r} column was given; the _qc_pass variants are NaN.",
                UserWarning,
                stacklevel=2,
            )
            mask = np.zeros(stats.means.shape[0], dtype=bool)
    return signal_background_from_stats(
        stats,
        norm_percentile=norm_percentile,
        top_fraction=top_fraction,
        cell_subset=cell_subset,
        qc_mask=mask,
    )
