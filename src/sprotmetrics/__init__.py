"""Quality metrics for imaging-based spatial proteomics.

Three levels of API, and most users want the first:

* **Quantities** -- one function per scientific quantity, vector-valued where the
  quantity is per cell or per marker (:func:`marker_coverage`, :func:`rel_sni`).
* **Metric ids** -- :func:`compute` applies the panel aggregation the registry
  declares, so ``compute("median_marker_coverage", adata)`` is a single number.
* **Families** -- :func:`basic_statistics`, :func:`signal_background` and friends
  return every id a benchmark component emits, including ``_qc_pass`` variants.

:func:`list_metrics` is the catalogue: what each metric measures, its range, and
whether a larger value is better.
"""

from importlib.metadata import version

from sprotmetrics._basic import (
    basic_statistics,
    cell_area,
    marker_coverage,
    n_cells,
    n_markers,
    qc_pass_rate,
    total_intensity,
)
from sprotmetrics._covariates import (
    covariate_dependence,
    covariate_pc_correlation,
    marker_variance_evenness,
    mean_variance_correlation,
    variance_explained_by_covariate,
)
from sprotmetrics._markers import (
    marker_bimodality,
    marker_cv,
    marker_distribution,
    marker_dynamic_range,
    marker_mean,
    marker_table,
    marker_zero_fraction,
)
from sprotmetrics._normalization import (
    marker_variance_ratio,
    normalization_diagnostics,
    total_cv,
    total_cv_ratio,
)
from sprotmetrics._panel import (
    dapi_positive_ratio,
    mutually_exclusive_coexpression_ratio,
    panel_consistency,
    replicate_correlation,
    replicate_cv,
    resolve_exclusive_pairs,
    resolve_nuclear_markers,
    segmentation_quality,
)
from sprotmetrics._pixelstats import CellPixelStats, cell_pixel_stats
from sprotmetrics._registry import (
    Direction,
    Inputs,
    MetricInfo,
    Reference,
    Status,
    Strength,
    Task,
    compute,
    get_metric,
    list_metrics,
    to_openproblems_info,
)
from sprotmetrics._signal import (
    cellular_specificity,
    intracell_coverage,
    marker_gini,
    nuclear_signal_containment,
    rel_sni,
    signal_background,
    signal_background_from_stats,
    signal_drop_on_aggregation,
    signal_to_background_ratio,
)
from sprotmetrics._spatial import resolve_coordinates, spatial_uniformity, spillover_ratio, tile_cv

__all__ = [
    "CellPixelStats",
    "Direction",
    "Inputs",
    "MetricInfo",
    "Reference",
    "Status",
    "Strength",
    "Task",
    "basic_statistics",
    "cell_area",
    "cell_pixel_stats",
    "cellular_specificity",
    "compute",
    "covariate_dependence",
    "covariate_pc_correlation",
    "dapi_positive_ratio",
    "get_metric",
    "intracell_coverage",
    "list_metrics",
    "marker_bimodality",
    "marker_coverage",
    "marker_cv",
    "marker_distribution",
    "marker_dynamic_range",
    "marker_gini",
    "marker_mean",
    "marker_table",
    "marker_variance_evenness",
    "marker_variance_ratio",
    "marker_zero_fraction",
    "mean_variance_correlation",
    "mutually_exclusive_coexpression_ratio",
    "n_cells",
    "n_markers",
    "normalization_diagnostics",
    "nuclear_signal_containment",
    "panel_consistency",
    "qc_pass_rate",
    "rel_sni",
    "replicate_correlation",
    "replicate_cv",
    "resolve_coordinates",
    "resolve_exclusive_pairs",
    "resolve_nuclear_markers",
    "segmentation_quality",
    "signal_background",
    "signal_background_from_stats",
    "signal_drop_on_aggregation",
    "signal_to_background_ratio",
    "spatial_uniformity",
    "spillover_ratio",
    "tile_cv",
    "to_openproblems_info",
    "total_cv",
    "total_cv_ratio",
    "total_intensity",
    "variance_explained_by_covariate",
]

__version__ = version("sprotmetrics")
