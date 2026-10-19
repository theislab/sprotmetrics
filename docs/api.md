# API

```{eval-rst}
.. module:: sprotmetrics
.. currentmodule:: sprotmetrics
```

Three levels, and most analysis wants the first.

**Quantities** are one function per scientific quantity, vector-valued where the
quantity is per cell or per marker. `marker_coverage(adata)` returns one number
*per marker*, so you can see which marker is dragging the panel down rather than
only that it is down.

**Metric ids** apply the panel aggregation the catalogue declares:
`compute("median_marker_coverage", adata)` is a single number, and
`compute("min_marker_coverage", adata)` is the worst marker.

**Families** return every id a benchmark component emits, including the
`_qc_pass` variants, and are what a pipeline step should call.

## The catalogue

Every metric records what it measures, its range, and whether a larger value is
better. `maximize` is `None` wherever ranking on the metric would be unsound —
a saturated detection rate, or a spread that could equally be biology or noise.

```{eval-rst}
.. autosummary::
    :toctree: generated

    list_metrics
    get_metric
    compute
    to_openproblems_info
    MetricInfo
    Task
    Reference
    Inputs
    Direction
    Strength
    Status
```

## Families

```{eval-rst}
.. autosummary::
    :toctree: generated

    basic_statistics
    marker_distribution
    normalization_diagnostics
    covariate_dependence
    segmentation_quality
    panel_consistency
    spatial_uniformity
    cluster_separability
    signal_background
    segmentation_agreement
```

## Segmentation

Reference-free unless noted: these ask whether the data satisfies properties it
must have if cells were delineated correctly.

```{eval-rst}
.. autosummary::
    :toctree: generated

    dapi_positive_ratio
    replicate_correlation
    replicate_cv
    mutually_exclusive_coexpression_ratio
    spillover_ratio
    nuclear_signal_containment
```

Against reference masks, when you have them:

```{eval-rst}
.. autosummary::
    :toctree: generated

    foreground_agreement
    object_f1
    detection_table
    object_overlaps
    match_objects
    split_merge_stats
    label_areas
```

## Image correction and signal quality

Computed from pixels rather than from the cell table, and so the only metrics
that can see signal which never entered a mask.

```{eval-rst}
.. autosummary::
    :toctree: generated

    cell_pixel_stats
    CellPixelStats
    rel_sni
    signal_to_background_ratio
    cellular_specificity
    intracell_coverage
    marker_gini
    tile_cv
```

## Marker distribution and normalization

```{eval-rst}
.. autosummary::
    :toctree: generated

    marker_table
    marker_mean
    marker_cv
    marker_coverage
    marker_zero_fraction
    marker_dynamic_range
    marker_bimodality
    total_cv
    total_cv_ratio
    marker_variance_ratio
    mean_variance_correlation
    marker_variance_evenness
```

## Aggregation and technical covariates

```{eval-rst}
.. autosummary::
    :toctree: generated

    total_intensity
    cell_area
    variance_explained_by_covariate
    covariate_pc_correlation
    signal_drop_on_aggregation
```

## Cell populations

```{eval-rst}
.. autosummary::
    :toctree: generated

    silhouette_cell_type
    clustering_stability
    nuclear_enrichment
    localisation_agreement
```

## Dataset statistics

```{eval-rst}
.. autosummary::
    :toctree: generated

    n_cells
    n_markers
    qc_pass_rate
```

## Panel configuration

Panel knowledge is configuration, not a second dataset. Supply it and the
metrics that need it work; omit it and they return `NaN` rather than guessing.

```{eval-rst}
.. autosummary::
    :toctree: generated

    resolve_nuclear_markers
    resolve_exclusive_pairs
    resolve_coordinates
```

## SpatialData

Thin wrappers that resolve element names and delegate. `spatialdata` is an
optional dependency: `pip install sprotmetrics[spatialdata]`.

```{eval-rst}
.. currentmodule:: sprotmetrics.sd
.. autosummary::
    :toctree: generated

    cell_pixel_stats
    signal_background
    segmentation_agreement
    compute_all
```
