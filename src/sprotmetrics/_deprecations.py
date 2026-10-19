"""Metrics this package deliberately does not compute, and why.

Registering them keeps the reasoning queryable: "why is there no SNR metric?" is
answered by ``list_metrics(include_deprecated=True)`` rather than by reading a
spreadsheet or rediscovering the algebra. :func:`sprotmetrics.compute` raises for
these and names the replacement.

The ``superseded_by`` entries are redundancies found while cataloguing the ISP
benchmark; the rest are metrics whose published definition does not transfer.
"""

from __future__ import annotations

from sprotmetrics._registry import (
    Direction,
    Inputs,
    Reference,
    Status,
    Task,
    register_placeholder,
)

register_placeholder(
    metric_id="snr_per_protein",
    label="Per-marker signal-to-noise ratio",
    summary="Mean over standard deviation, per marker.",
    family="marker_distribution",
    task=(Task.NORMALIZATION,),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX,
    superseded_by="median_marker_cv",
    origin="sprot protein_properties.snr_per_protein",
    notes=(
        "Exactly 1 / cv, so it carries no information beyond median_marker_cv. "
        "Reporting both would double-count one quantity in any aggregate score."
    ),
)

register_placeholder(
    metric_id="cv_per_protein",
    label="Per-marker coefficient of variation",
    summary="Standard deviation over mean, per marker.",
    family="marker_distribution",
    task=(Task.NORMALIZATION,),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX,
    superseded_by="median_marker_cv",
    origin="sprot protein_properties.cv_per_protein",
    notes="The same quantity under another name.",
)

register_placeholder(
    metric_id="dynamic_range_per_protein",
    label="Per-marker dynamic range",
    summary="Maximum minus minimum intensity, per marker.",
    family="marker_distribution",
    task=(Task.IMAGE_CORRECTION,),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX,
    superseded_by="median_marker_dynamic_range",
    origin="sprot protein_properties.dynamic_range_per_protein",
    notes=(
        "max - min is defined by two pixels and so is set by hot pixels. The "
        "replacement uses a q99 - q01 spread, which is robust to them."
    ),
)

register_placeholder(
    metric_id="sparsity_per_protein",
    label="Per-marker fraction of exact zeros",
    summary="Fraction of cells at exactly zero, per marker.",
    family="marker_distribution",
    task=(Task.IMAGE_CORRECTION,),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX,
    superseded_by="median_marker_zero_fraction",
    origin="sprot protein_properties.sparsity_per_protein",
    notes=(
        "Exact zeros are rare in fluorescence intensities, so this reads ~0 for "
        "every method. The replacement takes a threshold."
    ),
)

register_placeholder(
    metric_id="gene_pair_cosine_similarity",
    label="Similarity between two aggregation methods' outputs",
    summary="Cosine similarity between the matrices two runs produce.",
    family="aggregation",
    task=(Task.AGGREGATION,),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX,
    origin="sprot aggregating/compute.py",
    notes=(
        "Compares two runs against each other rather than scoring one, so it "
        "cannot be a metric in a benchmark that evaluates runs independently. "
        "It needs a cross-run comparison stage, which is a workflow change."
    ),
)

register_placeholder(
    metric_id="variance_explained_by_total_counts",
    label="Variance explained by single-cell technical covariates",
    summary="R^2 of expression on total counts, percent mitochondrial and batch.",
    family="technical_covariates",
    task=(Task.NORMALIZATION,),
    reference=Reference.REFERENCE_SCRNA,
    inputs=Inputs.MATRIX_OBS,
    superseded_by="mean_variance_explained_by_covariate_counts",
    origin="sprot technical_correction.variance_explained_by_covariates",
    notes=(
        "Those covariates do not exist in imaging data: there is no library "
        "size and no mitochondrial fraction. Cell area is the analogous "
        "technical driver and is what the replacement regresses on."
    ),
)

register_placeholder(
    metric_id="hpa_localisation_agreement",
    label="Agreement with Human Protein Atlas subcellular priors",
    summary="Match between the observed subcellular pattern and an external atlas.",
    family="marker_characterisation",
    task=(Task.MARKER_CHARACTERISATION,),
    reference=Reference.EXTERNAL_ATLAS,
    inputs=Inputs.IMAGE_LABELS,
    direction=Direction.HIGHER,
    min=0,
    max=1,
    status=Status.EXPERIMENTAL,
    origin="Metrics-2.xlsx quality_metrics row 7",
    notes=(
        "Computable with nuclear_enrichment, which this package provides, plus "
        "a table of expected localisations. The atlas itself is not bundled: it "
        "is a licensed external resource and the mapping from its labels to a "
        "given antibody panel is panel-specific. Supply priors to "
        "localisation_agreement()."
    ),
)
