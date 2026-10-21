"""Generate the metric reference page from the registry at build time.

The catalogue already records what each metric measures, its range, which
pipeline stage it scores and whether a larger value is better. Writing that out
again by hand would guarantee the page and the code disagree within a release,
so the page is generated and never committed.

Figures are shared between metrics: several of them rest on one idea -- a
data-driven positivity split, a ring of local background -- and share the
picture that explains it. :data:`FIGURES` maps a quantity function to that
picture, and the same figure may appear under several metrics.
"""

from __future__ import annotations

import pathlib
import textwrap
from typing import Any

#: quantity function -> (figure stem, caption)
FIGURES: dict[str, tuple[str, str]] = {
    "spillover_ratio": ("spillover", "Signal crossing a cell boundary."),
    "mutually_exclusive_coexpression_ratio": (
        "exclusive-coexpression",
        "A mask spanning two cells is positive for both lineages.",
    ),
    "nuclear_signal_containment": (
        "nuclear-containment",
        "Containment rises with mask size, whatever the mask contains.",
    ),
    "object_f1": ("object-matching", "Agreement against reference masks."),
    "foreground_agreement": ("object-matching", "Agreement against reference masks."),
    "replicate_correlation": (
        "replicate-concordance",
        "The same nucleus, imaged in two cycles.",
    ),
    "replicate_cv": ("replicate-concordance", "The same nucleus, imaged in two cycles."),
    "tile_cv": ("tile-cv", "Brightness that depends on position on the slide."),
    "dapi_positive_ratio": ("otsu-positivity", "Positivity comes from the data."),
    "marker_bimodality": ("otsu-positivity", "Positivity comes from the data."),
    "signal_to_background_ratio": ("mask-and-ring", "Background is sampled locally."),
    "cellular_specificity": ("mask-and-ring", "Background is sampled locally."),
    "rel_sni": ("rank-matched-curves", "The signal integral is an area between curves."),
    "intracell_coverage": (
        "intracell-coverage",
        "Bright and extensive, against bright and sparse.",
    ),
    "signal_drop_on_aggregation": (
        "signal-drop",
        "Pixels that were positive, summarised to nothing.",
    ),
    "nuclear_enrichment": ("nuclear-enrichment", "Which compartment holds the signal."),
    "localisation_agreement": ("nuclear-enrichment", "Which compartment holds the signal."),
    "marker_coverage": ("marker-coverage", "The fraction of cells carrying a marker."),
    "marker_zero_fraction": ("marker-coverage", "The fraction of cells carrying a marker."),
    "variance_explained_by_covariate": (
        "area-dependence",
        "Expression that tracks cell size rather than biology.",
    ),
    "covariate_pc_correlation": (
        "area-dependence",
        "Expression that tracks cell size rather than biology.",
    ),
    "marker_variance_evenness": (
        "variance-evenness",
        "Whether a few markers own all the variance.",
    ),
    "total_cv_ratio": ("normalization-spread", "What a per-cell normalization does."),
    "total_cv": ("normalization-spread", "What a per-cell normalization does."),
    "silhouette_cell_type": ("cluster-separability", "Do known populations sit apart?"),
    "clustering_stability": ("cluster-separability", "Do known populations sit apart?"),
    "marker_gini": ("marker-gini", "Concentration, which is descriptive rather than good."),
}

#: Family order on the page, and the heading each gets.
FAMILIES: tuple[tuple[str, str], ...] = (
    ("segmentation_quality", "Segmentation"),
    ("segmentation_reference", "Segmentation against reference masks"),
    ("signal_background", "Signal and background"),
    ("marker_distribution", "Marker distribution"),
    ("normalization_diagnostics", "Normalization"),
    ("technical_covariates", "Technical covariates"),
    ("cluster_separability", "Cell populations"),
    ("marker_characterisation", "Marker characterisation"),
    ("basic", "Dataset and object statistics"),
)

DIRECTION_WORDS = {
    "higher is better": "higher is better",
    "lower is better": "lower is better",
    "descriptive": "descriptive — no preferred value",
    "ambiguous": "ambiguous — direction is unsafe",
}


def _fmt_bound(value: float) -> str:
    import numpy as np

    if value == np.inf:
        return "∞"
    if value == -np.inf:
        return "−∞"
    if value != value:
        return "—"
    return f"{value:g}"


def _metric_block(info: Any, ids: list[str]) -> list[str]:
    """Markdown for one quantity function and every metric id it produces."""
    lines = [f"### `{info.func.__name__}`", "", f"**{info.label}.**", ""]

    description = (info.description or "").strip()
    if description:
        # The first line is the summary; the rest is indented to the docstring's
        # body, which Markdown would otherwise read as a code block.
        first, _, rest = description.partition("\n")
        lines += [first.strip(), ""]
        rest = textwrap.dedent(rest).strip()
        if rest:
            lines += [rest, ""]

    figure = FIGURES.get(info.func.__name__)
    if figure:
        stem, caption = figure
        lines += [
            "```{figure} /_static/metrics/" + stem + ".svg",
            ":alt: " + caption,
            ":width: 100%",
            "",
            caption,
            "```",
            "",
        ]

    strength = "" if info.strength.value == "strong" else f", {info.strength.value} evidence"
    lines += [
        "| | |",
        "| --- | --- |",
        f"| Direction | {DIRECTION_WORDS[info.direction.value]}{strength} |",
        f"| Range | {_fmt_bound(info.min)} to {_fmt_bound(info.max)} |",
        f"| Scores | {', '.join(t.value for t in info.task)} |",
        f"| Needs | {info.reference.value} |",
        f"| Metric ids | {', '.join('`' + i + '`' for i in ids)} |",
    ]
    if info.status.value != "stable":
        lines.append(f"| Status | {info.status.value} |")
    if info.references:
        lines.append("| References | " + ", ".join(f"[{d}](https://doi.org/{d})" for d in info.references) + " |")
    lines.append("")

    if info.notes:
        lines += ["```{warning}" if info.strength.value == "weak" else "```{note}", info.notes, "```", ""]
    return lines


def build_page() -> str:
    """Render the whole reference page."""
    import sprotmetrics as spm

    catalogue = spm.list_metrics(include_deprecated=True)
    by_function: dict[str, list[str]] = {}
    infos: dict[str, Any] = {}
    for metric_id in catalogue.index:
        info = spm.get_metric(metric_id)
        if info.func is None:
            continue
        key = info.func.__name__
        by_function.setdefault(key, []).append(metric_id)
        infos[key] = info

    out = [
        "# Metric reference",
        "",
        "Every metric this package computes, what it measures, and how to read it.",
        "",
        "The same information is available programmatically through",
        "{func}`~sprotmetrics.list_metrics`, which is what the benchmark uses; this",
        "page is generated from it, so the two cannot drift apart.",
        "",
        "Several metrics share a figure, because several of them rest on one idea.",
        "",
    ]

    seen: set[str] = set()
    for family, heading in FAMILIES:
        keys = [k for k, i in infos.items() if i.family == family]
        if not keys:
            continue
        out += [f"## {heading}", ""]
        for key in sorted(keys, key=lambda k: infos[k].label):
            out += _metric_block(infos[key], by_function[key])
            seen.add(key)

    leftover = [k for k in infos if k not in seen]
    if leftover:
        out += ["## Other", ""]
        for key in sorted(leftover):
            out += _metric_block(infos[key], by_function[key])

    retired = catalogue[catalogue["status"] == "deprecated"]
    if len(retired):
        out += [
            "## Recorded but not computed",
            "",
            "Metrics considered and rejected, kept queryable so the reasoning does not",
            "have to be rediscovered. Asking for one raises an error naming its",
            "replacement.",
            "",
            "| Metric | Instead use | Why |",
            "| --- | --- | --- |",
        ]
        for metric_id, row in retired.iterrows():
            replacement = f"`{row['superseded_by']}`" if row["superseded_by"] else "—"
            out.append(f"| `{metric_id}` | {replacement} | {row['notes']} |")
        out.append("")

    return "\n".join(out)


def _on_builder_inited(app) -> None:
    path = pathlib.Path(app.srcdir) / "metrics.md"
    path.write_text(build_page())


def setup(app):
    """Register the generator."""
    app.connect("builder-inited", _on_builder_inited)
    return {"parallel_read_safe": True, "version": "1.0"}
