"""Metrics that need to know what the markers *are*.

Reference-free, but not assumption-free: they ask whether the data satisfies
properties it must have if cells were delineated correctly -- every object holds
a nucleus, nuclear stains repeated across imaging cycles agree within a cell,
incompatible lineage markers are rarely co-positive.

Panel knowledge is configuration, not a second dataset. When it is absent these
return NaN rather than guessing for an unknown panel.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from sprotmetrics._registry import Direction, Inputs, Reference, Strength, Task, metric
from sprotmetrics._typing import Mask, MatrixLike, Sequence
from sprotmetrics._utils import (
    DEFAULT_QC_COLUMN,
    marker_names,
    positive_mask,
    resolve_subset,
    to_matrix,
)

__all__ = [
    "dapi_positive_ratio",
    "mutually_exclusive_coexpression_ratio",
    "panel_consistency",
    "segmentation_quality",
    "replicate_correlation",
    "replicate_cv",
    "resolve_exclusive_pairs",
    "resolve_nuclear_markers",
]

FAMILY = "segmentation_quality"

#: Substrings that identify a nuclear channel when none is named explicitly.
NUCLEAR_HINTS = ("dna", "dapi", "hoechst", "syto")


def _split_csv(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if value.strip().lower() in {"", "null", "none"}:
            return []
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def resolve_nuclear_markers(markers: Sequence[str], requested: Sequence[str] | str | None = None) -> list[int]:
    """Indices of the nuclear channels, named explicitly or matched against the hint list.

    The default hints are ``dna``, ``dapi``, ``hoechst`` and ``syto``, matched
    case-insensitively as substrings of the marker name.
    """
    names = [str(m) for m in markers]
    wanted = _split_csv(requested)
    if wanted:
        return [names.index(w) for w in wanted if w in names]
    return [i for i, m in enumerate(names) if any(h in m.lower() for h in NUCLEAR_HINTS)]


def resolve_exclusive_pairs(
    markers: Sequence[str], requested: Sequence[tuple[str, str]] | str | None = None
) -> list[tuple[int, int]]:
    """Index pairs for mutually exclusive markers, skipping any not in the panel."""
    names = [str(m) for m in markers]
    pairs: list[tuple[int, int]] = []
    if requested is None:
        return pairs
    specs = _split_csv(requested) if isinstance(requested, str) else list(requested)
    for spec in specs:
        if isinstance(spec, str):
            if ":" not in spec:
                raise ValueError(f"exclusive pairs must look like 'A:B', got {spec!r}")
            a, b = (s.strip() for s in spec.split(":", 1))
        else:
            a, b = (str(s).strip() for s in spec)
        if a in names and b in names:
            pairs.append((names.index(a), names.index(b)))
    return pairs


def _matrix(data, layer, subset):
    X = to_matrix(data, layer, dtype=np.float64)
    mask = resolve_subset(data, subset)
    return X if mask is None else X[mask]


@metric(
    name="dapi_positive_ratio",
    label="Fraction of objects containing nuclear signal",
    family=FAMILY,
    task=(Task.SEGMENTATION,),
    reference=Reference.PANEL_KNOWLEDGE,
    inputs=Inputs.MATRIX,
    min=0,
    max=1,
    direction=Direction.HIGHER,
    origin="Metrics-2.xlsx row 8 (no prior code)",
)
def dapi_positive_ratio(
    data: MatrixLike,
    nuclear_markers: Sequence[str] | str | None = None,
    *,
    layer: str | None = "counts",
    subset: Mask = None,
) -> float:
    """Fraction of segmented objects positive for the averaged nuclear channels.

    Every segmented cell should contain a nucleus; objects that do not are debris
    or mask fragments.
    """
    X = _matrix(data, layer, subset)
    idx = resolve_nuclear_markers(marker_names(data, n_markers=X.shape[1]), nuclear_markers)
    if not idx or X.shape[0] == 0:
        return float("nan")
    return float(np.mean(positive_mask(X[:, idx].mean(axis=1))))


@metric(
    name="replicate_correlation",
    label="Agreement between repeated nuclear stains",
    family=FAMILY,
    task=(Task.SEGMENTATION, Task.NORMALIZATION),
    reference=Reference.PANEL_KNOWLEDGE,
    inputs=Inputs.MATRIX,
    min=-1,
    max=1,
    direction=Direction.HIGHER,
    strength=Strength.WEAK,
    origin="sprot protein_contribution.equivalent_proteins_correlation (adapted)",
    notes=(
        "Sum aggregation inflates this to ~0.999 because every channel then scales "
        "with cell area, so it is weak evidence on the aggregation axis."
    ),
)
def replicate_correlation(
    data: MatrixLike,
    nuclear_markers: Sequence[str] | str | None = None,
    *,
    layer: str | None = "counts",
    subset: Mask = None,
) -> float:
    """Mean pairwise correlation between nuclear stains repeated across imaging cycles.

    The same nuclei imaged twice should agree; disagreement means masks are
    picking up different amounts of each restain.
    """
    X = _matrix(data, layer, subset)
    idx = resolve_nuclear_markers(marker_names(data, n_markers=X.shape[1]), nuclear_markers)
    if len(idx) < 2 or X.shape[0] == 0:
        return float("nan")
    reps = X[:, idx]
    corrs = [
        float(np.corrcoef(reps[:, i], reps[:, j])[0, 1])
        for i, j in combinations(range(reps.shape[1]), 2)
        if np.std(reps[:, i]) > 0 and np.std(reps[:, j]) > 0
    ]
    return float(np.mean(corrs)) if corrs else float("nan")


@metric(
    name="replicate_cv",
    label="Per-cell spread across repeated nuclear stains",
    family=FAMILY,
    task=(Task.SEGMENTATION,),
    reference=Reference.PANEL_KNOWLEDGE,
    inputs=Inputs.MATRIX,
    min=0,
    max=np.inf,
    direction=Direction.LOWER,
    aggregations=("median",),
)
def replicate_cv(
    data: MatrixLike,
    nuclear_markers: Sequence[str] | str | None = None,
    *,
    layer: str | None = "counts",
    subset: Mask = None,
) -> np.ndarray:
    """Coefficient of variation across the repeated nuclear stains, per cell.

    The per-cell counterpart of :func:`replicate_correlation`: a mask that strays
    off its nucleus picks up different amounts of each restain.
    """
    X = _matrix(data, layer, subset)
    idx = resolve_nuclear_markers(marker_names(data, n_markers=X.shape[1]), nuclear_markers)
    if len(idx) < 2 or X.shape[0] == 0:
        return np.array([], dtype=float)
    reps = X[:, idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        means = reps.mean(axis=1)
        return np.where(means > 0, reps.std(axis=1) / np.maximum(means, 1e-12), np.nan)


@metric(
    name="mutually_exclusive_coexpression_ratio",
    label="Co-positivity of incompatible lineage markers",
    family=FAMILY,
    task=(Task.SEGMENTATION, Task.IMAGE_CORRECTION),
    reference=Reference.PANEL_KNOWLEDGE,
    inputs=Inputs.MATRIX,
    min=0,
    max=1,
    direction=Direction.LOWER,
    origin="Metrics-2.xlsx row 6 (no prior code)",
)
def mutually_exclusive_coexpression_ratio(
    data: MatrixLike,
    pairs: Sequence[tuple[str, str]] | str | None = None,
    *,
    layer: str | None = "counts",
    subset: Mask = None,
) -> float:
    """Cells co-positive for incompatible markers, over cells positive for either.

    Normalising by the union rather than by all cells stops the value shrinking
    simply because few cells express either marker. A merged mask spanning an
    epithelial and an immune cell shows up here.
    """
    X = _matrix(data, layer, subset)
    resolved = resolve_exclusive_pairs(marker_names(data, n_markers=X.shape[1]), pairs)
    if not resolved or X.shape[0] == 0:
        return float("nan")
    rates = []
    for i, j in resolved:
        pi, pj = positive_mask(X[:, i]), positive_mask(X[:, j])
        union = np.logical_or(pi, pj).sum()
        if union > 0:
            rates.append(float(np.logical_and(pi, pj).sum() / union))
    return float(np.mean(rates)) if rates else float("nan")


def panel_consistency(
    data: MatrixLike,
    *,
    layer: str | None = "counts",
    nuclear_markers: Sequence[str] | str | None = None,
    exclusive_pairs: Sequence[tuple[str, str]] | str | None = None,
    subset: Mask = None,
    qc_variants: bool | str = False,
    qc_column: str = DEFAULT_QC_COLUMN,
) -> dict[str, float]:
    """Panel-knowledge segmentation checks, reported together."""

    def once(rows: Mask) -> dict[str, float]:
        return {
            "dapi_positive_ratio": dapi_positive_ratio(data, nuclear_markers, layer=layer, subset=rows),
            "replicate_correlation": replicate_correlation(data, nuclear_markers, layer=layer, subset=rows),
            "median_replicate_cv": _median(replicate_cv(data, nuclear_markers, layer=layer, subset=rows)),
            "mutually_exclusive_coexpression_ratio": mutually_exclusive_coexpression_ratio(
                data, exclusive_pairs, layer=layer, subset=rows
            ),
        }

    out = once(subset)
    if not qc_variants:
        return out
    column = qc_variants if isinstance(qc_variants, str) else qc_column
    passed = resolve_subset(data, column)
    rows = resolve_subset(data, subset)
    if passed is not None and rows is not None:
        passed = passed & rows
    if passed is not None and passed.any():
        over_qc = once(passed)
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


def _median(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmedian(arr)) if arr.size and np.isfinite(arr).any() else float("nan")


def segmentation_quality(
    data: MatrixLike,
    *,
    layer: str | None = "counts",
    nuclear_markers: Sequence[str] | str | None = None,
    exclusive_pairs: Sequence[tuple[str, str]] | str | None = None,
    coordinates: str | None = "spatial",
    n_neighbors: int = 6,
    max_cells: int = 5000,
    random_state: int = 0,
    subset: Mask = None,
    qc_variants: bool | str = False,
    qc_column: str = DEFAULT_QC_COLUMN,
) -> dict[str, float]:
    """Reference-free segmentation quality: the benchmark's five ids, in its order.

    Combines the panel-knowledge checks here with
    :func:`sprotmetrics.spillover_ratio`, which needs centroids rather than panel
    knowledge.
    """
    from sprotmetrics._spatial import spillover_ratio

    def once(rows: Mask) -> dict[str, float]:
        return {
            "dapi_positive_ratio": dapi_positive_ratio(data, nuclear_markers, layer=layer, subset=rows),
            "replicate_correlation": replicate_correlation(data, nuclear_markers, layer=layer, subset=rows),
            "median_replicate_cv": _median(replicate_cv(data, nuclear_markers, layer=layer, subset=rows)),
            "mutually_exclusive_coexpression_ratio": mutually_exclusive_coexpression_ratio(
                data, exclusive_pairs, layer=layer, subset=rows
            ),
            "spillover_ratio": spillover_ratio(
                data,
                coordinates,
                layer=layer,
                n_neighbors=n_neighbors,
                max_cells=max_cells,
                random_state=random_state,
                subset=rows,
            ),
        }

    out = once(subset)
    if not qc_variants:
        return out
    column = qc_variants if isinstance(qc_variants, str) else qc_column
    passed = resolve_subset(data, column)
    rows = resolve_subset(data, subset)
    if passed is not None and rows is not None:
        passed = passed & rows
    if passed is not None and passed.any():
        over_qc = once(passed)
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
