"""Subcellular localisation: does each marker sit where it is supposed to?

The one metric family needing a reference outside the sample. The reference is
*not* bundled: the Human Protein Atlas is a licensed external resource, and
mapping its labels onto a particular antibody panel is panel-specific work a
package cannot do blindly. :func:`localisation_agreement` therefore takes the
expected localisations as an argument, and :func:`nuclear_enrichment` -- the
measurement it scores -- is useful on its own without any atlas at all.
"""

from __future__ import annotations

import numpy as np

from sprotmetrics._pixelstats import CellPixelStats
from sprotmetrics._registry import Direction, Inputs, Reference, Status, Task, metric
from sprotmetrics._typing import Mapping, Sequence

__all__ = ["localisation_agreement", "nuclear_enrichment"]

FAMILY = "marker_characterisation"


@metric(
    name="nuclear_enrichment",
    label="Nuclear over cytoplasmic intensity",
    family=FAMILY,
    task=(Task.MARKER_CHARACTERISATION,),
    reference=Reference.NONE,
    inputs=Inputs.IMAGE_LABELS,
    min=0,
    max=np.inf,
    direction=Direction.NONE,
    aggregations=("median",),
    status=Status.EXPERIMENTAL,
    qc_variant=False,
)
def nuclear_enrichment(
    image: np.ndarray,
    cell_labels: np.ndarray,
    nuclear_labels: np.ndarray,
    markers: Sequence[str] | None = None,
) -> np.ndarray:
    """Mean intensity inside nuclei over mean intensity in the rest of the cell, per marker.

    Above 1 is a nuclear marker, below 1 a membrane or cytoplasmic one.
    Descriptive on its own -- which pattern is *correct* depends on the protein,
    which is what :func:`localisation_agreement` brings in.
    """
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[None, :, :]
    cell = np.squeeze(np.asarray(cell_labels)) > 0
    nucleus = np.squeeze(np.asarray(nuclear_labels)) > 0
    if image.shape[1:] != cell.shape or cell.shape != nucleus.shape:
        raise ValueError("image, cell_labels and nuclear_labels must share spatial shape")

    inside_nucleus = cell & nucleus
    cytoplasm = cell & ~nucleus
    out = np.full(image.shape[0], np.nan)
    if not inside_nucleus.any() or not cytoplasm.any():
        return out
    for c in range(image.shape[0]):
        cyto_mean = float(image[c][cytoplasm].astype(float).mean())
        if cyto_mean > 0:
            out[c] = float(image[c][inside_nucleus].astype(float).mean()) / cyto_mean
    return out


def localisation_agreement(
    enrichment: np.ndarray | CellPixelStats,
    expected: Mapping[str, str],
    markers: Sequence[str] | None = None,
    *,
    nuclear_threshold: float = 1.2,
    cytoplasmic_threshold: float = 0.8,
) -> float:
    """Fraction of markers whose observed localisation matches the expected one.

    Parameters
    ----------
    enrichment
        Per-marker nuclear-over-cytoplasmic ratios from :func:`nuclear_enrichment`.
    expected
        Marker name to expected compartment, one of ``"nuclear"``,
        ``"cytoplasmic"`` or ``"membrane"``. Supply this from the Human Protein
        Atlas or from your own panel annotation; markers absent from it are
        skipped rather than guessed at.
    nuclear_threshold, cytoplasmic_threshold
        Ratios above and below which a marker is called nuclear or
        cytoplasmic. Between them it is called ambiguous and counts as a
        mismatch, so a preprocessing choice that blurs the compartments is
        penalised rather than scoring by luck.
    """
    values = np.asarray(enrichment, dtype=float)
    names = list(markers) if markers is not None else None
    if names is None:
        raise ValueError("markers is required to match enrichment values to expected labels")
    if len(names) != values.size:
        raise ValueError(f"got {len(names)} marker names for {values.size} enrichment values")

    hits = total = 0
    for name, value in zip(names, values, strict=True):
        want = expected.get(name)
        if want is None or not np.isfinite(value):
            continue
        total += 1
        if value >= nuclear_threshold:
            observed = "nuclear"
        elif value <= cytoplasmic_threshold:
            observed = "cytoplasmic"
        else:
            observed = "ambiguous"
        # Membrane and cytoplasmic are both "not nuclear" at this resolution:
        # separating them needs a membrane mask the pipeline does not produce.
        if observed == want or (observed == "cytoplasmic" and want == "membrane"):
            hits += 1
    return float(hits / total) if total else float("nan")
