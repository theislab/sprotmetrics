
"""Metrics for matching cell and nuclear instance segmentations."""

from __future__ import annotations

import numpy as np
from skimage.segmentation import find_boundaries

from sprotmetrics._registry import Direction, Inputs, Reference, Task, metric

__all__ = ["fraction_of_matched_cells_and_nuclei"]

FAMILY = "segmentation_quality"


def _as_labels(labels: np.ndarray, name: str) -> np.ndarray:
    """Validate and normalize a 2D instance-label mask."""
    labels = np.asarray(labels)

    if labels.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")

    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"{name} must contain integer labels.")

    if np.any(labels < 0):
        raise ValueError(f"{name} must contain non-negative labels.")

    return labels.astype(np.int64, copy=False)


@metric(
    name="fraction_of_matched_cells_and_nuclei",
    label="Fraction of matched cells and nuclei",
    family=FAMILY,
    task=(Task.SEGMENTATION,),
    reference=Reference.NONE,
    inputs=Inputs.CELL_NUCLEI_LABELS,
    min=0,
    max=1,
    direction=Direction.HIGHER,
    qc_variant=False,
    origin="CellSegmentationEvaluator",
)
def fraction_of_matched_cells_and_nuclei(
    cell_labels: np.ndarray,
    nuclear_labels: np.ndarray,
) -> float:
    """Fraction of cells and nuclei that can be matched one-to-one.

    A nucleus is considered a candidate for a cell if it overlaps the cell
    interior, defined as the cell mask excluding its inner boundary. Each
    cell is matched to the available nucleus with the largest fraction of
    nuclear pixels inside that interior.

    Matching is greedy in ascending cell-label order, and each object can
    participate in at most one match.
    """
    cells = _as_labels(cell_labels, "cell_labels")
    nuclei = _as_labels(nuclear_labels, "nuclear_labels")

    if cells.shape != nuclei.shape:
        raise ValueError("cell_labels and nuclear_labels must have the same shape.")

    cell_ids = np.unique(cells)
    cell_ids = cell_ids[cell_ids > 0]

    nuclear_ids = np.unique(nuclei)
    nuclear_ids = nuclear_ids[nuclear_ids > 0]

    n_cells = len(cell_ids)
    n_nuclei = len(nuclear_ids)

    n_objects = n_cells + n_nuclei
    if n_objects == 0:
        return float("nan")

    # Exclude the inner boundary of each cell from its interior.
    boundaries = find_boundaries(cells, mode="inner")
    cell_interiors = (cells > 0) & ~boundaries

    nuclear_areas = {
        int(nucleus_id): int(np.count_nonzero(nuclei == nucleus_id))
        for nucleus_id in nuclear_ids
    }

    matched_nuclei: set[int] = set()
    n_matched = 0

    for cell_id in cell_ids:
        interior = cell_interiors & (cells == cell_id)

        if not np.any(interior):
            continue

        overlapping_nuclei = nuclei[interior]
        candidate_ids, overlap_counts = np.unique(
            overlapping_nuclei[overlapping_nuclei > 0],
            return_counts=True,
        )

        candidates = [
            (int(nucleus_id), int(overlap_count))
            for nucleus_id, overlap_count in zip(candidate_ids, overlap_counts)
            if int(nucleus_id) not in matched_nuclei
        ]

        if not candidates:
            continue

        # Maximize the fraction of the nucleus contained in the cell interior.
        best_nucleus, _ = max(
            candidates,
            key=lambda item: item[1] / nuclear_areas[item[0]],
        )

        matched_nuclei.add(best_nucleus)
        n_matched += 1

    n_unmatched_cells = n_cells - n_matched
    n_unmatched_nuclei = n_nuclei - n_matched

    denominator = n_matched + n_unmatched_cells + n_unmatched_nuclei
    return float(n_matched / denominator)