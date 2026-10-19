"""Metrics that need cell positions rather than panel knowledge.

Both ask whether intensity varies with where a cell sits rather than with what it
is: :func:`spillover_ratio` over a few microns, :func:`tile_cv` over the whole
slide. They live together because they share an input, not because they share a
pipeline stage.
"""

from __future__ import annotations

import numpy as np

from sprotmetrics._registry import Direction, Inputs, Reference, Status, Task, metric
from sprotmetrics._typing import Mask, MatrixLike, VectorLike
from sprotmetrics._utils import DEFAULT_QC_COLUMN, positive_mask, resolve_subset, to_matrix

__all__ = ["resolve_coordinates", "spatial_uniformity", "spillover_ratio", "tile_cv"]

FAMILY = "segmentation_quality"


def resolve_coordinates(data: MatrixLike, coordinates: VectorLike | None = "spatial") -> np.ndarray | None:
    """Per-cell coordinates from ``obsm``, else ``obs['centroid_y'/'centroid_x']``."""
    if coordinates is not None and not isinstance(coordinates, str):
        return np.asarray(coordinates, dtype=float)
    if not hasattr(data, "obs"):
        return None
    key = coordinates or "spatial"
    obsm = getattr(data, "obsm", {})
    if key in obsm:
        return np.asarray(obsm[key], dtype=float)
    if {"centroid_y", "centroid_x"}.issubset(data.obs.columns):
        return data.obs[["centroid_y", "centroid_x"]].to_numpy(dtype=float)
    return None


def _matrix(data, layer, subset):
    X = to_matrix(data, layer, dtype=np.float64)
    mask = resolve_subset(data, subset)
    return (X, mask) if mask is None else (X[mask], mask)


@metric(
    name="spillover_ratio",
    label="Signal bleed into neighbouring cells",
    family=FAMILY,
    task=(Task.SEGMENTATION,),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX_OBS,
    min=0,
    max=np.inf,
    direction=Direction.LOWER,
    origin="Metrics-2.xlsx row 7 (no prior code)",
    notes="1.0 means no spillover. Needs roughly 200 cells before it is defined.",
)
def spillover_ratio(
    data: MatrixLike,
    coordinates: VectorLike | None = "spatial",
    *,
    layer: str | None = "counts",
    n_neighbors: int = 6,
    max_cells: int = 5000,
    random_state: int = 0,
    subset: Mask = None,
) -> float:
    """Intensity of negative cells touching a positive cell, over those that do not.

    A marker positive in one cell should be absent from its neighbours. Masks
    that over-expand across a boundary raise this above 1.
    """
    from scipy.spatial import cKDTree

    X, mask = _matrix(data, layer, subset)
    coords = resolve_coordinates(data, coordinates)
    if coords is None or X.shape[0] == 0:
        return float("nan")
    if mask is not None:
        coords = coords[mask]
    n_cells = X.shape[0]
    if n_cells <= n_neighbors + 1:
        return float("nan")

    idx = np.arange(n_cells)
    if max_cells > 0 and n_cells > max_cells:
        idx = np.sort(np.random.default_rng(random_state).choice(n_cells, size=max_cells, replace=False))
    sub_coords, sub_X = coords[idx], X[idx]
    tree = cKDTree(sub_coords)
    # +1 because the first neighbour a tree returns is the cell itself.
    _, nn = tree.query(sub_coords, k=min(n_neighbors + 1, len(idx)))
    nn = np.atleast_2d(nn)[:, 1:]

    ratios = []
    for c in range(sub_X.shape[1]):
        col = sub_X[:, c]
        pos = positive_mask(col)
        neg = ~pos
        if pos.sum() == 0 or neg.sum() == 0:
            continue
        touches_pos = pos[nn].any(axis=1)
        near, far = neg & touches_pos, neg & ~touches_pos
        if near.sum() == 0 or far.sum() == 0:
            continue
        far_mean = float(np.mean(col[far]))
        if far_mean > 0:
            ratios.append(float(np.mean(col[near]) / far_mean))
    return float(np.median(ratios)) if ratios else float("nan")


@metric(
    name="tile_cv",
    label="Variation of marker intensity across the slide",
    family=FAMILY,
    task=(Task.IMAGE_CORRECTION,),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX_OBS,
    min=0,
    max=np.inf,
    direction=Direction.LOWER,
    aggregations=("median", "max"),
    status=Status.EXPERIMENTAL,
    origin="Metrics-2.xlsx row 10",
    notes="Targets flat-field correction, which nothing in the benchmark could see.",
)
def tile_cv(
    data: MatrixLike,
    coordinates: VectorLike | None = "spatial",
    *,
    layer: str | None = "counts",
    n_tiles: int | tuple[int, int] = 4,
    min_cells_per_tile: int = 20,
    subset: Mask = None,
) -> np.ndarray:
    """Coefficient of variation of a marker's mean across a grid of tiles, per marker.

    Illumination shading and stitching seams make one region of the slide
    systematically brighter. Real tissue is spatially structured too, so a high
    value is suggestive rather than conclusive -- but a *fall* after flat-field
    correction is evidence the correction worked.
    """
    X, mask = _matrix(data, layer, subset)
    coords = resolve_coordinates(data, coordinates)
    if coords is None or X.shape[0] == 0:
        return np.array([], dtype=float)
    if mask is not None:
        coords = coords[mask]

    ny, nx = (n_tiles, n_tiles) if isinstance(n_tiles, int) else n_tiles
    edges = []
    for axis, count in ((0, ny), (1, nx)):
        values = coords[:, axis]
        lo, hi = float(values.min()), float(values.max())
        span = hi - lo
        if span <= 0:
            return np.array([], dtype=float)
        # The top edge is nudged outward so the maximum lands inside the last bin.
        edges.append(np.digitize(values, np.linspace(lo, hi, count + 1)[1:-1]))
    tile = edges[0] * nx + edges[1]

    means = []
    for t in np.unique(tile):
        rows = tile == t
        if rows.sum() >= min_cells_per_tile:
            means.append(X[rows].mean(axis=0))
    if len(means) < 2:
        return np.array([], dtype=float)
    stacked = np.vstack(means)
    panel_mean = stacked.mean(axis=0)
    return np.where(panel_mean > 0, stacked.std(axis=0) / np.maximum(panel_mean, 1e-12), np.nan)


def spatial_uniformity(
    data: MatrixLike,
    coordinates: VectorLike | None = "spatial",
    *,
    layer: str | None = "counts",
    n_neighbors: int = 6,
    n_tiles: int | tuple[int, int] = 4,
    max_cells: int = 5000,
    random_state: int = 0,
    subset: Mask = None,
    qc_variants: bool | str = False,
    qc_column: str = DEFAULT_QC_COLUMN,
) -> dict[str, float]:
    """Position-dependent intensity checks, reported together."""

    def once(rows: Mask) -> dict[str, float]:
        cvs = tile_cv(data, coordinates, layer=layer, n_tiles=n_tiles, subset=rows)
        finite = cvs[np.isfinite(cvs)] if cvs.size else cvs
        return {
            "spillover_ratio": spillover_ratio(
                data,
                coordinates,
                layer=layer,
                n_neighbors=n_neighbors,
                max_cells=max_cells,
                random_state=random_state,
                subset=rows,
            ),
            "median_tile_cv": float(np.median(finite)) if finite.size else float("nan"),
            "max_tile_cv": float(np.max(finite)) if finite.size else float("nan"),
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
