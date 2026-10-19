"""One pass over the image, reused by every pixel-level metric.

Portions adapted from theislab/sprot_best_practices (BSD-3-Clause,
Copyright (c) 2023 Theis Lab); see NOTICE and
LICENSES/BSD-3-Clause-theislab-sprot_best_practices.txt.

Building these statistics is O(C x Y x X) with a per-channel Otsu. On a
3194 x 2537 mosaic that is the whole cost of the family, so it happens once and
every metric in :mod:`sprotmetrics._signal` is a cheap reduction over the result.
Users holding one image and wanting several metrics should build it themselves
and pass it around.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from sprotmetrics._typing import Mask, Sequence
from sprotmetrics._utils import otsu_threshold

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["CellPixelStats", "cell_pixel_stats"]


@dataclass(frozen=True, slots=True)
class CellPixelStats:
    """Per-cell and per-channel pixel statistics, plus the proximal-background sample."""

    markers: tuple[str, ...]
    areas: np.ndarray
    means: np.ndarray
    prop_positive: np.ndarray
    thresholds: np.ndarray
    present: np.ndarray
    inside_sum: np.ndarray
    total_sum: np.ndarray
    inside_mean: np.ndarray
    inside_positive_fraction: np.ndarray
    ring_mean: np.ndarray
    ring_positive_fraction: np.ndarray
    background_curves: tuple[np.ndarray | None, ...]
    n_background_pseudocells: int
    image_shape: tuple[int, ...]
    #: dtype the image was stored in. Multiplexed images are written float32, and
    #: the signal-to-background ratio is formed at that precision rather than in
    #: float64 (see :func:`sprotmetrics.signal_to_background_ratio`).
    intensity_dtype: np.dtype
    params: Mapping[str, Any]
    cell_subset: np.ndarray | None = None

    @property
    def n_objects(self) -> int:
        """Number of labelled objects currently selected."""
        return int(self.selection.sum())

    @property
    def selection(self) -> np.ndarray:
        """Boolean mask of the cells this view covers."""
        return self.present if self.cell_subset is None else (self.cell_subset & self.present)

    def subset(self, cells: Mask) -> CellPixelStats:
        """Restrict to `cells`, keeping every image-level quantity unchanged.

        Thresholds, ring statistics and the background curves deliberately do not
        move. A QC filter removes cells, not pixels: the background a cell is
        compared against is a property of the image, and recomputing it over a
        subset would make the filtered value incomparable with the unfiltered one.
        This is why the ``_qc_pass`` variants cannot be left to the caller --
        slicing the label image first would re-threshold on different pixels.
        """
        if cells is None:
            return replace(self, cell_subset=None)
        mask = np.asarray(cells)
        if mask.dtype != bool:
            idx, mask = mask.astype(int), np.zeros(self.means.shape[0], dtype=bool)
            mask[idx] = True
        if mask.shape[0] != self.means.shape[0]:
            raise ValueError(f"cells has length {mask.shape[0]}, expected {self.means.shape[0]} labels")
        return replace(self, cell_subset=mask)

    def to_frame(self) -> pd.DataFrame:
        """Per-cell mean and positive-pixel fraction for the selected cells."""
        sel = self.selection
        frame = pd.DataFrame({"label": np.flatnonzero(sel) + 1, "area": self.areas[sel]}).set_index("label")
        for j, name in enumerate(self.markers):
            frame[f"{name}__mean"] = self.means[sel, j]
            frame[f"{name}__prop_positive"] = self.prop_positive[sel, j]
        return frame


def cell_pixel_stats(
    image: np.ndarray,
    labels: np.ndarray,
    markers: Sequence[str] | None = None,
    *,
    dilation_iterations: int = 3,
    min_background_pseudocells: int = 3,
    random_state: int = 0,
) -> CellPixelStats:
    """Summarise an image under a label mask, with a local background sample.

    Background is drawn from a ring immediately outside the masks rather than
    from all non-cell pixels, so the comparison stays local instead of being
    diluted by empty slide.

    Parameters
    ----------
    image
        Channel-by-y-by-x array. A 2-D array is promoted to a single channel.
    labels
        Two-dimensional integer mask, 0 for background.
    markers
        Channel names; defaults to ``marker_0 ...``.
    dilation_iterations
        How far to grow the masks when sampling proximal background, in pixels.
    min_background_pseudocells
        Fewest cell-sized background patches that make a rank-matched curve
        meaningful. Below this the curve is ``None`` and the metrics that need it
        report NaN -- but :attr:`CellPixelStats.n_background_pseudocells` records
        the count, so a degenerate crop is visible rather than silently empty.
    random_state
        Seed for background sampling, which is otherwise irreproducible.
    """
    from scipy.ndimage import binary_dilation

    image = np.asarray(image)
    if image.ndim == 2:
        image = image[None, :, :]
    if image.ndim != 3:
        raise ValueError(f"expected a channel-y-x image, got shape {image.shape}")
    labels = np.squeeze(np.asarray(labels)).astype(np.int64, copy=False)
    if labels.ndim != 2:
        raise ValueError(f"expected a 2D label mask, got shape {labels.shape}")
    if image.shape[1:] != labels.shape:
        raise ValueError(f"image spatial shape {image.shape[1:]} does not match labels shape {labels.shape}")

    names = tuple(markers) if markers is not None else tuple(f"marker_{i}" for i in range(image.shape[0]))
    if len(names) != image.shape[0]:
        raise ValueError(f"got {len(names)} marker names for {image.shape[0]} channels")

    flat_labels = labels.ravel()
    inside = flat_labels > 0
    labs = flat_labels[inside]
    n_labels = int(flat_labels.max())
    areas_all = np.bincount(labs, minlength=n_labels + 1).astype(float)
    areas = areas_all[1:]
    present = areas > 0

    n_channels = image.shape[0]
    means = np.zeros((n_labels, n_channels), dtype=float)
    prop_positive = np.zeros((n_labels, n_channels), dtype=float)
    thresholds = np.zeros(n_channels, dtype=float)

    # Vectorised with bincount: the prototype masked the whole image once per
    # cell, which on a full mosaic is billions of comparisons.
    for c in range(n_channels):
        vals = image[c].ravel()[inside].astype(float)
        sums = np.bincount(labs, weights=vals, minlength=n_labels + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            means[:, c] = np.where(present, sums[1:] / np.maximum(areas, 1.0), np.nan)
        thr = otsu_threshold(vals)
        thresholds[c] = thr
        # Fraction of a cell's own pixels above threshold: punctate noise gives a
        # bright mean with low coverage, real staining gives both.
        pos_counts = np.bincount(labs[vals > thr], minlength=n_labels + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            prop_positive[:, c] = np.where(present, pos_counts[1:] / np.maximum(areas, 1.0), np.nan)

    cell_mask = labels > 0
    dilated = binary_dilation(cell_mask, iterations=max(1, int(dilation_iterations)))
    proximal = dilated & ~cell_mask

    n_cells = int(present.sum())
    avg_cell_size = max(1, int(cell_mask.sum() / n_cells) if n_cells else 100)
    rng = np.random.default_rng(random_state)

    curves: list[np.ndarray | None] = []
    n_drawn = 0
    inside_sum = np.zeros(n_channels)
    total_sum = np.zeros(n_channels)
    inside_mean = np.full(n_channels, np.nan)
    inside_pos = np.full(n_channels, np.nan)
    ring_mean = np.full(n_channels, np.nan)
    ring_pos = np.full(n_channels, np.nan)

    for c in range(n_channels):
        channel = image[c]
        in_native = channel[cell_mask]
        ring = channel[proximal].astype(float)
        inside_sum[c] = float(in_native.astype(float).sum())
        total_sum[c] = float(channel.astype(float).sum())
        if in_native.size:
            # Averaged in the image's own dtype, not upcast first. Images are
            # stored float32, and taking this mean in float64 shifts the
            # signal-to-background ratio in the seventh decimal -- enough to
            # break a bit-for-bit comparison against the benchmark's published
            # numbers, which is the only gate that can prove this port correct.
            inside_mean[c] = float(np.mean(in_native))
            inside_pos[c] = float(np.mean(in_native > thresholds[c]))
        if ring.size:
            ring_mean[c] = float(ring.mean())
            ring_pos[c] = float(np.mean(ring > thresholds[c]))

        n_samples = min(n_cells, ring.size // avg_cell_size)
        if n_samples >= min_background_pseudocells:
            picked = rng.choice(ring, size=(n_samples, avg_cell_size), replace=False)
            curves.append(np.sort(picked.mean(axis=1)))
            n_drawn = n_samples
        else:
            curves.append(None)

    return CellPixelStats(
        markers=names,
        areas=areas,
        means=means,
        prop_positive=prop_positive,
        thresholds=thresholds,
        present=present,
        inside_sum=inside_sum,
        total_sum=total_sum,
        inside_mean=inside_mean,
        inside_positive_fraction=inside_pos,
        ring_mean=ring_mean,
        ring_positive_fraction=ring_pos,
        background_curves=tuple(curves),
        n_background_pseudocells=int(n_drawn),
        image_shape=tuple(image.shape),
        intensity_dtype=image.dtype,
        params={
            "dilation_iterations": int(dilation_iterations),
            "random_state": int(random_state),
            "min_background_pseudocells": int(min_background_pseudocells),
        },
    )
