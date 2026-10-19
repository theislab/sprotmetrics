"""Direction tests: does each metric move the way its registration claims?

These are the assertions that cannot be generated from the registry, because
they encode what the metric is *for*. Each builds a deliberately degraded input
and checks the metric notices.
"""

from __future__ import annotations

import numpy as np

import sprotmetrics as spm

from .conftest import EXCLUSIVE_PAIR, MARKERS, NUCLEAR_MARKERS, build_image, build_labels, build_table


def _dilate(labels: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Grow every mask, as an over-expanding segmentation does."""
    from scipy.ndimage import grey_dilation

    return grey_dilation(labels, size=(2 * iterations + 1, 2 * iterations + 1))


def test_added_background_lowers_signal_to_background():
    image, labels = build_image(), build_labels()
    noisy = image.copy()
    noisy[:, labels == 0] += 200.0  # raise the background only

    clean_stats = spm.cell_pixel_stats(image, labels, MARKERS, min_background_pseudocells=1)
    noisy_stats = spm.cell_pixel_stats(noisy, labels, MARKERS, min_background_pseudocells=1)

    clean = np.nanmedian(spm.signal_to_background_ratio(clean_stats))
    noisy_ratio = np.nanmedian(spm.signal_to_background_ratio(noisy_stats))
    assert noisy_ratio < clean


def test_over_expanded_masks_capture_more_nuclear_signal():
    """Containment rises with mask size, which is why it needs a counterweight."""
    image, labels = build_image(), build_labels()
    tight = spm.cell_pixel_stats(image, labels, MARKERS, min_background_pseudocells=1)
    loose = spm.cell_pixel_stats(image, _dilate(labels), MARKERS, min_background_pseudocells=1)
    assert spm.nuclear_signal_containment(loose, NUCLEAR_MARKERS) > spm.nuclear_signal_containment(
        tight, NUCLEAR_MARKERS
    )


def test_replicate_correlation_falls_when_the_restains_disagree():
    good = build_table()
    bad = build_table()
    rng = np.random.default_rng(1)
    # Break the link between the two nuclear channels.
    bad.layers["counts"][:, MARKERS.index("DNA_2")] = rng.lognormal(3.0, 1.0, size=bad.n_obs).astype(np.float32)

    assert spm.replicate_correlation(bad, NUCLEAR_MARKERS) < spm.replicate_correlation(good, NUCLEAR_MARKERS)


def test_co_expressing_incompatible_markers_raises_the_exclusivity_ratio():
    exclusive = build_table(n_cells=80)
    merged = build_table(n_cells=80)
    a, b = (MARKERS.index(name) for name in EXCLUSIVE_PAIR)
    half = exclusive.n_obs // 2

    # Exclusive: each half is positive for one marker only.
    exclusive.layers["counts"][:half, a] = 1000.0
    exclusive.layers["counts"][half:, a] = 1.0
    exclusive.layers["counts"][:half, b] = 1.0
    exclusive.layers["counts"][half:, b] = 1000.0

    # Merged: the same half is positive for both, as a mask spanning two cells
    # would make it. Both markers stay bimodal, so positivity is still defined.
    merged.layers["counts"][:half, [a, b]] = 1000.0
    merged.layers["counts"][half:, [a, b]] = 1.0

    assert spm.mutually_exclusive_coexpression_ratio(
        merged, [EXCLUSIVE_PAIR]
    ) > spm.mutually_exclusive_coexpression_ratio(exclusive, [EXCLUSIVE_PAIR])


def test_area_scaled_expression_raises_covariate_dependence():
    """Sum aggregation makes every channel scale with area; mean does not."""
    mean_like = build_table()
    sum_like = build_table()
    area = sum_like.obs["area"].to_numpy()[:, None]
    sum_like.layers["counts"] = (sum_like.layers["counts"] * area).astype(np.float32)

    mean_r2 = np.mean(spm.variance_explained_by_covariate(mean_like, layer="counts"))
    sum_r2 = np.mean(spm.variance_explained_by_covariate(sum_like, layer="counts"))
    assert sum_r2 > mean_r2


def test_normalization_compresses_total_intensity_spread():
    table = build_table()
    ratio = spm.total_cv_ratio(table)
    assert ratio < 1.0


def test_bimodality_falls_when_the_populations_are_merged():
    separable = build_table(with_celltypes=True)
    flat = build_table(with_celltypes=True)
    rng = np.random.default_rng(0)
    # Replace a structured marker with pure noise: no two populations left.
    flat.layers["counts"][:, MARKERS.index("CD45")] = rng.normal(100.0, 1.0, size=flat.n_obs).astype(np.float32)

    j = MARKERS.index("CD45")
    assert spm.marker_bimodality(flat)[j] < spm.marker_bimodality(separable)[j]


def test_tile_cv_rises_with_an_illumination_gradient():
    # Enough cells that a 2x2 grid clears min_cells_per_tile.
    flat = build_table(n_cells=200)
    shaded = build_table(n_cells=200)
    # A left-to-right gradient, as uncorrected illumination produces.
    gradient = 1.0 + shaded.obs["centroid_x"].to_numpy()[:, None] / 16.0
    shaded.layers["counts"] = (shaded.layers["counts"] * gradient).astype(np.float32)

    assert np.nanmedian(spm.tile_cv(shaded, n_tiles=2)) > np.nanmedian(spm.tile_cv(flat, n_tiles=2))
