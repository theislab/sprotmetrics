"""Small deterministic fixtures with structure the metrics are supposed to find.

Modelled on the ISP benchmark's own ``src/testing/isp_fixtures.py``, but written
fresh and in memory: the benchmark's copy is wired into five other component
tests through Viash ``test_resources``, so it stays where it is rather than
becoming a dependency of this package.

The point of the panel shape is that each metric has something to detect. Two
nuclear replicates make replicate concordance meaningful; a mutually exclusive
immune/epithelial pair makes co-expression meaningful; a quarter of the cells are
small *and* dim so that QC has something to reject.
"""

from __future__ import annotations

import numpy as np
import pytest

IMAGE_SHAPE = (4, 64, 64)
MARKERS = ["DNA_1", "DNA_2", "CD45", "ECAD"]
NUCLEAR_MARKERS = ["DNA_1", "DNA_2"]
EXCLUSIVE_PAIR = ("CD45", "ECAD")

# Deliberately varied radii, so area-based filtering has something to remove.
_BLOBS = [(14, 14, 5), (14, 46, 5), (46, 14, 5), (46, 46, 4), (30, 30, 2)]


def build_image(seed: int = 0) -> np.ndarray:
    """Channel-y-x image with bright nuclear blobs on a dim noisy background."""
    rng = np.random.default_rng(seed)
    img = (rng.random(IMAGE_SHAPE) * 10.0).astype(np.float32)
    yy, xx = np.ogrid[: IMAGE_SHAPE[1], : IMAGE_SHAPE[2]]
    for cy, cx, radius in _BLOBS:
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        # Both nuclear channels light the same blobs, as restains of the same
        # nuclei do, with a little independent noise so their correlation is
        # high but not exactly 1.
        img[0][disk] += 500.0
        img[1][disk] += 500.0 + rng.normal(0.0, 5.0, size=int(disk.sum())).astype(np.float32)
    # One lineage marker per half of the field, so the exclusive pair is genuinely
    # exclusive and co-positivity should come out near zero.
    for (cy, cx, radius), channel in zip(_BLOBS[:4], [2, 3, 2, 3], strict=False):
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        img[channel][disk] += 300.0
    return img


def build_labels() -> np.ndarray:
    """Label mask matching :func:`build_image`, numbered 1..n with no gaps."""
    labels = np.zeros(IMAGE_SHAPE[1:], dtype=np.int32)
    yy, xx = np.ogrid[: IMAGE_SHAPE[1], : IMAGE_SHAPE[2]]
    for i, (cy, cx, radius) in enumerate(_BLOBS, start=1):
        labels[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2] = i
    return labels


def build_table(n_cells: int = 40, seed: int = 0, *, with_celltypes: bool = False):
    """Cell-by-protein AnnData matching the benchmark's aggregation output contract."""
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(seed)
    X = rng.lognormal(mean=3.0, sigma=1.0, size=(n_cells, len(MARKERS))).astype(np.float32)

    # DNA_2 is the same nuclei restained, so it tracks DNA_1 closely. Leaving them
    # independent would make replicate concordance read ~0 on data that is fine.
    i0, i1 = MARKERS.index("DNA_1"), MARKERS.index("DNA_2")
    X[:, i1] = np.abs(X[:, i0] * rng.normal(1.0, 0.05, size=n_cells).astype(np.float32))

    cell_types = None
    if with_celltypes:
        # Groups differing in marker *composition*, not overall brightness: a
        # uniform shift would be undone by per-cell normalization.
        group = rng.integers(0, 2, size=n_cells)
        X[group == 1, MARKERS.index("CD45")] *= 20.0
        X[group == 0, MARKERS.index("ECAD")] *= 20.0
        cell_types = np.where(group == 1, "type_a", "type_b")

    # A fixed quarter are debris: small *and* dim, which is what QC exists to
    # catch. A fixed fraction rather than a lucky draw keeps assertions stable.
    n_debris = max(1, n_cells // 4)
    is_debris = np.zeros(n_cells, dtype=bool)
    is_debris[:n_debris] = True
    X[is_debris] *= 0.05

    area = np.where(is_debris, rng.uniform(5, 40, n_cells), rng.uniform(80, 300, n_cells))
    obs = pd.DataFrame(
        {
            "area": area,
            "centroid_y": rng.uniform(0, 64, n_cells),
            "centroid_x": rng.uniform(0, 64, n_cells),
            "passed_QC": ~is_debris,
        },
        index=[str(i + 1) for i in range(n_cells)],
    )
    if cell_types is not None:
        obs["cell_type"] = pd.Categorical(cell_types)

    adata = ad.AnnData(X=X.copy(), obs=obs)
    adata.var_names = MARKERS
    adata.layers["counts"] = X.copy()
    # arcsinh, as the benchmark's cytometry-style normalization does.
    adata.layers["normalized"] = np.arcsinh(X / 5.0).astype(np.float32)
    return adata


@pytest.fixture
def image() -> np.ndarray:
    return build_image()


@pytest.fixture
def labels() -> np.ndarray:
    return build_labels()


@pytest.fixture
def table():
    return build_table()


@pytest.fixture
def table_with_celltypes():
    return build_table(with_celltypes=True)


@pytest.fixture
def stats(image, labels):
    from sprotmetrics import cell_pixel_stats

    return cell_pixel_stats(image, labels, MARKERS, min_background_pseudocells=1)
