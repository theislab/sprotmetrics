"""Separability and stability of the cell populations preprocessing leaves behind.

:func:`silhouette_cell_type` needs annotations the ISP pipeline never produced,
which is why the benchmark's ``cluster_separability`` component was implemented
but never wired into the workflow. :func:`clustering_stability` is the
annotation-free counterpart: it asks whether the structure survives resampling,
which needs no labels at all.
"""

from __future__ import annotations

import numpy as np

from sprotmetrics._registry import Direction, Inputs, Reference, Status, Task, metric
from sprotmetrics._typing import Mask, MatrixLike, VectorLike
from sprotmetrics._utils import DEFAULT_QC_COLUMN, resolve_subset, to_matrix

__all__ = ["cluster_separability", "clustering_stability", "silhouette_cell_type"]

FAMILY = "cluster_separability"


def _matrix(data: MatrixLike, layer: str | None, subset: Mask) -> np.ndarray:
    X = to_matrix(data, layer, fallback_layer="counts", dtype=np.float32, nan_to_num=True)
    mask = resolve_subset(data, subset)
    return X if mask is None else X[mask]


@metric(
    name="silhouette_cell_type",
    label="Cluster separability of annotated cell types",
    family=FAMILY,
    task=(Task.NORMALIZATION,),
    reference=Reference.CELL_TYPE_LABELS,
    inputs=Inputs.MATRIX,
    min=-1,
    max=1,
    direction=Direction.HIGHER,
)
def silhouette_cell_type(
    data: MatrixLike,
    labels: VectorLike = "cell_type",
    *,
    layer: str | None = "normalized",
    metric_name: str = "euclidean",
    max_cells: int = 5000,
    random_state: int = 0,
    subset: Mask = None,
) -> float:
    """Silhouette score of annotated cell types in marker space.

    Preprocessing that preserves biology should leave cell types separable.
    Returns NaN where the score is undefined -- no annotation, or fewer than two
    labels surviving the subsample -- rather than a number that looks like one.
    """
    from sklearn.metrics import silhouette_score

    X = _matrix(data, layer, subset)
    values = labels
    if isinstance(labels, str):
        if not hasattr(data, "obs") or labels not in data.obs.columns:
            return float("nan")
        values = np.asarray(data.obs[labels].to_numpy())
        rows = resolve_subset(data, subset)
        if rows is not None:
            values = values[rows]
    values = np.asarray(values)

    if X.shape[0] == 0 or len(set(values.tolist())) < 2:
        return float("nan")
    if max_cells > 0 and X.shape[0] > max_cells:
        idx = np.random.RandomState(random_state).choice(X.shape[0], size=max_cells, replace=False)
        X, values = X[idx], values[idx]
    # Subsampling can leave a single label behind, which silhouette rejects.
    if len(set(values.tolist())) < 2:
        return float("nan")
    return float(silhouette_score(X, values, metric=metric_name))


@metric(
    name="clustering_stability",
    label="Stability of clustering under resampling",
    family=FAMILY,
    task=(Task.ALL_AXES,),
    reference=Reference.NONE,
    inputs=Inputs.MATRIX,
    min=0,
    max=1,
    direction=Direction.HIGHER,
    status=Status.EXPERIMENTAL,
    references=("10.1371/journal.pcbi.1013350",),
    origin="Metrics-2.xlsx row 5",
)
def clustering_stability(
    data: MatrixLike,
    *,
    layer: str | None = "normalized",
    n_clusters: int = 10,
    n_resamples: int = 10,
    subsample_fraction: float = 0.8,
    score: str = "ari",
    max_cells: int = 5000,
    random_state: int = 0,
    subset: Mask = None,
) -> float:
    """Mean agreement between clusterings of overlapping subsamples.

    Needs no annotation, which is what makes it usable here. Bruhns et al. (2025)
    calibrate how downstream clustering degrades as segmentation quality falls,
    so an observed value can in principle be read back as an implied segmentation
    F1 -- as close to a reference comparison as one gets without reference masks.

    The most expensive metric in the package: it clusters `n_resamples` times.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    X = _matrix(data, layer, subset)
    n_obs = X.shape[0]
    if n_obs < n_clusters * 2 or n_resamples < 2:
        return float("nan")

    rng = np.random.default_rng(random_state)
    if max_cells > 0 and n_obs > max_cells:
        X = X[np.sort(rng.choice(n_obs, size=max_cells, replace=False))]
        n_obs = X.shape[0]

    size = max(n_clusters * 2, int(n_obs * subsample_fraction))
    assignments = []
    for i in range(n_resamples):
        idx = np.sort(rng.choice(n_obs, size=min(size, n_obs), replace=False))
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state + i)
        full = np.full(n_obs, -1)
        full[idx] = km.fit_predict(X[idx])
        assignments.append(full)

    agree = adjusted_rand_score if score == "ari" else adjusted_mutual_info_score
    scores = []
    for i in range(len(assignments)):
        for j in range(i + 1, len(assignments)):
            # Only cells both subsamples happened to draw can be compared.
            both = (assignments[i] >= 0) & (assignments[j] >= 0)
            if both.sum() >= n_clusters:
                scores.append(float(agree(assignments[i][both], assignments[j][both])))
    return float(np.mean(scores)) if scores else float("nan")


def cluster_separability(
    data: MatrixLike,
    *,
    layer: str | None = "normalized",
    labels: VectorLike = "cell_type",
    max_cells: int = 5000,
    random_state: int = 0,
    subset: Mask = None,
    qc_variants: bool | str = False,
    qc_column: str = DEFAULT_QC_COLUMN,
) -> dict[str, float]:
    """Cluster separability of annotated cell types, with the benchmark's id."""

    def once(rows: Mask) -> dict[str, float]:
        return {
            "silhouette_cell_type": silhouette_cell_type(
                data,
                labels,
                layer=layer,
                max_cells=max_cells,
                random_state=random_state,
                subset=rows,
            )
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
