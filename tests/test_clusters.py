import numpy as np
import pytest
from sklearn.cluster import KMeans

import sprotmetrics as spm


def test_cluster_weighted_marker_cv_returns_float(table):
    value = spm.cluster_weighted_marker_cv(table)

    assert isinstance(value, float)
    assert np.isnan(value) or 0 <= value <= 1


def test_cluster_weighted_marker_cv_is_deterministic(table):
    first = spm.cluster_weighted_marker_cv(table)
    second = spm.cluster_weighted_marker_cv(table)

    assert first == pytest.approx(second, nan_ok=True)


def test_cluster_weighted_marker_cv_returns_nan_for_fewer_than_10_cells():
    X = np.ones((9, 2))

    value = spm.cluster_weighted_marker_cv(X, layer=None)

    assert np.isnan(value)


def test_cluster_weighted_marker_cv_matches_reference():
    X = np.array([
        [1, 8],
        [2, 3],
        [3, 9],
        [4, 2],
        [5, 7],
        [6, 1],
        [7, 6],
        [8, 4],
        [9, 10],
        [10, 5],
    ], dtype=np.float32)

    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

    scores = []

    for k in range(1, 11):
        if k == 1:
            labels = np.zeros(len(X), dtype=int)
        else:
            labels = KMeans(
                n_clusters=k,
                random_state=777,
                n_init=10,
            ).fit_predict(X_scaled)

        cluster_cvs = []
        cluster_sizes = []

        for cluster in range(k):
            cluster_X = X[labels == cluster]
            if len(cluster_X) == 0:
                continue

            means = cluster_X.mean(axis=0)
            stds = cluster_X.std(axis=0)

            cvs = np.where(
                stds == 0,
                0.0,
                np.divide(
                    stds,
                    means,
                    out=np.full_like(stds, np.nan),
                    where=cluster_X.sum(axis=0) != 0,
                ),
            )

            cluster_cvs.append(np.nanmean(cvs))
            cluster_sizes.append(len(cluster_X))

        scores.append(
            np.average(cluster_cvs, weights=cluster_sizes)
        )

    expected = 1.0 / (1.0 + np.average(scores))
    actual = spm.cluster_weighted_marker_cv(X, layer=None)

    assert actual == pytest.approx(expected, abs=0.003)
