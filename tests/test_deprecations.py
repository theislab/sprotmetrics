"""Metrics the package records but does not compute."""

from __future__ import annotations

import pytest

import sprotmetrics as spm
from sprotmetrics._registry import Status


@pytest.mark.parametrize("metric_id", ["snr_per_protein", "cv_per_protein", "dynamic_range_per_protein"])
def test_redundant_metrics_are_recorded_with_a_replacement(metric_id):
    info = spm.get_metric(metric_id)
    assert info.status is Status.DEPRECATED
    assert info.superseded_by
    assert info.notes, "a deprecation without a reason is just an absence"


def test_computing_a_deprecated_metric_names_the_replacement():
    with pytest.raises(KeyError, match="median_marker_cv"):
        spm.compute("snr_per_protein", None)


def test_deprecated_metrics_are_absent_from_the_default_catalogue():
    assert "snr_per_protein" not in spm.list_metrics().index
    assert "snr_per_protein" in spm.list_metrics(include_deprecated=True).index
