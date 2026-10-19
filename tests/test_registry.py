"""Contract tests driven by the registry itself.

Because every metric declares its bounds, direction and family, one sweep can
assert the whole catalogue rather than restating it metric by metric -- and a
metric added later is covered the moment it is registered.
"""

from __future__ import annotations

import numpy as np
import pytest

import sprotmetrics as spm
from sprotmetrics._registry import QC_SUFFIX, Status


def test_catalogue_is_not_empty():
    frame = spm.list_metrics()
    assert len(frame) > 30
    assert frame.index.is_unique


@pytest.mark.parametrize("metric_id", sorted(spm.list_metrics().index))
def test_declared_bounds_are_coherent(metric_id):
    info = spm.get_metric(metric_id)
    assert info.min <= info.max
    assert info.label and info.summary
    assert info.family


def test_every_id_has_exactly_one_producer():
    """Two families emitting one id from duplicated code is how they drift apart."""
    frame = spm.list_metrics(include_deprecated=True)
    assert frame.index.is_unique


def test_qc_variants_resolve_to_their_base():
    frame = spm.list_metrics(include_qc_variants=True)
    for metric_id in frame.index:
        if metric_id.endswith(f"_{QC_SUFFIX}"):
            base = spm.get_metric(metric_id[: -len(QC_SUFFIX) - 1])
            assert base.qc_variant


def test_invariant_metrics_carry_no_qc_variant():
    """n_markers cannot change and qc_pass_rate is 1 among passing cells."""
    for metric_id in ("n_markers", "qc_pass_rate"):
        assert not spm.get_metric(metric_id).qc_variant
    frame = spm.list_metrics(include_qc_variants=True)
    assert "n_markers_qc_pass" not in frame.index
    assert "qc_pass_rate_qc_pass" not in frame.index


def test_descriptive_metrics_are_never_exported_for_ranking():
    """The OpenProblems schema cannot say "no direction", so these must be dropped."""
    exported = {m["name"] for m in spm.to_openproblems_info()["metrics"]}
    for metric_id in spm.list_metrics().index:
        if spm.get_metric(metric_id).maximize is None:
            assert metric_id not in exported


def test_openproblems_entries_have_the_required_fields():
    required = {"name", "label", "summary", "description", "min", "max", "maximize"}
    entries = spm.to_openproblems_info(family="segmentation_quality")["metrics"]
    assert entries
    for entry in entries:
        assert required <= set(entry)
        assert isinstance(entry["maximize"], bool)


def test_descriptive_metric_refuses_to_emit_a_direction():
    info = spm.get_metric("n_cells")
    assert info.maximize is None
    with pytest.raises(ValueError, match="no way to express"):
        info.to_openproblems()


def test_deprecated_metrics_are_hidden_but_findable():
    visible = spm.list_metrics()
    everything = spm.list_metrics(include_deprecated=True)
    assert len(everything) >= len(visible)
    for metric_id in everything.index:
        if spm.get_metric(metric_id).status is Status.DEPRECATED:
            assert metric_id not in visible.index


def test_compute_applies_the_registered_aggregation(table):
    panel = spm.marker_coverage(table, threshold=0.1)
    assert spm.compute("median_marker_coverage", table, threshold=0.1) == pytest.approx(float(np.median(panel)))
    assert spm.compute("min_marker_coverage", table, threshold=0.1) == pytest.approx(float(np.min(panel)))


def test_compute_on_a_qc_variant_restricts_to_passing_cells(table):
    whole = spm.compute("median_area", table)
    passing = spm.compute("median_area_qc_pass", table)
    # A quarter of the fixture is deliberately small debris.
    assert passing > whole


def test_unknown_metric_id_is_an_error():
    with pytest.raises(KeyError):
        spm.get_metric("no_such_metric")
