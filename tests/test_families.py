"""Family functions: the output contract, and the QC-variant pairing invariant."""

from __future__ import annotations

import numpy as np
import pytest

import sprotmetrics as spm

from .conftest import EXCLUSIVE_PAIR, MARKERS, NUCLEAR_MARKERS

MATRIX_FAMILIES = {
    "basic_statistics": lambda t: spm.basic_statistics(t, qc_variants=True),
    "marker_distribution": lambda t: spm.marker_distribution(t, qc_variants=True),
    "normalization_diagnostics": lambda t: spm.normalization_diagnostics(t, qc_variants=True),
    "covariate_dependence": lambda t: spm.covariate_dependence(t, qc_variants=True),
    "segmentation_quality": lambda t: spm.segmentation_quality(
        t, nuclear_markers=NUCLEAR_MARKERS, exclusive_pairs=[EXCLUSIVE_PAIR], qc_variants=True
    ),
}


@pytest.mark.parametrize("name", sorted(MATRIX_FAMILIES))
def test_family_returns_finite_or_nan_floats(name, table):
    values = MATRIX_FAMILIES[name](table)
    assert values
    for key, value in values.items():
        assert isinstance(value, float), key


@pytest.mark.parametrize("name", sorted(MATRIX_FAMILIES))
def test_every_metric_has_a_qc_twin_unless_it_cannot(name, table):
    values = MATRIX_FAMILIES[name](table)
    bases = [k for k in values if not k.endswith("_qc_pass")]
    for base in bases:
        if base in {"n_markers", "qc_pass_rate"}:
            assert f"{base}_qc_pass" not in values
        else:
            assert f"{base}_qc_pass" in values, base


@pytest.mark.parametrize("name", sorted(MATRIX_FAMILIES))
def test_qc_variants_are_omitted_by_default(name, table):
    values = {
        "basic_statistics": spm.basic_statistics,
        "marker_distribution": spm.marker_distribution,
        "normalization_diagnostics": spm.normalization_diagnostics,
        "covariate_dependence": spm.covariate_dependence,
        "segmentation_quality": spm.segmentation_quality,
    }[name](table)
    assert not any(k.endswith("_qc_pass") for k in values)


def test_missing_qc_column_warns_and_yields_nan(table):
    del table.obs["passed_QC"]
    with pytest.warns(UserWarning, match="_qc_pass"):
        values = spm.marker_distribution(table, qc_variants=True)
    qc = {k: v for k, v in values.items() if k.endswith("_qc_pass")}
    assert qc and all(np.isnan(v) for v in qc.values())
    # The all-cell values must be untouched by the absence of QC.
    assert not np.isnan(values["median_marker_cv"])


def test_panel_metrics_return_nan_rather_than_guessing(table):
    """Without the panel configuration there is no right answer, so do not invent one."""
    values = spm.segmentation_quality(table)
    assert np.isnan(values["mutually_exclusive_coexpression_ratio"])


def test_signal_background_contract(image, labels, table):
    values = spm.signal_background(image, labels, MARKERS)
    assert set(values) == {
        "median_rel_sni",
        "min_rel_sni",
        "median_intracell_coverage",
        "median_marker_gini",
        "median_signal_to_background_ratio",
        "median_cellular_specificity",
    }
    assert 0 <= values["median_marker_gini"] <= 1


def test_marker_table_exposes_the_per_marker_detail(table):
    frame = spm.marker_table(table)
    assert list(frame.index) == MARKERS
    assert {"mean", "cv", "coverage", "dynamic_range"} <= set(frame.columns)
    assert (frame["coverage"] >= 0).all() and (frame["coverage"] <= 1).all()
