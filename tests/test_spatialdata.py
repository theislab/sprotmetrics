"""The SpatialData layer: does it resolve elements and give the same numbers?"""

from __future__ import annotations

import numpy as np
import pytest

import sprotmetrics as spm

from .conftest import MARKERS, build_image, build_labels

sd = pytest.importorskip("spatialdata")
sd_models = pytest.importorskip("spatialdata.models")
import sprotmetrics.sd as spmsd  # noqa: E402


@pytest.fixture
def sdata():
    return sd.SpatialData(
        images={
            "fixture_image": sd_models.Image2DModel.parse(build_image(), dims=("c", "y", "x"), c_coords=list(MARKERS))
        },
        labels={"fixture_cell": sd_models.Labels2DModel.parse(build_labels(), dims=("y", "x"))},
    )


def test_wrapper_matches_the_array_api(sdata):
    direct = spm.signal_background(build_image(), build_labels(), MARKERS)
    through_sdata = spmsd.signal_background(sdata)
    assert set(direct) == set(through_sdata)
    for key, value in direct.items():
        assert value == pytest.approx(through_sdata[key], nan_ok=True), key


def test_channel_names_come_from_the_c_coordinate(sdata):
    stats = spmsd.cell_pixel_stats(sdata)
    assert list(stats.markers) == MARKERS


def test_single_element_is_inferred_without_a_key(sdata):
    assert spmsd.cell_pixel_stats(sdata).n_objects > 0


def test_ambiguous_elements_name_the_candidates(sdata):
    sdata.labels["second"] = sd_models.Labels2DModel.parse(build_labels(), dims=("y", "x"))
    with pytest.raises(ValueError, match="name the one you mean"):
        spmsd.cell_pixel_stats(sdata)


def test_unknown_key_lists_what_is_available(sdata):
    with pytest.raises(KeyError, match="fixture_image"):
        spmsd.cell_pixel_stats(sdata, image_key="nope")


def test_compute_all_joins_the_catalogue(sdata):
    frame = spmsd.compute_all(sdata)
    assert not frame.empty
    assert {"value", "family", "direction", "maximize"} <= set(frame.columns)
    assert frame["family"].notna().any()


def test_segmentation_agreement_against_itself(sdata):
    sdata.labels["copy"] = sd_models.Labels2DModel.parse(build_labels(), dims=("y", "x"))
    scores = spmsd.segmentation_agreement(sdata, "fixture_cell", "copy")
    assert scores["foreground_dice"] == pytest.approx(1.0)
    assert scores["object_f1_at_iou_50"] == pytest.approx(1.0)


def test_reference_scoring_sees_over_expansion():
    from scipy.ndimage import grey_dilation

    reference = build_labels()
    scores = spm.segmentation_agreement(grey_dilation(reference, size=(5, 5)), reference)
    assert scores["foreground_recall"] > scores["foreground_precision"]
    assert np.isfinite(scores["object_f1_at_iou_50"])
