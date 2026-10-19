"""Localisation metrics, on an image that actually has subcellular structure."""

from __future__ import annotations

import numpy as np
import pytest

import sprotmetrics as spm

MARKERS = ["HISTONE", "MEMBRANE"]


@pytest.fixture
def compartments():
    """A nuclear marker inside the nucleus and a membrane marker in the rim."""
    size = 64
    yy, xx = np.ogrid[:size, :size]
    radius = (yy - 32) ** 2 + (xx - 32) ** 2
    cell = (radius <= 14**2).astype(np.int32)
    nucleus = radius <= 7**2

    image = np.full((2, size, size), 10.0, dtype=np.float32)
    image[0][nucleus] = 500.0  # histone: nuclear
    image[1][cell.astype(bool) & ~nucleus] = 500.0  # membrane marker: not nuclear
    return image, cell, nucleus


def test_enrichment_separates_the_two_compartments(compartments):
    image, cell, nucleus = compartments
    values = spm.nuclear_enrichment(image, cell, nucleus, MARKERS)
    assert values[0] > 1.5, "a nuclear marker should be enriched in the nucleus"
    assert values[1] < 0.5, "a membrane marker should be depleted from it"


def test_agreement_is_perfect_when_the_priors_match(compartments):
    image, cell, nucleus = compartments
    values = spm.nuclear_enrichment(image, cell, nucleus, MARKERS)
    expected = {"HISTONE": "nuclear", "MEMBRANE": "membrane"}
    assert spm.localisation_agreement(values, expected, MARKERS) == pytest.approx(1.0)


def test_agreement_falls_when_the_priors_are_wrong(compartments):
    image, cell, nucleus = compartments
    values = spm.nuclear_enrichment(image, cell, nucleus, MARKERS)
    swapped = {"HISTONE": "membrane", "MEMBRANE": "nuclear"}
    assert spm.localisation_agreement(values, swapped, MARKERS) == pytest.approx(0.0)


def test_markers_without_a_prior_are_skipped_not_guessed(compartments):
    image, cell, nucleus = compartments
    values = spm.nuclear_enrichment(image, cell, nucleus, MARKERS)
    assert spm.localisation_agreement(values, {"HISTONE": "nuclear"}, MARKERS) == pytest.approx(1.0)
    assert np.isnan(spm.localisation_agreement(values, {}, MARKERS))
