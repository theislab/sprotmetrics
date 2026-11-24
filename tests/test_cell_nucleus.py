
import numpy as np
import pytest

from sprotmetrics import fraction_of_matched_cells_and_nuclei


def test_perfectly_matched_cells_and_nuclei():
    cells = np.zeros((10, 10), dtype=int)
    nuclei = np.zeros((10, 10), dtype=int)

    cells[1:9, 1:9] = 1
    nuclei[3:6, 3:6] = 1

    # One matched pair: 1 / (1 + 1)
    assert fraction_of_matched_cells_and_nuclei(cells, nuclei) == pytest.approx(1.0)


def test_unmatched_cell():
    cells = np.zeros((10, 10), dtype=int)
    nuclei = np.zeros((10, 10), dtype=int)

    cells[1:4, 1:4] = 1

    assert fraction_of_matched_cells_and_nuclei(cells, nuclei) == 0.0


def test_unmatched_nucleus():
    cells = np.zeros((10, 10), dtype=int)
    nuclei = np.zeros((10, 10), dtype=int)

    nuclei[1:4, 1:4] = 1

    assert fraction_of_matched_cells_and_nuclei(cells, nuclei) == 0.0


def test_two_cells_and_two_nuclei():
    cells = np.zeros((20, 20), dtype=int)
    nuclei = np.zeros((20, 20), dtype=int)

    cells[1:9, 1:9] = 1
    cells[11:19, 11:19] = 2
    nuclei[3:6, 3:6] = 1
    nuclei[13:16, 13:16] = 2

    assert fraction_of_matched_cells_and_nuclei(cells, nuclei) == pytest.approx(1.0)


def test_one_cell_with_two_nuclei():
    cells = np.zeros((10, 10), dtype=int)
    nuclei = np.zeros((10, 10), dtype=int)

    cells[1:9, 1:9] = 1
    nuclei[2:4, 2:4] = 1
    nuclei[6:8, 6:8] = 2

    # One cell can match only one of the two nuclei.
    assert fraction_of_matched_cells_and_nuclei(cells, nuclei) == pytest.approx(0.5)


def test_nucleus_partly_outside_cell():
    cells = np.zeros((10, 10), dtype=int)
    nuclei = np.zeros((10, 10), dtype=int)

    cells[2:8, 2:8] = 1
    nuclei[3:7, 6:9] = 1

    # The nucleus overlaps the cell interior and can still be matched.
    assert fraction_of_matched_cells_and_nuclei(cells, nuclei) == pytest.approx(1.0)


def test_nonconsecutive_labels():
    cells = np.zeros((10, 10), dtype=int)
    nuclei = np.zeros((10, 10), dtype=int)

    cells[1:9, 1:9] = 7
    nuclei[3:6, 3:6] = 42

    assert fraction_of_matched_cells_and_nuclei(cells, nuclei) == pytest.approx(1.0)


def test_background_is_ignored():
    cells = np.zeros((10, 10), dtype=int)
    nuclei = np.zeros((10, 10), dtype=int)

    assert np.isnan(fraction_of_matched_cells_and_nuclei(cells, nuclei))


def test_shape_mismatch_raises():
    cells = np.zeros((10, 10), dtype=int)
    nuclei = np.zeros((8, 10), dtype=int)

    with pytest.raises(ValueError, match="same shape"):
        fraction_of_matched_cells_and_nuclei(cells, nuclei)


def test_non_integer_labels_raise():
    cells = np.zeros((10, 10), dtype=float)
    nuclei = np.zeros((10, 10), dtype=int)

    with pytest.raises(TypeError, match="integer"):
        fraction_of_matched_cells_and_nuclei(cells, nuclei)


def test_negative_labels_raise():
    cells = np.zeros((10, 10), dtype=int)
    nuclei = np.zeros((10, 10), dtype=int)
    cells[0, 0] = -1

    with pytest.raises(ValueError, match="non-negative"):
        fraction_of_matched_cells_and_nuclei(cells, nuclei)
