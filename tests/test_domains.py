"""WDM grids describe the real, active degrees of freedom without a backend."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest


def test_wdm_grid_is_keyword_only_and_immutable():
    from enchilada.domains import WDMGrid

    with pytest.raises(TypeError):
        WDMGrid(4, 8)
    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=8)
    with pytest.raises(FrozenInstanceError):
        grid.num_frequency_divisions = 8
    assert grid.array_shape == (5, 8)
    assert grid.num_time_samples == 32


@pytest.mark.parametrize("name", ["num_frequency_divisions", "num_time_divisions"])
@pytest.mark.parametrize("value", [0, -2, 3, True, False, 4.0, np.float64(4.0)])
def test_wdm_grid_rejects_invalid_divisions(name, value):
    from enchilada.domains import WDMGrid

    arguments = {"num_frequency_divisions": 4, "num_time_divisions": 8, name: value}
    with pytest.raises(ValueError, match=name):
        WDMGrid(**arguments)


def test_numpy_integer_grid_counts_and_independent_active_masks():
    from enchilada.domains import WDMGrid

    grid = WDMGrid(num_frequency_divisions=np.int64(4), num_time_divisions=np.int32(8))
    mask = grid.active_mask
    assert mask.dtype == bool
    assert mask.shape == (5, 8)
    assert int(mask.sum()) == 32
    np.testing.assert_array_equal(mask[1:-1], True)
    np.testing.assert_array_equal(mask[[0, -1], ::2], True)
    np.testing.assert_array_equal(mask[[0, -1], 1::2], False)
    mask[:] = False
    assert int(grid.active_mask.sum()) == 32


@pytest.mark.parametrize(
    "arguments",
    [
        {"num_frequency_divisions": 4},
        {"num_time_divisions": 8},
        {"num_frequency_divisions": 4, "num_time_divisions": 8},
    ],
)
def test_resolve_wdm_grid_derives_only_the_missing_division(arguments):
    from enchilada.domains import WDMGrid, resolve_wdm_grid

    assert resolve_wdm_grid(32, **arguments) == WDMGrid(
        num_frequency_divisions=4, num_time_divisions=8
    )


@pytest.mark.parametrize(
    "num_time_samples,arguments",
    [
        (32, {}),
        (32, {"num_frequency_divisions": 4, "num_time_divisions": 4}),
        (32, {"num_frequency_divisions": 6}),
        (12, {"num_frequency_divisions": 4}),
        (32, {"num_frequency_divisions": True}),
        (32, {"num_time_divisions": 4.0}),
        (0, {"num_frequency_divisions": 4}),
        (-32, {"num_frequency_divisions": 4}),
        (True, {"num_frequency_divisions": 4}),
        (32.0, {"num_frequency_divisions": 4}),
    ],
)
def test_resolve_wdm_grid_rejects_ambiguous_or_incompatible_counts(
    num_time_samples, arguments
):
    from enchilada.domains import resolve_wdm_grid

    with pytest.raises(ValueError):
        resolve_wdm_grid(num_time_samples, **arguments)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_wdm_coefficients_keep_active_values_and_input_buffers(dtype):
    from enchilada.domains import WDMGrid, check_wdm_coefficients

    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=8)
    coefficients = np.zeros(grid.array_shape, dtype=dtype)
    coefficients[grid.active_mask] = np.arange(grid.num_time_samples)
    original = coefficients.copy()
    check_wdm_coefficients({"A": coefficients}, grid, "data")
    np.testing.assert_array_equal(coefficients, original)


@pytest.mark.parametrize(
    "coefficients,error,match",
    [
        (np.zeros((4, 8)), ValueError, "shape"),
        (np.zeros(40), ValueError, "shape"),
        (np.zeros((5, 8), complex), TypeError, "real"),
        (np.zeros((5, 8), int), TypeError, "floating"),
        (np.ones((5, 8)), ValueError, "inactive"),
    ],
)
def test_wdm_coefficients_reject_data_that_cannot_round_trip(
    coefficients, error, match
):
    from enchilada.domains import WDMGrid, check_wdm_coefficients

    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=8)
    with pytest.raises(error, match=match):
        check_wdm_coefficients({"A": coefficients}, grid, "data")


def test_wdm_coefficients_reject_even_tiny_inactive_values_without_projection():
    from enchilada.domains import WDMGrid, check_wdm_coefficients

    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=8)
    coefficients = np.zeros(grid.array_shape)
    coefficients[0, 1] = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="inactive"):
        check_wdm_coefficients({"A": coefficients}, grid, "data")
    assert coefficients[0, 1] != 0.0
