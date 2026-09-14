"""Regression cases found while independently reviewing object translation."""

import importlib.util

import numpy as np
import pytest

from enchilada import BlockResult, DataCovariance, Wheel
from enchilada.translation import transform


@pytest.mark.parametrize("operation", ["apply", "solve", "project", "quadratic_form"])
def test_translated_wdm_covariance_rejects_inactive_data_before_backend_use(
    observed, operation
):
    covariance = transform(
        DataCovariance.from_variance(observed, 1.0),
        "wdm",
        num_frequency_divisions=4,
    )
    values = {
        name: np.zeros(covariance.wdm_grid.array_shape)
        for name in observed.channel_names
    }
    values["A"][0, 1] = 1.0
    with pytest.raises(ValueError, match="inactive"):
        getattr(covariance, operation)(values)


def test_noise_only_wdm_result_can_reuse_grid_with_compatible_time_reference(observed):
    covariance = transform(
        DataCovariance.from_variance(observed, 1.0),
        "wdm",
        num_frequency_divisions=4,
    )
    original = BlockResult(noise_covariance=covariance, sampler_state={"step": 3})
    converted = transform(original, "wdm", reference_data=observed)
    assert converted.noise_covariance.wdm_grid == covariance.wdm_grid
    assert converted.sampler_state == original.sampler_state
    assert converted.tdi_signal_contribution is None
    assert converted.noise_covariance is not covariance


def test_omitted_domain_preserves_inputs_while_explicit_domain_aligns_them(observed):
    received = []

    class RecordInputs:
        def __init__(self, name):
            self.name = name

        def sample(
            self, conditional_residual, noise_covariance, current_block_result, *, rng
        ):
            received.append(
                (
                    "sample",
                    self.name,
                    conditional_residual.data_domain,
                    noise_covariance.data_domain,
                )
            )
            return current_block_result

    covariance = transform(DataCovariance.from_variance(observed, 2.0), "frequency")
    wheel = Wheel(observed, covariance)
    wheel.add(RecordInputs("default"), initial_block_result=BlockResult())
    wheel.add(
        RecordInputs("explicit"), initial_block_result=BlockResult(), data_domain="time"
    )
    assert received == []
    wheel.run(1)
    assert received == [
        ("sample", "default", "time", "frequency"),
        ("sample", "explicit", "time", "time"),
    ]
    assert wheel.noise_covariance.data_domain == "frequency"


@pytest.mark.parametrize(
    "canonical_domain",
    [
        "time",
        pytest.param(
            "wdm",
            marks=pytest.mark.skipif(
                importlib.util.find_spec("wdm") is None,
                reason="local WDM backend is optional",
            ),
        ),
    ],
)
def test_rejected_translated_output_rolls_back_mutated_inputs_and_rng(
    observed, canonical_domain
):
    canonical = (
        observed
        if canonical_domain == "time"
        else transform(observed, "wdm", num_frequency_divisions=8)
    )
    fail = [True]

    class MutateInputs:
        name = "mutate"

        def sample(
            self, conditional_residual, noise_covariance, current_block_result, *, rng
        ):
            draw = rng.normal()
            if fail[0]:
                conditional_residual.channel_data["A"][:] = 1e9
                native = noise_covariance.source_covariance
                native.covariance_matrix.setflags(write=True)
                native.covariance_matrix[:] = -1.0
                current_block_result.sampler_state["coordinates"][0] = -8.0
                signal = {
                    name: np.zeros_like(values)
                    for name, values in conditional_residual.channel_data.items()
                }
                signal["A"][0] = 1j
                return BlockResult(tdi_signal_contribution=signal)
            return BlockResult(
                sampler_state={"draw": draw, "coordinates": np.array([3.0])}
            )

    initial_covariance = DataCovariance.from_variance(observed, 2.0)
    actual = Wheel(canonical, initial_covariance, random_seed=7)
    control = Wheel(canonical, initial_covariance, random_seed=7)
    for wheel in (actual, control):
        wheel.add(
            MutateInputs(),
            initial_block_result=BlockResult(
                sampler_state={"draw": 0.0, "coordinates": np.array([3.0])}
            ),
            data_domain="frequency",
        )
    previous_draw = actual.ledger["mutate"].sampler_state["draw"]
    with pytest.raises(ValueError, match="DC coefficient must be real"):
        actual.run(1)
    assert actual.ledger["mutate"].sampler_state["draw"] == previous_draw
    np.testing.assert_array_equal(
        actual.ledger["mutate"].sampler_state["coordinates"], [3.0]
    )
    for name in canonical.channel_names:
        np.testing.assert_array_equal(
            actual.observed_data.channel_data[name], canonical.channel_data[name]
        )
        np.testing.assert_array_equal(
            actual.residual().channel_data[name], canonical.channel_data[name]
        )
    np.testing.assert_array_equal(
        actual.noise_covariance.covariance_matrix,
        initial_covariance.covariance_matrix,
    )
    fail[0] = False
    actual.run(1)
    control.run(1)
    assert (
        actual.ledger["mutate"].sampler_state["draw"]
        == control.ledger["mutate"].sampler_state["draw"]
    )
