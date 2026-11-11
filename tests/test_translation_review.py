"""Regression cases found while independently reviewing object translation."""

import importlib.util

import numpy as np
import pytest

from enchilada import BlockState, Covariance, Wheel
from enchilada.translation import transform


@pytest.mark.skipif(
    importlib.util.find_spec("wdm") is None, reason="local WDM backend is optional"
)
@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="covariance translation seam, design spec section 6",
)
def test_noise_only_wdm_result_can_reuse_grid_with_compatible_time_reference(observed):
    covariance = transform(
        Covariance.from_variance(observed, 1.0),
        "wdm",
        num_frequency_divisions=4,
        reference_data=observed,
    )
    original = BlockState(covariance=covariance, parameters={"variance": 1.0})
    wdm_reference = transform(observed, "wdm", num_frequency_divisions=4)
    converted = transform(original, "wdm", reference_data=wdm_reference)
    assert converted.parameters == original.parameters
    assert converted.template is None
    assert converted.covariance is not covariance


def test_omitted_domain_and_explicit_same_domain_both_receive_the_covariance(observed):
    received = []

    class RecordInputs:
        def __init__(self, name):
            self.name = name

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            covariance.check_compatible(conditional_residual)
            received.append(("sample", self.name, conditional_residual.data_domain))
            return current_block_state

    wheel = Wheel(observed, Covariance.from_variance(observed, 2.0))
    wheel.add(RecordInputs("default"), initial_block_state=BlockState())
    wheel.add(
        RecordInputs("explicit"), initial_block_state=BlockState(), data_domain="time"
    )
    assert received == []
    wheel.run(1)
    assert received == [("sample", "default", "time"), ("sample", "explicit", "time")]
    assert wheel.covariance.noise_variance("A", observed) == 2.0


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

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            amplitude = rng.normal()
            if fail[0]:
                conditional_residual.channel_data["A"][:] = 1e9
                current_block_state.parameters["coordinates"][0] = -8.0
                signal = {
                    name: np.zeros_like(values)
                    for name, values in conditional_residual.channel_data.items()
                }
                signal["A"][0] = 1j
                return BlockState(template=signal)
            return BlockState(
                parameters={"amplitude": amplitude, "coordinates": np.array([3.0])}
            )

    actual = Wheel(canonical, random_seed=7)
    control = Wheel(canonical, random_seed=7)
    for wheel in (actual, control):
        wheel.add(
            MutateInputs(),
            initial_block_state=BlockState(
                parameters={"amplitude": 0.0, "coordinates": np.array([3.0])}
            ),
            data_domain="frequency",
        )
    previous_amplitude = actual.ledger["mutate"].parameters["amplitude"]
    with pytest.raises(ValueError, match="DC coefficient must be real"):
        actual.run(1)
    assert actual.ledger["mutate"].parameters["amplitude"] == previous_amplitude
    np.testing.assert_array_equal(
        actual.ledger["mutate"].parameters["coordinates"], [3.0]
    )
    for name in canonical.channel_names:
        np.testing.assert_array_equal(
            actual.observed_data.channel_data[name], canonical.channel_data[name]
        )
        np.testing.assert_array_equal(
            actual.residual().channel_data[name], canonical.channel_data[name]
        )
    fail[0] = False
    actual.run(1)
    control.run(1)
    assert (
        actual.ledger["mutate"].parameters["amplitude"]
        == control.ledger["mutate"].parameters["amplitude"]
    )
