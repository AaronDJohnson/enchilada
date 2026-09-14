"""Translate data, covariance, and results, then run a mixed-domain Wheel.

From the enchilada checkout, with the local WDM checkout beside it:
    uv sync --python 3.13
    uv pip install ../wdm
    uv run --no-sync python examples/domain_translation.py

This example needs only enchilada and the optional local WDM backend.
The zero-signal blocks exercise representation handoffs without fitting a model.
"""

import json
from dataclasses import dataclass

import numpy as np

from enchilada import (
    BlockResult,
    DataCovariance,
    L1Data,
    TranslatedCovariance,
    Wheel,
    transform,
)


@dataclass(frozen=True)
class DomainEcho:
    """A fixed zero-signal model recording the representation it receives."""

    name: str
    expected_domain: str

    def _check_input(self, conditional_residual, noise_covariance):
        if conditional_residual.data_domain != self.expected_domain:
            raise ValueError(f"{self.name}: unexpected data representation")
        if (
            noise_covariance is None
            or noise_covariance.data_domain != self.expected_domain
        ):
            raise ValueError(f"{self.name}: covariance did not follow the data")

    def sample(
        self, conditional_residual, noise_covariance, current_block_result, *, rng
    ):
        self._check_input(conditional_residual, noise_covariance)
        return BlockResult(
            sampler_state={
                "num_sample_calls": (
                    current_block_result.sampler_state["num_sample_calls"] + 1
                )
            },
            metadata={"input_domain": conditional_residual.data_domain},
        )


def maximum_error(expected, actual):
    return max(
        float(np.max(np.abs(expected[channel] - actual[channel])))
        for channel in expected
    )


def run_example():
    rng = np.random.default_rng(7)
    observed = L1Data(
        channel_data={name: rng.normal(size=128) for name in ("A", "E")},
        sample_rate_hz=0.5,  # dt = 2 seconds exercises normalization explicitly
        channel_names=("A", "E"),
        tdi_generation="2.0",
        physical_observable="strain",
    )
    noise = DataCovariance.from_variance(
        reference_data=observed, time_sample_variance=0.25
    )

    spectrum = transform(observed, "frequency")
    wdm_data = transform(observed, "wdm", num_frequency_divisions=8)
    alternate_grid = transform(observed, "wdm", num_time_divisions=8)
    frequency_noise = transform(noise, "frequency")
    wdm_noise = transform(noise, "wdm", num_frequency_divisions=8)
    assert isinstance(wdm_noise, TranslatedCovariance)

    # A result has no observation-grid metadata: identify its source explicitly.
    signal = observed.block_result(
        tdi_signal_contribution={
            name: 0.1 * values for name, values in observed.channel_data.items()
        },
        model_parameters={"amplitude": 0.1},
    )
    wdm_signal = transform(
        signal, "wdm", num_frequency_divisions=8, reference_data=observed
    )
    restored_signal = transform(wdm_signal, "time", reference_data=wdm_data)

    quadratic_forms = {
        "time": noise.quadratic_form(observed.channel_data),
        "frequency": frequency_noise.quadratic_form(spectrum.channel_data),
        "wdm": wdm_noise.quadratic_form(wdm_data.channel_data),
    }
    np.testing.assert_allclose(
        list(quadratic_forms.values()), quadratic_forms["time"], rtol=1e-12
    )

    # Wheel keeps time as the canonical grid while preparing each block's inputs.
    wheel = Wheel(observed, initial_noise_covariance=noise, random_seed=3)
    for domain in ("time", "frequency", "wdm"):
        initial = BlockResult(
            sampler_state={"num_sample_calls": 0},
            metadata={"input_domain": domain},
        )
        options = {"num_frequency_divisions": 8} if domain == "wdm" else {}
        wheel.add(
            DomainEcho(domain, domain),
            initial_block_result=initial,
            data_domain=domain,
            **options,
        )
    wheel.run(num_cycles=2)
    results = wheel.ledger.snapshot()

    return {
        "wdm_shapes": {
            "frequency_selector": list(wdm_data.channel_data["A"].shape),
            "time_selector": list(alternate_grid.channel_data["A"].shape),
        },
        "frequency_roundtrip_max_error": maximum_error(
            observed.channel_data, transform(spectrum, "time").channel_data
        ),
        "wdm_roundtrip_max_error": maximum_error(
            observed.channel_data, transform(wdm_data, "time").channel_data
        ),
        "result_roundtrip_max_error": maximum_error(
            signal.tdi_signal_contribution, restored_signal.tdi_signal_contribution
        ),
        "result_model_parameters": restored_signal.model_parameters,
        "quadratic_forms": quadratic_forms,
        "block_domains": [
            result.metadata["input_domain"] for result in results.values()
        ],
        "canonical_domain": wheel.working_residual.data_domain,
        "residual_max_error": maximum_error(
            observed.channel_data, wheel.working_residual.channel_data
        ),
        "sample_calls": {
            name: result.sampler_state["num_sample_calls"]
            for name, result in results.items()
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2))
