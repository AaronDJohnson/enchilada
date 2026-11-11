"""Translate data and block states, then run a mixed-domain Wheel.

From the enchilada checkout, with the local WDM checkout beside it:
    uv sync --python 3.13
    uv pip install ../wdm
    uv run --no-sync python examples/domain_translation.py

This example needs only enchilada and the optional local WDM backend.
The zero-signal blocks exercise representation handoffs without fitting a model.
"""

import json
from dataclasses import dataclass, field

import numpy as np

from enchilada import BlockState, L1Data, Wheel, transform


@dataclass
class DomainEcho:
    """A fixed zero-signal model recording the representation it receives."""

    name: str
    expected_domain: str

    @dataclass
    class State:
        num_sample_calls: int = 0
        input_domain: str | None = None

    state: State = field(default_factory=State, init=False)

    def _check_input(self, conditional_residual, covariance):
        if conditional_residual.data_domain != self.expected_domain:
            raise ValueError(f"{self.name}: unexpected data representation")
        # Covariance translation across domains arrives with the translation
        # layer (design spec section 6); this walkthrough runs without one.
        if covariance is not None:
            raise ValueError(f"{self.name}: expected no covariance in this walkthrough")

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        self._check_input(conditional_residual, covariance)
        self.state.num_sample_calls += 1
        self.state.input_domain = conditional_residual.data_domain
        return BlockState()


def maximum_error(expected, actual):
    return max(
        float(np.max(np.abs(expected[channel] - actual[channel])))
        for channel in expected
    )


def run_example():
    rng = np.random.default_rng(7)
    observed = L1Data(
        {name: rng.normal(size=128) for name in ("A", "E")},
        sample_rate_hz=0.5,  # dt = 2 seconds exercises normalization explicitly
        tdi_generation="2.0",
        physical_observable="strain",
    )

    spectrum = transform(observed, "frequency")
    wdm_data = transform(observed, "wdm", num_frequency_divisions=8)
    alternate_grid = transform(observed, "wdm", num_time_divisions=8)

    # A result has no observation-grid metadata: identify its source explicitly.
    signal = observed.block_state(
        template={name: 0.1 * values for name, values in observed.channel_data.items()},
        parameters={"amplitude": 0.1},
    )
    wdm_signal = transform(
        signal, "wdm", num_frequency_divisions=8, reference_data=observed
    )
    restored_signal = transform(wdm_signal, "time", reference_data=wdm_data)

    # Wheel keeps time as the canonical grid while preparing each block's inputs.
    wheel = Wheel(observed, random_seed=3)
    blocks = {}
    for domain in ("time", "frequency", "wdm"):
        initial = BlockState()
        block = DomainEcho(domain, domain)
        blocks[domain] = block
        options = {"num_frequency_divisions": 8} if domain == "wdm" else {}
        wheel.add(
            block,
            initial_block_state=initial,
            data_domain=domain,
            **options,
        )
    wheel.run(num_cycles=2)

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
            signal.template, restored_signal.template
        ),
        "result_parameters": restored_signal.parameters,
        "block_domains": [block.state.input_domain for block in blocks.values()],
        "canonical_domain": wheel.working_residual.data_domain,
        "residual_max_error": maximum_error(
            observed.channel_data, wheel.working_residual.channel_data
        ),
        "sample_calls": {
            name: block.state.num_sample_calls for name, block in blocks.items()
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2))
