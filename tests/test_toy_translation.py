"""The time-domain toy preserves its likelihood across Wheel representations."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from enchilada import Wheel, transform


@pytest.fixture
def toy():
    path = Path(__file__).resolve().parent.parent / "examples" / "toy_fit.py"
    spec = importlib.util.spec_from_file_location("toy_fit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("canonical_domain", ["frequency", "wdm"])
@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="covariance translation seam, design spec section 6; "
    "assumes the layer returns the diagonal form",
)
def test_toy_time_blocks_preserve_draws_on_translated_observations(
    toy, canonical_domain
):
    if canonical_domain == "wdm":
        pytest.importorskip("wdm", reason="WDM examples require the optional backend")
    data = toy.make_observed()
    options = {"num_frequency_divisions": 8} if canonical_domain == "wdm" else {}
    translated_data = transform(data, canonical_domain, **options)
    chains = []
    wheels = []
    for observed in (data, translated_data):
        initialization_rng = np.random.default_rng(42)
        wheel = Wheel(observed, random_seed=42)
        for name, frequency in (("slow", 0.004), ("fast", 0.011)):
            block = toy.SineBlock(name, frequency)
            wheel.add(
                block,
                initial_block_state=toy.draw_sine_prior(
                    block, data, rng=initialization_rng
                ),
                data_domain="time",
            )
        noise_block = toy.WhiteNoiseBlock("noise")
        wheel.add(
            noise_block,
            initial_block_state=toy.draw_white_noise_prior(
                noise_block, data, rng=initialization_rng
            ),
            data_domain="time",
        )
        chain = []

        def collect(cycle, campaign, samples=chain):
            samples.append(
                [
                    campaign.ledger["slow"].parameters["amplitude"],
                    campaign.ledger["fast"].parameters["amplitude"],
                    campaign.ledger["noise"].parameters["sigma"],
                ]
            )

        wheel.run(6, on_cycle_complete=collect)
        chains.append(chain)
        wheels.append(wheel)

    np.testing.assert_allclose(chains[1], chains[0], rtol=1e-12, atol=1e-12)
    # Read the translated wheel's covariance back on the time grid of the
    # time-domain reference wheel, so WDM storage is comparable too.
    restored_covariance = transform(
        wheels[1].covariance, "time", reference_data=translated_data
    )
    assert restored_covariance.noise_variance("A", data) == pytest.approx(
        wheels[0].covariance.noise_variance("A", data), rel=1e-12
    )
    restored_residual = transform(wheels[1].working_residual, "time")
    np.testing.assert_allclose(
        restored_residual.channel_data["A"],
        wheels[0].working_residual.channel_data["A"],
        rtol=1e-11,
        atol=1e-11,
    )
