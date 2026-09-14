"""The time-domain toy preserves its likelihood across Wheel representations."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from enchilada import DataCovariance, TranslatedCovariance, Wheel, transform


@pytest.fixture
def toy():
    path = Path(__file__).resolve().parent.parent / "examples" / "toy_fit.py"
    spec = importlib.util.spec_from_file_location("toy_fit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sine_rejects_translated_statistical_mask_with_likelihood_error(toy):
    class FlatPSD:
        def psd(self, frequencies, channel=None):
            return np.ones_like(frequencies)

    data = toy.make_observed()
    # The PSD adapter excludes Fourier DC, a nonlocal exclusion in time.
    # A sample-wise white-noise likelihood must not silently count that mode.
    noise = transform(DataCovariance.from_psd(data, FlatPSD()), "time")
    wheel = Wheel(data, initial_noise_covariance=noise, random_seed=0)
    block = toy.SineBlock("slow", 0.004)
    wheel.add(
        block,
        initial_block_result=toy.draw_sine_prior(
            block, data, rng=np.random.default_rng(0)
        ),
        data_domain="time",
    )
    previous = wheel.ledger["slow"]
    with pytest.raises(ValueError, match="time-domain.*all samples active"):
        wheel.run(1)
    assert wheel.ledger["slow"].model_parameters == previous.model_parameters
    np.testing.assert_array_equal(
        wheel.ledger["slow"].tdi_signal_contribution["A"],
        previous.tdi_signal_contribution["A"],
    )


@pytest.mark.parametrize("canonical_domain", ["frequency", "wdm"])
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
                initial_block_result=toy.draw_sine_prior(
                    block, data, rng=initialization_rng
                ),
                data_domain="time",
            )
        noise_block = toy.WhiteNoiseBlock("noise")
        wheel.add(
            noise_block,
            initial_block_result=toy.draw_white_noise_prior(
                noise_block, data, rng=initialization_rng
            ),
            data_domain="time",
        )
        chain = []

        def collect(cycle, campaign, samples=chain):
            samples.append(
                [
                    campaign.ledger["slow"].model_parameters["amplitude"],
                    campaign.ledger["fast"].model_parameters["amplitude"],
                    campaign.ledger["noise"].model_parameters["sigma"],
                ]
            )

        wheel.run(6, on_cycle_complete=collect)
        chains.append(chain)
        wheels.append(wheel)

    np.testing.assert_allclose(chains[1], chains[0], rtol=1e-12, atol=1e-12)
    assert isinstance(wheels[1].noise_covariance, TranslatedCovariance)
    assert wheels[1].noise_covariance.data_domain == canonical_domain
    assert wheels[1].noise_covariance.source_covariance.data_domain == "time"
    restored_residual = transform(wheels[1].working_residual, "time")
    np.testing.assert_allclose(
        restored_residual.channel_data["A"],
        wheels[0].working_residual.channel_data["A"],
        rtol=1e-11,
        atol=1e-11,
    )
