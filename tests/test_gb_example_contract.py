"""GB example boundaries; waveform substitution is not native GBGPU validation."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from enchilada import DataCovariance, L1Data, Wheel, transform


@pytest.fixture
def gb_model(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples" / "gb_model.py"
    spec = importlib.util.spec_from_file_location("gb_example_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The optional native waveform library is outside this contract test.
    monkeypatch.setattr(module, "_new_gb", object)

    def waveform(gb, params, angles, duration, dt, width):
        signal = params[0] * np.exp(1j * params[3]) * np.ones(width)
        return 8, signal, 0.5 * signal

    monkeypatch.setattr(module, "gb_template", waveform)
    return module


@pytest.fixture
def spectral_context():
    data = L1Data(
        channel_data={"A": np.zeros(33, dtype=complex)},
        sample_rate_hz=1.0,
        num_time_samples=64,
        channel_names=("A",),
        tdi_generation="1.5",
        physical_observable="fractional_frequency",
        data_domain="frequency",
    )

    class UnitPSD:
        def psd(self, frequencies, channel=None):
            return np.ones_like(frequencies)

    return data, DataCovariance.from_psd(data, UnitPSD())


def make_block(model):
    return model.GBBlock(
        bounds=[[0.1, 2.0], [0.0, 2.0], [0.0, 0.5], [0.0, 6.0]],
        angles=(0.0, 0.0, 0.0, 0.0),
        n_walkers=12,
        steps_per_cycle=2,
        band=4,
    )


def test_gb_prior_keeps_continuation_on_its_block(gb_model, spectral_context):
    from dataclasses import fields

    block = make_block(gb_model)
    other = make_block(gb_model)
    assert hasattr(block, "state")
    assert block.state is not other.state
    data, noise = spectral_context
    published = gb_model.draw_gb_prior(block, data, noise, rng=np.random.default_rng(0))
    assert {field.name for field in fields(published)} == {
        "template",
        "covariance_matrix",
        "parameters",
    }
    assert block.state.coords["model_0"].shape == (1, 12, 1, 4)
    assert block.state.samples.shape == (0, 4)
    assert block.state.updates == 0
    assert other.state.coords == {}


@pytest.mark.parametrize("translated", [False, True])
def test_gb_rejects_covariance_without_native_frequency_psd(
    gb_model, spectral_context, translated
):
    data, _ = spectral_context
    time_noise = DataCovariance.from_variance(transform(data, "time"), 1.0)
    noise = transform(time_noise, "frequency") if translated else time_noise
    with pytest.raises(ValueError, match="native frequency.*covariance"):
        gb_model.draw_gb_prior(
            make_block(gb_model), data, noise, rng=np.random.default_rng(0)
        )


def test_gb_rejects_covariance_on_a_different_grid(gb_model, spectral_context):
    data, noise = spectral_context
    with pytest.raises(ValueError, match="sample_rate_hz.*match"):
        gb_model.draw_gb_prior(
            make_block(gb_model),
            data,
            replace(noise, sample_rate_hz=2.0),
            rng=np.random.default_rng(0),
        )


@pytest.mark.parametrize("storage_domain", ["time", "wdm"])
def test_gb_frequency_registration_restores_native_spectral_covariance(
    gb_model, spectral_context, storage_domain
):
    if storage_domain == "wdm":
        pytest.importorskip("wdm")
    data, noise = spectral_context
    options = {"num_frequency_divisions": 8} if storage_domain == "wdm" else {}
    observed = transform(data, storage_domain, **options)
    wheel = Wheel(
        observed,
        initial_noise_covariance=transform(noise, storage_domain, **options),
        random_seed=0,
    )
    block = make_block(gb_model)
    initial = gb_model.draw_gb_prior(block, data, noise, rng=np.random.default_rng(0))

    class ContextProbe:
        """Check the GB likelihood inputs without requiring optional Eryn."""

        name = block.name

        def sample(
            self, conditional_residual, noise_covariance, current_block_state, *, rng
        ):
            block._context(conditional_residual, noise_covariance)
            return current_block_state

    wheel.add(ContextProbe(), initial_block_state=initial, data_domain="frequency")
    wheel.run(1)
    result = wheel.ledger["gb"]
    assert result.template["A"].shape == observed.channel_data["A"].shape
    frequency_result = transform(result, "frequency", reference_data=observed)
    assert np.linalg.norm(frequency_result.template["A"]) > 0
    np.testing.assert_allclose(
        transform(wheel.residual(), "frequency").channel_data["A"],
        -frequency_result.template["A"],
        atol=1e-12,
    )
