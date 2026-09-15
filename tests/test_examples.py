"""The shipped examples run, and the toy fit actually converges to truth."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def load_example(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_runs_clean():
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / "demo.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "residual RMS" in proc.stdout


def test_toy_fit_converges_to_truth():
    toy = load_example("toy_fit")
    results = toy.run_toy_fit(n_cycles=200, burn_in=80, seed=0)
    truth = {
        "slow": toy.TRUTH["slow"],
        "fast": toy.TRUTH["fast"],
        "noise": toy.TRUTH["sigma"],
    }
    for name, (mean, std) in results.items():
        assert std > 0.0
        assert mean == pytest.approx(truth[name], abs=max(5 * std, 0.05)), (
            f"{name}: posterior {mean} +/- {std} vs truth {truth[name]}"
        )


def test_notebook_code_cells_execute():
    nb = json.loads((EXAMPLES / "demo.ipynb").read_text())
    code = "\n\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )
    pytest.importorskip("scipy", reason="notebook orbit cells need scipy")
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("notebook_path", sorted(EXAMPLES.glob("*.ipynb")))
def test_all_notebooks_contain_valid_python(notebook_path):
    """Check even optional notebooks when their scientific stack is absent."""
    notebook = json.loads(notebook_path.read_text())
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells, f"{notebook_path.name} has no executable example"
    for index, cell in enumerate(code_cells, 1):
        compile("".join(cell["source"]), f"{notebook_path}:cell-{index}", "exec")


def test_toy_prior_is_random_and_campaigns_share_configuration_only():
    from enchilada import Wheel

    toy = load_example("toy_fit")
    signal = toy.SineBlock(name="slow", freq=0.004)
    noise = toy.WhiteNoiseBlock(name="noise")
    observed = toy.make_observed()
    campaigns = [Wheel(observed, random_seed=seed) for seed in (42, 42, 43)]
    for wheel, initialization_seed in zip(campaigns, (42, 42, 43), strict=True):
        initialization_rng = np.random.default_rng(initialization_seed)
        wheel.add(
            signal,
            initial_block_result=toy.draw_sine_prior(
                signal, observed, rng=initialization_rng
            ),
        )
        wheel.add(
            noise,
            initial_block_result=toy.draw_white_noise_prior(
                noise, observed, rng=initialization_rng
            ),
        )
    priors = [w.ledger["slow"].model_parameters["amplitude"] for w in campaigns]
    assert priors[0] == priors[1]
    assert priors[0] != priors[2]
    assert priors[0] != 0.0
    a_before, n_before = vars(signal).copy(), vars(noise).copy()
    campaigns[0].run(3)
    campaigns[1].run(3)
    assert (
        campaigns[0].ledger["slow"].model_parameters
        == campaigns[1].ledger["slow"].model_parameters
    )
    assert (
        campaigns[0].ledger["noise"].model_parameters
        == campaigns[1].ledger["noise"].model_parameters
    )
    assert vars(signal) == a_before
    assert vars(noise) == n_before


def test_sine_conditional_includes_declared_amplitude_prior():
    from enchilada import DataCovariance, L1Data

    toy = load_example("toy_fit")
    data = L1Data(
        channel_data={"A": np.array([0.0, 4.0, 0.0, -4.0])},
        sample_rate_hz=1.0,
        channel_names=("A",),
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
    )
    noise = DataCovariance.from_variance(data, 2.0)
    signal = toy.SineBlock("signal", 0.25, prior_std=0.5)
    prior = toy.draw_sine_prior(signal, data, rng=np.random.default_rng(0))
    draws = np.array(
        [
            signal.sample(
                data, noise, prior, rng=np.random.default_rng(seed)
            ).model_parameters["amplitude"]
            for seed in range(4000)
        ]
    )
    # <s,s>=2, <s,d>=8: precision=2/2 + 1/0.25=5, mean=(8/2)/5.
    assert draws.mean() == pytest.approx(0.8, abs=0.025)
    assert draws.var() == pytest.approx(0.2, abs=0.015)


def test_sine_rejects_covariance_its_likelihood_cannot_use():
    from dataclasses import replace

    from enchilada import DataCovariance

    toy = load_example("toy_fit")
    observed = toy.make_observed()
    data = replace(
        observed,
        channel_names=("A", "E"),
        channel_data={"A": observed.channel_data["A"], "E": observed.channel_data["A"]},
    )
    signal = toy.SineBlock("signal", 0.004)
    noise = DataCovariance.from_variance(data, 1.0)
    varying = noise.covariance_matrix.copy()
    varying[:, 0, 1] = varying[:, 1, 0] = 0.5
    prior = toy.draw_sine_prior(signal, data, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="uncorrelated"):
        signal.sample(
            data,
            replace(noise, covariance_matrix=varying),
            prior,
            rng=np.random.default_rng(1),
        )


@pytest.fixture
def gb_model_with_test_waveform(monkeypatch):
    # Only the external waveform is replaced with a small deterministic model.
    model = load_example("gb_model")
    monkeypatch.setattr(model, "_new_gb", object)

    def test_waveform(gb, params, angles, Tobs, dt, NB):
        amp, f0, fdot, phi0 = params
        si = 5 + int(f0)
        bins = np.arange(NB)
        h = amp * np.exp(1j * (phi0 + fdot * bins))
        return si, h, 0.5 * h

    monkeypatch.setattr(model, "gb_template", test_waveform)
    return model


@pytest.fixture
def gb_model_with_eryn(gb_model_with_test_waveform):
    # Exercise real Eryn continuation without requiring GBGPU's native backend.
    pytest.importorskip(
        "eryn.ensemble", reason="GB sampler continuation requires optional Eryn"
    )
    return gb_model_with_test_waveform


def gb_test_context():
    from enchilada import DataCovariance, L1Data

    data = L1Data(
        channel_data={"A": np.ones(33, dtype=complex)},
        sample_rate_hz=1.0,
        num_time_samples=64,
        channel_names=("A",),
        tdi_generation="1.5",
        physical_observable="fractional_frequency",
        data_domain="frequency",
    )

    class UnitPSD:
        def psd(self, freqs, channel=None):
            return np.ones_like(freqs)

    return data, DataCovariance.from_psd(data, UnitPSD())


@pytest.mark.parametrize("num_time_samples", [63, 64])
def test_gb_injection_noise_matches_real_fft_statistics(
    gb_model_with_test_waveform, num_time_samples
):
    from enchilada import L1Data

    class ConstantPSD:
        def psd(self, frequencies):
            return np.full_like(frequencies, 2.0)

    duration = float(num_time_samples)
    draws = []
    for seed in range(2000):
        channel_data, _ = gb_model_with_test_waveform.inject_gb(
            truth=(0.0, 0.0, 0.0, 0.0),
            angles=(0.0, 0.0, 0.0, 0.0),
            Tobs=duration,
            dt=1.0,
            n_samples=num_time_samples,
            channels=("A", "E"),
            noise=ConstantPSD(),
            NB=4,
            seed=seed,
        )
        draws.extend(channel_data.values())
    spectra = np.asarray(draws)
    np.testing.assert_array_equal(spectra[:, 0], 0.0)
    # Interior complex bins split S/(2 df) equally between real and imaginary.
    component_variance = 2.0 * duration / 4.0
    assert spectra[:, 1].real.var() == pytest.approx(component_variance, rel=0.08)
    assert spectra[:, 1].imag.var() == pytest.approx(component_variance, rel=0.08)
    if num_time_samples % 2 == 0:
        # Nyquist has one real degree of freedom with the full S/(2 df) variance.
        np.testing.assert_array_equal(spectra[:, -1].imag, 0.0)
        assert spectra[:, -1].real.var() == pytest.approx(
            2.0 * component_variance, rel=0.08
        )
    else:
        assert spectra[:, -1].real.var() == pytest.approx(component_variance, rel=0.08)
        assert spectra[:, -1].imag.var() == pytest.approx(component_variance, rel=0.08)
    data = L1Data(
        channel_data=channel_data,
        sample_rate_hz=1.0,
        num_time_samples=num_time_samples,
        channel_names=("A", "E"),
        tdi_generation="1.5",
        physical_observable="fractional_frequency",
        data_domain="frequency",
    )
    for channel_name in data.channel_names:
        np.testing.assert_allclose(
            data.to_time().to_frequency().channel_data[channel_name],
            data.channel_data[channel_name],
            atol=1e-12,
        )


def test_gb_eryn_restart_and_global_rng_isolation(gb_model_with_eryn):
    from copy import deepcopy

    gb = gb_model_with_eryn
    data, noise = gb_test_context()
    settings = dict(
        bounds=[[0.1, 2.0], [0.0, 2.0], [0.0, 0.5], [0.0, 6.0]],
        angles=(0.0, 0.0, 0.0, 0.0),
        n_walkers=12,
        band=4,
    )
    short = gb.GBBlock(**settings, steps_per_cycle=2)
    long = gb.GBBlock(**settings, steps_per_cycle=4)
    original_config = deepcopy(vars(short))
    prior = gb.draw_gb_prior(short, data, noise, rng=np.random.default_rng(2))
    prior_copy = deepcopy(prior.sampler_state["coords"])
    global_rng_before = np.random.get_state()
    first = short.sample(data, noise, prior, rng=np.random.default_rng(3))
    # A fresh block must continue from the BlockResult without retained resources.
    restarted = gb.GBBlock(**settings, steps_per_cycle=2).sample(
        data, noise, first, rng=np.random.default_rng(999)
    )
    uninterrupted = long.sample(data, noise, prior, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(
        restarted.sampler_state["coords"]["model_0"],
        uninterrupted.sampler_state["coords"]["model_0"],
    )
    np.testing.assert_array_equal(
        prior.sampler_state["coords"]["model_0"], prior_copy["model_0"]
    )
    assert vars(short) == original_config
    assert restarted.sampler_state["samples"].shape == (24, 4)
    np.testing.assert_array_equal(np.random.get_state()[1], global_rng_before[1])
    assert np.random.get_state()[2:] == global_rng_before[2:]


def test_gb_recomputes_likelihood_for_changed_residual(gb_model_with_eryn):
    from dataclasses import replace

    gb = gb_model_with_eryn
    data, noise = gb_test_context()
    block = gb.GBBlock(
        bounds=[[0.1, 2.0], [0.0, 2.0], [0.0, 0.5], [0.0, 6.0]],
        angles=(0.0, 0.0, 0.0, 0.0),
        n_walkers=12,
        steps_per_cycle=3,
        band=4,
    )
    prior = gb.draw_gb_prior(block, data, noise, rng=np.random.default_rng(4))
    first = block.sample(data, noise, prior, rng=np.random.default_rng(5))
    changed = replace(data, channel_data={"A": 1e5 * data.channel_data["A"]})
    updated = block.sample(changed, noise, first, rng=np.random.default_rng(6))
    same = block.sample(data, noise, first, rng=np.random.default_rng(6))
    assert not np.array_equal(
        updated.sampler_state["coords"]["model_0"],
        same.sampler_state["coords"]["model_0"],
    )


@pytest.mark.parametrize("invalid", ["frequency_covariance", "inactive_samples"])
def test_sine_requires_fully_observed_time_white_noise(invalid):
    from dataclasses import replace

    from enchilada import DataCovariance

    toy = load_example("toy_fit")
    data = toy.make_observed()
    block = toy.SineBlock("slow", 0.004)
    prior = toy.draw_sine_prior(block, data, rng=np.random.default_rng(0))
    noise = DataCovariance.from_variance(data, 1.0)
    if invalid == "frequency_covariance":

        class UnitPSD:
            def psd(self, f, channel=None):
                return np.ones_like(f)

        noise = DataCovariance.from_psd(data, UnitPSD())
    else:
        active = noise.active_mask.copy()
        active[0] = False
        noise = replace(noise, active_mask=active)
    with pytest.raises(ValueError, match="time-domain.*all samples active"):
        block.sample(data, noise, prior, rng=np.random.default_rng(1))


def test_white_noise_prior_requires_time_data():
    toy = load_example("toy_fit")
    block = toy.WhiteNoiseBlock("noise")
    with pytest.raises(ValueError, match="time-domain"):
        toy.draw_white_noise_prior(
            block, toy.make_observed().to_frequency(), rng=np.random.default_rng(0)
        )
