"""Two conjugate Gibbs source blocks plus sampled white noise.

Blocks retain fixed model configuration. Wheel owns the current parameters,
BlockResult state, covariance, and sampling random stream. Caller-side helpers draw
complete initial results using a separate RNG; an on_cycle_complete callback collects
posterior chains outside the blocks. These blocks request time-domain inputs;
Wheel can also run them on Fourier or WDM observations using data_domain="time".
Their likelihood requires native, fully active white time noise. Run with:

    uv run --no-sync python examples/toy_fit.py
"""

import numpy as np

from enchilada import BlockResult, DataCovariance, L1Data, TranslatedCovariance, Wheel


class SineBlock:
    """A known-frequency sinusoid with amplitude prior N(0, prior_std**2).

    The conditional combines this proper Gaussian prior with the white-noise
    likelihood. Its waveform basis is derived from the supplied grid each call.
    """

    def __init__(self, name: str, freq: float, prior_std: float = 10.0):
        if not np.isfinite(prior_std) or prior_std <= 0:
            raise ValueError("prior_std must be finite and positive")
        self.name, self.freq, self.prior_std = name, freq, prior_std

    def _basis(self, conditional_residual: L1Data) -> np.ndarray:
        if conditional_residual.data_domain != "time":
            raise ValueError("SineBlock requires time-domain data")
        t = (
            conditional_residual.start_time_gps
            + np.arange(conditional_residual.num_time_samples) * conditional_residual.dt
        )
        return np.sin(2.0 * np.pi * self.freq * t)

    def _render(self, conditional_residual: L1Data, amplitude: float) -> BlockResult:
        basis = self._basis(conditional_residual)
        return conditional_residual.block_result(
            {ch: amplitude * basis for ch in conditional_residual.channel_names},
            model_parameters={"amplitude": amplitude},
        )

    def sample(
        self,
        conditional_residual: L1Data,
        noise_covariance: DataCovariance | TranslatedCovariance | None,
        current_block_result: BlockResult,
        *,
        rng: np.random.Generator,
    ) -> BlockResult:
        if noise_covariance is None:
            raise ValueError("SineBlock requires a white-noise covariance")
        if (
            noise_covariance.data_domain != "time"
            or noise_covariance.active_mask is None
            or not noise_covariance.active_mask.all()
        ):
            raise ValueError(
                "SineBlock requires native time-domain noise with all samples active; "
                "translated covariance can contain nonlocal correlations or exclusions"
            )
        if np.any(
            noise_covariance.covariance_matrix[
                :, ~np.eye(len(conditional_residual.channel_names), dtype=bool)
            ]
        ):
            raise ValueError("SineBlock requires uncorrelated channels")
        basis = self._basis(conditional_residual)
        # The toy uses a single channel with constant white-noise variance.
        # Include every channel if the same independent model is used on more.
        precision = 1.0 / self.prior_std**2
        weighted_data = 0.0
        for ch in conditional_residual.channel_names:
            variance = noise_covariance.noise_variance(channel_name=ch)
            precision += float(basis @ basis) / variance
            weighted_data += (
                float(conditional_residual.channel_data[ch] @ basis) / variance
            )
        amplitude = rng.normal(weighted_data / precision, np.sqrt(1.0 / precision))
        return self._render(conditional_residual, float(amplitude))


class WhiteNoiseBlock:
    """White-noise variance with a proper inverse-gamma(shape, scale) prior."""

    def __init__(self, name: str, shape: float = 2.0, scale: float = 1.0):
        if not np.isfinite([shape, scale]).all() or min(shape, scale) <= 0:
            raise ValueError(
                "inverse-gamma shape and scale must be finite and positive"
            )
        self.name, self.shape, self.scale = name, shape, scale

    def _render(self, conditional_residual: L1Data, variance: float) -> BlockResult:
        if conditional_residual.data_domain != "time":
            raise ValueError("WhiteNoiseBlock requires time-domain data")
        return BlockResult(
            noise_covariance=DataCovariance.from_variance(
                reference_data=conditional_residual, time_sample_variance=variance
            ),
            model_parameters={"sigma": float(np.sqrt(variance))},
        )

    def sample(
        self,
        conditional_residual: L1Data,
        noise_covariance: DataCovariance | TranslatedCovariance | None,
        current_block_result: BlockResult,
        *,
        rng: np.random.Generator,
    ) -> BlockResult:
        if conditional_residual.data_domain != "time":
            raise ValueError("WhiteNoiseBlock requires time-domain data")
        # The noise block sees observations minus every signal contribution.
        n_total = sum(arr.size for arr in conditional_residual.channel_data.values())
        ssr = sum(
            float(arr @ arr) for arr in conditional_residual.channel_data.values()
        )
        shape = self.shape + 0.5 * n_total
        scale = self.scale + 0.5 * ssr
        return self._render(conditional_residual, float(scale / rng.gamma(shape)))


def draw_sine_prior(
    block: SineBlock, reference_data: L1Data, *, rng: np.random.Generator
) -> BlockResult:
    """Caller-side prior draw on the time grid requested by this model."""
    return block._render(reference_data, float(rng.normal(0.0, block.prior_std)))


def draw_white_noise_prior(
    block: WhiteNoiseBlock, reference_data: L1Data, *, rng: np.random.Generator
) -> BlockResult:
    """Prepare a complete initial noise result before registering the block."""
    return block._render(reference_data, float(block.scale / rng.gamma(block.shape)))


TRUTH = {"slow": 3.0, "fast": 2.0, "sigma": 0.5}


def make_observed(seed: int = 0) -> L1Data:
    """Two sinusoids in white noise on a single channel."""
    rng = np.random.default_rng(seed)
    fs, n = 0.1, 4096
    t = np.arange(n) / fs
    data = (
        TRUTH["slow"] * np.sin(2.0 * np.pi * 0.004 * t)
        + TRUTH["fast"] * np.sin(2.0 * np.pi * 0.011 * t)
        + rng.normal(0.0, TRUTH["sigma"], n)
    )
    return L1Data(
        channel_data={"A": data},
        sample_rate_hz=fs,
        channel_names=("A",),
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
        start_time_gps=0.0,
    )


def run_toy_fit(n_cycles: int = 300, burn_in: int = 100, seed: int = 0):
    """Run the fit; return {name: (posterior_mean, posterior_std)}."""
    if not 0 <= burn_in < n_cycles:
        raise ValueError("burn_in must be nonnegative and smaller than n_cycles")
    observed = make_observed(seed)
    # Distinct child streams keep initialization and sampling reproducible
    # without replaying each other's draws or the observation-noise stream.
    initialization_seed, sampling_seed = np.random.SeedSequence(seed).spawn(2)
    initialization_rng = np.random.default_rng(initialization_seed)
    wheel = Wheel(observed, random_seed=int(sampling_seed.generate_state(1)[0]))
    for name, frequency in (("slow", 0.004), ("fast", 0.011)):
        block = SineBlock(name=name, freq=frequency)
        wheel.add(
            block,
            initial_block_result=draw_sine_prior(
                block, observed, rng=initialization_rng
            ),
            data_domain="time",
        )
    noise = WhiteNoiseBlock(name="noise")
    wheel.add(
        noise,
        initial_block_result=draw_white_noise_prior(
            noise, observed, rng=initialization_rng
        ),
        data_domain="time",
    )
    chains: dict[str, list[float]] = {name: [] for name in ("slow", "fast", "noise")}

    def collect(cycle: int, w: Wheel) -> None:
        for name in chains:
            parameter = "sigma" if name == "noise" else "amplitude"
            chains[name].append(w.ledger[name].model_parameters[parameter])
        if (cycle + 1) % 100 == 0:
            rms = float(np.sqrt(np.mean(w.residual().channel_data["A"] ** 2)))
            print(
                f"cycle {cycle + 1:4d}: full-residual RMS = {rms:.4f}  "
                f"(sigma draw = {chains['noise'][-1]:.4f})"
            )

    wheel.run(n_cycles, on_cycle_complete=collect)
    results = {}
    for name, samples in chains.items():
        chain = np.asarray(samples[burn_in:])
        results[name] = (float(chain.mean()), float(chain.std()))
    return results


def main() -> None:
    print(
        f"truth: slow={TRUTH['slow']}, fast={TRUTH['fast']}, sigma={TRUTH['sigma']}\n"
    )
    results = run_toy_fit()
    print()
    for name, (mean, std) in results.items():
        key = "sigma" if name == "noise" else name
        print(f"{name:6s} posterior: {mean:.4f} +/- {std:.4f}   (truth {TRUTH[key]})")


if __name__ == "__main__":
    main()
