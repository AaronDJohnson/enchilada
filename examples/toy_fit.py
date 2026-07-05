"""A complete (toy) blocked-Gibbs fit: two sources plus sampled noise.

Where examples/demo.py shows the plumbing with no-op segments, this example
runs a real Gibbs sampler to convergence on synthetic data, demonstrating the
parts of the protocol the demo leaves out:

- a *signal* segment with a conjugate amplitude draw (`SineSegment`),
  whitening against the current noise via `residual.noise_psd`;
- a *noise* segment (`WhiteNoiseSegment`) whose `noise_model` implements the
  `psd(freqs[, channel])` contract, refreshed by the Wheel every sweep;
- state ownership: each segment keeps its parameters, RNG, and posterior
  chain as plain instance attributes -- the Wheel never sees them, and you
  read results directly off the segment objects you constructed;
- progress reporting through `Wheel.run`'s `on_sweep` callback.

The data are one channel of two sinusoids in white noise -- physically a toy
(amplitudes in arbitrary units), but the Gibbs structure is exactly the real
thing. Run it with:

    uv run python examples/toy_fit.py
"""

import numpy as np

from turntable import Residuals, Wheel


class FlatNoise:
    """White-noise model satisfying the turntable noise contract."""

    def __init__(self, sigma: float, sample_rate: float):
        self.sigma = sigma
        self._fs = sample_rate

    def psd(self, freqs, channel=None):
        # one-sided PSD of white noise with per-sample std `sigma`
        return np.full_like(freqs, 2.0 * self.sigma**2 / self._fs)


class SineSegment:
    """One sinusoidal source with a conjugate Gibbs draw for its amplitude.

    The frequency is treated as known; the amplitude posterior given white
    noise of variance sigma^2 is N(<d, s>/<s, s>, sigma^2/<s, s>), which we
    sample exactly -- no Metropolis machinery needed for the toy.

    Everything this sampler is -- current amplitude, RNG, chain -- is a
    plain instance attribute. The Wheel only ever sees the contribution.
    """

    def __init__(self, name: str, freq: float, seed: int):
        self.name = name
        self.freq = freq
        self.amplitude = 0.0
        self.chain: list[float] = []
        self._rng = np.random.default_rng(seed)
        self._basis: dict[str, np.ndarray] | None = None

    def start(self, observed: Residuals):
        t = observed.epoch + np.arange(observed.n_samples) * observed.dt
        self._basis = {
            ch: np.sin(2.0 * np.pi * self.freq * t) for ch in observed.channels
        }
        return self._contribution()

    def step(self, residual: Residuals):
        # per-sample noise variance from the threaded noise model:
        # sigma^2 = S_onesided * fs / 2 for white noise
        psd = residual.noise_psd(residual.channels[0])
        sigma2 = float(psd[1]) * residual.fs / 2.0
        ch = residual.channels[0]
        s = self._basis[ch]
        # `residual` excludes this segment's own contribution
        ss = float(s @ s)
        mean = float(residual.tdi[ch] @ s) / ss
        self.amplitude = self._rng.normal(mean, np.sqrt(sigma2 / ss))
        self.chain.append(self.amplitude)
        return self._contribution()

    def _contribution(self):
        assert self._basis is not None
        return {ch: self.amplitude * s for ch, s in self._basis.items()}


class WhiteNoiseSegment:
    """Noise segment: conjugate inverse-gamma draw for the white-noise sigma."""

    def __init__(self, name: str, seed: int):
        self.name = name
        self.sigma = 1.0
        self.chain: list[float] = []
        self._rng = np.random.default_rng(seed)
        self._zeros: dict[str, np.ndarray] | None = None
        self._fs: float | None = None

    def start(self, observed: Residuals):
        self._zeros = {
            ch: np.zeros_like(observed.tdi[ch]) for ch in observed.channels
        }
        self._fs = observed.fs
        return {ch: arr.copy() for ch, arr in self._zeros.items()}

    def step(self, residual: Residuals):
        # residual here is data minus every signal segment's model: pure noise
        n_total = sum(arr.size for arr in residual.tdi.values())
        ssr = sum(float(arr @ arr) for arr in residual.tdi.values())
        # inverse-gamma(a, b) posterior with a weak IG(2, 1) prior
        a = 2.0 + 0.5 * n_total
        b = 1.0 + 0.5 * ssr
        self.sigma = float(np.sqrt(b / self._rng.gamma(a)))
        self.chain.append(self.sigma)
        # noise removes nothing from the data; its influence travels
        # through Residuals.noise, not through residual subtraction
        assert self._zeros is not None
        return {ch: arr.copy() for ch, arr in self._zeros.items()}

    def noise_model(self):
        assert self._fs is not None
        return FlatNoise(self.sigma, self._fs)


TRUTH = {"slow": 3.0, "fast": 2.0, "sigma": 0.5}


def make_observed(seed: int = 0) -> Residuals:
    """Two sinusoids in white noise on a single channel."""
    rng = np.random.default_rng(seed)
    fs, n = 0.1, 4096
    t = np.arange(n) / fs
    data = (
        TRUTH["slow"] * np.sin(2.0 * np.pi * 0.004 * t)
        + TRUTH["fast"] * np.sin(2.0 * np.pi * 0.011 * t)
        + rng.normal(0.0, TRUTH["sigma"], n)
    )
    return Residuals(
        tdi={"A": data},
        sample_rate=fs,
        n_samples=n,
        channels=("A",),
        tdi_generation="2.0",
        observable="fractional_frequency",
        epoch=0.0,
    )


def run_toy_fit(n_sweeps: int = 300, burn_in: int = 100, seed: int = 0):
    """Run the fit; returns {name: (posterior_mean, posterior_std)}."""
    slow = SineSegment(name="slow", freq=0.004, seed=seed + 1)
    fast = SineSegment(name="fast", freq=0.011, seed=seed + 2)
    noise = WhiteNoiseSegment(name="noise", seed=seed + 3)

    wheel = Wheel(make_observed(seed))
    wheel.add(slow)
    wheel.add(fast)
    wheel.add(noise)

    def progress(iteration: int, w: Wheel) -> None:
        if (iteration + 1) % 100 == 0:
            rms = float(np.sqrt(np.mean(w.residual().tdi["A"] ** 2)))
            print(
                f"sweep {iteration + 1:4d}: full-residual RMS = {rms:.4f}  "
                f"(sigma draw = {noise.sigma:.4f})"
            )

    wheel.run(n_sweeps, on_sweep=progress)

    # chains live on the segment objects we constructed -- read them directly
    results = {}
    for seg in (slow, fast, noise):
        chain = np.asarray(seg.chain[burn_in:])
        results[seg.name] = (float(chain.mean()), float(chain.std()))
    return results


def main() -> None:
    print(f"truth: slow={TRUTH['slow']}, fast={TRUTH['fast']}, "
          f"sigma={TRUTH['sigma']}\n")
    results = run_toy_fit()
    print()
    truth_by_name = {"slow": TRUTH["slow"], "fast": TRUTH["fast"],
                     "sigma": TRUTH["sigma"]}
    for name, (mean, std) in results.items():
        key = "sigma" if name == "noise" else name
        print(f"{name:6s} posterior: {mean:.4f} +/- {std:.4f}   "
              f"(truth {truth_by_name[key]})")


if __name__ == "__main__":
    main()
