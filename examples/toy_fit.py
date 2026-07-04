"""A complete (toy) blocked-Gibbs fit: two sources plus sampled noise.

Where examples/demo.py shows the plumbing with no-op segments, this example
runs a real Gibbs sampler to convergence on synthetic data, demonstrating the
parts of the protocol the demo leaves out:

- a *signal* segment with a conjugate amplitude draw (`SineSegment`),
  whitening against the current noise via `residual.noise_psd`;
- a *noise* segment (`WhiteNoiseSegment`) whose `noise_model` implements the
  `psd(freqs[, channel])` contract, refreshed by the Wheel every sweep;
- posterior chains accumulated in each segment's opaque `State`;
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
    """

    def __init__(self, name: str, freq: float, seed: int):
        self.name = name
        self.freq = freq
        self._seed = seed
        self._basis: dict[str, np.ndarray] | None = None

    def initial_state(self, observed: Residuals):
        t = observed.epoch + np.arange(observed.n_samples) * observed.dt
        self._basis = {
            ch: np.sin(2.0 * np.pi * self.freq * t) for ch in observed.channels
        }
        catalog = {"amplitude": 0.0}
        state = {"rng": np.random.default_rng(self._seed), "chain": []}
        return catalog, state

    def step(self, residual: Residuals, state):
        # per-sample noise variance from the threaded noise model:
        # sigma^2 = S_onesided * fs / 2 for white noise
        psd = residual.noise_psd(residual.channels[0])
        sigma2 = float(psd[1]) * residual.fs / 2.0
        ch = residual.channels[0]
        s = self._basis[ch]
        # `residual` excludes this segment's own render, so add nothing back
        ss = float(s @ s)
        mean = float(residual.tdi[ch] @ s) / ss
        amplitude = state["rng"].normal(mean, np.sqrt(sigma2 / ss))
        state["chain"].append(amplitude)
        return {"amplitude": amplitude}, state

    def render(self, catalog):
        assert self._basis is not None
        return {ch: catalog["amplitude"] * s for ch, s in self._basis.items()}


class WhiteNoiseSegment:
    """Noise segment: conjugate inverse-gamma draw for the white-noise sigma."""

    def __init__(self, name: str, seed: int):
        self.name = name
        self._seed = seed
        self._shapes: dict[str, np.ndarray] | None = None
        self._fs: float | None = None

    def initial_state(self, observed: Residuals):
        self._shapes = {
            ch: np.zeros_like(observed.tdi[ch]) for ch in observed.channels
        }
        self._fs = observed.fs
        catalog = {"sigma": 1.0}
        state = {"rng": np.random.default_rng(self._seed), "chain": []}
        return catalog, state

    def step(self, residual: Residuals, state):
        # residual here is data minus every signal segment's model: pure noise
        n_total = sum(arr.size for arr in residual.tdi.values())
        ssr = sum(float(arr @ arr) for arr in residual.tdi.values())
        # inverse-gamma(a, b) posterior with a weak IG(2, 1) prior
        a = 2.0 + 0.5 * n_total
        b = 1.0 + 0.5 * ssr
        sigma = float(np.sqrt(b / state["rng"].gamma(a)))
        state["chain"].append(sigma)
        return {"sigma": sigma}, state

    def render(self, catalog):
        # noise removes nothing from the data; its contribution travels
        # through Residuals.noise, not through residual subtraction
        assert self._shapes is not None
        return {ch: arr.copy() for ch, arr in self._shapes.items()}

    def noise_model(self, catalog):
        assert self._fs is not None
        return FlatNoise(catalog["sigma"], self._fs)


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
    wheel = Wheel(make_observed(seed))
    wheel.add(SineSegment(name="slow", freq=0.004, seed=seed + 1))
    wheel.add(SineSegment(name="fast", freq=0.011, seed=seed + 2))
    wheel.add(WhiteNoiseSegment(name="noise", seed=seed + 3))

    def progress(iteration: int, w: Wheel) -> None:
        if (iteration + 1) % 100 == 0:
            rms = float(np.sqrt(np.mean(w.residual().tdi["A"] ** 2)))
            print(
                f"sweep {iteration + 1:4d}: full-residual RMS = {rms:.4f}  "
                f"(sigma draw = {w.catalog('noise')['sigma']:.4f})"
            )

    wheel.run(n_sweeps, on_sweep=progress)

    results = {}
    for name in ("slow", "fast", "noise"):
        chain = np.asarray(wheel.state(name)["chain"][burn_in:])
        results[name] = (float(chain.mean()), float(chain.std()))
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
