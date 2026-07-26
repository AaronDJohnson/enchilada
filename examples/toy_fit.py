"""A complete (toy) blocked-Gibbs fit: two sources plus sampled noise.

Where examples/demo.py shows the plumbing with no-op segments, this example
runs a real Gibbs sampler to convergence on synthetic data, demonstrating the
parts of the protocol the demo leaves out:

- a signal segment (`SineSegment.step`) that fits the residual it is handed --
  already the data minus every other segment -- directly, then subtracts its
  new model and returns; the Wheel keeps the ledger, so there is no add-back;
- a *noise* segment (`WhiteNoiseSegment`) that removes nothing from the data
  and instead returns the residual with an updated `noise` object, which
  signal segments read through `residual.noise_psd`;
- state ownership: each segment keeps its parameters, RNG, current model,
  and posterior chain as plain instance attributes -- the Wheel never sees
  them, and you read results directly off the segment objects you built;
- progress reporting through `Wheel.run`'s `on_sweep` callback.

The data are one channel of two sinusoids in white noise -- physically a toy
(amplitudes in arbitrary units), but the Gibbs structure is exactly the real
thing. Run it with:

    uv run python examples/toy_fit.py
"""

from dataclasses import replace

import numpy as np

from turntable import Residuals, Wheel


class FlatNoise:
    """White-noise model satisfying the turntable noise contract."""

    def __init__(self, sigma: float, sample_rate: float):
        self.sigma = sigma
        self._fs = sample_rate

    def psd(self, freqs, channel=None):
        # one-sided PSD of white noise with per-sample std `sigma`
        # (see Residuals.noise_psd for the pinned normalization)
        return np.full_like(freqs, 2.0 * self.sigma**2 / self._fs)


class SineSegment:
    """One sinusoidal source with a conjugate Gibbs draw for its amplitude.

    The frequency is treated as known; the amplitude posterior given white
    noise of variance sigma^2 is N(<d, s>/<s, s>, sigma^2/<s, s>), which we
    sample exactly -- no Metropolis machinery needed for the toy.

    Everything this sampler is -- current amplitude, RNG, chain, and its own
    model basis -- is a plain instance attribute. The Wheel only ever sees
    the residual.
    """

    def __init__(self, name: str, freq: float, seed: int):
        self.name = name
        self.freq = freq
        self.amplitude = 0.0
        self.chain: list[float] = []
        self._rng = np.random.default_rng(seed)
        self._basis: dict[str, np.ndarray] | None = None

    def start(self, residual: Residuals) -> Residuals:
        t = residual.epoch + np.arange(residual.n_samples) * residual.dt
        self._basis = {
            ch: np.sin(2.0 * np.pi * self.freq * t) for ch in residual.channels
        }
        # initial amplitude is zero, so we subtract nothing: pass through
        return residual

    def step(self, residual: Residuals) -> Residuals:
        ch = residual.channels[0]
        s = self._basis[ch]
        # per-sample noise variance from the threaded noise model:
        # sigma^2 = S_onesided * fs / 2 for white noise
        sigma2 = float(residual.noise_psd(ch)[1]) * residual.fs / 2.0

        # `residual` is already the data minus every OTHER segment -- fit it
        # directly (no add-back), then subtract our new model and return.
        data_for_me = residual.tdi[ch]
        ss = float(s @ s)
        mean = float(data_for_me @ s) / ss
        self.amplitude = self._rng.normal(mean, np.sqrt(sigma2 / ss))
        self.chain.append(self.amplitude)

        new_tdi = dict(residual.tdi)
        new_tdi[ch] = data_for_me - self.amplitude * s
        return replace(residual, tdi=new_tdi)


class WhiteNoiseSegment:
    """Noise segment: conjugate inverse-gamma draw for the white-noise sigma.

    Removes nothing from the data; it returns the residual with an updated
    `noise` model, which the signal segments read via `residual.noise_psd`.
    """

    def __init__(self, name: str, seed: int):
        self.name = name
        self.sigma = 1.0
        self.chain: list[float] = []
        self._rng = np.random.default_rng(seed)

    def start(self, residual: Residuals) -> Residuals:
        return replace(residual, noise=FlatNoise(self.sigma, residual.fs))

    def step(self, residual: Residuals) -> Residuals:
        # residual here is data minus every signal segment's model: pure noise
        n_total = sum(arr.size for arr in residual.tdi.values())
        ssr = sum(float(arr @ arr) for arr in residual.tdi.values())
        # inverse-gamma(a, b) posterior with a weak IG(2, 1) prior
        a = 2.0 + 0.5 * n_total
        b = 1.0 + 0.5 * ssr
        self.sigma = float(np.sqrt(b / self._rng.gamma(a)))
        self.chain.append(self.sigma)
        return replace(residual, noise=FlatNoise(self.sigma, residual.fs))


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
        channels=("A",),   # n_samples derived from the array
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
    wheel.add(noise)  # steps last each sweep: sees data minus both signals

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
