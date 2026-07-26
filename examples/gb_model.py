"""Galactic-binary model pieces for the turntable example.

**Nothing in this file is turntable.** It is the user-supplied side of the
example — the parts a source-class group brings to a global fit:

- ``FixedLISANoise`` — a noise model exposing ``psd(freqs[, channel])``;
- ``gb_template`` / ``waveform`` / ``scatter`` / ``inner`` — GBGPU waveform and
  inner-product helpers (pure functions, no global state; ``waveform`` is the
  one-off convenience wrapper used for diagnostics and plots);
- ``inject_gb`` — builds a synthetic frequency-domain dataset (signal + noise);
- ``GBSegment`` — the turntable ``Segment`` implementation. It reads everything
  it needs (channels, grid, noise PSD) off the ``residual`` the Wheel hands it,
  so it never reaches back into this module's or the notebook's scope for run
  settings.

The notebook imports these and drives them through turntable's ``Residuals`` and
``Wheel``; keeping them here makes it obvious which code is turntable and which
is the model plugged into it.

Requires the LISA stack: ``gbgpu`` (master), ``eryn`` (dev),
``lisaanalysistools``. The companion notebook additionally plots with
``matplotlib`` and ``corner``.
"""

from dataclasses import replace

import numpy as np
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer
from eryn.priors.analytical import UniformDistribution
from eryn.state import State
from gbgpu.gbgpu import GBGPU
from lisatools.sensitivity import A1TDISens, get_sensitivity


# ---------------------------------------------------------------- noise model
class FixedLISANoise:
    """One-sided LISA PSD from lisatools ``get_sensitivity`` (A == E).

    The only contract a turntable noise object needs: ``psd(freqs[, channel])``.
    """

    def psd(self, f, channel=None):
        return get_sensitivity(np.asarray(f), sens_fn=A1TDISens)


# ---------------------------------------------------------------- waveform helpers
def gb_template(gb, params, angles, Tobs, dt, NB):
    """GB (amp, f0, fdot, phi0) + fixed angles -> (start_ind, hA, hE)."""
    amp, f0, fdot, phi0 = params
    iota, psi, lam, beta = angles
    gb.run_wave(amp, f0, fdot, 0.0, phi0, iota, psi, lam, beta, N=NB, T=Tobs, dt=dt)
    return int(gb.start_inds[0]), np.asarray(gb.A[0]), np.asarray(gb.E[0])


def waveform(params, angles, Tobs, dt, NB=128):
    """Convenience one-off template (own GBGPU instance) -> (start_ind, hA, hE).

    For hot loops (a likelihood) pass a reused ``gb`` to :func:`gb_template`
    instead; this is for diagnostics/plots.
    """
    return gb_template(GBGPU(), params, angles, Tobs, dt, NB)


def scatter(start_ind, cols, n_rfft, chans):
    """Narrowband template columns -> full one-sided rfft-grid dict."""
    out = {ch: np.zeros(n_rfft, dtype=complex) for ch in chans}
    for ch in chans:
        out[ch][start_ind:start_ind + len(cols[ch])] = cols[ch]
    return out


def inner(d, h, S, band, df):
    """Noise-weighted inner product ``4 df Re sum |d-h|^2 / S`` over a band."""
    r = d[band] - h
    return 4.0 * df * np.sum((np.abs(r) ** 2) / S[band]).real


# ---------------------------------------------------------------- data generation
def inject_gb(truth, angles, Tobs, dt, n_samples, channels, noise, NB=128, seed=42):
    """Synthesize one GB in stationary noise on the one-sided rfft grid.

    Note this dataset never exists as a time series -- GBGPU emits narrowband
    frequency-domain templates directly -- which is why the caller must state
    ``n_samples`` when wrapping it in a ``Residuals``. Data loaded as a time
    series instead gets it derived, and ``Residuals.to_frequency()`` carries it.

    Returns ``(tdi, info)``: ``tdi`` is the channel->array dict to wrap in a
    ``Residuals``; ``info`` carries ``band``, the noiseless ``signal``, the optimal ``snr``, and
    the template's ``start_ind`` for plotting/diagnostics.
    """
    df = 1.0 / Tobs
    freqs = np.fft.rfftfreq(n_samples, dt)
    n_rfft = freqs.size

    gb = GBGPU()
    si, hA, hE = gb_template(gb, truth, angles, Tobs, dt, NB)
    signal = scatter(si, {"A": hA, "E": hE}, n_rfft, channels)

    S = np.empty(n_rfft)
    S[0] = np.inf
    S[1:] = noise.psd(freqs[1:])

    rng = np.random.default_rng(seed)
    tdi = {}
    for ch in channels:
        draw = np.sqrt(S / (4.0 * df)) * (
            rng.standard_normal(n_rfft) + 1j * rng.standard_normal(n_rfft))
        draw[0] = 0.0
        tdi[ch] = signal[ch] + draw

    band = slice(si, si + NB)
    snr = np.sqrt(sum(inner(np.zeros(n_rfft), h, S, band, df) for h in (hA, hE)))
    return tdi, {"band": band, "signal": signal, "snr": snr, "start_ind": si}


# ---------------------------------------------------------------- the segment
class GBSegment:
    """turntable ``Segment``: one galactic binary, sampled with Eryn.

    Implements ``name`` / ``start`` / ``step``. Reads channels, the frequency
    grid, and the noise PSD off the ``residual`` it is handed (never from module
    scope). The residual is already the data minus every other segment, so it
    advances Eryn against it directly (no add-back), then returns the residual
    with its new point-estimate model subtracted. The posterior chain lives on
    the object (``self.chain``).

    Samples ``amp, f0, fdot, phi0``; the four sky/orientation ``angles`` are held
    fixed (enlarge ``init``/``bounds``/``angles`` to sample them too).

    ``name`` is a constructor argument so several of these can share one Wheel
    (each with its own band and sampler) -- Wheel rejects duplicate names.
    ``params`` is the current point estimate; ``chain`` the posterior samples.
    """

    #: channels this model can produce; GBGPU gives the A and E TDI variables
    SUPPORTED_CHANNELS = ("A", "E")

    def __init__(self, init, bounds, angles, name="gb", n_walkers=24,
                 steps_per_sweep=40, band=128, seed=0):
        self.name = name
        self.init = np.asarray(init, float)
        b = np.asarray(bounds, float)
        self.lo, self.hi = b[:, 0], b[:, 1]
        self.angles, self.NB = angles, band
        self.nw, self.k = n_walkers, steps_per_sweep
        self.ndim = self.init.size
        self.rng = np.random.default_rng(seed)
        self.chain = None
        self.params = self.init.copy()
        self._data = self._model = self._sampler = self._state = None

    def _read_context(self, residual):
        # ---- everything the physics needs comes from the residual ----
        # Check the conventions we cannot honour rather than assuming them:
        # this model emits a frequency-domain A/E fractional-frequency signal.
        if residual.domain != "frequency":
            raise ValueError(
                f"{self.name}: this GB model produces frequency-domain templates, "
                f"but the run is domain={residual.domain!r}"
            )
        if residual.observable != "fractional_frequency":
            raise ValueError(
                f"{self.name}: GBGPU emits fractional-frequency TDI, but the run "
                f"declares observable={residual.observable!r}"
            )
        unsupported = set(residual.channels) - set(self.SUPPORTED_CHANNELS)
        if unsupported:
            raise ValueError(
                f"{self.name}: this model only produces "
                f"{list(self.SUPPORTED_CHANNELS)}, but the run has channels "
                f"{list(residual.channels)} (unsupported: {sorted(unsupported)})"
            )
        self.chans = residual.channels
        self.Tobs, self.dt, self.df = residual.Tobs, residual.dt, residual.df
        self.n_rfft = residual.tdi[self.chans[0]].shape[0]
        self.gb = GBGPU()
        self.S = {ch: residual.noise_psd(ch) for ch in self.chans}  # noise via turntable

    def _template(self, params):
        return gb_template(self.gb, params, self.angles, self.Tobs, self.dt, self.NB)

    def _render(self, params):
        si, hA, hE = self._template(params)
        return scatter(si, {"A": hA, "E": hE}, self.n_rfft, self.chans)

    def _logl(self, x):  # Eryn maps this over walkers
        si, hA, hE = self._template(np.asarray(x).ravel())
        b = slice(si, si + self.NB)
        cols = {"A": hA, "E": hE}
        return -0.5 * sum(inner(self._data[ch], cols[ch], self.S[ch], b, self.df)
                          for ch in self.chans)

    def start(self, residual):
        self._read_context(residual)
        self._model = self._render(self.params)
        self._sampler = EnsembleSampler(
            self.nw, self.ndim, self._logl,
            priors={"model_0": ProbDistContainer(
                {i: UniformDistribution(self.lo[i], self.hi[i])
                 for i in range(self.ndim)})})
        r = self.rng.standard_normal((1, self.nw, 1, self.ndim))
        p0 = np.clip(self.init + 0.01 * (self.hi - self.lo) * r,
                     self.lo + 1e-9 * (self.hi - self.lo),
                     self.hi - 1e-9 * (self.hi - self.lo))
        self._state = State(p0)
        return replace(residual, tdi={ch: residual.tdi[ch] - self._model[ch]
                                      for ch in self.chans})

    def step(self, residual):
        self.S = {ch: residual.noise_psd(ch) for ch in self.chans}   # current noise, off the residual
        # `residual` is already the data minus every OTHER segment -- fit it
        # directly (no add-back); the Wheel keeps the ledger.
        self._data = {ch: residual.tdi[ch] for ch in self.chans}
        self._state = self._sampler.run_mcmc(self._state, self.k, progress=False)
        self.chain = self._sampler.get_chain()["model_0"].reshape(-1, self.ndim)
        self.params = self.chain[-self.nw:].mean(0)                        # point estimate
        self._model = self._render(self.params)
        return replace(residual, tdi={ch: residual.tdi[ch] - self._model[ch]
                                      for ch in self.chans})
