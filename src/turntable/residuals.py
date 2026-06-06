from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np


@dataclass(frozen=True)
class Residuals:
    """TDI data plus the fixed settings that say how to interpret it.

    One `Residuals` is constructed at the top of a run to hold the observed
    data and the campaign settings (sample rate, channels, epoch, ...). The
    Wheel produces new `Residuals` each Gibbs iteration with the same
    metadata fields but freshly computed `tdi` -- the data with every other
    segment's current model subtracted.

    Fields:
        tdi: Channel name -> 1D array of samples. Keys match `channels`;
            each array has length `n_samples`.
        sample_rate: Samples per second, in Hz.
        n_samples: Number of samples per channel.
        channels: TDI channel names in this run, e.g. ("A", "E", "T").
        tdi_generation: TDI generation string, e.g. "1.5" or "2.0".
        epoch: GPS seconds corresponding to sample index 0.
        noise: The current noise/covariance model the residual should be
            whitened against, or `None`. Opaque to the Wheel (like `Catalog`
            and `State`): a noise segment defines its own type and the Wheel
            threads it here so signal segments can weight their likelihood by
            the *current* noise estimate instead of a hardcoded PSD. `None`
            when no noise segment is registered. See `segment.NoiseSegment`.
        orbit: The LISA constellation ephemeris the data was produced with --
            the spacecraft positions every segment must share to build its
            response (see `turntable.orbits.Orbit`). A *fixed* property of the
            dataset, like `epoch`/`tdi_generation`: set once on the observed
            data and the Wheel threads it unchanged (it is never sampled).
            Opaque to the Wheel, exactly like `noise`; segments read
            `residual.orbit` rather than constructing their own, so every piece
            uses the *same* constellation. `None` lets a segment fall back to
            its own default orbit (back-compatible with orbit-less runs).

    Derived properties are exposed under both descriptive long names and
    the short symbols LISA papers use. Both spellings return the same value
    -- pick whichever reads better in context. Call `Residuals.aliases()`
    for the full long-to-short table.
    """

    tdi: dict[str, np.ndarray]
    sample_rate: float
    n_samples: int
    channels: tuple[str, ...]
    tdi_generation: str
    epoch: float
    noise: Any | None = None
    orbit: Any | None = None

    # ---- descriptive (long) names ---------------------------------------

    @property
    def observation_time(self) -> float:
        """Total observation time in seconds. Shadowed by `Tobs`."""
        return self.n_samples / self.sample_rate

    @property
    def sample_interval(self) -> float:
        """Seconds between consecutive samples. Shadowed by `dt`."""
        return 1.0 / self.sample_rate

    @property
    def frequency_resolution(self) -> float:
        """Width of a Fourier bin, in Hz. Shadowed by `df`."""
        return 1.0 / self.observation_time

    @property
    def nyquist_frequency(self) -> float:
        """Nyquist frequency, in Hz. Shadowed by `fny`."""
        return self.sample_rate / 2.0

    # ---- conventional short-name shadows --------------------------------

    @property
    def Tobs(self) -> float:
        """LISA shorthand for `observation_time` (seconds)."""
        return self.observation_time

    @property
    def fs(self) -> float:
        """LISA shorthand for `sample_rate` (Hz)."""
        return self.sample_rate

    @property
    def dt(self) -> float:
        """LISA shorthand for `sample_interval` (seconds)."""
        return self.sample_interval

    @property
    def N(self) -> int:
        """LISA shorthand for `n_samples`."""
        return self.n_samples

    @property
    def df(self) -> float:
        """LISA shorthand for `frequency_resolution` (Hz)."""
        return self.frequency_resolution

    @property
    def fny(self) -> float:
        """LISA shorthand for `nyquist_frequency` (Hz)."""
        return self.nyquist_frequency

    @property
    def t0(self) -> float:
        """LISA shorthand for `epoch` (GPS seconds)."""
        return self.epoch

    # ---- noise variance (assembled on this run's grid) ------------------

    def noise_psd(self) -> np.ndarray | None:
        """One-sided noise PSD on this run's rfft grid, or ``None``.

        The frequency-domain noise piece: the per-bin variance a Fourier-domain
        segment whitens against. Turntable assembles it on the run's frequency
        grid (from `n_samples`/`sample_rate`) so segments never recompute the
        normalization. DC (bin 0) is ``+inf`` (zero weight). Returns ``None``
        when no noise model is set (`self.noise is None`).

        Requires the threaded noise object to expose ``psd(freqs) -> ndarray``.
        """
        if self.noise is None:
            return None
        freqs = np.fft.rfftfreq(self.n_samples, d=self.sample_interval)
        psd = np.empty(freqs.shape)
        psd[0] = np.inf
        psd[1:] = self.noise.psd(freqs[1:])
        return psd

    def noise_wdm_variance(self, n_layers: int) -> np.ndarray | None:
        """Per-pixel WDM noise variance grid, shape ``(n_layers+1, N/n_layers)``.

        The time-frequency noise piece for a wavelet-domain segment. Turntable
        supplies the time/grid parameters (time-pixel count, sample interval,
        epoch) so a segment passes only its wavelet frequency resolution
        ``n_layers``; the per-pixel variance and its normalization are assembled
        here, not in the segment. Returns ``None`` when no noise model is set.

        Requires the threaded noise object to expose
        ``wdm_variance(n_layers, n_time, dt, epoch) -> ndarray``.
        """
        if self.noise is None:
            return None
        n_time = self.n_samples // n_layers
        return self.noise.wdm_variance(
            n_layers, n_time, self.sample_interval, self.epoch
        )

    # ---- discoverability ------------------------------------------------

    ALIASES: ClassVar[dict[str, str]] = {
        "observation_time": "Tobs",
        "sample_rate": "fs",
        "sample_interval": "dt",
        "n_samples": "N",
        "frequency_resolution": "df",
        "nyquist_frequency": "fny",
        "epoch": "t0",
    }
    """Long-name -> short-name table. Both spellings are valid attributes
    on every `Residuals` instance and return the same value."""

    @classmethod
    def aliases(cls) -> dict[str, str]:
        """Return a copy of the long-name -> short-name table.

        Useful for users learning the convention:

            >>> for long, short in Residuals.aliases().items():
            ...     print(f"{long:24s} = {short}")
        """
        return dict(cls.ALIASES)

    # ---- typo catcher ---------------------------------------------------

    _TYPOS: ClassVar[dict[str, str]] = {
        "T_obs": "Tobs",
        "t_obs": "Tobs",
        "Tobservation": "Tobs",
        "f_s": "fs",
        "F_s": "fs",
        "d_t": "dt",
        "D_t": "dt",
        "N_samples": "n_samples",
        "n_Samples": "n_samples",
        "Nsamples": "n_samples",
        "d_f": "df",
        "f_ny": "fny",
        "F_ny": "fny",
        "f_nyquist": "nyquist_frequency",
        "nyquist": "nyquist_frequency",
        "t_0": "t0",
        "T_0": "t0",
    }

    def __getattr__(self, name: str):
        # Only fires when normal attribute lookup fails, so this catches
        # common misspellings of the long/short names above.
        if name.startswith("_"):
            raise AttributeError(name)
        suggestion = self._TYPOS.get(name)
        if suggestion is not None:
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name!r}; "
                f"did you mean {suggestion!r}? "
                f"See {type(self).__name__}.aliases() for the full table."
            )
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")
