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

    Every field below is part of the cross-group data contract, and
    `__post_init__` validates the whole object on every construction
    (including the `replace(...)` the Wheel performs each sweep): tdi keys
    must equal `channels`, array lengths must match `domain`/`n_samples`,
    and an attached `orbit` must span the observation. Inconsistent data
    fails loudly at construction, not deep inside a sampler.

    Fields:
        tdi: Channel name -> 1D array. Keys match `channels` exactly. In the
            time domain (`domain="time"`) each array holds `n_samples` real
            samples. In the frequency domain (`domain="frequency"`) each
            array holds the one-sided spectrum on the rfft grid of the
            underlying time series -- length `n_samples // 2 + 1`, with the
            continuous-transform normalization `dt * np.fft.rfft(x)`.
        sample_rate: Samples per second, in Hz.
        n_samples: Number of *time-domain* samples per channel -- always,
            even when `domain="frequency"` (it pins the duration, so
            `Tobs`/`df`/`dt` and the PSD grid stay well defined).
        channels: TDI channel names in this run, e.g. ("A", "E", "T"). A
            name implies the campaign's agreed channel definition *including
            normalization* -- e.g. A = (Z - X)/sqrt(2), E = (X - 2Y + Z)/sqrt(6),
            T = (X + Y + Z)/sqrt(3). Segments building these combinations
            differently (unnormalized variants differ by sqrt(2)/sqrt(3)
            factors) must not join the same run.
        tdi_generation: TDI generation string, e.g. "1.5" or "2.0".
        observable: What the samples physically are. Recommended values:
            "fractional_frequency" (relative frequency deviation dnu/nu, the
            LDC / lisainstrument default), "phase" (radians), "strain".
            Campaign-specific strings are allowed; every segment reads this
            one field, so agreement is by construction -- state it once,
            correctly, rather than letting each group assume its own.
        domain: "time" (default) or "frequency". Selects the tdi
            representation described above; segment renders must match it
            (the Wheel validates render shapes against `tdi`).
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
    observable: str
    epoch: float
    domain: str = "time"
    noise: Any | None = None
    orbit: Any | None = None

    DOMAINS: ClassVar[tuple[str, ...]] = ("time", "frequency")
    """Valid values for `domain`."""

    RECOMMENDED_OBSERVABLES: ClassVar[tuple[str, ...]] = (
        "fractional_frequency",
        "phase",
        "strain",
    )
    """Common values for `observable`; other campaign-agreed strings are fine."""

    # ---- consistency validation ------------------------------------------

    def __post_init__(self) -> None:
        """Validate the data contract; runs on every construction/replace."""
        self._validate_settings()
        self._validate_tdi()
        self._validate_orbit_span()

    def _validate_settings(self) -> None:
        """Scalar run settings: counts, rates, and convention strings."""
        if not isinstance(self.n_samples, (int, np.integer)) or self.n_samples <= 0:
            raise ValueError(
                f"n_samples must be a positive integer, got {self.n_samples!r}"
            )
        if not np.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError(
                f"sample_rate must be a positive finite number in Hz, "
                f"got {self.sample_rate!r}"
            )
        if not np.isfinite(self.epoch):
            raise ValueError(f"epoch must be finite GPS seconds, got {self.epoch!r}")
        if not isinstance(self.tdi_generation, str) or not self.tdi_generation:
            raise ValueError(
                f"tdi_generation must be a non-empty string, "
                f"got {self.tdi_generation!r}"
            )
        if not isinstance(self.observable, str) or not self.observable:
            raise ValueError(
                f"observable must be a non-empty string saying what the TDI samples "
                f"physically are, got {self.observable!r}; recommended values: "
                f"{', '.join(self.RECOMMENDED_OBSERVABLES)}"
            )
        if self.domain not in self.DOMAINS:
            raise ValueError(
                f"domain must be one of {self.DOMAINS}, got {self.domain!r}"
            )

    def _validate_tdi(self) -> None:
        """tdi keys match channels; every array lives on this domain's grid."""
        if not isinstance(self.tdi, dict):
            raise TypeError(
                f"tdi must be a dict of channel -> array, "
                f"got {type(self.tdi).__name__}"
            )
        if not self.channels:
            raise ValueError("channels must be a non-empty tuple of channel names")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError(f"channels contains duplicates: {self.channels}")
        if set(self.tdi) != set(self.channels):
            missing = sorted(set(self.channels) - set(self.tdi))
            extra = sorted(set(self.tdi) - set(self.channels))
            raise ValueError(
                f"tdi keys must match channels exactly; "
                f"missing {missing}, unexpected {extra}"
            )
        expected = self.n_samples if self.domain == "time" else self.n_samples // 2 + 1
        for ch in self.channels:
            arr = self.tdi[ch]
            if not isinstance(arr, np.ndarray) or arr.ndim != 1:
                raise TypeError(
                    f"tdi[{ch!r}] must be a 1-D numpy array, "
                    f"got {type(arr).__name__}"
                )
            if arr.shape[0] != expected:
                raise ValueError(
                    f"tdi[{ch!r}] has length {arr.shape[0]}, expected {expected} for "
                    f"domain={self.domain!r} with n_samples={self.n_samples} "
                    f"(n_samples always counts time-domain samples; frequency-domain "
                    f"arrays live on the rfft grid of length n_samples // 2 + 1)"
                )
            if self.domain == "time" and np.iscomplexobj(arr):
                raise TypeError(
                    f"tdi[{ch!r}] is complex but domain='time'; time-domain TDI is "
                    f"real (did you mean domain='frequency'?)"
                )

    def _validate_orbit_span(self) -> None:
        """A tabulated orbit (one exposing t_range) must cover the data span."""
        if self.orbit is None:
            return
        t_range = getattr(self.orbit, "t_range", None)
        if t_range is None:
            return
        t_lo, t_hi = float(t_range[0]), float(t_range[1])
        data_end = self.epoch + self.n_samples / self.sample_rate
        if t_lo > self.epoch or t_hi < data_end:
            raise ValueError(
                f"orbit ephemeris spans [{t_lo}, {t_hi}] s but the data spans "
                f"[{self.epoch}, {data_end}] s; every segment would need spacecraft "
                f"positions outside the tabulated ephemeris (mismatched epoch "
                f"conventions? GPS vs zero-based times?)"
            )

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

    def noise_psd(self, channel: str | None = None) -> np.ndarray | None:
        """One-sided noise PSD on this run's rfft grid, or ``None``.

        The frequency-domain noise piece: the per-bin variance a Fourier-domain
        segment whitens against. Turntable assembles it on the run's frequency
        grid (from `n_samples`/`sample_rate`) so segments never recompute the
        normalization. DC (bin 0) is ``+inf`` (zero weight). Returns ``None``
        when no noise model is set (`self.noise is None`).

        Pass `channel` (e.g. `"A"`, `"E"`, `"T"`) for the per-channel PSD -- LISA's
        A and E share a PSD but T (the null channel) differs, so a likelihood that
        models T must weight it by its own noise. `channel=None` (the default)
        calls the model's plain `psd(freqs)`, preserving back-compatibility with
        noise objects that expose only an A/E PSD.

        Requires the threaded noise object to expose ``psd(freqs[, channel])``.
        The grid has length ``n_samples // 2 + 1``, so in a frequency-domain
        run (`domain="frequency"`) it lines up bin-for-bin with the tdi arrays.
        """
        if self.noise is None:
            return None
        if not callable(getattr(self.noise, "psd", None)):
            raise TypeError(
                f"noise object {type(self.noise).__name__} does not expose "
                f"psd(freqs[, channel]); the model threaded on Residuals.noise must "
                f"implement it to serve frequency-domain segments "
                f"(see NoiseSegment.noise_model for the noise contract)"
            )
        freqs = np.fft.rfftfreq(self.n_samples, d=self.sample_interval)
        psd = np.empty(freqs.shape)
        psd[0] = np.inf
        psd[1:] = (
            self.noise.psd(freqs[1:])
            if channel is None
            else self.noise.psd(freqs[1:], channel)
        )
        return psd

    def noise_wdm_variance(
        self, n_layers: int, channel: str | None = None
    ) -> np.ndarray | None:
        """Per-pixel WDM noise variance grid, shape ``(n_layers+1, N/n_layers)``.

        The time-frequency noise piece for a wavelet-domain segment. Turntable
        supplies the time/grid parameters (time-pixel count, sample interval,
        epoch) so a segment passes only its wavelet frequency resolution
        ``n_layers``; the per-pixel variance and its normalization are assembled
        here, not in the segment. Returns ``None`` when no noise model is set.

        `channel` selects the per-channel variance (see :meth:`noise_psd`);
        `channel=None` (default) calls the model's plain `wdm_variance`.

        Requires the threaded noise object to expose
        ``wdm_variance(n_layers, n_time, dt, epoch[, channel]) -> ndarray``,
        and ``n_layers`` to divide ``n_samples`` exactly (otherwise trailing
        samples would be silently dropped from the grid).
        """
        if self.noise is None:
            return None
        if not callable(getattr(self.noise, "wdm_variance", None)):
            raise TypeError(
                f"noise object {type(self.noise).__name__} does not expose "
                f"wdm_variance(n_layers, n_time, dt, epoch[, channel]); the model "
                f"threaded on Residuals.noise must implement it to serve "
                f"wavelet-domain segments (see NoiseSegment.noise_model)"
            )
        if self.n_samples % n_layers:
            raise ValueError(
                f"n_layers={n_layers} must divide n_samples={self.n_samples} exactly; "
                f"otherwise the WDM grid covers only {n_layers * (self.n_samples // n_layers)} "
                f"of {self.n_samples} samples and the remainder is silently unweighted"
            )
        n_time = self.n_samples // n_layers
        if channel is None:
            return self.noise.wdm_variance(
                n_layers, n_time, self.sample_interval, self.epoch
            )
        return self.noise.wdm_variance(
            n_layers, n_time, self.sample_interval, self.epoch, channel
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
