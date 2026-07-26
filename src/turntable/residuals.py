from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, Never

import numpy as np


@dataclass(frozen=True, eq=False)
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

            **Omit it for time-domain data**: the arrays are exactly that
            long, so it is read off them and stating it again is duplication.

            **Frequency-domain data must state it.** An rfft of a length-n
            real series has `n // 2 + 1` bins, which loses the parity of n:
            513 bins are consistent with n=1024 *and* n=1025, and those imply
            different `Tobs` and `df`. The data cannot answer the question, so
            turntable asks rather than guessing (guessing would silently
            mis-scale every frequency-domain weight).
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
            representation described above; the residual a segment returns
            must keep it (`Residuals` validates the tdi shapes).
        epoch: GPS seconds corresponding to sample index 0. Defaults to
            ``0.0`` -- fine for synthetic data with no absolute-time
            reference. Set it for real data: it anchors the constellation
            response (spacecraft positions at `epoch + n*dt`), the orbit-span
            check, and the frequency-domain phase reference. Shadowed by `t0`.
        noise: The current noise/covariance model the residual should be
            whitened against, or `None`. Opaque to the Wheel (like a
            segment's own state): a noise segment defines its own type and
            puts it here on the residual it returns, and the Wheel carries
            that residual to every other segment, so signal segments can
            weight their likelihood by the *current* noise estimate instead
            of a hardcoded PSD. `None` when no noise model is set. See
            `segment.NoiseSegment`.
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
    -- pick whichever reads better in context, but prefer *one* consistently
    within a given segment or script so readers are not tracking two
    vocabularies. Call `Residuals.aliases()` for the full long-to-short table.
    """

    tdi: dict[str, np.ndarray]
    sample_rate: float
    channels: tuple[str, ...]
    tdi_generation: str
    observable: str
    #: Number of *time-domain* samples. Leave it out (the ``0`` sentinel) for
    #: time-domain data and it is read off the tdi arrays, which carry it
    #: exactly. Frequency-domain data must state it -- see the class docstring.
    n_samples: int = 0
    epoch: float = 0.0
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
        """Validate the data contract; runs on every construction/replace.

        Ordered by dependency: conventions first (so `domain` is known), then
        the tdi structure (so an array length can be read), then `n_samples`
        (derived from that length, or required), then the length and orbit
        checks that need it.
        """
        self._validate_conventions()
        self._validate_tdi_structure()
        self._resolve_and_check_n_samples()
        self._validate_tdi_lengths()
        self._validate_orbit_span()

    def _validate_conventions(self) -> None:
        """Scalar run settings: rates, epoch, and convention strings."""
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

    def _validate_tdi_structure(self) -> None:
        """tdi is a dict of 1-D arrays whose keys are exactly `channels`."""
        if not isinstance(self.tdi, dict):
            raise TypeError(
                f"tdi must be a dict of channel -> array, got {type(self.tdi).__name__}"
            )
        if isinstance(self.channels, (list, tuple)) and not isinstance(
            self.channels, tuple
        ):
            # normalise here: a list would otherwise compare unequal to the
            # tuple a segment returns, and the Wheel would blame the segment
            # for "changing a run setting" it never touched.
            object.__setattr__(self, "channels", tuple(self.channels))
        if not isinstance(self.channels, tuple):
            raise TypeError(
                f"channels must be a tuple of channel names, "
                f"got {type(self.channels).__name__}"
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
        for ch in self.channels:
            arr = self.tdi[ch]
            if not isinstance(arr, np.ndarray) or arr.ndim != 1:
                raise TypeError(
                    f"tdi[{ch!r}] must be a 1-D numpy array, got {type(arr).__name__}"
                )

    def _resolve_and_check_n_samples(self) -> None:
        """Fill in `n_samples` from the data where that is exact.

        Time domain: read it off the arrays, which carry it exactly. Frequency
        domain: it cannot be recovered from the data (see the `n_samples` field
        docstring for why), so it must have been stated. Also validates an
        explicitly supplied value, hence the name.
        """
        if self.n_samples == 0:  # sentinel: not supplied
            first = self.tdi[self.channels[0]]
            if self.domain == "time":
                object.__setattr__(self, "n_samples", int(first.shape[0]))
            else:
                n_bins = int(first.shape[0])
                raise ValueError(
                    f"n_samples must be given when domain='frequency': the "
                    f"{n_bins}-bin rfft grid does not determine it (it is "
                    f"consistent with n_samples={2 * (n_bins - 1)} and "
                    f"={2 * n_bins - 1}, which imply different Tobs and df). "
                    f"Pass the number of time-domain samples the spectrum came "
                    f"from; only time-domain data can have it derived."
                )
        if (
            not isinstance(self.n_samples, (int, np.integer))
            or isinstance(self.n_samples, bool)  # True would pass as 1
            or self.n_samples <= 0
        ):
            raise ValueError(
                f"n_samples must be a positive integer, got {self.n_samples!r}"
            )

    def _validate_tdi_lengths(self) -> None:
        """Every array lives on this domain's grid, and is real in the time
        domain."""
        expected = self.n_samples if self.domain == "time" else self.n_samples // 2 + 1
        for ch in self.channels:
            arr = self.tdi[ch]
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
            # dtype is part of the contract too: integer or object arrays would
            # otherwise be accepted here and then fail deep inside the Wheel's
            # ledger arithmetic with a raw numpy casting error.
            if not np.issubdtype(arr.dtype, np.inexact):
                raise TypeError(
                    f"tdi[{ch!r}] has dtype {arr.dtype}; TDI must be floating or "
                    f"complex (an integer or object array cannot carry a residual "
                    f"-- convert with .astype(float) first)"
                )

    def _validate_orbit_span(self) -> None:
        """A tabulated orbit (one exposing t_range) must cover the data span."""
        if self.orbit is None:
            return
        t_range = getattr(self.orbit, "t_range", None)
        if t_range is None:
            return
        t_lo, t_hi = float(t_range[0]), float(t_range[1])
        # The samples sit at epoch + n*dt for n in [0, n_samples-1], so the last
        # one is at epoch + (n_samples-1)*dt -- NOT epoch + Tobs. Requiring the
        # ephemeris to reach epoch + Tobs would reject an orbit tabulated on the
        # data's own sample grid.
        #
        # Multiply `sample_interval` exactly as every grid builder here does
        # (`epoch + arange(n) * dt`, see NumericOrbit.from_hdf5) rather than
        # dividing by sample_rate: the two differ by an ulp for rates whose dt
        # is not exactly representable, and this is an exact float comparison,
        # so the mismatch would spuriously reject a data-grid orbit.
        #
        # This is a coarse check for gross epoch mismatches, not a guarantee: a
        # segment applying TDI light-travel delays evaluates retarded times
        # slightly outside [epoch, last_sample] and needs margin (and
        # NumericOrbit.positions raises if asked beyond its table).
        last_sample = self.epoch + (self.n_samples - 1) * self.sample_interval
        if t_lo > self.epoch or t_hi < last_sample:
            raise ValueError(
                f"orbit ephemeris spans [{t_lo}, {t_hi}] s but the data samples "
                f"span [{self.epoch}, {last_sample}] s; every segment would need "
                f"spacecraft positions outside the tabulated ephemeris (mismatched "
                f"epoch conventions? GPS vs zero-based times?)"
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

    # ---- domain transforms (the campaign's FFT convention, in code) ------

    def to_frequency(self) -> "Residuals":
        """This same dataset as a one-sided spectrum (``domain="frequency"``).

        Applies the campaign's Fourier convention -- ``X(f) = dt * rfft(x)``,
        the one :meth:`noise_psd` is normalized against -- so groups cannot
        disagree about it: it is executed here rather than described. Everything
        else rides along unchanged, including ``n_samples``, which is what makes
        the transform invertible (see :meth:`to_time`).

        Data enters a campaign as a time series, so this is the normal way to
        get a frequency-domain residual: build the `Residuals` from the time
        series (where `n_samples` is read off the arrays) and transform. You
        then never state `n_samples` by hand at all.

        Returns ``self`` unchanged if already in the frequency domain.
        """
        if self.domain == "frequency":
            return self
        tdi = {
            ch: self.sample_interval * np.fft.rfft(arr) for ch, arr in self.tdi.items()
        }
        return replace(self, tdi=tdi, domain="frequency")

    def to_time(self) -> "Residuals":
        """This same dataset as a time series (``domain="time"``).

        Inverts :meth:`to_frequency` exactly -- ``x = irfft(X / dt, n)`` -- for
        *either* parity of ``n``, because ``n_samples`` travelled with the data.
        A bare spectrum with no ``n_samples`` cannot be inverted this way: its
        ``n // 2 + 1`` bins are consistent with both ``2*(bins-1)`` and
        ``2*bins-1``, and choosing wrong silently resamples the series.

        Returns ``self`` unchanged if already in the time domain.
        """
        if self.domain == "time":
            return self
        tdi = {
            ch: np.fft.irfft(arr / self.sample_interval, n=self.n_samples)
            for ch, arr in self.tdi.items()
        }
        return replace(self, tdi=tdi, domain="time")

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

        Normalization (the convention every group in a run must share -- pinned
        here for the same reason `observable`/`domain` are). ``psd`` is the
        **one-sided** PSD in units of ``[observable]**2 / Hz``, tied to the
        ``dt * rfft(x)`` frequency spectrum (see the `tdi` field) by

            E[ |X(f)|**2 ] = (Tobs / 2) * S(f)      (interior bins)

        so a frequency-domain segment's per-bin weight is
        ``|X(f)|**2 / ((Tobs/2) S)`` for the interior bins. Two bins are not
        interior: DC (set to ``+inf`` here, so it carries zero weight) and, when
        ``n_samples`` is even, the Nyquist bin, which is purely real and carries
        one degree of freedom rather than two -- weight it half, or drop it, or
        the likelihood over-counts that single bin by 2x.

        For the **time-domain per-sample variance**, call
        :meth:`noise_variance`, which does this weighting for you and is
        therefore independent of the parity of ``n_samples``.

        For white noise ``S = 2 sigma**2 / fs``; the noise object carries its own
        ``fs`` (the ``psd(freqs)`` call passes only frequencies), so it computes
        ``S`` from the ``sigma`` it holds.
        """
        if self.noise is None:
            return None
        if not callable(getattr(self.noise, "psd", None)):
            raise TypeError(
                f"noise object {type(self.noise).__name__} does not expose "
                f"psd(freqs[, channel]); the model a noise segment puts on "
                f"Residuals.noise must implement it to serve frequency-domain "
                f"segments (see segment.NoiseSegment for the noise contract)"
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

    def noise_variance(self, channel: str | None = None) -> float | None:
        """Per-sample time-domain variance implied by the noise model, or None.

        The quantity a *time-domain* segment needs for its likelihood weight,
        so it does not have to re-derive it from the PSD grid. Integrates the
        one-sided PSD over this run's grid, excluding DC (which carries no
        variance for zero-mean data) and half-weighting the Nyquist bin when
        `n_samples` is even, because that bin carries one degree of freedom
        rather than two.

        That weighting is what makes the answer independent of the parity of
        `n_samples`: it returns the variance of the zero-mean series, which for
        white noise is ``sigma**2 * (1 - 1/n_samples)`` -- the naive
        ``sum(psd[1:]) * df`` instead lands on ``sigma**2`` for even n and
        ``sigma**2 (1-1/n)`` for odd n, i.e. it disagrees with itself across
        parities.

        Requires the same `psd(freqs[, channel])` contract as
        :meth:`noise_psd`, and raises the same error if it is missing.
        """
        psd = self.noise_psd(channel)
        if psd is None:
            return None
        interior = psd[1:]  # bin 0 is +inf by construction; DC holds no variance
        weights = np.ones(interior.shape)
        if self.n_samples % 2 == 0:
            weights[-1] = 0.5  # Nyquist: 1 degree of freedom, not 2
        return float(np.sum(interior * weights) * self.frequency_resolution)

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

    # Hidden from type checkers on purpose. This class ships py.typed, so its
    # annotations are load-bearing for consumers -- and *any* __getattr__ tells
    # a checker that every attribute name exists, which would statically
    # legitimise the very typos the runtime table below catches. Annotating it
    # `-> Never` does not help: `Never` is the bottom type, assignable to
    # everything, so `x: int = residual.Tobbs` type-checks clean. With the
    # method invisible at type-check time, mypy reports
    #   "Residuals" has no attribute "Tobbs"; maybe "Tobs"?
    # i.e. statically what the runtime does dynamically, while `hasattr` and
    # ordinary attribute access keep working at runtime.
    if not TYPE_CHECKING:  # pragma: no branch - always true at runtime

        def __getattr__(self, name: str) -> Never:
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
            # Not a known typo either -- still point at the alias table, so an
            # unfamiliar spelling leads somewhere instead of dead-ending.
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name!r}; "
                f"see {type(self).__name__}.aliases() for the derived quantities "
                f"(and their LISA short names)."
            )
