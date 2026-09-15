from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Never

import numpy as np

from enchilada.block_result import BlockResult, check_on_grid, check_rfft_endpoints
from enchilada.covariance import DataCovariance
from enchilada.domains import WDMGrid, check_wdm_coefficients
from enchilada.orbits import Orbit
from enchilada.translated_covariance import TranslatedCovariance


@dataclass(frozen=True, eq=False)
class L1Data:
    """L1 data plus the fixed settings that say how to interpret it.

    One `L1Data` object is constructed at the top of a run to hold the observed
    data and the campaign settings (sample rate, channels, epoch, ...). The
    same type carries the observed data and every residual: the Wheel hands
    each block an `L1Data` with the same metadata but `channel_data` recomputed
    as the data minus every other block's current signal. What a block returns is
    a `BlockResult` (see `block_result` and `zero_block_result`), not an `L1Data`.

    Every field below is part of the cross-group data contract, and
    `__post_init__` validates the whole object on every construction
    (including the `replace(...)` the Wheel performs each cycle): `channel_data`
    keys must equal `channel_names`, array lengths must match `data_domain` and
    `num_time_samples`, and an attached `orbit_ephemeris` must span the observation.
    Inconsistent data fails loudly at construction, not deep inside a sampler.

    Fields:
        channel_data: Channel name -> array. Keys match `channel_names` exactly.
            In the time domain (`data_domain="time"`) each array holds
            `num_time_samples` real samples. In the frequency domain
            (`data_domain="frequency"`) each array holds the one-sided spectrum
            on the rfft grid of the underlying time series -- length
            `num_time_samples // 2 + 1`, with the continuous-transform
            normalization `dt * np.fft.rfft(x)`. DC is real, and the final
            Nyquist coefficient is real when `num_time_samples` is even.
            WDM arrays are real coefficient grids shaped
            ``(num_frequency_divisions + 1, num_time_divisions)``; inactive
            edge coefficients must be exactly zero.
        sample_rate_hz: Samples per second, in Hz.
        num_time_samples: Number of *time-domain* samples per channel -- always,
            in every data domain (it pins the duration, so
            `Tobs`/`df`/`dt` and the PSD grid stay well defined).

            **Omit it for time-domain data**: the arrays are exactly that
            long, so it is read off them and stating it again is duplication.

            **Frequency-domain data must specify it.** An rfft of a length-n
            real series has `n // 2 + 1` bins, which loses the parity of n:
            513 bins are consistent with n=1024 *and* n=1025, and those imply
            different `Tobs` and `df`.
            For WDM data it is derived from ``wdm_grid`` when omitted.
        channel_names: TDI channel names in this run, e.g. ("A", "E", "T").
        tdi_generation: TDI generation string, e.g. "1.5" or "2.0".
        physical_observable: Physical interpretation of the data. Recommended values:
            "fractional_frequency" (relative frequency deviation dnu/nu, the
            LDC / lisainstrument default), "phase" (radians), "strain".
            Campaign-specific strings are allowed; every block reads this
            one field, so agreement is by construction -- state it once,
            correctly, rather than letting each group assume its own.
        data_domain: "time" (default), "frequency", or "wdm". Selects the channel_data
            representation described above. A block's TDI signal contribution
            must use the same representation and grid.
        wdm_grid: Required WDM division counts when ``data_domain="wdm"``;
            must be None in other domains. Sampling interval and epoch retain
            their meanings for the underlying time series. Keyword-only.
        start_time_gps: GPS seconds corresponding to sample index 0. Defaults to
            ``0.0`` -- fine for synthetic data with no absolute-time
            reference. Set it for real data: it anchors the constellation
            response (spacecraft positions at `start_time_gps + n*dt`), the
            orbit-span check, and the frequency-domain phase reference.
            Shadowed by `t0`.
        orbit_ephemeris: The LISA constellation ephemeris  --
            the spacecraft positions every block must share to build its
            response (see `enchilada.orbits.Orbit`). Currently a *fixed* property
            of the dataset, like `start_time_gps`/`tdi_generation`: set it once
            on the observed data and the Wheel copies it unchanged. Blocks read
            `data.orbit_ephemeris` so every piece uses the same constellation.
            `None` means no ephemeris is supplied; blocks requiring one must
            provide a configured orbit or reject the data.

    Noise covariance is supplied separately to the Wheel and to each block as
    a `DataCovariance`; it is not part of the observation container.

    Derived properties are exposed under both descriptive long names and
    the short symbols LISA papers use. Both spellings return the same value
    -- pick whichever reads better in context, but prefer *one* consistently
    within a given block or script so readers are not tracking two
    vocabularies. Call `L1Data.aliases()` for the full long-to-short table.

    Equality is identity (`eq=False`). A generated `__eq__` would compare the
    channel_data arrays elementwise and raise "truth value of an array is ambiguous",
    so `r1 == r2` is True only for the same object. Use `numpy.allclose`
    on the arrays to compare contents.

    """

    channel_data: dict[str, np.ndarray]
    sample_rate_hz: float
    channel_names: tuple[str, ...]
    tdi_generation: str
    physical_observable: str
    num_time_samples: int = 0
    start_time_gps: float = 0.0
    data_domain: str = "time"
    orbit_ephemeris: Orbit | None = None
    wdm_grid: WDMGrid | None = field(default=None, kw_only=True)

    DOMAINS: ClassVar[tuple[str, ...]] = ("time", "frequency", "wdm")
    """Valid values for `data_domain`."""

    RECOMMENDED_OBSERVABLES: ClassVar[tuple[str, ...]] = (
        "fractional_frequency",
        "phase",
        "strain",
    )
    """Common `physical_observable` values; campaign-specific strings are allowed."""

    # ---- consistency validation ------------------------------------------

    def __post_init__(self) -> None:
        """Validate the data contract; runs on every construction/replace.

        Ordered by dependency: conventions first (so `data_domain` is known),
        then the channel arrays (so an array length can be read), then
        `num_time_samples` (derived from that length, or required), then the
        length and orbit checks that need it.
        """
        self._validate_conventions()
        self._validate_channel_data_structure()
        self._resolve_and_check_num_time_samples()
        self._validate_channel_data_lengths()
        self._validate_orbit_span()

    def _validate_conventions(self) -> None:
        """Scalar run settings: rates, epoch, and convention strings."""
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError(
                f"sample_rate_hz must be a positive finite number in Hz, "
                f"got {self.sample_rate_hz!r}"
            )
        if not np.isfinite(self.start_time_gps):
            raise ValueError(
                f"start_time_gps must be finite GPS seconds, "
                f"got {self.start_time_gps!r}"
            )
        if not isinstance(self.tdi_generation, str) or not self.tdi_generation:
            raise ValueError(
                f"tdi_generation must be a non-empty string, "
                f"got {self.tdi_generation!r}"
            )
        if (
            not isinstance(self.physical_observable, str)
            or not self.physical_observable
        ):
            raise ValueError(
                f"physical_observable must be a non-empty string saying what "
                f"the TDI samples physically are, got {self.physical_observable!r}; "
                f"recommended values: "
                f"{', '.join(self.RECOMMENDED_OBSERVABLES)}"
            )
        if self.data_domain not in self.DOMAINS:
            raise ValueError(
                f"data_domain must be one of {self.DOMAINS}, got {self.data_domain!r}"
            )
        if self.data_domain == "wdm":
            if self.wdm_grid is None:
                raise ValueError("wdm_grid is required when data_domain='wdm'")
            if not isinstance(self.wdm_grid, WDMGrid):
                raise TypeError("wdm_grid must be a WDMGrid")
        elif self.wdm_grid is not None:
            raise ValueError("wdm_grid must be None outside data_domain='wdm'")

    def _validate_channel_data_structure(self) -> None:
        """Require channel arrays with the rank of the declared representation."""
        if not isinstance(self.channel_data, dict):
            raise TypeError(
                f"channel_data must be a dict of channel -> array, "
                f"got {type(self.channel_data).__name__}"
            )
        if isinstance(self.channel_names, (list, tuple)) and not isinstance(
            self.channel_names, tuple
        ):
            # Store channel names as an immutable sequence for grid comparisons.
            object.__setattr__(self, "channel_names", tuple(self.channel_names))
        if not isinstance(self.channel_names, tuple):
            raise TypeError(
                f"channel_names must be a tuple of channel names, "
                f"got {type(self.channel_names).__name__}"
            )
        if not self.channel_names:
            raise ValueError("channel_names must be a non-empty tuple of channel names")
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError(f"channel_names contains duplicates: {self.channel_names}")
        if set(self.channel_data) != set(self.channel_names):
            missing = sorted(set(self.channel_names) - set(self.channel_data))
            extra = sorted(set(self.channel_data) - set(self.channel_names))
            raise ValueError(
                f"channel_data keys must match channel_names exactly; "
                f"missing {missing}, unexpected {extra}"
            )
        for ch in self.channel_names:
            arr = self.channel_data[ch]
            expected_ndim = 2 if self.data_domain == "wdm" else 1
            if not isinstance(arr, np.ndarray) or arr.ndim != expected_ndim:
                raise TypeError(
                    f"channel_data[{ch!r}] must be a {expected_ndim}-D numpy array, "
                    f"got {type(arr).__name__}"
                )

    def _resolve_and_check_num_time_samples(self) -> None:
        """Fill in `num_time_samples` from the data where that is exact.

        Time domain: read it off the arrays.
        WDM domain: the grid's division product fixes it exactly.
        Frequency domain: it cannot be recovered from the data (see the
        `num_time_samples` field docstring for why), so it must have been stated.
        Also validates an explicitly supplied value.
        """
        if self.num_time_samples == 0:  # sentinel: not supplied
            first = self.channel_data[self.channel_names[0]]
            if self.data_domain == "time":
                object.__setattr__(self, "num_time_samples", int(first.shape[0]))
            elif self.data_domain == "wdm":
                assert self.wdm_grid is not None  # checked with the conventions
                object.__setattr__(
                    self, "num_time_samples", self.wdm_grid.num_time_samples
                )
            else:
                n_bins = int(first.shape[0])
                raise ValueError(
                    f"num_time_samples must be given when data_domain='frequency': the "
                    f"{n_bins}-bin rfft grid does not determine it (it is "
                    f"consistent with num_time_samples={2 * (n_bins - 1)} and "
                    f"={2 * n_bins - 1}, which imply different Tobs and df). "
                    f"Pass the number of time-domain samples the spectrum came "
                    f"from; only time-domain data can have it derived."
                )
        if (
            not isinstance(self.num_time_samples, (int, np.integer))
            or isinstance(self.num_time_samples, bool)  # True would pass as 1
            or self.num_time_samples <= 0
        ):
            raise ValueError(
                f"num_time_samples must be a positive integer, "
                f"got {self.num_time_samples!r}"
            )
        if (
            self.wdm_grid is not None
            and self.num_time_samples != self.wdm_grid.num_time_samples
        ):
            raise ValueError(
                "num_time_samples must equal the WDM grid division product"
            )

    def _validate_channel_data_lengths(self) -> None:
        """Check each array's length and dtype against the data domain's grid."""
        if self.data_domain == "wdm":
            assert self.wdm_grid is not None  # checked with the conventions
            check_wdm_coefficients(self.channel_data, self.wdm_grid, "channel_data")
            return
        expected = (
            self.num_time_samples
            if self.data_domain == "time"
            else self.num_time_samples // 2 + 1
        )
        for ch in self.channel_names:
            arr = self.channel_data[ch]
            if arr.shape[0] != expected:
                raise ValueError(
                    f"channel_data[{ch!r}] has length {arr.shape[0]}, "
                    f"expected {expected} for data_domain={self.data_domain!r} "
                    f"with num_time_samples={self.num_time_samples} "
                    f"(num_time_samples always counts time-domain samples; "
                    f"frequency-domain "
                    f"arrays live on the rfft grid of length num_time_samples // 2 + 1)"
                )
            if self.data_domain == "time" and np.iscomplexobj(arr):
                raise TypeError(
                    f"channel_data[{ch!r}] is complex but data_domain='time'; "
                    f"time-domain TDI is real (did you mean data_domain='frequency'?)"
                )
            if self.data_domain == "frequency" and not np.iscomplexobj(arr):
                raise TypeError(
                    f"channel_data[{ch!r}] is real but data_domain='frequency'; "
                    f"a one-sided spectrum is complex. Accepting a real array "
                    f"here would let "
                    f"a block return `spectrum.real` and be silently credited "
                    f"with the whole imaginary part as its model."
                )
            # dtype is part of the contract too: integer or object arrays would
            # otherwise be accepted here and then fail deep inside the Wheel's
            # ledger arithmetic with a raw numpy casting error.
            if not np.issubdtype(arr.dtype, np.inexact):
                raise TypeError(
                    f"channel_data[{ch!r}] has dtype {arr.dtype}; "
                    f"TDI must be floating or complex (an integer or object array "
                    f"cannot carry a residual "
                    f"-- convert with .astype(float) first)"
                )
        if self.data_domain == "frequency":
            check_rfft_endpoints(
                self.channel_data,
                num_time_samples=self.num_time_samples,
                context_label="channel_data",
            )

    def _validate_orbit_span(self) -> None:
        """Check that a tabulated orbit (one exposing time_range_gps) covers the data
        span."""
        if self.orbit_ephemeris is None:
            return
        time_range_gps = getattr(self.orbit_ephemeris, "time_range_gps", None)
        if time_range_gps is None:
            return
        orbit_start_time_gps, orbit_end_time_gps = (
            float(time_range_gps[0]),
            float(time_range_gps[1]),
        )
        # The samples sit at start_time_gps + n*dt for n in [0, N-1], so the last
        # one is at start_time_gps + (N-1)*dt, one sample before start_time_gps + Tobs.
        # Requiring that extra sample would reject an orbit tabulated on the
        # data's own sample grid.
        #
        # Multiply `sample_interval_s` exactly as every grid builder here does
        # (`start_time_gps + arange(n) * dt`, see NumericOrbit.from_hdf5) rather than
        # dividing by sample_rate_hz: the two differ by an ulp for rates whose dt
        # is not exactly representable, and this is an exact float comparison,
        # so the mismatch would spuriously reject a data-grid orbit.
        #
        # This is a coarse check for gross epoch mismatches, not a guarantee: a
        # block applying TDI light-travel delays evaluates retarded times
        # slightly outside [start_time_gps, last_sample_time_gps] and needs margin (and
        # NumericOrbit.positions raises if asked beyond its table).
        last_sample_time_gps = (
            self.start_time_gps + (self.num_time_samples - 1) * self.sample_interval_s
        )
        if (
            orbit_start_time_gps > self.start_time_gps
            or orbit_end_time_gps < last_sample_time_gps
        ):
            raise ValueError(
                f"orbit_ephemeris spans "
                f"[{orbit_start_time_gps}, {orbit_end_time_gps}] s "
                f"but the data samples span "
                f"[{self.start_time_gps}, {last_sample_time_gps}] s; "
                f"every block would "
                f"need spacecraft positions outside the tabulated ephemeris "
                f"(mismatched epoch conventions? GPS vs zero-based times?)"
            )

    # ---- descriptive (long) names ---------------------------------------

    @property
    def observation_duration_s(self) -> float:
        """Total observation duration in seconds. Shadowed by `Tobs`."""
        return self.num_time_samples / self.sample_rate_hz

    @property
    def sample_interval_s(self) -> float:
        """Seconds between consecutive samples. Shadowed by `dt`."""
        return 1.0 / self.sample_rate_hz

    @property
    def frequency_resolution_hz(self) -> float:
        """Width of a Fourier bin, in Hz. Shadowed by `df`."""
        return 1.0 / self.observation_duration_s

    @property
    def nyquist_frequency_hz(self) -> float:
        """Nyquist frequency, in Hz. Shadowed by `fny`."""
        return self.sample_rate_hz / 2.0

    # ---- conventional short-name shadows --------------------------------

    @property
    def Tobs(self) -> float:
        """LISA shorthand for `observation_duration_s` (seconds)."""
        return self.observation_duration_s

    @property
    def fs(self) -> float:
        """LISA shorthand for `sample_rate_hz` (Hz)."""
        return self.sample_rate_hz

    @property
    def dt(self) -> float:
        """LISA shorthand for `sample_interval_s` (seconds)."""
        return self.sample_interval_s

    @property
    def N(self) -> int:
        """LISA shorthand for `num_time_samples`."""
        return self.num_time_samples

    @property
    def df(self) -> float:
        """LISA shorthand for `frequency_resolution_hz` (Hz)."""
        return self.frequency_resolution_hz

    @property
    def fny(self) -> float:
        """LISA shorthand for `nyquist_frequency_hz` (Hz)."""
        return self.nyquist_frequency_hz

    @property
    def t0(self) -> float:
        """LISA shorthand for `start_time_gps` (GPS seconds)."""
        return self.start_time_gps

    # ---- domain transforms (fixing the campaign's FFT convention) ------

    def to_frequency(self) -> "L1Data":
        """Transform the dataset to a one-sided dft (``data_domain="frequency"``).

        Applies the campaign's Fourier convention -- ``X(f) = dt * rfft(x)``,
        consistent with `DataCovariance.from_psd`. ``num_time_samples`` is preserved
        to ensure the transform is invertible (see :meth:`to_time`).

        Data enters a campaign as a time series, so this is the normal way to
        get a frequency-domain residual: build `L1Data` from the time
        series (where `num_time_samples` is read off the arrays) and transform. You
        then never state `num_time_samples` by hand at all.

        Revalidates the source and returns independently owned channel arrays,
        including when the source is already in the frequency domain.
        """
        from enchilada.translation import transform

        return transform(self, "frequency")

    def to_time(self) -> "L1Data":
        """Transform the dataset to a time series (``data_domain="time"``).

        Inverts :meth:`to_frequency` exactly -- ``x = irfft(X / dt, n)`` -- for
        *either* parity of ``n``, because ``num_time_samples`` travelled with the data.
        A bare spectrum with no ``num_time_samples`` cannot be inverted this way: its
        ``n // 2 + 1`` bins are consistent with both ``2*(bins-1)`` and
        ``2*bins-1``, and choosing wrong silently resamples the series.

        Revalidates the source and returns independently owned channel arrays,
        including when the source is already in the time domain.
        """
        from enchilada.translation import transform

        return transform(self, "time")

    # ---- block results ------------------------------------------------------

    def block_result(
        self,
        tdi_signal_contribution: dict[str, np.ndarray] | None = None,
        *,
        noise_covariance: DataCovariance | TranslatedCovariance | None = None,
        model_parameters: dict[str, Any] | None = None,
        sampler_state: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BlockResult:
        """Build the `BlockResult` a block returns, checked against this grid.

        `tdi_signal_contribution` sums this block's sources in each TDI channel,
        with this residual's channel names, lengths, and real/complex dtype.
        None means zero contribution, removing any signal previously returned
        by this block. The Wheel performs all subtraction on the block's behalf.
        `noise_covariance` optionally replaces the run's full covariance.

            def sample(self, conditional_residual, noise_covariance,
                       current_block_result, *, rng):
                ...fit using the supplied data, covariance, and sampler state...
                return conditional_residual.block_result(
                    tdi_signal_contribution=signals,
                    model_parameters=params, sampler_state=continuation,
                )

        `model_parameters` describes the physical model used to produce the
        estimates. `sampler_state` preserves algorithm continuation information.
        Together with `metadata`, these form the complete snapshot stored in
        the ledger and supplied on the next sample call.

        Validated here, where a wrong length or dtype is the block author's
        mistake to see, rather than on return. The arrays are taken as given
        (not copied): the Wheel copies what it records, so reusing your own
        buffer between cycles is safe.
        """
        result = BlockResult(
            tdi_signal_contribution=tdi_signal_contribution,
            noise_covariance=noise_covariance,
            model_parameters={} if model_parameters is None else model_parameters,
            sampler_state={} if sampler_state is None else sampler_state,
            metadata={} if metadata is None else metadata,
        )
        if result.tdi_signal_contribution is not None:
            check_on_grid(
                result.tdi_signal_contribution,
                channel_names=self.channel_names,
                num_time_samples=self.num_time_samples,
                data_domain=self.data_domain,
                context_label="tdi_signal_contribution",
                wdm_grid=self.wdm_grid,
            )
        return result

    def zero_block_result(
        self,
        *,
        noise_covariance: DataCovariance | TranslatedCovariance | None = None,
        model_parameters: dict[str, Any] | None = None,
        sampler_state: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BlockResult:
        """A `BlockResult` with a zero signal on this grid: nothing to subtract.

        Allocates fresh zero arrays matching each channel's shape and dtype.
        Use this when explicit arrays are useful. A source-free or noise-only
        block can instead leave `tdi_signal_contribution=None` without allocating
        any arrays:

            return BlockResult(model_parameters={"sources": []})
            return BlockResult(noise_covariance=covariance)
        """
        return self.block_result(
            {ch: np.zeros_like(arr) for ch, arr in self.channel_data.items()},
            noise_covariance=noise_covariance,
            model_parameters=model_parameters,
            sampler_state=sampler_state,
            metadata=metadata,
        )

    # ---- discoverability ------------------------------------------------

    ALIASES: ClassVar[dict[str, str]] = {
        "observation_duration_s": "Tobs",
        "sample_rate_hz": "fs",
        "sample_interval_s": "dt",
        "num_time_samples": "N",
        "frequency_resolution_hz": "df",
        "nyquist_frequency_hz": "fny",
        "start_time_gps": "t0",
    }
    """Long-name -> short-name table. Both spellings are valid attributes
    on every `L1Data` instance and return the same value."""

    @classmethod
    def aliases(cls) -> dict[str, str]:
        """Return a copy of the long-name -> short-name table.

        Useful for users learning the convention:

            >>> for long, short in L1Data.aliases().items():
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
        "N_samples": "num_time_samples",
        "n_Samples": "num_time_samples",
        "Nsamples": "num_time_samples",
        "d_f": "df",
        "f_ny": "fny",
        "F_ny": "fny",
        "f_nyquist": "nyquist_frequency_hz",
        "nyquist": "nyquist_frequency_hz",
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
    #   "L1Data" has no attribute "Tobbs"; maybe "Tobs"?
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
