from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Never

import numpy as np

from enchilada.block_state import BlockState, check_on_grid, check_rfft_endpoints
from enchilada.covariance import DataCovariance
from enchilada.domains import WDMGrid, check_wdm_coefficients
from enchilada.orbits import Orbit
from enchilada.translated_covariance import TranslatedCovariance


@dataclass(frozen=True, eq=False, init=False)
class L1Data:
    """Observations or residuals with their sampling grid and physical conventions.

    Construction validates channel names, array shapes and dtypes, conventions,
    and any orbit's declared time bounds. Wheel supplies each block with a
    conditional residual and accepts a `BlockState` in return.
    Only `channel_data` may be positional; all settings are keyword-only.

    Fields:
        channel_data: Channel name -> real or complex floating-point array, with
            keys matching `channel_names`. Time arrays contain `num_time_samples`
            real values.
            Frequency arrays contain `num_time_samples // 2 + 1` complex values
            with normalization ``dt * rfft(x)``; DC and even-length Nyquist are
            real. WDM arrays contain real coefficients shaped
            ``(num_frequency_divisions + 1, num_time_divisions)``; inactive
            edge coefficients must be exactly zero.
        sample_rate_hz: Samples per second, in Hz.
        num_time_samples: Underlying time sample count per channel in every
            domain. When omitted, derived from time or WDM array shapes.
            Required for frequency input because an rfft length is ambiguous
            between even and odd time sample counts.
        channel_names: Ordered TDI channel names, e.g. ("A", "E", "T"). Omitted
            or None uses `channel_data` key order. An explicit order overrides
            dictionary order and must match covariance matrix channel axes.
        tdi_generation: TDI generation string, e.g. "1.5" or "2.0".
        physical_observable: Data meaning, typically "fractional_frequency"
            (dnu/nu), "phase" (radians), or "strain". Custom strings are allowed.
        data_domain: "time" (default), "frequency", or "wdm". Returned block
            signals must match their input residual's representation and grid.
        wdm_grid: Read-only metadata derived from WDM array shapes; None in
            other domains. Managed automatically, not a constructor argument.
            Sample rate and epoch still refer to the time series.
        start_time_gps: GPS seconds at sample index zero; defaults to 0.0.
            Defines sample times ``start_time_gps + n*dt``. Alias: `t0`.
        orbit_ephemeris: Optional `Orbit`, shared by reference with blocks and
            treated as immutable. Declared `time_range_gps` must cover all sample
            times. Blocks requiring an orbit must supply one or reject its absence.

    Fields are frozen, but the supplied arrays and dictionary are retained without
    copying. Wheel makes its own snapshots. Covariance is supplied separately as
    `DataCovariance`, `TranslatedCovariance`, or None. See :meth:`aliases` for
    scientific shorthand. Equality compares identity; compare arrays explicitly
    for numerical equality.
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
    wdm_grid: WDMGrid | None = field(default=None, init=False)

    DOMAINS: ClassVar[tuple[str, ...]] = ("time", "frequency", "wdm")
    """Valid values for `data_domain`."""

    RECOMMENDED_OBSERVABLES: ClassVar[tuple[str, ...]] = (
        "fractional_frequency",
        "phase",
        "strain",
    )
    """Common `physical_observable` values; campaign-specific strings are allowed."""

    def __init__(
        self,
        channel_data: dict[str, np.ndarray],
        *,
        sample_rate_hz: float,
        tdi_generation: str,
        physical_observable: str,
        channel_names: tuple[str, ...] | None = None,
        num_time_samples: int = 0,
        start_time_gps: float = 0.0,
        data_domain: str = "time",
        orbit_ephemeris: Orbit | None = None,
    ) -> None:
        # Accept optional names at construction while exposing a resolved tuple.
        object.__setattr__(self, "channel_data", channel_data)
        object.__setattr__(self, "sample_rate_hz", sample_rate_hz)
        object.__setattr__(self, "channel_names", channel_names)
        object.__setattr__(self, "tdi_generation", tdi_generation)
        object.__setattr__(self, "physical_observable", physical_observable)
        object.__setattr__(self, "num_time_samples", num_time_samples)
        object.__setattr__(self, "start_time_gps", start_time_gps)
        object.__setattr__(self, "data_domain", data_domain)
        object.__setattr__(self, "orbit_ephemeris", orbit_ephemeris)
        self.__post_init__()

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

    def _validate_channel_data_structure(self) -> None:
        """Require channel arrays with the rank of the declared representation."""
        if not isinstance(self.channel_data, dict):
            raise TypeError(
                f"channel_data must be a dict of channel -> array, "
                f"got {type(self.channel_data).__name__}"
            )
        if self.channel_names is None:
            object.__setattr__(self, "channel_names", tuple(self.channel_data))
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
        """Derive WDM metadata and fill in the time sample count where exact.

        Time domain: read it off the arrays.
        WDM domain: coefficient shape fixes the grid and its division product.
        Frequency domain: it cannot be recovered from the data (see the
        `num_time_samples` field docstring for why), so it must have been stated.
        Also validates an explicitly supplied value.
        """
        first = self.channel_data[self.channel_names[0]]
        grid = None
        if self.data_domain == "wdm":
            grid = WDMGrid(
                num_frequency_divisions=first.shape[0] - 1,
                num_time_divisions=first.shape[1],
            )
        object.__setattr__(self, "wdm_grid", grid)
        if self.num_time_samples == 0:  # sentinel: not supplied
            if self.data_domain == "time":
                object.__setattr__(self, "num_time_samples", int(first.shape[0]))
            elif self.data_domain == "wdm":
                assert self.wdm_grid is not None
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
                    f"from."
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
            assert self.wdm_grid is not None
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
        """Total observation duration in seconds. Alias: `Tobs`."""
        return self.num_time_samples / self.sample_rate_hz

    @property
    def sample_interval_s(self) -> float:
        """Seconds between consecutive samples. Alias: `dt`."""
        return 1.0 / self.sample_rate_hz

    @property
    def frequency_resolution_hz(self) -> float:
        """Width of a Fourier bin, in Hz. Alias: `df`."""
        return 1.0 / self.observation_duration_s

    @property
    def nyquist_frequency_hz(self) -> float:
        """Nyquist frequency, in Hz. Alias: `fny`."""
        return self.sample_rate_hz / 2.0

    # ---- conventional short-name aliases --------------------------------

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
        """Return a frequency representation from time, frequency, or WDM data.

        Uses ``X = dt * rfft(x)`` and preserves `num_time_samples`. Delegates to
        :func:`enchilada.transform`, revalidating the source and copying channel
        arrays even when already in the frequency domain.
        """
        from enchilada.translation import transform

        return transform(self, "frequency")

    def to_time(self) -> "L1Data":
        """Return a time representation from time, frequency, or WDM data.

        Fourier input uses ``irfft(X / dt, n=num_time_samples)``, preserving even
        and odd lengths. Delegates to :func:`enchilada.transform`, revalidating
        the source and copying channel arrays even when already in the time domain.
        """
        from enchilada.translation import transform

        return transform(self, "time")

    # ---- block states ------------------------------------------------------

    def block_state(
        self,
        template: dict[str, np.ndarray] | None = None,
        *,
        covariance_matrix: DataCovariance | TranslatedCovariance | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> BlockState:
        """Build a `BlockState` and validate its signal against this grid.

        The summed TDI signal must match this object's channels, shape, and
        real/complex representation. None means zero contribution, replacing
        any prior signal. Optional `covariance_matrix` replaces the full covariance
        on adoption; its compatibility is checked by Wheel, not this helper.

        Physical parameters follow the `BlockState` contract.
        Supplied arrays and dictionaries are retained without copying; Wheel
        copies the state when adopting it.
        """
        state = BlockState(
            template=template,
            covariance_matrix=covariance_matrix,
            parameters={} if parameters is None else parameters,
        )
        if state.template is not None:
            check_on_grid(
                state.template,
                channel_names=self.channel_names,
                num_time_samples=self.num_time_samples,
                data_domain=self.data_domain,
                context_label="template",
                wdm_grid=self.wdm_grid,
            )
        return state

    def zero_block_state(
        self,
        *,
        covariance_matrix: DataCovariance | TranslatedCovariance | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> BlockState:
        """A `BlockState` with a zero signal on this grid: nothing to subtract.

        Allocates fresh zero arrays matching each channel's shape and dtype.
        Use this when explicit arrays are useful. A source-free or noise-only
        block can instead leave `template=None` without allocating
        any arrays:

            return BlockState(parameters={"sources": []})
            return BlockState(covariance_matrix=covariance)
        """
        return self.block_state(
            {ch: np.zeros_like(arr) for ch, arr in self.channel_data.items()},
            covariance_matrix=covariance_matrix,
            parameters=parameters,
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
