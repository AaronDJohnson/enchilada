from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Never

import numpy as np

from enchilada.block_state import BlockState, check_on_grid, check_rfft_endpoints
from enchilada.covariance import Covariance
from enchilada.domains import WDMGrid, check_wdm_coefficients

LINKS: tuple[int, ...] = (12, 23, 31, 13, 32, 21)
"""Row order of `L1Data.light_travel_times_s`; ``ij`` is received at i from j.

This is the lisaorbits and LDC/Mojito ordering of directed links.
"""

TTL_COEFFICIENT_NAMES: tuple[str, ...] = tuple(
    f"c_{link}_{angle}_{end}"
    for link in LINKS
    for angle in ("eta", "phi")
    for end in ("rx", "tx")
)
"""Mojito TTL coefficient dataset names, preserved as dictionary keys."""

CD1L_CHANNELS: tuple[str, ...] = ("X", "Y", "Z")
"""TDI channels `L1Data.from_cd1l` reads, from the ``tdis/X2`` etc. datasets."""


@dataclass(frozen=True, eq=False, init=False)
class L1Data:
    """Observations or residuals with their sampling grid and physical conventions.

    Construction validates channel names, array shapes and dtypes, conventions,
    and raw orbit table sampling and time coverage. Wheel supplies each block with a
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
            dictionary order and is the channel order a `Covariance` uses.
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
        spacecraft_positions_m: Optional raw orbit table, shaped
            ``(3, num_orbit_samples, 3)`` (spacecraft, time, xyz), in metres and
            in the frame of its source; Mojito files store BCRS/ICRS equatorial
            coordinates. Requires `orbit_start_time_gps` and
            `orbit_sample_interval_s`.
        spacecraft_velocities_m_per_s: Optional raw velocities in metres per
            second, shaped and timed like `spacecraft_positions_m`.
        orbit_start_time_gps: GPS time of the first orbit sample.
        orbit_sample_interval_s: Seconds between orbit samples. Orbit sample
            times are `orbit_sample_times_gps`.
        light_travel_times_s: Optional raw directed light travel times in
            seconds, shaped ``(6, num_light_travel_time_samples)`` with rows in
            `LINKS` order, indexed by reception time. Requires
            `light_travel_time_start_time_gps` and
            `light_travel_time_sample_interval_s`.
        light_travel_time_derivatives: Optional dimensionless d(ltt)/dt,
            with the same shape, link order, and sampling as light_travel_times_s.
        scet_wrt_tcb_s: Optional raw spacecraft clock offsets in seconds,
            shaped (3, num_clock_samples), ordered spacecraft 1, 2, 3.
        clock_start_time_gps: First clock table sample time.
        clock_sample_interval_s: Clock table sampling interval in seconds.
        ttl_coefficients: Optional mapping of the 24 Mojito coefficient names
            (c_{link}_{eta|phi}_{rx|tx}) to 1-D raw arrays in source units.
            All coefficient arrays share the TTL sampling and sample count.
        ttl_start_time_gps: First TTL coefficient sample time.
        ttl_sample_interval_s: TTL coefficient sampling interval in seconds.
        light_travel_time_start_time_gps: GPS time of the first light travel time.
        light_travel_time_sample_interval_s: Seconds between light travel
            times. Their sample times are `light_travel_time_sample_times_gps`.

    Each raw orbit table must span every data sample time. Its values are taken
    as given: no interpolation, rotation, or resampling. Blocks share the arrays
    by reference and must treat them as immutable.

    Fields are frozen, but the supplied arrays and dictionary are retained without
    copying. Wheel makes its own snapshots. Covariance is supplied separately as
    a `Covariance` or None. See :meth:`aliases` for
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
    spacecraft_positions_m: np.ndarray | None = None
    spacecraft_velocities_m_per_s: np.ndarray | None = None
    orbit_start_time_gps: float | None = None
    orbit_sample_interval_s: float | None = None
    light_travel_times_s: np.ndarray | None = None
    light_travel_time_start_time_gps: float | None = None
    light_travel_time_sample_interval_s: float | None = None
    light_travel_time_derivatives: np.ndarray | None = None
    scet_wrt_tcb_s: np.ndarray | None = None
    clock_start_time_gps: float | None = None
    clock_sample_interval_s: float | None = None
    ttl_coefficients: dict[str, np.ndarray] | None = None
    ttl_start_time_gps: float | None = None
    ttl_sample_interval_s: float | None = None
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
        spacecraft_positions_m: np.ndarray | None = None,
        spacecraft_velocities_m_per_s: np.ndarray | None = None,
        orbit_start_time_gps: float | None = None,
        orbit_sample_interval_s: float | None = None,
        light_travel_times_s: np.ndarray | None = None,
        light_travel_time_start_time_gps: float | None = None,
        light_travel_time_sample_interval_s: float | None = None,
        light_travel_time_derivatives: np.ndarray | None = None,
        scet_wrt_tcb_s: np.ndarray | None = None,
        clock_start_time_gps: float | None = None,
        clock_sample_interval_s: float | None = None,
        ttl_coefficients: dict[str, np.ndarray] | None = None,
        ttl_start_time_gps: float | None = None,
        ttl_sample_interval_s: float | None = None,
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
        object.__setattr__(self, "spacecraft_positions_m", spacecraft_positions_m)
        object.__setattr__(
            self, "spacecraft_velocities_m_per_s", spacecraft_velocities_m_per_s
        )
        object.__setattr__(self, "orbit_start_time_gps", orbit_start_time_gps)
        object.__setattr__(self, "orbit_sample_interval_s", orbit_sample_interval_s)
        object.__setattr__(self, "light_travel_times_s", light_travel_times_s)
        object.__setattr__(
            self, "light_travel_time_start_time_gps", light_travel_time_start_time_gps
        )
        object.__setattr__(
            self,
            "light_travel_time_sample_interval_s",
            light_travel_time_sample_interval_s,
        )
        object.__setattr__(
            self, "light_travel_time_derivatives", light_travel_time_derivatives
        )
        object.__setattr__(self, "scet_wrt_tcb_s", scet_wrt_tcb_s)
        object.__setattr__(self, "clock_start_time_gps", clock_start_time_gps)
        object.__setattr__(self, "clock_sample_interval_s", clock_sample_interval_s)
        object.__setattr__(self, "ttl_coefficients", ttl_coefficients)
        object.__setattr__(self, "ttl_start_time_gps", ttl_start_time_gps)
        object.__setattr__(self, "ttl_sample_interval_s", ttl_sample_interval_s)
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
        self._validate_raw_orbit_tables()

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

    def _validate_raw_orbit_tables(self) -> None:
        """Check raw orbit tables' structure and that each spans every sample time.

        Constant time: table values are not scanned, because every replace
        revalidates and a full-rate light-travel-time table is large.
        """
        last_sample_time_gps = (
            self.start_time_gps + (self.num_time_samples - 1) * self.sample_interval_s
        )
        positions = self.spacecraft_positions_m
        velocities = self.spacecraft_velocities_m_per_s
        num_orbit_samples = None
        if positions is not None:
            _check_raw_table(positions, "spacecraft_positions_m", ndim=3)
            if positions.shape[0] != 3 or positions.shape[2] != 3:
                raise ValueError(
                    "spacecraft_positions_m must have shape (3, num_orbit_samples, 3) "
                    f"(spacecraft, time, xyz), got {positions.shape}"
                )
            num_orbit_samples = positions.shape[1]
        if velocities is not None:
            if positions is None:
                raise ValueError(
                    "spacecraft_velocities_m_per_s requires spacecraft_positions_m"
                )
            _check_raw_table(velocities, "spacecraft_velocities_m_per_s", ndim=3)
            if velocities.shape != positions.shape:
                raise ValueError(
                    f"spacecraft_velocities_m_per_s has shape {velocities.shape}; "
                    f"it must match spacecraft_positions_m {positions.shape}"
                )
        _check_raw_sampling(
            table_label="spacecraft_positions_m",
            prefix="orbit",
            num_samples=num_orbit_samples,
            start_time_gps=self.orbit_start_time_gps,
            sample_interval_s=self.orbit_sample_interval_s,
            data_span_gps=(self.start_time_gps, last_sample_time_gps),
        )
        light_travel_times = self.light_travel_times_s
        num_light_travel_time_samples = None
        if light_travel_times is not None:
            _check_raw_table(light_travel_times, "light_travel_times_s", ndim=2)
            if light_travel_times.shape[0] != len(LINKS):
                raise ValueError(
                    "light_travel_times_s must have one row per link in "
                    f"{LINKS}, got shape {light_travel_times.shape}"
                )
            num_light_travel_time_samples = light_travel_times.shape[1]
        _check_raw_sampling(
            table_label="light_travel_times_s",
            prefix="light_travel_time",
            num_samples=num_light_travel_time_samples,
            start_time_gps=self.light_travel_time_start_time_gps,
            sample_interval_s=self.light_travel_time_sample_interval_s,
            data_span_gps=(self.start_time_gps, last_sample_time_gps),
        )

        derivatives = self.light_travel_time_derivatives
        if derivatives is not None:
            if light_travel_times is None:
                raise ValueError(
                    "light_travel_time_derivatives requires light_travel_times_s"
                )
            _check_raw_table(derivatives, "light_travel_time_derivatives", ndim=2)
            if derivatives.shape != light_travel_times.shape:
                raise ValueError(
                    "light_travel_time_derivatives must match light_travel_times_s"
                )
        clocks = self.scet_wrt_tcb_s
        if clocks is not None:
            _check_raw_table(clocks, "scet_wrt_tcb_s", ndim=2)
            if clocks.shape[0] != 3:
                raise ValueError("scet_wrt_tcb_s must have three spacecraft rows")
        _check_raw_sampling(
            table_label="scet_wrt_tcb_s",
            prefix="clock",
            num_samples=None if clocks is None else clocks.shape[1],
            start_time_gps=self.clock_start_time_gps,
            sample_interval_s=self.clock_sample_interval_s,
            data_span_gps=(self.start_time_gps, last_sample_time_gps),
        )
        coefficients = self.ttl_coefficients
        num_ttl_samples = None
        if coefficients is not None:
            if not isinstance(coefficients, dict) or set(coefficients) != set(
                TTL_COEFFICIENT_NAMES
            ):
                raise ValueError(
                    "ttl_coefficients must contain all 24 Mojito coefficient names"
                )
            for name, values in coefficients.items():
                _check_raw_table(values, f"ttl_coefficients[{name!r}]", ndim=1)
                if num_ttl_samples is not None and values.size != num_ttl_samples:
                    raise ValueError(
                        "ttl_coefficients must have matching sample counts"
                    )
                num_ttl_samples = values.size
        _check_raw_sampling(
            table_label="ttl_coefficients",
            prefix="ttl",
            num_samples=num_ttl_samples,
            start_time_gps=self.ttl_start_time_gps,
            sample_interval_s=self.ttl_sample_interval_s,
            data_span_gps=(self.start_time_gps, last_sample_time_gps),
        )

    # ---- raw orbit tables -------------------------------------------------

    @property
    def orbit_sample_times_gps(self) -> np.ndarray | None:
        """GPS time of each orbit sample, or None without an orbit table."""
        if self.spacecraft_positions_m is None:
            return None
        assert self.orbit_start_time_gps is not None
        assert self.orbit_sample_interval_s is not None
        return np.asarray(
            self.orbit_start_time_gps
            + np.arange(self.spacecraft_positions_m.shape[1])
            * self.orbit_sample_interval_s,
            dtype=np.float64,
        )

    @property
    def light_travel_time_sample_times_gps(self) -> np.ndarray | None:
        """GPS reception time of each light-travel-time sample, or None."""
        if self.light_travel_times_s is None:
            return None
        assert self.light_travel_time_start_time_gps is not None
        assert self.light_travel_time_sample_interval_s is not None
        return np.asarray(
            self.light_travel_time_start_time_gps
            + np.arange(self.light_travel_times_s.shape[1])
            * self.light_travel_time_sample_interval_s,
            dtype=np.float64,
        )

    @property
    def clock_sample_times_gps(self) -> np.ndarray | None:
        """Sample times of the raw clock tables, or None."""
        if self.scet_wrt_tcb_s is None:
            return None
        assert self.clock_start_time_gps is not None
        assert self.clock_sample_interval_s is not None
        return (
            self.clock_start_time_gps
            + np.arange(self.scet_wrt_tcb_s.shape[1]) * self.clock_sample_interval_s
        )

    @property
    def ttl_sample_times_gps(self) -> np.ndarray | None:
        """Sample times of the raw ttl tables, or None."""
        if self.ttl_coefficients is None:
            return None
        assert self.ttl_start_time_gps is not None
        assert self.ttl_sample_interval_s is not None
        return (
            self.ttl_start_time_gps
            + np.arange(next(iter(self.ttl_coefficients.values())).size)
            * self.ttl_sample_interval_s
        )

    # ---- readers ----------------------------------------------------------

    @classmethod
    def from_cd1l(cls, file_path: str, *, duration_s: float | None = None) -> "L1Data":
        """Read a Mojito CD1-L brick: X, Y, Z TDI-2 with its raw orbit tables.

        Channels are ``tdis/X2``, ``Y2`` and ``Z2`` divided by the file's
        ``laser_frequency`` attribute, giving fractional frequency, sampled per
        ``tdis/sampling`` (``fs``, ``t0``, ``size``). The orbit tables are copied
        as stored, without rotation or resampling, and made read-only:
        ``orbits/sc_position_i`` and, when present, ``sc_velocity_i`` (BCRS/ICRS
        equatorial metres and metres per second) with ``orbits/sampling``; and
        ``ltts/ltt_ij`` in `LINKS` order with ``ltts/sampling``. Light travel
        times are cut to the samples covering the data read, plus one. The
        tables are the only orbit representation. Optional ltt_derivative_ij,
        scet_wrt_tcb/scet_wrt_tcb_i, and ttl/ttl_coefficients/c_* tables are
        also loaded unchanged and read-only. Clock and TTL sampling are separate;
        these tables are cut to the data span plus one sample as for ltts.
        Missing optional groups leave fields None; incomplete groups raise.
        TTL tables can be very large; duration_s limits the loaded prefix.
        Requires h5py.

        Args:
            file_path: Path to the CD1-L HDF5 file.
            duration_s: Read only the first ``ceil(duration_s * fs)`` samples;
                None reads the whole brick.
        """
        import h5py  # optional: the cd1l extra

        with h5py.File(file_path, "r") as cd1l_file:
            tdi_sampling = cd1l_file["tdis/sampling"].attrs
            sample_rate_hz = float(tdi_sampling["fs"])
            start_time_gps = float(tdi_sampling["t0"])
            available_samples = int(tdi_sampling["size"])
            num_time_samples = (
                available_samples
                if duration_s is None
                else int(np.ceil(duration_s * sample_rate_hz))
            )
            if not 0 < num_time_samples <= available_samples:
                raise ValueError(
                    f"duration_s={duration_s!r} selects {num_time_samples} samples; "
                    f"the file holds between 1 and {available_samples}"
                )
            laser_frequency_hz = float(cd1l_file.attrs["laser_frequency"])
            channel_data = {
                channel: np.asarray(
                    cd1l_file[f"tdis/{channel}2"][:num_time_samples], dtype=float
                )
                / laser_frequency_hz
                for channel in CD1L_CHANNELS
            }

            orbit_tables: dict[str, Any] = {}
            if "orbits" in cd1l_file:
                orbit_group = cd1l_file["orbits"]
                orbit_sampling = orbit_group["sampling"].attrs
                orbit_tables["orbit_start_time_gps"] = float(orbit_sampling["t0"])
                orbit_tables["orbit_sample_interval_s"] = float(orbit_sampling["dt"])
                orbit_tables["spacecraft_positions_m"] = _read_only(
                    np.stack(
                        [
                            np.asarray(orbit_group[f"sc_position_{sc}"], dtype=float)
                            for sc in (1, 2, 3)
                        ]
                    )
                )
                if all(f"sc_velocity_{sc}" in orbit_group for sc in (1, 2, 3)):
                    orbit_tables["spacecraft_velocities_m_per_s"] = _read_only(
                        np.stack(
                            [
                                np.asarray(
                                    orbit_group[f"sc_velocity_{sc}"], dtype=float
                                )
                                for sc in (1, 2, 3)
                            ]
                        )
                    )
            if "ltts" in cd1l_file:
                ltt_group = cd1l_file["ltts"]
                ltt_sampling = ltt_group["sampling"].attrs
                ltt_start_time_gps = float(ltt_sampling["t0"])
                ltt_sample_interval_s = float(ltt_sampling["dt"])
                last_sample_time_gps = start_time_gps + (num_time_samples - 1) * (
                    1.0 / sample_rate_hz
                )
                # Samples up to the last data time, plus one against rounding.
                needed = (
                    int(
                        np.ceil(
                            (last_sample_time_gps - ltt_start_time_gps)
                            / ltt_sample_interval_s
                        )
                    )
                    + 2
                )
                stop = min(int(ltt_sampling["size"]), max(needed, 1))
                orbit_tables["light_travel_time_start_time_gps"] = ltt_start_time_gps
                orbit_tables["light_travel_time_sample_interval_s"] = (
                    ltt_sample_interval_s
                )
                orbit_tables["light_travel_times_s"] = _read_only(
                    np.stack(
                        [
                            np.asarray(ltt_group[f"ltt_{link}"][:stop], dtype=float)
                            for link in LINKS
                        ]
                    )
                )
                derivative_names = [f"ltt_derivative_{link}" for link in LINKS]
                if any(name in ltt_group for name in derivative_names):
                    orbit_tables["light_travel_time_derivatives"] = _read_only(
                        np.stack(
                            [
                                np.asarray(ltt_group[name][:stop], dtype=float)
                                for name in derivative_names
                            ]
                        )
                    )
            for group_path, prefix, field_name, names in (
                (
                    "scet_wrt_tcb",
                    "clock",
                    "scet_wrt_tcb_s",
                    tuple(f"scet_wrt_tcb_{sc}" for sc in (1, 2, 3)),
                ),
                (
                    "ttl/ttl_coefficients",
                    "ttl",
                    "ttl_coefficients",
                    TTL_COEFFICIENT_NAMES,
                ),
            ):
                if group_path not in cd1l_file:
                    continue
                group = cd1l_file[group_path]
                sampling = group["sampling"].attrs
                table_t0 = float(sampling["t0"])
                table_dt = float(sampling["dt"])
                if (
                    not np.isfinite(table_t0)
                    or not np.isfinite(table_dt)
                    or table_dt <= 0
                ):
                    raise ValueError(
                        f"{group_path} sampling must be finite with positive dt"
                    )
                last_time = start_time_gps + (num_time_samples - 1) * (
                    1.0 / sample_rate_hz
                )
                stop = min(
                    int(sampling["size"]),
                    max(1, int(np.ceil((last_time - table_t0) / table_dt)) + 2),
                )
                orbit_tables[f"{prefix}_start_time_gps"] = table_t0
                orbit_tables[f"{prefix}_sample_interval_s"] = table_dt
                tables = {
                    name: _read_only(np.asarray(group[name][:stop], dtype=float))
                    for name in names
                }
                orbit_tables[field_name] = (
                    tables
                    if prefix == "ttl"
                    else _read_only(np.stack(list(tables.values())))
                )
        return cls(
            channel_data,
            channel_names=CD1L_CHANNELS,
            sample_rate_hz=sample_rate_hz,
            start_time_gps=start_time_gps,
            tdi_generation="2.0",
            physical_observable="fractional_frequency",
            **orbit_tables,
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
        covariance: Covariance | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> BlockState:
        """Build a `BlockState` and validate its signal against this grid.

        The summed TDI signal must match this object's channels, shape, and
        real/complex representation. None means zero contribution, replacing
        any prior signal. Optional `covariance` replaces the full covariance
        on adoption; its compatibility is checked by Wheel, not this helper.

        Physical parameters follow the `BlockState` contract.
        Supplied arrays and dictionaries are retained without copying; Wheel
        copies the state when adopting it.
        """
        state = BlockState(
            template=template,
            covariance=covariance,
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
        covariance: Covariance | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> BlockState:
        """A `BlockState` with a zero signal on this grid: nothing to subtract.

        Allocates fresh zero arrays matching each channel's shape and dtype.
        Use this when explicit arrays are useful. A source-free or noise-only
        block can instead leave `template=None` without allocating
        any arrays:

            return BlockState(parameters={"sources": []})
            return BlockState(covariance=covariance)
        """
        return self.block_state(
            {ch: np.zeros_like(arr) for ch, arr in self.channel_data.items()},
            covariance=covariance,
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


def _check_raw_table(values: object, label: str, *, ndim: int) -> None:
    """Require a floating-point ndarray of rank ``ndim``, without scanning values."""
    if not isinstance(values, np.ndarray) or not np.issubdtype(
        values.dtype, np.floating
    ):
        raise TypeError(f"{label} must be a floating-point numpy array")
    if values.ndim != ndim:
        raise ValueError(f"{label} must be {ndim}-D, got shape {values.shape}")


def _check_raw_sampling(
    *,
    table_label: str,
    prefix: str,
    num_samples: int | None,
    start_time_gps: float | None,
    sample_interval_s: float | None,
    data_span_gps: tuple[float, float],
) -> None:
    """Pair a raw table with its sampling and require it to span the data."""
    start_label = f"{prefix}_start_time_gps"
    interval_label = f"{prefix}_sample_interval_s"
    if num_samples is None:
        if start_time_gps is not None or sample_interval_s is not None:
            raise ValueError(
                f"{start_label} and {interval_label} describe {table_label}, "
                f"which was not given"
            )
        return
    if start_time_gps is None or sample_interval_s is None:
        raise ValueError(f"{table_label} requires {start_label} and {interval_label}")
    if not np.isfinite(start_time_gps):
        raise ValueError(f"{start_label} must be finite, got {start_time_gps!r}")
    if not np.isfinite(sample_interval_s) or sample_interval_s <= 0:
        raise ValueError(
            f"{interval_label} must be positive and finite, got {sample_interval_s!r}"
        )
    end_time_gps = start_time_gps + (num_samples - 1) * sample_interval_s
    data_start_gps, data_end_gps = data_span_gps
    if start_time_gps > data_start_gps or end_time_gps < data_end_gps:
        raise ValueError(
            f"{table_label} spans [{start_time_gps}, {end_time_gps}] s but the "
            f"data samples span [{data_start_gps}, {data_end_gps}] s "
            f"(mismatched epoch conventions? GPS vs zero-based times?)"
        )


def _read_only(values: np.ndarray) -> np.ndarray:
    """Mark a freshly read array read-only; blocks share it by reference."""
    values.setflags(write=False)
    return values
