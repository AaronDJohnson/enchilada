"""LISA constellation ephemerides carried on `L1Data.orbit_ephemeris`.

The orbit is a *fixed property of the dataset* -- the spacecraft positions the
data was produced with -- that every block must share to build its response.
enchilada carries an orbit object opaquely on `L1Data.orbit_ephemeris`
and never interprets it; this module defines the contract (`Orbit`) and the
concrete forms a dataset can supply.

Two shapes, per the user's data:

* **Analytic** -- a closed-form constellation a block generates itself (each
  piece's own `AnalyticOrbit` in `global_fit_pieces`); used for self-consistent
  synthetic tests. enchilada does not implement one; it only defines the
  protocol such an object satisfies.
* **Numerical** -- :class:`NumericOrbit`, a tabulated ephemeris (spacecraft
  positions on a coarse time grid) cubic-spline-interpolated to any requested
  time. This is what real / LDC / Mojito data ships, and the analogue of
  GLASS's ``interpolate_orbits`` (numerical orbit files + GSL cubic splines).

Both *analytic* and *numerical* ephemerides produced by the **lisaorbits**
package are ingested through :meth:`NumericOrbit.from_lisaorbits` (an analytic
lisaorbits orbit is sampled onto a grid; a numerical one is read from its file),
so a single tabulated type covers every dataset.

Frame: the response works in **ecliptic** Cartesian metres. Ephemerides given in
equatorial / ICRS (e.g. Mojito spacecraft positions) are rotated on load
(``coordinate_frame="equatorial"``).

The heavy dependencies (`scipy`, `h5py`, `lisaorbits`) are imported lazily so
enchilada's core stays dependency-free.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

SPEED_OF_LIGHT_M_PER_S = 299792458.0  # m/s
# J2000 mean obliquity of the ecliptic, used for the equatorial-to-ecliptic rotation.
_J2000_OBLIQUITY_RAD = 0.40909280422232897
_DEFAULT_NOMINAL_ARM_LENGTH_M = 2.5e9  # m, fallback if a table is degenerate


@runtime_checkable
class Orbit(Protocol):
    """What a block relies on from a constellation ephemeris.

    Attributes:
        nominal_arm_length_m: Nominal arm length in metres, used for TDI delays.
        transfer_frequency_hz: Transfer frequency in Hz, conventionally
            ``c / (2 pi nominal_arm_length_m)``.

    Method:
        positions(times_gps): spacecraft positions at absolute GPS times on the
            same clock as the data. Returns ``(x, y, z)`` in ecliptic metres,
            each shaped ``(3, num_query_times)`` (spacecraft, time).

    An implementation must cover every sample time, ``start_time_gps + n*dt`` for n in
    ``[0, num_time_samples-1]`` -- up to ``start_time_gps + (num_time_samples-1)*dt``,
    one sample interval short of ``start_time_gps + Tobs`` -- and in practice
    should carry margin beyond that, since a block applying TDI light-travel
    delays evaluates retarded times slightly outside the sample span. It should
    raise rather than extrapolate outside its domain of validity, as
    :class:`NumericOrbit` does. Tabulated implementations should also expose
    ``time_range_gps``. When present, `L1Data` checks it covers the sample span, so
    an epoch mismatch fails before any sampling starts.
    """

    nominal_arm_length_m: float
    transfer_frequency_hz: float

    def positions(
        self, times_gps: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Spacecraft positions at `times_gps`, in the ecliptic frame.

        Args:
            times_gps: GPS times in seconds on the same clock as
                `L1Data.start_time_gps`. These are absolute mission times,
                not offsets from the data's start. Any shape; implementations
                broadcast over it.

        Returns:
            `(x, y, z)` -- one array per *coordinate*, not per spacecraft --
            each of shape `(3, num_query_times)` in metres, indexed
            `[spacecraft, time]`. Ecliptic frame, so an implementation
            holding equatorial/ICRS tables must rotate before returning
            (see `NumericOrbit.from_arrays(coordinate_frame="equatorial")`).

        Implementations must raise rather than extrapolate outside their
        domain of validity: a silently wrong constellation shifts every
        block's response and is near-impossible to diagnose downstream.
        """
        ...


def _equatorial_to_ecliptic(
    spacecraft_positions_m: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rotate position vectors (..., 3) from equatorial/ICRS to ecliptic."""
    obliquity_rad = _J2000_OBLIQUITY_RAD
    rotation_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(obliquity_rad), np.sin(obliquity_rad)],
            [0.0, -np.sin(obliquity_rad), np.cos(obliquity_rad)],
        ]
    )
    return spacecraft_positions_m @ rotation_matrix.T


class NumericOrbit:
    """Tabulated LISA ephemeris, cubic-spline-interpolated in time.

    The numerical-orbit analogue of GLASS's ``interpolate_orbits``: store
    spacecraft positions on a (typically coarse) time grid and interpolate them
    to whatever times a waveform needs. Satisfies :class:`Orbit`.
    Tabulated arrays are copied into read-only storage, so callers may reuse
    their input buffers without changing the ephemeris.

    Args:
        sample_times_gps: Ascending sample times in GPS seconds, shaped
            ``(num_time_samples,)`` and on the same clock as the data.
        spacecraft_positions_m: Positions in ecliptic metres, shaped
            ``(3, num_time_samples, 3)`` (spacecraft, time, xyz). Use the loaders
            for other coordinate frames.
        nominal_arm_length_m: Nominal arm length in metres; defaults to the
            mean of the three arms over the tabulated times.
        transfer_frequency_hz: Transfer frequency in Hz; defaults to
            ``c / (2 pi nominal_arm_length_m)``.
    """

    def __init__(
        self,
        sample_times_gps: NDArray[np.float64],
        spacecraft_positions_m: NDArray[np.float64],
        *,
        nominal_arm_length_m: float | None = None,
        transfer_frequency_hz: float | None = None,
    ) -> None:
        from scipy.interpolate import CubicSpline  # lazy: keep enchilada dep-free

        self._sample_times_gps = np.array(
            sample_times_gps, dtype=float, order="C", copy=True
        )
        if (
            self._sample_times_gps.ndim != 1
            or self._sample_times_gps.size < 2
            or not np.isfinite(self._sample_times_gps).all()
            or np.any(np.diff(self._sample_times_gps) <= 0)
        ):
            raise ValueError(
                "sample_times_gps must be a finite, strictly increasing 1-D array "
                "with at least two times"
            )
        spacecraft_positions_m = np.array(
            spacecraft_positions_m, dtype=float, order="C", copy=True
        )
        if spacecraft_positions_m.shape != (
            3,
            self._sample_times_gps.size,
            3,
        ):
            raise ValueError(
                f"spacecraft_positions_m must be "
                f"(3, {self._sample_times_gps.size}, 3), "
                f"got {spacecraft_positions_m.shape}"
            )
        if not np.isfinite(spacecraft_positions_m).all():
            raise ValueError("spacecraft_positions_m must contain finite positions")
        self._spacecraft_positions_m = spacecraft_positions_m
        # one spline per (spacecraft, coordinate); evaluated together per call
        self._position_spline = CubicSpline(
            self._sample_times_gps, spacecraft_positions_m, axis=1
        )
        if nominal_arm_length_m is None:
            arm_lengths_m = [
                np.linalg.norm(
                    spacecraft_positions_m[first_spacecraft]
                    - spacecraft_positions_m[second_spacecraft],
                    axis=-1,
                )
                for first_spacecraft, second_spacecraft in ((0, 1), (0, 2), (1, 2))
            ]
            mean_arm_length_m = float(np.mean(arm_lengths_m))
            # A degenerate table (zero/coincident positions) yields a finite
            # zero mean arm, which is just as unusable as a NaN one.
            nominal_arm_length_m = (
                mean_arm_length_m
                if np.isfinite(mean_arm_length_m) and mean_arm_length_m > 0.0
                else _DEFAULT_NOMINAL_ARM_LENGTH_M
            )
        self.nominal_arm_length_m = float(nominal_arm_length_m)
        if not np.isfinite(self.nominal_arm_length_m) or self.nominal_arm_length_m <= 0:
            raise ValueError("nominal_arm_length_m must be positive and finite")
        self.transfer_frequency_hz = (
            float(transfer_frequency_hz)
            if transfer_frequency_hz is not None
            else SPEED_OF_LIGHT_M_PER_S / (2.0 * np.pi * self.nominal_arm_length_m)
        )
        if (
            not np.isfinite(self.transfer_frequency_hz)
            or self.transfer_frequency_hz <= 0
        ):
            raise ValueError("transfer_frequency_hz must be positive and finite")
        self._sample_times_gps.setflags(write=False)
        self._spacecraft_positions_m.setflags(write=False)

    @property
    def time_range_gps(self) -> tuple[float, float]:
        """Inclusive start and end of the tabulated grid in GPS seconds."""
        return (float(self._sample_times_gps[0]), float(self._sample_times_gps[-1]))

    def positions(
        self, times_gps: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Interpolate at ``times_gps``, returning ``(x, y, z)`` in metres.

        Each coordinate array has shape ``(3, num_query_times)``.

        Raises ValueError for nonfinite times or times outside the tabulated
        grid (`time_range_gps`):
        cubic splines extrapolate polynomially and silently drift off the
        real orbit, so out-of-span queries -- typically an epoch mismatch
        between data and ephemeris -- fail loudly instead.
        """
        times_gps = np.atleast_1d(np.asarray(times_gps, dtype=float))
        if not np.isfinite(times_gps).all():
            raise ValueError("times_gps must contain only finite query times")
        min_time_gps, max_time_gps = self.time_range_gps
        if times_gps.size and (
            times_gps.min() < min_time_gps or times_gps.max() > max_time_gps
        ):
            raise ValueError(
                f"requested times span [{times_gps.min()}, {times_gps.max()}] s "
                f"but the tabulated ephemeris covers "
                f"[{min_time_gps}, {max_time_gps}] s; "
                f"refusing to extrapolate "
                f"(mismatched epoch conventions? GPS vs zero-based times?)"
            )
        # Shape: (spacecraft, query time, xyz coordinate).
        interpolated_positions_m = self._position_spline(times_gps)
        return (
            interpolated_positions_m[..., 0],
            interpolated_positions_m[..., 1],
            interpolated_positions_m[..., 2],
        )

    # ---- loaders --------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        sample_times_gps: NDArray[np.float64],
        spacecraft_positions_m: NDArray[np.float64],
        *,
        coordinate_frame: str = "ecliptic",
        nominal_arm_length_m: float | None = None,
        transfer_frequency_hz: float | None = None,
    ) -> NumericOrbit:
        """Build from in-memory arrays.

        Args:
            sample_times_gps: Ascending GPS times, shaped ``(num_time_samples,)``.
            spacecraft_positions_m: Positions in metres, shaped
                ``(3, num_time_samples, 3)`` (spacecraft, time, xyz).
            coordinate_frame: "ecliptic" (default) or "equatorial";
                equatorial positions are rotated to ecliptic on load.
            nominal_arm_length_m: Optional override for the mean arm length.
            transfer_frequency_hz: Optional override for the transfer frequency.
        """
        spacecraft_positions_m = np.asarray(spacecraft_positions_m, dtype=float)
        if coordinate_frame == "equatorial":
            spacecraft_positions_m = _equatorial_to_ecliptic(spacecraft_positions_m)
        elif coordinate_frame != "ecliptic":
            raise ValueError(
                f"coordinate_frame must be 'ecliptic' or 'equatorial', "
                f"got {coordinate_frame!r}"
            )
        return cls(
            np.asarray(sample_times_gps, float),
            spacecraft_positions_m,
            nominal_arm_length_m=nominal_arm_length_m,
            transfer_frequency_hz=transfer_frequency_hz,
        )

    @classmethod
    def from_hdf5(
        cls,
        file_path: str,
        *,
        group_path: str = "orbits",
        position_dataset_names: tuple[str, str, str] = (
            "sc_position_1",
            "sc_position_2",
            "sc_position_3",
        ),
        coordinate_frame: str = "equatorial",
        nominal_arm_length_m: float | None = None,
    ) -> NumericOrbit:
        """Load a tabulated ephemeris from an HDF5 file.

        Defaults match the LDC/Mojito L1 layout: the group at ``group_path`` holds a
        ``sampling`` sub-object with ``t0``/``dt``/``size`` attributes and three
        ``sc_position_i`` datasets of shape ``(n, 3)`` in equatorial/ICRS metres.

        Args:
            file_path: Path to the HDF5 ephemeris file.
            group_path: Group containing sampling metadata and positions.
            position_dataset_names: Dataset names for spacecraft 1, 2, and 3,
                relative to group_path, in that order.
            coordinate_frame: Frame of the stored coordinates; "equatorial"
                by default, or "ecliptic" to skip rotation.
            nominal_arm_length_m: Optional override for the mean arm length.
        """
        import h5py  # lazy

        with h5py.File(file_path, "r") as orbit_file:
            orbit_group = orbit_file[group_path]
            sampling_metadata = orbit_group["sampling"]
            start_time_gps = float(sampling_metadata.attrs["t0"])
            sample_interval_s = float(sampling_metadata.attrs["dt"])
            num_time_samples = int(sampling_metadata.attrs["size"])
            spacecraft_positions_m = np.stack(
                [
                    np.asarray(orbit_group[dataset_name])
                    for dataset_name in position_dataset_names
                ],
                axis=0,
            )
        sample_times_gps = np.asarray(
            start_time_gps + np.arange(num_time_samples) * sample_interval_s,
            dtype=np.float64,
        )
        return cls.from_arrays(
            sample_times_gps,
            spacecraft_positions_m,
            coordinate_frame=coordinate_frame,
            nominal_arm_length_m=nominal_arm_length_m,
        )

    @classmethod
    def from_lisaorbits(
        cls,
        lisaorbits_model: object,
        sample_times_gps: NDArray[np.float64],
        *,
        coordinate_frame: str = "equatorial",
    ) -> NumericOrbit:
        """Tabulate a **lisaorbits** model at ``sample_times_gps``.

        Works for both analytic (e.g. ``KeplerianOrbits``) and numerical
        (``OEMOrbits``/``InterpolatedOrbits``) lisaorbits orbits -- both expose
        ``compute_position(t)`` -- evaluated on ``sample_times_gps`` and wrapped as a
        :class:`NumericOrbit`. For a lisaorbits *file* on disk, prefer
        :meth:`from_hdf5` with the file's dataset names.

        lisaorbits positions are in the BCRS/equatorial frame (validated against
        lisaorbits 3.0.3: the guiding-centre z swings by ``AU sin eps`` over a
        year), so ``coordinate_frame`` defaults to ``"equatorial"`` and positions
        are rotated to ecliptic on load.

        Args:
            lisaorbits_model: Orbit object exposing compute_position(t).
            sample_times_gps: GPS times to pass to the model, shaped
                ``(num_time_samples,)``. The model must use the same clock.
            coordinate_frame: Frame returned by the model, either "equatorial"
                (default) or "ecliptic".
        """
        sample_times_gps = np.asarray(sample_times_gps, dtype=float)
        compute_position = getattr(lisaorbits_model, "compute_position", None)
        if not callable(compute_position):
            raise TypeError(
                "expected a lisaorbits orbit with compute_position(t); pass a "
                "(3, n, 3) array to NumericOrbit.from_arrays, or use from_hdf5 for "
                "a lisaorbits file."
            )
        # The external API orders its axes as (time, spacecraft, xyz).
        sampled_positions_m = np.asarray(compute_position(sample_times_gps))
        num_time_samples = sample_times_gps.size
        if sampled_positions_m.shape == (num_time_samples, 3, 3):
            spacecraft_positions_m = np.moveaxis(sampled_positions_m, 0, 1)
        elif sampled_positions_m.shape == (num_time_samples, 9):
            # some versions flatten the (spacecraft, xyz) pair
            spacecraft_positions_m = np.moveaxis(
                sampled_positions_m.reshape(num_time_samples, 3, 3), 0, 1
            )
        else:
            # never reshape blindly: a wrong shape would silently scramble the
            # spacecraft/time/coordinate axes and produce a plausible-looking
            # but meaningless constellation.
            raise ValueError(
                f"compute_position(times) returned shape {sampled_positions_m.shape}; "
                f"expected ({num_time_samples}, 3, 3) -- (time, spacecraft, xyz) -- "
                f"or a flattened ({num_time_samples}, 9). "
                f"Pass a (3, {num_time_samples}, 3) array to NumericOrbit.from_arrays "
                f"instead if your source uses another layout."
            )
        return cls.from_arrays(
            sample_times_gps, spacecraft_positions_m, coordinate_frame=coordinate_frame
        )
