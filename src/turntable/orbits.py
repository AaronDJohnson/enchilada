"""LISA constellation ephemerides carried on `Residuals.orbit`.

The orbit is a *fixed property of the dataset* -- the spacecraft positions the
data was produced with -- that every segment must share to build its response.
turntable carries an orbit object opaquely on `Residuals.orbit` (like `noise`)
and never interprets it; this module defines the contract (`Orbit`) and the
concrete forms a dataset can supply.

Two shapes, per the user's data:

* **Analytic** -- a closed-form constellation a segment generates itself (each
  piece's own `AnalyticOrbit` in `global_fit_pieces`); used for self-consistent
  synthetic tests. turntable does not implement one; it only defines the
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
(``frame="equatorial"``).

The heavy dependencies (`scipy`, `h5py`, `lisaorbits`) are imported lazily so
turntable's core stays dependency-free.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

CLIGHT = 299792458.0  # m/s
# J2000 mean obliquity of the ecliptic [rad]; equatorial -> ecliptic is R_x(+eps).
_OBLIQUITY = 0.40909280422232897
_NOMINAL_ARMLENGTH = 2.5e9  # m, fallback if a table is degenerate


@runtime_checkable
class Orbit(Protocol):
    """What a segment relies on from a constellation ephemeris.

    Attributes:
        L: Nominal armlength [m] (a scalar; the TDI delay length).
        fstar: Transfer frequency ``c / (2 pi L)`` [Hz].

    Method:
        positions(t): spacecraft positions at time(s) ``t`` [s, absolute, same
            epoch convention as the data]. Returns ``(x, y, z)``, each of shape
            ``(3, len(t))`` (spacecraft, time), in **ecliptic** metres.

    An implementation must cover the full data span ``[epoch, epoch + Tobs]``
    and should raise (not extrapolate) outside its domain of validity, as
    :class:`NumericOrbit` does. Tabulated implementations should also expose
    ``t_range``; when present, `Residuals` checks it against the data span at
    construction, so an epoch mismatch fails before any sampling starts.
    """

    L: float
    fstar: float

    def positions(
        self, t: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...


def _equatorial_to_ecliptic(pos: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotate position vectors (..., 3) from equatorial/ICRS to ecliptic."""
    eps = _OBLIQUITY
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(eps), np.sin(eps)],
            [0.0, -np.sin(eps), np.cos(eps)],
        ]
    )
    return pos @ rx.T


class NumericOrbit:
    """Tabulated LISA ephemeris, cubic-spline-interpolated in time.

    The numerical-orbit analogue of GLASS's ``interpolate_orbits``: store
    spacecraft positions on a (typically coarse) time grid and interpolate them
    to whatever times a waveform needs. Satisfies :class:`Orbit`.

    Args:
        times: ``(n,)`` sample times [s], ascending, in the data's epoch.
        positions: ``(3, n, 3)`` spacecraft positions [m] -- (spacecraft, time,
            xyz). Already in ecliptic coordinates (use the loaders for other
            frames).
        L: nominal armlength [m]; defaults to the time-mean of the three arms.
        fstar: transfer frequency [Hz]; defaults to ``c / (2 pi L)``.
    """

    def __init__(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        L: float | None = None,
        fstar: float | None = None,
    ) -> None:
        from scipy.interpolate import CubicSpline  # lazy: keep turntable dep-free

        self._t = np.ascontiguousarray(times, dtype=float)
        pos = np.ascontiguousarray(positions, dtype=float)
        if pos.shape[0] != 3 or pos.shape[1] != self._t.size or pos.shape[2] != 3:
            raise ValueError(
                f"positions must be (3, {self._t.size}, 3), got {pos.shape}"
            )
        self._pos = pos
        # one spline per (spacecraft, coordinate); evaluated together per call
        self._spline = CubicSpline(self._t, pos, axis=1)
        if L is None:
            arms = [
                np.linalg.norm(pos[a] - pos[b], axis=-1)
                for a, b in ((0, 1), (0, 2), (1, 2))
            ]
            mean_arm = float(np.mean(arms))
            # A degenerate table (zero/coincident positions) yields a finite
            # zero mean arm, which is just as unusable as a NaN one.
            L = (
                mean_arm
                if np.isfinite(mean_arm) and mean_arm > 0.0
                else _NOMINAL_ARMLENGTH
            )
        self.L = float(L)
        self.fstar = (
            float(fstar) if fstar is not None else CLIGHT / (2.0 * np.pi * self.L)
        )

    @property
    def t_range(self) -> tuple[float, float]:
        """``(t_min, t_max)`` of the tabulated grid [s]."""
        return (float(self._t[0]), float(self._t[-1]))

    def positions(
        self, t: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Interpolated positions at ``t``; ``(x, y, z)`` each ``(3, len(t))``.

        Raises ValueError for times outside the tabulated grid (`t_range`):
        cubic splines extrapolate polynomially and silently drift off the
        real orbit, so out-of-span queries -- typically an epoch mismatch
        between data and ephemeris -- fail loudly instead.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        t_lo, t_hi = self.t_range
        if t.size and (t.min() < t_lo or t.max() > t_hi):
            raise ValueError(
                f"requested times span [{t.min()}, {t.max()}] s but the tabulated "
                f"ephemeris covers [{t_lo}, {t_hi}] s; refusing to extrapolate "
                f"(mismatched epoch conventions? GPS vs zero-based times?)"
            )
        p = self._spline(t)  # (3 sc, len(t), 3 xyz)
        return p[..., 0], p[..., 1], p[..., 2]

    # ---- loaders --------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        times: NDArray[np.float64],
        sc_positions: NDArray[np.float64],
        *,
        frame: str = "ecliptic",
        L: float | None = None,
        fstar: float | None = None,
    ) -> NumericOrbit:
        """Build from in-memory arrays.

        Args:
            times: ``(n,)`` times [s].
            sc_positions: ``(3, n, 3)`` positions [m] (spacecraft, time, xyz).
            frame: ``"ecliptic"`` (default) or ``"equatorial"`` (rotated on load).
        """
        pos = np.asarray(sc_positions, dtype=float)
        if frame == "equatorial":
            pos = _equatorial_to_ecliptic(pos)
        elif frame != "ecliptic":
            raise ValueError(f"frame must be 'ecliptic' or 'equatorial', got {frame!r}")
        return cls(np.asarray(times, float), pos, L=L, fstar=fstar)

    @classmethod
    def from_hdf5(
        cls,
        path: str,
        *,
        group: str = "orbits",
        position_datasets: tuple[str, str, str] = (
            "sc_position_1",
            "sc_position_2",
            "sc_position_3",
        ),
        frame: str = "equatorial",
        L: float | None = None,
    ) -> NumericOrbit:
        """Load a tabulated ephemeris from an HDF5 file.

        Defaults match the LDC/Mojito L1 layout: a ``group`` holding a
        ``sampling`` sub-object with ``t0``/``dt``/``size`` attributes and three
        ``sc_position_i`` datasets of shape ``(n, 3)`` in equatorial/ICRS metres.
        """
        import h5py  # lazy

        with h5py.File(path, "r") as f:
            g = f[group]
            s = g["sampling"]
            t0 = float(s.attrs["t0"])
            dt = float(s.attrs["dt"])
            n = int(s.attrs["size"])
            pos = np.stack([np.asarray(g[d]) for d in position_datasets], axis=0)
        times = np.asarray(t0 + np.arange(n) * dt, dtype=np.float64)
        return cls.from_arrays(times, pos, frame=frame, L=L)

    @classmethod
    def from_lisaorbits(
        cls,
        orbits: object,
        times: NDArray[np.float64],
        *,
        frame: str = "equatorial",
    ) -> NumericOrbit:
        """Sample a **lisaorbits** orbit object onto ``times`` and tabulate it.

        Works for both analytic (e.g. ``KeplerianOrbits``) and numerical
        (``OEMOrbits``/``InterpolatedOrbits``) lisaorbits orbits -- both expose
        ``compute_position(t)`` -- evaluated on ``times`` and wrapped as a
        :class:`NumericOrbit`. For a lisaorbits *file* on disk, prefer
        :meth:`from_hdf5` with the file's dataset names.

        lisaorbits positions are in the BCRS/equatorial frame (validated against
        lisaorbits 3.0.3: the guiding-centre z swings by ``AU sin eps`` over a
        year), so ``frame`` defaults to ``"equatorial"`` and they are rotated to
        ecliptic on load.
        """
        times = np.asarray(times, dtype=float)
        fn = getattr(orbits, "compute_position", None)
        if not callable(fn):
            raise TypeError(
                "expected a lisaorbits orbit with compute_position(t); pass a "
                "(3, n, 3) array to NumericOrbit.from_arrays, or use from_hdf5 for "
                "a lisaorbits file."
            )
        raw = np.asarray(fn(times))  # expected (len(t), 3 sc, 3 xyz)
        n = times.size
        if raw.shape == (n, 3, 3):
            pos = np.moveaxis(raw, 0, 1)  # -> (3 sc, n, 3 xyz)
        elif raw.shape == (n, 9):
            # some versions flatten the (spacecraft, xyz) pair
            pos = np.moveaxis(raw.reshape(n, 3, 3), 0, 1)
        else:
            # never reshape blindly: a wrong shape would silently scramble the
            # spacecraft/time/coordinate axes and produce a plausible-looking
            # but meaningless constellation.
            raise ValueError(
                f"compute_position(times) returned shape {raw.shape}; expected "
                f"({n}, 3, 3) -- (time, spacecraft, xyz) -- or a flattened "
                f"({n}, 9). Pass a (3, {n}, 3) array to NumericOrbit.from_arrays "
                f"instead if your source uses another layout."
            )
        return cls.from_arrays(times, pos, frame=frame)
