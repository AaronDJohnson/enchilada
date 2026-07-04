"""NumericOrbit: interpolation, frames, loaders, and refusal to extrapolate."""

import numpy as np
import pytest

pytest.importorskip("scipy", reason="orbits need the numeric-orbits extra")

from turntable import NumericOrbit
from turntable.orbits import _NOMINAL_ARMLENGTH, _OBLIQUITY, Orbit

AU = 1.495978707e11
YEAR = 365.25 * 86400.0


def circular_table(n=200, span=30 * 86400.0, spread=0.0167):
    t = np.linspace(0.0, span, n)
    pos = np.zeros((3, n, 3))
    for sc in range(3):
        ang = 2 * np.pi * t / YEAR + sc * spread
        pos[sc, :, 0] = AU * np.cos(ang)
        pos[sc, :, 1] = AU * np.sin(ang)
    return t, pos


class TestConstruction:
    def test_satisfies_orbit_protocol(self):
        orb = NumericOrbit(*circular_table())
        assert isinstance(orb, Orbit)
        assert orb.fstar == pytest.approx(299792458.0 / (2 * np.pi * orb.L))

    def test_shape_error_is_descriptive(self):
        t = np.linspace(0.0, 1.0, 10)
        with pytest.raises(ValueError, match=r"positions must be \(3, 10, 3\)"):
            NumericOrbit(t, np.zeros((10, 3, 3)))

    def test_degenerate_table_falls_back_to_nominal_armlength(self):
        t = np.linspace(0.0, 1.0, 10)
        orb = NumericOrbit(t, np.zeros((3, 10, 3)))
        assert orb.L == _NOMINAL_ARMLENGTH
        assert np.isfinite(orb.fstar)  # no ZeroDivisionError


class TestInterpolation:
    def test_matches_truth_between_nodes(self):
        t, pos = circular_table()
        orb = NumericOrbit(t, pos)
        query = np.array([12345.0, 1.7e6, 2.5e6])
        x, y, z = orb.positions(query)
        ang = 2 * np.pi * query / YEAR  # spacecraft 0
        np.testing.assert_allclose(x[0], AU * np.cos(ang), atol=1.0)
        np.testing.assert_allclose(y[0], AU * np.sin(ang), atol=1.0)

    def test_refuses_to_extrapolate(self):
        orb = NumericOrbit(*circular_table(span=30 * 86400.0))
        with pytest.raises(ValueError, match="refusing to extrapolate"):
            orb.positions(np.array([90 * 86400.0]))
        with pytest.raises(ValueError, match="refusing to extrapolate"):
            orb.positions(np.array([-1.0]))

    def test_endpoints_are_in_span(self):
        t, pos = circular_table()
        orb = NumericOrbit(t, pos)
        orb.positions(np.array([t[0], t[-1]]))


class TestFrames:
    def test_equatorial_rotation_matches_hand_rotation(self):
        t, pos = circular_table()
        eps = _OBLIQUITY
        orb = NumericOrbit.from_arrays(t, pos, frame="equatorial")
        x, y, z = orb.positions(t[:5])
        # hand-rotate spacecraft 0: R_x(+eps) applied to (x, y, z)_equatorial
        exp_y = pos[0, :5, 1] * np.cos(eps) + pos[0, :5, 2] * np.sin(eps)
        exp_z = -pos[0, :5, 1] * np.sin(eps) + pos[0, :5, 2] * np.cos(eps)
        np.testing.assert_allclose(x[0], pos[0, :5, 0], atol=1e-3)
        np.testing.assert_allclose(y[0], exp_y, atol=1e-3)
        np.testing.assert_allclose(z[0], exp_z, atol=1e-3)

    def test_unknown_frame_rejected(self):
        t, pos = circular_table()
        with pytest.raises(ValueError, match="frame"):
            NumericOrbit.from_arrays(t, pos, frame="galactic")


class TestLoaders:
    def test_from_hdf5_round_trip(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        t, pos = circular_table(n=50)
        dt = t[1] - t[0]
        path = tmp_path / "orbit.h5"
        with h5py.File(path, "w") as f:
            g = f.create_group("orbits")
            s = g.create_group("sampling")
            s.attrs["t0"], s.attrs["dt"], s.attrs["size"] = t[0], dt, t.size
            for i in range(3):
                g.create_dataset(f"sc_position_{i + 1}", data=pos[i])
        orb = NumericOrbit.from_hdf5(str(path), frame="ecliptic")
        assert orb.t_range == (t[0], t[-1])
        x, y, z = orb.positions(t[:3])
        np.testing.assert_allclose(x[0], pos[0, :3, 0], atol=1e-3)

    def test_from_lisaorbits_regression(self):
        """Pin the validation from commit 955a9e0: KeplerianOrbits round-trips
        through the tabulation/spline to well under a metre."""
        lisaorbits = pytest.importorskip("lisaorbits")
        t = np.linspace(0.0, 10 * 86400.0, 100)
        source = lisaorbits.KeplerianOrbits()
        orb = NumericOrbit.from_lisaorbits(source, t)
        assert isinstance(orb, Orbit)
        assert orb.L == pytest.approx(2.5e9, rel=0.01)
        # spline vs direct evaluation, rotated to ecliptic, off the nodes
        query = t[:-1] + np.diff(t) / 2.0
        x, y, z = orb.positions(query)
        raw = np.asarray(source.compute_position(query))  # (t, sc, xyz)
        eps = _OBLIQUITY
        exp_x = raw[:, 0, 0]
        exp_y = raw[:, 0, 1] * np.cos(eps) + raw[:, 0, 2] * np.sin(eps)
        exp_z = -raw[:, 0, 1] * np.sin(eps) + raw[:, 0, 2] * np.cos(eps)
        err = np.max(np.abs(np.stack([x[0] - exp_x, y[0] - exp_y, z[0] - exp_z])))
        assert err < 1.0  # metres; measured ~2e-2 m against lisaorbits 3.0.3
