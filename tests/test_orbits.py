"""NumericOrbit: interpolation, frames, loaders, and refusal to extrapolate."""

import numpy as np
import pytest

pytest.importorskip("scipy", reason="orbits need the numeric-orbits extra")

from enchilada import NumericOrbit
from enchilada.orbits import _DEFAULT_NOMINAL_ARM_LENGTH_M, _J2000_OBLIQUITY_RAD, Orbit

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
    def test_input_buffers_cannot_change_the_tabulated_ephemeris(self):
        times, positions = circular_table(n=8, span=3.0)
        orbit = NumericOrbit(times, positions)
        query = np.array([1.5])
        expected = tuple(values.copy() for values in orbit.positions(query))

        times += 100.0
        positions[:] = 0.0

        assert orbit.time_range_gps == (0.0, 3.0)
        for actual, original in zip(orbit.positions(query), expected, strict=True):
            np.testing.assert_array_equal(actual, original)
        with pytest.raises(ValueError, match="refusing to extrapolate"):
            orbit.positions(np.array([101.0]))
        assert not np.shares_memory(times, orbit._sample_times_gps)
        assert not np.shares_memory(positions, orbit._spacecraft_positions_m)
        assert not orbit._sample_times_gps.flags.writeable
        assert not orbit._spacecraft_positions_m.flags.writeable

    @pytest.mark.parametrize("shape", [(3,), (3, 4), (3, 4, 3, 3)])
    def test_position_array_requires_exactly_three_axes(self, shape):
        with pytest.raises(ValueError, match="spacecraft_positions_m must be"):
            NumericOrbit(np.arange(4.0), np.zeros(shape))

    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    @pytest.mark.parametrize("field", ["nominal_arm_length_m", "transfer_frequency_hz"])
    def test_physical_scale_overrides_must_be_positive_and_finite(self, field, value):
        with pytest.raises(ValueError, match=field):
            NumericOrbit(*circular_table(n=8), **{field: value})

    def test_satisfies_orbit_protocol(self):
        orb = NumericOrbit(*circular_table())
        assert isinstance(orb, Orbit)
        assert orb.transfer_frequency_hz == pytest.approx(
            299792458.0 / (2 * np.pi * orb.nominal_arm_length_m)
        )

    def test_shape_error_is_descriptive(self):
        t = np.linspace(0.0, 1.0, 10)
        with pytest.raises(
            ValueError, match=r"spacecraft_positions_m must be \(3, 10, 3\)"
        ):
            NumericOrbit(t, np.zeros((10, 3, 3)))

    def test_explicit_arm_length_and_transfer_frequency_are_honored(self):
        t, pos = circular_table(n=20)
        orb = NumericOrbit(
            sample_times_gps=t,
            spacecraft_positions_m=pos,
            nominal_arm_length_m=2.5e9,
            transfer_frequency_hz=0.019,
        )
        assert orb.nominal_arm_length_m == 2.5e9
        assert (
            orb.transfer_frequency_hz == 0.019
        )  # not recomputed from nominal_arm_length_m

    def test_degenerate_table_falls_back_to_nominal_armlength(self):
        t = np.linspace(0.0, 1.0, 10)
        orb = NumericOrbit(t, np.zeros((3, 10, 3)))
        assert orb.nominal_arm_length_m == _DEFAULT_NOMINAL_ARM_LENGTH_M
        assert np.isfinite(orb.transfer_frequency_hz)  # no ZeroDivisionError


class TestInterpolation:
    @pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
    def test_nonfinite_query_times_are_rejected_before_bounds_check(self, invalid):
        orbit = NumericOrbit(*circular_table(n=8, span=3.0))
        with pytest.raises(ValueError, match="times_gps.*finite"):
            orbit.positions(np.array([invalid, 100.0]))

    def test_matches_truth_between_nodes(self):
        t, pos = circular_table()
        orb = NumericOrbit(t, pos)
        query = np.array([12345.0, 1.7e6, 2.5e6])
        x, y, z = orb.positions(times_gps=query)
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
        eps = _J2000_OBLIQUITY_RAD
        orb = NumericOrbit.from_arrays(
            sample_times_gps=t,
            spacecraft_positions_m=pos,
            coordinate_frame="equatorial",
        )
        x, y, z = orb.positions(t[:5])
        # hand-rotate spacecraft 0: R_x(+eps) applied to (x, y, z)_equatorial
        exp_y = pos[0, :5, 1] * np.cos(eps) + pos[0, :5, 2] * np.sin(eps)
        exp_z = -pos[0, :5, 1] * np.sin(eps) + pos[0, :5, 2] * np.cos(eps)
        np.testing.assert_allclose(x[0], pos[0, :5, 0], atol=1e-3)
        np.testing.assert_allclose(y[0], exp_y, atol=1e-3)
        np.testing.assert_allclose(z[0], exp_z, atol=1e-3)

    def test_unknown_frame_rejected(self):
        t, pos = circular_table()
        with pytest.raises(ValueError, match="coordinate_frame"):
            NumericOrbit.from_arrays(t, pos, coordinate_frame="galactic")


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
        orb = NumericOrbit.from_hdf5(file_path=str(path), coordinate_frame="ecliptic")
        assert orb.time_range_gps == (t[0], t[-1])
        x, y, z = orb.positions(t[:3])
        np.testing.assert_allclose(x[0], pos[0, :3, 0], atol=1e-3)

    def test_from_lisaorbits_regression(self):
        """Pin the validation from commit 955a9e0: KeplerianOrbits round-trips
        through the tabulation/spline to well under a metre."""
        lisaorbits = pytest.importorskip("lisaorbits")
        t = np.linspace(0.0, 10 * 86400.0, 100)
        source = lisaorbits.KeplerianOrbits()
        orb = NumericOrbit.from_lisaorbits(lisaorbits_model=source, sample_times_gps=t)
        assert isinstance(orb, Orbit)
        assert orb.nominal_arm_length_m == pytest.approx(2.5e9, rel=0.01)
        # spline vs direct evaluation, rotated to ecliptic, off the nodes
        query = t[:-1] + np.diff(t) / 2.0
        x, y, z = orb.positions(times_gps=query)
        raw = np.asarray(source.compute_position(query))  # (t, sc, xyz)
        eps = _J2000_OBLIQUITY_RAD
        exp_x = raw[:, 0, 0]
        exp_y = raw[:, 0, 1] * np.cos(eps) + raw[:, 0, 2] * np.sin(eps)
        exp_z = -raw[:, 0, 1] * np.sin(eps) + raw[:, 0, 2] * np.cos(eps)
        err = np.max(np.abs(np.stack([x[0] - exp_x, y[0] - exp_y, z[0] - exp_z])))
        assert err < 1.0  # metres; measured ~2e-2 m against lisaorbits 3.0.3


class TestFromLisaorbitsGuard:
    """from_lisaorbits must never reshape a wrong-shaped array blindly."""

    class Stub:
        def __init__(self, out):
            self._out = out

        def compute_position(self, t):
            return self._out

    def test_rejects_object_without_compute_position(self):
        with pytest.raises(TypeError, match="compute_position"):
            NumericOrbit.from_lisaorbits(object(), np.linspace(0.0, 10.0, 5))

    def test_accepts_time_spacecraft_xyz(self):
        t = np.linspace(0.0, 10.0, 5)
        raw = np.zeros((t.size, 3, 3))
        raw[:, :, 0] = 1.0e9  # x
        orb = NumericOrbit.from_lisaorbits(
            self.Stub(raw), t, coordinate_frame="ecliptic"
        )
        x, _, _ = orb.positions(t)
        assert x.shape == (3, t.size)

    def test_accepts_flattened_nine_columns(self):
        t = np.linspace(0.0, 10.0, 5)
        raw = np.zeros((t.size, 3, 3))
        raw[:, :, 0] = 1.0e9
        flat = raw.reshape(t.size, 9)
        orb = NumericOrbit.from_lisaorbits(
            self.Stub(flat), t, coordinate_frame="ecliptic"
        )
        x, _, _ = orb.positions(t)
        np.testing.assert_allclose(x, 1.0e9)

    @pytest.mark.parametrize(
        "bad",
        [
            (5, 3, 4),  # wrong trailing axis
            (3, 5, 3),  # already transposed -- must NOT be silently accepted
            (5, 8),  # wrong flattened width
            (15, 3),  # 2-D of the right size but wrong layout
        ],
    )
    def test_rejects_any_other_shape(self, bad):
        t = np.linspace(0.0, 10.0, 5)
        with pytest.raises(ValueError, match="compute_position"):
            NumericOrbit.from_lisaorbits(self.Stub(np.zeros(bad)), t)


def test_from_hdf5_preserves_a_gps_scale_epoch(tmp_path):
    """t0 must survive the load, not be silently zeroed or offset."""
    h5py = pytest.importorskip("h5py")
    t0, dt, n = 1.2345e9, 15.0, 64  # GPS-scale start
    _, pos = circular_table(n=n)
    path = tmp_path / "orbit_gps.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("orbits")
        s = g.create_group("sampling")
        s.attrs["t0"], s.attrs["dt"], s.attrs["size"] = t0, dt, n
        for i in range(3):
            g.create_dataset(f"sc_position_{i + 1}", data=pos[i])
    orb = NumericOrbit.from_hdf5(file_path=str(path), coordinate_frame="ecliptic")
    assert orb.time_range_gps[0] == pytest.approx(t0)
    assert orb.time_range_gps[1] == pytest.approx(t0 + (n - 1) * dt)
