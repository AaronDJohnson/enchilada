"""Raw orbit tables on L1Data, and the Mojito CD1-L reader that fills them."""

from dataclasses import replace

import numpy as np
import pytest

from enchilada import L1Data, Wheel, transform
from enchilada.data import CD1L_CHANNELS, LINKS

FS = 0.4
T0 = 1.0e8
N = 64
LASER_FREQUENCY_HZ = 2.816e14
ORBIT_T0 = T0 - 1.0e6
ORBIT_DT = 1.0e5
ORBIT_SIZE = 30


def raw_tables(*, ltt_samples=N + 1, ltt_t0=T0, orbit_t0=ORBIT_T0):
    rng = np.random.default_rng(3)
    return {
        "spacecraft_positions_m": rng.normal(size=(3, ORBIT_SIZE, 3)) * 1e11,
        "spacecraft_velocities_m_per_s": rng.normal(size=(3, ORBIT_SIZE, 3)) * 3e4,
        "orbit_start_time_gps": orbit_t0,
        "orbit_sample_interval_s": ORBIT_DT,
        "light_travel_times_s": 8.3 + 1e-3 * rng.normal(size=(6, ltt_samples)),
        "light_travel_time_start_time_gps": ltt_t0,
        "light_travel_time_sample_interval_s": 1 / FS,
    }


def observed(**tables):
    return L1Data(
        {ch: np.zeros(N) for ch in "XYZ"},
        sample_rate_hz=FS,
        start_time_gps=T0,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
        **tables,
    )


class TestL1DataTables:
    def test_tables_are_optional(self):
        data = observed()
        assert data.spacecraft_positions_m is None
        assert data.light_travel_times_s is None
        assert data.orbit_sample_times_gps is None
        assert data.light_travel_time_sample_times_gps is None

    def test_tables_are_kept_as_given_with_their_sample_times(self):
        tables = raw_tables()
        data = observed(**tables)
        for name, value in tables.items():
            assert getattr(data, name) is value
        np.testing.assert_array_equal(
            data.orbit_sample_times_gps, ORBIT_T0 + np.arange(ORBIT_SIZE) * ORBIT_DT
        )
        np.testing.assert_array_equal(
            data.light_travel_time_sample_times_gps, T0 + np.arange(N + 1) / FS
        )

    def test_replace_transform_and_wheel_share_the_tables(self):
        tables = raw_tables()
        data = observed(**tables)
        trimmed = replace(
            data,
            channel_data={ch: np.zeros(N - 8) for ch in "XYZ"},
            num_time_samples=N - 8,
            start_time_gps=T0 + 4 / FS,
        )
        spectrum = transform(trimmed, "frequency")
        wheel = Wheel(spectrum, random_seed=0)
        for carried in (trimmed, spectrum, wheel.observed_data, wheel.working_residual):
            assert carried.light_travel_times_s is tables["light_travel_times_s"]
            assert carried.spacecraft_positions_m is tables["spacecraft_positions_m"]

    @pytest.mark.parametrize(
        ("tables", "match"),
        [
            ({"ltt_t0": T0 + 1.0}, "light_travel_times_s spans"),
            ({"ltt_samples": N - 1}, "light_travel_times_s spans"),
            ({"orbit_t0": T0 + 1.0}, "spacecraft_positions_m spans"),
            ({"orbit_t0": T0 - 1e8}, "spacecraft_positions_m spans"),
        ],
    )
    def test_each_table_must_span_the_data(self, tables, match):
        with pytest.raises(ValueError, match=match):
            observed(**raw_tables(**tables))

    def test_an_epoch_shift_by_replace_is_caught(self):
        data = observed(**raw_tables())
        with pytest.raises(ValueError, match="mismatched epoch"):
            replace(data, start_time_gps=T0 - 10.0)

    @pytest.mark.parametrize(
        ("drop", "match"),
        [
            ("orbit_start_time_gps", "requires orbit_start_time_gps"),
            ("orbit_sample_interval_s", "requires orbit_start_time_gps"),
            ("light_travel_time_start_time_gps", "requires light_travel_time_start"),
            ("spacecraft_positions_m", "requires spacecraft_positions_m"),
        ],
    )
    def test_tables_and_their_sampling_come_together(self, drop, match):
        tables = raw_tables()
        del tables[drop]
        with pytest.raises(ValueError, match=match):
            observed(**tables)

    def test_sampling_without_a_table_is_rejected(self):
        tables = raw_tables()
        del tables["light_travel_times_s"]
        with pytest.raises(ValueError, match="which was not given"):
            observed(**tables)

    @pytest.mark.parametrize(
        ("name", "value", "match"),
        [
            ("spacecraft_positions_m", np.zeros((ORBIT_SIZE, 3, 3)), r"\(3, num_orbit"),
            ("spacecraft_positions_m", np.zeros((3, ORBIT_SIZE)), "must be 3-D"),
            (
                "spacecraft_velocities_m_per_s",
                np.zeros((3, ORBIT_SIZE - 1, 3)),
                "must match spacecraft_positions_m",
            ),
            ("light_travel_times_s", np.full((5, N + 1), 8.3), "one row per link"),
            ("light_travel_times_s", np.full(N + 1, 8.3), "must be 2-D"),
            ("orbit_sample_interval_s", 0.0, "positive and finite"),
            ("light_travel_time_sample_interval_s", np.nan, "positive and finite"),
            ("orbit_start_time_gps", np.inf, "must be finite"),
        ],
    )
    def test_structure_is_validated(self, name, value, match):
        with pytest.raises(ValueError, match=match):
            observed(**{**raw_tables(), name: value})

    @pytest.mark.parametrize(
        "value", [np.ones((6, N + 1), dtype=int), [[8.3] * (N + 1)] * 6]
    )
    def test_tables_must_be_floating_arrays(self, value):
        with pytest.raises(TypeError, match="floating-point numpy array"):
            observed(**{**raw_tables(), "light_travel_times_s": value})


def write_cd1l(path, *, velocities=True, ltts=True, orbits=True):
    """A miniature CD1-L brick in the Mojito layout."""
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(5)
    contents = {}
    with h5py.File(path, "w") as f:
        f.attrs["laser_frequency"] = LASER_FREQUENCY_HZ
        tdis = f.create_group("tdis")
        sampling = tdis.create_group("sampling").attrs
        sampling["fs"], sampling["dt"], sampling["t0"], sampling["size"] = (
            FS,
            1 / FS,
            T0,
            N,
        )
        for name in ("X2", "Y2", "Z2", "A2"):
            contents[name] = rng.normal(size=N) * 1e-3
            tdis.create_dataset(name, data=contents[name])
        if orbits:
            group = f.create_group("orbits")
            sampling = group.create_group("sampling").attrs
            sampling["t0"], sampling["dt"], sampling["size"] = (
                ORBIT_T0,
                ORBIT_DT,
                ORBIT_SIZE,
            )
            kinds = ("position", "velocity") if velocities else ("position",)
            for kind in kinds:
                for sc in (1, 2, 3):
                    contents[f"sc_{kind}_{sc}"] = rng.normal(size=(ORBIT_SIZE, 3))
                    group.create_dataset(
                        f"sc_{kind}_{sc}", data=contents[f"sc_{kind}_{sc}"]
                    )
        if ltts:
            group = f.create_group("ltts")
            sampling = group.create_group("sampling").attrs
            sampling["t0"], sampling["dt"], sampling["size"] = T0, 1 / FS, N
            for link in LINKS:
                contents[f"ltt_{link}"] = 8.3 + 1e-3 * rng.normal(size=N)
                group.create_dataset(f"ltt_{link}", data=contents[f"ltt_{link}"])
    return contents


class TestFromCD1L:
    def test_reads_fractional_frequency_xyz_tdi2(self, tmp_path):
        path = tmp_path / "brick.h5"
        contents = write_cd1l(path)
        data = L1Data.from_cd1l(str(path))
        assert data.channel_names == CD1L_CHANNELS == ("X", "Y", "Z")
        assert (data.fs, data.t0, data.N) == (FS, T0, N)
        assert data.tdi_generation == "2.0"
        assert data.physical_observable == "fractional_frequency"
        assert data.data_domain == "time"
        assert data.orbit_ephemeris is None
        for ch in "XYZ":
            np.testing.assert_array_equal(
                data.channel_data[ch], contents[f"{ch}2"] / LASER_FREQUENCY_HZ
            )

    def test_orbit_tables_are_copied_raw_and_read_only(self, tmp_path):
        path = tmp_path / "brick.h5"
        contents = write_cd1l(path)
        data = L1Data.from_cd1l(str(path))
        for sc in (1, 2, 3):
            np.testing.assert_array_equal(
                data.spacecraft_positions_m[sc - 1], contents[f"sc_position_{sc}"]
            )
            np.testing.assert_array_equal(
                data.spacecraft_velocities_m_per_s[sc - 1],
                contents[f"sc_velocity_{sc}"],
            )
        for row, link in zip(data.light_travel_times_s, LINKS, strict=True):
            np.testing.assert_array_equal(row, contents[f"ltt_{link}"])
        assert (data.orbit_start_time_gps, data.orbit_sample_interval_s) == (
            ORBIT_T0,
            ORBIT_DT,
        )
        assert data.light_travel_time_start_time_gps == T0
        assert data.light_travel_time_sample_interval_s == 1 / FS
        for table in (
            data.spacecraft_positions_m,
            data.spacecraft_velocities_m_per_s,
            data.light_travel_times_s,
        ):
            assert not table.flags.writeable

    def test_duration_reads_a_prefix_and_the_ltts_covering_it(self, tmp_path):
        path = tmp_path / "brick.h5"
        contents = write_cd1l(path)
        data = L1Data.from_cd1l(str(path), duration_s=20.0)  # ceil(20 * 0.4) = 8
        assert data.N == 8
        np.testing.assert_array_equal(
            data.channel_data["X"], contents["X2"][:8] / LASER_FREQUENCY_HZ
        )
        assert data.light_travel_times_s.shape == (6, 9)
        np.testing.assert_array_equal(
            data.light_travel_times_s[0], contents["ltt_12"][:9]
        )
        assert data.light_travel_time_sample_times_gps[-1] >= T0 + 7 / FS

    @pytest.mark.parametrize("duration_s", [0.0, -5.0, (N + 1) / FS])
    def test_duration_must_select_samples_in_the_brick(self, tmp_path, duration_s):
        path = tmp_path / "brick.h5"
        write_cd1l(path)
        with pytest.raises(ValueError, match="duration_s"):
            L1Data.from_cd1l(str(path), duration_s=duration_s)

    @pytest.mark.parametrize(
        ("kwargs", "absent"),
        [
            ({"velocities": False}, ("spacecraft_velocities_m_per_s",)),
            ({"ltts": False}, ("light_travel_times_s",)),
            (
                {"orbits": False},
                ("spacecraft_positions_m", "spacecraft_velocities_m_per_s"),
            ),
        ],
    )
    def test_missing_groups_leave_fields_none(self, tmp_path, kwargs, absent):
        path = tmp_path / "brick.h5"
        write_cd1l(path, **kwargs)
        data = L1Data.from_cd1l(str(path))
        for name in absent:
            assert getattr(data, name) is None
