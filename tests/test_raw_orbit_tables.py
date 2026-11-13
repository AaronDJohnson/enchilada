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
    def test_orbit_objects_are_not_accepted(self):
        with pytest.raises(TypeError, match="orbit_ephemeris"):
            observed(orbit_ephemeris=object())

    @pytest.mark.parametrize("sample_rate_hz", [3.0, 6.0, 7.0, 0.5, 0.1])
    def test_raw_tables_cover_exact_data_grid(self, sample_rate_hz):
        positions = np.zeros((3, 32, 3))
        data = L1Data(
            {"X": np.zeros(32)},
            sample_rate_hz=sample_rate_hz,
            tdi_generation="2.0",
            physical_observable="fractional_frequency",
            spacecraft_positions_m=positions,
            orbit_start_time_gps=0.0,
            orbit_sample_interval_s=1 / sample_rate_hz,
        )
        assert data.spacecraft_positions_m is positions
        with pytest.raises(ValueError, match="spans"):
            replace(data, spacecraft_positions_m=positions[:, :-1, :])

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


def auxiliary_tables():
    from enchilada.data import TTL_COEFFICIENT_NAMES

    return {
        "light_travel_time_derivatives": np.zeros((6, N + 1)),
        "scet_wrt_tcb_s": np.zeros((3, N + 1)),
        "clock_start_time_gps": T0,
        "clock_sample_interval_s": 1 / FS,
        "ttl_coefficients": {name: np.zeros(N + 1) for name in TTL_COEFFICIENT_NAMES},
        "ttl_start_time_gps": T0,
        "ttl_sample_interval_s": 1 / FS,
    }


def test_auxiliary_tables_shared_through_transforms_and_wheel():
    tables = auxiliary_tables()
    data = observed(**raw_tables(), **tables)
    wheel = Wheel(data, random_seed=0)
    for carried in (
        replace(data),
        transform(data, "frequency"),
        wheel.working_residual,
    ):
        for name, value in tables.items():
            assert getattr(carried, name) is value
    for times in (data.clock_sample_times_gps, data.ttl_sample_times_gps):
        np.testing.assert_array_equal(times, T0 + np.arange(N + 1) / FS)
    empty = observed()
    assert empty.clock_sample_times_gps is None
    assert empty.ttl_sample_times_gps is None


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("light_travel_time_derivatives", np.zeros((6, N)), "must match"),
        ("light_travel_time_derivatives", np.zeros((6, N + 1), dtype=int), "floating"),
        ("scet_wrt_tcb_s", np.zeros((2, N + 1)), "three spacecraft"),
        ("scet_wrt_tcb_s", np.zeros((3, N - 1)), "spans"),
        ("clock_start_time_gps", None, "requires"),
        ("clock_sample_interval_s", 0.0, "positive"),
        ("ttl_coefficients", {}, "all 24"),
        ("ttl_coefficients", [], "all 24"),
        ("ttl_start_time_gps", T0 + 1, "spans"),
        ("ttl_sample_interval_s", None, "requires"),
    ],
)
def test_auxiliary_table_validation(field, value, match):
    tables = {**raw_tables(), **auxiliary_tables(), field: value}
    with pytest.raises((ValueError, TypeError), match=match):
        observed(**tables)


def test_derivatives_require_ltts_and_ttl_lengths_match():
    with pytest.raises(ValueError, match="requires light_travel_times_s"):
        observed(light_travel_time_derivatives=np.zeros((6, N)))
    tables = auxiliary_tables()
    tables["ttl_coefficients"]["c_12_eta_rx"] = np.zeros(N)
    with pytest.raises(ValueError, match="matching sample counts"):
        observed(**raw_tables(), **tables)


def add_auxiliary_groups(path):
    h5py = pytest.importorskip("h5py")

    from enchilada.data import TTL_COEFFICIENT_NAMES

    contents = {}
    with h5py.File(path, "a") as f:
        for link in LINKS:
            name = f"ltts/ltt_derivative_{link}"
            contents[name] = np.arange(N, dtype=float) * link * 1e-10
            f.create_dataset(name, data=contents[name])
        for path, names, dt, count in (
            ("scet_wrt_tcb", [f"scet_wrt_tcb_{sc}" for sc in (1, 2, 3)], 10.0, 18),
            ("ttl/ttl_coefficients", TTL_COEFFICIENT_NAMES, 0.25, 660),
        ):
            group = f.create_group(path)
            group.create_group("sampling").attrs.update(t0=T0 - 5, dt=dt, size=count)
            for i, name in enumerate(names):
                values = np.arange(count, dtype=float) + i
                contents[f"{path}/{name}"] = values
                group.create_dataset(name, data=values)
    return contents


@pytest.mark.parametrize("duration_s", [None, 20.0])
def test_reads_auxiliary_tables_with_independent_sampling(tmp_path, duration_s):
    path = tmp_path / "brick.h5"
    write_cd1l(path)
    contents = add_auxiliary_groups(path)
    data = L1Data.from_cd1l(str(path), duration_s=duration_s)
    last = T0 + (data.N - 1) / FS
    assert data.light_travel_time_derivatives.shape == data.light_travel_times_s.shape
    for row, link in zip(data.light_travel_time_derivatives, LINKS, strict=True):
        np.testing.assert_array_equal(
            row, contents[f"ltts/ltt_derivative_{link}"][: row.size]
        )
    assert data.clock_start_time_gps == data.ttl_start_time_gps == T0 - 5
    assert data.clock_sample_interval_s == 10.0
    assert data.ttl_sample_interval_s == 0.25
    assert data.clock_sample_times_gps[-1] >= last
    assert data.ttl_sample_times_gps[-1] >= last
    assert (
        data.ttl_coefficients["c_12_eta_rx"].size
        == int(np.ceil((last - (T0 - 5)) / 0.25)) + 2
    )
    for sc, row in enumerate(data.scet_wrt_tcb_s, 1):
        np.testing.assert_array_equal(
            row, contents[f"scet_wrt_tcb/scet_wrt_tcb_{sc}"][: row.size]
        )
    for name, values in data.ttl_coefficients.items():
        np.testing.assert_array_equal(
            values, contents[f"ttl/ttl_coefficients/{name}"][: values.size]
        )
        assert not values.flags.writeable
    assert not data.scet_wrt_tcb_s.flags.writeable
    assert not data.light_travel_time_derivatives.flags.writeable


def test_missing_auxiliary_groups_are_optional(tmp_path):
    path = tmp_path / "brick.h5"
    write_cd1l(path)
    data = L1Data.from_cd1l(str(path))
    for field in auxiliary_tables():
        assert getattr(data, field) is None


@pytest.mark.parametrize(
    "missing",
    [
        "ltts/ltt_derivative_12",
        "scet_wrt_tcb/scet_wrt_tcb_2",
        "ttl/ttl_coefficients/c_12_eta_rx",
    ],
)
def test_incomplete_auxiliary_groups_fail(tmp_path, missing):
    h5py = pytest.importorskip("h5py")

    path = tmp_path / "brick.h5"
    write_cd1l(path)
    add_auxiliary_groups(path)
    with h5py.File(path, "a") as f:
        del f[missing]
    with pytest.raises(KeyError):
        L1Data.from_cd1l(str(path))


def test_invalid_auxiliary_file_sampling_fails(tmp_path):
    h5py = pytest.importorskip("h5py")

    path = tmp_path / "brick.h5"
    write_cd1l(path)
    add_auxiliary_groups(path)
    with h5py.File(path, "a") as f:
        f["ttl/ttl_coefficients/sampling"].attrs["dt"] = 0.0
    with pytest.raises(ValueError, match="positive dt"):
        L1Data.from_cd1l(str(path))


def noise_tables():
    rng = np.random.default_rng(42)
    return {
        "noise_estimates_xyz": rng.normal(size=(3, 4, 3, 3))
        + 1j * rng.normal(size=(3, 4, 3, 3)),
        "noise_estimate_start_time_gps": T0 - 5,
        "noise_estimate_sample_interval_s": 60.0,
        "noise_estimate_min_frequency_hz": 1e-5,
        "noise_estimate_max_frequency_hz": 1.0,
        "laser_frequency_hz": LASER_FREQUENCY_HZ,
    }


def test_raw_noise_estimates_keep_values_grids_and_references():
    tables = noise_tables()
    data = observed(**tables)
    wheel = Wheel(data, random_seed=0)
    assert wheel.covariance is None
    for carried in (
        replace(data),
        transform(data, "frequency"),
        wheel.observed_data,
        wheel.working_residual,
    ):
        for name, value in tables.items():
            assert getattr(carried, name) is value
    np.testing.assert_array_equal(
        data.noise_estimate_sample_times_gps, T0 - 5 + np.arange(3) * 60
    )
    np.testing.assert_allclose(
        data.noise_estimate_frequencies_hz, np.geomspace(1e-5, 1, 4)
    )
    # Estimates need not have samples at the observation endpoints.
    assert data.noise_estimate_sample_times_gps[-1] < data.t0 + (data.N - 1) * data.dt
    empty = observed()
    assert empty.noise_estimates_xyz is None
    assert empty.noise_estimate_frequencies_hz is None
    assert empty.noise_estimate_sample_times_gps is None


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("noise_estimates_xyz", None, "requires noise_estimates"),
        ("noise_estimate_start_time_gps", None, "requires finite"),
        ("noise_estimate_start_time_gps", np.nan, "requires finite"),
        ("noise_estimate_sample_interval_s", 0, "dt > 0"),
        ("noise_estimate_min_frequency_hz", 0, "fmin < fmax"),
        ("noise_estimate_max_frequency_hz", 1e-6, "fmin < fmax"),
        ("laser_frequency_hz", -1, "positive and finite"),
        ("laser_frequency_hz", np.inf, "positive and finite"),
    ],
)
def test_raw_noise_metadata_validation(field, value, match):
    with pytest.raises(ValueError, match=match):
        observed(**{**noise_tables(), field: value})


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (np.zeros((3, 4, 3, 3), dtype=int), "floating-point or complex"),
        ([], "floating-point or complex"),
        (np.zeros((3, 4, 3)), "must have shape"),
        (np.zeros((3, 4, 6, 6)), "must have shape"),
        (np.zeros((0, 4, 3, 3)), "at least one time"),
        (np.zeros((3, 1, 3, 3)), "two frequencies"),
    ],
)
def test_raw_noise_matrix_validation(value, match):
    tables = noise_tables()
    tables["noise_estimates_xyz"] = value
    with pytest.raises((TypeError, ValueError), match=match):
        observed(**tables)


def add_noise_estimates(path):
    h5py = pytest.importorskip("h5py")
    tables = noise_tables()
    with h5py.File(path, "a") as f:
        group = f.create_group("noise_estimates")
        group.create_group("sampling").attrs.update(t0=T0 - 5, dt=60.0, size=3)
        group.create_group("log_frequency_sampling").attrs.update(
            fmin=1e-5, fmax=1.0, size=4
        )
        group.create_dataset("XYZ", data=tables["noise_estimates_xyz"])
        # Invalid unused datasets prove the loader only inspects XYZ.
        group.create_dataset("AET", data=np.zeros(1))
        group.create_dataset("eta", data=np.zeros(1))
    return tables


@pytest.mark.parametrize(("duration_s", "expected_count"), [(None, 3), (1.0, 3)])
def test_noise_estimates_loaded_without_normalization(
    tmp_path, duration_s, expected_count
):
    path = tmp_path / "noise.h5"
    write_cd1l(path)
    tables = add_noise_estimates(path)
    data = L1Data.from_cd1l(str(path), duration_s=duration_s)
    assert data.laser_frequency_hz == LASER_FREQUENCY_HZ
    np.testing.assert_array_equal(
        data.noise_estimates_xyz, tables["noise_estimates_xyz"][:expected_count]
    )
    assert not data.noise_estimates_xyz.flags.writeable
    for name, value in tables.items():
        if name != "noise_estimates_xyz":
            assert getattr(data, name) == value


def test_noise_estimate_prefix_avoids_reading_whole_table(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "noise.h5"
    write_cd1l(path)
    add_noise_estimates(path)
    with h5py.File(path, "a") as f:
        f["noise_estimates/sampling"].attrs["t0"] = T0
    data = L1Data.from_cd1l(str(path), duration_s=1)
    assert data.noise_estimates_xyz.shape[0] == 2


@pytest.mark.parametrize(
    "defect", ["missing_matrix", "frequency_size", "time_size", "zero_dt"]
)
def test_malformed_noise_groups_fail(tmp_path, defect):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "noise.h5"
    write_cd1l(path)
    add_noise_estimates(path)
    with h5py.File(path, "a") as f:
        group = f["noise_estimates"]
        if defect == "missing_matrix":
            del group["XYZ"]
        elif defect == "frequency_size":
            group["log_frequency_sampling"].attrs["size"] = 5
        elif defect == "time_size":
            group["sampling"].attrs["size"] = 5
        else:
            group["sampling"].attrs["dt"] = 0
    with pytest.raises((ValueError, KeyError)):
        L1Data.from_cd1l(str(path))
