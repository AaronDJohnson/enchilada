"""Object conversion preserves physical grids, numerical values and owned state."""

import importlib.util
from dataclasses import replace

import numpy as np
import pytest

from enchilada import BlockResult, DataCovariance, L1Data, WDMGrid
from enchilada.translation import transform

requires_wdm = pytest.mark.skipif(
    importlib.util.find_spec("wdm") is None, reason="local WDM backend is optional"
)


def observations(num_time_samples=64, sample_interval_s=0.25):
    rng = np.random.default_rng(17)
    return L1Data(
        channel_data={name: rng.normal(size=num_time_samples) for name in ("A", "E")},
        channel_names=("A", "E"),
        sample_rate_hz=1 / sample_interval_s,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
        start_time_gps=100.0,
    )


@pytest.mark.parametrize("num_time_samples", [1, 63, 64])
def test_fourier_round_trip_uses_continuous_coefficients(num_time_samples):
    data = observations(num_time_samples)
    spectrum = transform(data, "frequency")
    restored = transform(spectrum, "time")
    for name in data.channel_names:
        np.testing.assert_allclose(
            spectrum.channel_data[name],
            data.sample_interval_s * np.fft.rfft(data.channel_data[name]),
        )
        np.testing.assert_allclose(
            restored.channel_data[name], data.channel_data[name], atol=1e-14
        )
    assert restored.start_time_gps == data.start_time_gps
    assert restored.num_time_samples == num_time_samples
    assert spectrum.sample_rate_hz == data.sample_rate_hz
    assert spectrum.channel_names == data.channel_names
    assert spectrum.physical_observable == data.physical_observable


@pytest.mark.parametrize("conversion", ["transform", "to_time"])
@pytest.mark.parametrize(("amplitude", "sample_rate_hz"), [(1e-42, 1e-4), (1e38, 1e4)])
def test_frequency_normalization_preserves_complex64_dynamic_range(
    conversion, amplitude, sample_rate_hz
):
    spectrum = np.array([0, amplitude, 0], dtype=np.complex64)
    source = L1Data(
        channel_data={"A": spectrum},
        channel_names=("A",),
        sample_rate_hz=sample_rate_hz,
        num_time_samples=4,
        tdi_generation="2.0",
        physical_observable="strain",
        data_domain="frequency",
    )
    # This bin is a cosine: four samples are [a/(2*dt), 0, -a/(2*dt), 0].
    peak = float(spectrum[1].real) * sample_rate_hz / 2
    expected = np.array([peak, 0, -peak, 0])
    with np.errstate(over="raise", invalid="raise"):
        restored = (
            transform(source, "time") if conversion == "transform" else source.to_time()
        )
    np.testing.assert_allclose(restored.channel_data["A"], expected, rtol=1e-12, atol=0)


def test_same_domain_transform_is_an_independent_snapshot():
    data = observations()
    copied = transform(data, "time")
    copied.channel_data["A"][:] = 0
    assert np.any(data.channel_data["A"])
    covariance = DataCovariance.from_variance(data, 2.0)
    copied_covariance = transform(covariance, "time")
    assert not np.shares_memory(
        copied_covariance.covariance_matrix, covariance.covariance_matrix
    )


def test_result_conversion_preserves_state_without_aliasing():
    data = observations()
    original = data.block_result(
        data.channel_data,
        noise_covariance=DataCovariance.from_variance(data, 3.0),
        model_parameters={"amplitude": np.array([2.0])},
        sampler_state={"iteration": 3},
        metadata={"source": "test"},
    )
    converted = transform(original, "frequency", reference_data=data)
    spectrum = transform(data, "frequency")
    for name in data.channel_names:
        np.testing.assert_allclose(
            converted.tdi_signal_contribution[name], spectrum.channel_data[name]
        )
    assert converted.sampler_state == original.sampler_state
    assert converted.metadata == original.metadata
    assert converted.noise_covariance.data_domain == "frequency"
    converted.model_parameters["amplitude"][0] = 7
    assert original.model_parameters["amplitude"][0] == 2


def test_noise_only_and_state_only_results_need_no_signal_grid():
    data = observations()
    result = BlockResult(noise_covariance=DataCovariance.from_variance(data, 1.0))
    converted = transform(result, "frequency")
    assert converted.tdi_signal_contribution is None
    assert converted.noise_covariance.data_domain == "frequency"
    state = BlockResult(sampler_state={"iteration": 1})
    copied = transform(state, "frequency")
    assert copied is not state
    assert copied.sampler_state == state.sampler_state


def test_conversion_rejects_missing_or_mismatched_reference():
    data = observations()
    result = data.block_result(data.channel_data)
    with pytest.raises(ValueError, match="reference_data"):
        transform(result, "frequency")
    with pytest.raises(ValueError, match="length|shape"):
        transform(result, "frequency", reference_data=observations(32))
    with pytest.raises(ValueError, match="sample_rate_hz"):
        transform(
            DataCovariance.from_variance(data, 1.0),
            "frequency",
            reference_data=replace(data, sample_rate_hz=2.0),
        )


@pytest.mark.parametrize("target", ["", "wavelet", "Frequency"])
def test_unknown_domains_are_rejected(target):
    with pytest.raises(ValueError, match="target_domain"):
        transform(observations(), target)


def test_conversion_checks_nonfinite_data_and_irrelevant_grid_options():
    data = observations()
    with pytest.raises(ValueError, match="wdm"):
        transform(data, "frequency", num_frequency_divisions=8)
    with pytest.raises(TypeError, match="L1Data|BlockResult|covariance"):
        transform(np.ones(8), "frequency")
    data.channel_data["A"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        transform(data, "frequency")


@requires_wdm
@pytest.mark.parametrize("source_domain", ["time", "frequency"])
@pytest.mark.parametrize("sample_interval_s", [0.125, 10.0])
def test_actual_wdm_backend_and_all_round_trip_routes(source_domain, sample_interval_s):
    import wdm

    data = observations(sample_interval_s=sample_interval_s)
    source = transform(data, source_domain)
    converted = transform(source, "wdm", num_frequency_divisions=8)
    assert converted.wdm_grid.num_frequency_divisions == 8
    assert converted.wdm_grid.num_time_divisions == 8
    for name in data.channel_names:
        expected = wdm.forward_time(data.channel_data[name], 8, 8, sample_interval_s)
        np.testing.assert_allclose(converted.channel_data[name], expected, atol=1e-14)
        np.testing.assert_array_equal(
            converted.channel_data[name][~converted.wdm_grid.active_mask], 0.0
        )
        assert converted.channel_data[name].dtype == np.float64
    for target in ("time", "frequency"):
        restored = transform(converted, target)
        expected = transform(data, target)
        assert restored.wdm_grid is None
        for name in data.channel_names:
            np.testing.assert_allclose(
                restored.channel_data[name], expected.channel_data[name], atol=1e-12
            )
    np.testing.assert_allclose(
        converted.to_time().channel_data["A"], data.channel_data["A"], atol=1e-14
    )
    np.testing.assert_allclose(
        converted.to_frequency().channel_data["A"],
        transform(data, "frequency").channel_data["A"],
        atol=1e-12,
    )


@requires_wdm
def test_wdm_division_selectors_regrid_without_resampling():
    data = observations()
    first = transform(data, "wdm", num_frequency_divisions=8)
    same = transform(data, "wdm", num_time_divisions=8)
    regridded = transform(first, "wdm", num_time_divisions=4)
    direct = transform(data, "wdm", num_frequency_divisions=16)
    assert regridded.channel_data["A"].shape == (17, 4)
    assert regridded.num_time_samples == data.num_time_samples
    np.testing.assert_allclose(first.channel_data["A"], same.channel_data["A"])
    np.testing.assert_allclose(regridded.channel_data["A"], direct.channel_data["A"])
    copied = transform(first, "wdm")
    assert copied.wdm_grid == first.wdm_grid
    assert not np.shares_memory(copied.channel_data["A"], first.channel_data["A"])


def test_first_wdm_conversion_requires_a_valid_division_count():
    with pytest.raises(ValueError, match="division"):
        transform(observations(), "wdm")
    with pytest.raises(ValueError, match="divis|even"):
        transform(observations(63), "wdm", num_time_divisions=8)


def test_missing_wdm_backend_has_a_local_install_hint(monkeypatch):
    import enchilada._transforms as backend

    def unavailable(name):
        raise ModuleNotFoundError("No module named 'wdm'", name="wdm")

    monkeypatch.setattr(backend.importlib, "import_module", unavailable)
    with pytest.raises(ImportError, match="local.*wdm|wdm.*local"):
        transform(observations(), "wdm", num_time_divisions=8)


def test_joint_wdm_result_reuses_signal_grid_for_covariance():
    data = observations()
    grid = WDMGrid(num_frequency_divisions=8, num_time_divisions=8)
    reference = replace(
        data,
        data_domain="wdm",
        wdm_grid=grid,
        channel_data={name: np.zeros(grid.array_shape) for name in data.channel_names},
    )
    result = reference.block_result(
        reference.channel_data, noise_covariance=DataCovariance.from_variance(data, 2.0)
    )
    converted = transform(result, "wdm", reference_data=reference)
    assert converted.noise_covariance.wdm_grid == grid
    assert converted.noise_covariance.data_domain == "wdm"
    assert not np.shares_memory(
        converted.tdi_signal_contribution["A"], result.tdi_signal_contribution["A"]
    )


def test_backend_dependency_failure_is_not_reported_as_missing_wdm(monkeypatch):
    import enchilada._transforms as backend

    def unavailable_dependency(name):
        raise ModuleNotFoundError("No module named 'scipy'", name="scipy")

    monkeypatch.setattr(backend.sys, "version_info", (3, 13))
    monkeypatch.setattr(backend.importlib, "import_module", unavailable_dependency)
    with pytest.raises(ModuleNotFoundError, match="scipy"):
        transform(observations(), "wdm", num_time_divisions=8)


def test_unrelated_wdm_package_and_unsupported_python_have_clear_errors(monkeypatch):
    import enchilada._transforms as backend

    monkeypatch.setattr(backend.sys, "version_info", (3, 13))
    monkeypatch.setattr(backend.importlib, "import_module", lambda name: object())
    with pytest.raises(ImportError, match="not the expected LISA"):
        transform(observations(), "wdm", num_time_divisions=8)
    monkeypatch.setattr(backend.sys, "version_info", (3, 12))
    with pytest.raises(ImportError, match="Python >=3.13"):
        transform(observations(), "wdm", num_time_divisions=8)
