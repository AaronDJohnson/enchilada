"""Exact covariance changes of basis, checked against small real matrices."""

import numpy as np
import pytest

from enchilada import DataCovariance


def test_real_native_operations_accept_real_valued_complex_covariance_storage():
    covariance = DataCovariance(
        np.full((2, 1, 1), 2.0 + 0j),
        channel_names=("A",),
        sample_rate_hz=1.0,
        num_time_samples=2,
        data_domain="time",
    )
    values = {"A": np.array([1.0, 2.0])}
    assert not np.iscomplexobj(covariance.apply(values)["A"])
    assert not np.iscomplexobj(covariance.solve(values)["A"])
    assert covariance.quadratic_form(values) == pytest.approx(2.5)


def test_native_wdm_covariance_uses_only_structural_active_pixels():
    from enchilada.domains import WDMGrid

    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=4)
    covariance = DataCovariance(
        np.full((*grid.array_shape, 1, 1), 2.0),
        channel_names=("A",),
        sample_rate_hz=2.0,
        num_time_samples=16,
        data_domain="wdm",
        wdm_grid=grid,
    )
    np.testing.assert_array_equal(covariance.active_mask, grid.active_mask)
    values = {"A": grid.active_mask.astype(float)}
    solved = covariance.solve(values)["A"]
    np.testing.assert_array_equal(solved, 0.5 * grid.active_mask)
    assert covariance.degrees_of_freedom == 16
    assert covariance.log_determinant() == pytest.approx(16 * np.log(2))
    with pytest.raises(ValueError, match="inactive WDM"):
        DataCovariance(
            covariance.covariance_matrix,
            channel_names=("A",),
            sample_rate_hz=2.0,
            num_time_samples=16,
            data_domain="wdm",
            wdm_grid=grid,
            active_mask=np.ones(grid.array_shape, dtype=bool),
        )


def test_native_wdm_covariance_rejects_complex_cross_channel_values():
    from enchilada.domains import WDMGrid

    grid = WDMGrid(num_frequency_divisions=2, num_time_divisions=2)
    matrix = np.broadcast_to(
        np.array([[2.0, 1j], [-1j, 2.0]]), (*grid.array_shape, 2, 2)
    )
    with pytest.raises(ValueError, match="must be real"):
        DataCovariance(
            matrix,
            channel_names=("A", "E"),
            sample_rate_hz=2.0,
            num_time_samples=4,
            data_domain="wdm",
            wdm_grid=grid,
        )


def test_native_time_precision_projects_excluded_samples():
    covariance = DataCovariance(
        np.array([[[2.0, 0.5], [0.5, 3.0]], [[0.0, 0.0], [0.0, 0.0]]]),
        channel_names=("A", "E"),
        sample_rate_hz=2.0,
        num_time_samples=2,
        data_domain="time",
        active_mask=np.array([True, False]),
    )
    values = {"A": np.array([1.0, 100.0]), "E": np.array([2.0, -100.0])}
    solved = covariance.solve(values)
    expected = np.linalg.solve(covariance.covariance_matrix[0], [1.0, 2.0])
    np.testing.assert_allclose([solved["A"][0], solved["E"][0]], expected)
    assert solved["A"][1] == solved["E"][1] == 0
    applied = covariance.apply(solved)
    projected = covariance.project(values)
    for channel in values:
        np.testing.assert_allclose(applied[channel], projected[channel])
    assert covariance.quadratic_form(values) == pytest.approx(
        np.dot([1.0, 2.0], expected)
    )
    assert covariance.log_determinant() == pytest.approx(
        np.linalg.slogdet(covariance.covariance_matrix[0])[1]
    )
    assert covariance.degrees_of_freedom == 2


@pytest.mark.parametrize("num_time_samples", [1, 5, 6])
def test_frequency_quadratic_counts_real_endpoint_degrees(num_time_samples):
    num_bins = num_time_samples // 2 + 1
    covariance = DataCovariance(
        np.full((num_bins, 1, 1), 3.0),
        channel_names=("A",),
        sample_rate_hz=2.0,
        num_time_samples=num_time_samples,
    )
    spectrum = np.full(num_bins, 1.0 + 2.0j)
    spectrum[0] = 1.0
    if num_time_samples % 2 == 0:
        spectrum[-1] = 1.0
    degrees = np.full(num_bins, 2)
    degrees[0] = 1
    if num_time_samples % 2 == 0:
        degrees[-1] = 1
    assert covariance.quadratic_form({"A": spectrum}) == pytest.approx(
        np.sum(degrees * abs(spectrum) ** 2 / 3.0)
    )
    assert covariance.degrees_of_freedom == num_time_samples
    assert covariance.log_determinant() == pytest.approx(num_time_samples * np.log(3.0))


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("scale", [1.0, 1e-240, 1e240])
@pytest.mark.parametrize("masked", [False, True])
def test_complex_frequency_logdet_is_finite_without_runtime_warnings(scale, masked):
    matrices = scale * np.array(
        [
            [[2.0, 1.0], [1.0, 3.0]],
            [[2.0, 1j], [-1j, 3.0]],
            [[2.0, 1.0], [1.0, 3.0]],
        ],
        dtype=complex,
    )
    active = np.array([not masked, True, not masked])
    matrices[~active] = 0
    covariance = DataCovariance(
        matrices,
        channel_names=("A", "E"),
        sample_rate_hz=2.0,
        num_time_samples=4,
        active_mask=active,
    )
    # Each 2x2 matrix has determinant 5*scale**2. The interior Fourier bin
    # counts twice, and unmasked real DC/Nyquist endpoints each count once.
    weight = 2 if masked else 4
    expected = weight * (np.log(5.0) + 2 * np.log(scale))
    assert covariance.log_determinant() == pytest.approx(expected, rel=1e-12)


def test_frequency_to_time_retains_nonlocal_correlations_and_masks():
    from enchilada.translated_covariance import translate_covariance

    covariance = DataCovariance(
        np.array([[[0.0]], [[3.0]], [[5.0]]]),
        channel_names=("A",),
        sample_rate_hz=2.0,
        num_time_samples=4,
        active_mask=np.array([False, True, True]),
    )
    translated = translate_covariance(covariance, "time")
    # Columns independently synthesize real quadrature coordinates, including
    # the sqrt(2) interior-bin weighting in the declared Fourier metric.
    synthesis = np.column_stack(
        [
            np.fft.irfft(np.array(z) / 0.5, n=4)
            for z in (
                [1.0, 0.0, 0.0],
                [0.0, 1 / np.sqrt(2), 0.0],
                [0.0, 1j / np.sqrt(2), 0.0],
                [0.0, 0.0, 1.0],
            )
        ]
    )
    dense = synthesis @ np.diag([0.0, 3.0, 3.0, 5.0]) @ synthesis.T
    values = {"A": np.array([1.0, 2.0, -1.0, 3.0])}
    np.testing.assert_allclose(translated.apply(values)["A"], dense @ values["A"])
    np.testing.assert_allclose(
        translated.solve(values)["A"], np.linalg.pinv(dense) @ values["A"]
    )
    assert translated.active_mask is None
    assert translated.degrees_of_freedom == 3
    assert translated.log_determinant() == pytest.approx(np.log([3.0, 3.0, 5.0]).sum())
    restored = translate_covariance(translated, "frequency")
    assert isinstance(restored, DataCovariance)
    np.testing.assert_array_equal(
        restored.covariance_matrix, covariance.covariance_matrix
    )
    np.testing.assert_array_equal(restored.active_mask, covariance.active_mask)


def _pack(values, domain, grid, num_time_samples):
    """Independent real-coordinate convention used by the dense references."""
    rows = []
    for array in values.values():
        if domain == "time":
            row = array
        elif domain == "wdm":
            row = array[grid.active_mask]
        else:
            row = np.empty(num_time_samples)
            row[0] = array[0].real
            row[-1] = array[-1].real
            row[1:-1:2] = np.sqrt(2) * array[1:-1].real
            row[2:-1:2] = np.sqrt(2) * array[1:-1].imag
        rows.append(row)
    return np.concatenate(rows)


def _unpack(vector, domain, grid, num_time_samples):
    values = {}
    for name, row in zip(("A", "E"), vector.reshape(2, num_time_samples), strict=True):
        if domain == "time":
            array = row.copy()
        elif domain == "wdm":
            array = np.zeros(grid.array_shape)
            array[grid.active_mask] = row
        else:
            array = np.empty(num_time_samples // 2 + 1, dtype=complex)
            array[0] = row[0]
            array[-1] = row[-1]
            array[1:-1] = (row[1:-1:2] + 1j * row[2:-1:2]) / np.sqrt(2)
        values[name] = array
    return values


def _through_time(values, source_domain, target_domain, grid, dt, num_time_samples):
    if "wdm" in (source_domain, target_domain):
        wdm = pytest.importorskip("wdm")
    output = {}
    for name, array in values.items():
        if source_domain == "time":
            time_values = array
        elif source_domain == "frequency":
            time_values = np.fft.irfft(array / dt, n=num_time_samples)
        else:
            time_values = wdm.inverse_time(
                array, grid.num_frequency_divisions, grid.num_time_divisions, dt
            )
        if target_domain == "time":
            output[name] = time_values.copy()
        elif target_domain == "frequency":
            output[name] = dt * np.fft.rfft(time_values)
        else:
            output[name] = wdm.forward_time(
                time_values, grid.num_frequency_divisions, grid.num_time_divisions, dt
            )
    return output


def _dense_native(covariance):
    """Assemble only small reference matrices, without production operators."""
    count = covariance.num_time_samples
    dense = np.zeros((2 * count, 2 * count))
    matrices = covariance.covariance_matrix
    active = covariance.active_mask
    if covariance.data_domain == "wdm":
        matrices = matrices[covariance.wdm_grid.active_mask]
        active = active[covariance.wdm_grid.active_mask]
    for index, matrix in enumerate(matrices):
        if not active[index]:
            continue
        if covariance.data_domain != "frequency":
            indices = np.arange(2) * count + index
            dense[np.ix_(indices, indices)] = matrix.real
        elif index in (0, count // 2):
            indices = np.arange(2) * count + (0 if index == 0 else count - 1)
            dense[np.ix_(indices, indices)] = matrix.real
        else:
            real_indices = np.arange(2) * count + 2 * index - 1
            imaginary_indices = real_indices + 1
            dense[np.ix_(real_indices, real_indices)] = matrix.real
            dense[np.ix_(imaginary_indices, imaginary_indices)] = matrix.real
            dense[np.ix_(real_indices, imaginary_indices)] = -matrix.imag
            dense[np.ix_(imaginary_indices, real_indices)] = matrix.imag
    return dense


@pytest.mark.parametrize("source_domain", ["time", "frequency", "wdm"])
@pytest.mark.parametrize("target_domain", ["time", "frequency", "wdm"])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("scale", [1.0, 1e-40])
def test_covariance_change_of_basis_matches_dense_reference(
    source_domain, target_domain, masked, scale
):
    from enchilada.domains import WDMGrid
    from enchilada.translated_covariance import translate_covariance

    if "wdm" in (source_domain, target_domain):
        pytest.importorskip("wdm")
    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=4)
    count, dt = grid.num_time_samples, 0.3
    rng = np.random.default_rng(274)
    shape = (
        grid.array_shape
        if source_domain == "wdm"
        else (count if source_domain == "time" else count // 2 + 1,)
    )
    factors = rng.normal(size=(*shape, 2, 2))
    if source_domain == "frequency":
        factors = factors.astype(complex)
        factors[1:-1] += 1j * rng.normal(size=factors[1:-1].shape)
    matrices = scale * (factors @ factors.conj().swapaxes(-1, -2) + np.eye(2))
    active = grid.active_mask.copy() if source_domain == "wdm" else np.ones(shape, bool)
    if masked:
        active.flat[0] = False
        active.flat[3] = False
    covariance = DataCovariance(
        matrices,
        channel_names=("A", "E"),
        sample_rate_hz=1 / dt,
        num_time_samples=count,
        data_domain=source_domain,
        wdm_grid=grid if source_domain == "wdm" else None,
        active_mask=active,
    )
    translated = translate_covariance(
        covariance, target_domain, grid if target_domain == "wdm" else None
    )
    basis = np.eye(2 * count)
    transform = np.column_stack(
        [
            _pack(
                _through_time(
                    _unpack(row, source_domain, grid, count),
                    source_domain,
                    target_domain,
                    grid,
                    dt,
                    count,
                ),
                target_domain,
                grid,
                count,
            )
            for row in basis
        ]
    )
    dense = transform @ _dense_native(covariance) @ transform.T / scale
    vector = rng.normal(size=2 * count)
    values = _unpack(vector, target_domain, grid, count)
    actual_covariance = (
        _pack(translated.apply(values), target_domain, grid, count) / scale
    )
    inverse = np.linalg.pinv(dense, hermitian=True)
    actual_precision = (
        _pack(translated.solve(values), target_domain, grid, count) * scale
    )
    np.testing.assert_allclose(
        actual_covariance, dense @ vector, rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        actual_precision, inverse @ vector, rtol=2e-12, atol=2e-12
    )
    projected = _pack(translated.project(values), target_domain, grid, count)
    np.testing.assert_allclose(projected, dense @ inverse @ vector, atol=3e-12)
    assert translated.quadratic_form(values) * scale == pytest.approx(
        vector @ inverse @ vector, rel=2e-12
    )
    eigenvalues = np.linalg.eigvalsh(dense)
    positive = eigenvalues[eigenvalues > 1e-11 * eigenvalues.max()]
    assert translated.degrees_of_freedom == positive.size
    expected_logdet = np.log(positive).sum() + positive.size * np.log(scale)
    assert translated.log_determinant() == pytest.approx(expected_logdet, abs=3e-11)


@pytest.mark.parametrize("operation", ["apply", "solve", "project", "quadratic_form"])
@pytest.mark.parametrize(
    "invalid", [np.array([1j, 0j, 0j]), np.zeros(2, complex), np.zeros(3)]
)
def test_translated_operations_validate_frequency_inputs_before_inverse(
    operation, invalid
):
    from enchilada.translated_covariance import translate_covariance

    covariance = DataCovariance(
        np.ones((4, 1, 1)),
        channel_names=("A",),
        sample_rate_hz=2.0,
        num_time_samples=4,
        data_domain="time",
    )
    translated = translate_covariance(covariance, "frequency")
    with pytest.raises((ValueError, TypeError), match="values"):
        getattr(translated, operation)({"A": invalid})


def test_empty_covariance_subspace_has_zero_precision_and_zero_logdet():
    from enchilada.translated_covariance import translate_covariance

    covariance = DataCovariance(
        np.zeros((4, 1, 1)),
        channel_names=("A",),
        sample_rate_hz=2.0,
        num_time_samples=4,
        data_domain="time",
        active_mask=np.zeros(4, bool),
    )
    for model, values in (
        (covariance, {"A": np.ones(4)}),
        (translate_covariance(covariance, "frequency"), {"A": np.ones(3, complex)}),
    ):
        assert model.degrees_of_freedom == 0
        assert model.log_determinant() == model.quadratic_form(values) == 0
        for operation in (model.apply, model.solve, model.project):
            np.testing.assert_array_equal(
                operation(values)["A"], np.zeros_like(values["A"])
            )


def test_translation_copies_native_buffers_and_recovers_original_grid_after_regrid():
    from enchilada.domains import WDMGrid
    from enchilada.translated_covariance import (
        TranslatedCovariance,
        translate_covariance,
    )

    first = WDMGrid(num_frequency_divisions=4, num_time_divisions=4)
    second = WDMGrid(num_frequency_divisions=2, num_time_divisions=8)
    covariance = DataCovariance(
        np.full((*first.array_shape, 1, 1), 2.0),
        channel_names=("A",),
        sample_rate_hz=2.0,
        num_time_samples=16,
        data_domain="wdm",
        wdm_grid=first,
    )
    regridded = translate_covariance(covariance, "wdm", second)
    assert isinstance(regridded, TranslatedCovariance)
    intermediate = translate_covariance(regridded, "frequency")
    assert isinstance(intermediate.source_covariance, DataCovariance)
    restored = translate_covariance(intermediate, "wdm", first)
    assert isinstance(restored, DataCovariance)
    assert restored.wdm_grid == first
    for snapshot in (
        regridded.source_covariance,
        intermediate.source_covariance,
        restored,
    ):
        np.testing.assert_array_equal(
            snapshot.covariance_matrix, covariance.covariance_matrix
        )
        np.testing.assert_array_equal(snapshot.active_mask, covariance.active_mask)
        assert not np.shares_memory(
            snapshot.covariance_matrix, covariance.covariance_matrix
        )
    covariance.covariance_matrix.setflags(write=True)
    covariance.covariance_matrix[:] = -2
    np.testing.assert_array_equal(
        restored.covariance_matrix, np.full_like(restored.covariance_matrix, 2)
    )


def test_translated_copy_revalidates_and_trusted_copy_keeps_owned_buffers(monkeypatch):
    from enchilada.translated_covariance import copy_covariance, translate_covariance

    covariance = DataCovariance(
        np.ones((4, 1, 1)),
        channel_names=("A",),
        sample_rate_hz=2.0,
        num_time_samples=4,
        data_domain="time",
    )
    translated = translate_covariance(covariance, "frequency")
    with monkeypatch.context() as patch:

        def unexpected_factorization(*args, **kwargs):
            raise AssertionError("owned snapshots should not repeat factorization")

        patch.setattr(np.linalg, "cholesky", unexpected_factorization)
        snapshot = copy_covariance(translated, validated=True)
    assert not np.shares_memory(
        snapshot.source_covariance.covariance_matrix,
        translated.source_covariance.covariance_matrix,
    )
    assert not snapshot.source_covariance.covariance_matrix.flags.writeable
    translated.source_covariance.covariance_matrix.setflags(write=True)
    translated.source_covariance.covariance_matrix[:] = -1
    with pytest.raises(ValueError, match="positive definite"):
        copy_covariance(translated)


def test_wdm_regrid_keeps_exact_noise_weights_and_covariance_action():
    wdm = pytest.importorskip("wdm")
    from enchilada.domains import WDMGrid
    from enchilada.translated_covariance import translate_covariance

    first = WDMGrid(num_frequency_divisions=4, num_time_divisions=4)
    second = WDMGrid(num_frequency_divisions=2, num_time_divisions=8)
    variances = 1.0 + np.arange(20).reshape(first.array_shape)
    active = first.active_mask.copy()
    active[1, 1] = False
    covariance = DataCovariance(
        variances[..., None, None],
        channel_names=("A",),
        sample_rate_hz=2.0,
        num_time_samples=16,
        data_domain="wdm",
        wdm_grid=first,
        active_mask=active,
    )
    rng = np.random.default_rng(552)
    native_values = rng.normal(size=first.array_shape)
    native_values[~first.active_mask] = 0
    regridded_values = wdm.forward_time(
        wdm.inverse_time(native_values, 4, 4, 0.5), 2, 8, 0.5
    )
    regridded_values[~second.active_mask] = 0
    translated = translate_covariance(covariance, "wdm", second)
    assert translated.quadratic_form({"A": regridded_values}) == pytest.approx(
        np.sum(native_values[active] ** 2 / variances[active]), rel=1e-12
    )
    expected = wdm.forward_time(
        wdm.inverse_time(native_values * variances * active, 4, 4, 0.5), 2, 8, 0.5
    )
    np.testing.assert_allclose(
        translated.apply({"A": regridded_values})["A"], expected, atol=2e-13
    )


def test_translated_covariance_refuses_misleading_diagonal_accessors():
    from enchilada.translated_covariance import translate_covariance

    covariance = DataCovariance(
        np.arange(1.0, 5.0).reshape(4, 1, 1),
        channel_names=("A",),
        sample_rate_hz=2.0,
        num_time_samples=4,
        data_domain="time",
    )
    translated = translate_covariance(covariance, "frequency")
    with pytest.raises(AttributeError, match="cross-point correlations"):
        _ = translated.covariance_matrix
    with pytest.raises(ValueError, match="native frequency"):
        translated.noise_psd()
    with pytest.raises(ValueError, match="cross-point covariance"):
        translated.noise_variance()


@pytest.mark.parametrize(
    "domain,grid,error",
    [
        ("spectrogram", None, "data_domain"),
        ("wdm", None, "wdm_grid"),
        ("time", "unexpected", "wdm_grid"),
    ],
)
def test_translated_covariance_rejects_inconsistent_target_grid(domain, grid, error):
    from enchilada.translated_covariance import translate_covariance

    covariance = DataCovariance(
        np.ones((4, 1, 1)),
        channel_names=("A",),
        sample_rate_hz=2.0,
        num_time_samples=4,
        data_domain="time",
    )
    with pytest.raises(ValueError, match=error):
        translate_covariance(covariance, domain, grid)


@pytest.mark.parametrize("translated", [False, True])
@pytest.mark.parametrize(
    "invalid",
    [{"E": np.ones(4)}, {"A": [1.0, 2.0, 3.0, 4.0]}, {"A": np.full(4, np.nan)}],
)
def test_covariance_operands_cannot_change_channels_or_hide_invalid_samples(
    translated, invalid
):
    from enchilada.translated_covariance import translate_covariance

    covariance = DataCovariance(
        np.ones((4, 1, 1)),
        channel_names=("A",),
        sample_rate_hz=2.0,
        num_time_samples=4,
        data_domain="time",
    )
    model = translate_covariance(covariance, "frequency") if translated else covariance
    with pytest.raises((ValueError, TypeError), match="values"):
        model.apply(invalid)


def test_wdm_covariance_metadata_must_match_native_sample_count():
    from enchilada.domains import WDMGrid
    from enchilada.translated_covariance import translate_covariance

    grid = WDMGrid(num_frequency_divisions=2, num_time_divisions=2)
    covariance = DataCovariance(
        np.ones((8, 1, 1)),
        channel_names=("A",),
        sample_rate_hz=2.0,
        num_time_samples=8,
        data_domain="time",
    )
    with pytest.raises(ValueError, match="num_time_samples"):
        translate_covariance(covariance, "wdm", grid)
    with pytest.raises(ValueError, match="wdm_grid"):
        DataCovariance(
            np.ones((8, 1, 1)),
            channel_names=("A",),
            sample_rate_hz=2.0,
            num_time_samples=8,
            data_domain="wdm",
        )
    with pytest.raises(ValueError, match="num_time_samples"):
        DataCovariance(
            np.ones((*grid.array_shape, 1, 1)),
            channel_names=("A",),
            sample_rate_hz=2.0,
            num_time_samples=8,
            data_domain="wdm",
            wdm_grid=grid,
        )
    with pytest.raises(ValueError, match="wdm_grid"):
        DataCovariance(
            np.ones((8, 1, 1)),
            channel_names=("A",),
            sample_rate_hz=2.0,
            num_time_samples=8,
            data_domain="time",
            wdm_grid=grid,
        )
    native_wdm = DataCovariance(
        np.ones((*grid.array_shape, 1, 1)),
        channel_names=("A",),
        sample_rate_hz=2.0,
        num_time_samples=4,
        data_domain="wdm",
        wdm_grid=grid,
    )
    with pytest.raises(ValueError, match="native time or frequency"):
        native_wdm.noise_variance()


def test_frequency_quadratic_preserves_small_complex64_operands():
    from enchilada.translation import transform

    covariance = DataCovariance(
        np.full((4, 1, 1), 1e-92),
        channel_names=("A",),
        sample_rate_hz=1e-4,
        num_time_samples=4,
        data_domain="time",
    )
    translated = transform(covariance, "frequency")
    spectrum = np.array([0, 1e-42, 0], dtype=np.complex64)
    # The real cosine has two nonzero samples, each with amplitude a/(2*dt).
    peak = float(spectrum[1].real) / 2e4
    expected = 2 * peak**2 / 1e-92
    assert translated.quadratic_form({"A": spectrum}) == pytest.approx(
        expected, rel=1e-12, abs=0
    )


def test_frequency_operand_is_widened_before_wdm_normalization():
    pytest.importorskip("wdm")
    from enchilada.domains import WDMGrid
    from enchilada.translated_covariance import translate_covariance

    grid = WDMGrid(num_frequency_divisions=2, num_time_divisions=2)
    covariance = DataCovariance(
        np.ones((*grid.array_shape, 1, 1)),
        channel_names=("A",),
        sample_rate_hz=1e-4,
        num_time_samples=4,
        data_domain="wdm",
        wdm_grid=grid,
    )
    translated = translate_covariance(covariance, "frequency")
    spectrum = np.array([0, 1e-42, 0], dtype=np.complex64)
    expected = translated.quadratic_form({"A": spectrum.astype(np.complex128)})
    assert expected > 0
    np.testing.assert_allclose(
        translated.quadratic_form({"A": spectrum}), expected, rtol=1e-12, atol=0
    )


@pytest.mark.parametrize("num_time_samples", [1, 2, 3, 5])
@pytest.mark.parametrize("sample_interval_s", [0.3, 4.0])
def test_fft_covariance_roundtrip_preserves_tiny_and_odd_sample_grids(
    num_time_samples, sample_interval_s
):
    from enchilada.translated_covariance import translate_covariance

    rng = np.random.default_rng(722)
    factors = rng.normal(size=(num_time_samples, 2, 2))
    matrices = factors @ factors.swapaxes(-1, -2) + np.eye(2)
    covariance = DataCovariance(
        matrices,
        channel_names=("A", "E"),
        sample_rate_hz=1 / sample_interval_s,
        num_time_samples=num_time_samples,
        data_domain="time",
    )
    values = {
        name: rng.normal(size=num_time_samples) for name in covariance.channel_names
    }
    spectra = {
        name: sample_interval_s * np.fft.rfft(array) for name, array in values.items()
    }
    translated = translate_covariance(covariance, "frequency")
    assert translated.quadratic_form(spectra) == pytest.approx(
        covariance.quadratic_form(values), rel=1e-12
    )
    expected_logdet = covariance.log_determinant() + 2 * num_time_samples * np.log(
        num_time_samples * sample_interval_s**2
    )
    assert translated.log_determinant() == pytest.approx(expected_logdet, abs=1e-12)
    restored = translated.apply(translated.solve(spectra))
    for name in spectra:
        np.testing.assert_allclose(
            restored[name], spectra[name], rtol=2e-12, atol=2e-12
        )
        assert restored[name][0].imag == 0
        if num_time_samples % 2 == 0:
            assert restored[name][-1].imag == 0
