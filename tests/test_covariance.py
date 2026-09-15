"""Covariance units, correlations, and ownership at the noise boundary."""

from dataclasses import replace

import numpy as np
import pytest

from conftest import make_observed
from enchilada import DataCovariance
from enchilada.data import L1Data


def data(n=8, fs=2.0):
    return L1Data(
        channel_data={"A": np.zeros(n), "E": np.zeros(n)},
        channel_names=("A", "E"),
        sample_rate_hz=fs,
        tdi_generation="2.0",
        physical_observable="strain",
    )


def covariance(covariance_matrix, **overrides):
    from enchilada.covariance import DataCovariance

    settings = dict(channel_names=("A", "E"), sample_rate_hz=2.0, num_time_samples=8)
    settings.update(overrides)
    return DataCovariance(covariance_matrix, **settings)


def test_psd_becomes_actual_coefficient_covariance_and_keeps_channel_order():
    from enchilada.covariance import DataCovariance

    class Noise:
        def psd(self, frequencies, channel=None):
            return np.full_like(frequencies, 3.0 if channel == "A" else 7.0)

    result = DataCovariance.from_psd(reference_data=data(), noise_psd_model=Noise())
    # T=4 s: the Fourier coefficient covariance is twice the one-sided PSD.
    np.testing.assert_array_equal(result.covariance_matrix[1:, 0, 0], 6.0)
    np.testing.assert_array_equal(result.covariance_matrix[1:, 1, 1], 14.0)
    np.testing.assert_array_equal(result.covariance_matrix[:, 0, 1], 0.0)
    np.testing.assert_array_equal(result.active_mask, [False, True, True, True, True])
    np.testing.assert_array_equal(
        result.noise_psd(channel_name="E"), [np.inf, 7, 7, 7, 7]
    )
    assert result.data_domain == "frequency"
    assert result.noise_variance(channel_name="A") == pytest.approx(2.625)


@pytest.mark.parametrize(("n", "variance"), [(8, 0.4375), (9, 4 / 9)])
def test_psd_variance_respects_even_and_odd_endpoint_weights(n, variance):
    from enchilada.covariance import DataCovariance

    class SharedNoise:
        def psd(self, frequencies):
            return np.full_like(frequencies, 0.5)

    result = DataCovariance.from_psd(data(n), SharedNoise())
    assert result.noise_variance() == pytest.approx(variance)


def test_matrix_preserves_channel_correlations_for_a_joint_likelihood():
    covariance_matrix = np.tile([[2.0, -1.0], [-1.0, 2.0]], (5, 1, 1))
    result = covariance(covariance_matrix)
    residual = np.array([1.0, 1.0])
    assert residual @ np.linalg.solve(
        result.covariance_matrix[1], residual
    ) == pytest.approx(2)


def test_complex_cross_spectra_are_allowed_for_interior_bins():
    covariance_matrix = np.tile(np.eye(2, dtype=complex), (5, 1, 1))
    covariance_matrix[1, 0, 1], covariance_matrix[1, 1, 0] = 0.2j, -0.2j
    result = covariance(covariance_matrix)
    assert result.covariance_matrix[1, 0, 1] == 0.2j


@pytest.mark.parametrize("scale", [1.0, 1e-40])
def test_nonhermitian_matrices_are_rejected_at_physical_noise_scales(scale):
    covariance_matrix = np.tile(np.array([[2.0, 1.0], [0.0, 2.0]]) * scale, (5, 1, 1))
    with pytest.raises(ValueError, match="Hermitian"):
        covariance(covariance_matrix)


@pytest.mark.parametrize("scale", [1.0, 1e-40])
@pytest.mark.parametrize("transpose", [False, True])
def test_hermitian_check_respects_each_channels_variance(scale, transpose):
    matrix = np.array([[1.0, 1e-14], [0.0, 1e-40]]) * scale
    if transpose:
        matrix = matrix.T
    with pytest.raises(ValueError, match="Hermitian"):
        covariance(np.tile(matrix, (5, 1, 1)))


@pytest.mark.parametrize("scale", [1.0, 1e-40])
@pytest.mark.parametrize("transpose", [False, True])
def test_accepted_roundoff_is_stored_as_exact_hermitian_covariance(scale, transpose):
    matrix = np.array([[1.0, 0.5e-20 * (1 + 1e-13)], [0.5e-20, 1e-40]]) * scale
    if transpose:
        matrix = matrix.T
    result = covariance(np.tile(matrix, (5, 1, 1)))
    stored = result.covariance_matrix
    np.testing.assert_array_equal(stored, stored.conj().swapaxes(-1, -2))
    assert np.linalg.eigvalsh(stored).min() > 0
    assert stored[0, 0, 1] == pytest.approx(
        (matrix[0, 1] + matrix[1, 0]) / 2, rel=1e-15, abs=0
    )


@pytest.mark.parametrize("scale", [1.0, 1e-40])
def test_positive_definiteness_is_checked_after_roundoff_is_symmetrized(scale):
    # The lower triangle alone is positive definite, but their mean is not.
    matrix = np.array([[1.0, 1 + 6e-13], [1 - 2e-13, 1.0]]) * scale
    with pytest.raises(ValueError, match="positive definite"):
        covariance(np.tile(matrix, (5, 1, 1)))


@pytest.mark.parametrize("diagonal", [1e-300, np.nextafter(0.0, 1.0), 1e308])
def test_exact_hermitian_extreme_scales_do_not_overflow_or_underflow(diagonal):
    matrix = np.tile(np.diag([diagonal, diagonal]), (5, 1, 1))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = covariance(matrix)
    np.testing.assert_array_equal(result.covariance_matrix, matrix)


@pytest.mark.parametrize(
    "variances", [(1e308, 1e-308), (np.nextafter(0.0, 1.0), 1e308)]
)
@pytest.mark.parametrize("transpose", [False, True])
def test_valid_correlations_survive_extreme_channel_scale_ratios(variances, transpose):
    channel_scales = np.sqrt(variances)
    cross_covariance = (0.25 * max(channel_scales)) * min(channel_scales)
    matrix = np.array(
        [
            [variances[0], cross_covariance * (1 + 1e-13)],
            [cross_covariance, variances[1]],
        ]
    )
    if transpose:
        matrix = matrix.T
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        stored = covariance(np.tile(matrix, (5, 1, 1))).covariance_matrix[0]
        normalized = (stored / channel_scales[:, None]) / channel_scales[None, :]
    np.testing.assert_array_equal(stored, stored.T)
    np.testing.assert_allclose(normalized, [[1.0, 0.25], [0.25, 1.0]], rtol=1e-12)
    assert np.linalg.eigvalsh(normalized).min() > 0


def test_complex_roundoff_is_symmetrized_without_changing_zero_inactive_points():
    matrix = np.tile(np.eye(2, dtype=complex), (5, 1, 1))
    matrix[0] = 0
    matrix[1, 0, 0] += 1e-14j
    matrix[1, 0, 1] = 0.2 + 0.3j
    matrix[1, 1, 0] = 0.2 - 0.3j + 1e-14j
    result = covariance(matrix, active_mask=np.array([False, True, True, True, True]))
    stored = result.covariance_matrix
    np.testing.assert_array_equal(stored, stored.conj().swapaxes(-1, -2))
    np.testing.assert_array_equal(stored[0], 0)
    assert np.linalg.eigvalsh(stored[result.active_mask]).min() > 0


@pytest.mark.parametrize(
    "covariance_matrix", [[[1.0, 2.0], [2.0, 1.0]], [[1, 1], [1, 1]]]
)
def test_nonpositive_covariance_is_rejected(covariance_matrix):
    with pytest.raises(ValueError, match="positive definite"):
        covariance(np.tile(np.array(covariance_matrix, dtype=float), (5, 1, 1)))


def test_masked_points_may_be_zero_and_never_produce_a_finite_psd_weight():
    covariance_matrix = np.tile(np.eye(2), (5, 1, 1))
    covariance_matrix[2] = 0
    result = covariance(
        covariance_matrix, active_mask=np.array([True, True, False, True, True])
    )
    assert result.noise_psd("A")[2] == np.inf
    assert np.isfinite(result.covariance_matrix).all()


@pytest.mark.parametrize("data_domain", ["time", "frequency"])
def test_real_sample_points_reject_imaginary_covariance(data_domain):
    size = 8 if data_domain == "time" else 5
    covariance_matrix = np.tile(np.array([[2, 1j], [-1j, 2]]), (size, 1, 1))
    with pytest.raises(ValueError, match="real"):
        covariance(covariance_matrix, data_domain=data_domain)


def test_odd_frequency_last_bin_may_have_complex_cross_spectrum():
    covariance_matrix = np.tile(np.eye(2, dtype=complex), (5, 1, 1))
    covariance_matrix[-1, 0, 1], covariance_matrix[-1, 1, 0] = 0.2j, -0.2j
    assert (
        covariance(covariance_matrix, num_time_samples=9).covariance_matrix[-1, 0, 1]
        == 0.2j
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_time_samples": True}, "num_time_samples"),
        ({"sample_rate_hz": np.nan}, "sample_rate_hz"),
        ({"start_time_gps": np.inf}, "start_time_gps"),
        ({"data_domain": "spectrogram"}, "data_domain"),
        ({"channel_names": ("A", "A")}, "channel_names"),
        ({"active_mask": np.ones(4, dtype=bool)}, "active_mask"),
    ],
)
def test_grid_metadata_and_mask_are_validated(overrides, message):
    with pytest.raises((ValueError, TypeError), match=message):
        covariance(np.tile(np.eye(2), (5, 1, 1)), **overrides)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_covariance_is_rejected_even_when_masked(bad):
    covariance_matrix = np.tile(np.eye(2), (5, 1, 1))
    covariance_matrix[0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        covariance(
            covariance_matrix, active_mask=np.array([False, True, True, True, True])
        )


def test_shape_must_match_grid_and_channels():
    with pytest.raises(ValueError, match="shape"):
        covariance(np.tile(np.eye(2), (4, 1, 1)))


@pytest.mark.parametrize(
    "field", ["sample_rate_hz", "num_time_samples", "start_time_gps", "channel_names"]
)
def test_matching_shape_does_not_hide_metadata_mismatches(field):
    result = covariance(np.tile(np.eye(2), (5, 1, 1)))
    observed = data()
    overrides = {
        "sample_rate_hz": {"sample_rate_hz": 4.0},
        "num_time_samples": {
            "num_time_samples": 9,
            "channel_data": {ch: np.zeros(9) for ch in ("A", "E")},
        },
        "start_time_gps": {"start_time_gps": 1.0},
        "channel_names": {"channel_names": ("E", "A")},
    }
    with pytest.raises(ValueError, match=field):
        result.check_compatible(reference_data=replace(observed, **overrides[field]))
    result.check_compatible(observed)
    result.check_compatible(observed.to_frequency())


def test_constructor_and_copy_own_their_buffers():
    covariance_matrix = np.tile(np.eye(2), (5, 1, 1))
    active_mask = np.ones(5, dtype=bool)
    result = covariance(covariance_matrix, active_mask=active_mask)
    covariance_matrix[:] = 99
    active_mask[:] = False
    np.testing.assert_array_equal(result.covariance_matrix[:, 0, 0], 1)
    assert result.active_mask.all()
    copied = result.copy()
    assert not np.shares_memory(result.covariance_matrix, copied.covariance_matrix)
    assert not np.shares_memory(result.active_mask, copied.active_mask)


def test_owned_covariance_snapshot_skips_revalidation_but_copies_buffers(monkeypatch):
    result = covariance(np.tile(np.eye(2), (5, 1, 1)))

    def unexpected_factorization(*args, **kwargs):
        raise AssertionError("an owned validated snapshot needs no factorization")

    monkeypatch.setattr(np.linalg, "cholesky", unexpected_factorization)
    copied = result._copy_validated()
    np.testing.assert_array_equal(copied.covariance_matrix, result.covariance_matrix)
    np.testing.assert_array_equal(copied.active_mask, result.active_mask)
    assert not copied.covariance_matrix.flags.writeable
    assert not copied.active_mask.flags.writeable
    copied.covariance_matrix.setflags(write=True)
    copied.covariance_matrix[0] *= 2
    copied.active_mask.setflags(write=True)
    copied.active_mask[0] = False
    np.testing.assert_array_equal(result.covariance_matrix[0], np.eye(2))
    assert result.active_mask.all()


def test_public_covariance_copy_revalidates_caller_mutations():
    result = covariance(np.tile(np.eye(2), (5, 1, 1)))
    result.covariance_matrix.setflags(write=True)
    result.covariance_matrix[1, 0, 0] = -1
    with pytest.raises(ValueError, match="positive definite"):
        result.copy()


@pytest.mark.parametrize("variance", [2.0, {"A": 2.0, "E": 3.0}])
def test_time_variance_constructor_uses_variance_without_spectral_scaling(variance):
    from enchilada.covariance import DataCovariance

    result = DataCovariance.from_variance(
        reference_data=data(), time_sample_variance=variance
    )
    assert result.data_domain == "time"
    assert result.covariance_matrix.shape == (8, 2, 2)
    assert result.noise_variance("A") == 2
    assert result.noise_variance("E") == (3 if isinstance(variance, dict) else 2)
    with pytest.raises(ValueError, match="frequency"):
        result.noise_psd()


def test_nonstationary_time_variance_cannot_silently_be_averaged():
    covariance_matrix = np.tile(np.eye(2), (8, 1, 1))
    covariance_matrix[2, 0, 0] = 2
    with pytest.raises(ValueError, match="stationary"):
        covariance(covariance_matrix, data_domain="time").noise_variance("A")


def test_unknown_channel_is_rejected():
    with pytest.raises(ValueError, match="channel"):
        covariance(np.tile(np.eye(2), (5, 1, 1))).noise_psd("T")


def test_psd_model_errors_are_not_swallowed_by_channel_dispatch():
    from enchilada.covariance import DataCovariance

    class Broken:
        def psd(self, frequencies, channel=None):
            raise TypeError("model bug")

    with pytest.raises(TypeError, match="model bug"):
        DataCovariance.from_psd(data(), Broken())


@pytest.mark.parametrize(
    ("values", "error"),
    [
        (0.0, ValueError),
        (-1e-40, ValueError),
        (np.inf, ValueError),
        (np.nan, ValueError),
        (np.ones(2), ValueError),
        (1 + 0j, TypeError),
        ("invalid", TypeError),
    ],
)
def test_psd_adapter_rejects_invalid_noise_before_consumption(values, error):
    from enchilada.covariance import DataCovariance

    class Noise:
        def psd(self, frequencies, channel=None):
            return values

    with pytest.raises(error, match="PSD"):
        DataCovariance.from_psd(data(), Noise())


def test_psd_adapter_requires_the_noise_interface():
    from enchilada.covariance import DataCovariance

    with pytest.raises(TypeError, match="psd"):
        DataCovariance.from_psd(data(), object())


def test_one_sample_psd_model_has_only_an_excluded_dc_bin():
    from enchilada.covariance import DataCovariance

    class Noise:
        def psd(self, frequencies, channel=None):
            raise AssertionError("the empty positive-frequency grid needs no model")

    result = DataCovariance.from_psd(data(1), Noise())
    assert result.noise_variance() == 0
    np.testing.assert_array_equal(result.covariance_matrix, np.zeros((1, 2, 2)))


@pytest.mark.parametrize("variance", [{"A": 1.0}, 1 + 1j, [2.0, 3.0], "2"])
def test_variance_constructor_requires_real_scalars_and_exact_channels(variance):
    from enchilada.covariance import DataCovariance

    with pytest.raises((TypeError, ValueError), match="variance"):
        DataCovariance.from_variance(data(), variance)


def test_time_variance_needs_an_active_sample():
    result = covariance(
        np.zeros((8, 2, 2)), data_domain="time", active_mask=np.zeros(8, dtype=bool)
    )
    with pytest.raises(ValueError, match="active"):
        result.noise_variance()


@pytest.mark.parametrize(
    "covariance_matrix", [np.full((5, 2, 2), "bad"), np.ones((5, 2, 2), bool)]
)
def test_covariance_rejects_nonnumeric_and_boolean_values(covariance_matrix):
    with pytest.raises(TypeError, match="numeric"):
        covariance(covariance_matrix)


def test_active_dc_gets_one_real_degree_of_freedom_in_spectral_variance():
    # Every Fourier coefficient has variance 1. With T=4 the PSD is 0.5;
    # four real sample variances' worth of weighting gives 0.5 * 4 * df=0.5.
    result = covariance(np.tile(np.eye(2), (5, 1, 1)))
    assert result.noise_variance() == 0.5


def test_tiny_valid_correlated_covariance_is_accepted():
    covariance_matrix = np.tile(np.array([[2.0, 0.5], [0.5, 1.0]]) * 1e-40, (5, 1, 1))
    np.testing.assert_array_equal(
        covariance(covariance_matrix).covariance_matrix, covariance_matrix
    )


class FlatPSD:
    """Noise model: flat one-sided PSD, T channel twice A/E."""

    def psd(self, freqs, channel=None):
        return np.full_like(freqs, 2.0 if channel == "T" else 1.0)


class TestPsdChannelGrids:
    def test_psd_grid_matches_rfft(self, observed):
        covariance = DataCovariance.from_psd(observed, FlatPSD())
        psd = covariance.noise_psd()
        assert psd.shape == (observed.num_time_samples // 2 + 1,)
        assert psd[0] == np.inf
        assert np.all(psd[1:] == 1.0)

    def test_channel_dispatch(self, observed):
        covariance = DataCovariance.from_psd(observed, FlatPSD())
        assert covariance.noise_psd(channel_name="T")[1] == 2.0
        assert covariance.noise_psd(channel_name="A")[1] == 1.0

    def test_psd_grid_aligns_with_frequency_domain_data(self, rng):
        n = 64
        obs = make_observed(
            rng,
            data_domain="frequency",
            channel_data={ch: np.zeros(n // 2 + 1, complex) for ch in ("A", "E", "T")},
        )
        covariance = DataCovariance.from_psd(obs, FlatPSD())
        assert covariance.noise_psd().shape == obs.channel_data["A"].shape
        assert obs.df == 1.0 / obs.Tobs


class TestWhiteNoisePsdVariance:
    class White:
        def __init__(self, sigma, fs):
            self.sigma, self.fs = sigma, fs

        def psd(self, f, channel=None):
            return np.full_like(f, 2.0 * self.sigma**2 / self.fs)

    def _covariance(self, n, fs=4.0):
        return DataCovariance.from_psd(data(n, fs), self.White(0.7, fs))

    @pytest.mark.parametrize("n", [2048, 2049])
    def test_independent_of_the_parity_of_n(self, n):
        # the naive sum(psd[1:])*df lands on sigma**2 for even n and
        # sigma**2 (1-1/n) for odd n; the weighted form agrees with itself
        var = self._covariance(n).noise_variance(channel_name="A")
        assert var == pytest.approx(0.49 * (1 - 1 / n), rel=1e-12)

    def test_matches_the_empirical_variance(self):
        n, fs, sigma = 1 << 14, 4.0, 0.7
        r = self._covariance(n, fs=fs)
        rng = np.random.default_rng(0)
        emp = float(np.mean([np.var(rng.normal(0, sigma, n)) for _ in range(40)]))
        assert r.noise_variance() == pytest.approx(emp, rel=0.02)


class TestPsdFrequencyGrid:
    """Pin the actual PSD evaluation grid numerically."""

    class RampPSD:
        """Frequency-dependent, so a wrong grid cannot pass unnoticed."""

        def psd(self, f, channel=None):
            return 1.0 + np.asarray(f)

    @pytest.mark.parametrize(("n", "fs"), [(64, 0.5), (65, 2.0), (1024, 0.1)])
    def test_psd_is_evaluated_on_the_rfft_grid(self, n, fs):
        r = L1Data(
            channel_data={"A": np.zeros(n)},
            sample_rate_hz=fs,
            channel_names=("A",),
            tdi_generation="1.5",
            physical_observable="strain",
        )
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        psd = DataCovariance.from_psd(r, self.RampPSD()).noise_psd()
        assert psd[0] == np.inf
        np.testing.assert_allclose(psd[1:], 1.0 + freqs[1:], rtol=1e-12)


class TestNoisePsdSanity:
    """Invalid PSD estimates are rejected before they can reach a likelihood."""

    @pytest.mark.parametrize(
        ("bad", "what"),
        [
            (np.nan, "NaN from an ill-conditioned Whittle fit"),
            (np.inf, "inf from a divide-by-zero in the model"),
            (-1e-40, "negative from a least-squares PSD in a low-power band"),
            (0.0, "exactly zero: 1/S is inf, so it is not whitenable either"),
        ],
    )
    def test_a_psd_that_is_not_finite_and_positive_is_refused(self, rng, bad, what):
        class BrokenNoise:
            def psd(self, freqs, channel=None):
                out = np.full(freqs.shape, 1e-40)
                out[2] = bad
                return out

        r = make_observed(rng)
        with pytest.raises(ValueError, match="finite and strictly positive"):
            DataCovariance.from_psd(r, BrokenNoise())

    def test_a_finite_positive_psd_passes_through_untouched(self, rng):
        """The guard must not reject the DC bin it sets to +inf itself."""

        class GoodNoise:
            def psd(self, freqs, channel=None):
                return np.full(freqs.shape, 3e-41)

        psd = DataCovariance.from_psd(make_observed(rng), GoodNoise()).noise_psd()
        assert psd[0] == np.inf  # DC carries zero weight by construction
        np.testing.assert_allclose(psd[1:], 3e-41, rtol=1e-14, atol=0.0)
