"""Residuals: the data contract validates itself and assembles noise grids."""

from dataclasses import replace

import numpy as np
import pytest

from conftest import make_observed
from turntable import Residuals


class TestPostInitValidation:
    def test_valid_construction(self, rng):
        obs = make_observed(rng)
        assert obs.Tobs == obs.n_samples / obs.sample_rate

    def test_observable_is_required(self, rng):
        with pytest.raises(TypeError, match="observable"):
            Residuals(
                tdi={"A": np.zeros(8)},
                sample_rate=1.0,
                n_samples=8,
                channels=("A",),
                tdi_generation="2.0",
                epoch=0.0,
            )

    def test_empty_observable_rejected(self, rng):
        with pytest.raises(ValueError, match="fractional_frequency"):
            make_observed(rng, observable="")

    def test_tdi_keys_must_match_channels(self, rng):
        with pytest.raises(ValueError, match="missing.*'T'.*unexpected.*'X'"):
            make_observed(
                rng,
                tdi={"A": np.zeros(64), "E": np.zeros(64), "X": np.zeros(64)},
            )

    def test_array_length_must_match_n_samples(self, rng):
        with pytest.raises(ValueError, match="length 99, expected 64"):
            make_observed(
                rng,
                tdi={"A": np.zeros(64), "E": np.zeros(99), "T": np.zeros(64)},
            )

    def test_complex_time_domain_rejected(self, rng):
        with pytest.raises(TypeError, match="domain='frequency'"):
            make_observed(
                rng, tdi={ch: np.zeros(64, complex) for ch in ("A", "E", "T")}
            )

    def test_unknown_domain_rejected(self, rng):
        with pytest.raises(ValueError, match="domain"):
            make_observed(rng, domain="fourier")

    @pytest.mark.parametrize("sample_rate", [0.0, -1.0, float("nan")])
    def test_bad_sample_rate_rejected(self, rng, sample_rate):
        with pytest.raises(ValueError, match="sample_rate"):
            make_observed(rng, sample_rate=sample_rate)

    def test_frequency_domain_arrays_live_on_rfft_grid(self, rng):
        n = 64
        good = make_observed(
            rng,
            domain="frequency",
            tdi={ch: np.zeros(n // 2 + 1, complex) for ch in ("A", "E", "T")},
        )
        assert good.domain == "frequency"
        with pytest.raises(ValueError, match="rfft grid"):
            make_observed(
                rng,
                domain="frequency",
                tdi={ch: np.zeros(n, complex) for ch in ("A", "E", "T")},
            )

    def test_replace_revalidates(self, observed):
        with pytest.raises(ValueError, match="length"):
            replace(observed, tdi={ch: np.zeros(3) for ch in observed.channels})


class TestAliases:
    def test_long_and_short_names_agree(self, observed):
        for long, short in Residuals.aliases().items():
            assert getattr(observed, long) == getattr(observed, short)

    def test_typo_catcher_suggests(self, observed):
        with pytest.raises(AttributeError, match="did you mean 'Tobs'"):
            _ = observed.T_obs


class FlatPSD:
    """Noise model: flat one-sided PSD, T channel twice A/E."""

    def psd(self, freqs, channel=None):
        return np.full_like(freqs, 2.0 if channel == "T" else 1.0)


class TestNoiseGrids:
    def test_none_without_noise_model(self, observed):
        assert observed.noise_psd() is None

    def test_psd_grid_matches_rfft(self, observed):
        obs = replace(observed, noise=FlatPSD())
        psd = obs.noise_psd()
        assert psd.shape == (observed.n_samples // 2 + 1,)
        assert psd[0] == np.inf
        assert np.all(psd[1:] == 1.0)

    def test_channel_dispatch(self, observed):
        obs = replace(observed, noise=FlatPSD())
        assert obs.noise_psd("T")[1] == 2.0
        assert obs.noise_psd("A")[1] == 1.0

    def test_psd_grid_aligns_with_frequency_domain_data(self, rng):
        n = 64
        obs = make_observed(
            rng,
            domain="frequency",
            tdi={ch: np.zeros(n // 2 + 1, complex) for ch in ("A", "E", "T")},
            noise=FlatPSD(),
        )
        assert obs.noise_psd().shape == obs.tdi["A"].shape
        assert obs.df == 1.0 / obs.Tobs

    def test_contract_error_names_the_interface(self, observed):
        obs = replace(observed, noise=object())
        with pytest.raises(TypeError, match=r"psd\(freqs\[, channel\]\)"):
            obs.noise_psd()


class TestOrbitSpanCheck:
    class StubOrbit:
        t_range = (0.0, 100.0)

    def test_orbit_covering_data_accepted(self, rng):
        obs = make_observed(rng, orbit=self.StubOrbit(), sample_rate=1.0)
        assert obs.orbit is not None  # 64 s of data inside [0, 100]

    @pytest.mark.parametrize("epoch", [-1.0, 50.0])
    def test_orbit_not_covering_data_rejected(self, rng, epoch):
        with pytest.raises(ValueError, match="outside the tabulated ephemeris"):
            make_observed(rng, orbit=self.StubOrbit(), sample_rate=1.0, epoch=epoch)

    def test_orbit_without_t_range_skips_check(self, rng):
        obs = make_observed(rng, orbit=object(), epoch=1e9)
        assert obs.orbit is not None
