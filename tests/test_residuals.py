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

    def test_epoch_is_optional_defaults_to_zero(self, rng):
        obs = Residuals(
            tdi={"A": np.zeros(8)},
            sample_rate=1.0,
            n_samples=8,
            channels=("A",),
            tdi_generation="2.0",
            observable="fractional_frequency",
        )  # no epoch supplied
        assert obs.epoch == 0.0
        assert obs.t0 == 0.0

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

    def test_orbit_tabulated_on_the_data_grid_accepted(self, rng):
        # 64 samples at 1 Hz from epoch 0 -> last sample at t = 63 s. An orbit
        # tabulated on exactly that grid covers every time a segment can ask
        # for, so it must be accepted (it spans [0, 63], not [0, Tobs=64]).
        class GridOrbit:
            t_range = (0.0, 63.0)

        obs = make_observed(rng, orbit=GridOrbit(), sample_rate=1.0, epoch=0.0)
        assert obs.orbit is not None

    # (n_samples, sample_rate) pairs where (n-1)*dt and (n-1)/fs are NOT
    # bit-identical -- i.e. exactly where computing the bound by division
    # spuriously rejects a grid-tabulated orbit -- plus an exactly-representable
    # non-unit rate so the dt scaling itself is pinned.
    @pytest.mark.parametrize(
        ("n_samples", "sample_rate"),
        [(32, 3.0), (32, 6.0), (32, 7.0), (128, 3.0), (64, 0.5), (64, 0.1)],
    )
    def test_data_grid_orbit_accepted_at_any_rate(self, rng, n_samples, sample_rate):
        """An orbit tabulated the documented way is accepted at every rate.

        The grid idiom is `epoch + arange(n) * dt` (NumericOrbit.from_hdf5), so
        the check must compute its bound the same way; deriving it as
        (n-1)/sample_rate differs by an ulp at these rates and rejects.
        """

        class GridOrbit:
            def __init__(self, t_end):
                self.t_range = (0.0, t_end)

        dt = 1.0 / sample_rate
        last = 0.0 + (n_samples - 1) * dt
        obs = make_observed(
            rng,
            n_samples=n_samples,
            orbit=GridOrbit(last),
            sample_rate=sample_rate,
            epoch=0.0,
        )
        assert obs.orbit is not None
        # a full sample interval short is still rejected, at every rate
        with pytest.raises(ValueError, match="outside the tabulated ephemeris"):
            make_observed(
                rng,
                n_samples=n_samples,
                orbit=GridOrbit(last - dt),
                sample_rate=sample_rate,
                epoch=0.0,
            )

    def test_orbit_ending_before_the_last_sample_rejected(self, rng):
        # one hair short of the last sample (63 s) must still be caught
        class ShortOrbit:
            t_range = (0.0, 62.9)

        with pytest.raises(ValueError, match="outside the tabulated ephemeris"):
            make_observed(rng, orbit=ShortOrbit(), sample_rate=1.0, epoch=0.0)

    def test_orbit_without_t_range_skips_check(self, rng):
        obs = make_observed(rng, orbit=object(), epoch=1e9)
        assert obs.orbit is not None


class TestRemainingValidationBranches:
    """Negative cases for the scalar/tdi guards not covered above."""

    def test_n_samples_must_be_a_positive_int(self, observed):
        # 0 is the "derive it" sentinel, so the invalid values are negatives
        # and non-integers
        with pytest.raises(ValueError, match="n_samples must be a positive integer"):
            replace(observed, n_samples=-4)
        with pytest.raises(ValueError, match="n_samples must be a positive integer"):
            replace(observed, n_samples=8.0)  # float is not an int

    def test_epoch_must_be_finite(self, observed):
        with pytest.raises(ValueError, match="epoch must be finite"):
            replace(observed, epoch=float("nan"))

    def test_tdi_generation_must_be_a_non_empty_string(self, observed):
        with pytest.raises(ValueError, match="tdi_generation must be a non-empty"):
            replace(observed, tdi_generation="")

    def test_tdi_must_be_a_dict(self, observed):
        with pytest.raises(TypeError, match="tdi must be a dict"):
            replace(observed, tdi=[1.0, 2.0])

    def test_channels_must_be_non_empty(self, observed):
        with pytest.raises(ValueError, match="channels must be a non-empty"):
            replace(observed, channels=(), tdi={})

    def test_channels_must_not_contain_duplicates(self, rng):
        with pytest.raises(ValueError, match="channels contains duplicates"):
            make_observed(rng, channels=("A", "A"), tdi={"A": np.zeros(64)})

    def test_tdi_values_must_be_1d_arrays(self, observed):
        with pytest.raises(TypeError, match="must be a 1-D numpy array"):
            replace(observed, tdi={ch: [0.0] * 64 for ch in observed.channels})

    def test_private_attribute_miss_raises_plain_attributeerror(self, observed):
        with pytest.raises(AttributeError):
            _ = observed._not_a_field

    def test_unknown_attribute_points_at_the_alias_table(self, observed):
        with pytest.raises(AttributeError, match=r"aliases\(\)"):
            _ = observed.wobble


class TestNSamplesDerivation:
    """n_samples is read off the data where that is exact, required where not."""

    def _kwargs(self, **over):
        base = dict(
            sample_rate=0.1,
            channels=("A", "E"),
            tdi_generation="2.0",
            observable="fractional_frequency",
        )
        base.update(over)
        return base

    def test_time_domain_derives_from_the_arrays(self):
        r = Residuals(tdi={ch: np.zeros(1024) for ch in ("A", "E")}, **self._kwargs())
        assert r.n_samples == 1024
        assert r.Tobs == 1024 / 0.1
        assert r.df == 1.0 / r.Tobs

    def test_explicit_value_still_honoured_and_checked(self):
        kw = self._kwargs()
        tdi = {ch: np.zeros(1024) for ch in ("A", "E")}
        assert Residuals(tdi=tdi, n_samples=1024, **kw).n_samples == 1024
        with pytest.raises(ValueError, match="expected 512"):
            Residuals(tdi=tdi, n_samples=512, **kw)

    def test_frequency_domain_requires_it_and_says_why(self):
        # 513 bins are consistent with n=1024 and n=1025 -- the parity is lost,
        # so turntable asks instead of guessing
        with pytest.raises(ValueError, match="does not determine it") as exc:
            Residuals(
                tdi={ch: np.zeros(513, complex) for ch in ("A", "E")},
                domain="frequency",
                **self._kwargs(),
            )
        msg = str(exc.value)
        assert "n_samples=1024" in msg and "=1025" in msg

    @pytest.mark.parametrize("n", [1024, 1025])
    def test_frequency_domain_accepts_either_parity_when_stated(self, n):
        r = Residuals(
            tdi={ch: np.zeros(n // 2 + 1, complex) for ch in ("A", "E")},
            domain="frequency",
            n_samples=n,
            **self._kwargs(),
        )
        assert r.n_samples == n
        assert r.Tobs == n / 0.1  # the two parities really do differ

    def test_derived_value_survives_replace(self):
        r = Residuals(tdi={ch: np.zeros(64) for ch in ("A", "E")}, **self._kwargs())
        r2 = replace(r, tdi={ch: np.ones(64) for ch in ("A", "E")})
        assert r2.n_samples == 64

    def test_derivation_still_validates_every_channel(self):
        # derived from the first channel, but a ragged second one is caught
        with pytest.raises(ValueError, match="has length 60, expected 64"):
            Residuals(tdi={"A": np.zeros(64), "E": np.zeros(60)}, **self._kwargs())


class TestDomainTransforms:
    """to_frequency/to_time carry n_samples, so the round trip is exact."""

    def _time_residual(self, n, fs=0.2, **over):
        rng = np.random.default_rng(0)
        kw = dict(
            tdi={ch: rng.standard_normal(n) for ch in ("A", "E")},
            sample_rate=fs,
            channels=("A", "E"),
            tdi_generation="1.5",
            observable="fractional_frequency",
        )
        kw.update(over)
        return Residuals(**kw)

    @pytest.mark.parametrize("n", [1024, 1025])  # both parities
    def test_round_trip_is_exact(self, n):
        t = self._time_residual(n)
        f = t.to_frequency()
        assert f.domain == "frequency"
        assert f.tdi["A"].size == n // 2 + 1
        back = f.to_time()
        assert back.domain == "time"
        for ch in t.channels:
            np.testing.assert_allclose(back.tdi[ch], t.tdi[ch], atol=1e-12)

    @pytest.mark.parametrize("n", [1024, 1025])
    def test_n_samples_is_carried_not_restated(self, n):
        # never passed by hand: derived from the arrays, then carried across
        t = self._time_residual(n)
        f = t.to_frequency()
        assert t.n_samples == f.n_samples == n
        assert f.Tobs == t.Tobs and f.df == t.df and f.dt == t.dt

    def test_transforms_are_idempotent_no_ops(self):
        t = self._time_residual(64)
        assert t.to_time() is t
        f = t.to_frequency()
        assert f.to_frequency() is f

    def test_noise_and_orbit_ride_along(self):
        class Noise:
            def psd(self, f, channel=None):
                return np.ones_like(f)

        noise = Noise()
        t = self._time_residual(64, noise=noise)
        assert t.to_frequency().noise is noise

    def test_transform_convention_matches_noise_psd_normalization(self):
        """E[|X(f)|^2] == (Tobs/2) * S(f) for X = dt*rfft(x).

        Pins that to_frequency's convention and noise_psd's normalization are
        the same convention -- in code, not just in the docstrings.
        """
        fs, n, sigma = 0.2, 1 << 13, 0.7

        class White:
            def psd(self, f, channel=None):
                return np.full_like(f, 2.0 * sigma**2 / fs)

        rng = np.random.default_rng(1)
        acc = None
        trials = 40
        for _ in range(trials):
            t = Residuals(
                tdi={"A": rng.normal(0.0, sigma, n)},
                sample_rate=fs,
                channels=("A",),
                tdi_generation="1.5",
                observable="fractional_frequency",
                noise=White(),
            )
            power = np.abs(t.to_frequency().tdi["A"]) ** 2
            acc = power if acc is None else acc + power
        measured = acc / trials
        predicted = 0.5 * t.Tobs * t.noise_psd("A")
        interior = slice(10, -10)
        ratio = float(np.mean(measured[interior] / predicted[interior]))
        assert ratio == pytest.approx(1.0, abs=0.05)


class TestDtypeAndTypeContract:
    """dtype and container types are part of the validated contract."""

    def test_integer_tdi_rejected_at_construction(self, rng):
        # would otherwise fail deep inside the Wheel's ledger arithmetic
        with pytest.raises(TypeError, match="must be floating or complex"):
            make_observed(rng, tdi={ch: np.arange(64) for ch in ("A", "E", "T")})

    def test_object_dtype_rejected(self, rng):
        with pytest.raises(TypeError, match="must be floating or complex"):
            make_observed(
                rng, tdi={ch: np.zeros(64, dtype=object) for ch in ("A", "E", "T")}
            )

    def test_float32_is_allowed(self, rng):
        obs = make_observed(
            rng, tdi={ch: np.zeros(64, np.float32) for ch in ("A", "E", "T")}
        )
        assert obs.tdi["A"].dtype == np.float32

    def test_channels_of_a_wrong_container_type_rejected(self, rng):
        # a set has no order, so it cannot define the channel sequence
        with pytest.raises(TypeError, match="channels must be a tuple"):
            make_observed(rng, channels={"A", "E", "T"})

    def test_channels_list_is_coerced_to_tuple(self, rng):
        # a list would otherwise compare unequal to the tuple a segment returns,
        # and the Wheel would blame the segment for changing a run setting
        obs = make_observed(rng, channels=["A", "E", "T"])
        assert obs.channels == ("A", "E", "T")
        assert isinstance(obs.channels, tuple)

    def test_bool_n_samples_rejected(self, observed):
        with pytest.raises(ValueError, match="n_samples must be a positive integer"):
            replace(observed, n_samples=True)

    def test_equality_is_identity_and_hashing_works(self, rng):
        # dataclass eq over a dict of arrays used to raise a raw numpy error
        a, b = make_observed(rng), make_observed(rng)
        assert a != b and a == a
        assert isinstance(hash(a), int)


class TestNoiseVariance:
    class White:
        def __init__(self, sigma, fs):
            self.sigma, self.fs = sigma, fs

        def psd(self, f, channel=None):
            return np.full_like(f, 2.0 * self.sigma**2 / self.fs)

    def _residual(self, n, fs=4.0, **over):
        kw = dict(
            tdi={"A": np.zeros(n)},
            sample_rate=fs,
            channels=("A",),
            tdi_generation="1.5",
            observable="fractional_frequency",
            noise=self.White(0.7, fs),
        )
        kw.update(over)
        return Residuals(**kw)

    def test_none_without_a_noise_model(self, observed):
        assert observed.noise_variance() is None

    @pytest.mark.parametrize("n", [2048, 2049])
    def test_independent_of_the_parity_of_n(self, n):
        # the naive sum(psd[1:])*df lands on sigma**2 for even n and
        # sigma**2 (1-1/n) for odd n; the weighted form agrees with itself
        var = self._residual(n).noise_variance()
        assert var == pytest.approx(0.49 * (1 - 1 / n), rel=1e-12)

    def test_matches_the_empirical_variance(self):
        n, fs, sigma = 1 << 14, 4.0, 0.7
        r = self._residual(n, fs=fs)
        rng = np.random.default_rng(0)
        emp = float(np.mean([np.var(rng.normal(0, sigma, n)) for _ in range(40)]))
        assert r.noise_variance() == pytest.approx(emp, rel=0.02)

    def test_contract_error_matches_noise_psd(self, observed):
        obs = replace(observed, noise=object())
        with pytest.raises(TypeError, match="does not expose"):
            obs.noise_variance()


class TestPsdGridAndAliases:
    """Pin the actual frequency grid and the derived quantities numerically."""

    class RampPSD:
        """Frequency-dependent, so a wrong grid cannot pass unnoticed."""

        def psd(self, f, channel=None):
            return 1.0 + np.asarray(f)

    @pytest.mark.parametrize(("n", "fs"), [(64, 0.5), (65, 2.0), (1024, 0.1)])
    def test_psd_is_evaluated_on_the_rfft_grid(self, n, fs):
        r = Residuals(
            tdi={"A": np.zeros(n)},
            sample_rate=fs,
            channels=("A",),
            tdi_generation="1.5",
            observable="strain",
            noise=self.RampPSD(),
        )
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        psd = r.noise_psd()
        assert psd[0] == np.inf
        np.testing.assert_allclose(psd[1:], 1.0 + freqs[1:], rtol=1e-12)

    def test_derived_quantities_are_numerically_right(self, rng):
        obs = make_observed(rng, n_samples=64, sample_rate=0.5)
        assert obs.nyquist_frequency == pytest.approx(0.25)  # fs / 2
        assert obs.sample_interval == pytest.approx(2.0)  # 1 / fs
        assert obs.observation_time == pytest.approx(128.0)  # n / fs
        assert obs.frequency_resolution == pytest.approx(1 / 128.0)  # 1 / Tobs


class TestNoisePsdSanity:
    """A noise segment leaves tdi untouched, so the Wheel's finiteness guard
    never sees a bad fit -- the damage travels through the noise object. These
    pin the only place it can be caught."""

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

        r = make_observed(rng, noise=BrokenNoise())
        with pytest.raises(ValueError, match="non-finite or non-positive"):
            r.noise_psd()

    def test_the_message_blames_the_noise_model_by_name(self, rng):
        class WhittleFit:
            def psd(self, freqs, channel=None):
                return np.full(freqs.shape, np.nan)

        r = make_observed(rng, noise=WhittleFit())
        with pytest.raises(ValueError, match=r"noise model WhittleFit\.psd"):
            r.noise_psd()

    def test_a_finite_positive_psd_passes_through_untouched(self, rng):
        """The guard must not reject the DC bin it sets to +inf itself."""

        class GoodNoise:
            def psd(self, freqs, channel=None):
                return np.full(freqs.shape, 3e-41)

        psd = make_observed(rng, noise=GoodNoise()).noise_psd()
        assert psd[0] == np.inf  # DC carries zero weight by construction
        assert np.all(psd[1:] == 3e-41)
