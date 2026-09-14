"""L1Data validates channel samples, metadata, and Fourier conventions."""

from dataclasses import replace

import numpy as np
import pytest

from conftest import make_observed
from enchilada import BlockResult, DataCovariance, L1Data


class TestWDMData:
    def test_grid_supplies_time_sample_count_and_preserves_campaign_metadata(self):
        from enchilada.domains import WDMGrid

        grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=8)
        data = L1Data(
            channel_data={"A": np.zeros(grid.array_shape)},
            sample_rate_hz=2.0,
            channel_names=("A",),
            tdi_generation="2.0",
            physical_observable="fractional_frequency",
            data_domain="wdm",
            start_time_gps=100.0,
            wdm_grid=grid,
        )
        assert data.num_time_samples == 32
        assert data.Tobs == 16.0
        assert data.dt == 0.5
        assert data.start_time_gps == 100.0
        assert data.wdm_grid is grid
        assert replace(data, num_time_samples=32).num_time_samples == 32
        with pytest.raises(ValueError, match="num_time_samples"):
            replace(data, num_time_samples=64)

    def test_wdm_requires_grid_metadata(self, observed):
        with pytest.raises(ValueError, match="wdm_grid"):
            replace(observed, data_domain="wdm")

    def test_other_domains_reject_wdm_grid_metadata(self, observed):
        from enchilada.domains import WDMGrid

        grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=16)
        for data in (observed, observed.to_frequency()):
            with pytest.raises(ValueError, match="wdm_grid"):
                replace(data, wdm_grid=grid)

    def test_wdm_grid_must_have_the_public_type(self, observed):
        with pytest.raises(TypeError, match="WDMGrid"):
            replace(observed, data_domain="wdm", wdm_grid=(4, 16))

    @pytest.mark.parametrize("invalid", ["shape", "complex", "integer", "inactive"])
    def test_wdm_validates_each_channel(self, observed, invalid):
        from enchilada.domains import WDMGrid

        grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=16)
        channels = {ch: np.zeros(grid.array_shape) for ch in observed.channel_names}
        error = ValueError
        if invalid == "shape":
            channels["E"] = np.zeros((5, 8))
        elif invalid == "complex":
            channels["E"] = channels["E"].astype(complex)
            error = TypeError
        elif invalid == "integer":
            channels["E"] = channels["E"].astype(int)
            error = TypeError
        else:
            channels["E"][0, 1] = 1.0
        with pytest.raises(error, match="E"):
            replace(observed, data_domain="wdm", wdm_grid=grid, channel_data=channels)


class TestPostInitValidation:
    def test_valid_construction(self, rng):
        obs = make_observed(rng)
        assert obs.Tobs == obs.num_time_samples / obs.sample_rate_hz

    def test_noise_model_cannot_be_bundled_with_observations(self, observed):
        with pytest.raises(TypeError, match="noise"):
            replace(observed, noise=object())

    def test_physical_observable_is_required(self, rng):
        with pytest.raises(TypeError, match="physical_observable"):
            L1Data(
                channel_data={"A": np.zeros(8)},
                sample_rate_hz=1.0,
                num_time_samples=8,
                channel_names=("A",),
                tdi_generation="2.0",
                start_time_gps=0.0,
            )

    def test_start_time_gps_is_optional_defaults_to_zero(self, rng):
        obs = L1Data(
            channel_data={"A": np.zeros(8)},
            sample_rate_hz=1.0,
            num_time_samples=8,
            channel_names=("A",),
            tdi_generation="2.0",
            physical_observable="fractional_frequency",
        )  # no start_time_gps supplied
        assert obs.start_time_gps == 0.0
        assert obs.t0 == 0.0

    def test_empty_physical_observable_rejected(self, rng):
        with pytest.raises(ValueError, match="fractional_frequency"):
            make_observed(rng, physical_observable="")

    def test_channel_data_keys_must_match_channel_names(self, rng):
        with pytest.raises(ValueError, match="missing.*'T'.*unexpected.*'X'"):
            make_observed(
                rng,
                channel_data={"A": np.zeros(64), "E": np.zeros(64), "X": np.zeros(64)},
            )

    def test_array_length_must_match_num_time_samples(self, rng):
        with pytest.raises(ValueError, match="length 99, expected 64"):
            make_observed(
                rng,
                channel_data={"A": np.zeros(64), "E": np.zeros(99), "T": np.zeros(64)},
            )

    def test_complex_time_domain_rejected(self, rng):
        with pytest.raises(TypeError, match="data_domain='frequency'"):
            make_observed(
                rng, channel_data={ch: np.zeros(64, complex) for ch in ("A", "E", "T")}
            )

    def test_unknown_data_domain_rejected(self, rng):
        with pytest.raises(ValueError, match="data_domain"):
            make_observed(rng, data_domain="fourier")

    @pytest.mark.parametrize("sample_rate_hz", [0.0, -1.0, float("nan")])
    def test_bad_sample_rate_hz_rejected(self, rng, sample_rate_hz):
        with pytest.raises(ValueError, match="sample_rate_hz"):
            make_observed(rng, sample_rate_hz=sample_rate_hz)

    def test_frequency_domain_arrays_live_on_rfft_grid(self, rng):
        n = 64
        good = make_observed(
            rng,
            data_domain="frequency",
            channel_data={ch: np.zeros(n // 2 + 1, complex) for ch in ("A", "E", "T")},
        )
        assert good.data_domain == "frequency"
        with pytest.raises(ValueError, match="rfft grid"):
            make_observed(
                rng,
                data_domain="frequency",
                channel_data={ch: np.zeros(n, complex) for ch in ("A", "E", "T")},
            )

    def test_replace_revalidates(self, observed):
        with pytest.raises(ValueError, match="length"):
            replace(
                observed,
                channel_data={ch: np.zeros(3) for ch in observed.channel_names},
            )


class TestAliases:
    def test_long_and_short_names_agree(self, observed):
        for long, short in L1Data.aliases().items():
            assert getattr(observed, long) == getattr(observed, short)

    def test_typo_catcher_suggests(self, observed):
        with pytest.raises(AttributeError, match="did you mean 'Tobs'"):
            _ = observed.T_obs


class TestOrbitSpanCheck:
    class StubOrbit:
        time_range_gps = (0.0, 100.0)

    def test_orbit_ephemeris_covering_data_accepted(self, rng):
        obs = make_observed(rng, orbit_ephemeris=self.StubOrbit(), sample_rate_hz=1.0)
        assert obs.orbit_ephemeris is not None  # 64 s of data inside [0, 100]

    @pytest.mark.parametrize("start_time_gps", [-1.0, 50.0])
    def test_orbit_ephemeris_not_covering_data_rejected(self, rng, start_time_gps):
        with pytest.raises(ValueError, match="outside the tabulated ephemeris"):
            make_observed(
                rng,
                orbit_ephemeris=self.StubOrbit(),
                sample_rate_hz=1.0,
                start_time_gps=start_time_gps,
            )

    def test_orbit_ephemeris_tabulated_on_the_data_grid_accepted(self, rng):
        # 64 samples at 1 Hz starting at GPS time 0 -> last sample at t = 63 s. An orbit
        # tabulated on exactly that grid covers every time a block can ask
        # for, so it must be accepted (it spans [0, 63], not [0, Tobs=64]).
        class GridOrbit:
            time_range_gps = (0.0, 63.0)

        obs = make_observed(
            rng, orbit_ephemeris=GridOrbit(), sample_rate_hz=1.0, start_time_gps=0.0
        )
        assert obs.orbit_ephemeris is not None

    # (num_time_samples, sample_rate_hz) pairs where (n-1)*dt and (n-1)/fs are NOT
    # bit-identical -- i.e. exactly where computing the bound by division
    # spuriously rejects a grid-tabulated orbit -- plus an exactly-representable
    # non-unit rate so the dt scaling itself is pinned.
    @pytest.mark.parametrize(
        ("num_time_samples", "sample_rate_hz"),
        [(32, 3.0), (32, 6.0), (32, 7.0), (128, 3.0), (64, 0.5), (64, 0.1)],
    )
    def test_data_grid_orbit_ephemeris_accepted_at_any_rate(
        self, rng, num_time_samples, sample_rate_hz
    ):
        """An orbit tabulated the documented way is accepted at every rate.

        The grid idiom is `start_time_gps + arange(n) * dt` (NumericOrbit.from_hdf5), so
        the check must compute its bound the same way; deriving it as
        (n-1)/sample_rate_hz differs by an ulp at these rates and rejects.
        """

        class GridOrbit:
            def __init__(self, t_end):
                self.time_range_gps = (0.0, t_end)

        dt = 1.0 / sample_rate_hz
        last = 0.0 + (num_time_samples - 1) * dt
        obs = make_observed(
            rng,
            num_time_samples=num_time_samples,
            orbit_ephemeris=GridOrbit(last),
            sample_rate_hz=sample_rate_hz,
            start_time_gps=0.0,
        )
        assert obs.orbit_ephemeris is not None
        # a full sample interval short is still rejected, at every rate
        with pytest.raises(ValueError, match="outside the tabulated ephemeris"):
            make_observed(
                rng,
                num_time_samples=num_time_samples,
                orbit_ephemeris=GridOrbit(last - dt),
                sample_rate_hz=sample_rate_hz,
                start_time_gps=0.0,
            )

    def test_orbit_ephemeris_ending_before_the_last_sample_rejected(self, rng):
        # one hair short of the last sample (63 s) must still be caught
        class ShortOrbit:
            time_range_gps = (0.0, 62.9)

        with pytest.raises(ValueError, match="outside the tabulated ephemeris"):
            make_observed(
                rng,
                orbit_ephemeris=ShortOrbit(),
                sample_rate_hz=1.0,
                start_time_gps=0.0,
            )

    def test_orbit_ephemeris_without_time_range_gps_skips_check(self, rng):
        obs = make_observed(rng, orbit_ephemeris=object(), start_time_gps=1e9)
        assert obs.orbit_ephemeris is not None


class TestRemainingValidationBranches:
    """Negative cases for the scalar/channel_data guards not covered above."""

    def test_num_time_samples_must_be_a_positive_int(self, observed):
        # 0 is the "derive it" sentinel, so the invalid values are negatives
        # and non-integers
        with pytest.raises(
            ValueError, match="num_time_samples must be a positive integer"
        ):
            replace(observed, num_time_samples=-4)
        with pytest.raises(
            ValueError, match="num_time_samples must be a positive integer"
        ):
            replace(observed, num_time_samples=8.0)  # float is not an int

    def test_start_time_gps_must_be_finite(self, observed):
        with pytest.raises(ValueError, match="start_time_gps must be finite"):
            replace(observed, start_time_gps=float("nan"))

    def test_tdi_generation_must_be_a_non_empty_string(self, observed):
        with pytest.raises(ValueError, match="tdi_generation must be a non-empty"):
            replace(observed, tdi_generation="")

    def test_channel_data_must_be_a_dict(self, observed):
        with pytest.raises(TypeError, match="channel_data must be a dict"):
            replace(observed, channel_data=[1.0, 2.0])

    def test_channel_names_must_be_non_empty(self, observed):
        with pytest.raises(ValueError, match="channel_names must be a non-empty"):
            replace(observed, channel_names=(), channel_data={})

    def test_channel_names_must_not_contain_duplicates(self, rng):
        with pytest.raises(ValueError, match="channel_names contains duplicates"):
            make_observed(
                rng, channel_names=("A", "A"), channel_data={"A": np.zeros(64)}
            )

    def test_channel_data_values_must_be_1d_arrays(self, observed):
        with pytest.raises(TypeError, match="must be a 1-D numpy array"):
            replace(
                observed, channel_data={ch: [0.0] * 64 for ch in observed.channel_names}
            )

    def test_private_attribute_miss_raises_plain_attributeerror(self, observed):
        with pytest.raises(AttributeError):
            _ = observed._not_a_field

    def test_unknown_attribute_points_at_the_alias_table(self, observed):
        with pytest.raises(AttributeError, match=r"aliases\(\)"):
            _ = observed.wobble


class TestNSamplesDerivation:
    """num_time_samples is read off the data where that is exact, required where not."""

    def _kwargs(self, **over):
        base = dict(
            sample_rate_hz=0.1,
            channel_names=("A", "E"),
            tdi_generation="2.0",
            physical_observable="fractional_frequency",
        )
        base.update(over)
        return base

    def test_time_domain_derives_from_the_arrays(self):
        r = L1Data(
            channel_data={ch: np.zeros(1024) for ch in ("A", "E")}, **self._kwargs()
        )
        assert r.num_time_samples == 1024
        assert r.Tobs == 1024 / 0.1
        assert r.df == 1.0 / r.Tobs

    def test_explicit_value_still_honoured_and_checked(self):
        kw = self._kwargs()
        channel_data = {ch: np.zeros(1024) for ch in ("A", "E")}
        assert (
            L1Data(
                channel_data=channel_data, num_time_samples=1024, **kw
            ).num_time_samples
            == 1024
        )
        with pytest.raises(ValueError, match="expected 512"):
            L1Data(channel_data=channel_data, num_time_samples=512, **kw)

    def test_frequency_domain_requires_it_and_says_why(self):
        # 513 bins are consistent with n=1024 and n=1025 -- the parity is lost,
        # so enchilada asks instead of guessing
        with pytest.raises(ValueError, match="does not determine it") as exc:
            L1Data(
                channel_data={ch: np.zeros(513, complex) for ch in ("A", "E")},
                data_domain="frequency",
                **self._kwargs(),
            )
        msg = str(exc.value)
        assert "num_time_samples=1024" in msg and "=1025" in msg

    @pytest.mark.parametrize("n", [1024, 1025])
    def test_frequency_domain_accepts_either_parity_when_stated(self, n):
        r = L1Data(
            channel_data={ch: np.zeros(n // 2 + 1, complex) for ch in ("A", "E")},
            data_domain="frequency",
            num_time_samples=n,
            **self._kwargs(),
        )
        assert r.num_time_samples == n
        assert r.Tobs == n / 0.1  # the two parities really do differ

    def test_derived_value_survives_replace(self):
        r = L1Data(
            channel_data={ch: np.zeros(64) for ch in ("A", "E")}, **self._kwargs()
        )
        r2 = replace(r, channel_data={ch: np.ones(64) for ch in ("A", "E")})
        assert r2.num_time_samples == 64

    def test_derivation_still_validates_every_channel(self):
        # derived from the first channel, but a ragged second one is caught
        with pytest.raises(ValueError, match="has length 60, expected 64"):
            L1Data(
                channel_data={"A": np.zeros(64), "E": np.zeros(60)}, **self._kwargs()
            )


class TestDomainTransforms:
    """to_frequency/to_time carry num_time_samples, so the round trip is exact."""

    @pytest.mark.parametrize(
        "num_time_samples,index,endpoint",
        [(1, 0, "DC"), (4, 0, "DC"), (4, -1, "Nyquist"), (5, 0, "DC")],
    )
    def test_imaginary_rfft_endpoints_are_rejected(
        self, num_time_samples, index, endpoint
    ):
        spectrum = np.zeros(num_time_samples // 2 + 1, dtype=complex)
        spectrum[index] = 1 + 2j
        with pytest.raises(ValueError, match=rf"{endpoint}.*real"):
            L1Data(
                channel_data={"A": spectrum},
                sample_rate_hz=0.2,
                num_time_samples=num_time_samples,
                channel_names=("A",),
                tdi_generation="2.0",
                physical_observable="strain",
                data_domain="frequency",
            )
        assert spectrum[index] == 1 + 2j  # validation never projects the input

    @pytest.mark.parametrize("num_time_samples", [1, 4, 5])
    def test_valid_frequency_inputs_round_trip_without_losing_components(
        self, num_time_samples
    ):
        spectrum = np.arange(num_time_samples // 2 + 1, dtype=complex) + 1
        if num_time_samples > 1:
            spectrum[1:] += 2j
        if num_time_samples % 2 == 0:
            spectrum[-1] = spectrum[-1].real
        observed = L1Data(
            channel_data={"A": spectrum},
            sample_rate_hz=0.2,
            num_time_samples=num_time_samples,
            channel_names=("A",),
            tdi_generation="2.0",
            physical_observable="strain",
            data_domain="frequency",
        )
        np.testing.assert_allclose(
            observed.to_time().to_frequency().channel_data["A"], spectrum
        )

    def _time_residual(self, n, fs=0.2, **over):
        rng = np.random.default_rng(0)
        kw = dict(
            channel_data={ch: rng.standard_normal(n) for ch in ("A", "E")},
            sample_rate_hz=fs,
            channel_names=("A", "E"),
            tdi_generation="1.5",
            physical_observable="fractional_frequency",
        )
        kw.update(over)
        return L1Data(**kw)

    @pytest.mark.parametrize("n", [1024, 1025])  # both parities
    def test_round_trip_is_exact(self, n):
        t = self._time_residual(n)
        f = t.to_frequency()
        assert f.data_domain == "frequency"
        assert f.channel_data["A"].size == n // 2 + 1
        back = f.to_time()
        assert back.data_domain == "time"
        for ch in t.channel_names:
            np.testing.assert_allclose(
                back.channel_data[ch], t.channel_data[ch], atol=1e-12
            )

    @pytest.mark.parametrize("n", [1024, 1025])
    def test_num_time_samples_is_carried_not_restated(self, n):
        # never passed by hand: derived from the arrays, then carried across
        t = self._time_residual(n)
        f = t.to_frequency()
        assert t.num_time_samples == f.num_time_samples == n
        assert f.Tobs == t.Tobs and f.df == t.df and f.dt == t.dt

    @pytest.mark.parametrize("domain", ["time", "frequency"])
    def test_same_domain_conversion_returns_independent_arrays(self, domain):
        source = self._time_residual(64)
        if domain == "frequency":
            source = source.to_frequency()
        converted = getattr(source, f"to_{domain}")()
        for name in source.channel_names:
            original = source.channel_data[name].copy()
            np.testing.assert_array_equal(converted.channel_data[name], original)
            converted.channel_data[name][:] = 0
            np.testing.assert_array_equal(source.channel_data[name], original)

    @pytest.mark.parametrize("source_domain", ["time", "frequency"])
    @pytest.mark.parametrize("target_domain", ["time", "frequency"])
    @pytest.mark.parametrize("invalid", ["length", "nonfinite"])
    def test_convenience_conversions_revalidate_mutated_channels(
        self, source_domain, target_domain, invalid
    ):
        source = self._time_residual(5)
        if source_domain == "frequency":
            source = source.to_frequency()
        if invalid == "length":
            # Mutate every channel so a stacked FFT could silently pad/truncate.
            for name in source.channel_names:
                source.channel_data[name] = source.channel_data[name][:-1]
        else:
            source.channel_data["A"][1] = np.nan
        with pytest.raises(ValueError, match="length|finite"):
            getattr(source, f"to_{target_domain}")()

    @pytest.mark.parametrize("target_domain", ["time", "frequency"])
    @pytest.mark.parametrize("index", [0, -1], ids=["DC", "Nyquist"])
    def test_convenience_conversions_reject_mutated_fourier_endpoints(
        self, target_domain, index
    ):
        source = self._time_residual(4).to_frequency()
        source.channel_data["A"][index] = 1j
        with pytest.raises(ValueError, match="coefficient must be real"):
            getattr(source, f"to_{target_domain}")()

    def test_orbit_ephemeris_is_preserved_by_domain_transforms(self):
        orbit = object()
        t = self._time_residual(64, orbit_ephemeris=orbit)
        assert t.to_frequency().orbit_ephemeris is orbit
        assert t.to_frequency().to_time().orbit_ephemeris is orbit

    def test_transform_convention_matches_coefficient_covariance(self):
        """E[|X(f)|^2] == (Tobs/2) * S(f) for X = dt*rfft(x).

        Compare simulated Fourier power with the covariance constructed from
        its known one-sided PSD to check the normalization end to end.
        """
        fs, n, sigma = 0.2, 1 << 13, 0.7

        class White:
            def psd(self, f, channel=None):
                return np.full_like(f, 2.0 * sigma**2 / fs)

        rng = np.random.default_rng(1)
        acc = None
        trials = 40
        for _ in range(trials):
            t = L1Data(
                channel_data={"A": rng.normal(0.0, sigma, n)},
                sample_rate_hz=fs,
                channel_names=("A",),
                tdi_generation="1.5",
                physical_observable="fractional_frequency",
            )
            power = np.abs(t.to_frequency().channel_data["A"]) ** 2
            acc = power if acc is None else acc + power
        measured = acc / trials
        noise_covariance = DataCovariance.from_psd(t, White())
        predicted = noise_covariance.covariance_matrix[:, 0, 0].real
        interior = slice(10, -10)
        ratio = float(np.mean(measured[interior] / predicted[interior]))
        assert ratio == pytest.approx(1.0, abs=0.05)


class TestDtypeAndTypeContract:
    """dtype and container types are part of the validated contract."""

    def test_integer_channel_data_rejected_at_construction(self, rng):
        # would otherwise fail deep inside the Wheel's ledger arithmetic
        with pytest.raises(TypeError, match="must be floating or complex"):
            make_observed(
                rng, channel_data={ch: np.arange(64) for ch in ("A", "E", "T")}
            )

    def test_object_dtype_rejected(self, rng):
        with pytest.raises(TypeError, match="must be floating or complex"):
            make_observed(
                rng,
                channel_data={ch: np.zeros(64, dtype=object) for ch in ("A", "E", "T")},
            )

    def test_float32_is_allowed(self, rng):
        obs = make_observed(
            rng, channel_data={ch: np.zeros(64, np.float32) for ch in ("A", "E", "T")}
        )
        assert obs.channel_data["A"].dtype == np.float32

    def test_channel_names_of_a_wrong_container_type_rejected(self, rng):
        # a set has no order, so it cannot define the channel sequence
        with pytest.raises(TypeError, match="channel_names must be a tuple"):
            make_observed(rng, channel_names={"A", "E", "T"})

    def test_channel_names_list_is_coerced_to_tuple(self, rng):
        # a list would otherwise compare unequal to the tuple a block returns,
        # and the Wheel would blame the block for changing a run setting
        obs = make_observed(rng, channel_names=["A", "E", "T"])
        assert obs.channel_names == ("A", "E", "T")
        assert isinstance(obs.channel_names, tuple)

    def test_bool_num_time_samples_rejected(self, observed):
        with pytest.raises(
            ValueError, match="num_time_samples must be a positive integer"
        ):
            replace(observed, num_time_samples=True)

    def test_equality_is_identity_and_hashing_works(self, rng):
        # dataclass eq over a dict of arrays used to raise a raw numpy error
        a, b = make_observed(rng), make_observed(rng)
        assert a != b and a == a
        assert isinstance(hash(a), int)


class TestDerivedQuantities:
    """Pin the derived quantities numerically."""

    def test_derived_quantities_are_numerically_right(self, rng):
        obs = make_observed(rng, num_time_samples=64, sample_rate_hz=0.5)
        assert obs.nyquist_frequency_hz == pytest.approx(0.25)  # fs / 2
        assert obs.sample_interval_s == pytest.approx(2.0)  # 1 / fs
        assert obs.observation_duration_s == pytest.approx(128.0)  # n / fs
        assert obs.frequency_resolution_hz == pytest.approx(1 / 128.0)  # 1 / Tobs


class TestBlockResultFactories:
    """Block-result factories build what a block returns on this data's grid."""

    def test_block_result_carries_the_arrays_given(self, observed):
        tdi = {ch: np.full_like(arr, 2.0) for ch, arr in observed.channel_data.items()}
        t = observed.block_result(tdi)
        assert isinstance(t, BlockResult)
        for ch in observed.channel_names:
            np.testing.assert_array_equal(t.tdi_signal_contribution[ch], 2.0)
        assert t.noise_covariance is None  # a signal block publishes nothing

    def test_block_result_arrays_are_taken_not_copied(self, observed):
        arrs = {ch: np.zeros_like(a) for ch, a in observed.channel_data.items()}
        t = observed.block_result(arrs)
        assert (
            t.tdi_signal_contribution["A"] is arrs["A"]
        )  # the block may reuse its own buffer

    @pytest.mark.parametrize(
        "bad, match",
        [
            ({"A": np.zeros(64)}, "must match the run's channels"),
            (
                {ch: np.zeros(9) for ch in ("A", "E", "T")},
                "length 9, expected 64",
            ),
            (
                {ch: np.zeros(64, complex) for ch in ("A", "E", "T")},
                "complex but data_domain='time'",
            ),
        ],
    )
    def test_off_grid_block_result_is_refused_where_it_is_built(
        self, observed, bad, match
    ):
        with pytest.raises((ValueError, TypeError), match=match):
            observed.block_result(bad)

    def test_zero_block_result_matches_shapes_and_dtypes(self, observed):
        z = observed.zero_block_result()
        assert isinstance(z, BlockResult)
        for ch in observed.channel_names:
            assert (
                z.tdi_signal_contribution[ch].shape == observed.channel_data[ch].shape
            )
            assert (
                z.tdi_signal_contribution[ch].dtype == observed.channel_data[ch].dtype
            )
            np.testing.assert_array_equal(z.tdi_signal_contribution[ch], 0.0)

    def test_arrays_are_fresh_and_independent(self, observed):
        snapshot = observed.channel_data["A"].copy()
        z1, z2 = observed.zero_block_result(), observed.zero_block_result()
        assert z1.tdi_signal_contribution["A"] is not observed.channel_data["A"]
        z1.tdi_signal_contribution["A"][0] = 1.0
        np.testing.assert_array_equal(observed.channel_data["A"], snapshot)  # untouched
        np.testing.assert_array_equal(
            z2.tdi_signal_contribution["A"], 0.0
        )  # its own arrays

    def test_frequency_domain_block_result_is_complex_on_the_rfft_grid(self, rng):
        n = 64
        obs = make_observed(
            rng,
            data_domain="frequency",
            channel_data={
                ch: rng.standard_normal(n // 2 + 1) + 0j for ch in ("A", "E", "T")
            },
        )
        z = obs.zero_block_result()
        assert np.iscomplexobj(
            z.tdi_signal_contribution["A"]
        ) and z.tdi_signal_contribution["A"].shape == (n // 2 + 1,)
        # and a real array on that grid is refused, so `.real` cannot slip through
        with pytest.raises(TypeError, match="real but data_domain='frequency'"):
            obs.block_result({ch: np.zeros(n // 2 + 1) for ch in obs.channel_names})


class TestBlockResultClass:
    """The container itself, for block results built without the factories."""

    def test_with_noise_covariance_returns_a_new_block_result(self, observed):
        z = observed.zero_block_result()
        model = DataCovariance.from_variance(observed, 1.0)
        published = z.with_noise_covariance(model)
        assert published.noise_covariance is model
        assert z.noise_covariance is None  # frozen: the original is unchanged
        assert published.tdi_signal_contribution is z.tdi_signal_contribution

    def test_equality_is_identity_not_elementwise(self, observed):
        a, b = observed.zero_block_result(), observed.zero_block_result()
        assert a == a and a != b  # would raise "truth value ambiguous" otherwise

    @pytest.mark.parametrize(
        "bad, exc, match",
        [
            ("not a dict", TypeError, "must be a dict"),
            ({}, ValueError, "is empty"),
            ({"A": [0.0, 1.0]}, TypeError, "must be a 1-D or 2-D numpy array"),
            ({"A": np.zeros((2, 2, 2))}, TypeError, "must be a 1-D or 2-D numpy array"),
            ({"A": np.zeros(4, int)}, TypeError, "must be floating or complex"),
        ],
    )
    def test_malformed_containers_are_refused(self, bad, exc, match):
        with pytest.raises(exc, match=match):
            BlockResult(tdi_signal_contribution=bad)
