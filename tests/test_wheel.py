"""Wheel: the ledger, data-minus-others handoff, and boundary validation."""

from dataclasses import replace

import numpy as np
import pytest

from conftest import make_observed
from turntable import Residuals, Wheel
from turntable.testing import EchoSegment


class ConstSegment:
    """Subtracts a constant from the residual it is handed -- no add-back.

    The residual it receives is already the data minus every other segment, so
    it just subtracts its (constant) model. Its ledger entry is that constant.
    """

    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.steps = 0

    def start(self, residual):
        return self._subtract(residual)

    def step(self, residual):
        self.steps += 1
        return self._subtract(residual)

    def _subtract(self, residual):
        return replace(
            residual, tdi={ch: arr - self.value for ch, arr in residual.tdi.items()}
        )


class FlatPSD:
    def __init__(self, level=1.0):
        self.level = level

    def psd(self, freqs, channel=None):
        return np.full_like(freqs, self.level)


class NoiseSeg:
    """Noise segment: sets residual.noise, level tracks its step count."""

    def __init__(self, name):
        self.name = name
        self.steps = 0

    def start(self, residual):
        return replace(residual, noise=FlatPSD(0.0))

    def step(self, residual):
        self.steps += 1
        return replace(residual, noise=FlatPSD(float(self.steps)))


class TestLedger:
    def test_full_residual_subtracts_every_model(self, rng):
        obs = make_observed(rng)
        wheel = Wheel(obs)
        wheel.add(ConstSegment("a", 1.0))
        wheel.add(ConstSegment("b", 10.0))
        wheel.add(ConstSegment("c", 100.0))
        wheel.run(3)
        full = wheel.residual()
        for ch in obs.channels:
            np.testing.assert_allclose(full.tdi[ch], obs.tdi[ch] - 111.0)

    def test_exclude_leaves_that_segment_in(self, rng):
        obs = make_observed(rng)
        wheel = Wheel(obs)
        wheel.add(ConstSegment("a", 1.0))
        wheel.add(ConstSegment("b", 10.0))
        wheel.add(ConstSegment("c", 100.0))
        wheel.run(2)
        # residual(exclude=b) = data minus a and c (110), b left in
        seen = wheel.residual(exclude="b")
        for ch in obs.channels:
            np.testing.assert_allclose(seen.tdi[ch], obs.tdi[ch] - 101.0)

    def test_contribution_is_the_derived_model(self, rng):
        obs = make_observed(rng)
        wheel = Wheel(obs)
        wheel.add(ConstSegment("a", 7.0))
        wheel.run(2)
        for ch in obs.channels:
            np.testing.assert_allclose(wheel.contribution("a")[ch], 7.0)

    def test_ledger_derivation_survives_in_place_return(self, rng):
        # a segment that mutates its handed arrays in place AND returns the
        # same object still gets its contribution derived correctly, because
        # the Wheel diffs against a pristine snapshot.
        obs = make_observed(rng)

        class InPlace:
            name = "ip"

            def start(self, residual):
                return residual  # zero contribution

            def step(self, residual):
                for ch in residual.tdi:
                    residual.tdi[ch] -= 3.0  # mutate in place
                return residual  # return the SAME object

        wheel = Wheel(obs)
        wheel.add(InPlace())
        wheel.run(2)
        for ch in obs.channels:
            np.testing.assert_allclose(wheel.contribution("ip")[ch], 3.0)
            np.testing.assert_allclose(wheel.residual().tdi[ch], obs.tdi[ch] - 3.0)

    def test_unknown_exclude_or_contribution_rejected(self, observed):
        wheel = Wheel(observed)
        wheel.add(ConstSegment("a", 1.0))
        with pytest.raises(ValueError, match="unknown segment"):
            wheel.residual(exclude="ghost")
        with pytest.raises(ValueError, match="unknown segment"):
            wheel.contribution("ghost")

    def test_observed_stays_pristine(self, rng):
        obs = make_observed(rng)
        snapshot = {ch: arr.copy() for ch, arr in obs.tdi.items()}

        class Mutator(ConstSegment):
            def step(self, residual):
                residual.tdi[next(iter(residual.tdi))][:] = -999.0  # mutate the copy
                return super().step(residual)

        wheel = Wheel(obs)
        wheel.add(Mutator("mut", 0.0))
        wheel.run(2)
        for ch, arr in snapshot.items():
            np.testing.assert_array_equal(obs.tdi[ch], arr)  # observed untouched

    def test_segments_keep_their_own_state(self, observed):
        seg = ConstSegment("a", 1.0)
        wheel = Wheel(observed)
        wheel.add(seg)
        wheel.run(4)
        assert seg.steps == 4  # sampler state lives on the object, not the Wheel

    def test_zero_segments_is_a_noop(self, observed):
        wheel = Wheel(observed)
        wheel.run(5)
        for ch in observed.channels:
            np.testing.assert_array_equal(wheel.residual().tdi[ch], observed.tdi[ch])

    def test_run_rejects_bad_iteration_counts(self, observed):
        wheel = Wheel(observed)
        with pytest.raises(ValueError, match="n_iterations"):
            wheel.run(-1)
        with pytest.raises(ValueError, match="n_iterations"):
            wheel.run(True)  # bool is not an iteration count

    def test_run_accepts_numpy_integers(self, observed):
        wheel = Wheel(observed)
        wheel.add(ConstSegment("a", 1.0))
        wheel.run(np.int64(2))


class TestNoAddBack:
    def test_each_segment_sees_data_minus_others(self, rng):
        # with two non-trivial segments, each must be handed the data minus the
        # OTHER (never itself); record what each sees at step time.
        obs = make_observed(rng)
        seen = {}

        class Recorder(ConstSegment):
            def step(self, residual):
                seen[self.name] = {ch: residual.tdi[ch].copy() for ch in residual.tdi}
                return super().step(residual)

        wheel = Wheel(obs)
        wheel.add(Recorder("a", 2.0))
        wheel.add(Recorder("b", 5.0))
        wheel.run(1)
        # a sees data minus b (5); b sees data minus a (2)
        for ch in obs.channels:
            np.testing.assert_allclose(seen["a"][ch], obs.tdi[ch] - 5.0)
            np.testing.assert_allclose(seen["b"][ch], obs.tdi[ch] - 2.0)


class TestAtomicRegistration:
    def test_duplicate_name_rejected(self, observed):
        wheel = Wheel(observed)
        wheel.add(EchoSegment(name="x"))
        with pytest.raises(ValueError, match="already registered"):
            wheel.add(EchoSegment(name="x"))

    def test_empty_name_rejected(self, observed):
        with pytest.raises(ValueError, match="non-empty"):
            Wheel(observed).add(EchoSegment(name=""))

    def test_failed_start_leaves_wheel_untouched(self, observed):
        class BadStart(ConstSegment):
            def start(self, residual):
                return "not a residual"

        wheel = Wheel(observed)
        wheel.add(ConstSegment("ok", 1.0))
        with pytest.raises(TypeError, match="bad.start must return a Residuals"):
            wheel.add(BadStart("bad", 0.0))
        with pytest.raises(ValueError, match="unknown segment"):
            wheel.contribution("bad")  # not registered
        wheel.contribution("ok")  # the good one still is
        wheel.run(1)  # still healthy


class TestReturnedResidualValidation:
    def test_non_residual_return_named(self, observed):
        class Bad(ConstSegment):
            def step(self, residual):
                return {"A": np.zeros(1)}  # a dict, not a Residuals

        wheel = Wheel(observed)
        wheel.add(Bad("bad", 0.0))
        with pytest.raises(TypeError, match="bad.step must return a Residuals"):
            wheel.run(1)

    def test_changed_run_setting_rejected(self, observed):
        class Cheat(ConstSegment):
            def step(self, residual):
                return replace(residual, tdi_generation="9.9")

        wheel = Wheel(observed)
        wheel.add(Cheat("cheat", 0.0))
        with pytest.raises(
            ValueError, match="changed the run setting 'tdi_generation'"
        ):
            wheel.run(1)

    def test_bad_tdi_shape_raises_via_residuals(self, observed):
        class Drifter(ConstSegment):
            def step(self, residual):
                return replace(
                    residual, tdi={ch: np.zeros(2) for ch in residual.channels}
                )

        wheel = Wheel(observed)
        wheel.add(Drifter("drift", 0.0))
        with pytest.raises(ValueError, match="length 2, expected"):
            wheel.run(1)

    def test_changed_orbit_rejected(self, observed):
        class Cheat(ConstSegment):
            def step(self, residual):
                return replace(residual, orbit=object())

        wheel = Wheel(observed)
        wheel.add(Cheat("cheat", 0.0))
        with pytest.raises(ValueError, match="changed the orbit"):
            wheel.run(1)


class TestNoiseViaResidual:
    def test_noise_rides_the_residual_and_refreshes(self, observed):
        wheel = Wheel(observed)
        wheel.add(NoiseSeg("noise"))
        assert wheel.residual().noise.level == 0.0  # set in start
        wheel.run(2)
        assert wheel.residual().noise.level == 2.0  # refreshed each step

    def test_noise_segment_has_zero_ledger_entry(self, observed):
        wheel = Wheel(observed)
        wheel.add(NoiseSeg("noise"))
        wheel.run(1)
        for ch in observed.channels:
            np.testing.assert_array_equal(
                wheel.contribution("noise")[ch], np.zeros_like(observed.tdi[ch])
            )

    def test_start_sees_noise_from_earlier_segment(self, observed):
        seen = {}

        class Recorder(ConstSegment):
            def start(self, residual):
                seen["noise"] = residual.noise
                return super().start(residual)

        wheel = Wheel(observed)
        wheel.add(NoiseSeg("noise"))
        wheel.add(Recorder("rec", 0.0))
        assert isinstance(seen["noise"], FlatPSD)

    def test_step_sees_current_noise(self, observed):
        levels = []

        class Recorder(ConstSegment):
            def step(self, residual):
                levels.append(residual.noise.level)
                return super().step(residual)

        wheel = Wheel(observed)
        wheel.add(NoiseSeg("noise"))  # steps first each sweep
        wheel.add(Recorder("rec", 0.0))
        wheel.run(3)
        assert levels == [1.0, 2.0, 3.0]

    def test_fixed_noise_passes_through(self, rng):
        obs = make_observed(rng, noise=FlatPSD(7.0))
        wheel = Wheel(obs)
        wheel.add(EchoSegment("echo"))
        wheel.run(2)
        assert wheel.residual().noise.level == 7.0  # unchanged, threaded along


class TestOnSweep:
    def test_callback_called_per_sweep_with_wheel(self, observed):
        calls = []
        wheel = Wheel(observed)
        wheel.add(ConstSegment("a", 1.0))
        wheel.run(3, on_sweep=lambda i, w: calls.append((i, w is wheel)))
        assert calls == [(0, True), (1, True), (2, True)]


class TestBoundaryGuards:
    """The Wheel refuses returns that would silently corrupt other segments."""

    def test_dropping_the_noise_model_raises(self, rng):
        obs = make_observed(rng, noise=FlatPSD(3.0))

        class Rebuilder(ConstSegment):
            def step(self, residual):
                # rebuilds instead of using replace() -> loses noise silently
                return Residuals(
                    tdi=residual.tdi,
                    sample_rate=residual.sample_rate,
                    channels=residual.channels,
                    tdi_generation=residual.tdi_generation,
                    observable=residual.observable,
                    n_samples=residual.n_samples,
                    epoch=residual.epoch,
                    domain=residual.domain,
                )

        wheel = Wheel(obs)
        wheel.add(Rebuilder("r", 0.0))
        with pytest.raises(ValueError, match="dropped the noise model"):
            wheel.run(1)

    def test_non_finite_return_raises_and_names_the_channel(self, observed):
        class Blowup(ConstSegment):
            def step(self, residual):
                return replace(
                    residual, tdi={ch: arr * np.nan for ch, arr in residual.tdi.items()}
                )

        wheel = Wheel(observed)
        wheel.add(Blowup("boom", 0.0))
        with pytest.raises(ValueError, match="non-finite sample"):
            wheel.run(1)

    def test_missing_step_is_caught_at_registration(self, observed):
        class NoStep:
            name = "nostep"

            def start(self, residual):
                return residual

        with pytest.raises(TypeError, match="does not implement step"):
            Wheel(observed).add(NoStep())

    def test_withdrawing_a_model_warns(self, observed):
        class Fickle(ConstSegment):
            def __init__(self, name, value):
                super().__init__(name, value)
                self.calls = 0

            def step(self, residual):
                self.calls += 1
                if self.calls == 2:
                    return residual  # "nothing changed" -- silently withdraws
                return super().step(residual)

        wheel = Wheel(observed)
        wheel.add(Fickle("f", 1.0))
        with pytest.warns(RuntimeWarning, match="returned the residual unchanged"):
            wheel.run(2)

    def test_a_genuinely_zero_model_does_not_warn(self, observed):
        # EchoSegment contributes zero on every step; that is not a withdrawal
        import warnings as _w

        wheel = Wheel(observed)
        wheel.add(EchoSegment("echo"))
        with _w.catch_warnings():
            _w.simplefilter("error")  # any warning becomes a failure
            wheel.run(3)

    def test_complex_model_against_real_frequency_data_promotes(self, rng):
        # a real-dtype spectrum is a valid Residuals; the ledger must promote
        # rather than raise a raw numpy casting error
        obs = make_observed(
            rng,
            domain="frequency",
            tdi={ch: np.zeros(33) for ch in ("A", "E", "T")},
            n_samples=64,
        )

        class Complexify(ConstSegment):
            def step(self, residual):
                return replace(
                    residual,
                    tdi={ch: arr - (0.5 + 0.5j) for ch, arr in residual.tdi.items()},
                )

        wheel = Wheel(obs)
        wheel.add(Complexify("c", 0.0))
        wheel.run(1)
        assert np.iscomplexobj(wheel.residual().tdi["A"])

    @pytest.mark.parametrize("field", Wheel._INVARIANT)
    def test_every_invariant_run_setting_is_guarded(self, rng, field):
        """Each field in _INVARIANT must actually be enforced, not just listed."""
        obs = make_observed(rng, sample_rate=1.0)
        bad = {
            "channels": ("A",),
            "n_samples": obs.n_samples * 2,
            "sample_rate": 2.0,
            "tdi_generation": "9.9",
            "observable": "phase",
            "domain": "frequency",
            "epoch": 12345.0,
        }[field]

        class Cheat(ConstSegment):
            def step(self, residual):
                kw = {field: bad}
                if field in ("channels", "n_samples", "domain"):
                    # keep tdi self-consistent so Residuals' own checks pass and
                    # the Wheel's invariant check is what fires
                    if field == "channels":
                        kw["tdi"] = {"A": residual.tdi["A"]}
                    elif field == "n_samples":
                        kw["tdi"] = {ch: np.zeros(bad) for ch in residual.channels}
                    else:
                        kw["tdi"] = {
                            ch: np.zeros(residual.n_samples // 2 + 1, complex)
                            for ch in residual.channels
                        }
                return replace(residual, **kw)

        wheel = Wheel(obs)
        wheel.add(Cheat("cheat", 0.0))
        with pytest.raises(ValueError, match=f"changed the run setting {field!r}"):
            wheel.run(1)
