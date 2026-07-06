"""Wheel: residual passing, atomic registration, and boundary validation."""

from dataclasses import replace

import numpy as np
import pytest

from conftest import make_observed
from turntable import Wheel
from turntable.testing import EchoSegment


class ConstSegment:
    """Subtracts a constant offset on every channel each step, with the
    add-back done correctly; keeps its own step counter."""

    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.steps = 0
        self._applied = 0.0  # what this segment currently has subtracted

    def start(self, residual):
        return self._resubtract(residual)

    def step(self, residual):
        self.steps += 1
        return self._resubtract(residual)

    def _resubtract(self, residual):
        # add back our old model, subtract the (same) new one: net constant
        new = {
            ch: arr + self._applied - self.value for ch, arr in residual.tdi.items()
        }
        self._applied = self.value
        return replace(residual, tdi=new)


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


class TestResidualPassing:
    def test_running_residual_subtracts_every_model(self, rng):
        obs = make_observed(rng)
        wheel = Wheel(obs)
        wheel.add(ConstSegment("a", 1.0))
        wheel.add(ConstSegment("b", 10.0))
        wheel.add(ConstSegment("c", 100.0))
        wheel.run(3)
        full = wheel.residual()
        for ch in obs.channels:
            np.testing.assert_allclose(full.tdi[ch], obs.tdi[ch] - 111.0)

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
        assert seg.steps == 4  # state lives on the object, not the Wheel

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
        wheel.add(EchoSegment(name="ok"))
        base = wheel.residual()
        with pytest.raises(TypeError, match="bad.start must return a Residuals"):
            wheel.add(BadStart("bad", 0.0))
        assert wheel.residual() is base  # running residual unchanged
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
                # illegally change a fixed run setting
                return replace(residual, tdi_generation="9.9")

        wheel = Wheel(observed)
        wheel.add(Cheat("cheat", 0.0))
        with pytest.raises(ValueError, match="changed the run setting 'tdi_generation'"):
            wheel.run(1)

    def test_changed_channels_rejected(self, rng):
        obs = make_observed(rng)

        class Dropper:
            name = "drop"

            def start(self, residual):
                return residual

            def step(self, residual):
                # return a Residuals with a different channel set
                return replace(
                    residual,
                    tdi={"A": residual.tdi["A"]},
                    channels=("A",),
                )

        wheel = Wheel(obs)
        wheel.add(Dropper())
        with pytest.raises(ValueError, match="changed the run setting 'channels'"):
            wheel.run(1)

    def test_bad_tdi_shape_raises_via_residuals(self, observed):
        class Drifter:
            name = "drift"

            def start(self, residual):
                return residual

            def step(self, residual):
                # Residuals.__post_init__ rejects the wrong length itself
                return replace(
                    residual, tdi={ch: np.zeros(2) for ch in residual.channels}
                )

        wheel = Wheel(observed)
        wheel.add(Drifter())
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
