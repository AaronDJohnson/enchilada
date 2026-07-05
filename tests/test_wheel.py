"""Wheel: Gibbs bookkeeping, atomic registration, and boundary validation."""

import numpy as np
import pytest

from conftest import make_observed
from turntable import Wheel
from turntable.testing import EchoSegment


class ConstSegment:
    """Contributes a constant offset on every channel; keeps its own counter."""

    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.steps = 0
        self._zeros = None

    def start(self, observed):
        self._zeros = {
            ch: np.zeros_like(observed.tdi[ch]) for ch in observed.channels
        }
        return self._contribution()

    def step(self, residual):
        self.steps += 1
        return self._contribution()

    def _contribution(self):
        return {ch: np.full_like(a, self.value) for ch, a in self._zeros.items()}


class FlatPSD:
    def __init__(self, level=1.0):
        self.level = level

    def psd(self, freqs, channel=None):
        return np.full_like(freqs, self.level)


class NoiseSeg(ConstSegment):
    """Noise segment whose model level tracks its step count."""

    def __init__(self, name):
        super().__init__(name, 0.0)

    def noise_model(self):
        return FlatPSD(level=float(self.steps))


class TestGibbsBookkeeping:
    def test_residual_excludes_all_other_contributions(self, rng):
        obs = make_observed(rng)
        wheel = Wheel(obs)
        wheel.add(ConstSegment("a", 1.0))
        wheel.add(ConstSegment("b", 10.0))
        wheel.add(ConstSegment("c", 100.0))
        wheel.run(3)
        for name, others in [("a", 110.0), ("b", 101.0), ("c", 11.0)]:
            seen = wheel.residual(exclude=name)
            for ch in obs.channels:
                np.testing.assert_allclose(seen.tdi[ch], obs.tdi[ch] - others)

    def test_full_residual_subtracts_everything(self, rng):
        obs = make_observed(rng)
        wheel = Wheel(obs)
        wheel.add(ConstSegment("a", 1.0))
        wheel.add(ConstSegment("b", 10.0))
        full = wheel.residual()
        for ch in obs.channels:
            np.testing.assert_allclose(full.tdi[ch], obs.tdi[ch] - 11.0)

    def test_observed_data_never_mutated(self, rng):
        obs = make_observed(rng)
        snapshot = {ch: arr.copy() for ch, arr in obs.tdi.items()}

        class Mutator(ConstSegment):
            def start(self, observed):
                for arr in observed.tdi.values():
                    arr[:] = -999.0
                return super().start(observed)

        wheel = Wheel(obs)
        wheel.add(Mutator("mut", 0.0))
        wheel.run(1)
        for ch, arr in snapshot.items():
            np.testing.assert_array_equal(obs.tdi[ch], arr)

    def test_segments_keep_their_own_state(self, observed):
        seg = ConstSegment("a", 1.0)
        wheel = Wheel(observed)
        wheel.add(seg)
        wheel.run(4)
        assert seg.steps == 4  # state lives on the object, not the Wheel

    def test_zero_segments_is_a_noop(self, observed):
        Wheel(observed).run(5)

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

    def test_second_noise_segment_leaves_wheel_untouched(self, observed):
        wheel = Wheel(observed)
        wheel.add(NoiseSeg("n1"))
        with pytest.raises(ValueError, match="at most one noise segment"):
            wheel.add(NoiseSeg("n2"))
        with pytest.raises(ValueError, match="unknown segment"):
            wheel.residual(exclude="n2")  # rejected segment is not registered
        wheel.residual(exclude="n1")  # the accepted one is
        wheel.run(1)  # still healthy

    def test_bad_noise_model_leaves_wheel_untouched(self, observed):
        class BadNoise(ConstSegment):
            def noise_model(self):
                return object()

        wheel = Wheel(observed)
        with pytest.raises(TypeError, match="neither psd"):
            wheel.add(BadNoise("bad", 0.0))
        with pytest.raises(ValueError, match="unknown segment"):
            wheel.residual(exclude="bad")
        assert wheel.residual().noise is None  # its model was not threaded
        wheel.run(1)

    def test_bad_contribution_leaves_wheel_untouched(self, observed):
        class BadStart(ConstSegment):
            def start(self, observed):
                super().start(observed)
                return {ch: np.zeros(3) for ch in self._zeros}

        wheel = Wheel(observed)
        with pytest.raises(ValueError, match="has shape"):
            wheel.add(BadStart("bad", 0.0))
        with pytest.raises(ValueError, match="unknown segment"):
            wheel.residual(exclude="bad")
        wheel.run(1)

    def test_malformed_start_return(self, observed):
        class Bad(ConstSegment):
            def start(self, observed):
                return "not a contribution"

        with pytest.raises(TypeError, match="bad.start must return"):
            Wheel(observed).add(Bad("bad", 0.0))


class TestNoiseThreading:
    def test_noise_model_threaded_and_refreshed(self, observed):
        wheel = Wheel(observed)
        wheel.add(NoiseSeg("noise"))
        assert wheel.residual().noise.level == 0.0
        wheel.run(2)
        assert wheel.residual().noise.level == 2.0  # refreshed after each step

    def test_start_sees_threaded_noise(self, observed):
        seen = {}

        class Recorder(ConstSegment):
            def start(self, observed):
                seen["noise"] = observed.noise
                return super().start(observed)

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


class TestRunPathValidation:
    def test_contribution_drift_raises_immediately(self, observed):
        class Drifter(ConstSegment):
            drift = False

            def step(self, residual):
                super().step(residual)
                if self.drift:
                    return {ch: np.zeros(2) for ch in self._zeros}
                return self._contribution()

        d = Drifter("d", 0.0)
        wheel = Wheel(observed)
        wheel.add(d)
        wheel.run(1)
        d.drift = True
        with pytest.raises(ValueError, match="has shape"):
            wheel.run(1)

    def test_missing_channel_named(self, observed):
        class Dropper(ConstSegment):
            def step(self, residual):
                out = super().step(residual) or self._contribution()
                del out["E"]
                return out

        wheel = Wheel(observed)
        wheel.add(Dropper("drop", 0.0))
        with pytest.raises(ValueError, match="missing channels.*'E'"):
            wheel.run(1)

    def test_malformed_step_return_named(self, observed):
        class Bad(ConstSegment):
            def step(self, residual):
                return ([], {})

        wheel = Wheel(observed)
        wheel.add(Bad("bad", 0.0))
        with pytest.raises(TypeError, match="bad.step must return"):
            wheel.run(1)


class TestOnSweep:
    def test_callback_called_per_sweep_with_wheel(self, observed):
        calls = []
        wheel = Wheel(observed)
        wheel.add(ConstSegment("a", 1.0))
        wheel.run(3, on_sweep=lambda i, w: calls.append((i, w is wheel)))
        assert calls == [(0, True), (1, True), (2, True)]

    def test_unknown_exclude_rejected(self, observed):
        wheel = Wheel(observed)
        with pytest.raises(ValueError, match="unknown segment"):
            wheel.residual(exclude="ghost")
