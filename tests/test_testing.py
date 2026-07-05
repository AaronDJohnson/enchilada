"""turntable.testing: EchoSegment and the check_segment conformance helper."""

import numpy as np
import pytest

from conftest import make_observed
from turntable.testing import EchoSegment, check_segment


class TestCheckSegment:
    def test_conforming_segment_passes(self, observed):
        check_segment(EchoSegment(name="echo"), observed)

    def test_conforming_in_frequency_domain(self, rng):
        n = 64
        obs = make_observed(
            rng,
            domain="frequency",
            tdi={
                ch: rng.standard_normal(n // 2 + 1) + 0j
                for ch in ("A", "E", "T")
            },
        )
        check_segment(EchoSegment(name="echo"), obs)

    def test_mutating_step_caught(self, observed):
        class Mutator(EchoSegment):
            def step(self, residual):
                residual.tdi["A"][:] = 0.0
                return super().step(residual)

        with pytest.raises(ValueError, match="mutated residual"):
            check_segment(Mutator(name="mut"), observed)

    def test_malformed_step_caught(self, observed):
        class Bad(EchoSegment):
            def step(self, residual):
                return ([], {})  # old-style pair, not a contribution

        with pytest.raises(TypeError, match="must return"):
            check_segment(Bad(name="bad"), observed)

    def test_bad_contribution_shape_caught(self, observed):
        class Bad(EchoSegment):
            def step(self, residual):
                super().step(residual)
                return {ch: np.zeros(1) for ch in self._zeros}

        with pytest.raises(ValueError, match="has shape"):
            check_segment(Bad(name="bad"), observed)

    def test_nan_data_does_not_false_positive_mutation(self, rng):
        obs = make_observed(rng)
        obs.tdi["A"][10:20] = np.nan  # gap-filled data is legitimate
        check_segment(EchoSegment(name="echo"), obs)

    def test_conforming_noise_segment_passes(self, observed):
        class FlatPSD:
            def psd(self, freqs, channel=None):
                return np.full_like(freqs, 1.0)

        class NoiseSeg(EchoSegment):
            def noise_model(self):
                return FlatPSD()

        check_segment(NoiseSeg(name="noise"), observed)

    def test_noise_model_violating_contract_caught(self, observed):
        class BadNoise(EchoSegment):
            def noise_model(self):
                return object()

        with pytest.raises(TypeError, match="neither psd"):
            check_segment(BadNoise(name="bad"), observed)


class TestEchoSegment:
    def test_keeps_its_own_step_counter(self, observed):
        from turntable import Wheel

        echo = EchoSegment(name="echo")
        wheel = Wheel(observed)
        wheel.add(echo)
        wheel.run(3)
        assert echo.steps == 3
