"""turntable.testing: EchoSegment and the check_segment conformance helper."""

from dataclasses import replace

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

    def test_non_residual_return_caught(self, observed):
        class Bad(EchoSegment):
            def step(self, residual):
                return {"A": np.zeros(1)}  # not a Residuals

        with pytest.raises(TypeError, match="must return a Residuals"):
            check_segment(Bad(name="bad"), observed)

    def test_changed_run_setting_caught(self, observed):
        class Cheat(EchoSegment):
            def step(self, residual):
                return replace(residual, observable="strain")

        with pytest.raises(ValueError, match="changed the run setting"):
            check_segment(Cheat(name="cheat"), observed)

    def test_conforming_noise_segment_passes(self, observed):
        class FlatPSD:
            def psd(self, freqs, channel=None):
                return np.full_like(freqs, 1.0)

        class NoiseSeg(EchoSegment):
            def step(self, residual):
                return replace(residual, noise=FlatPSD())

        check_segment(NoiseSeg(name="noise"), observed)

    def test_noise_model_violating_contract_caught(self, observed):
        class BadNoise(EchoSegment):
            def step(self, residual):
                return replace(residual, noise=object())

        with pytest.raises(TypeError, match="neither psd"):
            check_segment(BadNoise(name="bad"), observed)

    def test_wavelet_only_noise_segment_passes(self, observed):
        # a noise model exposing only wdm_variance is conformant (psd OR wdm)
        class WdmOnly:
            def wdm_variance(self, n_layers, n_time, dt, epoch, channel=None):
                return np.ones((n_layers + 1, n_time))

        class WdmNoiseSeg(EchoSegment):
            def step(self, residual):
                return replace(residual, noise=WdmOnly())

        check_segment(WdmNoiseSeg(name="wdm"), observed)


class TestEchoSegment:
    def test_keeps_its_own_step_counter(self, observed):
        from turntable import Wheel

        echo = EchoSegment(name="echo")
        wheel = Wheel(observed)
        wheel.add(echo)
        wheel.run(3)
        assert echo.steps == 3

    def test_passes_residual_through_unchanged(self, observed):
        from turntable import Wheel

        wheel = Wheel(observed)
        wheel.add(EchoSegment(name="echo"))
        wheel.run(1)
        for ch in observed.channels:
            np.testing.assert_array_equal(
                wheel.residual().tdi[ch], observed.tdi[ch]
            )
