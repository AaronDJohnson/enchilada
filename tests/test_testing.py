"""turntable.testing: EchoBlock and the check_block conformance helper."""

from dataclasses import replace

import numpy as np
import pytest

from conftest import make_observed
from turntable.testing import EchoBlock, check_block


class TestCheckBlock:
    def test_conforming_block_passes(self, observed):
        check_block(EchoBlock(name="echo"), observed)

    def test_conforming_in_frequency_domain(self, rng):
        n = 64
        obs = make_observed(
            rng,
            domain="frequency",
            tdi={ch: rng.standard_normal(n // 2 + 1) + 0j for ch in ("A", "E", "T")},
        )
        check_block(EchoBlock(name="echo"), obs)

    def test_non_residual_return_caught(self, observed):
        class Bad(EchoBlock):
            def block_update(self, residual):
                return {"A": np.zeros(1)}  # not a Residuals

        with pytest.raises(TypeError, match="must return a Residuals"):
            check_block(Bad(name="bad"), observed)

    def test_changed_run_setting_caught(self, observed):
        class Cheat(EchoBlock):
            def block_update(self, residual):
                return replace(residual, observable="strain")

        with pytest.raises(ValueError, match="changed the run setting"):
            check_block(Cheat(name="cheat"), observed)

    def test_conforming_noise_block_passes(self, observed):
        class FlatPSD:
            def psd(self, freqs, channel=None):
                return np.full_like(freqs, 1.0)

        class NoiseSeg(EchoBlock):
            def block_update(self, residual):
                return replace(residual, noise=FlatPSD())

        check_block(NoiseSeg(name="noise"), observed)

    def test_noise_model_violating_contract_caught(self, observed):
        class BadNoise(EchoBlock):
            def block_update(self, residual):
                return replace(residual, noise=object())

        with pytest.raises(TypeError, match="does not expose"):
            check_block(BadNoise(name="bad"), observed)


class TestEchoBlock:
    def test_keeps_its_own_update_counter(self, observed):
        from turntable import Wheel

        echo = EchoBlock(name="echo")
        wheel = Wheel(observed)
        wheel.add(echo)
        wheel.run(3)
        assert echo.updates == 3

    def test_passes_residual_through_unchanged(self, observed):
        from turntable import Wheel

        wheel = Wheel(observed)
        wheel.add(EchoBlock(name="echo"))
        wheel.run(1)
        for ch in observed.channels:
            np.testing.assert_array_equal(wheel.residual().tdi[ch], observed.tdi[ch])


class TestCheckBlockStrictness:
    def test_a_withdrawing_block_fails_the_conformance_check(self, observed):
        """The Wheel warns mid-run; the pre-campaign gate must refuse."""
        from turntable import ModelWithdrawnWarning, replace

        class Forgetful(EchoBlock):
            def __init__(self, name):
                super().__init__(name)
                self.calls = 0

            def start(self, residual):
                return replace(
                    residual,
                    tdi={ch: arr - 1.0 for ch, arr in residual.tdi.items()},
                )

            def block_update(self, residual):
                return residual  # forgot to re-subtract: model leaves the fit

        with pytest.raises(ModelWithdrawnWarning):
            check_block(Forgetful(name="forgetful"), observed)
