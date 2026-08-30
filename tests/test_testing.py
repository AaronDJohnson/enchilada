"""enchilada.testing: EchoBlock and the check_block conformance helper."""

from dataclasses import replace

import numpy as np
import pytest

from conftest import make_observed
from enchilada import Template
from enchilada.testing import EchoBlock, check_block


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

    def test_non_template_return_caught(self, observed):
        class Bad(EchoBlock):
            def update(self, residual):
                return {"A": np.zeros(1)}  # not a Template

        with pytest.raises(TypeError, match="must return a Template"):
            check_block(Bad(name="bad"), observed)

    def test_off_grid_template_caught(self, observed):
        class Drifter(EchoBlock):
            def update(self, residual):
                return Template(tdi={ch: np.zeros(3) for ch in residual.channels})

        with pytest.raises(ValueError, match="length 3, expected"):
            check_block(Drifter(name="drift"), observed)

    def test_conforming_noise_block_passes(self, observed):
        class FlatPSD:
            def psd(self, freqs, channel=None):
                return np.full_like(freqs, 1.0)

        class FlatNoiseBlock(EchoBlock):
            def update(self, residual):
                return residual.zero_template().with_noise(FlatPSD())

        check_block(FlatNoiseBlock(name="noise"), observed)

    def test_noise_model_violating_contract_caught(self, observed):
        class BadNoise(EchoBlock):
            def update(self, residual):
                return residual.zero_template().with_noise(object())

        with pytest.raises(TypeError, match="does not expose"):
            check_block(BadNoise(name="bad"), observed)


class TestEchoBlock:
    def test_keeps_its_own_update_counter(self, observed):
        from enchilada import Wheel

        echo = EchoBlock(name="echo")
        wheel = Wheel(observed)
        wheel.add(echo)
        wheel.run(3)
        assert echo.updates == 3

    def test_contributes_a_zero_template(self, observed):
        from enchilada import Wheel

        wheel = Wheel(observed)
        wheel.add(EchoBlock(name="echo"))
        wheel.run(1)
        for ch in observed.channels:
            np.testing.assert_array_equal(wheel.residual().tdi[ch], observed.tdi[ch])


class TestCheckBlockStrictness:
    def test_an_unmigrated_block_fails_the_conformance_check(self, observed):
        """A block written for the pre-template contract must fail here,
        before it reaches a shared campaign."""

        class Unmigrated(EchoBlock):
            def update(self, residual):
                return replace(  # residual minus my template: the old contract
                    residual,
                    tdi={ch: arr - 1.0 for ch, arr in residual.tdi.items()},
                )

        with pytest.raises(TypeError, match="must return a Template"):
            check_block(Unmigrated(name="old"), observed)
