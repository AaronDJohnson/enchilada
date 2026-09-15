"""enchilada.testing: EchoBlock and the check_block conformance helper."""

import numpy as np
import pytest

from conftest import make_observed
from enchilada import BlockResult, DataCovariance
from enchilada.testing import EchoBlock, check_block


class TestCheckBlock:
    def test_conforming_block_passes(self, observed):
        check_block(
            block_under_test=EchoBlock(name="echo"),
            observed_data=observed,
            initial_block_result=BlockResult(sampler_state={"num_sample_calls": 0}),
            num_cycles=2,
            initial_noise_covariance=DataCovariance.from_variance(observed, 1.0),
            random_seed=0,
        )

    def test_conforming_in_frequency_domain(self, rng):
        n = 64
        obs = make_observed(
            rng,
            data_domain="frequency",
            channel_data={
                ch: rng.standard_normal(n // 2 + 1) + 0j for ch in ("A", "E", "T")
            },
        )
        check_block(
            EchoBlock(name="echo"),
            obs,
            initial_block_result=BlockResult(sampler_state={"num_sample_calls": 0}),
        )

    def test_non_block_result_return_caught(self, observed):
        class Bad(EchoBlock):
            def sample(
                self,
                conditional_residual,
                noise_covariance,
                current_block_result,
                *,
                rng,
            ):
                return {"A": np.zeros(1)}  # not a BlockResult

        with pytest.raises(TypeError, match="must return a BlockResult"):
            check_block(Bad(name="bad"), observed, initial_block_result=BlockResult())

    def test_off_grid_block_result_caught(self, observed):
        class Drifter(EchoBlock):
            def sample(
                self,
                conditional_residual,
                noise_covariance,
                current_block_result,
                *,
                rng,
            ):
                return BlockResult(
                    tdi_signal_contribution={
                        ch: np.zeros(3) for ch in conditional_residual.channel_names
                    }
                )

        with pytest.raises(ValueError, match="length 3, expected"):
            check_block(
                Drifter(name="drift"), observed, initial_block_result=BlockResult()
            )

    def test_conforming_noise_block_passes(self, observed):
        class FlatPSD:
            def psd(self, freqs, channel=None):
                return np.full_like(freqs, 1.0)

        class FlatNoiseBlock(EchoBlock):
            def sample(
                self,
                conditional_residual,
                noise_covariance,
                current_block_result,
                *,
                rng,
            ):
                from enchilada import DataCovariance

                return conditional_residual.block_result().with_noise_covariance(
                    DataCovariance.from_psd(conditional_residual, FlatPSD())
                )

        check_block(
            FlatNoiseBlock(name="noise"), observed, initial_block_result=BlockResult()
        )

    def test_noise_model_violating_contract_caught(self, observed):
        class BadNoise(EchoBlock):
            def sample(
                self,
                conditional_residual,
                noise_covariance,
                current_block_result,
                *,
                rng,
            ):
                return conditional_residual.block_result().with_noise_covariance(
                    object()
                )

        with pytest.raises(TypeError, match="DataCovariance"):
            check_block(
                BadNoise(name="bad"), observed, initial_block_result=BlockResult()
            )

    def test_initial_block_result_is_required(self, observed):
        with pytest.raises(TypeError, match="initial_block_result"):
            check_block(EchoBlock(name="echo"), observed)

    @pytest.mark.parametrize("invalid", [None, object()])
    def test_initial_block_result_must_be_a_block_result(self, observed, invalid):
        with pytest.raises(TypeError, match="initial_block_result.*BlockResult"):
            check_block(EchoBlock(name="echo"), observed, initial_block_result=invalid)


class TestEchoBlock:
    def test_keeps_sampling_counter_in_block_result(self, observed):
        from enchilada import Wheel

        echo = EchoBlock(name="echo")
        wheel = Wheel(observed)
        wheel.add(
            echo,
            initial_block_result=BlockResult(sampler_state={"num_sample_calls": 0}),
        )
        wheel.run(3)
        assert wheel.ledger["echo"].sampler_state["num_sample_calls"] == 3
        assert vars(echo) == {"name": "echo"}

    def test_contributes_a_zero_signal(self, observed):
        from enchilada import Wheel

        wheel = Wheel(observed)
        wheel.add(
            EchoBlock(name="echo"),
            initial_block_result=BlockResult(sampler_state={"num_sample_calls": 0}),
        )
        wheel.run(1)
        for ch in observed.channel_names:
            np.testing.assert_array_equal(
                wheel.residual().channel_data[ch], observed.channel_data[ch]
            )


def test_echo_initial_state_is_independent_of_other_campaigns(observed, rng):
    echo = EchoBlock(name="echo")
    initial = BlockResult(sampler_state={"num_sample_calls": 0})
    updated = echo.sample(
        conditional_residual=observed,
        noise_covariance=None,
        current_block_result=initial,
        rng=rng,
    )
    assert initial.sampler_state["num_sample_calls"] == 0
    assert updated.sampler_state["num_sample_calls"] == 1
    fresh = BlockResult(sampler_state={"num_sample_calls": 0})
    restarted = echo.sample(observed, None, fresh, rng=rng)
    assert fresh.sampler_state["num_sample_calls"] == 0
    assert restarted.sampler_state["num_sample_calls"] == 1
    assert not hasattr(echo, "draw_prior")
    assert vars(echo) == {"name": "echo"}
