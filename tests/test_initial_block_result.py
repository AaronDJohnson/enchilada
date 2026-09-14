"""Explicit block starts bypass prior draws and use normal atomic adoption."""

import importlib.util
import warnings
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from enchilada import (
    Block,
    BlockResult,
    DataCovariance,
    L1Data,
    NoiseOverwrittenWarning,
    WDMGrid,
    Wheel,
    transform,
)
from enchilada.testing import check_block

requires_wdm = pytest.mark.skipif(
    importlib.util.find_spec("wdm") is None, reason="local WDM backend is optional"
)


@pytest.fixture
def observed():
    return L1Data(
        channel_data={"A": np.full(64, 5.0)},
        channel_names=("A",),
        sample_rate_hz=0.4,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
    )


class SeededBlock:
    name = "seeded"

    def sample(
        self, conditional_residual, noise_covariance, current_block_result, *, rng
    ):
        assert current_block_result.model_parameters["amplitude"] == 2.0
        assert (
            current_block_result.tdi_signal_contribution["A"].shape
            == conditional_residual.channel_data["A"].shape
        )
        time_result = transform(
            current_block_result, "time", reference_data=conditional_residual
        )
        np.testing.assert_allclose(time_result.tdi_signal_contribution["A"], 2.0)
        time_residual = transform(conditional_residual, "time")
        np.testing.assert_allclose(time_residual.channel_data["A"], 5.0)
        return replace(
            current_block_result,
            sampler_state={
                "counter": current_block_result.sampler_state["counter"] + 1,
                "draw": rng.normal(),
            },
        )


def injected_result(observed):
    return observed.block_result(
        {"A": np.full(64, 2.0)},
        model_parameters={"amplitude": 2.0},
        sampler_state={"counter": 0},
        metadata={"injection": np.array([1, 2])},
    )


def test_sample_only_block_implements_protocol(observed):
    block: Block = SeededBlock()
    assert isinstance(block, Block)
    wheel = Wheel(observed)
    wheel.add(block, initial_block_result=injected_result(observed))
    wheel.run(1)
    assert wheel.ledger["seeded"].sampler_state["counter"] == 1


@pytest.mark.parametrize("operation", ["add", "check_block"])
@pytest.mark.parametrize("initial_arguments", [{}, {"initial_block_result": None}])
def test_initial_result_is_required_even_if_block_can_draw_prior(
    observed, operation, initial_arguments
):
    class CanDrawPrior(SeededBlock):
        def draw_prior(self, conditional_residual, noise_covariance, *, rng):
            return injected_result(observed)

    wheel = Wheel(observed)
    with pytest.raises(TypeError, match="initial_block_result"):
        if operation == "add":
            wheel.add(CanDrawPrior(), **initial_arguments)
        else:
            check_block(CanDrawPrior(), observed, **initial_arguments)
    assert len(wheel.ledger) == 0


def test_registration_does_not_access_model_initialization(observed):
    class UnusedPrior(SeededBlock):
        @property
        def draw_prior(self):
            raise AssertionError("registration must not look up initialization hooks")

    wheel = Wheel(observed)
    wheel.add(UnusedPrior(), initial_block_result=injected_result(observed))
    wheel.run(1)
    assert wheel.ledger["seeded"].sampler_state["counter"] == 1


@pytest.mark.parametrize(
    ("canonical_domain", "block_domain"),
    [
        ("time", None),
        ("frequency", "time"),
        ("time", "frequency"),
        pytest.param("time", "wdm", marks=requires_wdm),
        pytest.param("wdm", "wdm", marks=requires_wdm),
    ],
)
def test_seed_skips_prior_and_round_trips_through_selected_domain(
    observed, canonical_domain, block_domain
):
    canonical_options = (
        {"num_frequency_divisions": 8} if canonical_domain == "wdm" else {}
    )
    canonical = transform(observed, canonical_domain, **canonical_options)
    block_options = {"num_time_divisions": 4} if block_domain == "wdm" else {}
    initial = transform(
        injected_result(observed),
        block_domain or canonical_domain,
        reference_data=observed,
        **block_options,
    )
    wheel = Wheel(canonical, random_seed=23)
    rng_before = deepcopy(wheel._rng.bit_generator.state)
    wheel.add(
        SeededBlock(),
        initial_block_result=initial,
        data_domain=block_domain,
        **block_options,
    )
    assert wheel._rng.bit_generator.state == rng_before
    assert wheel.ledger["seeded"].sampler_state == {"counter": 0}
    np.testing.assert_allclose(transform(wheel.residual(), "time").channel_data["A"], 3)
    np.testing.assert_array_equal(
        wheel.observed_data.channel_data["A"], canonical.channel_data["A"]
    )
    assert wheel.contribution("seeded")["A"].shape == canonical.channel_data["A"].shape
    wheel.run(1)
    assert wheel.ledger["seeded"].sampler_state == {
        "counter": 1,
        "draw": np.random.default_rng(23).normal(),
    }
    assert initial.sampler_state == {"counter": 0}


def test_seed_snapshot_is_independent_including_noise_and_nested_state(observed):
    initial = replace(
        injected_result(observed),
        noise_covariance=DataCovariance.from_variance(observed, 0.25),
    )
    wheel = Wheel(observed)
    wheel.add(SeededBlock(), initial_block_result=initial)
    initial.tdi_signal_contribution["A"][:] = 100
    initial.model_parameters["amplitude"] = 100
    initial.sampler_state["counter"] = 100
    initial.metadata["injection"][:] = 100
    initial.noise_covariance.covariance_matrix.setflags(write=True)
    initial.noise_covariance.covariance_matrix[:] = 100
    accepted = wheel.ledger["seeded"]
    assert accepted.model_parameters == {"amplitude": 2.0}
    assert accepted.sampler_state == {"counter": 0}
    np.testing.assert_array_equal(accepted.metadata["injection"], [1, 2])
    np.testing.assert_array_equal(wheel.contribution("seeded")["A"], 2)
    assert wheel.noise_covariance.noise_variance() == 0.25


@pytest.mark.parametrize("block_domain", [None, "frequency"])
@pytest.mark.parametrize(
    "failure", ["type", "shape", "nonfinite", "covariance", "copy"]
)
def test_invalid_seed_leaves_registration_and_rng_unchanged(
    observed, failure, block_domain
):
    class CannotCopy:
        def __deepcopy__(self, memo):
            raise RuntimeError("seed state cannot be copied")

    invalid = transform(
        injected_result(observed), block_domain or "time", reference_data=observed
    )
    if failure == "type":
        invalid = {"amplitude": 2.0}
    elif failure == "shape":
        invalid = replace(invalid, tdi_signal_contribution={"A": np.zeros(3)})
    elif failure == "nonfinite":
        invalid.tdi_signal_contribution["A"][1] = np.nan
    elif failure == "covariance":
        invalid = replace(
            invalid,
            noise_covariance=DataCovariance.from_variance(
                replace(observed, sample_rate_hz=0.8), 1.0
            ),
        )
    else:
        invalid = replace(invalid, sampler_state={"resource": CannotCopy()})
    wheel = Wheel(observed, random_seed=7)
    before = deepcopy(wheel._rng.bit_generator.state)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        wheel.add(SeededBlock(), initial_block_result=invalid, data_domain=block_domain)
    assert len(wheel.ledger) == 0
    assert wheel._blocks == []
    assert wheel._block_domains == {}
    assert wheel._rng.bit_generator.state == before
    np.testing.assert_array_equal(wheel.residual().channel_data["A"], 5)
    # The same name can be retried, and no discarded prior draw consumed RNG.
    wheel.add(SeededBlock(), initial_block_result=injected_result(observed))
    wheel.run(1)
    assert (
        wheel.ledger["seeded"].sampler_state["draw"]
        == np.random.default_rng(7).normal()
    )


def test_failed_seed_translation_does_not_register_block(observed, monkeypatch):
    def unavailable_backend():
        raise ImportError("test WDM backend unavailable")

    monkeypatch.setattr("enchilada._transforms._load_wdm", unavailable_backend)
    grid = WDMGrid(num_frequency_divisions=8, num_time_divisions=8)
    initial = BlockResult(tdi_signal_contribution={"A": np.zeros(grid.array_shape)})
    wheel = Wheel(observed, random_seed=7)
    rng_before = deepcopy(wheel._rng.bit_generator.state)
    with pytest.raises(ImportError, match="test WDM backend unavailable"):
        wheel.add(
            SeededBlock(),
            initial_block_result=initial,
            data_domain="wdm",
            num_frequency_divisions=8,
        )
    assert len(wheel.ledger) == 0
    assert wheel._blocks == []
    assert wheel._block_domains == {}
    assert wheel._rng.bit_generator.state == rng_before
    np.testing.assert_array_equal(wheel.residual().channel_data["A"], 5)
    wheel.add(SeededBlock(), initial_block_result=injected_result(observed))
    wheel.run(1)


@pytest.mark.parametrize("with_noise", [False, True])
def test_seed_without_signal_can_sample_in_selected_domain(observed, with_noise):
    class RetainResult(SeededBlock):
        def sample(
            self, conditional_residual, noise_covariance, current_block_result, *, rng
        ):
            assert conditional_residual.data_domain == "frequency"
            assert current_block_result.tdi_signal_contribution is None
            assert current_block_result.model_parameters == {"variance": 0.25}
            if with_noise:
                assert current_block_result.noise_covariance.data_domain == "frequency"
                assert noise_covariance.data_domain == "frequency"
                assert noise_covariance.quadratic_form(
                    conditional_residual.channel_data
                ) == pytest.approx(64 * 25 / 0.25)
            return replace(current_block_result, sampler_state={"counter": 1})

    initial = BlockResult(
        noise_covariance=DataCovariance.from_variance(observed, 0.25)
        if with_noise
        else None,
        model_parameters={"variance": 0.25},
    )
    wheel = Wheel(observed)
    wheel.add(RetainResult(), initial_block_result=initial, data_domain="frequency")
    wheel.run(1)
    assert wheel.ledger["seeded"].sampler_state == {"counter": 1}
    assert initial.sampler_state == {}
    np.testing.assert_array_equal(wheel.residual().channel_data["A"], 5)


def test_seed_translation_does_not_transform_unrelated_observations(observed):
    # The pristine data's FFT would overflow, but the first block explains it
    # exactly. The second block's zero seed and conditional residual are valid.
    loud_observed = replace(observed, channel_data={"A": np.full(64, 1e307)})
    wheel = Wheel(loud_observed)
    first, second = SeededBlock(), SeededBlock()
    first.name, second.name = "loud", "zero"
    wheel.add(
        first,
        initial_block_result=loud_observed.block_result(loud_observed.channel_data),
    )
    wheel.add(
        second,
        initial_block_result=BlockResult(
            tdi_signal_contribution={"A": np.zeros(33, complex)}
        ),
        data_domain="frequency",
    )
    assert list(wheel.ledger) == ["loud", "zero"]
    np.testing.assert_array_equal(wheel.contribution("zero")["A"], 0)
    np.testing.assert_array_equal(wheel.residual().channel_data["A"], 0)


def test_noise_only_seed_establishes_covariance_owner(observed):
    class DropsNoise(SeededBlock):
        def sample(
            self, conditional_residual, noise_covariance, current_block_result, *, rng
        ):
            return BlockResult(model_parameters={"variance": 0.25})

    seed = BlockResult(
        noise_covariance=DataCovariance.from_variance(observed, 0.25),
        model_parameters={"variance": 0.25},
    )
    wheel = Wheel(observed)
    wheel.add(DropsNoise(), initial_block_result=seed)
    np.testing.assert_array_equal(wheel.residual().channel_data["A"], 5)
    assert wheel.noise_covariance.noise_variance() == 0.25
    with pytest.raises(ValueError, match="owns the current covariance"):
        wheel.run(1)
    assert wheel.ledger["seeded"].noise_covariance is not None


def test_seeded_noise_takeover_warning_can_fail_atomically(observed):
    wheel = Wheel(observed, random_seed=3)
    initial = BlockResult(noise_covariance=DataCovariance.from_variance(observed, 1.0))
    first, second = SeededBlock(), SeededBlock()
    first.name, second.name = "first", "second"
    wheel.add(first, initial_block_result=initial)
    new_noise = BlockResult(
        noise_covariance=DataCovariance.from_variance(observed, 2.0)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", NoiseOverwrittenWarning)
        with pytest.raises(NoiseOverwrittenWarning):
            wheel.add(second, initial_block_result=new_noise, data_domain="frequency")
    assert list(wheel.ledger) == ["first"]
    assert "second" not in wheel._block_domains
    assert wheel.noise_covariance.noise_variance() == 1.0


def test_check_block_can_exercise_an_explicit_injection(observed):
    check_block(
        SeededBlock(),
        observed,
        initial_block_result=injected_result(observed),
        num_cycles=2,
    )
