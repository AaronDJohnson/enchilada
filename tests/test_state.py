"""Campaign state is explicit and a failed call cannot advance it."""

from dataclasses import replace

import numpy as np
import pytest

from enchilada import DataCovariance, Wheel


def counter_initial_result(observed):
    return observed.zero_block_result(
        model_parameters={"a": 0.0}, sampler_state={"n": 0}
    )


class Counter:
    name = "counter"

    def sample(
        self, conditional_residual, noise_covariance, current_block_result, *, rng
    ):
        current_block_result.sampler_state["n"] += 1
        current_block_result.model_parameters["a"] = rng.normal()
        return current_block_result


def test_ledger_carries_initial_result_and_complete_state(observed):
    wheel = Wheel(observed, random_seed=4)
    block = Counter()
    wheel.add(block, initial_block_result=counter_initial_result(observed))
    assert wheel.ledger["counter"].sampler_state == {"n": 0}
    initial = wheel.ledger["counter"].model_parameters["a"]
    wheel.run(2)
    assert wheel.ledger["counter"].sampler_state == {"n": 2}
    assert wheel.ledger["counter"].model_parameters["a"] != initial
    assert vars(block) == {}


def test_pristine_working_and_input_are_independent(observed):
    original = observed.channel_data["A"].copy()
    wheel = Wheel(observed)
    observed.channel_data["A"][:] = 100
    wheel.observed_data.channel_data["A"][:] = 200
    wheel.working_residual.channel_data["A"][:] = 300
    np.testing.assert_array_equal(wheel.observed_data.channel_data["A"], original)
    np.testing.assert_array_equal(wheel.working_residual.channel_data["A"], original)


def test_block_result_state_is_copied_on_adoption_and_read(observed):
    initial = replace(
        counter_initial_result(observed),
        sampler_state={"n": 0, "nested": [np.ones(2)]},
    )
    wheel = Wheel(observed)
    wheel.add(Counter(), initial_block_result=initial)
    initial.sampler_state["nested"][0][:] = 10
    wheel.ledger["counter"].sampler_state["nested"][0][:] = 20
    np.testing.assert_array_equal(
        wheel.ledger["counter"].sampler_state["nested"][0], [1, 1]
    )


def test_failed_sample_rolls_back_state_data_and_rng(observed):
    fail = [True]

    class FailsOnce(Counter):
        def sample(
            self, conditional_residual, noise_covariance, current_block_result, *, rng
        ):
            result = super().sample(
                conditional_residual, noise_covariance, current_block_result, rng=rng
            )
            conditional_residual.channel_data["A"][:] = 999
            if fail[0]:
                fail[0] = False
                return replace(
                    result,
                    tdi_signal_contribution={
                        ch: np.full_like(x, np.nan)
                        for ch, x in result.tdi_signal_contribution.items()
                    },
                )
            return result

    actual, expected = Wheel(observed, random_seed=5), Wheel(observed, random_seed=5)
    actual.add(FailsOnce(), initial_block_result=counter_initial_result(observed))
    expected.add(Counter(), initial_block_result=counter_initial_result(observed))
    before = actual.ledger["counter"].model_parameters.copy()
    with pytest.raises(ValueError, match="non-finite"):
        actual.run(1)
    assert actual.ledger["counter"].model_parameters == before
    assert actual.ledger["counter"].sampler_state == {"n": 0}
    np.testing.assert_array_equal(
        actual.working_residual.channel_data["A"], observed.channel_data["A"]
    )
    actual.run(1)
    expected.run(1)
    assert (
        actual.ledger["counter"].model_parameters
        == expected.ledger["counter"].model_parameters
    )


def test_one_block_can_be_shared_between_campaigns(observed):
    block = Counter()
    one, two = Wheel(observed, random_seed=7), Wheel(observed, random_seed=7)
    initial = counter_initial_result(observed)
    one.add(block, initial_block_result=initial)
    two.add(block, initial_block_result=initial)
    one.run(2)
    two.run(1)
    two.run(1)
    assert one.ledger["counter"].sampler_state == two.ledger["counter"].sampler_state
    assert (
        one.ledger["counter"].model_parameters == two.ledger["counter"].model_parameters
    )


def test_retained_rng_reference_cannot_advance_campaign(observed):
    retained = []

    class RetainsRng(Counter):
        def sample(
            self, conditional_residual, noise_covariance, current_block_result, *, rng
        ):
            retained.append(rng)
            return super().sample(
                conditional_residual, noise_covariance, current_block_result, rng=rng
            )

    actual, expected = Wheel(observed, random_seed=11), Wheel(observed, random_seed=11)
    actual.add(RetainsRng(), initial_block_result=counter_initial_result(observed))
    expected.add(Counter(), initial_block_result=counter_initial_result(observed))
    actual.run(1)
    expected.run(1)
    retained[0].normal(size=100)
    actual.run(1)
    expected.run(1)
    assert (
        actual.ledger["counter"].model_parameters
        == expected.ledger["counter"].model_parameters
    )


@pytest.mark.parametrize("failure", ["copy", "covariance", "exception"])
def test_failed_adoption_does_not_advance_rng_or_state(observed, failure):
    fail = [True]

    class CannotCopy:
        def __deepcopy__(self, memo):
            raise RuntimeError("cannot copy sampler resource")

    class Invalid(Counter):
        def sample(
            self, conditional_residual, noise_covariance, current_block_result, *, rng
        ):
            result = super().sample(
                conditional_residual, noise_covariance, current_block_result, rng=rng
            )
            if fail[0]:
                fail[0] = False
                if failure == "copy":
                    return replace(result, sampler_state={"resource": CannotCopy()})
                if failure == "exception":
                    raise RuntimeError("sampler failed")
                wrong_grid = replace(
                    conditional_residual,
                    start_time_gps=conditional_residual.start_time_gps + 1,
                )
                return result.with_noise_covariance(
                    DataCovariance.from_variance(wrong_grid, 1)
                )
            return result

    actual, expected = Wheel(observed, random_seed=12), Wheel(observed, random_seed=12)
    actual.add(Invalid(), initial_block_result=counter_initial_result(observed))
    expected.add(Counter(), initial_block_result=counter_initial_result(observed))
    with pytest.raises((RuntimeError, ValueError)):
        actual.run(1)
    assert actual.ledger["counter"].sampler_state == {"n": 0}
    assert actual.noise_covariance is None
    actual.run(1)
    expected.run(1)
    assert (
        actual.ledger["counter"].model_parameters
        == expected.ledger["counter"].model_parameters
    )


def test_covariance_input_and_output_snapshots_are_isolated(observed):
    covariance = DataCovariance.from_variance(observed, 1)
    wheel = Wheel(observed, initial_noise_covariance=covariance)
    covariance.covariance_matrix.setflags(write=True)
    covariance.covariance_matrix[:] *= 2
    inspected = wheel.noise_covariance
    inspected.covariance_matrix.setflags(write=True)
    inspected.covariance_matrix[:] *= 3

    class EditsNoise(Counter):
        def sample(
            self, conditional_residual, noise_covariance, current_block_result, *, rng
        ):
            noise_covariance.covariance_matrix.setflags(write=True)
            noise_covariance.covariance_matrix[:] *= 4
            return current_block_result

    wheel.add(EditsNoise(), initial_block_result=counter_initial_result(observed))
    wheel.run(1)
    assert wheel.noise_covariance.noise_variance("A") == 1


def test_residual_overflow_is_rejected_before_registration(observed):
    data = replace(
        observed,
        channel_data={
            ch: np.full_like(x, 1e308) for ch, x in observed.channel_data.items()
        },
    )

    wheel = Wheel(data)
    with pytest.raises(ValueError, match="non-finite"):
        wheel.add(
            Counter(),
            initial_block_result=data.block_result(
                {ch: -x for ch, x in data.channel_data.items()}
            ),
        )
    assert len(wheel.ledger) == 0
    np.testing.assert_array_equal(
        wheel.working_residual.channel_data["A"], data.channel_data["A"]
    )
