"""Published estimates and campaign RNG are isolated from block-local state."""

from dataclasses import dataclass, replace

import numpy as np
import pytest

from enchilada import DataCovariance, Wheel


def counter_initial_result(observed):
    return observed.zero_block_state(parameters={"a": 0.0})


class Counter:
    name = "counter"

    @dataclass
    class State:
        n: int = 0

    def __init__(self):
        self.state = self.State()

    def sample(
        self, conditional_residual, noise_covariance, current_block_state, *, rng
    ):
        self.state.n += 1
        current_block_state.parameters["a"] = rng.normal()
        return current_block_state


def test_ledger_carries_estimates_and_block_keeps_continuation(observed):
    wheel = Wheel(observed, random_seed=4)
    block = Counter()
    wheel.add(block, initial_block_state=counter_initial_result(observed))
    assert block.state.n == 0
    initial = wheel.ledger["counter"].parameters["a"]
    wheel.run(2)
    assert block.state.n == 2
    assert wheel.ledger["counter"].parameters["a"] != initial


def test_pristine_working_and_input_are_independent(observed):
    original = observed.channel_data["A"].copy()
    wheel = Wheel(observed)
    observed.channel_data["A"][:] = 100
    wheel.observed_data.channel_data["A"][:] = 200
    wheel.working_residual.channel_data["A"][:] = 300
    np.testing.assert_array_equal(wheel.observed_data.channel_data["A"], original)
    np.testing.assert_array_equal(wheel.working_residual.channel_data["A"], original)


def test_block_state_parameters_are_copied_on_adoption_and_read(observed):
    initial = replace(
        counter_initial_result(observed),
        parameters={"a": 0.0, "amplitudes": [np.ones(2)]},
    )
    wheel = Wheel(observed)
    wheel.add(Counter(), initial_block_state=initial)
    initial.parameters["amplitudes"][0][:] = 10
    wheel.ledger["counter"].parameters["amplitudes"][0][:] = 20
    np.testing.assert_array_equal(
        wheel.ledger["counter"].parameters["amplitudes"][0], [1, 1]
    )


def test_failed_sample_rolls_back_published_state_data_and_rng(observed):
    fail = [True]

    class FailsOnce(Counter):
        def sample(
            self, conditional_residual, noise_covariance, current_block_state, *, rng
        ):
            result = super().sample(
                conditional_residual, noise_covariance, current_block_state, rng=rng
            )
            conditional_residual.channel_data["A"][:] = 999
            if fail[0]:
                fail[0] = False
                return replace(
                    result,
                    template={
                        ch: np.full_like(x, np.nan) for ch, x in result.template.items()
                    },
                )
            return result

    actual, expected = Wheel(observed, random_seed=5), Wheel(observed, random_seed=5)
    block = FailsOnce()
    actual.add(block, initial_block_state=counter_initial_result(observed))
    expected.add(Counter(), initial_block_state=counter_initial_result(observed))
    before = actual.ledger["counter"].parameters.copy()
    with pytest.raises(ValueError, match="non-finite"):
        actual.run(1)
    assert actual.ledger["counter"].parameters == before
    assert block.state.n == 1
    np.testing.assert_array_equal(
        actual.working_residual.channel_data["A"], observed.channel_data["A"]
    )
    actual.run(1)
    expected.run(1)
    assert actual.ledger["counter"].parameters == expected.ledger["counter"].parameters


def test_separate_blocks_keep_independent_campaign_continuation(observed):
    first, second = Counter(), Counter()
    one, two = Wheel(observed, random_seed=7), Wheel(observed, random_seed=7)
    initial = counter_initial_result(observed)
    one.add(first, initial_block_state=initial)
    two.add(second, initial_block_state=initial)
    one.run(2)
    assert first.state.n == 2
    assert second.state.n == 0
    two.run(1)
    two.run(1)
    assert first.state.n == second.state.n == 2
    assert one.ledger["counter"].parameters == two.ledger["counter"].parameters


def test_retained_rng_reference_cannot_advance_campaign(observed):
    retained = []

    class RetainsRng(Counter):
        def sample(
            self, conditional_residual, noise_covariance, current_block_state, *, rng
        ):
            retained.append(rng)
            return super().sample(
                conditional_residual, noise_covariance, current_block_state, rng=rng
            )

    actual, expected = Wheel(observed, random_seed=11), Wheel(observed, random_seed=11)
    actual.add(RetainsRng(), initial_block_state=counter_initial_result(observed))
    expected.add(Counter(), initial_block_state=counter_initial_result(observed))
    actual.run(1)
    expected.run(1)
    retained[0].normal(size=100)
    actual.run(1)
    expected.run(1)
    assert actual.ledger["counter"].parameters == expected.ledger["counter"].parameters


@pytest.mark.parametrize("failure", ["copy", "covariance", "exception"])
def test_failed_adoption_preserves_rng_and_published_state(observed, failure):
    fail = [True]

    class CannotCopy:
        def __deepcopy__(self, memo):
            raise RuntimeError("cannot copy parameter")

    class Invalid(Counter):
        def sample(
            self, conditional_residual, noise_covariance, current_block_state, *, rng
        ):
            result = super().sample(
                conditional_residual, noise_covariance, current_block_state, rng=rng
            )
            if fail[0]:
                fail[0] = False
                if failure == "copy":
                    return replace(result, parameters={"amplitude": CannotCopy()})
                if failure == "exception":
                    raise RuntimeError("sampler failed")
                wrong_grid = replace(
                    conditional_residual,
                    start_time_gps=conditional_residual.start_time_gps + 1,
                )
                return result.with_covariance_matrix(
                    DataCovariance.from_variance(wrong_grid, 1)
                )
            return result

    actual, expected = Wheel(observed, random_seed=12), Wheel(observed, random_seed=12)
    block = Invalid()
    actual.add(block, initial_block_state=counter_initial_result(observed))
    expected.add(Counter(), initial_block_state=counter_initial_result(observed))
    with pytest.raises((RuntimeError, ValueError)):
        actual.run(1)
    assert actual.ledger["counter"].parameters == {"a": 0.0}
    assert block.state.n == 1
    assert actual.noise_covariance is None
    actual.run(1)
    expected.run(1)
    assert actual.ledger["counter"].parameters == expected.ledger["counter"].parameters


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
            self, conditional_residual, noise_covariance, current_block_state, *, rng
        ):
            noise_covariance.covariance_matrix.setflags(write=True)
            noise_covariance.covariance_matrix[:] *= 4
            return current_block_state

    wheel.add(EditsNoise(), initial_block_state=counter_initial_result(observed))
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
            initial_block_state=data.block_state(
                {ch: -x for ch, x in data.channel_data.items()}
            ),
        )
    assert len(wheel.ledger) == 0
    np.testing.assert_array_equal(
        wheel.working_residual.channel_data["A"], data.channel_data["A"]
    )
