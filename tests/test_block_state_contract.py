"""Physical estimates cross the Wheel boundary; model internals stay on blocks."""

from dataclasses import dataclass, field

import numpy as np

from enchilada import BlockState, DataCovariance, Wheel


def test_block_local_state_survives_calls_without_entering_the_ledger(observed):
    class LocalResource:
        def __deepcopy__(self, memo):
            raise AssertionError("Wheel must not copy resources owned by a block")

    class StatefulSignal:
        name = "signal"

        @dataclass
        class State:
            num_calls: int = 0
            metadata: dict = field(default_factory=dict)

        def __init__(self):
            self.state = self.State(metadata={"resource": LocalResource()})

        def sample(
            self, conditional_residual, noise_covariance, current_block_state, *, rng
        ):
            self.state.num_calls += 1
            amplitude = current_block_state.parameters["amplitude"] + 1.0
            return conditional_residual.block_state(
                template={
                    name: np.full_like(values, amplitude)
                    for name, values in conditional_residual.channel_data.items()
                },
                covariance_matrix=DataCovariance.from_variance(
                    conditional_residual, 2.0
                ),
                parameters={"amplitude": amplitude},
            )

    block = StatefulSignal()
    initial = observed.zero_block_state(parameters={"amplitude": 0.0})
    wheel = Wheel(observed, random_seed=42)
    wheel.add(block, initial_block_state=initial)
    wheel.run(2)

    accepted = wheel.ledger["signal"]
    assert isinstance(accepted, BlockState)
    assert block.state.num_calls == 2
    assert isinstance(block.state.metadata["resource"], LocalResource)
    assert accepted.parameters == {"amplitude": 2.0}
    for name in observed.channel_names:
        np.testing.assert_array_equal(accepted.template[name], 2.0)
        np.testing.assert_allclose(
            wheel.working_residual.channel_data[name], observed.channel_data[name] - 2.0
        )
        assert accepted.covariance_matrix.noise_variance(name) == 2.0

    accepted.parameters["amplitude"] = 100.0
    accepted.template["A"][:] = 100.0
    assert wheel.ledger["signal"].parameters == {"amplitude": 2.0}
    np.testing.assert_array_equal(wheel.contribution("signal")["A"], 2.0)
