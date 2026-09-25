"""Ledger snapshots expose stable, independently owned estimate collections."""

import numpy as np

from enchilada import BlockState, Wheel


class AmplitudeUpdate:
    name = "amplitudes"

    def sample(
        self, conditional_residual, noise_covariance, current_block_state, *, rng
    ):
        current_block_state.parameters["amplitudes"] += 1
        return current_block_state


def test_ledger_identity_is_reflexive_and_snapshots_are_stable(observed):
    wheel = Wheel(observed)
    wheel.add(
        AmplitudeUpdate(),
        initial_block_state=BlockState(parameters={"amplitudes": np.array([1.0])}),
    )
    assert wheel.ledger == wheel.ledger
    snapshot = wheel.ledger.snapshot()
    entry = snapshot["amplitudes"]
    assert ("amplitudes", entry) in snapshot.items()
    wheel.run(1)
    np.testing.assert_array_equal(entry.parameters["amplitudes"], [1])
    entry.parameters["amplitudes"][:] = 100
    snapshot.clear()
    np.testing.assert_array_equal(
        wheel.ledger["amplitudes"].parameters["amplitudes"], [2]
    )


def test_empty_and_populated_snapshot_names_follow_registration(observed):
    wheel = Wheel(observed)
    assert wheel.ledger.snapshot() == {}
    wheel.add(
        AmplitudeUpdate(),
        initial_block_state=BlockState(parameters={"amplitudes": np.array([1.0])}),
    )
    assert list(wheel.ledger.snapshot()) == list(wheel.ledger) == ["amplitudes"]
