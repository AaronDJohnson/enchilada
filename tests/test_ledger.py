"""Ledger snapshots expose stable, independently owned result collections."""

import numpy as np

from enchilada import BlockResult, Wheel


class Counter:
    name = "counter"

    def sample(
        self, conditional_residual, noise_covariance, current_block_result, *, rng
    ):
        current_block_result.sampler_state["values"] += 1
        return current_block_result


def test_ledger_identity_is_reflexive_and_snapshots_are_stable(observed):
    wheel = Wheel(observed)
    wheel.add(
        Counter(),
        initial_block_result=BlockResult(sampler_state={"values": np.array([1.0])}),
    )
    assert wheel.ledger == wheel.ledger
    snapshot = wheel.ledger.snapshot()
    entry = snapshot["counter"]
    assert ("counter", entry) in snapshot.items()
    wheel.run(1)
    np.testing.assert_array_equal(entry.sampler_state["values"], [1])
    entry.sampler_state["values"][:] = 100
    snapshot.clear()
    np.testing.assert_array_equal(wheel.ledger["counter"].sampler_state["values"], [2])


def test_empty_and_populated_snapshot_names_follow_registration(observed):
    wheel = Wheel(observed)
    assert wheel.ledger.snapshot() == {}
    wheel.add(
        Counter(),
        initial_block_result=BlockResult(sampler_state={"values": np.array([1.0])}),
    )
    assert list(wheel.ledger.snapshot()) == list(wheel.ledger) == ["counter"]
