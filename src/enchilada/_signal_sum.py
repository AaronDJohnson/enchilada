"""Persistent pairwise sums of internally owned, validated block signals.

Replacing a leaf copies only its path to the root. An uncommitted candidate can
therefore be discarded without changing accepted state. Arrays are never mutated
here, and absent signals need no zero arrays. Wheel keeps this tree private.
"""

from dataclasses import dataclass

import numpy as np

Signal = dict[str, np.ndarray] | None


def _combine(left: Signal, right: Signal) -> Signal:
    if left is None:
        return right
    if right is None:
        return left
    # Do not round a subtree of float32 signals before it meets a float64
    # observation or another wider signal elsewhere in the tree.
    with np.errstate(over="ignore", invalid="ignore"):
        return {
            name: np.add(
                values,
                right[name],
                dtype=np.result_type(values, right[name], np.float64),
            )
            for name, values in left.items()
        }


@dataclass(frozen=True, eq=False)
class _Node:
    total: Signal = None
    left: "_Node | None" = None
    right: "_Node | None" = None


def _total(node: _Node | None) -> Signal:
    return None if node is None else node.total


def _replace(node: _Node | None, index: int, capacity: int, signal: Signal) -> _Node:
    if capacity == 1:
        return _Node(signal)
    midpoint = capacity // 2
    left = None if node is None else node.left
    right = None if node is None else node.right
    if index < midpoint:
        left = _replace(left, index, midpoint, signal)
    else:
        right = _replace(right, index - midpoint, midpoint, signal)
    return _Node(_combine(_total(left), _total(right)), left, right)


def _excluding(node: _Node | None, index: int, capacity: int) -> Signal:
    if node is None or capacity == 1:
        return None
    midpoint = capacity // 2
    if index < midpoint:
        return _combine(_excluding(node.left, index, midpoint), _total(node.right))
    return _combine(
        _total(node.left), _excluding(node.right, index - midpoint, midpoint)
    )


@dataclass(frozen=True, eq=False)
class SignalSum:
    """A private aggregate indexed by nonnegative block registration positions."""

    _root: _Node | None = None
    _capacity: int = 1

    @property
    def total(self) -> Signal:
        return _total(self._root)

    def with_signal(self, index: int, signal: Signal) -> "SignalSum":
        root, capacity = self._root, self._capacity
        while index >= capacity:
            root = _Node(_total(root), root)
            capacity *= 2
        return SignalSum(_replace(root, index, capacity, signal), capacity)

    def excluding(self, index: int) -> Signal:
        return _excluding(self._root, index, self._capacity)
