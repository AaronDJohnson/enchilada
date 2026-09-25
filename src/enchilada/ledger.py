"""Read-only access to the complete block states accepted by the orchestrator."""

from collections.abc import Iterator
from copy import deepcopy

from enchilada.block_state import BlockState


class Ledger:
    """Read-only access to each block's latest accepted BlockState.

    Wheel alone records entries. Indexing deep-copies a state; `snapshot()`
    deep-copies all states into a dictionary. Iteration yields block names.
    Ledger equality uses object identity; compare state values explicitly.

    This stores current physical estimates. Sampler continuation and diagnostics
    stay on each block. Collect posterior history through Wheel's
    on_cycle_complete callback.
    """

    def __init__(self) -> None:
        self._states_by_block_name: dict[str, BlockState] = {}

    def __getitem__(self, block_name: str) -> BlockState:
        return deepcopy(self._states_by_block_name[block_name])

    def __iter__(self) -> Iterator[str]:
        return iter(self._states_by_block_name)

    def __len__(self) -> int:
        return len(self._states_by_block_name)

    def __contains__(self, block_name: object) -> bool:
        return block_name in self._states_by_block_name

    def snapshot(self) -> dict[str, BlockState]:
        """Copy all accepted states into an independently owned dictionary."""
        return deepcopy(self._states_by_block_name)
