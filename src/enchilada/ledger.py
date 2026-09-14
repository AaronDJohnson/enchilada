"""Read-only access to the complete block results accepted by the orchestrator."""

from collections.abc import Iterator
from copy import deepcopy

from enchilada.block_result import BlockResult


class Ledger:
    """One complete BlockResult per stable block name, including optional estimates.

    `model_parameters` describes the physical model. `sampler_state` preserves
    algorithm continuation information independently of those physical values.

    Inspection returns a deep copy so modifying a diagnostic cannot affect the
    next sample call. Wheel alone records entries. This stores current state, not an
    unbounded posterior history; collect samples with Wheel's on_cycle_complete hook.

    The ledger is an identity-comparing store, not a value-comparing Mapping.
    Use `snapshot()` for a stable dictionary to iterate with items/keys/values.
    Results contain arbitrary sampler state and arrays; compare their physical
    values explicitly when comparing different snapshots.
    """

    def __init__(self) -> None:
        self._results_by_block_name: dict[str, BlockResult] = {}

    def __getitem__(self, block_name: str) -> BlockResult:
        return deepcopy(self._results_by_block_name[block_name])

    def __iter__(self) -> Iterator[str]:
        return iter(self._results_by_block_name)

    def __len__(self) -> int:
        return len(self._results_by_block_name)

    def __contains__(self, block_name: object) -> bool:
        return block_name in self._results_by_block_name

    def snapshot(self) -> dict[str, BlockResult]:
        """Copy all accepted results into an independently owned dictionary."""
        return deepcopy(self._results_by_block_name)
