"""The boundary between a model's local state and its published estimates."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from enchilada.block_state import BlockState
from enchilada.covariance import DataCovariance
from enchilada.data import L1Data
from enchilada.translated_covariance import TranslatedCovariance


@runtime_checkable
class Block(Protocol):
    """A model that publishes physical estimates while owning its sampler state.

    Keep sampler continuation and diagnostics in a block-owned State object.
    BlockState contains only template, covariance_matrix, and physical parameters.
    `Wheel.add` requires an explicit `initial_block_state`; it calls no
    initialization hook. `sample` advances the latest accepted
    `current_block_state` and returns a complete new state.

    `conditional_residual` is the observed data minus every OTHER block's current
    signal; this block's signal is still present. The block does not perform
    cross-block subtraction or add-back. `noise_covariance` is the current
    DataCovariance, TranslatedCovariance, or None.

    Signal-only, noise-only, and joint models share this interface. A None signal
    means zero contribution, replacing any previous signal.
    The current covariance owner must return its complete covariance every call;
    other blocks can omit covariance without changing it. Wheel keeps one
    covariance owner; publish combined noise components together.

    Each call receives independent snapshots of the conditional residual,
    covariance, and current state. Use the supplied RNG; Wheel commits its
    progress with the state. Treat the shared orbit as immutable.
    Wheel does not copy or roll back block-local state; blocks manage their own
    failure recovery. Use separate block instances for independent campaigns.
    Register a preferred data_domain with Wheel.add to receive matching inputs.
    A translated covariance exposes exact operators; it may correlate samples.
    Validate unsupported conventions explicitly.

    Static type checkers validate signatures; runtime protocol checks only test
    member presence. Neither establishes statistical correctness.
    """

    if TYPE_CHECKING:

        @property
        def name(self) -> str:
            """Stable registration name; immutable configurations are supported."""
            ...
    else:
        # Avoid an inherited property setter blocking dataclass construction
        # when an implementation explicitly subclasses this protocol.
        name: str

    def sample(
        self,
        conditional_residual: L1Data,
        noise_covariance: DataCovariance | TranslatedCovariance | None,
        current_block_state: BlockState,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Advance the supplied state and return the complete new block state."""
        ...
