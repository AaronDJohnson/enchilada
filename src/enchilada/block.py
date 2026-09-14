"""The stateless boundary between a model and its orchestrator."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from enchilada.block_result import BlockResult
from enchilada.covariance import DataCovariance
from enchilada.data import L1Data
from enchilada.translated_covariance import TranslatedCovariance


@runtime_checkable
class Block(Protocol):
    """A configured model with explicit, orchestrator-owned sampling state.

    Keep fixed model configuration on the instance. Put every changing parameter,
    walker, tuning value, and continuation record in the returned BlockResult.
    Use the supplied RNG; Wheel owns it and commits its progress with the result.
    A single configured block can therefore participate in independent Wheels.

    Prepare a complete `initial_block_result` in campaign setup and supply it to
    `Wheel.add`. It must contain consistent estimates and all state needed to
    sample. Initialization can use an injection, a chosen estimate, or a separate
    model-specific prior-drawing helper. Wheel never calls an initialization hook.
    `sample` advances `current_block_result`
    against `conditional_residual` and `noise_covariance`, returning a complete
    new BlockResult. A call may perform several internal sampling steps.

    `conditional_residual` is the observed data minus every OTHER block's current
    signal; this block's signal is still present. The block does not perform
    cross-block subtraction or add-back. `noise_covariance` is the current
    covariance or None. `current_block_result` is this block's latest accepted
    result from the ledger, including `model_parameters` and `sampler_state`.
    Model parameters describe the physical estimates; sampler state preserves
    algorithm internals such as walkers and proposal tuning. A simple sampler
    may leave sampler_state empty.

    Signal-only, noise-only, and joint models all use this interface. Return
    the summed TDI signal in `BlockResult.tdi_signal_contribution` and a replacement
    DataCovariance in `BlockResult.noise_covariance`. Both estimates are optional.
    A noise-only block returns `BlockResult(noise_covariance=covariance, ...)`.
    A None signal means zero contribution, replacing any previous signal.
    The current covariance owner must return its complete covariance every call;
    other blocks can omit covariance without changing it. Wheel keeps one
    covariance owner; publish combined noise components together.

    Each call receives independent snapshots of the conditional residual,
    covariance, and current result. The orbit is a shared immutable resource.
    Register a preferred data_domain with Wheel.add to receive matching inputs.
    A translated covariance exposes exact operators; it may correlate samples.
    Validate unsupported conventions explicitly.

    This protocol checks method shape only. It cannot prove that user code is
    stateless or that its sampling transitions are statistically correct.
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
        current_block_result: BlockResult,
        *,
        rng: np.random.Generator,
    ) -> BlockResult:
        """Advance the supplied state and return the complete new block result."""
        ...
