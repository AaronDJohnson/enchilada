"""Helpers for checking the Wheel/Block exchange contract end to end."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from enchilada.block import Block
from enchilada.block_state import BlockState
from enchilada.covariance import DataCovariance
from enchilada.data import L1Data
from enchilada.translated_covariance import TranslatedCovariance


class EchoBlock:
    """Zero-signal diagnostic block that prints residual RMS and advances a counter.

    Register with ``BlockState()`` and inspect ``block.state.num_sample_calls``
    for the counter. Diagnostics belong to ``block.state.metadata``. Each call
    publishes an empty physical state. Use a separate instance per campaign.
    """

    @dataclass
    class State:
        """Mutable continuation state and diagnostics owned by this block."""

        num_sample_calls: int = 0
        metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(self, name: str):
        self.name = name
        self.state = self.State()

    def sample(
        self,
        conditional_residual: L1Data,
        noise_covariance: DataCovariance | TranslatedCovariance | None,
        current_block_state: BlockState,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Print the residual RMS, advance the local counter, and publish no signal."""
        first_channel_name = conditional_residual.channel_names[0]
        residual_rms = float(
            np.sqrt(
                np.mean(
                    np.abs(conditional_residual.channel_data[first_channel_name]) ** 2
                )
            )
        )
        num_sample_calls = self.state.num_sample_calls
        print(
            f"[{self.name}] sample {num_sample_calls}: "
            f"residual RMS on {first_channel_name!r} = {residual_rms:.4e}"
        )
        self.state.num_sample_calls += 1
        return BlockState()


def check_block(
    block_under_test: Block,
    observed_data: L1Data,
    num_cycles: int = 2,
    *,
    initial_block_state: BlockState,
    initial_noise_covariance: DataCovariance | TranslatedCovariance | None = None,
    random_seed: int | None = 0,
) -> None:
    """Run a block on a scratch Wheel and validate each returned BlockState.

    Adopts the required ``initial_block_state``, then calls ``sample`` for
    ``num_cycles``, passing accepted state into each call. Wheel validates the
    signal grid and any published DataCovariance or TranslatedCovariance.

    Args:
        block_under_test: Configured block whose exchange contract is checked.
        observed_data: L1 observations and shared run settings for the scratch run.
        num_cycles: Number of sampling cycles after initialization; defaults to 2.
        initial_block_state: Required starting physical estimates on the observation
            grid. Initialize any sampler continuation state on the block itself.
        initial_noise_covariance: Starting DataCovariance or TranslatedCovariance
            for the scratch Wheel. None leaves it unset until a block supplies it.
        random_seed: Seed for the scratch Wheel's NumPy generator. Defaults to
            0 for reproducibility; None requests nondeterministic initialization.

    This advances the supplied block's local state. It checks the exchange
    contract, not statistical correctness or block-local failure recovery.
    Test posterior recovery and independent campaigns separately;
    ``examples/toy_fit.py`` provides a pattern.
    """
    from enchilada.wheel import Wheel

    wheel = Wheel(
        observed_data=observed_data,
        initial_noise_covariance=initial_noise_covariance,
        random_seed=random_seed,
    )
    wheel.add(
        block_to_register=block_under_test,
        initial_block_state=initial_block_state,
    )
    wheel.run(num_cycles=num_cycles)
