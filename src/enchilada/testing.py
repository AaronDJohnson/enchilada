"""Helpers for checking the stateless Wheel/Block protocol end to end."""

import numpy as np

from enchilada.block import Block
from enchilada.block_result import BlockResult
from enchilada.covariance import DataCovariance
from enchilada.data import L1Data
from enchilada.translated_covariance import TranslatedCovariance


class EchoBlock:
    """No-op block that prints its input and carries its counter in the BlockResult.

    The instance holds only a name. Register it with an explicit initial result,
    such as ``BlockResult(sampler_state={"num_sample_calls": 0})``. Independent
    Wheels own independent counters; inspect ``wheel.ledger[name].sampler_state``.
    """

    def __init__(self, name: str):
        self.name = name

    def sample(
        self,
        conditional_residual: L1Data,
        noise_covariance: DataCovariance | TranslatedCovariance | None,
        current_block_result: BlockResult,
        *,
        rng: np.random.Generator,
    ) -> BlockResult:
        """Print the conditional residual RMS and return the advanced counter."""
        first_channel_name = conditional_residual.channel_names[0]
        residual_rms = float(
            np.sqrt(
                np.mean(
                    np.abs(conditional_residual.channel_data[first_channel_name]) ** 2
                )
            )
        )
        num_sample_calls = current_block_result.sampler_state["num_sample_calls"]
        print(
            f"[{self.name}] sample {num_sample_calls}: "
            f"residual RMS on {first_channel_name!r} = {residual_rms:.4e}"
        )
        return BlockResult(sampler_state={"num_sample_calls": num_sample_calls + 1})


def check_block(
    block_under_test: Block,
    observed_data: L1Data,
    num_cycles: int = 2,
    *,
    initial_block_result: BlockResult,
    initial_noise_covariance: DataCovariance | TranslatedCovariance | None = None,
    random_seed: int | None = 0,
) -> None:
    """Run a block on a scratch Wheel and validate each returned BlockResult.

    Adopts the required ``initial_block_result``, followed by ``num_cycles``
    calls to ``sample``,
    including the parameter/state round trip. Wheel checks the signal grid and
    any published DataCovariance. The supplied block should retain configuration
    only, so this check can precede its registration in a scientific campaign.

    Args:
        block_under_test: Configured block whose exchange contract is checked.
        observed_data: L1 observations and shared run settings for the scratch run.
        num_cycles: Number of sampling cycles after initialization; defaults to 2.
        initial_block_result: Required complete starting result on the observation
            grid, including the continuation state needed by sample. Cannot be None.
        initial_noise_covariance: Starting covariance for the scratch Wheel.
            None leaves covariance unset. A noise block may supply covariance
            during the check.
        random_seed: Seed for the scratch Wheel's NumPy generator. Defaults to
            0 for reproducibility; None requests nondeterministic initialization.

    This checks the protocol, not statistical correctness or whether hidden
    sampler state has been left on the block. Test posterior recovery and
    independent campaigns separately; ``examples/toy_fit.py`` provides a pattern.
    """
    from enchilada.wheel import Wheel

    wheel = Wheel(
        observed_data=observed_data,
        initial_noise_covariance=initial_noise_covariance,
        random_seed=random_seed,
    )
    wheel.add(
        block_to_register=block_under_test,
        initial_block_result=initial_block_result,
    )
    wheel.run(num_cycles=num_cycles)
