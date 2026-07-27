"""Helpers for sanity-checking the turntable interface.

Importable utilities for verifying that the Wheel/Block protocol works
end-to-end without needing a real waveform model:

- :class:`EchoBlock` -- a no-op block that prints what the Wheel hands
  it; useful for watching the plumbing.
- :func:`check_block` -- a conformance check for your own Block
  implementation; run it in your test suite before plugging into a shared
  campaign.
"""

import numpy as np

from turntable.block import Block
from turntable.residuals import Residuals


class EchoBlock:
    """No-op block that prints what the Wheel passes to it.

    Subtracts nothing (its model is zero), so it returns the residual it was
    handed unchanged and other blocks are unaffected. Keeps its own update
    counter as internal state -- like any real block, its internals are
    its own business. Useful for verifying that the Gibbs ring wires
    blocks together correctly and that residual metadata survives the
    round trip.
    """

    def __init__(self, name: str):
        self.name = name
        self.updates = 0

    def start(self, residual: Residuals) -> Residuals:
        """Print the run settings the Wheel handed over, and subtract nothing.

        Registration is where a real block would read the conventions off the
        residual and set its sampler up; this one just shows you what was
        available at that moment.
        """
        print(
            f"[{self.name}] start: "
            f"fs={residual.fs} Hz, Tobs={residual.Tobs:.1f} s, "
            f"N={residual.N}, channels={residual.channels}, "
            f"observable={residual.observable!r}, domain={residual.domain!r}"
        )
        return residual

    def update(self, residual: Residuals) -> Residuals:
        """Print the RMS of the residual handed over, and return it unchanged.

        Because the model stays zero, the ledger entry stays zero and no other
        block is affected -- so the RMS you see is the residual as it stands
        after every other block has subtracted its model this cycle.
        """
        ch0 = residual.channels[0]
        # abs() so the RMS is real for frequency-domain (complex) data too
        rms = float(np.sqrt(np.mean(np.abs(residual.tdi[ch0]) ** 2)))
        print(
            f"[{self.name}] update {self.updates}: residual RMS on {ch0!r} = {rms:.4e}"
        )
        self.updates += 1
        return residual


def check_block(block: Block, observed: Residuals, n_cycles: int = 2) -> None:
    """Conformance check for a `Block` implementation.

    Drives the full Wheel protocol against `observed` on a scratch Wheel and
    verifies:

    - `start` returns a valid `Residuals` that keeps every run setting
      unchanged (only `tdi`/`noise` may move);
    - each of `n_cycles` block updates does the same (mid-run drift raises);
    - if the block sets a noise model on the residual (a noise block),
      that model satisfies the consumption contract -- `residual.noise_psd`
      succeeds rather than raising for want of a `psd` method.

    It also escalates the `ModelWithdrawnWarning` the Wheel merely warns about,
    so a block that stops re-subtracting its model fails here rather than
    quietly leaving the fit later. (If your block legitimately drops to a
    zero model during these cycles -- a death move -- give it a starting state
    that does not, or filter the warning around your own call.)

    It does not check the residual bookkeeping -- the `Wheel` owns that, so
    there is no cross-block arithmetic in a block to get wrong (see the
    `Wheel` docstring for the ledger). Whether your *sampler* recovers truth
    is still yours to verify; `examples/toy_fit.py` is the pattern.

    Raises with a pointed message at the first violation; returns quietly
    when the block conforms. Run this in your own test suite before
    plugging a block into a shared campaign:

        from turntable.testing import check_block
        check_block(MyBlock(name="ucb"), toy_observed)
    """
    import warnings

    from turntable.wheel import ModelWithdrawnWarning, Wheel

    wheel = Wheel(observed)
    # The Wheel only *warns* when a model is withdrawn, because mid-run it
    # cannot tell a legitimate death move from a forgotten re-subtraction. A
    # pre-campaign conformance check is exactly where that heuristic should be
    # strict, so escalate it here.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ModelWithdrawnWarning)
        wheel.add(block)  # start(): validated for a well-formed residual
        wheel.run(n_cycles)  # update() x n_cycles: each return validated

    # if this block threads a noise model, it must satisfy the contract signal
    # blocks consume it through: a callable psd(freqs[, channel])
    result = wheel.residual()
    noise = result.noise
    if noise is not None and noise is not observed.noise:
        result.noise_psd()  # exercises psd; raises TypeError if it is missing
