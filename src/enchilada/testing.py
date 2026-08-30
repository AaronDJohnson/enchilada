"""Helpers for sanity-checking the enchilada interface.

Importable utilities for verifying that the Wheel/Block protocol works
end-to-end without needing a real waveform model:

- :class:`EchoBlock` -- a no-op block that prints what the Wheel hands
  it; useful for watching the plumbing.
- :func:`check_block` -- a conformance check for your own Block
  implementation; run it in your test suite before plugging into a shared
  campaign.
"""

import numpy as np

from enchilada.block import Block
from enchilada.data import L1Data
from enchilada.template import Template


class EchoBlock:
    """No-op block that prints what the Wheel passes to it.

    Claims nothing (its template is zero), so other blocks are unaffected.
    Keeps its own update counter as internal state -- like any real block,
    its internals are its own business. Useful for verifying that the Gibbs
    ring wires blocks together correctly and that residual metadata survives
    the round trip.
    """

    def __init__(self, name: str):
        self.name = name
        self.updates = 0

    def start(self, residual: L1Data) -> Template:
        """Print the run settings the Wheel handed over; return a zero template.

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
        return residual.zero_template()

    def update(self, residual: L1Data) -> Template:
        """Print the RMS of the residual handed over; return a zero template.

        Because the template stays zero, the ledger entry stays zero and no
        other block is affected -- so the RMS you see is the residual as it
        stands with every other block's current template subtracted.
        """
        ch0 = residual.channels[0]
        # abs() so the RMS is real for frequency-domain (complex) data too
        rms = float(np.sqrt(np.mean(np.abs(residual.tdi[ch0]) ** 2)))
        print(
            f"[{self.name}] update {self.updates}: residual RMS on {ch0!r} = {rms:.4e}"
        )
        self.updates += 1
        return residual.zero_template()


def check_block(block: Block, observed: L1Data, n_cycles: int = 2) -> None:
    """Conformance check for a `Block` implementation.

    Drives the full Wheel protocol against `observed` on a scratch Wheel and
    verifies:

    - `start` returns a `Template` on the run's grid (right channels, right
      lengths, real or complex to match the domain);
    - each of `n_cycles` block updates does the same (mid-run drift raises);
    - if the block publishes a noise model (a noise block), that model
      satisfies the consumption contract -- `residual.noise_psd` succeeds
      rather than raising for want of a `psd` method.

    A block still written for the pre-template contract -- returning the
    residual with its template subtracted -- fails the first check with a
    `TypeError`, here rather than in a shared campaign.

    It does not check the residual bookkeeping -- the `Wheel` owns that, so
    there is no cross-block arithmetic in a block to get wrong (see the
    `Wheel` docstring for the ledger). Whether your *sampler* recovers truth
    is still yours to verify; `examples/toy_fit.py` is the pattern.

    Raises with a pointed message at the first violation; returns quietly
    when the block conforms. Run this in your own test suite before
    plugging a block into a shared campaign:

        from enchilada.testing import check_block
        check_block(MyBlock(name="ucb"), toy_observed)
    """
    from enchilada.wheel import Wheel

    wheel = Wheel(observed)
    wheel.add(block)  # start(): validated for a well-formed template
    wheel.run(n_cycles)  # update() x n_cycles: each return validated

    # if this block threads a noise model, it must satisfy the contract signal
    # blocks consume it through: a callable psd(freqs[, channel])
    result = wheel.residual()
    noise = result.noise
    if noise is not None and noise is not observed.noise:
        result.noise_psd()  # exercises psd; raises TypeError if it is missing
