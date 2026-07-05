"""Helpers for sanity-checking the turntable interface.

Importable utilities for verifying that the Wheel/Segment protocol works
end-to-end without needing a real waveform model:

- :class:`EchoSegment` -- a no-op segment that prints what the Wheel hands
  it; useful for watching the plumbing.
- :func:`check_segment` -- a conformance check for your own Segment
  implementation; run it in your test suite before plugging into a shared
  campaign.
"""

import numpy as np

from turntable.residuals import Residuals
from turntable.segment import Segment


class EchoSegment:
    """No-op segment that prints what the Wheel passes to it.

    Contributes zeros, so other segments' residuals are unaffected. Keeps
    its own step counter as internal state -- like any real segment, its
    internals are its own business. Useful for verifying that the Gibbs
    loop wires segments together correctly and that residual metadata
    survives the round trip.
    """

    def __init__(self, name: str):
        self.name = name
        self.steps = 0
        # zero templates matching the observed representation (time- or
        # frequency-domain shapes and dtypes), captured at start
        self._zeros: dict[str, np.ndarray] | None = None

    def start(self, observed: Residuals):
        self._zeros = {
            ch: np.zeros_like(observed.tdi[ch]) for ch in observed.channels
        }
        print(
            f"[{self.name}] start: "
            f"fs={observed.fs} Hz, Tobs={observed.Tobs:.1f} s, "
            f"N={observed.N}, channels={observed.channels}, "
            f"observable={observed.observable!r}, domain={observed.domain!r}"
        )
        return {ch: arr.copy() for ch, arr in self._zeros.items()}

    def step(self, residual: Residuals):
        ch0 = residual.channels[0]
        # abs() so the RMS is real for frequency-domain (complex) data too
        rms = float(np.sqrt(np.mean(np.abs(residual.tdi[ch0]) ** 2)))
        print(
            f"[{self.name}] step {self.steps}: "
            f"residual RMS on {ch0!r} = {rms:.4e}"
        )
        self.steps += 1
        assert self._zeros is not None
        return {ch: arr.copy() for ch, arr in self._zeros.items()}


def check_segment(
    segment: Segment, observed: Residuals, n_sweeps: int = 2
) -> None:
    """Conformance check for a `Segment` implementation.

    Drives the full Wheel protocol against `observed` on a scratch Wheel:
    registration (`start` + contribution validation), `n_sweeps` Gibbs
    sweeps with per-step contribution re-validation, and -- for noise
    segments -- the `noise_model` contract. Additionally checks that `step`
    does not mutate the residual it is handed (the Wheel defends itself
    with copies, but a mutating segment is relying on that defense instead
    of the contract).

    Raises with a pointed message at the first violation; returns quietly
    when the segment conforms. Run this in your own test suite before
    plugging a segment into a shared campaign:

        from turntable.testing import check_segment
        check_segment(MySegment(name="ucb"), toy_observed)
    """
    from turntable.wheel import Wheel  # local import: avoid circularity

    wheel = Wheel(observed)
    wheel.add(segment)  # validates name, initial contribution, noise contract
    wheel.run(n_sweeps)

    name = segment.name

    # one out-of-band step: the contribution must validate, and the input
    # residual must come back untouched
    residual = wheel.residual(exclude=name)
    before = {ch: arr.copy() for ch, arr in residual.tdi.items()}
    contribution = segment.step(residual)
    wheel._validate_contribution(contribution, name, "step")
    for ch, arr in before.items():
        # equal_nan: gap-filled data may legitimately contain NaNs
        if not np.array_equal(residual.tdi[ch], arr, equal_nan=True):
            raise ValueError(
                f"{name}.step mutated residual.tdi[{ch!r}] in place; segments "
                f"must treat the residual as read-only and return their model "
                f"through their contribution (the Wheel hands out copies, but "
                f"other orchestration contexts may not)"
            )
