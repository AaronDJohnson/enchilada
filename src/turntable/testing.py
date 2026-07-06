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

    Subtracts nothing (its model is zero), so it returns the residual it was
    handed unchanged and other segments are unaffected. Keeps its own step
    counter as internal state -- like any real segment, its internals are
    its own business. Useful for verifying that the Gibbs ring wires
    segments together correctly and that residual metadata survives the
    round trip.
    """

    def __init__(self, name: str):
        self.name = name
        self.steps = 0

    def start(self, residual: Residuals) -> Residuals:
        print(
            f"[{self.name}] start: "
            f"fs={residual.fs} Hz, Tobs={residual.Tobs:.1f} s, "
            f"N={residual.N}, channels={residual.channels}, "
            f"observable={residual.observable!r}, domain={residual.domain!r}"
        )
        return residual

    def step(self, residual: Residuals) -> Residuals:
        ch0 = residual.channels[0]
        # abs() so the RMS is real for frequency-domain (complex) data too
        rms = float(np.sqrt(np.mean(np.abs(residual.tdi[ch0]) ** 2)))
        print(
            f"[{self.name}] step {self.steps}: "
            f"residual RMS on {ch0!r} = {rms:.4e}"
        )
        self.steps += 1
        return residual


def check_segment(
    segment: Segment, observed: Residuals, n_sweeps: int = 2
) -> None:
    """Conformance check for a `Segment` implementation.

    Drives the full Wheel protocol against `observed` on a scratch Wheel and
    verifies:

    - `start` returns a valid `Residuals` that keeps every run setting
      unchanged (only `tdi`/`noise` may move);
    - each of `n_sweeps` `step` calls does the same (mid-run drift raises);
    - if the segment sets a noise model on the residual (a noise segment),
      that model satisfies the consumption contract -- `residual.noise_psd`
      succeeds rather than raising for want of a `psd`/`wdm_variance` method.

    What it does *not* check is the add-back (that `step` re-adds your own
    previous model before sampling): a forgotten add-back produces a
    perfectly well-formed residual, so no generic check can catch it without
    risking false positives on correct segments. Guard it yourself with a
    known-truth recovery test; `examples/toy_fit.py` is the reference
    pattern.

    Raises with a pointed message at the first violation; returns quietly
    when the segment conforms. Run this in your own test suite before
    plugging a segment into a shared campaign:

        from turntable.testing import check_segment
        check_segment(MySegment(name="ucb"), toy_observed)
    """
    from turntable.wheel import Wheel  # local import: avoid circularity

    wheel = Wheel(observed)
    wheel.add(segment)  # start(): validated for a well-formed returned residual
    wheel.run(n_sweeps)  # step() x n_sweeps: each return validated

    # if this segment threads a noise model, it must satisfy the contract signal
    # segments consume it through -- at least ONE of psd / wdm_variance
    result = wheel.residual()
    noise = result.noise
    if noise is not None and noise is not observed.noise:
        has_psd = callable(getattr(noise, "psd", None))
        has_wdm = callable(getattr(noise, "wdm_variance", None))
        if not (has_psd or has_wdm):
            raise TypeError(
                f"{segment.name} put a noise model ({type(noise).__name__}) on the "
                f"residual that exposes neither psd(freqs[, channel]) nor "
                f"wdm_variance(n_layers, n_time, dt, epoch[, channel]); signal "
                f"segments consume it through Residuals.noise_psd / "
                f"noise_wdm_variance, so it must implement at least one "
                f"(see segment.NoiseSegment)"
            )
        if has_psd:
            result.noise_psd()  # exercise the real frequency-domain consumption
