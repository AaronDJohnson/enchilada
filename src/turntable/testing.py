"""Helpers for sanity-checking the turntable interface.

Importable utilities for verifying that the Wheel/Segment protocol works
end-to-end without needing a real waveform model.
"""

import numpy as np

from turntable.residuals import Residuals


class EchoSegment:
    """No-op segment that prints what the Wheel passes to it.

    Returns a zero contribution, so other segments' residuals are unaffected.
    Useful for verifying that the Gibbs loop wires segments together
    correctly and that residual metadata survives the round trip.
    """

    def __init__(self, name: str):
        self.name = name
        # zero templates matching the observed representation (time- or
        # frequency-domain shapes and dtypes), captured at registration
        self._zeros: dict[str, np.ndarray] | None = None

    def initial_state(self, observed: Residuals):
        self._zeros = {
            ch: np.zeros_like(observed.tdi[ch]) for ch in observed.channels
        }
        print(
            f"[{self.name}] initial_state: "
            f"fs={observed.fs} Hz, Tobs={observed.Tobs:.1f} s, "
            f"N={observed.N}, channels={observed.channels}, "
            f"observable={observed.observable!r}, domain={observed.domain!r}"
        )
        return ([], {"step": 0})

    def step(self, residual: Residuals, state):
        ch0 = residual.channels[0]
        # abs() so the RMS is real for frequency-domain (complex) data too
        rms = float(np.sqrt(np.mean(np.abs(residual.tdi[ch0]) ** 2)))
        print(
            f"[{self.name}] step {state['step']}: "
            f"residual RMS on {ch0!r} = {rms:.4e}"
        )
        return ([], {"step": state["step"] + 1})

    def render(self, catalog):
        assert self._zeros is not None
        return {ch: arr.copy() for ch, arr in self._zeros.items()}
