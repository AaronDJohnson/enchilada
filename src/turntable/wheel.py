from dataclasses import replace

import numpy as np

from turntable.residuals import Residuals
from turntable.segment import Catalog, Segment, State


class Wheel:
    """Orchestrates a blocked-Gibbs global fit across registered segments.

    The Wheel holds the observed data and each segment's current catalog.
    On every iteration it visits each segment, hands it the data with every
    other segment's current model subtracted, lets it sample, and caches
    the new contribution for the next visit.

    Noise. The residual each segment sees carries the current noise model on
    `residual.noise`. Two ways to supply it:

    * Fixed noise -- set it once on the observed data
      (`observed = replace(observed, noise=...)`); the Wheel threads that one
      model to every segment and never changes it. Register no noise segment.
    * Sampled noise -- register a noise segment (a Segment that also defines
      `noise_model`); the Wheel threads its current model and refreshes it
      after every noise step, overriding any fixed `observed.noise`.

    Typical use:

        observed = Residuals(tdi=..., sample_rate=..., n_samples=...,
                             channels=("A", "E", "T"),
                             tdi_generation="2.0", epoch=0.0,
                             noise=fixed_noise_model)  # optional fixed noise
        wheel = Wheel(observed)
        wheel.add(ucb_segment)
        wheel.add(mbhb_segment)
        wheel.add(noise_segment)  # optional; overrides the fixed noise
        wheel.run(n_iterations=1000)

    Catalogs and states are accessible via `wheel.catalog(name)` and
    `wheel.state(name)` for checkpointing or post-hoc analysis.
    """

    def __init__(self, observed: Residuals):
        """Args:
            observed: TDI data with the run settings attached. Every
                residual the Wheel produces inherits these settings.
        """
        self.observed = observed
        self._segments: list[Segment] = []
        self._catalogs: dict[str, Catalog] = {}
        self._states: dict[str, State] = {}
        self._renders: dict[str, dict[str, np.ndarray]] = {}
        # The noise model threaded onto every residual. Defaults to the fixed
        # `observed.noise` (may be None); a registered noise segment (one that
        # defines `noise_model`) overrides it and refreshes it each step.
        self._noise_segment: str | None = None
        self._noise = observed.noise

    def add(self, segment: Segment) -> None:
        """Register a segment. Calls its `initial_state` and caches its
        starting contribution."""
        if segment.name in self._catalogs:
            raise ValueError(f"segment name {segment.name!r} already registered")
        catalog, state = segment.initial_state(self.observed)
        render = segment.render(catalog)
        self._validate_render(render, segment.name)
        self._segments.append(segment)
        self._catalogs[segment.name] = catalog
        self._states[segment.name] = state
        self._renders[segment.name] = render
        if hasattr(segment, "noise_model"):
            if self._noise_segment is not None:
                raise ValueError(
                    f"noise segment {segment.name!r} cannot be registered: "
                    f"{self._noise_segment!r} already provides the noise model "
                    "(a Wheel supports at most one noise segment)"
                )
            self._noise_segment = segment.name
            self._noise = segment.noise_model(catalog)

    def run(self, n_iterations: int) -> None:
        """Drive the Gibbs loop for `n_iterations` sweeps over all segments."""
        for _ in range(n_iterations):
            for seg in self._segments:
                residual = self._residual_excluding(seg.name)
                catalog, state = seg.step(residual, self._states[seg.name])
                self._catalogs[seg.name] = catalog
                self._states[seg.name] = state
                self._renders[seg.name] = seg.render(catalog)
                if seg.name == self._noise_segment:
                    self._noise = seg.noise_model(catalog)

    def catalog(self, name: str) -> Catalog:
        """Current catalog from the named segment."""
        return self._catalogs[name]

    def state(self, name: str) -> State:
        """Current internal state of the named segment."""
        return self._states[name]

    def _residual_excluding(self, name: str) -> Residuals:
        tdi = {ch: self.observed.tdi[ch].copy() for ch in self.observed.channels}
        for other, render in self._renders.items():
            if other == name:
                continue
            for ch in tdi:
                tdi[ch] -= render[ch]
        return replace(self.observed, tdi=tdi, noise=self._noise)

    def _validate_render(
        self, arrays: dict[str, np.ndarray], segment_name: str
    ) -> None:
        missing = set(self.observed.channels) - arrays.keys()
        if missing:
            raise ValueError(
                f"{segment_name}.render missing channels: {sorted(missing)}"
            )
        for ch in self.observed.channels:
            if arrays[ch].shape != (self.observed.n_samples,):
                raise ValueError(
                    f"{segment_name}.render[{ch!r}] has shape "
                    f"{arrays[ch].shape}, expected ({self.observed.n_samples},)"
                )
