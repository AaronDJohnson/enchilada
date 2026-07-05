from collections.abc import Callable
from dataclasses import replace

import numpy as np

from turntable.residuals import Residuals
from turntable.segment import Contribution, Segment


class Wheel:
    """Passes residuals between registered segments. That's the whole job.

    The Wheel holds the observed data and one thing per segment: its latest
    TDI contribution -- the residual ledger it needs to hand each segment
    the data with every *other* segment's model subtracted. Everything else
    about a segment (parameters, RNG, chains, checkpoints) lives inside the
    segment itself; the Wheel never sees it.

    Consistency checking. The Wheel enforces the protocol contract at every
    boundary: `add` validates the whole registration before mutating any
    state (a segment that fails validation is not half-registered),
    contributions are shape-checked against the observed data both at
    registration and after *every* step (a mid-run drift raises instead of
    silently broadcast-corrupting other segments' residuals), and a noise
    segment's `noise_model` must return an object satisfying the noise
    contract (`psd(freqs[, channel])` and/or `wdm_variance(...)` -- see
    `segment.NoiseSegment`).

    Noise. The residual each segment sees carries the current noise model on
    `residual.noise`. Two ways to supply it:

    * Fixed noise -- set it once on the observed data
      (`observed = replace(observed, noise=...)`); the Wheel threads that one
      model to every segment and never changes it. Register no noise segment.
    * Sampled noise -- register a noise segment (a Segment that also defines
      `noise_model`); the Wheel threads its current model and refreshes it
      after every noise step, overriding any fixed `observed.noise`. A segment
      only sees the noise model in `start` if it is registered *after* the
      noise segment (before it, `observed.noise` is still the fixed model or
      `None`); order does not matter once `run` starts, since `step` always
      sees the current model. Add the noise segment first if a signal
      segment seeds itself from the noise estimate in `start`.

    Typical use:

        observed = Residuals(tdi=..., sample_rate=..., n_samples=...,
                             channels=("A", "E", "T"),
                             tdi_generation="2.0",
                             observable="fractional_frequency",
                             epoch=0.0,
                             noise=fixed_noise_model)  # optional fixed noise
        wheel = Wheel(observed)
        wheel.add(ucb_segment)
        wheel.add(mbhb_segment)
        wheel.add(noise_segment)  # optional; overrides the fixed noise
        wheel.run(n_iterations=1000)

    The current full residual is available via `wheel.residual()`. For
    anything about a segment's internals -- its current parameters, its
    chain -- ask the segment object itself; you constructed it, you hold it.
    """

    def __init__(self, observed: Residuals):
        """Args:
            observed: TDI data with the run settings attached. Every
                residual the Wheel produces inherits these settings.
        """
        self.observed = observed
        self._segments: list[Segment] = []
        # The residual ledger: each segment's latest contribution, exactly
        # what is needed to build "data minus everyone else".
        self._contributions: dict[str, Contribution] = {}
        # The noise model threaded onto every residual. Defaults to the fixed
        # `observed.noise` (may be None); a registered noise segment (one that
        # defines `noise_model`) overrides it and refreshes it each step.
        self._noise_segment: str | None = None
        self._noise = observed.noise

    def add(self, segment: Segment) -> None:
        """Register a segment: call its `start` and record its contribution.

        All validation happens before any Wheel state changes, so a failed
        `add` leaves the Wheel exactly as it was. `start` receives the
        observed data with the *current* noise model threaded on
        `observed.noise` (i.e. a previously registered noise segment's model
        overrides any fixed noise, as documented on the class).
        """
        name = segment.name
        if not isinstance(name, str) or not name:
            raise ValueError(f"segment name must be a non-empty string, got {name!r}")
        if name in self._contributions:
            raise ValueError(f"segment name {name!r} already registered")
        is_noise = hasattr(segment, "noise_model")
        if is_noise and self._noise_segment is not None:
            raise ValueError(
                f"noise segment {name!r} cannot be registered: "
                f"{self._noise_segment!r} already provides the noise model "
                "(a Wheel supports at most one noise segment)"
            )
        # Hand over copies of the data arrays: a segment that mutates what it
        # was given (in any language wrapper) must not corrupt the Wheel's
        # observed data, which is the ground truth every residual is built from.
        contribution = segment.start(
            replace(
                self.observed,
                tdi={ch: arr.copy() for ch, arr in self.observed.tdi.items()},
                noise=self._noise,
            )
        )
        self._validate_contribution(contribution, name, "start")
        if is_noise:
            noise = segment.noise_model()
            self._validate_noise_model(noise, name)
        # every check passed -- commit atomically
        self._segments.append(segment)
        self._contributions[name] = contribution
        if is_noise:
            self._noise_segment = name
            self._noise = noise

    def run(
        self,
        n_iterations: int,
        on_sweep: Callable[[int, "Wheel"], None] | None = None,
    ) -> None:
        """Drive the Gibbs loop for `n_iterations` sweeps over all segments.

        Each segment's contribution is re-validated after every step, so a
        contribution whose shape or channels drift mid-run raises
        immediately instead of corrupting other segments' residuals.

        The Wheel stores nothing about a segment beyond its latest
        contribution: chains, checkpoints, and diagnostics are the
        segment's own business.

        Args:
            n_iterations: Number of full sweeps over all segments.
            on_sweep: Optional progress/checkpoint hook, called as
                `on_sweep(iteration, self)` after each completed sweep
                (`iteration` counts from 0). Read `residual()` off the
                wheel -- or anything you like off your own segment
                objects -- to log, plot, or checkpoint; the hook must not
                mutate the wheel. Equivalent to calling `run(1)` in your
                own loop -- the Gibbs chain is identical either way.
        """
        if (
            not isinstance(n_iterations, (int, np.integer))
            or isinstance(n_iterations, bool)
            or n_iterations < 0
        ):
            raise ValueError(
                f"n_iterations must be a non-negative integer, got {n_iterations!r}"
            )
        for iteration in range(n_iterations):
            for seg in self._segments:
                contribution = seg.step(self.residual(exclude=seg.name))
                self._validate_contribution(contribution, seg.name, "step")
                if seg.name == self._noise_segment:
                    noise = seg.noise_model()
                    self._validate_noise_model(noise, seg.name)
                    self._noise = noise
                self._contributions[seg.name] = contribution
            if on_sweep is not None:
                on_sweep(iteration, self)

    def residual(self, exclude: str | None = None) -> Residuals:
        """The observed data minus registered segments' current contributions.

        With no argument, every segment's contribution is subtracted -- the
        full residual, useful for convergence checks and diagnostics. Pass
        `exclude` to leave that one segment's contribution in the data:
        `residual(exclude="ucb")` is exactly what the "ucb" segment sees in
        `step`. The current noise model rides on `.noise`.
        """
        if exclude is not None and exclude not in self._contributions:
            raise ValueError(
                f"unknown segment {exclude!r}; registered: "
                f"{sorted(self._contributions)}"
            )
        tdi = {ch: self.observed.tdi[ch].copy() for ch in self.observed.channels}
        for other, contribution in self._contributions.items():
            if other == exclude:
                continue
            for ch in tdi:
                tdi[ch] -= contribution[ch]
        return replace(self.observed, tdi=tdi, noise=self._noise)

    def _validate_contribution(
        self, arrays: object, segment_name: str, method: str
    ) -> None:
        if not isinstance(arrays, dict):
            raise TypeError(
                f"{segment_name}.{method} must return a contribution "
                f"(dict of channel -> array), got {type(arrays).__name__}"
            )
        missing = set(self.observed.channels) - arrays.keys()
        if missing:
            raise ValueError(
                f"{segment_name}.{method} contribution missing channels: "
                f"{sorted(missing)}"
            )
        for ch in self.observed.channels:
            # A contribution must match the observed data's representation:
            # the arrays in observed.tdi are length n_samples in the time
            # domain or n_samples // 2 + 1 on the rfft grid in the frequency
            # domain (see Residuals.domain). Validating against observed.tdi
            # keeps the Wheel agnostic to which one this run uses; the
            # residual subtraction is a plain elementwise op for either.
            expected = self.observed.tdi[ch].shape
            if arrays[ch].shape != expected:
                raise ValueError(
                    f"{segment_name}.{method} contribution[{ch!r}] has shape "
                    f"{arrays[ch].shape}, expected {expected} "
                    f"({self.observed.domain}-domain run)"
                )

    def _validate_noise_model(self, model: object, segment_name: str) -> None:
        if model is None:
            raise ValueError(
                f"{segment_name}.noise_model returned None; it must return the "
                f"noise model implied by the segment's current state"
            )
        if not (
            callable(getattr(model, "psd", None))
            or callable(getattr(model, "wdm_variance", None))
        ):
            raise TypeError(
                f"{segment_name}.noise_model returned {type(model).__name__}, which "
                f"exposes neither psd(freqs[, channel]) nor "
                f"wdm_variance(n_layers, n_time, dt, epoch[, channel]); signal "
                f"segments consume the noise model through Residuals.noise_psd / "
                f"noise_wdm_variance, so it must implement at least one "
                f"(see segment.NoiseSegment)"
            )
