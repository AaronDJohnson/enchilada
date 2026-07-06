from collections.abc import Callable
from dataclasses import replace
from typing import ClassVar

import numpy as np

from turntable.residuals import Residuals
from turntable.segment import Segment


class Wheel:
    """Passes one residual around a ring of segments. That's the whole job.

    The Wheel holds a single running residual and nothing else about the
    fit. On every visit it hands the current residual to a segment and takes
    back the segment's new residual -- it does no arithmetic and knows
    nothing about any segment's model, parameters, chain, or noise. Blocked
    Gibbs falls out of the ring: each segment adds its own model back,
    resamples, and subtracts the new one (see `segment.Segment`), so the
    residual handed to the next segment already reflects every update so far.

    Everything about a segment -- its model, its sampler state, its noise
    model -- lives inside the segment that owns it. The Wheel keeps only
    `observed` (the pristine input, for reference) and the running
    `residual()`.

    Consistency checking. The Wheel validates each segment fully before it
    changes any state (a segment whose `start` fails is not registered), and
    after every `start`/`step` it checks the returned object is a `Residuals`
    that kept the run settings unchanged -- only `tdi` and `noise` may move.
    `Residuals` itself validates that the returned `tdi` keeps the right
    channels and shapes, so a mid-run drift raises immediately instead of
    silently corrupting the residual the next segment sees.

    Noise. Signal segments whiten against `residual.noise`. Two ways to
    supply it, both just residual contents the Wheel passes along untouched:

    * Fixed noise -- set it once on the observed data
      (`observed = replace(observed, noise=...)`); it rides the residual to
      every segment and never changes.
    * Sampled noise -- register a noise segment (one that returns the
      residual with an updated `noise`; see `segment.NoiseSegment`). Every
      segment stepped after it in a sweep sees the refreshed estimate. A
      segment sees a noise model in its own `start` only if it is registered
      after whoever set it.

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
        wheel.add(noise_segment)  # optional; a segment that edits residual.noise
        wheel.run(n_iterations=1000)

    The current running residual (the data minus every segment's latest
    model) is available via `wheel.residual()`. For anything about a
    segment's internals -- its parameters, its chain -- ask the segment
    object itself; you constructed it, you hold it.
    """

    # run settings a segment must not change: it may only move tdi and noise
    _INVARIANT: ClassVar[tuple[str, ...]] = (
        "channels",
        "n_samples",
        "sample_rate",
        "tdi_generation",
        "observable",
        "domain",
        "epoch",
    )

    def __init__(self, observed: Residuals):
        """Args:
            observed: TDI data with the run settings attached. The running
                residual starts here and inherits these settings for the
                whole run.
        """
        self.observed = observed
        self._segments: list[Segment] = []
        self._names: set[str] = set()
        # the one piece of state: the observed data minus every registered
        # segment's current model (with the current noise on `.noise`)
        self._residual = observed

    def add(self, segment: Segment) -> None:
        """Register a segment: call its `start` and adopt the residual it
        returns.

        All validation happens before any Wheel state changes, so a failed
        `add` leaves the Wheel exactly as it was. `start` receives the
        current running residual (carrying any noise model set by a segment
        already registered).
        """
        name = segment.name
        if not isinstance(name, str) or not name:
            raise ValueError(f"segment name must be a non-empty string, got {name!r}")
        if name in self._names:
            raise ValueError(f"segment name {name!r} already registered")
        new_residual = segment.start(self._handoff())
        self._validate_returned(new_residual, name, "start")
        # every check passed -- commit atomically
        self._segments.append(segment)
        self._names.add(name)
        self._residual = new_residual

    def run(
        self,
        n_iterations: int,
        on_sweep: Callable[[int, "Wheel"], None] | None = None,
    ) -> None:
        """Drive the Gibbs loop for `n_iterations` sweeps over all segments.

        Each sweep hands the running residual to every segment in turn,
        adopting each returned residual before moving to the next. The
        returned residual is validated every step, so a segment that drifts
        (wrong shape, changed run settings) raises immediately.

        The Wheel stores nothing about a segment beyond its effect on the
        residual: chains, checkpoints, and diagnostics are the segment's own
        business.

        Args:
            n_iterations: Number of full sweeps over all segments.
            on_sweep: Optional progress/checkpoint hook, called as
                `on_sweep(iteration, self)` after each completed sweep
                (`iteration` counts from 0). Read `residual()` off the
                wheel -- or anything you like off your own segment
                objects -- to log, plot, or checkpoint; the hook must not
                mutate the wheel. Equivalent to calling `run(1)` in your own
                loop -- the Gibbs chain is identical either way.
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
                new_residual = seg.step(self._handoff())
                self._validate_returned(new_residual, seg.name, "step")
                self._residual = new_residual
            if on_sweep is not None:
                on_sweep(iteration, self)

    def residual(self) -> Residuals:
        """The current running residual: observed data minus every segment's
        latest model, with the current noise model on `.noise`."""
        return self._residual

    def _handoff(self) -> Residuals:
        """A fresh copy of the running residual to hand a segment, so it may
        mutate its arrays without touching the Wheel's own state."""
        return replace(
            self._residual,
            tdi={ch: arr.copy() for ch, arr in self._residual.tdi.items()},
        )

    def _validate_returned(
        self, returned: object, segment_name: str, method: str
    ) -> None:
        if not isinstance(returned, Residuals):
            raise TypeError(
                f"{segment_name}.{method} must return a Residuals "
                f"(the updated residual), got {type(returned).__name__}"
            )
        for field in self._INVARIANT:
            if getattr(returned, field) != getattr(self.observed, field):
                raise ValueError(
                    f"{segment_name}.{method} changed the run setting {field!r} "
                    f"({getattr(self.observed, field)!r} -> {getattr(returned, field)!r}); "
                    f"a segment may only update tdi and noise, not the fixed run "
                    f"settings"
                )
        if returned.orbit is not self.observed.orbit:
            raise ValueError(
                f"{segment_name}.{method} changed the orbit; it is a fixed "
                f"property of the dataset and must be passed through unchanged"
            )
