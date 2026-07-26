import warnings
from collections.abc import Callable
from dataclasses import replace
from typing import ClassVar

import numpy as np

from turntable.residuals import Residuals
from turntable.segment import Segment


class Wheel:
    """Runs a blocked-Gibbs global fit by handing each segment a clean residual.

    The Wheel keeps the pristine observed data and a **ledger** -- one entry
    per segment holding that segment's current model contribution (its summed
    waveform, as a channel -> array dict). From those it can form any residual
    by subtraction, and it hands each segment exactly the residual that segment
    should fit: the observed data minus **every other** segment's current model
    (never the segment's own). The segment fits against that, subtracts its new
    model, and returns the updated residual; the Wheel reads the segment's new
    ledger entry straight off the difference between what it handed out and what
    came back.

    Why this shape. Because a segment is only ever shown the data with its own
    model already removed, there is no "add-back" to remember and no way to
    forget one -- the classic silent failure of residual passing is structurally
    impossible here. The ledger is a required, automatically-consistent product
    of every return; the segment never does the cross-segment arithmetic.

    What lives where. The segment owns its *sampler* state -- parameters, RNG,
    chain, checkpoints -- and the Wheel never touches it. The Wheel owns the
    *residual* state -- the pristine data and the per-segment contribution
    ledger -- and does all the differencing. (This is the split the GLASS
    global fit uses: blocks own their samplers, the framework owns the residual
    bookkeeping.)

    Consistency checking. `add` validates a segment fully before recording it
    (a `start` that fails leaves the Wheel untouched), and every `start`/`step`
    return is checked to be a `Residuals` that kept the fixed run settings --
    only `tdi` and `noise` may move. `Residuals` itself validates the returned
    tdi shapes, so a mid-run drift raises immediately.

    Noise. Signal segments whiten against `residual.noise`. Two ways to supply
    it:

    * Fixed noise -- set it once on the observed data
      (`observed = replace(observed, noise=...)`); it rides every handed
      residual and never changes.
    * Sampled noise -- register a noise segment (one that returns the residual
      with an updated `noise` and its tdi untouched, so its ledger entry is
      zero; see `segment.NoiseSegment`). Every segment stepped after it sees the
      refreshed estimate.

    Typical use:

        observed = Residuals(tdi=..., sample_rate=...,
                             channels=("A", "E", "T"),
                             tdi_generation="2.0",
                             observable="fractional_frequency",
                             noise=fixed_noise_model)  # optional fixed noise
        # (n_samples is read off the arrays for time-domain data)
        wheel = Wheel(observed)
        wheel.add(ucb_segment)
        wheel.add(mbhb_segment)
        wheel.add(noise_segment)  # optional; a segment that edits residual.noise
        wheel.run(n_iterations=1000)

    `wheel.residual()` is the full residual (data minus every segment);
    `wheel.residual(exclude=name)` is the residual that segment sees. For a
    segment's internals -- its parameters, its chain -- ask the segment object
    you constructed and hold.
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
        """Start a run from the observed data.

        Args:
            observed: TDI data with the run settings attached. Kept pristine;
                every residual the Wheel forms starts from it.
        """
        self.observed = observed
        self._segments: list[Segment] = []
        # the ledger: name -> that segment's current contribution (summed model)
        self._ledger: dict[str, dict[str, np.ndarray]] = {}
        # the current noise model threaded onto every handed residual
        self._noise = observed.noise

    def add(self, segment: Segment) -> None:
        """Register a segment: call its `start` and record its contribution.

        `start` is handed the data minus every segment already registered
        (carrying the current noise). All validation happens before the Wheel
        records anything, so a failed `add` leaves the Wheel exactly as it was.
        """
        name = segment.name
        if not isinstance(name, str) or not name:
            raise ValueError(f"segment name must be a non-empty string, got {name!r}")
        if name in self._ledger:
            raise ValueError(f"segment name {name!r} already registered")
        for method in ("start", "step"):
            # check both up front: a missing `step` would otherwise register
            # cleanly and die mid-sweep, after other ledger entries had moved
            if not callable(getattr(segment, method, None)):
                raise TypeError(
                    f"segment {name!r} does not implement {method}(residual); a "
                    f"Segment needs `name`, `start` and `step` "
                    f"(see turntable.segment.Segment)"
                )
        handed = self.residual()  # data minus segments registered so far
        returned = segment.start(self._mutable(handed))
        self._validate_returned(returned, name, "start")
        # every check passed -- commit atomically
        self._segments.append(segment)
        self._adopt(name, handed, returned, "start")

    def run(
        self,
        n_iterations: int,
        on_sweep: Callable[[int, "Wheel"], None] | None = None,
    ) -> None:
        """Drive the Gibbs loop for `n_iterations` sweeps over all segments.

        Each sweep visits every segment in turn, hands it the data minus every
        *other* segment's current model, validates what it returns, and updates
        that segment's ledger entry from the difference.

        Args:
            n_iterations: Number of full sweeps over all segments.
            on_sweep: Optional progress/checkpoint hook, called as
                `on_sweep(iteration, self)` after each completed sweep
                (`iteration` counts from 0). Read `residual()` off the wheel --
                or anything off your own segment objects -- to log or
                checkpoint; the hook must not mutate the wheel. Equivalent to
                calling `run(1)` in your own loop.
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
                handed = self.residual(exclude=seg.name)  # data minus OTHERS
                returned = seg.step(self._mutable(handed))
                self._validate_returned(returned, seg.name, "step")
                self._adopt(seg.name, handed, returned, "step")
            if on_sweep is not None:
                on_sweep(iteration, self)

    def residual(self, exclude: str | None = None) -> Residuals:
        """A residual formed from the ledger, with the current noise on `.noise`.

        With no argument: the full residual, observed data minus every
        segment's current model. Pass `exclude=name` for the residual that
        segment sees -- observed data minus every *other* segment's model.
        Fresh arrays each call, so callers may mutate freely.
        """
        if exclude is not None and exclude not in self._ledger:
            raise ValueError(
                f"unknown segment {exclude!r}; registered: {sorted(self._ledger)}"
            )
        tdi = {ch: self.observed.tdi[ch].copy() for ch in self.observed.channels}
        for name, contribution in self._ledger.items():
            if name == exclude:
                continue
            for ch in tdi:
                # out-of-place: promotes dtype rather than raising if a
                # contribution is wider than the observed array
                tdi[ch] = tdi[ch] - contribution[ch]
        return replace(self.observed, tdi=tdi, noise=self._noise)

    def contribution(self, name: str) -> dict[str, np.ndarray]:
        """The named segment's current ledger entry (its summed model)."""
        if name not in self._ledger:
            raise ValueError(
                f"unknown segment {name!r}; registered: {sorted(self._ledger)}"
            )
        return {ch: arr.copy() for ch, arr in self._ledger[name].items()}

    def _mutable(self, residual: Residuals) -> Residuals:
        """A copy the segment may mutate freely, leaving `residual` pristine so
        the Wheel can diff against it even if the segment returns it in place."""
        return replace(
            residual, tdi={ch: arr.copy() for ch, arr in residual.tdi.items()}
        )

    def _contribution(
        self, handed: Residuals, returned: Residuals
    ) -> dict[str, np.ndarray]:
        """A segment's model = what it was handed minus what it returned."""
        return {ch: handed.tdi[ch] - returned.tdi[ch] for ch in self.observed.channels}

    def _adopt(
        self, name: str, handed: Residuals, returned: Residuals, method: str
    ) -> None:
        """Record a segment's new ledger entry and the noise it threaded.

        Warns if a model that was previously non-zero has become exactly zero.
        The ledger is derived (handed minus returned), so a segment that hands
        the residual straight back -- a plausible way to mean "nothing changed
        this sweep", e.g. a cadenced block, an all-rejected sweep, or a wrapper
        whose external process failed -- silently withdraws its model from the
        fit. A segment must re-subtract its current model on *every* step.
        """
        contribution = self._contribution(handed, returned)
        previous = self._ledger.get(name)
        if (
            previous is not None
            and self._all_zero(contribution)
            and not self._all_zero(previous)
        ):
            warnings.warn(
                f"{name}.{method} returned the residual unchanged, so its model "
                f"went from non-zero to zero and has left the fit. A segment must "
                f"re-subtract its current model every step, even when its "
                f"parameters did not move (the ledger is derived from what you "
                f"return, not remembered).",
                RuntimeWarning,
                stacklevel=4,
            )
        self._ledger[name] = contribution
        self._noise = returned.noise

    @staticmethod
    def _all_zero(contribution: dict[str, np.ndarray]) -> bool:
        return all(not np.any(arr) for arr in contribution.values())

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
                    f"({getattr(self.observed, field)!r} -> "
                    f"{getattr(returned, field)!r}); a segment may only update "
                    f"tdi and noise, not the fixed run settings"
                )
        if returned.orbit is not self.observed.orbit:
            raise ValueError(
                f"{segment_name}.{method} changed the orbit; it is a fixed "
                f"property of the dataset and must be passed through unchanged"
            )
        # Losing the noise model is never intentional, and it is silent: every
        # segment stepped afterwards would whiten against nothing. Guard it the
        # same way the orbit is guarded -- a segment that rebuilds a Residuals
        # from scratch (rather than using `replace`) drops it by accident.
        if self._noise is not None and returned.noise is None:
            raise ValueError(
                f"{segment_name}.{method} dropped the noise model (a model was "
                f"set, and the returned residual has noise=None); build the "
                f"result with `replace(residual, ...)` so noise and orbit ride "
                f"along, or return `replace(residual, noise=your_model)` if you "
                f"are the noise segment"
            )
        # The last silent cross-segment failure: a blown-up sampler returning
        # NaN/inf would otherwise be recorded as that segment's model and handed
        # to every segment stepped later in the sweep.
        for ch in self.observed.channels:
            arr = returned.tdi[ch]
            if not np.isfinite(arr).all():
                n_bad = int((~np.isfinite(arr)).sum())
                raise ValueError(
                    f"{segment_name}.{method} returned {n_bad} non-finite "
                    f"sample(s) in channel {ch!r} (NaN or inf); the residual "
                    f"would poison every segment stepped after it. Check the "
                    f"sampler's proposal and its noise weighting."
                )
