import warnings
from collections.abc import Callable
from dataclasses import replace
from typing import ClassVar

import numpy as np

from turntable.block import Block
from turntable.residuals import Residuals


class ModelWithdrawnWarning(RuntimeWarning):
    """A block's model went from non-zero to exactly zero in one update.

    The ledger is *derived* (what a block was handed minus what it returned),
    so the Wheel cannot tell these two apart from the outside:

    * the block's model is legitimately zero now -- a reversible-jump block
      whose last source died, or a cadenced block with nothing to contribute
      this cycle. Nothing is wrong.
    * the block failed to re-subtract the model it still believes it has --
      a wrapper whose external process errored, an all-rejected cycle returned
      as "no change" -- and its model has silently left the fit.

    It warns rather than raises because it is a heuristic about intent. If the
    first case is yours, silence it precisely::

        warnings.filterwarnings("ignore", category=turntable.ModelWithdrawnWarning)

    `turntable.testing.check_block` escalates it to an error, on the grounds
    that a conformance check should be strict where a running fit should not.
    """


class Wheel:
    """Runs a blocked-Gibbs global fit by handing each block a clean residual.

    The Wheel keeps the pristine observed data and a **ledger** -- one entry
    per block holding that block's current model contribution (its summed
    waveform, as a channel -> array dict). From those it can form any residual
    by subtraction, and it hands each block exactly the residual that block
    should fit: the observed data minus **every other** block's current model
    (never the block's own). The block fits against that, subtracts its new
    model, and returns the updated residual; the Wheel reads the block's new
    ledger entry straight off the difference between what it handed out and what
    came back.

    Why this shape. Because a block is only ever shown the data with its own
    model already removed, there is no "add-back" to remember and no way to
    forget one -- the classic silent failure of residual passing is structurally
    impossible here. The ledger is a required, automatically-consistent product
    of every return; the block never does the cross-block arithmetic.

    What lives where. The block owns its *sampler* state -- parameters, RNG,
    chain, checkpoints -- and the Wheel never touches it. The Wheel owns the
    *residual* state -- the pristine data and the per-block contribution
    ledger -- and does all the differencing. (This is the split the GLASS
    global fit uses: blocks own their samplers, the framework owns the residual
    bookkeeping.)

    Consistency checking. `add` validates a block fully before recording it
    (`name`, `start` and `block_update`; a `start` that fails leaves the Wheel
    untouched), and every `start`/`block_update` return must be a `Residuals` that

    * kept the fixed run settings (`_INVARIANT`) -- only `tdi` and `noise` move;
    * kept the same `orbit` object;
    * did not drop a noise model that was set;
    * contains no NaN or inf.

    `Residuals` itself re-validates shapes and dtypes, so a mid-run drift
    raises immediately. One failure is only a warning, because it cannot be
    diagnosed from outside: see `ModelWithdrawnWarning`.

    Noise. Signal blocks whiten against `residual.noise`. Two ways to supply
    it:

    * Fixed noise -- set it once on the observed data
      (`observed = replace(observed, noise=...)`); it rides every handed
      residual and never changes.
    * Sampled noise -- register a noise block (one that returns the residual
      with an updated `noise` and its tdi untouched, so its ledger entry is
      zero; see `block.NoiseBlock`). Every block updated after it sees the
      refreshed estimate.

    Typical use:

        observed = Residuals(tdi=..., sample_rate=...,
                             channels=("A", "E", "T"),
                             tdi_generation="2.0",
                             observable="fractional_frequency",
                             noise=fixed_noise_model)  # optional fixed noise
        # (n_samples is read off the arrays for time-domain data)
        wheel = Wheel(observed)
        wheel.add(ucb_block)
        wheel.add(mbhb_block)
        wheel.add(noise_block)  # optional; a block that edits residual.noise
        wheel.run(n_cycles=1000)

    `wheel.residual()` is the full residual (data minus every block);
    `wheel.residual(exclude=name)` is the residual that block sees. For a
    block's internals -- its parameters, its chain -- ask the block object
    you constructed and hold.
    """

    # run settings a block must not change: it may only move tdi and noise
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
        for ch in observed.channels:
            # Otherwise the first block to touch it gets blamed by the
            # finiteness guard below for data that was already broken. NaN is
            # also the natural way a user marks gaps today, and gap support is
            # not in the contract yet -- so say that plainly here.
            if not np.isfinite(observed.tdi[ch]).all():
                n_bad = int((~np.isfinite(observed.tdi[ch])).sum())
                raise ValueError(
                    f"observed.tdi[{ch!r}] has {n_bad} non-finite sample(s); the "
                    f"data itself is not usable as a residual. If these mark "
                    f"gaps or excised glitches, note that turntable has no "
                    f"data-quality mask yet (see the Residuals docstring); "
                    f"fill or trim them before starting a run."
                )
        self.observed = observed
        self._blocks: list[Block] = []
        # the ledger: name -> that block's current contribution (summed model)
        self._ledger: dict[str, dict[str, np.ndarray]] = {}
        # the current noise model threaded onto every handed residual
        self._noise = observed.noise

    def add(self, block: Block) -> None:
        """Register a block: call its `start` and record its contribution.

        `start` is handed the data minus every block already registered
        (carrying the current noise). All validation happens before the Wheel
        records anything, so a failed `add` leaves the Wheel exactly as it was.
        """
        name = getattr(block, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"block name must be a non-empty string, got {name!r}; every "
                f"Block needs a `name` unique within the Wheel "
                f"(see turntable.block.Block)"
            )
        if name in self._ledger:
            raise ValueError(f"block name {name!r} already registered")
        for method in ("start", "block_update"):
            # check both up front: a missing `block_update` would otherwise register
            # cleanly and die mid-cycle, after other ledger entries had moved
            if not callable(getattr(block, method, None)):
                raise TypeError(
                    f"block {name!r} does not implement {method}(residual); a "
                    f"Block needs `name`, `start` and `block_update` "
                    f"(see turntable.block.Block)"
                )
        handed = self.residual()  # data minus blocks registered so far
        returned = block.start(self._mutable(handed))
        self._validate_returned(returned, name, "start")
        # every check passed -- commit atomically
        self._blocks.append(block)
        self._adopt(name, handed, returned, "start")

    def run(
        self,
        n_cycles: int,
        on_cycle: Callable[[int, "Wheel"], None] | None = None,
    ) -> None:
        """Drive the blocked-Gibbs loop for `n_cycles` cycles of the wheel.

        One full cycle visits every block once, handing each the data minus
        every *other* block's current model, validating what it returns, and
        updating that block's ledger entry from the difference. That is the
        unit with statistical meaning: only after a complete cycle is every
        block conditioned on the current value of all the others.

        Three nested scales, three words, so no name does double duty:

        * a **cycle** -- one pass over every block (this loop);
        * a **block update** -- one block's `block_update()` call within it;
        * a **step** -- what a block's own sampler does, many times, inside a
          single `block_update()` call.

        Two notes for readers coming from elsewhere. The Monte Carlo
        literature calls a cycle a *sweep*. GLASS uses `cycle` for something
        different -- the number of repeat updates given to one module -- so
        when comparing notes, turntable's cycle is GLASS's outer Gibbs loop,
        not its `cycle` variable.

        Args:
            n_cycles: Number of full cycles of the wheel over all blocks.
            on_cycle: Optional progress/checkpoint hook, called as
                `on_cycle(cycle, self)` after each completed cycle
                (`cycle` counts from 0). Read `residual()` off the wheel --
                or anything off your own block objects -- to log or
                checkpoint; the hook must not mutate the wheel. Equivalent to
                calling `run(1)` in your own loop.
        """
        if (
            not isinstance(n_cycles, (int, np.integer))
            or isinstance(n_cycles, bool)
            or n_cycles < 0
        ):
            raise ValueError(
                f"n_cycles must be a non-negative integer, got {n_cycles!r}"
            )
        for cycle in range(n_cycles):
            for block in self._blocks:
                handed = self.residual(exclude=block.name)  # data minus OTHERS
                returned = block.block_update(self._mutable(handed))
                self._validate_returned(returned, block.name, "block_update")
                self._adopt(block.name, handed, returned, "block_update")
            if on_cycle is not None:
                on_cycle(cycle, self)

    def residual(self, exclude: str | None = None) -> Residuals:
        """A residual formed from the ledger, with the current noise on `.noise`.

        With no argument: the full residual, observed data minus every
        block's current model. Pass `exclude=name` for the residual that
        block sees -- observed data minus every *other* block's model.
        Fresh arrays each call, so callers may mutate freely.
        """
        if exclude is not None and exclude not in self._ledger:
            raise ValueError(
                f"unknown block {exclude!r}; registered: {sorted(self._ledger)}"
            )
        # Promote once, up front, to whatever dtype the observed data and every
        # subtracted model share -- then the accumulation below can stay
        # in-place. (Subtracting out-of-place per block would also promote,
        # but allocates a fresh array per block per channel, which is the
        # hot loop: run() calls this once per block per cycle.)
        entries = [c for name, c in self._ledger.items() if name != exclude]
        tdi = {}
        for ch in self.observed.channels:
            base = self.observed.tdi[ch]
            dtype = (
                np.result_type(base, *(e[ch] for e in entries))
                if entries
                else base.dtype
            )
            tdi[ch] = base.astype(dtype, copy=True)
        for entry in entries:
            for ch in tdi:
                tdi[ch] -= entry[ch]
        return replace(self.observed, tdi=tdi, noise=self._noise)

    def contribution(self, name: str) -> dict[str, np.ndarray]:
        """The named block's current ledger entry (its summed model)."""
        if name not in self._ledger:
            raise ValueError(
                f"unknown block {name!r}; registered: {sorted(self._ledger)}"
            )
        return {ch: arr.copy() for ch, arr in self._ledger[name].items()}

    def _mutable(self, residual: Residuals) -> Residuals:
        """A copy the block may mutate freely, leaving `residual` pristine so
        the Wheel can diff against it even if the block returns it in place."""
        return replace(
            residual, tdi={ch: arr.copy() for ch, arr in residual.tdi.items()}
        )

    def _contribution(
        self, handed: Residuals, returned: Residuals
    ) -> dict[str, np.ndarray]:
        """A block's model = what it was handed minus what it returned."""
        return {ch: handed.tdi[ch] - returned.tdi[ch] for ch in self.observed.channels}

    def _adopt(
        self, name: str, handed: Residuals, returned: Residuals, method: str
    ) -> None:
        """Record a block's new ledger entry and the noise it threaded.

        Warns (`ModelWithdrawnWarning`) if a model that was previously non-zero
        has become exactly zero -- which may be a legitimate death move or a
        block that forgot to re-subtract itself. See that class for why the
        Wheel cannot distinguish them and how to silence it.
        """
        contribution = self._contribution(handed, returned)
        previous = self._ledger.get(name)
        if (
            previous is not None
            and self._all_zero(contribution)
            and not self._all_zero(previous)
        ):
            warnings.warn(
                f"{name}.{method}: this block's model went from non-zero to "
                f"exactly zero, so it now contributes nothing to the fit. If that "
                f"is intentional (a death move to zero sources, or a cycle with "
                f"nothing to contribute) this is fine -- silence it with "
                f"warnings.filterwarnings('ignore', "
                f"category=turntable.ModelWithdrawnWarning). If not, remember the "
                f"ledger is derived from what you return, not remembered: "
                f"re-subtract your current model on every block update.",
                ModelWithdrawnWarning,
                # _adopt -> run/add -> the user's call: 3 frames
                stacklevel=3,
            )
        self._ledger[name] = contribution
        self._noise = returned.noise

    @staticmethod
    def _all_zero(contribution: dict[str, np.ndarray]) -> bool:
        return all(not np.any(arr) for arr in contribution.values())

    def _validate_returned(
        self, returned: object, block_name: str, method: str
    ) -> None:
        """Refuse a return that would corrupt the run, in five checks.

        Type, then the fixed run settings, then orbit identity, then the noise
        model, then finiteness -- ordered cheapest-and-most-fundamental first
        so the message a block author sees names the most basic thing they
        got wrong.
        """
        if not isinstance(returned, Residuals):
            raise TypeError(
                f"{block_name}.{method} must return a Residuals "
                f"(the updated residual), got {type(returned).__name__}"
            )
        for field in self._INVARIANT:
            if getattr(returned, field) != getattr(self.observed, field):
                raise ValueError(
                    f"{block_name}.{method} changed the run setting {field!r} "
                    f"({getattr(self.observed, field)!r} -> "
                    f"{getattr(returned, field)!r}); a block may only update "
                    f"tdi and noise, not the fixed run settings"
                )
        if returned.orbit is not self.observed.orbit:
            raise ValueError(
                f"{block_name}.{method} changed the orbit; it is a fixed "
                f"property of the dataset and must be passed through unchanged"
            )
        # Losing the noise model is never intentional, and it is silent: every
        # block updated afterwards would whiten against nothing. Guard it the
        # same way the orbit is guarded -- a block that rebuilds a Residuals
        # from scratch (rather than using `replace`) drops it by accident.
        if self._noise is not None and returned.noise is None:
            raise ValueError(
                f"{block_name}.{method} dropped the noise model (a model was "
                f"set, and the returned residual has noise=None); build the "
                f"result with `replace(residual, ...)` so noise and orbit ride "
                f"along, or return `replace(residual, noise=your_model)` if you "
                f"are the noise block"
            )
        # The last silent cross-block failure: a blown-up sampler returning
        # NaN/inf would otherwise be recorded as that block's model and handed
        # to every block updated later in the cycle.
        for ch in self.observed.channels:
            arr = returned.tdi[ch]
            if not np.isfinite(arr).all():
                n_bad = int((~np.isfinite(arr)).sum())
                raise ValueError(
                    f"{block_name}.{method} returned {n_bad} non-finite "
                    f"sample(s) in channel {ch!r} (NaN or inf); the residual "
                    f"would poison every block updated after it. Check the "
                    f"sampler's proposal and its noise weighting."
                )
