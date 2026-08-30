import warnings
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from enchilada.block import Block
from enchilada.data import L1Data
from enchilada.template import Template, check_on_grid


class NoiseOverwrittenWarning(RuntimeWarning):
    """Two blocks are writing `L1Data.noise`, so one is losing.

    `noise` is a single slot: whichever block writes last owns the model every
    block sees afterwards. That is fine when one noise block owns it, and fine
    when a noise block takes over the model the dataset arrived with. It is
    almost never what you want when two blocks each maintain a component --
    instrument noise and the galactic confusion foreground, say -- because the
    Wheel does not combine them, it just keeps the last one.

    Fix it by sampling both components inside a single noise block that
    publishes one combined model, or by treating the foreground as a signal
    block that returns it as a template (where the ledger *does* combine
    contributions). If you really do mean to hand ownership between blocks,
    silence it precisely::

        warnings.filterwarnings(
            "ignore", category=enchilada.NoiseOverwrittenWarning
        )
    """


class Wheel:
    """Runs a blocked-Gibbs global fit by handing each block a clean residual.

    The Wheel keeps the pristine observed data and a **ledger** -- one entry
    per block holding that block's current **template**: the signal it has
    fit, summed over all its sources, as a channel -> array dict. From those
    it forms any residual by subtraction, and it hands each block exactly the
    residual that block should fit: the observed data minus **every other**
    block's current template (never the block's own). The block fits against
    that and returns its new template; the Wheel records it in the ledger and
    subtracts it on the block's behalf whenever it forms a residual.

    Why this shape. A block is only ever shown the data with its own template
    already removed, so there is no "add-back" for it to remember -- and it
    does no cross-block arithmetic at all: it returns what it found, and the
    Wheel owns every subtraction. Because the template is returned rather than
    recovered from a difference, it is stored at full precision, and the
    classic silent failure of residual passing -- handing the residual straight
    back and having that read as "nothing to subtract" -- is caught exactly
    (see `_validate_returned`) instead of guessed at.

    What lives where. The block owns its *sampler* state -- parameters, RNG,
    chain, checkpoints -- and the Wheel never touches it. The Wheel owns the
    *residual* state -- the pristine data and the per-block template ledger --
    and does all the differencing. (This is the split the GLASS global fit
    uses: blocks own their samplers, the framework owns the residual
    bookkeeping.)

    Consistency checking. `add` validates a block fully before recording it
    (`name`, `start` and `update`; a `start` that fails leaves the Wheel
    untouched), and every `start`/`update` must return a `Template` that lives
    on the run's grid (the right channels, lengths and real/complex-ness) and
    contains no NaN or inf.

    The return type does most of this work. Because a template is not an
    `L1Data`, it cannot carry the run settings or the orbit, so neither can
    drift in transit and neither needs guarding; and a block still written for
    the old contract -- returning the residual with its template subtracted --
    is a `TypeError` at registration rather than a fit that is quietly wrong
    (that return is a valid-looking `L1Data`, so no array check could catch
    it). One failure remains a warning, because it cannot be proven wrong from
    outside: a second block publishing a noise model
    (`NoiseOverwrittenWarning`).

    Noise. Signal blocks whiten against `residual.noise`. Two ways to supply
    it:

    * Fixed noise -- set it once on the observed data
      (`observed = replace(observed, noise=...)`); it rides every handed
      residual and never changes.
    * Sampled noise -- register a noise block (one that returns
      `residual.zero_template().with_noise(model)`, so its ledger entry is
      zero; see `block.NoiseBlock`). Every block updated after it sees the
      refreshed estimate.

    Typical use:

        observed = L1Data(tdi=..., sample_rate=...,
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

    `wheel.residual()` is the full residual (data minus every block's
    template); `wheel.residual(exclude=name)` is the residual that block sees;
    `wheel.contribution(name)` is that block's current template. For a
    block's internals -- its parameters, its chain -- ask the block object
    you constructed and hold.
    """

    def __init__(self, observed: L1Data):
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
                    f"gaps or excised glitches, note that enchilada has no "
                    f"data-quality mask yet (see the L1Data docstring); "
                    f"fill or trim them before starting a run."
                )
        self.observed = observed
        self._blocks: list[Block] = []
        # the ledger: name -> that block's current template (its summed signal)
        self._ledger: dict[str, dict[str, np.ndarray]] = {}
        # the current noise model threaded onto every handed residual, and the
        # block that last wrote it (None = the model the dataset arrived with)
        self._noise = observed.noise
        self._noise_owner: str | None = None

    def add(self, block: Block) -> None:
        """Register a block: call its `start` and record its template.

        `start` is handed the data minus every block already registered
        (carrying the current noise). All validation happens before the Wheel
        records anything, so a failed `add` leaves the Wheel exactly as it was.
        """
        name = getattr(block, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"block name must be a non-empty string, got {name!r}; every "
                f"Block needs a `name` unique within the Wheel "
                f"(see enchilada.block.Block)"
            )
        if name in self._ledger:
            raise ValueError(f"block name {name!r} already registered")
        for method in ("start", "update"):
            # check both up front: a missing `update` would otherwise register
            # cleanly and die mid-cycle, after other ledger entries had moved
            if not callable(getattr(block, method, None)):
                raise TypeError(
                    f"block {name!r} does not implement {method}(residual); a "
                    f"Block needs `name`, `start` and `update` "
                    f"(see enchilada.block.Block)"
                )
        # residual() already builds fresh arrays, so the block may mutate
        # what it is handed; the ledger and `observed` are untouched either way
        returned = block.start(self.residual())
        self._validate_returned(returned, name, "start")
        # every check passed -- commit atomically
        self._blocks.append(block)
        self._adopt(name, returned, "start")

    def run(
        self,
        n_cycles: int,
        on_cycle: Callable[[int, "Wheel"], None] | None = None,
    ) -> None:
        """Drive the blocked-Gibbs loop for `n_cycles` cycles of the wheel.

        One full cycle visits every block once, handing each the data minus
        every *other* block's current template, validating the template it
        returns, and recording it in the ledger. That is the unit with
        statistical meaning: only after a complete cycle is every block
        conditioned on the current value of all the others.

        Three nested scales, three words, so no name does double duty:

        * a **cycle** -- one pass over every block (this loop);
        * a **block update** -- one block's `update()` call within it;
        * a **step** -- what a block's own sampler does, many times, inside a
          single `update()` call.

        Two notes for readers coming from elsewhere. The Monte Carlo
        literature calls a cycle a *sweep*. GLASS uses `cycle` for something
        different -- the number of repeat updates given to one module -- so
        when comparing notes, enchilada's cycle is GLASS's outer Gibbs loop,
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
                returned = block.update(handed)
                self._validate_returned(returned, block.name, "update")
                self._adopt(block.name, returned, "update")
            if on_cycle is not None:
                on_cycle(cycle, self)

    def residual(self, exclude: str | None = None) -> L1Data:
        """A residual formed from the ledger, with the current noise on `.noise`.

        With no argument: the full residual, observed data minus every
        block's current template. Pass `exclude=name` for the residual that
        block sees -- observed data minus every *other* block's template.
        Fresh arrays each call, so callers may mutate freely.
        """
        if exclude is not None and exclude not in self._ledger:
            raise ValueError(
                f"unknown block {exclude!r}; registered: {sorted(self._ledger)}"
            )
        # Promote once, up front, to whatever dtype the observed data and every
        # subtracted template share -- then the accumulation below can stay
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
        """The named block's current ledger entry: its template, as a copy."""
        if name not in self._ledger:
            raise ValueError(
                f"unknown block {name!r}; registered: {sorted(self._ledger)}"
            )
        return {ch: arr.copy() for ch, arr in self._ledger[name].items()}

    def _adopt(self, name: str, returned: Template, method: str) -> None:
        """Record a block's template in the ledger, and any noise it publishes.

        The ledger takes a *copy*: the block may keep (and later overwrite)
        the arrays it returned, and the ledger must not move with them.
        """
        self._ledger[name] = {
            ch: returned.tdi[ch].copy() for ch in self.observed.channels
        }
        if returned.noise is not None:  # this block published a model
            if self._noise_owner is not None and self._noise_owner != name:
                warnings.warn(
                    f"{name}.{method} replaced the noise model that "
                    f"{self._noise_owner!r} owns. `L1Data.noise` is a single "
                    f"slot -- the Wheel does not combine noise models, so "
                    f"{self._noise_owner!r}'s is now gone and every block sees "
                    f"only {name!r}'s. If you are modelling two components, "
                    f"publish one combined model from a single noise block "
                    f"(see enchilada.NoiseOverwrittenWarning).",
                    NoiseOverwrittenWarning,
                    # _adopt -> run/add -> the user's call: 3 frames
                    stacklevel=3,
                )
            self._noise_owner = name
            self._noise = returned.noise

    def _validate_returned(
        self, returned: object, block_name: str, method: str
    ) -> None:
        """Refuse a return that would corrupt the run, in three checks.

        Type, then the grid, then finiteness -- ordered
        cheapest-and-most-fundamental first so the message a block author sees
        names the most basic thing they got wrong. This is the whole list: a
        `Template` carries no run settings and no orbit, so there is nothing
        else left to drift.
        """
        if not isinstance(returned, Template):
            extra = (
                " -- a block returns only its own template now, never the "
                "residual with that template subtracted; build it with "
                "`residual.template({...})`, or `residual.zero_template()` "
                "when there is nothing to subtract"
                if isinstance(returned, L1Data)
                else ""
            )
            raise TypeError(
                f"{block_name}.{method} must return a Template (the signal this "
                f"block claims), got {type(returned).__name__}{extra}"
            )
        check_on_grid(
            returned.tdi,
            channels=self.observed.channels,
            n_samples=self.observed.n_samples,
            domain=self.observed.domain,
            what=f"{block_name}.{method} template tdi",
        )
        # A blown-up sampler returning NaN/inf would otherwise be recorded as
        # that block's template and handed to every block updated later in
        # the cycle.
        for ch in self.observed.channels:
            arr = returned.tdi[ch]
            if not np.isfinite(arr).all():
                n_bad = int((~np.isfinite(arr)).sum())
                raise ValueError(
                    f"{block_name}.{method} returned {n_bad} non-finite "
                    f"sample(s) in channel {ch!r} (NaN or inf) in its template; "
                    f"it would poison every block updated after it. Check the "
                    f"sampler's proposal and its noise weighting."
                )
