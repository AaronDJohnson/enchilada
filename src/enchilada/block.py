from typing import Protocol, runtime_checkable

from enchilada.data import L1Data
from enchilada.template import Template


@runtime_checkable
class Block(Protocol):
    """Plug a sampler into the Wheel by implementing this interface.

    The Wheel hands you a residual and takes back your template. It calls
    two methods on you:

    - `start` once, when you are registered. Read the run settings and
      conventions off the residual, set yourself up, and return your initial
      template (`residual.zero_template()` if you start from nothing).
    - `update` once each cycle -- one **block update**. The residual you
      receive is the observed data with **every other** block's template
      already subtracted -- **but not your own**. So it is exactly the data
      your source class must explain: fit against it directly, then return
      your new template. You subtract nothing and add nothing back.

    What a template is. The signal you currently claim, summed over all your
    sources, on the residual's grid -- a `Template`, built from the residual
    you were handed: `residual.template({ch: ...})`, or
    `residual.zero_template()` when it is zero. The Wheel keeps every block's
    template in a ledger and does all the arithmetic: it subtracts yours from
    the data on your behalf when forming what every other block sees, and
    leaves it out of what *you* see. You never do cross-block arithmetic --
    there is no add-back to forget and no subtraction to get wrong. (This is
    why the Wheel, not the block, owns the residual bookkeeping; you still own
    everything about your sampler.)

    A template is deliberately a different type from the residual. Returning
    the residual with your template subtracted -- the pre-template contract --
    would otherwise be an ordinary-looking return that silently made every
    other block fit the wrong thing; as a distinct type it is a `TypeError` at
    registration instead. It also means a template cannot carry the run
    settings or the orbit, so you cannot accidentally change either.

    What is yours. Your parameters, your RNG, your posterior chain, your
    proposal tuning, your checkpoints, and your current template all live
    inside your object (or the external process it wraps). The Wheel never
    sees, stores, or restores your sampler state; it only records the template
    you return, so it can form the next residual.

    The pattern:

        from enchilada import replace   # or: from dataclasses import replace

        def update(self, residual):
            template = {}
            for ch in residual.channels:
                data = residual.tdi[ch]                  # already data minus OTHERS
                self.amplitude = draw_against(data, self.waveform[ch])  # your sample
                template[ch] = self.amplitude * self.waveform[ch]       # what you claim
            return residual.template(template)

    Noise is not special. A block that models the noise instead of a signal
    claims no signal, so its template is zero; it returns
    `residual.zero_template().with_noise(my_model)`, and signal blocks
    read the model back through `residual.noise_psd` (per-bin weight) or
    `residual.noise_variance` (per-sample variance, for a time-domain
    likelihood). See `NoiseBlock`.

    `isinstance(x, Block)` is a *shape* check, not a semantic one. `Block` is
    runtime-checkable, so it tests only that `x` has `name`, `start` and
    `update` -- and those are common enough names that a progress bar or an
    online learner can pass. Use it to catch an obviously wrong object early;
    do not read a pass as "this is a block." `Wheel.add` does not rely on it
    (it checks each method itself and validates what `start` returns).

    Implementation notes:
        - `name` must be unique within a Wheel; it identifies your block in
          diagnostics and error messages.
        - Your template must live on the grid you were handed: the same
          channels, the same lengths, real in the time domain and complex in
          the frequency domain. `residual.template(...)` checks that where you
          build it; the Wheel checks it again on return.
        - The `tdi` arrays you are handed are freshly built for you each call,
          so you may reuse or overwrite them if convenient. The Wheel copies
          the template it records, so you may also keep and overwrite your own
          template buffer between cycles.
        - `residual.noise` and `residual.orbit` are shared by reference --
          treat them as immutable. A noise block publishes a *new* model with
          `residual.zero_template().with_noise(...)` rather than mutating the
          one it was handed; a signal block leaves `Template.noise` unset (the
          default), which is not the same as clearing it.
        - `residual.noise` may be `None`: either no noise model is set, or on
          the first cycle you are updated before the noise block is
          (registration order). Both `noise_psd()` and `noise_variance()`
          return `None` in that case -- guard for it rather than assuming a
          model is present.
        - Read the data conventions off the residual instead of assuming them:
          `residual.observable`, `residual.domain`, `residual.channels`. If
          your sampler only supports one convention, check these in `start` and
          raise.
        - For samplers in another language, write a thin Python wrapper that
          shells out, writes/reads files, and implements this protocol. The
          wrapper (or the process behind it) carries all the state; the Wheel
          cannot tell the difference.
    """

    name: str

    def start(self, residual: L1Data) -> Template:
        """Join a run: set yourself up and return your initial template.

        Called once when the block is added to a Wheel. `residual` is the
        observed data minus the templates of any blocks already registered,
        with the current noise model on `residual.noise`. Read run settings
        off it, set your sampler up, and return your initial template --
        `residual.zero_template()` if you start with no sources.
        """
        ...

    def update(self, residual: L1Data) -> Template:
        """Perform one block update: revise your fit, return your template.

        Args:
            residual: The observed data with every **other** block's current
                template subtracted -- not your own. This is the data your
                source class must explain; fit against it directly (no
                add-back) and return your new template. Run settings
                (`residual.fs`, `residual.Tobs`, ...) are as in `start`; the
                current noise model rides on `residual.noise`.

        Returns:
            Your new template as an `L1Data` on the same grid: `tdi` holds the
            signal you now claim (zeros for a noise block, which instead
            updates `noise`). The Wheel records it in the ledger and subtracts
            it from the data on your behalf when forming the next residual.
        """
        ...


class NoiseBlock(Block, Protocol):
    """Convention for a `Block` that models the noise, not a signal.

    Structurally identical to `Block` -- a noise block implements the same
    `start`/`update` -- but by convention its template is zero (so its ledger
    entry is zero) and what it carries instead is an updated `noise` object:

        def update(self, residual):
            model = self.estimate_noise(residual.tdi)  # residual is ~pure noise
            return residual.zero_template().with_noise(model)

    The `noise` object it puts on its template is consumed by signal blocks
    through `L1Data.noise_psd` (and `L1Data.noise_variance`, which
    integrates it for time-domain use), so it must expose

    - ``psd(freqs[, channel]) -> ndarray`` -- the one-sided PSD (see
      `L1Data.noise_psd` for the pinned normalization convention).

    `isinstance(block, NoiseBlock)` cannot tell you anything -- it returns True
    for *every* block. That is not a bug to fix: a noise block declares no
    method a signal block lacks, because the difference between them is what
    they do with `tdi` and `noise`, not their shape. Use this protocol as
    documentation and as a type annotation; to find the noise block in a
    campaign, track which one you registered for that job.

    That contract is enforced where the model is consumed (`noise_psd` raises
    if it is missing), not by the Wheel, which stays entirely noise-agnostic.
    The Wheel threads the updated noise onto every residual it forms after, so
    every block updated later sees the refreshed estimate. `Template.noise`
    defaults to `None`, which means "I publish no model" rather than "clear the
    model": a signal block that leaves it unset does not disturb yours.
    """
