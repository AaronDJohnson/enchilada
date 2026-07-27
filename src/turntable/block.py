from typing import Protocol, runtime_checkable

from turntable.residuals import Residuals


@runtime_checkable
class Block(Protocol):
    """Plug a sampler into the Wheel by implementing this interface.

    The Wheel hands you a residual and takes back your updated one. It calls
    two methods on you:

    - `start` once, when you are registered. Read the run settings and
      conventions off the residual, set yourself up, subtract your initial
      model, and return the updated residual (return it unchanged if you
      start from nothing).
    - `step` once per turn of the wheel. The residual you receive is the observed
      data with **every other** block's current model already subtracted --
      **but not your own**. So it is exactly the data your source class must
      explain: fit against it directly, subtract your new model, and return
      the result. There is nothing to add back.

    No add-back. The Wheel keeps a ledger of every block's current model and
    hands you the data minus *everyone else*, so your own model is never in
    what you receive. You fit it as-is. When you return, the Wheel reads your
    new ledger entry straight off the difference between what it handed you and
    what you returned -- you never do the cross-block arithmetic, and there
    is no add-back to forget. (This is why the Wheel, not the block, owns the
    residual bookkeeping; you still own everything about your sampler.)

    What is yours. Your parameters, your RNG, your posterior chain, your
    proposal tuning, your checkpoints, and your current model all live inside
    your object (or the external process it wraps). The Wheel never sees,
    stores, or restores your sampler state; it only records the summed model
    you produce, so it can form the next residual.

    The pattern:

        from turntable import replace   # or: from dataclasses import replace

        def step(self, residual):
            tdi = dict(residual.tdi)             # keep the channels you do not touch
            for ch in residual.channels:
                data = tdi[ch]                   # already data minus OTHERS
                self.amplitude = draw_against(data, self.template)  # your sample
                tdi[ch] = data - self.amplitude * self.template     # subtract yours
            return replace(residual, tdi=tdi)

    Noise is not special. A block that models the noise instead of a signal
    removes nothing from `tdi`; it returns the residual with an updated `noise`
    object -- `replace(residual, noise=my_model)` -- so its ledger entry is
    zero, and signal blocks read the model back through `residual.noise_psd`
    (per-bin weight) or `residual.noise_variance` (per-sample variance, for a
    time-domain likelihood).
    See `NoiseBlock`.

    Implementation notes:
        - `name` must be unique within a Wheel; it identifies your block in
          diagnostics and error messages.
        - Return a `Residuals` with the same run settings you were handed
          (`channels`, `n_samples`, `sample_rate`, `domain`, `epoch`,
          `tdi_generation`, `observable`, `orbit`); only `tdi` and `noise` may
          change. The Wheel validates this after `start` and every `step`, and
          `Residuals` itself validates that your `tdi` keeps the right keys and
          shapes.
        - You are handed a fresh copy of the `tdi` arrays each call, so you may
          mutate them in place if convenient; just return the result. The
          `noise` and `orbit` objects are shared by reference -- treat them as
          immutable, swapping via `replace(residual, noise=...)` rather than
          mutating in place.
        - `residual.noise` may be `None`: no noise model is set, or you step
          before the noise block does on the first turn (registration
          order). Both `noise_psd()` and `noise_variance()` return `None` in
          that case -- guard for it rather than assuming a model is present.
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

    def start(self, residual: Residuals) -> Residuals:
        """Join a run: set yourself up and return the residual you produce.

        Called once when the block is added to a Wheel. `residual` is the
        observed data minus the models of any blocks already registered,
        with the current noise model on `residual.noise`. Read run settings
        off it, subtract your initial model, and return the updated residual --
        unchanged if you start with no model.
        """
        ...

    def step(self, residual: Residuals) -> Residuals:
        """Advance one turn of the wheel and return the updated residual.

        Args:
            residual: The observed data with every **other** block's current
                model subtracted -- not your own. This is the data your source
                class must explain; fit against it directly (no add-back),
                subtract your new model, and return the result. Run settings
                (`residual.fs`, `residual.Tobs`, ...) are as in `start`; the
                current noise model rides on `residual.noise`.

        Returns:
            The updated residual, with your new model subtracted (or, for a
            noise block, with `noise` updated). The Wheel derives your new
            ledger entry from what changed and forms the next residual.
        """
        ...


class NoiseBlock(Block, Protocol):
    """Convention for a `Block` that models the noise, not a signal.

    Structurally identical to `Block` -- a noise block implements the same
    `start`/`step` -- but by convention it leaves `tdi` untouched (so its
    ledger entry is zero) and instead returns the residual with an updated
    `noise` object:

        def step(self, residual):
            model = self.estimate_noise(residual.tdi)  # residual is ~pure noise
            return replace(residual, noise=model)

    The `noise` object it puts on the residual is consumed by signal blocks
    through `Residuals.noise_psd` (and `Residuals.noise_variance`, which
    integrates it for time-domain use), so it must expose

    - ``psd(freqs[, channel]) -> ndarray`` -- the one-sided PSD (see
      `Residuals.noise_psd` for the pinned normalization convention).

    `isinstance(seg, NoiseBlock)` cannot tell you anything -- it returns True
    for *every* block. That is not a bug to fix: a noise block declares no
    method a signal block lacks, because the difference between them is what
    they do with `tdi` and `noise`, not their shape. Use this protocol as
    documentation and as a type annotation; to find the noise block in a
    campaign, track which one you registered for that job.

    That contract is enforced where the model is consumed (`noise_psd` raises
    if it is missing), not by the Wheel, which stays entirely noise-agnostic.
    The Wheel threads the updated noise onto every residual it forms after, so
    every block stepped later sees the refreshed estimate.
    """
