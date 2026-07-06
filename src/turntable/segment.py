from typing import Protocol

from turntable.residuals import Residuals


class Segment(Protocol):
    """Plug a sampler into the Wheel by implementing this interface.

    The Wheel passes one residual around the ring and does no arithmetic of
    its own: it hands you the current residual, and you hand back the new
    one. It calls two methods on you:

    - `start` once, when you are registered. Read the run settings and
      conventions off the residual, set yourself up, subtract your initial
      model, and return the updated residual (return it unchanged if you
      start from nothing).
    - `step` each Gibbs iteration. You receive the current residual -- the
      observed data with *every* segment's current model subtracted,
      including your own. Recover the data you fit by adding your own
      current model back, sample, subtract your new model, and return the
      updated residual.

    The residual is the only thing that crosses the boundary. Everything
    about you -- your parameters, your RNG, your posterior chain, your
    proposal tuning, your checkpoints, and crucially *your own current
    model* -- lives inside your object (or the external process it wraps).
    The Wheel never sees, stores, or restores any of it; it holds only the
    single running residual.

    The add-back. Because the residual you receive already has your own
    model subtracted, you must add it back before sampling, or you will fit
    the data-minus-yourself and your model will collapse. The pattern:

        def step(self, residual):
            s = self.template                       # your model's basis
            mine_old = self.amplitude * s           # what you last subtracted
            data = residual.tdi[ch] + mine_old      # data minus *others*
            self.amplitude = draw_against(data, s)  # your conditional sample
            mine_new = self.amplitude * s
            new_tdi = {ch: residual.tdi[ch] + mine_old - mine_new, ...}
            return replace(residual, tdi=new_tdi)

    A forgotten add-back still produces a well-formed residual, so
    `turntable.testing.check_segment` cannot catch it -- guard it yourself
    with a known-truth recovery test (`examples/toy_fit.py` is the pattern).

    Noise is not special. A segment that models the noise instead of a
    signal removes nothing from `tdi`; it returns the residual with an
    updated `noise` object -- `replace(residual, noise=my_model)` -- and
    signal segments read it back through `residual.noise_psd`. The Wheel
    does not know or care which segments are noise. See `NoiseSegment` for
    the convention.

    Implementation notes:
        - `name` must be unique within a Wheel; it identifies your segment
          in diagnostics and error messages.
        - Return a `Residuals` with the same run settings you were handed
          (`channels`, `n_samples`, `sample_rate`, `domain`, `epoch`,
          `tdi_generation`, `observable`, `orbit`); only `tdi` and `noise`
          may change. The Wheel validates this after `start` and every
          `step`, and `Residuals` itself validates that your `tdi` arrays
          keep the right keys and shapes.
        - You are handed a fresh copy of the `tdi` arrays each call, so you
          may mutate them in place if that is convenient; just return the
          result. The `noise` and `orbit` objects are shared by reference,
          though -- treat them as immutable, swapping via
          `replace(residual, noise=...)` rather than mutating in place.
        - `residual.noise` may be `None`: no noise model is set, or you step
          before the noise segment does on the first sweep (registration
          order). Guard `noise_psd()`/`residual.noise` for `None` rather
          than assuming a model is always present.
        - Read the data conventions off the residual instead of assuming
          them: `residual.observable` says what the samples physically are,
          `residual.domain` which representation this run uses, and
          `residual.channels` the (normalized) channel definitions. If your
          sampler only supports one convention, check these in `start` and
          raise.
        - For samplers in another language, write a thin Python wrapper
          that shells out, writes/reads files, and implements this
          protocol. The wrapper (or the process behind it) carries all the
          state; the Wheel cannot tell the difference.
    """

    name: str

    def start(self, residual: Residuals) -> Residuals:
        """Join a run: set yourself up and return the residual you produce.

        Called once when the segment is added to a Wheel. `residual` is the
        current running residual (the observed data minus the models of any
        segments already registered, with the current noise model on
        `residual.noise`). Read run settings off it, subtract your initial
        model, and return the updated residual -- unchanged if you start
        with no model.
        """
        ...

    def step(self, residual: Residuals) -> Residuals:
        """Advance one Gibbs iteration and return the updated residual.

        Args:
            residual: The observed data with every segment's current model
                subtracted, *including your own*. Add your own current model
                back to recover the data for your source class alone (the
                "add-back"; see the class docstring), sample against that,
                subtract your new model, and return the result. Run settings
                (`residual.fs`, `residual.Tobs`, ...) are as in `start`; the
                current noise model rides on `residual.noise`.

        Returns:
            The updated residual, with your new model subtracted (or, for a
            noise segment, with `noise` updated). The Wheel passes it
            straight to the next segment.
        """
        ...


class NoiseSegment(Segment, Protocol):
    """Convention for a `Segment` that models the noise, not a signal.

    Structurally identical to `Segment` -- a noise segment implements the
    same `start`/`step` -- but by convention it leaves `tdi` untouched and
    instead returns the residual with an updated `noise` object:

        def step(self, residual):
            model = self.estimate_noise(residual.tdi)  # residual is ~pure noise
            return replace(residual, noise=model)

    The `noise` object it puts on the residual is consumed by signal
    segments through `Residuals.noise_psd`, so it must expose

    - ``psd(freqs[, channel]) -> ndarray`` -- the one-sided PSD (see
      `Residuals.noise_psd` for the pinned normalization convention).

    That contract is enforced where the model is consumed (`noise_psd`
    raises if it is missing), not by the Wheel, which stays entirely
    noise-agnostic. Because the noise rides on the passed residual, every
    segment stepped after the noise segment in a sweep automatically sees
    the refreshed estimate.
    """
