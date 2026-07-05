from typing import Any, Protocol

import numpy as np

from turntable.residuals import Residuals

Contribution = dict[str, np.ndarray]
"""A segment's current TDI model: one array per channel in
`observed.channels`, matching the observed arrays' representation (see
`Residuals.domain`). This is the only thing a segment ever hands the Wheel."""


class Segment(Protocol):
    """Plug a sampler into the Wheel by implementing this interface.

    The protocol is deliberately minimal: the Wheel passes residuals, and
    that's it. It calls two methods on you:

    - `start` once, when you are registered. Read the run settings and
      conventions off `observed`, set up whatever you need, and return your
      initial contribution (zeros if you start from nothing).
    - `step` each Gibbs iteration, with the data minus every other
      segment's current contribution. Sample however you like, update
      yourself, and return your new contribution.

    Everything about you is *yours*. Your parameters, your RNG, your
    posterior chain, your proposal tuning all live inside your object (or
    the external process it wraps) -- the Wheel never sees, stores, or
    restores them. The only things the Wheel remembers are each segment's
    latest contribution (the arrays it needs to build residuals -- the
    residual ledger) and the current noise model. Checkpointing and
    restarts are likewise yours: persist and reload yourself however fits
    your sampler; a restarted run is just `start` returning your reloaded
    contribution.

    Implementation notes:
        - `name` must be unique within a Wheel; it identifies your segment
          in diagnostics and error messages.
        - Contributions must have the same keys as `observed.channels`,
          each array matching the observed data's representation (see
          `Residuals.domain`): length `observed.n_samples` in a time-domain
          run, length `observed.n_samples // 2 + 1` on the rfft grid in a
          frequency-domain run. The Wheel validates this at registration
          and after every step.
        - The Wheel holds your returned contribution until your next
          `step`, so treat it as handed over: only write into those arrays
          inside the `step` that returns them.
        - Read the data conventions off the residual instead of assuming
          them: `residual.observable` says what the samples physically are,
          `residual.domain` which representation this run uses, and
          `residual.channels` the (normalized) channel definitions. If your
          sampler only supports one convention, check these in `start` and
          raise.
        - Treat the residual you are handed as read-only; return your model
          through your contribution instead of mutating in place
          (`turntable.testing.check_segment` enforces this).
        - For samplers in another language, write a thin Python wrapper
          that shells out, writes/reads files, and implements this
          protocol. The wrapper (or the process behind it) carries all the
          state; the Wheel cannot tell the difference.
    """

    name: str

    def start(self, observed: Residuals) -> Contribution:
        """Join a run: set yourself up and return your initial contribution.

        Called once when the segment is added to a Wheel. `observed` is the
        full observed-data `Residuals` (with the current noise model, if
        any, on `observed.noise`); read run settings off it and use the
        data itself to seed your initial guess if helpful. Return zeros if
        you start with no model.
        """
        ...

    def step(self, residual: Residuals) -> Contribution:
        """Advance one Gibbs iteration and return your updated contribution.

        Args:
            residual: Data with every other segment's current contribution
                subtracted. Sample against this as if it were the observed
                data for your source class alone. Run settings
                (`residual.fs`, `residual.Tobs`, ...) are the same as on
                the `observed` you received in `start`; the current noise
                model rides on `residual.noise`.

        Returns:
            Your new TDI contribution, which the Wheel subtracts so other
            segments see a clean residual.
        """
        ...


class NoiseSegment(Segment, Protocol):
    """A `Segment` that also models the noise/covariance, not a signal.

    A noise segment fits the power of the residual after every signal segment
    has subtracted its model. It implements the full `Segment` contract *and*
    one extra method, `noise_model`. The Wheel detects that method, and each
    iteration threads the returned model onto `Residuals.noise` for every other
    segment's `step`, so signal segments whiten against the current noise
    estimate.

    Because noise removes nothing from the data, its contributions are zero
    arrays (one per `observed.channels`); its influence on the global fit
    travels through `Residuals.noise`, not through residual subtraction.
    """

    def noise_model(self) -> Any:
        """Return the noise/covariance model implied by your current state.

        The returned object is placed on `Residuals.noise` and consumed by
        signal segments through `Residuals.noise_psd` /
        `Residuals.noise_wdm_variance`, so it must expose at least one of:

        - ``psd(freqs[, channel]) -> ndarray`` -- one-sided PSD on the given
          frequencies, optionally per channel (A/E share a PSD, T differs);
        - ``wdm_variance(n_layers, n_time, dt, epoch[, channel]) -> ndarray``
          -- per-pixel WDM variance grid for wavelet-domain segments.

        The Wheel validates this contract when the model is produced (at
        registration and after every noise step). Called after `start` and
        after every `step`.
        """
        ...
