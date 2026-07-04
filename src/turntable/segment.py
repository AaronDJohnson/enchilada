from typing import Any, Protocol

import numpy as np

from turntable.residuals import Residuals

Catalog = Any
"""A segment's current best-guess source list. Opaque to the Wheel -- the
segment defines its own type (a list of dicts, a numpy structured array,
a path to a file, whatever fits). Only the segment's own `render` reads it."""

State = Any
"""A segment's internal MCMC state carried across iterations: RNG, chain
position, proposal statistics, anything. Opaque to the Wheel.

Two contract points live here explicitly:

- **Your RNG is yours.** Create it in `initial_state`, carry it in State,
  advance it in `step`. The Wheel never seeds, copies, or advances it, so a
  run's reproducibility is exactly the reproducibility of each segment's
  State handling.
- **Your chain is yours.** The Wheel keeps only the *latest*
  (catalog, state) per segment -- there is no history on the Wheel side. A
  segment that wants a posterior chain accumulates it inside State (or
  writes it to disk from `step`).
"""


class Segment(Protocol):
    """Plug a sampler into the Wheel by implementing this interface.

    The Wheel does not care how you sample, what your waveform model looks
    like, or what language your sampler is written in. It calls three
    methods on you:

    - `initial_state` once, when you are registered. The observed data is
      handed in so you can read run settings (`fs`, `Tobs`, `channels`, ...)
      off it and, if useful, seed your first catalog from the data itself.
    - `step` each Gibbs iteration, with the data minus every other
      segment's current model.
    - `render` whenever the Wheel needs your current contribution to
      subtract from the data so other segments see a clean residual.

    Implementation notes:
        - `name` must be unique within a Wheel; it identifies your segment
          in checkpoints and diagnostics.
        - `render` must return a dict with the same keys as
          `observed.channels`, each array matching the observed data's
          representation (see `Residuals.domain`): length
          `observed.n_samples` in a time-domain run, length
          `observed.n_samples // 2 + 1` on the rfft grid in a
          frequency-domain run. The Wheel validates this at registration
          and after every step.
        - Read the data conventions off the residual instead of assuming
          them: `residual.observable` says what the samples physically are,
          `residual.domain` which representation this run uses, and
          `residual.channels` the (normalized) channel definitions. If your
          sampler only supports one convention, check these in
          `initial_state` and raise.
        - Own your RNG and your chain: both live in your opaque `State`
          (see the `State` docstring for the exact contract).
        - For samplers in another language, write a thin Python wrapper
          that shells out, writes/reads files, and implements this
          protocol. The Wheel cannot tell the difference.
    """

    name: str

    def initial_state(self, observed: Residuals) -> tuple[Catalog, State]:
        """Return the starting catalog and state for this segment.

        Called once when the segment is added to a Wheel. `observed` is the
        full observed-data `Residuals`; read run settings off it and use
        the data itself to seed your initial guess if helpful.
        """
        ...

    def step(self, residual: Residuals, state: State) -> tuple[Catalog, State]:
        """Advance this segment by one Gibbs iteration.

        Args:
            residual: Data with every other segment's current model
                subtracted. Sample against this as if it were the
                observed data for your source class alone. Run settings
                (`residual.fs`, `residual.Tobs`, ...) are the same as on
                the `observed` you received in `initial_state`.
            state: Whatever you returned as `State` last time (or from
                `initial_state` on the first call).

        Returns:
            (catalog, state) -- your updated source list and the state
            to carry into the next iteration.
        """
        ...

    def render(self, catalog: Catalog) -> dict[str, np.ndarray]:
        """Produce this segment's TDI contribution to the data.

        Must return one array per channel in `observed.channels`, matching
        the observed arrays' representation and length (`n_samples` samples
        in a time-domain run; `n_samples // 2 + 1` rfft bins in a
        frequency-domain run -- see `Residuals.domain`). The Wheel subtracts
        this from the observed data to form the residual seen by other
        segments, and validates the shapes at registration and after every
        step.
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

    Because noise removes nothing from the data, `render` returns zero arrays
    (one per `observed.channels`); its contribution to the global fit travels
    through `Residuals.noise`, not through residual subtraction.
    """

    def noise_model(self, catalog: Catalog) -> Any:
        """Return the noise/covariance model implied by `catalog`.

        The returned object is placed on `Residuals.noise` and consumed by
        signal segments through `Residuals.noise_psd` /
        `Residuals.noise_wdm_variance`, so it must expose at least one of:

        - ``psd(freqs[, channel]) -> ndarray`` -- one-sided PSD on the given
          frequencies, optionally per channel (A/E share a PSD, T differs);
        - ``wdm_variance(n_layers, n_time, dt, epoch[, channel]) -> ndarray``
          -- per-pixel WDM variance grid for wavelet-domain segments.

        The Wheel validates this contract when the model is produced (at
        registration and after every noise step). Called after
        `initial_state` and after every `step`.
        """
        ...
