"""What a block returns: the signal it currently claims.

A block is handed an `L1Data` residual and returns a `Template` -- its own
signal, summed over all its sources, on that residual's grid. The Wheel keeps
every block's template in a ledger and does all the subtraction itself, so a
block never performs cross-block arithmetic.

The return type is deliberately *not* `L1Data`. A block that still does the
old thing -- subtracting its template and returning the residual -- produces a
perfectly well-formed `L1Data` that no array check can distinguish from a real
template, and every block downstream would then be fitting the wrong thing.
Returning a distinct type makes that migration error a `TypeError` at
`Wheel.add`, before a campaign starts, rather than a quiet corruption of the
fit. It also means a template cannot carry the run settings, the orbit, or an
unset noise slot, so none of those can drift or be dropped in transit -- the
checks the Wheel used to need for that are gone with them.

Build one from the residual you were handed, which knows the grid:

    return residual.template({ch: amplitude * basis[ch] for ch in ...})
    return residual.zero_template()                       # nothing to subtract
    return residual.zero_template().with_noise(model)     # a noise block
"""

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True, eq=False)
class Template:
    """A block's current signal, plus the noise model if it publishes one.

    Fields:
        tdi: Channel name -> 1D array, on the grid of the residual the block
            was handed: the same channels, the same lengths, and real or
            complex to match its `domain`. This is the signal the Wheel
            subtracts from the data on the block's behalf -- not the data with
            it removed.
        noise: A noise/covariance model to publish to every other block, or
            `None` (the default) for a signal block, which publishes nothing.
            Only a noise block sets it; see `block.NoiseBlock`. Leaving it
            `None` does not clear the current model -- a signal block simply
            has no opinion about the noise.

    Prefer `L1Data.template` / `L1Data.zero_template` to constructing this
    directly: they fill in the grid from the residual and validate against it,
    so a wrong length or dtype fails where the mistake is rather than on
    return. Construction here checks only the shape of the container itself.

    Equality is identity (`eq=False`), for the same reason as `L1Data`: a
    generated `__eq__` would compare the arrays elementwise and raise "truth
    value of an array is ambiguous". Use `numpy.allclose` on the arrays.
    """

    tdi: dict[str, np.ndarray]
    noise: Any | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tdi, dict):
            raise TypeError(
                f"Template.tdi must be a dict of channel -> array, got "
                f"{type(self.tdi).__name__}"
            )
        if not self.tdi:
            raise ValueError(
                "Template.tdi is empty; a template covers every channel of the "
                "run (use residual.zero_template() when you have nothing to "
                "subtract)"
            )
        for ch, arr in self.tdi.items():
            if not isinstance(arr, np.ndarray) or arr.ndim != 1:
                raise TypeError(
                    f"Template.tdi[{ch!r}] must be a 1-D numpy array, got "
                    f"{type(arr).__name__}"
                )
            if not np.issubdtype(arr.dtype, np.inexact):
                raise TypeError(
                    f"Template.tdi[{ch!r}] has dtype {arr.dtype}; a template must "
                    f"be floating or complex (convert with .astype(float) first)"
                )

    def with_noise(self, noise: Any) -> "Template":
        """This template carrying `noise`, the model a noise block publishes.

        The way a noise block returns its estimate, since its template itself
        is zero::

            return residual.zero_template().with_noise(my_model)

        Signal blocks read the model back through `L1Data.noise_psd` (per-bin
        weight) or `L1Data.noise_variance` (per-sample variance), so it must
        expose ``psd(freqs[, channel])``.
        """
        return replace(self, noise=noise)


def check_on_grid(
    tdi: dict[str, np.ndarray],
    *,
    channels: tuple[str, ...],
    n_samples: int,
    domain: str,
    what: str,
) -> None:
    """Raise unless `tdi` lives on the grid described by the other arguments.

    Shared by `L1Data.template` (so a block fails where it builds a bad
    template) and the Wheel (so it fails on return even for a template built
    by hand). `what` names the thing being checked in the message.
    """
    if set(tdi) != set(channels):
        missing = sorted(set(channels) - set(tdi))
        extra = sorted(set(tdi) - set(channels))
        raise ValueError(
            f"{what} keys must match the run's channels exactly; "
            f"missing {missing}, unexpected {extra}"
        )
    expected = n_samples if domain == "time" else n_samples // 2 + 1
    for ch in channels:
        arr = tdi[ch]
        if arr.shape[0] != expected:
            raise ValueError(
                f"{what}[{ch!r}] has length {arr.shape[0]}, expected {expected} "
                f"for domain={domain!r} with n_samples={n_samples} (n_samples "
                f"always counts time-domain samples; frequency-domain arrays "
                f"live on the rfft grid of length n_samples // 2 + 1)"
            )
        if domain == "time" and np.iscomplexobj(arr):
            raise TypeError(
                f"{what}[{ch!r}] is complex but domain='time'; time-domain TDI "
                f"is real (did you mean domain='frequency'?)"
            )
        if domain == "frequency" and not np.iscomplexobj(arr):
            raise TypeError(
                f"{what}[{ch!r}] is real but domain='frequency'; a one-sided "
                f"spectrum is complex. Accepting a real array here would let a "
                f"block return `spectrum.real` and be silently credited with "
                f"the whole imaginary part as its template."
            )
