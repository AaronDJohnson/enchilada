"""A block's published template, covariance, and physical parameters."""

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from enchilada.covariance import DataCovariance
from enchilada.domains import WDMGrid, check_wdm_coefficients
from enchilada.translated_covariance import TranslatedCovariance


@dataclass(frozen=True, eq=False)
class BlockState:
    """A complete snapshot of a block's physical estimates.

    `template` is the sum of this block's modeled signals in each
    TDI channel. None means zero contribution, removing any previous signal.
    Explicit arrays must cover every channel on the block's input grid.
    Wheel translates them to the observation grid when storing the ledger.

    `parameters` holds the physical values used to produce the template and
    covariance. Values must support deepcopy: Wheel snapshots them on adoption
    and before each call. These are complete snapshots, not incremental changes.
    Sampler continuation state, diagnostics, and resources belong to the block's
    own State object; they are not published to the ledger.

    `covariance_matrix` is a DataCovariance or TranslatedCovariance replacing the
    full covariance; components are not summed.
    A signal block can omit it. The current covariance owner must supply its
    complete covariance on every call. A noise-only state needs no signal arrays.
    Data conventions and orbit stay with L1Data.
    Construction checks the container; Wheel checks the grid and finiteness.
    """

    template: dict[str, np.ndarray] | None = None
    covariance_matrix: DataCovariance | TranslatedCovariance | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        signal = self.template
        if signal is not None:
            if not isinstance(signal, dict):
                raise TypeError(
                    "BlockState.template must be a dict of channel -> array or None"
                )
            if not signal:
                raise ValueError("BlockState.template is empty; use None for no signal")
            for ch, arr in signal.items():
                if not isinstance(arr, np.ndarray) or arr.ndim not in (1, 2):
                    raise TypeError(
                        f"BlockState.template[{ch!r}] must be a 1-D or 2-D numpy array"
                    )
                if not np.issubdtype(arr.dtype, np.inexact):
                    raise TypeError(
                        f"BlockState.template[{ch!r}] has "
                        f"dtype {arr.dtype}; a signal must be floating or complex"
                    )
        if self.covariance_matrix is not None and not isinstance(
            self.covariance_matrix, (DataCovariance, TranslatedCovariance)
        ):
            raise TypeError(
                "BlockState.covariance_matrix must be a DataCovariance "
                "or TranslatedCovariance"
            )
        if not isinstance(self.parameters, dict):
            raise TypeError("BlockState.parameters must be a dict")

    def with_covariance_matrix(
        self, covariance_matrix: DataCovariance | TranslatedCovariance
    ) -> "BlockState":
        """Return a shallow replacement with the supplied covariance.

        Other fields remain shared. Publication occurs only when Wheel adopts
        the returned state.
        """
        return replace(self, covariance_matrix=covariance_matrix)


def check_rfft_endpoints(
    channel_data: dict[str, np.ndarray],
    *,
    num_time_samples: int,
    context_label: str,
) -> None:
    """Require the real-valued endpoints of spectra for real time series.

    Call after validating channel array shapes and dtypes. DC is always real;
    only an even-length time series has a Nyquist bin at the final index.
    Reject invalid components instead of letting irfft silently discard them.
    """
    for channel_name, spectrum in channel_data.items():
        if spectrum[0].imag != 0:
            raise ValueError(
                f"{context_label}[{channel_name!r}] DC coefficient must be real"
            )
        if num_time_samples % 2 == 0 and spectrum[-1].imag != 0:
            raise ValueError(
                f"{context_label}[{channel_name!r}] Nyquist coefficient must be "
                "real for even num_time_samples"
            )


def check_on_grid(
    template: dict[str, np.ndarray],
    *,
    channel_names: tuple[str, ...],
    num_time_samples: int,
    data_domain: str,
    context_label: str,
    wdm_grid: WDMGrid | None = None,
) -> None:
    """Require `template` to match the supplied channel names and grid.

    Shared by `L1Data.block_state` (so a block fails where it builds a bad
    signal) and the Wheel (so it fails on return even for a BlockState built
    by hand). `context_label` names the thing being checked in the message.
    """
    if set(template) != set(channel_names):
        missing = sorted(set(channel_names) - set(template))
        extra = sorted(set(template) - set(channel_names))
        raise ValueError(
            f"{context_label} keys must match the run's channels exactly; "
            f"missing {missing}, unexpected {extra}"
        )
    if data_domain == "wdm":
        if wdm_grid is None:
            raise ValueError(f"{context_label} requires wdm_grid for data_domain='wdm'")
        if wdm_grid.num_time_samples != num_time_samples:
            raise ValueError(
                f"{context_label} WDM grid does not match num_time_samples"
            )
        check_wdm_coefficients(template, wdm_grid, context_label)
        return
    if wdm_grid is not None:
        raise ValueError(f"{context_label} wdm_grid requires data_domain='wdm'")
    expected = num_time_samples if data_domain == "time" else num_time_samples // 2 + 1
    for ch in channel_names:
        arr = template[ch]
        if arr.ndim != 1:
            raise TypeError(
                f"{context_label}[{ch!r}] must be a 1-D numpy array "
                f"for data_domain={data_domain!r}"
            )
        if arr.shape[0] != expected:
            raise ValueError(
                f"{context_label}[{ch!r}] has length {arr.shape[0]}, "
                f"expected {expected} for data_domain={data_domain!r} "
                f"with num_time_samples={num_time_samples} "
                f"(num_time_samples "
                f"always counts time-domain samples; frequency-domain arrays "
                f"live on the rfft grid of length num_time_samples // 2 + 1)"
            )
        if data_domain == "time" and np.iscomplexobj(arr):
            raise TypeError(
                f"{context_label}[{ch!r}] is complex but data_domain='time'; "
                f"time-domain TDI is real (did you mean data_domain='frequency'?)"
            )
        if data_domain == "frequency" and not np.iscomplexobj(arr):
            raise TypeError(
                f"{context_label}[{ch!r}] is real but data_domain='frequency'; "
                f"a one-sided spectrum is complex. "
                f"Accepting a real array here would let a "
                f"block return `spectrum.real` and be silently credited with "
                f"the whole imaginary part as its signal contribution."
            )
    if data_domain == "frequency":
        check_rfft_endpoints(
            template,
            num_time_samples=num_time_samples,
            context_label=context_label,
        )
