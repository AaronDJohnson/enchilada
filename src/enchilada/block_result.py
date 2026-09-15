"""A block's signal, optional noise estimate, and complete sampler state."""

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from enchilada.covariance import DataCovariance
from enchilada.domains import WDMGrid, check_wdm_coefficients
from enchilada.translated_covariance import TranslatedCovariance


@dataclass(frozen=True, eq=False)
class BlockResult:
    """A complete initial state or the result of a sampling call.

    `tdi_signal_contribution` is the sum of this block's modeled signals in each
    TDI channel. None means zero contribution, removing any previous signal.
    Explicit arrays must cover every channel on the block's input grid.
    Wheel translates them to the observation grid when storing the ledger.

    `model_parameters` holds the physical values used to produce the signal
    and covariance. `sampler_state` holds algorithm continuation information
    (walkers, proposal tuning, counters, external RNG state); simple samplers
    can leave it empty. These are complete snapshots, not incremental changes.
    `metadata` holds diagnostics and provenance. These dictionaries must support
    deepcopy; the Wheel snapshots them both on adoption and before a block call.
    Keep live processes, open files, and model resources outside these dictionaries.

    `noise_covariance` replaces the full covariance; components are not summed.
    A signal block can omit it. The current covariance owner must supply its
    complete covariance on every call. A noise-only result needs no signal arrays.
    Data conventions and orbit stay with L1Data.
    Construction checks the container; Wheel checks the grid and finiteness.
    """

    tdi_signal_contribution: dict[str, np.ndarray] | None = None
    noise_covariance: DataCovariance | TranslatedCovariance | None = None
    model_parameters: dict[str, Any] = field(default_factory=dict)
    sampler_state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        signal = self.tdi_signal_contribution
        if signal is not None:
            if not isinstance(signal, dict):
                raise TypeError(
                    "BlockResult.tdi_signal_contribution must be a dict of "
                    "channel -> array or None"
                )
            if not signal:
                raise ValueError(
                    "BlockResult.tdi_signal_contribution is empty; "
                    "use None for no signal"
                )
            for ch, arr in signal.items():
                if not isinstance(arr, np.ndarray) or arr.ndim not in (1, 2):
                    raise TypeError(
                        f"BlockResult.tdi_signal_contribution[{ch!r}] must be "
                        "a 1-D or 2-D numpy array"
                    )
                if not np.issubdtype(arr.dtype, np.inexact):
                    raise TypeError(
                        f"BlockResult.tdi_signal_contribution[{ch!r}] has "
                        f"dtype {arr.dtype}; a signal must be floating or complex"
                    )
        if self.noise_covariance is not None and not isinstance(
            self.noise_covariance, (DataCovariance, TranslatedCovariance)
        ):
            raise TypeError(
                "BlockResult.noise_covariance must be a DataCovariance "
                "or TranslatedCovariance"
            )
        for name in ("model_parameters", "sampler_state", "metadata"):
            if not isinstance(getattr(self, name), dict):
                raise TypeError(f"BlockResult.{name} must be a dict")

    def with_noise_covariance(
        self, noise_covariance: DataCovariance | TranslatedCovariance
    ) -> "BlockResult":
        """Publish a replacement covariance along with this signal and state."""
        return replace(self, noise_covariance=noise_covariance)


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
    tdi_signal_contribution: dict[str, np.ndarray],
    *,
    channel_names: tuple[str, ...],
    num_time_samples: int,
    data_domain: str,
    context_label: str,
    wdm_grid: WDMGrid | None = None,
) -> None:
    """Require `tdi_signal_contribution` to match the supplied channel names and grid.

    Shared by `L1Data.block_result` (so a block fails where it builds a bad
    signal) and the Wheel (so it fails on return even for a BlockResult built
    by hand). `context_label` names the thing being checked in the message.
    """
    if set(tdi_signal_contribution) != set(channel_names):
        missing = sorted(set(channel_names) - set(tdi_signal_contribution))
        extra = sorted(set(tdi_signal_contribution) - set(channel_names))
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
        check_wdm_coefficients(tdi_signal_contribution, wdm_grid, context_label)
        return
    if wdm_grid is not None:
        raise ValueError(f"{context_label} wdm_grid requires data_domain='wdm'")
    expected = num_time_samples if data_domain == "time" else num_time_samples // 2 + 1
    for ch in channel_names:
        arr = tdi_signal_contribution[ch]
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
            tdi_signal_contribution,
            num_time_samples=num_time_samples,
            context_label=context_label,
        )
