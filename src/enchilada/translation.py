"""One object-level interface for time, Fourier and WDM representations."""

from copy import deepcopy
from dataclasses import replace
from typing import overload

from enchilada._transforms import transform_channels
from enchilada.block_result import BlockResult, check_on_grid
from enchilada.covariance import DataCovariance
from enchilada.data import L1Data
from enchilada.domains import WDMGrid, resolve_wdm_grid
from enchilada.translated_covariance import (
    TranslatedCovariance,
    copy_covariance,
    translate_covariance,
)


def _target_grid(
    target_domain: str,
    num_time_samples: int,
    source_wdm_grid: WDMGrid | None,
    num_frequency_divisions: int | None,
    num_time_divisions: int | None,
) -> WDMGrid | None:
    if target_domain not in L1Data.DOMAINS:
        raise ValueError(f"target_domain must be one of {L1Data.DOMAINS}")
    if target_domain != "wdm":
        if num_frequency_divisions is not None or num_time_divisions is not None:
            raise ValueError("division counts apply only to target_domain='wdm'")
        return None
    if num_frequency_divisions is None and num_time_divisions is None:
        if source_wdm_grid is not None:
            return source_wdm_grid
    return resolve_wdm_grid(
        num_time_samples,
        num_frequency_divisions=num_frequency_divisions,
        num_time_divisions=num_time_divisions,
    )


@overload
def transform(
    value: L1Data,
    target_domain: str,
    *,
    num_frequency_divisions: int | None = None,
    num_time_divisions: int | None = None,
    reference_data: L1Data | None = None,
) -> L1Data: ...


@overload
def transform(
    value: DataCovariance | TranslatedCovariance,
    target_domain: str,
    *,
    num_frequency_divisions: int | None = None,
    num_time_divisions: int | None = None,
    reference_data: L1Data | None = None,
) -> DataCovariance | TranslatedCovariance: ...


@overload
def transform(
    value: BlockResult,
    target_domain: str,
    *,
    num_frequency_divisions: int | None = None,
    num_time_divisions: int | None = None,
    reference_data: L1Data | None = None,
) -> BlockResult: ...


def transform(
    value: L1Data | DataCovariance | TranslatedCovariance | BlockResult,
    target_domain: str,
    *,
    num_frequency_divisions: int | None = None,
    num_time_divisions: int | None = None,
    reference_data: L1Data | None = None,
) -> L1Data | DataCovariance | TranslatedCovariance | BlockResult:
    """Return an independent representation of data, covariance or a block result.

    Args:
        value: Object to translate. A BlockResult with a signal needs
            reference_data because its observation grid is intentionally external.
        target_domain: "time", "frequency", or "wdm".
        num_frequency_divisions: WDM frequency divisions (Nf, yielding Nf+1 rows).
            Specify this or num_time_divisions on first conversion into WDM.
        num_time_divisions: WDM time divisions. The other count is derived from
            num_time_samples; both counts must be positive even integers. Both
            may be supplied if consistent. WDM-to-WDM defaults to its current grid.
        reference_data: Source signal grid for a BlockResult. If supplied for a
            covariance, its channel names, sampling grid and epoch must agree.

    Preserves physical metadata, orbit, model parameters and sampler state.
    Signal arrays follow dt*rfft and the WDM backend's normalized-edge convention.
    Covariance conversion preserves correlations and statistical exclusions;
    a TranslatedCovariance provides exact operators instead of a diagonal grid.
    WDM requires the separately installed local backend and Python >=3.13.
    """
    if target_domain not in L1Data.DOMAINS:
        raise ValueError(f"target_domain must be one of {L1Data.DOMAINS}")
    if target_domain != "wdm" and (
        num_frequency_divisions is not None or num_time_divisions is not None
    ):
        raise ValueError("division counts apply only to target_domain='wdm'")
    if reference_data is not None and not isinstance(reference_data, L1Data):
        raise TypeError("reference_data must be L1Data")
    if isinstance(value, L1Data):
        source = replace(value)  # Revalidate potentially edited input buffers.
        grid = _target_grid(
            target_domain,
            source.num_time_samples,
            source.wdm_grid,
            num_frequency_divisions,
            num_time_divisions,
        )
        converted = transform_channels(
            source.channel_data,
            source_domain=source.data_domain,
            target_domain=target_domain,
            num_time_samples=source.num_time_samples,
            sample_rate_hz=source.sample_rate_hz,
            source_wdm_grid=source.wdm_grid,
            target_wdm_grid=grid,
        )
        return replace(
            source, channel_data=converted, data_domain=target_domain, wdm_grid=grid
        )
    if isinstance(value, (DataCovariance, TranslatedCovariance)):
        value = copy_covariance(value)
        if reference_data is not None:
            value.check_compatible(reference_data)
        grid = _target_grid(
            target_domain,
            value.num_time_samples,
            value.wdm_grid,
            num_frequency_divisions,
            num_time_divisions,
        )
        return translate_covariance(value, target_domain, target_wdm_grid=grid)
    if isinstance(value, BlockResult):
        source_result = deepcopy(value)
        source_result.__post_init__()
        signal = source_result.tdi_signal_contribution
        if signal is not None:
            if reference_data is None:
                raise ValueError(
                    "a BlockResult signal requires its source reference_data"
                )
            source_reference = replace(reference_data)
            check_on_grid(
                signal,
                channel_names=source_reference.channel_names,
                num_time_samples=source_reference.num_time_samples,
                data_domain=source_reference.data_domain,
                wdm_grid=source_reference.wdm_grid,
                context_label="BlockResult.tdi_signal_contribution",
            )
            grid = _target_grid(
                target_domain,
                source_reference.num_time_samples,
                source_reference.wdm_grid,
                num_frequency_divisions,
                num_time_divisions,
            )
            signal = transform_channels(
                signal,
                source_domain=source_reference.data_domain,
                target_domain=target_domain,
                num_time_samples=source_reference.num_time_samples,
                sample_rate_hz=source_reference.sample_rate_hz,
                source_wdm_grid=source_reference.wdm_grid,
                target_wdm_grid=grid,
            )
        covariance = source_result.noise_covariance
        if covariance is not None:
            # A covariance carries its own source representation, which can
            # differ from the signal's. Share the chosen target grid instead.
            if (
                signal is not None
                and reference_data is not None
                and target_domain == "wdm"
            ):
                grid = _target_grid(
                    target_domain,
                    reference_data.num_time_samples,
                    reference_data.wdm_grid,
                    num_frequency_divisions,
                    num_time_divisions,
                )
                assert grid is not None  # target_domain is WDM here.
                num_frequency_divisions = grid.num_frequency_divisions
                num_time_divisions = grid.num_time_divisions
            covariance = transform(
                covariance,
                target_domain,
                num_frequency_divisions=num_frequency_divisions,
                num_time_divisions=num_time_divisions,
                reference_data=reference_data,
            )
        return replace(
            source_result, tdi_signal_contribution=signal, noise_covariance=covariance
        )
    raise TypeError("transform expects L1Data, a covariance, or BlockResult")
