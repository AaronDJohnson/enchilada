"""One object-level interface for time, Fourier and WDM representations."""

from copy import deepcopy
from dataclasses import replace
from typing import overload

from enchilada._transforms import transform_channels
from enchilada.block_state import BlockState, check_on_grid
from enchilada.covariance import Covariance
from enchilada.data import L1Data
from enchilada.domains import WDMGrid, resolve_wdm_grid


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
    value: Covariance,
    target_domain: str,
    *,
    num_frequency_divisions: int | None = None,
    num_time_divisions: int | None = None,
    reference_data: L1Data | None = None,
) -> Covariance: ...


@overload
def transform(
    value: BlockState,
    target_domain: str,
    *,
    num_frequency_divisions: int | None = None,
    num_time_divisions: int | None = None,
    reference_data: L1Data | None = None,
) -> BlockState: ...


def transform(
    value: L1Data | Covariance | BlockState,
    target_domain: str,
    *,
    num_frequency_divisions: int | None = None,
    num_time_divisions: int | None = None,
    reference_data: L1Data | None = None,
) -> L1Data | Covariance | BlockState:
    """Return an independent representation of data, covariance or a block state.

    Args:
        value: Object to translate. A BlockState with a signal or covariance
            needs reference_data because its grid is intentionally external.
        target_domain: "time", "frequency", or "wdm".
        num_frequency_divisions: WDM frequency divisions (Nf, yielding Nf+1 rows).
            Specify this or num_time_divisions on first conversion into WDM.
        num_time_divisions: WDM time divisions. The other count is derived from
            num_time_samples; both counts must be positive even integers. Both
            may be supplied if consistent. WDM-to-WDM defaults to its current grid.
        reference_data: Required for a Covariance, as the L1Data on its current
            grid, and for a BlockState with a signal or covariance, as their
            source grid.

    Preserves observation metadata, orbit, and published physical parameters.
    Signal arrays follow dt*rfft and the WDM backend's normalized-edge convention.
    Covariance translation between domains is a seam that raises
    `NotImplementedError` until the translation layer is installed (design spec
    section 6).
    WDM coefficient transforms require the separately installed local backend
    and Python >=3.13. Same-grid copies and covariance construction do not.
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
        if value.data_domain == "wdm":
            # Validate against the stored grid before construction derives it anew.
            value._validate_channel_data_lengths()
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
        return replace(source, channel_data=converted, data_domain=target_domain)
    if isinstance(value, Covariance):
        # The seam for the covariance translation layer (design spec section 6).
        if reference_data is None:
            raise ValueError(
                "transform needs reference_data for a Covariance: the L1Data on "
                "its current grid"
            )
        # Unbound call: copy without trusting a subclass's overridden methods,
        # then validate the plain Covariance that results.
        copied = Covariance.copy(value)
        copied.check_compatible(reference_data)
        grid = _target_grid(
            target_domain,
            reference_data.num_time_samples,
            reference_data.wdm_grid,
            num_frequency_divisions,
            num_time_divisions,
        )
        # A time or frequency target and reference both have grid None.
        if (
            target_domain == reference_data.data_domain
            and grid == reference_data.wdm_grid
        ):
            return copied
        raise NotImplementedError(
            "covariance translation between domains is not implemented; see "
            "docs/superpowers/specs/2026-11-10-covariance-object-design.md section 6"
        )
    if isinstance(value, BlockState):
        source_state = deepcopy(value)
        source_state.__post_init__()
        signal = source_state.template
        if signal is not None:
            if reference_data is None:
                raise ValueError(
                    "a BlockState signal requires its source reference_data"
                )
            if reference_data.data_domain == "wdm":
                reference_data._validate_channel_data_lengths()
            source_reference = replace(reference_data)
            check_on_grid(
                signal,
                channel_names=source_reference.channel_names,
                num_time_samples=source_reference.num_time_samples,
                data_domain=source_reference.data_domain,
                wdm_grid=source_reference.wdm_grid,
                context_label="BlockState.template",
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
        covariance = source_state.covariance
        if covariance is not None:
            if reference_data is None:
                raise ValueError(
                    "a BlockState covariance requires its source reference_data"
                )
            covariance = transform(
                covariance,
                target_domain,
                num_frequency_divisions=num_frequency_divisions,
                num_time_divisions=num_time_divisions,
                reference_data=reference_data,
            )
        return replace(source_state, template=signal, covariance=covariance)
    raise TypeError("transform expects L1Data, Covariance, or BlockState")
