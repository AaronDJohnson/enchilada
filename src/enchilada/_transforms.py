"""Array transforms shared by object translation and covariance operators."""

import importlib
import sys

import numpy as np

from enchilada.domains import WDMGrid


def _load_wdm():
    if sys.version_info < (3, 13):
        raise ImportError("The local wdm backend requires Python >=3.13")
    try:
        backend = importlib.import_module("wdm")
    except ModuleNotFoundError as exc:
        if exc.name != "wdm":
            raise
        raise ImportError(
            "WDM conversion requires the local wdm package. Install your checkout "
            "with `uv pip install /path/to/wdm` (Python >=3.13), then use "
            "`uv run --no-sync` to preserve the prepared environment."
        ) from exc
    if not all(
        callable(getattr(backend, name, None))
        for name in ("forward_many", "inverse_many")
    ):
        raise ImportError(
            "The imported wdm module is not the expected LISA transform backend; "
            "install the local wdm checkout with forward_many and inverse_many."
        )
    return backend


def transform_channels(
    channel_data: dict[str, np.ndarray],
    *,
    source_domain: str,
    target_domain: str,
    num_time_samples: int,
    sample_rate_hz: float,
    source_wdm_grid: WDMGrid | None = None,
    target_wdm_grid: WDMGrid | None = None,
) -> dict[str, np.ndarray]:
    """Transform validated channel grids, returning independently owned arrays.

    WDM uses the local backend's normalized edges and supported double precision.
    No model state or noise interpretation belongs in this array-only boundary.
    """
    if source_domain not in ("time", "frequency", "wdm") or target_domain not in (
        "time",
        "frequency",
        "wdm",
    ):
        raise ValueError("source_domain and target_domain must name supported domains")
    for name, values in channel_data.items():
        if not np.isfinite(values).all():
            raise ValueError(f"channel_data[{name!r}] must contain finite values")
    if source_domain == target_domain and source_wdm_grid == target_wdm_grid:
        return {name: values.copy() for name, values in channel_data.items()}
    sample_interval_s = 1.0 / sample_rate_hz
    channel_names = tuple(channel_data)
    stacked = np.stack([channel_data[name] for name in channel_names])
    backend = None
    if "wdm" in (source_domain, target_domain):
        backend = _load_wdm()
    if source_domain == "time":
        if target_domain == "wdm":
            stacked = stacked.astype(np.float64, copy=False)
        raw_spectra = np.fft.rfft(stacked, axis=-1)
    elif source_domain == "frequency":
        # Normalize in at least double precision so complex64 coefficients do
        # not underflow or overflow before the inverse transform sees them.
        stacked = stacked.astype(
            np.result_type(stacked.dtype, np.complex128), copy=False
        )
        raw_spectra = stacked / sample_interval_s
    else:
        if source_wdm_grid is None:
            raise ValueError("WDM source requires wdm_grid")
        assert backend is not None
        raw_spectra = backend.inverse_many(
            stacked.astype(np.float64, copy=False),
            source_wdm_grid.num_frequency_divisions,
            source_wdm_grid.num_time_divisions,
            sample_interval_s,
        )
        # Real WDM coefficients represent a real series. Canonicalize numerical
        # roundoff in these mathematically real Fourier endpoints.
        raw_spectra[:, 0] = raw_spectra[:, 0].real
        raw_spectra[:, -1] = raw_spectra[:, -1].real
    if target_domain == "time":
        converted = np.fft.irfft(raw_spectra, n=num_time_samples, axis=-1)
    elif target_domain == "frequency":
        converted = sample_interval_s * raw_spectra
    else:
        if target_wdm_grid is None:
            raise ValueError("WDM target requires wdm_grid")
        assert backend is not None
        converted = backend.forward_many(
            raw_spectra.astype(np.complex128, copy=False),
            target_wdm_grid.num_frequency_divisions,
            target_wdm_grid.num_time_divisions,
            sample_interval_s,
        )
        converted[:, ~target_wdm_grid.active_mask] = 0.0
    if not np.isfinite(converted).all():
        raise ValueError("domain transformation produced non-finite coefficients")
    return {
        name: values.copy()
        for name, values in zip(channel_names, converted, strict=True)
    }
