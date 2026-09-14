"""Exact covariance changes of basis without dense cross-sample matrices."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from enchilada.covariance import DataCovariance, _validate_covariance_values
from enchilada.domains import WDMGrid

if TYPE_CHECKING:
    from enchilada.data import L1Data


def _validate_target(
    data_domain: str, wdm_grid: WDMGrid | None, num_time_samples: int
) -> None:
    if data_domain not in ("time", "frequency", "wdm"):
        raise ValueError("data_domain must be 'time', 'frequency', or 'wdm'")
    if data_domain == "wdm":
        if not isinstance(wdm_grid, WDMGrid):
            raise ValueError("wdm_grid must be a WDMGrid for WDM covariance")
        if wdm_grid.num_time_samples != num_time_samples:
            raise ValueError("wdm_grid must match num_time_samples")
    elif wdm_grid is not None:
        raise ValueError("wdm_grid is only valid for data_domain='wdm'")


def _log_coordinate_scale(
    data_domain: str, sample_rate_hz: float, num_time_samples: int
) -> float:
    # Each domain uses a scaled orthogonal map from real time samples. The
    # frequency coordinates pack sqrt(2)*Re/Im internally, with real endpoints.
    if data_domain == "time":
        return 0.0
    log_dt = -np.log(sample_rate_hz)
    if data_domain == "wdm":
        return float(0.5 * log_dt)
    return float(log_dt + 0.5 * np.log(num_time_samples))


@dataclass(frozen=True, eq=False)
class TranslatedCovariance:
    """An exact operator backed by an independently owned native covariance.

    Translation generally correlates different samples, bins, or WDM pixels.
    Such covariance has no per-point ``covariance_matrix`` or boolean target
    ``active_mask``. :meth:`project` carries the native statistical exclusions
    through the change of basis, including nonlocal excluded directions.
    WDM structural edge positions are described separately by ``wdm_grid``.

    ``apply`` and ``solve`` act on real time/WDM coordinates or sqrt(2)-weighted
    real Fourier quadratures. ``solve`` is precision on the retained subspace;
    excluded data are ignored. Determinants are pseudodeterminants in those
    same coordinates. No covariance information or native mask is discarded.

    Only transforms and per-point channel operations are evaluated. The native
    covariance, not a dense transformed matrix or live backend object, is stored.
    """

    source_covariance: DataCovariance
    data_domain: str = field(kw_only=True)
    wdm_grid: WDMGrid | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.source_covariance, DataCovariance):
            raise TypeError("source_covariance must be a native DataCovariance")
        _validate_target(
            self.data_domain, self.wdm_grid, self.source_covariance.num_time_samples
        )
        object.__setattr__(
            self, "source_covariance", DataCovariance.copy(self.source_covariance)
        )

    @property
    def channel_names(self) -> tuple[str, ...]:
        return self.source_covariance.channel_names

    @property
    def sample_rate_hz(self) -> float:
        return self.source_covariance.sample_rate_hz

    @property
    def num_time_samples(self) -> int:
        return self.source_covariance.num_time_samples

    @property
    def start_time_gps(self) -> float:
        return self.source_covariance.start_time_gps

    @property
    def active_mask(self) -> None:
        """No pointwise target mask; use project for native-basis exclusions."""
        return None

    @property
    def covariance_matrix(self) -> np.ndarray:
        raise AttributeError(
            "translated covariance has cross-point correlations; "
            "use apply, solve, or quadratic_form instead of covariance_matrix"
        )

    @property
    def degrees_of_freedom(self) -> int:
        return self.source_covariance.degrees_of_freedom

    def check_compatible(self, reference_data: "L1Data") -> None:
        DataCovariance.check_compatible(self.source_covariance, reference_data)

    def _transform(
        self, values: dict[str, np.ndarray], *, to_native: bool
    ) -> dict[str, np.ndarray]:
        from enchilada._transforms import transform_channels

        if to_native:
            _validate_covariance_values(
                values,
                channel_names=self.channel_names,
                num_time_samples=self.num_time_samples,
                data_domain=self.data_domain,
                wdm_grid=self.wdm_grid,
            )
        native = self.source_covariance
        return transform_channels(
            values,
            source_domain=self.data_domain if to_native else native.data_domain,
            target_domain=native.data_domain if to_native else self.data_domain,
            num_time_samples=self.num_time_samples,
            sample_rate_hz=self.sample_rate_hz,
            source_wdm_grid=self.wdm_grid if to_native else native.wdm_grid,
            target_wdm_grid=native.wdm_grid if to_native else self.wdm_grid,
        )

    def _log_scale_ratio(self) -> float:
        return _log_coordinate_scale(
            self.data_domain, self.sample_rate_hz, self.num_time_samples
        ) - _log_coordinate_scale(
            self.source_covariance.data_domain,
            self.sample_rate_hz,
            self.num_time_samples,
        )

    def apply(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply the exact projected covariance in this representation."""
        native_values = self._transform(values, to_native=True)
        native_result = self.source_covariance.apply(native_values)
        result = self._transform(native_result, to_native=False)
        scale = np.exp(2.0 * self._log_scale_ratio())
        return {name: array * scale for name, array in result.items()}

    def solve(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply the exact precision, projecting out native excluded directions."""
        native_values = self._transform(values, to_native=True)
        native_result = self.source_covariance.solve(native_values)
        result = self._transform(native_result, to_native=False)
        scale = np.exp(-2.0 * self._log_scale_ratio())
        return {name: array * scale for name, array in result.items()}

    def project(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Project onto the retained subspace, which can span multiple pixels."""
        native_values = self._transform(values, to_native=True)
        return self._transform(
            self.source_covariance.project(native_values), to_native=False
        )

    def quadratic_form(self, values: dict[str, np.ndarray]) -> float:
        """Representation-invariant Gaussian quadratic on retained coordinates."""
        return self.source_covariance.quadratic_form(
            self._transform(values, to_native=True)
        )

    def log_determinant(self) -> float:
        """Log-pseudodeterminant, including the real-coordinate Jacobian."""
        return (
            self.source_covariance.log_determinant()
            + 2.0 * self.degrees_of_freedom * self._log_scale_ratio()
        )

    def noise_psd(self, channel_name: str | None = None) -> np.ndarray:
        raise ValueError(
            "noise_psd requires native frequency covariance; "
            "use covariance operations or translate back to its native basis"
        )

    def noise_variance(self, channel_name: str | None = None) -> float:
        raise ValueError(
            "noise_variance does not describe cross-point covariance; "
            "use covariance operations or translate back to its native basis"
        )

    def copy(self) -> "TranslatedCovariance":
        """Copy and revalidate the native covariance and target grid."""
        return TranslatedCovariance(
            self.source_covariance, data_domain=self.data_domain, wdm_grid=self.wdm_grid
        )

    def _copy_validated(self) -> "TranslatedCovariance":
        """Copy only an orchestrator-owned, already validated snapshot."""
        snapshot = object.__new__(TranslatedCovariance)
        object.__setattr__(
            snapshot,
            "source_covariance",
            DataCovariance._copy_validated(self.source_covariance),
        )
        object.__setattr__(snapshot, "data_domain", self.data_domain)
        object.__setattr__(snapshot, "wdm_grid", self.wdm_grid)
        return snapshot


def copy_covariance(
    value: DataCovariance | TranslatedCovariance, *, validated: bool = False
) -> DataCovariance | TranslatedCovariance:
    """Copy either built-in representation without trusting overridden methods.

    ``validated=True`` is reserved for orchestrator-owned snapshots that were
    never exposed to callers. External covariance must always be revalidated.
    """
    if isinstance(value, DataCovariance):
        return (
            DataCovariance._copy_validated(value)
            if validated
            else DataCovariance.copy(value)
        )
    if isinstance(value, TranslatedCovariance):
        return (
            TranslatedCovariance._copy_validated(value)
            if validated
            else TranslatedCovariance.copy(value)
        )
    raise TypeError("covariance must be a DataCovariance or TranslatedCovariance")


def translate_covariance(
    covariance: DataCovariance | TranslatedCovariance,
    target_domain: str,
    target_wdm_grid: WDMGrid | None = None,
) -> DataCovariance | TranslatedCovariance:
    """Change representation exactly, retaining the original native covariance.

    Repeated translations keep one native covariance, never a chain of views.
    Returning to its native domain and WDM grid recovers a validated native copy.
    """
    if isinstance(covariance, TranslatedCovariance):
        native = covariance.source_covariance
    elif isinstance(covariance, DataCovariance):
        native = covariance
    else:
        raise TypeError("covariance must be a DataCovariance or TranslatedCovariance")
    _validate_target(target_domain, target_wdm_grid, native.num_time_samples)
    if native.data_domain == target_domain and native.wdm_grid == target_wdm_grid:
        return DataCovariance.copy(native)
    return TranslatedCovariance(
        native, data_domain=target_domain, wdm_grid=target_wdm_grid
    )
