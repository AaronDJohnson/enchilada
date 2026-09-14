"""Representation metadata independent of any optional transform backend."""

from dataclasses import dataclass

import numpy as np


def _positive_even_integer(value: int, name: str) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, bool)
        or value <= 0
        or value % 2
    ):
        raise ValueError(f"{name} must be a positive even integer, got {value!r}")
    return int(value)


@dataclass(frozen=True, kw_only=True)
class WDMGrid:
    """The WDM frequency and time divisions of an unchanged observation.

    Both division counts are positive even integers. The coefficient array has
    ``num_frequency_divisions + 1`` rows, including both edge bands. The two
    edge bands contain coefficients only at even time indices; every interior
    coefficient is active. Sampling interval and epoch belong to ``L1Data``.
    """

    num_frequency_divisions: int
    num_time_divisions: int

    def __post_init__(self) -> None:
        for name in ("num_frequency_divisions", "num_time_divisions"):
            object.__setattr__(
                self, name, _positive_even_integer(getattr(self, name), name)
            )

    @property
    def array_shape(self) -> tuple[int, int]:
        """Frequency rows, including both edges, followed by time columns."""
        return self.num_frequency_divisions + 1, self.num_time_divisions

    @property
    def num_time_samples(self) -> int:
        """Number of samples in the underlying real time series."""
        return self.num_frequency_divisions * self.num_time_divisions

    @property
    def active_mask(self) -> np.ndarray:
        """Return an independent mask of the grid's physical coefficients."""
        active = np.ones(self.array_shape, dtype=bool)
        active[[0, -1], 1::2] = False
        return active


def resolve_wdm_grid(
    num_time_samples: int,
    *,
    num_frequency_divisions: int | None = None,
    num_time_divisions: int | None = None,
) -> WDMGrid:
    """Derive a missing division count without padding or resampling the data.

    Supply at least one count. If both are supplied their product must equal
    ``num_time_samples``. Inferred counts obey the same positive-even contract
    as explicit counts.
    """
    if (
        not isinstance(num_time_samples, (int, np.integer))
        or isinstance(num_time_samples, bool)
        or num_time_samples <= 0
    ):
        raise ValueError("num_time_samples must be a positive integer")
    num_time_samples = int(num_time_samples)
    if num_frequency_divisions is None and num_time_divisions is None:
        raise ValueError(
            "supply num_frequency_divisions or num_time_divisions for the WDM grid"
        )
    if num_frequency_divisions is not None:
        num_frequency_divisions = _positive_even_integer(
            num_frequency_divisions, "num_frequency_divisions"
        )
    if num_time_divisions is not None:
        num_time_divisions = _positive_even_integer(
            num_time_divisions, "num_time_divisions"
        )
    if num_frequency_divisions is None:
        assert num_time_divisions is not None
        if num_time_samples % num_time_divisions:
            raise ValueError("num_time_divisions must divide num_time_samples exactly")
        num_frequency_divisions = num_time_samples // num_time_divisions
    if num_time_divisions is None:
        if num_time_samples % num_frequency_divisions:
            raise ValueError(
                "num_frequency_divisions must divide num_time_samples exactly"
            )
        num_time_divisions = num_time_samples // num_frequency_divisions
    grid = WDMGrid(
        num_frequency_divisions=num_frequency_divisions,
        num_time_divisions=num_time_divisions,
    )
    if grid.num_time_samples != num_time_samples:
        raise ValueError("WDM division product must equal num_time_samples")
    return grid


def check_wdm_coefficients(
    channel_data: dict[str, np.ndarray],
    wdm_grid: WDMGrid,
    context_label: str,
) -> None:
    """Require real coefficient grids with exactly zero inactive entries.

    The WDM inverse discards inactive coefficients. Reject them here rather
    than silently losing caller data. The translator zeros its own numerical
    inactive tails before constructing an ``L1Data``. Active-value finiteness
    remains the responsibility of the Wheel and transform boundaries.
    """
    inactive = ~wdm_grid.active_mask
    for channel_name, coefficients in channel_data.items():
        label = f"{context_label}[{channel_name!r}]"
        if not isinstance(coefficients, np.ndarray):
            raise TypeError(f"{label} must be a numpy array")
        if coefficients.shape != wdm_grid.array_shape:
            raise ValueError(
                f"{label} must have WDM shape {wdm_grid.array_shape}, "
                f"got {coefficients.shape}"
            )
        if not np.issubdtype(coefficients.dtype, np.floating):
            raise TypeError(f"{label} WDM coefficients must be real floating values")
        if np.any(coefficients[inactive] != 0):
            raise ValueError(f"{label} inactive WDM coefficients must be exactly zero")
