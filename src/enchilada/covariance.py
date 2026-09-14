"""Channel covariance on an explicitly identified data grid."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from inspect import signature
from typing import TYPE_CHECKING, Any

import numpy as np

from enchilada.domains import WDMGrid

if TYPE_CHECKING:
    from enchilada.data import L1Data


def _validate_covariance_values(
    values: dict[str, np.ndarray],
    *,
    channel_names: tuple[str, ...],
    num_time_samples: int,
    data_domain: str,
    wdm_grid: WDMGrid | None,
) -> None:
    """Validate operands before a transform can discard invalid coordinates."""
    from enchilada.block_result import check_on_grid

    if not isinstance(values, dict) or set(values) != set(channel_names):
        raise ValueError("values keys must match covariance channel_names exactly")
    for array in values.values():
        if not isinstance(array, np.ndarray) or array.dtype.kind not in "fc":
            raise TypeError("values must contain floating-point numpy arrays")
        if not np.isfinite(array).all():
            raise ValueError("values must contain finite floating-point numbers")
    check_on_grid(
        values,
        channel_names=channel_names,
        num_time_samples=num_time_samples,
        data_domain=data_domain,
        context_label="values",
        wdm_grid=wdm_grid,
    )


@dataclass(frozen=True, eq=False)
class DataCovariance:
    """Covariance ``E[x_i x_j.conj()]`` between channels at each grid point.

    ``covariance_matrix`` has shape ``(n_points, n_channels, n_channels)``. Points are
    independent: this representation supports channel correlations, but not
    correlations between different times or Fourier bins. In the time domain,
    points are the ``num_time_samples`` real samples. In the frequency domain, they
    are the ``num_time_samples // 2 + 1`` coefficients of ``dt * rfft(x)``. These are
    coefficient covariances, **not** spectral densities. :meth:`from_psd`
    converts the established one-sided PSD convention explicitly.

    ``active_mask`` identifies points used in inference: True includes a point,
    False excludes it. None includes all points. Every matrix is finite and
    Hermitian; active matrices must additionally be positive definite.
    Inactive points may have zero covariance, avoiding infinities in matrix
    arithmetic. DC and even-length Nyquist coefficients are real.

    Symmetry is checked relative to each pair of channel standard deviations.
    Accepted numerical roundoff is symmetrized before testing positive
    definiteness, so the stored matrix is exactly Hermitian.

    In the WDM domain the leading matrix dimensions are ``(Nf+1, Nt)``;
    structural edge pixels are excluded by ``wdm_grid.active_mask``. Frequency
    coefficients are proper complex random variables at interior bins (zero
    pseudocovariance), with real DC and even-length Nyquist coefficients.

    Operations use real coordinates: frequency interiors are packed as
    ``sqrt(2) * Re(Z), sqrt(2) * Im(Z)`` and endpoints as ``Re(Z)``. This
    convention counts each real degree of freedom once in quadratic forms and
    determinants. Inactive points are projected out, not zero-variance data
    constraints. Input arrays are copied and made read-only.

    Fields:
        covariance_matrix: Channel covariance at each sample, Fourier bin, or
            WDM pixel, shaped (*grid_shape, num_channels, num_channels).
        channel_names: Ordered TDI channel names for both matrix channel axes.
        sample_rate_hz: Sampling frequency of the underlying time series, in Hz.
        num_time_samples: Number of underlying time samples per channel, even
            when the covariance is represented in the frequency domain.
        data_domain: "time", "frequency" (default), or "wdm"; determines the grid
            points and the real/complex coefficient convention.
        start_time_gps: GPS seconds at sample index zero; defaults to 0.0 for
            synthetic data without an absolute-time reference.
        active_mask: Boolean array with one entry per grid point. None selects
            all physical points and is normalized to an explicit mask.
        wdm_grid: Required frequency/time division metadata for WDM covariance;
            omitted in time and frequency representations.
    """

    covariance_matrix: np.ndarray
    channel_names: tuple[str, ...] = field(kw_only=True)
    sample_rate_hz: float = field(kw_only=True)
    num_time_samples: int = field(kw_only=True)
    data_domain: str = field(default="frequency", kw_only=True)
    start_time_gps: float = field(default=0.0, kw_only=True)
    active_mask: np.ndarray | None = field(default=None, kw_only=True)
    wdm_grid: WDMGrid | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.data_domain not in ("time", "frequency", "wdm"):
            raise ValueError("data_domain must be 'time', 'frequency', or 'wdm'")
        if (
            not isinstance(self.num_time_samples, (int, np.integer))
            or isinstance(self.num_time_samples, bool)
            or self.num_time_samples <= 0
        ):
            raise ValueError("num_time_samples must be a positive integer")
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive and finite, in Hz")
        if not np.isfinite(self.start_time_gps):
            raise ValueError("start_time_gps must be finite GPS seconds")
        if not isinstance(self.channel_names, (tuple, list)) or not self.channel_names:
            raise ValueError("channel_names must be a non-empty ordered sequence")
        channels = tuple(self.channel_names)
        if any(not isinstance(ch, str) or not ch for ch in channels) or len(
            set(channels)
        ) != len(channels):
            raise ValueError("channel_names must contain unique non-empty names")
        object.__setattr__(self, "channel_names", channels)
        grid_shape: tuple[int, ...]
        if self.data_domain == "wdm":
            if not isinstance(self.wdm_grid, WDMGrid):
                raise ValueError("wdm_grid must be a WDMGrid for WDM covariance")
            if self.wdm_grid.num_time_samples != self.num_time_samples:
                raise ValueError("wdm_grid must match num_time_samples")
            grid_shape = self.wdm_grid.array_shape
            structural_mask = self.wdm_grid.active_mask
        else:
            if self.wdm_grid is not None:
                raise ValueError("wdm_grid is only valid for data_domain='wdm'")
            grid_shape = (
                self.num_time_samples
                if self.data_domain == "time"
                else self.num_time_samples // 2 + 1,
            )
            structural_mask = np.ones(grid_shape, dtype=bool)
        raw = np.asarray(self.covariance_matrix)
        if raw.dtype.kind not in "fciu":
            raise TypeError(
                "covariance_matrix must contain real or complex numeric values"
            )
        matrix = np.array(raw, dtype=complex if np.iscomplexobj(raw) else float)
        expected = (*grid_shape, len(channels), len(channels))
        if matrix.shape != expected:
            raise ValueError(
                f"covariance_matrix shape must be {expected}, got {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(
                "covariance_matrix must be finite, including inactive points"
            )
        active = (
            structural_mask.copy()
            if self.active_mask is None
            else np.array(self.active_mask, copy=True)
        )
        if active.dtype != np.dtype(bool) or active.shape != grid_shape:
            raise ValueError(
                f"active_mask must be a boolean mask with shape {grid_shape}"
            )
        if np.any(active & ~structural_mask):
            raise ValueError("active_mask cannot include inactive WDM edge pixels")
        # Validate all points with one common channel-matrix implementation.
        matrix = matrix.reshape((-1, len(channels), len(channels)))
        active = active.reshape(-1)

        adjoint = matrix.conj().swapaxes(-1, -2)
        channel_scales = np.sqrt(np.abs(matrix.diagonal(axis1=-2, axis2=-1).real))
        larger_scale = np.maximum(
            channel_scales[:, :, None], channel_scales[:, None, :]
        )
        smaller_scale = np.minimum(
            channel_scales[:, :, None], channel_scales[:, None, :]
        )
        # Test symmetry in channel-normalized units. A single loud channel must
        # not hide a significant asymmetry in a quiet channel's covariance.
        # Multiply the tolerance into the larger standard deviation first so
        # the geometric scale neither overflows nor needlessly underflows.
        with np.errstate(over="ignore", under="ignore"):
            symmetry_tolerance = (1e-12 * larger_scale) * smaller_scale
            asymmetry = np.abs(matrix - adjoint)
        if np.any(asymmetry > symmetry_tolerance):
            raise ValueError("covariance_matrix must be Hermitian at every grid point")
        real_points = matrix if self.data_domain != "frequency" else matrix[[0]]
        if self.data_domain == "frequency" and self.num_time_samples % 2 == 0:
            real_points = matrix[[0, -1]]
        if np.any(real_points.imag != 0):
            raise ValueError(
                "covariance of real samples, DC, Nyquist and WDM coefficients "
                "must be real"
            )

        # Canonicalize tolerated roundoff before checking positive definiteness.
        # Cholesky reads only one triangle; storing both input triangles would
        # otherwise leave downstream solves using a different matrix. Compute
        # each mean once and reflect it so the stored result is exactly Hermitian.
        row_indices, column_indices = np.triu_indices(len(channels), k=1)
        upper = matrix[:, row_indices, column_indices]
        reflected_lower = matrix[:, column_indices, row_indices].conj()
        with np.errstate(under="ignore"):
            symmetric_upper = upper + (reflected_lower - upper) * 0.5
        matrix[:, row_indices, column_indices] = symmetric_upper
        matrix[:, column_indices, row_indices] = symmetric_upper.conj()
        diagonal_indices = np.arange(len(channels))
        matrix[:, diagonal_indices, diagonal_indices] = matrix.diagonal(
            axis1=-2, axis2=-1
        ).real
        try:
            np.linalg.cholesky(matrix[active])
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "active covariance matrices must be positive definite"
            ) from exc
        matrix = matrix.reshape(expected)
        active = active.reshape(grid_shape)
        matrix.setflags(write=False)
        active.setflags(write=False)
        object.__setattr__(self, "covariance_matrix", matrix)
        object.__setattr__(self, "active_mask", active)

    @classmethod
    def from_psd(
        cls, reference_data: "L1Data", noise_psd_model: Any
    ) -> "DataCovariance":
        """Evaluate ``noise_psd_model.psd(freqs[, channel])`` on the positive rfft grid.

        Returns frequency covariance even when ``reference_data`` is a time series.
        The one-sided density ``S`` becomes ``C = (Tobs / 2) * S``. The same
        covariance conversion applies at Nyquist, whose coefficient is real
        and has one degree of freedom. DC is zero and inactive, enforcing the
        zero-mean convention. Channels are independent in this adapter;
        construct a matrix directly for correlated channels.
        """
        method = getattr(noise_psd_model, "psd", None)
        if not callable(method):
            raise TypeError("noise model must expose psd(freqs[, channel])")
        frequencies = np.fft.rfftfreq(
            reference_data.num_time_samples, d=1.0 / reference_data.sample_rate_hz
        )[1:]
        matrix = np.zeros(
            (
                frequencies.size + 1,
                len(reference_data.channel_names),
                len(reference_data.channel_names),
            )
        )
        for index, channel in enumerate(reference_data.channel_names):
            if frequencies.size == 0:
                continue
            # Bind before calling: a TypeError inside a model must propagate,
            # rather than being mistaken for its accepting only frequencies.
            try:
                signature(method).bind(frequencies, channel)
            except TypeError:
                psd_values = method(frequencies)
            else:
                psd_values = method(frequencies, channel)
            psd_values = np.asarray(psd_values)
            if np.iscomplexobj(psd_values) or psd_values.dtype.kind not in "fiu":
                raise TypeError("noise model PSD must contain real numeric values")
            if psd_values.shape not in ((), frequencies.shape):
                raise ValueError("noise model PSD shape must match the frequency grid")
            if not np.isfinite(psd_values).all() or np.any(psd_values <= 0):
                raise ValueError("noise model PSD must be finite and strictly positive")
            matrix[1:, index, index] = (
                reference_data.num_time_samples / (2 * reference_data.sample_rate_hz)
            ) * psd_values
        active = np.ones(matrix.shape[0], dtype=bool)
        active[0] = False
        return cls(
            matrix,
            channel_names=reference_data.channel_names,
            sample_rate_hz=reference_data.sample_rate_hz,
            num_time_samples=reference_data.num_time_samples,
            start_time_gps=reference_data.start_time_gps,
            data_domain="frequency",
            active_mask=active,
        )

    @classmethod
    def from_variance(
        cls, reference_data: "L1Data", time_sample_variance: float | Mapping[str, float]
    ) -> "DataCovariance":
        """Construct independent white time noise with constant channel variances.

        ``time_sample_variance`` is a scalar shared by all channels, or a mapping with
        exactly the run's channel names. No FFT or density scaling is applied.
        """
        if isinstance(time_sample_variance, Mapping):
            if set(time_sample_variance) != set(reference_data.channel_names):
                raise ValueError(
                    "time_sample_variance keys must match channels exactly"
                )
            diagonal = np.asarray(
                [time_sample_variance[ch] for ch in reference_data.channel_names]
            )
        else:
            scalar = np.asarray(time_sample_variance)
            if scalar.ndim != 0 or scalar.dtype.kind not in "fiu":
                raise ValueError(
                    "time_sample_variance must be a real scalar for each channel"
                )
            diagonal = np.full(len(reference_data.channel_names), time_sample_variance)
        if (
            diagonal.shape != (len(reference_data.channel_names),)
            or diagonal.dtype.kind not in "fiu"
        ):
            raise ValueError(
                "time_sample_variance must be a real scalar for each channel"
            )
        matrix = np.zeros(
            (
                reference_data.num_time_samples,
                len(reference_data.channel_names),
                len(reference_data.channel_names),
            )
        )
        indices = np.arange(len(reference_data.channel_names))
        matrix[:, indices, indices] = diagonal
        return cls(
            matrix,
            channel_names=reference_data.channel_names,
            sample_rate_hz=reference_data.sample_rate_hz,
            num_time_samples=reference_data.num_time_samples,
            start_time_gps=reference_data.start_time_gps,
            data_domain="time",
        )

    def check_compatible(self, reference_data: "L1Data") -> None:
        """Require the same channels, sampling grid, and epoch as ``reference_data``.

        Representation may differ: a time-domain block can consume a spectral
        model through :meth:`noise_variance`. Matrix operations must still
        honor this object's explicit ``data_domain``; this check does not transform.
        """
        for name in (
            "channel_names",
            "sample_rate_hz",
            "num_time_samples",
            "start_time_gps",
        ):
            if getattr(self, name) != getattr(reference_data, name):
                raise ValueError(f"covariance {name} does not match the data")

    def _channel_index(self, channel_name: str | None) -> int:
        if channel_name is None:
            return 0
        if channel_name not in self.channel_names:
            raise ValueError(f"unknown covariance channel {channel_name!r}")
        return self.channel_names.index(channel_name)

    def noise_psd(self, channel_name: str | None = None) -> np.ndarray:
        """One-sided marginal PSD; inactive bins are infinite (zero weight).

        Defaults to the first channel. Cross-channel correlations remain in
        ``covariance_matrix``; marginal PSDs alone do not define a joint likelihood.
        Requires frequency covariance and applies ``S = 2 C / Tobs``.
        """
        if self.data_domain != "frequency":
            raise ValueError("noise_psd requires frequency covariance")
        index = self._channel_index(channel_name)
        psd = self.covariance_matrix[:, index, index].real * (
            2 * self.sample_rate_hz / self.num_time_samples
        )
        assert self.active_mask is not None  # normalized by construction
        psd[~self.active_mask] = np.inf
        return psd

    def noise_variance(self, channel_name: str | None = None) -> float:
        """Marginal variance for time-domain weighting; first channel by default.

        Spectral covariance is integrated over active bins with half weight at
        DC and even-length Nyquist. With the default inactive DC this preserves
        the zero-mean convention: white noise gives
        ``sigma**2 * (1 - 1/num_time_samples)``. This scalar does not account for
        temporal or cross-channel correlations in a full likelihood.

        Time covariance must have constant diagonal across active samples;
        nonstationary variance cannot be reduced to one weight silently.
        """
        if self.data_domain == "wdm":
            raise ValueError(
                "noise_variance requires native time or frequency covariance; "
                "use covariance operations for WDM noise"
            )
        index = self._channel_index(channel_name)
        assert self.active_mask is not None  # normalized by construction
        if self.data_domain == "time":
            active_sample_variances = self.covariance_matrix[
                self.active_mask, index, index
            ].real
            if active_sample_variances.size == 0:
                raise ValueError("noise_variance requires active time samples")
            if not np.allclose(
                active_sample_variances, active_sample_variances[0], rtol=1e-12, atol=0
            ):
                raise ValueError("noise_variance requires stationary time variance")
            return float(active_sample_variances[0])
        psd = self.noise_psd(channel_name)
        frequency_bin_weights = np.ones(psd.size)
        frequency_bin_weights[0] = 0.5
        if self.num_time_samples % 2 == 0:
            frequency_bin_weights[-1] = 0.5
        return float(
            np.sum(psd[self.active_mask] * frequency_bin_weights[self.active_mask])
            * (self.sample_rate_hz / self.num_time_samples)
        )

    def copy(self) -> "DataCovariance":
        """Copy and revalidate a covariance that its caller may have modified."""
        return DataCovariance(
            self.covariance_matrix,
            channel_names=self.channel_names,
            sample_rate_hz=self.sample_rate_hz,
            num_time_samples=self.num_time_samples,
            data_domain=self.data_domain,
            start_time_gps=self.start_time_gps,
            active_mask=self.active_mask,
            wdm_grid=self.wdm_grid,
        )

    def _copy_validated(self) -> "DataCovariance":
        """Copy an owned, already validated snapshot without matrix factorization.

        Only use this for orchestrator-owned objects that have never been handed
        to callers. Public input and block-returned covariance must pass through
        :meth:`copy` first: read-only NumPy arrays alone do not prove validity.
        """
        assert self.active_mask is not None
        matrix = self.covariance_matrix.copy()
        active = self.active_mask.copy()
        matrix.setflags(write=False)
        active.setflags(write=False)
        snapshot = object.__new__(DataCovariance)
        for name, value in (
            ("covariance_matrix", matrix),
            ("active_mask", active),
            ("channel_names", self.channel_names),
            ("sample_rate_hz", self.sample_rate_hz),
            ("num_time_samples", self.num_time_samples),
            ("data_domain", self.data_domain),
            ("start_time_gps", self.start_time_gps),
            ("wdm_grid", self.wdm_grid),
        ):
            object.__setattr__(snapshot, name, value)
        return snapshot

    def _stack_values(self, values: dict[str, np.ndarray]) -> np.ndarray:
        """Validate a native-domain vector and put channels on the final axis."""
        _validate_covariance_values(
            values,
            channel_names=self.channel_names,
            num_time_samples=self.num_time_samples,
            data_domain=self.data_domain,
            wdm_grid=self.wdm_grid,
        )
        return np.stack([values[name] for name in self.channel_names], axis=-1)

    def _unstack_values(self, values: np.ndarray) -> dict[str, np.ndarray]:
        return {
            name: values[..., index].copy()
            for index, name in enumerate(self.channel_names)
        }

    def _degree_weights(self) -> np.ndarray:
        assert self.active_mask is not None
        weights = np.ones(self.active_mask.shape, dtype=int)
        if self.data_domain == "frequency":
            weights[1:] = 2
            if self.num_time_samples % 2 == 0:
                weights[-1] = 1
        return weights

    @property
    def degrees_of_freedom(self) -> int:
        """Number of retained real coordinates, including every channel."""
        assert self.active_mask is not None
        return int(np.sum(self._degree_weights()[self.active_mask])) * len(
            self.channel_names
        )

    def apply(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply covariance on the retained subspace; excluded outputs are zero."""
        stacked = self._stack_values(values)
        assert self.active_mask is not None
        matrix = (
            self.covariance_matrix
            if self.data_domain == "frequency"
            else self.covariance_matrix.real
        )
        result = np.zeros_like(stacked, dtype=np.result_type(stacked, matrix))
        result[self.active_mask] = np.einsum(
            "...ij,...j->...i",
            matrix[self.active_mask],
            stacked[self.active_mask],
        )
        return self._unstack_values(result)

    def solve(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply precision after projection, ignoring excluded coordinates.

        This is the Moore-Penrose inverse on the retained real-coordinate
        subspace, not a constraint that excluded data must be zero.
        """
        stacked = self._stack_values(values)
        assert self.active_mask is not None
        matrix = (
            self.covariance_matrix
            if self.data_domain == "frequency"
            else self.covariance_matrix.real
        )
        result = np.zeros_like(stacked, dtype=np.result_type(stacked, matrix))
        result[self.active_mask] = np.linalg.solve(
            matrix[self.active_mask],
            stacked[self.active_mask, ..., None],
        )[..., 0]
        return self._unstack_values(result)

    def project(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Retain included native coordinates and zero excluded ones."""
        stacked = self._stack_values(values)
        assert self.active_mask is not None
        stacked[~self.active_mask] = 0
        return self._unstack_values(stacked)

    def quadratic_form(self, values: dict[str, np.ndarray]) -> float:
        """Return the exact real Gaussian quadratic, including real FFT endpoints."""
        stacked = self._stack_values(values)
        solved = self._stack_values(self.solve(values))
        return float(
            np.sum(self._degree_weights()[..., None] * (stacked.conj() * solved).real)
        )

    def log_determinant(self) -> float:
        """Log-pseudodeterminant in normalized real coordinates, excluding masks.

        Interior Fourier bins use sqrt(2)-weighted real and imaginary parts.
        The empty retained subspace has determinant one, hence returns zero.
        """
        assert self.active_mask is not None
        # Active matrices are positive definite. Their Cholesky diagonals are
        # positive real values even when cross-channel covariance is complex.
        factors = np.linalg.cholesky(self.covariance_matrix[self.active_mask])
        diagonals = factors.diagonal(axis1=-2, axis2=-1).real
        logdet = 2 * np.log(diagonals).sum(axis=-1)
        return float(np.sum(self._degree_weights()[self.active_mask] * logdet))
