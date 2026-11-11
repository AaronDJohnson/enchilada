"""Noise covariance as one array per channel pair, read through channel attributes."""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from inspect import signature
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from enchilada.data import L1Data

FORMS = ("diagonal", "full")
"""How pair arrays are read: one value per point, or a matrix between points."""

AET_FROM_XYZ = np.array(
    [
        [-1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)],
        [1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0)],
        [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
    ]
)
"""Rows A, E, T from columns X, Y, Z: the orthonormal LDC combination."""

BASES = {"XYZ": ("X", "Y", "Z"), "AET": ("A", "E", "T")}


def split_pair_name(
    name: str, channel_names: tuple[str, ...]
) -> tuple[str, str] | None:
    """Resolve ``"X"`` to ``("X", "X")`` and ``"XY"`` to ``("X", "Y")``.

    Returns None when ``name`` is neither a channel nor a concatenation of two
    channels. Raises ValueError when more than one split works.
    """
    if name in channel_names:
        return (name, name)
    candidates = [
        (name[:cut], name[cut:])
        for cut in range(1, len(name))
        if name[:cut] in channel_names and name[cut:] in channel_names
    ]
    if len(candidates) > 1:
        raise ValueError(
            f"{name!r} is ambiguous between channel pairs {candidates}; "
            "use a (first, second) tuple or pair(first, second)"
        )
    return candidates[0] if candidates else None


def _checked_variance(value: object, label: str) -> float:
    """Return ``value`` as a float, requiring a finite positive real number."""
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "fiu":  # kind "b" is excluded
        raise TypeError(f"{label} must be a real number, got {value!r}")
    level = float(array)
    if not np.isfinite(level) or level <= 0:
        raise ValueError(f"{label} must be finite and positive, got {value!r}")
    return level


@dataclass(frozen=True, eq=False)
class Covariance:
    """Noise covariance stored as one array per channel pair.

    ``pairs`` maps ``(first, second)`` in channel order to an array. Every
    channel's own ``(ch, ch)`` entry is required; an absent off-diagonal pair
    means zero cross-covariance. All arrays share one shape and one ``form``:

    * ``form="diagonal"``: the data's channel shape ``S``; one value per point.
    * ``form="full"``: ``S + S``; covariance between every pair of points.

    Read it through channel attributes: ``cov.X`` is ``pair("X", "X")``,
    ``cov.XY`` is ``pair("X", "Y")`` and ``cov.YX`` its conjugate transpose.
    With channels exactly X, Y, Z the names A, E, T and their pairs are
    derived on demand, and the reverse; see :meth:`in_basis`. A derivation that
    is not numerically positive definite raises ``ValueError``, so
    ``hasattr(cov, "X")`` can raise ``ValueError`` for such an object; this is
    deliberate so the numerical failure is not hidden.

    The object carries no grid. Everything that depends on the domain takes
    the ``L1Data`` the covariance accompanies: :meth:`check_compatible`,
    :meth:`like`, :meth:`from_psd`, :meth:`from_variance`, :meth:`noise_psd`,
    :meth:`noise_variance` and ``enchilada.transform``.

    Construction checks what needs no grid: required diagonals, known
    channels, one orientation per pair, finite floating arrays of one shape,
    real nonnegative diagonals in the diagonal form, Hermitian (possibly
    complex) diagonal blocks with real nonnegative point-diagonal in the full
    form, and positive definiteness over the channels with nonzero variance at
    each point. Stored arrays are independent read-only copies. Equality is
    identity; compare arrays explicitly.
    """

    pairs: dict[tuple[str, str], np.ndarray]
    channel_names: tuple[str, ...] = field(kw_only=True)
    form: str = field(default="diagonal", kw_only=True)
    _derived: "Covariance | None" = field(
        default=None, init=False, repr=False, compare=False
    )

    # ---- construction -----------------------------------------------------

    def __post_init__(self) -> None:
        channels = self._validated_channel_names()
        object.__setattr__(self, "channel_names", channels)
        if self.form not in FORMS:
            raise ValueError(f"form must be one of {FORMS}, got {self.form!r}")
        if not isinstance(self.pairs, Mapping):
            raise TypeError("pairs must be a dict of (first, second) -> numpy array")
        stored: dict[tuple[str, str], np.ndarray] = {}
        for key, value in self.pairs.items():
            pair_key, reversed_orientation = self._canonical_key(key)
            if pair_key in stored:
                raise ValueError(f"pair {pair_key!r} is given in both orientations")
            array = self._validated_array(pair_key, value)
            stored[pair_key] = (
                self._conjugate_transpose(array) if reversed_orientation else array
            )
        missing = [ch for ch in channels if (ch, ch) not in stored]
        if missing:
            raise ValueError(
                f"pairs is missing the diagonal entry for channels {missing}"
            )
        shapes = sorted({arr.shape for arr in stored.values()})
        if len(shapes) != 1:
            raise ValueError(f"all pair arrays must share one shape, got {shapes}")
        (shape,) = shapes
        if self.form == "full":
            half = len(shape) // 2
            if len(shape) % 2 or shape[:half] != shape[half:]:
                raise ValueError(
                    "form='full' arrays need shape S + S for one grid shape S, "
                    f"got {shape}"
                )
        if self.form == "diagonal":
            for ch in channels:
                if np.iscomplexobj(stored[(ch, ch)]):
                    raise ValueError(f"diagonal pair {(ch, ch)!r} must be real")
            for ch in channels:
                if np.any(stored[(ch, ch)] < 0):
                    raise ValueError(f"diagonal pair {(ch, ch)!r} must be nonnegative")
            self._check_positive_diagonal_form(stored)
        else:
            stored = self._symmetrised_full_form(stored)
            self._check_block_positive(self._assembled_full_matrix(stored)[None])
        for arr in stored.values():
            arr.setflags(write=False)
        object.__setattr__(self, "pairs", stored)

    def _validated_channel_names(self) -> tuple[str, ...]:
        names = self.channel_names
        if not isinstance(names, (tuple, list)) or any(
            not isinstance(ch, str) for ch in names
        ):
            raise TypeError("channel_names must be a tuple of channel-name strings")
        if not names:
            raise ValueError("channel_names must be a non-empty tuple of channel names")
        channels = tuple(names)
        if any(not ch for ch in channels) or len(set(channels)) != len(channels):
            raise ValueError("channel_names must contain unique non-empty strings")
        reserved = set(dir(type(self))) | _FIELD_NAMES
        clashing = [ch for ch in channels if ch in reserved]
        if clashing:
            raise ValueError(
                f"channel names {clashing} collide with Covariance attributes"
            )
        return channels

    def _canonical_key(self, key: object) -> tuple[tuple[str, str], bool]:
        """Return the upper-triangle key and whether ``key`` was reversed."""
        if not (
            isinstance(key, tuple)
            and len(key) == 2
            and all(isinstance(name, str) for name in key)
        ):
            raise TypeError(
                "pair keys must be (first, second) tuples of channel names, "
                f"got {key!r}"
            )
        for name in key:
            if name not in self.channel_names:
                raise ValueError(
                    f"unknown channel {name!r} in pair {key!r}; "
                    f"channels are {self.channel_names}"
                )
        first, second = key
        if self.channel_names.index(first) <= self.channel_names.index(second):
            return (first, second), False
        return (second, first), True

    @staticmethod
    def _validated_array(pair_key: tuple[str, str], value: object) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"pair {pair_key!r} must be a numpy array, got {type(value).__name__}"
            )
        if value.dtype.kind not in "fc":
            raise TypeError(
                f"pair {pair_key!r} must have a floating or complex dtype, "
                f"got {value.dtype}"
            )
        # Convert first: a masked array's isfinite ignores masked entries, and
        # the conversion below drops the mask but keeps the values beneath it.
        array = np.array(value, copy=True)
        if not np.isfinite(array).all():
            raise ValueError(f"pair {pair_key!r} must be finite")
        return array

    def _conjugate_transpose(self, array: np.ndarray) -> np.ndarray:
        if self.form == "diagonal":
            return np.conj(array)
        half = array.ndim // 2
        return np.conj(np.transpose(array, (*range(half, array.ndim), *range(half))))

    def _lookup(
        self, stored: Mapping[tuple[str, str], np.ndarray], first: str, second: str
    ) -> np.ndarray | None:
        if (first, second) in stored:
            return stored[(first, second)]
        if (second, first) in stored:
            return self._conjugate_transpose(stored[(second, first)])
        return None

    def _check_positive_diagonal_form(
        self, stored: dict[tuple[str, str], np.ndarray]
    ) -> None:
        """Per point, the channel matrix must be positive definite where active."""
        channels = self.channel_names
        if all(first == second for first, second in stored):
            return  # independent channels: nonnegative diagonals suffice
        count = len(channels)
        points = stored[(channels[0], channels[0])].size
        dtype = complex if any(np.iscomplexobj(a) for a in stored.values()) else float
        matrices = np.zeros((points, count, count), dtype=dtype)
        for i, first in enumerate(channels):
            for j, second in enumerate(channels):
                values = self._lookup(stored, first, second)
                if values is not None:
                    matrices[:, i, j] = values.reshape(-1)
        self._check_block_positive(matrices)

    @staticmethod
    def _check_block_positive(matrices: np.ndarray) -> None:
        """Each matrix must be positive definite over its nonzero-diagonal entries.

        An entry with zero variance must have a zero row and column: it is a
        structurally inactive point, not a data constraint.
        """
        diagonals = matrices.diagonal(axis1=-2, axis2=-1).real
        inactive = diagonals == 0
        if np.any(inactive):
            rows = np.where(inactive[:, :, None], matrices, 0)
            cols = np.where(inactive[:, None, :], matrices, 0)
            if np.any(rows != 0) or np.any(cols != 0):
                raise ValueError(
                    "a channel with zero variance at a point must have zero "
                    "cross-covariance there"
                )
        if not inactive.any():
            Covariance._require_positive_definite(matrices)
            return
        width = inactive.shape[1]
        if width < 64:
            codes = inactive @ (1 << np.arange(width, dtype=np.int64))
        else:  # too wide for one integer code (e.g. full form): group whole rows
            codes = np.unique(inactive, axis=0, return_inverse=True)[1].reshape(-1)
        for code in np.unique(codes):
            keep = np.flatnonzero(~inactive[np.argmax(codes == code)])
            if keep.size == 0:
                continue
            Covariance._require_positive_definite(
                matrices[codes == code][:, keep[:, None], keep]
            )

    @staticmethod
    def _require_positive_definite(matrices: np.ndarray) -> None:
        try:
            np.linalg.cholesky(matrices)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "covariance must be positive definite at every active point"
            ) from exc

    def _assembled_full_matrix(
        self, stored: Mapping[tuple[str, str], np.ndarray]
    ) -> np.ndarray:
        """Stack full-form pair matrices into one (channels*points)^2 matrix."""
        channels = self.channel_names
        sample = stored[(channels[0], channels[0])]
        points = int(np.prod(sample.shape[: sample.ndim // 2]))
        dtype = complex if any(np.iscomplexobj(a) for a in stored.values()) else float
        block = np.zeros((len(channels) * points, len(channels) * points), dtype=dtype)
        for i, first in enumerate(channels):
            for j, second in enumerate(channels):
                values = self._lookup(stored, first, second)
                if values is not None:
                    block[
                        i * points : (i + 1) * points, j * points : (j + 1) * points
                    ] = values.reshape(points, points)
        return block

    def _symmetrised_full_form(
        self, stored: dict[tuple[str, str], np.ndarray]
    ) -> dict[tuple[str, str], np.ndarray]:
        """Require Hermitian diagonal blocks and store them exactly Hermitian."""
        result = dict(stored)
        for ch in self.channel_names:
            values = stored[(ch, ch)]
            points = int(np.prod(values.shape[: values.ndim // 2]))
            matrix = values.reshape(points, points)
            scale = np.sqrt(np.abs(np.diagonal(matrix).real))
            with np.errstate(over="ignore", under="ignore"):
                tolerance = (1e-12 * np.maximum.outer(scale, scale)) * np.minimum.outer(
                    scale, scale
                )
                asymmetry = np.abs(matrix - matrix.conj().T)
            if np.any(asymmetry > tolerance):
                raise ValueError(f"diagonal pair {(ch, ch)!r} must be Hermitian")
            with np.errstate(under="ignore"):
                symmetric = 0.5 * (matrix + matrix.conj().T)
            result[(ch, ch)] = symmetric.reshape(values.shape)
        return result

    # ---- access -----------------------------------------------------------

    def pair(self, first: str, second: str) -> np.ndarray:
        """Covariance between two channels: stored, conjugate-transposed, or zero.

        A stored pair returns its read-only array; never write to it.
        """
        for name in (first, second):
            if name not in self.channel_names:
                raise ValueError(
                    f"unknown channel {name!r}; channels are {self.channel_names}"
                )
        found = self._lookup(self.pairs, first, second)
        if found is None:
            return np.zeros_like(self.pairs[(first, first)])
        return found

    def __getattr__(self, name: str) -> np.ndarray:
        # Only reached when normal lookup fails, so fields and methods win.
        if name.startswith("_") or name in _FIELD_NAMES:
            raise AttributeError(name)
        channels = self.channel_names
        try:
            resolved = split_pair_name(name, channels)
        except ValueError as exc:
            raise AttributeError(str(exc)) from None
        if resolved is not None:
            return self.pair(*resolved)
        derived_basis = self._other_basis()
        if derived_basis is not None:
            if split_pair_name(name, BASES[derived_basis]) is not None:
                return getattr(self.in_basis(derived_basis), name)
        valid = [*channels, *(a + b for a in channels for b in channels if a != b)]
        if derived_basis is not None:
            derived_channels = BASES[derived_basis]
            valid += [
                *derived_channels,
                *(a + b for a in derived_channels for b in derived_channels if a != b),
            ]
        raise AttributeError(
            f"Covariance has no attribute {name!r}; channel attributes are {valid}"
        )

    # ---- the accompanying data ---------------------------------------------

    @classmethod
    def like(
        cls, reference_data: "L1Data", pairs: Mapping[Any, np.ndarray]
    ) -> "Covariance":
        """Build a covariance on ``reference_data``'s grid from channel-keyed arrays.

        Keys are ``"A"``, ``"AE"`` (resolved by the attribute rule) or
        ``(first, second)`` tuples; two keys naming the same pair raise
        ``ValueError``. ``form`` follows the array shape against the data's
        channel shape. The result passes :meth:`check_compatible`.
        """
        channels = tuple(reference_data.channel_names)
        resolved: dict[Any, np.ndarray] = {}
        origins: dict[Any, Any] = {}
        for key, value in pairs.items():
            target = key
            if isinstance(key, str):
                target = split_pair_name(key, channels)
                if target is None:
                    raise ValueError(
                        f"key {key!r} is neither a channel nor a pair of channels "
                        f"from {channels}"
                    )
            if target in origins:
                raise ValueError(
                    f"keys {origins[target]!r} and {key!r} both name pair {target!r}"
                )
            origins[target] = key
            resolved[target] = value
        if not resolved:
            raise ValueError(
                "pairs must contain at least every channel's diagonal entry"
            )
        grid_shape = reference_data.channel_data[channels[0]].shape
        sample_shape = np.shape(next(iter(resolved.values())))
        if sample_shape == grid_shape:
            form = "diagonal"
        elif sample_shape == grid_shape + grid_shape:
            form = "full"
        else:
            raise ValueError(
                f"pair arrays must have the data's channel shape {grid_shape} "
                f"(diagonal form) or {grid_shape + grid_shape} (full form), "
                f"got {sample_shape}"
            )
        covariance = cls(resolved, channel_names=channels, form=form)
        covariance.check_compatible(reference_data)
        return covariance

    @classmethod
    def from_psd(cls, reference_data: "L1Data", noise_psd_model: Any) -> "Covariance":
        """Independent channels from ``noise_psd_model.psd(freqs[, channel])``.

        ``reference_data`` must be frequency-domain data. Every rfft bin,
        including ``f = 0`` and Nyquist, stores ``C = (Tobs / 2) * S(f_k)``; the
        model must be finite and positive there (floor it at low frequency).
        """
        if reference_data.data_domain != "frequency":
            raise ValueError(
                "from_psd needs frequency-domain reference_data; "
                "pass data.to_frequency()"
            )
        method = getattr(noise_psd_model, "psd", None)
        if not callable(method):
            raise TypeError("noise model must expose psd(freqs[, channel])")
        frequencies = np.fft.rfftfreq(
            reference_data.num_time_samples, d=1.0 / reference_data.sample_rate_hz
        )
        half_observation = reference_data.num_time_samples / (
            2.0 * reference_data.sample_rate_hz
        )
        pairs: dict[tuple[str, str], np.ndarray] = {}
        for channel in reference_data.channel_names:
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
                raise ValueError(
                    "noise model PSD shape must match the rfft grid, including DC"
                )
            if not np.isfinite(psd_values).all() or np.any(psd_values <= 0):
                raise ValueError(
                    "noise model PSD must be finite and strictly positive at every "
                    "bin, including f = 0; floor the model at low frequency"
                )
            pairs[(channel, channel)] = np.array(
                np.broadcast_to(half_observation * psd_values, frequencies.shape),
                dtype=float,
            )
        return cls(pairs, channel_names=tuple(reference_data.channel_names))

    @classmethod
    def from_variance(
        cls, reference_data: "L1Data", time_sample_variance: float | Mapping[str, float]
    ) -> "Covariance":
        """Independent white time noise with one constant variance per channel.

        ``reference_data`` must be time-domain data. ``time_sample_variance``
        is a scalar shared by all channels or a mapping with exactly the run's
        channel names; each value must be a finite positive real number
        (``TypeError`` for a non-real value, ``ValueError`` otherwise).
        """
        if reference_data.data_domain != "time":
            raise ValueError("from_variance needs time-domain reference_data")
        channels = tuple(reference_data.channel_names)
        if isinstance(time_sample_variance, Mapping):
            if set(time_sample_variance) != set(channels):
                raise ValueError(
                    "time_sample_variance keys must match channels exactly"
                )
            levels = [
                _checked_variance(
                    time_sample_variance[ch], f"time_sample_variance[{ch!r}]"
                )
                for ch in channels
            ]
        else:
            level = _checked_variance(time_sample_variance, "time_sample_variance")
            levels = [level] * len(channels)
        shape = reference_data.channel_data[channels[0]].shape
        covariance = cls(
            {
                (ch, ch): np.full(shape, level)
                for ch, level in zip(channels, levels, strict=True)
            },
            channel_names=channels,
        )
        covariance.check_compatible(reference_data)
        return covariance

    def check_compatible(self, reference_data: "L1Data") -> None:
        """Require this covariance to describe ``reference_data``'s grid and domain.

        Checks channel order, the shape rule for ``form``, real arrays outside
        the frequency domain, real entries where both coefficients are real
        (the DC and even-length Nyquist entries of a diagonal-form pair; the
        DC/Nyquist corner entries of a full-form block), positive variances in
        time and frequency, and WDM variances that are zero exactly at
        structurally inactive pixels.
        """
        if tuple(reference_data.channel_names) != self.channel_names:
            raise ValueError(
                f"covariance channel_names {self.channel_names} do not match the "
                f"data's {tuple(reference_data.channel_names)}"
            )
        grid_shape = reference_data.channel_data[self.channel_names[0]].shape
        expected = grid_shape if self.form == "diagonal" else grid_shape + grid_shape
        shape = next(iter(self.pairs.values())).shape
        if shape != expected:
            raise ValueError(
                f"covariance pair arrays have shape {shape}; form={self.form!r} on "
                f"this {reference_data.data_domain} grid requires {expected}"
            )
        domain = reference_data.data_domain
        if domain != "frequency":
            for key, arr in self.pairs.items():
                if np.iscomplexobj(arr):
                    raise ValueError(
                        f"pair {key!r} must be real for data_domain={domain!r}"
                    )
        else:
            even = reference_data.num_time_samples % 2 == 0
            for key, arr in self.pairs.items():
                if not np.iscomplexobj(arr):
                    continue
                # Only entries pairing two real coefficients must be real: the
                # DC (and even-N Nyquist) entry of a diagonal-form pair, and the
                # DC/Nyquist corner entries of a full-form block. A full-form
                # DC row pairs a real coefficient with complex ones elsewhere.
                real_bins = [0] + ([-1] if even else [])
                if self.form == "diagonal":
                    endpoints = [arr[k] for k in real_bins]
                else:
                    endpoints = [arr[k, m] for k in real_bins for m in real_bins]
                if any(np.any(np.imag(values) != 0) for values in endpoints):
                    raise ValueError(
                        f"pair {key!r} must be real where both coefficients are "
                        "real (DC and even-length Nyquist)"
                    )
        for ch in self.channel_names:
            variances = self._point_variances(ch)
            if domain == "wdm":
                assert reference_data.wdm_grid is not None
                active = reference_data.wdm_grid.active_mask
                if np.any(variances[~active] != 0):
                    raise ValueError(
                        f"pair {(ch, ch)!r} must be zero at structurally inactive "
                        "WDM pixels"
                    )
                if np.any(variances[active] <= 0):
                    raise ValueError(
                        f"pair {(ch, ch)!r} must be positive at active WDM pixels"
                    )
            elif np.any(variances <= 0):
                raise ValueError(
                    f"pair {(ch, ch)!r} must be positive at every point for "
                    f"data_domain={domain!r}"
                )

    def _point_variances(self, channel: str) -> np.ndarray:
        values = self.pairs[(channel, channel)]
        if self.form == "diagonal":
            return values
        half = values.ndim // 2
        points = int(np.prod(values.shape[:half]))
        # Hermitian blocks have a real point-diagonal; drop the zero imaginary part.
        diagonal = values.reshape(points, points).diagonal().real
        return diagonal.reshape(values.shape[:half])

    def noise_psd(self, channel: str, reference_data: "L1Data") -> np.ndarray:
        """One-sided marginal PSD ``S = 2 C / Tobs`` at every bin.

        ``reference_data`` must be the frequency-domain data this covariance
        accompanies. Cross-channel terms stay in the pairs.
        """
        if reference_data.data_domain != "frequency":
            raise ValueError("noise_psd needs frequency-domain reference_data")
        self.check_compatible(reference_data)
        if channel not in self.channel_names:
            raise ValueError(
                f"unknown channel {channel!r}; channels are {self.channel_names}"
            )
        observation_length = (
            reference_data.num_time_samples / reference_data.sample_rate_hz
        )
        return np.array(self._point_variances(channel), dtype=float) * (
            2.0 / observation_length
        )

    def noise_variance(self, channel: str, reference_data: "L1Data") -> float:
        """Marginal time-sample variance of one channel.

        Time data: the constant variance (``ValueError`` if not constant).
        Frequency data: the trapezoid integral of the PSD over all bins, with
        half weight at DC and at Nyquist for even lengths, so white noise reads
        back as exactly ``sigma**2``. WDM data is not supported.
        """
        if reference_data.data_domain == "wdm":
            raise ValueError("noise_variance needs time or frequency reference_data")
        self.check_compatible(reference_data)
        if channel not in self.channel_names:
            raise ValueError(
                f"unknown channel {channel!r}; channels are {self.channel_names}"
            )
        if reference_data.data_domain == "time":
            variances = self._point_variances(channel)
            if not np.allclose(variances, variances.flat[0], rtol=1e-12, atol=0):
                raise ValueError("noise_variance requires stationary time variance")
            return float(variances.flat[0])
        psd = self.noise_psd(channel, reference_data)
        weights = np.ones(psd.size)
        weights[0] = 0.5
        if reference_data.num_time_samples % 2 == 0:
            weights[-1] = 0.5
        frequency_resolution = (
            reference_data.sample_rate_hz / reference_data.num_time_samples
        )
        return float(np.sum(psd * weights) * frequency_resolution)

    # ---- copies -----------------------------------------------------------

    def copy(self) -> "Covariance":
        """Copy and revalidate, for arrays a caller may have edited in place."""
        return Covariance(
            {key: arr.copy() for key, arr in self.pairs.items()},
            channel_names=self.channel_names,
            form=self.form,
        )

    def _copy_validated(self) -> "Covariance":
        """Copy an owned, already validated object without re-running checks.

        Only for orchestrator-owned objects never handed to callers; public
        input and block-returned covariance go through :meth:`copy`.
        """
        snapshot = object.__new__(Covariance)
        copied = {key: arr.copy() for key, arr in self.pairs.items()}
        for arr in copied.values():
            arr.setflags(write=False)
        object.__setattr__(snapshot, "pairs", copied)
        object.__setattr__(snapshot, "channel_names", self.channel_names)
        object.__setattr__(snapshot, "form", self.form)
        object.__setattr__(snapshot, "_derived", None)
        return snapshot

    def __getstate__(self) -> dict[str, Any]:
        # deepcopy and pickle carry the fields only; the cache is rebuilt.
        return {
            "pairs": self.pairs,
            "channel_names": self.channel_names,
            "form": self.form,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Restored arrays are fresh and writable; keep the read-only promise.
        pairs = dict(state["pairs"])
        for arr in pairs.values():
            arr.setflags(write=False)
        object.__setattr__(self, "pairs", pairs)
        object.__setattr__(self, "channel_names", state["channel_names"])
        object.__setattr__(self, "form", state["form"])
        object.__setattr__(self, "_derived", None)

    # ---- derived basis ----------------------------------------------------

    def _basis_name(self) -> str | None:
        current = frozenset(self.channel_names)
        for basis, names in BASES.items():
            if current == frozenset(names):
                return basis
        return None

    def _other_basis(self) -> str | None:
        basis = self._basis_name()
        if basis is None:
            return None
        return "AET" if basis == "XYZ" else "XYZ"

    def in_basis(self, basis: str) -> "Covariance":
        """Return this covariance in the ``"XYZ"`` or ``"AET"`` basis.

        Available only when the channels are exactly X, Y, Z or exactly A, E, T.
        The derived object has all six pairs stored, the same ``form``, channel
        order ``("A", "E", "T")`` or ``("X", "Y", "Z")``, and is computed once.
        Asking for the current basis returns ``self``. Derived full-form diagonal
        blocks are symmetrised exactly, because the summation order differs
        between an entry and its mirror.

        Derivation needs the rotated covariance to be numerically positive
        definite; when one channel's variance is below float64 roundoff of the
        others (T << A at the lowest LISA frequencies) ``in_basis`` and the
        derived attributes raise ``ValueError`` with that explanation.
        ``hasattr(cov, "X")`` can therefore raise ``ValueError`` for such an
        object; this is deliberate so the numerical failure is not hidden.
        """
        if basis not in BASES:
            raise ValueError(f"basis must be one of {tuple(BASES)}, got {basis!r}")
        current = self._basis_name()
        if current is None:
            raise ValueError(
                "deriving a basis needs channels exactly X, Y, Z or A, E, T; "
                f"got {self.channel_names}"
            )
        if basis == current:
            return self
        if self._derived is None:
            try:
                derived = self._derive(current, basis)
            except ValueError as exc:
                raise ValueError(
                    f"cannot derive the {basis} basis from {current}: the rotated "
                    "covariance is not numerically positive definite (one channel's "
                    "variance is below the roundoff of the others, e.g. T << A at "
                    f"low frequency): {exc}"
                ) from exc
            object.__setattr__(derived, "_derived", self)
            object.__setattr__(self, "_derived", derived)
        assert self._derived is not None
        return self._derived

    def _derive(self, current: str, basis: str) -> "Covariance":
        source, target = BASES[current], BASES[basis]
        matrix = AET_FROM_XYZ if basis == "AET" else AET_FROM_XYZ.T
        pairs: dict[tuple[str, str], np.ndarray] = {}
        for a, first in enumerate(target):
            for b, second in enumerate(target[a:], start=a):
                total: np.ndarray | float = 0.0
                for i, source_first in enumerate(source):
                    for j, source_second in enumerate(source):
                        weight = matrix[a, i] * matrix[b, j]
                        if weight != 0.0:
                            total = total + weight * self.pair(
                                source_first, source_second
                            )
                assert isinstance(total, np.ndarray)
                if first != second:
                    pairs[(first, second)] = total
                elif self.form == "diagonal":
                    pairs[(first, second)] = total.real
                else:
                    pairs[(first, second)] = 0.5 * (
                        total + self._conjugate_transpose(total)
                    )
        return Covariance(pairs, channel_names=target, form=self.form)


_FIELD_NAMES = frozenset(f.name for f in fields(Covariance))
