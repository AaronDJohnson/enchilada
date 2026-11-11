"""Noise covariance as one array per channel pair, read through channel attributes."""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from inspect import signature
from typing import TYPE_CHECKING, Any

import numpy as np

from enchilada.domains import WDMGrid

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
        ``(first, second)`` tuples. ``form`` follows the array shape against the
        data's channel shape. The result passes :meth:`check_compatible`.
        """
        channels = tuple(reference_data.channel_names)
        resolved: dict[tuple[str, str], np.ndarray] = {}
        for key, value in pairs.items():
            if isinstance(key, str):
                split = split_pair_name(key, channels)
                if split is None:
                    raise ValueError(
                        f"key {key!r} is neither a channel nor a pair of channels "
                        f"from {channels}"
                    )
                resolved[split] = value
            else:
                resolved[key] = value
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
        channel names.
        """
        if reference_data.data_domain != "time":
            raise ValueError("from_variance needs time-domain reference_data")
        channels = tuple(reference_data.channel_names)
        if isinstance(time_sample_variance, Mapping):
            if set(time_sample_variance) != set(channels):
                raise ValueError(
                    "time_sample_variance keys must match channels exactly"
                )
            levels = [float(time_sample_variance[ch]) for ch in channels]
        else:
            scalar = np.asarray(time_sample_variance)
            if scalar.ndim != 0 or scalar.dtype.kind not in "fiu":
                raise ValueError(
                    "time_sample_variance must be a real scalar for each channel"
                )
            levels = [float(scalar)] * len(channels)
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

    # ---- derived basis (Task 2) -------------------------------------------

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


def _validate_covariance_values(
    values: dict[str, np.ndarray],
    *,
    channel_names: tuple[str, ...],
    num_time_samples: int,
    data_domain: str,
    wdm_grid: WDMGrid | None,
) -> None:
    """Validate operands before a transform can discard invalid coordinates."""
    from enchilada.block_state import check_on_grid

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

    Grid points are independent; channels at each point may be correlated.
    Time grids contain ``num_time_samples`` real samples. Fourier grids contain
    ``num_time_samples // 2 + 1`` coefficients of ``dt * rfft(x)``. WDM grids have
    shape ``(Nf+1, Nt)``, with structural edge pixels excluded by
    ``wdm_grid.active_mask``.

    Frequency covariances describe proper complex interior coefficients (zero
    pseudocovariance), with real DC and even-length Nyquist coefficients. These
    are coefficient covariances, not spectral densities; :meth:`from_psd`
    converts the one-sided PSD convention.

    Every matrix is finite and Hermitian; active matrices are positive definite.
    Inactive matrices may be zero. Stored covariance and mask arrays are
    independent, read-only copies.
    Symmetry is checked relative to each pair of channel standard deviations.
    Accepted numerical roundoff is symmetrized before testing positive
    definiteness, so the stored matrix is exactly Hermitian.

    Array operations accept channel dictionaries of real time/WDM arrays or
    complex rFFT arrays. Their mathematical normalization uses real coordinates:
    ``sqrt(2) * Re(Z), sqrt(2) * Im(Z)`` for frequency interiors and ``Re(Z)`` for
    endpoints. Callers do not pack these coordinates themselves. Quadratic forms
    and determinants count each real degree of freedom once. Inactive points
    are projected out, not treated as zero-variance data constraints.

    Fields:
        covariance_matrix: Channel covariance at each sample, Fourier bin, or
            WDM pixel, shaped ``(*grid_shape, num_channels, num_channels)``.
        channel_names: Ordered TDI channel names for both matrix channel axes.
        sample_rate_hz: Sampling frequency of the underlying time series, in Hz.
        num_time_samples: Number of underlying time samples per channel in every
            representation.
        data_domain: "time", "frequency" (default), or "wdm"; determines the grid
            points and the real/complex coefficient convention.
        start_time_gps: GPS seconds at sample index zero; defaults to 0.0 for
            synthetic data without an absolute-time reference.
        active_mask: Boolean array per grid point; True includes, False excludes.
            None selects all physical points and becomes an explicit mask.
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
        and has one degree of freedom. DC covariance is stored as zero and
        excluded from the likelihood; input data need not have zero mean.
        Channels are independent in this adapter; construct a matrix directly
        for correlated channels.
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

        Representation may differ. This checks metadata only and does not transform
        arrays or verify that a likelihood supports the covariance representation.
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
