# Covariance Object Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `DataCovariance` and `TranslatedCovariance` with one grid-free `Covariance` value object read through channel attributes (`cov.X`, `cov.XY`, `cov.A`, `cov.AE`), with the A, E, T basis derived on demand from X, Y, Z.

**Architecture:** `Covariance` stores one array per channel pair plus the channel order and a `form` flag ("diagonal" or "full"); everything domain-dependent takes the `L1Data` the covariance accompanies. The new class is built beside the old one first (Tasks 1–4) so the suite stays green, then the package switches over in one commit (Task 5), then notebooks and docs follow. Covariance translation is left as a seam in `transform()` that raises `NotImplementedError` across domains until the user's translation layer lands.

**Tech Stack:** Python ≥3.12, NumPy only in the core; pytest, ruff, mypy. Run tools as `.venv/bin/python -m pytest` etc. (`uv run` fails in this sandbox).

**Spec:** `docs/superpowers/specs/2026-11-10-covariance-object-design.md` (read it first; section numbers below refer to it).

**User decisions (already made):**
- Covariance is a separate object, `Covariance`, with attribute access per channel and per pair, including both XYZ and AET names.
- One stored basis; the other basis is derived on demand; derived names exist only when the channel set is exactly {X,Y,Z} or {A,E,T}.
- A, E, T use the orthonormal LDC matrix (spec 3.5).
- Internals are one array per channel pair with the shape rule, not a stacked `(points, ch, ch)` block.
- The object carries no grid fields; only `pairs`, `channel_names`, `form`.
- The user's own translation layer will implement the covariance case of `transform()` later; this change ships the seam only.
- Sections 1–2 of the spec and the written spec were approved verbatim.

**Refinements fixed by this plan (not in the spec's words, consistent with its intent):**
- `copy()` revalidates (as today's `DataCovariance.copy`); `_copy_validated()` is the Wheel's fast path for objects it owns.
- Same-domain `transform()` of a `Covariance` returns an independent copy, as the `L1Data` branch does and as `test_same_domain_transform_is_an_independent_snapshot` expects. Content is unchanged.
- `Wheel.add` with a block domain and a noise-only initial state treats the covariance as being on the block's grid (a reference in that domain is built from the observations), so the seam applies uniformly.
- Six tests, not four, exercise covariance translation through the Wheel and get the strict `xfail`; `examples/domain_translation.py` loses its covariance part instead of being xfailed, so its test keeps passing on the data and state parts.
- `FixedLISANoise.psd` in the GB example floors its frequencies because `from_psd` now evaluates f = 0.
- The Task 1 and Task 2 code blocks below predate their review fixes (masked-array finiteness, positivity grouping, complex Hermitian full-form blocks); HEAD is authoritative for what was built.
- Full-form diagonal blocks may be complex Hermitian (frequency-domain `E[Z_k Z_l*]`); only the diagonal form requires real diagonal pairs. Found during Task 2; spec 4.1 and 1.4 amended to match.

---

## File structure

| File | Responsibility after this plan |
|---|---|
| `src/enchilada/covariance.py` | `Covariance` only: fields, validation, `pair`, attribute access, `in_basis`, `like`, `from_psd`, `from_variance`, `noise_psd`, `noise_variance`, `check_compatible`, `copy` |
| `src/enchilada/translated_covariance.py` | **deleted** |
| `src/enchilada/translation.py` | `transform` dispatch; the covariance branch is the seam (identity same-domain, `NotImplementedError` otherwise) |
| `src/enchilada/wheel.py` | `initial_covariance`, `.covariance`, passes `reference_data` when translating the covariance |
| `src/enchilada/block.py`, `block_state.py`, `data.py`, `testing.py`, `__init__.py` | renames `covariance_matrix`→`covariance`, `noise_covariance`→`covariance`, export `Covariance` |
| `tests/test_covariance.py` | rewritten for the new class (built as `tests/test_covariance_object.py` in Tasks 1–3, renamed in Task 5) |
| `tests/test_covariance_translation.py` | **deleted** |
| other tests, `examples/*.py`, `benchmarks/benchmark_wheel.py` | renamed and rewritten call sites (Task 5) |
| `examples/*.ipynb` | code cells migrated (Task 6) |
| `README.md`, `CHANGELOG.md` | rewritten covariance passages (Task 7) |

Commit after every task. Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Never `git add -A`: the working tree also holds the user's unrelated uncommitted edits to `README.md`, `pyproject.toml` and `docs/wheel.svg`. Stage only the files each task names. The three uncommitted files are one piece of work (the diagram, its sdist include, the image), so Task 7 first commits them on their own, then edits the README; the task's report tells the user that commit was made. Task 1 commits this plan and its task file alongside the first code.

Verification commands used throughout:

```bash
.venv/bin/python -m pytest -q                       # full suite (currently 576 passed, 40 skipped)
.venv/bin/python -m ruff check src tests examples benchmarks
.venv/bin/python -m ruff format --check src tests examples benchmarks
.venv/bin/python -m mypy
```

---

### Task 1: `Covariance` core — fields, validation, `pair()`, attribute access, `copy()`

**Goal:** Add the `Covariance` class to `src/enchilada/covariance.py` beside the existing `DataCovariance`, implementing spec sections 1.1, 1.3 (without the derived-basis fallback), 4.1 and `copy`.

**Files:**
- Modify: `src/enchilada/covariance.py` (insert the new code between the imports and `_validate_covariance_values`; leave `DataCovariance` untouched in this task)
- Test: `tests/test_covariance_object.py` (new)

**Acceptance Criteria:**
- [ ] `Covariance({("A","A"): a, ("E","E"): e}, channel_names=("A","E"))` constructs; `cov.A is cov.pairs[("A","A")]`; arrays are read-only copies.
- [ ] A reversed key `("E","A")` is stored under `("A","E")` as its conjugate transpose; giving both orientations raises `ValueError`.
- [ ] Missing diagonal, unknown channel, non-tuple key, non-array value, integer dtype, non-finite value, complex diagonal, negative diagonal, mixed shapes, `form="full"` with a non-`S+S` shape, and a non-Hermitian full diagonal block each raise with the message in the code below.
- [ ] `cov.YX` equals `np.conj(cov.XY)` for the diagonal form and `np.conj(cov.XY.T)` for the full form; an absent pair returns zeros shaped like the diagonal.
- [ ] Multi-letter channel names resolve through `pair()`; an ambiguous attribute name raises `AttributeError` naming the candidates; an unknown name raises `AttributeError` listing valid names.
- [ ] A point with zero variance in one channel but a nonzero cross term there raises; a per-point channel matrix that is not positive definite raises.
- [ ] A channel named like a class attribute (e.g. `"pairs"`, `"copy"`) is rejected at construction.
- [ ] Existing suite still passes (nothing else changed).

**Verify:** `.venv/bin/python -m pytest tests/test_covariance_object.py -q` → all pass; `.venv/bin/python -m pytest -q` → 576 + new tests pass, 40 skipped.

**Steps:**

- [ ] **Step 1: Write the failing tests**

Create `tests/test_covariance_object.py`:

```python
"""Covariance: one array per channel pair, read through channel attributes."""

import numpy as np
import pytest

from enchilada.covariance import Covariance

AE = ("A", "E")


def diagonal_pairs(n=8, a=1.0, e=4.0):
    return {("A", "A"): np.full(n, a), ("E", "E"): np.full(n, e)}


def full_pairs(n=4):
    a = np.diag(np.full(n, 2.0)) + 0.5 * np.eye(n, k=1) + 0.5 * np.eye(n, k=-1)
    e = np.diag(np.arange(1.0, n + 1))
    ae = 0.1 * np.eye(n)
    return {("A", "A"): a, ("E", "E"): e, ("A", "E"): ae}


class TestConstruction:
    def test_diagonal_entries_are_stored_as_read_only_copies(self):
        pairs = diagonal_pairs()
        cov = Covariance(pairs, channel_names=AE)
        assert cov.form == "diagonal"
        assert cov.channel_names == AE
        assert not cov.pairs[("A", "A")].flags.writeable
        assert not np.shares_memory(cov.pairs[("A", "A")], pairs[("A", "A")])
        np.testing.assert_array_equal(cov.A, 1.0)
        np.testing.assert_array_equal(cov.E, 4.0)

    def test_reversed_key_is_stored_in_canonical_orientation(self):
        pairs = diagonal_pairs()
        pairs[("E", "A")] = np.full(8, 0.5 + 0.25j)
        cov = Covariance(pairs, channel_names=AE)
        assert set(cov.pairs) == {("A", "A"), ("E", "E"), ("A", "E")}
        np.testing.assert_array_equal(cov.pairs[("A", "E")], 0.5 - 0.25j)
        np.testing.assert_array_equal(cov.EA, 0.5 + 0.25j)

    def test_both_orientations_are_rejected(self):
        pairs = diagonal_pairs()
        pairs[("A", "E")] = np.zeros(8)
        pairs[("E", "A")] = np.zeros(8)
        with pytest.raises(ValueError, match="both orientations"):
            Covariance(pairs, channel_names=AE)

    @pytest.mark.parametrize(
        "mutate, error, match",
        [
            (lambda p: p.pop(("E", "E")), ValueError, "missing the diagonal"),
            (lambda p: p.update({("T", "T"): np.ones(8)}), ValueError, "unknown channel"),
            (lambda p: p.update({"A": np.ones(8)}), TypeError, "tuples"),
            (lambda p: p.update({("A", "A"): [1.0] * 8}), TypeError, "numpy array"),
            (lambda p: p.update({("A", "A"): np.ones(8, dtype=int)}), TypeError, "dtype"),
            (lambda p: p.update({("A", "A"): np.full(8, np.nan)}), ValueError, "finite"),
            (lambda p: p.update({("A", "A"): np.ones(8, complex)}), ValueError, "must be real"),
            (lambda p: p.update({("A", "A"): -np.ones(8)}), ValueError, "nonnegative"),
            (lambda p: p.update({("A", "E"): np.zeros(7)}), ValueError, "one shape"),
        ],
    )
    def test_structural_rules(self, mutate, error, match):
        pairs = diagonal_pairs()
        mutate(pairs)
        with pytest.raises(error, match=match):
            Covariance(pairs, channel_names=AE)

    def test_unknown_form_and_bad_full_shape(self):
        with pytest.raises(ValueError, match="form must be"):
            Covariance(diagonal_pairs(), channel_names=AE, form="banded")
        with pytest.raises(ValueError, match="S \\+ S"):
            Covariance(diagonal_pairs(), channel_names=AE, form="full")

    def test_full_form_requires_hermitian_diagonal_blocks(self):
        pairs = full_pairs()
        pairs[("A", "A")][0, 1] = 5.0  # breaks symmetry
        with pytest.raises(ValueError, match="Hermitian"):
            Covariance(pairs, channel_names=AE, form="full")

    def test_full_form_symmetrises_roundoff_exactly(self):
        pairs = full_pairs()
        pairs[("A", "A")][0, 1] += 1e-15
        cov = Covariance(pairs, channel_names=AE, form="full")
        np.testing.assert_array_equal(cov.A, cov.A.T)

    def test_zero_variance_point_must_have_zero_cross_terms(self):
        pairs = diagonal_pairs()
        pairs[("A", "A")] = pairs[("A", "A")].copy()
        pairs[("A", "A")][3] = 0.0
        pairs[("A", "E")] = np.full(8, 0.1)
        with pytest.raises(ValueError, match="zero cross-covariance"):
            Covariance(pairs, channel_names=AE)
        pairs[("A", "E")] = pairs[("A", "E")].copy()
        pairs[("A", "E")][3] = 0.0
        cov = Covariance(pairs, channel_names=AE)
        assert cov.A[3] == 0.0

    def test_channel_matrices_must_be_positive_definite(self):
        pairs = diagonal_pairs(a=1.0, e=1.0)
        pairs[("A", "E")] = np.full(8, 1.5)  # |C_AE|^2 > C_AA C_EE
        with pytest.raises(ValueError, match="positive definite"):
            Covariance(pairs, channel_names=AE)

    def test_full_form_block_matrix_must_be_positive_definite(self):
        pairs = full_pairs()
        pairs[("A", "E")] = 3.0 * np.eye(4)
        with pytest.raises(ValueError, match="positive definite"):
            Covariance(pairs, channel_names=AE, form="full")

    @pytest.mark.parametrize("name", ["pairs", "copy", "form", "pair"])
    def test_channel_names_cannot_shadow_attributes(self, name):
        with pytest.raises(ValueError, match="collide"):
            Covariance({(name, name): np.ones(4)}, channel_names=(name,))

    def test_channel_names_must_be_unique_nonempty_strings(self):
        with pytest.raises(ValueError, match="unique non-empty"):
            Covariance({("A", "A"): np.ones(4)}, channel_names=("A", "A"))
        with pytest.raises(ValueError, match="non-empty tuple"):
            Covariance({}, channel_names=())


class TestAccess:
    def test_absent_cross_term_reads_as_zero(self):
        cov = Covariance(diagonal_pairs(), channel_names=AE)
        np.testing.assert_array_equal(cov.AE, 0.0)
        np.testing.assert_array_equal(cov.EA, 0.0)
        assert cov.AE.shape == (8,)
        assert cov.pair("A", "E") is not cov.pair("A", "E")  # fresh zeros

    def test_reversed_pair_is_conjugate_transpose(self):
        pairs = diagonal_pairs()
        pairs[("A", "E")] = np.full(8, 0.5 + 0.25j)
        cov = Covariance(pairs, channel_names=AE)
        np.testing.assert_array_equal(cov.EA, np.conj(cov.AE))
        full = Covariance(full_pairs(), channel_names=AE, form="full")
        np.testing.assert_array_equal(full.EA, full.AE.T)

    def test_full_form_conjugate_transpose_swaps_point_axes(self):
        rng = np.random.default_rng(3)
        ae = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        a = np.diag(np.full(4, 50.0))
        e = np.diag(np.full(4, 50.0))
        cov = Covariance(
            {("A", "A"): a, ("E", "E"): e, ("A", "E"): ae}, channel_names=AE, form="full"
        )
        np.testing.assert_array_equal(cov.EA, ae.conj().T)

    def test_stored_pairs_return_the_read_only_array_itself(self):
        cov = Covariance(diagonal_pairs(), channel_names=AE)
        assert cov.A is cov.pairs[("A", "A")]
        with pytest.raises(ValueError):
            cov.A[0] = 7.0

    def test_multi_letter_names_use_pair(self):
        cov = Covariance(
            {("X1", "X1"): np.ones(4), ("Y1", "Y1"): np.ones(4), ("X1", "Y1"): np.full(4, 0.2)},
            channel_names=("X1", "Y1"),
        )
        np.testing.assert_array_equal(cov.X1Y1, 0.2)
        np.testing.assert_array_equal(cov.pair("Y1", "X1"), 0.2)

    def test_ambiguous_attribute_names_are_errors(self):
        # "AAA" splits as ("A", "AA") and ("AA", "A"); both are valid pairs.
        cov = Covariance(
            {("A", "A"): np.ones(2), ("AA", "AA"): np.ones(2)}, channel_names=("A", "AA")
        )
        with pytest.raises(AttributeError, match="ambiguous"):
            _ = cov.AAA
        np.testing.assert_array_equal(cov.pair("A", "AA"), 0.0)

    def test_unknown_attribute_lists_valid_names(self):
        cov = Covariance(diagonal_pairs(), channel_names=AE)
        with pytest.raises(AttributeError, match="'T'.*channel attributes.*'AE'"):
            _ = cov.T
        assert not hasattr(cov, "T")
        assert hasattr(cov, "AE")

    def test_pair_rejects_unknown_channels(self):
        cov = Covariance(diagonal_pairs(), channel_names=AE)
        with pytest.raises(ValueError, match="unknown channel 'T'"):
            cov.pair("A", "T")


class TestCopy:
    def test_copy_is_independent_and_revalidated(self):
        cov = Covariance(diagonal_pairs(), channel_names=AE)
        copied = cov.copy()
        assert copied is not cov
        assert not np.shares_memory(copied.A, cov.A)
        np.testing.assert_array_equal(copied.A, cov.A)
        assert copied.channel_names == cov.channel_names and copied.form == cov.form

    def test_copy_catches_edited_arrays(self):
        cov = Covariance(diagonal_pairs(), channel_names=AE)
        cov.pairs[("A", "A")].setflags(write=True)
        cov.pairs[("A", "A")][:] = -1.0
        with pytest.raises(ValueError, match="nonnegative"):
            cov.copy()

    def test_validated_copy_skips_checks_but_copies_buffers(self):
        cov = Covariance(diagonal_pairs(), channel_names=AE)
        snapshot = cov._copy_validated()
        assert not np.shares_memory(snapshot.A, cov.A)
        assert not snapshot.A.flags.writeable
        np.testing.assert_array_equal(snapshot.A, cov.A)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_covariance_object.py -q`
Expected: `ImportError: cannot import name 'Covariance'`.

- [ ] **Step 3: Implement the class**

In `src/enchilada/covariance.py`, change the module docstring and imports to:

```python
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
```

Then insert the class directly after `split_pair_name` (before `_validate_covariance_values`):

```python
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
    derived on demand, and the reverse; see :meth:`in_basis`.

    The object carries no grid. Everything that depends on the domain takes
    the ``L1Data`` the covariance accompanies: :meth:`check_compatible`,
    :meth:`like`, :meth:`from_psd`, :meth:`from_variance`, :meth:`noise_psd`,
    :meth:`noise_variance` and ``enchilada.transform``.

    Construction checks what needs no grid: required diagonals, known
    channels, one orientation per pair, finite floating arrays of one shape,
    real nonnegative diagonals, Hermitian full-form diagonal blocks, and
    positive definiteness over the channels with nonzero variance at each
    point. Stored arrays are independent read-only copies. Equality is
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
        for ch in channels:
            if np.iscomplexobj(stored[(ch, ch)]):
                raise ValueError(f"diagonal pair {(ch, ch)!r} must be real")
        if self.form == "diagonal":
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
        if not isinstance(names, (tuple, list)) or not names:
            raise ValueError("channel_names must be a non-empty tuple of channel names")
        channels = tuple(names)
        if any(not isinstance(ch, str) or not ch for ch in channels) or len(
            set(channels)
        ) != len(channels):
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
        if not np.isfinite(value).all():
            raise ValueError(f"pair {pair_key!r} must be finite")
        return np.array(value, copy=True)

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
        for pattern in np.unique(inactive, axis=0):
            keep = ~pattern
            if not keep.any():
                continue
            selected = matrices[np.all(inactive == pattern, axis=1)][:, keep][
                :, :, keep
            ]
            try:
                np.linalg.cholesky(selected)
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
        """Require Hermitian diagonal blocks and store them exactly symmetric."""
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
            result[(ch, ch)] = symmetric.real.reshape(values.shape)
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
        raise NotImplementedError  # Task 2


_FIELD_NAMES = frozenset(f.name for f in fields(Covariance))
```

`_FIELD_NAMES` is referenced inside methods, so defining it after the class is fine.

- [ ] **Step 4: Run the new tests and the whole suite**

Run: `.venv/bin/python -m pytest tests/test_covariance_object.py -q`
Expected: all pass.
Run: `.venv/bin/python -m pytest -q && .venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format --check src tests && .venv/bin/python -m mypy`
Expected: 576 + new passed, 40 skipped; ruff and mypy clean. If `ruff format --check` complains, run `.venv/bin/python -m ruff format src/enchilada/covariance.py tests/test_covariance_object.py`.

- [ ] **Step 5: Commit**

```bash
git add src/enchilada/covariance.py tests/test_covariance_object.py docs/superpowers/plans/2026-11-10-covariance-object.md docs/superpowers/plans/2026-11-10-covariance-object.md.tasks.json
git commit -m "Add Covariance: per-pair arrays with channel attribute access

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Derived basis — `in_basis`, caching, attribute fallback

**Goal:** Implement spec 1.4: XYZ ⇄ AET through the orthonormal LDC matrix, computed once and cached, reachable through attributes.

**Files:**
- Modify: `src/enchilada/covariance.py` (replace the `in_basis` stub; add `_derive`)
- Test: `tests/test_covariance_object.py`

**Acceptance Criteria:**
- [ ] For XYZ channels, `cov.A`, `cov.E`, `cov.T`, `cov.AE`, `cov.AT`, `cov.ET`, `cov.EA` match `M C Mᵀ` computed explicitly per point, for both forms.
- [ ] `cov.in_basis("AET").in_basis("XYZ") is cov`; `cov.in_basis("XYZ") is cov`; repeated `in_basis("AET")` returns the same object.
- [ ] White equal noise in XYZ gives equal diagonals and zero cross terms in AET.
- [ ] With channels A, E only, `cov.X` raises `AttributeError` mentioning the triple and `in_basis("XYZ")` raises `ValueError`.
- [ ] `in_basis("ABC")` raises `ValueError`.

**Verify:** `.venv/bin/python -m pytest tests/test_covariance_object.py -q` → all pass.

**Steps:**

- [ ] **Step 1: Add the failing tests** to `tests/test_covariance_object.py`. Put the `import`/`from` lines at the top of the file with the existing imports (ruff E402); append only the helpers and classes:

```python
from enchilada.covariance import AET_FROM_XYZ

XYZ = ("X", "Y", "Z")
AET = ("A", "E", "T")


def xyz_pairs(n=6, with_cross=True):
    rng = np.random.default_rng(5)
    pairs = {
        ("X", "X"): np.full(n, 2.0),
        ("Y", "Y"): np.full(n, 3.0),
        ("Z", "Z"): np.full(n, 4.0),
    }
    if with_cross:
        pairs[("X", "Y")] = rng.uniform(-0.5, 0.5, n) + 1j * rng.uniform(-0.5, 0.5, n)
        pairs[("X", "Z")] = rng.uniform(-0.5, 0.5, n) + 1j * rng.uniform(-0.5, 0.5, n)
        pairs[("Y", "Z")] = rng.uniform(-0.5, 0.5, n) + 1j * rng.uniform(-0.5, 0.5, n)
    return pairs


def explicit_aet(cov):
    """Reference: per point, M C M^T with C the 3x3 channel matrix."""
    n = cov.X.size
    out = {}
    for p in range(n):
        c = np.array([[cov.pair(i, j)[p] for j in XYZ] for i in XYZ])
        out[p] = AET_FROM_XYZ @ c @ AET_FROM_XYZ.T
    return out


class TestDerivedBasis:
    def test_aet_matches_explicit_matrix_product(self):
        cov = Covariance(xyz_pairs(), channel_names=XYZ)
        reference = explicit_aet(cov)
        aet = cov.in_basis("AET")
        for a, first in enumerate(AET):
            for b, second in enumerate(AET):
                expected = np.array([reference[p][a, b] for p in range(cov.X.size)])
                np.testing.assert_allclose(aet.pair(first, second), expected, atol=1e-14)
                # The same values through the derived attribute names.
                name = first if a == b else first + second
                np.testing.assert_allclose(getattr(cov, name), expected, atol=1e-14)
        np.testing.assert_allclose(cov.EA, np.conj(cov.AE))

    def test_full_form_derivation_is_elementwise(self):
        n = 3
        base = np.diag(np.full(n, 5.0)) + 0.5 * np.eye(n, k=1) + 0.5 * np.eye(n, k=-1)
        pairs = {
            ("X", "X"): base, ("Y", "Y"): 2 * base, ("Z", "Z"): 3 * base,
            ("X", "Z"): 0.1 * np.eye(n),
        }
        cov = Covariance(pairs, channel_names=XYZ, form="full")
        aet = cov.in_basis("AET")
        assert aet.form == "full"
        m = AET_FROM_XYZ
        expected_aa = sum(
            m[0, i] * m[0, j] * cov.pair(XYZ[i], XYZ[j]) for i in range(3) for j in range(3)
        )
        np.testing.assert_allclose(aet.A, expected_aa, atol=1e-14)

    def test_round_trip_and_caching(self):
        cov = Covariance(xyz_pairs(), channel_names=XYZ)
        aet = cov.in_basis("AET")
        assert aet.channel_names == ("A", "E", "T")
        assert cov.in_basis("AET") is aet
        assert cov.in_basis("XYZ") is cov
        assert aet.in_basis("XYZ") is cov
        back = Covariance(dict(aet.pairs), channel_names=aet.channel_names).in_basis("XYZ")
        for first in XYZ:
            for second in XYZ:
                np.testing.assert_allclose(
                    back.pair(first, second), cov.pair(first, second), atol=1e-14
                )

    def test_white_equal_noise_stays_white(self):
        cov = Covariance(
            {(ch, ch): np.full(4, 7.0) for ch in XYZ}, channel_names=XYZ
        )
        np.testing.assert_allclose(cov.A, 7.0)
        np.testing.assert_allclose(cov.E, 7.0)
        np.testing.assert_allclose(cov.T, 7.0)
        np.testing.assert_allclose(cov.AE, 0.0, atol=1e-14)
        np.testing.assert_allclose(cov.AT, 0.0, atol=1e-14)
        np.testing.assert_allclose(cov.ET, 0.0, atol=1e-14)

    def test_any_channel_order_works(self):
        pairs = xyz_pairs()
        cov = Covariance(pairs, channel_names=("Z", "X", "Y"))
        reference = Covariance(pairs, channel_names=XYZ)
        np.testing.assert_allclose(cov.A, reference.A)
        np.testing.assert_allclose(cov.ET, reference.ET)

    def test_derived_names_need_the_full_triple(self):
        cov = Covariance(diagonal_pairs(), channel_names=AE)
        with pytest.raises(AttributeError, match="channel attributes"):
            _ = cov.X
        with pytest.raises(ValueError, match="exactly X, Y, Z or A, E, T"):
            cov.in_basis("XYZ")
        with pytest.raises(ValueError, match="basis must be one of"):
            Covariance(xyz_pairs(), channel_names=XYZ).in_basis("ABC")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_covariance_object.py -q -k Derived`
Expected: failures with `NotImplementedError`.

- [ ] **Step 3: Implement**

Replace the `in_basis` stub in `src/enchilada/covariance.py` with:

```python
    def in_basis(self, basis: str) -> "Covariance":
        """Return this covariance in the ``"XYZ"`` or ``"AET"`` basis.

        Available only when the channels are exactly X, Y, Z or exactly A, E, T.
        The derived object has all six pairs stored, the same ``form``, channel
        order ``("A", "E", "T")`` or ``("X", "Y", "Z")``, and is computed once.
        Asking for the current basis returns ``self``.
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
            derived = self._derive(current, basis)
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
                            total = total + weight * self.pair(source_first, source_second)
                assert isinstance(total, np.ndarray)
                pairs[(first, second)] = total.real if first == second else total
        return Covariance(pairs, channel_names=target, form=self.form)
```

- [ ] **Step 4: Run tests, lint, types**

Run: `.venv/bin/python -m pytest tests/test_covariance_object.py -q && .venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format --check src tests && .venv/bin/python -m mypy`
Expected: all pass, clean.

- [ ] **Step 5: Commit**

```bash
git add src/enchilada/covariance.py tests/test_covariance_object.py
git commit -m "Derive the A, E, T basis from X, Y, Z on demand

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Data-bound helpers — `like`, `check_compatible`, `from_psd`, `from_variance`, `noise_psd`, `noise_variance`

**Goal:** Implement spec 1.2, 1.5, 3.2–3.4 and 4.2 on `Covariance`.

**Files:**
- Modify: `src/enchilada/covariance.py`
- Test: `tests/test_covariance_object.py`

**Acceptance Criteria:**
- [ ] `Covariance.like(data, {"A": a, "E": e, "AE": ae})` resolves string keys, infers `form` from the data's channel shape, and runs `check_compatible`.
- [ ] `from_psd` requires frequency data, stores `(Tobs/2)·S(f_k)` at every bin including DC, and raises on a non-finite or non-positive PSD anywhere (message mentions `f = 0`).
- [ ] `from_variance` requires time data; accepts a scalar or a per-channel mapping.
- [ ] `noise_psd("A", data)` inverts `from_psd` exactly at every bin; `noise_variance("A", data)` on a `from_variance` object returns the variance; on a `from_psd(white)` object returns exactly `sigma**2`.
- [ ] `check_compatible` enforces channel order, shape per form, real arrays outside frequency, real entries where both coefficients are real in frequency (DC/Nyquist of a diagonal-form pair; the DC/Nyquist corners of a full-form block), positive diagonals in time/frequency, and WDM zeros exactly at structurally inactive pixels.

**Verify:** `.venv/bin/python -m pytest tests/test_covariance_object.py -q` → all pass.

**Steps:**

- [ ] **Step 1: Add failing tests** to `tests/test_covariance_object.py`. Put the `import`/`from` lines at the top of the file with the existing imports (ruff E402); append only the helpers and classes:

```python
from conftest import make_observed

from enchilada import L1Data, WDMGrid


class FlatPSD:
    def __init__(self, level):
        self.level = level

    def psd(self, frequencies, channel=None):
        return np.full_like(frequencies, self.level, dtype=float)


class TestLike:
    def test_string_keys_and_form_inference(self, rng):
        data = make_observed(rng, channel_names=AE)
        n = data.num_time_samples
        cov = Covariance.like(data, {"A": np.ones(n), "E": np.full(n, 4.0), "AE": np.full(n, 0.5)})
        assert cov.form == "diagonal"
        np.testing.assert_array_equal(cov.AE, 0.5)
        full = Covariance.like(data, {"A": np.eye(n), "E": np.eye(n)})
        assert full.form == "full"

    def test_bad_keys_and_shapes(self, rng):
        data = make_observed(rng, channel_names=AE)
        n = data.num_time_samples
        with pytest.raises(ValueError, match="neither a channel nor a pair"):
            Covariance.like(data, {"A": np.ones(n), "E": np.ones(n), "AT": np.ones(n)})
        with pytest.raises(ValueError, match="channel shape"):
            Covariance.like(data, {"A": np.ones(n + 1), "E": np.ones(n + 1)})
        with pytest.raises(ValueError, match="at least every channel"):
            Covariance.like(data, {})


class TestFactoriesAndReadBack:
    def test_from_variance_scalar_and_mapping(self, rng):
        data = make_observed(rng, channel_names=AE)
        cov = Covariance.from_variance(data, 0.25)
        np.testing.assert_array_equal(cov.A, 0.25)
        assert cov.noise_variance("A", data) == 0.25
        cov = Covariance.from_variance(data, {"A": 1.0, "E": 4.0})
        np.testing.assert_array_equal(cov.E, 4.0)
        with pytest.raises(ValueError, match="time-domain"):
            Covariance.from_variance(data.to_frequency(), 1.0)
        with pytest.raises(ValueError, match="keys must match"):
            Covariance.from_variance(data, {"A": 1.0})

    def test_from_psd_fills_every_bin_including_dc(self, rng):
        data = make_observed(rng, channel_names=AE).to_frequency()
        cov = Covariance.from_psd(data, FlatPSD(2.0))
        tobs = data.num_time_samples / data.sample_rate_hz
        np.testing.assert_allclose(cov.A, tobs / 2 * 2.0)
        assert cov.A.shape == (data.num_time_samples // 2 + 1,)
        np.testing.assert_allclose(cov.noise_psd("A", data), 2.0)

    def test_from_psd_rejects_non_frequency_data_and_bad_models(self, rng):
        data = make_observed(rng, channel_names=AE)
        with pytest.raises(ValueError, match="frequency-domain"):
            Covariance.from_psd(data, FlatPSD(1.0))
        spectrum = data.to_frequency()

        class Diverging:
            def psd(self, f, channel=None):
                with np.errstate(divide="ignore"):
                    return 1.0 / f**2

        with pytest.raises(ValueError, match="f = 0"):
            Covariance.from_psd(spectrum, Diverging())
        with pytest.raises(TypeError, match="psd"):
            Covariance.from_psd(spectrum, object())

    @pytest.mark.parametrize("n", [63, 64])
    def test_white_noise_variance_round_trips_exactly(self, rng, n):
        data = make_observed(rng, num_time_samples=n, channel_names=("A",))
        sigma2 = 0.09
        spectrum = data.to_frequency()
        cov = Covariance.from_psd(spectrum, FlatPSD(2 * sigma2 / data.sample_rate_hz))
        assert cov.noise_variance("A", spectrum) == pytest.approx(sigma2, rel=1e-12)

    def test_noise_variance_needs_stationary_time_and_noise_psd_needs_frequency(self, rng):
        data = make_observed(rng, channel_names=AE)
        n = data.num_time_samples
        cov = Covariance.like(data, {"A": np.linspace(1, 2, n), "E": np.ones(n)})
        with pytest.raises(ValueError, match="stationary"):
            cov.noise_variance("A", data)
        with pytest.raises(ValueError, match="frequency-domain"):
            cov.noise_psd("A", data)


class TestCheckCompatible:
    def test_channel_order_and_shape(self, rng):
        data = make_observed(rng, channel_names=AE)
        cov = Covariance.from_variance(data, 1.0)
        with pytest.raises(ValueError, match="channel_names"):
            cov.check_compatible(make_observed(rng, channel_names=("E", "A")))
        with pytest.raises(ValueError, match="shape"):
            cov.check_compatible(make_observed(rng, num_time_samples=32, channel_names=AE))
        with pytest.raises(ValueError, match="shape"):
            cov.check_compatible(data.to_frequency())

    def test_time_arrays_must_be_real_and_positive(self, rng):
        data = make_observed(rng, channel_names=AE)
        n = data.num_time_samples
        complex_cross = Covariance(
            {("A", "A"): np.ones(n), ("E", "E"): np.ones(n), ("A", "E"): np.full(n, 0.1j)},
            channel_names=AE,
        )
        with pytest.raises(ValueError, match="must be real"):
            complex_cross.check_compatible(data)
        zero = np.ones(n)
        zero[0] = 0.0
        with pytest.raises(ValueError, match="positive at every point"):
            Covariance({("A", "A"): zero, ("E", "E"): np.ones(n)}, channel_names=AE).check_compatible(data)

    def test_frequency_endpoints_must_be_real(self, rng):
        spectrum = make_observed(rng, channel_names=AE).to_frequency()
        k = spectrum.num_time_samples // 2 + 1
        cross = np.full(k, 0.1 + 0.1j)
        cov = Covariance({("A", "A"): np.ones(k), ("E", "E"): np.ones(k), ("A", "E"): cross}, channel_names=AE)
        with pytest.raises(ValueError, match="both coefficients are real"):
            cov.check_compatible(spectrum)
        cross = cross.copy()
        cross[0] = 0.1
        cross[-1] = 0.1
        Covariance({("A", "A"): np.ones(k), ("E", "E"): np.ones(k), ("A", "E"): cross}, channel_names=AE).check_compatible(spectrum)

    def test_wdm_zeros_exactly_at_inactive_pixels(self):
        grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=8)
        coefficients = np.zeros(grid.array_shape)
        # L1Data infers wdm_grid from the (num_frequency_divisions + 1, num_time_divisions) shape.
        data = L1Data(
            {"A": coefficients},
            sample_rate_hz=1.0,
            tdi_generation="2.0",
            physical_observable="fractional_frequency",
            data_domain="wdm",
        )
        variances = np.where(grid.active_mask, 2.0, 0.0)
        Covariance({("A", "A"): variances}, channel_names=("A",)).check_compatible(data)
        with pytest.raises(ValueError, match="inactive WDM pixels"):
            Covariance({("A", "A"): np.full(grid.array_shape, 2.0)}, channel_names=("A",)).check_compatible(data)
        leaky = variances.copy()
        leaky[1, 0] = 0.0
        with pytest.raises(ValueError, match="positive at active WDM"):
            Covariance({("A", "A"): leaky}, channel_names=("A",)).check_compatible(data)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_covariance_object.py -q -k "Like or Factories or CheckCompatible"`
Expected: `AttributeError`s for the missing methods.

- [ ] **Step 3: Implement** — add these methods to `Covariance` (after `__getattr__`, before the copy section):

```python
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
            raise ValueError("pairs must contain at least every channel's diagonal entry")
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
                raise ValueError("time_sample_variance keys must match channels exactly")
            levels = [float(time_sample_variance[ch]) for ch in channels]
        else:
            scalar = np.asarray(time_sample_variance)
            if scalar.ndim != 0 or scalar.dtype.kind not in "fiu":
                raise ValueError(
                    "time_sample_variance must be a real scalar for each channel"
                )
            levels = [float(scalar)] * len(channels)
        shape = reference_data.channel_data[channels[0]].shape
        return cls(
            {(ch, ch): np.full(shape, level) for ch, level in zip(channels, levels)},
            channel_names=channels,
        )

    def check_compatible(self, reference_data: "L1Data") -> None:
        """Require this covariance to describe ``reference_data``'s grid and domain.

        Checks channel order, the shape rule for ``form``, real arrays outside
        the frequency domain, real values at DC and even-length Nyquist for
        complex frequency pairs, positive variances in time and frequency, and
        WDM variances that are zero exactly at structurally inactive pixels.
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
                    endpoints = [arr[k, l] for k in real_bins for l in real_bins]
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
            raise ValueError(f"unknown channel {channel!r}; channels are {self.channel_names}")
        observation_length = reference_data.num_time_samples / reference_data.sample_rate_hz
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
            raise ValueError(f"unknown channel {channel!r}; channels are {self.channel_names}")
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
        frequency_resolution = reference_data.sample_rate_hz / reference_data.num_time_samples
        return float(np.sum(psd * weights) * frequency_resolution)
```

- [ ] **Step 4: Run tests, lint, types**

Run: `.venv/bin/python -m pytest tests/test_covariance_object.py -q && .venv/bin/python -m pytest -q && .venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format --check src tests && .venv/bin/python -m mypy`
Expected: all pass, clean. Wrap long lines if `ruff format --check` complains (`ruff format` the two files).

- [ ] **Step 5: Commit**

```bash
git add src/enchilada/covariance.py tests/test_covariance_object.py
git commit -m "Covariance factories, read-back helpers and check_compatible

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Translation seam for `Covariance` in `transform()`

**Goal:** Implement spec section 6 for the new class while the old classes still work: same domain and grid returns an independent copy; any other target raises `NotImplementedError`; `reference_data` is required.

**Files:**
- Modify: `src/enchilada/translation.py`
- Test: `tests/test_translation.py` (append)

**Acceptance Criteria:**
- [ ] `transform(cov, "time", reference_data=time_data)` returns a `Covariance` with equal arrays that share no memory with `cov`.
- [ ] `transform(cov, "frequency", reference_data=time_data)` raises `NotImplementedError` whose message names the spec file and "section 6".
- [ ] `transform(cov, "time")` without `reference_data` raises `ValueError` mentioning `reference_data`.
- [ ] A covariance incompatible with `reference_data` raises `ValueError` from `check_compatible` before any domain check.
- [ ] WDM to WDM with different division counts raises `NotImplementedError`; with the same grid returns a copy.

**Verify:** `.venv/bin/python -m pytest tests/test_translation.py -q` → all pass.

**Steps:**

- [ ] **Step 1: Add failing tests** at the end of `tests/test_translation.py`:

```python
from enchilada.covariance import Covariance


def test_covariance_same_domain_transform_is_an_independent_copy():
    data = observations()
    cov = Covariance.from_variance(data, 2.0)
    copied = transform(cov, "time", reference_data=data)
    assert isinstance(copied, Covariance)
    assert copied is not cov
    np.testing.assert_array_equal(copied.A, cov.A)
    assert not np.shares_memory(copied.A, cov.A)


def test_covariance_cross_domain_transform_is_the_seam():
    data = observations()
    cov = Covariance.from_variance(data, 2.0)
    with pytest.raises(NotImplementedError, match="section 6"):
        transform(cov, "frequency", reference_data=data)
    with pytest.raises(ValueError, match="reference_data"):
        transform(cov, "time")
    with pytest.raises(ValueError, match="shape"):
        transform(cov, "time", reference_data=observations(32))


@requires_wdm
def test_covariance_wdm_regrid_is_the_seam_but_same_grid_copies():
    data = observations()
    wdm = transform(data, "wdm", num_frequency_divisions=8)
    variances = np.where(wdm.wdm_grid.active_mask, 1.0, 0.0)
    cov = Covariance.like(wdm, {name: variances for name in wdm.channel_names})
    same = transform(cov, "wdm", num_frequency_divisions=8, reference_data=wdm)
    np.testing.assert_array_equal(same.A, cov.A)
    with pytest.raises(NotImplementedError):
        transform(cov, "wdm", num_frequency_divisions=4, reference_data=wdm)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_translation.py -q -k covariance_`
Expected: `TypeError: transform expects L1Data, a covariance, or BlockState` for the new class.

- [ ] **Step 3: Implement**

In `src/enchilada/translation.py` add the import `from enchilada.covariance import Covariance, DataCovariance` (replacing the existing `DataCovariance` import) and insert this branch in `transform` **before** `if isinstance(value, (DataCovariance, TranslatedCovariance)):`:

```python
    if isinstance(value, Covariance):
        # The seam for the covariance translation layer (design spec section 6).
        if reference_data is None:
            raise ValueError(
                "transform needs reference_data for a Covariance: the L1Data on "
                "its current grid"
            )
        value.check_compatible(reference_data)
        if target_domain == reference_data.data_domain:
            grid = _target_grid(
                target_domain,
                reference_data.num_time_samples,
                reference_data.wdm_grid,
                num_frequency_divisions,
                num_time_divisions,
            )
            if target_domain != "wdm" or grid == reference_data.wdm_grid:
                return value.copy()
        raise NotImplementedError(
            "covariance translation between domains is not implemented; see "
            "docs/superpowers/specs/2026-11-10-covariance-object-design.md section 6"
        )
```

Add an overload for the new class next to the existing ones:

```python
@overload
def transform(
    value: Covariance,
    target_domain: str,
    *,
    num_frequency_divisions: int | None = None,
    num_time_divisions: int | None = None,
    reference_data: L1Data | None = None,
) -> Covariance: ...
```

and widen the implementation's `value` and return annotations to include `Covariance`.

- [ ] **Step 4: Run tests, lint, types**

Run: `.venv/bin/python -m pytest tests/test_translation.py -q && .venv/bin/python -m pytest -q && .venv/bin/python -m ruff check src tests && .venv/bin/python -m mypy`
Expected: all pass, clean.

- [ ] **Step 5: Commit**

```bash
git add src/enchilada/translation.py tests/test_translation.py
git commit -m "Add the transform() seam for Covariance translation

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Switch the package to `Covariance`; delete the old classes; migrate tests, example scripts and the benchmark

**Goal:** Spec sections 2, 6 (interim behaviour) and 7 for source, tests, `examples/*.py` and `benchmarks/`. After this task nothing in the repository except the notebooks, README and CHANGELOG mentions `DataCovariance`, `TranslatedCovariance`, `covariance_matrix`, `noise_covariance` or `active_mask` (other than `WDMGrid.active_mask`).

**Files:**
- Modify: `src/enchilada/__init__.py`, `block.py`, `block_state.py`, `data.py`, `testing.py`, `wheel.py`, `translation.py`, `covariance.py`
- Delete: `src/enchilada/translated_covariance.py`, `tests/test_covariance_translation.py`, `tests/test_covariance.py`
- Rename: `tests/test_covariance_object.py` → `tests/test_covariance.py`
- Modify: every test file listed in 5.4; `examples/toy_fit.py`, `examples/gb_model.py`, `examples/domain_translation.py`; `benchmarks/benchmark_wheel.py`

**Acceptance Criteria:**
- [ ] `grep -rn "DataCovariance\|TranslatedCovariance\|covariance_matrix\|noise_covariance\|initial_noise_covariance" src tests examples/*.py benchmarks` prints nothing.
- [ ] `enchilada.__all__` equals the list in 5.1 and `tests/test_typing.py` pins it.
- [ ] Full suite passes with exactly these strict xfails (and the WDM/GB skips as before): `test_initial_block_state.py::test_seed_without_signal_can_sample_in_selected_domain[True]`, `test_wheel_translation.py::test_wdm_noise_publication_is_usable_in_every_domain`, `test_translation_review.py::test_noise_only_wdm_result_can_reuse_grid_with_compatible_time_reference`, `test_toy_translation.py::test_toy_time_blocks_preserve_draws_on_translated_observations[frequency|wdm]`, `test_gb_example_contract.py::test_gb_frequency_registration_restores_native_spectral_covariance[time|wdm]`.
- [ ] `examples/toy_fit.py`, `examples/domain_translation.py` (with WDM backend) and `benchmarks/benchmark_wheel.py` run.
- [ ] ruff, ruff format and mypy clean.

**Verify:** `.venv/bin/python -m pytest -q` → passes with the xfails above; `.venv/bin/python -m pytest -q --cov --cov-report=term-missing` ≥ 95 %.

**Steps:**

- [ ] **Step 5.1: Source modules**

`src/enchilada/__init__.py`: replace the two covariance imports with `from enchilada.covariance import Covariance`, delete the `translated_covariance` import, and set

```python
__all__ = [
    "Block",
    "Covariance",
    "Ledger",
    "NoiseOverwrittenWarning",
    "NumericOrbit",
    "Orbit",
    "L1Data",
    "BlockState",
    "Wheel",
    "WDMGrid",
    "__version__",
    "replace",
    "transform",
]
```

`src/enchilada/block_state.py`: imports become `from enchilada.covariance import Covariance` (drop the `translated_covariance` import). Field and method:

```python
    template: dict[str, np.ndarray] | None = None
    covariance: Covariance | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
```

```python
        if self.covariance is not None and not isinstance(self.covariance, Covariance):
            raise TypeError("BlockState.covariance must be a Covariance or None")
```

```python
    def with_covariance(self, covariance: Covariance) -> "BlockState":
        """Return a shallow replacement with the supplied covariance.

        Other fields remain shared. Publication occurs only when Wheel adopts
        the returned state.
        """
        return replace(self, covariance=covariance)
```

Docstring paragraph: replace the `covariance_matrix` paragraph with

```
    `covariance` is a Covariance replacing the current noise model; components
    are not summed. A signal block can omit it. The current covariance owner
    must supply its complete covariance on every call, on the grid of the data
    it received. A noise-only state needs no signal arrays.
```

`src/enchilada/block.py`: imports `from enchilada.covariance import Covariance` only (drop `translated_covariance`); signature

```python
    def sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        current_block_state: BlockState,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
```

Docstring: `noise_covariance` → `covariance`; replace "`noise_covariance` is the current DataCovariance, TranslatedCovariance, or None." with "`covariance` is the current `Covariance` on the same grid as `conditional_residual`, or None." Delete the sentence "A translated covariance exposes exact operators; it may correlate samples."

`src/enchilada/data.py`: imports `from enchilada.covariance import Covariance` (drop `translated_covariance`); in `block_state` and `zero_block_state` rename the keyword `covariance_matrix` → `covariance` (parameter, docstring, the `BlockState(...)` call and the `self.block_state(...)` call), type `Covariance | None`; in the class docstring replace "`DataCovariance`, `TranslatedCovariance`, or None" with "a `Covariance` or None" and "must match covariance matrix channel axes" with "and is the channel order a `Covariance` uses"; in `zero_block_state`'s docstring example use `BlockState(covariance=covariance)`.

`src/enchilada/testing.py`: import `Covariance`; `EchoBlock.sample(self, conditional_residual, covariance, current_block_state, *, rng)`; `check_block(..., initial_covariance: Covariance | None = None, ...)` passing `initial_covariance=initial_covariance` to `Wheel`; docstrings accordingly ("Starting Covariance for the scratch Wheel").

`src/enchilada/wheel.py`:

```python
from enchilada.covariance import Covariance
```
(remove the `translated_covariance` import). Constructor:

```python
    def __init__(
        self,
        observed_data: L1Data,
        initial_covariance: Covariance | None = None,
        *,
        random_seed: int | None = None,
    ) -> None:
        ...
        if initial_covariance is not None and not isinstance(
            initial_covariance, Covariance
        ):
            raise TypeError("initial_covariance must be a Covariance or None")
        if initial_covariance is not None:
            initial_covariance = initial_covariance.copy()
            initial_covariance.check_compatible(observed_data)
        self._observed_data = self._copy_data(observed_data)
        self._working_residual = self._copy_data(observed_data)
        self._covariance = initial_covariance
```

Property:

```python
    @property
    def covariance(self) -> Covariance | None:
        """The current covariance, as an independent copy."""
        return None if self._covariance is None else self._covariance._copy_validated()
```

`add`: error text `"a Block needs name and sample(conditional_residual, covariance, current_block_state, *, rng)."`; after the existing `if domain_selection is not None and initial_block_state.template is not None:` block add

```python
        elif domain_selection is not None and initial_block_state.covariance is not None:
            # A noise-only seed is on the block's grid. Only that grid is needed,
            # so build an empty reference instead of transforming observations.
            domain, grid = domain_selection
            source_reference = replace(
                self._observed_data,
                channel_data={
                    ch: _empty_channel(domain, self._observed_data.num_time_samples, grid)
                    for ch in self._observed_data.channel_names
                },
                data_domain=domain,
            )
```

with this module-level helper placed after `DomainSelection = ...`:

```python
def _empty_channel(
    domain: str, num_time_samples: int, grid: WDMGrid | None
) -> np.ndarray:
    """Zeros with the channel shape and dtype of ``domain`` on this grid."""
    if domain == "time":
        return np.zeros(num_time_samples)
    if domain == "frequency":
        return np.zeros(num_time_samples // 2 + 1, dtype=complex)
    assert grid is not None
    return np.zeros(grid.array_shape)
```

`_block_inputs` return type `tuple[L1Data, Covariance | None]`; body:

```python
        conditional_residual = self.residual(exclude_block_name=exclude_block_name)
        if domain_selection is None:
            return conditional_residual, self.covariance
        domain, grid = domain_selection
        divisions = None if grid is None else grid.num_frequency_divisions
        conditional_residual = transform(
            conditional_residual, domain, num_frequency_divisions=divisions
        )
        covariance = None
        if self._covariance is not None:
            # transform() validates against the observation grid and returns a
            # copy, so the owned object is never exposed.
            covariance = transform(
                self._covariance,
                domain,
                num_frequency_divisions=divisions,
                reference_data=self._observed_data,
            )
        return conditional_residual, covariance
```

`run`: `conditional_residual, covariance = self._block_inputs(...)` and `block.sample(conditional_residual=conditional_residual, covariance=covariance, current_block_state=current_block_state, rng=candidate_rng)`.

`_adopt`:

```python
        candidate_covariance = self._covariance
        candidate_noise_owner_block_name = self._noise_owner_block_name
        if (
            block_name == candidate_noise_owner_block_name
            and candidate_block_state.covariance is None
        ):
            raise ValueError(
                f"{block_name}.{source_method_name} must return its complete "
                "covariance because it owns the current covariance"
            )
        if candidate_block_state.covariance is not None:
            # Revalidate: callers can edit arrays inside a frozen dataclass.
            candidate_covariance = candidate_block_state.covariance.copy()
            candidate_covariance.check_compatible(self._observed_data)
            candidate_block_state = replace(
                candidate_block_state, covariance=candidate_covariance
            )
            candidate_noise_owner_block_name = block_name
```
and at the commit point `self._covariance, self._noise_owner_block_name = (candidate_covariance, candidate_noise_owner_block_name)`. Docstrings: `initial_noise_covariance` → `initial_covariance`, "DataCovariance or TranslatedCovariance" → "Covariance", "Covariance carries its own source grid metadata." → "A seed covariance is on the block's grid, like its template."

`src/enchilada/translation.py`: remove the `DataCovariance`/`TranslatedCovariance` imports, overload and branch (keep the `Covariance` branch from Task 4); in the `BlockState` branch replace `source_state.covariance_matrix` with `source_state.covariance`, and `replace(source_state, template=signal, covariance_matrix=covariance)` with `covariance=covariance`; drop the sentence in the docstring about `TranslatedCovariance` and replace it with "Covariance translation between domains is a seam that raises `NotImplementedError` until the translation layer is installed (design spec section 6)." The final `raise TypeError("transform expects L1Data, Covariance, or BlockState")`.

`src/enchilada/covariance.py`: delete `_validate_covariance_values` and the whole `DataCovariance` class; delete the now-unused `from enchilada.domains import WDMGrid` import if nothing uses it.

Delete `src/enchilada/translated_covariance.py`.

- [ ] **Step 5.2: Mechanical renames across tests, examples and benchmarks**

Save this as `$TMPDIR/rename_covariance.py` and run it with `.venv/bin/python $TMPDIR/rename_covariance.py` from the repo root:

```python
import re
from pathlib import Path

ROOT = Path(".")
FILES = [
    *sorted(ROOT.glob("tests/*.py")),
    *sorted(ROOT.glob("examples/*.py")),
    *sorted(ROOT.glob("benchmarks/*.py")),
]
SKIP = {"tests/test_covariance.py", "tests/test_covariance_translation.py",
        "tests/test_covariance_object.py"}
RULES = [
    (r"\bDataCovariance\b", "Covariance"),
    (r"\binitial_noise_covariance\b", "initial_covariance"),
    (r"\bwith_covariance_matrix\b", "with_covariance"),
    (r"\bcovariance_matrix\b", "covariance"),
    (r"\bnoise_covariance\b", "covariance"),
    (r"Covariance \| TranslatedCovariance \| None", "Covariance | None"),
    (r",\s*TranslatedCovariance\b", ""),
    (r"\bTranslatedCovariance,\s*", ""),
    (r"\.noise_variance\(\)", '.noise_variance("A", observed)'),
    (r'\.noise_variance\("A"\)', '.noise_variance("A", observed)'),
    (r"\.noise_variance\(channel_name=ch\)", ".noise_variance(ch, conditional_residual)"),
    (r"\.noise_psd\(channel_name=ch\)", ".noise_psd(ch, conditional_residual)"),
]
for path in FILES:
    if str(path) in SKIP:
        continue
    text = original = path.read_text()
    for pattern, replacement in RULES:
        text = re.sub(pattern, replacement, text)
    if text != original:
        path.write_text(text)
        print("rewrote", path)
```

Then `git rm tests/test_covariance.py tests/test_covariance_translation.py` and `git mv tests/test_covariance_object.py tests/test_covariance.py`.

- [ ] **Step 5.3: Example scripts and benchmark (hand edits after the script)**

`examples/toy_fit.py`: import line `from enchilada import BlockState, Covariance, L1Data, Wheel`. Replace `SineBlock.sample`'s validation (from `if covariance is None:` through the `uncorrelated` check) with:

```python
        if covariance is None:
            raise ValueError("SineBlock requires a white-noise covariance")
        if conditional_residual.data_domain != "time" or covariance.form != "diagonal":
            raise ValueError(
                "SineBlock requires time-domain white noise with one variance per sample"
            )
        try:
            covariance.check_compatible(conditional_residual)
        except ValueError as exc:
            raise ValueError(
                "SineBlock requires time-domain white noise on the data grid"
            ) from exc
        channels = conditional_residual.channel_names
        if any(
            np.any(covariance.pair(first, second))
            for first in channels
            for second in channels
            if first != second
        ):
            raise ValueError("SineBlock requires uncorrelated channels")
```

`WhiteNoiseBlock._render`: `BlockState(covariance=Covariance.from_variance(reference_data=conditional_residual, time_sample_variance=variance), ...)` (the script already renamed the keyword; check it). Module docstring: "Their likelihood requires native, fully active white time noise." → "Their likelihood requires white time noise with one variance per sample."

`examples/gb_model.py`: import `from enchilada import BlockState, Covariance, L1Data`. `FixedLISANoise`:

```python
class FixedLISANoise:
    """One-sided LISA PSD from lisatools ``get_sensitivity`` (A == E).

    The only contract an enchilada noise object needs: ``psd(freqs[, channel])``.
    ``Covariance.from_psd`` evaluates the model at every rfft bin including
    ``f = 0``, so frequencies are floored at ``minimum_frequency_hz``.
    """

    def __init__(self, minimum_frequency_hz: float = 1e-5):
        self.minimum_frequency_hz = minimum_frequency_hz

    def psd(self, f, channel=None):
        from lisatools.sensitivity import A1TDISens, get_sensitivity

        floored = np.maximum(np.asarray(f, dtype=float), self.minimum_frequency_hz)
        return get_sensitivity(floored, sens_fn=A1TDISens)
```

`GBBlock._context`: replace from `if covariance is None:` to the `return _new_gb(), {...}` with:

```python
        if covariance is None:
            raise ValueError(f"{self.name}: GBBlock requires a noise covariance")
        try:
            covariance.check_compatible(conditional_residual)
        except ValueError as exc:
            raise ValueError(
                f"{self.name}: this narrowband likelihood requires a native "
                "frequency-domain covariance on the data grid"
            ) from exc
        if covariance.form != "diagonal":
            raise ValueError(f"{self.name}: this likelihood needs one PSD value per bin")
        channels = conditional_residual.channel_names
        if any(
            np.any(covariance.pair(first, second))
            for first in channels
            for second in channels
            if first != second
        ):
            raise ValueError(
                f"{self.name}: this likelihood requires uncorrelated channels"
            )
        return _new_gb(), {
            ch: covariance.noise_psd(ch, conditional_residual) for ch in channels
        }
```
Class docstring: delete "generic translated covariance operators need a different likelihood. Wheel can restore a native frequency covariance that was stored in time or WDM." and write "The covariance must be on the frequency grid of the data it accompanies."

`examples/domain_translation.py`: imports `from enchilada import BlockState, L1Data, Wheel, transform`; delete `noise = ...`, `frequency_noise = ...`, `wdm_noise = ...`, the `assert isinstance(wdm_noise, ...)`, the whole `quadratic_forms` block, and `"quadratic_forms": quadratic_forms,` from the report; `wheel = Wheel(observed, random_seed=3)`; `DomainEcho._check_input` becomes

```python
    def _check_input(self, conditional_residual, covariance):
        if conditional_residual.data_domain != self.expected_domain:
            raise ValueError(f"{self.name}: unexpected data representation")
        # Covariance translation across domains arrives with the translation
        # layer (design spec section 6); this walkthrough runs without one.
        if covariance is not None:
            raise ValueError(f"{self.name}: expected no covariance in this walkthrough")
```
Module docstring first line: "Translate data and block states, then run a mixed-domain Wheel."

`benchmarks/benchmark_wheel.py`: the script handled it (`Covariance.from_variance(observed, 1.0)`, `covariance` parameter). Confirm with `grep -n Covariance benchmarks/benchmark_wheel.py`.

- [ ] **Step 5.4: Test rewrites (after the script)**

For each file, make the listed edits. Everything not listed was handled by the script.

`tests/test_typing.py`: import list `Block, L1Data, Covariance, BlockState, Wheel, WDMGrid, transform`; lines 44–47 become

```python
noise = Covariance.from_variance(r, 1.0)
assert_type(transform(noise, "time", reference_data=r), Covariance)
assert_type(transform(BlockState(), "time"), BlockState)
wheel = Wheel(r, noise)
```
and `test_public_surface_is_pinned` pins the `__all__` list from 5.1.

`tests/test_wheel.py`:
- `test_initial_psd_model_must_be_converted_explicitly`: `match="initial_covariance.*Covariance"`; `spectrum = observed.to_frequency()`; `covariance = Covariance.from_psd(spectrum, psd_model)`; `wheel = Wheel(spectrum, initial_covariance=covariance)`; final assert `np.testing.assert_array_equal(wheel.covariance.A, covariance.A)`.
- `test_fixed_covariance_is_passed_explicitly`: final assert `np.testing.assert_array_equal(wheel.covariance.A, covariance.A)`.
- `test_a_signal_block_does_not_disturb_the_noise_model`: `obs = make_observed(rng).to_frequency()`; `expected = wheel.covariance.A.copy()`; final `np.testing.assert_array_equal(wheel.covariance.A, expected)`.
- `test_initial_covariance_subclass_cannot_skip_grid_validation`:

```python
        class UncheckedCovariance(Covariance):
            def check_compatible(self, reference_data):
                pass

        mismatched_grid = make_observed(np.random.default_rng(1), num_time_samples=32)
        covariance = Covariance.from_variance(mismatched_grid, 1.0)
        unchecked = UncheckedCovariance(
            dict(covariance.pairs), channel_names=covariance.channel_names
        )
        with pytest.raises(ValueError, match="shape"):
            Wheel(observed, initial_covariance=unchecked)
```
- Every `wheel.covariance.noise_variance("A", observed)` the script produced inside a test whose data variable is not `observed` (grep `noise_variance("A", observed)` and check each enclosing function): in `TestExplicitNoise.test_initial_noise_is_available_to_explicit_initialization` keep `observed`; in `test_sample_sees_current_noise` the recorder's `levels.append(covariance.noise_variance("A", observed))` → use `conditional_residual`.
- `assert isinstance(covariance, Covariance)` stays valid.

`tests/test_state.py` `test_covariance_input_and_output_snapshots_are_isolated`:

```python
    covariance = Covariance.from_variance(observed, 1)
    wheel = Wheel(observed, initial_covariance=covariance)
    covariance.pairs[("A", "A")].setflags(write=True)
    covariance.pairs[("A", "A")][:] *= 2
    inspected = wheel.covariance
    inspected.pairs[("A", "A")].setflags(write=True)
    inspected.pairs[("A", "A")][:] *= 3

    class EditsNoise(Counter):
        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            covariance.pairs[("A", "A")].setflags(write=True)
            covariance.pairs[("A", "A")][:] *= 4
            return current_block_state
```
(final assert already `wheel.covariance.noise_variance("A", observed) == 1`).

`tests/test_initial_block_state.py`:
- lines 173–174 → `initial.covariance.pairs[("A", "A")].setflags(write=True)` / `initial.covariance.pairs[("A", "A")][:] = 100`.
- `test_seed_without_signal_can_sample_in_selected_domain`: parametrize `[False, pytest.param(True, marks=pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="covariance translation seam, design spec section 6"))]`; inside `if with_noise:` replace the three asserts with `current_block_state.covariance.check_compatible(conditional_residual)` and `covariance.check_compatible(conditional_residual)`. A seed covariance is on the block's grid, so build it there (otherwise `Wheel.add` fails the shape check with `ValueError` before reaching the seam and the strict xfail reports a failure):
```python
    # White noise with variance 0.25 on the block's frequency grid:
    # C = (Tobs / 2) * (2 sigma^2 / fs) = N sigma^2 / fs^2 = 64 * 0.25 / 0.25.
    spectrum = observed.to_frequency()
    initial = BlockState(
        covariance=Covariance.like(
            spectrum, {ch: np.full(33, 64.0) for ch in spectrum.channel_names}
        )
        if with_noise
        else None,
        parameters={"variance": 0.25},
    )
```
- `test_seeded_noise_takeover_warning_can_fail_atomically`: drop `data_domain="frequency"` from the `wheel.add(second, ...)` call (the assertion `"second" not in wheel._block_domains` still holds).

`tests/test_block_state.py`: delete `test_block_states_accept_exact_translated_covariance` and its parametrize decorator. In `test_rejected_noise_result_preserves_signal_covariance_state_and_rng`, add `from conftest import make_observed` to the imports, change the failing branch to

```python
                if failure == "wrong_grid":
                    wrong_grid = make_observed(np.random.default_rng(0), num_time_samples=32)
                    covariance = Covariance.from_variance(wrong_grid, 2)
                return replace(result, covariance=covariance)
```

and the expectation to `error = "must return.*covariance" if failure == "omitted" else "shape"`. The final `actual.covariance.noise_variance("A", observed) == 1` is produced by the rename script.

`tests/test_block_state_contract.py` line 58: `accepted.covariance.noise_variance(name, observed) == 2.0`.

`tests/test_data.py`:
- `test_inferred_channel_order_aligns_covariance_axes`: replace the two covariance lines with
```python
        covariance = Covariance.from_variance(data, {"A": 1.0, "E": 4.0})
        assert covariance.channel_names == ("E", "A")
        assert covariance.E[0] == 4 and covariance.A[0] == 1
        whitened = sum(
            float(np.sum(data.channel_data[ch] ** 2 / covariance.pair(ch, ch)))
            for ch in data.channel_names
        )
        assert whitened == pytest.approx(8.0)
```
- `test_explicit_channel_order_overrides_dictionary_order`: `assert covariance.A[0] == 1 and covariance.E[0] == 4` and `covariance.check_compatible(data)`.
- PSD test (line ~626): `noise_covariance = Covariance.from_psd(t.to_frequency(), White())` → script renamed the variable to `covariance`; `predicted = covariance.A`.
- line 701: `assert t.covariance is None`.
- `test_with_covariance_matrix_returns_a_new_block_state` → rename test to `test_with_covariance_returns_a_new_block_state`; body already renamed by the script.

`tests/test_testing.py`: `FlatNoiseBlock` publishes `Covariance.from_psd(conditional_residual, FlatPSD())`, so call `check_block(FlatNoiseBlock(name="noise"), observed.to_frequency(), initial_block_state=BlockState())`; `BadNoise` expectation `match="Covariance"`.

`tests/test_translation.py`:
- `test_same_domain_transform_is_an_independent_snapshot`: `copied_covariance = transform(covariance, "time", reference_data=data)`; `assert not np.shares_memory(copied_covariance.A, covariance.A)`.
- `test_result_conversion_preserves_state_without_aliasing`: remove `covariance=Covariance.from_variance(data, 3.0),` from the state and the `assert converted.covariance...` line.
- `test_noise_only_and_parameters_only_states_need_no_signal_grid` becomes:
```python
def test_noise_only_state_needs_reference_data_and_parameters_only_does_not():
    data = observations()
    result = BlockState(covariance=Covariance.from_variance(data, 1.0))
    with pytest.raises(ValueError, match="reference_data"):
        transform(result, "frequency")
    with pytest.raises(NotImplementedError, match="section 6"):
        transform(result, "frequency", reference_data=data)
    same = transform(result, "time", reference_data=data)
    np.testing.assert_array_equal(same.covariance.A, 1.0)
    state = BlockState(parameters={"amplitude": 1.0})
    copied = transform(state, "frequency")
    assert copied is not state
    assert copied.parameters == state.parameters
```
- `test_conversion_rejects_missing_or_mismatched_reference`: last `with` block → `with pytest.raises(ValueError, match="shape"): transform(Covariance.from_variance(data, 1.0), "time", reference_data=observations(32))`.
- `test_joint_wdm_result_reuses_signal_grid_for_covariance` → rename to `test_joint_wdm_result_converts_signal_with_a_wdm_reference`; remove `covariance=Covariance.from_variance(data, 2.0)` and the two `converted.covariance...` asserts; keep the template no-aliasing assert.

`tests/test_toy_translation.py`: import `from enchilada import Wheel, transform`; delete `test_sine_rejects_translated_statistical_mask_with_likelihood_error` (the DC mask no longer exists); on `test_toy_time_blocks_preserve_draws_on_translated_observations` add `@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="covariance translation seam, design spec section 6")` and replace lines 96–98 with `assert wheels[1].covariance is not None`.

`tests/test_translation_example.py`: `examples/domain_translation.py` no longer reports `quadratic_forms`, so delete the five-line `np.testing.assert_allclose(list(report["quadratic_forms"].values()), ...)` call. Everything else in the report is unchanged.

`tests/test_translation_review.py`: import `from enchilada import BlockState, Covariance, Wheel`. Delete `test_translated_wdm_covariance_rejects_inactive_data_before_backend_use`. Rewrite:

```python
@pytest.mark.skipif(importlib.util.find_spec("wdm") is None, reason="local WDM backend is optional")
@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="covariance translation seam, design spec section 6")
def test_noise_only_wdm_result_can_reuse_grid_with_compatible_time_reference(observed):
    covariance = transform(
        Covariance.from_variance(observed, 1.0), "wdm", num_frequency_divisions=4,
        reference_data=observed,
    )
    original = BlockState(covariance=covariance, parameters={"variance": 1.0})
    wdm_reference = transform(observed, "wdm", num_frequency_divisions=4)
    converted = transform(original, "wdm", reference_data=wdm_reference)
    assert converted.parameters == original.parameters
    assert converted.template is None
    assert converted.covariance is not covariance


def test_omitted_domain_and_explicit_same_domain_both_receive_the_covariance(observed):
    received = []

    class RecordInputs:
        def __init__(self, name):
            self.name = name

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            covariance.check_compatible(conditional_residual)
            received.append(("sample", self.name, conditional_residual.data_domain))
            return current_block_state

    wheel = Wheel(observed, Covariance.from_variance(observed, 2.0))
    wheel.add(RecordInputs("default"), initial_block_state=BlockState())
    wheel.add(RecordInputs("explicit"), initial_block_state=BlockState(), data_domain="time")
    assert received == []
    wheel.run(1)
    assert received == [("sample", "default", "time"), ("sample", "explicit", "time")]
    assert wheel.covariance.noise_variance("A", observed) == 2.0
```
In `test_rejected_translated_output_rolls_back_mutated_inputs_and_rng`: delete the three `native...` lines inside `if fail[0]:`, construct the wheels as `Wheel(canonical, random_seed=7)` (no covariance), delete `initial_covariance = ...` and the final covariance equality assert.

`tests/test_wheel_translation.py`: `ConstantSignal.sample`: delete `assert covariance.data_domain == self.expected_domain`. `test_mixed_blocks_receive_matching_inputs_and_return_canonical_states`: `wheel = Wheel(canonical, random_seed=42)` (no covariance; the seam would raise for the non-canonical blocks). `test_wdm_noise_publication_is_usable_in_every_domain`: keep any existing skip marker, add the strict xfail marker, and rewrite the body:

```python
    class PixelNoise:
        name = "noise"

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            covariance.check_compatible(conditional_residual)
            assert isinstance(current_block_state.covariance, Covariance)
            return current_block_state

    data = observations()
    wdm_data = transform(data, "wdm", num_frequency_divisions=8)
    grid = wdm_data.wdm_grid
    variances = np.where(grid.active_mask, 1.0, 0.0)
    variances[1:-1] *= np.arange(1, grid.num_frequency_divisions)[:, None]
    initial = BlockState(
        covariance=Covariance.like(wdm_data, {name: variances for name in data.channel_names})
    )
    wheel = Wheel(data)
    wheel.add(PixelNoise(), initial_block_state=initial, data_domain="wdm", num_frequency_divisions=8)
    assert wheel.covariance is not None
    assert wheel.ledger["noise"].template is None
    wheel.run(2)
    np.testing.assert_array_equal(wheel.residual().channel_data["A"], data.channel_data["A"])
```
`test_failed_translated_result_preserves_accepted_state_and_rng[covariance_grid]`: the block is in the frequency domain, so a returned covariance built on `replace(conditional_residual, sample_rate_hz=2.0)` has the right shape and the seam raises `NotImplementedError` on the way back. Change the failing return to `BlockState(covariance=Covariance.from_variance(transform(conditional_residual, "time"), 1.0))` (a time-shaped covariance on frequency data: the seam runs `check_compatible` first and fails with "shape") and the `pytest.raises(ValueError)` stays. The two wheels there carry no initial covariance, so `_block_inputs` never reaches the seam.

`tests/test_examples.py`:
- `test_sine_rejects_covariance_its_likelihood_cannot_use`: replace everything from `noise = Covariance.from_variance(data, 1.0)` to the end of the function with
```python
    n = data.num_time_samples
    noise = Covariance.like(
        data, {"A": np.ones(n), "E": np.ones(n), "AE": np.full(n, 0.5)}
    )
    prior = toy.draw_sine_prior(signal, data, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="uncorrelated"):
        signal.sample(data, noise, prior, rng=np.random.default_rng(1))
```
  (`replace` stays imported: `data` is still built with it; the local `from enchilada import Covariance` comes from the rename script.)
- `test_sine_requires_fully_observed_time_white_noise`: replace the whole function, including its parametrize, with
```python
def test_sine_requires_time_white_noise_on_the_data_grid():
    from enchilada import Covariance

    toy = load_example("toy_fit")
    data = toy.make_observed()
    block = toy.SineBlock("slow", 0.004)
    prior = toy.draw_sine_prior(block, data, rng=np.random.default_rng(0))

    class UnitPSD:
        def psd(self, f, channel=None):
            return np.ones_like(f)

    noise = Covariance.from_psd(data.to_frequency(), UnitPSD())
    with pytest.raises(ValueError, match="time-domain white noise on the data grid"):
        block.sample(data, noise, prior, rng=np.random.default_rng(1))
```

`tests/test_gb_example_contract.py`:
- `test_gb_rejects_covariance_without_native_frequency_psd`: drop the `translated` parametrize and parameter; body
```python
    data, _ = spectral_context
    time_noise = Covariance.from_variance(transform(data, "time"), 1.0)
    with pytest.raises(ValueError, match="native frequency.*covariance"):
        gb_model.draw_gb_prior(
            make_block(gb_model), data, time_noise, rng=np.random.default_rng(0)
        )
```
- `test_gb_rejects_covariance_on_a_different_grid`: the object carries no sample rate, so the mismatch is a shorter grid:
```python
    data, _ = spectral_context
    time_data = transform(data, "time")
    shorter = replace(
        time_data,
        channel_data={ch: arr[:-2] for ch, arr in time_data.channel_data.items()},
        num_time_samples=time_data.num_time_samples - 2,
    )
    noise = Covariance.from_psd(transform(shorter, "frequency"), UnitPSD())
    with pytest.raises(ValueError, match="native frequency.*covariance"):
        gb_model.draw_gb_prior(
            make_block(gb_model), data, noise, rng=np.random.default_rng(0)
        )
```
- `test_gb_frequency_registration_restores_native_spectral_covariance`: add `@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="covariance translation seam, design spec section 6")` below the parametrize, and pass `initial_covariance=transform(noise, storage_domain, reference_data=data, **options)` (the `pytest.importorskip("wdm")` inside the body still skips the WDM case without the backend).

`tests/test_block_tutorial.py`, `tests/test_ledger.py`, `tests/test_block_tutorial.py`, `tests/test_examples.py` (other tests), `tests/test_testing.py` (other tests), `tests/test_domains.py`: script only; verify by reading the diff.

- [ ] **Step 5.5: Run everything**

```bash
grep -rn "DataCovariance\|TranslatedCovariance\|covariance_matrix\|noise_covariance" src tests examples/*.py benchmarks || echo CLEAN
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q -rx | grep -i xfail      # lists the six expected xfails
.venv/bin/python -m ruff check src tests examples benchmarks
.venv/bin/python -m ruff format src tests examples benchmarks
.venv/bin/python -m mypy
.venv/bin/python examples/toy_fit.py | head -3
.venv/bin/python benchmarks/benchmark_wheel.py --help 2>/dev/null || .venv/bin/python benchmarks/benchmark_wheel.py | head -3
```
Expected: CLEAN; all pass except the six strict xfails; lint and types clean; the two scripts run. Iterate on failures by reading the failing test and fixing either the test's migration or the implementation, keeping the spec's rules.

- [ ] **Step 5.6: Commit**

```bash
git add -u src tests examples/toy_fit.py examples/gb_model.py examples/domain_translation.py benchmarks/benchmark_wheel.py
git add tests/test_covariance.py
git commit -m "Switch the package to Covariance; remove DataCovariance and TranslatedCovariance

BlockState.covariance, Wheel(initial_covariance=...), Wheel.covariance and
sample(conditional_residual, covariance, current_block_state, *, rng)
replace the old names. Covariance translation across domains is a seam in
transform() that raises NotImplementedError until the translation layer
lands; the six Wheel tests that need it are strict xfails.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Notebooks

**Goal:** Migrate the code cells of the four notebooks to the new API; keep them compiling (`test_all_notebooks_contain_valid_python`) and keep `create_block.ipynb` runnable end to end.

**Files:**
- Modify: `examples/create_block.ipynb`, `examples/domain_translation.ipynb`, `examples/gb_block_eryn.ipynb`, `examples/demo.ipynb` (markdown mentions only)

**Acceptance Criteria:**
- [ ] No notebook cell mentions `DataCovariance`, `TranslatedCovariance`, `covariance_matrix`, `noise_covariance`, `initial_noise_covariance`, `.solve(`, `.apply(`, `quadratic_form`, `degrees_of_freedom`.
- [ ] Concatenating `create_block.ipynb`'s code cells and running them with `.venv/bin/python -c` exits 0; with the WDM backend installed, `tests/test_translation_example.py::test_translation_notebook_runs_all_cells_with_real_backend` passes.
- [ ] `tests/test_block_tutorial.py` and `tests/test_examples.py` pass.
- [ ] Edited cells have their `outputs` cleared and `execution_count` set to null; markdown cells that describe removed behaviour are updated.

**Verify:** `.venv/bin/python -m pytest tests/test_examples.py tests/test_block_tutorial.py -q` and the concatenated run below.

**Steps:**

- [ ] **Step 1: Apply the edits with a script.** Save as `$TMPDIR/migrate_notebooks.py`, run with `.venv/bin/python $TMPDIR/migrate_notebooks.py` from the repo root. Rules apply in list order to every cell (code and markdown), so notebook-specific rules that mention `noise_covariance` come before the generic rename.

```python
import json
from pathlib import Path


def edit(path, replacements, replace_cells=()):
    nb = json.loads(Path(path).read_text())
    touched = set()
    for index, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"])
        new = src
        for cell_index, body in replace_cells:
            if cell_index == index:
                new = body
        for old, repl in replacements:
            if old in new:
                new = new.replace(old, repl)
        if new != src:
            cell["source"] = new.splitlines(keepends=True)
            if cell["cell_type"] == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
            touched.add(index)
    Path(path).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(path, "edited cells", sorted(touched))


# Generic renames; applied after each notebook's specific rules.
COMMON = [
    ("    DataCovariance,\n", "    Covariance,\n"),
    ("    TranslatedCovariance,\n", ""),
    ("DataCovariance", "Covariance"),
    ("initial_noise_covariance", "initial_covariance"),
    ("covariance_matrix", "covariance"),
    ("noise_covariance", "covariance"),
]

CREATE_BLOCK_CELL_22 = '''from enchilada import transform

spectrum = transform(observed, "frequency")
sigma2 = covariance.noise_variance("A", observed)
sample_rate_hz = observed.sample_rate_hz


class FlatPSD:
    def __init__(self, level):
        self.level = level

    def psd(self, frequencies, channel=None):
        return np.full_like(frequencies, self.level, dtype=float)


# The same white noise expressed as a one-sided PSD on the frequency grid.
# Covariance.from_psd evaluates the model at every bin, including f = 0, and
# stores (Tobs / 2) * S there; noise_psd inverts it exactly.
spectral_noise = Covariance.from_psd(spectrum, FlatPSD(2 * sigma2 / sample_rate_hz))
np.testing.assert_allclose(
    spectral_noise.noise_psd("A", spectrum), 2 * sigma2 / sample_rate_hz
)
np.testing.assert_allclose(spectral_noise.noise_variance("A", spectrum), sigma2)
assert spectral_noise.A.shape == spectrum.channel_data["A"].shape
# Handing a frequency-domain covariance to a time-domain block needs the
# covariance translation layer (design spec section 6); until it lands, keep a
# campaign and its covariance in one domain.
print("PSD construction checks passed")
'''

edit(
    "examples/create_block.ipynb",
    [
        # SineAmplitudeBlock.sample (cell 8): divide by the per-sample variance.
        (
            '        if noise_covariance is None or noise_covariance.data_domain != "time":\n'
            '            raise ValueError("This sampler needs a covariance in the time domain")\n',
            '        if noise_covariance is None or noise_covariance.form != "diagonal":\n'
            '            raise ValueError("This sampler needs one noise variance per time sample")\n',
        ),
        (
            '        precision_basis = noise_covariance.solve({"A": basis})["A"]\n',
            "        precision_basis = basis / noise_covariance.A\n",
        ),
        (
            '        precision_data = noise_covariance.solve(conditional_residual.channel_data)["A"]\n',
            '        precision_data = conditional_residual.channel_data["A"] / noise_covariance.A\n',
        ),
        # cell 26: noise_variance now takes the data it describes.
        (
            '    joint_campaign.noise_covariance.noise_variance("A"),',
            '    joint_campaign.noise_covariance.noise_variance("A", observed),',
        ),
        # cell 6 (markdown): the operator sentence.
        (
            "`C⁺` is implemented by `noise_covariance.solve(...)`; it applies precision on the\n"
            "retained subspace. Using the operator preserves correlations and statistical\n"
            "exclusions, including those carried by a `TranslatedCovariance`. The proper prior",
            "`C⁺` is the per-sample precision `1 / covariance.A`: this block requires one\n"
            "white-noise variance per time sample (`form=\"diagonal\"`). The proper prior",
        ),
    ]
    + COMMON,
    replace_cells=[(22, CREATE_BLOCK_CELL_22)],
)

DT_CELL_12 = '''# Covariance translation between representations is supplied by the
# translation layer (design spec section 6). Until it is installed, the
# covariance stays on the observation grid and this walkthrough translates
# data and block states only.
print("Covariance stays on the", observed.data_domain, "grid:", covariance.A.shape)
'''

edit(
    "examples/domain_translation.ipynb",
    [
        ("    covariance_matrix=noise_covariance,\n", ""),
        ("assert isinstance(restored_result.covariance_matrix, DataCovariance)\n", ""),
        (
            "np.testing.assert_array_equal(\n"
            "    restored_result.covariance_matrix.covariance_matrix,\n"
            "    noise_covariance.covariance_matrix,\n"
            ")\n",
            "",
        ),
        (
            'print("WDM covariance representation:", wdm_result.covariance_matrix.data_domain)\n',
            "",
        ),
        (
            "        assert noise_covariance is not None\n"
            "        assert noise_covariance.data_domain == self.expected_domain\n",
            "        # Covariance translation across domains arrives with the translation\n"
            "        # layer (design spec section 6); this walkthrough runs without one.\n"
            "        assert noise_covariance is None\n",
        ),
        (
            "wheel = Wheel(observed, initial_noise_covariance=noise_covariance, random_seed=3)",
            "wheel = Wheel(observed, random_seed=3)",
        ),
    ]
    + COMMON,
    replace_cells=[(12, DT_CELL_12)],
)

edit(
    "examples/gb_block_eryn.ipynb",
    [
        (
            " If observations and a native spectral covariance are stored in time or WDM, "
            "Wheel restores that native spectral covariance for the GB block.",
            "",
        ),
        ("noise via explicit DataCovariance", "noise via explicit Covariance"),
    ]
    + COMMON,
)

edit(
    "examples/demo.ipynb",
    [("`noise_covariance.noise_variance()`", "`covariance.noise_variance(channel, data)`")]
    + COMMON,
)
```

Then update by hand the markdown cells whose prose describes removed behaviour, keeping each to a sentence or two:
- `create_block.ipynb` cell 21 ("Wheel translates the conditional residual, covariance, ..."): covariance translation is pending the translation layer; this tutorial keeps the campaign in the time domain. Cell 23 ("Covariance in a new domain need not be ..."): same point. Cell 24: `BlockState(covariance=..., parameters=...)` wording is produced by the script; check it reads well.
- `domain_translation.ipynb` cell 11 (`TranslatedCovariance`, `quadratic_form`, `log_determinant`): replace the paragraph with "Covariance translation between representations is the job of a separate translation layer; `transform` on a `Covariance` raises `NotImplementedError` across domains until it is installed, so this walkthrough translates data and block states only." Cell 13 (`active_mask` prose): replace with "`wdm_data.wdm_grid.active_mask` describes the grid's structurally present coefficients; a WDM `Covariance` is zero exactly there."
- `gb_block_eryn.ipynb` cells 0, 5, 7, 13 are handled by the script; read them once for grammar.

- [ ] **Step 2: Check**

```bash
grep -n "DataCovariance\|TranslatedCovariance\|covariance_matrix\|noise_covariance\|\.solve(\|\.apply(\|quadratic_form\|degrees_of_freedom\|active_mask=" examples/*.ipynb || echo CLEAN
.venv/bin/python - <<'EOF'
import json, subprocess, sys
nb = json.load(open("examples/create_block.ipynb"))
code = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
sys.exit(subprocess.run([sys.executable, "-c", code], cwd="examples").returncode)
EOF
.venv/bin/python -m pytest tests/test_examples.py tests/test_block_tutorial.py tests/test_translation_example.py -q
.venv/bin/python -m ruff check examples
```
Expected: CLEAN, exit 0, tests pass (the translation example tests skip without the WDM backend), ruff clean. `domain_translation.ipynb` and `gb_block_eryn.ipynb` need optional dependencies; only their compilation is tested locally.

- [ ] **Step 3: Commit**

```bash
git add examples/create_block.ipynb examples/domain_translation.ipynb examples/gb_block_eryn.ipynb examples/demo.ipynb
git commit -m "Migrate notebooks to Covariance

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: README and CHANGELOG

**Goal:** Spec section 7 docs: the README describes the attribute API, shape rule, derived basis and seam; CHANGELOG 0.3.0 replaces its `DataCovariance`/`TranslatedCovariance` bullets.

**Files:**
- Commit first, unchanged: `README.md` (the user's uncommitted diagram hunk), `pyproject.toml` (its sdist include), `docs/wheel.svg`
- Modify: `README.md`, `CHANGELOG.md`

**Acceptance Criteria:**
- [ ] `grep -n "DataCovariance\|TranslatedCovariance\|covariance_matrix\|noise_covariance\|active_mask\|quadratic_form" README.md CHANGELOG.md` prints nothing except the `wheel.svg` alt text and the CHANGELOG `[0.1.0]`/`[0.2.0]` history sections.
- [ ] `tests/test_docs.py` and `tests/test_packaging_docs.py` pass.
- [ ] The diagram work is its own commit, made before the documentation edits, so the README never references an untracked image.

**Verify:** `.venv/bin/python -m pytest tests/test_docs.py tests/test_packaging_docs.py -q`

**Steps:**

- [ ] **Step 0: Commit the user's diagram work on its own.** `git diff README.md` is a five-line addition (the wheel figure), `git diff pyproject.toml` adds `docs/**/*.svg` to the sdist, and `docs/wheel.svg` is the image. Look at both diffs, then:

```bash
git add README.md pyproject.toml docs/wheel.svg
git commit -m "Add the global-fit wheel diagram to the README

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

If either diff contains anything other than the diagram work, stop and report instead of committing.

- [ ] **Step 1: README edits** (line numbers are from the committed README; the user's uncommitted diagram edit may shift them, so anchor on the quoted text):

Line 21 alt text: "…owns the data, residual, noise covariance, and ledger" → "…owns the data, residual, covariance, and ledger".

Lines 154–160:
```
- `sample(conditional_residual, covariance, current_block_state, *, rng) -> BlockState`:
  sample against the data minus every other block's latest signal and return
  the new physical estimate.

`conditional_residual` is an `L1Data`; `covariance` is a `Covariance` on the same
grid as `conditional_residual`, or `None`; and `current_block_state` is that block's
previous accepted `BlockState`.
```

Table line 190: ``| `covariance` | Optional `Covariance`; `None` means no covariance publication. |``

Lines 236 and 244: keep "covariance"; replace "Optional covariance uses its own grid metadata and the normal conversion" with "An optional covariance is on the block's grid, like its template".

Replace lines 255–335 (from "`L1Data` holds channel arrays…" through "…through covariance operators.") with:

````markdown
`L1Data` holds channel arrays and observation conventions. Supply noise as an
explicit `Covariance` when constructing Wheel:

```python
from enchilada import Covariance, Wheel

noise = Covariance.from_variance(observed, time_sample_variance=0.25)
wheel = Wheel(
    observed_data=observed,
    initial_covariance=noise,
    random_seed=42,
)
```

With `initial_covariance=None`, Wheel starts without a covariance; a block can
publish one. Convert a one-sided PSD model with
`Covariance.from_psd(spectrum, noise_psd_model=model)`, where `spectrum` is
frequency-domain data and the model provides `psd(frequencies)` or
`psd(frequencies, channel)` returning finite positive one-sided densities at
every rfft bin including `f = 0` (floor the model at low frequency).
Signal-only, noise-only, and joint signal/noise models all implement `Block`.
A noise-only block publishes covariance directly:

```python
from enchilada import BlockState

new_block_state = BlockState(
    covariance=covariance,
    parameters={"sigma": sigma},
)
```

Publishing covariance makes that block the current covariance owner. The owner
must return its full covariance on every subsequent call, even when unchanged;
omitting it raises `ValueError`. Other blocks may omit `covariance` to preserve
the current model. Joint signal/noise states publish both in one `BlockState`;
validation and adoption are atomic. `state.with_covariance(covariance)` builds a
new state with that covariance.

A `Covariance` is read through its channels. `noise.A`, `noise.E`, `noise.T`
are each channel's own covariance; `noise.AE`, `noise.ET`, `noise.AT` are the
cross terms, with `noise.EA` the conjugate transpose of `noise.AE`; an absent
cross term reads as zero. When the channels are exactly X, Y, Z the names
A, E, T and their pairs are derived on demand with the orthonormal LDC
combination `A = (Z − X)/√2`, `E = (X − 2Y + Z)/√6`, `T = (X + Y + Z)/√3`, and
the reverse holds for A, E, T data; `noise.in_basis("AET")` returns the whole
derived object. `noise.pair("X1", "Y1")` is the method form for channel names
that do not split cleanly.

Each pair is one array on the grid of the data the covariance accompanies.
With channel shape `S` (`(N,)` time, `(N//2 + 1,)` frequency, `(Nf + 1, Nt)`
WDM), shape `S` means one value per point (`form="diagonal"`) and shape `S + S`
a full matrix between points (`form="full"`). Build one from arrays with
`Covariance.like(data, {"A": var_a, "E": var_e, "AE": cross})`; the form is
inferred from the data. Coefficients use `dt * rfft(x)`; a one-sided PSD becomes
`C = (Tobs / 2) S` at every bin, and `noise.noise_psd("A", spectrum)` inverts it.
`noise.noise_variance("A", data)` gives the per-sample variance from stationary
time covariance or by integrating a frequency covariance.

The object carries no grid; `noise.check_compatible(data)` checks channel
order, shape, real/complex kind and positivity against the data it accompanies,
and Wheel runs it on every adoption. Covariance translation between domains is
a seam in `transform()` that currently raises `NotImplementedError`; a
translation layer plugs in there.
````

Table rows (lines 339 and 344): ``| Constructor | `Wheel(observed_data, initial_covariance=None, *, random_seed=None)` |`` and ``| Check a block | `check_block(block_under_test, observed_data, num_cycles=2, *, initial_block_state, initial_covariance=None, random_seed=0)` |``. Line 346: `wheel.covariance`.

Lines 376–396: `initial_covariance=None,` in the example; prose unchanged otherwise.

Lines 437–451: replace the paragraph and code block from "Translate covariance with the same grid choice." through "…not invariant under a scaled transform." with:

```markdown
Covariance translation between representations is not built in. `transform` on
a `Covariance` needs `reference_data` (the data on its current grid), returns an
independent copy for the same domain and grid, and raises `NotImplementedError`
for any other target until a translation layer implements that branch. Until
then, keep a campaign's covariance on its observation grid and give blocks that
publish covariance the observation domain.
```

Lines 494–497: ``L1Data.block_state(template=None, *, covariance=None, parameters=None)`` and ``L1Data.zero_block_state(*, covariance=None, parameters=None)``.

Line 538: "`Covariance.from_psd`". Line 545–546: "An explicit tuple can choose a different order, as in the A/E example above. Its names must match the dictionary keys and are the channel order a `Covariance` uses." Line 559: "Covariance publications are checked against the observation grid and the positivity rules before any state is committed."

Line 611: "for converting data and block states between representations."

Lines 640–641: delete "; `DataCovariance.active_mask` only selects covariance points for inference" (end the sentence at "windowing."). Line 642: "`Wheel.covariance` is a single slot". Lines 652–654: replace the "No automatic diagonal approximation" bullet with "- **No covariance translation yet.** `transform` raises `NotImplementedError` for a `Covariance` across domains; the translation layer that fills that seam is separate work."

Lines 668–670: "Defensive state and covariance copies still cost memory and work. Internal copies of validated covariance skip revalidation; constructors and new covariance publications retain it."

- [ ] **Step 2: CHANGELOG edits** in the `[0.3.0] — Unreleased` section:

Delete the "Native WDM covariance and exact `TranslatedCovariance` operators…" bullet and the "`DataCovariance` stores coefficient covariance…" bullet from **Added**. In the `Ledger` bullet write "optional `template` and `covariance`". In the `transform` bullet write "converts observations and block states between time, frequency, and WDM representations; covariance translation is a seam that raises `NotImplementedError` until a translation layer is installed." In the "Per-block representation selection" bullet drop "covariance," and "and covariance weights".

Add to **Added**:

```
- `Covariance` stores noise covariance as one array per channel pair and is
  read through channel attributes: `cov.X`, `cov.XY`, `cov.A`, `cov.AE`, with
  reversed pairs as conjugate transposes and absent cross terms as zero. With
  channels exactly X, Y, Z the A, E, T basis is derived on demand through the
  orthonormal LDC combination, and the reverse; `in_basis` returns the whole
  derived object. Arrays follow a shape rule on the data's grid (`form`
  "diagonal" or "full"). `like`, `from_psd`, `from_variance`, `noise_psd`,
  `noise_variance` and `check_compatible` take the `L1Data` the covariance
  accompanies; the object carries no grid.
```

In **Changed**: replace the `BlockState` bullet's second and third sentences with "`BlockState` publishes only `template`, `covariance`, and `parameters`, with `with_covariance` for a replacement covariance." Rename in the `Blocks require only…` bullet to `sample(conditional_residual, covariance, current_block_state, *, rng)`. In the Wheel bullet: "the current `covariance`" and drop "Internal covariance snapshots avoid repeated factorization while new covariance publications retain validation." Replace the "Covariance is explicit:" bullet with:

```
- Covariance is explicit: pass `initial_covariance` to Wheel or publish it from
  a block; `None` starts a campaign without one. `Covariance.from_psd(spectrum,
  model)` needs frequency-domain data and fills every bin including DC with
  `(Tobs / 2) S(f)`; `from_variance(data, variance)` needs time-domain data.
  `L1Data.block_state`/`zero_block_state` take `covariance=`.
```

Delete from **Fixed** the "Covariance log determinants use Cholesky factors…" bullet; rewrite the "Covariance symmetry checks scale to each channel pair…" bullet as "Full-form covariance blocks are checked for Hermitian symmetry per channel with pair-scaled tolerance and stored exactly symmetric." In **Removed** replace the first bullet with:

```
- Noise storage and weighting helpers on observation data, the `DataCovariance`
  stacked point × channel × channel matrix with its `active_mask` and inactive
  DC bin, the `TranslatedCovariance` operator, and the `apply`, `solve`,
  `project`, `quadratic_form`, `log_determinant` and `degrees_of_freedom`
  methods. Covariance is one `Covariance` object managed by Wheel.
```

- [ ] **Step 3: Verify and commit**

```bash
grep -n "DataCovariance\|TranslatedCovariance\|covariance_matrix\|noise_covariance\|active_mask\|quadratic_form" README.md CHANGELOG.md
.venv/bin/python -m pytest tests/test_docs.py tests/test_packaging_docs.py -q
git add README.md CHANGELOG.md
git commit -m "Document the Covariance object and the translation seam

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
Expected grep hits: only CHANGELOG lines below `## [0.2.0]`/`## [0.1.0]`. Tell the user that Step 0 committed their diagram work (README, pyproject.toml, docs/wheel.svg) as its own commit.

---

### Task 8: Final verification

**Goal:** Prove the whole change meets the repository's gates.

**Files:** none new.

**Acceptance Criteria:**
- [ ] `.venv/bin/python -m pytest -q --cov --cov-report=term-missing` passes with coverage ≥ 95 %.
- [ ] ruff check, ruff format --check, mypy clean.
- [ ] `UV_CACHE_DIR="$TMPDIR/uv-cache" uv build --sdist` succeeds and `tests/test_packaging_docs.py` passes from the unpacked sdist (the CI sdist lane).
- [ ] `git status --short` is clean apart from status updates to the plan's `.tasks.json`.

**Verify:**

```bash
.venv/bin/python -m pytest -q --cov --cov-report=term-missing | tail -15
.venv/bin/python -m ruff check src tests examples benchmarks && .venv/bin/python -m ruff format --check src tests examples benchmarks && .venv/bin/python -m mypy
UV_CACHE_DIR="$TMPDIR/uv-cache" uv build --sdist && tar tzf dist/enchilada-0.3.0.tar.gz | grep -c "tests/test_covariance.py"
```

**Steps:**

- [ ] **Step 1:** Run the three commands above. If coverage is below 95 %, add tests for the uncovered lines of `covariance.py` reported by `--cov-report=term-missing` (likely error branches) to `tests/test_covariance.py`, then rerun.
- [ ] **Step 2:** Commit any coverage additions: `git add tests/test_covariance.py && git commit -m "Cover remaining Covariance validation branches" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"`.
- [ ] **Step 3:** Report to the user: the commit list (including the diagram commit from Task 7), the six strict xfails waiting on the translation layer, and the follow-ups that were out of scope (uv pin bump, PR #5 on GitHub, pushing `name_changes`).
