# Covariance object design

Date: 2026-11-10
Status: approved in discussion, awaiting written review
Branch: `name_changes` (contains the `BlockState` rename and the CI hardening merge)

## Goal

Replace `DataCovariance` and `TranslatedCovariance` with one small value object,
`Covariance`, that

- stores the noise covariance as one array per channel pair,
- is read through channel attributes: `cov.X`, `cov.Y`, `cov.XY`, `cov.A`,
  `cov.E`, `cov.T`, `cov.AE`, `cov.ET`, `cov.AT`, and so on,
- derives the A, E, T basis from X, Y, Z on demand, and the reverse,
- carries no grid metadata; everything that depends on the domain or grid takes
  the `L1Data` the covariance accompanies.

Domain translation of the covariance is out of scope. The user has a translation
layer that will be added after this change lands. This document fixes only the
seam it plugs into (section 6).

## Decisions made in discussion

| Question | Decision |
|---|---|
| Separate object or plain arrays | Separate object, `Covariance` |
| How the second basis exists | One stored basis; the other derived on demand |
| A, E, T definition | Orthonormal LDC form, see section 3.5 |
| Internal layout | One array per channel pair with a shape rule, not a stacked `(points, ch, ch)` block |
| Grid metadata on the object | None. Only `pairs`, `channel_names`, `form` |
| Seam for the translation layer | The covariance branch of `enchilada.translation.transform` |
| Interim behaviour of the seam | Same domain passes through; different domain raises `NotImplementedError` |

## 1. The `Covariance` object

Module: `src/enchilada/covariance.py`. `@dataclass(frozen=True, eq=False)`, like
`L1Data`; `eq=False` because a dict field has no usable generated hash and
equality of covariances is array comparison the caller does explicitly.

### 1.1 Fields

```python
pairs: dict[tuple[str, str], np.ndarray]   # upper triangle in channel order
channel_names: tuple[str, ...]             # keyword-only
form: str = "diagonal"                     # keyword-only; "diagonal" or "full"
```

`pairs` holds one read-only array per stored channel pair, keyed `(i, j)` with
`channel_names.index(i) <= channel_names.index(j)`. Every `(ch, ch)` entry is
required. Off-diagonal entries are optional; an absent pair means zero
cross-covariance.

`form` says how each array is read. With `S` the shape of one channel's data on
the accompanying grid (`(N,)` time, `(N//2 + 1,)` frequency, `(Nf + 1, Nt)`
WDM):

| `form` | array shape | meaning |
|---|---|---|
| `"diagonal"` | `S` | one value per point, `E[x_i conj(x_i)]` or the cross term at that point |
| `"full"` | `S + S` | covariance between every pair of points |

All pairs on one object share one shape and therefore one form. `form` is a
field rather than inferred because a bare 2-D array could be either a WDM
diagonal or a time-domain full matrix, and the conjugate transpose (section
1.3) and basis derivation (section 1.4) must know which.

A private cache slot (`field(init=False, repr=False, compare=False)`) holds the
derived-basis object once computed.

### 1.2 Construction

```python
Covariance(pairs, *, channel_names, form="diagonal")
Covariance.like(data, pairs)            # channel order and form from data
Covariance.from_psd(data, psd_model)    # frequency-domain data only
Covariance.from_variance(data, variance)  # time-domain data only
```

`like(data, pairs)` takes `channel_names` from `data.channel_names`, accepts
string keys (`"A"`, `"AE"`) resolved by the attribute rule in 1.3 as well as
tuple keys, infers `form` from the array shape against the data's channel
shape, and calls `check_compatible(data)` before returning. An ambiguous string
key raises `ValueError` naming the candidate splits.

`from_psd(data, model)` is today's factory with two changes: `data` must be in
the frequency domain (otherwise `ValueError` pointing at `data.to_frequency()`),
and every bin including DC gets `(Tobs / 2) * S(f_k)`. Channels are independent;
no off-diagonal pairs are stored. `from_variance(data, variance)` requires
time-domain data and produces a constant time diagonal per channel, accepting a
scalar or a per-channel dict as today.

### 1.3 Attribute access

Field and method names always win; `__getattr__` runs only when normal lookup
fails, which is why a channel name equal to any attribute of the class is
rejected at construction.

Resolution of an attribute `name`:

1. `name in channel_names` returns `self.pair(name, name)`.
2. Otherwise every split `name = a + b` with both halves in `channel_names` is a
   candidate. Exactly one candidate returns `self.pair(a, b)`. More than one
   raises `AttributeError` telling the caller to use `pair()`.
3. No candidate and a derived basis exists (1.4): the name is resolved on the
   derived object.
4. Otherwise `AttributeError` listing the valid names: the channel names, the
   pair names, and the derived names when they exist.

`pair(a, b)` returns the stored array when `(a, b)` is the canonical
orientation, the conjugate transpose of the stored array when `(b, a)` is, and
zeros shaped like `pair(a, a)` when neither is stored. Conjugate transpose
means `np.conj(x)` for `form == "diagonal"` and conjugation plus swapping the
two halves of the axes for `form == "full"`.

Stored arrays are returned as the read-only arrays themselves, not copies.
Computed arrays are fresh. Callers never mutate what they get back.

### 1.4 Derived basis

`in_basis(name)` with `name` in `{"XYZ", "AET"}` returns a `Covariance` in that
basis. It is available only when `set(channel_names)` is exactly
`{"X", "Y", "Z"}` or exactly `{"A", "E", "T"}`; any other channel set raises
`ValueError` saying the full triple is required. Asking for the current basis
returns `self`. The derived object is computed once, cached, and has all six
pairs stored, channel order `("A", "E", "T")` or `("X", "Y", "Z")`, and the same
`form`.

For XYZ to AET with matrix `M` (section 3.5), every derived pair is

    C'_ab = sum_{i,j} M[a,i] * M[b,j] * pair(i, j)

over all ordered `(i, j)`, using `pair()` so that reversed orientations and
absent pairs are handled. The sum is elementwise in the point indices, so the
same expression serves both forms. AET to XYZ uses the transpose, which is the
inverse because `M` is orthogonal. In the diagonal form the derived diagonals
take `.real`; in the full form the derived diagonal blocks stay Hermitian.

### 1.5 Methods kept

| Method | Change |
|---|---|
| `noise_psd(channel, data)` | needs frequency-domain `data` for `Tobs`; returns `2 * C_kk / Tobs` at every bin, no infinities |
| `noise_variance(channel, data)` | time data: the constant diagonal, `ValueError` if not constant; frequency data: trapezoid integral of the PSD over all bins with half weight at DC and even-length Nyquist, so white noise reads back as exactly `sigma**2` |
| `copy()` | new object with independent read-only copies; skips validation through a private fast path, as today |
| `check_compatible(data)` | the domain-dependent validation, section 4.2 |

### 1.6 Removed

`active_mask` and the inactive DC bin; `apply`, `solve`, `project`,
`quadratic_form`, `log_determinant`, `degrees_of_freedom`; the module
`translated_covariance.py` with `TranslatedCovariance`, `copy_covariance` and
`translate_covariance`. No example block calls any of them. A block that needs a
whitened inner product writes it over the pair arrays; a free function can be
added when one is needed.

## 2. Wheel, `BlockState`, and the block contract

### 2.1 Names

| Today | After |
|---|---|
| `BlockState.covariance_matrix` | `BlockState.covariance: Covariance \| None` |
| `BlockState.with_covariance_matrix(...)` | `BlockState.with_covariance(...)` |
| `Wheel(observed_data, initial_noise_covariance=None, *, random_seed=None)` | `Wheel(observed_data, initial_covariance=None, *, random_seed=None)` |
| `Wheel.noise_covariance` | `Wheel.covariance` |
| `Block.sample(conditional_residual, noise_covariance, current_block_state, *, rng)` | `Block.sample(conditional_residual, covariance, current_block_state, *, rng)` |
| `check_block(..., initial_noise_covariance=...)` | `check_block(..., initial_covariance=...)` |
| `L1Data.block_state(..., covariance_matrix=...)` and `zero_block_state` | keyword `covariance=` |

`NoiseOverwrittenWarning` keeps its name and meaning.

### 2.2 Behaviour

Unchanged in structure. On construction the Wheel checks `isinstance(...,
Covariance)`, calls `check_compatible(observed_data)`, and stores a copy. The
covariance a block receives is on the block's selected grid, obtained through
the `transform` seam (section 6) with the observed data as `reference_data`. A covariance a block returns is translated
back to the observation grid through the same seam when the block's domain
differs, then validated with `check_compatible(observed_data)` and stored as a
copy. The owner rule is unchanged: one block owns the covariance, must return it
every turn, and a second publisher triggers `NoiseOverwrittenWarning`. All
validation happens before the atomic commit, so a bad return leaves the Wheel
unchanged. `Wheel.covariance` returns a copy. The Ledger snapshots `BlockState`
whole.

### 2.3 What a block sees

`covariance.A`, `covariance.AE`, `covariance.noise_psd("A",
conditional_residual)`, and the derived basis when the triple is present. A
block that needs uncorrelated channels checks that each off-diagonal pair is
absent or zero; this replaces the hand-rolled stacked-matrix check in
`examples/gb_model.py`.

## 3. Conventions

### 3.1 Frequency coefficients

Coefficients are `dt * rfft(x)`. Interior coefficients are proper complex with
zero pseudocovariance. DC and even-length Nyquist coefficients are real.

### 3.2 PSD to covariance

`C_kk = (Tobs / 2) * S(f_k)` at every bin including `f_0 = 0` and Nyquist,
where `S` is the one-sided density and `Tobs = N * dt`. This is the exact
coefficient variance at the endpoints as well as the interior; the endpoint
factor some texts show belongs to the integral of the one-sided density, not to
the coefficient variance.

### 3.3 PSD read-back

`S(f_k) = 2 * C_kk / Tobs` at every bin.

### 3.4 Variance read-back from a frequency covariance

`sigma**2 = df * sum_k w_k * S(f_k)`, `df = 1 / Tobs`, `w_k = 1/2` at DC and at
Nyquist when `N` is even, `1` elsewhere.

### 3.5 A, E, T

```
[A]   [ -1/sqrt(2)     0        1/sqrt(2) ] [X]
[E] = [  1/sqrt(6)  -2/sqrt(6)  1/sqrt(6) ] [Y]
[T]   [  1/sqrt(3)   1/sqrt(3)  1/sqrt(3) ] [Z]
```

Orthonormal, so the inverse is the transpose, and white equal noise in X, Y, Z
is white and equal in A, E, T.

## 4. Validation and errors

### 4.1 At construction, needing no grid

- `channel_names`: non-empty tuple of unique non-empty strings, none equal to an
  attribute of the class.
- `pairs`: a dict whose keys are 2-tuples of known channel names. Unknown names
  raise `ValueError` listing the channels. Non-tuple keys raise `TypeError`.
  Keys are normalised to canonical orientation; both orientations of one pair
  raise `ValueError`. Every `(ch, ch)` must be present.
- Values: `np.ndarray` of floating or complex floating dtype (`TypeError`
  otherwise), all finite, all of one shape (`ValueError`). Diagonal form: each
  channel's own pair is real and nonnegative; full form: each channel's own
  block is Hermitian (complex allowed) with a real nonnegative point-diagonal,
  stored exactly Hermitian. Stored arrays are independent read-only copies.
- `form` is `"diagonal"` or `"full"`. For `"full"` the shape must be `S + S`,
  each diagonal pair must be Hermitian under the axis swap within roundoff and
  is stored exactly symmetrised, as today.
- Positivity: at each point, the channels with a nonzero diagonal form a
  positive-definite matrix (for `"full"`, over the assembled block matrix of
  those entries); channels with a zero diagonal at a point have zero in every
  pair involving them at that point. Zero diagonals are allowed only so that
  WDM structural edge pixels can be represented; `check_compatible` confirms
  they occur nowhere else.

### 4.2 `check_compatible(data)`

- `data.channel_names == channel_names` in order.
- Array shape equals the data's channel shape (`form == "diagonal"`) or that
  shape doubled (`form == "full"`).
- Time and WDM: every array real dtype. Frequency: off-diagonal pairs, and in
  the full form each channel's own block, may be complex; entries that pair two
  real coefficients are real (the DC entry, and the Nyquist entry when `N` is
  even, of a diagonal-form pair; the DC/Nyquist corner entries of a full-form
  block). Other entries in the DC or Nyquist row of a full-form block pair a
  real coefficient with a complex one and may be complex.
- Time and frequency: diagonals positive everywhere. WDM: diagonals zero exactly
  at `data.wdm_grid`'s structurally inactive pixels and positive elsewhere.

All failures are `ValueError` naming the pair and the rule.

### 4.3 Error types

`TypeError` for wrong kinds of objects, `ValueError` for structural and
numerical rules, `AttributeError` for unknown attribute names,
`NotImplementedError` from the translation seam (section 6).

## 5. Testing

- `tests/test_covariance.py` is rewritten: construction rules of 4.1 one by
  one; attribute access for both forms including the conjugate transpose of
  reversed pairs; multi-letter channel names through `pair()` and the ambiguity
  error; derived basis against an explicit matrix product for both forms, the
  AET to XYZ round trip, white equal noise staying white, the triple
  requirement, and caching; `like()` with string keys; `from_psd` and
  `from_variance` numerics including the DC convention; `noise_psd` and
  `noise_variance` round trips; `check_compatible` for each rule of 4.2;
  `copy()` independence and read-only arrays; the attribute-collision rule.
- Wheel, block-state, ledger, typing, testing-helper and example-contract tests
  take the renames of 2.1. The owner rule and rollback tests are unchanged in
  substance.
- `tests/test_covariance_translation.py` is deleted with the module it tests.
- The four tests that exercise covariance translation through the Wheel
  (`test_translation_example.py::test_translation_example_preserves_data_and_noise_weight_across_domains`,
  `test_wheel_translation.py::test_wdm_noise_publication_is_usable_in_every_domain`,
  and the two tests in `test_translation_review.py`) are rewritten to the new
  API and marked `pytest.mark.xfail(raises=NotImplementedError, strict=True)`.
  When the translation layer lands they fail as unexpected passes and the
  marker is removed.
- `examples/gb_model.py` uses the attribute API; `tests/test_examples.py` keeps
  running every script.
- Coverage stays at or above the configured 95 percent.

## 6. Translation seam

The seam is the covariance case of the existing function

```python
transform(
    value,
    target_domain,
    *,
    num_frequency_divisions=None,
    num_time_divisions=None,
    reference_data=None,
)
```

in `enchilada.translation`. For a `Covariance` value `reference_data` becomes
required, since the object has no domain of its own; it is the `L1Data` on the
covariance's current grid, and `ValueError` is raised when it is missing. The
Wheel already calls `transform` for the residual, the block state and the
covariance; the one change in `wheel.py` is that the covariance calls pass
`reference_data` (the observed data on the way out to a block, the block's
residual on the way back). Nothing further in `wheel.py` changes when the layer
arrives.

Interim behaviour in this change:

- `target_domain` equal to `reference_data.data_domain` (and, for WDM, the same
  division counts) returns `value` unchanged.
- Any other target raises
  `NotImplementedError("covariance translation between domains is not implemented; see docs/superpowers/specs/2026-11-10-covariance-object-design.md section 6")`.

Contract the layer must meet, which is what the deleted tests used to prove:

1. Same-domain, same-grid translation is the identity.
2. For `form == "full"` the result is exact: `C' = T C T^H` for the linear map
   `T` between the two representations, using the real-coordinate
   normalisation `sqrt(2) * Re`, `sqrt(2) * Im` for frequency interiors and
   `Re` at the endpoints.
3. The whitened quadratic form of any residual is invariant under translation
   of a `form == "full"` covariance together with that residual.
4. A round trip to another domain and back returns the original arrays on the
   original grid, including odd `N` and tiny grids.
5. WDM structurally inactive pixels are zero rows and columns in the result and
   are never used as data.
6. For `form == "diagonal"` the layer documents whether it returns the diagonal
   of the exact result or promotes to `form == "full"`; either is acceptable,
   and the choice is the layer's to make.

## 7. Migration and deletion

Rewritten: `src/enchilada/covariance.py`.

Deleted: `src/enchilada/translated_covariance.py`,
`tests/test_covariance_translation.py`.

Edited: `wheel.py`, `block.py`, `block_state.py`, `translation.py`, `data.py`,
`testing.py`, `__init__.py` (export `Covariance`, drop `DataCovariance` and
`TranslatedCovariance`).

Callers: `examples/toy_fit.py`, `examples/gb_model.py`,
`examples/domain_translation.py` (the covariance part of the demo is removed
with a note pointing at section 6), the four notebooks (`demo`, `create_block`,
`domain_translation`, `gb_block_eryn`; code cells edited, outputs of edited
cells cleared), `benchmarks/benchmark_wheel.py`, the twenty test files that
reference the old names.

Docs: README covariance section rewritten around the attribute API, the shape
rule, the derived basis and the seam. CHANGELOG 0.3.0 is unreleased, so its
existing `DataCovariance` and `TranslatedCovariance` bullets are replaced by
one `Changed` entry for `Covariance` and the renames in 2.1 and one `Removed`
entry for the deletions in 1.6, with a note that covariance translation is
pending. Version stays 0.3.0.

## Non-goals

- Converting the data itself between the XYZ and AET bases.
- Any covariance translation mathematics.
- Likelihood helpers on the object.
