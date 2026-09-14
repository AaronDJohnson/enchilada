# Domain Translation Implementation Plan

> **For agentic workers:** Use subagent-driven-development to implement the independent data/grid and covariance tasks while the parent implements translation and orchestration.

**Goal:** Convert enchilada objects between time, frequency, and the local WDM representation through one public function, with configurable dependent WDM division counts.

**Architecture:** `transform(value, target_domain, ...)` owns conversion. `WDMGrid` records validated divisions. Exact translated covariance retains its native representation and applies transforms around native covariance operations. Wheel can select a block's representation at registration and translate its inputs and returned result atomically.

**Tech Stack:** Python, NumPy, optional local `wdm` (Python >=3.13 and SciPy); pytest, Ruff, mypy.

**Spec:** The user request and `docs/domain-translation.md`, updated alongside the implementation.

## Global constraints

- Preserve the existing working tree and version 0.2.0; no commits or publication.
- GBGPU/Eryn remain example-only dependencies.
- Core time/frequency operations remain NumPy-only and support Python >=3.12.
- WDM is imported lazily from the separately installed local package; do not embed a machine-specific path or depend on an unrelated PyPI distribution.
- `num_time_samples = num_frequency_divisions * num_time_divisions`; both division counts are positive even integers. Derive either missing count and reject inconsistent counts. Never pad or truncate implicitly.
- WDM arrays are real `(num_frequency_divisions + 1, num_time_divisions)` arrays. Inactive structural edge entries are zero. Time and frequency conventions remain unchanged.
- Preserve channel order, sampling rate, sample count, epoch, orbit, parameters, sampler state, and metadata. All returned objects are independent snapshots.
- Covariance translation is exact within the native covariance model, including proper complex Fourier channel correlations and masks. No silent diagonal approximation or full-observation dense matrix.

## Task 1: WDM grid and object contracts

Files: `src/enchilada/domains.py`, `data.py`, `block_result.py`; corresponding tests.

- [x] Write failing tests for division derivation, invalid parity/counts, WDM shape, dtype, inactive entries and sample-count consistency.
- [x] Implement frozen keyword-only `WDMGrid(num_frequency_divisions, num_time_divisions)` with `array_shape`, `num_time_samples`, and independent `active_mask`.
- [x] Implement `resolve_wdm_grid(num_time_samples, *, num_frequency_divisions=None, num_time_divisions=None)`.
- [x] Add keyword-only `L1Data.wdm_grid` and native WDM validation. Require the grid only for WDM data; derive an omitted sample count from it.
- [x] Permit 2D BlockResult signals and validate them against the supplied WDM grid in `check_on_grid`.
- [x] Keep existing `to_time`/`to_frequency` convenience methods and delegate their WDM cases to `transform`; do not add `to_wdm` methods to every class.
- [x] Run focused tests, lint, and formatting.

## Task 2: Exact covariance operations and translation

Files: `src/enchilada/covariance.py`, `translated_covariance.py`; covariance tests.

- [x] Write failing tests for native time, frequency and WDM covariance action, precision, active projection, quadratic form, effective degrees of freedom and log-pseudodeterminant.
- [x] Add native WDM covariance shaped `(Nf+1, Nt, channels, channels)` with a matching statistical mask contained in the structural mask.
- [x] Implement `apply`, `solve`, `project`, `quadratic_form`, `log_determinant`, and `degrees_of_freedom` for native covariances.
- [x] Implement immutable `TranslatedCovariance` holding an owned native covariance and target grid. Expose a nonlocal projection instead of pretending exclusions are a pixel mask.
- [x] Use normalized real-coordinate scales: time `1`, frequency `dt*sqrt(N)`, WDM `sqrt(dt)`. For map `T=aQ`, covariance action is `a**2*T*C*T^-1`, precision action `T*C^+*T^-1/a**2`, and log-pseudodeterminant shifts by `2*rank*log(abs(a))`.
- [x] Frequency operations use one real coordinate at DC/even Nyquist and two sqrt(2)-scaled real/imaginary coordinates elsewhere; assume proper complex interior coefficients.
- [x] Preserve masks in the native basis. Translation back to the native grid returns a native covariance copy. Flatten successive translations.
- [x] Implement validated and trusted internal copying without allowing an external subclass to bypass validation.
- [x] Compare operators with independently assembled small dense references, including masks and complex cross-channel covariance; test copy isolation and inverse translation.

## Task 3: One public wrapper and backend boundary

Files: `src/enchilada/_transforms.py`, `translation.py`, `__init__.py`; `tests/test_translation.py`.

- [x] Write failing tests for `transform(data, "frequency")`, `transform(data, "wdm", num_frequency_divisions=8)`, covariance and BlockResult conversion, same-domain copies, and invalid requests.
- [x] Implement `transform_channels` separately from object classes. Use `dt*rfft` for enchilada spectra, divide by `dt` before WDM forward and multiply after inverse. Promote WDM operations to supported float64/complex128; zero only structural inactive output entries.
- [x] Implement `transform(value, target_domain, *, num_frequency_divisions=None, num_time_divisions=None, reference_data=None)` for L1Data, native/translated covariance, and BlockResult. A result with signal needs its source `reference_data` because BlockResult intentionally carries no observation grid.
- [x] Resolve target WDM divisions once. Reuse source WDM grid when no new count is specified; require a count on first conversion into WDM.
- [x] Keep optional imports lazy and provide a useful missing-backend/Python-version error. Preserve unrelated import failures rather than hiding them.
- [x] Export only the wrapper and useful grid/covariance types, with consumer typing tests.
- [x] Verify non-unit sample intervals, every conversion route, WDM regridding, state preservation, and fresh output buffers against the actual local WDM source.

## Task 4: Orchestrator-owned representation selection

Files: `src/enchilada/wheel.py`, `block.py`; `tests/test_wheel_translation.py`.

- [x] Write failing tests for mixed time/frequency/WDM blocks and returned noise models.
- [x] Extend `Wheel.add` with keyword-only `data_domain`, `num_frequency_divisions`, and `num_time_divisions`. An omitted domain keeps the existing handoff contract; explicit selection translates residual, covariance, and previous signal together.
- [x] Store only each block's fixed domain/grid selection. Convert its result back to the canonical observation grid before adoption. Keep block-local numerical parameters and sampler state untouched by transforms.
- [x] Validate translated results and covariance before committing ledger, sum tree, residual, owner and RNG. Failed registration must not retain a domain selection.
- [x] Exercise returned WDM signal shape failures, covariance ownership, rollback and retry, and shared block configurations across campaigns.

## Task 5: Documentation, example and final validation

- [x] Replace the proposed-design document with the implemented API, exact covariance semantics, masks, units, installation instructions and examples.
- [x] Add a small NumPy/WDM-only example showing both division selectors, object conversion and mixed-domain Wheel use.
- [x] Run the existing suite without WDM and the complete suite with the real local backend. Explain any optional skips and retain the coverage gate.
- [x] Run Ruff, formatting, mypy, PEP 517 build and packaged-source/wheel checks. Review numerical conventions and transaction boundaries independently.
- [x] Record remaining limitations accurately; WDM does not imply data-gap support, cross-point independence, or a diagonal likelihood.

Implementation verified: 517 tests pass with the local WDM backend (97.41% coverage);
483 pass without it (95.70% coverage). Packaged-source tests and installed-wheel
smoke pass. Ruff and mypy pass. See `docs/quality-review.md` for evidence and limits.
