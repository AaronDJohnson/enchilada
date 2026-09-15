# Quality reassessment — 2026-09-14

This assessment covers the current working tree after fixing the review findings.
Version 0.2.0 remains unreleased. `Block` requires `name` and `sample`;
`Wheel.add` and `check_block` require a complete `initial_block_result`.

Overall judgment: **about 8.5–9/10 for an alpha scientific-orchestration library**.
The confirmed code, typing, documentation, packaging, and workflow findings are
resolved. The remaining limits concern hosted validation, optional external
examples, and scale. Scores are engineering judgments, not percentages or a
guarantee about scientific models supplied by blocks.

| Metric | Before / 10 | Now / 10 | Basis |
| --- | ---: | ---: | --- |
| Correctness | 8 | 9 | Conversion validation and precision bugs fixed; covariance normalization checked against independent references. |
| Ease of use | 8.5 | 9 | Uniform conversion/copy behavior, explicit starting results, working frozen-block typing, and executable tutorials. |
| Readability | 8.5 | 8.5 | Descriptive public names and documented ownership; numerical and transaction code still needs careful reading. |
| Reliability | 8.5 | 9 | Atomic adoption, defensive snapshots, reproducible continuation, and new mutation/precision regressions. |
| Testing | 8.5 | 9 | 97.38% coverage, consumer typing, real WDM checks, and isolated minimum/newest-interpreter core checks. |
| Maintainability | 8 | 8.5 | One shared conversion path and a reusable WDM workflow reduce duplicated behavior. |
| Documentation | 8.5 | 9 | Correct covariance-domain contract, explicit copy semantics, restored template, and checked source-archive links. |
| Performance and scalability | 7.5 | 7.5 | Cached signal sums remain useful; full-array/state copies and mission-scale memory are still constraints. |
| Dependency and installation discipline | 8 | 9 | NumPy-only core with a verified 1.26.4 floor; optional WDM revision pinned; built artifacts validated. |
| Readiness for an alpha release | 7 | 8.5 | Local checks pass and publication is correctly gated; the final hosted matrix still needs to run. |

## Resolved findings

1. **Release publication guard.** Publication now requires both a push event and
   a tag ref. Manual dispatches, including dispatches targeting tags, stop after
   artifact validation. The release build also requires the core-minimum and WDM
   integration jobs.
2. **Conversion validation and ownership.** `L1Data.to_time()` and
   `to_frequency()` delegate to `transform()`. Mutated lengths, nonfinite values,
   and imaginary Fourier endpoints are rejected. Both conveniences return
   independent channel arrays even when already in the requested domain.
3. **Frozen-block typing.** `Block.name` is read-only to type checkers. Its runtime
   annotation avoids an inherited property interfering with dataclass creation.
   Consumer and runtime regressions cover structural implementations and explicit
   frozen subclasses, including registration and sampling.
4. **Fourier precision.** Frequency inputs are promoted to at least complex128
   before normalization. Analytic tests cover complex64 underflow and overflow,
   including the previously zeroed covariance quadratic form.
5. **Covariance log determinants.** Minimum-version testing exposed NumPy 1.26
   warnings for valid complex matrices, reproducible outside enchilada. The
   positive-definite covariance calculation now uses Cholesky factors. Analytic
   masked and correlated cases span scales from `1e-240` to `1e240`; the full core
   suite passes with runtime warnings promoted to errors.
6. **Documentation and source archives.** The translation guide distinguishes
   explicit block-domain covariance conversion from default native-domain
   retention. Archives include linked specifications, templates, OpenSpec files,
   and workflows. The missing scientific template is restored, a nonexistent
   feature-selection claim is removed, and historical removed-source links point
   to their recorded commit. A recursive link test runs in the extracted archive.
7. **Core dependency floor.** The declared minimum is NumPy 1.26.4 on Python 3.12.
   A separate job pins it without orbit dependencies; numeric-orbit minimum
   resolution has a separate CI job. The lockfile matches the metadata.
8. **Reproducible optional WDM integration.** CI and release call a shared workflow
   pinned to WDM 0.2.0 at `6f145b4076862ea4786150a54d5a1f28af3c7055`.
   Backend version/API and plotting imports are required, and any skipped test in
   its numerical/example suite fails that gate. WDM remains outside core dependencies.

## Fresh verification

- **Python 3.13.7 / NumPy 2.5.1 with the real WDM source:** 593 passed,
  2 optional Eryn skips, **97.38% branch-inclusive coverage**. This exercises
  numerical translation, covariance, orchestration, and executable examples.
- **Isolated Python 3.12.11 / NumPy 1.26.4 core:** 519 passed, 42 optional
  skips with `-W error::RuntimeWarning`; no runtime warnings. SciPy, h5py,
  lisaorbits, and WDM are absent from that environment.
- **Python 3.14.7 / NumPy 2.5.3 core:** 519 passed, 42 optional skips with
  runtime warnings treated as errors. Source is loaded from this checkout.
- **Installed pinned WDM integration:** the exact workflow selection passes
  135 tests with zero skips, including the executable translation walkthroughs.
  A wheel built from the recorded revision with hatchling 1.30.1 imports from
  site-packages as WDM 0.2.0. Python 3.13.14, NumPy 2.5.1, SciPy 1.18.0, and
  matplotlib 3.11.1 pass the workflow preflight; all 50 installed distributions
  pass dependency compatibility checks.
- **Final release artifacts:** forced PEP 517 wheel and sdist builds pass. The
  extracted source passes all 519 core tests, with 42 optional skips and runtime
  warnings treated as errors. Its recursive README links resolve. A fresh
  installed wheel passes version/`py.typed`, Fourier round trip and invalid-DC
  rejection, covariance quadratic invariance, and Wheel registration/execution
  checks using only NumPy 1.26.4 as a runtime dependency.
- **Static checks:** Ruff lint and formatting pass across 46 files; mypy passes
  all 14 package modules and the consumer regression. Actionlint 1.7.12 accepts
  all three workflows. Lock consistency and whitespace checks pass.

## Remaining limits and release steps

- Run the final revision through hosted Linux/macOS CI before tagging. Local
  macOS checks and workflow linting do not establish a hosted matrix pass.
  This includes fetching the pinned WDM checkout: the remote revision exists,
  but anonymous repository access was not verified after that check was canceled.
- Native GBGPU/Eryn inference remains unverified here. Those are deliberately
  external example dependencies and do not gate the core release. The two skips
  are stated explicitly; no successful native inference run is claimed.
- Mission-scale memory and runtime have not been established. The existing local
  benchmark and its limits are described in [performance.md](performance.md).
  These fixes do not justify increasing the performance score.
- Blocks remain responsible for the validity of their model and sampler.
  Exchange-contract tests cannot establish arbitrary posterior correctness.

No tags, publication, commits, or changes to the user's WDM checkout were made.
Earlier assessments below are historical and concern older snapshots.

---

# Quality reassessment — 2026-09-13

The earlier review addressed the findings listed below in the working tree on
`codex/orchestrator-owned-state`. The scored assessment and historical checks
predate the WDM implementation; current translation evidence follows here.
Version 0.2.0 remains unreleased. The existing GBGPU/Eryn waveform example is
retained; no binary_galactica migration was made. GBGPU, Eryn, and LISA Analysis
Tools are optional example requirements only. They are absent from package
dependencies, extras, the lockfile, and core imports. Their installation and
native integration status do not gate an enchilada release.

## Example compatibility checks — 2026-09-13

The examples now declare their supported input domains explicitly where their
likelihood depends on a particular representation. The conjugate toy requires
native, fully active, independent white time noise; the GB example requires a
native frequency covariance for its narrowband PSD likelihood. Unsupported
translated operators raise a descriptive ValueError instead of failing while
accessing a per-point matrix or mask. Neither example silently discards
cross-point covariance.

- `demo.py` and all eight code cells in `demo.ipynb` execute from both the
  repository root and the examples directory.
- `toy_fit.py` recovers the injected amplitudes and noise scale. Tests verify
  matching draws and residuals when its time-domain blocks run on Fourier or WDM
  observations through Wheel's explicit domain selection.
- `domain_translation.py` and all nine code cells in its notebook execute against
  the actual local WDM backend. Round trips, covariance quadratic forms, complete
  result conversion, and canonical ledger behavior pass their assertions. The
  notebook's plots were rendered and inspected.
- Automated checks now compile every shipped notebook and execute the WDM
  notebook when its optional backend and plotting dependency are available.
  Notebook execution uses ordinary Python cells in sequence; a live Jupyter
  frontend launch was not tested.
- The full suite with local WDM passes **529 tests**, with two optional Eryn skips
  and **97.41%** branch-inclusive coverage. Ruff lint, formatting, and mypy pass.
  Without WDM or Eryn, the focused example suite passes **20 tests** with five
  optional skips.

The GB contract tests substitute only the waveform backend. They cover the new
covariance guard, grid compatibility, and restoration of a native spectral
covariance from time/WDM storage. A native GBGPU installation was attempted in an
isolated temporary environment; the sandbox download failed DNS resolution and
the escalated retry was canceled. The full GB notebook inference run therefore
remains unverified. A separate cached-only Eryn attempt still skipped both
continuation tests because its plotting imports lacked seaborn. The shared
environment and package dependencies were unchanged.

## Translation verification — 2026-09-13

The public `transform` wrapper now supports time, frequency, and the local WDM
backend for L1Data, covariance, and BlockResult. Wheel can select a representation
and dependent division counts for each block. Exact covariance operations preserve
cross-point correlations and native statistical exclusions. See the
[domain translation guide](domain-translation.md).

| Current check | Result |
| --- | --- |
| Python 3.13.7 / NumPy 2.5.1, local WDM source on PYTHONPATH | 517 passed, 2 optional Eryn skips; 97.41% branch-inclusive coverage |
| Same checkout without WDM | 483 passed, 36 optional skips; 95.70% coverage |
| Extracted final source archive with local WDM | 517 passed, 2 skipped |
| Ruff lint / formatting / mypy | Passed; 14 source modules checked by mypy |
| Forced PEP 517 sdist and wheel build | Passed, including new modules, example, tests, guide, and py.typed |
| Installed wheel outside the checkout | Core-only import, Fourier round trip, WDM data/covariance conversion, and selected-domain Wheel smoke passed |

The installed-wheel smoke used a fresh environment for enchilada and reused the
development NumPy/SciPy installations. WDM was loaded directly from the user's
local checkout; no unpublished backend was fetched from PyPI. The native backend
is optional, requires Python >=3.13, and is not currently installed by hosted CI.
Its numerical checks compare all nine domain pairs with independent small dense
references, including masks, cross-channel correlations, tiny covariance scales,
and Fourier endpoints. Regression tests also cover mutation isolation, failed
translation rollback, RNG retries, grid reuse, and precision promotion before WDM
normalization.

The Python 3.12/3.14 environments tried for this new snapshot lacked pytest, so
they did not execute the new suite. The earlier compatibility checks below apply
to the pre-translation snapshot. Hosted CI and minimum-dependency validation remain
outstanding. One previously identified release-workflow issue also remains:
`publish.if` needs to require a push event as well as a tag, so a manual dispatch
on a tag cannot publish. This translation change does not alter that workflow.

## Earlier scored assessment

Scores are engineering judgments out of 10, relative to this package's stated
scope as an alpha orchestration library. They are not certification of scientific
models supplied by blocks.

| Metric | Before | After | Reason |
| --- | ---: | ---: | --- |
| Correctness | 6 | 8.5 | Covariance, orbit isolation, FFT endpoints, and residual aggregation now have targeted regression coverage. |
| Usability | 7 | 8 | Explicit Ledger snapshots and corrected installation instructions remove misleading behavior and setup failures. |
| Readability | 8 | 8.5 | Public names and state ownership are clear; signal aggregation is isolated in a small private module. |
| Testing | 8 | 9 | Core coverage is 98.74%; source artifacts and three Python versions were exercised, with optional gaps stated below. |
| Scalability | 6 | 7.5 | Cached sums and covariance snapshots improve measured cycles substantially; full arrays and sampler state still require copies. |
| Release readiness | 5 | 7.5 | PEP 517 builds and packaged-source tests pass; hosted CI and minimum dependencies still need validation. |

## Resolved findings

- Covariance symmetry is checked against each channel pair's scale. Accepted
  roundoff is symmetrized before Cholesky, and the exact validated covariance is
  stored consistently in Wheel and Ledger. Covariance subclasses cannot bypass
  the constructor's observation-grid check.
- NumericOrbit copies its time and position buffers, validates their shape and
  finiteness, and rejects nonfinite query times before checking bounds. Caller
  mutations can no longer separate reported bounds from interpolation knots.
- Frequency data and signal results reject imaginary DC and even-length Nyquist
  coefficients. Odd-length final bins remain complex. GB noise injection now
  draws a real Nyquist coefficient with the full required variance.
- Ledger is an explicit store with identity equality and defensive access.
  `snapshot()` supplies a stable copied dictionary; it no longer inherits Mapping
  behavior that made even self-equality fail.
- A persistent balanced tree caches accepted signal sums. Candidate changes are
  discarded on failure, and residuals remain anchored to pristine observations.
  Summing contributions before subtraction fixes the reproduced cancellation
  failure; aggregation uses at least float64/complex128 precision. Internal
  covariance snapshots avoid repeated Cholesky while external input is validated.
- CI preserves the environment selected by its install step with `--no-sync`.
  Release builds test the actual source archive before uploading it. The declared
  backend and workflow uv version agree, and builds explicitly exercise PEP 517.
- Notebook setup installs project extras together, then external requirements,
  and launches without resynchronizing. The optional GB requirements constrain
  NumPy below 2.4 because released Eryn 1.2.6 calls `numpy.in1d`, which
  [NumPy 2.4 removed](https://numpy.org/devdocs/release/2.4.0-notes.html#removed-numpy-in1d).
  Core NumPy support is unchanged.

## Earlier executed verification

| Environment or check | Result |
| --- | --- |
| Checkout, Python 3.13.7 / NumPy 2.5.1 | 337 passed, 2 optional Eryn tests skipped; 98.74% branch-inclusive core coverage |
| Final extracted source archive, same environment | 337 passed, 2 skipped |
| Source archive, existing Python 3.12.12 / NumPy 2.5.3 environment | 333 passed, 6 skipped: Eryn, h5py/lisaorbits, and mypy unavailable |
| Source archive, existing Python 3.14.7 / NumPy 2.4.6 environment | 334 passed, 5 skipped: Eryn and h5py/lisaorbits unavailable |
| GB examples with released Eryn 1.2.6 / NumPy 2.3.5 | All 13 example tests passed; waveform backend substituted in these tests |
| Ruff lint and formatting | Passed across source, tests, examples, and benchmark |
| Mypy | Passed for all 10 source files |
| Workflow YAML and shell syntax; lock consistency; whitespace | Passed |
| Forced PEP 517 sdist and wheel build | Passed using uv_build 0.12.1 |
| Installed final wheel outside checkout | Version, py.typed, covariance initialization, sampling, Ledger snapshot, and residual smoke checks passed |

The installed-wheel smoke used a fresh environment for enchilada and reused
NumPy from the existing development environment. The Python 3.12 and 3.14 checks
used existing environments; they are compatibility evidence, not clean
dependency-resolution tests. Minimum-dependency environment setup did not finish
after network restrictions prevented installation, so no minimum-version pass is
claimed.

The GB tests exercised the released Eryn wheel, including restart continuity,
RNG isolation, and likelihood refresh, with a deterministic substitute waveform.
An independently cached development build has the same version string but
different source; its passing results were not used as evidence for the released
requirements. Released Eryn emits deprecation warnings under NumPy 2.3.5 and
fails under NumPy 2.5.1, which motivated the example-only constraint.

## Performance and release limits

At 64 blocks and 32,768 samples in each of three channels, the benchmark improved
from 0.567 to 0.071 seconds per cycle, about 8 times faster. The sum cache trades
additional memory for fewer array additions. Full methodology and workload limits
are recorded in [performance.md](performance.md).

This is suitable for an alpha release candidate after the configured hosted CI
gates pass. Hosted Linux/macOS jobs and minimum-dependency execution have not run
for these working-tree changes. Native GBGPU/lisaanalysistools execution remains
unverified for the optional example; it is not a package release requirement.
Mission-scale memory and throughput also need representative workloads.
Domain translation/WDM is now implemented and verified as described above. No
commit, tag, or package publication was performed.
