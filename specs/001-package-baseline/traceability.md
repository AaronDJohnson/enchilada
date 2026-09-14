# Baseline Evidence and Traceability

**Specification**: [enchilada Package Baseline](spec.md)
**Baseline commit**: `dd412c3d47d3a209b632f43ee3220215936a31cc`
**Reviewed**: 2026-09-08

The specification describes the current source tree, including changes newer than the `0.1.0` release. Executable source and tests determine current behavior when historical prose disagrees. The [contract reference](contracts/package-contract.md) records public details; this file records how those claims can be checked.

## Evidence Categories

- **Automated** means an existing test exercises the stated area. A test class may cover only part of a grouped requirement; qualifications below identify the remaining limits.
- **Source review** means the behavior is visible in the implementation or checked-in configuration, without claiming a dedicated automated regression test.
- **Manual integration** means the example or delivery action requires a separate environment or execution and was not performed for this specification.

## Requirement Coverage

Test references below use class or function names within the linked file. Every functional requirement appears in the table; this is a coverage map for the specification, not a claim of exhaustive test coverage.

| Requirements | User stories | Source and existing evidence |
| --- | --- | --- |
| FR-001, FR-002, FR-003, FR-004 | 2 | [data.py](../../src/enchilada/data.py); automated: [test_data.py](../../tests/test_data.py) `TestPostInitValidation`, `TestRemainingValidationBranches`, `TestNSamplesDerivation`, `TestDtypeAndTypeContract`. |
| FR-005 | 2, 3, 4 | [data.py](../../src/enchilada/data.py), [wheel.py](../../src/enchilada/wheel.py); automated portions: `TestDomainTransforms`, [test_wheel.py](../../tests/test_wheel.py) `TestNoiseViaResidual`; preservation of all metadata and orbit identity is also source-reviewed. |
| FR-006 | 2 | [test_data.py](../../tests/test_data.py) `TestAliases`, `TestPsdGridAndAliases`; alias-copy behavior in [data.py](../../src/enchilada/data.py). |
| FR-007 | 2 | [test_data.py](../../tests/test_data.py) `TestDomainTransforms`, including parity, no-op identity, and normalization fixtures. |
| FR-008 | 4 | [test_data.py](../../tests/test_data.py) `TestOrbitSpanCheck`, including a table ending exactly at the last sample and several sample rates. |
| FR-009 | 2 | [test_wheel.py](../../tests/test_wheel.py) `TestObservedDataIsChecked`. |
| FR-010, FR-011 | 1, 6 | [block.py](../../src/enchilada/block.py), [wheel.py](../../src/enchilada/wheel.py); automated: `TestAtomicRegistration`, `TestBoundaryGuards`, `TestDocumentedContracts`; stable names after registration are a caller responsibility, not an enforced rename API. |
| FR-012, FR-013 | 1, 5 | [test_wheel.py](../../tests/test_wheel.py) `TestAtomicRegistration.test_failed_start_leaves_wheel_untouched`, `TestDocumentedContracts.test_start_is_handed_data_minus_already_registered_blocks`; general initialization-exception behavior also follows from source ordering. |
| FR-014, FR-015, FR-016 | 1, 6 | [template.py](https://github.com/AaronDJohnson/enchilada/blob/dd412c3d47d3a209b632f43ee3220215936a31cc/src/enchilada/template.py); automated: [test_data.py](../../tests/test_data.py) `TestTemplateFactories`, `TestTemplateClass`; [test_wheel.py](../../tests/test_wheel.py) `TestReturnedTemplateValidation`, `TestBoundaryGuards`, `TestDocumentedContracts`. |
| FR-017 | 1, 2, 6 | Contributor contract in [block.py](../../src/enchilada/block.py), private state exercised by `TestLedger.test_blocks_keep_their_own_state`; convention guards demonstrated in [gb_model.py](../../examples/gb_model.py), whose external stack is not run by CI. |
| FR-018, FR-019, FR-020 | 1 | [test_wheel.py](../../tests/test_wheel.py) `TestLedger`, `TestNoAddBack`, `TestNoiseViaResidual`; changing-signal adoption within one cycle also follows from [wheel.py](../../src/enchilada/wheel.py). |
| FR-021 | 1, 5 | [test_wheel.py](../../tests/test_wheel.py) `TestLedger` covers rejected counts, integer scalars, and no-block operation; zero-cycle callback behavior is source-reviewed. |
| FR-022, FR-023, FR-024 | 5 | [test_wheel.py](../../tests/test_wheel.py) `TestLedger`, `TestDocumentedContracts`, `TestBoundaryGuards.test_wider_model_promotes_rather_than_raising`; isolation from later edits to a returned template buffer is source-reviewed in ledger adoption. |
| FR-025 | 5 | [test_wheel.py](../../tests/test_wheel.py) `TestOnCycle`; per-invocation index reset and no-block callback behavior are source-reviewed. |
| FR-026, FR-027 | 5 | Return rejection exercised in [test_wheel.py](../../tests/test_wheel.py) `TestReturnedTemplateValidation`, `TestBoundaryGuards`; source ordering establishes partial-cycle state retention, exception propagation, and skipped later updates/callbacks. Whole-cycle rollback is not implemented. |
| FR-028, FR-029, FR-030 | 3 | [test_wheel.py](../../tests/test_wheel.py) `TestNoiseViaResidual`, `TestNoiseOwnership`, `TestBoundaryGuards.test_a_signal_block_does_not_disturb_the_noise_model`. |
| FR-031, FR-032, FR-033 | 3 | [test_data.py](../../tests/test_data.py) `TestNoiseGrids`, `TestPsdGridAndAliases`, `TestNoisePsdSanity`, `TestNoiseVariance`, and `TestDomainTransforms.test_transform_convention_matches_noise_psd_normalization`; the one-sample case is source-reviewed. |
| FR-034, FR-035 | 4 | [orbits.py](../../src/enchilada/orbits.py), [block.py](../../src/enchilada/block.py); automated: [test_orbits.py](../../tests/test_orbits.py) `TestConstruction.test_satisfies_orbit_protocol`, `TestInterpolation`, `TestFrames`; shared-object immutability is a contributor responsibility. |
| FR-036, FR-037, FR-038 | 4 | [test_orbits.py](../../tests/test_orbits.py) `TestConstruction`, `TestInterpolation`, `TestFrames`, `TestLoaders`, `TestFromLisaorbitsGuard`, `test_from_hdf5_preserves_a_gps_scale_epoch`. |
| FR-039, FR-040, FR-041 | 6 | [testing.py](../../src/enchilada/testing.py); automated: [test_testing.py](../../tests/test_testing.py) `TestCheckBlock`, `TestCheckBlockStrictness`, `TestEchoBlock`; final-model-only noise checking and real block-state mutation are source-reviewed limitations. |
| FR-042, FR-043 | 7 | [test_examples.py](../../tests/test_examples.py) `test_demo_runs_clean`, `test_notebook_code_cells_execute`, `test_toy_fit_converges_to_truth`; [test_docs.py](../../tests/test_docs.py) checks README examples and keywords. |
| FR-044 | 7 | Source review/manual integration: [galactic-binary notebook](../../examples/gb_block_eryn.ipynb), [gb_model.py](../../examples/gb_model.py), [external requirements](../../examples/requirements-gb.txt). Its scientific recovery and upstream platform availability were not verified in this run. |
| FR-045 | 2, 4, 5, 6 | Error-message assertions across [test_data.py](../../tests/test_data.py), [test_wheel.py](../../tests/test_wheel.py), [test_orbits.py](../../tests/test_orbits.py), and [test_testing.py](../../tests/test_testing.py); [test_typing.py](../../tests/test_typing.py) `test_runtime_attribute_hints_still_work`. |
| FR-046 | 8 | [pyproject.toml](../../pyproject.toml), lazy imports in [orbits.py](../../src/enchilada/orbits.py), and the core-only job in [CI](../../.github/workflows/ci.yml). Source/configuration review; no new isolated core-only installation was created for the spec. |
| FR-047 | 8 | [exports](../../src/enchilada/__init__.py), [typing marker](../../src/enchilada/py.typed), [test_typing.py](../../tests/test_typing.py); installed-artifact checks configured in [CI](../../.github/workflows/ci.yml). |
| FR-048 | 8 | Source/configuration review: `source-include` in [pyproject.toml](../../pyproject.toml) and the independent sdist job in [CI](../../.github/workflows/ci.yml); no build or unpacked-sdist run performed for this documentation task. |
| FR-049 | 8 | [CI](../../.github/workflows/ci.yml), [pyproject.toml](../../pyproject.toml); the local suite and coverage gate were executed, while cross-platform, lower-bound resolution, and artifact jobs are configuration evidence only. |
| FR-050 | 8 | Source/configuration review: [release.yml](../../.github/workflows/release.yml), including dependencies between jobs, tag/version validation, artifact smoke check, and publication condition. No publication was attempted. |

## Outcome Verification

| Outcome | Acceptance evidence and precise limits |
| --- | --- |
| SC-001 | `TestLedger`, `TestNoAddBack`, and initialization residual tests compare actual arrays with direct subtraction. Default `assert_allclose` comparisons use `rtol=1e-7`, `atol=0`; changing-signal order is also visible in the update loop. |
| SC-002 | Existing counter and callback tests independently cover update counts and completed-cycle notifications. The combined three-block/four-cycle scenario is a specification acceptance case, not a separately named existing test. |
| SC-003 | Existing tests compare original and copied arrays after mutations; copying on adoption is visible in source. A dedicated post-return buffer-reuse regression test is not present. |
| SC-004 | `TestDomainTransforms.test_round_trip_is_exact`: `N=1024,1025`, `rtol=1e-7` (default), `atol=1e-12`; metadata retention is partly tested and otherwise source-reviewed. |
| SC-005 | `TestNoiseVariance.test_independent_of_the_parity_of_n`: `N=2048,2049`, relative tolerance `1e-12`. Fourier normalization: seed 1, 40 realizations, `N=8192`, `fs=0.2`, `sigma=0.7`, mean interior power ratio within 0.05 of 1. |
| SC-006 | Invalid-input categories are parameterized in the data, Wheel, template, orbit-loader, and conformance tests. These do not assert exhaustive validation of arbitrary malformed Python objects. |
| SC-007 | Interpolation fixture uses a 200-node, 30-day circular table and three between-node queries; X/Y comparisons use `rtol=1e-7`, `atol=1` metre. Frame-conversion comparisons use `rtol=1e-7`, `atol=1e-3` metre. Separate tests cover both endpoints and queries before/after the table. |
| SC-008 | `test_toy_fit_converges_to_truth`: slow amplitude 3, fast amplitude 2, noise standard deviation 0.5; seed 0, 200 cycles, 80 burn-in; each mean within `max(5 * posterior_std, 0.05)` of truth. This is a deterministic example regression, not a guarantee for third-party samplers. |
| SC-009 | Minimal script, notebook code cells, and conforming helper tests; the notebook's orbit cells require SciPy and skip when unavailable. |
| SC-010 | Checked-in CI and release validation paths, plus public-surface/type tests. Building, installing a wheel, running from an sdist, and the full platform matrix require their configured jobs; the local test result below does not certify them. |

## Baseline Qualifications and Follow-up Boundaries

These observations are recorded to prevent a later plan from assuming guarantees the current package does not provide. They are not approved feature work or changes to the runtime.

1. **Documentation migration drift.** `Block.update` has the correct `Template` annotation but its return prose still names `L1Data`; `Wheel` has a paragraph incorrectly saying a block's own template is removed from what it sees. Some toy-fit method annotations, the notebook's suggested noise extension, and older changelog prose still use the previous return model. The README also mentions a dropped-noise error removed by the template contract. Use the current signatures, implementation, and tests, as captured in this spec.
2. **Shared objects remain mutable.** Observed arrays, attached models, and orbit references are not deep copies. Caller and contributor ownership rules are necessary. Changing a registered block's name or mutating campaign structure inside a callback is unsupported.
3. **Conformance is deliberately narrow.** The helper uses the caller's actual block instance and checks only the final changed noise object's default spectrum. It does not test earlier noise publications, every channel, sampler recovery, or arbitrary external-process behavior.
4. **Failure atomicity is limited.** Registration validation occurs before committing membership, but adoption itself is not transactional. Escalating `NoiseOverwrittenWarning` to an exception can leave a new signal entry recorded before the noise update completes. Whole-cycle failures also retain earlier successful updates. These limitations must be considered before planning stronger recovery guarantees.
5. **Scientific declarations are not fully validated.** Channel normalization and physical observable semantics are campaign agreements. Direct frequency-domain data is not checked for real-valued DC/Nyquist coefficients. Orbit constructors do not normalize all malformed-shape errors or validate all explicit arm/transfer overrides.
6. **Capability boundaries remain explicit.** No gap masks/window convention, combined noise components, general cross-channel covariance interface, lifecycle finalizer, block removal, parallel execution, or persistent campaign checkpoint is supplied. These need separate feature specifications if desired.
7. **Delivery evidence has an execution boundary.** The external LISA example is outside CI. The recorded upstream wheel matrix and historical orchestration timings are not current benchmark or installation results. The release workflow also lacks main CI's separate unpacked-sdist job.
8. **Test coverage is not semantic completeness.** The baseline suite reaches 100% branch-inclusive coverage, but several behavioral combinations above are supported only by source review. New work affecting those boundaries should choose independent acceptance checks appropriate to its scope.

## Validation Record

On 2026-09-08, in the existing macOS/Python 3.13.7 environment:

```sh
.venv/bin/python -m pytest -q --cov --cov-report=term-missing
```

Result: **219 passed, 1 skipped**, with **100% measured branch-inclusive coverage** (403 statements, 146 branches). The skip is the intentional retired-vocabulary check on the test file that itself enumerates the retired names. Numerical-orbit tests and the demonstration notebook ran in this environment. The galactic-binary integration notebook is not part of that suite.

After adding the README entry, `.venv/bin/python -m pytest -q tests/test_docs.py` passed with **33 passed, 1 intentionally skipped**.

Specification validation confirmed the mandatory section order, eight user stories with independent tests and acceptance scenarios, 50 unique functional requirements, 10 unique outcomes, complete ID mapping, 94 resolving local Markdown links, 61 resolving named test references, and a valid active feature path. No unfilled template placeholders were found. Source review supplied the semantic checks recorded in the [quality checklist](checklists/requirements.md).
