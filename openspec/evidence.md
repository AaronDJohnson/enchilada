# OpenSpec baseline evidence and limitations

**Written and checked**: 2026-09-11
**Runtime baseline**: `dd412c3d47d3a209b632f43ee3220215936a31cc`
**Specification status**: Draft retrospective contract; scientific review pending

This record accompanies the [OpenSpec comparison](comparison.md). All 50 original
`FR-NNN` identifiers map one-to-one to named OpenSpec requirements. The existing
[Spec Kit traceability record](../specs/001-package-baseline/traceability.md) retains
its detailed test-name mappings and baseline qualifications.

The specifications describe expected baseline behavior and contributor obligations.
A scenario can describe a meaningful acceptance case without an existing dedicated
test. The suite result below must not be read as 64 independently executed OpenSpec
scenarios or as certification of arbitrary scientific models.

## Evidence by capability

| Capability / mapping | Source and existing tests | Important limits |
| --- | --- | --- |
| [Observation data](specs/observation-data/spec.md), FR-001 through FR-009 | [data.py](../src/enchilada/data.py), [wheel.py](../src/enchilada/wheel.py), [test_data.py](../tests/test_data.py), [test_wheel.py](../tests/test_wheel.py) | Metadata validation does not establish physical channel meaning. Full metadata preservation also relies on source review. The impulse scenario is an independent arithmetic example, not a newly added regression test. |
| [Block orchestration](specs/block-orchestration/spec.md), FR-010 through FR-027 | [block.py](../src/enchilada/block.py), [template.py](https://github.com/AaronDJohnson/enchilada/blob/dd412c3d47d3a209b632f43ee3220215936a31cc/src/enchilada/template.py), [wheel.py](../src/enchilada/wheel.py), [test_wheel.py](../tests/test_wheel.py) | Existing tests cover many constituent behaviors. The complete changing-signal/buffer-reuse case is not a dedicated existing test. Convention rejection and stable names include contributor obligations. |
| [Noise weighting](specs/noise-weighting/spec.md), FR-028 through FR-033 | [data.py](../src/enchilada/data.py), [wheel.py](../src/enchilada/wheel.py), [test_data.py](../tests/test_data.py), [test_wheel.py](../tests/test_wheel.py) | Scalar PSD results use NumPy broadcasting; arbitrary output shapes and channel names do not receive extra semantic validation. The helper does not establish all channels' scientific validity. |
| [Orbit ephemerides](specs/orbit-ephemerides/spec.md), FR-034 through FR-038 | [orbits.py](../src/enchilada/orbits.py), [test_orbits.py](../tests/test_orbits.py) | Tests exercise selected tables and queries, not a general response-error budget. Explicit arm/transfer overrides and every malformed rank are not uniformly guarded. |
| [Contributor validation](specs/contributor-validation/spec.md), FR-039 through FR-045 | [testing.py](../src/enchilada/testing.py), [test_testing.py](../tests/test_testing.py), [test_examples.py](../tests/test_examples.py), [test_docs.py](../tests/test_docs.py), [gb_model.py](../examples/gb_model.py) | The helper mutates its supplied block and checks only the final changed default PSD. Toy recovery is a fixed regression. The external galactic-binary integration is outside the suite. |
| [Package delivery](specs/package-delivery/spec.md), FR-046 through FR-050 | [metadata](../pyproject.toml), [exports](../src/enchilada/__init__.py), [typing tests](../tests/test_typing.py), [CI](../.github/workflows/ci.yml), [release workflow](../.github/workflows/release.yml) | Build/install/platform/release claims describe configured checks. This documentation task did not run the complete CI matrix, build distribution artifacts, or publish a release. |

## Preserve the original acceptance scope

| Original outcome | OpenSpec location | Evidence qualification |
| --- | --- | --- |
| SC-001 | Conditional residual and adoption requirements, FR-019/020 | Existing subtraction tests plus source ordering; combined changing-signal case remains a specified scenario. |
| SC-002 | Ordered cycles and callbacks, FR-018/025 | Existing counter and callback tests exercise parts; the combined three-block/four-cycle fixture is not a separately named test. |
| SC-003 | Array isolation, FR-023 | Diagnostic-copy tests and adoption source review; no dedicated post-publication buffer-reuse test was added. |
| SC-004 | Fourier conversion, FR-007 | Existing even/odd round trips use `rtol=1e-7`, `atol=1e-12`; general physical transform conventions also need independent checks. |
| SC-005 | Spectral scale and variance, FR-032/033 | Fixed seeded spectral-power regression and analytical parity-specific variance tests. The latter's `pytest.approx(rel=1e-12)` also uses a default absolute floor of `1e-12`; the effective variance bound is the larger term. The spectral 5% bound is not a calibrated general stochastic acceptance procedure. |
| SC-006 | Guarded inputs throughout the capabilities | Tested invalid categories and source guards; not an exhaustive malformed-object or scientific-input validation matrix. |
| SC-007 | Orbit interpolation and frame conversion, FR-036/037 | Existing fixtures use both absolute and relative tolerances. At orbital coordinate scales, `rtol=1e-7` can dominate the metre-valued absolute tolerance; a metre-level accuracy guarantee does not follow. |
| SC-008 | Toy regression, FR-043 | Seed 0, 200 cycles, 80 burn-in; means within `max(5*posterior_std,0.05)` and positive spreads. This does not check joint covariance or Monte Carlo precision. |
| SC-009 | Helper and minimal examples, FR-039/041/042 | Existing helper/script/notebook checks; optional notebook dependencies were available in the recorded run. |
| SC-010 | Delivery, FR-046 through FR-050 | Configured compatibility and artifact paths; only the local existing suite was executed for this comparison. |

## Scientific and engineering gaps remain visible

1. **The shared state needs a scientific interpretation.** Correct additive
   residuals do not by themselves establish conditional posterior sampling. The
   [galactic-binary example](../examples/gb_model.py) renders its publication from
   a final walker-mean estimate. A scientist must decide whether a block's output
   represents a sample, an estimate, or another quantity before interpreting a run.
2. **Joint inference needs independent evidence.** SCI-002 / VAL-002 in the
   [scientific worked example](../specs/scientific-software-template/example.md)
   proposes a correlated Gaussian oracle and moment checks. The fixture,
   calibration, and acceptance run remain unimplemented and unverified here.
3. **Copying does not make all shared objects immutable.** Ledger signals and
   returned diagnostics have copy guarantees; original observations, noise, and
   orbits remain shared. Contributors must preserve their scientific meaning.
4. **A failed cycle can contain partial progress.** Earlier updates remain, block
   state is not rolled back, and promoting a noise-ownership warning to an
   exception can interrupt adoption after a signal is stored. Recovery must account
   for the actual failure boundary.
5. **Conventions need scientific review.** Finite values and matching shapes do not
   establish correct units, channel normalization, frames, priors, or likelihoods.
   Gap handling, automatic component-noise composition, general cross-channel
   covariance, parallel scheduling, and persistent campaign checkpoints are absent.
6. **Historical prose may disagree with the source.** The existing traceability
   record identifies return-type and residual-description drift. This comparison
   follows the current executable interfaces and preserves those qualifications.

These gaps apply to both documentation formats. They are not silently resolved by
reformatting the requirements or by accepting the documents as a comparison draft.

## Verification record

OpenSpec CLI **1.13.0**, Node.js **24.4.1**, initialized using
`openspec init --tools none --no-animation`. Strict validation command:

```sh
OPENSPEC_TELEMETRY=0 npm exec --yes --package=@fission-ai/openspec@1.13.0 -- openspec validate --specs --strict --no-interactive --json
```

Result: **6 of 6 capability specs valid; zero findings**. The versioned CLI was
downloaded into a temporary npm cache and invoked from that cached installation.
Requirement mapping confirmed **50 unique original FR identifiers**, exactly one
per OpenSpec requirement, across **64 scenarios**. This is document validation.

Existing package suite, macOS, Python **3.13.7**, NumPy **2.5.1**, pytest **9.1.1**:

```sh
.venv/bin/python -m pytest -q --cov --cov-report=term-missing
```

Result: **219 passed, 1 skipped in 8.48 seconds**, with **100% measured
branch-inclusive coverage** (403 statements, 146 branches). The skip is the
intentional vocabulary check on the test file that contains the retired names.
Numerical-orbit tests and the demonstration notebook ran. No new scientific
acceptance harness was implemented as part of this documentation comparison.
