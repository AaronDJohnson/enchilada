# Compare the Enchilada specifications

This OpenSpec version describes the same package baseline as the existing Spec Kit
specification: commit `dd412c3d47d3a209b632f43ee3220215936a31cc`, metadata version
0.1.0, including the unreleased template-return behavior. It was written on
2026-09-11 with the official OpenSpec CLI 1.13.0. The capability specifications are
draft retrospective contracts, with 50 requirements and 64 scenarios.

Start with [block orchestration](specs/block-orchestration/spec.md), then compare it
with the [Spec Kit package baseline](../specs/001-package-baseline/spec.md) and the
[scientific worked example](../specs/scientific-software-template/example.md).
The [evidence record](evidence.md) distinguishes specified behavior, existing tests,
and scientific checks that remain unverified.

## OpenSpec capability index

| Capability | Main scientific or operational concern | Spec Kit requirements | Requirements / scenarios |
| --- | --- | --- | --- |
| [Observation data](specs/observation-data/spec.md) | Units, sample grid, parity, Fourier scale, finite input, orbit coverage | FR-001 through FR-009 | 9 / 11 |
| [Block orchestration](specs/block-orchestration/spec.md) | Conditional residuals, update order, shared state, copies, partial failure | FR-010 through FR-027 | 18 / 21 |
| [Noise weighting](specs/noise-weighting/spec.md) | Noise ownership, one-sided PSD, DC/Nyquist, variance integration | FR-028 through FR-033 | 6 / 8 |
| [Orbit ephemerides](specs/orbit-ephemerides/spec.md) | Clock/frame consistency, interpolation range, loader axes and defaults | FR-034 through FR-038 | 5 / 8 |
| [Contributor validation](specs/contributor-validation/spec.md) | What conformance and example tests establish; useful diagnostics | FR-039 through FR-045 | 7 / 9 |
| [Package delivery](specs/package-delivery/spec.md) | Core-only use, typing, reproducible build/test paths, release conditions | FR-046 through FR-050 | 5 / 7 |

Each requirement includes its exact `FR-NNN` mapping. Keeping the full package
scope makes the comparison fair; delivery details are in their own capability so a
scientific review can begin with the computations.

## What changes between the two approaches

| Question | Existing Spec Kit documents | This OpenSpec version |
| --- | --- | --- |
| How do I navigate the package? | The original baseline starts with user stories and a package-wide requirement list. | Six capability files let the reader start with one computation or boundary. |
| Where are the equations? | The baseline uses a companion contract; the later scientific template puts equations and numerical contracts in the main document. | Equations and scientifically observable conventions appear with their capabilities and requirements. |
| What describes acceptance? | User-story acceptance scenarios in the baseline; explicit claim and validation cards in the scientific template. | Every named requirement has concrete GIVEN/WHEN/THEN scenarios. Evidence is recorded separately. |
| How does future work differ from current behavior? | The baseline recommends separate feature specs; the scientific example clearly labels its proposed checks. | Current capability specs live in `openspec/specs/`; proposed deltas belong in a separate `openspec/changes/<change>/` directory. |
| How is scientific rigor encouraged? | The scientific template explicitly prompts for assumptions, error budgets, independent references, and evidence status. | Project context and authoring rules in `config.yaml` preserve those concerns; the native scenario format alone does not require a complete scientific argument. |
| What has the tool verified? | The existing documents have structural/reference checks described in their records. | OpenSpec's strict CLI validation passes all six capability specs. This checks document structure, not the correctness of the mathematics or implementation. |

The layout follows the official [OpenSpec writing guide](https://github.com/Fission-AI/OpenSpec/blob/main/docs/writing-specs.md)
and [CLI reference](https://openspec.dev/docs/cli). Scientific equations are treated
as definitions of observable behavior; internal implementation choices can remain
in a future change's design document.

## Compare exactly the same residual calculation

The scientific worked example uses observation 10, initial signals A=2 and B=3,
then updates A to 4 and B to 1. A receives 7, B receives 6, and the final residual
is 5. Later overwriting A's publication buffer must not change the stored signal.

| Scientific example | Matching OpenSpec requirement |
| --- | --- |
| Current conditional residual, part of SCI-001 / VAL-001 | [Supply the current conditional residual](specs/block-orchestration/spec.md#requirement-supply-the-current-conditional-residual), FR-019 |
| Same-cycle adoption, part of SCI-001 / VAL-001 | [Adopt the published signal before subsequent updates](specs/block-orchestration/spec.md#requirement-adopt-the-published-signal-before-subsequent-updates), FR-020 |
| Publication-buffer isolation, part of SCI-001 / VAL-001 | [Isolate adopted and inspected signal arrays](specs/block-orchestration/spec.md#requirement-isolate-adopted-and-inspected-signal-arrays), FR-023 |
| Joint-posterior recovery, SCI-002 / VAL-002 | A proposed scientific integration test; not promoted into this current-behavior baseline |

The formats express the same accounting requirement differently. Neither a
scenario heading nor a claim card says that the associated check has actually run.
The exact combined VAL-001 fixture remains distinct from the existing tests that
exercise parts of its behavior.

One numerical clarification is made explicit here: the existing variance test's
`pytest.approx(rel=1e-12)` also admits its default absolute tolerance of `1e-12`.
The OpenSpec scenario records both terms. This clarifies the test's existing
acceptance rule; the runtime and test code are unchanged.

The correlated-Gaussian fixture in the scientific example would be suitable for a
future OpenSpec change: its proposal would explain the missing joint-moment
evidence, its delta spec would define acceptance, its design would justify the
statistical procedure, and its tasks would implement and calibrate the checks.
This comparison does not create or approve that runtime/test change.

## Implications for scientific authors

OpenSpec gives a compact place to say what a component must do and name the cases
that would expose an error. Our scientific template supplies more explicit prompts
for numerical reasoning and evidence. Both still need help surfacing requirements
an author may not know to ask about.

The [project configuration](config.yaml) therefore asks future authors to inspect
source/tests, distinguish caller obligations from enforced checks, retain equations
and tolerances, and explain engineering rules through their scientific consequences.
These are authoring instructions. They do not implement the guided interview,
automatic engineering assessment, or CI enforcement discussed earlier.

For this package, I would review OpenSpec's orchestration and noise capabilities
for readability, then use the scientific template's error-budget and evidence
sections to assess whether the requirements have enough scientific support.

## Reproduce the format checks

The CLI was initialized with `openspec init --tools none --no-animation`; no agent
integration was installed. The following commands use the pinned CLI from a local
npm cache, downloading it on first use if necessary. Run them at the project root
with Node.js 20.19.0 or newer:

```sh
OPENSPEC_TELEMETRY=0 npm exec --yes --package=@fission-ai/openspec@1.13.0 -- openspec list --specs
OPENSPEC_TELEMETRY=0 npm exec --yes --package=@fission-ai/openspec@1.13.0 -- openspec validate --specs --strict --no-interactive
```

For reading, use Markdown preview or the configured spec-view dashboard. In
spec-view, navigate with its document list; repository-relative Markdown links are
not translated into viewer routes. The OpenSpec comparison, evidence, and six
capabilities are included alongside the original documents.
