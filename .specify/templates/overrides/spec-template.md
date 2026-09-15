# Scientific Software Specification: [Capability]

**Specification ID / revision**: [identifier / revision]
**Created / updated**: [dates]
**Status**: Draft
**Scope type**: [new requirement / retrospective contract / proposed validation]
**Source baseline**: [commit and any uncommitted changes]
**Scientific reviewer**: [role; pending unless review is recorded]

<!-- Copy this template into specs/<feature>/spec.md. Replace placeholders and
remove authoring comments. Keep only applicable sections and validation profiles.
Do not invent assumptions, tolerances, references, results, or approvals.
Record unresolved scientific choices as OPEN and claims as unverified until
the stated acceptance evidence exists. A small capability can have one equation,
one claim, one independent check, and a short evidence record. -->

## 1. Scientific purpose and scope

[Define the computed quantity, its intended use, supported physical or numerical
regime, and exclusions. Separate current behavior from proposed requirements.
Give a small representative calculation.]

## 2. Mathematical contract

| Symbol / input | Meaning and units | Shape, representation, and convention |
| --- | --- | --- |
| [quantity] | [definition] | [coordinates, normalization, ordering] |

[State equations, assumptions, boundary conditions, and reference sources.
Distinguish the exact target from numerical or statistical approximations.
Specify valid inputs and behavior outside the supported domain.]

## 3. Algorithm and state contract

[Describe steps that determine the scientific result, including update order,
state ownership, initialization, continuation, randomness, and invariants.
State how failures affect accepted state. Distinguish scientific constraints
from implementation choices. For inference, identify the target distribution
and explain why the specified transitions preserve it.]

## 4. Numerical contract and error budget

[Specify precision, conditioning, normalization, error sources, and how errors
propagate to the claimed quantity. Justify tolerances from the scientific use or
reference uncertainty. Address cancellation, singularities, endpoints, and
convergence where relevant. Do not select tolerances by observing a failing run.]

## 5. Scientific requirements

### SCI-001 — [Falsifiable claim]

- **Requirement**: [quantity or invariant that MUST hold in the stated regime]
- **Scientific consequence**: [what a violation changes]
- **Assumptions and dependencies**: [conditions and related claims]
- **Acceptance**: VAL-001.
- **Evidence status**: unverified.
- **Evidence references and limits**: [recorded evidence or explicit gap]

## 6. Validation and acceptance

### VAL-001 — [Independent check]

- **Claims and assessment type**: [claim IDs; verification / physical validation]
- **Fixture**: [complete inputs, units, shapes, precision, parameters, seeds]
- **Independent reference**: [analytical result, independent implementation, or
  measurements; explain independence and reference uncertainty]
- **Metric and acceptance rule**: [measurable threshold and its justification]
- **Execution**: [command/test identifier, or planned if no executable exists]
- **Failure sensitivity**: [plausible scientific errors this check detects]
- **Coverage and limits**: [regime checked and conclusions it cannot support]

[For stochastic claims, specify initialization, independent replicates, burn-in,
dependence-aware Monte Carlo error, relevant joint statistics, precision targets,
false rejection, and sensitivity to meaningful errors. Fix the procedure before
acceptance runs. Insufficient precision is inconclusive, not a pass.]

[For deterministic approximations, specify convergence studies or error bounds
where needed. A round trip alone is not an independent normalization reference.]

## 7. Reproducibility and evidence

[State whether reproducibility means exact, numerical within tolerance, or
statistical. Record source/spec revisions, environment, commands, hardware where
relevant, RNG configuration, and retained raw outputs.]

| Evidence ID | Claims / cases | Baseline and procedure | Observed result / artifact | Limits |
| --- | --- | --- | --- | --- |
| EV-001 | SCI-001 / VAL-001 | [revision and procedure] | Not run | [missing evidence] |

Keep planned, partial, skipped, inconclusive, failed, and passed evidence distinct.
Passing API tests or a documentation checklist does not establish physical validity.

## 8. Open decisions and scientific change control

| Open decision or limitation | Scientific consequence / affected claims | Resolution criterion and responsible role |
| --- | --- | --- |
| OPEN: [unknown] | [consequence] | [evidence needed and owner] |

[Explain how changes to assumptions, algorithms, reference data, or tolerances are
reviewed. Preserve previous evidence and mark it stale when its basis changes.]

**Readiness decision**: [design readiness and evidence status separately; draft
until the required review and validation are recorded]
