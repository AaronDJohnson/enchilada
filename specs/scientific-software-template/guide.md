# Writing a scientific software specification

Start with the [reusable template](../../.specify/templates/overrides/spec-template.md)
and the [Enchilada worked example](example.md). The template is intended for
numerical libraries, simulation, inference, and scientific data processing. Its
central unit is a scientific claim with a defined domain and an acceptance method.

## Choose a useful scope

Specify one coherent computational capability. A small package can share one
specification. For a larger package, put common units, data representations, and
mathematical assumptions in a shared contract and give each capability its own
claims and validation cases. Link those definitions to prevent divergent copies.

Keep a small specification small: one equation, one claim, one independent check,
and a short evidence record can be sufficient. Include interfaces and operational
details when they affect scientific use. For example, a buffer-copy guarantee can
protect sampler state; a package's release naming convention rarely explains its
numerical behavior.

## Write and use the specification

1. Copy the template into the relevant `specs/<feature>/spec.md`. Keep the template
   reusable; fill the copy. Replace placeholders and remove its authoring comments.
2. State the mathematical quantity, supported regime, assumptions, and conventions.
   For existing software, separate what it currently does from what it should do.
3. Describe algorithm steps and shared state wherever they affect the result.
   Distinguish a scientific constraint from an implementation choice.
4. Write a small set of `SCI-NNN` claims. Give each a `VAL-NNN` case with explicit
   inputs, an independent reference, a measurable threshold, and a reason for it.
5. Identify genuinely unknown decisions as `OPEN`, with their consequences and
   resolution criteria. Remove inapplicable validation profiles. Keep evidence
   statuses `unverified` until the specified evidence actually exists.
6. Review the design, implement the necessary checks, run them, and record results
   against the exact baseline. An agreed design can be ready for implementation
   while its scientific claims remain unverified.

The example illustrates both deterministic accounting and a proposed statistical
acceptance case. It deliberately exposes missing evidence rather than turning
existing API tests into a claim of posterior correctness.

## Using it with Spec Kit

The canonical template is already placed at
`.specify/templates/overrides/spec-template.md`. Current Spec Kit documentation
describes that location as the highest-priority project template override and
provides `specify preset resolve spec-template` to inspect resolution. An initialized
Spec Kit project and an installed CLI are prerequisites for those commands; this
repository's documentation setup does not install them.
See [upstream preset resolution](https://github.github.io/spec-kit/reference/presets.html).

In another initialized project, copy the template to the same override location,
preserving any existing customization. For manual use, copy it directly into a
feature specification; no CLI is necessary.

**Template resolution does not replace command instructions.** The current
[upstream specify command](https://github.com/github/spec-kit/blob/main/templates/commands/specify.md)
also asks for user stories, business-oriented checks, and reasonable defaults for
unspecified details. Review or customize the installed authoring command before
using this scientific template in an automated workflow. In particular, mathematical
equations, numerical algorithms that define the result, and physical conventions
must remain in the specification, and unknown scientific choices must remain open.

This is a template and an authoring guide. It does not install a preset package,
replace all Spec Kit commands, or enforce CI gates. A future command/preset should
carry the same scientific rules through specification, planning, tasks, and review.

## Authoring instruction for an assistant

Use this alongside the template when asking an assistant to draft a specification:

```text
Write a scientific software specification using the supplied template. Inspect
the relevant code, tests, and primary scientific references before making claims.
Keep equations, assumptions, conventions, algorithmic invariants, and numerical
acceptance criteria in the specification. Separate observed behavior, intended
requirements, and recorded evidence.

Do not invent scientific assumptions, tolerances, references, results, or approvals.
Record unresolved choices as OPEN with their scientific consequences. Give each
claim a falsifiable validation case and explain the independence and uncertainty
of its reference. Mark planned, partial, skipped, and inconclusive evidence honestly.
For stochastic claims, address joint behavior where relevant, Monte Carlo error,
precision, false rejection, and sensitivity to meaningful errors.

Keep the document proportional to the capability. Remove inapplicable profiles.
Do not change source code, reference results, or tolerances just to make the
specification appear satisfied. Report remaining evidence gaps.
```

## Review for scientific substance

An independent reviewer should be able to reconstruct a small calculation from the
specification, identify a plausible error the checks would catch, and determine
exactly which conclusions the recorded evidence supports. A passing checklist is
evidence of document review only. Physical validity still needs suitable scientific
evidence, and some mathematical claims need an argument or proof alongside tests.

Read in a Markdown preview or through the repository's configured spec-view. The
equations use plain-text notation so their meaning survives viewers without a
mathematics renderer. In spec-view, use the dashboard's document links to move
between files; its renderer does not translate repository-relative Markdown links
into viewer routes.
