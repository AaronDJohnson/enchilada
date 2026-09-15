# enchilada Specifications

For new scientific work, use the [scientific software template](../.specify/templates/overrides/spec-template.md),
its [authoring guide](scientific-software-template/guide.md), and the
[Enchilada worked example](scientific-software-template/example.md). They put
mathematical meaning, algorithmic invariants, numerical error, and independent
validation at the center, with explicit evidence gaps and claim status.

An [OpenSpec version and comparison guide](../openspec/comparison.md) covers the
same package baseline in six capability specifications, with one-to-one mappings
to all 50 original requirements and a separate evidence record.

The [package baseline](001-package-baseline/spec.md) records the historical package on branch `template-return` at commit `dd412c3d47d3a209b632f43ee3220215936a31cc`, when project metadata read `0.1.0` and blocks returned `Template` objects. It predates the current `BlockResult` and explicit-initialization interfaces. Use the [README](../README.md) and [block tutorial](../examples/create_block.ipynb) for the current API.

| Artifact | Purpose |
| --- | --- |
| [Specification](001-package-baseline/spec.md) | Eight prioritized user journeys, acceptance scenarios, 50 functional requirements, 10 measurable outcomes, and explicit scope assumptions. |
| [Package contract](001-package-baseline/contracts/package-contract.md) | Exact public interfaces, data ownership, residual equations, Fourier/noise conventions, orbit loaders, and delivery configuration. |
| [Evidence and traceability](001-package-baseline/traceability.md) | Links requirements to source/tests and records validation limits and existing documentation drift. |
| [Quality checklist](001-package-baseline/checklists/requirements.md) | Specification review results, separate from runtime test results. |

The baseline follows the official [Spec Kit template](https://github.com/github/spec-kit/blob/main/templates/spec-template.md), with technical details in a companion contract. It is an inventory of current behavior, not a certification of scientific correctness. The new scientific template brings those mathematical and numerical contracts into the main specification.

## Using the Baseline

Use the specification and contract to understand that historical snapshot. Treat its evidence record as an inventory of checks and limits at the recorded commit, not as current validation.

For future work, create a separate numbered feature specification and cite the baseline requirement IDs it changes. A plan for a new feature should distinguish retained behavior from proposed additions; this retrospective baseline is not a task list to rebuild the package.

The baseline is available directly in `specs/001-package-baseline`. No active Spec Kit feature selection is configured by these documentation artifacts.

These are repository specification artifacts. Running interactive Spec Kit commands additionally requires its CLI and the chosen agent integration. The [upstream setup guide](https://github.com/github/spec-kit#-get-started) describes that separate setup. This artifact set does not imply those commands are installed or that a project constitution has been ratified.
