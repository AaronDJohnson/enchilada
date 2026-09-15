# Optional Block Results Implementation Plan

> Historical implementation plan. Some interface decisions recorded here were
> superseded; see the [current design](../specs/2026-09-11-orchestrator-state-design.md)
> for the supported API.

> **For agentic workers:** Use subagent-driven development for the independent migrations and review; keep the core behavior change and its tests together.

**Goal:** Let signal, noise, and joint blocks return complete results with descriptive fields and optional physical outputs.

**Architecture:** `BlockResult` carries optional `tdi_signal_contribution` and `noise_covariance`, plus `model_parameters`, `sampler_state`, and `metadata`. Wheel alone interprets and adopts these values. An absent signal is zero; the current covariance owner must return its full covariance every time.

**Tech Stack:** Python >=3.12, NumPy, pytest, Ruff, mypy.

**Spec:** User-approved design in the conversation: two optional estimates, the final name `tdi_signal_contribution`, and separate physical parameters and sampler continuation state.

## Constraints

- Work on the existing `codex/orchestrator-owned-state` branch, preserving all prior edits.
- Keep version 0.2.0 for this ongoing architectural release.
- Keep `L1Data.noise` and `Wheel.noise` unchanged. This optional-output change retained Wheel's constructor keywords; a subsequent naming migration uses `observed_data`, `initial_noise_covariance`, and `random_seed`, while preserving argument order and defaults.
- A subsequent Wheel method migration uses `add(block_to_register)`, `run(num_cycles, on_cycle_complete=None)`, `residual(exclude_block_name=None)`, and `contribution(block_name)`. The callback receives `(cycle_index, wheel)` after a full cycle accepts; its zero-based index is local to each run, and a partially failed cycle does not invoke it.
- This optional-output change retained DataCovariance's fields; a subsequent naming migration uses `covariance_matrix`, `active_mask`, `channel_names`, `sample_rate_hz`, `num_time_samples`, `data_domain`, and `start_time_gps` without changing covariance behavior.
- Preserve canonical grid validation, independent snapshots, and atomic adoption of result/covariance/residual/RNG.
- Preserve the existing warning when another block replaces the covariance owner. Do not add covariance components or domain adapters.
- Keep historical `openspec/`, `specs/`, and released changelog entries unchanged.
- Leave changes uncommitted for user review.

## Task 1: Optional results and orchestrator behavior

**Files:** `src/enchilada/block_result.py`, `data.py`, `wheel.py`, `block.py`, `ledger.py`; `tests/test_block_result.py`.

**Interface:**

```python
BlockResult(
    tdi_signal_contribution=None,
    noise_covariance=None,
    model_parameters={},
    sampler_state={},
    metadata={},
)
```

All dictionaries use independent default factories. Explicit signal dictionaries must remain nonempty and valid for the full channel grid. `None` is the only omitted-signal sentinel.

- [x] Write and run tests demonstrating noise-only adoption without signal arrays, signal removal/restoration, and rollback when a covariance owner omits its estimate or returns an incompatible covariance.
- [x] Rename fields and `with_noise_covariance(noise_covariance)`. Preserve constructor container validation and Wheel's defensive covariance validation.
- [x] Let `L1Data.block_result(tdi_signal_contribution=None, *, noise_covariance=None, model_parameters=None, sampler_state=None, metadata=None)` build optional or joint results. Retain explicit zeros in `zero_block_result`, with the same keyword fields.
- [x] Filter absent signals during residual formation. `Wheel.contribution(block_name)` returns fresh zero arrays on the observation grid for absent signals.
- [x] Reject omitted covariance from the current noise owner before any accepted state changes. Keep non-owner omission and single-owner replacement behavior.
- [x] Run the new behavioral tests and review core changes.

## Task 2: Migrate consumers and documentation

**Files:** Existing `tests/`, `examples/*.py`, `examples/*.ipynb`, `src/enchilada/testing.py`, README, current design documents, unreleased changelog.

- [x] Migrate BlockResult access, constructors, replacements, and helper keywords; preserve external sampler APIs and unrelated covariance/noise fields.
- [x] Make noise-only and no-op examples return absent signals. Retain physical parameters separately from all sampler continuation state.
- [x] Document absent-signal semantics, complete covariance snapshots, optional outputs, and all field/helper renames.
- [x] Run existing tests including runnable README examples and optional Eryn continuation coverage.

## Task 3: Review and verify

- [x] Independently review cross-file integration for retired result fields, accidental external API renames, and partial adoption.
- [x] Run `.venv/bin/python -m pytest -q --cov --cov-report=term-missing` (coverage floor 95%).
- [x] Run Ruff lint/format checks, mypy, and `git diff --check`.
- [x] Report completed behavior and verification results, including any optional dependency skips.

## Verification results

- The six new behavioral cases failed before implementation because the new optional-result constructor was unavailable, then passed with the core change.
- Full suite: 308 passed, 3 skipped; coverage 98.82% against the 95% floor.
- Optional real-Eryn examples: 21 passed. Toy fit recovered amplitudes 2.9866 and 2.0019 and noise sigma 0.4997 for injected values 3, 2, and 0.5.
- Ruff lint and formatting, mypy, and whitespace checks passed.
- Independent core review found no actionable issues in optional-output handling, ownership, snapshots, rollback, or typing.
- Version 0.2.0 wheel and sdist built successfully. Built-wheel smoke verified optional results, covariance factory/helper, ledger continuation, and py.typed.
