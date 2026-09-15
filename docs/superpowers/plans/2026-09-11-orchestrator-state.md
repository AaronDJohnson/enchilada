# Orchestrator-owned state implementation plan

> Historical implementation plan. Some interface decisions recorded here were
> superseded; see the [current design](../specs/2026-09-11-orchestrator-state-design.md)
> for the supported API.

**Goal:** Move campaign state into Wheel and expose a stateless block contract.
**Architecture:** Explicit covariance and complete `BlockResult` snapshots, committed
atomically by Wheel. Ledger owns accepted snapshots; working residual is rebuilt
from pristine observations.
**Tech stack:** Python 3.12+, NumPy, pytest, mypy, Ruff.
**Spec:** ../specs/2026-09-11-orchestrator-state-design.md

## Global constraints

No new runtime dependencies. Preserve existing user edits. WDM is deferred.
Blocks may retain fixed configuration, but all changing state is explicit.
L1Data uses `channel_data`, `channel_names`, `sample_rate_hz`, `num_time_samples`,
`start_time_gps`, `data_domain`, `physical_observable`, and `orbit_ephemeris`.
Its `block_result(tdi_signal_contribution=...)` builder stores optional signal
arrays in `BlockResult.tdi_signal_contribution`. A subsequent covariance naming
migration aligns grid fields with L1Data and uses `covariance_matrix` and
`active_mask`; it does not change covariance behavior. Results also expose optional
`noise_covariance`, physical
`model_parameters`, algorithm continuation `sampler_state`, and `metadata`.
Each result is a complete snapshot. An absent signal removes the previous signal;
the current covariance owner must publish its full covariance on every call.
Noise-only blocks return covariance directly; `zero_block_result()` remains an
explicit zero-array helper, and `with_noise_covariance()` attaches covariance to
a new result. Signal and covariance outputs are adopted atomically.
A subsequent orbit naming migration uses `nominal_arm_length_m`,
`transfer_frequency_hz`, `time_range_gps`, and `positions(times_gps)`, with
descriptive constructor/loader arguments. Scientific behavior and external
lisaorbits/HDF5 schema names remain unchanged.
A subsequent Wheel naming migration uses `observed_data`,
`initial_noise_covariance`, and `random_seed` at construction;
`add(block_to_register)`, `run(num_cycles, on_cycle_complete=None)`,
`residual(exclude_block_name=None)`, and `contribution(block_name)` preserve
argument order and behavior. The callback receives `(cycle_index, wheel)` only
after a complete cycle accepts, with a zero-based index local to each run.
A partially failed cycle does not invoke it.

## Tasks

- [x] Covariance: create covariance.py and test_covariance.py. Test PSD scaling,
  Hermitian positivity at small physical scales, metadata, masks and copy isolation
  before implementing the matrix container and constructors.
- [x] Orchestration: expand `BlockResult` and L1Data's `block_result` /
  `zero_block_result` builders, add Ledger,
  replace block protocol, and implement transactional Wheel updates. First test
  `wheel.ledger['a'].sampler_state['updates'] == 2` after run(2), input d-minus-others,
  failed-return rollback, RNG retry and mutation isolation.
- [x] Migrate helpers, examples, notebook cells, and README. Use fixed block
  configuration, `draw_prior(conditional_residual, noise_covariance, *, rng)`, and
  `sample(conditional_residual, noise_covariance, current_block_result, *, rng)`.
  Move diagnostics to ledger/on_cycle_complete. `sampler_state["updates"]` counts accepted
  `sample` calls; a call may perform multiple sampler steps.
  Validate demo execution, notebook execution, and toy posterior recovery.
- [x] Review and verify the final diff. Run `.venv/bin/python -m pytest -q --cov`,
  `.venv/bin/ruff check src tests examples`, `.venv/bin/ruff format --check src
  tests examples`, and `.venv/bin/mypy`. Resolve failures before reporting results.

## Decisions

Use the existing workspace so the IDE sees changes and user modifications remain
in place. Keep changes uncommitted for review. A new implementation supersedes the
old stateful API rather than silently supporting two state ownership models.

## Verification and review of the orchestrator-state baseline

The optional-output change has its own implementation and verification plan in
[2026-09-13-optional-block-results.md](2026-09-13-optional-block-results.md).

- Full suite: 299 passed, 3 skipped; coverage 98.79% (required 95%).
- Ruff lint/format, mypy, and git diff whitespace checks pass.
- Wheel and sdist built offline. Importing the built wheel validates the new
  exports, py.typed, covariance, and two-cycle ledger state continuation.
- Independent review found no remaining blockers. Unsupported covariance/domain
  guards in toy blocks were corrected during review.
- Real cached Eryn tests exercise restart reproducibility and likelihood refresh
  with a deterministic substitute waveform. The native GBGPU example/notebook
  is not verified in this environment because its backend is unavailable.
- A pre-existing injection helper draws complex Nyquist noise for even N; the GB
  example likelihood excludes the endpoint. It was not changed by this refactor.
- uv 0.12.1 built successfully but warned that the repository's declared
  uv-build range is >=0.11.32,<0.12.0; build dependency bounds were not changed.

The WDM recommendation is recorded in ../../domain-translation.md. It explains
Fourier scaling, WDM grid/masks, and why exact transformed noise needs structured
cross-point covariance beyond the current matrix stack.
