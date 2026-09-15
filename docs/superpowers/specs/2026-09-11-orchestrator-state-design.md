# Orchestrator-owned state

Wheel owns two L1Data copies, an optional DataCovariance noise matrix, a Ledger
of complete `BlockResult` records, and the campaign RNG. Stateless blocks initialize
from their priors. The [domain translation guide](../../domain-translation.md)
describes time, frequency, and WDM conversion at block boundaries.

## Ownership and protocol

Construct Wheel with `Wheel(observed_data, initial_noise_covariance=None, *,
random_seed=None)`, register with `add(block_to_register, *, data_domain=None,
num_frequency_divisions=None, num_time_divisions=None)`, and sample with
`run(num_cycles, on_cycle_complete=None)`. Read conditional data through
`residual(exclude_block_name=None)` and signals through `contribution(block_name)`.
Public snapshot properties are `observed_data`, `working_residual`,
`noise_covariance`, and `ledger`.
`Block.name` identifies a registered block; the block methods receive a NumPy
random generator through `rng`.

Wheel owns pristine observations, the current full residual, covariance, complete
per-block results, and a NumPy RNG. Block instances retain fixed configuration
only. `draw_prior(conditional_residual, noise_covariance, *, rng) -> BlockResult`
runs at registration;
`sample(conditional_residual, noise_covariance, current_block_result, *, rng) -> BlockResult`
runs once per cycle. A prior
initializer draws parameters from the model's declared prior, initializes sampler
state, and publishes the applicable signal and covariance outputs. A zero-source
prior may omit its signal. Every `sample` call receives the previous accepted
`BlockResult`, including `model_parameters`, `sampler_state`, and `metadata`.
`conditional_residual` is `L1Data`; `noise_covariance` is
`DataCovariance | TranslatedCovariance | None`; `current_block_result` is the accepted record.
The call can run multiple internal sampler steps. EchoBlock's
`sampler_state["num_sample_calls"]` counter counts accepted `sample` calls,
not individual sampler draws.

One Block protocol covers signal-only, noise-only, and joint signal/noise models.
The optional signal and covariance outputs determine each block's contribution.

`BlockResult` has optional `tdi_signal_contribution` (the sum of its signal
templates) and optional `noise_covariance`, plus `model_parameters`, `sampler_state`,
and `metadata` dictionaries. Model parameters describe the physical model;
sampler state contains algorithm continuation information, such as walkers,
adaptation, and external RNG state. Metadata holds diagnostics. Each result is a
complete snapshot. Omitted `model_parameters`, `sampler_state`, and `metadata`
default to empty dictionaries. An omitted signal means zero and removes any
previous signal from that block. A noise-only block can
return `BlockResult(noise_covariance=...)` directly. Wheel snapshots results with
deepcopy. Opaque external resources are not sampler state: wrappers recreate
resources per call and return portable state. Chain history is optional and is
normally collected with `on_cycle_complete` to avoid copying a growing chain per call.

Ledger is a read-only snapshot store keyed by stable registration names. It
supports indexing, iteration over names, membership checks, and length.
`ledger.snapshot()` returns an independent dictionary of all complete
`BlockResult` records; dictionary methods such as `items()` operate on that
snapshot. The Ledger itself compares by identity. Compare physical arrays and
sampler state explicitly when comparing results from different campaigns.
Reads and adoption copy nested data. `contribution(block_name)` returns signal
arrays, materializing zeros for an absent contribution; `ledger[name]` returns
the full `BlockResult`. Internally, Ledger stores accepted snapshots in
`_results_by_block_name`.

The type is exported by `enchilada` and `enchilada.block_result`. L1Data's
`block_result(tdi_signal_contribution=None, *, noise_covariance=None,
model_parameters=None, sampler_state=None, metadata=None)` validates optional
signal arrays. `zero_block_result(...)` supplies explicit zero arrays. The returned
record provides `with_noise_covariance(...)`. A signal template is a physical
waveform; a block result contains the block's complete accepted state.

L1Data names its observation fields `channel_data`, `channel_names`,
`sample_rate_hz`, `num_time_samples`, `start_time_gps`, `data_domain`,
`physical_observable`, `tdi_generation`, `orbit_ephemeris`, and the optional
`wdm_grid` required for a WDM representation.
`block_result(tdi_signal_contribution=...)` validates
signal arrays on that observation grid and stores them in
`BlockResult.tdi_signal_contribution`.
Scientific shorthand is supported notation: `fs`, `N`, and `t0` refer to
`sample_rate_hz`, `num_time_samples`, and `start_time_gps`. The derived quantities are `observation_duration_s` (`Tobs`), `sample_interval_s`
(`dt`), `frequency_resolution_hz` (`df`), and `nyquist_frequency_hz` (`fny`).
The shared signal validator is `check_on_grid(tdi_signal_contribution, *,
channel_names, num_time_samples, data_domain, context_label, wdm_grid=None)`.

For observed d and block signals T_j, full residual is d - sum(T_j); input to i is
d - sum(j != i, T_j). A persistent balanced tree caches signal sums, so replacing
or excluding one block takes O(log B) array additions for B blocks. Residuals are
formed from pristine observations and the appropriate cached sums, avoiding
incremental add-back drift. Candidate tree paths are committed atomically with
the corresponding result. The tree uses up to one aggregate array per channel
at each internal node. Array dtypes promote as needed.

Blocks receive copies, including covariance and block result state. Internally
owned covariance snapshots copy previously validated arrays without repeating
Cholesky factorization; public construction and covariance publication retain
full validation. The orbit remains a shared, immutable scientific resource.
Inspection cannot edit accepted arrays or state.

The orbit interface uses `nominal_arm_length_m`, `transfer_frequency_hz`, and
`positions(times_gps)`. Tabulated ephemerides expose `time_range_gps`, which L1Data
checks against its sample span. NumericOrbit construction and array loading use
`sample_times_gps` and `spacecraft_positions_m`; loader options use
`coordinate_frame`, `file_path`, `group_path`, `position_dataset_names`, and
`lisaorbits_model`. HDF5 loaders read the external `t0`, `dt`, `size`, and
`sc_position_*` schema; lisaorbits loading calls its `compute_position` method.

On each call copy the RNG. Validate and copy the return, check covariance metadata,
and prepare the next residual before committing the complete result, noise, RNG,
and membership. Joint signal and covariance outputs are adopted atomically.
Failure leaves that block's prior accepted state and RNG unchanged; previously
accepted updates in the same cycle remain. Noise is a single replaceable model
with a warning for a second owner. Warnings-as-errors must precede commit.

`on_cycle_complete(cycle_index, wheel)` runs only after every block in a cycle
has accepted its result. The cycle index starts at zero for each `run` call.
A partially failed cycle does not invoke the callback; earlier successful block
results remain committed. A callback exception occurs after the complete cycle
has committed.

## Covariance

DataCovariance holds actual coefficient covariance E[x x†] in `covariance_matrix`,
with shape point × channel × channel in time/frequency, and
`(Nf + 1, Nt, channels, channels)` for native WDM covariance. Its grid fields match
L1Data's names: `channel_names`, `sample_rate_hz`, `num_time_samples`,
`start_time_gps`, `data_domain`, and `wdm_grid`. The boolean `active_mask` marks
points used in inference with `True`; its default `None` selects all points in
time/frequency and all structurally active WDM coefficients. Covariance
helper signatures use `from_psd(reference_data, noise_psd_model)`,
`from_variance(reference_data, time_sample_variance)`, and
`check_compatible(reference_data)`. The covariance weighting accessors
`noise_psd(channel_name=None)` and `noise_variance(channel_name=None)` expose
marginal weights. L1Data carries observations and conventions; covariance and
noise weighting belong to DataCovariance.
This initial representation supports cross-channel correlation but
assumes independence across sample/bin indices. It does not silently represent
arbitrary temporal correlation.

Constructors support time variances and one-sided PSD models. PSD conversion uses
C_k = (Tobs/2) S_k for dt*rfft coefficients. DC is inactive with finite zero matrix
entries; no infinite entries are stored. Active matrices must be finite Hermitian
positive definite, with scale-relative checks suitable for very small LISA units.
Noise weighting excludes DC and applies the real-FFT Nyquist convention.

Pass a DataCovariance to `Wheel(observed_data, initial_noise_covariance=...)` to initialize
noise weighting. With `initial_noise_covariance=None`, Wheel starts without a
covariance, and a block can publish one. PSD models are converted explicitly with
`DataCovariance.from_psd(reference_data, noise_psd_model)`; Wheel does not infer
covariance from the observations. Blocks receive the current covariance as an
explicit argument and publish it through `BlockResult.noise_covariance`.
Once a block owns the current covariance, every
subsequent result from it must include its full covariance, even if unchanged;
omission raises `ValueError`. Other blocks may omit covariance to preserve the
current global value. The single-owner policy warns on takeover and does not
combine independent noise components. `Wheel.noise_covariance` exposes the
current covariance as a defensive snapshot.

## Scope and validation

EchoBlock, the toy fit, and the GB example demonstrate the stateless block
contract and explicit continuation state. The runtime core requires NumPy;
external samplers and waveform packages belong to optional examples.
A checkpoint file format, parallel scheduling, and automatic combination of
independent noise components are outside the current scope.

Test prior/state round trips, unchanged block configuration, independent campaigns,
data/state/covariance alias isolation, failure rollback including RNG and warnings,
registration order, precision, covariance validation, and toy posterior recovery.
Cover signal-only, noise-only, state-only, and joint results, omitted-signal
removal, required covariance publication by the current owner, and atomic rollback.
Run the repository's pytest, lint, format, and mypy checks.

## Domain translation

`transform` converts observations, covariance, and complete results between time,
frequency, and WDM representations. Wheel can explicitly select a representation
and WDM division counts for each block. Inputs are converted together; returned
outputs are converted back to the canonical grid before atomic adoption.
Native covariance remains an independent-point model, while TranslatedCovariance
preserves induced cross-point correlations with exact operators. Parameters,
sampler state, and metadata are copied without reinterpretation. The optional
local WDM backend requires Python 3.13 or newer and is imported lazily; core
time/frequency operations remain NumPy-only. See the
[implemented guide](../../domain-translation.md) for masks, normalization, and
installation.
