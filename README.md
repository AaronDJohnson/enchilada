# enchilada

[![CI](https://github.com/AaronDJohnson/enchilada/actions/workflows/ci.yml/badge.svg)](https://github.com/AaronDJohnson/enchilada/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/AaronDJohnson/enchilada/blob/main/LICENSE)

Blocked-Gibbs global-fit orchestration for LISA.

*The whole enchilada*: a global fit infers every source population and the
instrument noise **together**, because none of them can be measured cleanly
without the others. This is the layer that makes that one joint fit out of
many separately-owned pieces.

A `Wheel` owns pristine L1 observations, a working residual, optional noise
covariance, and a `Ledger` containing each block's complete `BlockResult`:
optional signal and covariance outputs, model parameters, sampler state, and
diagnostics. A block keeps fixed configuration and receives its state explicitly
on every call. The Wheel supplies the data minus all other blocks' signals, then
validates and records the returned block result.
Waveforms, likelihoods, and prior/conditional sampling remain model code.

## Install

Requires Python ≥ 3.12 (floor set by lisaorbits).

```sh
pip install enchilada      # or: uv add enchilada
```

To work on enchilada itself, clone it and use [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/AaronDJohnson/enchilada.git
cd enchilada
uv sync
```

or with pip: `pip install -e .`

The core package depends only on NumPy (`>=1.26.4`). GBGPU, Eryn, and LISA Analysis Tools
are confined to the optional galactic-binary example. Installing enchilada or
any of its extras does not install them; running that example requires the
separate `examples/requirements-gb.txt` installation described below.

Loading tabulated spacecraft
ephemerides (`enchilada.orbits.NumericOrbit`) needs the extra:

```sh
uv sync --extra numeric-orbits   # adds h5py, scipy, lisaorbits

# For the notebooks, install both extras together:
uv sync --extra numeric-orbits --extra examples
uv run --no-sync jupyter lab examples/demo.ipynb
```

## Quickstart

```sh
uv run --no-sync python examples/demo.py
```

runs three Gibbs cycles over two no-op `EchoBlock`s on synthetic data —
enough to watch the Wheel hand each block its residual. The whole thing is:

<!-- runnable -->
```python
import numpy as np
from enchilada import BlockResult, L1Data, Wheel
from enchilada.testing import EchoBlock

rng = np.random.default_rng(0)
num_time_samples = 1024
channel_names = ("A", "E", "T")

# One frozen object holds the TDI arrays and the run settings everyone shares.
observed = L1Data(
    channel_data={ch: rng.standard_normal(num_time_samples) for ch in channel_names},
    sample_rate_hz=0.1,
    channel_names=channel_names,
    tdi_generation="2.0",
    physical_observable="fractional_frequency",
)  # num_time_samples is read off the arrays; start_time_gps defaults to 0.0

ucb = EchoBlock(name="ucb")
mbhb = EchoBlock(name="mbhb")

wheel = Wheel(observed_data=observed)
wheel.add(
    block_to_register=ucb,
    initial_block_result=BlockResult(sampler_state={"num_sample_calls": 0}),
)
wheel.add(
    block_to_register=mbhb,
    initial_block_result=BlockResult(sampler_state={"num_sample_calls": 0}),
)
wheel.run(num_cycles=3)

wheel.residual()  # the running residual: observed minus every block's template
# 3 accepted sample() calls, stored in the ledger
wheel.ledger["ucb"].sampler_state["num_sample_calls"]
```

[`examples/demo.ipynb`](https://github.com/AaronDJohnson/enchilada/blob/main/examples/demo.ipynb) is the same walkthrough with
commentary, plus the `L1Data` long/short name aliases (`Tobs`, `fs`, `dt`,
...), the typo catcher, and attaching a constellation ephemeris. For a real
(toy) sampler — two conjugate-Gibbs source blocks plus a sampled
white-noise block, converging to known truth — run
[`examples/toy_fit.py`](https://github.com/AaronDJohnson/enchilada/blob/main/examples/toy_fit.py).

To write your own block, work through the
[Block authoring tutorial](examples/create_block.ipynb). It builds a Metropolis
sampler from scratch, starts its main campaign at a known injection, checks its
posterior, demonstrates continuation and domain selection, and adds a noise-only
block using the same interface.

For a **real LISA source class**, [`examples/gb_block_eryn.ipynb`](https://github.com/AaronDJohnson/enchilada/blob/main/examples/gb_block_eryn.ipynb)
fits an injected galactic binary through the Wheel using GBGPU waveforms, an
Eryn sampler reconstructed from explicit continuation state, and a fixed LISA noise PSD from LISA
Analysis Tools. Everything that is *not* enchilada lives in
[`examples/gb_model.py`](https://github.com/AaronDJohnson/enchilada/blob/main/examples/gb_model.py), so the notebook shows only the
enchilada touchpoints. That example needs an external LISA stack — none of it
an enchilada dependency, so it is not exercised by CI:

```sh
uv sync --extra numeric-orbits --extra examples
uv pip install -r examples/requirements-gb.txt
uv run --no-sync jupyter lab examples/gb_block_eryn.ipynb
```

Run these commands from the checkout root, in this order. Sync the project extras
first, then install the external stack: a later exact `uv sync` removes packages
that are outside the project lockfile. `--no-sync` launches Jupyter in that prepared
environment. The example requirements constrain NumPy to `<2.4`: released Eryn
1.2.6 calls `numpy.in1d`, which was [removed in NumPy 2.4](https://numpy.org/devdocs/release/2.4.0-notes.html#removed-numpy-in1d).
This constraint applies only to the optional example environment.

Every piece of it ships wheels, so nothing compiles. But `gbgpu` and
`lisaanalysistools` publish *no* sdist, which fixes the example's window
narrower than enchilada's own: **Python 3.12–3.13, on Linux or Apple-silicon
macOS**. Intel macOS, Windows, and 3.14 have no wheels, and pip reports that as
`No matching distribution found` rather than a build error — see the header of
[`examples/requirements-gb.txt`](https://github.com/AaronDJohnson/enchilada/blob/main/examples/requirements-gb.txt)
for the full matrix. Wheel availability and dependency resolution do not establish
native GBGPU runtime validation. Its outputs are not committed; run it to populate them.

## Plugging in your sampler

Implement the stateless `Block` protocol:

- `name`: unique within a Wheel.
- `sample(conditional_residual, noise_covariance, current_block_result, *, rng) -> BlockResult`:
  advance the supplied state against the data minus every other block's latest
  signal.

`conditional_residual` is an `L1Data`; `noise_covariance` is a `DataCovariance`,
`TranslatedCovariance`, or `None`; and `current_block_result` is that block's
previous accepted `BlockResult`.
Use the supplied NumPy RNG for draws.
The orchestrator commits its progress only with a valid result. Keep changing
parameters, walkers, adaptation, and external sampler RNG state in the block result;
block instance attributes are fixed configuration.

A `BlockResult` is a complete snapshot with five fields:

| Field | Contents |
| --- | --- |
| `tdi_signal_contribution` | Optional dictionary of channel arrays containing the **sum** of this block's signal templates. `None` means zero signal. |
| `noise_covariance` | Optional full native or translated covariance publication. |
| `model_parameters` | Physical model state, such as amplitudes, source counts, or noise levels. |
| `sampler_state` | Algorithm continuation state, such as walkers, adaptation, counters, or external sampler RNG state. |
| `metadata` | Diagnostics such as log likelihoods. |

Each return replaces the block's previous snapshot. Omitted `model_parameters`,
`sampler_state`, and `metadata` default to empty dictionaries.
A missing signal contribution removes any signal that block previously
published. A zero-source or state-only block can return a result with no signal;
`conditional_residual.zero_block_result(sampler_state=...)` constructs explicit
zero arrays when they are useful.

For example, after calculating a new amplitude:

```python
new_block_result = conditional_residual.block_result(
    tdi_signal_contribution={"A": amplitude * basis},
    model_parameters={"amplitude": amplitude},
    sampler_state={
        "num_sample_calls": current_block_result.sampler_state["num_sample_calls"] + 1
    },
    metadata={"log_likelihood": log_likelihood},
)
```

One `sample()` call is one block's turn in a Wheel cycle. It may perform many
internal sampler steps. EchoBlock's `sampler_state["num_sample_calls"]` counts
accepted calls, not individual sampler draws.

The dictionaries must support deep copying. Keep open files, processes, and GPU
resources outside them; recreate resources inside a call from explicit numerical
state. A returned `L1Data` is rejected: blocks return a `BlockResult`.

Every `Wheel.add` requires a complete `initial_block_result`; omitting it or passing
`None` is an error. The application chooses its starting estimate and constructs
the result before registration. For a known injection:

```python
initial_result = observed.block_result(
    tdi_signal_contribution=injected_signal,
    model_parameters=injected_parameters,
    sampler_state=initial_sampler_state,
)
wheel.add(
    block_to_register=signal_block,
    initial_block_result=initial_result,
    data_domain="time",
)
```

Registration validates and adopts this result without calling model code or
consuming Wheel RNG draws. Build `initial_result` with the applicable signal or
covariance outputs, model parameters, and continuation state that this block's
`sample` expects. Optional prior initialization belongs in application or model
setup: use a separate setup RNG to draw parameters and construct their complete
result. The Block protocol requires only `name` and `sample`. The
[Block authoring tutorial](examples/create_block.ipynb) shows a complete example.
Supply signal arrays on the requested block domain and WDM grid, or the observation
grid if no domain is selected. Wheel converts them to the observation grid for
storage. Optional covariance uses its own grid metadata and the normal conversion
and ownership rules; noise-only initial results are supported too.

Wheel validates and defensively copies the result before adopting it atomically.
The model must check that its parameters, rendered signal, and algorithm state
agree and that the parameters lie within its prior support; Wheel cannot establish
these scientific conditions. The initial estimate changes the ledger and residual,
while preserving the observations and declared prior. Subsequent calls sample normally; starting
at an injection does not hold its parameters fixed. This initializes one block;
it does not restore a full campaign checkpoint or the Wheel RNG state.

`L1Data` holds channel arrays and observation conventions. Supply noise as an
explicit covariance when constructing Wheel:

```python
from enchilada import DataCovariance, Wheel

noise = DataCovariance.from_variance(reference_data=observed, time_sample_variance=0.25)
wheel = Wheel(
    observed_data=observed,
    initial_noise_covariance=noise,
    random_seed=42,
)
```

With `initial_noise_covariance=None`, Wheel starts without a covariance; a block
can publish one. Convert a one-sided PSD model explicitly with
`DataCovariance.from_psd(reference_data=observed, noise_psd_model=model)`. The model
provides `psd(frequencies)` or `psd(frequencies, channel)`, returning positive
one-sided densities on the supplied positive-frequency grid.
Signal-only, noise-only, and joint signal/noise models all implement `Block`.
A noise-only block can publish covariance directly:

```python
from enchilada import BlockResult

new_block_result = BlockResult(
    noise_covariance=covariance,
    model_parameters={"sigma": sigma},
    sampler_state={
        "num_sample_calls": current_block_result.sampler_state["num_sample_calls"] + 1
    },
)
```

Publishing covariance makes that block the current covariance owner. The owner
must return its full covariance on every subsequent call, even when unchanged;
omitting it raises `ValueError`. Other blocks may omit `noise_covariance` to
preserve the current global covariance. Joint signal/noise results publish both
outputs in one `BlockResult`; validation and adoption are atomic.
`result.with_noise_covariance(covariance)` builds a new result with that covariance.

`DataCovariance.noise_psd(channel_name="A")` exposes a marginal PSD from frequency
covariance. `noise_variance(channel_name="A")` gives a scalar time-domain weight
from stationary time covariance or integrated spectral covariance. These marginal
weights do not capture cross-channel correlations; those likelihoods must use the
full covariance matrix. Use `noise.check_compatible(reference_data)` to check
channel ordering, sample count, sample rate, and start time.

`DataCovariance.covariance_matrix` holds actual coefficient covariances `E[x x†]`.
Time and frequency matrices have shape **point × channel × channel**:
`(num_points, num_channels, num_channels)`. Native WDM matrices have shape
`(num_frequency_divisions + 1, num_time_divisions, num_channels, num_channels)`.
`active_mask` has the same leading point dimensions; `True` marks points used in
inference. Its default `None` selects all points in time/frequency and all
structurally active WDM coefficients. Cross-domain covariance uses an exact
`TranslatedCovariance` operator rather than independent target-pixel matrices.

For an existing array of coefficient covariances on the observation grid:

```python
from enchilada import DataCovariance

noise = DataCovariance(
    covariance_matrix=coefficient_covariance,
    channel_names=observed.channel_names,
    sample_rate_hz=observed.sample_rate_hz,
    num_time_samples=observed.num_time_samples,
    data_domain=observed.data_domain,
    start_time_gps=observed.start_time_gps,
    active_mask=None,
)
```

Fourier coefficients use `dt * rfft(x)`; a one-sided PSD becomes `C = (Tobs / 2) S`.
The PSD constructor marks DC inactive and stores finite zeros there. All matrices
must be finite and Hermitian; matrices at active points must also be positive
definite. The initial representation supports
cross-channel correlations at each sample/bin and assumes independent points.
The native matrix stack does not directly represent arbitrary correlations between
points. Translation preserves any cross-point correlations induced by the change
of representation through covariance operators.

Wheel uses descriptive arguments at each entry point:

| Entry point | Signature |
| --- | --- |
| Constructor | `Wheel(observed_data, initial_noise_covariance=None, *, random_seed=None)` |
| Register a block | `add(block_to_register, *, initial_block_result, data_domain=None, num_frequency_divisions=None, num_time_divisions=None)` |
| Sample complete cycles | `run(num_cycles, on_cycle_complete=None)` |
| Read a residual | `residual(exclude_block_name=None)` |
| Read a block's signal | `contribution(block_name)` |
| Check a block | `check_block(block_under_test, observed_data, num_cycles=2, *, initial_block_result, initial_noise_covariance=None, random_seed=0)` |

Inspect `wheel.observed_data`, `wheel.working_residual`, `wheel.noise_covariance`, and
`wheel.ledger[name]`; each returns defensive copies of changing data. The orbit is
a shared immutable resource. `wheel.contribution(block_name=name)` returns just the summed
signal, including zero arrays for an omitted contribution, and
`wheel.residual(exclude_block_name=name)` gives that block's conditional data.
Ledger supports `ledger[name]`, iteration over block names, membership checks, and
`len(ledger)`. Use `ledger.snapshot()` for an independent dictionary of all current
results, including normal dictionary methods such as `items()`. A Ledger compares
by identity; compare result arrays and state explicitly when comparing campaigns.

Collect posterior history with `run(num_cycles, on_cycle_complete=callback)`;
Ledger retains current state only. The callback receives `(cycle_index, wheel)`
after every block in a cycle has accepted its result. The index is zero-based
and restarts at zero for each `run` call. A partially failed cycle does not invoke
the callback.

A failed call, invalid result, failed copy, or noise ownership warning promoted to
an exception leaves that block's accepted state, covariance, residual, and RNG
unchanged. Earlier successful updates in the same cycle remain committed. Retrying
is explicit; the Wheel does not retry or roll back a whole cycle automatically.

Before joining a campaign, exercise the protocol against synthetic data:

```python
from enchilada.testing import check_block

check_block(
    block_under_test=MyBlock(name="ucb"),
    observed_data=toy_observed,
    num_cycles=2,
    initial_noise_covariance=None,
    initial_block_result=toy_initial_result,
    random_seed=0,
)
```

This checks registration of the required complete `initial_block_result` and the
requested number of sampling cycles. Build `toy_initial_result` with the model
parameters, signal or covariance, and continuation state that `MyBlock.sample`
expects. Supply its signal in the domain of `observed_data`; `check_block` does not
select a separate block domain. The optional initial
covariance and random seed configure the test campaign; the defaults are shown
above. Each model must independently validate its
prior, likelihood, and sampler; see the conjugate example in `examples/toy_fit.py`.

## Time, frequency, and WDM translation

Use one wrapper for observations, covariance, and block results:

```python
from enchilada import transform

spectrum = transform(observed, "frequency")
wdm_data = transform(observed, "wdm", num_frequency_divisions=8)
wdm_alternate = transform(observed, "wdm", num_time_divisions=8)
restored = transform(wdm_data, "time")
```

WDM requires both division counts to be positive and even, with
`num_frequency_divisions * num_time_divisions == num_time_samples`. Supply either
count on the first conversion and the other is derived; both can be supplied if
consistent. WDM-to-WDM conversion without new counts reuses the grid.
`wdm_data.wdm_grid` is a frozen `WDMGrid`; each channel contains a real `float64`
array of shape `(num_frequency_divisions + 1, num_time_divisions)`, with zero
entries outside the grid's structural `active_mask`.

Time/Fourier transforms use only NumPy. WDM uses the unpublished local `wdm`
checkout and requires Python ≥3.13, NumPy `>=2.1,<3`, and SciPy `>=1.14.1,<2`.
With sibling checkouts, run from the enchilada directory:

```sh
uv sync --python 3.13
uv pip install ../wdm
uv run --no-sync python examples/domain_translation.py
```

Adjust `../wdm` to its local location. Install it after project sync and use
`--no-sync` to preserve the environment. This does not change the core package's
Python ≥3.12 or NumPy-only requirements.
For the recorded WDM version and a reproducible installation pinned to the
integration-test revision, see the [backend installation guide](docs/domain-translation.md#install-the-optional-wdm-backend).

Translate covariance with the same grid choice. Exact covariance operations
preserve the quadratic form across representations:

```python
noise = DataCovariance.from_variance(observed, 0.25)
wdm_noise = transform(noise, "wdm", num_frequency_divisions=8)
q_time = noise.quadratic_form(observed.channel_data)
q_wdm = wdm_noise.quadratic_form(wdm_data.channel_data)
```

`DataCovariance` and `TranslatedCovariance` provide `apply`, `solve`, `project`,
`quadratic_form`, `log_determinant`, and `degrees_of_freedom`. A translated
covariance preserves cross-point correlations and native exclusions. Its
`active_mask` is `None`; use `project` for the possibly nonlocal active subspace
rather than assuming independent target pixels. Log determinants include the
coordinate Jacobian; they are not invariant under a scaled transform.

Select a representation when registering a block:

```python
wheel.add(
    frequency_block,
    initial_block_result=initial_frequency_result,
    data_domain="frequency",
)
wheel.add(
    wdm_block,
    initial_block_result=initial_wdm_result,
    data_domain="wdm",
    num_frequency_divisions=8,
)
```

Each initial result must contain the state its block expects, with any signal
arrays on that block's selected domain and WDM grid.
Wheel converts the conditional residual, covariance, and previous signal into
the selected representation, then converts returned outputs back to the canonical
observation grid before atomic adoption. Omitting `data_domain` keeps the default
handoff. To convert a standalone signal-bearing result, supply its source grid:
`transform(result, "wdm", num_frequency_divisions=8, reference_data=observed)`.
Model parameters, sampler state, and metadata are copied without reinterpretation.
See the [domain translation guide](docs/domain-translation.md) for normalization,
mask semantics, native WDM covariance, and examples.
The [translation notebook](examples/domain_translation.ipynb) walks through the
same API with plots, round-trip checks, and blocks using different domains.

## Conventions and consistency checking

Cross-group runs fail through silently mismatched conventions, so enchilada
makes every convention an explicit, validated part of `L1Data`.

`channel_data` holds the observed or residual arrays. The other fields are
`channel_names`, `sample_rate_hz`, `num_time_samples`, `start_time_gps`,
`data_domain`, `physical_observable`, `tdi_generation`, `orbit_ephemeris`, and
`wdm_grid` (required only for WDM data).
`L1Data.block_result(tdi_signal_contribution=None, *, noise_covariance=None,
model_parameters=None, sampler_state=None, metadata=None)` builds a block result
and validates supplied signal arrays against this observation grid.

Scientific shorthand is part of the API:

| Quantity | Descriptive name | Scientific alias |
| --- | --- | --- |
| Sample rate in Hz | `sample_rate_hz` | `fs` |
| Time sample count | `num_time_samples` | `N` |
| Start time in GPS seconds | `start_time_gps` | `t0` |
| Observation duration in seconds | `observation_duration_s` | `Tobs` |
| Sample interval in seconds | `sample_interval_s` | `dt` |
| Fourier bin width in Hz | `frequency_resolution_hz` | `df` |
| Nyquist frequency in Hz | `nyquist_frequency_hz` | `fny` |

- `physical_observable` (required) — what the TDI samples physically are:
  `"fractional_frequency"`, `"phase"`, `"strain"`, or a campaign-agreed
  string. Every block reads this one field instead of assuming.
- `data_domain` — `"time"` (default, `num_time_samples` real samples per channel) or
  `"frequency"` (one-sided `dt * rfft(x)` spectra of length
  `num_time_samples // 2 + 1`). `num_time_samples` always counts time-domain
  samples, so `Tobs`/`df`/`dt` and the PSD grid stay well defined. `"wdm"` uses the
  two-dimensional real coefficient layout recorded by `wdm_grid`. A returned
  signal must match the representation and grid handed to that block; Wheel
  translates it back when an explicit block representation was selected.
- `num_time_samples` — **you should never have to state it.** Data enters a campaign
  as a time series, where the arrays carry it exactly, so enchilada reads it
  off them; `residual.to_frequency()` then carries it across the transform:

  ```python
  observed = L1Data(
      channel_data=time_series,
      sample_rate_hz=fs,
      channel_names=("A", "E"),
      tdi_generation="1.5",
      physical_observable="fractional_frequency",
  )
  spectrum = observed.to_frequency()  # num_time_samples rides along
  ```

  `to_frequency()`/`to_time()` use `transform()` to validate the source and apply
  the campaign's Fourier convention, `X(f) = dt * rfft(x)`, also used by
  `DataCovariance.from_psd`. They return independent channel arrays even when
  already in the requested domain. The round trip preserves either parity of n
  to numerical precision. Only a spectrum built with no time-domain provenance
  (e.g. straight from a frequency-domain waveform generator) has to state
  `num_time_samples`, because `n // 2 + 1` bins fit both n=1024 and n=1025,
  which mean different `Tobs` and `df`.
- `channel_names` — names imply the campaign's normalized definitions
  (e.g. A = (Z − X)/√2); see the `L1Data` docstring.

And it checks consistency at every boundary, failing loudly rather than
producing quietly wrong science:

- `L1Data` validates itself on every construction: `channel_data` keys must equal
  `channel_names`, array lengths must match `data_domain`/`num_time_samples`,
  and an attached orbit must span the observation (catching GPS-vs-zero-based epoch
  mismatches at construction, not mid-run).
- `Wheel` checks `name` and `sample` before registration, then validates
  every supplied initial or returned signal's channels, lengths, real/complex
  kind, and finiteness.
  Covariance publications are checked against the observation grid and matrix
  constraints before any state is committed.
- `NumericOrbit.positions` refuses to extrapolate outside its tabulated
  ephemeris instead of returning cubic-polynomial garbage.

## Orbits

The constellation ephemeris the data was produced with rides on
`L1Data.orbit_ephemeris` so every block builds its response from the *same*
spacecraft positions. `enchilada.orbits.NumericOrbit` tabulates and
cubic-spline-interpolates an ephemeris, with loaders for LDC/Mojito-style
HDF5 files (`from_hdf5`) and lisaorbits objects (`from_lisaorbits`); both
need the `numeric-orbits` extra.

`Orbit` exposes `nominal_arm_length_m`, `transfer_frequency_hz`, and
`positions(times_gps)`. `NumericOrbit.time_range_gps` gives the tabulated span,
which `L1Data` checks against its sample times.

To load arrays, supply GPS sample times with shape `(num_samples,)` and spacecraft
positions in metres with shape `(3, num_samples, 3)` (spacecraft, time, coordinate):

```python
from enchilada import NumericOrbit

orbit_ephemeris = NumericOrbit.from_arrays(
    sample_times_gps=sample_times_gps,
    spacecraft_positions_m=spacecraft_positions_m,
    coordinate_frame="ecliptic",
)
x_m, y_m, z_m = orbit_ephemeris.positions(times_gps=requested_times_gps)
```

The direct constructor uses `NumericOrbit(sample_times_gps,
spacecraft_positions_m, *, nominal_arm_length_m=None, transfer_frequency_hz=None)`
and expects ecliptic coordinates. `from_arrays` accepts the same optional arm
length and transfer frequency, plus `coordinate_frame="ecliptic"` by default.
Use `coordinate_frame="equatorial"` to rotate equatorial input on loading.

`from_hdf5(file_path, *, group_path="orbits", position_dataset_names=...,
coordinate_frame="equatorial", nominal_arm_length_m=None)` reads the HDF5
sampling attributes `t0`, `dt`, and `size`. The position dataset names
default to `("sc_position_1", "sc_position_2", "sc_position_3")`.
`from_lisaorbits(lisaorbits_model, sample_times_gps, *,
coordinate_frame="equatorial")` samples the external model through its
`compute_position` method. See the module docstring in
[`src/enchilada/orbits.py`](https://github.com/AaronDJohnson/enchilada/blob/main/src/enchilada/orbits.py) for frames and
conventions.

## Development

The [specification archive](specs/README.md) records the earlier package baseline
in Spec Kit format, with acceptance scenarios and requirement-to-test traceability.
It predates the current BlockResult interface; use this README and the
[block authoring tutorial](examples/create_block.ipynb) for the current API.

For scientific work, use the [scientific specification template](.specify/templates/overrides/spec-template.md)
and [authoring guide with a worked example](specs/scientific-software-template/guide.md).

```sh
uv sync --extra numeric-orbits   # dev group (pytest, pytest-cov, ruff, mypy)
uv run --no-sync pytest                    # full suite, incl. examples and orbit loaders
uv run --no-sync pytest --cov --cov-report=term-missing   # coverage (gate: 95%)
uv run --no-sync mypy                      # enchilada ships py.typed; keep it honest
uv run --no-sync ruff check src tests examples benchmarks
uv run --no-sync ruff format --check src tests examples benchmarks   # CI gates on this too
```

CI runs lint, formatting, mypy, the suite behind a 95% coverage gate, artifact
builds, and an installed-wheel smoke test across Python 3.12/3.13 on Linux and
macOS, with core-only checks through Python 3.14. A separate Python 3.12 job tests
the core at NumPy 1.26.4 without orbit dependencies; another job resolves the
numeric-orbit dependency floors. The optional WDM integration job installs a
pinned backend revision and runs the numerical and notebook checks. Tag pushes
matching `v*` run these gates before PyPI Trusted Publishing. Manual release
dispatches build and validate artifacts without publishing, including on tags. See
[CHANGELOG.md](https://github.com/AaronDJohnson/enchilada/blob/main/CHANGELOG.md) for release notes.

## Known limitations

Deliberate scope decisions, recorded so they are choices rather than
oversights:

- **No data-quality / gap mask.** Wheel requires complete finite observation arrays.
  Fill or trim gaps before constructing a campaign. A future data-gap interface
  must define how masking affects residual arithmetic, sampling grids, and
  windowing; `DataCovariance.active_mask` only selects covariance points for
  inference.
- **One noise model at a time.** `Wheel.noise_covariance` is a single slot, so two
  noise blocks (say instrument noise and galactic confusion) cannot each own a
  component and have enchilada combine them — the last block to write it
  wins. Sample them inside one noise block that publishes a combined model,
  or treat the confusion foreground as a signal block that returns it as a
  template, where the ledger combines contributions. A second block writing the
  covariance slot raises
  `NoiseOverwrittenWarning`. Blocks other than the current owner may omit a
  covariance publication to preserve the model; the owner must return its full
  covariance. Ownership changes remain warnings because they may be deliberate.
- **No automatic diagonal approximation.** A transformed covariance can correlate
  different time samples, Fourier bins, or WDM pixels. Use its exact operations;
  native independent-point matrices express a separate modeling choice.
- **No checkpoint file format or lifecycle end.** Current sampler state lives in
  the ledger, but durable serialization and resource cleanup remain explicit.

## Scaling

Wheel stores pristine observations, a working residual, and each block's current
result. A persistent balanced tree caches sums of accepted signals. Updating or
excluding one block takes O(log B) array additions for B registered blocks;
residuals are formed from pristine observations, avoiding cumulative add-back
drift. The tree uses up to one aggregate array per channel at each internal node.

Defensive state and covariance copies still cost memory and work. Internal copies
of validated covariance avoid repeating matrix factorizations; constructors and
new covariance publications retain validation. Keep unbounded chains in a callback
sink, outside `BlockResult`. See the [performance measurements](docs/performance.md)
and `benchmarks/benchmark_wheel.py` for a reproducible workload and memory costs.

## Status

0.2.0 — in development, with stateless blocks and
orchestrator-owned state. This is still an alpha; pin a version for a running
campaign. MIT licensed. Issues and questions welcome.
