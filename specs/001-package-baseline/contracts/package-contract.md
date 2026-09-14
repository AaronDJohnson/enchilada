# Package Contract: enchilada

This companion to the [baseline specification](../spec.md) records the exact public interfaces and scientific conventions at commit `dd412c3d47d3a209b632f43ee3220215936a31cc`. It is a reference for compatibility and acceptance checks, not an implementation plan.

## Public Surface

The package root exports `Block`, `NoiseBlock`, `NoiseOverwrittenWarning`, `NumericOrbit`, `Orbit`, `L1Data`, `Template`, `Wheel`, `__version__`, and `replace`. The replacement helper is `dataclasses.replace`; the version comes from installed distribution metadata, falling back to `0+unknown` when that metadata is absent. The package includes `py.typed`.

`EchoBlock` and `check_block` are imported from `enchilada.testing`. `Block` and `Orbit` support structural runtime checks. Such checks establish member presence only; they do not validate scientific behavior. `NoiseBlock` adds no distinguishing members to `Block`, so runtime checks cannot identify a block's noise role.

Sources: [exports](../../../src/enchilada/__init__.py), [protocols](../../../src/enchilada/block.py), [typing tests](../../../tests/test_typing.py).

## Observation and Residual Data

### Construction

`L1Data(tdi, sample_rate, channels, tdi_generation, observable, n_samples=0, epoch=0.0, domain="time", noise=None, orbit=None)`

Construct by keyword. `L1Data` is a frozen dataclass with identity equality; comparing objects does not compare array contents. Its channel dictionary, arrays, noise model, and orbit are not deeply frozen or defensively copied at construction. `replace` constructs a new object and runs validation again.

| Field | Contract |
| --- | --- |
| `tdi` | Dictionary from channel name to a one-dimensional NumPy array; keys exactly match `channels`. |
| `sample_rate` | Positive finite sampling rate in Hz. |
| `channels` | Non-empty tuple without duplicates; a supplied list is normalized to a tuple. Channel names are campaign-provided labels. |
| `tdi_generation` | Non-empty string, such as `"1.5"` or `"2.0"`; no enumeration of physically supported generations is enforced. |
| `observable` | Required non-empty string. Recommended values are `"fractional_frequency"`, `"phase"`, and `"strain"`; campaign-specific strings are allowed. |
| `n_samples` | Resolved positive integer time-sample count, also when storing a spectrum. Zero is the omitted-value sentinel: derive in the time domain and reject in the frequency domain. |
| `epoch` | Finite seconds on the campaign's absolute clock, conventionally GPS seconds; zero is useful for synthetic data. |
| `domain` | Exactly `"time"` or `"frequency"`. |
| `noise` | Current shared noise object, or `None`. Its spectrum contract is checked on consumption. |
| `orbit` | Shared orbit object, or `None`. A `t_range`, if provided, is checked against the sample span. |

Time-domain arrays contain `N` real floating-point samples. Frequency-domain arrays contain `N // 2 + 1` complex samples. Integer and object arrays are unsupported. Floating-point precision need not be identical across arrays or templates.

`L1Data` validates structure and conventions but does not reject non-finite sample values itself. `Wheel(observed)` rejects non-finite observed data, and each block return is checked separately. Container construction does not establish physical correctness of the observable, channel normalization, or TDI generation. No channel-basis transform is supplied.

### Grid and aliases

| Descriptive name | Short name | Value |
| --- | --- | --- |
| `observation_time` | `Tobs` | `N / fs`, in seconds |
| `sample_rate` | `fs` | Samples per second |
| `sample_interval` | `dt` | `1 / fs`, in seconds |
| `n_samples` | `N` | Number of time samples |
| `frequency_resolution` | `df` | `1 / Tobs`, in Hz |
| `nyquist_frequency` | `fny` | `fs / 2`, in Hz |
| `epoch` | `t0` | Time of the first sample |

`L1Data.aliases()` returns a copy of the name mapping. `DOMAINS`, `RECOMMENDED_OBSERVABLES`, and `ALIASES` expose the documented convention tables. Common misspellings raise `AttributeError` with a suggested name; other public-name misses point to the alias table. Unknown private attributes raise ordinary `AttributeError`. Typing information does not turn unknown attributes into valid dynamic members.

Samples are at `epoch + k * dt`, for `k = 0, ..., N - 1`. The last sample is `epoch + (N - 1) * dt`, not `epoch + Tobs`. The orbit-span check uses that same multiplication convention to avoid rejecting an exactly matching grid through rounding differences.

### Representation conversion

- `to_frequency() -> L1Data`: compute `X = dt * rfft(x)` independently for every channel and set `domain="frequency"`.
- `to_time() -> L1Data`: compute `x = irfft(X / dt, n=N)` independently for every channel and set `domain="time"`.
- A conversion to the current domain returns the same object.
- A real conversion preserves `N`, sample rate, epoch, channel ordering, TDI generation, observable, and the noise/orbit references.
- The positive-frequency grid is `rfftfreq(N, d=dt)`, with bin spacing `df`. An odd-length transform has no exact Nyquist bin.
- The round-trip guarantee starts from valid real time data. Arbitrary complex DC or even-length Nyquist coefficients are not checked for real-signal consistency; inverse transformation cannot preserve their imaginary components.
- No additional epoch-dependent phase factor, resampling, window, or gap treatment is applied.

Source: [data.py](../../../src/enchilada/data.py).

## Templates and Blocks

### Template interfaces

| Interface | Behavior |
| --- | --- |
| `Template(tdi, noise=None)` | Frozen dataclass with identity equality. Checks a non-empty dictionary containing one-dimensional floating or complex arrays. Construction alone does not check the campaign grid or finiteness. |
| `residual.template(tdi) -> Template` | Builds a template against the residual's channel keys, lengths, and real/complex representation; keeps the supplied arrays without copying. |
| `residual.zero_template() -> Template` | Allocates fresh zero arrays with each residual channel's shape and dtype; publishes no noise model. |
| `template.with_noise(noise) -> Template` | Creates a new template container carrying the supplied noise reference and sharing the existing signal arrays. |

A template represents the sum of one block's current modeled sources. All campaign channels must be present, including channels where the signal is zero. The template has no grid metadata of its own; the residual factory and campaign boundary establish grid agreement. Finite values are enforced at the campaign boundary, not by the factories.

### Block interfaces

`Block.name: str`

`Block.start(residual: L1Data) -> Template`

`Block.update(residual: L1Data) -> Template`

The name is a non-empty string unique within its Wheel and must remain stable. A block validates any unsupported observable, channel, domain, generation, orbit, or noise requirements during initialization. The package cannot inspect the physical meaning of a returned waveform.

`start` initializes participation and returns an initial template. `update` advances the author's inference work and returns the new template. A noise-only block follows the same protocol and returns `residual.zero_template().with_noise(new_model)`.

The block owns its sampler state, including parameters, randomness, posterior history, proposal adaptation, its own template buffer, checkpoint files, and any wrapped external process. There is no serialization or process-launch protocol in the package and no `close`/`finalize` lifecycle method.

Sources: [template.py](https://github.com/AaronDJohnson/enchilada/blob/dd412c3d47d3a209b632f43ee3220215936a31cc/src/enchilada/template.py), [block.py](../../../src/enchilada/block.py).

## Wheel Lifecycle and Accounting

| Interface | Behavior |
| --- | --- |
| `Wheel(observed: L1Data)` | Reject non-finite observed samples; retain the observation reference and its initial noise reference; begin with no blocks or contributions. |
| `add(block: Block) -> None` | Validate name, uniqueness, and both callable operations; call `start`; validate the returned template; record block order, copied signal, and any noise publication. |
| `run(n_cycles: int, on_cycle=None) -> None` | Update each block once per cycle in registration order; accept Python/NumPy integer counts, excluding Booleans, with count at least zero. |
| `residual(exclude: str \| None = None) -> L1Data` | Return fresh arrays for the full residual or the residual leaving out subtraction of the named block; attach the current noise model. |
| `contribution(name: str) -> dict[str, ndarray]` | Return a copied dictionary of the named block's adopted signal arrays. |
| `observed` | Public reference to the original observation; callers must preserve its contents. |

Let `d` denote an observed channel array and `T_j` the last accepted template for block `j`. Independently for each channel:

```text
Full residual:                 r = d - sum_j(T_j)
Residual handed to block i:   r_i = d - sum_{j != i}(T_j)
After a successful update:   T_i = copy(returned_template_i)
```

Noise updates affect the noise slot, not this subtraction. A zero template is an explicit valid contribution. The ledger stores the returned signal directly rather than reconstructing it from differences, avoiding cancellation when the signal is much smaller than the observation.

During registration, only already registered blocks contribute to the residual. During a cycle, blocks later in the order see the accepted changes from earlier blocks in that cycle. Consecutive `run` calls continue with the same blocks, ledger, and noise; there is no restart or new initialization.

`on_cycle(cycle, wheel)` runs after a completed pass, with indices starting at zero for each `run` call. The callback may inspect the Wheel and the caller's block objects but must not mutate the campaign. With no blocks, requested cycles still generate callbacks. With zero cycles, none are called.

### Ownership and failure boundaries

- Each handed residual contains freshly formed arrays. Blocks may reuse or overwrite those arrays to construct their own template.
- Ledger adoption copies each template array. Editing a returned buffer later does not edit the stored contribution. Contribution inspection also returns copies.
- Residual arithmetic promotes each channel to a dtype capable of representing the observed array and the included contributions. The original observation and ledger dtypes remain unchanged.
- The original observed arrays, noise objects, and orbit object remain shared references. The caller preserves observed data; all participants treat shared models as immutable. Frozen dataclasses do not enforce deep immutability.
- Initialization or return-validation failure occurs before committing registration. This protects Wheel membership and accounting, not side effects inside a block.
- A failed update or invalid return leaves that block's prior ledger entry in place and prevents later updates and the incomplete-cycle callback. Earlier successful updates and all block-internal mutations remain.
- Callback exceptions propagate after that cycle's updates have been adopted. There is no automatic retry, checkpoint recovery, transaction across a whole cycle, or lifecycle cleanup.
- Registration/adoption is not transactional for exceptions raised while committing state. In particular, promoting `NoiseOverwrittenWarning` to an exception can interrupt adoption after the signal entry has been recorded. This limitation is recorded in [traceability](../traceability.md).

Source: [wheel.py](../../../src/enchilada/wheel.py).

## Noise Contract

Noise objects expose `psd(freqs[, channel])`. `noise_psd()` calls `psd(positive_frequencies)`; `noise_psd(channel)` passes the channel as the second positional argument. A model accepting only the first form supports the default accessor but not necessarily channel-specific access. The package does not validate channel names at this accessor.

`L1Data.noise_psd(channel=None) -> ndarray | None` returns the one-sided spectrum on the observation's `N // 2 + 1` grid, in `[observable]² / Hz`. The model sees positive frequencies only. The returned DC entry is set to positive infinity, and every non-DC value must be finite and greater than zero. Assignment follows NumPy broadcasting; scalar model output can fill the positive-frequency grid. No additional output-shape validation is promised beyond that assignment.

For an interior bin under the package's transform convention:

```text
X(f) = dt * rfft(x)
E[|X(f)|²] = (Tobs / 2) * S(f)
White noise: S(f) = 2 * sigma² / fs
```

DC has zero likelihood weight. For even `N`, the Nyquist coefficient has one real degree of freedom; a frequency-domain likelihood must half-weight or omit it rather than treating it as an interior complex bin. Likelihood construction remains outside the package.

`L1Data.noise_variance(channel=None) -> float | None` computes:

```text
variance = df * sum_{k=1..floor(N/2)} w_k * S_k
w_k = 1, except w_(N/2) = 1/2 when N is even
```

For white noise this is `sigma² * (1 - 1/N)`, the expected variance after excluding the sample mean. A one-sample observation has no positive-frequency bins and therefore zero variance. With no model, both accessors return `None`. Missing callable `psd` raises `TypeError`; invalid non-DC values raise `ValueError`.

The noise slot starts with `observed.noise`. `Template.noise is None` publishes nothing and cannot clear the current slot. A non-`None` publication replaces it and establishes the publishing block as owner. Repeated publications by that owner and the first block publication over initial observed noise are silent. A different publishing block emits `NoiseOverwrittenWarning`, a `RuntimeWarning` subclass naming both owners; under normal warning handling its model becomes current. Multiple component models must be combined by their owner, not by the Wheel.

Sources: [data.py](../../../src/enchilada/data.py), [wheel.py](../../../src/enchilada/wheel.py).

## Orbit Contract and Loaders

`Orbit` exposes `L` in metres, `fstar` in Hz, and `positions(t) -> (x, y, z)`. For a one-dimensional vector of `m` query times, each returned coordinate has shape `(3, m)`, indexed by spacecraft then time. Queries use absolute seconds on the observation's clock. Output is in ecliptic Cartesian metres. `NumericOrbit` also accepts a scalar query, producing coordinate arrays of shape `(3, 1)`.

An optional `t_range = (first_time, last_time)` enables the observation's coarse coverage check. Without that member the early check is skipped. Response authors must allow margin for retarded-time queries beyond the observation span.

| Constructor or loader | Inputs and defaults |
| --- | --- |
| `NumericOrbit(times, positions, *, L=None, fstar=None)` | Ascending times of shape `(n,)`, positions of shape `(3, n, 3)` in ecliptic metres; constructs cubic-spline interpolation. Valid tables need at least two finite, strictly increasing times and finite positions. |
| `from_arrays(times, sc_positions, *, frame="ecliptic", L=None, fstar=None)` | Same axes; accepts ecliptic or equatorial input. |
| `from_hdf5(path, *, group="orbits", position_datasets=("sc_position_1", "sc_position_2", "sc_position_3"), frame="equatorial", L=None)` | Reads the named group, its `sampling` object's `t0`, `dt`, and `size` attributes, and three `(n, 3)` position datasets; constructs times as `t0 + arange(size) * dt`. |
| `from_lisaorbits(orbits, times, *, frame="equatorial")` | Calls `compute_position(times)`; accepts `(n, 3, 3)` or flattened `(n, 9)` with time first, then converts to spacecraft-first storage. Other returned layouts are rejected. |

Spline construction supplies validation for time ordering and finiteness. Error types/messages for every malformed table rank or override value are not normalized by the package. An explicit `L` or `fstar` is accepted where offered without a separate positivity/consistency check.

Without `L`, the nominal arm length is the mean of the three pairwise spacecraft separations over the table. A non-positive or non-finite mean falls back to `2.5e9` metres. Without `fstar`, the transfer frequency is `299792458 / (2 * pi * L)` Hz. `t_range` reports the first and last tabulated times. Queries strictly outside those endpoints raise `ValueError` rather than extrapolate; endpoint queries are accepted.

Equatorial conversion uses the baseline obliquity `epsilon = 0.40909280422232897` radians:

```text
x_ecliptic = x_equatorial
y_ecliptic = cos(epsilon) * y_equatorial + sin(epsilon) * z_equatorial
z_ecliptic = -sin(epsilon) * y_equatorial + cos(epsilon) * z_equatorial
```

The numerical orbit supports caller-supplied analytic objects by sampling them through the external-orbit loader. There is no built-in analytic constellation model, spacecraft-velocity interface, or light-travel-delay solver.

Source: [orbits.py](../../../src/enchilada/orbits.py).

## Conformance and Examples

`check_block(block, observed, n_cycles=2) -> None` constructs a scratch Wheel, registers the actual supplied block, runs the requested cycles, and then evaluates `noise_psd()` if the final noise object is non-`None` and is not the original `observed.noise` object. Errors propagate; a conforming block returns normally.

The helper mutates the supplied block's real internal state and does not clone or restore it. Use a disposable block instance. It checks signal returns on every exercised call, but checks only the final changed noise model through the default channel interface. It does not validate every published noise model, every channel, unchanged initial noise, convergence, posterior quality, or protocol behavior on data it was not given.

`EchoBlock(name)` starts with `updates=0`, prints run context during `start`, and prints the first channel's RMS magnitude during `update`. It increments its counter once per update and always returns a zero template. The RMS uses absolute magnitude so frequency-domain diagnostics remain real.

| Artifact | Role and validation scope |
| --- | --- |
| [Minimal script](../../../examples/demo.py) | Three cycles, two no-op blocks, synthetic A/E/T data; exercised by tests. |
| [Annotated notebook](../../../examples/demo.ipynb) | The same integration plus aliases, typo hints, and an orbit; code cells exercised when SciPy is available. |
| [Toy fit](../../../examples/toy_fit.py) | Two sinusoidal signal blocks and sampled white noise, with private chains and a progress callback; seeded truth recovery is tested. Default run: 300 cycles, 100 burn-in, seed 0; acceptance fixture: 200/80/0. |
| [Galactic-binary notebook](../../../examples/gb_block_eryn.ipynb) and [model](../../../examples/gb_model.py) | Example A/E frequency-domain fractional-frequency model with GBGPU waveforms, an Eryn sampler, and fixed LISA noise. Four source parameters vary while sky/orientation angles are fixed. External stack and scientific recovery are outside automated CI. |

Source: [testing.py](../../../src/enchilada/testing.py). Example classes are not part of the package's supported public exports.

## Packaging and Quality Configuration

These values describe the checked-in project configuration, not an independently verified inventory of current upstream releases.

| Area | Baseline configuration |
| --- | --- |
| Package | `enchilada`, metadata version `0.1.0`, MIT license, Python `>=3.12` |
| Core runtime | `numpy>=1.23` |
| `numeric-orbits` extra | `h5py>=3.0`, `lisaorbits>=3.0.3`, `scipy>=1.8`; orbit-related imports are lazy |
| `examples` extra | `jupyterlab>=4` |
| Development tools | mypy, pytest, pytest-cov, ruff, with floors declared in project metadata |
| Build | `uv_build>=0.11.32,<0.12.0`; wheel includes the typing marker; sdist also includes Python tests, example scripts/notebooks/requirements, and changelog |
| Full CI | Python 3.12/3.13 on Linux and macOS; lint, format, mypy, tests at a 95% branch-inclusive coverage floor, build, installed-wheel smoke check |
| Core-only CI | Python 3.12/3.13/3.14 on Linux without numerical-orbit extras; optional tests skip when their dependencies are unavailable |
| Dependency floors | Python 3.12 on Linux, resolving direct dependencies to declared lower bounds and running the suite |
| Source distribution CI | Unpack an sdist into a separate tree, install its dependencies, and run its packaged suite |
| Release | `v*` tags run the full test matrix, core-only and floor gates; require parsed tag/project version agreement; build and smoke-check artifacts; publish through PyPI Trusted Publishing. Manual workflow dispatch builds without publishing. |

The release workflow does not include the separate sdist-test job from main CI. This distinction is intentional in the description here; no unexecuted gate is implied.

The external galactic-binary example uses [requirements-gb.txt](../../../examples/requirements-gb.txt), not a package extra. That file records Python 3.12–3.13 on Linux x86_64/aarch64 or macOS arm64 as its supported wheel configurations, with no recorded support for Intel macOS, Windows, or Python 3.14. Recheck upstream distribution availability before expanding that example's support claim.

The package coordinates local in-memory data. It supplies no campaign data files, persistence service, automatic background work, authentication system, or network transport. External blocks may provide their own transport or processes.

For `B` blocks, `C` channels, and `M` stored samples or frequency bins per channel, the current residual accounting uses approximately `O(B² C M)` arithmetic per cycle and `O(B C M)` ledger storage, excluding block internals and temporary arrays. This is an implementation characteristic, not a required algorithm. The README's older laptop timings predate the template-return change and are not acceptance thresholds.

Sources: [project configuration](../../../pyproject.toml), [CI](../../../.github/workflows/ci.yml), [release workflow](../../../.github/workflows/release.yml), [README](../../../README.md), [changelog](../../../CHANGELOG.md).
