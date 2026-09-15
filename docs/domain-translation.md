# Domain translation

Use `transform` to convert observations, covariance, and block results between
`"time"`, `"frequency"`, and `"wdm"`. Wheel can also select a representation for
each registered block and perform these conversions at its boundary.

## Install the optional WDM backend

Time/Fourier translation uses NumPy and works with the core package on Python
3.12 or newer. WDM translation additionally uses the separately developed
[`AaronDJohnson/wdm`](https://github.com/AaronDJohnson/wdm) backend. The tested
backend is version `0.2.0`, revision
`6f145b4076862ea4786150a54d5a1f28af3c7055`. It requires Python 3.13 or newer,
NumPy `>=2.1,<3`, and SciPy `>=1.14.1,<2`; its isolated build uses
`hatchling==1.30.1`. WDM stays optional and is not a core dependency or an
enchilada extra.

From the enchilada checkout, install the tested revision explicitly:

```sh
uv sync --python 3.13 --locked --extra numeric-orbits
uv pip install "wdm @ git+https://github.com/AaronDJohnson/wdm.git@6f145b4076862ea4786150a54d5a1f28af3c7055"
uv run --no-sync python examples/domain_translation.py
uv run --no-sync pytest -q tests/test_translation_example.py
```

The numeric-orbits extra supplies the notebook's plotting dependencies as well
as its orbit support. For local backend development, use an editable checkout
alongside enchilada instead:

```sh
uv sync --python 3.13
uv pip install -e ../wdm
uv run --no-sync python examples/domain_translation.py
```

Adjust `../wdm` to the local checkout location. Install it after syncing the
project, and use `--no-sync` to preserve the prepared environment. An exact
`uv sync` removes packages outside the project lockfile. Source checkouts without
installed distribution metadata may report `0.0.0.dev0`; runtime compatibility
checks require the backend's batched transform API rather than a version string.
Other backend revisions need numerical revalidation before being treated as
supported.

The [WDM integration workflow](../.github/workflows/wdm.yml), called by CI and
release gates, checks out this exact revision on Python 3.13, installs it into the
locked test environment, and checks the installed version and required API before
running the numerical translation suite, script, and notebook. Any skipped test
in that integration suite fails the gate. Core-only jobs keep WDM optional and
may skip those tests.

## One wrapper for all three object types

```python
from enchilada import transform

spectrum = transform(observed, "frequency")
wdm_data = transform(observed, "wdm", num_frequency_divisions=8)
restored = transform(wdm_data, "time")
```

The complete signature is:

```text
transform(
    value,
    target_domain,
    *,
    num_frequency_divisions=None,
    num_time_divisions=None,
    reference_data=None,
)
```

`value` can be `L1Data`, `DataCovariance`, `TranslatedCovariance`, or `BlockResult`.
Conversions return independent snapshots, including when the requested domain is
the current domain. They preserve channel order, sample rate, underlying sample
count, GPS start time, physical observable, TDI generation, and the shared orbit
resource. Numerical round trips agree to floating-point precision.

A `BlockResult` deliberately has no observation-grid fields. Supply its source
`L1Data` through `reference_data` when translating a result containing signal
arrays:

```python
wdm_result = transform(
    current_block_result,
    "wdm",
    num_frequency_divisions=8,
    reference_data=observed,
)
```

Its signal and covariance are translated together; `model_parameters`,
`sampler_state`, and `metadata` are copied without changing their numerical
interpretation. An absent signal stays absent. A result containing only covariance
can use that covariance's own grid metadata.

## Choose the WDM grid

The two division counts obey

```text
num_time_samples = num_frequency_divisions * num_time_divisions
```

Both counts must be positive even integers. On the first conversion into WDM,
provide either count and the wrapper derives the other. Providing both is allowed
only when they agree with the existing time sample count. No implicit padding,
trimming, or guessing is performed.

For 128 time samples, either of these chooses a valid grid:

```python
fine_time = transform(observed, "wdm", num_frequency_divisions=8)  # 8 * 16
fine_frequency = transform(observed, "wdm", num_time_divisions=8)  # 16 * 8
```

Each WDM channel is a real `float64` array of shape
`(num_frequency_divisions + 1, num_time_divisions)`. The extra frequency row is
part of the Wilson basis's edge layout. Structurally inactive edge entries are
zero; they do not add degrees of freedom.

`L1Data.wdm_grid` is a frozen `WDMGrid` with `num_frequency_divisions`,
`num_time_divisions`, `num_time_samples`, `array_shape`, and an independent
`active_mask`. Its mask identifies the structural coefficients belonging to the
basis. A WDM `L1Data` must carry this metadata; time and frequency data do not.

Calling `transform(wdm_data, "wdm")` reuses its grid and returns a copy. Supplying
a new division count requests regridding through the same underlying data.
`L1Data.to_time()` and `to_frequency()` are available as conveniences;
`transform` is the common interface for observations, covariance, and results.

## Covariance stays an exact operator

A native `DataCovariance` stores channel covariance independently at each point
of its declared representation. Its matrix shape is `(N, C, C)` in time,
`(N // 2 + 1, C, C)` in frequency, and `(Nf + 1, Nt, C, C)` in WDM. Native WDM
covariance carries `wdm_grid`; its statistical `active_mask` can select a subset
of the structurally active coefficients.

Transforming an independent-point covariance can introduce correlations between
different points. `TranslatedCovariance` preserves the native covariance and
applies the coordinate transforms around its operations. It does not create a
full-observation dense matrix or silently substitute independent target-pixel
variances. Converting back to the original native grid returns a native
`DataCovariance` copy. Repeated translations keep one native model.

Both covariance types provide these operations on dictionaries of channel arrays:

| Operation | Meaning |
| --- | --- |
| `apply(values)` | Apply the covariance operator. |
| `solve(values)` | Apply its inverse on the active subspace, with zero weight in excluded directions. |
| `project(values)` | Project onto the active subspace. |
| `quadratic_form(values)` | Evaluate the covariance-weighted squared residual. |
| `log_determinant()` | Log-pseudodeterminant over active real coordinates. |
| `degrees_of_freedom` | Number of active real coordinates. |

`apply`, `solve`, and `project` return channel dictionaries on the covariance's
current grid. `check_compatible(reference_data)` checks the shared sampling and
channel metadata. `TranslatedCovariance.active_mask` is `None`: the translated
mask can describe a nonlocal subspace rather than a set of pixels, and `project`
is the appropriate interface. The WDM structural mask alone cannot describe a
translated Fourier DC exclusion or other native statistical masks.

```python
from enchilada import DataCovariance, transform

noise = DataCovariance.from_variance(observed, 0.25)
wdm_noise = transform(noise, "wdm", num_frequency_divisions=8)
q_time = noise.quadratic_form(observed.channel_data)
q_wdm = wdm_noise.quadratic_form(wdm_data.channel_data)
```

`q_time` and `q_wdm` agree to numerical precision. Native marginal PSD/variance
helpers are conveniences for their supported representations; they do not
replace the full covariance operator in a correlated likelihood.

Enchilada spectra use `X = dt * rfft(x)`. The wrapper accounts for the WDM backend's
unscaled-rFFT input convention and uses its supported double precision. Covariance
transforms with the coordinate map as `C_new = A C A†`. In normalized real
coordinates, Fourier interiors contribute separate `sqrt(2)`-scaled real and
imaginary coordinates; DC and even-length Nyquist each contribute one real
coordinate. The corresponding coordinate scales are `1` for time, `dt * sqrt(N)`
for frequency, and `sqrt(dt)` for WDM. Covariance, inverse action, projection, and
log-pseudodeterminant all account for these scales. A quadratic form is invariant;
a log determinant includes the coordinate Jacobian and is not generally invariant.

## Give each block its own representation

```python
wheel.add(
    time_block, initial_block_result=initial_time_result, data_domain="time"
)
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

`Wheel.add(block_to_register, *, initial_block_result, data_domain=None,
num_frequency_divisions=None, num_time_divisions=None)` records a fixed
representation selection for that block.
An omitted domain keeps the default handoff contract. Explicit selection converts
its conditional residual, covariance, and previous signal together before calling
`sample`. Blocks receive `L1Data` plus native or translated
covariance, and use the same `BlockResult` contract in every representation.

Registration requires a complete `initial_block_result`; omitting it or passing
`None` is an error. Wheel validates and adopts that result without calling model
code or consuming Wheel RNG draws. The application chooses the initial estimate
and prepares the state its block expects. Any optional prior draw belongs in
application or model setup, using a separate setup RNG. The initial result's
signal arrays must already use the selected block domain and WDM grid,
or the observation grid when `data_domain` is omitted. For example, prepare a
frequency-domain signal using that domain's conventions before registration:

```python
block_data = transform(observed, "frequency")
initial_result = block_data.block_result(
    tdi_signal_contribution=initial_frequency_signal,
    model_parameters=initial_parameters,
    sampler_state=initial_sampler_state,
)
wheel.add(
    frequency_block,
    initial_block_result=initial_result,
    data_domain="frequency",
)
```

The result has no signal-grid metadata of its own, so Wheel interprets it using
the registration choice. Optional covariance carries its own grid metadata and
may start in a different domain; Wheel converts it normally. This also permits a
noise-only initial result. Supply all model and continuation fields expected by
`sample`; parameters, signal, and sampler state must describe the same starting
estimate. Wheel checks the exchange contract, not physical consistency or prior
support. The Block protocol requires only `name` and `sample`.

Signals in the ledger and `Wheel.working_residual` use the canonical observation
grid. An explicit `Wheel.add(data_domain=...)` selection also converts returned
covariance to that grid before adoption, including covariance in a supplied
initial result. With the default handoff (`data_domain=None`), covariance retains
its declared native domain. Parameters and sampler
continuation state remain block-local numerical values; the wrapper does not
reinterpret them. Invalid output or a failed
conversion leaves that block's accepted result, residual, covariance, and RNG
unchanged.

Starting from an injection changes the initial estimate and residual; it preserves
the observations and prior, and the block continues sampling normally. This is
one block's initialization, not a full campaign checkpoint or a Wheel RNG restore.

Complete-result rules still apply: an absent signal removes the previous signal;
the current covariance owner must publish its full covariance on every call;
other blocks can omit covariance to preserve the global value. Covariance
ownership is a single replaceable model, with a warning on takeover.

The runnable `examples/domain_translation.py` shows both WDM division selectors,
round trips, covariance-weight invariance, and a Wheel containing time, Fourier,
and WDM blocks. Its optional test runs against the real WDM backend.

For an interactive walkthrough with plots, open
[`examples/domain_translation.ipynb`](../examples/domain_translation.ipynb).
The notebook includes setup commands and self-contained cells for each step.
For a walkthrough of implementing the block itself, start with the
[Block authoring tutorial](../examples/create_block.ipynb).

## Limits

Translation preserves the supplied native covariance model. It does not infer a
noise model, introduce a diagonal approximation, or establish an arbitrary dense
covariance model from independent-point inputs. Native WDM covariance is an
explicit independent-pixel model; transforming a different native model need not
produce that structure.

WDM translation does not add data-gap handling, automatic combination of noise
components, durable checkpoints, or parallel block scheduling. Choose division
counts and the native covariance model for the scientific problem, and benchmark
operator calls at the intended observation size.
