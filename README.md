# turntable

[![CI](https://github.com/AaronDJohnson/turntable/actions/workflows/ci.yml/badge.svg)](https://github.com/AaronDJohnson/turntable/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Blocked-Gibbs global-fit orchestration for LISA.

A LISA global fit has to jointly infer many source populations (galactic
binaries, massive black-hole binaries, ...) plus the instrument noise, with
each block typically owned by a different group and sampler. turntable is the
orchestration layer — and only that. A `Wheel` cycles over registered
`Segment`s, handing each one the observed data with every *other* segment's
current model subtracted, so each sampler works against a clean residual. No
waveforms, no likelihoods, and no samplers live here; those belong to the
segments, which can wrap code written in any language.

## Install

Requires Python ≥ 3.12 (floor set by lisaorbits). With
[uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/AaronDJohnson/turntable.git
cd turntable
uv sync
```

or with pip: `pip install -e .`

The core package depends only on numpy. Loading tabulated spacecraft
ephemerides (`turntable.orbits.NumericOrbit`) needs the extra:

```sh
uv sync --extra numeric-orbits   # adds h5py, scipy, lisaorbits
uv sync --extra examples         # adds jupyterlab for the demo notebook
```

## Quickstart

```sh
uv run python examples/demo.py
```

runs three Gibbs sweeps over two no-op `EchoSegment`s on synthetic data —
enough to watch the Wheel hand each segment its residual. The whole thing is:

```python
import numpy as np
from turntable import Residuals, Wheel
from turntable.testing import EchoSegment

rng = np.random.default_rng(0)
n_samples = 1024
channels = ("A", "E", "T")

# One frozen object holds the TDI arrays and the run settings everyone shares.
observed = Residuals(
    tdi={ch: rng.standard_normal(n_samples) for ch in channels},
    sample_rate=0.1,
    n_samples=n_samples,
    channels=channels,
    tdi_generation="2.0",
    observable="fractional_frequency",
    epoch=0.0,
)

ucb = EchoSegment(name="ucb")
mbhb = EchoSegment(name="mbhb")

wheel = Wheel(observed)
wheel.add(ucb)
wheel.add(mbhb)
wheel.run(n_iterations=3)

wheel.residual()   # observed minus every segment's current contribution
ucb.steps          # segment internals live on YOUR objects, not the Wheel
```

[`examples/demo.ipynb`](examples/demo.ipynb) is the same walkthrough with
commentary, plus the `Residuals` long/short name aliases (`Tobs`, `fs`, `dt`,
...), the typo catcher, and attaching a constellation ephemeris. For a real
(toy) sampler — two conjugate-Gibbs source segments plus a sampled
white-noise segment, converging to known truth — run
[`examples/toy_fit.py`](examples/toy_fit.py).

## Plugging in your sampler

The Wheel passes residuals — that's it. Implement the two-method `Segment`
protocol — see the docstrings in
[`src/turntable/segment.py`](src/turntable/segment.py) for the full contract:

- `name` — unique within a Wheel; identifies you in diagnostics and errors.
- `start(observed) -> {channel: array}` — called once at registration; read
  run settings off `observed`, set yourself up, and return your initial TDI
  contribution (zeros if you start from nothing).
- `step(residual) -> {channel: array}` — one Gibbs iteration against the
  data with every other segment's contribution already subtracted; update
  yourself and return your new contribution.

A segment that also models the noise implements one extra method,
`noise_model()`; the Wheel threads its return value onto `Residuals.noise`
so every other segment whitens against the current noise estimate (see
`NoiseSegment` and `Residuals.noise_psd`).

Everything about your sampler is *yours*: parameters, RNG, posterior
chains, checkpoints all live inside your segment object (or the external
process it wraps) — the Wheel never sees them, and never restores them.
The only bookkeeping the Wheel does is the residual ledger: each segment's
latest contribution, which is exactly what it needs to hand out residuals.
To log progress or checkpoint, pass an `on_sweep` callback to `run` (or
equivalently call `run(1)` in your own loop) and read `wheel.residual()` —
or anything at all off your own segment objects — between sweeps.

The Wheel does not care how you sample or what language your sampler is
written in — a thin Python wrapper that shells out, moves files, and
implements these methods is indistinguishable from a native segment.

Before plugging a segment into a shared campaign, run the conformance check
in your own test suite:

```python
from turntable.testing import check_segment
check_segment(MySegment(name="ucb"), toy_observed)
```

It drives the full protocol on a scratch Wheel and raises a pointed error at
the first violation (wrong contribution shapes, malformed returns, mutating
the residual, a noise model missing the contract).

## Conventions and consistency checking

Cross-group runs fail through silently mismatched conventions, so turntable
makes every convention an explicit, validated part of `Residuals`:

- `observable` (required) — what the TDI samples physically are:
  `"fractional_frequency"`, `"phase"`, `"strain"`, or a campaign-agreed
  string. Every segment reads this one field instead of assuming.
- `domain` — `"time"` (default, `n_samples` real samples per channel) or
  `"frequency"` (one-sided `dt * rfft(x)` spectra of length
  `n_samples // 2 + 1`). `n_samples` always counts time-domain samples, so
  `Tobs`/`df`/`dt` and the PSD grid stay well defined in both. Segment
  contributions must match the observed representation.
- `channels` — names imply the campaign's normalized definitions
  (e.g. A = (Z − X)/√2); see the `Residuals` docstring.

And it checks consistency at every boundary, failing loudly rather than
producing quietly wrong science:

- `Residuals` validates itself on every construction: tdi keys must equal
  `channels`, array lengths must match `domain`/`n_samples`, and an attached
  orbit must span the observation (catching GPS-vs-zero-based epoch
  mismatches at construction, not mid-run).
- The `Wheel` validates each segment fully **before** registering it (a
  failed `add` changes nothing), re-validates every contribution after every
  step (mid-run shape drift raises instead of silently broadcast-corrupting
  other segments' residuals), and checks that a noise segment's model
  actually satisfies the noise contract (`psd(freqs[, channel])` and/or
  `wdm_variance(...)`).
- `NumericOrbit.positions` refuses to extrapolate outside its tabulated
  ephemeris instead of returning cubic-polynomial garbage.

## Orbits

The constellation ephemeris the data was produced with rides on
`Residuals.orbit` so every segment builds its response from the *same*
spacecraft positions. `turntable.orbits.NumericOrbit` tabulates and
cubic-spline-interpolates an ephemeris, with loaders for LDC/Mojito-style
HDF5 files (`from_hdf5`) and lisaorbits objects (`from_lisaorbits`); both
need the `numeric-orbits` extra. See the module docstring in
[`src/turntable/orbits.py`](src/turntable/orbits.py) for frames and
conventions.

## Development

```sh
uv sync --extra numeric-orbits   # dev group (pytest, ruff) installs by default
uv run pytest                    # full suite, incl. examples and orbit loaders
uv run ruff check src tests examples
```

CI runs lint, tests, and artifact builds on Python 3.12 and 3.13 for every
push and pull request. See [CHANGELOG.md](CHANGELOG.md) for release notes.

## Status

Pre-release (0.1.0, unreleased) under active development: interfaces may
still move until the first tag. MIT licensed. Issues and questions welcome.
