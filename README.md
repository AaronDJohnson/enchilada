# turntable

[![CI](https://github.com/AaronDJohnson/turntable/actions/workflows/ci.yml/badge.svg)](https://github.com/AaronDJohnson/turntable/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Blocked-Gibbs global-fit orchestration for LISA.

A LISA global fit has to jointly infer many source populations (galactic
binaries, massive black-hole binaries, ...) plus the instrument noise, with
each block typically owned by a different group and sampler. turntable is the
orchestration layer — and only that. A `Wheel` keeps the pristine data and a
**ledger** of each segment's current model, and hands every registered
`Segment` the data minus *every other* segment — exactly the residual that
segment should fit. The segment fits it, subtracts its new model, and returns;
the Wheel reads the segment's new ledger entry off the difference. So blocked
Gibbs falls out of the ring, and there is no "add-back" for a segment to
forget. No waveforms, no likelihoods, and no samplers live here; those belong
to the segments (which own their sampler state and can wrap code in any
language), while the Wheel owns only the residual bookkeeping.

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
uv sync --extra examples         # adds jupyterlab for the notebooks
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
    channels=channels,
    tdi_generation="2.0",
    observable="fractional_frequency",
)   # n_samples is read off the arrays; epoch defaults to 0.0

ucb = EchoSegment(name="ucb")
mbhb = EchoSegment(name="mbhb")

wheel = Wheel(observed)
wheel.add(ucb)
wheel.add(mbhb)
wheel.run(n_iterations=3)

wheel.residual()   # the running residual: observed minus every segment's model
ucb.steps          # segment internals live on YOUR objects, not the Wheel
```

[`examples/demo.ipynb`](examples/demo.ipynb) is the same walkthrough with
commentary, plus the `Residuals` long/short name aliases (`Tobs`, `fs`, `dt`,
...), the typo catcher, and attaching a constellation ephemeris. For a real
(toy) sampler — two conjugate-Gibbs source segments plus a sampled
white-noise segment, converging to known truth — run
[`examples/toy_fit.py`](examples/toy_fit.py).

For a **real LISA source class**, [`examples/gb_segment_eryn.ipynb`](examples/gb_segment_eryn.ipynb)
fits an injected galactic binary through the Wheel using GBGPU waveforms, an
Eryn sampler living inside the segment, and a fixed LISA noise PSD from LISA
Analysis Tools. Everything that is *not* turntable lives in
[`examples/gb_model.py`](examples/gb_model.py), so the notebook shows only the
turntable touchpoints. That example needs the external LISA stack (`gbgpu`,
`eryn`, `lisaanalysistools`) plus `matplotlib`/`corner` for its plots — none of
which are turntable dependencies, so it is not exercised by CI. Its outputs are
not committed; run it to populate them.

## Plugging in your sampler

Implement the two-method `Segment` protocol — see the docstrings in
[`src/turntable/segment.py`](src/turntable/segment.py) for the full contract:

- `name` — unique within a Wheel; identifies you in diagnostics and errors.
- `start(residual) -> residual` — called once at registration; read the run
  settings off the residual, set yourself up, subtract your initial model,
  and return the updated residual (return it unchanged if you start from
  nothing).
- `step(residual) -> residual` — one Gibbs iteration. The residual you receive
  is the data with every **other** segment's model subtracted — *not* your
  own. So it is exactly the data your source class must explain: fit it
  directly, subtract your new model, and return the result. There is no
  add-back; the Wheel keeps the ledger and derives your new entry from what
  you return.

A segment that models the noise instead of a signal removes nothing from the
data; it returns the residual with an updated `noise` object —
`replace(residual, noise=my_model)` (so its ledger entry is zero) — and signal
segments read it back through `Residuals.noise_psd`. The Wheel stays entirely
noise-agnostic (see `NoiseSegment`).

Everything about your sampler is *yours*: parameters, RNG, posterior chains,
checkpoints, and your own current model all live inside your segment object
(or the external process it wraps) — the Wheel never sees or restores them. It
owns only the residual bookkeeping (the pristine data and the per-segment
ledger). To log progress or checkpoint,
pass an `on_sweep` callback to `run` (or equivalently call `run(1)` in your
own loop) and read `wheel.residual()` — or anything off your own segment
objects — between sweeps.

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
the first violation (a `start`/`step` that returns something other than a
valid `Residuals`, changes a fixed run setting, or — for a noise segment —
puts a model on the residual that fails the noise contract). It needn't check
the residual bookkeeping — the Wheel owns that — but whether your *sampler*
recovers truth is still yours to verify; `examples/toy_fit.py` is the pattern.

## Conventions and consistency checking

Cross-group runs fail through silently mismatched conventions, so turntable
makes every convention an explicit, validated part of `Residuals`:

- `observable` (required) — what the TDI samples physically are:
  `"fractional_frequency"`, `"phase"`, `"strain"`, or a campaign-agreed
  string. Every segment reads this one field instead of assuming.
- `domain` — `"time"` (default, `n_samples` real samples per channel) or
  `"frequency"` (one-sided `dt * rfft(x)` spectra of length
  `n_samples // 2 + 1`). `n_samples` always counts time-domain samples, so
  `Tobs`/`df`/`dt` and the PSD grid stay well defined in both. The residual a
  segment returns must keep the observed representation.
- `n_samples` — **omit it for time-domain data**; the arrays carry it exactly,
  so turntable reads it off them. Frequency-domain data must state it: an rfft
  loses the parity of n (513 bins fit both n=1024 and n=1025, which mean
  different `Tobs` and `df`), so turntable asks rather than silently guessing.
- `channels` — names imply the campaign's normalized definitions
  (e.g. A = (Z − X)/√2); see the `Residuals` docstring.

And it checks consistency at every boundary, failing loudly rather than
producing quietly wrong science:

- `Residuals` validates itself on every construction: tdi keys must equal
  `channels`, array lengths must match `domain`/`n_samples`, and an attached
  orbit must span the observation (catching GPS-vs-zero-based epoch
  mismatches at construction, not mid-run).
- The `Wheel` validates each segment fully **before** registering it (a
  failed `add` changes nothing), and re-validates the residual returned by
  every `start`/`step`: it must be a `Residuals` that kept the fixed run
  settings, and `Residuals` itself rejects wrong tdi shapes — so a mid-run
  drift raises immediately instead of corrupting the next segment's residual.
  A noise model is checked where it is consumed (`noise_psd` raises if it
  lacks a `psd` method).
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
uv sync --extra numeric-orbits   # dev group (pytest, pytest-cov, ruff, mypy)
uv run pytest                    # full suite, incl. examples and orbit loaders
uv run pytest --cov --cov-report=term-missing   # coverage (gate: 95%)
uv run mypy                      # turntable ships py.typed; keep it honest
uv run ruff check src tests examples
```

CI runs lint, mypy, the suite behind a 95% coverage gate, artifact builds, and
an installed-wheel smoke test across Python 3.12/3.13 on both Linux and macOS
for every push and pull request. See
[CHANGELOG.md](CHANGELOG.md) for release notes.

## Status

Pre-release (0.1.0, unreleased) under active development: interfaces may
still move until the first tag. MIT licensed. Issues and questions welcome.
