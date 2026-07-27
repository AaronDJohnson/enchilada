# turntable

[![CI](https://github.com/AaronDJohnson/turntable/actions/workflows/ci.yml/badge.svg)](https://github.com/AaronDJohnson/turntable/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/AaronDJohnson/turntable/blob/main/LICENSE)

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

Requires Python ≥ 3.12 (floor set by lisaorbits).

```sh
pip install turntable      # or: uv add turntable
```

To work on turntable itself, clone it and use [uv](https://docs.astral.sh/uv/):

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

[`examples/demo.ipynb`](https://github.com/AaronDJohnson/turntable/blob/main/examples/demo.ipynb) is the same walkthrough with
commentary, plus the `Residuals` long/short name aliases (`Tobs`, `fs`, `dt`,
...), the typo catcher, and attaching a constellation ephemeris. For a real
(toy) sampler — two conjugate-Gibbs source segments plus a sampled
white-noise segment, converging to known truth — run
[`examples/toy_fit.py`](https://github.com/AaronDJohnson/turntable/blob/main/examples/toy_fit.py).

For a **real LISA source class**, [`examples/gb_segment_eryn.ipynb`](https://github.com/AaronDJohnson/turntable/blob/main/examples/gb_segment_eryn.ipynb)
fits an injected galactic binary through the Wheel using GBGPU waveforms, an
Eryn sampler living inside the segment, and a fixed LISA noise PSD from LISA
Analysis Tools. Everything that is *not* turntable lives in
[`examples/gb_model.py`](https://github.com/AaronDJohnson/turntable/blob/main/examples/gb_model.py), so the notebook shows only the
turntable touchpoints. That example needs the external LISA stack (`gbgpu`,
`eryn`, `lisaanalysistools`) plus `matplotlib`/`corner` for its plots — none of
which are turntable dependencies, so it is not exercised by CI. Its outputs are
not committed; run it to populate them.

## Plugging in your sampler

Implement the two-method `Segment` protocol — see the docstrings in
[`src/turntable/segment.py`](https://github.com/AaronDJohnson/turntable/blob/main/src/turntable/segment.py) for the full contract:

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

`replace` is re-exported for convenience (`from turntable import replace`), since
every segment needs it to return an updated residual.

A segment that models the noise instead of a signal removes nothing from the
data; it returns the residual with an updated `noise` object —
`replace(residual, noise=my_model)` (so its ledger entry is zero) — and signal
segments read it back through `Residuals.noise_psd` for a frequency-domain
weight, or `Residuals.noise_variance` for the per-sample variance a time-domain
likelihood needs (turntable does the PSD integration, including the Nyquist
weighting, so the answer does not depend on the parity of `n_samples`).

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
- `n_samples` — **you should never have to state it.** Data enters a campaign
  as a time series, where the arrays carry it exactly, so turntable reads it
  off them; `residual.to_frequency()` then carries it across the transform:

  ```python
  observed = Residuals(tdi=time_series, sample_rate=fs, channels=("A", "E"),
                       tdi_generation="1.5", observable="fractional_frequency")
  spectrum = observed.to_frequency()      # n_samples rides along
  ```

  `to_frequency()`/`to_time()` apply the campaign's Fourier convention —
  `X(f) = dt * rfft(x)`, the one `noise_psd` is normalized against — so it is
  *executed* rather than merely documented, and the round trip is exact for
  either parity of n. Only a spectrum built with no time-domain provenance
  (e.g. straight from a frequency-domain waveform generator) has to state
  `n_samples`, because `n // 2 + 1` bins fit both n=1024 and n=1025, which mean
  different `Tobs` and `df`.
- `channels` — names imply the campaign's normalized definitions
  (e.g. A = (Z − X)/√2); see the `Residuals` docstring.

And it checks consistency at every boundary, failing loudly rather than
producing quietly wrong science:

- `Residuals` validates itself on every construction: tdi keys must equal
  `channels`, array lengths must match `domain`/`n_samples`, and an attached
  orbit must span the observation (catching GPS-vs-zero-based epoch
  mismatches at construction, not mid-run).
- The `Wheel` validates each segment fully **before** registering it (`name`,
  `start` *and* `step`, so a failed `add` changes nothing), and re-validates the
  residual returned by every `start`/`step`: it must be a `Residuals` that kept
  the fixed run settings, must not have dropped the noise model, and must be
  finite — a NaN from a blown-up sampler is refused rather than handed to every
  segment stepped after it. `Residuals` itself rejects wrong tdi shapes *and
  dtypes*, so a mid-run drift raises immediately instead of corrupting the next
  segment's residual. A noise model is checked where it is consumed
  (`noise_psd`/`noise_variance` raise if it lacks a `psd` method).
- Because the ledger is *derived* from what a segment returns, a segment that
  hands the residual straight back withdraws its model from the fit. That is
  almost never intended, so the Wheel warns when a previously non-zero model
  becomes exactly zero: re-subtract your current model on every step, even when
  your parameters did not move.
- `NumericOrbit.positions` refuses to extrapolate outside its tabulated
  ephemeris instead of returning cubic-polynomial garbage.

## Orbits

The constellation ephemeris the data was produced with rides on
`Residuals.orbit` so every segment builds its response from the *same*
spacecraft positions. `turntable.orbits.NumericOrbit` tabulates and
cubic-spline-interpolates an ephemeris, with loaders for LDC/Mojito-style
HDF5 files (`from_hdf5`) and lisaorbits objects (`from_lisaorbits`); both
need the `numeric-orbits` extra. See the module docstring in
[`src/turntable/orbits.py`](https://github.com/AaronDJohnson/turntable/blob/main/src/turntable/orbits.py) for frames and
conventions.

## Development

```sh
uv sync --extra numeric-orbits   # dev group (pytest, pytest-cov, ruff, mypy)
uv run pytest                    # full suite, incl. examples and orbit loaders
uv run pytest --cov --cov-report=term-missing   # coverage (gate: 95%)
uv run mypy                      # turntable ships py.typed; keep it honest
uv run ruff check src tests examples
uv run ruff format --check src tests examples   # CI gates on this too
```

CI runs lint, formatting, mypy, the suite behind a 95% coverage gate, artifact
builds, and an installed-wheel smoke test across Python 3.12/3.13 on Linux and
macOS; plus a core-only leg (numpy alone, through 3.14) and a leg that resolves
to the declared dependency floors, so both claims are tested rather than
asserted. Tagging `v*` runs the same gate and publishes via PyPI Trusted
Publishing. See
[CHANGELOG.md](https://github.com/AaronDJohnson/turntable/blob/main/CHANGELOG.md) for release notes.

## Known limitations

Deliberate scope decisions, recorded so they are choices rather than
oversights:

- **No data-quality / gap mask.** Every sample is treated as carrying
  information. Real LISA data has scheduled gaps (antenna repointing) and
  excised glitches, and a mask is exactly the kind of convention that belongs
  in `Residuals` — otherwise each group invents its own. It is left out while
  the datasets in play are gap-free, because a field nobody exercises would be
  guessed at rather than designed. **TODO: add it as soon as the simulated data
  grows gaps** — see the "Deliberately not in the contract yet" section of the
  `Residuals` docstring for the specific decisions it involves (representation,
  whether the Wheel's arithmetic must respect it, what the PSD grid means over
  a gap, and whether windowing becomes a campaign convention too).
- **One noise model at a time.** `Residuals.noise` is a single slot, so two
  noise blocks (say instrument noise and galactic confusion) cannot each own a
  component and have turntable combine them — the last segment to write it
  wins. Sample them inside one noise segment that publishes a combined model,
  or treat the confusion foreground as a signal segment that subtracts from
  `tdi`. Dropping the model entirely is an error, but one noise segment
  silently overwriting another's is not yet detected.

## Status

Pre-release (0.1.0, unreleased) under active development: interfaces may
still move until the first tag. MIT licensed. Issues and questions welcome.
