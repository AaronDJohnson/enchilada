# Changelog

All notable changes to turntable are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow
[SemVer](https://semver.org/) once tagged.

## [0.1.0] — unreleased

First working release of the blocked-Gibbs orchestration layer.

### Added
- `Residuals`: the frozen cross-group data contract — TDI arrays plus run
  settings, with a required `observable` field (`domain` defaults to `"time"`,
  `epoch` to `0.0`), long/short name aliases (`Tobs`, `fs`, `dt`, ...), a typo
  catcher, and full self-validation on every construction (tdi/channels
  consistency, per-domain array lengths, orbit-must-span-data).
  `n_samples` is derived from the tdi arrays for time-domain data (where they
  carry it exactly) and required only for frequency-domain data, where the
  rfft grid loses the parity of n — 513 bins are consistent with n=1024 and
  n=1025, which imply different `Tobs`/`df`, so it is asked for rather than
  guessed. This moved `n_samples` after the required fields, so construct
  `Residuals` by keyword (positional construction changed shape).
  `noise_variance()` gives the per-sample variance a time-domain likelihood
  needs (PSD integrated over the grid, Nyquist half-weighted, so it does not
  depend on the parity of `n_samples`). `dtype` is part of the validated
  contract (floating or complex; integer and object arrays are refused at
  construction rather than failing inside the Wheel), `channels` is normalised
  to a tuple, and equality is identity-based so comparing two `Residuals` no
  longer raises a numpy error.
  `to_frequency()` / `to_time()` transform a dataset between representations,
  carrying `n_samples` (so the round trip is exact for either parity) and
  applying the campaign's `X(f) = dt * rfft(x)` convention in code rather than
  in prose -- the same convention `noise_psd` is normalized against, which a
  test now pins. Since data enters as a time series and transforms from there,
  `n_samples` never needs stating by hand in the normal workflow.
- Vocabulary, settled before the first release so none of it needs a
  deprecation period. Three nested scales get three words, and no word does
  double duty:
  a **cycle** is one pass over every block (`Wheel.run(n_cycles=...)`,
  `on_cycle=...`, `check_block(n_cycles=...)`);
  a **block update** is one block's turn within that pass
  (`Block.block_update(residual)`);
  a **step** is what a block's own sampler does, many times, inside a single
  `block_update()` call (`steps_per_cycle` in the GB example).
  The unit itself is a **block** (`Block`, `NoiseBlock`, `EchoBlock`,
  `check_block`, `turntable.block`) -- the word blocked Gibbs already uses for
  a jointly-updated group of parameters. "Iteration" and "sweep" are not used
  as API names: the first could mean any of the three scales, and the second
  invites confusion with a block's own sampler steps. Note for anyone reading
  GLASS alongside this: GLASS's `cycle` means repeat updates of a single
  module, which is not what turntable calls a cycle.
- `Block` protocol — two methods, `start(residual)` and `block_update(residual)`,
  each returning the updated residual. The residual handed to a block is
  the data minus every *other* block (its own model excluded), so the
  block fits it directly and subtracts its new model — there is no
  add-back to forget. Everything else — parameters, RNG, chains, checkpoints,
  and the block's own current model — is block-internal state the Wheel
  never sees. Noise is not special: a noise block returns the residual with
  an updated `noise` object (a zero ledger entry), consumed via
  `Residuals.noise_psd` (documented interface `psd(freqs[, channel])`).
- `Wheel` boundary guards: `add` verifies `name`/`start`/`block_update` before
  registering anything; a returned residual may not drop the noise model
  (symmetric with the existing orbit check) and may not contain NaN/inf, which
  would otherwise be recorded as that block's model and handed to every
  block updated later; and because the ledger is derived, the Wheel warns
  when a previously non-zero model becomes exactly zero (a block that returns
  the residual unchanged silently withdraws itself from the fit).
- `Wheel`: the Gibbs ring, owning the pristine data and a per-block ledger
  (each block's current model). It hands each block the data minus every
  other model and derives that block's new ledger entry from what it
  returns, so the residual bookkeeping — and the add-back — lives in the
  framework, not the block (the split the GLASS global fit uses). Atomic
  registration, per-update validation that the returned residual kept the fixed
  run settings, `residual(exclude=...)` and `contribution(name)` accessors,
  and an `on_cycle` callback on `run`.
- `NumericOrbit`: tabulated ephemerides with cubic-spline interpolation,
  loaders for LDC/Mojito HDF5 files and lisaorbits objects
  (validated against lisaorbits 3.0.3), equatorial-to-ecliptic frame
  rotation, and a hard refusal to extrapolate outside the tabulated span.
- `turntable.testing`: `EchoBlock` and the `check_block` conformance
  helper for third-party block implementations.
- Examples: `examples/demo.py` (plumbing walkthrough), `examples/demo.ipynb`
  (annotated notebook incl. orbits), `examples/toy_fit.py` (a converging
  two-source + sampled-noise Gibbs fit), and
  `examples/gb_block_eryn.ipynb` + `examples/gb_model.py` -- a real LISA
  source class (GBGPU waveforms, an Eryn sampler inside the block, a fixed
  LISA Analysis Tools PSD) recovering an injected galactic binary through the
  Wheel. That one needs an external stack (`gbgpu`, `eryn`,
  `lisaanalysistools`, `matplotlib`, `corner`) that is deliberately not a
  turntable dependency, so it is not exercised by CI.
- Ergonomics: `turntable.replace` re-exports `dataclasses.replace`, so updating
  a residual needs no second import; `Block` is `runtime_checkable`, so
  `isinstance(obj, Block)` is a usable registration check. (`NoiseBlock`
  deliberately is *not*: it declares no member beyond `Block`, so a
  runtime check against it would return True for every block.)
- `ModelWithdrawnWarning`: the withdrawal heuristic raises a named, filterable
  category rather than a bare `RuntimeWarning`, since a legitimate RJMCMC death
  move is indistinguishable from a forgotten re-subtraction from outside the
  block. Filter it if your sampler does death moves; `check_block`
  escalates it to an error, because a conformance check is exactly where the
  strict reading belongs.
- Packaging and release: installable via uv or pip (uv_build backend),
  numpy-only core, `numeric-orbits` and `examples` extras, MIT license, PEP 561
  `py.typed` marker (with `check_untyped_defs`, so unannotated bodies are
  checked too -- consumers' type checkers trust these annotations). CI runs
  ruff (lint + format), mypy, and the suite behind a 95% coverage gate on
  Python 3.12/3.13 across Linux and macOS, plus a numpy-only leg through 3.14,
  a dependency-floors leg, an installed-wheel smoke test, and an sdist leg that
  unpacks the archive and runs the packaged suite from it -- so "the sdist is
  self-testing" is verified rather than asserted. Tagging `v*` re-runs the
  whole gate, checks the tag against the project version as parsed versions,
  and publishes through PyPI Trusted Publishing (no token exists to leak).
