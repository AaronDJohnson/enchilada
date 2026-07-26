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
- `Segment` protocol — two methods, `start(residual)` and `step(residual)`,
  each returning the updated residual. The residual handed to a segment is
  the data minus every *other* segment (its own model excluded), so the
  segment fits it directly and subtracts its new model — there is no
  add-back to forget. Everything else — parameters, RNG, chains, checkpoints,
  and the segment's own current model — is segment-internal state the Wheel
  never sees. Noise is not special: a noise segment returns the residual with
  an updated `noise` object (a zero ledger entry), consumed via
  `Residuals.noise_psd` (documented interface `psd(freqs[, channel])`).
- `Wheel`: the Gibbs ring, owning the pristine data and a per-segment ledger
  (each segment's current model). It hands each segment the data minus every
  other model and derives that segment's new ledger entry from what it
  returns, so the residual bookkeeping — and the add-back — lives in the
  framework, not the segment (the split the GLASS global fit uses). Atomic
  registration, per-step validation that the returned residual kept the fixed
  run settings, `residual(exclude=...)` and `contribution(name)` accessors,
  and an `on_sweep` callback on `run`.
- `NumericOrbit`: tabulated ephemerides with cubic-spline interpolation,
  loaders for LDC/Mojito HDF5 files and lisaorbits objects
  (validated against lisaorbits 3.0.3), equatorial-to-ecliptic frame
  rotation, and a hard refusal to extrapolate outside the tabulated span.
- `turntable.testing`: `EchoSegment` and the `check_segment` conformance
  helper for third-party segment implementations.
- Examples: `examples/demo.py` (plumbing walkthrough), `examples/demo.ipynb`
  (annotated notebook incl. orbits), `examples/toy_fit.py` (a converging
  two-source + sampled-noise Gibbs fit), and
  `examples/gb_segment_eryn.ipynb` + `examples/gb_model.py` -- a real LISA
  source class (GBGPU waveforms, an Eryn sampler inside the segment, a fixed
  LISA Analysis Tools PSD) recovering an injected galactic binary through the
  Wheel. That one needs an external stack (`gbgpu`, `eryn`,
  `lisaanalysistools`, `matplotlib`, `corner`) that is deliberately not a
  turntable dependency, so it is not exercised by CI.
- Packaging: installable via uv or pip (uv_build backend), numpy-only core,
  `numeric-orbits` and `examples` extras, MIT license, PEP 561 `py.typed`
  marker (the Protocol/dataclass annotations are checkable by consumers), and
  CI running ruff, mypy, the suite behind a 95% coverage gate, and an
  installed-wheel smoke test on Python 3.12/3.13 across Linux and macOS.
