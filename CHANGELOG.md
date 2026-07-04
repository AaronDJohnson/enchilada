# Changelog

All notable changes to turntable are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow
[SemVer](https://semver.org/) once tagged.

## [0.1.0] — unreleased

First working release of the blocked-Gibbs orchestration layer.

### Added
- `Residuals`: the frozen cross-group data contract — TDI arrays plus run
  settings, with required `observable` and `domain` fields, long/short name
  aliases (`Tobs`, `fs`, `dt`, ...), a typo catcher, and full self-validation
  on every construction (tdi/channels consistency, per-domain array lengths,
  orbit-must-span-data).
- `Segment` protocol (`initial_state` / `step` / `render`, plus
  `noise_model` for noise segments) with explicitly documented contracts:
  render shapes per domain, RNG and posterior-chain ownership in `State`,
  and the noise-model interface (`psd(freqs[, channel])` /
  `wdm_variance(...)`).
- `Wheel`: the Gibbs loop, with atomic registration, per-step render
  re-validation, noise threading and refresh, a public
  `residual(exclude=...)` accessor, and an `on_sweep` progress/checkpoint
  callback on `run`.
- `NumericOrbit`: tabulated ephemerides with cubic-spline interpolation,
  loaders for LDC/Mojito HDF5 files and lisaorbits objects
  (validated against lisaorbits 3.0.3), equatorial-to-ecliptic frame
  rotation, and a hard refusal to extrapolate outside the tabulated span.
- `turntable.testing`: `EchoSegment` and the `check_segment` conformance
  helper for third-party segment implementations.
- Examples: `examples/demo.py` (plumbing walkthrough), `examples/demo.ipynb`
  (annotated notebook incl. orbits), `examples/toy_fit.py` (a converging
  two-source + sampled-noise Gibbs fit).
- Packaging: installable via uv or pip (uv_build backend), numpy-only core,
  `numeric-orbits` and `examples` extras, MIT license, CI (ruff + pytest on
  Python 3.12/3.13).
