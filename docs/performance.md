# Wheel performance

Measurements on 2026-09-13 used macOS 14.3.1 on Apple silicon, Python 3.13.7,
and NumPy 2.5.1. The baseline was the wheel built for the quality review before
the signal-sum cache and covariance snapshot changes.

Run from an installed checkout:

```sh
python benchmarks/benchmark_wheel.py
```

The benchmark uses three time-domain channels, 32,768 samples per channel,
constant signal blocks, and a fixed covariance. It warms up for one cycle,
then reports the median of three runs of three cycles. Timing includes input
snapshots, returned-result validation, signal accounting, and residual creation.
It does not measure a waveform generator or an inference algorithm.

| Blocks | Before, seconds/cycle | After, seconds/cycle | Speedup |
| --- | ---: | ---: | ---: |
| 4 | 0.02585 | 0.00193 | 13.4× |
| 16 | 0.10782 | 0.01383 | 7.8× |
| 64 | 0.56712 | 0.07125 | 8.0× |

These are local measurements, not performance guarantees. Other runs on the
same machine vary with allocator state and background load. To compare revisions,
run this same script against separately installed wheels with the same interpreter
and NumPy version. `--samples`, `--blocks`, `--cycles`, and `--repeats` control
the workload.

Wheel now caches a balanced tree of signal sums. Replacing or excluding a block
requires a logarithmic number of array additions in the number of registered
blocks. Each full residual is still formed by subtracting the aggregate from
pristine observations. This avoids accumulated incremental-subtraction drift
and preserves observations when large signal contributions cancel each other.
Sums use at least float64/complex128 precision, including for float32 signals.
Ordinary floating-point rounding and ill-conditioned cancellation remain possible.

The cache costs additional memory: up to one aggregate channel vector per
internal tree node, on top of the ledger and input/output copies. At 64 blocks
in this benchmark, the dense float64 aggregate arrays require about 49.5 MiB.
The script also prints process peak RSS in the platform's native units (bytes
on macOS, KiB on Linux); this is cumulative across cases and includes allocator
retention, so it is not an isolated per-case memory estimate.

New covariance publications still undergo full validation and Cholesky checks.
Copies of an already validated, privately owned covariance copy its arrays
without repeating factorization. Public input never uses this shortcut.
Other costs remain: accepted ledger dictionaries are copied, sampler state is
deep-copied, and full channel/covariance snapshots are passed to each block.
Large campaigns should benchmark their actual data sizes and sampler states;
the measured speedup does not establish mission-scale capacity.
