"""Measure complete sampling cycles with fixed signals and covariance.

Run from an installed checkout: python benchmarks/benchmark_wheel.py
Use --samples and --blocks to explore a different workload. Timings include
defensive snapshots and validation; these are measurements, not test thresholds.
"""

import argparse
import json
import resource
import statistics
import time

import numpy as np

from enchilada import BlockResult, DataCovariance, L1Data, Wheel


class FixedSignal:
    def __init__(self, name):
        self.name = name

    def sample(
        self, conditional_residual, noise_covariance, current_block_result, *, rng
    ):
        return current_block_result


def measure(num_blocks, num_samples, num_cycles, repeats):
    observed = L1Data(
        channel_data={name: np.ones(num_samples) for name in ("A", "E", "T")},
        channel_names=("A", "E", "T"),
        sample_rate_hz=1.0,
        tdi_generation="2.0",
        physical_observable="strain",
    )
    covariance = DataCovariance.from_variance(observed, 1.0)
    wheel = Wheel(observed, covariance, random_seed=0)
    for index in range(num_blocks):
        initial = BlockResult(
            tdi_signal_contribution={
                name: np.full_like(values, 0.5 / num_blocks)
                for name, values in observed.channel_data.items()
            }
        )
        wheel.add(FixedSignal(str(index)), initial_block_result=initial)
    wheel.run(1)
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        wheel.run(num_cycles)
        elapsed.append((time.perf_counter() - start) / num_cycles)
    np.testing.assert_allclose(wheel.residual().channel_data["A"], 0.5)
    return {
        "blocks": num_blocks,
        "samples_per_channel": num_samples,
        "channels": 3,
        "median_seconds_per_cycle": statistics.median(elapsed),
        "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=32768)
    parser.add_argument("--blocks", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()
    for num_blocks in arguments.blocks:
        print(
            json.dumps(
                measure(
                    num_blocks, arguments.samples, arguments.cycles, arguments.repeats
                )
            ),
            flush=True,
        )
