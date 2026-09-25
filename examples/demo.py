"""Minimal enchilada demo: the L1Data / Block / Wheel plumbing, end to end.

Runs three blocked-Gibbs cycles over two no-op EchoBlocks on synthetic
data -- no real waveforms or MCMC, just enough to watch the Wheel hand each
block its conditional residual. Run it with:

    uv run --no-sync python examples/demo.py

See examples/demo.ipynb for the same walkthrough with commentary.
"""

import numpy as np

from enchilada import BlockState, L1Data, Wheel
from enchilada.testing import EchoBlock

# One frozen object holds the TDI arrays and the run settings everyone
# in the run agrees on.
rng = np.random.default_rng(0)
n_samples = 1024
channels = ("A", "E", "T")

observed = L1Data(
    {ch: rng.standard_normal(n_samples) for ch in channels},
    sample_rate_hz=0.1,
    tdi_generation="2.0",
    physical_observable="fractional_frequency",
    start_time_gps=0.0,
)  # Channel order and num_time_samples are inferred from the dictionary and arrays.

print(f"observed: N={observed.N}, fs={observed.fs} Hz, Tobs={observed.Tobs:.0f} s\n")

# Each EchoBlock keeps its call counter on the block and returns an empty
# physical BlockState, so every residual is just the observed data.
ucb = EchoBlock(name="ucb")
mbhb = EchoBlock(name="mbhb")

wheel = Wheel(observed, random_seed=0)
wheel.add(ucb, initial_block_state=BlockState())
wheel.add(mbhb, initial_block_state=BlockState())

wheel.run(num_cycles=3)

print(
    f"\nucb took {ucb.state.num_sample_calls} "
    "sampling calls; "
    f"mbhb took {mbhb.state.num_sample_calls} sampling calls"
)
print(
    "full residual RMS:",
    float(np.sqrt(np.mean(wheel.residual().channel_data["A"] ** 2))),
)
