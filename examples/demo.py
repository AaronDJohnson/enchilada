"""Minimal enchilada demo: the L1Data / Block / Wheel plumbing, end to end.

Runs three blocked-Gibbs cycles over two no-op EchoBlocks on synthetic
data -- no real waveforms or MCMC, just enough to watch the Wheel hand each
block its conditional residual. Run it with:

    uv run --no-sync python examples/demo.py

See examples/demo.ipynb for the same walkthrough with commentary.
"""

import numpy as np

from enchilada import BlockResult, L1Data, Wheel
from enchilada.testing import EchoBlock

# One frozen object holds the TDI arrays and the run settings everyone
# in the run agrees on.
rng = np.random.default_rng(0)
n_samples = 1024
channels = ("A", "E", "T")

observed = L1Data(
    channel_data={ch: rng.standard_normal(n_samples) for ch in channels},
    sample_rate_hz=0.1,
    channel_names=channels,  # num_time_samples is inferred from time-domain arrays
    tdi_generation="2.0",
    physical_observable="fractional_frequency",
    start_time_gps=0.0,
)

print(f"observed: N={observed.N}, fs={observed.fs} Hz, Tobs={observed.Tobs:.0f} s\n")

# Each EchoBlock prints what the Wheel hands it and returns a state-only BlockResult,
# so the residuals every block sees are just the observed data. The Wheel keeps
# each block's changing state in its ledger.
ucb = EchoBlock(name="ucb")
mbhb = EchoBlock(name="mbhb")

wheel = Wheel(observed, random_seed=0)
wheel.add(ucb, initial_block_result=BlockResult(sampler_state={"num_sample_calls": 0}))
wheel.add(mbhb, initial_block_result=BlockResult(sampler_state={"num_sample_calls": 0}))

wheel.run(num_cycles=3)

print(
    f"\nucb took {wheel.ledger['ucb'].sampler_state['num_sample_calls']} "
    "sampling calls; "
    f"mbhb took {wheel.ledger['mbhb'].sampler_state['num_sample_calls']} sampling calls"
)
print(
    "full residual RMS:",
    float(np.sqrt(np.mean(wheel.residual().channel_data["A"] ** 2))),
)
