"""Minimal turntable demo: the Residuals / Segment / Wheel plumbing, end to end.

Runs three blocked-Gibbs sweeps over two no-op EchoSegments on synthetic
data -- no real waveforms or MCMC, just enough to watch the Wheel hand each
segment its residual. Run it with:

    uv run python examples/demo.py

See examples/demo.ipynb for the same walkthrough with commentary.
"""

import numpy as np

from turntable import Residuals, Wheel
from turntable.testing import EchoSegment

# One frozen object holds the TDI arrays and the run settings everyone
# in the run agrees on.
rng = np.random.default_rng(0)
n_samples = 1024
channels = ("A", "E", "T")

observed = Residuals(
    tdi={ch: rng.standard_normal(n_samples) for ch in channels},
    sample_rate=0.1,
    channels=channels,   # n_samples is read off the arrays in the time domain
    tdi_generation="2.0",
    observable="fractional_frequency",
    epoch=0.0,
)

print(f"observed: N={observed.N}, fs={observed.fs} Hz, Tobs={observed.Tobs:.0f} s\n")

# Each EchoSegment prints what the Wheel hands it and contributes zeros, so
# the residuals every segment sees are just the observed data. Segments keep
# their own state -- hold on to the objects to read it back afterwards.
ucb = EchoSegment(name="ucb")
mbhb = EchoSegment(name="mbhb")

wheel = Wheel(observed)
wheel.add(ucb)
wheel.add(mbhb)

wheel.run(n_iterations=3)

print(f"\nucb took {ucb.steps} steps; mbhb took {mbhb.steps} steps")
print("full residual RMS:",
      float(np.sqrt(np.mean(wheel.residual().tdi["A"] ** 2))))
