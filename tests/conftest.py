import numpy as np
import pytest

from enchilada import L1Data


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def make_observed(rng, *, n_samples=64, channels=("A", "E", "T"), **overrides):
    """A small, valid time-domain L1Data to build tests on."""
    fields = dict(
        tdi={ch: rng.standard_normal(n_samples) for ch in channels},
        sample_rate=0.5,
        n_samples=n_samples,
        channels=channels,
        tdi_generation="2.0",
        observable="fractional_frequency",
        epoch=0.0,
    )
    fields.update(overrides)
    return L1Data(**fields)


@pytest.fixture
def observed(rng):
    return make_observed(rng)


def const_template(residual, value):
    """A template holding `value` in every sample of every channel."""
    return residual.template(
        {ch: np.full_like(arr, value) for ch, arr in residual.tdi.items()}
    )
