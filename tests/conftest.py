import numpy as np
import pytest

from enchilada import L1Data


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def make_observed(
    rng, *, num_time_samples=64, channel_names=("A", "E", "T"), **overrides
):
    """A small, valid time-domain L1Data to build tests on."""
    fields = dict(
        channel_data={
            ch: rng.standard_normal(num_time_samples) for ch in channel_names
        },
        sample_rate_hz=0.5,
        num_time_samples=num_time_samples,
        channel_names=channel_names,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
        start_time_gps=0.0,
    )
    fields.update(overrides)
    return L1Data(**fields)


@pytest.fixture
def observed(rng):
    return make_observed(rng)


def const_block_result(residual, value):
    """A block result holding `value` in every sample of every channel."""
    return residual.block_result(
        {ch: np.full_like(arr, value) for ch, arr in residual.channel_data.items()}
    )
