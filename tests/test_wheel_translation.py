"""Wheel keeps accepted signals on its observation grid.

Explicit block domain selections also return covariance to that grid, through
the covariance seam in transform() (design spec section 6).
"""

import importlib.util
from dataclasses import dataclass, replace

import numpy as np
import pytest

from enchilada import BlockState, Covariance, L1Data, WDMGrid, Wheel, transform
from enchilada.testing import EchoBlock

requires_wdm = pytest.mark.skipif(
    importlib.util.find_spec("wdm") is None, reason="local WDM backend is optional"
)


def observations():
    return L1Data(
        channel_data={name: np.full(64, 5.0) for name in ("A", "E")},
        channel_names=("A", "E"),
        sample_rate_hz=0.4,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
        start_time_gps=12.0,
    )


class ConstantSignal:
    @dataclass
    class State:
        updates: int = 0

    def __init__(self, name, amplitude, expected_domain):
        self.name = name
        self.amplitude = amplitude
        self.expected_domain = expected_domain
        self.state = self.State()

    def _render(self, residual):
        assert residual.data_domain == self.expected_domain
        time_data = transform(residual, "time")
        result = time_data.block_state(
            {
                name: np.full_like(values, self.amplitude)
                for name, values in time_data.channel_data.items()
            },
            parameters={"amplitude": self.amplitude},
        )
        grid_options = (
            {}
            if residual.wdm_grid is None
            else {"num_frequency_divisions": residual.wdm_grid.num_frequency_divisions}
        )
        return transform(
            result, residual.data_domain, reference_data=time_data, **grid_options
        )

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        previous = transform(
            current_block_state, "time", reference_data=conditional_residual
        )
        np.testing.assert_allclose(previous.template["A"], self.amplitude, atol=1e-13)
        residual = transform(conditional_residual, "time")
        np.testing.assert_allclose(
            residual.channel_data["A"], 1.5 + self.amplitude, atol=1e-12
        )
        self.state.updates += 1
        return self._render(conditional_residual)


@pytest.mark.parametrize(
    "canonical_domain", ["time", "frequency", pytest.param("wdm", marks=requires_wdm)]
)
@pytest.mark.parametrize(
    "third_domain", ["frequency", pytest.param("wdm", marks=requires_wdm)]
)
def test_mixed_blocks_receive_matching_inputs_and_return_canonical_states(
    canonical_domain, third_domain
):
    data = observations()
    options = {"num_frequency_divisions": 8} if canonical_domain == "wdm" else {}
    canonical = transform(data, canonical_domain, **options)
    wheel = Wheel(canonical, random_seed=42)
    domains = ("time", "frequency", third_domain)
    blocks = {}
    for name, amplitude, domain in zip(
        ("one", "two", "half"), (1.0, 2.0, 0.5), domains, strict=True
    ):
        options = {"num_time_divisions": 4} if domain == "wdm" else {}
        block = ConstantSignal(name, amplitude, domain)
        blocks[name] = block
        block_data = transform(data, domain, **options)
        wheel.add(
            block,
            initial_block_state=block._render(block_data),
            data_domain=domain,
            **options,
        )
    wheel.run(3)
    assert wheel.working_residual.data_domain == canonical_domain
    restored = transform(wheel.residual(), "time")
    np.testing.assert_allclose(restored.channel_data["A"], 1.5, atol=1e-12)
    for name in ("one", "two", "half"):
        assert blocks[name].state.updates == 3
        assert wheel.contribution(name)["A"].shape == canonical.channel_data["A"].shape


@requires_wdm
@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="covariance translation seam, design spec section 6",
)
def test_wdm_noise_publication_is_usable_in_every_domain():
    class PixelNoise:
        name = "noise"

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            covariance.check_compatible(conditional_residual)
            assert isinstance(current_block_state.covariance, Covariance)
            return current_block_state

    data = observations()
    wdm_data = transform(data, "wdm", num_frequency_divisions=8)
    grid = wdm_data.wdm_grid
    variances = np.where(grid.active_mask, 1.0, 0.0)
    variances[1:-1] *= np.arange(1, grid.num_frequency_divisions)[:, None]
    initial = BlockState(
        covariance=Covariance.like(
            wdm_data, {name: variances for name in data.channel_names}
        )
    )
    wheel = Wheel(data)
    wheel.add(
        PixelNoise(),
        initial_block_state=initial,
        data_domain="wdm",
        num_frequency_divisions=8,
    )
    assert wheel.covariance is not None
    assert wheel.ledger["noise"].template is None
    wheel.run(2)
    np.testing.assert_array_equal(
        wheel.residual().channel_data["A"], data.channel_data["A"]
    )


@pytest.mark.parametrize("failure", ["signal_shape", "covariance_grid"])
def test_failed_translated_result_preserves_accepted_state_and_rng(failure):
    fail = [True]

    class RandomSignal:
        name = "random"

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            value = rng.normal()
            if fail[0]:
                if failure == "signal_shape":
                    return BlockState(template={"A": np.zeros(4, complex)})
                return BlockState(
                    covariance=Covariance.from_variance(
                        transform(conditional_residual, "time"), 1.0
                    )
                )
            return BlockState(parameters={"amplitude": value})

    actual, expected = (
        Wheel(observations(), random_seed=7),
        Wheel(observations(), random_seed=7),
    )
    for wheel in (actual, expected):
        wheel.add(
            RandomSignal(),
            initial_block_state=BlockState(parameters={"amplitude": 0.0}),
            data_domain="frequency",
        )
    before = actual.ledger["random"].parameters.copy()
    with pytest.raises(ValueError):
        actual.run(1)
    assert actual.ledger["random"].parameters == before
    fail[0] = False
    actual.run(1)
    expected.run(1)
    assert actual.ledger["random"].parameters == expected.ledger["random"].parameters
    np.testing.assert_array_equal(
        actual.residual().channel_data["A"], observations().channel_data["A"]
    )


def test_invalid_domain_selection_does_not_register_or_sample():
    class NeverCalled:
        name = "unregistered"

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            raise AssertionError

    wheel = Wheel(observations())
    with pytest.raises(ValueError, match="data_domain|target_domain"):
        wheel.add(
            NeverCalled(), initial_block_state=BlockState(), data_domain="unknown"
        )
    with pytest.raises(ValueError, match="data_domain|wdm"):
        wheel.add(NeverCalled(), initial_block_state=BlockState(), num_time_divisions=8)
    assert len(wheel.ledger) == 0


def test_initial_wdm_signal_must_match_selected_grid_before_backend(monkeypatch):
    def unexpected_backend():
        raise AssertionError("invalid initial WDM grid reached the optional backend")

    monkeypatch.setattr("enchilada._transforms._load_wdm", unexpected_backend)
    data = observations()
    wheel = Wheel(data)
    # Both grids represent 64 samples, but selected (Nf, Nt) = (4, 16)
    # requires shape (5, 16), not the (9, 8) coefficients of an (8, 8) grid.
    initial = BlockState(
        template={name: np.zeros((9, 8)) for name in data.channel_names}
    )
    with pytest.raises(ValueError, match="shape"):
        wheel.add(
            ConstantSignal("wrong_grid", 1.0, "wdm"),
            initial_block_state=initial,
            data_domain="wdm",
            num_frequency_divisions=4,
            num_time_divisions=16,
        )
    assert len(wheel.ledger) == 0
    for name in data.channel_names:
        np.testing.assert_array_equal(
            wheel.residual().channel_data[name], data.channel_data[name]
        )


def test_wheel_rejects_mutated_wdm_buffers_with_same_time_sample_count():
    data = L1Data(
        {name: np.zeros((5, 16)) for name in ("A", "E")},
        sample_rate_hz=0.4,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
        data_domain="wdm",
    )
    original_grid = data.wdm_grid
    for name in data.channel_names:
        data.channel_data[name] = np.zeros((9, 8))
    with pytest.raises(ValueError, match="shape"):
        Wheel(data)
    assert data.wdm_grid is original_grid


@pytest.mark.parametrize("domain", ["time", "frequency"])
def test_explicit_domain_revalidates_mutated_initial_signal_arrays(domain):
    data = observations()
    wheel = Wheel(data)
    initial = transform(data, domain).zero_block_state()
    initial.template["A"] = initial.template["A"].tolist()
    with pytest.raises(TypeError, match="numpy array"):
        wheel.add(
            ConstantSignal("invalid_array", 1.0, domain),
            initial_block_state=initial,
            data_domain=domain,
        )
    assert len(wheel.ledger) == 0


def test_block_in_another_domain_reaches_the_covariance_seam_at_run():
    """The Wheel hands the covariance to transform() with the observations as
    reference_data; until the layer lands that raises (design spec section 6)."""

    class NeverSampled:
        name = "frequency_block"

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            raise AssertionError("the seam must raise before sample is called")

    data = observations()
    wheel = Wheel(data, Covariance.from_variance(data, 0.25), random_seed=3)
    wheel.add(NeverSampled(), initial_block_state=BlockState(), data_domain="frequency")
    rng_before = wheel._rng.bit_generator.state
    with pytest.raises(NotImplementedError, match="section 6"):
        wheel.run(1)
    assert wheel._rng.bit_generator.state == rng_before
    assert wheel.covariance.noise_variance("A", data) == 0.25


def test_covariance_returned_in_another_domain_reaches_the_seam_at_run():
    """A covariance returned on the block's grid must go back to the observation
    grid; until the layer lands that raises and commits nothing (section 6)."""

    class FrequencyNoise:
        name = "frequency_noise"

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            assert conditional_residual.data_domain == "frequency"
            level = 1.0 + abs(rng.normal())
            return BlockState(
                covariance=Covariance.like(
                    conditional_residual,
                    {
                        name: np.full(values.shape, level)
                        for name, values in conditional_residual.channel_data.items()
                    },
                ),
                parameters={"level": level},
            )

    data = observations()
    wheel = Wheel(data, random_seed=3)
    wheel.add(
        FrequencyNoise(),
        initial_block_state=BlockState(parameters={"level": 0.0}),
        data_domain="frequency",
    )
    rng_before = wheel._rng.bit_generator.state
    with pytest.raises(NotImplementedError, match="section 6"):
        wheel.run(1)
    assert wheel.covariance is None
    assert wheel._rng.bit_generator.state == rng_before
    entry = wheel.ledger["frequency_noise"]
    assert entry.parameters == {"level": 0.0}
    assert entry.template is None and entry.covariance is None
    np.testing.assert_array_equal(
        wheel.residual().channel_data["A"], data.channel_data["A"]
    )


@pytest.mark.parametrize("domain", ["frequency", "wdm"])
def test_noise_only_seed_in_another_domain_reaches_the_seam_at_registration(domain):
    """A seed covariance is on the block's grid; returning it to the observation
    grid is covariance translation, so registration fails and changes nothing."""
    data = observations()
    if domain == "frequency":
        reference = data.to_frequency()
        variances = np.ones(data.num_time_samples // 2 + 1)
        options = {}
    else:
        # Built by hand so the check needs no WDM backend.
        grid = WDMGrid(num_frequency_divisions=8, num_time_divisions=8)
        zeros = {name: np.zeros(grid.array_shape) for name in data.channel_names}
        reference = replace(data, channel_data=zeros, data_domain="wdm")
        variances = np.where(grid.active_mask, 1.0, 0.0)
        options = {"num_frequency_divisions": 8}
    seed = BlockState(
        covariance=Covariance.like(
            reference, {name: variances for name in data.channel_names}
        )
    )
    wheel = Wheel(data)
    with pytest.raises(NotImplementedError, match="section 6"):
        wheel.add(
            EchoBlock("noise"), initial_block_state=seed, data_domain=domain, **options
        )
    assert len(wheel.ledger) == 0
    assert wheel.covariance is None
