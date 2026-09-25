"""Wheel keeps accepted signals on its observation grid.

Explicit block domain selections also return covariance to that grid; the default
handoff preserves covariance in its declared representation.
"""

import importlib.util
from dataclasses import dataclass, replace

import numpy as np
import pytest

from enchilada import BlockState, DataCovariance, L1Data, Wheel, transform

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

    def sample(
        self, conditional_residual, noise_covariance, current_block_state, *, rng
    ):
        assert noise_covariance.data_domain == self.expected_domain
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
    wheel = Wheel(canonical, DataCovariance.from_variance(data, 0.25), random_seed=42)
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
def test_wdm_noise_publication_is_usable_in_every_domain():
    class PixelNoise:
        name = "noise"

        def sample(
            self, conditional_residual, noise_covariance, current_block_state, *, rng
        ):
            assert noise_covariance.data_domain == "wdm"
            assert isinstance(current_block_state.covariance_matrix, DataCovariance)
            return current_block_state

    data = observations()
    grid = transform(data, "wdm", num_frequency_divisions=8).wdm_grid
    variances = np.broadcast_to(np.eye(2), (*grid.array_shape, 2, 2)).copy()
    variances[1:-1] *= np.arange(1, grid.num_frequency_divisions)[:, None, None, None]
    initial = BlockState(
        covariance_matrix=DataCovariance(
            variances,
            channel_names=data.channel_names,
            sample_rate_hz=data.sample_rate_hz,
            num_time_samples=data.num_time_samples,
            start_time_gps=data.start_time_gps,
            data_domain="wdm",
            wdm_grid=grid,
        )
    )
    wheel = Wheel(data)
    wheel.add(
        PixelNoise(),
        initial_block_state=initial,
        data_domain="wdm",
        num_frequency_divisions=8,
    )
    assert wheel.noise_covariance.data_domain == "time"
    assert wheel.ledger["noise"].template is None
    before = wheel.noise_covariance.quadratic_form(data.channel_data)
    wheel.run(2)
    assert wheel.noise_covariance.quadratic_form(data.channel_data) == pytest.approx(
        before
    )
    np.testing.assert_array_equal(
        wheel.residual().channel_data["A"], data.channel_data["A"]
    )


@pytest.mark.parametrize("failure", ["signal_shape", "covariance_grid"])
def test_failed_translated_result_preserves_accepted_state_and_rng(failure):
    fail = [True]

    class RandomSignal:
        name = "random"

        def sample(
            self, conditional_residual, noise_covariance, current_block_state, *, rng
        ):
            value = rng.normal()
            if fail[0]:
                if failure == "signal_shape":
                    return BlockState(template={"A": np.zeros(4, complex)})
                wrong = replace(conditional_residual, sample_rate_hz=2.0)
                return BlockState(
                    covariance_matrix=DataCovariance.from_variance(wrong, 1.0)
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

        def sample(
            self, conditional_residual, noise_covariance, current_block_state, *, rng
        ):
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
