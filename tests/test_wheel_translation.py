"""Wheel owns translations and keeps accepted state on its observation grid."""

import importlib.util
from dataclasses import replace

import numpy as np
import pytest

from enchilada import BlockResult, DataCovariance, L1Data, Wheel, transform

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
    def __init__(self, name, amplitude, expected_domain):
        self.name = name
        self.amplitude = amplitude
        self.expected_domain = expected_domain

    def _render(self, residual, updates):
        assert residual.data_domain == self.expected_domain
        time_data = transform(residual, "time")
        result = time_data.block_result(
            {
                name: np.full_like(values, self.amplitude)
                for name, values in time_data.channel_data.items()
            },
            model_parameters={"amplitude": self.amplitude},
            sampler_state={"updates": updates},
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
        self, conditional_residual, noise_covariance, current_block_result, *, rng
    ):
        assert noise_covariance.data_domain == self.expected_domain
        previous = transform(
            current_block_result, "time", reference_data=conditional_residual
        )
        np.testing.assert_allclose(
            previous.tdi_signal_contribution["A"], self.amplitude, atol=1e-13
        )
        residual = transform(conditional_residual, "time")
        np.testing.assert_allclose(
            residual.channel_data["A"], 1.5 + self.amplitude, atol=1e-12
        )
        return self._render(
            conditional_residual, current_block_result.sampler_state["updates"] + 1
        )


@pytest.mark.parametrize(
    "canonical_domain", ["time", "frequency", pytest.param("wdm", marks=requires_wdm)]
)
@pytest.mark.parametrize(
    "third_domain", ["frequency", pytest.param("wdm", marks=requires_wdm)]
)
def test_mixed_blocks_receive_matching_inputs_and_return_canonical_results(
    canonical_domain, third_domain
):
    data = observations()
    options = {"num_frequency_divisions": 8} if canonical_domain == "wdm" else {}
    canonical = transform(data, canonical_domain, **options)
    wheel = Wheel(canonical, DataCovariance.from_variance(data, 0.25), random_seed=42)
    domains = ("time", "frequency", third_domain)
    for name, amplitude, domain in zip(
        ("one", "two", "half"), (1.0, 2.0, 0.5), domains, strict=True
    ):
        options = {"num_time_divisions": 4} if domain == "wdm" else {}
        block = ConstantSignal(name, amplitude, domain)
        block_data = transform(data, domain, **options)
        wheel.add(
            block,
            initial_block_result=block._render(block_data, 0),
            data_domain=domain,
            **options,
        )
    wheel.run(3)
    assert wheel.working_residual.data_domain == canonical_domain
    restored = transform(wheel.residual(), "time")
    np.testing.assert_allclose(restored.channel_data["A"], 1.5, atol=1e-12)
    for name in ("one", "two", "half"):
        assert wheel.ledger[name].sampler_state == {"updates": 3}
        assert wheel.contribution(name)["A"].shape == canonical.channel_data["A"].shape


@requires_wdm
def test_wdm_noise_publication_is_usable_in_every_domain():
    class PixelNoise:
        name = "noise"

        def sample(
            self, conditional_residual, noise_covariance, current_block_result, *, rng
        ):
            assert noise_covariance.data_domain == "wdm"
            assert isinstance(current_block_result.noise_covariance, DataCovariance)
            return current_block_result

    data = observations()
    grid = transform(data, "wdm", num_frequency_divisions=8).wdm_grid
    variances = np.broadcast_to(np.eye(2), (*grid.array_shape, 2, 2)).copy()
    variances[1:-1] *= np.arange(1, grid.num_frequency_divisions)[:, None, None, None]
    initial = BlockResult(
        noise_covariance=DataCovariance(
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
        initial_block_result=initial,
        data_domain="wdm",
        num_frequency_divisions=8,
    )
    assert wheel.noise_covariance.data_domain == "time"
    assert wheel.ledger["noise"].tdi_signal_contribution is None
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
            self, conditional_residual, noise_covariance, current_block_result, *, rng
        ):
            value = rng.normal()
            if fail[0]:
                if failure == "signal_shape":
                    return BlockResult(
                        tdi_signal_contribution={"A": np.zeros(4, complex)}
                    )
                wrong = replace(conditional_residual, sample_rate_hz=2.0)
                return BlockResult(
                    noise_covariance=DataCovariance.from_variance(wrong, 1.0)
                )
            return BlockResult(sampler_state={"value": value})

    actual, expected = (
        Wheel(observations(), random_seed=7),
        Wheel(observations(), random_seed=7),
    )
    for wheel in (actual, expected):
        wheel.add(
            RandomSignal(),
            initial_block_result=BlockResult(sampler_state={"value": 0.0}),
            data_domain="frequency",
        )
    before = actual.ledger["random"].sampler_state.copy()
    with pytest.raises(ValueError):
        actual.run(1)
    assert actual.ledger["random"].sampler_state == before
    fail[0] = False
    actual.run(1)
    expected.run(1)
    assert (
        actual.ledger["random"].sampler_state == expected.ledger["random"].sampler_state
    )
    np.testing.assert_array_equal(
        actual.residual().channel_data["A"], observations().channel_data["A"]
    )


def test_invalid_domain_selection_does_not_register_or_sample():
    class NeverCalled:
        name = "unregistered"

        def sample(
            self, conditional_residual, noise_covariance, current_block_result, *, rng
        ):
            raise AssertionError

    wheel = Wheel(observations())
    with pytest.raises(ValueError, match="data_domain|target_domain"):
        wheel.add(
            NeverCalled(), initial_block_result=BlockResult(), data_domain="unknown"
        )
    with pytest.raises(ValueError, match="data_domain|wdm"):
        wheel.add(
            NeverCalled(), initial_block_result=BlockResult(), num_time_divisions=8
        )
    assert len(wheel.ledger) == 0
