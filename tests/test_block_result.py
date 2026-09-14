"""Optional physical estimates retain complete, atomic sampler snapshots."""

from dataclasses import replace

import numpy as np
import pytest

from enchilada import BlockResult, DataCovariance, Wheel


@pytest.mark.parametrize("factory", ["direct", "data", "zero", "with_covariance"])
def test_block_results_accept_exact_translated_covariance(observed, factory):
    from enchilada.translated_covariance import TranslatedCovariance

    covariance = TranslatedCovariance(
        DataCovariance.from_variance(observed, 2.0), data_domain="frequency"
    )
    if factory == "direct":
        result = BlockResult(noise_covariance=covariance)
    elif factory == "data":
        result = observed.block_result(noise_covariance=covariance)
    elif factory == "zero":
        result = observed.zero_block_result(noise_covariance=covariance)
    else:
        result = BlockResult().with_noise_covariance(covariance)
    assert result.noise_covariance is covariance


def test_wdm_result_factory_accepts_active_signal_and_zero_factory_matches_grid(
    observed,
):
    from enchilada.domains import WDMGrid

    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=16)
    observed = replace(
        observed,
        data_domain="wdm",
        wdm_grid=grid,
        channel_data={ch: np.zeros(grid.array_shape) for ch in observed.channel_names},
    )
    signal = {ch: grid.active_mask.astype(float) for ch in observed.channel_names}
    result = observed.block_result(signal, sampler_state={"step": 1})
    assert result.sampler_state == {"step": 1}
    np.testing.assert_array_equal(result.tdi_signal_contribution["A"], signal["A"])
    zero = observed.zero_block_result()
    assert zero.tdi_signal_contribution["A"].shape == grid.array_shape
    np.testing.assert_array_equal(zero.tdi_signal_contribution["A"], 0.0)
    assert observed.block_result().tdi_signal_contribution is None


@pytest.mark.parametrize("invalid", ["shape", "complex", "inactive"])
def test_wdm_result_factory_rejects_incompatible_signal(observed, invalid):
    from enchilada.domains import WDMGrid

    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=16)
    observed = replace(
        observed,
        data_domain="wdm",
        wdm_grid=grid,
        channel_data={ch: np.zeros(grid.array_shape) for ch in observed.channel_names},
    )
    signal = {ch: np.zeros(grid.array_shape) for ch in observed.channel_names}
    error = ValueError
    if invalid == "shape":
        signal["A"] = np.zeros((5, 8))
    elif invalid == "complex":
        signal["A"] = signal["A"].astype(complex)
        error = TypeError
    else:
        signal["A"][-1, 1] = 1e-30
    with pytest.raises(error, match="A"):
        observed.block_result(signal)


def test_one_dimensional_domains_reject_two_dimensional_block_results(observed):
    for data in (observed, observed.to_frequency()):
        signal = {ch: arr[:, None] for ch, arr in data.channel_data.items()}
        with pytest.raises((TypeError, ValueError), match="1-D"):
            data.block_result(signal)


@pytest.mark.parametrize(
    "num_time_samples,index,endpoint",
    [(1, 0, "DC"), (4, 0, "DC"), (4, -1, "Nyquist"), (5, 0, "DC")],
)
def test_result_factory_rejects_imaginary_rfft_endpoints(
    observed, num_time_samples, index, endpoint
):
    observed = replace(
        observed,
        num_time_samples=num_time_samples,
        channel_data={ch: np.zeros(num_time_samples) for ch in observed.channel_names},
    ).to_frequency()
    signal = {ch: samples.copy() for ch, samples in observed.channel_data.items()}
    signal["A"][index] = 1j
    with pytest.raises(ValueError, match=rf"{endpoint}.*real"):
        observed.block_result(tdi_signal_contribution=signal)


@pytest.mark.parametrize("num_time_samples", [1, 4, 5])
def test_valid_rfft_signal_preserves_all_allowed_components(observed, num_time_samples):
    observed = replace(
        observed,
        num_time_samples=num_time_samples,
        channel_data={ch: np.zeros(num_time_samples) for ch in observed.channel_names},
    ).to_frequency()
    signal = {ch: samples.copy() for ch, samples in observed.channel_data.items()}
    signal["A"][0] = 1.0
    if num_time_samples > 1:
        signal["A"][-1] = 2.0 if num_time_samples % 2 == 0 else 2.0 + 3j

    class Signal:
        name = "signal"

        def sample(
            self, conditional_residual, noise_covariance, current_block_result, *, rng
        ):
            return current_block_result

    wheel = Wheel(observed)
    wheel.add(
        Signal(),
        initial_block_result=observed.block_result(tdi_signal_contribution=signal),
    )
    wheel.run(1)
    np.testing.assert_array_equal(wheel.contribution("signal")["A"], signal["A"])
    np.testing.assert_array_equal(
        wheel.working_residual.channel_data["A"], -signal["A"]
    )


def noise_initial_result(observed, *, signal_value=None):
    signal = None
    if signal_value is not None:
        signal = {
            ch: np.full_like(samples, signal_value)
            for ch, samples in observed.channel_data.items()
        }
    return observed.block_result(
        tdi_signal_contribution=signal,
        noise_covariance=DataCovariance.from_variance(observed, 1),
        model_parameters={"variance": 1},
        sampler_state={"steps": 0, "draw": 0.0},
    )


class NoiseOnly:
    name = "noise"

    def sample(
        self, conditional_residual, noise_covariance, current_block_result, *, rng
    ):
        variance = current_block_result.model_parameters["variance"] + 1
        return conditional_residual.block_result(
            noise_covariance=DataCovariance.from_variance(
                conditional_residual, variance
            ),
            model_parameters={"variance": variance},
            sampler_state={
                "steps": current_block_result.sampler_state["steps"] + 1,
                "draw": rng.normal(),
            },
        )


@pytest.mark.parametrize(
    "num_time_samples,index,endpoint",
    [(1, 0, "DC"), (4, 0, "DC"), (4, -1, "Nyquist"), (5, 0, "DC")],
)
def test_rejected_rfft_signal_preserves_estimates_state_and_rng(
    observed, num_time_samples, index, endpoint
):
    observed = replace(
        observed,
        num_time_samples=num_time_samples,
        channel_data={ch: np.zeros(num_time_samples) for ch in observed.channel_names},
    ).to_frequency()
    fail = [True]

    class InvalidOnce(NoiseOnly):
        def sample(
            self, conditional_residual, noise_covariance, current_block_result, *, rng
        ):
            result = super().sample(
                conditional_residual, noise_covariance, current_block_result, rng=rng
            )
            if fail[0]:
                fail[0] = False
                signal = {
                    ch: np.full_like(samples, 0.25)
                    for ch, samples in conditional_residual.channel_data.items()
                }
                signal["A"][index] += 1j
                return replace(result, tdi_signal_contribution=signal)
            return result

    actual, expected = Wheel(observed, random_seed=8), Wheel(observed, random_seed=8)
    initial = noise_initial_result(observed, signal_value=0.5)
    actual.add(InvalidOnce(), initial_block_result=initial)
    expected.add(NoiseOnly(), initial_block_result=initial)
    with pytest.raises(ValueError, match=rf"{endpoint}.*real"):
        actual.run(1)
    assert (
        actual.ledger["noise"].sampler_state == expected.ledger["noise"].sampler_state
    )
    assert actual.ledger["noise"].model_parameters == {"variance": 1}
    assert actual.noise_covariance.noise_variance("A") == 1
    np.testing.assert_array_equal(actual.contribution("noise")["A"], 0.5)
    np.testing.assert_array_equal(actual.working_residual.channel_data["A"], -0.5)

    actual.run(1)
    expected.run(1)
    assert (
        actual.ledger["noise"].sampler_state == expected.ledger["noise"].sampler_state
    )
    assert actual.noise_covariance.noise_variance("A") == 2
    np.testing.assert_array_equal(actual.working_residual.channel_data["A"], 0.0)


@pytest.mark.parametrize("domain", ["time", "frequency"])
def test_noise_only_result_preserves_data_without_signal_arrays(observed, domain):
    if domain == "frequency":
        observed = observed.to_frequency()
    wheel = Wheel(observed, random_seed=3)
    wheel.add(NoiseOnly(), initial_block_result=noise_initial_result(observed))
    wheel.run(2)

    result = wheel.ledger["noise"]
    assert result.tdi_signal_contribution is None
    assert result.model_parameters == {"variance": 3}
    assert result.sampler_state["steps"] == 2
    assert result.noise_covariance.noise_variance("A") == 3
    assert wheel.noise_covariance.noise_variance("A") == 3
    contribution = wheel.contribution("noise")
    assert set(contribution) == set(observed.channel_names)
    for channel, samples in observed.channel_data.items():
        np.testing.assert_array_equal(
            wheel.working_residual.channel_data[channel], samples
        )
        np.testing.assert_array_equal(contribution[channel], np.zeros_like(samples))
        assert contribution[channel].dtype == samples.dtype
    contribution["A"][:] = 100
    np.testing.assert_array_equal(wheel.contribution("noise")["A"], 0)
    assert wheel.ledger["noise"].tdi_signal_contribution is None


@pytest.mark.parametrize("domain", ["time", "frequency"])
def test_absent_signal_removes_previous_contribution_and_can_return(observed, domain):
    if domain == "frequency":
        observed = observed.to_frequency()

    def population_initial_result(data):
        return data.block_result(
            tdi_signal_contribution={
                ch: np.full_like(samples, 0.25)
                for ch, samples in data.channel_data.items()
            },
            model_parameters={"num_sources": 1},
            sampler_state={"steps": 0},
        )

    class Population:
        name = "population"

        def sample(
            self, conditional_residual, noise_covariance, current_block_result, *, rng
        ):
            for channel, samples in observed.channel_data.items():
                np.testing.assert_allclose(
                    conditional_residual.channel_data[channel], samples
                )
            steps = current_block_result.sampler_state["steps"] + 1
            if steps == 1:
                return BlockResult(
                    model_parameters={"num_sources": 0}, sampler_state={"steps": steps}
                )
            return replace(
                population_initial_result(conditional_residual),
                sampler_state={"steps": steps},
            )

    wheel = Wheel(observed)
    wheel.add(NoiseOnly(), initial_block_result=noise_initial_result(observed))
    wheel.add(Population(), initial_block_result=population_initial_result(observed))
    np.testing.assert_allclose(
        wheel.working_residual.channel_data["A"], observed.channel_data["A"] - 0.25
    )
    wheel.run(1)
    assert wheel.ledger["population"].tdi_signal_contribution is None
    assert wheel.ledger["population"].model_parameters == {"num_sources": 0}
    np.testing.assert_allclose(
        wheel.working_residual.channel_data["A"], observed.channel_data["A"]
    )
    assert wheel.noise_covariance.noise_variance("A") == 2
    wheel.run(1)
    assert wheel.ledger["population"].model_parameters == {"num_sources": 1}
    assert wheel.ledger["population"].sampler_state == {"steps": 2}
    np.testing.assert_allclose(
        wheel.working_residual.channel_data["A"], observed.channel_data["A"] - 0.25
    )


@pytest.mark.parametrize("failure", ["omitted", "wrong_grid"])
def test_rejected_noise_result_preserves_signal_covariance_state_and_rng(
    observed, failure
):
    fail = [True]

    class FailsOnce(NoiseOnly):
        def sample(
            self, conditional_residual, noise_covariance, current_block_result, *, rng
        ):
            result = super().sample(
                conditional_residual, noise_covariance, current_block_result, rng=rng
            )
            if fail[0]:
                fail[0] = False
                covariance = None
                if failure == "wrong_grid":
                    wrong_grid = replace(conditional_residual, start_time_gps=1)
                    covariance = DataCovariance.from_variance(wrong_grid, 2)
                return replace(result, noise_covariance=covariance)
            return result

    actual, expected = Wheel(observed, random_seed=8), Wheel(observed, random_seed=8)
    initial = noise_initial_result(observed, signal_value=0.5)
    actual.add(FailsOnce(), initial_block_result=initial)
    expected.add(NoiseOnly(), initial_block_result=initial)
    error = (
        "must return.*noise_covariance" if failure == "omitted" else "start_time_gps"
    )
    with pytest.raises(ValueError, match=error):
        actual.run(1)
    assert (
        actual.ledger["noise"].sampler_state == expected.ledger["noise"].sampler_state
    )
    assert actual.noise_covariance.noise_variance("A") == 1
    np.testing.assert_array_equal(actual.contribution("noise")["A"], 0.5)
    np.testing.assert_allclose(
        actual.working_residual.channel_data["A"], observed.channel_data["A"] - 0.5
    )
    actual.run(1)
    expected.run(1)
    assert (
        actual.ledger["noise"].sampler_state == expected.ledger["noise"].sampler_state
    )
    assert actual.ledger["noise"].tdi_signal_contribution is None
    assert actual.noise_covariance.noise_variance("A") == 2
    np.testing.assert_array_equal(
        actual.working_residual.channel_data["A"], observed.channel_data["A"]
    )
