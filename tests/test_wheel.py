"""Wheel: the block-result ledger, the data-minus-others handoff, and its guards."""

import warnings
from dataclasses import replace

import numpy as np
import pytest

from conftest import const_block_result, make_observed
from enchilada import BlockResult, DataCovariance, NoiseOverwrittenWarning, Wheel
from enchilada.testing import EchoBlock


def constant_initial_result(observed, value, *, dtype=None):
    signal = {
        ch: np.full_like(samples, value, dtype=dtype)
        for ch, samples in observed.channel_data.items()
    }
    return observed.block_result(
        signal,
        model_parameters={"value": value},
        sampler_state={"updates": 0},
        metadata={"model": "constant"},
    )


def noise_initial_result(observed, level=1.0):
    return BlockResult(
        sampler_state={"updates": 0},
        noise_covariance=DataCovariance.from_variance(observed, level),
    )


class ConstBlock:
    """Returns a block result with the same signal value in every sample.

    The residual it is handed is already the data minus every other block; a
    real block would fit it. This one ignores it and returns its (constant)
    template, so its ledger entry carries that constant signal.
    """

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def sample(
        self, conditional_residual, noise_covariance, current_block_result, *, rng
    ):
        return replace(
            current_block_result,
            tdi_signal_contribution=self._block_result(
                conditional_residual
            ).tdi_signal_contribution,
            sampler_state={
                "updates": current_block_result.sampler_state["updates"] + 1
            },
        )

    def _block_result(self, residual):
        return const_block_result(residual, self.value)


class FlatPSD:
    def __init__(self, level=1.0):
        self.level = level

    def psd(self, freqs, channel=None):
        return np.full_like(freqs, self.level)


class LevelNoiseBlock:
    """A covariance level driven by a sampling-call counter."""

    def __init__(self, name):
        self.name = name

    def sample(
        self, conditional_residual, noise_covariance, current_block_result, *, rng
    ):
        updates = current_block_result.sampler_state["updates"] + 1
        return conditional_residual.block_result(
            sampler_state={"updates": updates}
        ).with_noise_covariance(
            DataCovariance.from_variance(conditional_residual, float(updates + 1))
        )


class TestLedger:
    def test_full_residual_subtracts_every_signal_template(self, rng):
        obs = make_observed(rng)
        wheel = Wheel(observed_data=obs)
        wheel.add(
            block_to_register=ConstBlock("a", 1.0),
            initial_block_result=constant_initial_result(obs, 1.0),
        )
        wheel.add(
            ConstBlock("b", 10.0),
            initial_block_result=constant_initial_result(obs, 10.0),
        )
        wheel.add(
            ConstBlock("c", 100.0),
            initial_block_result=constant_initial_result(obs, 100.0),
        )
        wheel.run(num_cycles=3)
        full = wheel.residual()
        for ch in obs.channel_names:
            np.testing.assert_allclose(
                full.channel_data[ch], obs.channel_data[ch] - 111.0
            )

    def test_exclude_leaves_that_block_in(self, rng):
        obs = make_observed(rng)
        wheel = Wheel(obs)
        wheel.add(
            ConstBlock("a", 1.0), initial_block_result=constant_initial_result(obs, 1.0)
        )
        wheel.add(
            ConstBlock("b", 10.0),
            initial_block_result=constant_initial_result(obs, 10.0),
        )
        wheel.add(
            ConstBlock("c", 100.0),
            initial_block_result=constant_initial_result(obs, 100.0),
        )
        wheel.run(2)
        # residual(exclude_block_name=b) = data minus a and c (110), b left in
        seen = wheel.residual(exclude_block_name="b")
        for ch in obs.channel_names:
            np.testing.assert_allclose(
                seen.channel_data[ch], obs.channel_data[ch] - 101.0
            )

    def test_contribution_is_the_returned_signal_template(self, rng):
        obs = make_observed(rng)
        wheel = Wheel(obs)
        wheel.add(
            ConstBlock("a", 7.0), initial_block_result=constant_initial_result(obs, 7.0)
        )
        wheel.run(2)
        for ch in obs.channel_names:
            np.testing.assert_allclose(wheel.contribution(block_name="a")[ch], 7.0)

    def test_a_template_built_in_place_is_accepted(self, rng):
        # a block may overwrite the arrays it was handed with its template and
        # return the same object: the Wheel handed it copies, so the pristine
        # residual is still there to compare against, and observed is untouched
        obs = make_observed(rng)

        class InPlace:
            name = "ip"

            def sample(
                self,
                conditional_residual,
                noise_covariance,
                current_block_result,
                *,
                rng,
            ):
                for ch in conditional_residual.channel_data:
                    conditional_residual.channel_data[ch][:] = (
                        3.0  # overwrite the handed arrays
                    )
                return conditional_residual.block_result(
                    conditional_residual.channel_data
                )  # ...and claim them

        wheel = Wheel(obs)
        wheel.add(InPlace(), initial_block_result=BlockResult())
        wheel.run(2)
        for ch in obs.channel_names:
            np.testing.assert_allclose(wheel.contribution("ip")[ch], 3.0)
            np.testing.assert_allclose(
                wheel.residual().channel_data[ch], obs.channel_data[ch] - 3.0
            )

    def test_unknown_exclude_or_contribution_rejected(self, observed):
        wheel = Wheel(observed)
        wheel.add(
            ConstBlock("a", 1.0),
            initial_block_result=constant_initial_result(observed, 1.0),
        )
        with pytest.raises(ValueError, match="unknown block"):
            wheel.residual(exclude_block_name="ghost")
        with pytest.raises(ValueError, match="unknown block"):
            wheel.contribution("ghost")

    def test_observed_stays_pristine(self, rng):
        obs = make_observed(rng)
        snapshot = {ch: arr.copy() for ch, arr in obs.channel_data.items()}

        class Mutator(ConstBlock):
            def sample(
                self,
                conditional_residual,
                noise_covariance,
                current_block_result,
                *,
                rng,
            ):
                # Mutate the data handed to this call.
                conditional_residual.channel_data[
                    next(iter(conditional_residual.channel_data))
                ][:] = -999.0
                return super().sample(
                    conditional_residual,
                    noise_covariance,
                    current_block_result,
                    rng=rng,
                )

        wheel = Wheel(obs)
        wheel.add(
            Mutator("mut", 0.0), initial_block_result=constant_initial_result(obs, 0.0)
        )
        wheel.run(2)
        for ch, arr in snapshot.items():
            np.testing.assert_array_equal(
                obs.channel_data[ch], arr
            )  # observed untouched

    def test_sampler_state_lives_in_complete_ledger_block_results(self, observed):
        block = ConstBlock("a", 1.0)
        config = vars(block).copy()
        wheel = Wheel(observed)
        wheel.add(
            block, initial_block_result=constant_initial_result(observed, block.value)
        )
        assert wheel.ledger["a"].sampler_state == {"updates": 0}
        wheel.run(4)
        accepted = wheel.ledger["a"]
        assert isinstance(accepted, BlockResult)
        assert accepted.model_parameters == {"value": 1.0}
        assert accepted.sampler_state == {"updates": 4}
        assert accepted.metadata == {"model": "constant"}
        assert vars(block) == config

    def test_zero_blocks_is_a_noop(self, observed):
        wheel = Wheel(observed)
        wheel.run(5)
        for ch in observed.channel_names:
            np.testing.assert_array_equal(
                wheel.residual().channel_data[ch], observed.channel_data[ch]
            )

    def test_run_rejects_bad_turn_counts(self, observed):
        wheel = Wheel(observed)
        with pytest.raises(ValueError, match="num_cycles"):
            wheel.run(-1)
        with pytest.raises(ValueError, match="num_cycles"):
            wheel.run(True)  # bool is not a cycle count

    def test_run_accepts_numpy_integers(self, observed):
        wheel = Wheel(observed)
        wheel.add(
            ConstBlock("a", 1.0),
            initial_block_result=constant_initial_result(observed, 1.0),
        )
        wheel.run(np.int64(2))


class TestNoAddBack:
    def test_each_block_sees_data_minus_others(self, rng):
        # with two non-trivial blocks, each must be handed the data minus the
        # OTHER (never itself); record what each sees at sampling time.
        obs = make_observed(rng)
        seen = {}

        class Recorder(ConstBlock):
            def sample(
                self,
                conditional_residual,
                noise_covariance,
                current_block_result,
                *,
                rng,
            ):
                seen[self.name] = {
                    ch: conditional_residual.channel_data[ch].copy()
                    for ch in conditional_residual.channel_data
                }
                return super().sample(
                    conditional_residual,
                    noise_covariance,
                    current_block_result,
                    rng=rng,
                )

        wheel = Wheel(obs)
        wheel.add(
            Recorder("a", 2.0), initial_block_result=constant_initial_result(obs, 2.0)
        )
        wheel.add(
            Recorder("b", 5.0), initial_block_result=constant_initial_result(obs, 5.0)
        )
        wheel.run(1)
        # a sees data minus b (5); b sees data minus a (2)
        for ch in obs.channel_names:
            np.testing.assert_allclose(seen["a"][ch], obs.channel_data[ch] - 5.0)
            np.testing.assert_allclose(seen["b"][ch], obs.channel_data[ch] - 2.0)

    def test_samples_see_latest_signals_in_registration_order(self, observed):
        seen = []

        class Growing(ConstBlock):
            def sample(
                self,
                conditional_residual,
                noise_covariance,
                current_block_result,
                *,
                rng,
            ):
                seen.append((self.name, conditional_residual.channel_data["A"].copy()))
                updates = current_block_result.sampler_state["updates"] + 1
                return replace(
                    current_block_result,
                    tdi_signal_contribution=const_block_result(
                        conditional_residual, self.value + updates
                    ).tdi_signal_contribution,
                    sampler_state={"updates": updates},
                )

        wheel = Wheel(observed)
        wheel.add(
            Growing("a", 2.0),
            initial_block_result=constant_initial_result(observed, 2.0),
        )
        wheel.add(
            Growing("b", 5.0),
            initial_block_result=constant_initial_result(observed, 5.0),
        )
        wheel.run(2)
        assert [name for name, _ in seen] == ["a", "b", "a", "b"]
        for (_, handed), subtracted in zip(seen, [5.0, 3.0, 6.0, 4.0], strict=True):
            np.testing.assert_allclose(handed, observed.channel_data["A"] - subtracted)


class TestAtomicRegistration:
    def test_registration_adopts_state_without_invoking_block(self, observed):
        calls = []

        class PriorRecorder(ConstBlock):
            def draw_prior(self, conditional_residual, noise_covariance, *, rng):
                calls.append("prior")
                raise AssertionError("registration must not draw a prior")

            def sample(
                self,
                conditional_residual,
                noise_covariance,
                current_block_result,
                *,
                rng,
            ):
                calls.append("sample")
                return super().sample(
                    conditional_residual,
                    noise_covariance,
                    current_block_result,
                    rng=rng,
                )

        wheel = Wheel(observed)
        wheel.add(
            PriorRecorder("prior", 2.0),
            initial_block_result=constant_initial_result(observed, 2.0),
        )
        assert calls == []
        assert wheel.ledger["prior"].sampler_state == {"updates": 0}
        np.testing.assert_array_equal(wheel.contribution("prior")["A"], 2.0)
        wheel.run(2)
        assert calls == ["sample", "sample"]

    def test_duplicate_name_rejected(self, observed):
        wheel = Wheel(observed)
        wheel.add(
            EchoBlock(name="x"),
            initial_block_result=BlockResult(sampler_state={"num_sample_calls": 0}),
        )
        with pytest.raises(ValueError, match="already registered"):
            wheel.add(
                EchoBlock(name="x"),
                initial_block_result=BlockResult(sampler_state={"num_sample_calls": 0}),
            )

    def test_empty_name_rejected(self, observed):
        with pytest.raises(ValueError, match="non-empty"):
            Wheel(observed).add(
                EchoBlock(name=""),
                initial_block_result=BlockResult(sampler_state={"num_sample_calls": 0}),
            )

    def test_invalid_initial_result_leaves_wheel_untouched(self, observed):
        wheel = Wheel(observed)
        wheel.add(
            ConstBlock("ok", 1.0),
            initial_block_result=constant_initial_result(observed, 1.0),
        )
        with pytest.raises(TypeError, match="initial_block_result.*BlockResult"):
            wheel.add(ConstBlock("bad", 0.0), initial_block_result="not a result")
        with pytest.raises(ValueError, match="unknown block"):
            wheel.contribution("bad")  # not registered
        wheel.contribution("ok")  # the good one still is
        wheel.run(1)  # still healthy


class TestReturnedBlockResultValidation:
    @pytest.mark.parametrize("return_input_data", [False, True])
    def test_non_block_result_return_named(self, observed, return_input_data):
        class Bad(ConstBlock):
            def sample(
                self,
                conditional_residual,
                noise_covariance,
                current_block_result,
                *,
                rng,
            ):
                return conditional_residual if return_input_data else {"A": np.zeros(1)}

        wheel = Wheel(observed)
        wheel.add(
            Bad("bad", 0.0), initial_block_result=constant_initial_result(observed, 0.0)
        )
        with pytest.raises(TypeError, match="bad.sample must return a BlockResult"):
            wheel.run(1)

    def test_bad_tdi_shape_rejected(self, observed):
        class Drifter(ConstBlock):
            def sample(
                self,
                conditional_residual,
                noise_covariance,
                current_block_result,
                *,
                rng,
            ):
                return BlockResult(
                    tdi_signal_contribution={
                        ch: np.zeros(2) for ch in conditional_residual.channel_names
                    }
                )

        wheel = Wheel(observed)
        wheel.add(
            Drifter("drift", 0.0),
            initial_block_result=constant_initial_result(observed, 0.0),
        )
        with pytest.raises(ValueError, match="length 2, expected"):
            wheel.run(1)

    def test_missing_channel_rejected(self, observed):
        class Partial(ConstBlock):
            def sample(
                self,
                conditional_residual,
                noise_covariance,
                current_block_result,
                *,
                rng,
            ):
                return BlockResult(
                    tdi_signal_contribution={
                        "A": np.zeros(conditional_residual.num_time_samples)
                    }
                )

        wheel = Wheel(observed)
        wheel.add(
            Partial("partial", 0.0),
            initial_block_result=constant_initial_result(observed, 0.0),
        )
        with pytest.raises(ValueError, match="must match the run's channels"):
            wheel.run(1)


class TestExplicitNoise:
    def test_noise_covariance_refreshes_each_cycle(self, observed):
        wheel = Wheel(observed)
        wheel.add(
            LevelNoiseBlock("noise"),
            initial_block_result=noise_initial_result(observed),
        )
        assert wheel.noise_covariance.noise_variance() == 1.0
        wheel.run(2)
        assert wheel.noise_covariance.noise_variance() == 3.0
        assert wheel.ledger["noise"].noise_covariance.noise_variance() == 3.0

    def test_noise_block_contributes_zero_without_signal(self, observed):
        wheel = Wheel(observed)
        wheel.add(
            LevelNoiseBlock("noise"),
            initial_block_result=noise_initial_result(observed),
        )
        wheel.run(1)
        assert wheel.ledger["noise"].tdi_signal_contribution is None
        for ch in observed.channel_names:
            np.testing.assert_array_equal(
                wheel.contribution("noise")[ch],
                np.zeros_like(observed.channel_data[ch]),
            )

    def test_initial_noise_is_available_to_explicit_initialization(self, observed):
        wheel = Wheel(observed)
        wheel.add(
            LevelNoiseBlock("noise"),
            initial_block_result=noise_initial_result(observed),
        )
        covariance = wheel.noise_covariance
        initial = constant_initial_result(observed, covariance.noise_variance())
        wheel.add(
            ConstBlock("rec", covariance.noise_variance()), initial_block_result=initial
        )
        assert isinstance(covariance, DataCovariance)
        assert covariance.noise_variance() == 1.0
        np.testing.assert_array_equal(wheel.contribution("rec")["A"], 1.0)

    def test_sample_sees_current_noise(self, observed):
        levels = []

        class Recorder(ConstBlock):
            def sample(
                self,
                conditional_residual,
                noise_covariance,
                current_block_result,
                *,
                rng,
            ):
                levels.append(noise_covariance.noise_variance())
                return super().sample(
                    conditional_residual,
                    noise_covariance,
                    current_block_result,
                    rng=rng,
                )

        wheel = Wheel(observed)
        wheel.add(
            LevelNoiseBlock("noise"),
            initial_block_result=noise_initial_result(observed),
        )  # samples first each cycle
        wheel.add(
            Recorder("rec", 0.0),
            initial_block_result=constant_initial_result(observed, 0.0),
        )
        wheel.run(3)
        assert levels == [2.0, 3.0, 4.0]

    def test_initial_psd_model_must_be_converted_explicitly(self, observed):
        psd_model = FlatPSD(7.0)
        with pytest.raises(TypeError, match="initial_noise_covariance.*DataCovariance"):
            Wheel(observed, initial_noise_covariance=psd_model)
        covariance = DataCovariance.from_psd(observed, psd_model)
        wheel = Wheel(observed, initial_noise_covariance=covariance)
        wheel.add(
            EchoBlock("echo"),
            initial_block_result=BlockResult(sampler_state={"num_sample_calls": 0}),
        )
        wheel.run(2)
        np.testing.assert_array_equal(
            wheel.noise_covariance.covariance_matrix, covariance.covariance_matrix
        )

    def test_fixed_covariance_is_passed_explicitly(self, observed):
        covariance = DataCovariance.from_variance(observed, 7.0)
        wheel = Wheel(observed_data=observed, initial_noise_covariance=covariance)
        wheel.add(
            EchoBlock("echo"),
            initial_block_result=BlockResult(sampler_state={"num_sample_calls": 0}),
        )
        wheel.run(2)
        np.testing.assert_array_equal(
            wheel.noise_covariance.covariance_matrix, covariance.covariance_matrix
        )


class TestOnCycle:
    def test_callback_called_per_turn_with_wheel(self, observed):
        calls = []
        wheel = Wheel(observed)
        wheel.add(
            ConstBlock("a", 1.0),
            initial_block_result=constant_initial_result(observed, 1.0),
        )
        wheel.run(
            num_cycles=3, on_cycle_complete=lambda i, w: calls.append((i, w is wheel))
        )
        assert calls == [(0, True), (1, True), (2, True)]


class TestBoundaryGuards:
    """The Wheel refuses returns that would silently corrupt other blocks."""

    def test_a_signal_block_does_not_disturb_the_noise_model(self, rng):
        """A signal block can omit noise_covariance and preserve fixed noise."""
        obs = make_observed(rng)
        covariance = DataCovariance.from_psd(obs, FlatPSD(3.0))
        wheel = Wheel(obs, initial_noise_covariance=covariance)
        expected = wheel.noise_covariance.covariance_matrix.copy()
        wheel.add(
            ConstBlock("signal", 1.0),
            initial_block_result=constant_initial_result(obs, 1.0),
        )
        wheel.run(2)
        np.testing.assert_array_equal(
            wheel.noise_covariance.covariance_matrix, expected
        )

    def test_non_finite_return_raises_and_names_the_channel(self, observed):
        class Blowup(ConstBlock):
            def sample(
                self,
                conditional_residual,
                noise_covariance,
                current_block_result,
                *,
                rng,
            ):
                return conditional_residual.block_result(
                    {
                        ch: arr * np.nan
                        for ch, arr in conditional_residual.channel_data.items()
                    }
                )

        wheel = Wheel(observed)
        wheel.add(
            Blowup("boom", 0.0),
            initial_block_result=constant_initial_result(observed, 0.0),
        )
        with pytest.raises(ValueError, match="non-finite sample"):
            wheel.run(1)

    def test_missing_sample_is_caught_at_registration(self, observed):
        class NoSample:
            name = "nosample"

        with pytest.raises(TypeError, match="does not implement sample"):
            Wheel(observed).add(NoSample(), initial_block_result=BlockResult())

    def test_block_with_only_name_and_sample_can_register(self, observed):
        class NoPrior:
            name = "no_prior"

            def sample(
                self,
                conditional_residual,
                noise_covariance,
                current_block_result,
                *,
                rng,
            ):
                return current_block_result

        wheel = Wheel(observed)
        wheel.add(NoPrior(), initial_block_result=BlockResult())
        wheel.run(1)
        assert list(wheel.ledger) == ["no_prior"]

    def test_a_death_move_to_a_zero_template_is_silent(self, observed):
        # a reversible-jump block whose last source dies returns a zero
        # template explicitly; there is no heuristic left to trip
        import warnings as _w

        class Death(ConstBlock):
            def sample(
                self,
                conditional_residual,
                noise_covariance,
                current_block_result,
                *,
                rng,
            ):
                return conditional_residual.zero_block_result(
                    model_parameters={"value": 0.0}
                )

        wheel = Wheel(observed)
        wheel.add(
            Death("rj", 1.0),
            initial_block_result=constant_initial_result(observed, 1.0),
        )
        with _w.catch_warnings():
            _w.simplefilter("error")
            wheel.run(1)
        np.testing.assert_array_equal(wheel.contribution("rj")["A"], 0.0)

    def test_a_genuinely_zero_template_does_not_warn(self, observed):
        # EchoBlock contributes a zero template on every sampling call
        import warnings as _w

        wheel = Wheel(observed)
        wheel.add(
            EchoBlock("echo"),
            initial_block_result=BlockResult(sampler_state={"num_sample_calls": 0}),
        )
        with _w.catch_warnings():
            _w.simplefilter("error")  # any warning becomes a failure
            wheel.run(3)

    def test_real_frequency_domain_data_is_rejected(self, rng):
        # a real one-sided spectrum would let a block return `.real` and be
        # credited with the entire imaginary part as its model
        with pytest.raises(TypeError, match="real but data_domain='frequency'"):
            make_observed(
                rng,
                data_domain="frequency",
                channel_data={ch: np.zeros(33) for ch in ("A", "E", "T")},
                num_time_samples=64,
            )

    def test_wider_model_promotes_rather_than_raising(self, rng):
        # float32 observed + float64 template must promote, not fail the way
        # the old in-place subtraction did
        obs = make_observed(
            rng, channel_data={ch: np.zeros(64, np.float32) for ch in ("A", "E", "T")}
        )

        class Wider(ConstBlock):
            def sample(
                self,
                conditional_residual,
                noise_covariance,
                current_block_result,
                *,
                rng,
            ):
                return conditional_residual.block_result(
                    {
                        ch: np.ones(64, np.float64)
                        for ch in conditional_residual.channel_data
                    }
                )

        wheel = Wheel(obs)
        wheel.add(
            Wider("w", 0.0), initial_block_result=constant_initial_result(obs, 0.0)
        )
        wheel.run(1)
        assert wheel.residual().channel_data["A"].dtype == np.float64
        assert wheel.observed_data.channel_data["A"].dtype == np.float32  # left alone


def test_cancelling_signals_do_not_erase_the_observation(rng):
    observed = make_observed(
        rng, channel_data={name: np.ones(64) for name in ("A", "E", "T")}
    )
    wheel = Wheel(observed)
    wheel.add(
        ConstBlock("positive", 1e16),
        initial_block_result=constant_initial_result(observed, 1e16),
    )
    wheel.add(
        ConstBlock("negative", -1e16),
        initial_block_result=constant_initial_result(observed, -1e16),
    )
    np.testing.assert_array_equal(wheel.residual().channel_data["A"], np.ones(64))
    wheel.run(2)
    np.testing.assert_array_equal(wheel.residual().channel_data["A"], np.ones(64))


def test_cached_signal_sums_follow_updates_withdrawals_and_failures(rng):
    observed = make_observed(
        rng, channel_data={name: np.full(64, 100.0) for name in ("A", "E", "T")}
    )

    class Changing(ConstBlock):
        def sample(
            self, conditional_residual, noise_covariance, current_block_result, *, rng
        ):
            count = current_block_result.sampler_state["updates"] + 1
            if count == 2 and self.name == "4":
                # Signal leaves are finite individually, but their sum overflows.
                return conditional_residual.block_result(
                    {name: np.full(64, np.finfo(float).max) for name in ("A", "E", "T")}
                )
            if count == 2:
                return BlockResult(sampler_state={"updates": count})
            return conditional_residual.block_result(
                {name: np.full(64, self.value + count) for name in ("A", "E", "T")},
                sampler_state={"updates": count},
            )

    wheel = Wheel(observed)
    for index in range(9):
        wheel.add(
            Changing(str(index), float(index + 1)),
            initial_block_result=constant_initial_result(observed, float(index + 1)),
        )
    np.testing.assert_array_equal(wheel.residual().channel_data["A"], 55.0)
    wheel.run(1)
    np.testing.assert_array_equal(wheel.residual().channel_data["A"], 46.0)
    for index in range(9):
        np.testing.assert_array_equal(
            wheel.residual(exclude_block_name=str(index)).channel_data["A"],
            48.0 + index,
        )
    # Force an overflow in the candidate tree after its leaves pass validation.
    wheel.add(
        ConstBlock("overflow_guard", np.finfo(float).max),
        initial_block_result=constant_initial_result(observed, np.finfo(float).max),
    )
    before_failed_block = wheel.contribution("4")["A"]
    with pytest.raises(ValueError, match="non-finite"):
        wheel.run(1)
    np.testing.assert_array_equal(wheel.contribution("4")["A"], before_failed_block)
    # The first four withdrawals were accepted; the rejected candidate never
    # enters the cached sum. Excluding the enormous guard recovers the small sum.
    np.testing.assert_array_equal(
        wheel.residual(exclude_block_name="overflow_guard").channel_data["A"], 60.0
    )


def test_signal_aggregation_preserves_small_float32_contributions(rng):
    observed = make_observed(
        rng, channel_data={name: np.ones(64) for name in ("A", "E", "T")}
    )

    class NarrowSignal(ConstBlock):
        def _block_result(self, residual):
            return residual.block_result(
                {
                    name: np.full(64, self.value, dtype=np.float32)
                    for name in residual.channel_names
                }
            )

    wheel = Wheel(observed)
    for index, value in enumerate((1e8, 1.0, -1e8)):
        wheel.add(
            NarrowSignal(str(index), value),
            initial_block_result=constant_initial_result(
                observed, value, dtype=np.float32
            ),
        )
    np.testing.assert_array_equal(wheel.residual().channel_data["A"], 0.0)


class TestDocumentedContracts:
    """Guarantees the docstrings make that nothing else pins."""

    def test_explicit_initialization_can_read_current_residual(self, rng):
        """The caller can initialize a new block from the current full residual."""
        obs = make_observed(rng)
        seen = {}

        wheel = Wheel(obs)
        for name, value in [("first", 3.0), ("second", 5.0)]:
            residual = wheel.residual()
            seen[name] = residual.channel_data["A"].copy()
            wheel.add(
                ConstBlock(name, value),
                initial_block_result=constant_initial_result(residual, value),
            )
        # the first joiner sees raw data; the second sees it minus the first
        np.testing.assert_allclose(seen["first"], obs.channel_data["A"])
        np.testing.assert_allclose(seen["second"], obs.channel_data["A"] - 3.0)

    def test_residual_returns_fresh_arrays(self, rng):
        """residual()'s docstring: 'callers may mutate freely'."""
        obs = make_observed(rng)
        snapshot = obs.channel_data["A"].copy()
        wheel = Wheel(obs)  # no blocks: the aliasing case
        wheel.residual().channel_data["A"][0] = -999.0
        np.testing.assert_array_equal(obs.channel_data["A"], snapshot)
        wheel.add(
            ConstBlock("a", 1.0), initial_block_result=constant_initial_result(obs, 1.0)
        )
        wheel.residual(exclude_block_name="a").channel_data["A"][0] = -999.0
        np.testing.assert_array_equal(obs.channel_data["A"], snapshot)

    def test_contribution_returns_a_copy_of_the_ledger_entry(self, observed):
        wheel = Wheel(observed)
        wheel.add(
            ConstBlock("a", 2.0),
            initial_block_result=constant_initial_result(observed, 2.0),
        )
        got = wheel.contribution("a")
        got["A"][:] = 100.0
        np.testing.assert_allclose(wheel.contribution("a")["A"], 2.0)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize("channel", ["A", "E", "T"])
    def test_non_finite_is_caught_in_any_channel_and_any_form(self, rng, bad, channel):
        obs = make_observed(rng)

        class Blowup(ConstBlock):
            def sample(
                self,
                conditional_residual,
                noise_covariance,
                current_block_result,
                *,
                rng,
            ):
                tdi = {
                    ch: np.zeros_like(arr)
                    for ch, arr in conditional_residual.channel_data.items()
                }
                tdi[channel][3] = bad
                return conditional_residual.block_result(tdi)

        wheel = Wheel(obs)
        wheel.add(
            Blowup("b", 0.0), initial_block_result=constant_initial_result(obs, 0.0)
        )
        with pytest.raises(ValueError, match="non-finite sample"):
            wheel.run(1)

    def test_a_template_zero_in_one_channel_is_recorded_per_channel(self, rng):
        obs = make_observed(rng)

        def partly_zero_result(residual):
            # non-zero in A and E, zero in T
            return residual.block_result(
                {
                    ch: np.full_like(arr, 0.0 if ch == "T" else 1.0)
                    for ch, arr in residual.channel_data.items()
                }
            )

        class PartlyZero:
            name = "p"

            def sample(
                self,
                conditional_residual,
                noise_covariance,
                current_block_result,
                *,
                rng,
            ):
                return partly_zero_result(conditional_residual)

        wheel = Wheel(obs)
        wheel.add(PartlyZero(), initial_block_result=partly_zero_result(obs))
        wheel.run(2)
        np.testing.assert_allclose(wheel.contribution("p")["T"], 0.0)
        np.testing.assert_allclose(wheel.contribution("p")["A"], 1.0)

    def test_block_name_must_be_a_string_not_just_non_empty(self, observed):
        class Numbered(ConstBlock):
            pass

        block = Numbered("x", 0.0)
        block.name = 5
        with pytest.raises(ValueError, match="non-empty string"):
            Wheel(observed).add(
                block,
                initial_block_result=constant_initial_result(observed, block.value),
            )

    def test_missing_name_gets_the_protocol_message(self, observed):
        class Anonymous:
            def sample(
                self,
                conditional_residual,
                noise_covariance,
                current_block_result,
                *,
                rng,
            ):
                return conditional_residual.zero_block_result()

        with pytest.raises((ValueError, TypeError), match="name"):
            Wheel(observed).add(Anonymous(), initial_block_result=BlockResult())


class TestObservedDataIsChecked:
    """Wheel.__init__ checks the data itself, so the first block to touch it
    is not blamed by the return-value guard for damage it did not do."""

    def test_non_finite_observed_data_is_refused_at_construction(self, rng):
        obs = make_observed(rng)
        obs.channel_data["E"][7] = np.nan
        with pytest.raises(
            ValueError, match=r"observed_data.channel_data\['E'\] has non-finite"
        ):
            Wheel(obs)

    def test_the_message_points_at_the_missing_gap_support(self, rng):
        """NaN is how a user marks a gap today, and gaps are not in the
        contract yet -- the error has to say so or it reads as a bug."""
        obs = make_observed(rng)
        obs.channel_data["A"][:3] = np.nan
        with pytest.raises(ValueError, match="no data-quality mask"):
            Wheel(obs)

    def test_finite_data_constructs_normally(self, observed):
        assert Wheel(observed).residual() is not None

    def test_initial_covariance_subclass_cannot_skip_grid_validation(self, observed):
        class UncheckedCovariance(DataCovariance):
            def check_compatible(self, reference_data):
                pass

        mismatched_grid = replace(observed, sample_rate_hz=observed.sample_rate_hz * 2)
        covariance = DataCovariance.from_variance(mismatched_grid, 1.0)
        unchecked = UncheckedCovariance(**vars(covariance))
        with pytest.raises(ValueError, match="sample_rate_hz"):
            Wheel(observed, initial_noise_covariance=unchecked)


class TestNoiseOwnership:
    """`Wheel.noise_covariance` is a single slot, so a second writer replaces the first.
    The Wheel cannot forbid that (handing ownership over may be deliberate),
    but it must not stay silent either."""

    @staticmethod
    def _noise_block(name, level):
        class NoiseWriter:
            def __init__(self):
                self.name = name

            def sample(
                self,
                conditional_residual,
                noise_covariance,
                current_block_result,
                *,
                rng,
            ):
                # a real noise block re-estimates, so it publishes a NEW object
                # every cycle -- that must not read as an overwrite
                return conditional_residual.block_result().with_noise_covariance(
                    DataCovariance.from_variance(conditional_residual, level)
                )

        return NoiseWriter()

    def test_two_noise_blocks_warn_that_one_is_being_lost(self, observed):
        wheel = Wheel(observed)
        wheel.add(
            self._noise_block("instrument", 1.0),
            initial_block_result=noise_initial_result(observed, 1.0),
        )
        with pytest.warns(NoiseOverwrittenWarning, match="'instrument' owns"):
            wheel.add(
                self._noise_block("confusion", 2.0),
                initial_block_result=noise_initial_result(observed, 2.0),
            )

    def test_warning_as_error_rolls_back_registration(self, observed):
        wheel = Wheel(observed)
        wheel.add(
            self._noise_block("instrument", 1.0),
            initial_block_result=noise_initial_result(observed, 1.0),
        )
        before = wheel.working_residual.channel_data["A"].copy()
        with warnings.catch_warnings():
            warnings.simplefilter("error", NoiseOverwrittenWarning)
            with pytest.raises(NoiseOverwrittenWarning, match="'instrument' owns"):
                wheel.add(
                    self._noise_block("confusion", 2.0),
                    initial_block_result=noise_initial_result(observed, 2.0),
                )
            # Failed membership or owner adoption would break this next cycle.
            wheel.run(1)
        assert list(wheel.ledger) == ["instrument"]
        assert wheel.noise_covariance.noise_variance() == 1.0
        np.testing.assert_array_equal(wheel.working_residual.channel_data["A"], before)

    def test_warning_as_error_rolls_back_sample(self, observed):
        class Takeover(ConstBlock):
            def sample(
                self,
                conditional_residual,
                noise_covariance,
                current_block_result,
                *,
                rng,
            ):
                return conditional_residual.block_result(
                    const_block_result(
                        conditional_residual, 9.0
                    ).tdi_signal_contribution,
                    sampler_state={"updates": 1},
                ).with_noise_covariance(
                    DataCovariance.from_variance(conditional_residual, 2.0)
                )

        wheel = Wheel(observed)
        wheel.add(
            self._noise_block("instrument", 1.0),
            initial_block_result=noise_initial_result(observed, 1.0),
        )
        wheel.add(
            Takeover("takeover", 0.5),
            initial_block_result=constant_initial_result(observed, 0.5),
        )
        before = wheel.working_residual.channel_data["A"].copy()
        with warnings.catch_warnings():
            warnings.simplefilter("error", NoiseOverwrittenWarning)
            for _ in range(2):
                with pytest.raises(NoiseOverwrittenWarning, match="'instrument' owns"):
                    wheel.run(1)
        assert list(wheel.ledger) == ["instrument", "takeover"]
        assert wheel.noise_covariance.noise_variance() == 1.0
        assert wheel.ledger["takeover"].sampler_state == {"updates": 0}
        assert wheel.ledger["takeover"].noise_covariance is None
        np.testing.assert_array_equal(wheel.contribution("takeover")["A"], 0.5)
        np.testing.assert_array_equal(wheel.working_residual.channel_data["A"], before)

    def test_the_message_names_both_blocks_and_the_fix(self, observed):
        wheel = Wheel(observed)
        wheel.add(
            self._noise_block("a", 1.0),
            initial_block_result=noise_initial_result(observed, 1.0),
        )
        with pytest.warns(NoiseOverwrittenWarning) as rec:
            wheel.add(
                self._noise_block("b", 2.0),
                initial_block_result=noise_initial_result(observed, 2.0),
            )
        msg = str(rec[0].message)
        assert "b.initial_block_result" in msg and "'a'" in msg
        assert "single" in msg and "combined model" in msg

    def test_one_noise_block_re_estimating_every_cycle_is_silent(self, observed):
        """The common case: the same block writes a fresh model each cycle."""
        wheel = Wheel(observed)
        wheel.add(
            self._noise_block("noise", 1.0),
            initial_block_result=noise_initial_result(observed, 1.0),
        )
        wheel.add(
            ConstBlock("signal", 0.5),
            initial_block_result=constant_initial_result(observed, 0.5),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails this test
            wheel.run(3)

    def test_replacing_initial_covariance_is_silent(self, observed):
        """Initial covariance is not owned by a block."""
        covariance = DataCovariance.from_variance(observed, 9.0)
        wheel = Wheel(observed, initial_noise_covariance=covariance)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            wheel.add(
                self._noise_block("noise", 1.0),
                initial_block_result=noise_initial_result(observed, 1.0),
            )
        assert wheel.noise_covariance.noise_variance() == 1.0

    def test_a_signal_block_does_not_claim_noise_ownership(self, observed):
        """A signal-only result leaves the existing covariance owner unchanged."""
        wheel = Wheel(observed)
        wheel.add(
            ConstBlock("signal", 0.5),
            initial_block_result=constant_initial_result(observed, 0.5),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            wheel.add(
                self._noise_block("noise", 1.0),
                initial_block_result=noise_initial_result(observed, 1.0),
            )
            wheel.run(2)
