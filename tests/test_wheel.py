"""Wheel: the ledger, data-minus-others handoff, and boundary validation."""

import warnings
from dataclasses import replace

import numpy as np
import pytest

from conftest import make_observed
from turntable import (
    ModelWithdrawnWarning,
    NoiseOverwrittenWarning,
    Residuals,
    Wheel,
)
from turntable.testing import EchoBlock


class ConstBlock:
    """Subtracts a constant from the residual it is handed -- no add-back.

    The residual it receives is already the data minus every other block, so
    it just subtracts its (constant) model. Its ledger entry is that constant.
    """

    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.updates = 0

    def start(self, residual):
        return self._subtract(residual)

    def update(self, residual):
        self.updates += 1
        return self._subtract(residual)

    def _subtract(self, residual):
        return replace(
            residual, tdi={ch: arr - self.value for ch, arr in residual.tdi.items()}
        )


class FlatPSD:
    def __init__(self, level=1.0):
        self.level = level

    def psd(self, freqs, channel=None):
        return np.full_like(freqs, self.level)


class LevelNoiseBlock:
    """Noise block: sets residual.noise, level tracks its update count."""

    def __init__(self, name):
        self.name = name
        self.updates = 0

    def start(self, residual):
        return replace(residual, noise=FlatPSD(0.0))

    def update(self, residual):
        self.updates += 1
        return replace(residual, noise=FlatPSD(float(self.updates)))


class TestLedger:
    def test_full_residual_subtracts_every_model(self, rng):
        obs = make_observed(rng)
        wheel = Wheel(obs)
        wheel.add(ConstBlock("a", 1.0))
        wheel.add(ConstBlock("b", 10.0))
        wheel.add(ConstBlock("c", 100.0))
        wheel.run(3)
        full = wheel.residual()
        for ch in obs.channels:
            np.testing.assert_allclose(full.tdi[ch], obs.tdi[ch] - 111.0)

    def test_exclude_leaves_that_block_in(self, rng):
        obs = make_observed(rng)
        wheel = Wheel(obs)
        wheel.add(ConstBlock("a", 1.0))
        wheel.add(ConstBlock("b", 10.0))
        wheel.add(ConstBlock("c", 100.0))
        wheel.run(2)
        # residual(exclude=b) = data minus a and c (110), b left in
        seen = wheel.residual(exclude="b")
        for ch in obs.channels:
            np.testing.assert_allclose(seen.tdi[ch], obs.tdi[ch] - 101.0)

    def test_contribution_is_the_derived_model(self, rng):
        obs = make_observed(rng)
        wheel = Wheel(obs)
        wheel.add(ConstBlock("a", 7.0))
        wheel.run(2)
        for ch in obs.channels:
            np.testing.assert_allclose(wheel.contribution("a")[ch], 7.0)

    def test_ledger_derivation_survives_in_place_return(self, rng):
        # a block that mutates its handed arrays in place AND returns the
        # same object still gets its contribution derived correctly, because
        # the Wheel diffs against a pristine snapshot.
        obs = make_observed(rng)

        class InPlace:
            name = "ip"

            def start(self, residual):
                return residual  # zero contribution

            def update(self, residual):
                for ch in residual.tdi:
                    residual.tdi[ch] -= 3.0  # mutate in place
                return residual  # return the SAME object

        wheel = Wheel(obs)
        wheel.add(InPlace())
        wheel.run(2)
        for ch in obs.channels:
            np.testing.assert_allclose(wheel.contribution("ip")[ch], 3.0)
            np.testing.assert_allclose(wheel.residual().tdi[ch], obs.tdi[ch] - 3.0)

    def test_unknown_exclude_or_contribution_rejected(self, observed):
        wheel = Wheel(observed)
        wheel.add(ConstBlock("a", 1.0))
        with pytest.raises(ValueError, match="unknown block"):
            wheel.residual(exclude="ghost")
        with pytest.raises(ValueError, match="unknown block"):
            wheel.contribution("ghost")

    def test_observed_stays_pristine(self, rng):
        obs = make_observed(rng)
        snapshot = {ch: arr.copy() for ch, arr in obs.tdi.items()}

        class Mutator(ConstBlock):
            def update(self, residual):
                residual.tdi[next(iter(residual.tdi))][:] = -999.0  # mutate the copy
                return super().update(residual)

        wheel = Wheel(obs)
        wheel.add(Mutator("mut", 0.0))
        wheel.run(2)
        for ch, arr in snapshot.items():
            np.testing.assert_array_equal(obs.tdi[ch], arr)  # observed untouched

    def test_blocks_keep_their_own_state(self, observed):
        block = ConstBlock("a", 1.0)
        wheel = Wheel(observed)
        wheel.add(block)
        wheel.run(4)
        assert block.updates == 4  # sampler state lives on the object, not the Wheel

    def test_zero_blocks_is_a_noop(self, observed):
        wheel = Wheel(observed)
        wheel.run(5)
        for ch in observed.channels:
            np.testing.assert_array_equal(wheel.residual().tdi[ch], observed.tdi[ch])

    def test_run_rejects_bad_turn_counts(self, observed):
        wheel = Wheel(observed)
        with pytest.raises(ValueError, match="n_cycles"):
            wheel.run(-1)
        with pytest.raises(ValueError, match="n_cycles"):
            wheel.run(True)  # bool is not a cycle count

    def test_run_accepts_numpy_integers(self, observed):
        wheel = Wheel(observed)
        wheel.add(ConstBlock("a", 1.0))
        wheel.run(np.int64(2))


class TestNoAddBack:
    def test_each_block_sees_data_minus_others(self, rng):
        # with two non-trivial blocks, each must be handed the data minus the
        # OTHER (never itself); record what each sees at update time.
        obs = make_observed(rng)
        seen = {}

        class Recorder(ConstBlock):
            def update(self, residual):
                seen[self.name] = {ch: residual.tdi[ch].copy() for ch in residual.tdi}
                return super().update(residual)

        wheel = Wheel(obs)
        wheel.add(Recorder("a", 2.0))
        wheel.add(Recorder("b", 5.0))
        wheel.run(1)
        # a sees data minus b (5); b sees data minus a (2)
        for ch in obs.channels:
            np.testing.assert_allclose(seen["a"][ch], obs.tdi[ch] - 5.0)
            np.testing.assert_allclose(seen["b"][ch], obs.tdi[ch] - 2.0)


class TestAtomicRegistration:
    def test_duplicate_name_rejected(self, observed):
        wheel = Wheel(observed)
        wheel.add(EchoBlock(name="x"))
        with pytest.raises(ValueError, match="already registered"):
            wheel.add(EchoBlock(name="x"))

    def test_empty_name_rejected(self, observed):
        with pytest.raises(ValueError, match="non-empty"):
            Wheel(observed).add(EchoBlock(name=""))

    def test_failed_start_leaves_wheel_untouched(self, observed):
        class BadStart(ConstBlock):
            def start(self, residual):
                return "not a residual"

        wheel = Wheel(observed)
        wheel.add(ConstBlock("ok", 1.0))
        with pytest.raises(TypeError, match="bad.start must return a Residuals"):
            wheel.add(BadStart("bad", 0.0))
        with pytest.raises(ValueError, match="unknown block"):
            wheel.contribution("bad")  # not registered
        wheel.contribution("ok")  # the good one still is
        wheel.run(1)  # still healthy


class TestReturnedResidualValidation:
    def test_non_residual_return_named(self, observed):
        class Bad(ConstBlock):
            def update(self, residual):
                return {"A": np.zeros(1)}  # a dict, not a Residuals

        wheel = Wheel(observed)
        wheel.add(Bad("bad", 0.0))
        with pytest.raises(TypeError, match="bad.update must return a Residuals"):
            wheel.run(1)

    def test_changed_run_setting_rejected(self, observed):
        class Cheat(ConstBlock):
            def update(self, residual):
                return replace(residual, tdi_generation="9.9")

        wheel = Wheel(observed)
        wheel.add(Cheat("cheat", 0.0))
        with pytest.raises(
            ValueError, match="changed the run setting 'tdi_generation'"
        ):
            wheel.run(1)

    def test_bad_tdi_shape_raises_via_residuals(self, observed):
        class Drifter(ConstBlock):
            def update(self, residual):
                return replace(
                    residual, tdi={ch: np.zeros(2) for ch in residual.channels}
                )

        wheel = Wheel(observed)
        wheel.add(Drifter("drift", 0.0))
        with pytest.raises(ValueError, match="length 2, expected"):
            wheel.run(1)

    def test_changed_orbit_rejected(self, observed):
        class Cheat(ConstBlock):
            def update(self, residual):
                return replace(residual, orbit=object())

        wheel = Wheel(observed)
        wheel.add(Cheat("cheat", 0.0))
        with pytest.raises(ValueError, match="changed the orbit"):
            wheel.run(1)


class TestNoiseViaResidual:
    def test_noise_rides_the_residual_and_refreshes(self, observed):
        wheel = Wheel(observed)
        wheel.add(LevelNoiseBlock("noise"))
        assert wheel.residual().noise.level == 0.0  # set in start
        wheel.run(2)
        assert wheel.residual().noise.level == 2.0  # refreshed each update

    def test_noise_block_has_zero_ledger_entry(self, observed):
        wheel = Wheel(observed)
        wheel.add(LevelNoiseBlock("noise"))
        wheel.run(1)
        for ch in observed.channels:
            np.testing.assert_array_equal(
                wheel.contribution("noise")[ch], np.zeros_like(observed.tdi[ch])
            )

    def test_start_sees_noise_from_earlier_block(self, observed):
        seen = {}

        class Recorder(ConstBlock):
            def start(self, residual):
                seen["noise"] = residual.noise
                return super().start(residual)

        wheel = Wheel(observed)
        wheel.add(LevelNoiseBlock("noise"))
        wheel.add(Recorder("rec", 0.0))
        assert isinstance(seen["noise"], FlatPSD)

    def test_update_sees_current_noise(self, observed):
        levels = []

        class Recorder(ConstBlock):
            def update(self, residual):
                levels.append(residual.noise.level)
                return super().update(residual)

        wheel = Wheel(observed)
        wheel.add(LevelNoiseBlock("noise"))  # updates first each cycle
        wheel.add(Recorder("rec", 0.0))
        wheel.run(3)
        assert levels == [1.0, 2.0, 3.0]

    def test_fixed_noise_passes_through(self, rng):
        obs = make_observed(rng, noise=FlatPSD(7.0))
        wheel = Wheel(obs)
        wheel.add(EchoBlock("echo"))
        wheel.run(2)
        assert wheel.residual().noise.level == 7.0  # unchanged, threaded along


class TestOnCycle:
    def test_callback_called_per_turn_with_wheel(self, observed):
        calls = []
        wheel = Wheel(observed)
        wheel.add(ConstBlock("a", 1.0))
        wheel.run(3, on_cycle=lambda i, w: calls.append((i, w is wheel)))
        assert calls == [(0, True), (1, True), (2, True)]


class TestBoundaryGuards:
    """The Wheel refuses returns that would silently corrupt other blocks."""

    def test_dropping_the_noise_model_raises(self, rng):
        obs = make_observed(rng, noise=FlatPSD(3.0))

        class Rebuilder(ConstBlock):
            def update(self, residual):
                # rebuilds instead of using replace() -> loses noise silently
                return Residuals(
                    tdi=residual.tdi,
                    sample_rate=residual.sample_rate,
                    channels=residual.channels,
                    tdi_generation=residual.tdi_generation,
                    observable=residual.observable,
                    n_samples=residual.n_samples,
                    epoch=residual.epoch,
                    domain=residual.domain,
                )

        wheel = Wheel(obs)
        wheel.add(Rebuilder("r", 0.0))
        with pytest.raises(ValueError, match="dropped the noise model"):
            wheel.run(1)

    def test_non_finite_return_raises_and_names_the_channel(self, observed):
        class Blowup(ConstBlock):
            def update(self, residual):
                return replace(
                    residual, tdi={ch: arr * np.nan for ch, arr in residual.tdi.items()}
                )

        wheel = Wheel(observed)
        wheel.add(Blowup("boom", 0.0))
        with pytest.raises(ValueError, match="non-finite sample"):
            wheel.run(1)

    def test_missing_update_is_caught_at_registration(self, observed):
        class NoBlockUpdate:
            name = "noupdate"

            def start(self, residual):
                return residual

        with pytest.raises(TypeError, match="does not implement update"):
            Wheel(observed).add(NoBlockUpdate())

    def test_withdrawing_a_model_warns(self, observed):
        class Fickle(ConstBlock):
            def __init__(self, name, value):
                super().__init__(name, value)
                self.calls = 0

            def update(self, residual):
                self.calls += 1
                if self.calls == 2:
                    return residual  # "nothing changed" -- silently withdraws
                return super().update(residual)

        wheel = Wheel(observed)
        wheel.add(Fickle("f", 1.0))
        with pytest.warns(ModelWithdrawnWarning, match="contributes nothing"):
            wheel.run(2)

    def test_the_withdrawal_warning_points_at_the_caller(self, observed):
        # the location IS the payload: it tells the user which call did it
        class Fickle(ConstBlock):
            def update(self, residual):
                return residual

        wheel = Wheel(observed)
        wheel.add(ConstBlock("f", 1.0))
        wheel._blocks[0].__class__ = Fickle
        with pytest.warns(ModelWithdrawnWarning) as record:
            wheel.run(1)  # <- this line must be blamed
        assert record[0].filename == __file__, record[0].filename

    def test_a_legitimate_death_move_can_be_silenced_precisely(self, observed):
        # a reversible-jump block whose last source dies is CORRECT; the user
        # must be able to keep -W error while ignoring exactly this heuristic
        import warnings as _w

        class Death(ConstBlock):
            def update(self, residual):
                self.value = 0.0  # k -> 0 sources
                return self._subtract(residual)

        wheel = Wheel(observed)
        wheel.add(Death("rj", 1.0))
        with _w.catch_warnings():
            _w.simplefilter("error")
            _w.filterwarnings("ignore", category=ModelWithdrawnWarning)
            wheel.run(1)

    def test_a_genuinely_zero_model_does_not_warn(self, observed):
        # EchoBlock contributes zero on every update; that is not a withdrawal
        import warnings as _w

        wheel = Wheel(observed)
        wheel.add(EchoBlock("echo"))
        with _w.catch_warnings():
            _w.simplefilter("error")  # any warning becomes a failure
            wheel.run(3)

    def test_real_frequency_domain_data_is_rejected(self, rng):
        # a real one-sided spectrum would let a block return `.real` and be
        # credited with the entire imaginary part as its model
        with pytest.raises(TypeError, match="real but domain='frequency'"):
            make_observed(
                rng,
                domain="frequency",
                tdi={ch: np.zeros(33) for ch in ("A", "E", "T")},
                n_samples=64,
            )

    def test_wider_model_promotes_rather_than_raising(self, rng):
        # float32 observed + float64 model must promote, not fail the way the
        # old in-place subtraction did
        obs = make_observed(
            rng, tdi={ch: np.zeros(64, np.float32) for ch in ("A", "E", "T")}
        )

        class Wider(ConstBlock):
            def update(self, residual):
                return replace(
                    residual,
                    tdi={
                        ch: arr - np.ones(64, np.float64)
                        for ch, arr in residual.tdi.items()
                    },
                )

        wheel = Wheel(obs)
        wheel.add(Wider("w", 0.0))
        wheel.run(1)
        assert wheel.residual().tdi["A"].dtype == np.float64
        assert wheel.observed.tdi["A"].dtype == np.float32  # left alone

    def test_invariant_list_is_exactly_this(self):
        """Pinned literally: parametrizing over _INVARIANT would let a field be
        deleted from the tuple AND from its own test in one edit."""
        assert Wheel._INVARIANT == (
            "channels",
            "n_samples",
            "sample_rate",
            "tdi_generation",
            "observable",
            "domain",
            "epoch",
        )

    @pytest.mark.parametrize(
        "field",
        [
            "channels",
            "n_samples",
            "sample_rate",
            "tdi_generation",
            "observable",
            "domain",
            "epoch",
        ],
    )
    def test_every_invariant_run_setting_is_guarded(self, rng, field):
        """Each field must actually be enforced, not merely listed."""
        obs = make_observed(rng, sample_rate=1.0)
        bad = {
            "channels": ("A",),
            "n_samples": obs.n_samples * 2,
            "sample_rate": 2.0,
            "tdi_generation": "9.9",
            "observable": "phase",
            "domain": "frequency",
            "epoch": 12345.0,
        }[field]

        class Cheat(ConstBlock):
            def update(self, residual):
                kw = {field: bad}
                if field in ("channels", "n_samples", "domain"):
                    # keep tdi self-consistent so Residuals' own checks pass and
                    # the Wheel's invariant check is what fires
                    if field == "channels":
                        kw["tdi"] = {"A": residual.tdi["A"]}
                    elif field == "n_samples":
                        kw["tdi"] = {ch: np.zeros(bad) for ch in residual.channels}
                    else:
                        kw["tdi"] = {
                            ch: np.zeros(residual.n_samples // 2 + 1, complex)
                            for ch in residual.channels
                        }
                return replace(residual, **kw)

        wheel = Wheel(obs)
        wheel.add(Cheat("cheat", 0.0))
        with pytest.raises(ValueError, match=f"changed the run setting {field!r}"):
            wheel.run(1)


class TestDocumentedContracts:
    """Guarantees the docstrings make that nothing else pins."""

    def test_start_is_handed_data_minus_already_registered_blocks(self, rng):
        """add()'s docstring promise. A block that *reads* what it is handed
        is the only way to see this -- blocks that subtract a constant give
        the same ledger entry either way."""
        obs = make_observed(rng)
        seen = {}

        class Reader:
            def __init__(self, name, value):
                self.name, self.value = name, value

            def start(self, residual):
                seen[self.name] = residual.tdi["A"].copy()
                return replace(
                    residual,
                    tdi={ch: arr - self.value for ch, arr in residual.tdi.items()},
                )

            def update(self, residual):
                return replace(
                    residual,
                    tdi={ch: arr - self.value for ch, arr in residual.tdi.items()},
                )

        wheel = Wheel(obs)
        wheel.add(Reader("first", 3.0))
        wheel.add(Reader("second", 5.0))
        # the first joiner sees raw data; the second sees it minus the first
        np.testing.assert_allclose(seen["first"], obs.tdi["A"])
        np.testing.assert_allclose(seen["second"], obs.tdi["A"] - 3.0)

    def test_residual_returns_fresh_arrays(self, rng):
        """residual()'s docstring: 'callers may mutate freely'."""
        obs = make_observed(rng)
        snapshot = obs.tdi["A"].copy()
        wheel = Wheel(obs)  # no blocks: the aliasing case
        wheel.residual().tdi["A"][0] = -999.0
        np.testing.assert_array_equal(obs.tdi["A"], snapshot)
        wheel.add(ConstBlock("a", 1.0))
        wheel.residual(exclude="a").tdi["A"][0] = -999.0
        np.testing.assert_array_equal(obs.tdi["A"], snapshot)

    def test_contribution_returns_a_copy_of_the_ledger_entry(self, observed):
        wheel = Wheel(observed)
        wheel.add(ConstBlock("a", 2.0))
        got = wheel.contribution("a")
        got["A"][:] = 100.0
        np.testing.assert_allclose(wheel.contribution("a")["A"], 2.0)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize("channel", ["A", "E", "T"])
    def test_non_finite_is_caught_in_any_channel_and_any_form(self, rng, bad, channel):
        obs = make_observed(rng)

        class Blowup(ConstBlock):
            def update(self, residual):
                tdi = dict(residual.tdi)
                tdi[channel] = tdi[channel].copy()
                tdi[channel][3] = bad
                return replace(residual, tdi=tdi)

        wheel = Wheel(obs)
        wheel.add(Blowup("b", 0.0))
        with pytest.raises(ValueError, match="non-finite sample"):
            wheel.run(1)

    def test_withdrawal_is_judged_across_all_channels_not_any(self, rng):
        """_all_zero must be all(), not any(): a model that is zero in one
        channel and non-zero in others has NOT been withdrawn."""
        import warnings as _w

        obs = make_observed(rng)

        class PartlyZero:
            name = "p"

            def _model(self, residual):
                # non-zero in A and E, zero in T
                return replace(
                    residual,
                    tdi={
                        ch: (arr - 1.0 if ch != "T" else arr)
                        for ch, arr in residual.tdi.items()
                    },
                )

            def start(self, residual):
                return self._model(residual)

            def update(self, residual):
                return self._model(residual)

        wheel = Wheel(obs)
        wheel.add(PartlyZero())
        with _w.catch_warnings():
            _w.simplefilter("error")  # a spurious warning fails the test
            wheel.run(2)
        np.testing.assert_allclose(wheel.contribution("p")["T"], 0.0)
        np.testing.assert_allclose(wheel.contribution("p")["A"], 1.0)

    def test_a_real_withdrawal_is_caught_even_if_one_channel_was_always_zero(self, rng):
        obs = make_observed(rng)

        class ThenStops:
            name = "s"

            def __init__(self):
                self.calls = 0

            def start(self, residual):
                return replace(
                    residual,
                    tdi={
                        ch: (arr - 1.0 if ch != "T" else arr)
                        for ch, arr in residual.tdi.items()
                    },
                )

            def update(self, residual):
                return residual  # withdraws A and E

        wheel = Wheel(obs)
        wheel.add(ThenStops())
        with pytest.warns(ModelWithdrawnWarning):
            wheel.run(1)

    def test_block_name_must_be_a_string_not_just_non_empty(self, observed):
        class Numbered(ConstBlock):
            pass

        block = Numbered("x", 0.0)
        block.name = 5
        with pytest.raises(ValueError, match="non-empty string"):
            Wheel(observed).add(block)

    def test_missing_name_gets_the_protocol_message(self, observed):
        class Anonymous:
            def start(self, residual):
                return residual

            def update(self, residual):
                return residual

        with pytest.raises((ValueError, TypeError), match="name"):
            Wheel(observed).add(Anonymous())


class TestObservedDataIsChecked:
    """Wheel.__init__ checks the data itself, so the first block to touch it
    is not blamed by the return-value guard for damage it did not do."""

    def test_non_finite_observed_data_is_refused_at_construction(self, rng):
        obs = make_observed(rng)
        obs.tdi["E"][7] = np.nan
        with pytest.raises(ValueError, match=r"observed.tdi\['E'\] has 1 non-finite"):
            Wheel(obs)

    def test_the_message_points_at_the_missing_gap_support(self, rng):
        """NaN is how a user marks a gap today, and gaps are not in the
        contract yet -- the error has to say so or it reads as a bug."""
        obs = make_observed(rng)
        obs.tdi["A"][:3] = np.nan
        with pytest.raises(ValueError, match="no data-quality mask yet"):
            Wheel(obs)

    def test_finite_data_constructs_normally(self, observed):
        assert Wheel(observed).residual() is not None


class TestNoiseOwnership:
    """`Residuals.noise` is a single slot, so a second writer silently wins.
    The Wheel cannot forbid that (handing ownership over may be deliberate),
    but it must not stay silent either."""

    @staticmethod
    def _noise_block(name, level):
        class NoiseWriter:
            def __init__(self):
                self.name = name

            def start(self, residual):
                return replace(residual, noise=FlatPSD(level))

            def update(self, residual):
                # a real noise block re-estimates, so it returns a NEW object
                # every cycle -- that must not read as an overwrite
                return replace(residual, noise=FlatPSD(level))

        return NoiseWriter()

    def test_two_noise_blocks_warn_that_one_is_being_lost(self, observed):
        wheel = Wheel(observed)
        wheel.add(self._noise_block("instrument", 1.0))
        with pytest.warns(NoiseOverwrittenWarning, match="'instrument' owns"):
            wheel.add(self._noise_block("confusion", 2.0))

    def test_the_message_names_both_blocks_and_the_fix(self, observed):
        wheel = Wheel(observed)
        wheel.add(self._noise_block("a", 1.0))
        with pytest.warns(NoiseOverwrittenWarning) as rec:
            wheel.add(self._noise_block("b", 2.0))
        msg = str(rec[0].message)
        assert "b.start" in msg and "'a'" in msg
        assert "single" in msg and "combined model" in msg

    def test_one_noise_block_re_estimating_every_cycle_is_silent(self, observed):
        """The common case: the same block writes a fresh model each cycle."""
        wheel = Wheel(observed)
        wheel.add(self._noise_block("noise", 1.0))
        wheel.add(ConstBlock("signal", 0.5))
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails this test
            wheel.run(3)

    def test_taking_over_the_datasets_own_model_is_silent(self, rng):
        """A noise block replacing the model the data arrived with is the
        documented workflow, not an overwrite -- nobody owned it."""
        obs = make_observed(rng, noise=FlatPSD(9.0))
        wheel = Wheel(obs)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            wheel.add(self._noise_block("noise", 1.0))
        assert wheel.residual().noise.level == 1.0

    def test_a_block_passing_noise_through_does_not_claim_ownership(self, observed):
        """Signal blocks return the same noise object they were handed; that
        must not make them the owner and frame the real noise block as the
        intruder on the next cycle."""
        wheel = Wheel(observed)
        wheel.add(ConstBlock("signal", 0.5))  # passes noise through untouched
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            wheel.add(self._noise_block("noise", 1.0))
            wheel.run(2)
