"""Covariance: one array per channel pair, read through channel attributes."""

import numpy as np
import pytest

from enchilada.covariance import Covariance

AE = ("A", "E")


def diagonal_pairs(n=8, a=1.0, e=4.0):
    return {("A", "A"): np.full(n, a), ("E", "E"): np.full(n, e)}


def full_pairs(n=4):
    a = np.diag(np.full(n, 2.0)) + 0.5 * np.eye(n, k=1) + 0.5 * np.eye(n, k=-1)
    e = np.diag(np.arange(1.0, n + 1))
    ae = 0.1 * np.eye(n)
    return {("A", "A"): a, ("E", "E"): e, ("A", "E"): ae}


class TestConstruction:
    def test_diagonal_entries_are_stored_as_read_only_copies(self):
        pairs = diagonal_pairs()
        cov = Covariance(pairs, channel_names=AE)
        assert cov.form == "diagonal"
        assert cov.channel_names == AE
        assert not cov.pairs[("A", "A")].flags.writeable
        assert not np.shares_memory(cov.pairs[("A", "A")], pairs[("A", "A")])
        np.testing.assert_array_equal(cov.A, 1.0)
        np.testing.assert_array_equal(cov.E, 4.0)

    def test_reversed_key_is_stored_in_canonical_orientation(self):
        pairs = diagonal_pairs()
        pairs[("E", "A")] = np.full(8, 0.5 + 0.25j)
        cov = Covariance(pairs, channel_names=AE)
        assert set(cov.pairs) == {("A", "A"), ("E", "E"), ("A", "E")}
        np.testing.assert_array_equal(cov.pairs[("A", "E")], 0.5 - 0.25j)
        np.testing.assert_array_equal(cov.EA, 0.5 + 0.25j)

    def test_both_orientations_are_rejected(self):
        pairs = diagonal_pairs()
        pairs[("A", "E")] = np.zeros(8)
        pairs[("E", "A")] = np.zeros(8)
        with pytest.raises(ValueError, match="both orientations"):
            Covariance(pairs, channel_names=AE)

    @pytest.mark.parametrize(
        "mutate, error, match",
        [
            (lambda p: p.pop(("E", "E")), ValueError, "missing the diagonal"),
            (
                lambda p: p.update({("T", "T"): np.ones(8)}),
                ValueError,
                "unknown channel",
            ),
            (lambda p: p.update({"A": np.ones(8)}), TypeError, "tuples"),
            (lambda p: p.update({("A", "A"): [1.0] * 8}), TypeError, "numpy array"),
            (
                lambda p: p.update({("A", "A"): np.ones(8, dtype=int)}),
                TypeError,
                "dtype",
            ),
            (
                lambda p: p.update({("A", "A"): np.full(8, np.nan)}),
                ValueError,
                "finite",
            ),
            (
                lambda p: p.update({("A", "A"): np.ones(8, complex)}),
                ValueError,
                "must be real",
            ),
            (lambda p: p.update({("A", "A"): -np.ones(8)}), ValueError, "nonnegative"),
            (lambda p: p.update({("A", "E"): np.zeros(7)}), ValueError, "one shape"),
        ],
    )
    def test_structural_rules(self, mutate, error, match):
        pairs = diagonal_pairs()
        mutate(pairs)
        with pytest.raises(error, match=match):
            Covariance(pairs, channel_names=AE)

    def test_unknown_form_and_bad_full_shape(self):
        with pytest.raises(ValueError, match="form must be"):
            Covariance(diagonal_pairs(), channel_names=AE, form="banded")
        with pytest.raises(ValueError, match="S \\+ S"):
            Covariance(diagonal_pairs(), channel_names=AE, form="full")

    def test_full_form_requires_hermitian_diagonal_blocks(self):
        pairs = full_pairs()
        pairs[("A", "A")][0, 1] = 5.0  # breaks symmetry
        with pytest.raises(ValueError, match="Hermitian"):
            Covariance(pairs, channel_names=AE, form="full")

    def test_full_form_symmetrises_roundoff_exactly(self):
        pairs = full_pairs()
        pairs[("A", "A")][0, 1] += 1e-15
        cov = Covariance(pairs, channel_names=AE, form="full")
        np.testing.assert_array_equal(cov.A, cov.A.T)

    def test_zero_variance_point_must_have_zero_cross_terms(self):
        pairs = diagonal_pairs()
        pairs[("A", "A")] = pairs[("A", "A")].copy()
        pairs[("A", "A")][3] = 0.0
        pairs[("A", "E")] = np.full(8, 0.1)
        with pytest.raises(ValueError, match="zero cross-covariance"):
            Covariance(pairs, channel_names=AE)
        pairs[("A", "E")] = pairs[("A", "E")].copy()
        pairs[("A", "E")][3] = 0.0
        cov = Covariance(pairs, channel_names=AE)
        assert cov.A[3] == 0.0

    def test_channel_matrices_must_be_positive_definite(self):
        pairs = diagonal_pairs(a=1.0, e=1.0)
        pairs[("A", "E")] = np.full(8, 1.5)  # |C_AE|^2 > C_AA C_EE
        with pytest.raises(ValueError, match="positive definite"):
            Covariance(pairs, channel_names=AE)

    def test_full_form_block_matrix_must_be_positive_definite(self):
        pairs = full_pairs()
        pairs[("A", "E")] = 3.0 * np.eye(4)
        with pytest.raises(ValueError, match="positive definite"):
            Covariance(pairs, channel_names=AE, form="full")

    @pytest.mark.parametrize("name", ["pairs", "copy", "form", "pair"])
    def test_channel_names_cannot_shadow_attributes(self, name):
        with pytest.raises(ValueError, match="collide"):
            Covariance({(name, name): np.ones(4)}, channel_names=(name,))

    def test_channel_names_must_be_unique_nonempty_strings(self):
        with pytest.raises(ValueError, match="unique non-empty"):
            Covariance({("A", "A"): np.ones(4)}, channel_names=("A", "A"))
        with pytest.raises(ValueError, match="non-empty tuple"):
            Covariance({}, channel_names=())


class TestAccess:
    def test_absent_cross_term_reads_as_zero(self):
        cov = Covariance(diagonal_pairs(), channel_names=AE)
        np.testing.assert_array_equal(cov.AE, 0.0)
        np.testing.assert_array_equal(cov.EA, 0.0)
        assert cov.AE.shape == (8,)
        assert cov.pair("A", "E") is not cov.pair("A", "E")  # fresh zeros

    def test_reversed_pair_is_conjugate_transpose(self):
        pairs = diagonal_pairs()
        pairs[("A", "E")] = np.full(8, 0.5 + 0.25j)
        cov = Covariance(pairs, channel_names=AE)
        np.testing.assert_array_equal(cov.EA, np.conj(cov.AE))
        full = Covariance(full_pairs(), channel_names=AE, form="full")
        np.testing.assert_array_equal(full.EA, full.AE.T)

    def test_full_form_conjugate_transpose_swaps_point_axes(self):
        rng = np.random.default_rng(3)
        ae = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        a = np.diag(np.full(4, 50.0))
        e = np.diag(np.full(4, 50.0))
        cov = Covariance(
            {("A", "A"): a, ("E", "E"): e, ("A", "E"): ae},
            channel_names=AE,
            form="full",
        )
        np.testing.assert_array_equal(cov.EA, ae.conj().T)

    def test_stored_pairs_return_the_read_only_array_itself(self):
        cov = Covariance(diagonal_pairs(), channel_names=AE)
        assert cov.A is cov.pairs[("A", "A")]
        with pytest.raises(ValueError):
            cov.A[0] = 7.0

    def test_multi_letter_names_use_pair(self):
        cov = Covariance(
            {
                ("X1", "X1"): np.ones(4),
                ("Y1", "Y1"): np.ones(4),
                ("X1", "Y1"): np.full(4, 0.2),
            },
            channel_names=("X1", "Y1"),
        )
        np.testing.assert_array_equal(cov.X1Y1, 0.2)
        np.testing.assert_array_equal(cov.pair("Y1", "X1"), 0.2)

    def test_ambiguous_attribute_names_are_errors(self):
        # "AAA" splits as ("A", "AA") and ("AA", "A"); both are valid pairs.
        cov = Covariance(
            {("A", "A"): np.ones(2), ("AA", "AA"): np.ones(2)},
            channel_names=("A", "AA"),
        )
        with pytest.raises(AttributeError, match="ambiguous"):
            _ = cov.AAA
        np.testing.assert_array_equal(cov.pair("A", "AA"), 0.0)

    def test_unknown_attribute_lists_valid_names(self):
        cov = Covariance(diagonal_pairs(), channel_names=AE)
        with pytest.raises(AttributeError, match="'T'.*channel attributes.*'AE'"):
            _ = cov.T
        assert not hasattr(cov, "T")
        assert hasattr(cov, "AE")

    def test_pair_rejects_unknown_channels(self):
        cov = Covariance(diagonal_pairs(), channel_names=AE)
        with pytest.raises(ValueError, match="unknown channel 'T'"):
            cov.pair("A", "T")


class TestCopy:
    def test_copy_is_independent_and_revalidated(self):
        cov = Covariance(diagonal_pairs(), channel_names=AE)
        copied = cov.copy()
        assert copied is not cov
        assert not np.shares_memory(copied.A, cov.A)
        np.testing.assert_array_equal(copied.A, cov.A)
        assert copied.channel_names == cov.channel_names and copied.form == cov.form

    def test_copy_catches_edited_arrays(self):
        cov = Covariance(diagonal_pairs(), channel_names=AE)
        cov.pairs[("A", "A")].setflags(write=True)
        cov.pairs[("A", "A")][:] = -1.0
        with pytest.raises(ValueError, match="nonnegative"):
            cov.copy()

    def test_validated_copy_skips_checks_but_copies_buffers(self):
        cov = Covariance(diagonal_pairs(), channel_names=AE)
        snapshot = cov._copy_validated()
        assert not np.shares_memory(snapshot.A, cov.A)
        assert not snapshot.A.flags.writeable
        np.testing.assert_array_equal(snapshot.A, cov.A)
