"""Covariance: one array per channel pair, read through channel attributes."""

import copy
import pickle

import numpy as np
import pytest

from conftest import make_observed
from enchilada import L1Data, WDMGrid
from enchilada.covariance import AET_FROM_XYZ, Covariance

AE = ("A", "E")
XYZ = ("X", "Y", "Z")
AET = ("A", "E", "T")


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

    def test_canonical_orientation_follows_channel_order_not_alphabet(self):
        pairs = diagonal_pairs()
        pairs[("A", "E")] = np.full(8, 0.5 + 0.25j)
        cov = Covariance(pairs, channel_names=("E", "A"))
        assert set(cov.pairs) == {("E", "E"), ("A", "A"), ("E", "A")}
        np.testing.assert_array_equal(cov.pairs[("E", "A")], 0.5 - 0.25j)
        np.testing.assert_array_equal(cov.AE, 0.5 + 0.25j)

    def test_pairs_must_be_a_mapping(self):
        with pytest.raises(TypeError, match="dict of"):
            Covariance([("A", "A")], channel_names=("A",))

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
            (
                lambda p: p.update(
                    {("A", "A"): np.ma.masked_invalid([1.0, np.nan] + [1.0] * 6)}
                ),
                ValueError,
                "finite",
            ),
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

    def test_full_form_accepts_complex_hermitian_diagonal_blocks(self):
        pairs = full_pairs()
        block = np.diag(np.full(4, 5.0)) + 0.5j * (np.eye(4, k=1) - np.eye(4, k=-1))
        block[0, 1] += 1e-15  # roundoff asymmetry, symmetrised on storage
        block[2, 2] += 1e-15j  # roundoff on the point-diagonal
        pairs[("A", "A")] = block
        cov = Covariance(pairs, channel_names=AE, form="full")
        assert np.iscomplexobj(cov.A)
        np.testing.assert_array_equal(cov.A, cov.A.conj().T)
        np.testing.assert_array_equal(np.diagonal(cov.A).imag, 0.0)

    @pytest.mark.parametrize(
        "imaginary",
        [
            0.5j * (np.eye(4, k=1) + np.eye(4, k=-1)),  # not conjugate-symmetric
            0.5j * np.eye(4),  # imaginary point-diagonal
        ],
    )
    def test_full_form_rejects_non_hermitian_complex_blocks(self, imaginary):
        pairs = full_pairs()
        pairs[("A", "A")] = np.diag(np.full(4, 5.0)) + imaginary
        with pytest.raises(ValueError, match="Hermitian"):
            Covariance(pairs, channel_names=AE, form="full")

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
        with pytest.raises(TypeError, match="tuple of channel-name strings"):
            Covariance({("A", "A"): np.ones(4)}, channel_names="AE")
        with pytest.raises(TypeError, match="tuple of channel-name strings"):
            Covariance({("A", "A"): np.ones(4)}, channel_names=("A", 1))


def edge_pixel_pairs():
    """A (5, 6) WDM-like diagonal: row 0 is an edge band in every channel, row 4
    is zero in A only, and the A-E cross term lives where both are active."""
    a = np.ones((5, 6))
    a[[0, 4]] = 0.0
    e = np.full((5, 6), 2.0)
    e[0] = 0.0
    t = np.full((5, 6), 3.0)
    t[0] = 0.0
    ae = np.zeros((5, 6))
    ae[1:4] = 0.3
    return {("A", "A"): a, ("E", "E"): e, ("T", "T"): t, ("A", "E"): ae}


class TestInactivePoints:
    def test_structural_zeros_construct(self):
        cov = Covariance(edge_pixel_pairs(), channel_names=("A", "E", "T"))
        np.testing.assert_array_equal(cov.AE[cov.A == 0], 0.0)
        np.testing.assert_array_equal(cov.AE[1:4], 0.3)
        np.testing.assert_array_equal(cov.T[0], 0.0)

    def test_active_pixels_must_still_be_positive_definite(self):
        pairs = edge_pixel_pairs()
        pairs[("A", "E")][2, 3] = 1.5  # > sqrt(C_AA C_EE) = sqrt(2) there
        with pytest.raises(ValueError, match="positive definite"):
            Covariance(pairs, channel_names=("A", "E", "T"))

    def test_more_than_64_channels_group_inactive_points_exactly(self):
        # Bit codes would collide past 64 channels and skip point 1's check.
        names = tuple(f"c{i}" for i in range(65))
        pairs = {(ch, ch): np.ones(2) for ch in names}
        pairs[("c64", "c64")] = np.array([0.0, 1.0])  # inactive at point 0 only
        pairs[("c0", "c64")] = np.array([0.0, 1.5])  # not PD at point 1
        with pytest.raises(ValueError, match="positive definite"):
            Covariance(pairs, channel_names=names)
        pairs[("c0", "c64")] = np.array([0.0, 0.5])
        cov = Covariance(pairs, channel_names=names)
        assert cov.c64[0] == 0.0


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

    def test_uninitialised_instance_has_no_channel_attributes(self):
        assert not hasattr(object.__new__(Covariance), "A")

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

    def test_deepcopy_and_pickle_round_trip(self):
        pairs = diagonal_pairs()
        pairs[("A", "E")] = np.full(8, 0.5 + 0.25j)
        cov = Covariance(pairs, channel_names=AE)
        for restored in (copy.deepcopy(cov), pickle.loads(pickle.dumps(cov))):
            assert restored.channel_names == cov.channel_names
            assert restored.form == cov.form
            assert set(restored.pairs) == set(cov.pairs)
            for key, arr in cov.pairs.items():
                np.testing.assert_array_equal(restored.pairs[key], arr)


def xyz_pairs(n=6, with_cross=True):
    rng = np.random.default_rng(5)
    pairs = {
        ("X", "X"): np.full(n, 2.0),
        ("Y", "Y"): np.full(n, 3.0),
        ("Z", "Z"): np.full(n, 4.0),
    }
    if with_cross:
        pairs[("X", "Y")] = rng.uniform(-0.5, 0.5, n) + 1j * rng.uniform(-0.5, 0.5, n)
        pairs[("X", "Z")] = rng.uniform(-0.5, 0.5, n) + 1j * rng.uniform(-0.5, 0.5, n)
        pairs[("Y", "Z")] = rng.uniform(-0.5, 0.5, n) + 1j * rng.uniform(-0.5, 0.5, n)
    return pairs


def explicit_aet(cov):
    """Reference: per point, M C M^T with C the 3x3 channel matrix."""
    n = cov.X.size
    out = {}
    for p in range(n):
        c = np.array([[cov.pair(i, j)[p] for j in XYZ] for i in XYZ])
        out[p] = AET_FROM_XYZ @ c @ AET_FROM_XYZ.T
    return out


def full_xyz_pairs(n=3, complex_blocks=False):
    rng = np.random.default_rng(11)
    base = np.diag(np.full(n, 5.0)) + 0.5 * np.eye(n, k=1) + 0.5 * np.eye(n, k=-1)
    pairs = {
        ("X", "X"): base,
        ("Y", "Y"): 2 * base,
        ("Z", "Z"): 3 * base,
        ("X", "Y"): 0.2 * rng.standard_normal((n, n)),
        ("X", "Z"): 0.1 * np.eye(n),
        ("Y", "Z"): 0.2 * rng.standard_normal((n, n)),
    }
    if complex_blocks:
        twist = 0.3j * (np.eye(n, k=1) - np.eye(n, k=-1))  # Hermitian
        pairs[("X", "X")] = pairs[("X", "X")] + twist
        pairs[("Y", "Y")] = pairs[("Y", "Y")] - twist
        pairs[("Z", "Z")] = pairs[("Z", "Z")] + 2 * twist
        pairs[("X", "Y")] = pairs[("X", "Y")] + 0.2j * rng.standard_normal((n, n))
        pairs[("Y", "Z")] = pairs[("Y", "Z")] + 0.2j * rng.standard_normal((n, n))
    return pairs


def block_product_aet(cov):
    """Reference for the full form: (M kron I) C (M kron I)^T on the 3n x 3n C."""
    n = cov.X.shape[0]
    assembled = np.block([[cov.pair(i, j) for j in XYZ] for i in XYZ])
    rotation = np.kron(AET_FROM_XYZ, np.eye(n))
    product = rotation @ assembled @ rotation.T
    return {
        (first, second): product[a * n : (a + 1) * n, b * n : (b + 1) * n]
        for a, first in enumerate(AET)
        for b, second in enumerate(AET)
    }


def per_point_rotation(cov, source, matrix):
    """Reference: per flattened point, matrix @ C @ matrix.T over ``source``."""
    flat = {(i, j): cov.pair(i, j).reshape(-1) for i in source for j in source}
    points = flat[(source[0], source[0])].size
    return np.array(
        [
            matrix
            @ np.array([[flat[(i, j)][p] for j in source] for i in source])
            @ matrix.T
            for p in range(points)
        ]
    )


class TestDerivedBasis:
    def test_aet_matches_explicit_matrix_product(self):
        cov = Covariance(xyz_pairs(), channel_names=XYZ)
        reference = explicit_aet(cov)
        aet = cov.in_basis("AET")
        for a, first in enumerate(AET):
            for b, second in enumerate(AET):
                expected = np.array([reference[p][a, b] for p in range(cov.X.size)])
                np.testing.assert_allclose(
                    aet.pair(first, second), expected, atol=1e-14
                )
                # The same values through the derived attribute names.
                name = first if a == b else first + second
                np.testing.assert_allclose(getattr(cov, name), expected, atol=1e-14)
        np.testing.assert_allclose(cov.EA, np.conj(cov.AE))

    def test_full_form_derivation_is_elementwise(self):
        n = 3
        base = np.diag(np.full(n, 5.0)) + 0.5 * np.eye(n, k=1) + 0.5 * np.eye(n, k=-1)
        pairs = {
            ("X", "X"): base,
            ("Y", "Y"): 2 * base,
            ("Z", "Z"): 3 * base,
            ("X", "Z"): 0.1 * np.eye(n),
        }
        cov = Covariance(pairs, channel_names=XYZ, form="full")
        aet = cov.in_basis("AET")
        assert aet.form == "full"
        m = AET_FROM_XYZ
        expected_aa = sum(
            m[0, i] * m[0, j] * cov.pair(XYZ[i], XYZ[j])
            for i in range(3)
            for j in range(3)
        )
        np.testing.assert_allclose(aet.A, expected_aa, atol=1e-14)

    def test_full_form_attributes_match_block_product(self):
        cov = Covariance(full_xyz_pairs(), channel_names=XYZ, form="full")
        reference = block_product_aet(cov)
        assert not np.iscomplexobj(cov.in_basis("AET").A)  # real stays real
        for name in ("A", "E", "T", "AE", "AT", "ET", "EA"):
            key = (name, name) if len(name) == 1 else (name[0], name[1])
            np.testing.assert_allclose(getattr(cov, name), reference[key], atol=1e-14)

    def test_complex_full_form_keeps_hermitian_diagonal_blocks(self):
        cov = Covariance(
            full_xyz_pairs(complex_blocks=True), channel_names=XYZ, form="full"
        )
        reference = block_product_aet(cov)
        aet = cov.in_basis("AET")
        assert np.iscomplexobj(aet.A)
        for first in AET:
            for second in AET:
                np.testing.assert_allclose(
                    aet.pair(first, second), reference[(first, second)], atol=1e-14
                )

    @pytest.mark.parametrize("dtype", [float, complex])
    def test_small_variance_channel_converts_in_full_form(self, dtype):
        # LISA's T channel can sit ~1e-8 below X at low frequency. The derived
        # T block must still pass the Hermitian check scaled to its own size.
        n = 6
        rng = np.random.default_rng(13)

        def random_block():
            g = rng.standard_normal((n, n)).astype(dtype)
            if dtype is complex:
                g = g + 1j * rng.standard_normal((n, n))
            return g

        def hermitian_pd():
            g = random_block()
            return g @ g.conj().T + n * np.eye(n)

        a_blk, e_blk, t_blk = hermitian_pd(), hermitian_pd(), hermitian_pd()
        # A non-Hermitian A-E coupling makes the X, Y, Z cross blocks
        # non-Hermitian, so an entry and its mirror sum different values.
        ae_blk = 0.5 * random_block()
        zeros = np.zeros((n, n), dtype=dtype)
        aet_matrix = np.block(
            [
                [a_blk, ae_blk, zeros],
                [ae_blk.conj().T, e_blk, zeros],
                [zeros, zeros, 1e-8 * t_blk],
            ]
        )
        rotation = np.kron(AET_FROM_XYZ.T, np.eye(n))
        c = rotation @ aet_matrix @ rotation.T
        c = 0.5 * (c + c.conj().T)
        block = {
            (first, second): c[i * n : (i + 1) * n, j * n : (j + 1) * n]
            for i, first in enumerate(XYZ)
            for j, second in enumerate(XYZ)
            if i <= j
        }
        cov = Covariance(block, channel_names=XYZ, form="full")
        aet = cov.in_basis("AET")
        scale = np.abs(c).max()
        np.testing.assert_allclose(aet.T, 1e-8 * t_blk, rtol=0, atol=1e-14 * scale)
        reference = block_product_aet(cov)
        for first in AET:
            for second in AET:
                np.testing.assert_allclose(
                    aet.pair(first, second), reference[(first, second)], atol=1e-13
                )

    def test_round_trip_and_caching(self):
        cov = Covariance(xyz_pairs(), channel_names=XYZ)
        aet = cov.in_basis("AET")
        assert aet.channel_names == ("A", "E", "T")
        assert cov.in_basis("AET") is aet
        assert cov.in_basis("XYZ") is cov
        assert aet.in_basis("XYZ") is cov
        back = Covariance(dict(aet.pairs), channel_names=aet.channel_names).in_basis(
            "XYZ"
        )
        for first in XYZ:
            for second in XYZ:
                np.testing.assert_allclose(
                    back.pair(first, second), cov.pair(first, second), atol=1e-14
                )

    def test_white_equal_noise_stays_white(self):
        cov = Covariance({(ch, ch): np.full(4, 7.0) for ch in XYZ}, channel_names=XYZ)
        np.testing.assert_allclose(cov.A, 7.0)
        np.testing.assert_allclose(cov.E, 7.0)
        np.testing.assert_allclose(cov.T, 7.0)
        np.testing.assert_allclose(cov.AE, 0.0, atol=1e-14)
        np.testing.assert_allclose(cov.AT, 0.0, atol=1e-14)
        np.testing.assert_allclose(cov.ET, 0.0, atol=1e-14)

    def test_any_channel_order_works(self):
        pairs = xyz_pairs()
        cov = Covariance(pairs, channel_names=("Z", "X", "Y"))
        reference = Covariance(pairs, channel_names=XYZ)
        np.testing.assert_allclose(cov.A, reference.A)
        np.testing.assert_allclose(cov.ET, reference.ET)

    def test_aet_object_reads_xyz_names(self):
        relabel = dict(zip(XYZ, AET, strict=True))
        pairs = {(relabel[a], relabel[b]): v for (a, b), v in xyz_pairs().items()}
        aet = Covariance(pairs, channel_names=AET)
        np.testing.assert_array_equal(aet.XY, aet.in_basis("XYZ").pair("X", "Y"))
        reference = per_point_rotation(aet, AET, AET_FROM_XYZ.T)
        for a, first in enumerate(XYZ):
            for b, second in enumerate(XYZ):
                name = first if a == b else first + second
                np.testing.assert_allclose(
                    getattr(aet, name), reference[:, a, b], atol=1e-14
                )

    def test_copies_drop_the_cache(self):
        cov = Covariance(xyz_pairs(), channel_names=XYZ)
        derived = cov.in_basis("AET")
        for copied in (cov.copy(), cov._copy_validated()):
            assert copied.in_basis("AET") is not derived
            assert copied.in_basis("AET").in_basis("XYZ") is copied

    def test_wdm_shaped_diagonal_form_derives_per_point(self):
        pairs = {key: value.reshape(5, 4) for key, value in xyz_pairs(n=20).items()}
        assert np.iscomplexobj(pairs[("X", "Y")])
        cov = Covariance(pairs, channel_names=XYZ)
        aet = cov.in_basis("AET")
        reference = per_point_rotation(cov, XYZ, AET_FROM_XYZ)
        for a, first in enumerate(AET):
            for b, second in enumerate(AET):
                assert aet.pair(first, second).shape == (5, 4)
                np.testing.assert_allclose(
                    aet.pair(first, second).reshape(-1), reference[:, a, b], atol=1e-14
                )

    def test_derivation_below_float64_precision_explains_itself(self):
        # Many points, so at least one is indefinite whatever LAPACK's rounding.
        cov = Covariance(
            {
                ("A", "A"): np.linspace(1, 3, 64),
                ("E", "E"): np.linspace(2, 5, 64),
                ("T", "T"): np.full(64, 1e-20),
            },
            channel_names=("A", "E", "T"),
        )
        with pytest.raises(ValueError, match="cannot derive the XYZ basis"):
            _ = cov.X
        with pytest.raises(ValueError, match="cannot derive the XYZ basis"):
            cov.in_basis("XYZ")

    def test_derived_names_need_the_full_triple(self):
        cov = Covariance(diagonal_pairs(), channel_names=AE)
        with pytest.raises(AttributeError, match="channel attributes"):
            _ = cov.X
        with pytest.raises(ValueError, match="exactly X, Y, Z or A, E, T"):
            cov.in_basis("XYZ")
        with pytest.raises(ValueError, match="basis must be one of"):
            Covariance(xyz_pairs(), channel_names=XYZ).in_basis("ABC")


def single_channel_wdm_data():
    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=8)
    # L1Data infers wdm_grid from the
    # (num_frequency_divisions + 1, num_time_divisions) shape.
    data = L1Data(
        {"A": np.zeros(grid.array_shape)},
        sample_rate_hz=1.0,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
        data_domain="wdm",
    )
    return grid, data


def hermitian_block(rng, k, *, real_corners):
    """Complex Hermitian positive-definite (k, k) block for one channel."""
    b = rng.standard_normal((k, k)) + 1j * rng.standard_normal((k, k))
    block = b @ b.conj().T + k * np.eye(k)
    if real_corners:
        block[0, -1] = block[0, -1].real
        block[-1, 0] = np.conj(block[0, -1])
    return block


class FlatPSD:
    def __init__(self, level):
        self.level = level

    def psd(self, frequencies, channel=None):
        return np.full_like(frequencies, self.level, dtype=float)


class TestLike:
    def test_string_keys_and_form_inference(self, rng):
        data = make_observed(rng, channel_names=AE)
        n = data.num_time_samples
        cov = Covariance.like(
            data, {"A": np.ones(n), "E": np.full(n, 4.0), "AE": np.full(n, 0.5)}
        )
        assert cov.form == "diagonal"
        np.testing.assert_array_equal(cov.AE, 0.5)
        full = Covariance.like(data, {"A": np.eye(n), "E": np.eye(n)})
        assert full.form == "full"

    def test_bad_keys_and_shapes(self, rng):
        data = make_observed(rng, channel_names=AE)
        n = data.num_time_samples
        with pytest.raises(ValueError, match="neither a channel nor a pair"):
            Covariance.like(data, {"A": np.ones(n), "E": np.ones(n), "AT": np.ones(n)})
        with pytest.raises(ValueError, match="channel shape"):
            Covariance.like(data, {"A": np.ones(n + 1), "E": np.ones(n + 1)})
        with pytest.raises(ValueError, match="at least every channel"):
            Covariance.like(data, {})

    def test_two_keys_naming_one_pair_are_rejected(self, rng):
        data = make_observed(rng, channel_names=AE)
        n = data.num_time_samples
        ones = np.ones(n)
        with pytest.raises(ValueError, match=r"'A' and \('A', 'A'\) both name"):
            Covariance.like(data, {"A": ones, ("A", "A"): ones, "E": ones})
        with pytest.raises(ValueError, match=r"'AE' and \('A', 'E'\) both name"):
            Covariance.like(
                data,
                {"A": ones, "E": ones, "AE": 0.1 * ones, ("A", "E"): 0.1 * ones},
            )


class TestFactoriesAndReadBack:
    def test_from_variance_scalar_and_mapping(self, rng):
        data = make_observed(rng, channel_names=AE)
        cov = Covariance.from_variance(data, 0.25)
        np.testing.assert_array_equal(cov.A, 0.25)
        assert cov.noise_variance("A", data) == 0.25
        cov = Covariance.from_variance(data, {"A": 1.0, "E": 4.0})
        np.testing.assert_array_equal(cov.E, 4.0)
        with pytest.raises(ValueError, match="time-domain"):
            Covariance.from_variance(data.to_frequency(), 1.0)
        with pytest.raises(ValueError, match="keys must match"):
            Covariance.from_variance(data, {"A": 1.0})

    @pytest.mark.parametrize("variance", [0.0, -1.0, np.inf, np.nan])
    def test_from_variance_rejects_non_positive_or_non_finite(self, rng, variance):
        data = make_observed(rng, channel_names=AE)
        with pytest.raises(ValueError, match="time_sample_variance must be finite"):
            Covariance.from_variance(data, variance)

    def test_from_variance_rejects_negative_variance(self, rng):
        data = make_observed(rng, channel_names=AE)
        with pytest.raises(
            ValueError, match=r"time_sample_variance\['E'\] must be finite and positive"
        ):
            Covariance.from_variance(data, {"A": 1.0, "E": -1.0})

    @pytest.mark.parametrize("variance", ["2", 1 + 1j, [2.0, 3.0], True])
    def test_from_variance_requires_a_real_number(self, rng, variance):
        data = make_observed(rng, channel_names=AE)
        with pytest.raises(TypeError, match="time_sample_variance must be a real"):
            Covariance.from_variance(data, variance)

    def test_from_variance_mapping_requires_real_numbers(self, rng):
        data = make_observed(rng, channel_names=AE)
        with pytest.raises(
            TypeError, match=r"time_sample_variance\['A'\] must be a real number"
        ):
            Covariance.from_variance(data, {"A": "1.5", "E": 1.0})

    def test_from_psd_passes_each_channel_to_a_per_channel_model(self, rng):
        spectrum = make_observed(rng, channel_names=AE).to_frequency()

        class PerChannel:
            def psd(self, f, channel):
                return np.full_like(f, {"A": 3.0, "E": 7.0}[channel])

        cov = Covariance.from_psd(spectrum, PerChannel())
        tobs = spectrum.num_time_samples / spectrum.sample_rate_hz
        k = spectrum.num_time_samples // 2 + 1
        np.testing.assert_array_equal(cov.A, np.full(k, tobs / 2 * 3.0))
        np.testing.assert_array_equal(cov.E, np.full(k, tobs / 2 * 7.0))
        assert ("A", "E") not in cov.pairs

    def test_from_psd_accepts_a_frequency_only_model(self, rng):
        spectrum = make_observed(rng, channel_names=AE).to_frequency()

        class Shared:
            def psd(self, f):
                return 1.0 + f

        cov = Covariance.from_psd(spectrum, Shared())
        np.testing.assert_array_equal(cov.A, cov.E)
        f = np.fft.rfftfreq(spectrum.num_time_samples, d=1 / spectrum.sample_rate_hz)
        np.testing.assert_allclose(cov.noise_psd("A", spectrum), 1.0 + f)

    def test_from_psd_propagates_a_type_error_raised_inside_the_model(self, rng):
        spectrum = make_observed(rng, channel_names=AE).to_frequency()

        class Broken:
            # Fails only when given a channel, so retrying without the channel
            # after catching the TypeError would succeed and hide the bug.
            def psd(self, f, channel=None):
                if channel is not None:
                    raise TypeError("model bug")
                return np.ones_like(f)

        with pytest.raises(TypeError, match="model bug"):
            Covariance.from_psd(spectrum, Broken())

    @pytest.mark.parametrize(
        ("values", "error"),
        [
            (0, ValueError),
            (-1e-40, ValueError),
            (np.inf, ValueError),
            (np.nan, ValueError),
            (np.ones(2), ValueError),
            (1 + 0j, TypeError),
            ("invalid", TypeError),
        ],
    )
    def test_from_psd_rejects_invalid_model_values(self, rng, values, error):
        spectrum = make_observed(rng, channel_names=AE).to_frequency()

        class Noise:
            def psd(self, f, channel=None):
                return values

        with pytest.raises(error, match="PSD"):
            Covariance.from_psd(spectrum, Noise())

    def test_from_psd_fills_every_bin_including_dc(self, rng):
        data = make_observed(rng, channel_names=AE).to_frequency()
        cov = Covariance.from_psd(data, FlatPSD(2.0))
        tobs = data.num_time_samples / data.sample_rate_hz
        np.testing.assert_allclose(cov.A, tobs / 2 * 2.0)
        assert cov.A.shape == (data.num_time_samples // 2 + 1,)
        np.testing.assert_allclose(cov.noise_psd("A", data), 2.0)

    def test_from_psd_rejects_non_frequency_data_and_bad_models(self, rng):
        data = make_observed(rng, channel_names=AE)
        with pytest.raises(ValueError, match="frequency-domain"):
            Covariance.from_psd(data, FlatPSD(1.0))
        spectrum = data.to_frequency()

        class Diverging:
            def psd(self, f, channel=None):
                with np.errstate(divide="ignore"):
                    return 1.0 / f**2

        with pytest.raises(ValueError, match="f = 0"):
            Covariance.from_psd(spectrum, Diverging())
        with pytest.raises(TypeError, match="psd"):
            Covariance.from_psd(spectrum, object())

    @pytest.mark.parametrize("n", [63, 64])
    def test_white_noise_variance_round_trips_exactly(self, rng, n):
        data = make_observed(rng, num_time_samples=n, channel_names=("A",))
        sigma2 = 0.09
        spectrum = data.to_frequency()
        cov = Covariance.from_psd(spectrum, FlatPSD(2 * sigma2 / data.sample_rate_hz))
        assert cov.noise_variance("A", spectrum) == pytest.approx(sigma2, rel=1e-12)

    def test_noise_variance_needs_stationary_time_and_noise_psd_needs_frequency(
        self, rng
    ):
        data = make_observed(rng, channel_names=AE)
        n = data.num_time_samples
        cov = Covariance.like(data, {"A": np.linspace(1, 2, n), "E": np.ones(n)})
        with pytest.raises(ValueError, match="stationary"):
            cov.noise_variance("A", data)
        with pytest.raises(ValueError, match="frequency-domain"):
            cov.noise_psd("A", data)


class TestCheckCompatible:
    def test_channel_order_and_shape(self, rng):
        data = make_observed(rng, channel_names=AE)
        cov = Covariance.from_variance(data, 1.0)
        with pytest.raises(ValueError, match="channel_names"):
            cov.check_compatible(make_observed(rng, channel_names=("E", "A")))
        with pytest.raises(ValueError, match="shape"):
            cov.check_compatible(
                make_observed(rng, num_time_samples=32, channel_names=AE)
            )
        with pytest.raises(ValueError, match="shape"):
            cov.check_compatible(data.to_frequency())

    def test_time_arrays_must_be_real_and_positive(self, rng):
        data = make_observed(rng, channel_names=AE)
        n = data.num_time_samples
        complex_cross = Covariance(
            {
                ("A", "A"): np.ones(n),
                ("E", "E"): np.ones(n),
                ("A", "E"): np.full(n, 0.1j),
            },
            channel_names=AE,
        )
        with pytest.raises(ValueError, match="must be real"):
            complex_cross.check_compatible(data)
        zero = np.ones(n)
        zero[0] = 0.0
        with pytest.raises(ValueError, match="positive at every point"):
            Covariance(
                {("A", "A"): zero, ("E", "E"): np.ones(n)}, channel_names=AE
            ).check_compatible(data)

    def test_frequency_endpoints_must_be_real(self, rng):
        spectrum = make_observed(rng, channel_names=AE).to_frequency()
        k = spectrum.num_time_samples // 2 + 1
        cross = np.full(k, 0.1 + 0.1j)
        cov = Covariance(
            {("A", "A"): np.ones(k), ("E", "E"): np.ones(k), ("A", "E"): cross},
            channel_names=AE,
        )
        with pytest.raises(ValueError, match="both coefficients are real"):
            cov.check_compatible(spectrum)
        cross = cross.copy()
        cross[0] = 0.1
        cross[-1] = 0.1
        Covariance(
            {("A", "A"): np.ones(k), ("E", "E"): np.ones(k), ("A", "E"): cross},
            channel_names=AE,
        ).check_compatible(spectrum)

    def test_wdm_zeros_exactly_at_inactive_pixels(self):
        grid, data = single_channel_wdm_data()
        variances = np.where(grid.active_mask, 2.0, 0.0)
        Covariance({("A", "A"): variances}, channel_names=("A",)).check_compatible(data)
        with pytest.raises(ValueError, match="inactive WDM pixels"):
            Covariance(
                {("A", "A"): np.full(grid.array_shape, 2.0)}, channel_names=("A",)
            ).check_compatible(data)
        leaky = variances.copy()
        leaky[1, 0] = 0.0
        with pytest.raises(ValueError, match="positive at active WDM"):
            Covariance({("A", "A"): leaky}, channel_names=("A",)).check_compatible(data)

    def test_like_infers_diagonal_form_for_a_wdm_array(self):
        grid, data = single_channel_wdm_data()
        variances = np.where(grid.active_mask, 2.0, 0.0)
        assert Covariance.like(data, {"A": variances}).form == "diagonal"

    @pytest.mark.filterwarnings("error")  # a complex-to-float cast warns, not raises
    @pytest.mark.parametrize("n", [63, 64])
    def test_complex_hermitian_full_block_reads_back_a_real_psd(self, rng, n):
        spectrum = make_observed(
            rng, num_time_samples=n, channel_names=("A",)
        ).to_frequency()
        k = n // 2 + 1
        block = hermitian_block(rng, k, real_corners=n % 2 == 0)
        cov = Covariance({("A", "A"): block}, channel_names=("A",), form="full")
        cov.check_compatible(spectrum)
        psd = cov.noise_psd("A", spectrum)
        tobs = n / spectrum.sample_rate_hz
        assert psd.dtype == np.float64
        np.testing.assert_allclose(psd, 2 * np.diag(block).real / tobs)

    @pytest.mark.parametrize("n", [63, 64])
    def test_full_form_dc_row_may_pair_dc_with_complex_coefficients(self, rng, n):
        spectrum = make_observed(
            rng, num_time_samples=n, channel_names=("A",)
        ).to_frequency()
        k = n // 2 + 1
        block = hermitian_block(rng, k, real_corners=True)
        block[0, 1] = 1.0 + 1.0j
        block[1, 0] = 1.0 - 1.0j
        Covariance(
            {("A", "A"): block}, channel_names=("A",), form="full"
        ).check_compatible(spectrum)

    @pytest.mark.parametrize("n", [63, 64])
    def test_full_form_corner_must_be_real_only_when_nyquist_is_real(self, rng, n):
        spectrum = make_observed(
            rng, num_time_samples=n, channel_names=("A",)
        ).to_frequency()
        k = n // 2 + 1
        block = hermitian_block(rng, k, real_corners=True)
        block[0, -1] = block[0, -1].real + 0.5j
        block[-1, 0] = np.conj(block[0, -1])
        cov = Covariance({("A", "A"): block}, channel_names=("A",), form="full")
        if n % 2 == 0:
            with pytest.raises(ValueError, match="both coefficients are real"):
                cov.check_compatible(spectrum)
        else:
            cov.check_compatible(spectrum)  # the last bin is a complex interior

    @pytest.mark.parametrize("n", [63, 64])
    def test_diagonal_cross_pair_must_be_real_at_even_nyquist(self, rng, n):
        spectrum = make_observed(
            rng, num_time_samples=n, channel_names=AE
        ).to_frequency()
        k = n // 2 + 1
        cross = np.full(k, 0.1 + 0.1j)
        cross[0] = 0.1
        cov = Covariance(
            {("A", "A"): np.ones(k), ("E", "E"): np.ones(k), ("A", "E"): cross},
            channel_names=AE,
        )
        if n % 2 == 0:
            with pytest.raises(ValueError, match="both coefficients are real"):
                cov.check_compatible(spectrum)
        else:
            cov.check_compatible(spectrum)

    @pytest.mark.parametrize("n", [63, 64])
    def test_like_runs_check_compatible(self, rng, n):
        spectrum = make_observed(
            rng, num_time_samples=n, channel_names=AE
        ).to_frequency()
        k = n // 2 + 1
        with pytest.raises(ValueError, match="both coefficients are real"):
            Covariance.like(
                spectrum,
                {"A": np.ones(k), "E": np.ones(k), "AE": np.full(k, 0.1 + 0.1j)},
            )
