"""Covariance: one array per channel pair, read through channel attributes."""

import copy
import pickle

import numpy as np
import pytest

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
        cov = Covariance(
            {
                ("A", "A"): np.array([1.0]),
                ("E", "E"): np.array([2.0]),
                ("T", "T"): np.array([1e-17]),
            },
            channel_names=AET,
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
