# -*- coding: utf-8 -*-
"""Masking incomplete geometries across semirings.

Brain geometries are routinely incomplete: the cortical medial wall is
excluded from surface analysis, and volumetric work is restricted to a
grey-matter mask.  An edge that reaches a masked vertex / voxel must
contribute the **semiring identity** to every reduction -- so no signal
leaks in from the uninteresting region.

These tests verify the contract behind ``nitrix.sparse.ell_mask``: for
every built-in algebra with a ``(*)``-annihilator (all but EUCLIDEAN),
masking a reaching edge makes it a true no-op, *independent of where its
index points*.  The preferred call is ``ell_mask(..., semiring=sr)``,
which reads ``sr.annihilator`` (B8); the legacy ``identity=`` form is
deprecated and emits a ``DeprecationWarning``.  EUCLIDEAN is the
documented exception -- its squared-difference has no annihilator, so
``semiring=EUCLIDEAN`` raises -- and tests pin both the guard and the
old leak-via-identity limitation.
"""
from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.semiring import (
    BOOLEAN,
    EUCLIDEAN,
    LOG,
    REAL,
    TROPICAL_MAX_PLUS,
    TROPICAL_MIN_PLUS,
    semiring_ell_matmul,
)
from nitrix.sparse import ELL, ell_mask, grid_identity, mesh_k_ring_adjacency
from nitrix.sparse import icosphere


# Algebras whose (*)-annihilator equals semiring.identity, so masking via
# the identity is a guaranteed no-op.
MASKABLE = [REAL, LOG, TROPICAL_MAX_PLUS, TROPICAL_MIN_PLUS]


def _apply(ell, B, semiring):
    return semiring_ell_matmul(
        ell.values, ell.indices, B, semiring=semiring,
        n_cols=ell.n_cols, backend='jax',
    )


def _random_float_ell(n, k_max, identity, seed):
    rng = np.random.default_rng(seed)
    indices = jnp.asarray(rng.integers(0, n, (n, k_max)).astype(np.int32))
    values = jnp.asarray(rng.standard_normal((n, k_max)))
    return ELL(values=values, indices=indices, n_cols=n, identity=identity)


@pytest.mark.parametrize(
    'semiring', MASKABLE, ids=[s.name for s in MASKABLE],
)
def test_masked_neighbour_is_noop_independent_of_target(semiring):
    '''Masking every edge that reaches a masked vertex makes the masked
    vertices' features irrelevant: perturbing them (even hugely) leaves
    every output unchanged.  This is the medial-wall / grey-matter
    "don't blur in masked signal" guarantee.
    '''
    n, k_max, ncol = 12, 5, 4
    masked_vertices = jnp.asarray([3, 8])  # the "medial wall"
    ell = _random_float_ell(n, k_max, semiring.identity, seed=0)

    col_valid = np.ones(n, dtype=bool)
    col_valid[np.asarray(masked_vertices)] = False
    ell = ell_mask(ell, jnp.asarray(col_valid), semiring=semiring)

    rng = np.random.default_rng(1)
    B = jnp.asarray(rng.standard_normal((n, ncol)))
    # Perturb only the masked rows, by a huge amount.
    B_pert = B.at[masked_vertices].add(1e6)

    out = _apply(ell, B, semiring)
    out_pert = _apply(ell, B_pert, semiring)
    assert bool(jnp.all(jnp.isfinite(out)))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(out_pert))


@pytest.mark.parametrize(
    'semiring', MASKABLE, ids=[s.name for s in MASKABLE],
)
def test_masking_equals_reducing_over_unmasked_edges(semiring):
    '''Masking the last ELL column with the identity gives the same
    result as reducing over only the first k_max-1 columns.
    '''
    n, k_max, ncol = 9, 4, 3
    ell = _random_float_ell(n, k_max, semiring.identity, seed=2)

    # Edge mask: drop the last column everywhere.
    edge_valid = np.ones((n, k_max), dtype=bool)
    edge_valid[:, -1] = False
    masked = ell_mask(ell, jnp.asarray(edge_valid), semiring=semiring)

    # Reference: an ELL with only the first k_max-1 columns.
    ref = ELL(
        values=ell.values[:, :-1], indices=ell.indices[:, :-1],
        n_cols=n, identity=semiring.identity,
    )

    rng = np.random.default_rng(3)
    B = jnp.asarray(rng.standard_normal((n, ncol)))
    np.testing.assert_allclose(
        np.asarray(_apply(masked, B, semiring)),
        np.asarray(_apply(ref, B, semiring)),
        atol=1e-12,
    )


def test_boolean_masking_is_noop():
    '''BOOLEAN (OR-over-AND) masks via its identity ``False``.'''
    n, k_max, ncol = 8, 4, 3
    rng = np.random.default_rng(0)
    indices = jnp.asarray(rng.integers(0, n, (n, k_max)).astype(np.int32))
    values = jnp.asarray(rng.integers(0, 2, (n, k_max)).astype(bool))
    ell = ELL(values=values, indices=indices, n_cols=n, identity=False)

    masked_vertices = jnp.asarray([1, 5])
    col_valid = np.ones(n, dtype=bool)
    col_valid[np.asarray(masked_vertices)] = False
    ell = ell_mask(ell, jnp.asarray(col_valid), semiring=BOOLEAN)

    B = jnp.asarray(rng.integers(0, 2, (n, ncol)).astype(bool))
    B_pert = B.at[masked_vertices].set(jnp.logical_not(B[masked_vertices]))
    out = _apply(ell, B, BOOLEAN)
    out_pert = _apply(ell, B_pert, BOOLEAN)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(out_pert))


def test_euclidean_semiring_mask_raises_no_annihilator():
    '''EUCLIDEAN has no (*)-annihilator, so the safe ``semiring=`` path
    refuses to mask it (rather than silently leaking).  This is the B8
    guard: ``EUCLIDEAN.annihilator is None`` -> a clear error.
    '''
    n, k_max = 8, 4
    ell = _random_float_ell(n, k_max, EUCLIDEAN.identity, seed=0)
    col_valid = np.ones(n, dtype=bool)
    col_valid[[2, 6]] = False
    with pytest.raises(ValueError, match='annihilator'):
        ell_mask(ell, jnp.asarray(col_valid), semiring=EUCLIDEAN)


def test_euclidean_has_no_annihilator_limitation():
    '''EUCLIDEAN's (a-b)^2 has no annihilator: masking via its identity
    (0) does NOT zero the edge -- it injects B[idx]^2.  This test pins
    the documented limitation (masking EUCLIDEAN must drop columns
    structurally, not set a value).  The deprecated ``identity=`` form
    is the only way to even attempt it; ``semiring=EUCLIDEAN`` raises
    (see the test above).
    '''
    n, k_max, ncol = 8, 4, 3
    ell = _random_float_ell(n, k_max, EUCLIDEAN.identity, seed=0)
    masked_vertices = jnp.asarray([2, 6])
    col_valid = np.ones(n, dtype=bool)
    col_valid[np.asarray(masked_vertices)] = False
    with pytest.warns(DeprecationWarning):
        ell = ell_mask(ell, jnp.asarray(col_valid), identity=EUCLIDEAN.identity)

    rng = np.random.default_rng(1)
    B = jnp.asarray(rng.standard_normal((n, ncol)))
    B_pert = B.at[masked_vertices].add(1e3)
    out = _apply(ell, B, EUCLIDEAN)
    out_pert = _apply(ell, B_pert, EUCLIDEAN)
    # The masked target leaks: perturbing it changes the result.
    assert not bool(jnp.allclose(out, out_pert))


def test_wrong_identity_leaks_under_max_plus_footgun():
    '''Masking with the REAL identity (0) and reducing under max-plus
    leaks the masked signal (0 + B = B); masking with the max-plus
    identity (-inf) does not.  Demonstrates why the identity must match
    the semiring you reduce under.
    '''
    n, k_max, ncol = 10, 5, 3
    indices = jnp.asarray(
        np.random.default_rng(0).integers(0, n, (n, k_max)).astype(np.int32),
    )
    values = jnp.zeros((n, k_max))  # max-plus weights: 0 means "gather"
    base = ELL(values=values, indices=indices, n_cols=n, identity=-jnp.inf)

    masked_vertices = jnp.asarray([4])
    col_valid = np.ones(n, dtype=bool)
    col_valid[np.asarray(masked_vertices)] = False

    # The deprecated identity= form is exactly the footgun B8's semiring=
    # path closes: an explicit (mismatched) scalar with no annihilator
    # check.  Both calls warn.
    with pytest.warns(DeprecationWarning):
        wrong = ell_mask(base, jnp.asarray(col_valid), identity=0.0)      # WRONG
        right = ell_mask(base, jnp.asarray(col_valid), identity=-jnp.inf)  # RIGHT

    rng = np.random.default_rng(1)
    B = jnp.asarray(rng.standard_normal((n, ncol)))
    B_pert = B.at[masked_vertices].add(1e6)

    # Wrong identity: the masked target leaks into the max.
    w0 = _apply(wrong, B, TROPICAL_MAX_PLUS)
    w1 = _apply(wrong, B_pert, TROPICAL_MAX_PLUS)
    assert not bool(jnp.allclose(w0, w1))

    # Right identity: no leak.
    r0 = _apply(right, B, TROPICAL_MAX_PLUS)
    r1 = _apply(right, B_pert, TROPICAL_MAX_PLUS)
    np.testing.assert_array_equal(np.asarray(r0), np.asarray(r1))


def test_medial_wall_mask_on_icosphere_smoothing():
    '''Realistic surface case: a k-ring smoother (REAL, row-stochastic)
    with a masked medial wall does not blur masked-vertex signal into
    the cortex.
    '''
    m = icosphere(2)  # 162 vertices
    adj = mesh_k_ring_adjacency(m, k=1, binary=False)  # row-stochastic, REAL
    n = m.n_vertices
    rng = np.random.default_rng(0)
    medial = rng.choice(n, size=20, replace=False)
    col_valid = np.ones(n, dtype=bool)
    col_valid[medial] = False
    adj_m = ell_mask(adj, jnp.asarray(col_valid), semiring=REAL)

    signal = jnp.asarray(rng.standard_normal((n, 1)))
    # Put pathological values on the medial wall.
    signal_bad = signal.at[jnp.asarray(medial)].set(1e6)

    out = _apply(adj_m, signal, REAL)
    out_bad = _apply(adj_m, signal_bad, REAL)
    np.testing.assert_allclose(np.asarray(out), np.asarray(out_bad), atol=1e-9)


def test_grey_matter_mask_on_grid():
    '''Realistic volume case: a grid operator restricted to a grey-matter
    mask does not pull signal from out-of-mask voxels (REAL).
    '''
    from nitrix.sparse import regular_grid_stencil

    grid_shape = (6, 6)
    offsets = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]
    weights = jnp.array([0.2, 0.2, 0.2, 0.2, 0.2])
    op = regular_grid_stencil(grid_shape, offsets, weights, boundary='replicate')
    n = int(np.prod(grid_shape))

    rng = np.random.default_rng(0)
    gm = np.ones(n, dtype=bool)
    out_of_gm = rng.choice(n, size=8, replace=False)
    gm[out_of_gm] = False
    op_m = ell_mask(op, jnp.asarray(gm), semiring=REAL)

    x = jnp.asarray(rng.standard_normal((n, 1)))
    x_bad = x.at[jnp.asarray(out_of_gm)].set(1e6)
    out = _apply(op_m, x, REAL)
    out_bad = _apply(op_m, x_bad, REAL)
    np.testing.assert_allclose(np.asarray(out), np.asarray(out_bad), atol=1e-9)


def test_ell_mask_rejects_bad_mask_shape():
    ell = _random_float_ell(6, 3, REAL.identity, seed=0)
    with pytest.raises(ValueError, match='valid.shape'):
        ell_mask(ell, jnp.ones((7,), dtype=bool), semiring=REAL)


def test_ell_mask_differentiable_through_values():
    '''Masking is a jnp.where; gradients flow through the unmasked
    values and stop at the masked positions.
    '''
    ell = _random_float_ell(6, 3, REAL.identity, seed=0)
    col_valid = jnp.asarray(np.array([1, 1, 0, 1, 0, 1], dtype=bool))
    B = jnp.asarray(np.random.default_rng(1).standard_normal((6, 2)))

    def loss(values):
        e = ELL(values=values, indices=ell.indices, n_cols=6, identity=0.0)
        e = ell_mask(e, col_valid, semiring=REAL)
        return jnp.sum(_apply(e, B, REAL) ** 2)

    g = jax.grad(loss)(ell.values)
    assert g.shape == ell.values.shape
    assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# B8: annihilator field + semiring= masking path
# ---------------------------------------------------------------------------


def test_builtin_annihilator_fields():
    '''The annihilator coincides with identity for the maskable
    algebras and is None for EUCLIDEAN (no annihilator).
    '''
    assert REAL.annihilator == 0.0
    assert LOG.annihilator == -jnp.inf
    assert TROPICAL_MAX_PLUS.annihilator == -jnp.inf
    assert TROPICAL_MIN_PLUS.annihilator == jnp.inf
    assert BOOLEAN.annihilator is False
    assert EUCLIDEAN.annihilator is None


@pytest.mark.parametrize(
    'semiring', MASKABLE, ids=[s.name for s in MASKABLE],
)
def test_semiring_path_matches_deprecated_identity_path(semiring):
    '''``semiring=`` and the deprecated ``identity=semiring.identity``
    produce identical masked ELLs for the maskable algebras.
    '''
    n, k_max = 10, 4
    ell = _random_float_ell(n, k_max, semiring.identity, seed=7)
    col_valid = np.ones(n, dtype=bool)
    col_valid[[1, 4, 9]] = False
    cv = jnp.asarray(col_valid)

    via_semiring = ell_mask(ell, cv, semiring=semiring)
    with pytest.warns(DeprecationWarning):
        via_identity = ell_mask(ell, cv, identity=semiring.identity)
    np.testing.assert_array_equal(
        np.asarray(via_semiring.values), np.asarray(via_identity.values),
    )
    assert via_semiring.identity == via_identity.identity


def test_identity_kwarg_emits_deprecation_warning():
    ell = _random_float_ell(6, 3, REAL.identity, seed=0)
    col_valid = jnp.ones(6, dtype=bool)
    with pytest.warns(DeprecationWarning, match='deprecated'):
        ell_mask(ell, col_valid, identity=0.0)


def test_semiring_path_emits_no_warning():
    ell = _random_float_ell(6, 3, REAL.identity, seed=0)
    col_valid = jnp.ones(6, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        ell_mask(ell, col_valid, semiring=REAL)  # must not warn


def test_ell_mask_both_identity_and_semiring_raises():
    ell = _random_float_ell(6, 3, REAL.identity, seed=0)
    col_valid = jnp.ones(6, dtype=bool)
    with pytest.raises(ValueError, match='not both'):
        ell_mask(ell, col_valid, identity=0.0, semiring=REAL)


def test_ell_mask_neither_identity_nor_semiring_raises():
    ell = _random_float_ell(6, 3, REAL.identity, seed=0)
    col_valid = jnp.ones(6, dtype=bool)
    with pytest.raises(ValueError, match='exactly one'):
        ell_mask(ell, col_valid)
