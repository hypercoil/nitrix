# -*- coding: utf-8 -*-
"""Tests for ``nitrix.sparse.{grid, mesh}`` specialisations.

Coverage:

- **grid stencils**: 1-D / 2-D / 3-D Laplacian, identity, and
  general regular-grid stencils.  Scipy.ndimage.laplace parity at
  machine eps; periodic / reflect / replicate boundary modes.
- **mesh**: icosphere vertex / face counts at several subdivisions;
  unit-sphere invariant; 1-ring degree = 5 on the base icosahedron;
  cotangent Laplacian sends constants to zero (machine eps).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.ndimage as spnd

jax.config.update('jax_enable_x64', True)

from nitrix.semiring import REAL, semiring_ell_matmul
from nitrix.sparse import (
    ELL,
    grid_identity,
    grid_laplacian,
    icosphere,
    mesh_bary_upsample,
    mesh_cotangent_laplacian,
    mesh_k_ring_adjacency,
    mesh_pool_max,
    mesh_unpool_max,
    regular_grid_stencil,
)


def _apply_ell(op, x):
    return semiring_ell_matmul(
        op.values, op.indices, x, semiring=REAL,
        n_cols=op.n_cols, backend='jax',
    )


# ---------------------------------------------------------------------------
# grid: 1-D Laplacian
# ---------------------------------------------------------------------------


def test_grid_laplacian_1d_constant_is_zero():
    op = grid_laplacian((10,), boundary='replicate')
    x = jnp.ones((10, 1))
    np.testing.assert_allclose(_apply_ell(op, x), 0.0, atol=1e-13)


def test_grid_laplacian_1d_x_squared_gives_two_interior():
    '''d²x²/dx² = 2 exactly (interior); boundary uses replicate.'''
    op = grid_laplacian((5,), boundary='replicate')
    x = jnp.asarray([1.0, 4.0, 9.0, 16.0, 25.0])  # i^2 for i=1..5
    y = _apply_ell(op, x[:, None])[:, 0]
    # Interior: indices 1..3 see u(i-1) - 2 u(i) + u(i+1) = 2
    np.testing.assert_allclose(y[1:-1], 2.0, atol=1e-13)


def test_grid_laplacian_1d_periodic_wraps():
    op = grid_laplacian((5,), boundary='periodic')
    x = jnp.asarray([1.0, 4.0, 9.0, 16.0, 25.0])
    y = _apply_ell(op, x[:, None])[:, 0]
    # At i=0: u(4) - 2u(0) + u(1) = 25 - 2 + 4 = 27
    np.testing.assert_allclose(float(y[0]), 27.0, atol=1e-13)
    # At i=4: u(3) - 2u(4) + u(0) = 16 - 50 + 1 = -33
    np.testing.assert_allclose(float(y[-1]), -33.0, atol=1e-13)


# ---------------------------------------------------------------------------
# grid: 2-D Laplacian (scipy parity)
# ---------------------------------------------------------------------------


def test_grid_laplacian_2d_matches_scipy_replicate():
    rng = np.random.default_rng(0)
    img = rng.standard_normal((6, 6))
    op = grid_laplacian((6, 6), boundary='replicate')
    y = _apply_ell(op, jnp.asarray(img).reshape(-1, 1))[:, 0].reshape(6, 6)
    ref = spnd.laplace(img, mode='nearest')
    np.testing.assert_allclose(y, ref, atol=1e-13)


def test_grid_laplacian_2d_matches_scipy_periodic():
    rng = np.random.default_rng(0)
    img = rng.standard_normal((8, 8))
    op = grid_laplacian((8, 8), boundary='periodic')
    y = _apply_ell(op, jnp.asarray(img).reshape(-1, 1))[:, 0].reshape(8, 8)
    ref = spnd.laplace(img, mode='wrap')
    np.testing.assert_allclose(y, ref, atol=1e-13)


def test_grid_laplacian_2d_matches_scipy_reflect():
    rng = np.random.default_rng(0)
    img = rng.standard_normal((7, 7))
    op = grid_laplacian((7, 7), boundary='reflect')
    y = _apply_ell(op, jnp.asarray(img).reshape(-1, 1))[:, 0].reshape(7, 7)
    ref = spnd.laplace(img, mode='reflect')
    np.testing.assert_allclose(y, ref, atol=1e-13)


# ---------------------------------------------------------------------------
# grid: anisotropic spacing
# ---------------------------------------------------------------------------


def test_grid_laplacian_anisotropic_spacing():
    '''Per-axis spacing scales the per-axis Laplacian weight by 1/h^2.'''
    op = grid_laplacian((4, 4), spacing=(1.0, 2.0))
    # Centre weight = -2 * (1/1^2 + 1/2^2) = -2 * 1.25 = -2.5
    np.testing.assert_allclose(float(op.values[5, 0]), -2.5, atol=1e-13)


def test_grid_laplacian_3d_shape():
    op = grid_laplacian((4, 4, 4))
    # 4*4*4 = 64 voxels, 7-point stencil
    assert op.shape == (64, 64)
    assert op.k_max == 7


# ---------------------------------------------------------------------------
# grid: identity + general stencil
# ---------------------------------------------------------------------------


def test_grid_identity_acts_as_identity():
    op = grid_identity((5,))
    x = jnp.asarray([1.0, 4.0, 9.0, 16.0, 25.0])
    y = _apply_ell(op, x[:, None])[:, 0]
    np.testing.assert_allclose(y, x, atol=1e-13)


def test_regular_grid_stencil_general():
    '''Custom 2-D stencil: average of NN-4 neighbours (mean smoother).'''
    offsets = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    weights = jnp.array([0.25, 0.25, 0.25, 0.25])
    op = regular_grid_stencil(
        (4, 4), offsets, weights, boundary='replicate',
    )
    # Applied to a constant: still constant.
    x = jnp.ones((16, 1))
    np.testing.assert_allclose(_apply_ell(op, x), 1.0, atol=1e-13)


# ---------------------------------------------------------------------------
# mesh: icosphere construction
# ---------------------------------------------------------------------------


def test_icosphere_base_counts():
    m = icosphere(0)
    assert m.n_vertices == 12
    assert m.n_faces == 20


def test_icosphere_subdivision_counts():
    '''10 * 4^n + 2 vertices, 20 * 4^n faces.'''
    for n in (0, 1, 2, 3):
        m = icosphere(n)
        assert m.n_vertices == 10 * (4 ** n) + 2, (
            f'ico{n}: expected {10 * 4**n + 2} verts, got {m.n_vertices}'
        )
        assert m.n_faces == 20 * (4 ** n)


def test_icosphere_vertices_on_unit_sphere():
    m = icosphere(2)
    norms = jnp.linalg.norm(m.vertices, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# mesh: k-ring adjacency
# ---------------------------------------------------------------------------


def test_icosahedron_one_ring_degree_is_five():
    '''Every vertex of the base icosahedron has exactly 5 neighbours.'''
    m = icosphere(0)
    adj = mesh_k_ring_adjacency(m, k=1, binary=True)
    degrees = adj.values.sum(axis=-1)
    np.testing.assert_array_equal(np.asarray(degrees), np.full(12, 5))


def test_icosphere_subdivided_one_ring_max_degree():
    '''After subdivision: original vertices keep degree 5; new
    (edge-midpoint) vertices have degree 6.  So max degree is 6.
    '''
    m = icosphere(2)
    adj = mesh_k_ring_adjacency(m, k=1)
    degrees = adj.values.sum(axis=-1)
    assert int(degrees.max()) == 6
    assert int(degrees.min()) == 5


def test_mesh_k_ring_row_stochastic():
    '''binary=False rows sum to 1 (row-stochastic adjacency).'''
    m = icosphere(1)
    adj = mesh_k_ring_adjacency(m, k=1, binary=False)
    np.testing.assert_allclose(adj.values.sum(axis=-1), 1.0, atol=1e-6)


def test_mesh_adjacency_matvec_matches_dense():
    '''ELL adjacency matvec equals the dense brute-force version.'''
    m = icosphere(1)
    adj = mesh_k_ring_adjacency(m, k=1, binary=True)
    x = jax.random.normal(jax.random.key(0), (m.n_vertices, 1))
    y = _apply_ell(adj, x)

    # Build dense reference
    n = m.n_vertices
    dense = np.zeros((n, n))
    for i in range(n):
        for k_idx in range(adj.k_max):
            if float(adj.values[i, k_idx]) > 0:
                dense[i, int(adj.indices[i, k_idx])] = 1.0
    ref = dense @ np.asarray(x)
    np.testing.assert_allclose(np.asarray(y), ref, atol=1e-6)


# ---------------------------------------------------------------------------
# mesh: cotangent Laplacian
# ---------------------------------------------------------------------------


def test_mesh_cotangent_laplacian_sends_constants_to_zero():
    '''Row sums of any discrete Laplacian are zero (constants are
    in the null space).'''
    m = icosphere(2)
    L = mesh_cotangent_laplacian(m)
    ones = jnp.ones((m.n_vertices, 1))
    y = _apply_ell(L, ones)
    assert float(jnp.abs(y).max()) < 1e-5


def test_mesh_cotangent_laplacian_is_psd_smoke():
    '''Smoke check: per-row diagonal weight is positive (the assembled
    L is at least diagonally dominant in the PSD form).

    By construction the diagonal entry is in column 0.
    '''
    m = icosphere(1)
    L = mesh_cotangent_laplacian(m)
    diag_indices = L.indices[:, 0]
    diag_values = L.values[:, 0]
    self_match = jnp.arange(m.n_vertices) == diag_indices
    assert bool(jnp.all(self_match))
    assert bool(jnp.all(diag_values > 0))


def test_mesh_cotangent_differentiable():
    '''Cotangent Laplacian is constructed once at trace time; the
    matvec is differentiable through the values.  Verify that a
    trace-and-grad pipeline returns finite gradients w.r.t. the
    input signal.
    '''
    m = icosphere(1)
    L = mesh_cotangent_laplacian(m)
    def loss(x):
        y = _apply_ell(L, x[:, None])[:, 0]
        return jnp.sum(y ** 2)
    x = jax.random.normal(jax.random.key(0), (m.n_vertices,), dtype=L.values.dtype)
    g = jax.grad(loss)(x)
    assert g.shape == x.shape
    assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# Mesh convenience wrappers (mesh_pool_max / mesh_unpool_max / mesh_bary_upsample)
# ---------------------------------------------------------------------------


def test_mesh_pool_max_takes_window_max():
    '''mesh_pool_max applied with a cross-level ELL returns the
    per-coarse-vertex max of the named fine-vertex features.
    '''
    rng = np.random.default_rng(0)
    n_coarse, n_fine, win = 5, 20, 4
    # Each coarse vertex maps to win consecutive fine vertices.
    indices = jnp.asarray(np.stack([
        np.arange(win) + i * win for i in range(n_coarse)
    ]).astype(np.int32))
    values = jnp.zeros((n_coarse, win))  # ignored by mesh_pool_max
    cross_ell = ELL(
        values=values, indices=indices, n_cols=n_fine, identity=-jnp.inf,
    )

    fine_features = jnp.asarray(rng.standard_normal((n_fine, 3)))
    pooled = mesh_pool_max(cross_ell, fine_features)
    assert pooled.shape == (n_coarse, 3)
    # Verify each row equals the max of its 4 fine sources.
    for i in range(n_coarse):
        ref = jnp.max(fine_features[i * win:(i + 1) * win], axis=0)
        np.testing.assert_allclose(pooled[i], ref, atol=1e-13)


def test_mesh_bary_upsample_is_weighted_sum():
    '''mesh_bary_upsample computes the weighted sum over coarse
    sources, matching a hand-roll reference.
    '''
    rng = np.random.default_rng(0)
    n_coarse, n_fine, k = 8, 20, 3
    indices = jnp.asarray(
        rng.integers(0, n_coarse, (n_fine, k)).astype(np.int32),
    )
    # Random barycentric weights; not strictly normalised but the
    # primitive is a generic weighted-sum so unnormalised is fine.
    weights = jnp.asarray(rng.standard_normal((n_fine, k)))
    bary_ell = ELL(
        values=weights, indices=indices, n_cols=n_coarse, identity=0.0,
    )

    coarse_coords = jnp.asarray(rng.standard_normal((n_coarse, 3)))
    up = mesh_bary_upsample(bary_ell, coarse_coords)
    assert up.shape == (n_fine, 3)

    # Hand-roll
    ref = jnp.zeros((n_fine, 3))
    for i in range(n_fine):
        for p in range(k):
            j = int(indices[i, p])
            w = float(weights[i, p])
            ref = ref.at[i].add(w * coarse_coords[j])
    np.testing.assert_allclose(up, ref, atol=1e-13)


def test_mesh_unpool_max_inverts_pool_at_single_source():
    '''When each fine vertex has exactly one coarse source, the
    "max" reduction trivially gathers; unpool returns the coarse
    feature at each fine vertex.
    '''
    rng = np.random.default_rng(0)
    n_coarse, n_fine = 5, 15
    # Each fine vertex gets a single source coarse vertex.
    source = jnp.asarray(rng.integers(0, n_coarse, (n_fine,)).astype(np.int32))
    indices = source[:, None]  # (n_fine, 1)
    values = jnp.zeros((n_fine, 1))
    bary_ell = ELL(
        values=values, indices=indices,
        n_cols=n_coarse, identity=-jnp.inf,
    )

    coarse_features = jnp.asarray(rng.standard_normal((n_coarse, 4)))
    unp = mesh_unpool_max(bary_ell, coarse_features)
    assert unp.shape == (n_fine, 4)
    # Each fine vertex's output equals its source's coarse feature.
    for i in range(n_fine):
        np.testing.assert_allclose(unp[i], coarse_features[source[i]], atol=1e-13)


def test_mesh_bary_upsample_differentiable():
    '''Gradient through the weighted-sum upsampler flows back to coords.'''
    rng = np.random.default_rng(0)
    n_coarse, n_fine, k = 5, 12, 3
    indices = jnp.asarray(
        rng.integers(0, n_coarse, (n_fine, k)).astype(np.int32),
    )
    weights = jnp.asarray(rng.standard_normal((n_fine, k)))
    bary_ell = ELL(
        values=weights, indices=indices, n_cols=n_coarse, identity=0.0,
    )
    def loss(coords):
        return jnp.sum(mesh_bary_upsample(bary_ell, coords) ** 2)
    coords = jnp.asarray(rng.standard_normal((n_coarse, 3)))
    g = jax.grad(loss)(coords)
    assert g.shape == coords.shape
    assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# Icosphere hierarchy (Sprint B): icosphere_hierarchy /
# icosphere_cross_level_adjacency / icosphere_bary_upsampler
# ---------------------------------------------------------------------------


from nitrix.sparse import (  # noqa: E402
    IcosphereHierarchy,
    icosphere_bary_upsampler,
    icosphere_cross_level_adjacency,
    icosphere_hierarchy,
)
from nitrix.sparse import mesh_bary_upsample as _mesh_bary_upsample  # noqa: E402


def test_icosphere_hierarchy_length_and_counts():
    '''Hierarchy at max_level=3 has 4 meshes with the expected
    vertex counts (10 * 4^L + 2) and parent tables.
    '''
    h = icosphere_hierarchy(max_level=3)
    assert isinstance(h, IcosphereHierarchy)
    assert len(h) == 4
    assert h.max_level == 3
    expected = [12, 42, 162, 642]
    for L, m in enumerate(h.meshes):
        assert m.n_vertices == expected[L]
    # parents[0] is None; parents[L] has shape (n_verts_at_L, 2).
    assert h.parents[0] is None
    for L in (1, 2, 3):
        assert h.parents[L].shape == (expected[L], 2)


def test_hierarchy_parent_invariants():
    '''Parent invariants: coarse-original fine verts have parents
    (v, v); midpoint fine verts have parents (a, b) with a != b
    and a, b within range of the parent level.  Across the table,
    the first ``n_coarse`` rows are coarse-originals.
    '''
    h = icosphere_hierarchy(max_level=2)
    for L in (1, 2):
        n_coarse = h.meshes[L - 1].n_vertices
        n_fine = h.meshes[L].n_vertices
        parents = h.parents[L]
        # First n_coarse verts are coarse-originals
        for v in range(n_coarse):
            assert int(parents[v, 0]) == v
            assert int(parents[v, 1]) == v
        # Remaining verts are midpoints with a != b
        for v in range(n_coarse, n_fine):
            a, b = int(parents[v, 0]), int(parents[v, 1])
            assert a != b
            assert 0 <= a < n_coarse and 0 <= b < n_coarse
            assert a < b  # stored sorted


def test_cross_level_adjacency_row_contents():
    '''Row i of cross_level_adjacency(L, L+1) contains exactly the
    fine vertices descended from coarse vertex i: i itself plus
    every edge-midpoint whose parent edge is incident to i.
    '''
    h = icosphere_hierarchy(max_level=1)
    adj = icosphere_cross_level_adjacency(h, 0, 1)
    parents = np.asarray(h.parents[1])
    indices = np.asarray(adj.indices)
    for i in range(h.meshes[0].n_vertices):
        # Build the reference set
        ref = {i}  # self
        for v_fine in range(h.meshes[1].n_vertices):
            a, b = int(parents[v_fine, 0]), int(parents[v_fine, 1])
            if a == b:
                continue  # coarse-original
            if a == i or b == i:
                ref.add(v_fine)
        # Strip padding (which clones row[0])
        unique = set(indices[i].tolist())
        assert unique == ref, f'row {i}: got {unique}, expected {ref}'


def test_cross_level_adjacency_kmax():
    '''k_max(0->1) = 6 (icosahedron max degree 5 + 1 for self);
    k_max(L->L+1) for L >= 1 = 7 (subdivided max degree 6 + 1).
    '''
    h = icosphere_hierarchy(max_level=2)
    adj_01 = icosphere_cross_level_adjacency(h, 0, 1)
    adj_12 = icosphere_cross_level_adjacency(h, 1, 2)
    assert adj_01.indices.shape[1] == 6
    assert adj_12.indices.shape[1] == 7


def test_bary_upsampler_weights():
    '''Coincident-vertex rows have weights (1, 0); midpoint-vertex
    rows have weights (0.5, 0.5).  Indices in midpoint rows are
    exactly the parent pair.
    '''
    h = icosphere_hierarchy(max_level=1)
    bary = icosphere_bary_upsampler(h, 0, 1)
    values = np.asarray(bary.values)
    indices = np.asarray(bary.indices)
    parents = np.asarray(h.parents[1])
    n_coarse = h.meshes[0].n_vertices

    for v_fine in range(h.meshes[1].n_vertices):
        a, b = int(parents[v_fine, 0]), int(parents[v_fine, 1])
        if a == b:
            assert v_fine < n_coarse
            assert int(indices[v_fine, 0]) == a
            np.testing.assert_allclose(values[v_fine, 0], 1.0)
            np.testing.assert_allclose(values[v_fine, 1], 0.0)
        else:
            assert {int(indices[v_fine, 0]), int(indices[v_fine, 1])} == {a, b}
            np.testing.assert_allclose(values[v_fine, 0], 0.5)
            np.testing.assert_allclose(values[v_fine, 1], 0.5)


def test_bary_upsample_round_trip_recovers_pre_sphere_midpoints():
    '''Applying bary_upsample to the coarse vertex coordinates
    yields the (pre-sphere-projection) midpoints for the new
    fine vertices.  After re-normalising to the unit sphere,
    they equal the fine vertex coordinates exactly.
    '''
    h = icosphere_hierarchy(max_level=1)
    bary = icosphere_bary_upsampler(h, 0, 1)
    coarse_coords = h.meshes[0].vertices
    fine_coords = h.meshes[1].vertices
    up = _mesh_bary_upsample(bary, coarse_coords)
    # Re-project to unit sphere (the icosphere subdivision step also
    # does this normalisation after the midpoint average).
    up_norm = up / jnp.linalg.norm(up, axis=-1, keepdims=True)
    np.testing.assert_allclose(up_norm, fine_coords, atol=1e-6)


def test_cross_level_adjacency_rejects_non_consecutive():
    h = icosphere_hierarchy(max_level=3)
    with pytest.raises(ValueError, match='consecutive'):
        icosphere_cross_level_adjacency(h, 0, 2)


def test_bary_upsampler_rejects_non_consecutive():
    h = icosphere_hierarchy(max_level=3)
    with pytest.raises(ValueError, match='consecutive'):
        icosphere_bary_upsampler(h, 0, 2)


def test_pool_then_bary_with_hierarchy():
    '''mesh_pool_max on a cross-level adjacency takes per-coarse-vertex
    max of fine features; bary upsample lifts coarse features back to
    fine via barycentric interpolation.  End-to-end pipeline runs.
    '''
    from nitrix.sparse import mesh_pool_max
    h = icosphere_hierarchy(max_level=1)
    fine = h.meshes[1].vertices  # (42, 3)
    pool_ell = icosphere_cross_level_adjacency(h, 0, 1)
    coarse_pooled = mesh_pool_max(pool_ell, fine)
    assert coarse_pooled.shape == (12, 3)
    bary_ell = icosphere_bary_upsampler(h, 0, 1)
    fine_up = _mesh_bary_upsample(bary_ell, coarse_pooled)
    assert fine_up.shape == (42, 3)


def test_bary_upsampler_differentiable():
    '''Gradient through the icosphere bary upsampler flows back to
    coarse coordinates.
    '''
    h = icosphere_hierarchy(max_level=1)
    bary = icosphere_bary_upsampler(h, 0, 1)
    coarse = jnp.asarray(h.meshes[0].vertices)
    def loss(coords):
        return jnp.sum(_mesh_bary_upsample(bary, coords) ** 2)
    g = jax.grad(loss)(coarse)
    assert g.shape == coarse.shape
    assert bool(jnp.all(jnp.isfinite(g)))
