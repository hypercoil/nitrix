# -*- coding: utf-8 -*-
"""Tests for ``nitrix.sparse`` per-vertex mesh primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.sparse import (
    compute_vertex_normals,
    edge_face_adjacency,
    face_normals,
    icosphere,
    mesh_laplacian_smooth,
)


def test_vertex_normals_unit_length():
    m = icosphere(2)
    n = compute_vertex_normals(m.vertices, m.faces)
    assert n.shape == m.vertices.shape
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(n), axis=-1), 1.0, atol=1e-6
    )


def test_vertex_normals_radial_on_sphere():
    # On a unit sphere the outward vertex normal is the radial direction.
    m = icosphere(3)
    n = np.asarray(compute_vertex_normals(m.vertices, m.faces))
    v = np.asarray(m.vertices)
    radial = v / np.linalg.norm(v, axis=-1, keepdims=True)
    align = np.abs(np.sum(n * radial, axis=-1))  # |cos| ~ 1
    assert float(align.min()) > 0.95


# ---------------------------------------------------------------------------
# face_normals (unit per-face normals -- nimox mesh-loss substrate)
# ---------------------------------------------------------------------------


def test_face_normals_unit_length():
    m = icosphere(2)
    fn = face_normals(m.vertices, m.faces)
    assert fn.shape == (m.n_faces, 3)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(fn), axis=-1), 1.0, atol=1e-6
    )


def test_face_normals_outward_radial_on_sphere():
    # icosphere faces are outward-wound, so each unit face normal aligns with
    # the (outward) face-centroid direction.
    m = icosphere(3)
    fn = np.asarray(face_normals(m.vertices, m.faces))
    v = np.asarray(m.vertices)
    f = np.asarray(m.faces)
    centroid = v[f].mean(axis=1)
    radial = centroid / np.linalg.norm(centroid, axis=-1, keepdims=True)
    cos = np.sum(fn * radial, axis=-1)
    assert float(cos.min()) > 0.95  # outward, not just aligned


def test_face_normals_matches_manual_cross():
    rng = np.random.default_rng(0)
    verts = jnp.asarray(rng.normal(size=(6, 3)))
    faces = jnp.asarray([[0, 1, 2], [3, 4, 5]])
    fn = np.asarray(face_normals(verts, faces))
    v = np.asarray(verts)
    for k, (i0, i1, i2) in enumerate([(0, 1, 2), (3, 4, 5)]):
        cr = np.cross(v[i1] - v[i0], v[i2] - v[i0])
        cr = cr / np.linalg.norm(cr)
        np.testing.assert_allclose(fn[k], cr, atol=1e-6)


def test_face_normals_zero_area_is_zero_not_nan():
    # Three collinear vertices -> degenerate face -> zero normal, no NaN.
    verts = jnp.asarray([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
    faces = jnp.asarray([[0, 1, 2]])
    fn = np.asarray(face_normals(verts, faces))
    assert np.all(np.isfinite(fn))
    np.testing.assert_allclose(fn[0], 0.0, atol=1e-9)


def test_face_normals_differentiable():
    m = icosphere(1)

    def loss(v):
        return jnp.sum(face_normals(v, m.faces) ** 2)

    g = jax.grad(loss)(m.vertices)
    assert g.shape == m.vertices.shape
    assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# edge_face_adjacency (shared-edge face pairs -- normal-consistency topology)
# ---------------------------------------------------------------------------


def _edge_face_pairs_oracle(faces: np.ndarray) -> set:
    """Brute-force {frozenset(face_a, face_b)} for edges with exactly 2 faces."""
    from collections import defaultdict

    edge_faces = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces[(min(u, v), max(u, v))].append(fi)
    return {frozenset(fs) for fs in edge_faces.values() if len(fs) == 2}


def test_edge_face_adjacency_closed_mesh_counts():
    # A closed triangle mesh: every edge bounds exactly two faces, so the
    # pair count equals the unique-edge count = 3F/2 (Euler).
    m = icosphere(1)
    pairs = np.asarray(edge_face_adjacency(m.faces))
    assert pairs.shape == (3 * m.n_faces // 2, 2)
    assert np.all(pairs[:, 0] < pairs[:, 1])  # face_a < face_b
    assert int(pairs.max()) < m.n_faces


def test_edge_face_adjacency_matches_oracle():
    m = icosphere(2)
    pairs = np.asarray(edge_face_adjacency(m.faces))
    got = {frozenset((int(a), int(b))) for a, b in pairs}
    assert got == _edge_face_pairs_oracle(np.asarray(m.faces))


def test_edge_face_adjacency_two_triangles_share_one_edge():
    # Two triangles sharing edge (1, 2): exactly one face pair (0, 1).
    faces = jnp.asarray([[0, 1, 2], [1, 2, 3]])
    pairs = np.asarray(edge_face_adjacency(faces))
    assert pairs.shape == (1, 2)
    np.testing.assert_array_equal(pairs[0], [0, 1])


def test_edge_face_adjacency_boundary_edges_excluded():
    # A single triangle has three boundary edges (one face each) -> no pair.
    faces = jnp.asarray([[0, 1, 2]])
    pairs = np.asarray(edge_face_adjacency(faces))
    assert pairs.shape == (0, 2)


def test_edge_face_adjacency_non_manifold_edge_excluded():
    # Three faces sharing edge (0, 1) is non-manifold -> that edge yields no
    # pair (count >= 3); the other edges are all boundary -> empty.
    faces = jnp.asarray([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
    pairs = np.asarray(edge_face_adjacency(faces))
    assert pairs.shape == (0, 2)


def test_laplacian_smooth_lam_zero_is_identity():
    m = icosphere(2)
    out = mesh_laplacian_smooth(m.vertices, m.faces, lam=0.0, iterations=3)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(m.vertices))


def test_laplacian_smooth_pulls_outlier_to_neighbours():
    m = icosphere(2)
    v = m.vertices
    pushed = v.at[0].add(jnp.asarray([0.5, 0.0, 0.0], dtype=v.dtype))
    smoothed = mesh_laplacian_smooth(pushed, m.faces, lam=0.5, iterations=1)
    before = float(jnp.linalg.norm(pushed[0] - v[0]))
    after = float(jnp.linalg.norm(smoothed[0] - v[0]))
    # The perturbed vertex is pulled back toward its 1-ring.
    assert after < before
    assert smoothed.shape == v.shape


def test_laplacian_smooth_differentiable():
    m = icosphere(1)
    g = jax.grad(
        lambda v: jnp.sum(mesh_laplacian_smooth(v, m.faces, iterations=2) ** 2)
    )(m.vertices)
    assert g.shape == m.vertices.shape
    assert bool(jnp.all(jnp.isfinite(g)))
