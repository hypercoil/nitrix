# -*- coding: utf-8 -*-
"""Exact spatial broad-phase for point-to-triangle distance (C5a / AI-C5).

The uniform-grid + expanding-shell ``nearest_surface_distance`` must return the
**bit-exact** result of the brute-force scan (the keystone invariant -- a
pruning bug would silently corrupt SDF / thickness / resample).  Anchored on
grid-vs-brute parity (icosphere + a real fsaverage surface, near and far
queries, the brute-fallback path) and the analytic sphere oracle.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from _real_meshes import fsaverage_white

from nitrix.geometry._triangle_distance import nearest_surface_distance
from nitrix.sparse import Mesh, icosphere


def _mesh(scale: float = 10.0) -> Mesh:
    m = icosphere(4)  # 2562 vertices, ~5120 faces
    return Mesh(jnp.asarray(np.asarray(m.vertices) * scale), m.faces)


def test_grid_matches_brute_near_and_far() -> None:
    mesh = _mesh()
    rng = np.random.default_rng(0)
    v = np.asarray(mesh.vertices)
    near = v + rng.standard_normal(v.shape) * 0.5  # near-surface
    far = rng.standard_normal((400, 3)) * 40.0  # far (brute-fallback)
    q = np.concatenate([near, far], axis=0)
    db = nearest_surface_distance(q, mesh, method='brute')
    dg = nearest_surface_distance(q, mesh, method='grid', r_max=8)
    assert np.array_equal(db, dg)  # bit-exact parity


def test_auto_matches_brute() -> None:
    mesh = _mesh()
    rng = np.random.default_rng(1)
    q = np.asarray(mesh.vertices) + rng.standard_normal(
        (mesh.n_vertices, 3)
    ) * 0.3
    assert np.array_equal(
        nearest_surface_distance(q, mesh, method='auto'),
        nearest_surface_distance(q, mesh, method='brute'),
    )


def test_grid_fallback_resolves_far_queries_exactly() -> None:
    # A tiny r_max forces far queries through the brute fallback; result is
    # still exact (the fallback guarantees it).
    mesh = _mesh()
    rng = np.random.default_rng(2)
    q = rng.standard_normal((300, 3)) * 50.0  # all far
    dg = nearest_surface_distance(q, mesh, method='grid', r_max=1)
    db = nearest_surface_distance(q, mesh, method='brute')
    assert np.array_equal(dg, db)


def test_grid_on_surface_is_zero() -> None:
    mesh = _mesh()
    # Triangle centroids lie exactly on the surface -> distance 0.
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)
    centroids = v[f].mean(axis=1)
    d = nearest_surface_distance(centroids, mesh, method='grid')
    # ~0 up to the closest-point reconstruction roundoff (float64, ~1e-6).
    assert np.allclose(d, 0.0, atol=1e-5)
    # And bit-exact vs brute (the parity that actually matters).
    assert np.array_equal(d, nearest_surface_distance(centroids, mesh, method='brute'))


def test_analytic_sphere_distance() -> None:
    # Distance from a point at radius R to a unit-ish sphere mesh ~ |R - r|.
    mesh = _mesh(scale=10.0)  # radius ~10
    pts = np.array([[20.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]])
    d = nearest_surface_distance(pts, mesh, method='grid', r_max=12)
    # outside (R=20): ~10; inside (R=5): ~5; centre (R=0): ~10 (to the shell)
    db = nearest_surface_distance(pts, mesh, method='brute')
    assert np.array_equal(d, db)
    assert abs(d[0] - 10.0) < 0.2 and abs(d[1] - 5.0) < 0.2


def test_empty_query() -> None:
    mesh = _mesh()
    assert nearest_surface_distance(
        np.zeros((0, 3)), mesh, method='grid'
    ).shape == (0,)


def test_real_fsaverage_grid_matches_brute() -> None:
    # The real-data keystone: on an irregular cortical surface the grid path is
    # bit-exact vs brute for both near-surface and interior queries.
    v, f, _ = fsaverage_white()
    mesh = Mesh(jnp.asarray(v), jnp.asarray(f))
    rng = np.random.default_rng(3)
    vv = np.asarray(v)
    near = vv + rng.standard_normal(vv.shape).astype(np.float32) * 1.0
    far = (vv.mean(0) + rng.standard_normal((300, 3)) * 80.0).astype(np.float32)
    q = np.concatenate([near[:2000], far], axis=0)
    db = nearest_surface_distance(q, mesh, method='brute')
    dg = nearest_surface_distance(q, mesh, method='grid', r_max=8)
    assert np.array_equal(db, dg)
