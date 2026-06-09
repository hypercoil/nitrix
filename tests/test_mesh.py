# -*- coding: utf-8 -*-
"""Tests for ``nitrix.sparse`` per-vertex mesh primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.sparse import (
    compute_vertex_normals,
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
