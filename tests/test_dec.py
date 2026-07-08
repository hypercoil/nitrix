# -*- coding: utf-8 -*-
"""Tests for discrete exterior calculus on triangle meshes (geometry.dec).

The operators are pinned by the two DEC identities -- the exterior derivative
composes to zero (``d_1 d_0 == 0``) and the cotangent Laplacian factorises as
``d_0^T star_1 d_0`` -- and the Helmholtz-Hodge decomposition by its defining
properties: the parts sum to the input, are star_1-orthogonal, the exact part is
curl-free and the coexact part divergence-free, and the harmonic part vanishes on
a genus-0 sphere.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.geometry import (  # noqa: E402
    hodge_decompose,
    mesh_curl,
    mesh_divergence,
    mesh_gradient,
    mesh_star_k,
)
from nitrix.sparse import (  # noqa: E402
    Mesh,
    icosphere,
    mesh_cotangent_laplacian,
)
from nitrix.sparse.ell import ell_to_dense  # noqa: E402


def _mesh64(n_iter=2):
    m = icosphere(n_iter)
    return Mesh(vertices=m.vertices.astype(jnp.float64), faces=m.faces)


def _dense(ell):
    return np.asarray(ell_to_dense(ell))


def test_operator_shapes_and_euler():
    mesh = _mesh64(2)
    v, f = mesh.n_vertices, mesh.n_faces
    d0 = _dense(mesh_gradient(mesh))
    d1 = _dense(mesh_curl(mesh))
    e = d0.shape[0]
    assert d0.shape == (e, v)
    assert d1.shape == (f, e)
    assert v - e + f == 2  # Euler characteristic of the sphere
    assert _dense(mesh_star_k(mesh, 0)).shape == (v, v)
    assert _dense(mesh_star_k(mesh, 1)).shape == (e, e)
    assert _dense(mesh_star_k(mesh, 2)).shape == (f, f)
    assert _dense(mesh_divergence(mesh)).shape == (v, e)


def test_dd_is_zero():
    # d_1 d_0 = 0 exactly -- the discrete d^2 = 0 (orientation consistency).
    mesh = _mesh64(2)
    d0 = _dense(mesh_gradient(mesh))
    d1 = _dense(mesh_curl(mesh))
    assert np.max(np.abs(d1 @ d0)) == 0.0


def test_gradient_is_edge_difference():
    mesh = _mesh64(1)
    d0 = mesh_gradient(mesh)
    dense = _dense(d0)
    # each edge row has exactly a -1 and a +1
    for row in dense:
        assert sorted(row[row != 0].tolist()) == [-1.0, 1.0]


def test_cotan_laplacian_factorisation():
    # d_0^T star_1 d_0 == mesh_cotangent_laplacian (validates d_0 + star_1).
    mesh = _mesh64(2)
    d0 = _dense(mesh_gradient(mesh))
    star1 = np.diag(_dense(mesh_star_k(mesh, 1)))
    got = d0.T @ np.diag(star1) @ d0
    ref = _dense(mesh_cotangent_laplacian(mesh))
    np.testing.assert_allclose(got, ref, atol=1e-10)


def test_star_k_invalid_degree():
    import pytest

    with pytest.raises(ValueError):
        mesh_star_k(_mesh64(1), 3)


def _random_form(mesh, seed=0):
    e = mesh_gradient(mesh).n_rows
    return jnp.asarray(np.random.default_rng(seed).standard_normal(e))


def test_hodge_parts_sum_to_input():
    mesh = _mesh64(2)
    omega = _random_form(mesh)
    hd = hodge_decompose(omega, mesh)
    total = hd.exact + hd.coexact + hd.harmonic
    np.testing.assert_allclose(
        np.asarray(total), np.asarray(omega), atol=1e-10
    )


def test_hodge_parts_are_star1_orthogonal():
    mesh = _mesh64(2)
    star1 = np.diag(_dense(mesh_star_k(mesh, 1)))
    hd = hodge_decompose(_random_form(mesh, 1), mesh)
    ex, co = np.asarray(hd.exact), np.asarray(hd.coexact)
    assert abs(float(ex @ (star1 * co))) < 1e-6


def test_hodge_exact_is_curl_free_and_coexact_div_free():
    mesh = _mesh64(2)
    d1 = _dense(mesh_curl(mesh))
    star1 = np.diag(_dense(mesh_star_k(mesh, 1)))
    d0 = _dense(mesh_gradient(mesh))
    hd = hodge_decompose(_random_form(mesh, 2), mesh)
    assert np.max(np.abs(d1 @ np.asarray(hd.exact))) < 1e-6  # curl(exact) = 0
    # div(coexact) = d_0^T star_1 coexact = 0
    assert np.max(np.abs(d0.T @ (star1 * np.asarray(hd.coexact)))) < 1e-6


def test_hodge_harmonic_vanishes_on_sphere():
    # Genus-0 => no harmonic 1-forms => the harmonic part is (numerically) zero.
    mesh = _mesh64(2)
    omega = _random_form(mesh, 3)
    hd = hodge_decompose(omega, mesh)
    rel = float(jnp.linalg.norm(hd.harmonic) / jnp.linalg.norm(omega))
    assert rel < 1e-5


def test_hodge_is_differentiable_in_the_form():
    mesh = _mesh64(1)
    omega = _random_form(mesh, 4)
    g = jax.grad(lambda w: jnp.sum(hodge_decompose(w, mesh).exact ** 2))(omega)
    assert bool(jnp.all(jnp.isfinite(g)))
