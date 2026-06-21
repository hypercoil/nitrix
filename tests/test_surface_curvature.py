# -*- coding: utf-8 -*-
"""Mesh curvature (geometry-suite P1.1 / FR 12.6).

Anchored on the sphere analytic oracle (``H = 1/r``, ``K = 1/r^2``,
``kappa_1 = kappa_2 = 1/r``), the discrete Gauss-Bonnet identity
(``sum K_i A_i = 4 pi`` for a genus-0 surface), and -- on a real cortical
surface -- correlation with FreeSurfer ``?h.curv`` (a class/correlation check,
not bit-parity: nitrix uses the *convex-positive* convention, the opposite
sign to FreeSurfer's sulci-positive ``curv``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _real_meshes import fsaverage_white

from nitrix.geometry import (
    gaussian_curvature,
    mean_curvature,
    principal_curvatures,
)
from nitrix.geometry.surface import _cotangent_apply
from nitrix.sparse import (
    Mesh,
    apply_operator,
    icosphere,
    mesh_cotangent_laplacian,
    vertex_areas,
)


def _sphere(n_sub: int = 4, radius: float = 1.0) -> Mesh:
    m = icosphere(n_sub)
    return Mesh(vertices=m.vertices * radius, faces=m.faces)


# --------------------------------------------------------------------------- #
# Sphere analytic oracle
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize('radius', [1.0, 2.0])
def test_sphere_mean_curvature_is_one_over_r(radius: float) -> None:
    h = np.asarray(mean_curvature(_sphere(4, radius)))
    assert np.allclose(h, 1.0 / radius, atol=5e-3)


@pytest.mark.parametrize('radius', [1.0, 2.0])
def test_sphere_gaussian_curvature_is_one_over_r2(radius: float) -> None:
    k = np.asarray(gaussian_curvature(_sphere(4, radius)))
    assert np.allclose(k, 1.0 / radius**2, atol=1e-2)


def test_sphere_principal_curvatures_coincide() -> None:
    pk = np.asarray(principal_curvatures(_sphere(4, 1.0)))
    assert np.allclose(pk[:, 0], 1.0, atol=1e-2)
    assert np.allclose(pk[:, 1], 1.0, atol=1e-2)
    assert np.all(pk[:, 0] >= pk[:, 1] - 1e-6)  # kappa_1 >= kappa_2


def test_gauss_bonnet_on_sphere() -> None:
    mesh = icosphere(5)
    k = gaussian_curvature(mesh)
    a = vertex_areas(mesh)
    assert np.allclose(float(jnp.sum(k * a)), 4 * np.pi, rtol=1e-3)


def test_cotangent_apply_matches_host_operator() -> None:
    # The differentiable JAX cotangent apply agrees with the host-side ELL.
    mesh = icosphere(3)
    field = jax.random.normal(jax.random.PRNGKey(3), (mesh.n_vertices, 2))
    jax_lv = _cotangent_apply(mesh.vertices, mesh.faces, field)
    host_lv = apply_operator(mesh_cotangent_laplacian(mesh), field)
    assert np.allclose(np.asarray(jax_lv), np.asarray(host_lv), atol=1e-4)


def test_curvature_differentiable() -> None:
    mesh = icosphere(2)

    def loss(v: jax.Array) -> jax.Array:
        return jnp.sum(mean_curvature(Mesh(v, mesh.faces)) ** 2)

    g = jax.grad(loss)(mesh.vertices)
    assert g.shape == mesh.vertices.shape
    assert np.all(np.isfinite(np.asarray(g)))


# --------------------------------------------------------------------------- #
# Real cortical surface
# --------------------------------------------------------------------------- #


def test_real_gauss_bonnet_genus0() -> None:
    # The strongest real-mesh global check: a genus-0 white surface integrates
    # to 4*pi regardless of mesh quality (combinatorial Gauss-Bonnet).
    v, f, _ = fsaverage_white()
    mesh = Mesh(jnp.asarray(v), jnp.asarray(f))
    total = float(jnp.sum(gaussian_curvature(mesh) * vertex_areas(mesh)))
    assert np.allclose(total, 4 * np.pi, rtol=1e-2)


def test_real_mean_curvature_anticorrelates_with_fs_curv() -> None:
    # nitrix is convex-positive; FreeSurfer ?h.curv is sulci-positive ->
    # strong *anti*-correlation.  Documents the (intentional) convention.
    v, f, overlays = fsaverage_white()
    if 'curv' not in overlays:
        pytest.skip('no curv overlay')
    mesh = Mesh(jnp.asarray(v), jnp.asarray(f))
    h = np.asarray(mean_curvature(mesh))
    assert np.all(np.isfinite(h))
    corr = float(np.corrcoef(h, overlays['curv'])[0, 1])
    assert corr < -0.8


def test_real_gaussian_curvature_has_both_signs() -> None:
    # Real cortex has elliptic (K>0) and hyperbolic/saddle (K<0) regions.
    v, f, _ = fsaverage_white()
    k = np.asarray(gaussian_curvature(Mesh(jnp.asarray(v), jnp.asarray(f))))
    assert k.min() < 0.0 < k.max()
