# -*- coding: utf-8 -*-
"""Areal & strain distortion of a surface warp (geometry-suite P1.2 / GS-6).

The surface analogue of the volumetric Jacobian: ``areal_distortion`` =
``log2(A_warped / A_source)`` per vertex, ``strain_distortion`` = per-face
principal stretches.  Anchored on closed-form warps (isometry -> 0 / (1,1);
uniform scale s -> log2(s^2) / (s,s); anisotropic scale -> the two stretches)
and exercised on a real white -> sphere map.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _real_meshes import fsaverage_surface

from nitrix.geometry import areal_distortion, strain_distortion
from nitrix.sparse import Mesh, icosphere

# A flat two-triangle unit square in the xy-plane.
_SQUARE = Mesh(
    vertices=jnp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    ),
    faces=jnp.array([[0, 1, 2], [0, 2, 3]]),
)


def _rotated(mesh: Mesh, deg: float = 30.0) -> Mesh:
    th = np.deg2rad(deg)
    r = jnp.array(
        [
            [np.cos(th), -np.sin(th), 0.0],
            [np.sin(th), np.cos(th), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return Mesh(mesh.vertices @ r.T + jnp.array([5.0, -2.0, 1.0]), mesh.faces)


def test_isometry_has_no_distortion() -> None:
    mesh = icosphere(2)
    warped = _rotated(mesh)  # rigid: rotation + translation
    # float32 rigid-transform roundoff + degenerate (lambda_1 == lambda_2)
    # stretches -> ~1e-3 tolerance, not 1e-4.
    assert np.allclose(
        np.asarray(areal_distortion(mesh, warped)), 0.0, atol=3e-3
    )
    strain = np.asarray(strain_distortion(mesh, warped))
    assert np.allclose(strain, 1.0, atol=3e-3)


def test_uniform_scale() -> None:
    mesh = icosphere(2)
    s = 2.0
    warped = Mesh(mesh.vertices * s, mesh.faces)
    # area scales by s^2 -> log2(s^2) = 2 log2(s) (areal is well-conditioned).
    assert np.allclose(
        np.asarray(areal_distortion(mesh, warped)), np.log2(s**2), atol=1e-3
    )
    # strain is degenerate here (lambda_1 == lambda_2 == s); float32 disc
    # cancellation splits them by ~5e-4, so a looser tol.
    assert np.allclose(
        np.asarray(strain_distortion(mesh, warped)), s, atol=3e-3
    )


def test_anisotropic_scale_recovers_stretches() -> None:
    warped = Mesh(_SQUARE.vertices * jnp.array([3.0, 2.0, 1.0]), _SQUARE.faces)
    strain = np.asarray(strain_distortion(_SQUARE, warped))
    assert np.allclose(strain[:, 0], 3.0, atol=1e-5)  # lambda_1
    assert np.allclose(strain[:, 1], 2.0, atol=1e-5)  # lambda_2
    # areal: 2-D area scales by 3*2 = 6.
    assert np.allclose(
        np.asarray(areal_distortion(_SQUARE, warped)), np.log2(6.0), atol=1e-4
    )


def test_strain_ordering() -> None:
    warped = Mesh(_SQUARE.vertices * jnp.array([2.0, 5.0, 1.0]), _SQUARE.faces)
    strain = np.asarray(strain_distortion(_SQUARE, warped))
    assert np.all(strain[:, 0] >= strain[:, 1])  # lambda_1 >= lambda_2
    assert np.allclose(np.sort(strain[0])[::-1], [5.0, 2.0], atol=1e-5)


def test_topology_mismatch_raises() -> None:
    with pytest.raises(ValueError, match='share topology'):
        areal_distortion(icosphere(1), icosphere(2))
    with pytest.raises(ValueError, match='share topology'):
        strain_distortion(icosphere(1), icosphere(2))


def test_distortion_differentiable() -> None:
    mesh = icosphere(2)

    def loss(v: jax.Array) -> jax.Array:
        warped = Mesh(v, mesh.faces)
        return jnp.sum(areal_distortion(mesh, warped) ** 2) + jnp.sum(
            strain_distortion(mesh, warped) ** 2
        )

    g = jax.grad(loss)(mesh.vertices * 1.3)
    assert g.shape == mesh.vertices.shape
    assert np.all(np.isfinite(np.asarray(g)))


def test_real_white_to_sphere_distortion() -> None:
    # A real surface warp (white -> sphere share fsaverage5 topology): the
    # sphere flattens the folds, so distortion is finite and far from trivial.
    vw, fw = fsaverage_surface('white')
    vs, fs = fsaverage_surface('sphere')
    white = Mesh(jnp.asarray(vw), jnp.asarray(fw))
    sphere = Mesh(jnp.asarray(vs), jnp.asarray(fs))
    areal = np.asarray(areal_distortion(white, sphere))
    strain = np.asarray(strain_distortion(white, sphere))
    assert np.all(np.isfinite(areal)) and np.all(np.isfinite(strain))
    assert areal.std() > 0.1  # the warp genuinely distorts area
    assert np.all(strain[:, 0] >= strain[:, 1] - 1e-5)
