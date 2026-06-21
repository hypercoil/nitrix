# -*- coding: utf-8 -*-
"""Geodesic surface smoothing (geometry-suite P1.2 / GS-12).

Backward-Euler heat diffusion ``(M + tL) x = M x_0`` via ``linalg.krylov.cg``.
Anchored on the exact invariants (area-weighted mass conservation, constant
preservation, ``fwhm=0`` identity), monotone variance reduction with FWHM, and
a real fsaverage smoke test.  This is the nitrix-native geodesic smoother
(documented divergence from ``wb_command -metric-smoothing``, not parity).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from _real_meshes import fsaverage_white

from nitrix.geometry import surface_smooth
from nitrix.sparse import Mesh, icosphere, vertex_areas


def _noise(n: int, seed: int = 0) -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(seed), (n,))


def test_mass_is_conserved() -> None:
    # Heat diffusion conserves the area-weighted integral sum_i A_i x_i.
    mesh = icosphere(3)
    area = vertex_areas(mesh)
    x0 = _noise(mesh.n_vertices)
    x = surface_smooth(mesh, x0, fwhm=0.3)
    assert np.allclose(
        float(jnp.sum(area * x)), float(jnp.sum(area * x0)), rtol=1e-4
    )


def test_constant_is_preserved() -> None:
    mesh = icosphere(3)
    const = jnp.full((mesh.n_vertices,), 2.5)
    out = surface_smooth(mesh, const, fwhm=0.5)
    assert np.allclose(np.asarray(out), 2.5, atol=1e-4)


def test_zero_fwhm_is_identity() -> None:
    mesh = icosphere(3)
    x0 = _noise(mesh.n_vertices, seed=1)
    out = surface_smooth(mesh, x0, fwhm=0.0)
    assert np.allclose(np.asarray(out), np.asarray(x0), atol=1e-5)


def test_larger_fwhm_smooths_more() -> None:
    # Monotone: a larger kernel removes more high-frequency variance.
    mesh = icosphere(4)
    x0 = _noise(mesh.n_vertices, seed=2)
    v0 = float(jnp.var(x0))
    variances = [
        float(jnp.var(surface_smooth(mesh, x0, fwhm=fw)))
        for fw in (0.05, 0.1, 0.2, 0.4)
    ]
    assert variances[0] < v0  # any smoothing reduces variance
    assert all(b <= a + 1e-7 for a, b in zip(variances, variances[1:]))


def test_batched_matches_per_field() -> None:
    # Smoothing a stack of fields == smoothing each separately.
    mesh = icosphere(3)
    fields = jnp.stack([_noise(mesh.n_vertices, s) for s in (3, 4)])  # (2, n)
    both = surface_smooth(mesh, fields, fwhm=0.2)
    sep = jnp.stack(
        [surface_smooth(mesh, fields[i], fwhm=0.2) for i in (0, 1)]
    )
    assert np.allclose(np.asarray(both), np.asarray(sep), atol=1e-4)


def test_smoothing_differentiable() -> None:
    mesh = icosphere(2)
    x0 = _noise(mesh.n_vertices, seed=5)

    def loss(v: jax.Array) -> jax.Array:
        return jnp.sum(surface_smooth(mesh, v, fwhm=0.3) ** 2)

    g = jax.grad(loss)(x0)
    assert g.shape == x0.shape
    assert np.all(np.isfinite(np.asarray(g)))


def test_real_mesh_smoothing() -> None:
    # Smooth FS curv on the real white surface: finite, mass-conserving,
    # variance-reducing (mm-scale fp32, folded geometry).
    v, f, overlays = fsaverage_white()
    if 'curv' not in overlays:
        import pytest

        pytest.skip('no curv overlay')
    mesh = Mesh(jnp.asarray(v), jnp.asarray(f))
    area = vertex_areas(mesh)
    curv = jnp.asarray(overlays['curv'])
    out = surface_smooth(mesh, curv, fwhm=5.0)  # 5 mm FWHM
    assert np.all(np.isfinite(np.asarray(out)))
    assert float(jnp.var(out)) < float(jnp.var(curv))
    assert np.allclose(
        float(jnp.sum(area * out)), float(jnp.sum(area * curv)), rtol=1e-3
    )
