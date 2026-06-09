# -*- coding: utf-8 -*-
"""Tests for ``nitrix.register`` displacement-field regularisers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.geometry import identity_grid
from nitrix.register import (
    bending_energy,
    gradient_smoothness,
    jacobian_folding_penalty,
)


def _zero_field(shape):
    return jnp.zeros((*shape, len(shape)))


def test_gradient_smoothness_zero_for_constant_field():
    u = _zero_field((8, 8))
    assert float(gradient_smoothness(u)) == 0.0
    # A spatially-constant (non-zero) displacement also has zero gradient.
    u2 = jnp.ones((8, 8, 2)) * 3.0
    np.testing.assert_allclose(float(gradient_smoothness(u2)), 0.0, atol=1e-10)


def test_gradient_smoothness_linear_field_interior_value():
    grid = identity_grid((10, 10))  # (10,10,2), grid[i,j] = (i, j)
    slope = 0.05
    u = slope * grid  # du_c/dx_c = slope, cross-derivatives 0
    per_voxel = gradient_smoothness(u, reduction='none')
    # Interior voxel: ||grad u||^2 = ndim * slope^2.
    np.testing.assert_allclose(
        float(per_voxel[5, 5]), 2 * slope**2, atol=1e-8
    )
    assert float(gradient_smoothness(u)) > 0.0


def test_bending_energy_zero_for_linear_field_interior():
    # A linear (affine) flow has zero curvature; the discrete second
    # difference is exactly 0 in the interior (the replicate boundary makes
    # only the edge voxels nonzero -- inherent to finite-difference bending).
    grid = identity_grid((12, 12))
    u = 0.1 * grid
    be = bending_energy(u, reduction='none')
    np.testing.assert_allclose(np.asarray(be)[2:-2, 2:-2], 0.0, atol=1e-8)


def test_bending_energy_positive_for_quadratic_field():
    grid = identity_grid((12, 12)).astype(jnp.float64)
    x = grid[..., 0]
    u0 = 0.1 * x**2  # curvature in component 0 along axis 0
    u = jnp.stack([u0, jnp.zeros_like(u0)], axis=-1)
    be = bending_energy(u, reduction='none')
    lin = bending_energy(0.1 * grid, reduction='none')
    # Interior curvature is clearly nonzero, unlike the linear field.
    assert float(be[5, 5]) > 1e-3
    assert float(be[5, 5]) > float(lin[5, 5]) + 1e-3


def test_jacobian_folding_penalty_zero_for_identity():
    u = _zero_field((8, 8, 8))  # det J = 1 everywhere
    assert float(jacobian_folding_penalty(u)) == 0.0


def test_jacobian_folding_penalty_detects_fold():
    grid = identity_grid((10, 10))
    # u_x = -1.5 * x  ->  J_xx = 1 - 1.5 = -0.5, det J = -0.5 < 0 (fold).
    u0 = -1.5 * grid[..., 0]
    u = jnp.stack([u0, jnp.zeros_like(u0)], axis=-1)
    per_voxel = jacobian_folding_penalty(u, reduction='none')
    assert float(per_voxel[5, 5]) > 0.0  # interior is folded
    assert float(jacobian_folding_penalty(u)) > 0.0


def test_regularisers_reductions_and_shapes():
    u = jnp.asarray(
        np.random.default_rng(0).standard_normal((6, 7, 8, 3))
    )
    for fn in (gradient_smoothness, bending_energy, jacobian_folding_penalty):
        none = fn(u, reduction='none')
        assert none.shape == (6, 7, 8)
        np.testing.assert_allclose(
            float(fn(u, reduction='mean')), float(none.mean()), atol=1e-8
        )
        np.testing.assert_allclose(
            float(fn(u, reduction='sum')), float(none.sum()), atol=1e-6
        )


def test_regularisers_differentiable():
    u = jnp.asarray(np.random.default_rng(1).standard_normal((8, 8, 2)))
    for fn in (gradient_smoothness, bending_energy, jacobian_folding_penalty):
        g = jax.grad(lambda f: fn(f))(u)
        assert g.shape == u.shape
        assert bool(jnp.all(jnp.isfinite(g)))
