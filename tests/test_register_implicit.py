# -*- coding: utf-8 -*-
"""Tests for the implicit-function differentiable registration layer.

``register_implicit`` (single-level) + ``rigid_register_implicit`` /
``affine_register_implicit`` (coarse-to-fine, each level implicit).  Two
oracles: synthetic-warp **recovery** parity with the forward recipes, and an
**IFT-exact gradient vs finite-difference** directional check (the
differentiable-layer contract -- ``jax.grad`` through ``.matrix`` flows to the
images).
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from nitrix.geometry import (  # noqa: E402
    affine_exp,
    affine_grid,
    rigid_exp,
    spatial_transform,
)
from nitrix.metrics import ncc  # noqa: E402
from nitrix.register import (  # noqa: E402
    LNCC,
    RegistrationSpec,
    Rigid,
    affine_register_implicit,
    apply_transform,
    register_implicit,
    rigid_register,
    rigid_register_implicit,
)


def _blobs_2d(n=48):
    yy, xx = np.mgrid[0:n, 0:n].astype('float64')

    def blob(cy, cx, s, a):
        return a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))

    img = (
        blob(0.30 * n, 0.38 * n, 0.11 * n, 1.0)
        + blob(0.62 * n, 0.69 * n, 0.14 * n, 0.7)
        + blob(0.75 * n, 0.28 * n, 0.09 * n, 0.6)
        + blob(0.47 * n, 0.81 * n, 0.12 * n, 0.5)
    )
    return jnp.asarray(img)


def _warp_known(fixed, matrix):
    shape = fixed.shape
    center = (jnp.asarray(shape, dtype=fixed.dtype) - 1.0) / 2.0
    grid = affine_grid(matrix, shape, center=center)
    return spatial_transform(fixed[..., None], grid, mode='constant')[..., 0]


# ---------------------------------------------------------------------------
# Recovery (single-level + multi-level)
# ---------------------------------------------------------------------------


def test_register_implicit_single_level_rigid_recovery():
    # Single-level full-res SSD recovers a small rigid misalignment.
    fixed = _blobs_2d(48)
    true = jnp.asarray([0.06, 2.0, -1.5])
    moving = _warp_known(fixed, rigid_exp(true, ndim=2))
    res = register_implicit(moving, fixed, model=Rigid(), n_iters=60)
    assert float(ncc(res.warped, fixed)) > 0.98
    assert np.isclose(float(res.params[0]), -0.06, atol=0.02)
    assert res.matrix.shape == (3, 3)


def test_register_implicit_recovery_parity_with_forward_recipe():
    # The implicit single-level solve and a single-level forward rigid_register
    # reach the same recovery (same optimum, different backward).
    fixed = _blobs_2d(48)
    true = jnp.asarray([0.05, 1.5, -1.0])
    moving = _warp_known(fixed, rigid_exp(true, ndim=2))
    imp = register_implicit(moving, fixed, model=Rigid(), n_iters=60)
    fwd = rigid_register(
        moving,
        fixed,
        spec=RegistrationSpec(levels=1, iterations=60),
        method='forward',
    )
    assert np.allclose(
        np.asarray(imp.params), np.asarray(fwd.params), atol=2e-2
    )
    assert np.allclose(
        np.asarray(imp.matrix), np.asarray(fwd.matrix), atol=2e-2
    )


def test_rigid_register_implicit_multilevel_recovery():
    # Coarse-to-fine rigid (each level implicit) recovers a larger rotation +
    # translation that the single-level full-res solve would miss.
    fixed = _blobs_2d(64)
    true = jnp.asarray([0.18, 5.0, -4.0])
    moving = _warp_known(fixed, rigid_exp(true, ndim=2))
    res = rigid_register_implicit(
        moving, fixed, spec=RegistrationSpec(levels=3, iterations=40)
    )
    assert float(ncc(res.warped, fixed)) > 0.97
    assert np.isclose(float(res.params[0]), -0.18, atol=0.03)


def test_affine_register_implicit_multilevel_recovery():
    # Coarse-to-fine affine (each level implicit) recovers a larger transform a
    # single level would miss.
    fixed = _blobs_2d(64)
    gen = np.array([[0.07, 0.04], [-0.03, -0.05]])
    true = jnp.asarray(np.concatenate([gen.reshape(-1), [3.0, -2.0]]))
    moving = _warp_known(fixed, affine_exp(true, ndim=2))
    res = affine_register_implicit(
        moving, fixed, spec=RegistrationSpec(levels=3, iterations=40)
    )
    assert float(ncc(res.warped, fixed)) > 0.97
    assert res.params.shape == (6,)


def test_register_implicit_lncc_recovery():
    # The general-metric path (implicit_minimize) recovers under LNCC.
    fixed = _blobs_2d(48)
    true = jnp.asarray([0.05, 1.5, -1.0])
    moving = _warp_known(fixed, rigid_exp(true, ndim=2))
    res = register_implicit(
        moving, fixed, model=Rigid(), metric=LNCC(radius=4), n_iters=80
    )
    assert float(ncc(res.warped, fixed)) > 0.95


# ---------------------------------------------------------------------------
# Self-contained matrix convention (owns the centring conjugation)
# ---------------------------------------------------------------------------


def test_register_implicit_matrix_is_self_contained():
    # apply_transform with the returned .matrix reproduces .warped -- i.e. the
    # centre is baked in (the same convention as rigid_register / affine_register).
    fixed = _blobs_2d(48)
    true = jnp.asarray([0.05, 1.5, -1.0])
    moving = _warp_known(fixed, rigid_exp(true, ndim=2))
    res = register_implicit(moving, fixed, model=Rigid(), n_iters=60)
    reapplied = apply_transform(moving, res, method=None)
    np.testing.assert_allclose(
        np.asarray(reapplied), np.asarray(res.warped), atol=1e-5
    )


# ---------------------------------------------------------------------------
# Differentiable layer: IFT-exact gradient vs finite difference
# ---------------------------------------------------------------------------


def test_register_implicit_grad_through_images_finite():
    fixed = _blobs_2d(40)
    true = jnp.asarray([0.04, 1.0, -0.8])
    moving = _warp_known(fixed, rigid_exp(true, ndim=2))

    def loss(mv):
        res = register_implicit(mv, fixed, model=Rigid(), n_iters=60)
        return jnp.sum(res.matrix**2)

    g = jax.grad(loss)(moving)
    assert g.shape == moving.shape
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.sum(jnp.abs(g))) > 0  # not identically zero


def test_register_implicit_ift_grad_matches_finite_difference():
    # The headline contract: the IFT gradient of a scalar of the recovered
    # transform w.r.t. the moving image agrees with a finite-difference
    # directional derivative (exact at the converged optimum).
    rng = np.random.default_rng(0)
    fixed = _blobs_2d(36)
    true = jnp.asarray([0.03, 0.8, -0.6])
    moving = _warp_known(fixed, rigid_exp(true, ndim=2))

    def loss(mv):
        res = register_implicit(mv, fixed, model=Rigid(), n_iters=80)
        # A smooth scalar of the recovered parameters.
        return jnp.sum(jnp.asarray([1.0, 0.5, -0.3]) * res.params)

    g = jax.grad(loss)(moving)
    # Random unit direction; central difference along it.
    direction = jnp.asarray(rng.normal(size=moving.shape))
    direction = direction / jnp.linalg.norm(direction)
    eps = 1e-3
    fd = float(
        (loss(moving + eps * direction) - loss(moving - eps * direction))
        / (2 * eps)
    )
    analytic = float(jnp.sum(g * direction))
    # FD is a second-order approximation; the optimum must be well converged.
    assert abs(analytic - fd) < 1e-2 * (1 + abs(fd)), (analytic, fd)


def test_register_implicit_grad_init_is_zero():
    # dtheta*/dinit = 0 (the IFT property) -> grad of a result scalar w.r.t.
    # init_params is zero (the start does not move the optimum's gradient).
    fixed = _blobs_2d(40)
    true = jnp.asarray([0.04, 1.0, -0.8])
    moving = _warp_known(fixed, rigid_exp(true, ndim=2))
    init = jnp.zeros(3)

    def loss(p0):
        res = register_implicit(
            moving, fixed, model=Rigid(), n_iters=60, init_params=p0
        )
        return jnp.sum(res.params**2)

    g = jax.grad(loss)(init)
    np.testing.assert_allclose(np.asarray(g), 0.0, atol=1e-6)
