# -*- coding: utf-8 -*-
"""Tests for R3 — the differentiable-layer support.

- **linalg.implicit_least_squares**: the implicit-function-theorem
  gradient matching the analytic (linear) pseudo-inverse, finite
  differences (nonlinear), and the unrolled-LM gradient at convergence.
- **linalg.implicit_minimize**: the general scalar-objective IFT gradient
  matching the pseudo-inverse / finite differences, and making a non-SSD
  (LNCC) registration differentiable w.r.t. the moving image.
- **Recipe differentiability**: ``jax.grad`` flows through the SSD rigid
  recipe and the diffeomorphic Demons recipe w.r.t. the input image (the
  unrolled differentiable-layer path entense uses), yielding finite,
  non-trivial gradients.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nitrix.geometry import (  # noqa: E402
    affine_grid,
    rigid_exp,
    spatial_transform,
)
from nitrix.linalg import (  # noqa: E402
    implicit_least_squares,
    implicit_minimize,
    levenberg_marquardt,
)
from nitrix.register import (  # noqa: E402
    LNCC,
    DemonsSpec,
    RegistrationSpec,
    diffeomorphic_demons_register,
    rigid_register,
)


def _blobs_2d(n=16):
    yy, xx = np.mgrid[0:n, 0:n].astype('float64')

    def blob(cy, cx, s, a):
        return a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))

    return jnp.asarray(
        blob(0.35 * n, 0.4 * n, 0.2 * n, 1.0)
        + blob(0.65 * n, 0.6 * n, 0.22 * n, 0.7)
    )


# ---------------------------------------------------------------------------
# implicit_least_squares
# ---------------------------------------------------------------------------


def test_implicit_lstsq_linear_matches_pseudoinverse():
    rng = np.random.RandomState(0)
    a = jnp.asarray(rng.standard_normal((12, 3)))

    def loss(b):
        return implicit_least_squares(
            lambda d, x: a @ x - d, b, jnp.zeros(3), n_iters=25, cg_tol=1e-12
        ).sum()

    b = jnp.asarray(rng.standard_normal(12))
    g = np.asarray(jax.grad(loss)(b))
    analytic = np.asarray(np.linalg.pinv(np.asarray(a))).sum(axis=0)
    assert np.allclose(g, analytic, atol=1e-6)


def test_implicit_lstsq_nonlinear_fd_and_matches_unrolled():
    t = jnp.asarray(np.linspace(0.0, 2.0, 40))

    def resid(c, k):
        return jnp.exp(-k[0] * t) - c

    data = jnp.asarray(np.exp(-1.7 * np.asarray(t)))

    def imp(c):
        return implicit_least_squares(
            resid, c, jnp.asarray([0.5]), n_iters=40, cg_tol=1e-12
        )[0]

    g = np.asarray(jax.grad(imp)(data))
    rng = np.random.RandomState(1)
    d = rng.standard_normal(40)
    d /= np.linalg.norm(d)
    fd = (
        float(imp(data + 1e-6 * jnp.asarray(d)))
        - float(imp(data - 1e-6 * jnp.asarray(d)))
    ) / 2e-6
    assert np.isclose(float(g @ d), fd, rtol=1e-4, atol=1e-6)

    def unrolled(c):
        return levenberg_marquardt(
            lambda k: resid(c, k), jnp.asarray([0.5]), n_iters=60, cg_tol=1e-12
        ).params[0]

    assert np.allclose(g, np.asarray(jax.grad(unrolled)(data)), atol=1e-4)


def test_implicit_lstsq_x0_independent():
    a = jnp.asarray(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    b = jnp.asarray(np.array([1.0, 2.0, 3.0]))
    x1 = implicit_least_squares(
        lambda d, x: a @ x - d, b, jnp.zeros(2), n_iters=20
    )
    x2 = implicit_least_squares(
        lambda d, x: a @ x - d, b, 5.0 + jnp.zeros(2), n_iters=20
    )
    assert np.allclose(np.asarray(x1), np.asarray(x2), atol=1e-8)


# ---------------------------------------------------------------------------
# implicit_minimize (general scalar objective: the non-SSD differentiable
# layer)
# ---------------------------------------------------------------------------


def test_implicit_minimize_lstsq_matches_pseudoinverse():
    # On a least-squares objective the exact-Hessian backward must agree
    # with the analytic pseudo-inverse (and hence implicit_least_squares).
    rng = np.random.RandomState(0)
    a = jnp.asarray(rng.standard_normal((12, 3)))

    def loss(b):
        return implicit_minimize(
            lambda d, x: 0.5 * jnp.sum((a @ x - d) ** 2),
            b,
            jnp.zeros(3),
            maxiter=200,
            cg_tol=1e-12,
        ).sum()

    b = jnp.asarray(rng.standard_normal(12))
    g = np.asarray(jax.grad(loss)(b))
    analytic = np.asarray(np.linalg.pinv(np.asarray(a))).sum(axis=0)
    assert np.allclose(g, analytic, atol=1e-6)


def test_implicit_minimize_nonlinear_fd():
    t = jnp.asarray(np.linspace(0.0, 2.0, 40))

    def obj(c, k):
        return jnp.sum((jnp.exp(-k[0] * t) - c) ** 2)

    data = jnp.asarray(np.exp(-1.7 * np.asarray(t)))

    def imp(c):
        return implicit_minimize(
            obj, c, jnp.asarray([0.5]), maxiter=200, cg_tol=1e-12
        )[0]

    g = np.asarray(jax.grad(imp)(data))
    rng = np.random.RandomState(1)
    d = rng.standard_normal(40)
    d /= np.linalg.norm(d)
    fd = (
        float(imp(data + 1e-6 * jnp.asarray(d)))
        - float(imp(data - 1e-6 * jnp.asarray(d)))
    ) / 2e-6
    assert np.isclose(float(g @ d), fd, rtol=1e-4, atol=1e-6)


def test_lncc_registration_differentiable_via_implicit_minimize():
    # The headline capability: a non-SSD (LNCC) registration is a
    # differentiable layer through implicit_minimize -- jax.grad flows from
    # a scalar of the recovered warp back to the moving image.
    moving, fixed = _make_pair()
    center = (jnp.asarray(fixed.shape, dtype=fixed.dtype) - 1.0) / 2.0
    metric = LNCC(radius=3)

    def _warp(m, theta):
        grid = affine_grid(rigid_exp(theta, ndim=2), m.shape, center=center)
        return spatial_transform(m[..., None], grid, mode='nearest')[..., 0]

    def objective(m, theta):
        return metric.cost(_warp(m, theta), fixed)

    def loss(m):
        theta = implicit_minimize(
            objective, m, jnp.zeros(3), maxiter=60, ridge=1e-6
        )
        return jnp.sum(_warp(m, theta) ** 2)

    g = np.asarray(jax.grad(loss)(moving))
    assert np.all(np.isfinite(g))
    assert np.abs(g).sum() > 0.0


# ---------------------------------------------------------------------------
# recipe differentiability (unrolled differentiable layer)
# ---------------------------------------------------------------------------


def _make_pair(n=16, transform=(0.06, 1.0, -0.8)):
    fixed = _blobs_2d(n)
    center = (jnp.asarray(fixed.shape, dtype=fixed.dtype) - 1.0) / 2.0
    grid = affine_grid(
        rigid_exp(jnp.asarray(transform), ndim=2), fixed.shape, center=center
    )
    moving = spatial_transform(fixed[..., None], grid, mode='nearest')[..., 0]
    return moving, fixed


def test_rigid_recipe_early_exit_not_diff_fixed_is():
    # B2: mode='early_exit' (the single-pair IC while_loop) is NOT reverse-
    # differentiable -- jax.grad through it raises a loud, actionable error;
    # mode='fixed' (the default) is the reverse-differentiable fixed scan.
    moving, fixed = _make_pair()

    def loss_early(m):
        res = rigid_register(
            m,
            fixed,
            spec=RegistrationSpec(levels=1, iterations=5, mode='early_exit'),
        )
        return jnp.sum(res.warped**2)

    with pytest.raises(RuntimeError, match='implicit_least_squares'):
        jax.grad(loss_early)(moving)

    def loss_default(m):
        res = rigid_register(
            m, fixed, spec=RegistrationSpec(levels=1, iterations=5)
        )
        return jnp.sum(res.warped**2)

    g = np.asarray(jax.grad(loss_default)(moving))
    assert np.all(np.isfinite(g))
    assert np.abs(g).sum() > 0.0


def test_demons_recipe_differentiable_wrt_image():
    moving, fixed = _make_pair()

    def loss(m):
        res = diffeomorphic_demons_register(
            m, fixed, spec=DemonsSpec(levels=1, iterations=3, n_steps=4)
        )
        return jnp.sum(res.warped**2)

    g = np.asarray(jax.grad(loss)(moving))
    assert np.all(np.isfinite(g))
    assert np.abs(g).sum() > 0.0
