# -*- coding: utf-8 -*-
"""Tests for ``nitrix.stats.basis`` (penalised spline bases)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.stats.basis import bspline_basis, spline_design


def _x(seed=0, n=200):
    rng = np.random.default_rng(seed)
    return jnp.asarray(np.sort(rng.uniform(0.0, 1.0, n)))


def test_bspline_partition_of_unity():
    """Uncentered uniform B-spline rows sum to 1 (partition of unity)."""
    b = bspline_basis(_x(), 10, center=False)
    assert b.design.shape == (200, 10)
    np.testing.assert_allclose(
        np.asarray(jnp.sum(b.design, axis=1)), np.ones(200), atol=1e-12
    )


def test_difference_penalty_rank():
    """An order-``m`` difference penalty on ``k`` bases has rank ``k - m``
    (its null space is the degree-``m-1`` polynomials)."""
    for k, m in [(10, 2), (12, 1), (15, 3)]:
        b = bspline_basis(_x(), k, penalty_order=m, center=False)
        assert np.linalg.matrix_rank(np.asarray(b.penalty)) == k - m


def test_sum_to_zero_constraint():
    """Centering removes one column and makes the design sum to zero, with an
    orthonormal reparameterisation that annihilates the column-sum constraint."""
    b0 = bspline_basis(_x(), 10, center=False)
    b = bspline_basis(_x(), 10, center=True)
    assert b.design.shape == (200, 9)
    # columns of the centered design sum to ~0 over the data
    np.testing.assert_allclose(
        np.asarray(jnp.sum(b.design, axis=0)), np.zeros(9), atol=1e-10
    )
    Z = b.constraint
    np.testing.assert_allclose(np.asarray(Z.T @ Z), np.eye(9), atol=1e-12)
    col_sums = jnp.sum(b0.design, axis=0)
    np.testing.assert_allclose(
        np.asarray(col_sums @ Z), np.zeros(9), atol=1e-10
    )


def test_spline_design_reevaluation():
    """``spline_design`` rebuilds the (constrained) design at new points and
    reproduces the construction design on the original covariate."""
    x = _x()
    b = bspline_basis(x, 12, center=True)
    np.testing.assert_allclose(
        np.asarray(spline_design(b, x)), np.asarray(b.design), atol=1e-12
    )
    grid = jnp.linspace(0.05, 0.95, 50)
    g = spline_design(b, grid)
    assert g.shape == (50, b.dim)


def test_penalised_fit_recovers_smooth_function():
    """A penalised least-squares spline fit recovers a smooth signal from noisy
    data (the basis + penalty are fit-for-purpose)."""
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0.0, 1.0, 300))
    truth = np.sin(2 * np.pi * x)
    y = truth + rng.standard_normal(300) * 0.1
    b = bspline_basis(jnp.asarray(x), 20, center=True)
    B = np.asarray(b.design)
    S = np.asarray(b.penalty)
    lam = 1e-3
    beta = np.linalg.solve(B.T @ B + lam * S, B.T @ (y - y.mean()))
    fit = B @ beta + y.mean()
    # interior error well below the noise level
    interior = (x > 0.05) & (x < 0.95)
    assert float(np.sqrt(np.mean((fit[interior] - truth[interior]) ** 2))) < 0.05
