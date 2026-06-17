# -*- coding: utf-8 -*-
"""Tests for ``nitrix.stats.basis`` (penalised spline bases)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.stats.basis import (
    bspline_basis,
    spline_design,
    thinplate_regression_basis,
)


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


# ---------------------------------------------------------------------------
# Thin-plate regression spline (bs='tp')
# ---------------------------------------------------------------------------


def test_tprs_null_space_and_constraint():
    """TPRS penalty has rank k - M (the M-dim polynomial null space is
    unpenalised), and centering removes one column / sums to zero."""
    tp = thinplate_regression_basis(_x(), 15, penalty_order=2, center=False)
    assert tp.kind == 'tprs'
    # uncentered: k = 15, penalty rank = k - M = 13
    assert np.linalg.matrix_rank(np.asarray(tp.penalty)) == 15 - 2
    # penalty is PSD (positive-eigenvalue truncation)
    assert float(np.linalg.eigvalsh(np.asarray(tp.penalty)).min()) > -1e-8

    tpc = thinplate_regression_basis(_x(), 15, center=True)
    assert tpc.dim == 14
    np.testing.assert_allclose(
        np.asarray(jnp.sum(tpc.design, axis=0)), np.zeros(14), atol=1e-9
    )


def test_tprs_recovers_smooth_via_gam():
    """A TPRS GAM recovers a smooth function (the basis is fit-for-purpose
    through the same Fellner-Schall engine as P-splines)."""
    from nitrix.stats.gam import gam_fit, smooth_partial_effect

    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0.0, 1.0, 300))
    truth = np.sin(2 * np.pi * x)
    y = truth + rng.standard_normal(300) * 0.2
    tp = thinplate_regression_basis(jnp.asarray(x), 15)
    res = gam_fit(jnp.asarray(y[None, :]), [tp])
    eff, se = smooth_partial_effect(res, 0, tp, jnp.asarray(x))
    fit = float(res.coef[0, 0]) + np.asarray(eff[0])
    interior = (x > 0.05) & (x < 0.95)
    assert float(np.sqrt(np.mean((fit[interior] - truth[interior]) ** 2))) < 0.05
    assert 2.0 < float(res.edf[0, 0]) < float(tp.dim)
    assert (se > 0).all()


def test_tprs_knot_subsampling_and_reeval():
    """Large n subsamples knots to max_knots; spline_design re-evaluates."""
    rng = np.random.default_rng(4)
    x = jnp.asarray(np.sort(rng.uniform(0.0, 1.0, 500)))
    tp = thinplate_regression_basis(x, 20, max_knots=80)
    assert tp.knots.shape[0] == 80
    np.testing.assert_allclose(
        np.asarray(spline_design(tp, x)), np.asarray(tp.design), atol=1e-10
    )
    grid = jnp.linspace(0.1, 0.9, 40)
    assert spline_design(tp, grid).shape == (40, tp.dim)


def test_tprs_pytree_roundtrip():
    tp = thinplate_regression_basis(_x(), 12)
    leaves, treedef = jax.tree_util.tree_flatten(tp)
    rt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rt.kind == 'tprs' and rt.dim == tp.dim
    np.testing.assert_array_equal(np.asarray(rt.design), np.asarray(tp.design))
