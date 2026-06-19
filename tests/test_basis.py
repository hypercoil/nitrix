# -*- coding: utf-8 -*-
"""Tests for ``nitrix.stats.basis`` (penalised spline bases)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.stats.basis import (
    REBasis,
    SplineBasis,
    bspline_basis,
    by_factor_smooth,
    cyclic_cubic_basis,
    re_smooth,
    spline_design,
    tensor_product_basis,
    tensor_product_design,
    thinplate_regression_basis,
    varying_coefficient_smooth,
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
    assert (
        float(np.sqrt(np.mean((fit[interior] - truth[interior]) ** 2))) < 0.05
    )


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
    assert (
        float(np.sqrt(np.mean((fit[interior] - truth[interior]) ** 2))) < 0.05
    )
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


# ---------------------------------------------------------------------------
# Cyclic cubic P-spline (bs='cp')
# ---------------------------------------------------------------------------


def test_cyclic_periodicity_and_penalty_rank():
    """The cyclic design is periodic (design(lo) == design(hi)); the circular
    difference penalty has rank n_basis - 1 (null space = constants)."""
    from nitrix.stats.basis import cyclic_cubic_basis

    cb = cyclic_cubic_basis(_x(), 12, bounds=(0.0, 1.0), center=False)
    assert cb.kind == 'cyclic'
    assert np.linalg.matrix_rank(np.asarray(cb.penalty)) == 12 - 1
    d = spline_design(cb, jnp.asarray([0.0, 1.0]))
    np.testing.assert_allclose(np.asarray(d[0]), np.asarray(d[1]), atol=1e-12)


def test_cyclic_recovers_periodic_smooth_via_gam():
    from nitrix.stats.basis import cyclic_cubic_basis
    from nitrix.stats.gam import gam_fit, smooth_partial_effect

    rng = np.random.default_rng(5)
    x = np.sort(rng.uniform(0.0, 1.0, 300))
    truth = np.sin(2 * np.pi * x)  # periodic on [0, 1]
    y = truth + rng.standard_normal(300) * 0.2
    cb = cyclic_cubic_basis(jnp.asarray(x), 12, bounds=(0.0, 1.0))
    res = gam_fit(jnp.asarray(y[None, :]), [cb])
    eff, _ = smooth_partial_effect(res, 0, cb, jnp.asarray([0.0, 1.0]))
    # fitted smooth wraps continuously
    np.testing.assert_allclose(float(eff[0, 0]), float(eff[0, 1]), atol=1e-9)
    effx, _ = smooth_partial_effect(res, 0, cb, jnp.asarray(x))
    fit = float(res.coef[0, 0]) + np.asarray(effx[0])
    interior = (x > 0.05) & (x < 0.95)
    assert (
        float(np.sqrt(np.mean((fit[interior] - truth[interior]) ** 2))) < 0.05
    )


# ---------------------------------------------------------------------------
# Tensor-product (te / ti) interaction basis
# ---------------------------------------------------------------------------


def test_tensor_product_shapes_and_kronecker_penalties():
    """The tensor design is the row-wise Kronecker product and the two
    penalties are the Kronecker-embedded marginals S1 = P1 (x) I, S2 = I (x) P2."""
    rng = np.random.default_rng(0)
    x1 = jnp.asarray(rng.uniform(0, 1, 50))
    x2 = jnp.asarray(rng.uniform(0, 1, 50))
    m1 = bspline_basis(x1, 7, center=True)
    m2 = cyclic_cubic_basis(x2, 6, bounds=(0.0, 1.0))
    k1, k2 = m1.dim, m2.dim
    te = tensor_product_basis((m1, m2))
    assert te.dim == k1 * k2
    assert te.penalties.shape == (2, k1 * k2, k1 * k2)
    assert te.pen_eig.shape == (2, k1 * k2)

    # row-wise Kronecker design
    D1, D2 = np.asarray(m1.design), np.asarray(m2.design)
    ref = (D1[:, :, None] * D2[:, None, :]).reshape(50, k1 * k2)
    np.testing.assert_allclose(np.asarray(te.design), ref, atol=1e-12)

    # S1 = P1 (x) I,  S2 = I (x) P2
    P1, P2 = np.asarray(m1.penalty), np.asarray(m2.penalty)
    np.testing.assert_allclose(
        np.asarray(te.penalties[0]), np.kron(P1, np.eye(k2)), atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(te.penalties[1]), np.kron(np.eye(k1), P2), atol=1e-12
    )


def test_tensor_penalties_commute():
    """The marginal penalties commute (Kronecker structure) -- the property that
    lets them be simultaneously diagonalised for the elementwise FS trace."""
    rng = np.random.default_rng(1)
    m1 = bspline_basis(jnp.asarray(rng.uniform(0, 1, 40)), 6, center=True)
    m2 = bspline_basis(jnp.asarray(rng.uniform(0, 1, 40)), 5, center=True)
    te = tensor_product_basis((m1, m2))
    S1 = np.asarray(te.penalties[0])
    S2 = np.asarray(te.penalties[1])
    np.testing.assert_allclose(S1 @ S2, S2 @ S1, atol=1e-10)


def test_tensor_fs_trace_eigenvalues_match_dense_pinv():
    """The natural-parameterisation identity: tr(S_lambda^+ S_k) read from the
    precomputed Kronecker eigenvalues equals the dense pseudo-inverse trace, for
    arbitrary (anisotropic) lambda -- the load-bearing te correctness check."""
    rng = np.random.default_rng(2)
    m1 = bspline_basis(jnp.asarray(rng.uniform(0, 1, 60)), 7, center=True)
    m2 = bspline_basis(jnp.asarray(rng.uniform(0, 1, 60)), 6, center=True)
    te = tensor_product_basis((m1, m2))
    S = [np.asarray(te.penalties[0]), np.asarray(te.penalties[1])]
    E = [np.asarray(te.pen_eig[0]), np.asarray(te.pen_eig[1])]
    for lam in [(1.0, 1.0), (0.3, 5.0), (12.0, 0.05), (1e-4, 1e3)]:
        s_lambda = lam[0] * S[0] + lam[1] * S[1]
        s_pinv = np.linalg.pinv(s_lambda, rcond=1e-10, hermitian=True)
        s_eig = lam[0] * E[0] + lam[1] * E[1]
        for k in range(2):
            tr_dense = np.trace(s_pinv @ S[k])
            tr_eig = np.sum(
                np.where(
                    s_eig > 0, E[k] / np.where(s_eig > 0, s_eig, 1.0), 0.0
                )
            )
            np.testing.assert_allclose(tr_eig, tr_dense, atol=1e-7)


def test_tensor_product_design_reevaluation():
    """tensor_product_design rebuilds the row-wise tensor design on a fresh
    matched grid (used to render the interaction surface)."""
    rng = np.random.default_rng(3)
    m1 = bspline_basis(jnp.asarray(rng.uniform(0, 1, 50)), 6, center=True)
    m2 = bspline_basis(jnp.asarray(rng.uniform(0, 1, 50)), 5, center=True)
    te = tensor_product_basis((m1, m2))
    g1 = jnp.asarray(np.linspace(0.2, 0.8, 11))
    g2 = jnp.asarray(np.linspace(0.3, 0.7, 11))
    D = tensor_product_design(te, (g1, g2))
    d1 = np.asarray(spline_design(m1, g1))
    d2 = np.asarray(spline_design(m2, g2))
    ref = (d1[:, :, None] * d2[:, None, :]).reshape(11, m1.dim * m2.dim)
    np.testing.assert_allclose(np.asarray(D), ref, atol=1e-12)


# ---------------------------------------------------------------------------
# Random-effect smooth (bs='re') -- the GAMM block
# ---------------------------------------------------------------------------


def test_re_smooth_intercept_is_one_hot_with_identity_penalty():
    """A random intercept block is the one-hot indicator design with an
    identity ridge penalty over the factor levels."""
    g = jnp.asarray(np.array([0, 1, 2, 0, 1, 2, 0], dtype=np.int32))
    re = re_smooth(g, n_levels=3)
    assert isinstance(re, REBasis)
    assert re.dim == 3 and re.levels == 3
    assert re.design.shape == (7, 3)
    # one-hot rows
    np.testing.assert_array_equal(
        np.asarray(re.design), np.eye(3)[np.asarray(g)]
    )
    np.testing.assert_array_equal(np.asarray(re.penalty), np.eye(3))


def test_re_smooth_slope_scales_one_hot_by_covariate():
    """A random slope block is one_hot(g) * by, column-localised per level."""
    g = jnp.asarray(np.array([0, 1, 0, 1], dtype=np.int32))
    by = jnp.asarray(np.array([2.0, -1.0, 0.5, 3.0]))
    re = re_smooth(g, by=by, n_levels=2)
    expect = np.eye(2)[np.asarray(g)] * np.asarray(by)[:, None]
    np.testing.assert_allclose(np.asarray(re.design), expect, atol=1e-12)


def test_re_smooth_infers_levels_and_is_a_pytree():
    g = jnp.asarray(np.array([0, 3, 1, 2], dtype=np.int32))
    re = re_smooth(g)
    assert re.dim == 4  # inferred max + 1
    leaves, treedef = jax.tree_util.tree_flatten(re)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, REBasis) and rebuilt.dim == 4


def test_by_factor_smooth_returns_masked_blocks_per_level():
    """``by_factor_smooth`` returns one SplineBasis per level; each design loads
    only on its level's rows and shares the marginal penalty."""
    x = _x(n=300)
    by = jnp.asarray(np.tile([0, 1, 2], 100))
    blocks = by_factor_smooth(x, by, n_basis=8)
    assert len(blocks) == 3
    marginal = bspline_basis(x, 8)
    by_np = np.asarray(by)
    for level, blk in enumerate(blocks):
        assert isinstance(blk, SplineBasis)
        d = np.asarray(blk.design)
        # rows not in this level are exactly zero; rows in this level == marginal
        assert np.all(d[by_np != level] == 0.0)
        np.testing.assert_allclose(
            d[by_np == level],
            np.asarray(marginal.design)[by_np == level],
            atol=1e-12,
        )
        # penalty is the shared marginal penalty
        np.testing.assert_allclose(
            np.asarray(blk.penalty), np.asarray(marginal.penalty), atol=1e-12
        )


def test_by_factor_smooth_respects_n_levels_for_absent_levels():
    """``n_levels`` fixes the tuple length when a level is absent from the sample."""
    x = _x(n=100)
    by = jnp.asarray(np.zeros(100, dtype=np.int32))  # only level 0 present
    blocks = by_factor_smooth(x, by, n_basis=6, n_levels=3)
    assert len(blocks) == 3
    # the two absent levels have all-zero designs
    assert np.all(np.asarray(blocks[1].design) == 0.0)
    assert np.all(np.asarray(blocks[2].design) == 0.0)


def test_varying_coefficient_smooth_scales_design_by_covariate():
    """``varying_coefficient_smooth`` is the marginal design scaled row-wise by
    ``by``, with the penalty unchanged."""
    x = _x(n=200)
    rng = np.random.default_rng(2)
    z = jnp.asarray(rng.standard_normal(200))
    vc = varying_coefficient_smooth(x, z, n_basis=8)
    marginal = bspline_basis(x, 8)
    np.testing.assert_allclose(
        np.asarray(vc.design),
        np.asarray(marginal.design) * np.asarray(z)[:, None],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(vc.penalty), np.asarray(marginal.penalty), atol=1e-12
    )
