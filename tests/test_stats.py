# -*- coding: utf-8 -*-
"""Tests for ``nitrix.stats`` -- covariance, correlation, Fourier.

Coverage:

- **covariance**: match ``numpy.cov`` to machine eps on real and
  complex inputs; check Hermitian property of complex cov; ridge,
  weights (vector and matrix), and conditional cov paths.  The
  key concern -- "silently wrong on complex inputs" -- is
  explicitly tested via Hermiticity and ``np.cov`` parity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.stats import (
    ccorr,
    ccov,
    conditionalcorr,
    conditionalcov,
    corr,
    corrcoef,
    cov,
    gaussian_nll,
    kl_diagonal_gaussian,
    pairedcov,
    partialcorr,
    pca_fit,
    pca_inverse_transform,
    pca_transform,
    pcorr,
    precision,
)

# ---------------------------------------------------------------------------
# covariance: real-valued
# ---------------------------------------------------------------------------


def test_cov_matches_numpy_real():
    rng = np.random.default_rng(0)
    X_np = rng.standard_normal((5, 200))
    X = jnp.asarray(X_np)
    S = cov(X)
    S_ref = np.cov(X_np, bias=False)
    np.testing.assert_allclose(np.asarray(S), S_ref, atol=1e-13)


def test_cov_bias_matches_numpy():
    rng = np.random.default_rng(0)
    X_np = rng.standard_normal((5, 200))
    X = jnp.asarray(X_np)
    S = cov(X, bias=True)
    S_ref = np.cov(X_np, bias=True)
    np.testing.assert_allclose(np.asarray(S), S_ref, atol=1e-13)


def test_cov_ddof_overrides_bias():
    X = jnp.asarray(np.random.default_rng(0).standard_normal((4, 100)))
    np.testing.assert_allclose(
        cov(X, bias=True),
        cov(X, ddof=0),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        cov(X, bias=False),
        cov(X, ddof=1),
        atol=1e-14,
    )


def test_cov_correlation_diagonal_is_one():
    X = jnp.asarray(np.random.default_rng(0).standard_normal((5, 100)))
    R = corr(X)
    np.testing.assert_allclose(jnp.diagonal(R), 1.0, atol=1e-12)


def test_cov_ridge_adds_to_diagonal():
    X = jnp.asarray(np.random.default_rng(0).standard_normal((5, 100)))
    S = cov(X)
    S_ridge = cov(X, l2=1.0)
    np.testing.assert_allclose(S_ridge - S, jnp.eye(5), atol=1e-12)


def test_cov_vector_weights_uniform_matches_unweighted():
    X = jnp.asarray(np.random.default_rng(0).standard_normal((4, 100)))
    w = jnp.ones(100)
    np.testing.assert_allclose(cov(X), cov(X, weights=w), atol=1e-14)


def test_cov_non_uniform_weights_match_numpy_aweights():
    rng = np.random.default_rng(0)
    X_np = rng.standard_normal((4, 100))
    X = jnp.asarray(X_np)
    w_np = np.abs(rng.standard_normal(100)) + 0.5
    w = jnp.asarray(w_np)
    S = cov(X, weights=w)
    S_ref = np.cov(X_np, aweights=w_np, bias=False)
    np.testing.assert_allclose(np.asarray(S), S_ref, atol=1e-13)


def test_cov_pass_both_weight_args_raises():
    X = jnp.zeros((3, 10))
    w = jnp.ones(10)
    W = jnp.eye(10)
    with pytest.raises(ValueError, match='at most one of'):
        cov(X, weights=w, weight_matrix=W)


def test_cov_diagonal_weight_matrix_matches_vector_weights():
    """The full-``weight_matrix`` path must reduce to the (numpy-validated)
    vector-weight path when the matrix is ``diag(w)``.  This exercises the
    matrix branch of ``_cov_core`` and is the regression for SPEC §8's
    non-diagonal-weight bug (legacy code silently bypassed the matrix path;
    the new code computes it).
    """
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.standard_normal((4, 60)))
    w = jnp.asarray(np.abs(rng.standard_normal(60)) + 0.5)
    np.testing.assert_allclose(
        np.asarray(cov(X, weight_matrix=jnp.diag(w))),
        np.asarray(cov(X, weights=w)),
        atol=1e-12,
    )


def test_cov_nondiagonal_weight_matrix_is_symmetric_and_finite():
    """A genuinely non-diagonal symmetric coupling matrix produces a finite,
    symmetric covariance (the path the legacy implementation could not take).
    """
    rng = np.random.default_rng(1)
    X = jnp.asarray(rng.standard_normal((5, 40)))
    A = rng.standard_normal((40, 40))
    W = jnp.asarray(A + A.T)  # symmetric, non-diagonal
    S = cov(X, weight_matrix=W)
    assert bool(jnp.all(jnp.isfinite(S)))
    np.testing.assert_allclose(np.asarray(S), np.asarray(S).T, atol=1e-10)


# ---------------------------------------------------------------------------
# covariance: complex-valued (the silently-wrong concern)
# ---------------------------------------------------------------------------


def test_cov_complex_matches_numpy():
    """Complex cov must match np.cov to machine eps; this was the
    "silently wrong" failure mode in the legacy code.
    """
    rng = np.random.default_rng(0)
    n_chan, n_obs = 5, 100
    X_np = rng.standard_normal((n_chan, n_obs)) + 1j * rng.standard_normal(
        (n_chan, n_obs)
    )
    X = jnp.asarray(X_np)
    S = cov(X)
    S_ref = np.cov(X_np, bias=False)
    np.testing.assert_allclose(np.asarray(S), S_ref, atol=1e-13)


def test_cov_complex_is_hermitian():
    rng = np.random.default_rng(0)
    X_np = rng.standard_normal((5, 100)) + 1j * rng.standard_normal((5, 100))
    X = jnp.asarray(X_np)
    S = cov(X)
    np.testing.assert_allclose(S, S.conj().T, atol=1e-13)


def test_cov_complex_diagonal_is_real_positive():
    rng = np.random.default_rng(0)
    X = jnp.asarray(
        rng.standard_normal((5, 100)) + 1j * rng.standard_normal((5, 100))
    )
    d = jnp.diagonal(cov(X))
    np.testing.assert_allclose(d.imag, 0.0, atol=1e-15)
    assert bool(jnp.all(d.real > 0))


def test_corr_complex_diagonal_is_one():
    rng = np.random.default_rng(0)
    X = jnp.asarray(
        rng.standard_normal((5, 100)) + 1j * rng.standard_normal((5, 100))
    )
    R = corr(X)
    np.testing.assert_allclose(jnp.diagonal(R), 1.0, atol=1e-12)


def test_pairedcov_complex_matches_numpy_cross_block():
    rng = np.random.default_rng(0)
    n_obs = 100
    X_np = rng.standard_normal((5, n_obs)) + 1j * rng.standard_normal(
        (5, n_obs)
    )
    Y_np = rng.standard_normal((3, n_obs)) + 1j * rng.standard_normal(
        (3, n_obs)
    )
    P = pairedcov(jnp.asarray(X_np), jnp.asarray(Y_np))
    Z_np = np.concatenate([X_np, Y_np], axis=0)
    S_full = np.cov(Z_np, bias=False)
    P_ref = S_full[:5, 5:]
    np.testing.assert_allclose(np.asarray(P), P_ref, atol=1e-13)


# ---------------------------------------------------------------------------
# precision / partial / conditional
# ---------------------------------------------------------------------------


def test_precision_inverse_of_cov():
    X = jnp.asarray(np.random.default_rng(0).standard_normal((4, 200)))
    S = cov(X)
    P = precision(X)
    np.testing.assert_allclose(S @ P, jnp.eye(4), atol=1e-9)


def test_partialcorr_diagonal_is_one():
    X = jnp.asarray(np.random.default_rng(0).standard_normal((4, 200)))
    R = partialcorr(X)
    np.testing.assert_allclose(jnp.diagonal(R), 1.0, atol=1e-10)


def test_conditionalcov_residual_orthogonal_to_y():
    """After residualisation, the un-centered inner product
    ``X_residual @ Y.T`` is ~0 (orthogonality in the OLS sense,
    which does NOT include centering -- residualise has no
    implicit intercept term).
    """
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.standard_normal((5, 200)))
    Y = jnp.asarray(rng.standard_normal((2, 200)))
    from nitrix.linalg import residualise

    X_res = residualise(X, Y, rowvar=True)
    # X_res @ Y.T should be ~0 (orthogonal to Y's row span).
    gram = X_res @ Y.T
    assert float(jnp.abs(gram).max()) < 1e-10


def test_conditionalcov_shape_and_symmetry():
    """Conditional cov returns symmetric ``(c, c)`` for real input."""
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.standard_normal((5, 200)))
    Y = jnp.asarray(rng.standard_normal((2, 200)))
    S = conditionalcov(X, Y)
    assert S.shape == (5, 5)
    np.testing.assert_allclose(S, S.T, atol=1e-12)


def test_aliases_match_canonical():
    X = jnp.asarray(np.random.default_rng(0).standard_normal((4, 100)))
    Y = jnp.asarray(np.random.default_rng(1).standard_normal((2, 100)))
    np.testing.assert_allclose(corrcoef(X), corr(X), atol=1e-15)
    np.testing.assert_allclose(pcorr(X), partialcorr(X), atol=1e-15)
    np.testing.assert_allclose(ccov(X, Y), conditionalcov(X, Y), atol=1e-15)
    np.testing.assert_allclose(ccorr(X, Y), conditionalcorr(X, Y), atol=1e-15)


# ---------------------------------------------------------------------------
# gaussian: KL divergence + negative log-likelihood
# ---------------------------------------------------------------------------


def test_kl_diagonal_gaussian_zero_at_standard_normal():
    mean = jnp.zeros((4, 8))
    log_var = jnp.zeros((4, 8))
    kl = kl_diagonal_gaussian(mean, log_var, axis=-1, reduction='sum')
    np.testing.assert_allclose(np.asarray(kl), 0.0, atol=1e-12)


def test_kl_diagonal_gaussian_matches_closed_form():
    rng = np.random.default_rng(0)
    mean = rng.standard_normal((3, 5))
    log_var = rng.standard_normal((3, 5)) * 0.5
    ref = 0.5 * (mean**2 + np.exp(log_var) - 1.0 - log_var)
    out = kl_diagonal_gaussian(
        jnp.asarray(mean), jnp.asarray(log_var), reduction='none'
    )
    np.testing.assert_allclose(np.asarray(out), ref, atol=1e-10)


def test_kl_diagonal_gaussian_nonnegative():
    rng = np.random.default_rng(1)
    mean = jnp.asarray(rng.standard_normal((10, 16)))
    log_var = jnp.asarray(rng.standard_normal((10, 16)))
    kl = kl_diagonal_gaussian(mean, log_var, axis=-1, reduction='sum')
    assert float(kl.min()) >= -1e-9


def test_kl_sum_axis_shape():
    mean = jnp.zeros((6, 12))
    log_var = jnp.zeros((6, 12))
    per_sample = kl_diagonal_gaussian(mean, log_var, axis=-1, reduction='sum')
    assert per_sample.shape == (6,)


def test_gaussian_nll_matches_jax_norm_logpdf():
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.standard_normal((4, 7)))
    mean = jnp.asarray(rng.standard_normal((4, 7)))
    log_var = jnp.asarray(rng.standard_normal((4, 7)) * 0.3)
    sigma = jnp.exp(0.5 * log_var)
    ref = -jax.scipy.stats.norm.logpdf(x, loc=mean, scale=sigma)
    out = gaussian_nll(x, mean, log_var, reduction='none')
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), atol=1e-10)


def test_gaussian_nll_differentiable():
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.standard_normal((5, 9)))
    mean = jnp.asarray(rng.standard_normal((5, 9)))
    log_var = jnp.asarray(rng.standard_normal((5, 9)) * 0.2)
    g_mean, g_lv = jax.grad(
        lambda m, lv: gaussian_nll(x, m, lv), argnums=(0, 1)
    )(mean, log_var)
    assert bool(jnp.all(jnp.isfinite(g_mean)))
    assert bool(jnp.all(jnp.isfinite(g_lv)))


# ---------------------------------------------------------------------------
# pca
# ---------------------------------------------------------------------------


def test_pca_full_reconstruction_is_exact():
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.standard_normal((50, 8)))
    res = pca_fit(X)  # keep all min(n, d) = 8 components
    z = pca_transform(X, res.components, res.mean)
    x_rec = pca_inverse_transform(z, res.components, res.mean)
    np.testing.assert_allclose(np.asarray(x_rec), np.asarray(X), atol=1e-8)


def test_pca_components_orthonormal():
    rng = np.random.default_rng(1)
    X = jnp.asarray(rng.standard_normal((100, 6)))
    res = pca_fit(X)
    gram = res.components @ res.components.T
    np.testing.assert_allclose(np.asarray(gram), np.eye(6), atol=1e-8)


def test_pca_explained_variance_matches_numpy_eigh():
    rng = np.random.default_rng(2)
    X_np = rng.standard_normal((200, 5))
    res = pca_fit(jnp.asarray(X_np))
    cov_np = np.cov(X_np, rowvar=False, bias=False)
    eig = np.sort(np.linalg.eigvalsh(cov_np))[::-1]
    np.testing.assert_allclose(
        np.asarray(res.explained_variance), eig, atol=1e-8
    )
    # Descending.
    ev = np.asarray(res.explained_variance)
    assert bool(np.all(np.diff(ev) <= 1e-9))


def test_pca_transform_decorrelates():
    rng = np.random.default_rng(3)
    # Correlated features.
    base = rng.standard_normal((300, 3))
    mix = rng.standard_normal((3, 5))
    X = jnp.asarray(base @ mix)
    res = pca_fit(X, n_components=3)
    z = np.asarray(pca_transform(X, res.components, res.mean))
    c = np.cov(z, rowvar=False, bias=False)
    off = c - np.diag(np.diag(c))
    assert float(np.abs(off).max()) < 1e-6


def test_pca_sign_is_deterministic():
    rng = np.random.default_rng(4)
    X = jnp.asarray(rng.standard_normal((80, 7)))
    a = pca_fit(X)
    b = pca_fit(X)
    np.testing.assert_array_equal(
        np.asarray(a.components), np.asarray(b.components)
    )
    # Largest-magnitude entry of each component is non-negative.
    comp = np.asarray(a.components)
    rows = np.arange(comp.shape[0])
    lead = comp[rows, np.argmax(np.abs(comp), axis=1)]
    assert bool(np.all(lead >= 0))


def test_pca_n_components_subset_shape():
    rng = np.random.default_rng(5)
    X = jnp.asarray(rng.standard_normal((40, 10)))
    res = pca_fit(X, n_components=3)
    assert res.components.shape == (3, 10)
    assert res.explained_variance.shape == (3,)
    z = pca_transform(X, res.components, res.mean)
    assert z.shape == (40, 3)


def test_pca_runs_on_active_backend():
    # On the cuSolver-dead GPU, pca_fit must still succeed (safe_eigh
    # transparently falls back to CPU and returns to the caller).
    rng = np.random.default_rng(6)
    X_np = rng.standard_normal((60, 4))
    res = pca_fit(jnp.asarray(X_np))
    cov_np = np.cov(X_np, rowvar=False, bias=False)
    eig = np.sort(np.linalg.eigvalsh(cov_np))[::-1]
    np.testing.assert_allclose(
        np.asarray(res.explained_variance), eig, atol=1e-8
    )


# ---------------------------------------------------------------------------
# pca: randomized solver
# ---------------------------------------------------------------------------


def _low_rank(n, d, r, seed):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, r))
    v = rng.standard_normal((r, d))
    return jnp.asarray(u @ v)


def test_pca_randomized_matches_full_on_low_rank():
    X = _low_rank(200, 30, 5, seed=0)
    full = pca_fit(X, n_components=5, solver='full')
    rand = pca_fit(
        X,
        n_components=5,
        solver='randomized',
        key=jax.random.PRNGKey(0),
        n_power_iterations=4,
    )
    # Exactly rank-5 data: the randomized range finder captures the
    # whole signal subspace, so the top-5 variances match the exact fit.
    np.testing.assert_allclose(
        np.asarray(rand.explained_variance),
        np.asarray(full.explained_variance),
        rtol=1e-5,
        atol=1e-6,
    )


def test_pca_randomized_reconstructs_low_rank():
    X = _low_rank(150, 40, 6, seed=1)
    res = pca_fit(
        X, n_components=6, solver='randomized', key=jax.random.PRNGKey(1)
    )
    z = pca_transform(X, res.components, res.mean)
    x_rec = pca_inverse_transform(z, res.components, res.mean)
    np.testing.assert_allclose(np.asarray(x_rec), np.asarray(X), atol=1e-6)


def test_pca_randomized_components_orthonormal():
    X = _low_rank(120, 25, 8, seed=2)
    res = pca_fit(
        X, n_components=8, solver='randomized', key=jax.random.PRNGKey(2)
    )
    gram = res.components @ res.components.T
    np.testing.assert_allclose(np.asarray(gram), np.eye(8), atol=1e-6)


def test_pca_randomized_reproducible():
    X = _low_rank(100, 20, 4, seed=3)
    a = pca_fit(X, n_components=4, solver='randomized', key=jax.random.PRNGKey(7))
    b = pca_fit(X, n_components=4, solver='randomized', key=jax.random.PRNGKey(7))
    np.testing.assert_array_equal(
        np.asarray(a.components), np.asarray(b.components)
    )


def test_pca_randomized_requires_key():
    X = _low_rank(50, 10, 3, seed=4)
    with pytest.raises(ValueError, match='requires a PRNG key'):
        pca_fit(X, n_components=3, solver='randomized')


def test_pca_invalid_solver_raises():
    X = _low_rank(50, 10, 3, seed=5)
    with pytest.raises(ValueError, match='expected'):
        pca_fit(X, n_components=3, solver='bogus')


# ---------------------------------------------------------------------------
# pca: gram + auto solvers
# ---------------------------------------------------------------------------


def test_pca_gram_matches_full_n_lt_d():
    rng = np.random.default_rng(10)
    X = jnp.asarray(rng.standard_normal((40, 100)))  # n < d (CompCor regime)
    full = pca_fit(X, n_components=8, solver='full')
    gram = pca_fit(X, n_components=8, solver='gram')
    np.testing.assert_allclose(
        np.asarray(gram.explained_variance),
        np.asarray(full.explained_variance),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.abs(np.asarray(gram.components)),
        np.abs(np.asarray(full.components)),
        atol=1e-6,
    )
    # Scores (the CompCor regressors) reconstruct identically.
    zf = pca_transform(X, full.components, full.mean)
    zg = pca_transform(X, gram.components, gram.mean)
    np.testing.assert_allclose(np.asarray(zg), np.asarray(zf), atol=1e-6)


def test_pca_gram_matches_full_n_gt_d():
    rng = np.random.default_rng(11)
    X = jnp.asarray(rng.standard_normal((200, 20)))
    full = pca_fit(X, n_components=5, solver='full')
    gram = pca_fit(X, n_components=5, solver='gram')
    np.testing.assert_allclose(
        np.asarray(gram.explained_variance),
        np.asarray(full.explained_variance),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(gram.components), np.asarray(full.components), atol=1e-6
    )


def test_pca_gram_components_orthonormal():
    rng = np.random.default_rng(12)
    X = jnp.asarray(rng.standard_normal((25, 90)))
    res = pca_fit(X, n_components=10, solver='gram')
    gram = res.components @ res.components.T
    np.testing.assert_allclose(np.asarray(gram), np.eye(10), atol=1e-6)


def test_pca_auto_picks_gram_when_n_lt_d():
    rng = np.random.default_rng(13)
    X = jnp.asarray(rng.standard_normal((30, 120)))
    auto = pca_fit(X, n_components=6, solver='auto')
    gram = pca_fit(X, n_components=6, solver='gram')
    np.testing.assert_array_equal(
        np.asarray(auto.components), np.asarray(gram.components)
    )


def test_pca_auto_picks_full_when_n_ge_d():
    rng = np.random.default_rng(14)
    X = jnp.asarray(rng.standard_normal((120, 30)))
    auto = pca_fit(X, n_components=6, solver='auto')
    full = pca_fit(X, n_components=6, solver='full')
    np.testing.assert_array_equal(
        np.asarray(auto.components), np.asarray(full.components)
    )


def test_pca_gram_topk_shape():
    rng = np.random.default_rng(15)
    X = jnp.asarray(rng.standard_normal((50, 200)))
    res = pca_fit(X, n_components=4, solver='gram')
    assert res.components.shape == (4, 200)
    assert res.explained_variance.shape == (4,)
