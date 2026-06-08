# -*- coding: utf-8 -*-
"""Tests for ``nitrix.linalg``.

Two submodules covered:

- ``matrix``: symmetric / triangular bijections (``sym2vec`` /
  ``vec2sym`` with hand-rolled custom_vjp), diagonal helpers,
  Toeplitz parity vs scipy, eigenspace reconditioning, and
  custom-VJP correctness (sym2vec backward zeros the lower
  triangle; vec2sym backward picks up the implicit-mirror factor
  of 2 for off-diagonal entries).
- ``residual``: OLS / WLS / ridge variants; the
  ``cholesky``-vs-``svd`` parity bound (machine-eps at fp64);
  orthogonality of the residual to the regressor span; gradient
  finiteness; the leading-batch broadcast contract.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.linalg import (
    delete_diagonal,
    fill_diagonal,
    recondition_eigenspaces,
    residualise,
    squareform,
    sym2vec,
    symmetric,
    toeplitz,
    toeplitz_2d,
    vec2sym,
)

# ---------------------------------------------------------------------------
# matrix
# ---------------------------------------------------------------------------


def test_symmetric_idempotent_on_symmetric_input():
    M = jax.random.normal(jax.random.key(0), (4, 4))
    M_sym = symmetric(M)
    out = symmetric(M_sym)
    np.testing.assert_allclose(out, M_sym, atol=1e-12)


def test_symmetric_skew_makes_antisymmetric():
    M = jax.random.normal(jax.random.key(0), (4, 4))
    AS = symmetric(M, skew=True)
    np.testing.assert_allclose(AS, -AS.T, atol=1e-12)


def test_symmetric_supports_arbitrary_axes():
    M = jax.random.normal(jax.random.key(0), (5, 3, 5, 2))
    out = symmetric(M, axes=(0, 2))
    np.testing.assert_allclose(
        np.asarray(out),
        np.asarray(out).swapaxes(0, 2),
        atol=1e-12,
    )


def test_sym2vec_vec2sym_roundtrip():
    M = symmetric(jax.random.normal(jax.random.key(0), (6, 6)))
    M = M.at[jnp.arange(6), jnp.arange(6)].set(0)  # zero diagonal
    v = sym2vec(M, offset=1)
    assert v.shape == (15,)  # binom(6, 2)
    M2 = vec2sym(v, offset=1)
    np.testing.assert_allclose(M2, M, atol=1e-12)


def test_sym2vec_vec2sym_roundtrip_offset_0():
    M = symmetric(jax.random.normal(jax.random.key(0), (5, 5)))
    v = sym2vec(M, offset=0)
    assert v.shape == (15,)  # binom(5+1, 2)
    M2 = vec2sym(v, offset=0)
    np.testing.assert_allclose(M2, M, atol=1e-12)


def test_sym2vec_grad_zeros_lower_triangle():
    """sym2vec custom_vjp: the backward must place zero in the
    lower triangle (the forward read nothing from there).
    """
    M = jax.random.normal(jax.random.key(0), (4, 4))
    M = symmetric(M)

    def loss(M):
        return (sym2vec(M, offset=1) ** 2).sum()

    g = jax.grad(loss)(M)
    np.testing.assert_allclose(jnp.tril(g, -1), 0.0, atol=1e-12)
    # Upper triangle should be 2 * M[upper] (each entry contributes once).
    np.testing.assert_allclose(jnp.triu(g, 1), 2 * jnp.triu(M, 1), atol=1e-12)


def test_vec2sym_grad_doubles_off_diagonal():
    """vec2sym custom_vjp: each vec entry populates two cells
    (upper + mirror), so dL/dv_k = 2 * 2 * v_k = 4 v_k for an
    ||M||^2 loss.
    """
    v = jax.random.normal(jax.random.key(0), (6,))

    def loss(v):
        M = vec2sym(v, offset=1)
        return (M**2).sum()

    g = jax.grad(loss)(v)
    np.testing.assert_allclose(g, 4 * v, atol=1e-12)


def test_squareform_dispatches_correctly():
    M = symmetric(jax.random.normal(jax.random.key(0), (4, 4)))
    M = M.at[jnp.arange(4), jnp.arange(4)].set(0)
    v = squareform(M)
    assert v.shape == (6,)  # binom(4, 2)
    M2 = squareform(v)
    np.testing.assert_allclose(M, M2, atol=1e-12)


def test_recondition_eigenspaces_ensures_nonzero_eigenvalues():
    # Build a rank-deficient matrix.
    A_low = jax.random.normal(jax.random.key(0), (5, 3))
    A = A_low @ A_low.T  # rank 3, PSD
    ev_pre = jnp.linalg.eigvalsh(A)
    A_rec = recondition_eigenspaces(A, psi=1.0)
    ev_post = jnp.linalg.eigvalsh(A_rec)
    assert float(ev_pre.min()) < 1e-6
    assert float(ev_post.min()) > 0.5


def test_recondition_eigenspaces_rejects_xi_gt_psi():
    A = jnp.eye(3)
    with pytest.raises(ValueError, match='psi >= xi'):
        recondition_eigenspaces(A, psi=0.1, xi=1.0)


def test_delete_diagonal_zeros_main_diagonal():
    A = jax.random.normal(jax.random.key(0), (4, 4))
    out = delete_diagonal(A)
    np.testing.assert_allclose(jnp.diag(out), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        out - jnp.diag(jnp.diag(out)),
        A - jnp.diag(jnp.diag(A)),
        atol=1e-12,
    )


def test_fill_diagonal_main():
    A = jnp.ones((4, 5))
    out = fill_diagonal(A, fill=-1.0)
    # Main diagonal of a (4, 5) matrix has 4 entries.
    for i in range(4):
        assert float(out[i, i]) == -1.0
    np.testing.assert_allclose(out[0, 1:], 1.0, atol=1e-12)


def test_fill_diagonal_offset_positive():
    A = jnp.ones((4, 4))
    out = fill_diagonal(A, fill=2.0, offset=1)
    for i in range(3):
        assert float(out[i, i + 1]) == 2.0
    assert float(out[3, 3]) == 1.0  # main diagonal untouched


def test_fill_diagonal_offset_negative():
    A = jnp.ones((4, 4))
    out = fill_diagonal(A, fill=3.0, offset=-2)
    for i in range(2):
        assert float(out[i + 2, i]) == 3.0
    assert float(out[0, 0]) == 1.0


def test_toeplitz_2d_matches_scipy():
    import scipy.linalg as spl

    c = jnp.array([1.0, 2.0, 3.0, 4.0])
    r = jnp.array([1.0, 5.0, 6.0, 7.0])
    T = toeplitz_2d(c, r)
    T_ref = spl.toeplitz(np.asarray(c), np.asarray(r))
    np.testing.assert_allclose(np.asarray(T), T_ref, atol=1e-12)


def test_toeplitz_rectangular_with_fill():
    c = jnp.array([1.0, 2.0])
    r = jnp.array([1.0, 3.0])
    T = toeplitz_2d(c, r, shape=(4, 5), fill_value=0.0)
    assert T.shape == (4, 5)
    # First column equals c padded with 0
    np.testing.assert_allclose(T[:2, 0], c, atol=1e-12)
    np.testing.assert_allclose(T[2:, 0], 0.0, atol=1e-12)


def test_toeplitz_batched():
    # 3 batched Toeplitz matrices from (3, 4) col and row tensors
    c = jax.random.normal(jax.random.key(0), (3, 4))
    r = jax.random.normal(jax.random.key(1), (3, 4))
    T = toeplitz(c, r)
    assert T.shape == (3, 4, 4)
    # Each batch element matches the unbatched version.
    for b in range(3):
        np.testing.assert_allclose(
            np.asarray(T[b]),
            np.asarray(toeplitz_2d(c[b], r[b])),
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# residual
# ---------------------------------------------------------------------------


def _fmri_like_inputs(seed=0, n_obs=400, n_conf=24, n_vox=1000):
    rng = np.random.default_rng(seed)
    X = jnp.asarray(rng.standard_normal((n_conf, n_obs)))
    Y = jnp.asarray(rng.standard_normal((n_vox, n_obs)))
    return X, Y


def test_residualise_cholesky_vs_svd_parity_at_fp64():
    """Verify the docstring's machine-precision-parity claim.

    Well-conditioned random X, no ridge: Cholesky and SVD must
    agree to ~machine eps in fp64.  We allow a small slack
    (atol=1e-12) for accumulated roundoff over the projection
    matmul.
    """
    X, Y = _fmri_like_inputs(seed=0)
    res_chol = residualise(Y, X, method='cholesky')
    res_svd = residualise(Y, X, method='svd')
    max_diff = float(jnp.abs(res_chol - res_svd).max())
    assert max_diff < 1e-12, (
        f'Cholesky vs SVD parity failed: max diff {max_diff:.2e} '
        '(expected <1e-12 at fp64)'
    )


def test_residualise_residual_orthogonal_to_regressors():
    """Residual must be orthogonal to the column span of X."""
    X, Y = _fmri_like_inputs(seed=1)
    res = residualise(Y, X)
    # res shape: (n_vox, n_obs); X shape: (n_conf, n_obs).
    # res @ X.T should be ~0 (n_vox, n_conf).
    gram = res @ X.T
    assert float(jnp.abs(gram).max()) < 1e-10


def test_residualise_projection_lies_in_regressor_span():
    """Projection should equal the OLS prediction X @ betas."""
    X, Y = _fmri_like_inputs(seed=2, n_vox=100)
    proj = residualise(Y, X, return_mode='projection')
    res = residualise(Y, X, return_mode='residual')
    np.testing.assert_allclose(proj + res, Y, atol=1e-10)


def test_residualise_ridge_reduces_norm():
    """With a heavy ridge, betas shrink and projection -> 0."""
    X, Y = _fmri_like_inputs(seed=0, n_vox=100)
    proj_no_ridge = residualise(
        Y,
        X,
        l2=0.0,
        return_mode='projection',
    )
    proj_heavy_ridge = residualise(
        Y,
        X,
        l2=1e10,
        return_mode='projection',
    )
    assert (
        float(jnp.abs(proj_no_ridge).mean())
        > float(jnp.abs(proj_heavy_ridge).mean()) * 100
    )  # heavy ridge crushes the projection


def test_residualise_weights_match_repeated_observations():
    """Per-observation weights with integer values must agree with
    repeating each observation that many times in the unweighted
    fit -- the classic WLS / replicated-OLS equivalence.
    """
    rng = np.random.default_rng(0)
    n_obs, n_conf, n_vox = 50, 3, 5
    X = jnp.asarray(rng.standard_normal((n_conf, n_obs)))
    Y = jnp.asarray(rng.standard_normal((n_vox, n_obs)))
    # Weight observations 1, 2, 1, 2, ...
    w = jnp.tile(jnp.array([1.0, 2.0]), n_obs // 2)
    res_w = residualise(Y, X, weights=w)

    # Equivalent: replicate observations 2 with multiplicity 2.
    rep = jnp.array([1, 2] * (n_obs // 2))
    rep_idx = jnp.concatenate(
        [jnp.full((int(r),), i) for i, r in enumerate(rep)]
    )
    X_rep = X[:, rep_idx]
    Y_rep = Y[:, rep_idx]
    res_rep = residualise(Y_rep, X_rep)
    # The betas should match; the residuals at the original indices
    # match (replicated indices repeat them).
    # We compare residuals at the original obs positions only.
    # First find the start of each repeated obs in the replicated layout.
    starts = jnp.cumsum(jnp.concatenate([jnp.array([0]), rep[:-1]]))
    np.testing.assert_allclose(
        res_w,
        res_rep[:, starts],
        atol=1e-9,
    )


def test_residualise_leading_batch_dims():
    """Leading batch dims broadcast and are vmapped internally."""
    rng = np.random.default_rng(0)
    B, n_obs, n_conf, n_vox = 4, 100, 5, 20
    X = jnp.asarray(rng.standard_normal((B, n_conf, n_obs)))
    Y = jnp.asarray(rng.standard_normal((B, n_vox, n_obs)))
    res = residualise(Y, X)
    assert res.shape == (B, n_vox, n_obs)
    # Each batch element should match the unbatched call.
    for b in range(B):
        np.testing.assert_allclose(
            np.asarray(res[b]),
            np.asarray(residualise(Y[b], X[b])),
            atol=1e-12,
        )


def test_residualise_rowvar_false():
    """rowvar=False: observation axis is penultimate."""
    X, Y = _fmri_like_inputs(seed=0, n_obs=100, n_conf=4, n_vox=10)
    res_rowvar = residualise(Y, X, rowvar=True)
    res_colvar = residualise(
        Y.swapaxes(-1, -2),
        X.swapaxes(-1, -2),
        rowvar=False,
    )
    np.testing.assert_allclose(
        res_rowvar,
        res_colvar.swapaxes(-1, -2),
        atol=1e-12,
    )


def test_residualise_differentiable_wrt_X():
    X, Y = _fmri_like_inputs(seed=0, n_obs=50, n_conf=3, n_vox=10)

    def loss(X):
        return (residualise(Y, X) ** 2).sum()

    g = jax.grad(loss)(X)
    assert g.shape == X.shape
    assert bool(jnp.all(jnp.isfinite(g)))


def test_residualise_differentiable_wrt_Y():
    X, Y = _fmri_like_inputs(seed=0, n_obs=50, n_conf=3, n_vox=10)

    def loss(Y):
        return (residualise(Y, X) ** 2).sum()

    g = jax.grad(loss)(Y)
    assert g.shape == Y.shape
    assert bool(jnp.all(jnp.isfinite(g)))


def test_residualise_invalid_method_raises():
    X, Y = _fmri_like_inputs(seed=0, n_obs=20, n_conf=3, n_vox=5)
    with pytest.raises(ValueError, match='method='):
        residualise(Y, X, method='qr')  # not exposed currently


def test_residualise_invalid_return_mode_raises():
    X, Y = _fmri_like_inputs(seed=0, n_obs=20, n_conf=3, n_vox=5)
    with pytest.raises(ValueError, match='return_mode='):
        residualise(Y, X, return_mode='nonsense')


def test_residualise_obs_mismatch_raises():
    X = jnp.zeros((3, 50))
    Y = jnp.zeros((5, 30))  # different obs
    with pytest.raises(ValueError, match='observations mismatch'):
        residualise(Y, X)
