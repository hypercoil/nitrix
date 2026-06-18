# -*- coding: utf-8 -*-
"""Tests for ``nitrix.stats.connectivity`` (analytic shrinkage covariance)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.stats.connectivity import (
    ebic_score,
    glasso,
    glasso_path,
    ledoit_wolf,
    oas,
    shrunk_covariance,
)


def _has_sklearn() -> bool:
    try:
        import sklearn.covariance  # noqa: F401

        return True
    except ImportError:
        return False


needs_sklearn = pytest.mark.skipif(
    not _has_sklearn(), reason='sklearn not installed'
)


@needs_sklearn
@pytest.mark.parametrize('n,p', [(50, 8), (30, 30), (20, 40)])
def test_ledoit_wolf_matches_sklearn(n, p):
    from sklearn.covariance import ledoit_wolf as sk

    X = np.random.default_rng(n * p).standard_normal((n, p))
    cov, shr = ledoit_wolf(jnp.asarray(X))
    sk_cov, sk_shr = sk(X)
    np.testing.assert_allclose(float(shr), sk_shr, atol=1e-12)
    np.testing.assert_allclose(np.asarray(cov), sk_cov, atol=1e-12)


@needs_sklearn
@pytest.mark.parametrize('n,p', [(50, 8), (30, 30), (20, 40)])
def test_oas_matches_sklearn(n, p):
    from sklearn.covariance import oas as sk

    X = np.random.default_rng(n * p + 1).standard_normal((n, p))
    cov, shr = oas(jnp.asarray(X))
    sk_cov, sk_shr = sk(X)
    np.testing.assert_allclose(float(shr), sk_shr, atol=1e-12)
    np.testing.assert_allclose(np.asarray(cov), sk_cov, atol=1e-12)


def test_shrunk_is_spd_in_small_sample():
    """Shrinkage keeps the covariance SPD / invertible even at p >= n, where
    the raw empirical covariance is singular."""
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.standard_normal((20, 40)))
    for method in ('ledoit_wolf', 'oas'):
        cov = shrunk_covariance(X, method=method)
        # SPD check on CPU (numpy) -- jnp.linalg.eigvalsh is cuSOLVER and fails
        # on the broken-stack GPU; the estimator itself is cuSOLVER-free.
        assert float(np.linalg.eigvalsh(np.asarray(cov)).min()) > 0
    with pytest.raises(ValueError):
        shrunk_covariance(X, method='nope')


def test_batched_and_differentiable():
    rng = np.random.default_rng(2)
    Xb = jnp.asarray(rng.standard_normal((5, 50, 8)))
    covs = jax.vmap(lambda x: ledoit_wolf(x)[0])(Xb)
    assert covs.shape == (5, 8, 8)
    g = jax.grad(lambda X: jnp.sum(ledoit_wolf(X)[0]))(
        jnp.asarray(rng.standard_normal((40, 6)))
    )
    assert bool(jnp.all(jnp.isfinite(g)))


def test_assume_centered():
    """assume_centered skips the mean subtraction (matches sklearn's flag)."""
    rng = np.random.default_rng(3)
    X = jnp.asarray(rng.standard_normal((40, 5)))
    Xc = X - jnp.mean(X, axis=0, keepdims=True)
    cov_a, _ = ledoit_wolf(Xc, assume_centered=True)
    cov_b, _ = ledoit_wolf(X, assume_centered=False)
    np.testing.assert_allclose(
        np.asarray(cov_a), np.asarray(cov_b), atol=1e-12
    )


# ---------------------------------------------------------------------------
# Graphical LASSO
#
# sklearn.covariance.GraphicalLasso is NOT used as a bit-match oracle: its own
# returned solution can violate the KKT stationarity of the off-diagonal-only
# program (different penalty convention / looser tolerance), so it is an
# unreliable reference.  We validate against the *definition* instead -- the KKT
# conditions of the convex program, symmetry / positive-definiteness, the
# lambda=0 maximum-likelihood limit, and support recovery on a known graph.
# ---------------------------------------------------------------------------


def _sample_cov(n: int, p: int, seed: int) -> np.ndarray:
    """A well-conditioned (p, p) sample covariance from n>p Gaussian draws."""
    X = np.random.default_rng(seed).standard_normal((n, p))
    Xc = X - X.mean(0, keepdims=True)
    return (Xc.T @ Xc) / n


def _kkt_residual(theta: np.ndarray, S: np.ndarray, lam: float):
    """KKT slacks of the off-diagonal-only glasso at ``theta``.

    Working covariance ``W = theta^-1``.  Stationarity:
      * diagonal      ``W_jj == S_jj``,
      * active edges  ``|W_ij - S_ij| == lam`` (subgradient is sign),
      * inactive      ``|W_ij - S_ij| <= lam``.
    Returns ``(diag_err, active_dev, inactive_excess)``.
    """
    W = np.linalg.inv(theta)
    R = W - S
    diag_err = float(np.max(np.abs(np.diag(R))))
    off = ~np.eye(theta.shape[0], dtype=bool)
    active = (np.abs(theta) > 1e-8) & off
    inactive = (~active) & off
    active_dev = (
        float(np.max(np.abs(np.abs(R[active]) - lam))) if active.any() else 0.0
    )
    inactive_excess = (
        float(np.max(np.clip(np.abs(R[inactive]) - lam, 0.0, None)))
        if inactive.any()
        else 0.0
    )
    return diag_err, active_dev, inactive_excess


@pytest.mark.parametrize('lam', [0.05, 0.1, 0.25])
def test_glasso_satisfies_kkt(lam):
    """The recovered precision is the exact optimum of the convex program."""
    S = _sample_cov(60, 10, seed=11)
    theta = np.asarray(glasso(jnp.asarray(S), lam))
    diag_err, active_dev, inactive_excess = _kkt_residual(theta, S, lam)
    assert diag_err < 1e-8
    assert active_dev < 1e-7
    assert inactive_excess < 1e-9


@pytest.mark.parametrize('lam', [0.05, 0.1, 0.25])
def test_glasso_symmetric_and_pd(lam):
    S = _sample_cov(60, 10, seed=12)
    theta = np.asarray(glasso(jnp.asarray(S), lam))
    np.testing.assert_allclose(theta, theta.T, atol=1e-10)
    # SPD on CPU (eigvalsh is cuSOLVER and dies on the broken stack).
    assert float(np.linalg.eigvalsh(theta).min()) > 0


def test_glasso_zero_lambda_is_mle():
    """At lambda=0 the penalty vanishes -> Theta = S^-1 (the MLE precision)."""
    S = _sample_cov(200, 8, seed=13)
    theta = np.asarray(glasso(jnp.asarray(S), 0.0))
    np.testing.assert_allclose(theta, np.linalg.inv(S), atol=1e-6)


def test_glasso_monotone_sparsity():
    """Edge count is non-increasing in lambda (stronger penalty -> sparser)."""
    S = _sample_cov(60, 12, seed=14)
    lams = [0.02, 0.05, 0.1, 0.2, 0.4]
    counts = []
    for lam in lams:
        theta = np.asarray(glasso(jnp.asarray(S), lam))
        off = ~np.eye(theta.shape[0], dtype=bool)
        counts.append(int(np.sum(np.abs(theta[off]) > 1e-8) // 2))
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] < counts[0]  # the penalty actually bites


def test_glasso_support_recovery_block_independence():
    """Two independent variable blocks: at a moderate lambda the cross-block
    precision entries are driven to *exactly* zero (conditional independence),
    while within-block edges survive."""
    rng = np.random.default_rng(15)
    n, b = 800, 4
    # Two correlated blocks, independent of each other.
    A = rng.standard_normal((b, b))
    block_cov = A @ A.T + b * np.eye(b)
    L = np.linalg.cholesky(block_cov)
    Z = rng.standard_normal((n, 2 * b))
    X = np.empty_like(Z)
    X[:, :b] = Z[:, :b] @ L.T
    X[:, b:] = Z[:, b:] @ L.T
    Xc = X - X.mean(0, keepdims=True)
    S = (Xc.T @ Xc) / n
    theta = np.asarray(glasso(jnp.asarray(S), 0.5))
    cross = np.abs(theta[:b, b:])
    assert float(cross.max()) == 0.0  # soft-threshold zeros them exactly
    # within-block structure is retained.
    within = np.abs(theta[:b, :b][~np.eye(b, dtype=bool)])
    assert float(within.max()) > 0.0


def test_glasso_path_matches_individual_and_shape():
    S = _sample_cov(60, 9, seed=16)
    lams = jnp.asarray([0.3, 0.15, 0.05])
    thetas = glasso_path(jnp.asarray(S), lams)
    assert thetas.shape == (3, 9, 9)
    for i, lam in enumerate([0.3, 0.15, 0.05]):
        # Warm-started path converges to the same optimum as a cold solve.
        cold = glasso(jnp.asarray(S), lam)
        np.testing.assert_allclose(
            np.asarray(thetas[i]), np.asarray(cold), atol=1e-7
        )


def test_ebic_selects_sparse_truth():
    """EBIC over a path prefers a sparse model close to a sparse ground truth,
    not the dense lambda~0 end."""
    rng = np.random.default_rng(17)
    p, n = 8, 300
    # Tridiagonal (chain) precision -> a banded, sparse graph.
    theta_true = (
        np.diag(np.full(p, 2.0))
        + np.diag(np.full(p - 1, -0.6), 1)
        + np.diag(np.full(p - 1, -0.6), -1)
    )
    cov_true = np.linalg.inv(theta_true)
    Lc = np.linalg.cholesky(cov_true)
    X = rng.standard_normal((n, p)) @ Lc.T
    Xc = X - X.mean(0, keepdims=True)
    S = (Xc.T @ Xc) / n

    lams = jnp.asarray([0.4, 0.25, 0.15, 0.08, 0.03, 0.0])
    thetas = glasso_path(jnp.asarray(S), lams)
    scores = np.asarray(
        [
            float(ebic_score(thetas[i], jnp.asarray(S), n))
            for i in range(len(lams))
        ]
    )
    best = int(np.argmin(scores))
    assert np.all(np.isfinite(scores))
    # EBIC prefers an interior model -- neither the dense lambda=0 endpoint nor
    # the over-sparse lambda=0.4 endpoint.
    assert 0 < best < len(lams) - 1
    theta_hat = np.asarray(thetas[best])
    upper = np.triu(np.ones((p, p), bool), k=1)
    band = np.abs(np.add.outer(np.arange(p), -np.arange(p))) == 1
    true_edges = band & upper
    recovered = (np.abs(theta_hat) > 1e-8) & upper
    # Every true band edge is recovered (no false negatives) and the model stays
    # sparse: at most a couple of spurious off-band edges out of 21 candidates.
    assert bool(np.all(recovered[true_edges]))
    false_pos = int(np.sum(recovered & ~band))
    assert false_pos <= 2


def test_glasso_differentiable_and_cusolver_free():
    """Differentiable through the fixed iteration budget; the lowered HLO issues
    no cuSOLVER custom-call (rolled coordinate descent + rolled-Cholesky logdet)."""
    S = jnp.asarray(_sample_cov(40, 6, seed=18))
    g = jax.grad(lambda s: jnp.sum(glasso(s, 0.1)))(S)
    assert bool(jnp.all(jnp.isfinite(g)))

    import re

    tokens = (
        'cusolver',
        'potrf',
        'getrf',
        'geqrf',
        'gesvd',
        'cholesky',
        'eigh',
    )
    hlo = (
        jax.jit(lambda s: glasso(s, 0.1)).lower(S).compile().as_text().lower()
    )
    targets = re.findall(r'custom_call_target="([^"]+)"', hlo)
    assert not [t for t in targets if any(tok in t for tok in tokens)]
