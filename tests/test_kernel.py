# -*- coding: utf-8 -*-
"""Tests for ``nitrix.linalg.kernel`` -- the kernel primitives (ER7 coverage gap).

The primitives (``linear_kernel`` / ``linear_distance`` / ``parameterised_norm``
and the nonlinear ``rbf`` / ``gaussian`` / ``polynomial`` / ``sigmoid`` / ``cosine``
kernels) were exercised only indirectly via the GP suite; here they are anchored
directly against explicit numpy formulae and scikit-learn references, including
the diagonal (vector ``theta``) and full (matrix ``theta``) Mahalanobis metrics
and the batched (``...``) and jit paths.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp

from nitrix.linalg.kernel import (
    cosine_kernel,
    gaussian_kernel,
    linear_distance,
    linear_kernel,
    parameterised_norm,
    polynomial_kernel,
    rbf_kernel,
    sigmoid_kernel,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X0 = rng.standard_normal((6, 4))
    X1 = rng.standard_normal((5, 4))
    return X0, X1


# ---------------------------------------------------------------------------
# linear_kernel -- Euclidean and Mahalanobis (vector / matrix theta)
# ---------------------------------------------------------------------------


def test_linear_kernel_euclidean_and_self(data):
    X0, X1 = data
    np.testing.assert_allclose(
        np.asarray(linear_kernel(jnp.asarray(X0), jnp.asarray(X1))),
        X0 @ X1.T, atol=1e-10,
    )
    K = np.asarray(linear_kernel(jnp.asarray(X0)))  # self-kernel
    np.testing.assert_allclose(K, X0 @ X0.T, atol=1e-10)
    np.testing.assert_allclose(K, K.T, atol=1e-10)  # symmetric


def test_linear_kernel_vector_and_matrix_theta(data):
    X0, X1 = data
    theta_v = np.abs(np.random.default_rng(1).standard_normal(4)) + 0.1
    np.testing.assert_allclose(
        np.asarray(linear_kernel(jnp.asarray(X0), jnp.asarray(X1),
                                 theta=jnp.asarray(theta_v))),
        (X0 * theta_v) @ X1.T, atol=1e-10,
    )
    A = np.random.default_rng(2).standard_normal((4, 4))
    theta_m = A @ A.T  # SPD metric
    np.testing.assert_allclose(
        np.asarray(linear_kernel(jnp.asarray(X0), jnp.asarray(X1),
                                 theta=jnp.asarray(theta_m))),
        X0 @ theta_m @ X1.T, atol=1e-10,
    )


# ---------------------------------------------------------------------------
# linear_distance -- squared L2 / Mahalanobis, non-negativity, zero self-diag
# ---------------------------------------------------------------------------


def test_linear_distance_squared_euclidean(data):
    X0, X1 = data
    ref = ((X0[:, None, :] - X1[None, :, :]) ** 2).sum(-1)
    np.testing.assert_allclose(
        np.asarray(linear_distance(jnp.asarray(X0), jnp.asarray(X1))),
        ref, atol=1e-9,
    )
    # self-distance has an exactly-zero diagonal (clipped, never negative)
    D = np.asarray(linear_distance(jnp.asarray(X0)))
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-9)
    assert np.all(D >= 0.0)


def test_linear_distance_mahalanobis(data):
    X0, X1 = data
    theta_v = np.abs(np.random.default_rng(3).standard_normal(4)) + 0.1
    ref_v = (theta_v * (X0[:, None, :] - X1[None, :, :]) ** 2).sum(-1)
    np.testing.assert_allclose(
        np.asarray(linear_distance(jnp.asarray(X0), jnp.asarray(X1),
                                   theta=jnp.asarray(theta_v))),
        ref_v, atol=1e-9,
    )
    A = np.random.default_rng(4).standard_normal((4, 4))
    theta_m = A @ A.T
    diff = X0[:, None, :] - X1[None, :, :]
    ref_m = np.einsum('nmd,de,nme->nm', diff, theta_m, diff)
    np.testing.assert_allclose(
        np.asarray(linear_distance(jnp.asarray(X0), jnp.asarray(X1),
                                   theta=jnp.asarray(theta_m))),
        ref_m, atol=1e-8,
    )


# ---------------------------------------------------------------------------
# parameterised_norm -- unit rows (L2) and squared variant
# ---------------------------------------------------------------------------


def test_parameterised_norm_unit_rows(data):
    X0, _ = data
    Xn = np.asarray(parameterised_norm(jnp.asarray(X0)))
    np.testing.assert_allclose(np.linalg.norm(Xn, axis=-1), 1.0, atol=1e-7)
    # squared=True divides by norm^2 instead of norm
    Xs = np.asarray(parameterised_norm(jnp.asarray(X0), squared=True))
    np.testing.assert_allclose(
        Xs, X0 / (X0 ** 2).sum(-1, keepdims=True), atol=1e-7
    )


# ---------------------------------------------------------------------------
# Nonlinear kernels -- scikit-learn references
# ---------------------------------------------------------------------------


def test_rbf_matches_sklearn_and_unit_diagonal(data):
    from sklearn.metrics.pairwise import rbf_kernel as sk_rbf

    X0, X1 = data
    np.testing.assert_allclose(
        np.asarray(rbf_kernel(jnp.asarray(X0), jnp.asarray(X1), gamma=0.3)),
        sk_rbf(X0, X1, gamma=0.3), atol=1e-9,
    )
    # default gamma = 1/d (sklearn convention); self-kernel has unit diagonal
    np.testing.assert_allclose(
        np.asarray(rbf_kernel(jnp.asarray(X0))),
        sk_rbf(X0, gamma=1.0 / X0.shape[-1]), atol=1e-9,
    )
    np.testing.assert_allclose(np.diag(np.asarray(rbf_kernel(jnp.asarray(X0)))),
                               1.0, atol=1e-12)


def test_gaussian_kernel_is_rbf_reparameterised(data):
    X0, X1 = data
    sigma = 1.3
    np.testing.assert_allclose(
        np.asarray(gaussian_kernel(jnp.asarray(X0), jnp.asarray(X1), sigma=sigma)),
        np.asarray(rbf_kernel(jnp.asarray(X0), jnp.asarray(X1),
                              gamma=1.0 / (2.0 * sigma * sigma))),
        atol=1e-10,
    )


def test_polynomial_and_sigmoid_match_sklearn(data):
    from sklearn.metrics.pairwise import polynomial_kernel as sk_poly
    from sklearn.metrics.pairwise import sigmoid_kernel as sk_sig

    X0, X1 = data
    np.testing.assert_allclose(
        np.asarray(polynomial_kernel(jnp.asarray(X0), jnp.asarray(X1),
                                     gamma=0.2, order=3, r=1.0)),
        sk_poly(X0, X1, degree=3, gamma=0.2, coef0=1.0), atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(sigmoid_kernel(jnp.asarray(X0), jnp.asarray(X1),
                                  gamma=0.2, r=0.5)),
        sk_sig(X0, X1, gamma=0.2, coef0=0.5), atol=1e-9,
    )


def test_cosine_kernel_matches_sklearn_and_unit_self_diagonal(data):
    from sklearn.metrics.pairwise import cosine_similarity

    X0, X1 = data
    np.testing.assert_allclose(
        np.asarray(cosine_kernel(jnp.asarray(X0), jnp.asarray(X1))),
        cosine_similarity(X0, X1), atol=1e-7,
    )
    np.testing.assert_allclose(
        np.diag(np.asarray(cosine_kernel(jnp.asarray(X0)))), 1.0, atol=1e-7
    )


# ---------------------------------------------------------------------------
# Batched (...) leading dims and jit cleanliness
# ---------------------------------------------------------------------------


def test_kernels_batched_and_jit(data):
    rng = np.random.default_rng(5)
    Xb = rng.standard_normal((3, 6, 4))  # (batch=3, n=6, d=4)
    # batched rbf == stacked per-batch rbf
    Kb = np.asarray(rbf_kernel(jnp.asarray(Xb), gamma=0.3))
    assert Kb.shape == (3, 6, 6)
    for b in range(3):
        np.testing.assert_allclose(
            Kb[b], np.asarray(rbf_kernel(jnp.asarray(Xb[b]), gamma=0.3)),
            atol=1e-10,
        )
    # jit-clean
    f = jax.jit(lambda X: rbf_kernel(X, gamma=0.3))
    np.testing.assert_allclose(np.asarray(f(jnp.asarray(Xb))), Kb, atol=1e-10)
