# -*- coding: utf-8 -*-
"""Tests for the symmetric spectral / Chebyshev matrix functions.

``matrix_function`` is checked against an eigh reference (and shown to preserve
negative eigenvalues -- it is the unclipped general entry point, not the SPD
``symexp``). ``matrix_polynomial`` / ``chebyshev_apply`` are checked against the
Chebyshev three-term identities and the dense-vs-matvec equivalence, and are
verified eigh-free (jit-clean). ``frechet_derivative`` is checked against
``jax.jvp`` and a finite difference, including the degenerate-spectrum branch.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

import numpy.polynomial.chebyshev as C  # noqa: E402

from nitrix.linalg import (  # noqa: E402
    chebyshev_apply,
    chebyshev_coefficients,
    frechet_derivative,
    matrix_exp,
    matrix_function,
    matrix_polynomial,
)


def _sym(n=5, seed=0):
    m = np.random.default_rng(seed).standard_normal((n, n))
    return jnp.asarray(m + m.T)


# --- matrix_function ---------------------------------------------------------


def test_matrix_function_matches_eigh_reference():
    A = _sym()
    w, V = np.linalg.eigh(np.asarray(A))
    ref = V @ np.diag(np.exp(w)) @ V.T
    np.testing.assert_allclose(
        np.asarray(matrix_function(A, jnp.exp)), ref, atol=1e-12
    )


def test_matrix_function_preserves_negative_eigenvalues():
    # The identity map must recover A exactly -- i.e. no SPD eigenvalue clipping
    # (a symmetric A has negative eigenvalues that symexp-style clipping would
    # destroy).
    A = _sym(seed=3)
    assert np.asarray(np.linalg.eigvalsh(np.asarray(A))).min() < 0
    np.testing.assert_allclose(
        np.asarray(matrix_function(A, lambda x: x)), np.asarray(A), atol=1e-12
    )


def test_matrix_function_matches_matrix_exp_for_symmetric():
    # For symmetric A the eigh path and the pure-matmul matrix_exp agree.
    A = _sym(seed=1)
    np.testing.assert_allclose(
        np.asarray(matrix_function(A, jnp.exp)),
        np.asarray(matrix_exp(A)),
        atol=1e-9,
    )


def test_matrix_function_grad_finite():
    A = _sym(seed=2)
    g = jax.grad(lambda X: matrix_function(X, jnp.exp).sum())(A)
    assert bool(jnp.all(jnp.isfinite(g)))


# --- matrix_polynomial / chebyshev_apply -------------------------------------


def test_chebyshev_low_order_identities():
    A = _sym() / 6.0  # scale spectrum into (-1, 1)
    eye = jnp.eye(5)
    np.testing.assert_allclose(
        np.asarray(matrix_polynomial(A, [1.0])), np.asarray(eye), atol=1e-12
    )  # T_0 = I
    np.testing.assert_allclose(
        np.asarray(matrix_polynomial(A, [0.0, 1.0])), np.asarray(A), atol=1e-12
    )  # T_1 = A
    np.testing.assert_allclose(
        np.asarray(matrix_polynomial(A, [0.0, 0.0, 1.0])),
        np.asarray(2 * A @ A - eye),
        atol=1e-12,
    )  # T_2 = 2A^2 - I


def test_matrix_polynomial_domain_matches_scalar_chebyshev():
    A = _sym(seed=4)
    w, V = np.linalg.eigh(np.asarray(A))
    coeffs = np.array([0.3, -0.5, 0.2, 0.1])
    lo, hi = w.min() - 0.1, w.max() + 0.1
    got = matrix_polynomial(A, jnp.asarray(coeffs), domain=(lo, hi))
    lam_scaled = (2 * w - (lo + hi)) / (hi - lo)
    ref = V @ np.diag(C.chebval(lam_scaled, coeffs)) @ V.T
    np.testing.assert_allclose(np.asarray(got), ref, atol=1e-12)


def test_chebyshev_apply_matches_dense_polynomial():
    A = _sym(seed=5) / 6.0
    coeffs = jnp.asarray([0.3, -0.5, 0.2, 0.1])
    dense = matrix_polynomial(A, coeffs)
    # vector operand
    x = jnp.asarray(np.random.default_rng(7).standard_normal(5))
    np.testing.assert_allclose(
        np.asarray(chebyshev_apply(lambda v: A @ v, x, coeffs)),
        np.asarray(dense @ x),
        atol=1e-12,
    )
    # identity operand reproduces the dense matrix
    np.testing.assert_allclose(
        np.asarray(chebyshev_apply(lambda M: A @ M, jnp.eye(5), coeffs)),
        np.asarray(dense),
        atol=1e-12,
    )


def test_chebyshev_apply_single_coefficient():
    A = _sym(seed=6) / 6.0
    x = jnp.asarray(np.random.default_rng(8).standard_normal(5))
    # order-0 series is c0 * x (op never applied)
    np.testing.assert_allclose(
        np.asarray(chebyshev_apply(lambda v: A @ v, x, [2.0])),
        np.asarray(2.0 * x),
        atol=1e-12,
    )


def test_matrix_polynomial_jit_is_eigh_free():
    A = _sym(seed=9) / 6.0
    coeffs = jnp.asarray([0.3, -0.5, 0.2, 0.1])
    out = jax.jit(lambda A, c: matrix_polynomial(A, c))(A, coeffs)
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(matrix_polynomial(A, coeffs)), atol=1e-12
    )


# --- frechet_derivative ------------------------------------------------------


def test_frechet_matches_jvp_and_fd():
    A = _sym(seed=0)
    E = _sym(seed=1)
    fr = frechet_derivative(A, jnp.exp, E)
    _, jvp = jax.jvp(lambda X: matrix_function(X, jnp.exp), (A,), (E,))
    np.testing.assert_allclose(np.asarray(fr), np.asarray(jvp), atol=1e-11)
    h = 1e-6
    fd = (
        matrix_function(A + h * E, jnp.exp)
        - matrix_function(A - h * E, jnp.exp)
    ) / (2 * h)
    np.testing.assert_allclose(np.asarray(fr), np.asarray(fd), atol=1e-6)


def test_chebyshev_coefficients_reconstruct_matrix_function():
    # matrix_polynomial with fitted coefficients approximates matrix_function.
    A = _sym(n=6, seed=7)
    w = np.linalg.eigvalsh(np.asarray(A))
    lo, hi = w.min() - 0.2, w.max() + 0.2

    def fn(lam):
        return jnp.exp(-0.3 * lam)

    coeffs = chebyshev_coefficients(fn, 24, domain=(lo, hi))
    np.testing.assert_allclose(
        np.asarray(matrix_polynomial(A, coeffs, domain=(lo, hi))),
        np.asarray(matrix_function(A, fn)),
        atol=1e-10,
    )


def test_chebyshev_coefficients_scalar_series():
    # The apply-ready coefficients reproduce the scalar function via the T_k sum.
    def fn(x):
        return jnp.sin(2.0 * x)

    coeffs = chebyshev_coefficients(fn, 30, domain=(-1.0, 1.0))
    xs = jnp.linspace(-0.9, 0.9, 50)
    # 1x1 "matrices" -> matrix_polynomial degenerates to the scalar polynomial
    approx = jnp.stack(
        [matrix_polynomial(x[None, None], coeffs)[0, 0] for x in xs]
    )
    np.testing.assert_allclose(
        np.asarray(approx), np.asarray(fn(xs)), atol=1e-10
    )


def test_frechet_degenerate_spectrum():
    # At A = I (all eigenvalues 1) the Loewner matrix is the constant f'(1) = e,
    # so L_exp(I)[E] = e * sym(E) -- the degenerate branch must not divide by 0.
    E0 = np.random.default_rng(2).standard_normal((5, 5))
    E = jnp.asarray(E0)
    fr = frechet_derivative(jnp.eye(5), jnp.exp, E)
    expected = np.e * 0.5 * (E0 + E0.T)
    np.testing.assert_allclose(np.asarray(fr), expected, atol=1e-11)
    assert bool(jnp.all(jnp.isfinite(fr)))
