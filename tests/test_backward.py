# -*- coding: utf-8 -*-
"""G1 — finite-difference checks for the per-algebra backward kernels.

Per SPEC_UPDATE §3.1 each built-in algebra (except ``BOOLEAN``) ships
with a hand-derived backward that must pass finite-difference checks
at the pinned per-dtype tolerance.  Per IMPLEMENTATION_PLAN §3.1 G1,
failing this gate for a given algebra means that algebra ships
forward-only with a documented gradient raise.

Tests run at ``float64`` so the finite-difference reference is not
dominated by quantisation.  Subgradient algebras (TROPICAL_*) are
checked with inputs that have a unique argmax / argmin so the
subgradient is well-defined; ties are flagged separately.
``EUCLIDEAN`` is checked away from the ``C = 0`` singularity; the
guard at the singularity is asserted by checking that the gradient
is finite (not NaN) when ``A_row == B_col``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax.test_util as jtu

from nitrix.semiring import (
    BOOLEAN,
    EUCLIDEAN,
    LOG,
    REAL,
    TROPICAL_MAX_PLUS,
    TROPICAL_MIN_PLUS,
    semiring_ell_matmul,
    semiring_matmul,
)
from nitrix.sparse import ell_from_dense


jax.config.update('jax_enable_x64', True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matmul_loss(A, B, semiring, backend='jax'):
    return jnp.sum(
        semiring_matmul(A, B, semiring=semiring, backend=backend) ** 2,
    )


def _ell_loss(values, B, indices, semiring, n_cols, backend='jax'):
    return jnp.sum(
        semiring_ell_matmul(
            values, indices, B,
            semiring=semiring, n_cols=n_cols, backend=backend,
        ) ** 2,
    )


def _finite_diff(fn, x, eps=1e-5):
    '''Symmetric finite-difference gradient.

    Returns the same-shape gradient of ``fn`` w.r.t. ``x``, evaluated
    numerically.  Loops over flat indices; only used in tests, so the
    O(N) eval cost is fine.
    '''
    x_flat = x.reshape(-1)
    out = np.zeros_like(np.asarray(x_flat, dtype=np.float64))
    for i in range(x_flat.size):
        e = jnp.zeros_like(x_flat).at[i].set(eps)
        e = e.reshape(x.shape)
        f_plus = float(fn(x + e))
        f_minus = float(fn(x - e))
        out[i] = (f_plus - f_minus) / (2 * eps)
    return out.reshape(x.shape)


def _close(a, b, atol, rtol):
    return float(jnp.max(jnp.abs(a - b))), bool(
        jnp.allclose(a, b, atol=atol, rtol=rtol)
    )


# ---------------------------------------------------------------------------
# Dense matmul backwards
# ---------------------------------------------------------------------------


def _matmul_pair(seed, m=4, k=4, n=4, scale=1.0):
    ka, kb = jax.random.split(jax.random.key(seed))
    A = jax.random.normal(ka, (m, k), dtype=jnp.float64) * scale
    B = jax.random.normal(kb, (k, n), dtype=jnp.float64) * scale
    return A, B


def test_real_matmul_backward():
    A, B = _matmul_pair(0)
    gA, gB = jax.grad(_matmul_loss, argnums=(0, 1))(A, B, REAL)
    # Reference: ∂/∂A sum((AB)^2) = 2 * (AB) @ B^T
    AB = A @ B
    gA_ref = 2 * AB @ B.T
    gB_ref = 2 * A.T @ AB
    np.testing.assert_allclose(gA, gA_ref, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(gB, gB_ref, atol=1e-10, rtol=1e-10)


def test_log_matmul_backward_matches_finite_difference():
    A, B = _matmul_pair(1)
    gA, gB = jax.grad(_matmul_loss, argnums=(0, 1))(A, B, LOG)
    gA_fd = _finite_diff(lambda x: _matmul_loss(x, B, LOG), A)
    gB_fd = _finite_diff(lambda x: _matmul_loss(A, x, LOG), B)
    np.testing.assert_allclose(gA, gA_fd, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(gB, gB_fd, atol=1e-6, rtol=1e-6)


def test_tropical_max_plus_matmul_backward_matches_finite_difference():
    # Use inputs with a unique argmax (random normal is fine almost surely).
    A, B = _matmul_pair(2)
    gA, gB = jax.grad(_matmul_loss, argnums=(0, 1))(
        A, B, TROPICAL_MAX_PLUS,
    )
    # FD step must be smaller than the gap to the second-best k.
    gA_fd = _finite_diff(
        lambda x: _matmul_loss(x, B, TROPICAL_MAX_PLUS), A, eps=1e-4,
    )
    gB_fd = _finite_diff(
        lambda x: _matmul_loss(A, x, TROPICAL_MAX_PLUS), B, eps=1e-4,
    )
    np.testing.assert_allclose(gA, gA_fd, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(gB, gB_fd, atol=1e-6, rtol=1e-6)


def test_tropical_min_plus_matmul_backward_matches_finite_difference():
    A, B = _matmul_pair(3)
    gA, gB = jax.grad(_matmul_loss, argnums=(0, 1))(
        A, B, TROPICAL_MIN_PLUS,
    )
    gA_fd = _finite_diff(
        lambda x: _matmul_loss(x, B, TROPICAL_MIN_PLUS), A, eps=1e-4,
    )
    gB_fd = _finite_diff(
        lambda x: _matmul_loss(A, x, TROPICAL_MIN_PLUS), B, eps=1e-4,
    )
    np.testing.assert_allclose(gA, gA_fd, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(gB, gB_fd, atol=1e-6, rtol=1e-6)


def test_euclidean_matmul_backward_matches_finite_difference():
    # Random inputs almost surely keep C[i, j] > 0; the singularity guard
    # is tested separately below.
    A, B = _matmul_pair(4, scale=1.0)
    gA, gB = jax.grad(_matmul_loss, argnums=(0, 1))(A, B, EUCLIDEAN)
    gA_fd = _finite_diff(lambda x: _matmul_loss(x, B, EUCLIDEAN), A)
    gB_fd = _finite_diff(lambda x: _matmul_loss(A, x, EUCLIDEAN), B)
    np.testing.assert_allclose(gA, gA_fd, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(gB, gB_fd, atol=1e-6, rtol=1e-6)


def test_euclidean_singularity_guard_keeps_grad_finite():
    # Construct A and B so that C[2, 0] = 0 (row 2 of A equals column 0 of B
    # interpreted as a length-k vector).
    A, B = _matmul_pair(5)
    B = B.at[:, 0].set(A[2])
    gA, gB = jax.grad(_matmul_loss, argnums=(0, 1))(A, B, EUCLIDEAN)
    assert bool(jnp.all(jnp.isfinite(gA)))
    assert bool(jnp.all(jnp.isfinite(gB)))


def test_boolean_matmul_backward_rule_raises():
    # BOOLEAN ships with a backward *rule* that always raises -- the
    # only way a user can actually invoke it is by reaching into the
    # algebra's ``matmul_vjp``.  jax.grad cannot drive the rule from
    # bool inputs since the cast through ``> 0`` masks the gradient
    # before reaching ``semiring_matmul``, so we exercise the rule
    # directly to confirm the diagnostic.
    from nitrix.semiring._backward import boolean_matmul_vjp
    with pytest.raises(TypeError, match='BOOLEAN'):
        boolean_matmul_vjp(
            (jnp.zeros((4, 4)), jnp.zeros((4, 4)), jnp.zeros((4, 4))),
            jnp.zeros((4, 4)),
        )


def test_boolean_ell_matmul_backward_rule_raises():
    from nitrix.semiring._backward import boolean_ell_matmul_vjp
    with pytest.raises(TypeError, match='BOOLEAN'):
        boolean_ell_matmul_vjp(
            (
                jnp.zeros((4, 4)),
                jnp.zeros((4, 4), jnp.int32),
                jnp.zeros((4, 4)),
                jnp.zeros((4, 4)),
            ),
            jnp.zeros((4, 4)),
        )


# ---------------------------------------------------------------------------
# ELL backwards
# ---------------------------------------------------------------------------


def _ell_pair(seed, m=4, n=4, k_max=4, identity=0.0):
    Akey, Bkey = jax.random.split(jax.random.key(seed))
    A = jax.random.normal(Akey, (m, n), dtype=jnp.float64)
    B = jax.random.normal(Bkey, (n, n), dtype=jnp.float64)
    ell = ell_from_dense(A, k_max=k_max, identity=identity)
    return ell.values, ell.indices, B


def test_real_ell_matmul_backward():
    values, indices, B = _ell_pair(10)
    gV, gB = jax.grad(_ell_loss, argnums=(0, 1))(values, B, indices, REAL, 4)
    gV_fd = _finite_diff(
        lambda v: _ell_loss(v, B, indices, REAL, 4), values,
    )
    gB_fd = _finite_diff(
        lambda x: _ell_loss(values, x, indices, REAL, 4), B,
    )
    np.testing.assert_allclose(gV, gV_fd, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(gB, gB_fd, atol=1e-8, rtol=1e-8)


def test_log_ell_matmul_backward():
    values, indices, B = _ell_pair(11, identity=-jnp.inf)
    gV, gB = jax.grad(_ell_loss, argnums=(0, 1))(values, B, indices, LOG, 4)
    gV_fd = _finite_diff(
        lambda v: _ell_loss(v, B, indices, LOG, 4), values,
    )
    gB_fd = _finite_diff(
        lambda x: _ell_loss(values, x, indices, LOG, 4), B,
    )
    np.testing.assert_allclose(gV, gV_fd, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(gB, gB_fd, atol=1e-6, rtol=1e-6)


def test_tropical_max_plus_ell_matmul_backward():
    values, indices, B = _ell_pair(12, identity=-jnp.inf)
    gV, gB = jax.grad(_ell_loss, argnums=(0, 1))(
        values, B, indices, TROPICAL_MAX_PLUS, 4,
    )
    gV_fd = _finite_diff(
        lambda v: _ell_loss(v, B, indices, TROPICAL_MAX_PLUS, 4),
        values, eps=1e-4,
    )
    gB_fd = _finite_diff(
        lambda x: _ell_loss(values, x, indices, TROPICAL_MAX_PLUS, 4),
        B, eps=1e-4,
    )
    np.testing.assert_allclose(gV, gV_fd, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(gB, gB_fd, atol=1e-6, rtol=1e-6)


def test_tropical_min_plus_ell_matmul_backward():
    values, indices, B = _ell_pair(13, identity=jnp.inf)
    gV, gB = jax.grad(_ell_loss, argnums=(0, 1))(
        values, B, indices, TROPICAL_MIN_PLUS, 4,
    )
    gV_fd = _finite_diff(
        lambda v: _ell_loss(v, B, indices, TROPICAL_MIN_PLUS, 4),
        values, eps=1e-4,
    )
    gB_fd = _finite_diff(
        lambda x: _ell_loss(values, x, indices, TROPICAL_MIN_PLUS, 4),
        B, eps=1e-4,
    )
    np.testing.assert_allclose(gV, gV_fd, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(gB, gB_fd, atol=1e-6, rtol=1e-6)


def test_euclidean_ell_matmul_backward():
    values, indices, B = _ell_pair(14)
    gV, gB = jax.grad(_ell_loss, argnums=(0, 1))(
        values, B, indices, EUCLIDEAN, 4,
    )
    gV_fd = _finite_diff(
        lambda v: _ell_loss(v, B, indices, EUCLIDEAN, 4), values,
    )
    gB_fd = _finite_diff(
        lambda x: _ell_loss(values, x, indices, EUCLIDEAN, 4), B,
    )
    np.testing.assert_allclose(gV, gV_fd, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(gB, gB_fd, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Composition tests: forward output is unchanged when we route through
# custom_vjp.  Regression-guards a mis-wired forward function.
# ---------------------------------------------------------------------------


def test_forward_value_unchanged_by_custom_vjp_wrapping():
    A, B = _matmul_pair(20)
    for sr in [REAL, LOG, TROPICAL_MAX_PLUS, TROPICAL_MIN_PLUS, EUCLIDEAN]:
        with_grad = semiring_matmul(A, B, semiring=sr, backend='jax')
        # Build a forward-only Semiring (matmul_vjp=None) by re-tagging.
        import dataclasses
        fwd_only = dataclasses.replace(sr, matmul_vjp=None)
        without_grad = semiring_matmul(A, B, semiring=fwd_only, backend='jax')
        np.testing.assert_array_equal(with_grad, without_grad)


def test_batched_grad_real():
    # Backwards must compose with the batch-vmap.  Two-batch case.
    keyA, keyB = jax.random.split(jax.random.key(30))
    A = jax.random.normal(keyA, (3, 4, 4))
    B = jax.random.normal(keyB, (3, 4, 4))
    def loss(A, B):
        return semiring_matmul(A, B, semiring=REAL, backend='jax').sum()
    gA, gB = jax.grad(loss, argnums=(0, 1))(A, B)
    # Reference: same as un-batched.
    gA_ref = jnp.stack([jnp.ones((4, 4)) @ B[i].T for i in range(3)])
    np.testing.assert_allclose(gA, gA_ref, atol=1e-12)
