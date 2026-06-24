# -*- coding: utf-8 -*-
"""Backend-parity, identity-propagation, and numerical-stability tests
for ``nitrix.semiring.semiring_matmul``.

Per SPEC §10 and SPEC §10, these are the load-bearing tests
behind the G1 backward-kernel gate (forward correctness only at this
phase; backward kernels are 2.A.5).  Each test asserts:

1. Pallas Triton output matches the JAX reference exactly (no
   precision drop).
2. The JAX reference matches the naive broadcast formulation at the
   pinned tolerance.
3. Algebraic invariants hold under adversarial inputs (``-inf``
   identity, large magnitudes).

The Pallas path is exercised only when the host has an Ampere+ NVIDIA
GPU; the JAX path is exercised unconditionally.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.special
import numpy as np
import pytest

from nitrix._internal.backend import _HAS_AMPERE_NVIDIA
from nitrix.semiring import (
    BOOLEAN,
    EUCLIDEAN,
    LOG,
    REAL,
    TROPICAL_MAX_PLUS,
    TROPICAL_MIN_PLUS,
    reference_semiring_matmul,
    semiring_matmul,
)

pallas_only = pytest.mark.skipif(
    not _HAS_AMPERE_NVIDIA,
    reason='requires NVIDIA Ampere+ for Pallas Triton backend',
)


# ---------------------------------------------------------------------------
# Naive broadcast references (independent of our Semiring code).
# ---------------------------------------------------------------------------


def naive_real(A, B):
    return (A[:, :, None] * B[None, :, :]).sum(axis=1)


def naive_log(A, B):
    return jax.scipy.special.logsumexp(A[:, :, None] + B[None, :, :], axis=1)


def naive_tropical_max(A, B):
    return (A[:, :, None] + B[None, :, :]).max(axis=1)


def naive_tropical_min(A, B):
    return (A[:, :, None] + B[None, :, :]).min(axis=1)


def naive_euclidean(A, B):
    d = A[:, :, None] - B[None, :, :]
    return jnp.sqrt((d * d).sum(axis=1))


def naive_boolean(A, B):
    return jnp.any(A[:, :, None] & B[None, :, :], axis=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


SHAPES = [
    # (m, k, n) — all divisible by 16/32 for Pallas tile sweet spots.
    (32, 32, 32),
    (64, 64, 32),
    (128, 64, 64),
]


def _pair(key, m, k, n, dtype=jnp.float32, scale=1.0):
    ka, kb = jax.random.split(key)
    A = jax.random.normal(ka, (m, k), dtype=dtype) * scale
    B = jax.random.normal(kb, (k, n), dtype=dtype) * scale
    return A, B


# ---------------------------------------------------------------------------
# Forward correctness, all six algebras, JAX path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('shape', SHAPES)
def test_real_jax_matches_naive(shape):
    A, B = _pair(jax.random.key(0), *shape)
    got = reference_semiring_matmul(A, B, semiring=REAL)
    np.testing.assert_allclose(got, naive_real(A, B), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('shape', SHAPES)
def test_log_jax_matches_naive(shape):
    A, B = _pair(jax.random.key(1), *shape)
    got = reference_semiring_matmul(A, B, semiring=LOG)
    np.testing.assert_allclose(got, naive_log(A, B), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('shape', SHAPES)
def test_tropical_max_jax_matches_naive(shape):
    A, B = _pair(jax.random.key(2), *shape)
    got = reference_semiring_matmul(A, B, semiring=TROPICAL_MAX_PLUS)
    np.testing.assert_allclose(got, naive_tropical_max(A, B), atol=1e-5)


@pytest.mark.parametrize('shape', SHAPES)
def test_tropical_min_jax_matches_naive(shape):
    A, B = _pair(jax.random.key(3), *shape)
    got = reference_semiring_matmul(A, B, semiring=TROPICAL_MIN_PLUS)
    np.testing.assert_allclose(got, naive_tropical_min(A, B), atol=1e-5)


@pytest.mark.parametrize('shape', SHAPES)
def test_euclidean_jax_matches_naive(shape):
    A, B = _pair(jax.random.key(4), *shape)
    got = reference_semiring_matmul(A, B, semiring=EUCLIDEAN)
    np.testing.assert_allclose(
        got, naive_euclidean(A, B), atol=1e-4, rtol=1e-4
    )


def test_boolean_jax_matches_naive():
    A, B = _pair(jax.random.key(5), 32, 32, 32)
    Ab = A > 0
    Bb = B > 0
    got = reference_semiring_matmul(Ab, Bb, semiring=BOOLEAN)
    np.testing.assert_array_equal(got, naive_boolean(Ab, Bb))


# ---------------------------------------------------------------------------
# Backend parity: Pallas vs JAX exact equality at fp32.
# ---------------------------------------------------------------------------


@pallas_only
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize(
    'algebra',
    [REAL, LOG, TROPICAL_MAX_PLUS, TROPICAL_MIN_PLUS, EUCLIDEAN],
    ids=lambda s: s.name,
)
def test_pallas_matches_jax(shape, algebra):
    A, B = _pair(jax.random.key(10), *shape)
    out_pallas = semiring_matmul(A, B, semiring=algebra, backend='pallas-cuda')
    out_jax = semiring_matmul(A, B, semiring=algebra, backend='jax')
    # Both walk the K loop in the same order with the same monoid; expect
    # bitwise-equal results at fp32.
    np.testing.assert_array_equal(out_pallas, out_jax)


@pallas_only
def test_pallas_boolean_matches_jax():
    A, B = _pair(jax.random.key(11), 32, 32, 32)
    Ab, Bb = (A > 0).astype(jnp.bool_), (B > 0).astype(jnp.bool_)
    out_pallas = semiring_matmul(
        Ab, Bb, semiring=BOOLEAN, backend='pallas-cuda'
    )
    out_jax = semiring_matmul(Ab, Bb, semiring=BOOLEAN, backend='jax')
    np.testing.assert_array_equal(out_pallas, out_jax)


# ---------------------------------------------------------------------------
# Identity propagation -- the easiest way to silently break a streaming
# kernel is to forget the algebra's identity.
# ---------------------------------------------------------------------------


def test_tropical_max_neg_inf_propagates_jax():
    m, k, n = 16, 16, 16
    A = jnp.ones((m, k), jnp.float32)
    A = A.at[3].set(-jnp.inf)
    B = jax.random.normal(jax.random.key(20), (k, n), jnp.float32)
    out = reference_semiring_matmul(A, B, semiring=TROPICAL_MAX_PLUS)
    assert bool(jnp.all(jnp.isneginf(out[3])))
    expected = naive_tropical_max(A, B)
    np.testing.assert_allclose(out[:3], expected[:3], atol=1e-5)


@pallas_only
def test_tropical_max_neg_inf_propagates_pallas():
    m, k, n = 32, 32, 32
    A = jnp.ones((m, k), jnp.float32)
    A = A.at[3].set(-jnp.inf)
    B = jax.random.normal(jax.random.key(21), (k, n), jnp.float32)
    out = semiring_matmul(
        A,
        B,
        semiring=TROPICAL_MAX_PLUS,
        backend='pallas-cuda',
    )
    assert bool(jnp.all(jnp.isneginf(out[3])))
    assert bool(~jnp.any(jnp.isnan(out)))


def test_log_neg_inf_does_not_nan_jax():
    m, k, n = 16, 16, 16
    A = jnp.zeros((m, k), jnp.float32)
    A = A.at[5].set(-jnp.inf)
    B = jnp.zeros((k, n), jnp.float32)
    out = reference_semiring_matmul(A, B, semiring=LOG)
    assert bool(~jnp.any(jnp.isnan(out)))
    assert bool(jnp.all(jnp.isneginf(out[5])))


@pallas_only
def test_log_neg_inf_does_not_nan_pallas():
    m, k, n = 32, 32, 32
    A = jnp.zeros((m, k), jnp.float32)
    A = A.at[5].set(-jnp.inf)
    B = jnp.zeros((k, n), jnp.float32)
    out = semiring_matmul(A, B, semiring=LOG, backend='pallas-cuda')
    assert bool(~jnp.any(jnp.isnan(out)))
    assert bool(jnp.all(jnp.isneginf(out[5])))


# ---------------------------------------------------------------------------
# Numerical stability under adversarial magnitudes.
# ---------------------------------------------------------------------------


def test_log_large_magnitudes_finite_jax():
    A, B = _pair(jax.random.key(30), 32, 32, 32, scale=50.0)
    out = reference_semiring_matmul(A, B, semiring=LOG)
    assert bool(jnp.all(jnp.isfinite(out)))
    np.testing.assert_allclose(out, naive_log(A, B), atol=1e-3, rtol=1e-4)


@pallas_only
def test_log_large_magnitudes_finite_pallas():
    A, B = _pair(jax.random.key(31), 32, 32, 32, scale=50.0)
    out = semiring_matmul(A, B, semiring=LOG, backend='pallas-cuda')
    assert bool(jnp.all(jnp.isfinite(out)))


def test_euclidean_zero_distance():
    m, k, n = 16, 16, 16
    A = jax.random.normal(jax.random.key(40), (m, k))
    B = jax.random.normal(jax.random.key(41), (k, n))
    # Make column 0 of B equal to row 2 of A (treated as a length-k vector).
    B = B.at[:, 0].set(A[2])
    out = reference_semiring_matmul(A, B, semiring=EUCLIDEAN)
    np.testing.assert_allclose(out[2, 0], 0.0, atol=1e-5)
