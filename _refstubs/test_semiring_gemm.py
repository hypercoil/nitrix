'''Correctness tests for ``semiring_gemm``.

Each test checks the Pallas kernel against *two* independent ground truths:

1. A direct broadcasting formulation in pure JAX (no semiring abstraction).
   This guards against bugs in our ``Monoid`` / ``Semigroup`` glue.
2. ``reference_semiring_gemm`` -- exercises the same algebra but without the
   tiled accumulation. This guards against tile-boundary bugs in the kernel.

Tests run under ``interpret=True`` so they do not require a GPU.
'''

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from semiring_gemm import (
    euclidean_semiring,
    log_semiring,
    make_gemm_kernel,
    real_semiring,
    reference_semiring_gemm,
    tropical_max_plus_semiring,
    tropical_min_plus_semiring,
)


jax.config.update('jax_enable_x64', True)


# -----------------------------------------------------------------------------
# Direct broadcasting formulations (independent of our Semiring code).
# -----------------------------------------------------------------------------


def naive_real(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return (a[:, :, None] * b[None, :, :]).sum(axis=1)


def naive_tropical_max(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return (a[:, :, None] + b[None, :, :]).max(axis=1)


def naive_tropical_min(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return (a[:, :, None] + b[None, :, :]).min(axis=1)


def naive_log(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jax.scipy.special.logsumexp(
        a[:, :, None] + b[None, :, :], axis=1
    )


def naive_euclidean(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    diff = a[:, :, None] - b[None, :, :]
    return jnp.sqrt((diff * diff).sum(axis=1))


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


SHAPES = [
    # (m, p, n, block_m, block_k, block_n)
    (32, 32, 32, 16, 16, 16),
    (64, 32, 64, 32, 16, 32),
    (128, 64, 96, 64, 32, 32),
    (48, 80, 64, 24, 16, 32),
]


def _random_pair(
    key: jax.Array, m: int, p: int, n: int, dtype=jnp.float32
) -> tuple[jnp.ndarray, jnp.ndarray]:
    k_a, k_b = jax.random.split(key)
    a = jax.random.normal(k_a, (m, p), dtype=dtype)
    b = jax.random.normal(k_b, (p, n), dtype=dtype)
    return a, b


# -----------------------------------------------------------------------------
# Real semiring (sanity check vs. jnp.matmul).
# -----------------------------------------------------------------------------


@pytest.mark.parametrize('shape', SHAPES)
def test_real_semiring_matches_matmul(shape):
    m, p, n, bm, bk, bn = shape
    a, b = _random_pair(jax.random.key(0), m, p, n, dtype=jnp.float32)

    gemm = make_gemm_kernel(
        real_semiring(), m, n, p,
        block_m=bm, block_n=bn, block_k=bk, interpret=True,
    )
    got = gemm(a, b)

    np.testing.assert_allclose(got, naive_real(a, b), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(
        got, reference_semiring_gemm(real_semiring(), a, b),
        atol=1e-4, rtol=1e-4,
    )
    np.testing.assert_allclose(got, a @ b, atol=1e-4, rtol=1e-4)


# -----------------------------------------------------------------------------
# Tropical (max-plus, min-plus).
# -----------------------------------------------------------------------------


@pytest.mark.parametrize('shape', SHAPES)
def test_tropical_max_plus(shape):
    m, p, n, bm, bk, bn = shape
    a, b = _random_pair(jax.random.key(1), m, p, n)

    gemm = make_gemm_kernel(
        tropical_max_plus_semiring(), m, n, p,
        block_m=bm, block_n=bn, block_k=bk, interpret=True,
    )
    got = gemm(a, b)

    np.testing.assert_allclose(got, naive_tropical_max(a, b), atol=1e-5)
    np.testing.assert_allclose(
        got, reference_semiring_gemm(tropical_max_plus_semiring(), a, b),
        atol=1e-5,
    )


@pytest.mark.parametrize('shape', SHAPES)
def test_tropical_min_plus(shape):
    m, p, n, bm, bk, bn = shape
    a, b = _random_pair(jax.random.key(2), m, p, n)

    gemm = make_gemm_kernel(
        tropical_min_plus_semiring(), m, n, p,
        block_m=bm, block_n=bn, block_k=bk, interpret=True,
    )
    got = gemm(a, b)

    np.testing.assert_allclose(got, naive_tropical_min(a, b), atol=1e-5)
    np.testing.assert_allclose(
        got, reference_semiring_gemm(tropical_min_plus_semiring(), a, b),
        atol=1e-5,
    )


def test_tropical_max_plus_negative_inf_identity():
    '''The additive identity of max-plus is -inf; rows of -inf in A
    should propagate to rows of -inf in C.'''
    m, p, n = 16, 16, 16
    a = jnp.ones((m, p), dtype=jnp.float32)
    a = a.at[3].set(-jnp.inf)  # row 3 is -inf everywhere
    b = jax.random.normal(jax.random.key(11), (p, n))

    gemm = make_gemm_kernel(
        tropical_max_plus_semiring(), m, n, p,
        block_m=8, block_n=8, block_k=8, interpret=True,
    )
    got = gemm(a, b)

    assert jnp.all(jnp.isneginf(got[3]))
    # Other rows should match the naive formulation.
    expected = naive_tropical_max(a, b)
    np.testing.assert_allclose(got[:3], expected[:3], atol=1e-5)
    np.testing.assert_allclose(got[4:], expected[4:], atol=1e-5)


# -----------------------------------------------------------------------------
# Log semiring (logsumexp).  Stresses the auxiliary-state pathway.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize('shape', SHAPES)
def test_log_semiring(shape):
    m, p, n, bm, bk, bn = shape
    a, b = _random_pair(jax.random.key(3), m, p, n)

    gemm = make_gemm_kernel(
        log_semiring(), m, n, p,
        block_m=bm, block_n=bn, block_k=bk, interpret=True,
    )
    got = gemm(a, b)

    # logsumexp tolerance is a touch looser due to compounded exp/log.
    np.testing.assert_allclose(got, naive_log(a, b), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(
        got, reference_semiring_gemm(log_semiring(), a, b),
        atol=1e-4, rtol=1e-4,
    )


def test_log_semiring_large_magnitudes():
    '''Numerical stability: values with very large magnitudes that would
    overflow if naively exponentiated should still produce finite results,
    matching the running-max/running-sum reference.'''
    m, p, n = 32, 32, 32
    key = jax.random.key(99)
    a = jax.random.normal(key, (m, p)) * 50.0  # huge magnitudes
    b = jax.random.normal(jax.random.fold_in(key, 1), (p, n)) * 50.0

    gemm = make_gemm_kernel(
        log_semiring(), m, n, p,
        block_m=16, block_n=16, block_k=16, interpret=True,
    )
    got = gemm(a, b)

    assert jnp.all(jnp.isfinite(got))
    np.testing.assert_allclose(got, naive_log(a, b), atol=1e-3, rtol=1e-4)


def test_log_semiring_negative_inf_identity():
    '''-inf is the additive identity of the log semiring; mixing it with
    finite values must not produce NaN.'''
    m, p, n = 16, 16, 16
    a = jnp.zeros((m, p), dtype=jnp.float32)
    a = a.at[5].set(-jnp.inf)
    b = jnp.zeros((p, n), dtype=jnp.float32)

    gemm = make_gemm_kernel(
        log_semiring(), m, n, p,
        block_m=8, block_n=8, block_k=8, interpret=True,
    )
    got = gemm(a, b)

    assert not jnp.any(jnp.isnan(got))
    # Row 5 of a is all -inf, so row 5 of the output should be -inf
    # (logsumexp of all -inf entries).
    assert jnp.all(jnp.isneginf(got[5]))


# -----------------------------------------------------------------------------
# Euclidean semiring.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize('shape', SHAPES)
def test_euclidean_semiring(shape):
    m, p, n, bm, bk, bn = shape
    a, b = _random_pair(jax.random.key(4), m, p, n)

    gemm = make_gemm_kernel(
        euclidean_semiring(), m, n, p,
        block_m=bm, block_n=bn, block_k=bk, interpret=True,
    )
    got = gemm(a, b)

    np.testing.assert_allclose(got, naive_euclidean(a, b), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(
        got, reference_semiring_gemm(euclidean_semiring(), a, b),
        atol=1e-4, rtol=1e-4,
    )


def test_euclidean_semiring_zero_distance():
    '''A column of B equal to a row of A should yield 0 distance.'''
    m, p, n = 16, 16, 16
    a = jax.random.normal(jax.random.key(7), (m, p))
    # Build B so that column 0 equals (row 2 of A) transposed-as-vector.
    b = jax.random.normal(jax.random.key(8), (p, n))
    b = b.at[:, 0].set(a[2])

    gemm = make_gemm_kernel(
        euclidean_semiring(), m, n, p,
        block_m=8, block_n=8, block_k=8, interpret=True,
    )
    got = gemm(a, b)
    np.testing.assert_allclose(got[2, 0], 0.0, atol=1e-5)


# -----------------------------------------------------------------------------
# Builder validation.
# -----------------------------------------------------------------------------


def test_builder_rejects_indivisible_blocks():
    with pytest.raises(ValueError, match='block_m'):
        make_gemm_kernel(real_semiring(), 33, 32, 32, block_m=16)
    with pytest.raises(ValueError, match='block_n'):
        make_gemm_kernel(real_semiring(), 32, 33, 32, block_n=16)
    with pytest.raises(ValueError, match='block_k'):
        make_gemm_kernel(real_semiring(), 32, 32, 33, block_k=16)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
