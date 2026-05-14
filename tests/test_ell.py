# -*- coding: utf-8 -*-
"""Tests for ``nitrix.sparse.ell`` and ``nitrix.semiring.semiring_ell_matmul``.

Backend parity for ELL on the current JAX pin is one-sided: the Pallas
Triton ELL kernel raises ``PallasELLNotTileable`` unconditionally
because Triton does not lower the ``gather`` primitive (see
``bench/G0_ELL_REPORT.md``).  The tests assert the public dispatcher
falls back to JAX and emits exactly one ``NitrixBackendFallback``
warning per (shape, dtype) signature.
"""
from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nitrix._internal.backend import (
    NitrixBackendFallback,
    _HAS_AMPERE_NVIDIA,
    reset_fallback_state,
)
from nitrix.semiring import (
    LOG,
    REAL,
    TROPICAL_MAX_PLUS,
    reference_semiring_ell_matmul,
    semiring_ell_matmul,
)
from nitrix.sparse import ELL, ell_from_dense, ell_pad, ell_to_dense


pallas_only = pytest.mark.skipif(
    not _HAS_AMPERE_NVIDIA,
    reason='requires NVIDIA Ampere+ for Pallas Triton backend',
)


def _dense_pair(key, m, n, dtype=jnp.float32):
    ka, kb = jax.random.split(key)
    A = jax.random.normal(ka, (m, n), dtype=dtype)
    B = jax.random.normal(kb, (n, n), dtype=dtype)
    return A, B


# ---------------------------------------------------------------------------
# Format primitives
# ---------------------------------------------------------------------------


def test_ell_from_dense_round_trip_real():
    A = jax.random.normal(jax.random.key(0), (16, 16))
    ell = ell_from_dense(A, k_max=16)
    assert ell.shape == (16, 16)
    assert ell.k_max == 16
    # Scatter-add round-trip
    reconstructed = ell_to_dense(ell)
    np.testing.assert_allclose(reconstructed, A, atol=1e-6)


def test_ell_pad_appends_identity():
    m, k_actual, k_max, n_cols = 4, 3, 6, 8
    values = jnp.arange(m * k_actual, dtype=jnp.float32).reshape(m, k_actual)
    indices = jnp.zeros((m, k_actual), dtype=jnp.int32)
    ell = ell_pad(
        values, indices,
        k_max=k_max, n_cols=n_cols, identity=0.0, pad_index=0,
    )
    assert ell.values.shape == (m, k_max)
    # Tail is padded with identity
    np.testing.assert_array_equal(ell.values[:, k_actual:], 0.0)
    # Original entries preserved
    np.testing.assert_array_equal(ell.values[:, :k_actual], values)


def test_ell_pad_rejects_oversize():
    with pytest.raises(ValueError, match='k_actual'):
        ell_pad(
            jnp.zeros((2, 4)),
            jnp.zeros((2, 4), jnp.int32),
            k_max=3,
            n_cols=10,
        )


# ---------------------------------------------------------------------------
# Reference correctness across algebras
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('algebra,identity,naive_fn', [
    (REAL, 0.0, lambda A, B: A @ B),
    (
        LOG, -jnp.inf,
        lambda A, B: jax.scipy.special.logsumexp(
            A[:, :, None] + B[None, :, :], axis=1,
        ),
    ),
    (
        TROPICAL_MAX_PLUS, -jnp.inf,
        lambda A, B: (A[:, :, None] + B[None, :, :]).max(axis=1),
    ),
], ids=lambda a: getattr(a, 'name', str(a)))
def test_ell_matches_dense(algebra, identity, naive_fn):
    A, B = _dense_pair(jax.random.key(5), 16, 16)
    # Full ELL == dense A.
    ell = ell_from_dense(A, k_max=16, identity=identity)
    out = reference_semiring_ell_matmul(
        ell.values, ell.indices, B, semiring=algebra, n_cols=16,
    )
    if algebra is REAL:
        ref = jnp.matmul(A, B, precision='highest')
        np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)
    else:
        np.testing.assert_allclose(out, naive_fn(A, B), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Pallas fallback observability
# ---------------------------------------------------------------------------


@pallas_only
def test_ell_pallas_falls_back_with_warning():
    reset_fallback_state()
    A, B = _dense_pair(jax.random.key(50), 16, 16)
    ell = ell_from_dense(A, k_max=8)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        out = semiring_ell_matmul(
            ell.values, ell.indices, B,
            semiring=REAL, n_cols=16, backend='pallas-cuda',
        )
        n_fallback = sum(
            1 for w in ws if w.category is NitrixBackendFallback
        )
    # Pallas ELL kernel raises NotTileable on current JAX -> exactly one warning.
    assert n_fallback == 1
    # The output still matches the JAX path.
    out_jax = semiring_ell_matmul(
        ell.values, ell.indices, B,
        semiring=REAL, n_cols=16, backend='jax',
    )
    np.testing.assert_array_equal(out, out_jax)


@pallas_only
def test_ell_pallas_fallback_dedupes():
    reset_fallback_state()
    A, B = _dense_pair(jax.random.key(51), 16, 16)
    ell = ell_from_dense(A, k_max=8)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        for _ in range(3):
            semiring_ell_matmul(
                ell.values, ell.indices, B,
                semiring=REAL, n_cols=16, backend='pallas-cuda',
            )
        n = sum(1 for w in ws if w.category is NitrixBackendFallback)
    assert n == 1


# ---------------------------------------------------------------------------
# Batched ELL (vmap)
# ---------------------------------------------------------------------------


def test_ell_batched_jax():
    m, n = 8, 8
    key = jax.random.key(60)
    A0 = jax.random.normal(jax.random.fold_in(key, 0), (m, n))
    A1 = jax.random.normal(jax.random.fold_in(key, 1), (m, n))
    B0 = jax.random.normal(jax.random.fold_in(key, 2), (n, n))
    B1 = jax.random.normal(jax.random.fold_in(key, 3), (n, n))

    ell0 = ell_from_dense(A0, k_max=n)
    ell1 = ell_from_dense(A1, k_max=n)
    values = jnp.stack([ell0.values, ell1.values])
    indices = jnp.stack([ell0.indices, ell1.indices])
    Bs = jnp.stack([B0, B1])

    out = semiring_ell_matmul(
        values, indices, Bs, semiring=REAL, n_cols=n, backend='jax',
    )
    np.testing.assert_allclose(
        out[0], jnp.matmul(A0, B0, precision='highest'),
        atol=1e-4, rtol=1e-4,
    )
    np.testing.assert_allclose(
        out[1], jnp.matmul(A1, B1, precision='highest'),
        atol=1e-4, rtol=1e-4,
    )
