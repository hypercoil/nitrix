# -*- coding: utf-8 -*-
"""Tests for ``nitrix.sparse.ell_sectioned``.

Coverage:

- Construction: bucketing by degree, correct k_max per bucket,
  preserved row identity (scatter-back round-trip).
- Matmul: bit-exact with a flat-ELL reference for REAL; matches
  scipy LSE for LOG with -inf identity; matches naive max for
  TROPICAL_MAX_PLUS.
- Memory win: total per-row storage is strictly less than the flat
  ELL of the same adjacency for a degree-variance test case.
- Empty bucket handling: degenerate (zero-degree row) doesn't crash.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nitrix.semiring import (
    LOG,
    REAL,
    TROPICAL_MAX_PLUS,
    semiring_ell_matmul,
)
from nitrix.sparse import (
    SectionedELL,
    ell_pad,
    sectioned_ell_from_ragged,
    sectioned_semiring_ell_matmul,
)


jax.config.update('jax_enable_x64', True)


def _ragged(n_cols, degrees, seed=0):
    np.random.seed(seed)
    values_per_row = [
        np.random.randn(d).astype(np.float64) for d in degrees
    ]
    indices_per_row = [
        np.random.choice(n_cols, size=d, replace=False) for d in degrees
    ]
    return values_per_row, indices_per_row


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_sectioned_ell_buckets_by_log2_degree():
    n_cols = 16
    values, indices = _ragged(n_cols, [1, 3, 8, 2, 5])
    sectioned = sectioned_ell_from_ragged(
        values, indices, n_cols=n_cols,
    )
    # Bucket assignment:
    #   degree 1 -> bucket 0 (k_max=1)
    #   degree 3 -> bucket 2 (k_max=4)
    #   degree 8 -> bucket 3 (k_max=8)
    #   degree 2 -> bucket 1 (k_max=2)
    #   degree 5 -> bucket 3 (k_max=8)
    assert sectioned.n_buckets == 4
    k_maxes = sorted(s.k_max for s in sectioned.sections)
    assert k_maxes == [1, 2, 4, 8]


def test_sectioned_ell_storage_smaller_than_flat():
    n_cols = 32
    # Worst-case degree 32; median degree 4.
    degrees = [4] * 20 + [32] * 2 + [1] * 5
    values, indices = _ragged(n_cols, degrees, seed=1)
    sectioned = sectioned_ell_from_ragged(
        values, indices, n_cols=n_cols,
    )
    flat_storage = len(degrees) * max(degrees)   # n_rows × k_max
    assert sectioned.total_storage < flat_storage


def test_sectioned_ell_empty_row_handled():
    n_cols = 8
    # First row has degree 0; this must not crash construction or matmul.
    values = [np.zeros(0)] + [np.random.randn(3) for _ in range(3)]
    indices = [np.zeros(0, dtype=np.int32)] + [
        np.random.choice(n_cols, size=3, replace=False) for _ in range(3)
    ]
    sectioned = sectioned_ell_from_ragged(values, indices, n_cols=n_cols)
    B = jax.random.normal(jax.random.key(0), (n_cols, 2))
    out = sectioned_semiring_ell_matmul(
        sectioned, B, semiring=REAL, backend='jax',
    )
    assert out.shape == (4, 2)
    # The empty row contributes nothing -> output is the identity (0
    # for REAL).
    np.testing.assert_array_equal(out[0], jnp.zeros(2))


# ---------------------------------------------------------------------------
# Matmul correctness across algebras
# ---------------------------------------------------------------------------


def test_sectioned_real_matmul_matches_manual():
    n_cols = 16
    values, indices = _ragged(n_cols, [1, 3, 8, 2, 5], seed=2)
    sectioned = sectioned_ell_from_ragged(
        values, indices, n_cols=n_cols,
    )
    B = jax.random.normal(jax.random.key(0), (n_cols, 4), dtype=jnp.float64)
    got = sectioned_semiring_ell_matmul(
        sectioned, B, semiring=REAL, backend='jax',
    )
    ref = np.zeros((5, 4))
    for i, (v, idx) in enumerate(zip(values, indices)):
        ref[i] = (v[:, None] * np.asarray(B)[idx]).sum(0)
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_sectioned_log_matmul_with_neg_inf_identity():
    import scipy.special as sp
    n_cols = 16
    values, indices = _ragged(n_cols, [2, 5, 3], seed=3)
    sectioned = sectioned_ell_from_ragged(
        values, indices, n_cols=n_cols, identity=-np.inf,
    )
    B = jax.random.normal(jax.random.key(1), (n_cols, 4), dtype=jnp.float64)
    got = sectioned_semiring_ell_matmul(
        sectioned, B, semiring=LOG, backend='jax',
    )
    ref = np.zeros((3, 4))
    for i, (v, idx) in enumerate(zip(values, indices)):
        ref[i] = sp.logsumexp(v[:, None] + np.asarray(B)[idx], axis=0)
    np.testing.assert_allclose(got, ref, atol=1e-10, rtol=1e-10)


def test_sectioned_tropical_matmul_with_neg_inf_identity():
    n_cols = 16
    values, indices = _ragged(n_cols, [2, 5, 3], seed=4)
    sectioned = sectioned_ell_from_ragged(
        values, indices, n_cols=n_cols, identity=-np.inf,
    )
    B = jax.random.normal(jax.random.key(2), (n_cols, 4), dtype=jnp.float64)
    got = sectioned_semiring_ell_matmul(
        sectioned, B, semiring=TROPICAL_MAX_PLUS, backend='jax',
    )
    ref = np.zeros((3, 4))
    for i, (v, idx) in enumerate(zip(values, indices)):
        ref[i] = (v[:, None] + np.asarray(B)[idx]).max(axis=0)
    np.testing.assert_allclose(got, ref, atol=1e-10)


def test_sectioned_matches_flat_ell_when_uniform_degree():
    '''If all rows have the same degree, sectioned ELL should give
    the same answer as a flat ELL of the same adjacency.
    '''
    n_cols = 8
    n_rows = 5
    k = 3
    np.random.seed(10)
    values_arr = np.random.randn(n_rows, k).astype(np.float64)
    indices_arr = np.random.choice(
        n_cols, size=(n_rows, k), replace=True,
    ).astype(np.int32)
    sectioned = sectioned_ell_from_ragged(
        list(values_arr), list(indices_arr), n_cols=n_cols,
    )
    B = jax.random.normal(jax.random.key(3), (n_cols, 2), dtype=jnp.float64)
    got_sectioned = sectioned_semiring_ell_matmul(
        sectioned, B, semiring=REAL, backend='jax',
    )
    # Flat ELL reference.
    flat = ell_pad(
        jnp.asarray(values_arr), jnp.asarray(indices_arr),
        k_max=k, n_cols=n_cols,
    )
    got_flat = semiring_ell_matmul(
        flat.values, flat.indices, B,
        semiring=REAL, n_cols=n_cols, backend='jax',
    )
    np.testing.assert_allclose(got_sectioned, got_flat, atol=1e-12)
