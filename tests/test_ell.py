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
    _HAS_AMPERE_NVIDIA,
    NitrixBackendFallback,
    reset_fallback_state,
)
from nitrix.semiring import (
    LOG,
    REAL,
    TROPICAL_MAX_PLUS,
    reference_semiring_ell_matmul,
    reference_semiring_ell_rmatvec,
    semiring_ell_matmul,
    semiring_ell_rmatvec,
)
from nitrix.sparse import (
    ELL,
    ell_add_self_loops,
    ell_from_dense,
    ell_pad,
    ell_to_dense,
)

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
        values,
        indices,
        k_max=k_max,
        n_cols=n_cols,
        identity=0.0,
        pad_index=0,
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
# ell_add_self_loops
# ---------------------------------------------------------------------------


def _self_loop_fixture():
    # Row 0: two real edges; row 1: one real (3.) + one pad (0. = identity);
    # row 2: all pad.  identity = 0.0 (REAL).
    values = jnp.asarray([[1.0, 2.0], [3.0, 0.0], [0.0, 0.0]])
    indices = jnp.asarray([[1, 2], [0, 0], [0, 0]], dtype=jnp.int32)
    ell = ELL(values=values, indices=indices, n_cols=3, identity=0.0)
    edge_attr = jnp.asarray(
        [
            [[10.0, 11.0], [20.0, 21.0]],  # row 0: both valid
            [[30.0, 31.0], [99.0, 99.0]],  # row 1: slot 1 is pad
            [[88.0, 88.0], [77.0, 77.0]],  # row 2: all pad
        ]
    )
    return ell, edge_attr


def test_ell_add_self_loops_appends_self_edge():
    ell, _ = _self_loop_fixture()
    out, attr = ell_add_self_loops(ell)
    assert attr is None
    assert out.indices.shape == (3, 3)
    assert out.values.shape == (3, 3)
    # Self slot points at the row itself, with a non-identity marker value.
    np.testing.assert_array_equal(out.indices[:, -1], jnp.arange(3))
    np.testing.assert_array_equal(out.values[:, -1], 1.0)
    # Existing slots untouched.
    np.testing.assert_array_equal(out.values[:, :2], ell.values)
    np.testing.assert_array_equal(out.indices[:, :2], ell.indices)


def test_ell_add_self_loops_mean_fill_excludes_padding():
    ell, edge_attr = _self_loop_fixture()
    out, attr = ell_add_self_loops(ell, edge_attr, fill='mean')
    assert attr.shape == (3, 3, 2)
    # Original attributes preserved.
    np.testing.assert_array_equal(attr[:, :2, :], edge_attr)
    # Self-edge attr = mean over *valid* (non-pad) edges only.
    np.testing.assert_allclose(attr[0, -1], [15.0, 16.0])  # mean of both
    np.testing.assert_allclose(attr[1, -1], [30.0, 31.0])  # pad slot dropped
    np.testing.assert_allclose(attr[2, -1], [0.0, 0.0])  # all-pad -> 0


def test_ell_add_self_loops_add_and_zero_fill():
    ell, edge_attr = _self_loop_fixture()
    _, attr_add = ell_add_self_loops(ell, edge_attr, fill='add')
    np.testing.assert_allclose(attr_add[0, -1], [30.0, 32.0])
    np.testing.assert_allclose(attr_add[1, -1], [30.0, 31.0])
    _, attr_zero = ell_add_self_loops(ell, edge_attr, fill='zero')
    np.testing.assert_array_equal(attr_zero[:, -1, :], 0.0)


def test_ell_add_self_loops_rejects_misaligned_edge_attr():
    ell, _ = _self_loop_fixture()
    with pytest.raises(ValueError, match='edge_attr'):
        ell_add_self_loops(ell, jnp.zeros((3, 5, 2)))  # k_max mismatch


def test_ell_add_self_loops_is_jit_safe():
    ell, edge_attr = _self_loop_fixture()

    @jax.jit
    def run(values, indices, attr):
        e = ELL(values=values, indices=indices, n_cols=3, identity=0.0)
        out, out_attr = ell_add_self_loops(e, attr, fill='mean')
        return out.values, out.indices, out_attr

    values, indices, attr = run(ell.values, ell.indices, edge_attr)
    assert values.shape == (3, 3)
    np.testing.assert_allclose(attr[0, -1], [15.0, 16.0])


# ---------------------------------------------------------------------------
# Reference correctness across algebras
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'algebra,identity,naive_fn',
    [
        (REAL, 0.0, lambda A, B: A @ B),
        (
            LOG,
            -jnp.inf,
            lambda A, B: jax.scipy.special.logsumexp(
                A[:, :, None] + B[None, :, :],
                axis=1,
            ),
        ),
        (
            TROPICAL_MAX_PLUS,
            -jnp.inf,
            lambda A, B: (A[:, :, None] + B[None, :, :]).max(axis=1),
        ),
    ],
    ids=lambda a: getattr(a, 'name', str(a)),
)
def test_ell_matches_dense(algebra, identity, naive_fn):
    A, B = _dense_pair(jax.random.key(5), 16, 16)
    # Full ELL == dense A.
    ell = ell_from_dense(A, k_max=16, identity=identity)
    out = reference_semiring_ell_matmul(
        ell.values,
        ell.indices,
        B,
        semiring=algebra,
        n_cols=16,
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
            ell.values,
            ell.indices,
            B,
            semiring=REAL,
            n_cols=16,
            backend='pallas-cuda',
        )
        n_fallback = sum(1 for w in ws if w.category is NitrixBackendFallback)
    # Pallas ELL kernel raises NotTileable on current JAX -> exactly one warning.
    assert n_fallback == 1
    # The output still matches the JAX path.
    out_jax = semiring_ell_matmul(
        ell.values,
        ell.indices,
        B,
        semiring=REAL,
        n_cols=16,
        backend='jax',
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
                ell.values,
                ell.indices,
                B,
                semiring=REAL,
                n_cols=16,
                backend='pallas-cuda',
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
        values,
        indices,
        Bs,
        semiring=REAL,
        n_cols=n,
        backend='jax',
    )
    np.testing.assert_allclose(
        out[0],
        jnp.matmul(A0, B0, precision='highest'),
        atol=1e-4,
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        out[1],
        jnp.matmul(A1, B1, precision='highest'),
        atol=1e-4,
        rtol=1e-4,
    )


# ---------------------------------------------------------------------------
# semiring_ell_rmatvec -- the REAL adjoint (transpose) matvec
# ---------------------------------------------------------------------------


def test_rmatvec_equals_dense_transpose_matvec():
    m, n = 10, 10
    A = jax.random.normal(jax.random.key(7), (m, n))
    ell = ell_from_dense(A, k_max=3)  # asymmetric top-3
    Ad = np.asarray(ell_to_dense(ell))
    X = np.asarray(jax.random.normal(jax.random.key(8), (m, 4)))
    Y = semiring_ell_rmatvec(
        ell.values, ell.indices, jnp.asarray(X), semiring=REAL, n_cols=n
    )
    np.testing.assert_allclose(np.asarray(Y), Ad.T @ X, atol=1e-5)


def test_rmatvec_reference_matches_primitive():
    m, n = 12, 12
    A = jax.random.normal(jax.random.key(9), (m, n))
    ell = ell_from_dense(A, k_max=4)
    X = jax.random.normal(jax.random.key(10), (m, 3))
    ref = reference_semiring_ell_rmatvec(
        ell.values, ell.indices, X, semiring=REAL, n_cols=n
    )
    prim = semiring_ell_rmatvec(
        ell.values, ell.indices, X, semiring=REAL, n_cols=n
    )
    np.testing.assert_allclose(np.asarray(ref), np.asarray(prim), atol=1e-6)


def test_matvec_plus_rmatvec_is_symmetric_part():
    """½(A x + Aᵀ x) == sym(A) x -- the identity the spectral fix relies on."""
    n = 16
    A = jax.random.normal(jax.random.key(11), (n, n))
    ell = ell_from_dense(A, k_max=5)
    Ad = np.asarray(ell_to_dense(ell))
    X = np.asarray(jax.random.normal(jax.random.key(12), (n, 3)))
    Xj = jnp.asarray(X)
    Ax = semiring_ell_matmul(
        ell.values, ell.indices, Xj, semiring=REAL, n_cols=n, backend='jax'
    )
    Atx = semiring_ell_rmatvec(
        ell.values, ell.indices, Xj, semiring=REAL, n_cols=n
    )
    sym = 0.5 * (np.asarray(Ax) + np.asarray(Atx))
    np.testing.assert_allclose(sym, 0.5 * (Ad + Ad.T) @ X, atol=1e-5)


def test_rmatvec_vjp_is_exact_adjoint():
    """The VJP of the linear map ``Y = Aᵀ X`` is exact (dtype-robust, so it
    holds in both the float32-isolated and x64-suite regimes):

        d/dX  ⟨C, Y⟩ = A C                                   (the gather matmul)
        d/dv  ⟨C, Y⟩[i, p] = ⟨X[i, :], C[indices[i, p], :]⟩
    """
    m, n = 9, 9
    A = jax.random.normal(jax.random.key(13), (m, n))
    ell = ell_from_dense(A, k_max=3)
    Ad = np.asarray(ell_to_dense(ell))
    X = np.asarray(jax.random.normal(jax.random.key(14), (m, 2)))
    C = np.asarray(jax.random.normal(jax.random.key(15), (n, 2)))

    def fwd(values, Xin):
        return semiring_ell_rmatvec(
            values, ell.indices, Xin, semiring=REAL, n_cols=n
        )

    _, vjp = jax.vjp(fwd, ell.values, jnp.asarray(X))
    g_v, g_X = vjp(jnp.asarray(C))
    # grad w.r.t. X is the dense gather A @ C
    np.testing.assert_allclose(np.asarray(g_X), Ad @ C, atol=1e-4)
    # grad w.r.t. values: per-edge gather-dot of X with the scattered cotangent
    idx = np.asarray(ell.indices)
    gv_ref = np.einsum('ic,ipc->ip', X, C[idx])
    np.testing.assert_allclose(np.asarray(g_v), gv_ref, atol=1e-4)


def test_rmatvec_batched():
    m, n = 8, 8
    key = jax.random.key(15)
    A0 = jax.random.normal(jax.random.fold_in(key, 0), (m, n))
    A1 = jax.random.normal(jax.random.fold_in(key, 1), (m, n))
    X0 = np.asarray(jax.random.normal(jax.random.fold_in(key, 2), (m, 3)))
    X1 = np.asarray(jax.random.normal(jax.random.fold_in(key, 3), (m, 3)))
    ell0 = ell_from_dense(A0, k_max=n)
    ell1 = ell_from_dense(A1, k_max=n)
    values = jnp.stack([ell0.values, ell1.values])
    indices = jnp.stack([ell0.indices, ell1.indices])
    Xs = jnp.stack([jnp.asarray(X0), jnp.asarray(X1)])
    out = semiring_ell_rmatvec(values, indices, Xs, semiring=REAL, n_cols=n)
    np.testing.assert_allclose(
        out[0], np.asarray(ell_to_dense(ell0)).T @ X0, atol=1e-4
    )
    np.testing.assert_allclose(
        out[1], np.asarray(ell_to_dense(ell1)).T @ X1, atol=1e-4
    )


@pytest.mark.parametrize('algebra', [LOG, TROPICAL_MAX_PLUS])
def test_rmatvec_non_real_raises(algebra):
    ell = ell_from_dense(jax.random.normal(jax.random.key(16), (6, 6)), k_max=3)
    X = jnp.ones((6, 2))
    with pytest.raises(NotImplementedError, match='REAL'):
        semiring_ell_rmatvec(
            ell.values, ell.indices, X, semiring=algebra, n_cols=6
        )


def test_ell_from_dense_topk_breaks_symmetry_symmetrize_fixes_it():
    rng = np.random.default_rng(17)
    n = 20
    A = np.abs(rng.standard_normal((n, n)))
    A = A + A.T  # symmetric source
    np.fill_diagonal(A, 0.0)
    asym = np.asarray(ell_to_dense(ell_from_dense(jnp.asarray(A), k_max=4)))
    sym = np.asarray(
        ell_to_dense(ell_from_dense(jnp.asarray(A), k_max=4, symmetrize=True))
    )
    assert not np.allclose(asym, asym.T)  # the trap
    np.testing.assert_allclose(sym, sym.T, atol=1e-6)  # symmetrize fixes it
