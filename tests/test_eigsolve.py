# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for the ``nitrix.linalg._eigsolve`` extremal-eigensolver
dispatcher: the ``SolverSpec`` contract, the validity table / ``auto``
policy, the uniform top-k return, and differentiability through the
dispatcher.

The tight per-method kernel numerics (shift-invert / poly agreement with
``eigh`` at 1e-3 / 1e-2) are covered, on the real normalised-affinity
operator, by ``tests/test_graph.py``; here we use an affinity-spectrum
fixture (eigenvalues in ``[-1, 1]`` -- the regime shift-invert's ``sigma``
and poly's ``shift`` assume) and verify routing + mechanics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nitrix.linalg._eigsolve import EigPair, SolverSpec, eigsolve_top_k
from nitrix.sparse import ELL, SectionedELL, sectioned_ell_from_ragged
from nitrix.sparse.ell import ell_from_dense, ell_to_dense


def _affinity(n: int, seed: int = 0) -> jnp.ndarray:
    """Symmetric normalised affinity ``D^-1/2 A D^-1/2`` of a random
    Erdos-Renyi graph: spectrum in ``[-1, 1]`` with the largest eigenvalue
    ~1, the regime every iterative method here is tuned for."""
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < 0.5).astype(np.float64)
    A = np.triu(A, 1)
    A = A + A.T
    d = A.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    M = A * inv_sqrt[:, None] * inv_sqrt[None, :]
    return jnp.asarray(M)


def _eigh_top_k_reference(M: jnp.ndarray, k: int) -> np.ndarray:
    vals = np.linalg.eigvalsh(np.asarray(M))
    return np.sort(vals)[::-1][:k]


# ---------------------------------------------------------------------------
# SolverSpec
# ---------------------------------------------------------------------------


def test_solverspec_is_hashable_and_frozen():
    spec = SolverSpec.lobpcg(n_iters=50)
    assert hash(spec) == hash(SolverSpec.lobpcg(n_iters=50))
    assert spec != SolverSpec.lobpcg(n_iters=51)
    with pytest.raises(Exception):
        spec.n_iters = 99  # frozen


def test_solverspec_builders_set_method():
    assert SolverSpec.auto().method == 'auto'
    assert SolverSpec.eigh().method == 'eigh'
    assert SolverSpec.lobpcg().method == 'lobpcg'
    assert SolverSpec.shift_invert().method == 'shift_invert'
    assert SolverSpec.poly().method == 'poly'


# ---------------------------------------------------------------------------
# eigh method: uniform extremal top-k return
# ---------------------------------------------------------------------------


def test_eigh_returns_largest_first_top_k():
    M = _affinity(24)
    out = eigsolve_top_k(M, 4, spec=SolverSpec.eigh())
    assert isinstance(out, EigPair)
    assert out.values.shape == (4,)
    assert out.vectors.shape == (24, 4)
    np.testing.assert_allclose(
        np.asarray(out.values),
        _eigh_top_k_reference(M, 4),
        atol=1e-5,
    )
    # Largest-first.
    assert np.all(np.diff(np.asarray(out.values)) <= 1e-6)


def test_auto_dense_resolves_to_eigh():
    M = _affinity(24)
    auto = eigsolve_top_k(M, 3, spec=SolverSpec.auto())
    eigh = eigsolve_top_k(M, 3, spec=SolverSpec.eigh())
    np.testing.assert_allclose(
        np.asarray(auto.values),
        np.asarray(eigh.values),
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# Iterative methods recover the extremal eigenvalues
# ---------------------------------------------------------------------------


def test_lobpcg_dense_agrees_with_eigh():
    M = _affinity(48)
    out = eigsolve_top_k(M, 3, spec=SolverSpec.lobpcg(n_iters=400))
    np.testing.assert_allclose(
        np.sort(np.asarray(out.values))[::-1],
        _eigh_top_k_reference(M, 3),
        atol=1e-4,
    )


@pytest.mark.parametrize('method', ['shift_invert', 'poly'])
def test_shift_invert_and_poly_recover_top_eigenvalue(method):
    # Tight kernel numerics live in test_graph.py on the connectopy
    # operator; here we just confirm the dispatcher routes and the
    # approximate methods land near the extremal eigenvalue.
    M = _affinity(48)
    spec = getattr(SolverSpec, method)()
    out = eigsolve_top_k(M, 3, spec=spec)
    top = float(np.max(np.asarray(out.values)))
    assert abs(top - _eigh_top_k_reference(M, 1)[0]) < 5e-2


def test_auto_sparse_resolves_to_lobpcg_and_matches_dense():
    M = _affinity(48)
    ell = ell_from_dense(M)
    out = eigsolve_top_k(ell, 3, spec=SolverSpec.auto())
    np.testing.assert_allclose(
        np.sort(np.asarray(out.values))[::-1],
        _eigh_top_k_reference(M, 3),
        atol=1e-4,
    )


# ---------------------------------------------------------------------------
# Validity table
# ---------------------------------------------------------------------------


def test_eigh_rejects_sparse():
    ell = ell_from_dense(_affinity(16))
    with pytest.raises(ValueError, match="does not support 'ell'"):
        eigsolve_top_k(ell, 2, spec=SolverSpec.eigh())


@pytest.mark.parametrize('method', ['shift_invert', 'poly'])
def test_shift_invert_and_poly_serve_sparse(method):
    # Phase-4: shift-invert and poly serve ELL (forward = the ELL matvec,
    # backward = the same per-format projection as lobpcg).
    M = _affinity(48)
    ell = ell_from_dense(M)
    spec = getattr(SolverSpec, method)()
    out = eigsolve_top_k(ell, 3, spec=spec)
    top = float(np.max(np.asarray(out.values)))
    assert abs(top - _eigh_top_k_reference(M, 1)[0]) < 5e-2


def test_ell_shift_invert_grad_is_finite():
    """Differentiability holds for shift-invert on ELL via the shared
    per-format backward."""
    M = _affinity(40)
    ell = ell_from_dense(M)

    def loss(values):
        op = ELL(values, ell.indices, ell.n_cols, ell.identity)
        return eigsolve_top_k(
            op,
            3,
            spec=SolverSpec.shift_invert(),
        ).values.sum()

    g = jax.grad(loss)(ell.values)
    assert g.shape == ell.values.shape
    assert np.all(np.isfinite(np.asarray(g)))


# ---------------------------------------------------------------------------
# Full (format x method) matrix: jit/eager parity + grad-under-jit
# ---------------------------------------------------------------------------


def _spec(method):
    return {
        'eigh': SolverSpec.eigh(),
        'lobpcg': SolverSpec.lobpcg(n_iters=400),
        'shift_invert': SolverSpec.shift_invert(),
        'poly': SolverSpec.poly(),
    }[method]


def _sectioned(M):
    """SectionedELL of a dense affinity (ragged per-row neighbour lists)."""
    n = M.shape[0]
    Mnp = np.asarray(M)
    vals, idx = [], []
    for i in range(n):
        nz = np.nonzero(Mnp[i])[0]
        vals.append(Mnp[i, nz])
        idx.append(nz.astype(np.int32))
    return sectioned_ell_from_ragged(vals, idx, n_cols=n, identity=0.0)


def _leaf_and_rebuild(fmt, M):
    """``(leaf, rebuild)`` so ``rebuild(leaf)`` is the operand and ``leaf``
    is the differentiable array(s): dense ``M``, ELL ``values``, or the
    SectionedELL per-section ``values`` tuple.  Threading the leaf lets one
    parametrized body cover ``jit`` / ``grad`` for all three formats."""
    if fmt == 'dense':
        return M, (lambda x: x)
    if fmt == 'ell':
        ell = ell_from_dense(M)
        return ell.values, (
            lambda v: ELL(v, ell.indices, ell.n_cols, ell.identity)
        )
    sec = _sectioned(M)
    leaf = tuple(s.values for s in sec.sections)

    def rebuild(vt):
        sections = tuple(
            ELL(v, s.indices, s.n_cols, s.identity)
            for v, s in zip(vt, sec.sections)
        )
        return SectionedELL(
            sections,
            sec.row_groups,
            sec.n_rows,
            sec.n_cols,
            sec.identity,
        )

    return leaf, rebuild


# ``eigh`` is dense-only; the iterative methods cover every format (phase 4).
_CASES = [('dense', m) for m in ('eigh', 'lobpcg', 'shift_invert', 'poly')] + [
    (f, m)
    for f in ('ell', 'sectioned')
    for m in ('lobpcg', 'shift_invert', 'poly')
]


@pytest.mark.parametrize('fmt,method', _CASES)
def test_matrix_forward_jit_parity(fmt, method):
    """Every (format, method) cell is ``jit``-able: the jit forward matches
    eager (tight) and lands on the dense eigh reference (loose iterative
    floor)."""
    M = _affinity(48)
    leaf, rebuild = _leaf_and_rebuild(fmt, M)
    spec = _spec(method)

    def f(x):
        return eigsolve_top_k(rebuild(x), 3, spec=spec).values

    eager = np.asarray(f(leaf))
    jitted = np.asarray(jax.jit(f)(leaf))
    np.testing.assert_allclose(jitted, eager, atol=1e-5)
    np.testing.assert_allclose(
        np.sort(eager)[::-1],
        _eigh_top_k_reference(M, 3),
        atol=3e-2,
    )


@pytest.mark.parametrize('fmt,method', _CASES)
def test_matrix_grad_under_jit(fmt, method):
    """Every (format, method) cell is differentiable under ``jit``: the
    gradient is finite and ``jit(grad)`` matches eager ``grad``.  For the
    sparse formats the differentiable leaf is the per-section ``values``
    pytree."""
    M = _affinity(40)
    leaf, rebuild = _leaf_and_rebuild(fmt, M)
    spec = _spec(method)

    def loss(x):
        return eigsolve_top_k(rebuild(x), 3, spec=spec).values.sum()

    g_eager = jax.tree_util.tree_leaves(jax.grad(loss)(leaf))
    g_jit = jax.tree_util.tree_leaves(jax.jit(jax.grad(loss))(leaf))
    assert g_eager and len(g_eager) == len(g_jit)
    for ge, gj in zip(g_eager, g_jit):
        gj = np.asarray(gj)
        assert np.all(np.isfinite(gj))
        # jit vs eager agree up to fp32 reduction-order noise; the
        # CG/while-loop-based ``shift_invert`` is the loose end (its inner
        # iterations amplify tiny differences through the F-matrix).
        np.testing.assert_allclose(gj, np.asarray(ge), atol=2e-3)


# ---------------------------------------------------------------------------
# Differentiability through the dispatcher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('method', ['lobpcg', 'shift_invert', 'poly'])
def test_dense_grad_matches_eigh_grad(method):
    """The eigenvalue-sum gradient is Hellmann-Feynman (sum_i u_i u_i^T) and
    the backward is solver-independent, so every iterative method matches
    eigh's gradient and lives in the symmetric subspace."""
    M = _affinity(40)

    def loss(op, spec):
        return eigsolve_top_k(op, 3, spec=spec).values.sum()

    g_eigh = jax.grad(loss)(M, SolverSpec.eigh())
    g = jax.grad(loss)(M, _spec(method))
    np.testing.assert_allclose(np.asarray(g), np.asarray(g_eigh), atol=5e-3)
    np.testing.assert_allclose(np.asarray(g), np.asarray(g).T, atol=1e-6)


def test_ell_lobpcg_grad_is_pattern_projected_and_finite():
    M = _affinity(40)
    ell = ell_from_dense(M)

    def loss(values):
        op = ELL(values, ell.indices, ell.n_cols, ell.identity)
        return eigsolve_top_k(
            op,
            3,
            spec=SolverSpec.lobpcg(n_iters=400),
        ).values.sum()

    g = jax.grad(loss)(ell.values)
    assert g.shape == ell.values.shape
    assert np.all(np.isfinite(np.asarray(g)))


# ---------------------------------------------------------------------------
# promise_symmetry: the symmetric-part matvec for asymmetric (top-k) ELLs
# ---------------------------------------------------------------------------


def _asym_affinity_ell(n: int = 60, k: int = 5, seed: int = 0) -> ELL:
    """Top-k-per-row sparsification of a symmetric affinity -- an
    *asymmetric* stored ELL (the silent-failure case)."""
    return ell_from_dense(_affinity(n, seed), k_max=k)


def _sym_top_k_reference(ell: ELL, k: int) -> np.ndarray:
    """Largest-``k`` eigenvalues of ½(A + Aᵀ) for the stored ELL ``A``."""
    Ad = np.asarray(ell_to_dense(ell))
    return np.sort(np.linalg.eigvalsh(0.5 * (Ad + Ad.T)))[::-1][:k]


def test_solverspec_promise_symmetry_default_and_hashable():
    assert SolverSpec.lobpcg().promise_symmetry is True
    assert SolverSpec.auto().promise_symmetry is True
    off = SolverSpec.lobpcg(promise_symmetry=False)
    assert hash(off) == hash(SolverSpec.lobpcg(promise_symmetry=False))
    assert off != SolverSpec.lobpcg(promise_symmetry=True)


def test_promise_symmetry_false_matches_dense_symmetrised():
    ell = _asym_affinity_ell()
    Ad = np.asarray(ell_to_dense(ell))
    assert not np.allclose(Ad, Ad.T)  # the stored pattern is asymmetric
    ref = _sym_top_k_reference(ell, 3)
    out = eigsolve_top_k(
        ell, 3, spec=SolverSpec.lobpcg(n_iters=400, promise_symmetry=False)
    )
    np.testing.assert_allclose(
        np.sort(np.asarray(out.values))[::-1], ref, atol=1e-3
    )


def test_promise_symmetry_true_diverges_on_asymmetric_ell():
    """The trap: trusting symmetry on an asymmetric operator gives a
    different (wrong) spectrum than ½(A + Aᵀ)."""
    ell = _asym_affinity_ell()
    ref = _sym_top_k_reference(ell, 3)
    out = eigsolve_top_k(
        ell, 3, spec=SolverSpec.lobpcg(n_iters=400, promise_symmetry=True)
    )
    assert not np.allclose(
        np.sort(np.asarray(out.values))[::-1], ref, atol=1e-2
    )


def test_sectioned_promise_symmetry_false_matches_dense():
    ell = _asym_affinity_ell()
    Ad = np.asarray(ell_to_dense(ell))
    ref = _sym_top_k_reference(ell, 3)
    sec = _sectioned(jnp.asarray(Ad))  # ragged from the asymmetric pattern
    out = eigsolve_top_k(
        sec, 3, spec=SolverSpec.lobpcg(n_iters=400, promise_symmetry=False)
    )
    np.testing.assert_allclose(
        np.sort(np.asarray(out.values))[::-1], ref, atol=1e-3
    )


def test_promise_symmetry_false_requires_square_operator():
    rect = ELL(
        values=jnp.ones((6, 2)),
        indices=jnp.zeros((6, 2), jnp.int32),
        n_cols=8,
        identity=0.0,
    )
    with pytest.raises(ValueError, match='square'):
        eigsolve_top_k(
            rect, 1, spec=SolverSpec.lobpcg(promise_symmetry=False)
        )


def test_promise_symmetry_false_grad_is_finite():
    ell = _asym_affinity_ell()

    def loss(values):
        op = ELL(values, ell.indices, ell.n_cols, ell.identity)
        return eigsolve_top_k(
            op,
            3,
            spec=SolverSpec.lobpcg(n_iters=400, promise_symmetry=False),
        ).values.sum()

    g = jax.grad(loss)(ell.values)
    assert g.shape == ell.values.shape
    assert np.all(np.isfinite(np.asarray(g)))
