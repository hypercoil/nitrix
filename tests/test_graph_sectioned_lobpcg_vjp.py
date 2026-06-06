# -*- coding: utf-8 -*-
"""Per-section vs flat-ELL gradient parity for the LOBPCG SectionedELL VJP.

The load-bearing property: for the same underlying sparse adjacency
laid out as both (a) flat ELL with global ``k_max`` and (b)
SectionedELL bucketed by row degree, the gradient w.r.t. the
non-pad entries must agree (up to fp64 epsilon).  This guards
against a per-section indexing or scatter-back bug in the
implicit-VJP backward.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.linalg._eigsolve import (
    SolverSpec,
    _eig_top_k_ell,
    _eig_top_k_sectioned,
)
from nitrix.semiring import REAL, semiring_ell_matmul
from nitrix.sparse import sectioned_ell_from_ragged


def _make_varying_degree_graph(n=50, seed=7):
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n))
    # Ring base
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    # Random chords (more for higher indices to vary degree)
    for _ in range(40):
        i, j = rng.integers(0, n, 2)
        if i != j:
            A[i, j] = A[j, i] = 1.0
    A += 0.5 * np.eye(n)
    return A


def _to_flat_ell(A_np):
    n = A_np.shape[0]
    k_max = int((A_np != 0).sum(axis=1).max())
    values = np.zeros((n, k_max))
    indices = np.zeros((n, k_max), dtype=np.int32)
    for i in range(n):
        nz = np.nonzero(A_np[i])[0]
        values[i, : len(nz)] = A_np[i, nz]
        indices[i, : len(nz)] = nz
        if len(nz) < k_max:
            indices[i, len(nz):] = nz[0]
    return jnp.asarray(values), jnp.asarray(indices), k_max


def _to_sectioned(A_np):
    n = A_np.shape[0]
    ragged_values = []
    ragged_indices = []
    for i in range(n):
        nz = np.nonzero(A_np[i])[0]
        ragged_values.append(A_np[i, nz])
        ragged_indices.append(nz.astype(np.int32))
    return sectioned_ell_from_ragged(
        ragged_values, ragged_indices, n_cols=n, identity=0.0,
    )


def test_sectioned_vjp_matches_flat_per_entry():
    '''Per-section gradient at each non-pad ``(row, col)`` entry
    matches the flat-ELL gradient at the same ``(row, col)`` to
    machine-eps fp64.

    Verifies the per-section ``row_groups``-gather + einsum is
    correct.
    '''
    A_np = _make_varying_degree_graph()
    n = A_np.shape[0]
    values_flat, indices_flat, k_max = _to_flat_ell(A_np)
    sec = _to_sectioned(A_np)

    X0 = jax.random.normal(jax.random.key(0), (n, 3))
    target = jax.random.normal(jax.random.key(2), (3, 3))
    target = (target + target.T) / 2

    # Flat-ELL loss
    def loss_flat(values):
        _, U = _eig_top_k_ell(
            values, indices_flat, X0, n, SolverSpec.lobpcg(n_iters=500),
        )
        AU = semiring_ell_matmul(
            values, indices_flat, U,
            semiring=REAL, n_cols=n, backend='jax',
        )
        return jnp.trace(U.T @ AU @ target)

    g_flat = jax.grad(loss_flat)(values_flat)

    # SectionedELL loss (via the same matvec definition).
    from nitrix.linalg._eigsolve import _sectioned_matvec

    values_tuple = tuple(s.values for s in sec.sections)
    indices_tuple = tuple(s.indices for s in sec.sections)
    row_groups_tuple = tuple(jnp.asarray(rg) for rg in sec.row_groups)

    def loss_sec(values_tuple):
        _, U = _eig_top_k_sectioned(
            values_tuple, indices_tuple, row_groups_tuple, X0,
            n, SolverSpec.lobpcg(n_iters=500),
        )
        AU = _sectioned_matvec(
            values_tuple, indices_tuple, row_groups_tuple, U,
            n_cols=n, n_rows=n,
        )
        return jnp.trace(U.T @ AU @ target)

    g_sec = jax.grad(loss_sec)(values_tuple)

    # For each non-pad entry in each section, compare to the matching
    # flat-ELL entry by (row, col).
    max_diff = 0.0
    n_compared = 0
    for s_idx, (g_s, idx_s, row_idx_s) in enumerate(
        zip(g_sec, indices_tuple, row_groups_tuple)
    ):
        sec_values = sec.sections[s_idx].values
        n_rows_s, k_max_s = g_s.shape
        for i_local in range(n_rows_s):
            global_row = int(row_idx_s[i_local])
            for p in range(k_max_s):
                if float(sec_values[i_local, p]) == 0.0:
                    continue  # pad slot
                col = int(idx_s[i_local, p])
                for p_flat in range(k_max):
                    if (
                        int(indices_flat[global_row, p_flat]) == col
                        and float(values_flat[global_row, p_flat]) != 0
                    ):
                        diff = abs(
                            float(g_s[i_local, p])
                            - float(g_flat[global_row, p_flat])
                        )
                        max_diff = max(max_diff, diff)
                        n_compared += 1
                        break
    assert n_compared > 0, 'no per-entry comparison was made'
    assert max_diff < 1e-13, (
        f'sectioned vs flat per-entry max diff = {max_diff:.2e}'
    )


def test_sectioned_vjp_finite_under_grad():
    '''Smoke check: ``jax.grad`` through the sectioned LOBPCG returns
    finite per-section gradients.
    '''
    A_np = _make_varying_degree_graph()
    n = A_np.shape[0]
    sec = _to_sectioned(A_np)
    X0 = jax.random.normal(jax.random.key(0), (n, 3))
    target = jax.random.normal(jax.random.key(2), (3, 3))
    target = (target + target.T) / 2

    from nitrix.linalg._eigsolve import _sectioned_matvec

    values_tuple = tuple(s.values for s in sec.sections)
    indices_tuple = tuple(s.indices for s in sec.sections)
    row_groups_tuple = tuple(jnp.asarray(rg) for rg in sec.row_groups)

    def loss(values_tuple):
        _, U = _eig_top_k_sectioned(
            values_tuple, indices_tuple, row_groups_tuple, X0,
            n, SolverSpec.lobpcg(n_iters=500),
        )
        return jnp.trace(U.T @ U @ target)

    g = jax.grad(loss)(values_tuple)
    assert len(g) == len(values_tuple)
    for gi, vi in zip(g, values_tuple):
        assert gi.shape == vi.shape
        assert bool(jnp.all(jnp.isfinite(gi)))
