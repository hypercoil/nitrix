# -*- coding: utf-8 -*-
"""Phase-A regression guards for the LME size-dispatch / cuSOLVER bypass.

These pin the three properties the dispatch refactor exists to deliver, so a
future change cannot silently regress them:

1. **The per-voxel fit issues no cuSOLVER custom-call** at any ``p`` -- the
   closed-form (``p <= 2``) and hand-Cholesky + ``triangularSolve`` (``p > 2``)
   paths keep the compiled HLO on cuBLAS only.  This is the property that
   unblocks ``flame_two_level`` on the broken-cuSOLVER L4 and flattens the
   linear-in-``V`` compile (see
   ``docs/feature-requests/lme-family-tiny-linalg-gpu-block-and-perf.md`` and
   ``docs/feature-requests/gpu-cusolver-first-call-handle-failure.md``).
2. **Voxel-block chunking is numerically transparent** -- ``block=k`` agrees
   with the single-``vmap`` fit; it is a memory knob, not a result change.
3. **The general ``p > 2`` path is correct** -- the unrolled hand-Cholesky
   inverse / log-det matches ``jnp.linalg`` to machine precision, and a
   ``p = 5`` ``reml_fit`` runs and is finite.
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.stats._smalllinalg import spd_inv_logdet_chol, sym_eig_jacobi
from nitrix.stats.lme import flame_two_level, reml_fit
from nitrix.stats.lme._varcomp import VarCompSpec, fit_varcomp_diagonal

# cuSOLVER-backed routines whose custom-call targets must NOT appear in the
# per-voxel HLO (cuBLAS matmul / triangularSolve are fine).
_CUSOLVER_TOKENS = (
    'cusolver',
    'syevd',
    'potrf',
    'getrf',
    'geqrf',
    'gesvd',
    'cholesky',
    'eigh',
    'lu_decomposition',
)


def _custom_call_targets(hlo: str) -> list[str]:
    return sorted(set(re.findall(r'custom_call_target="([^"]+)"', hlo)))


def _cusolver_calls(hlo: str) -> list[str]:
    return [
        c
        for c in _custom_call_targets(hlo)
        if any(tok in c.lower() for tok in _CUSOLVER_TOKENS)
    ]


# ---------------------------------------------------------------------------
# 1. The per-voxel fit is cuSOLVER-free at every p
# ---------------------------------------------------------------------------


def test_varcomp_per_voxel_hlo_is_cusolver_free():
    """The compiled per-voxel REML core must issue no cuSOLVER custom-call.

    Closed-form for ``p in {1, 2}``; unrolled hand-Cholesky +
    ``triangularSolve`` (cuBLAS) for ``p > 2``.  A ``potrf`` / ``syevd`` /
    ``getrf`` here is the exact failure that skips ``flame_two_level`` on the
    broken-cuSOLVER GPU.
    """
    rng = np.random.default_rng(0)
    for p in (1, 2, 5):
        N, V, K = 40, 256, 2
        Y = jnp.asarray(rng.standard_normal((V, N)))
        X = jnp.asarray(rng.standard_normal((N, p)))
        B = jnp.asarray(
            np.stack([np.abs(rng.standard_normal(N)), np.ones(N)])
        )
        init = jnp.zeros((V, K))
        f = jax.jit(
            lambda Y, X, B, init: fit_varcomp_diagonal(
                Y, X, B, init, spec=VarCompSpec(n_iter=5)
            )
        )
        hlo = f.lower(Y, X, B, init).compile().as_text()
        assert not _cusolver_calls(hlo), (
            f'p={p}: per-voxel REML core must be cuSOLVER-free; found '
            f'{_cusolver_calls(hlo)}'
        )


def test_flame_hlo_is_cusolver_free():
    """``flame_two_level`` has no eigendecomposition, so its *entire* compiled
    program must be cuSOLVER-free (the end-to-end GPU-unblock guarantee)."""
    rng = np.random.default_rng(0)
    V, N, p = 512, 40, 2
    beta = jnp.asarray(rng.standard_normal((V, N)))
    var_within = jnp.asarray(np.abs(rng.standard_normal((V, N))) + 0.1)
    X_group = jnp.asarray(rng.standard_normal((N, p)))
    f = jax.jit(lambda b, vw, xg: flame_two_level(b, vw, xg, n_iter=5))
    hlo = f.lower(beta, var_within, X_group).compile().as_text()
    assert not _cusolver_calls(hlo), (
        f'flame_two_level must be entirely cuSOLVER-free; found '
        f'{_cusolver_calls(hlo)}'
    )


# ---------------------------------------------------------------------------
# 2. Voxel-block chunking is numerically transparent
# ---------------------------------------------------------------------------


def test_reml_block_chunking_matches_single_vmap():
    """``reml_fit(block=k)`` equals the single-``vmap`` fit (a memory knob)."""
    rng = np.random.default_rng(0)
    g, n_per, V = 4, 20, 23  # V not a multiple of the block on purpose
    N = g * n_per
    group = np.repeat(np.arange(g), n_per)
    X = jnp.asarray(np.column_stack([np.ones(N), rng.standard_normal(N)]))
    Z = np.zeros((N, g))
    for i in range(g):
        Z[group == i, i] = 1.0
    Z = jnp.asarray(Z)
    Y = jnp.asarray(rng.standard_normal((V, N)))

    full = reml_fit(Y, X, Z, n_iter=30)
    chunked = reml_fit(Y, X, Z, n_iter=30, block=8)
    np.testing.assert_allclose(
        np.asarray(chunked.theta_hat), np.asarray(full.theta_hat), atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(chunked.beta_hat), np.asarray(full.beta_hat), atol=1e-9
    )


def test_flame_block_chunking_matches_single_vmap():
    """``flame_two_level(block=k)`` equals the single-``vmap`` fit."""
    rng = np.random.default_rng(0)
    V, N, p = 19, 30, 2
    beta = jnp.asarray(rng.standard_normal((V, N)))
    var_within = jnp.asarray(np.abs(rng.standard_normal((V, N))) + 0.1)
    X_group = jnp.asarray(rng.standard_normal((N, p)))

    full = flame_two_level(beta, var_within, X_group, n_iter=20)
    chunked = flame_two_level(beta, var_within, X_group, n_iter=20, block=6)
    # block= only changes the vmap reduction order, so block vs single differ
    # by floating-point reassociation (~1e-8 on the variance components here),
    # not by the chunking maths.  atol=1e-6 stays ~50x above that reassociation
    # noise yet far below any real chunking bug (which mishandles whole voxels,
    # an O(1) error); a 1e-9 tolerance was below the reassociation floor.
    np.testing.assert_allclose(
        np.asarray(chunked.sigma_b_sq), np.asarray(full.sigma_b_sq), atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(chunked.gamma_hat), np.asarray(full.gamma_hat), atol=1e-6
    )


# ---------------------------------------------------------------------------
# 3. The general p > 2 path is correct
# ---------------------------------------------------------------------------


def test_rolled_cholesky_matches_reference():
    """Hand-Cholesky inverse + log-det match a numpy reference for SPD ``A``.

    The reference is numpy (CPU) on purpose: the whole point of the hand
    Cholesky is to avoid ``jnp.linalg.inv`` / ``slogdet`` (cuSOLVER), which can
    fail on the broken-stack GPU -- using them as the oracle would couple the
    test to the very routine the code routes around.
    """
    rng = np.random.default_rng(1)
    for p in (3, 4, 6, 8, 20):
        M = rng.standard_normal((p, p))
        A_np = M @ M.T + p * np.eye(p)
        inv, logdet = spd_inv_logdet_chol(jnp.asarray(A_np), p)
        np.testing.assert_allclose(
            np.asarray(inv), np.linalg.inv(A_np), atol=1e-10
        )
        np.testing.assert_allclose(
            float(logdet), float(np.linalg.slogdet(A_np)[1]), atol=1e-10
        )


def test_cholesky_pivot_floor_keeps_singular_input_finite():
    """The modified-Cholesky pivot floor keeps a degenerate (singular /
    boundary) system finite instead of producing NaN, while leaving a
    well-conditioned solve bit-unchanged."""
    from nitrix.stats._smalllinalg import small_inv_logdet

    rng = np.random.default_rng(0)
    # Well-conditioned: floor never activates -> matches numpy to machine eps.
    for p in (1, 2, 4, 8):
        m = rng.standard_normal((p, p))
        a = jnp.asarray(m @ m.T + p * np.eye(p))
        inv, _ = small_inv_logdet(a, p)
        np.testing.assert_allclose(
            np.asarray(inv), np.linalg.inv(np.asarray(a)), atol=1e-10
        )
    # Exactly singular (rank-1) and a zero scalar: must stay finite.
    for p in (2, 3, 5):
        u = rng.standard_normal((p, 1))
        inv, ld = small_inv_logdet(jnp.asarray(u @ u.T), p)
        assert bool(jnp.all(jnp.isfinite(inv))) and bool(jnp.isfinite(ld))
    inv1, ld1 = small_inv_logdet(jnp.zeros((1, 1)), 1)
    assert bool(jnp.all(jnp.isfinite(inv1))) and bool(jnp.isfinite(ld1))


def test_reml_collinear_design_stays_finite():
    """A rank-deficient fixed-effect design (collinear covariate) fits to a
    finite result rather than NaN-propagating through the per-voxel solve."""
    rng = np.random.default_rng(1)
    g, n_per, V = 5, 20, 4
    N = g * n_per
    group = np.repeat(np.arange(g), n_per)
    x = rng.standard_normal(N)
    # third column is an exact duplicate of the second -> rank-deficient X.
    X = jnp.asarray(np.column_stack([np.ones(N), x, x]))
    Z = np.zeros((N, g))
    for i in range(g):
        Z[group == i, i] = 1.0
    result = reml_fit(jnp.asarray(rng.standard_normal((V, N))), X, jnp.asarray(Z), n_iter=20)
    assert bool(jnp.all(jnp.isfinite(result.theta_hat)))
    assert bool(jnp.all(jnp.isfinite(result.beta_hat)))


def test_cholesky_graph_size_is_flat_in_p():
    """The rolled Cholesky keeps the graph ``O(p^2)`` -- a single ``while`` over
    columns -- so its top-level equation count is small and ~constant in ``p``.

    An unrolled Cholesky would emit ``O(p^3)`` equations and blow compile up
    cubically (the diagnosed GAM bottleneck); this guards the rolled form.
    """
    rng = np.random.default_rng(0)

    def n_eqns(p: int) -> int:
        m = rng.standard_normal((p, p))
        A = jnp.asarray(m @ m.T + p * np.eye(p))
        jaxpr = jax.make_jaxpr(lambda M: spd_inv_logdet_chol(M, p)[0])(A)
        return len(jaxpr.jaxpr.eqns)

    small, large = n_eqns(10), n_eqns(30)
    assert large < 40, f'graph too large ({large} eqns) -- Cholesky unrolled?'
    assert large <= small + 3, (
        f'graph grows with p ({small} -> {large}) -- Cholesky no longer rolled.'
    )


# ---------------------------------------------------------------------------
# 4. The cuSOLVER-free small symmetric eigensolver (sym_eig_jacobi)
# ---------------------------------------------------------------------------


def test_sym_eig_jacobi_matches_numpy():
    """Fixed-sweep cyclic Jacobi reproduces ``numpy.linalg.eigh`` (eigenvalues
    and a reconstruction) for small symmetric matrices.

    The oracle is numpy (CPU): ``sym_eig_jacobi`` exists precisely to avoid
    ``jnp.linalg.eigh`` (``syevd`` cuSOLVER), so the reference must not be it.
    """
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 4, 5, 6):
        m = rng.standard_normal((n, n))
        a = m @ m.T + 0.1 * np.eye(n)  # SPD, distinct eigenvalues
        d, vecs = sym_eig_jacobi(jnp.asarray(a), n)
        d, vecs = np.asarray(d), np.asarray(vecs)
        # eigenvalues match (Jacobi does not sort -> compare sorted)
        np.testing.assert_allclose(
            np.sort(d), np.linalg.eigvalsh(a), atol=1e-10
        )
        # orthonormal eigenvectors and an exact reconstruction
        np.testing.assert_allclose(vecs.T @ vecs, np.eye(n), atol=1e-10)
        np.testing.assert_allclose(
            vecs @ np.diag(d) @ vecs.T, a, atol=1e-9
        )


def test_sym_eig_jacobi_handles_diagonal_and_degenerate():
    """Exact-diagonal input (the ``apq == 0`` rotation guard) and a repeated
    eigenvalue both decompose cleanly."""
    # exact diagonal: no rotation should fire, eigenvalues are the diagonal
    d, vecs = sym_eig_jacobi(jnp.asarray(np.diag([5.0, 2.0, 9.0])), 3)
    np.testing.assert_allclose(np.sort(np.asarray(d)), [2.0, 5.0, 9.0], atol=1e-12)
    # degenerate (repeated eigenvalue): still orthonormal + reconstructs
    a = np.diag([3.0, 3.0, 7.0]) + 0.0
    d, vecs = sym_eig_jacobi(jnp.asarray(a), 3)
    d, vecs = np.asarray(d), np.asarray(vecs)
    np.testing.assert_allclose(vecs @ np.diag(d) @ vecs.T, a, atol=1e-10)
    np.testing.assert_allclose(vecs.T @ vecs, np.eye(3), atol=1e-10)


def test_sym_eig_jacobi_vmaps_and_is_cusolver_free():
    """The eigensolver runs under ``vmap`` and its compiled HLO issues no
    cuSOLVER custom-call (the property that lets ``lme_f_contrast`` form its
    denominator-df eigendirections inside a per-voxel batched computation)."""
    rng = np.random.default_rng(1)
    batch = np.stack(
        [
            (lambda b: b @ b.T + 0.05 * np.eye(3))(rng.standard_normal((3, 3)))
            for _ in range(64)
        ]
    )
    f = jax.jit(jax.vmap(lambda m: sym_eig_jacobi(m, 3)))
    d, vecs = f(jnp.asarray(batch))
    rec = np.einsum('vij,vj,vkj->vik', np.asarray(vecs), np.asarray(d), np.asarray(vecs))
    np.testing.assert_allclose(rec, batch, atol=1e-9)
    hlo = f.lower(jnp.asarray(batch)).compile().as_text()
    assert not _cusolver_calls(hlo), (
        f'sym_eig_jacobi must be cuSOLVER-free; found {_cusolver_calls(hlo)}'
    )


def test_reml_general_p_runs_and_is_finite():
    """A ``p = 5`` fixed-effect design fits (the general hand-Cholesky path)."""
    rng = np.random.default_rng(2)
    g, n_per, V = 6, 30, 4
    N = g * n_per
    group = np.repeat(np.arange(g), n_per)
    X = jnp.asarray(
        np.column_stack([np.ones(N)] + [rng.standard_normal(N) for _ in range(4)])
    )
    Z = np.zeros((N, g))
    for i in range(g):
        Z[group == i, i] = 1.0
    Z = jnp.asarray(Z)
    Y = jnp.asarray(rng.standard_normal((V, N)))
    result = reml_fit(Y, X, Z, n_iter=40)
    assert result.beta_hat.shape == (V, 5)
    assert bool(jnp.all(jnp.isfinite(result.beta_hat)))
    assert bool(jnp.all(jnp.isfinite(result.theta_hat)))
