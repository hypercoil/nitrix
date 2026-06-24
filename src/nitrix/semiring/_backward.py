# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
JAX-side backward rules for the built-in semiring algebras.

Per SPEC §4.1 (differentiability vocabulary), each built-in
ships with a hand-derived backward in the same algebra family.
Per IMPLEMENTATION_PLAN §5.2 (Phase 2.A.5 — SERIAL for G1), backward
is JAX-only at first GA; per-algebra Pallas backwards (2.A.7) come
after the forward Pallas path lands.

Returned shapes
---------------

For ``semiring_matmul`` with ``A: (m, k)`` and ``B: (k, n)``:

    grad_A: (m, k)
    grad_B: (k, n)

For ``semiring_ell_matmul`` with ``values: (m, k_max)``,
``indices: (m, k_max)``, ``B: (n_cols, ncol)``:

    grad_values: (m, k_max)
    grad_B:      (n_cols, ncol)

Numerical-stability notes per algebra
-------------------------------------

- **REAL**  - linear; backward is two transpose-matmuls and is as
  well-conditioned as the forward.
- **LOG**   - the softmax weight
  ``w[i, k, j] = exp(A[i, k] + B[k, j] - C[i, j])`` is bounded in
  ``[0, 1]`` whenever ``C`` is the true logsumexp; the streaming K
  loop keeps every intermediate at ``(M, N)``.  ``-inf`` rows
  propagate through ``_safe_exp_diff`` from ``algebras.py``.
- **TROPICAL_MAX/MIN_PLUS**  - subgradient; we route the upstream
  gradient through the argmax / argmin one-hot.  Ties are broken in
  favour of the *first* maximiser encountered (consistent with
  ``jnp.argmax`` default behaviour).
- **EUCLIDEAN**  - the gradient has a ``1 / C[i, j]`` factor that
  is undefined at ``C = 0``.  We use ``where(C > eps, ..., 0)`` so
  the backward sees a clean zero at the singularity rather than a
  ``nan``.  ``eps`` is the dtype-aware square-root sentinel.
- **BOOLEAN** - not differentiable; backward raises.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, cast

import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Num

from ._reference import reference_semiring_ell_rmatvec

# Residual / gradient tuple shapes (shape strings are documentation; the
# 4th ELL residual is ``C`` for the inexact algebras and ``n_cols`` for
# REAL -- it is unpacked into an unused slot there, so the alias fits).
_DenseResiduals = Tuple[
    Num[Array, 'm k'], Num[Array, 'k n'], Num[Array, 'm n']
]
_DenseGrads = Tuple[Num[Array, 'm k'], Num[Array, 'k n']]
_ELLResiduals = Tuple[
    Num[Array, 'm kmax'],
    Int[Array, 'm kmax'],
    Num[Array, 'n_cols ncol'],
    Num[Array, 'm ncol'],
]
_ELLGrads = Tuple[Num[Array, 'm kmax'], Num[Array, 'n_cols ncol']]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_div(
    num: Num[Array, '*shape'],
    den: Num[Array, '*shape'],
    eps: Optional[float] = None,
) -> Num[Array, '*shape']:
    """``num / den`` defined to be 0 wherever ``|den| <= eps``.

    Uses the "double-where with sentinel" trick to keep both forward
    and reverse-mode AD NaN-free.  ``eps`` defaults to a dtype-aware
    small constant.
    """
    if eps is None:
        eps = jnp.finfo(den.dtype).tiny * 4
    safe = jnp.abs(den) > eps
    safe_den = jnp.where(safe, den, jnp.ones_like(den))
    return jnp.where(safe, num / safe_den, jnp.zeros_like(num))


def _safe_exp_diff(
    x: Float[Array, '*shape'], m: Float[Array, '*shape']
) -> Float[Array, '*shape']:
    """``exp(x - m)``, with ``-inf`` rows propagating cleanly to 0.

    A copy of the ``algebras._safe_exp_diff`` helper, kept local so
    the backward module doesn't import from the forward algebra
    module (which would form a small cycle).
    """
    finite = jnp.isfinite(x)
    safe_diff = jnp.where(finite, x - m, jnp.zeros_like(x))
    return jnp.where(finite, jnp.exp(safe_diff), jnp.zeros_like(x))


# ===========================================================================
# REAL — inner product / linear matmul
# ===========================================================================


def real_matmul_vjp(
    residuals: _DenseResiduals, g_out: Num[Array, 'm n']
) -> _DenseGrads:
    """Backward of ``C = A @ B`` under the real semiring (2-D).

    Two transpose-matmuls; reuses ``jnp.matmul`` for the heavy lift
    (Pallas backward kernels are 2.A.7).  Batched inputs are handled
    upstream via ``jax.vmap``; this function operates on the 2-D
    core ``A: (m, k)``, ``B: (k, n)`` directly.
    """
    A, B, _C = residuals
    g_A = jnp.matmul(g_out, B.T, preferred_element_type=A.dtype)
    g_B = jnp.matmul(A.T, g_out, preferred_element_type=B.dtype)
    return g_A, g_B


# ===========================================================================
# LOG — softmax-weighted, streaming K
# ===========================================================================


def log_matmul_vjp(
    residuals: _DenseResiduals, g_out: Num[Array, 'm n']
) -> _DenseGrads:
    """Backward of ``C[i,j] = lse_k (A[i,k] + B[k,j])`` under the log semiring.

    The softmax weight ``w[i, k, j] = exp(A[i, k] + B[k, j] - C[i, j])``
    is bounded in ``[0, 1]`` because ``C`` is the logsumexp.  We
    stream over K so the per-step intermediate is only ``(M, N)``:

        for k:
            log_w_k = A[:, k:k+1] + B[k:k+1, :] - C            # (M, N)
            w_k     = safe_exp(log_w_k)
            contrib = g_out * w_k                                # (M, N)
            grad_A[:, k] = contrib.sum(axis=1)
            grad_B[k, :] = contrib.sum(axis=0)
    """
    A, B, C = residuals
    M, K = A.shape
    _, N = B.shape

    def body(kk: Int[Array, ''], carry: _DenseGrads) -> _DenseGrads:
        gA, gB = carry
        a_col = lax.dynamic_slice_in_dim(A, kk, 1, axis=1)  # (M, 1)
        b_row = lax.dynamic_slice_in_dim(B, kk, 1, axis=0)  # (1, N)
        log_w = a_col + b_row - C  # (M, N)
        w = _safe_exp_diff(log_w, jnp.zeros_like(log_w))
        contrib = g_out * w  # (M, N)
        gA = lax.dynamic_update_slice_in_dim(
            gA,
            contrib.sum(axis=1, keepdims=True),
            kk,
            axis=1,
        )
        gB = lax.dynamic_update_slice_in_dim(
            gB,
            contrib.sum(axis=0, keepdims=True),
            kk,
            axis=0,
        )
        return gA, gB

    gA0 = jnp.zeros_like(A)
    gB0 = jnp.zeros_like(B)
    gA, gB = lax.fori_loop(0, K, body, (gA0, gB0))
    return gA, gB


# ===========================================================================
# TROPICAL_MAX_PLUS — argmax-gather subgradient
# ===========================================================================


def _tropical_argk(
    A: Num[Array, 'm k'],
    B: Num[Array, 'k n'],
    *,
    monoid_init: float,
    better: Callable[
        [Num[Array, 'm n'], Num[Array, 'm n']], Bool[Array, 'm n']
    ],
) -> Int[Array, 'm n']:
    """Compute the per-(i, j) optimal k via a streaming K loop.

    ``monoid_init`` is the initial ``best_val`` (-inf for max, +inf
    for min).  ``better(cur, best)`` returns a boolean mask of where
    ``cur`` is strictly better than ``best``.  Ties are broken in
    favour of the smaller k (consistent with ``jnp.argmax``).
    """
    M, K = A.shape
    _, N = B.shape

    def body(
        kk: Int[Array, ''],
        state: Tuple[Num[Array, 'm n'], Int[Array, 'm n']],
    ) -> Tuple[Num[Array, 'm n'], Int[Array, 'm n']]:
        best_val, best_k = state
        a_col = lax.dynamic_slice_in_dim(A, kk, 1, axis=1)  # (M, 1)
        b_row = lax.dynamic_slice_in_dim(B, kk, 1, axis=0)  # (1, N)
        cur = a_col + b_row  # (M, N)
        is_better = better(cur, best_val)
        best_val = jnp.where(is_better, cur, best_val)
        best_k = jnp.where(is_better, kk, best_k)
        return best_val, best_k

    best_val0 = jnp.full((M, N), monoid_init, dtype=A.dtype)
    best_k0 = jnp.zeros((M, N), dtype=jnp.int32)
    # ``lax.fori_loop`` is typed as returning Any; restore the index array.
    _, best_k = lax.fori_loop(0, K, body, (best_val0, best_k0))
    return cast(Int[Array, 'm n'], best_k)


def _tropical_route(
    g_out: Num[Array, 'm n'], best_k: Int[Array, 'm n'], K: int
) -> _DenseGrads:
    """Route ``g_out`` through the one-hot ``[k == best_k[i, j]]``.

    Returns ``(gA, gB)`` of shapes ``(M, K)`` and ``(K, N)``
    respectively.  Implemented as a streaming K loop so we don't
    materialise the ``(M, K, N)`` one-hot mask.
    """
    M, N = g_out.shape

    def body(kk: Int[Array, ''], carry: _DenseGrads) -> _DenseGrads:
        gA, gB = carry
        mask = (best_k == kk).astype(g_out.dtype)  # (M, N)
        contrib = g_out * mask
        gA = lax.dynamic_update_slice_in_dim(
            gA,
            contrib.sum(axis=1, keepdims=True),
            kk,
            axis=1,
        )
        gB = lax.dynamic_update_slice_in_dim(
            gB,
            contrib.sum(axis=0, keepdims=True),
            kk,
            axis=0,
        )
        return gA, gB

    gA0 = jnp.zeros((M, K), dtype=g_out.dtype)
    gB0 = jnp.zeros((K, N), dtype=g_out.dtype)
    gA, gB = lax.fori_loop(0, K, body, (gA0, gB0))
    return gA, gB


def tropical_max_plus_matmul_vjp(
    residuals: _DenseResiduals, g_out: Num[Array, 'm n']
) -> _DenseGrads:
    """Argmax-gather subgradient for max-plus.

    ``∂C[i,j]/∂A[i,k] = 1 iff k = argmax_k (A[i,k] + B[k,j])``;
    same for ``B`` by symmetry.  Subgradient because the max is not
    differentiable at ties.
    """
    A, B, _C = residuals
    best_k = _tropical_argk(
        A,
        B,
        monoid_init=-jnp.inf,
        better=lambda cur, best: cur > best,
    )
    return _tropical_route(g_out, best_k, A.shape[1])


def tropical_min_plus_matmul_vjp(
    residuals: _DenseResiduals, g_out: Num[Array, 'm n']
) -> _DenseGrads:
    """Argmin-gather subgradient for min-plus."""
    A, B, _C = residuals
    best_k = _tropical_argk(
        A,
        B,
        monoid_init=jnp.inf,
        better=lambda cur, best: cur < best,
    )
    return _tropical_route(g_out, best_k, A.shape[1])


# ===========================================================================
# EUCLIDEAN — normalised-difference, sqrt-singularity-guarded
# ===========================================================================


def euclidean_matmul_vjp(
    residuals: _DenseResiduals, g_out: Num[Array, 'm n']
) -> _DenseGrads:
    """Backward of ``C[i,j] = sqrt(sum_k (A[i,k] - B[k,j])**2)``.

    Closed-form gradient with a ``1 / C`` factor; we guard at
    ``C[i, j] <= eps`` and set the contribution to 0 there (the
    function has a corner at zero distance, where the gradient is
    not defined; zero is the conventional choice).
    """
    A, B, C = residuals
    # h[i, j] = g_out[i, j] / C[i, j], safe at C ≈ 0
    h = _safe_div(g_out, C)
    # gA[i, k] = sum_j h[i, j] * (A[i, k] - B[k, j])
    #         = A[i, k] * sum_j h[i, j] - sum_j h[i, j] * B[k, j]
    #         = A * h.sum(j, keepdims) - h @ B.T
    grad_A = A * h.sum(axis=1, keepdims=True) - jnp.matmul(h, B.T)
    # gB[k, j] = sum_i h[i, j] * (B[k, j] - A[i, k])
    #         = B[k, j] * sum_i h[i, j] - sum_i h[i, j] * A[i, k]
    #         = B * h.sum(i, keepdims) - A.T @ h
    grad_B = B * h.sum(axis=0, keepdims=True) - jnp.matmul(A.T, h)
    return grad_A, grad_B


# ===========================================================================
# BOOLEAN — not differentiable
# ===========================================================================


def boolean_matmul_vjp(
    residuals: _DenseResiduals, g_out: Num[Array, 'm n']
) -> _DenseGrads:
    raise TypeError(
        'semiring=BOOLEAN is not differentiable.  Use REAL (or LOG '
        'with a soft-max relaxation) if you need a gradient.'
    )


# ===========================================================================
# ELL backwards
# ===========================================================================


def real_ell_matmul_vjp(
    residuals: _ELLResiduals, g_out: Num[Array, 'm ncol']
) -> _ELLGrads:
    """Backward of ``C[i,j] = sum_p values[i, p] * B[indices[i, p], j]``.

    Returns ``(grad_values, grad_B)``; ``indices`` is non-diff (passed
    through ``custom_vjp`` ``nondiff_argnums``).

    Gradient w.r.t. ``values``:
        grad_values[i, p] = sum_j g_out[i, j] * B[indices[i, p], j]

    Gradient w.r.t. ``B`` (scatter-add over the indices):
        grad_B[k, j]
            = sum_{(i, p) : indices[i, p] == k} values[i, p] * g_out[i, j]

    Implemented as a fori over p so the intermediates are ``(M, N)``
    rather than ``(M, k_max, N)``.
    """
    # REAL needed only as a call-time value (the gather/scatter is plain
    # arithmetic); the local import sidesteps the algebras -> _backward
    # load-time cycle, matching the package's deferred-import idiom.
    from .algebras import REAL

    values, indices, B, _n_cols = residuals
    _, kmax = values.shape
    n_cols, _ = B.shape

    def body(p: Int[Array, ''], g_values: Num[Array, 'm kmax']) -> Num[
        Array, 'm kmax'
    ]:
        idx_p = lax.dynamic_slice_in_dim(indices, p, 1, axis=1)[:, 0]  # (M,)
        gathered = B[idx_p]  # (M, N)
        # g_values[i, p] = (g_out[i, :] * gathered[i, :]).sum()
        gv_p = (g_out * gathered).sum(axis=1, keepdims=True)  # (M, 1)
        return lax.dynamic_update_slice_in_dim(g_values, gv_p, p, axis=1)

    g_values = lax.fori_loop(0, kmax, body, jnp.zeros_like(values))
    # grad_B is exactly the additive ELL adjoint Aᵀ @ g_out -- the single
    # source of truth shared with the symmetric matvec (see
    # reference_semiring_ell_rmatvec).  Cast back to B's dtype so the
    # returned cotangent dtype matches the primal (result_type may promote).
    g_B = reference_semiring_ell_rmatvec(
        values, indices, g_out, semiring=REAL, n_cols=n_cols
    ).astype(B.dtype)
    return g_values, g_B


def real_ell_rmatvec_vjp(
    residuals: Tuple[
        Num[Array, 'm kmax'], Int[Array, 'm kmax'], Num[Array, 'm ncol']
    ],
    g_out: Num[Array, 'n_cols ncol'],
) -> Tuple[Num[Array, 'm kmax'], Num[Array, 'm ncol']]:
    """Backward of ``Y = Aᵀ X`` (``reference_semiring_ell_rmatvec``, REAL).

    Returns ``(grad_values, grad_X)``; ``indices`` is non-diff (the public
    ``semiring_ell_rmatvec`` returns a zero for it).  The rmatvec is the
    adjoint of the gather matmul, so its own backward is the *matmul*
    forward plus the same gather-dot used for ``grad_values`` above:

        grad_X[i, j]      = Σ_p values[i, p] · g_out[indices[i, p], j]   (= A @ g_out)
        grad_values[i, p] = Σ_j X[i, j] · g_out[indices[i, p], j]
    """
    values, indices, X = residuals
    _, kmax = values.shape

    def body(
        p: Int[Array, ''],
        carry: Tuple[Num[Array, 'm kmax'], Num[Array, 'm ncol']],
    ) -> Tuple[Num[Array, 'm kmax'], Num[Array, 'm ncol']]:
        g_values, g_X = carry
        idx_p = lax.dynamic_slice_in_dim(indices, p, 1, axis=1)[:, 0]  # (M,)
        v_p = lax.dynamic_slice_in_dim(values, p, 1, axis=1)  # (M, 1)
        g_at = g_out[idx_p]  # (M, ncol)
        # grad_X gathers the gradient back through A; grad_values is the
        # per-edge gather-dot of X with the scattered cotangent.
        g_X = g_X + v_p * g_at
        gv_p = (X * g_at).sum(axis=1, keepdims=True)  # (M, 1)
        g_values = lax.dynamic_update_slice_in_dim(g_values, gv_p, p, axis=1)
        return g_values, g_X

    g_values0 = jnp.zeros_like(values)
    g_X0 = jnp.zeros_like(X)
    g_values, g_X = lax.fori_loop(0, kmax, body, (g_values0, g_X0))
    return g_values, g_X


def log_ell_matmul_vjp(
    residuals: _ELLResiduals, g_out: Num[Array, 'm ncol']
) -> _ELLGrads:
    """Backward of ``C[i,j] = lse_p (values[i, p] + B[indices[i, p], j])``.

    Softmax weight per ``(i, p, j)``::

        w[i, p, j] = exp(values[i, p] + B[indices[i, p], j] - C[i, j])

    Bounded in ``[0, 1]`` because ``C`` is the logsumexp.  Streamed
    over ``p`` so intermediates stay at ``(M, N)``.

    Gradient w.r.t. ``values``: sum over j of g_out * w
    Gradient w.r.t. ``B``: scatter-add over indices of g_out * w
    """
    values, indices, B, C = residuals
    m, kmax = values.shape
    n_cols, ncol = B.shape

    def body(p: Int[Array, ''], carry: _ELLGrads) -> _ELLGrads:
        g_values, g_B = carry
        idx_p = lax.dynamic_slice_in_dim(indices, p, 1, axis=1)[:, 0]  # (M,)
        v_p = lax.dynamic_slice_in_dim(values, p, 1, axis=1)  # (M, 1)
        b_rows = B[idx_p]  # (M, N)
        log_w = v_p + b_rows - C  # (M, N)
        w = _safe_exp_diff(log_w, jnp.zeros_like(log_w))
        contrib = g_out * w
        gv_p = contrib.sum(axis=1, keepdims=True)  # (M, 1)
        g_values = lax.dynamic_update_slice_in_dim(
            g_values,
            gv_p,
            p,
            axis=1,
        )
        g_B = g_B.at[idx_p].add(contrib)
        return g_values, g_B

    g_values0 = jnp.zeros_like(values)
    g_B0 = jnp.zeros_like(B)
    g_values, g_B = lax.fori_loop(0, kmax, body, (g_values0, g_B0))
    return g_values, g_B


def tropical_max_plus_ell_matmul_vjp(
    residuals: _ELLResiduals, g_out: Num[Array, 'm ncol']
) -> _ELLGrads:
    """Argmax-gather subgradient for ELL max-plus.

    Per output ``(i, j)``, the upstream gradient is routed entirely
    to the column ``p*[i, j]`` that achieved the maximum.  Same
    streaming pattern as the dense case but with per-row gather of
    ``B[indices[i, p], j]`` inside the K loop.
    """
    values, indices, B, _C = residuals
    m, kmax = values.shape
    n_cols, ncol = B.shape

    def body(
        p: Int[Array, ''],
        state: Tuple[Num[Array, 'm ncol'], Int[Array, 'm ncol']],
    ) -> Tuple[Num[Array, 'm ncol'], Int[Array, 'm ncol']]:
        best_val, best_p = state
        idx_p = lax.dynamic_slice_in_dim(indices, p, 1, axis=1)[:, 0]
        v_p = lax.dynamic_slice_in_dim(values, p, 1, axis=1)
        b_rows = B[idx_p]  # (M, N)
        cur = v_p + b_rows  # (M, N)
        is_better = cur > best_val
        best_val = jnp.where(is_better, cur, best_val)
        best_p = jnp.where(is_better, p, best_p)
        return best_val, best_p

    best_val0 = jnp.full((m, ncol), -jnp.inf, dtype=values.dtype)
    best_p0 = jnp.zeros((m, ncol), dtype=jnp.int32)
    _, best_p = lax.fori_loop(0, kmax, body, (best_val0, best_p0))

    def route(p: Int[Array, ''], carry: _ELLGrads) -> _ELLGrads:
        g_values, g_B = carry
        mask = (best_p == p).astype(g_out.dtype)  # (M, N)
        contrib = g_out * mask  # (M, N)
        gv_p = contrib.sum(axis=1, keepdims=True)
        g_values = lax.dynamic_update_slice_in_dim(
            g_values,
            gv_p,
            p,
            axis=1,
        )
        idx_p = lax.dynamic_slice_in_dim(indices, p, 1, axis=1)[:, 0]
        g_B = g_B.at[idx_p].add(contrib)
        return g_values, g_B

    g_values0 = jnp.zeros_like(values)
    g_B0 = jnp.zeros_like(B)
    g_values, g_B = lax.fori_loop(0, kmax, route, (g_values0, g_B0))
    return g_values, g_B


def tropical_min_plus_ell_matmul_vjp(
    residuals: _ELLResiduals, g_out: Num[Array, 'm ncol']
) -> _ELLGrads:
    values, indices, B, _C = residuals
    m, kmax = values.shape
    n_cols, ncol = B.shape

    def body(
        p: Int[Array, ''],
        state: Tuple[Num[Array, 'm ncol'], Int[Array, 'm ncol']],
    ) -> Tuple[Num[Array, 'm ncol'], Int[Array, 'm ncol']]:
        best_val, best_p = state
        idx_p = lax.dynamic_slice_in_dim(indices, p, 1, axis=1)[:, 0]
        v_p = lax.dynamic_slice_in_dim(values, p, 1, axis=1)
        b_rows = B[idx_p]
        cur = v_p + b_rows
        is_better = cur < best_val
        best_val = jnp.where(is_better, cur, best_val)
        best_p = jnp.where(is_better, p, best_p)
        return best_val, best_p

    best_val0 = jnp.full((m, ncol), jnp.inf, dtype=values.dtype)
    best_p0 = jnp.zeros((m, ncol), dtype=jnp.int32)
    _, best_p = lax.fori_loop(0, kmax, body, (best_val0, best_p0))

    def route(p: Int[Array, ''], carry: _ELLGrads) -> _ELLGrads:
        g_values, g_B = carry
        mask = (best_p == p).astype(g_out.dtype)
        contrib = g_out * mask
        gv_p = contrib.sum(axis=1, keepdims=True)
        g_values = lax.dynamic_update_slice_in_dim(
            g_values,
            gv_p,
            p,
            axis=1,
        )
        idx_p = lax.dynamic_slice_in_dim(indices, p, 1, axis=1)[:, 0]
        g_B = g_B.at[idx_p].add(contrib)
        return g_values, g_B

    g_values0 = jnp.zeros_like(values)
    g_B0 = jnp.zeros_like(B)
    g_values, g_B = lax.fori_loop(0, kmax, route, (g_values0, g_B0))
    return g_values, g_B


def euclidean_ell_matmul_vjp(
    residuals: _ELLResiduals, g_out: Num[Array, 'm ncol']
) -> _ELLGrads:
    """Backward of ``C[i,j] = sqrt(sum_p (values[i, p] - B[indices[i, p], j])**2)``.

    Streamed over p; per-step intermediate is ``(M, N)``.  Same
    sqrt-singularity guard as the dense case.
    """
    values, indices, B, C = residuals
    m, kmax = values.shape
    n_cols, ncol = B.shape

    h = _safe_div(g_out, C)  # (M, N)

    def body(p: Int[Array, ''], carry: _ELLGrads) -> _ELLGrads:
        g_values, g_B = carry
        idx_p = lax.dynamic_slice_in_dim(indices, p, 1, axis=1)[:, 0]
        v_p = lax.dynamic_slice_in_dim(values, p, 1, axis=1)  # (M, 1)
        b_rows = B[idx_p]  # (M, N)
        diff = v_p - b_rows  # (M, N)
        # grad_values[i, p] = sum_j h[i, j] * (values[i, p] - B[idx_p, j])
        gv_p = (h * diff).sum(axis=1, keepdims=True)  # (M, 1)
        g_values = lax.dynamic_update_slice_in_dim(
            g_values,
            gv_p,
            p,
            axis=1,
        )
        # grad_B[idx_p, j] += -h[i, j] * (values[i, p] - B[idx_p, j])
        contrib_B = -h * diff  # (M, N)
        g_B = g_B.at[idx_p].add(contrib_B)
        return g_values, g_B

    g_values0 = jnp.zeros_like(values)
    g_B0 = jnp.zeros_like(B)
    g_values, g_B = lax.fori_loop(0, kmax, body, (g_values0, g_B0))
    return g_values, g_B


def boolean_ell_matmul_vjp(
    residuals: _ELLResiduals, g_out: Num[Array, 'm ncol']
) -> _ELLGrads:
    raise TypeError(
        'semiring=BOOLEAN is not differentiable.  Use REAL (or LOG) '
        'if you need a gradient through ELL contraction.'
    )


__all__ = [
    'real_matmul_vjp',
    'log_matmul_vjp',
    'tropical_max_plus_matmul_vjp',
    'tropical_min_plus_matmul_vjp',
    'euclidean_matmul_vjp',
    'boolean_matmul_vjp',
    'real_ell_matmul_vjp',
    'real_ell_rmatvec_vjp',
    'log_ell_matmul_vjp',
    'tropical_max_plus_ell_matmul_vjp',
    'tropical_min_plus_ell_matmul_vjp',
    'euclidean_ell_matmul_vjp',
    'boolean_ell_matmul_vjp',
]
