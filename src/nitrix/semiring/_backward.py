# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
JAX-side backward rules for the built-in semiring algebras.

Each built-in algebra ships with a hand-derived backward rule in the same
algebra family, so that gradients respect the semiring's own arithmetic.
These rules are pure JAX; there is currently no per-algebra fused
(Pallas) backward, so the reverse pass falls back to JAX even where the
forward pass has a fused kernel.

Returned shapes
---------------

For :func:`~nitrix.semiring.semiring_matmul` with :math:`A` of shape
``(m, k)`` and :math:`B` of shape ``(k, n)``:

    grad_A: (m, k)
    grad_B: (k, n)

For :func:`~nitrix.semiring.semiring_ell_matmul` with ``values`` of shape
``(m, k_max)``, ``indices`` of shape ``(m, k_max)``, and :math:`B` of
shape ``(n_cols, ncol)``:

    grad_values: (m, k_max)
    grad_B:      (n_cols, ncol)

Numerical-stability notes per algebra
-------------------------------------

- **REAL** — linear; the backward is two transpose-matmuls and is as
  well-conditioned as the forward.
- **LOG** — the softmax weight
  :math:`w_{ikj} = \\exp(A_{ik} + B_{kj} - C_{ij})` is bounded in
  :math:`[0, 1]` whenever :math:`C` is the true log-sum-exp; the
  streaming K loop keeps every intermediate at shape ``(M, N)``.
  Rows containing :math:`-\\infty` propagate cleanly through
  :func:`_safe_exp_diff`.
- **TROPICAL_MAX/MIN_PLUS** — subgradient; the upstream gradient is
  routed through the argmax / argmin one-hot.  Ties are broken in
  favour of the *first* maximiser encountered (consistent with
  ``jnp.argmax`` default behaviour).
- **EUCLIDEAN** — the gradient carries a :math:`1 / C_{ij}` factor that
  is undefined at :math:`C = 0`.  A masked division sets the
  contribution to zero at the singularity rather than producing a
  ``nan``; the guard threshold is a dtype-aware small sentinel.
- **BOOLEAN** — not differentiable; the backward raises.
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
    """Elementwise division defined to be zero wherever the denominator vanishes.

    Computes ``num / den`` but returns zero wherever
    :math:`|\\mathrm{den}| \\le \\mathrm{eps}`.  Uses the
    "double-where with sentinel" trick — replacing the denominator with
    ones on the masked-out entries before dividing — to keep both the
    forward value and its reverse-mode gradient free of ``nan``.

    Parameters
    ----------
    num : Num[Array, '*shape']
        Numerator array.
    den : Num[Array, '*shape']
        Denominator array, broadcast-compatible with ``num``.
    eps : float, optional
        Magnitude below which a denominator entry is treated as zero.
        When ``None`` (the default), a dtype-aware small constant
        (``jnp.finfo(den.dtype).tiny * 4``) is used.

    Returns
    -------
    Num[Array, '*shape']
        The elementwise quotient, with zeros wherever
        :math:`|\\mathrm{den}| \\le \\mathrm{eps}`.
    """
    if eps is None:
        eps = jnp.finfo(den.dtype).tiny * 4
    safe = jnp.abs(den) > eps
    safe_den = jnp.where(safe, den, jnp.ones_like(den))
    return jnp.where(safe, num / safe_den, jnp.zeros_like(num))


def _safe_exp_diff(
    x: Float[Array, '*shape'], m: Float[Array, '*shape']
) -> Float[Array, '*shape']:
    """Compute :math:`\\exp(x - m)`, mapping non-finite entries cleanly to zero.

    Evaluates :math:`\\exp(x - m)` elementwise, but returns zero wherever
    ``x`` is not finite, so that :math:`-\\infty` entries (empty log-space
    rows) propagate to a hard zero rather than a ``nan``.  This is a local
    copy of the forward algebra module's helper of the same name, kept here
    so the backward module does not import from the forward module and
    thereby form an import cycle.

    Parameters
    ----------
    x : Float[Array, '*shape']
        Log-space values to exponentiate after subtracting ``m``.
    m : Float[Array, '*shape']
        Log-space offset (typically the per-entry maximum or the
        log-sum-exp), broadcast-compatible with ``x``.

    Returns
    -------
    Float[Array, '*shape']
        :math:`\\exp(x - m)` where ``x`` is finite, and zero elsewhere.
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
    """Backward of the real-semiring matmul :math:`C = A B` (2-D core).

    The real semiring is ordinary matrix multiplication, so the reverse
    rule is two transpose-matmuls,
    :math:`g_A = g_{\\mathrm{out}} B^{\\top}` and
    :math:`g_B = A^{\\top} g_{\\mathrm{out}}`, delegated to ``jnp.matmul``.
    Batched inputs are handled upstream via ``jax.vmap``; this function
    operates directly on the 2-D core.

    Parameters
    ----------
    residuals : tuple
        The forward residuals ``(A, B, C)``: the left operand :math:`A`
        of shape ``(m, k)``, the right operand :math:`B` of shape
        ``(k, n)``, and the (here unused) forward output :math:`C` of
        shape ``(m, n)``.
    g_out : Num[Array, 'm n']
        Upstream cotangent with respect to the output :math:`C`.

    Returns
    -------
    tuple of Array
        The pair ``(grad_A, grad_B)`` of cotangents, of shapes ``(m, k)``
        and ``(k, n)`` respectively.
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
    """Backward of the log-semiring matmul :math:`C_{ij} = \\log\\sum_k \\exp(A_{ik} + B_{kj})`.

    The gradient routes the upstream cotangent through the softmax weight
    :math:`w_{ikj} = \\exp(A_{ik} + B_{kj} - C_{ij})`, which is bounded in
    :math:`[0, 1]` because :math:`C` is the log-sum-exp over ``k``.  The
    computation streams over the contraction axis ``k`` so that each
    per-step intermediate is only ``(M, N)`` rather than materialising the
    full ``(M, K, N)`` weight tensor::

        for k:
            log_w_k = A[:, k:k+1] + B[k:k+1, :] - C            # (M, N)
            w_k     = safe_exp(log_w_k)
            contrib = g_out * w_k                              # (M, N)
            grad_A[:, k] = contrib.sum(axis=1)
            grad_B[k, :] = contrib.sum(axis=0)

    Parameters
    ----------
    residuals : tuple
        The forward residuals ``(A, B, C)``: the left operand :math:`A`
        of shape ``(m, k)``, the right operand :math:`B` of shape
        ``(k, n)``, and the forward output :math:`C` of shape ``(m, n)``
        (the log-sum-exp, reused as the softmax normaliser).
    g_out : Num[Array, 'm n']
        Upstream cotangent with respect to the output :math:`C`.

    Returns
    -------
    tuple of Array
        The pair ``(grad_A, grad_B)`` of cotangents, of shapes ``(m, k)``
        and ``(k, n)`` respectively.
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
    """Find, for each output entry :math:`(i, j)`, the optimal contraction index k.

    Sweeps the contraction axis ``k`` in a streaming loop, tracking the
    running best value :math:`A_{ik} + B_{kj}` and the index ``k`` that
    achieved it.  Ties are broken in favour of the smaller ``k``, since a
    later index only replaces the incumbent when it is *strictly* better
    (consistent with ``jnp.argmax`` default behaviour).

    Parameters
    ----------
    A : Num[Array, 'm k']
        Left operand.
    B : Num[Array, 'k n']
        Right operand.
    monoid_init : float
        Initial value of the running best: :math:`-\\infty` for a max
        (argmax) sweep, :math:`+\\infty` for a min (argmin) sweep.
    better : callable
        Predicate ``better(cur, best)`` returning a boolean mask of the
        entries where the candidate ``cur`` is strictly better than the
        incumbent ``best`` (``cur > best`` for max, ``cur < best`` for
        min).

    Returns
    -------
    Int[Array, 'm n']
        The optimal contraction index ``k`` per output entry, as int32.
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
    """Route the upstream gradient through the tropical argmax/argmin one-hot.

    For each output entry :math:`(i, j)`, the whole upstream gradient
    ``g_out[i, j]`` is sent to the single contraction index
    ``best_k[i, j]`` that achieved the optimum, and to zero elsewhere —
    this is the subgradient of the tropical (max-plus / min-plus) matmul.
    Implemented as a streaming loop over ``k`` so the ``(M, K, N)``
    one-hot mask is never materialised.

    Parameters
    ----------
    g_out : Num[Array, 'm n']
        Upstream cotangent with respect to the output.
    best_k : Int[Array, 'm n']
        Per-entry winning contraction index, as returned by
        :func:`_tropical_argk`.
    K : int
        Size of the contraction axis (number of columns of :math:`A` /
        rows of :math:`B`).

    Returns
    -------
    tuple of Array
        The pair ``(gA, gB)`` of cotangents, of shapes ``(M, K)`` and
        ``(K, N)`` respectively.
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
    """Argmax-gather subgradient for the dense max-plus matmul.

    Under the max-plus semiring, :math:`C_{ij} = \\max_k (A_{ik} + B_{kj})`,
    so :math:`\\partial C_{ij} / \\partial A_{ik} = 1` exactly when
    :math:`k = \\arg\\max_k (A_{ik} + B_{kj})`, and analogously for
    :math:`B`.  This is a subgradient because the maximum is not
    differentiable at ties; the winning ``k`` is found by
    :func:`_tropical_argk` and the gradient routed by
    :func:`_tropical_route`.

    Parameters
    ----------
    residuals : tuple
        The forward residuals ``(A, B, C)``: the left operand :math:`A`
        of shape ``(m, k)``, the right operand :math:`B` of shape
        ``(k, n)``, and the (here unused) forward output :math:`C` of
        shape ``(m, n)``.
    g_out : Num[Array, 'm n']
        Upstream cotangent with respect to the output :math:`C`.

    Returns
    -------
    tuple of Array
        The pair ``(grad_A, grad_B)`` of cotangents, of shapes ``(m, k)``
        and ``(k, n)`` respectively.
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
    """Argmin-gather subgradient for the dense min-plus matmul.

    Under the min-plus semiring, :math:`C_{ij} = \\min_k (A_{ik} + B_{kj})`,
    so the upstream gradient is routed to the contraction index that
    achieved the minimum (found by :func:`_tropical_argk`, routed by
    :func:`_tropical_route`).  This is a subgradient because the minimum
    is not differentiable at ties.

    Parameters
    ----------
    residuals : tuple
        The forward residuals ``(A, B, C)``: the left operand :math:`A`
        of shape ``(m, k)``, the right operand :math:`B` of shape
        ``(k, n)``, and the (here unused) forward output :math:`C` of
        shape ``(m, n)``.
    g_out : Num[Array, 'm n']
        Upstream cotangent with respect to the output :math:`C`.

    Returns
    -------
    tuple of Array
        The pair ``(grad_A, grad_B)`` of cotangents, of shapes ``(m, k)``
        and ``(k, n)`` respectively.
    """
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
    """Backward of the Euclidean matmul :math:`C_{ij} = \\sqrt{\\sum_k (A_{ik} - B_{kj})^2}`.

    Uses the closed-form gradient of the pairwise Euclidean distance,
    which carries a :math:`1 / C_{ij}` factor.  Where
    :math:`C_{ij} \\le \\mathrm{eps}` the division is guarded (via
    :func:`_safe_div`) and the contribution set to zero: the distance
    has a corner at zero, where the gradient is undefined, and zero is
    the conventional subgradient choice.

    Parameters
    ----------
    residuals : tuple
        The forward residuals ``(A, B, C)``: the left operand :math:`A`
        of shape ``(m, k)``, the right operand :math:`B` of shape
        ``(k, n)``, and the forward output :math:`C` of shape ``(m, n)``
        (the distance, reused in the :math:`1 / C` factor).
    g_out : Num[Array, 'm n']
        Upstream cotangent with respect to the output :math:`C`.

    Returns
    -------
    tuple of Array
        The pair ``(grad_A, grad_B)`` of cotangents, of shapes ``(m, k)``
        and ``(k, n)`` respectively.
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
    """Backward of the real-semiring ELL matmul :math:`C_{ij} = \\sum_p v_{ip}\\, B_{\\mathrm{idx}(i, p), j}`.

    The forward contracts a sparse ELL operand (per-row ``values`` and
    column ``indices``) against the dense operand :math:`B`.  The reverse
    rule is

    .. code-block::

        grad_values[i, p] = sum_j g_out[i, j] * B[indices[i, p], j]
        grad_B[k, j]      = sum_{(i, p): indices[i, p] == k}
                                values[i, p] * g_out[i, j]

    i.e. a per-edge gather-dot for ``grad_values`` and a scatter-add for
    ``grad_B``.  The ``grad_B`` term is the additive ELL adjoint
    :math:`A^{\\top} g_{\\mathrm{out}}`, computed by
    :func:`~nitrix.semiring._reference.reference_semiring_ell_rmatvec` so
    the matmul and the symmetric matvec share a single source of truth.
    The ``grad_values`` loop streams over ``p`` so intermediates are
    ``(M, N)`` rather than ``(M, k_max, N)``.  ``indices`` is
    non-differentiable and receives no cotangent.

    Parameters
    ----------
    residuals : tuple
        The forward residuals ``(values, indices, B, n_cols)``: the ELL
        ``values`` of shape ``(m, kmax)``, the column ``indices`` of
        shape ``(m, kmax)``, the dense operand :math:`B` of shape
        ``(n_cols, ncol)``, and the row count ``n_cols`` (unused here).
    g_out : Num[Array, 'm ncol']
        Upstream cotangent with respect to the output :math:`C`.

    Returns
    -------
    tuple of Array
        The pair ``(grad_values, grad_B)`` of cotangents, of shapes
        ``(m, kmax)`` and ``(n_cols, ncol)`` respectively.
    """
    # REAL needed only as a call-time value (the gather/scatter is plain
    # arithmetic); the local import sidesteps the algebras -> _backward
    # load-time cycle, matching the package's deferred-import idiom.
    from .algebras import REAL

    values, indices, B, _n_cols = residuals
    _, kmax = values.shape
    n_cols, _ = B.shape

    def body(
        p: Int[Array, ''], g_values: Num[Array, 'm kmax']
    ) -> Num[Array, 'm kmax']:
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
    """Backward of the real-semiring ELL rmatvec :math:`Y = A^{\\top} X`.

    The forward
    (:func:`~nitrix.semiring._reference.reference_semiring_ell_rmatvec`
    under the real semiring) applies the transpose of the ELL operator to
    the dense input :math:`X`.  Because the rmatvec is the adjoint of the
    gather matmul, its own reverse rule is the *matmul* forward for the
    ``X`` cotangent, plus the same per-edge gather-dot used for the
    ``values`` cotangent:

    .. code-block::

        grad_X[i, j]      = sum_p values[i, p] * g_out[indices[i, p], j]
        grad_values[i, p] = sum_j X[i, j] * g_out[indices[i, p], j]

    where the ``grad_X`` term equals :math:`A\\, g_{\\mathrm{out}}`.
    ``indices`` is non-differentiable; the public
    :func:`~nitrix.semiring.semiring_ell_rmatvec` returns a zero cotangent
    for it.

    Parameters
    ----------
    residuals : tuple
        The forward residuals ``(values, indices, X)``: the ELL
        ``values`` of shape ``(m, kmax)``, the column ``indices`` of
        shape ``(m, kmax)``, and the dense input :math:`X` of shape
        ``(m, ncol)``.
    g_out : Num[Array, 'n_cols ncol']
        Upstream cotangent with respect to the output :math:`Y`.

    Returns
    -------
    tuple of Array
        The pair ``(grad_values, grad_X)`` of cotangents, of shapes
        ``(m, kmax)`` and ``(m, ncol)`` respectively.
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
    """Backward of the log-semiring ELL matmul :math:`C_{ij} = \\log\\sum_p \\exp(v_{ip} + B_{\\mathrm{idx}(i, p), j})`.

    The gradient routes the upstream cotangent through the softmax weight
    per edge :math:`(i, p, j)`,

    .. code-block::

        w[i, p, j] = exp(values[i, p] + B[indices[i, p], j] - C[i, j])

    which is bounded in :math:`[0, 1]` because :math:`C` is the
    log-sum-exp over ``p``.  The ``values`` cotangent is the sum over
    ``j`` of ``g_out * w``, and the :math:`B` cotangent is a scatter-add
    of ``g_out * w`` over the column indices.  The loop streams over
    ``p`` so intermediates stay at ``(M, N)``.

    Parameters
    ----------
    residuals : tuple
        The forward residuals ``(values, indices, B, C)``: the ELL
        ``values`` of shape ``(m, kmax)``, the column ``indices`` of
        shape ``(m, kmax)``, the dense operand :math:`B` of shape
        ``(n_cols, ncol)``, and the forward output :math:`C` of shape
        ``(m, ncol)`` (the log-sum-exp, reused as the softmax
        normaliser).
    g_out : Num[Array, 'm ncol']
        Upstream cotangent with respect to the output :math:`C`.

    Returns
    -------
    tuple of Array
        The pair ``(grad_values, grad_B)`` of cotangents, of shapes
        ``(m, kmax)`` and ``(n_cols, ncol)`` respectively.
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
    """Argmax-gather subgradient for the max-plus ELL matmul.

    Under the max-plus semiring,
    :math:`C_{ij} = \\max_p (v_{ip} + B_{\\mathrm{idx}(i, p), j})`.  For
    each output entry :math:`(i, j)` the whole upstream gradient is routed
    to the single ELL column :math:`p^{*}(i, j)` that achieved the
    maximum, and scatter-added into :math:`B` at that column's index.
    This follows the same streaming pattern as the dense case, but with a
    per-row gather of ``B[indices[i, p], j]`` inside the sweep over ``p``.
    It is a subgradient because the maximum is not differentiable at ties.

    Parameters
    ----------
    residuals : tuple
        The forward residuals ``(values, indices, B, C)``: the ELL
        ``values`` of shape ``(m, kmax)``, the column ``indices`` of
        shape ``(m, kmax)``, the dense operand :math:`B` of shape
        ``(n_cols, ncol)``, and the (here unused) forward output
        :math:`C` of shape ``(m, ncol)``.
    g_out : Num[Array, 'm ncol']
        Upstream cotangent with respect to the output :math:`C`.

    Returns
    -------
    tuple of Array
        The pair ``(grad_values, grad_B)`` of cotangents, of shapes
        ``(m, kmax)`` and ``(n_cols, ncol)`` respectively.
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
    """Backward of the Euclidean ELL matmul :math:`C_{ij} = \\sqrt{\\sum_p (v_{ip} - B_{\\mathrm{idx}(i, p), j})^2}`.

    Uses the closed-form gradient of the sparse pairwise Euclidean
    distance, carrying a :math:`1 / C_{ij}` factor guarded by
    :func:`_safe_div` (the same square-root singularity guard as the
    dense case; the contribution is set to zero where the distance
    vanishes).  The loop streams over ``p`` so the per-step intermediate
    is ``(M, N)``; the ``values`` cotangent accumulates a per-edge
    gather-dot and the :math:`B` cotangent a scatter-add.

    Parameters
    ----------
    residuals : tuple
        The forward residuals ``(values, indices, B, C)``: the ELL
        ``values`` of shape ``(m, kmax)``, the column ``indices`` of
        shape ``(m, kmax)``, the dense operand :math:`B` of shape
        ``(n_cols, ncol)``, and the forward output :math:`C` of shape
        ``(m, ncol)`` (the distance, reused in the :math:`1 / C` factor).
    g_out : Num[Array, 'm ncol']
        Upstream cotangent with respect to the output :math:`C`.

    Returns
    -------
    tuple of Array
        The pair ``(grad_values, grad_B)`` of cotangents, of shapes
        ``(m, kmax)`` and ``(n_cols, ncol)`` respectively.
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
