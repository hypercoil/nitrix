# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Tiny dense SPD linear algebra, size-dispatched and cuSOLVER-free.

The mass-univariate fits in ``nitrix.stats`` (LME / GLM / GAM) all reduce, per
element, to a small ``(p, p)`` symmetric-positive-definite solve -- the
profiled fixed-effect normal equations, the IRLS / penalised normal equations,
the variance-component information matrix.  ``p`` is tiny (a handful of
fixed-effect columns or spline coefficients), but the solve sits inside a
``vmap`` over up to ~1M elements, so the choice of routine matters twice over:

- **Performance / compile.**  Differentiating a batched ``cholesky`` makes the
  XLA:CPU compile scale with the batch and inflates the autodiff tape.
- **GPU availability.**  On the dev L4 the cuSOLVER handle pool is dead --
  ``potrf`` (Cholesky), ``syevd`` (eigh), and even ``getrf`` (LU) fail to
  create a handle as the first cuSOLVER routine in a process (see
  ``docs/feature-requests/gpu-cusolver-first-call-handle-failure.md``).  Only
  cuBLAS (matmul, ``triangular_solve`` / ``trsm``) and the elementwise surface
  work.

So we issue **no cuSOLVER custom-call at any ``p``**: closed-form algebra for
``p in {1, 2}`` (the dominant designs), and for ``p > 2`` a hand-Cholesky whose
inverse goes through ``triangular_solve`` (cuBLAS ``trsm``).  Everything is
differentiable.

Why a *rolled* Cholesky
-----------------------

The Cholesky loop is run as a ``lax.fori_loop`` over the ``p`` columns, **not**
unrolled at trace time.  A trace-time unroll produces ``O(p^3)`` scalar ops in
the graph, so compile time grows cubically in ``p`` -- fine for LME's tiny
fixed-effect width (``p <= 5`` is ~0.4 s) but ruinous for GAM, where ``p = 1 +
sum(smooth dims)`` reaches 20-30 (the unrolled form measured ~29 s for a single
``p = 30`` inverse, dominating a ~96 s GAM compile).  Rolling the loop makes the
graph ``O(p^2)`` (one compiled body), so compile is ~flat in ``p`` (~0.4 s at
``p = 30``); runtime is unchanged (still ``O(p^3)`` flops).  The closed forms
for ``p in {1, 2}`` -- the LME hot path -- are untouched.
"""

from __future__ import annotations

from typing import Tuple, cast

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

__all__ = ['small_inv_logdet', 'spd_inv_logdet_chol', 'sym_eig_jacobi']

# Relative pivot floor (modified Cholesky).  A pivot / determinant is clamped
# to this fraction of the matrix's diagonal scale before a ``sqrt`` / division,
# so a near-singular or boundary system (where the caller's ridge was too small
# to keep the pivot positive against roundoff) yields a *regularised, finite*
# solve instead of a silent ``NaN``.  It sits ~4 orders below fp32 eps and is
# far below the smallest pivot of any well-conditioned matrix (cond < 1e12), so
# it never perturbs a healthy solve -- only rescues a degenerate one.
_PIVOT_REL_FLOOR = 1e-12


def spd_inv_logdet_chol(
    A: Float[Array, 'n n'], n: int
) -> Tuple[Float[Array, 'n n'], Float[Array, '']]:
    """SPD inverse + log-det via a rolled, cuSOLVER-free Cholesky.

    For ``n > 2`` we deliberately avoid ``jnp.linalg.{inv,cholesky,slogdet}``
    -- on the broken-cuSOLVER L4 their ``getrf`` / ``potrf`` custom-calls fail
    to create a handle (cause-unknown; see the GPU-availability FR).  Instead we
    factor ``A = L L^T`` column-by-column with a ``lax.fori_loop`` (rolled, so
    the graph stays ``O(n^2)`` -- see the module docstring), take the inverse
    through ``triangular_solve`` (cuBLAS ``trsm``, which *does* work on these
    stacks), and read the log-determinant off the factor diagonal.  No cuSOLVER
    routine is issued; everything is differentiable (``sqrt`` / division /
    ``trsm`` all have VJPs).

    The column update uses ``L @ L[j]``: at column ``j`` the not-yet-filled
    entries of ``L`` are zero, so the full dot automatically restricts to the
    ``k < j`` terms of the Cholesky-Banachiewicz recurrence.
    """
    idx = jnp.arange(n)
    # Modified-Cholesky pivot floor, relative to the matrix's diagonal scale so
    # a well-conditioned solve is unperturbed but a degenerate pivot stays
    # positive (finite, regularised) rather than producing sqrt(negative)=NaN.
    floor = (
        _PIVOT_REL_FLOOR * jnp.max(jnp.diagonal(A)) + jnp.finfo(A.dtype).tiny
    )

    def body(j: Array, L: Float[Array, 'n n']) -> Float[Array, 'n n']:
        Lj = lax.dynamic_index_in_dim(L, j, axis=0, keepdims=False)  # (n,)
        s = L @ Lj  # s_i = sum_{k<j} L[i,k] L[j,k]
        diag = jnp.sqrt(jnp.maximum(A[j, j] - s[j], floor))
        col = (A[:, j] - s) / diag
        new_col = jnp.where(idx == j, diag, jnp.where(idx > j, col, 0.0))
        return cast(
            Float[Array, 'n n'],
            lax.dynamic_update_index_in_dim(L, new_col, j, axis=1),
        )

    L = cast(
        Float[Array, 'n n'],
        lax.fori_loop(0, n, body, jnp.zeros((n, n), A.dtype)),
    )
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L)))
    eye = jnp.eye(n, dtype=A.dtype)
    l_inv = lax.linalg.triangular_solve(
        L, eye, left_side=True, lower=True, transpose_a=False
    )
    inv = l_inv.T @ l_inv  # A^{-1} = L^{-T} L^{-1}
    return inv, log_det


def small_inv_logdet(
    A: Float[Array, 'n n'], n: int
) -> Tuple[Float[Array, 'n n'], Float[Array, '']]:
    """Inverse and log-determinant of a small SPD ``(n, n)`` matrix.

    ``n`` is a Python int (a compile-time shape), so the branch is a
    Python-level ``if`` and **no branch issues a cuSOLVER custom-call**:

    - ``n == 1``  -- a reciprocal and a ``log``.
    - ``n == 2``  -- the explicit symmetric inverse ``[[c,-b],[-b,a]]/det`` and
      ``log(det)`` with ``det = a c - b^2``.
    - ``n > 2``   -- a rolled hand-Cholesky + ``triangular_solve`` (cuBLAS
      ``trsm``), see ``spd_inv_logdet_chol``.

    ``A`` is assumed SPD (the callers add a ridge).
    """
    tiny = jnp.finfo(A.dtype).tiny
    if n == 1:
        # Floor a non-positive / zero pivot (roundoff or constant input) so the
        # reciprocal and log stay finite; never perturbs an SPD scalar.
        a = jnp.maximum(A[0, 0], _PIVOT_REL_FLOOR * jnp.abs(A[0, 0]) + tiny)
        return (1.0 / a)[None, None], jnp.log(a)
    if n == 2:
        a = A[0, 0]
        b = A[0, 1]
        c = A[1, 1]
        # Floor the determinant relative to the diagonal product, so a
        # near-singular / indefinite 2x2 gives a finite regularised inverse.
        det = jnp.maximum(a * c - b * b, _PIVOT_REL_FLOOR * a * c + tiny)
        inv = jnp.array([[c, -b], [-b, a]], dtype=A.dtype) / det
        return inv, jnp.log(det)
    return spd_inv_logdet_chol(A, n)


def sym_eig_jacobi(
    A: Float[Array, 'n n'], n: int, n_sweeps: int = 8
) -> Tuple[Float[Array, 'n'], Float[Array, 'n n']]:
    """Symmetric eigendecomposition of a small ``(n, n)`` matrix, cuSOLVER-free.

    Returns ``(evals, evecs)`` with ``A approx evecs @ diag(evals) @ evecs.T``;
    the columns of ``evecs`` are orthonormal eigenvectors.  Eigenvalues are **not
    sorted** (cyclic Jacobi visits pairs in a fixed order).

    Why a hand-rolled Jacobi.  ``jnp.linalg.eigh`` issues a ``syevd`` cuSOLVER
    custom-call, which is dead on the dev L4 (see ``small_inv_logdet``); and
    ``safe_eigh``'s CPU fallback is eager (not ``jit`` / ``vmap`` -able), so it
    cannot run *inside* a per-element batched computation.  A fixed-sweep cyclic
    Jacobi is pure arithmetic (Givens rotations): jittable, ``vmap``-clean, and
    exact to machine precision in a handful of sweeps for the tiny ``n`` this
    serves (an ``L x L`` contrast covariance, ``L`` a few rows;
    ``lme_f_contrast``'s Fai-Cornelius denominator-df eigendirections).

    **Forward-only.**  The decomposition is for the value, not its gradient: a
    naive reverse-mode sweep accumulates ``1 / off-diagonal^2`` rotation-angle
    terms that overflow as the iteration converges (the standard hazard of
    differentiating an iterative eigensolver).  Its sole consumer -- the
    Satterthwaite F denominator df -- is not differentiated through (v3 §1.3), so
    that is by design; wrap a call in ``lax.stop_gradient`` if it ever feeds a
    differentiated path.

    Each rotation zeroes one off-diagonal ``A[p, q]`` via the orthogonal
    ``J(p, q)`` (the stable smaller-root angle), applied as ``A <- J^T A J`` and
    ``V <- V J``; ``n`` is a Python int, so the ``n(n-1)/2`` pair sweep is
    unrolled and ``n_sweeps`` cyclic passes drive the off-diagonal to zero
    quadratically.  ``n_sweeps=8`` reaches machine precision for ``n <= 7`` (the
    largest matrix this serves -- a ``7``-component block-Woodbury Hessian;
    empirically ``n = 7`` converges by ``6`` sweeps, ``n <= 5`` by ``5``).  The
    rotation is applied as the dense ``J^T A J`` matmul rather than an explicit
    rank-2 row/column update: the latter is fewer flops but unrolls into many
    small dynamic-update-slice fusions that the XLA:CPU JIT fails to materialise,
    and at this ``n`` the matmul cost is negligible.
    """
    A = 0.5 * (A + A.T)
    dtype = A.dtype
    tiny = jnp.finfo(dtype).tiny
    pairs = [(p, q) for p in range(n) for q in range(p + 1, n)]

    def one_sweep(
        carry: Tuple[Float[Array, 'n n'], Float[Array, 'n n']], _: Array
    ) -> Tuple[Tuple[Float[Array, 'n n'], Float[Array, 'n n']], None]:
        mat, vecs = carry
        for p, q in pairs:
            apq = mat[p, q]
            app = mat[p, p]
            aqq = mat[q, q]
            do_rot = jnp.abs(apq) > tiny  # skip an already-zero off-diagonal
            denom = jnp.where(do_rot, 2.0 * apq, 1.0)
            tau = (aqq - app) / denom
            sgn = jnp.where(tau >= 0, 1.0, -1.0)
            t = sgn / (
                jnp.abs(tau) + jnp.sqrt(tau * tau + 1.0)
            )  # smaller root
            t = jnp.where(do_rot, t, 0.0)
            cos = 1.0 / jnp.sqrt(t * t + 1.0)
            sin = t * cos
            rot = jnp.eye(n, dtype=dtype)
            rot = rot.at[p, p].set(cos)
            rot = rot.at[q, q].set(cos)
            rot = rot.at[p, q].set(sin)
            rot = rot.at[q, p].set(-sin)
            mat = rot.T @ mat @ rot
            vecs = vecs @ rot
        return (mat, vecs), None

    (mat, vecs), _ = lax.scan(
        one_sweep, (A, jnp.eye(n, dtype=dtype)), xs=None, length=n_sweeps
    )
    return jnp.diagonal(mat), vecs
