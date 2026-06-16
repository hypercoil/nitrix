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
``p in {1, 2}`` (the dominant designs), and for ``p > 2`` an unrolled
hand-Cholesky (plain ``jnp``, the loop unrolled at trace time -- ``p`` is a
compile-time int and tiny) whose inverse goes through ``triangular_solve``
(cuBLAS ``trsm``).  Everything is differentiable.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

__all__ = ['small_inv_logdet', 'unrolled_spd_inv_logdet']


def unrolled_spd_inv_logdet(
    A: Float[Array, 'n n'], n: int
) -> Tuple[Float[Array, 'n n'], Float[Array, '']]:
    """SPD inverse + log-det via a hand-written, unrolled Cholesky.

    For ``n > 2`` we deliberately avoid ``jnp.linalg.{inv,cholesky,slogdet}``
    -- on the broken-cuSOLVER L4 their ``getrf`` / ``potrf`` custom-calls fail
    to create a handle (cause-unknown; see the GPU-availability FR).  Instead
    we factor ``A = L L^T`` with the Cholesky-Banachiewicz recurrence written
    in plain ``jnp`` (the loop is unrolled at trace time -- ``n`` is a
    compile-time int and tiny), take the inverse through ``triangular_solve``
    (cuBLAS ``trsm``, which *does* work on these stacks), and read the
    log-determinant off the factor diagonal.  No cuSOLVER routine is issued at
    any point; the whole thing is differentiable (``sqrt`` / division /
    ``trsm`` all have VJPs).
    """
    cols: list[list[Array]] = []
    for j in range(n):
        diag = A[j, j]
        for k in range(j):
            ljk = cols[k][j]
            diag = diag - ljk * ljk
        ljj = jnp.sqrt(diag)
        col_entries = [jnp.zeros((), A.dtype)] * j + [ljj]
        for i in range(j + 1, n):
            s = A[i, j]
            for k in range(j):
                s = s - cols[k][i] * cols[k][j]
            col_entries.append(s / ljj)
        cols.append(col_entries)
    # L[i, j] = cols[j][i] (lower triangular).
    L = jnp.stack(
        [jnp.stack([cols[j][i] for j in range(n)]) for i in range(n)]
    )
    log_det = 2.0 * jnp.sum(jnp.log(jnp.stack([cols[j][j] for j in range(n)])))
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
    - ``n > 2``   -- an unrolled hand-Cholesky + ``triangular_solve`` (cuBLAS
      ``trsm``), see ``unrolled_spd_inv_logdet``.

    ``A`` is assumed SPD (the callers add a ridge).
    """
    if n == 1:
        a = A[0, 0]
        return (1.0 / a)[None, None], jnp.log(a)
    if n == 2:
        a = A[0, 0]
        b = A[0, 1]
        c = A[1, 1]
        det = a * c - b * b
        inv = jnp.array([[c, -b], [-b, a]], dtype=A.dtype) / det
        return inv, jnp.log(det)
    return unrolled_spd_inv_logdet(A, n)
