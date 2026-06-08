# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Matrix-free Krylov solvers.

``cg`` -- conjugate gradients for a **symmetric positive-definite**
system ``A x = b``, where ``A`` is supplied either as a dense matrix or
as a matrix-vector-product callable.  Because CG is built from matvecs
and inner products only -- no factorisation -- it never touches the
cuSolver handle pool, so it stays on the GPU on the stacks where
``cholesky`` / ``solve`` wedge (see ``_solver``).  That makes it the
device-resilient SPD path: the registration Gauss-Newton /
Levenberg-Marquardt normal equations ``(JᵀJ + λ·diag) δ = -Jᵀr`` are SPD
and small, so their per-iteration solve runs matrix-free on-device
instead of paying a host round-trip to a CPU Cholesky.

Differentiable: ``jax.scipy.sparse.linalg.cg`` carries its own
(implicit) differentiation rule, so gradients flow through the solve.

This is the first member of the §12.1 ``linalg.krylov`` family
(``minres`` / ``lsqr`` / ``bicgstab`` to follow); registration is the
named consumer that graduates it.
"""

from __future__ import annotations

from typing import Callable, Optional, Union, cast

from jax.scipy.sparse.linalg import cg as _jax_cg
from jaxtyping import Array, Float

__all__ = ['cg']

MatvecOrMatrix = Union[Float[Array, '... n n'], Callable[[Array], Array]]


def cg(
    a: MatvecOrMatrix,
    b: Float[Array, '... n'],
    *,
    x0: Optional[Float[Array, '... n']] = None,
    tol: float = 1e-6,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    l2: Union[float, Array] = 0.0,
) -> Float[Array, '... n']:
    """Solve the SPD system ``(A + l2 I) x = b`` by conjugate gradients.

    Parameters
    ----------
    a
        The SPD operator: either a dense matrix ``(..., n, n)`` or a
        callable computing the matrix-vector product ``v -> A @ v`` (the
        matrix is never materialised in the callable case).
    b
        Right-hand side, ``(..., n)``.
    x0
        Optional initial guess (default zeros).
    tol, atol
        Convergence tolerances; CG stops when
        ``norm(residual) <= max(tol * norm(b), atol)``.
    maxiter
        Maximum iterations (default: CG's own, ``10 * n``).  An SPD
        system converges in at most ``n`` exact steps.
    l2
        Non-negative Tikhonov / Levenberg-Marquardt ridge added to the
        diagonal (in matrix-free form, ``A v + l2 v``).  Default ``0``.
        May be a traced scalar (so the LM damping ``λ`` can vary inside
        a ``jit``); the ``l2 v`` term is added unconditionally, a no-op
        at ``l2 = 0``.

    Returns
    -------
    The solution ``x``, ``(..., n)``.

    Notes
    -----
    GPU-native and resilient to the cuSolver wedge (matvecs + dot
    products only).  Requires ``A`` to be SPD; for a general system use
    ``solve``, for a direct SPD factorisation use ``cho_solve``.
    """
    op: Callable[[Array], Array]
    if callable(a):
        op = a
    else:
        mat = a

        def _matmul(v: Array) -> Array:
            return mat @ v

        op = _matmul

    def matvec(v: Array) -> Array:
        return op(v) + l2 * v

    x, _ = _jax_cg(matvec, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter)
    return cast(Float[Array, '... n'], x)
