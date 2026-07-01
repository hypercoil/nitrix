# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Matrix-free Krylov solvers.

The :func:`cg` routine implements conjugate gradients for a symmetric
positive-definite system :math:`A x = b`, where :math:`A` is supplied
either as a dense matrix or as a matrix-vector-product callable.  Because
conjugate gradients is built from matrix-vector products and inner
products only -- with no factorisation -- it never touches the cuSolver
handle pool, so it stays on the GPU on the stacks where the direct dense
solvers wedge.  That makes it the device-resilient positive-definite
path: the registration Gauss-Newton / Levenberg-Marquardt normal
equations :math:`(J^{\\top} J + \\lambda \\operatorname{diag}) \\delta =
-J^{\\top} r` are symmetric positive-definite and small, so their
per-iteration solve runs matrix-free on-device instead of paying a host
round-trip to a CPU Cholesky.

The solve is differentiable: the underlying conjugate-gradient routine
carries its own (implicit) differentiation rule, so gradients flow
through the solve.
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
    """Solve the symmetric positive-definite system by conjugate gradients.

    Solves :math:`(A + \\lambda I) x = b` for a symmetric positive-definite
    operator :math:`A`, using only matrix-vector products and inner
    products.  No factorisation is formed, so the solve stays on the GPU
    and is resilient to the cuSolver wedge that can affect the direct
    dense solvers.

    Parameters
    ----------
    a
        The symmetric positive-definite operator: either a dense matrix
        of shape ``(..., n, n)`` or a callable computing the
        matrix-vector product :math:`v \\mapsto A v` (the matrix is never
        materialised in the callable case).
    b
        Right-hand side of shape ``(..., n)``.
    x0
        Optional initial guess of shape ``(..., n)`` (default zeros).
    tol, atol
        Relative and absolute convergence tolerances.  The iteration
        stops when :math:`\\lVert r \\rVert \\le \\max(\\mathtt{tol}
        \\cdot \\lVert b \\rVert, \\mathtt{atol})`, where :math:`r` is the
        residual.  Defaults ``1e-6`` and ``0.0``.
    maxiter
        Maximum number of iterations.  The default defers to the
        underlying solver (``10 * n``); a positive-definite system
        converges in at most :math:`n` exact steps.
    l2
        Non-negative Tikhonov / Levenberg-Marquardt ridge :math:`\\lambda`
        added to the diagonal (in matrix-free form, :math:`A v +
        \\lambda v`).  Default ``0``.  May be a traced scalar (so the
        Levenberg-Marquardt damping :math:`\\lambda` can vary inside a
        ``jit``); the :math:`\\lambda v` term is added unconditionally, a
        no-op at :math:`\\lambda = 0`.

    Returns
    -------
    Float[Array, '... n']
        The solution :math:`x`, of shape ``(..., n)``.

    Notes
    -----
    GPU-native and resilient to the cuSolver wedge, as it uses
    matrix-vector and dot products only.  Requires :math:`A` to be
    symmetric positive-definite; for a general system use :func:`solve`,
    and for a direct positive-definite factorisation use
    :func:`cho_solve`.
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
