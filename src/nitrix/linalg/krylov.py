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

from typing import Callable, Optional, Tuple, Union, cast

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import bicgstab as _jax_bicgstab
from jax.scipy.sparse.linalg import cg as _jax_cg
from jax.scipy.sparse.linalg import gmres as _jax_gmres
from jaxtyping import Array, Float

__all__ = ['bicgstab', 'cg', 'gmres', 'minres']

MatvecOrMatrix = Union[Float[Array, '... n n'], Callable[[Array], Array]]


def _as_op(
    a: MatvecOrMatrix, l2: Union[float, Array]
) -> Callable[[Array], Array]:
    """Resolve a matrix / matvec to an ``(A + l2 I)`` matvec closure."""
    if callable(a):
        op = a
    else:
        mat = a

        def op(v: Array) -> Array:
            return mat @ v

    def matvec(v: Array) -> Array:
        return op(v) + l2 * v

    return matvec


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
    matvec = _as_op(a, l2)
    x, _ = _jax_cg(matvec, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter)
    return cast(Float[Array, '... n'], x)


def bicgstab(
    a: MatvecOrMatrix,
    b: Float[Array, '... n'],
    *,
    x0: Optional[Float[Array, '... n']] = None,
    tol: float = 1e-6,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    l2: Union[float, Array] = 0.0,
) -> Float[Array, '... n']:
    r"""Solve a **non-symmetric** system by stabilised bi-conjugate gradients.

    Solves :math:`(A + \lambda I) x = b` for a general (non-symmetric) operator
    :math:`A` using only matrix-vector products -- the matrix-free counterpart of
    :func:`cg` when :math:`A` is not symmetric positive-definite (e.g. the
    resolvent :math:`(\alpha I - J)` of a non-symmetric Jacobian). GPU-native,
    cuSOLVER-free, and differentiable (the routine carries its own implicit rule).

    Parameters
    ----------
    a
        The operator: a dense ``(..., n, n)`` matrix or a matvec callable.
    b
        Right-hand side ``(..., n)``.
    x0
        Optional initial guess (default zeros).
    tol, atol
        Relative / absolute residual tolerances. Defaults ``1e-6`` / ``0``.
    maxiter
        Maximum iterations (defaults to the underlying solver's).
    l2
        Non-negative ridge :math:`\lambda` added in matrix-free form. Default
        ``0``.

    Returns
    -------
    Float[Array, '... n']
        The solution :math:`x`.
    """
    matvec = _as_op(a, l2)
    x, _ = _jax_bicgstab(matvec, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter)
    return cast(Float[Array, '... n'], x)


def gmres(
    a: MatvecOrMatrix,
    b: Float[Array, '... n'],
    *,
    x0: Optional[Float[Array, '... n']] = None,
    tol: float = 1e-6,
    atol: float = 0.0,
    restart: int = 20,
    maxiter: Optional[int] = None,
    l2: Union[float, Array] = 0.0,
) -> Float[Array, '... n']:
    r"""Solve a **non-symmetric** system by restarted GMRES.

    The generalised minimal-residual method: minimises the residual over the
    Krylov subspace, restarted every ``restart`` steps. A more robust (if more
    memory-hungry) non-symmetric solver than :func:`bicgstab`; matrix-free,
    GPU-native, differentiable.

    Parameters
    ----------
    a, b, x0, tol, atol, maxiter, l2
        As in :func:`bicgstab`.
    restart
        Krylov subspace size before a restart. Default ``20``.

    Returns
    -------
    Float[Array, '... n']
        The solution :math:`x`.
    """
    matvec = _as_op(a, l2)
    x, _ = _jax_gmres(
        matvec, b, x0=x0, tol=tol, atol=atol, restart=restart, maxiter=maxiter
    )
    return cast(Float[Array, '... n'], x)


def _minres_forward(
    matvec: Callable[[Array], Array],
    b: Float[Array, '... n'],
    maxiter: int,
    tol: float,
) -> Float[Array, '... n']:
    """MINRES iteration (Paige--Saunders), starting from zero.

    Symmetric (possibly indefinite) matrix-free solve via Lanczos
    tridiagonalisation + Givens QR of the residual; batches over leading axes.
    """
    eps = jnp.finfo(b.dtype).eps
    beta1 = jnp.linalg.norm(b, axis=-1)  # (...,)
    z = jnp.zeros_like(beta1)

    State = Tuple[
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
    ]
    init: State = (
        jnp.zeros((), jnp.int32),  # itn
        jnp.zeros_like(b),  # x
        b,  # r1
        b,  # r2
        z,  # oldb
        beta1,  # beta
        z,  # dbar
        z,  # epsln
        beta1,  # phibar
        -jnp.ones_like(beta1),  # cs
        z,  # sn
        jnp.zeros_like(b),  # w
        jnp.zeros_like(b),  # w2
    )

    def cond(state: State) -> Array:
        itn, phibar = state[0], state[8]
        return (itn < maxiter) & jnp.any(phibar > tol * beta1)

    def body(state: State) -> State:
        itn, x, r1, r2, oldb, beta, dbar, epsln, phibar, cs, sn, w, w2 = state
        itn = itn + 1
        v = r2 / beta[..., None]
        y = matvec(v)
        first = itn < 2
        safe_oldb = jnp.where(first, 1.0, oldb)
        y = y - jnp.where(first, 0.0, beta / safe_oldb)[..., None] * r1
        alfa = jnp.sum(v * y, axis=-1)
        y = y - (alfa / beta)[..., None] * r2
        r1 = r2
        r2 = y
        oldb = beta
        beta = jnp.linalg.norm(y, axis=-1)
        oldeps = epsln
        delta = cs * dbar + sn * alfa
        gbar = sn * dbar - cs * alfa
        epsln = sn * beta
        dbar = -cs * beta
        gamma = jnp.maximum(jnp.sqrt(gbar**2 + beta**2), eps)
        cs = gbar / gamma
        sn = beta / gamma
        phi = cs * phibar
        phibar = sn * phibar
        denom = 1.0 / gamma
        w1 = w2
        w2 = w
        w = (v - oldeps[..., None] * w1 - delta[..., None] * w2) * denom[
            ..., None
        ]
        x = x + phi[..., None] * w
        return (
            itn,
            x,
            r1,
            r2,
            oldb,
            beta,
            dbar,
            epsln,
            phibar,
            cs,
            sn,
            w,
            w2,
        )

    final = jax.lax.while_loop(cond, body, init)
    return final[1]


def minres(
    a: MatvecOrMatrix,
    b: Float[Array, '... n'],
    *,
    x0: Optional[Float[Array, '... n']] = None,
    tol: float = 1e-6,
    maxiter: Optional[int] = None,
    l2: Union[float, Array] = 0.0,
) -> Float[Array, '... n']:
    r"""Solve a **symmetric (possibly indefinite)** system by MINRES.

    Solves :math:`(A + \lambda I) x = b` for a symmetric operator :math:`A` that
    need not be positive-definite -- the case :func:`cg` cannot handle (a
    saddle-point / indefinite Hermitian system) -- by the minimum-residual method
    (Paige--Saunders): Lanczos tridiagonalisation with a Givens-QR residual
    minimisation, using only matrix-vector products. GPU-native and cuSOLVER-free.
    Differentiable via :func:`jax.lax.custom_linear_solve` (the implicit rule for
    a linear solve; the backward is a MINRES solve with the same symmetric
    operator).

    Parameters
    ----------
    a
        The symmetric operator: a dense ``(..., n, n)`` matrix or a matvec
        callable. Symmetry is assumed, not checked.
    b
        Right-hand side ``(..., n)``.
    x0
        Optional initial guess; the residual system is solved from it. Default
        zeros.
    tol
        Relative residual tolerance :math:`\lVert r \rVert \le
        \mathtt{tol}\,\lVert b \rVert`. Default ``1e-6``.
    maxiter
        Maximum iterations. Default ``5 n``.
    l2
        Non-negative ridge :math:`\lambda` added in matrix-free form. Default
        ``0``.

    Returns
    -------
    Float[Array, '... n']
        The solution :math:`x`.
    """
    matvec = _as_op(a, l2)
    n = b.shape[-1]
    iters = maxiter if maxiter is not None else 5 * n

    def solve(mv: Callable[[Array], Array], rhs: Array) -> Array:
        return _minres_forward(mv, rhs, iters, tol)

    if x0 is None:
        x = jax.lax.custom_linear_solve(matvec, b, solve, symmetric=True)
    else:
        residual = b - matvec(x0)
        x = x0 + jax.lax.custom_linear_solve(
            matvec, residual, solve, symmetric=True
        )
    return cast(Float[Array, '... n'], x)
