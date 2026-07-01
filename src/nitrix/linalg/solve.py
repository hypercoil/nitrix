# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Dense linear solves: general and SPD-Cholesky.

The public surface for the small dense solves the rest of the library
implied but never exposed first-class.  :func:`cho_solve` is the SPD
normal-equation solve behind the registration Gauss-Newton /
Levenberg-Marquardt step (:math:`(J^{\\top}J + \\lambda I)\\,\\delta =
-J^{\\top}r`) and any small symmetric-positive-definite system;
:func:`solve` is the general (non-symmetric) :math:`A x = b`.

Both batch over leading dimensions and are reverse-mode differentiable.
The right-hand side may be a vector ``(..., n)`` or a matrix
``(..., n, m)``; the trailing shape of the result matches.

Notes
-----
These delegate to the cuSolver-robust ``safe_cho_solve`` /
``safe_solve`` wrappers, which route the factorisation to a device
where it works (CPU on stacks with the broken cuSolver handle pool) and
move the result back.  On a healthy GPU stack there is no overhead;
the triangular solve (cuBLAS) stays on-device.
"""

from __future__ import annotations

from jaxtyping import Array, Float

from ._solver import safe_cho_solve, safe_solve

__all__ = ['solve', 'cho_solve']


def solve(
    a: Float[Array, '... n n'],
    b: Float[Array, '...'],
) -> Float[Array, '...']:
    """Solve the general linear system :math:`a x = b`.

    Batches over any leading dimensions and is reverse-mode
    differentiable.  The right-hand side may be a single vector or a
    stack of column vectors; the trailing shape of the solution matches
    that of ``b``.

    Parameters
    ----------
    a
        Square coefficient matrix, ``(..., n, n)``.
    b
        Right-hand side: ``(..., n)`` (vector) or ``(..., n, m)``
        (matrix of columns).

    Returns
    -------
    Float[Array, '...']
        Solution :math:`x` with the same trailing shape as ``b``.

    Notes
    -----
    For symmetric-positive-definite ``a`` prefer :func:`cho_solve` -- it
    is roughly twice as fast (one Cholesky plus two triangular solves
    versus a full LU) and numerically better-behaved on the normal
    equations.
    """
    return safe_solve(a, b)


def cho_solve(
    a: Float[Array, '... n n'],
    b: Float[Array, '...'],
    *,
    l2: float = 0.0,
) -> Float[Array, '...']:
    """Solve the SPD system :math:`(a + \\lambda I) x = b` via Cholesky.

    The optional ``l2`` ridge :math:`\\lambda` is added to the diagonal
    -- the standard Levenberg-Marquardt / Tikhonov damping -- so a
    single call covers both the Gauss-Newton (``l2 = 0``) and
    Levenberg-Marquardt (``l2 > 0``) normal-equation solves.  Batches
    over any leading dimensions and is reverse-mode differentiable.

    Parameters
    ----------
    a
        Symmetric positive-definite matrix, ``(..., n, n)``.  Symmetry
        is assumed, not enforced.
    b
        Right-hand side: ``(..., n)`` (vector) or ``(..., n, m)``
        (matrix of columns).
    l2
        Non-negative diagonal ridge added before factoring.  Default
        ``0`` (plain SPD solve).

    Returns
    -------
    Float[Array, '...']
        Solution :math:`x` with the same trailing shape as ``b``.

    Notes
    -----
    Returns ``NaN`` if :math:`a + \\lambda I` is not positive-definite
    (no Cholesky factor); add an ``l2 > 0`` ridge if the Gram matrix may
    be rank-deficient.
    """
    return safe_cho_solve(a, b, l2=l2)
