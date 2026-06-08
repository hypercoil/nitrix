# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Dense linear solves: general and SPD-Cholesky.

The public surface for the small dense solves the rest of the library
implied but never exposed first-class.  ``cho_solve`` is the SPD
normal-equation solve behind the registration Gauss-Newton /
Levenberg-Marquardt step (``(JᵀJ + λI) δ = -Jᵀr``) and any small
symmetric-positive-definite system; ``solve`` is the general
(non-symmetric) ``A x = b``.

Both batch over leading dimensions and are reverse-mode differentiable.
The right-hand side may be a vector ``(..., n)`` or a matrix
``(..., n, m)``; the trailing shape of the result matches.

Implementation note: these delegate to the cuSolver-robust
``_solver.safe_cho_solve`` / ``safe_solve`` wrappers, which route the
factorisation to a device where it works (CPU on stacks with the broken
cuSolver handle pool -- see ``_solver`` for the probe + adaptive-latch
machinery) and move the result back.  On a healthy GPU stack there is no
overhead; ``triangular_solve`` (cuBLAS) stays on-device.
"""

from __future__ import annotations

from jaxtyping import Array, Float

from ._solver import safe_cho_solve, safe_solve

__all__ = ['solve', 'cho_solve']


def solve(
    a: Float[Array, '... n n'],
    b: Float[Array, '...'],
) -> Float[Array, '...']:
    """Solve the general linear system ``a x = b``.

    Parameters
    ----------
    a
        Square coefficient matrix, ``(..., n, n)``.
    b
        Right-hand side: ``(..., n)`` (vector) or ``(..., n, m)``
        (matrix of columns).

    Returns
    -------
    Solution ``x`` with the same trailing shape as ``b``.

    Notes
    -----
    For symmetric-positive-definite ``a`` prefer ``cho_solve`` -- it is
    ~2x faster (one Cholesky + two triangular solves vs a full LU) and
    numerically better-behaved on the normal equations.
    """
    return safe_solve(a, b)


def cho_solve(
    a: Float[Array, '... n n'],
    b: Float[Array, '...'],
    *,
    l2: float = 0.0,
) -> Float[Array, '...']:
    """Solve the SPD system ``(a + l2 I) x = b`` via Cholesky.

    The optional ``l2`` ridge is added to the diagonal -- the standard
    Levenberg-Marquardt / Tikhonov damping -- so a single call covers
    both the Gauss-Newton (``l2 = 0``) and Levenberg-Marquardt
    (``l2 > 0``) normal-equation solves.

    Parameters
    ----------
    a
        Symmetric positive-definite matrix, ``(..., n, n)``.  Symmetry
        is assumed, not enforced.
    b
        Right-hand side: ``(..., n)`` (vector) or ``(..., n, m)``.
    l2
        Non-negative diagonal ridge added before factoring.  Default
        ``0`` (plain SPD solve).

    Returns
    -------
    Solution ``x`` with the same trailing shape as ``b``.

    Notes
    -----
    Returns ``NaN`` if ``a + l2 I`` is not positive-definite (no
    Cholesky factor); add an ``l2 > 0`` ridge if the Gram matrix may be
    rank-deficient.
    """
    return safe_cho_solve(a, b, l2=l2)
