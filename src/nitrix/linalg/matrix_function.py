# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Matrix functions of a *general* (non-symmetric) matrix.

The symmetric family (``symexp`` / ``symlog`` / ``symsqrt`` / ``sympower``
in ``linalg.spd``) applies a scalar function to the eigenvalues of a
symmetric matrix.  ``matrix_exp`` is the general-matrix counterpart: the
true matrix exponential ``e^A`` for an arbitrary square ``A``, needed to
exponentiate a non-symmetric Lie-algebra element -- specifically the
affine generator ``[[A, b], [0, 0]]`` whose exponential is the affine
transform ``geometry.transform.affine_exp`` applies (a rigid generator's
rotation block is skew-symmetric, also non-symmetric).

This is the first member of the §12.2 ``linalg.matrix_function`` family;
registration is the named consumer that graduates it.  ``matrix_log`` and
the named ``matrix_polynomial`` / ``frechet_derivative`` specialisations
remain on the §12.2 backlog.

Implementation: scaling-and-squaring Padé (``jax.scipy.linalg.expm``)
routed through the cuSolver-robust device pick (the Padé step is a
general ``lu`` solve, which wedges on the affected stacks -- see
``_solver.safe_expm``).  Differentiable; batches over leading dims.
"""

from __future__ import annotations

from jaxtyping import Array, Float

from ._solver import safe_expm

__all__ = ['matrix_exp']


def matrix_exp(
    a: Float[Array, '... n n'],
    *,
    max_squarings: int = 16,
) -> Float[Array, '... n n']:
    """Matrix exponential ``e^A`` of a general square matrix.

    Parameters
    ----------
    a
        Square matrix, ``(..., n, n)``.  Need not be symmetric.
    max_squarings
        Cap on the scaling-and-squaring doublings (Padé accuracy
        control); the default ``16`` covers the magnitudes that arise
        from registration generators with room to spare.

    Returns
    -------
    ``e^A``, same shape as ``a``.

    Notes
    -----
    Identities used as correctness anchors in the tests: ``e^0 = I``;
    ``det(e^A) = e^{tr A}``; for a skew-symmetric ``A``, ``e^A`` is a
    rotation (orthogonal, ``det = +1``).
    """
    return safe_expm(a, max_squarings=max_squarings)
