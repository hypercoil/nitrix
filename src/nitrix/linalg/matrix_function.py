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

Implementation: **scaling-and-squaring with a truncated Taylor series**.
Unlike ``jax.scipy.linalg.expm`` (Padé, which needs a linear solve and so
wedges / falls back to CPU on the affected cuSolver stacks), this path is
**pure matmul** -- GPU-native, ``jit`` / ``grad``-clean, and never mixes
devices inside a traced optimiser.  ``A`` is scaled by ``2^-s``, the
Taylor series of ``e^{A/2^s}`` is summed, and the result is squared ``s``
times.  With the defaults (``s = 8``, order ``12``) the scaled norm is
tiny for any ``‖A‖`` that arises from a registration generator, so the
result matches a reference ``expm`` to machine precision; raise them for
very large ``‖A‖``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = ['matrix_exp']


def matrix_exp(
    a: Float[Array, '... n n'],
    *,
    n_squarings: int = 8,
    taylor_order: int = 12,
) -> Float[Array, '... n n']:
    """Matrix exponential ``e^A`` of a general square matrix.

    Parameters
    ----------
    a
        Square matrix, ``(..., n, n)``.  Need not be symmetric.  Batches
        over leading dimensions.
    n_squarings
        Scaling-and-squaring doublings ``s``: ``A`` is exponentiated as
        ``(e^{A/2^s})^{2^s}``.  The default ``8`` keeps the scaled norm
        small for any registration generator.
    taylor_order
        Number of Taylor terms for ``e^{A/2^s}``.  Default ``12``.

    Returns
    -------
    ``e^A``, same shape as ``a``.

    Notes
    -----
    Pure matmul (no factorisation), so it is GPU-native and differentiable
    by autodiff.  Identities used as correctness anchors in the tests:
    ``e^0 = I``; ``det(e^A) = e^{tr A}``; for a skew-symmetric ``A``,
    ``e^A`` is a rotation (orthogonal, ``det = +1``).
    """
    n = a.shape[-1]
    eye = jnp.broadcast_to(jnp.eye(n, dtype=a.dtype), a.shape)
    scaled = a / float(2**n_squarings)
    # Taylor series e^B = I + B + B^2/2! + ... + B^q/q!.
    term = eye
    result = eye
    for k in range(1, taylor_order + 1):
        term = (term @ scaled) / k
        result = result + term
    # Square s times to undo the scaling.
    for _ in range(n_squarings):
        result = result @ result
    return result
