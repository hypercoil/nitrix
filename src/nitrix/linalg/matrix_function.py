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
registration is the named consumer that graduates it.  ``matrix_log`` joins it
here (its warranting consumer is the affine Fréchet/Karcher mean,
``geometry.algebra.transform_mean``); the named ``matrix_polynomial`` /
``frechet_derivative`` specialisations remain on the §12.2 backlog.

``matrix_exp`` implementation: **scaling-and-squaring with a truncated Taylor
series**.  Unlike ``jax.scipy.linalg.expm`` (Padé, which needs a linear solve
and so wedges / falls back to CPU on the affected cuSolver stacks), this path is
**pure matmul** -- GPU-native, ``jit`` / ``grad``-clean, and never mixes
devices inside a traced optimiser.  ``A`` is scaled by ``2^-s``, the
Taylor series of ``e^{A/2^s}`` is summed, and the result is squared ``s``
times.  With the defaults (``s = 8``, order ``12``) the scaled norm is
tiny for any ``‖A‖`` that arises from a registration generator, so the
result matches a reference ``expm`` to machine precision; raise them for
very large ``‖A‖``.

``matrix_log`` is the inverse: **inverse scaling-and-squaring** -- take ``s``
matrix square roots (Denman--Beavers) to drive ``A`` toward ``I``, sum the
Taylor series of ``log(I + X)`` on the small remainder, and multiply by
``2^s``.  Registration affines carry large translations (``‖A − I‖ ≫ 1``), which
rules out a pure-matmul (binomial) square root, so the Denman--Beavers step
routes its inverse through ``safe_inv`` (GPU-native on a healthy stack;
CPU-fallback on the wedged-cuSolver dev box).  Unlike ``matrix_exp`` this is
**not** for an optimiser hot loop -- it backs the offline Karcher mean -- so the
fallback is acceptable.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from ._solver import safe_inv

__all__ = ['matrix_exp', 'matrix_log']


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


def _db_sqrtm(
    m: Float[Array, '... n n'], iters: int
) -> Float[Array, '... n n']:
    """Matrix square root by the Denman--Beavers iteration (batched).

    ``Y_{k+1} = ½(Y_k + Z_k⁻¹)``, ``Z_{k+1} = ½(Z_k + Y_k⁻¹)`` with
    ``Y_0 = M``, ``Z_0 = I``; ``Y_k -> M^{1/2}`` quadratically.  The inverses
    route through ``safe_inv`` (CPU-fallback on the wedged cuSolver stack).
    """
    n = m.shape[-1]
    eye = jnp.broadcast_to(jnp.eye(n, dtype=m.dtype), m.shape)
    y, z = m, eye
    for _ in range(iters):
        y, z = 0.5 * (y + safe_inv(z)), 0.5 * (z + safe_inv(y))
    return y


def matrix_log(
    a: Float[Array, '... n n'],
    *,
    n_sqrt: int = 6,
    db_iters: int = 8,
    taylor_order: int = 12,
) -> Float[Array, '... n n']:
    """Matrix logarithm ``log A`` of a general square matrix (the inverse of
    :func:`matrix_exp`).

    Inverse scaling-and-squaring: ``log A = 2^s · log(A^{1/2^s})`` -- take ``s``
    Denman--Beavers square roots to drive ``A`` toward ``I``, sum the Taylor
    series of ``log(I + X)`` on the small remainder ``X = A^{1/2^s} − I``, and
    scale back by ``2^s``.

    Parameters
    ----------
    a
        Square matrix, ``(..., n, n)`` with no eigenvalue on the closed
        negative real axis (a registration affine / its relative transform
        qualifies).  Batches over leading dimensions.
    n_sqrt
        Number of square-root steps ``s``.  The default ``6`` brings a
        registration affine (rotation + ``≲ 60`` voxel translation) within the
        Taylor radius; raise it for larger transforms.
    db_iters
        Denman--Beavers iterations per square root (quadratic convergence).
    taylor_order
        Terms of the ``log(I + X)`` series.

    Returns
    -------
    ``log A``, same shape as ``a``.

    Notes
    -----
    Routes its inverses through ``safe_inv`` (GPU-native on a healthy stack,
    CPU-fallback otherwise).  Correctness anchor: ``matrix_log`` is the inverse
    of ``matrix_exp`` (round-trips to machine precision in the registration
    regime); ``log(I) = 0``; ``tr(log A) = log(det A)``.
    """
    n = a.shape[-1]
    eye = jnp.broadcast_to(jnp.eye(n, dtype=a.dtype), a.shape)
    root = a
    for _ in range(n_sqrt):
        root = _db_sqrtm(root, db_iters)
    x = root - eye
    xk = x
    result = x
    for k in range(2, taylor_order + 1):
        xk = xk @ x
        result = result + ((-1.0) ** (k + 1)) * xk / k
    return result * float(2**n_sqrt)
