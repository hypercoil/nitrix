# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Shared 1-D B-spline basis primitives.

The uniform cubic-B-spline weight evaluator and the difference (P-spline)
penalty are used by two otherwise-unrelated subsystems: ``bias._bspline`` (the
N4 multilevel-B-spline field smoother) and ``stats.basis`` (the GAM penalised
spline smooths).  They live here, in a neutral low-level home, so the closed
forms and the knot/penalty conventions have a single source of truth -- the
Cox--de Boor extension point and any convention fix are made once.

- :func:`uniform_bspline_weights` -- the ``degree + 1`` non-zero uniform
  B-spline basis weights at fractional positions ``t`` in :math:`[0, 1]`
  (a partition of unity).  Uniform (non-clamped) knots, matching ITK/ANTs N4.
- :func:`difference_penalty_1d` -- the order-``m`` finite-difference penalty
  :math:`D^{\top} D` (shape ``(n, n)``), the Eilers--Marx P-spline roughness
  penalty; the 1-D building block of the tensor-product penalty.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

__all__ = ['uniform_bspline_weights', 'difference_penalty_1d']


def uniform_bspline_weights(
    t: Float[Array, ' n'], degree: int
) -> Float[Array, 'n degree_plus_1']:
    """Evaluate the ``degree + 1`` non-zero uniform B-spline weights.

    For each fractional position ``t`` in :math:`[0, 1]`, returns the
    ``degree + 1`` non-zero uniform (non-clamped knot) B-spline basis weights
    covering that position, matching the ITK/ANTs N4 convention.  Closed forms
    are provided for the orders that matter in practice: degree 1 (linear),
    degree 2 (quadratic) and degree 3 (cubic, the N4 / GAM default).  These are
    the degree-1/2/3 specialisations of the uniform-knot Cox--de Boor
    recursion, which is the single extension point for higher orders.  The
    weights in each row sum to 1 (partition of unity).

    Parameters
    ----------
    t : Float[Array, ' n']
        Fractional positions in :math:`[0, 1]` at which to evaluate the basis,
        one per sample.
    degree : int
        Polynomial degree of the B-spline basis.  Supported values are 1, 2
        and 3; any other value raises :class:`NotImplementedError`.

    Returns
    -------
    Float[Array, 'n degree_plus_1']
        The ``degree + 1`` non-zero basis weights for each of the ``n`` input
        positions.  Each row sums to 1.
    """
    if degree == 1:
        return jnp.stack([1.0 - t, t], axis=-1)
    if degree == 2:
        return jnp.stack(
            [
                0.5 * (1.0 - t) ** 2,
                0.5 * (1.0 + 2.0 * t - 2.0 * t**2),
                0.5 * t**2,
            ],
            axis=-1,
        )
    if degree == 3:
        t2 = t**2
        t3 = t**3
        return jnp.stack(
            [
                (1.0 - t) ** 3 / 6.0,
                (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
                (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
                t3 / 6.0,
            ],
            axis=-1,
        )
    raise NotImplementedError(
        f'degree={degree!r} is unsupported; the uniform B-spline closed forms '
        'ship for degree 1, 2, 3 (3 = cubic is the N4 / ANTs default). Higher '
        'orders are the uniform-knot Cox--de Boor recursion -- add it here.'
    )


def difference_penalty_1d(
    n: int, order: int, dtype: Any
) -> Float[Array, 'n n']:
    r"""Build the order-``order`` difference (P-spline) penalty matrix.

    Constructs the Eilers--Marx P-spline roughness penalty :math:`D^{\top} D`,
    where ``D`` is the ``order``-th finite-difference operator acting on ``n``
    coefficients.  The resulting matrix has rank :math:`n - \text{order}`; its
    null space is spanned by the polynomials of degree :math:`\text{order} - 1`,
    which are therefore left unpenalised.  This is the 1-D building block of the
    tensor-product penalty.

    Parameters
    ----------
    n : int
        Number of spline coefficients, i.e. the side length of the returned
        square penalty matrix.
    order : int
        Order of the finite difference used to build ``D``.
    dtype : Any
        Floating-point dtype of the returned penalty matrix.

    Returns
    -------
    Float[Array, 'n n']
        The symmetric positive-semidefinite penalty matrix
        :math:`D^{\top} D` of shape ``(n, n)``.
    """
    diff = np.diff(np.eye(n), n=order, axis=0)
    return jnp.asarray(diff.T @ diff, dtype=dtype)
