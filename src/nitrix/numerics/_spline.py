# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Shared 1-D B-spline basis primitives.

The uniform cubic-B-spline weight evaluator and the difference (P-spline)
penalty are used by two otherwise-unrelated subsystems: ``bias._bspline`` (the
N4 multilevel-B-spline field smoother) and ``stats.basis`` (the GAM penalised
spline smooths).  They live here, in a neutral low-level home, so the closed
forms and the knot/penalty conventions have a single source of truth -- the
Cox--de Boor extension point and any convention fix are made once.

- ``uniform_bspline_weights(t, degree)`` -- the ``degree + 1`` non-zero
  uniform B-spline basis weights at fractional positions ``t in [0, 1]``
  (a partition of unity).  Uniform (non-clamped) knots, matching ITK/ANTs N4.
- ``difference_penalty_1d(n, order, dtype)`` -- the order-``m`` finite-
  difference penalty ``D^T D`` (``n, n``), the Eilers--Marx P-spline roughness
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
    """The ``degree + 1`` non-zero uniform B-spline weights at fractions ``t``.

    Closed forms for the orders that matter (1 linear, 2 quadratic, 3 cubic --
    the N4 / GAM default).  Rows sum to 1 (partition of unity).  Higher orders
    are the uniform-knot Cox--de Boor recursion (these are its degree-1/2/3
    specialisations) -- the single extension point.
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
    """The order-``order`` difference (P-spline) penalty ``D^T D`` (``n, n``).

    ``D`` is the ``order``-th finite-difference operator on ``n`` coefficients;
    ``D^T D`` has rank ``n - order`` (its null space is the degree-``order-1``
    polynomials).  The 1-D building block of the tensor-product penalty.
    """
    diff = np.diff(np.eye(n), n=order, axis=0)
    return jnp.asarray(diff.T @ diff, dtype=dtype)
