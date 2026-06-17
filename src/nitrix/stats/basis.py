# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Penalised spline bases for additive models.

A GAM smooth term ``s(x)`` is a spline basis expansion ``f(x) = B(x) beta``
with a roughness penalty ``beta^T S beta``; the smoothing parameter trades the
penalty off against fit (chosen by REML in ``gam.py``).  This module builds the
``(design, penalty)`` pair for a covariate vector and re-evaluates the design
at new covariate values (for plotting partial effects).

We ship the **P-spline** (Eilers & Marx): a uniform cubic B-spline design with
a discrete difference penalty -- ``mgcv``'s ``bs='ps'`` -- because it reuses
the uniform-B-spline machinery already validated for N4 (the closed-form weight
evaluator here mirrors ``bias._bspline._uniform_bspline_weights``) and gives an
exact reference to test against.

Identifiability
---------------

A smooth competes with the model intercept unless it is constrained to sum to
zero over the data (``sum_i f(x_i) = 0``, i.e. ``(1^T B) beta = 0``).  We
absorb that single constraint by reparameterising ``beta = Z alpha`` with ``Z``
an orthonormal basis of the constraint's null space, computed by a **single
Householder reflection** -- no ``qr`` (cuSOLVER) -- so the construction runs on
the broken-cuSOLVER GPU and stays differentiable.

Everything is value -> value and cuSOLVER-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

__all__ = ['SplineBasis', 'bspline_basis', 'spline_design']


# ---------------------------------------------------------------------------
# Uniform B-spline weights (mirrors bias._bspline._uniform_bspline_weights)
# ---------------------------------------------------------------------------


def _bspline_weights(
    t: Float[Array, ' n'], degree: int
) -> Float[Array, 'n degree_plus_1']:
    """The ``degree + 1`` non-zero uniform B-spline weights at fractions ``t``.

    Closed forms for the orders that matter (1 linear, 2 quadratic, 3 cubic --
    the default).  A partition of unity (rows sum to 1).
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
        f'degree={degree!r}: bspline_basis ships uniform B-splines of degree '
        '1, 2, 3 (3 = cubic is the default). Higher orders are the Cox--de '
        'Boor recursion (these are its degree-1/2/3 specialisations).'
    )


def _bspline_design(
    x: Float[Array, ' n'],
    n_basis: int,
    degree: int,
    lo: float,
    hi: float,
) -> Float[Array, 'n n_basis']:
    """Banded uniform-B-spline design ``B[i, a]`` at arbitrary positions ``x``.

    ``x`` is mapped to the parametric coordinate ``s in [0, n_spans]`` with
    ``n_spans = n_basis - degree``; each row has ``degree + 1`` non-zeros at
    columns ``span .. span + degree``.
    """
    n_spans = n_basis - degree
    if n_spans < 1:
        raise ValueError(
            f'n_basis={n_basis} too small for degree={degree}: need at least '
            f'degree + 1 = {degree + 1} basis functions.'
        )
    s = (x - lo) / (hi - lo) * n_spans
    span = jnp.clip(jnp.floor(s).astype(jnp.int32), 0, n_spans - 1)
    frac = s - span.astype(x.dtype)
    w = _bspline_weights(frac, degree)  # (n, degree + 1)
    n = x.shape[0]
    rows = jnp.arange(n)[:, None]
    cols = span[:, None] + jnp.arange(degree + 1)[None, :]
    design = jnp.zeros((n, n_basis), dtype=x.dtype)
    return design.at[rows, cols].add(w)


def _difference_penalty(
    n_basis: int, order: int, dtype: Any
) -> Float[Array, 'n_basis n_basis']:
    """The ``order``-th difference (P-spline) penalty ``D^T D`` (``k, k``)."""
    diff = np.diff(np.eye(n_basis), n=order, axis=0)
    return jnp.asarray(diff.T @ diff, dtype=dtype)


def _householder_null(
    c: Float[Array, ' k'],
) -> Float[Array, 'k k_minus_1']:
    """Orthonormal basis ``Z`` (``k, k-1``) of the null space of row ``c``.

    A single Householder reflection ``H`` with ``H c proportional to e_1``;
    ``Z = H[:, 1:]`` then satisfies ``c @ Z = 0`` with orthonormal columns.
    Pure elementwise algebra -- no ``qr`` / cuSOLVER -- and differentiable.
    """
    norm = jnp.linalg.norm(c)
    sign = jnp.where(c[0] >= 0, 1.0, -1.0)
    v = c.at[0].add(sign * norm)
    vv = jnp.dot(v, v)
    k = c.shape[0]
    eye = jnp.eye(k, dtype=c.dtype)
    h = eye - 2.0 * jnp.outer(v, v) / jnp.where(vv > 0, vv, 1.0)
    return h[:, 1:]


# ---------------------------------------------------------------------------
# Public container + builder
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SplineBasis:
    """A penalised spline smooth: design, penalty, and re-evaluation params.

    Attributes
    ----------
    design
        ``(n, k)`` basis design at the construction covariate (post-constraint).
    penalty
        ``(k, k)`` roughness penalty ``S`` (post-constraint).
    n_basis, degree, penalty_order, lo, hi
        The construction parameters, kept so the design can be rebuilt at new
        covariate values (``spline_design``).
    constraint
        ``(k0, k)`` sum-to-zero reparameterisation ``Z`` (``None`` if
        unconstrained); ``k = k0 - 1`` when present.
    """

    design: Float[Array, 'n k']
    penalty: Float[Array, 'k k']
    n_basis: int
    degree: int
    penalty_order: int
    lo: float
    hi: float
    constraint: Optional[Float[Array, 'k0 k']]

    @property
    def dim(self) -> int:
        """Number of (post-constraint) basis coefficients ``k``."""
        return self.design.shape[-1]

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Any, ...], Tuple[int, int, int, float, float]]:
        children = (self.design, self.penalty, self.constraint)
        aux = (self.n_basis, self.degree, self.penalty_order, self.lo, self.hi)
        return children, aux

    @classmethod
    def tree_unflatten(
        cls, aux: Tuple[int, int, int, float, float], children: Tuple[Any, ...]
    ) -> 'SplineBasis':
        design, penalty, constraint = children
        n_basis, degree, penalty_order, lo, hi = aux
        return cls(
            design=design,
            penalty=penalty,
            n_basis=n_basis,
            degree=degree,
            penalty_order=penalty_order,
            lo=lo,
            hi=hi,
            constraint=constraint,
        )


def bspline_basis(
    x: Float[Array, ' n'],
    n_basis: int = 10,
    *,
    degree: int = 3,
    penalty_order: int = 2,
    bounds: Optional[Tuple[float, float]] = None,
    center: bool = True,
) -> SplineBasis:
    """Build a P-spline smooth basis for covariate ``x``.

    Parameters
    ----------
    x
        ``(n,)`` covariate values.
    n_basis
        Number of B-spline basis functions ``k`` (``mgcv``'s ``k``; default
        ``10``).
    degree
        B-spline degree (default ``3``, cubic).
    penalty_order
        Order of the difference penalty (default ``2`` -- penalise curvature,
        the standard P-spline).
    bounds
        ``(lo, hi)`` knot range.  Defaults to a small margin around the data
        range so the boundary knots are not exactly at the extreme points.
    center
        Apply the sum-to-zero identifiability constraint (default ``True``);
        the returned basis then has ``k - 1`` columns.

    Returns
    -------
    ``SplineBasis`` (design, penalty, and re-evaluation parameters).
    """
    if not 1 <= penalty_order < n_basis:
        raise ValueError(
            f'penalty_order={penalty_order} must satisfy 1 <= order < '
            f'n_basis={n_basis}: an order >= n_basis gives an empty difference '
            'operator and a silently-zero (unpenalised) penalty.'
        )
    x = jnp.asarray(x)
    if bounds is None:
        lo = float(jnp.min(x))
        hi = float(jnp.max(x))
        margin = 1e-3 * (hi - lo) + 1e-6
        lo, hi = lo - margin, hi + margin
    else:
        lo, hi = float(bounds[0]), float(bounds[1])

    design = _bspline_design(x, n_basis, degree, lo, hi)
    penalty = _difference_penalty(n_basis, penalty_order, x.dtype)

    constraint: Optional[Array] = None
    if center:
        col_sums = jnp.sum(design, axis=0)  # (k,)
        constraint = _householder_null(col_sums)  # (k, k-1)
        design = design @ constraint
        penalty = constraint.T @ penalty @ constraint

    return SplineBasis(
        design=design,
        penalty=penalty,
        n_basis=n_basis,
        degree=degree,
        penalty_order=penalty_order,
        lo=lo,
        hi=hi,
        constraint=constraint,
    )


def spline_design(
    basis: SplineBasis, x: Float[Array, ' m']
) -> Float[Array, 'm k']:
    """Re-evaluate a ``SplineBasis`` design at new covariate values ``x``.

    Used to render a smooth's partial effect on a fresh grid.  Applies the same
    knot mapping and sum-to-zero constraint as the original construction.
    """
    design = _bspline_design(
        jnp.asarray(x), basis.n_basis, basis.degree, basis.lo, basis.hi
    )
    if basis.constraint is not None:
        design = design @ basis.constraint
    return design
