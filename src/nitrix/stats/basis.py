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
from typing import Any, Optional, Tuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from ..numerics._spline import difference_penalty_1d, uniform_bspline_weights

__all__ = [
    'SplineBasis',
    'bspline_basis',
    'thinplate_regression_basis',
    'spline_design',
]


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
    w = uniform_bspline_weights(frac, degree)  # (n, degree + 1)
    n = x.shape[0]
    rows = jnp.arange(n)[:, None]
    cols = span[:, None] + jnp.arange(degree + 1)[None, :]
    design = jnp.zeros((n, n_basis), dtype=x.dtype)
    return design.at[rows, cols].add(w)


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

    The re-evaluation at new covariate values factors as ``design(x) =
    raw_features(x) @ constraint`` for every basis ``kind``: only the
    (nonlinear) ``raw_features`` differs -- B-spline taps for ``'bspline'``, the
    radial + polynomial features for ``'tprs'`` (thin-plate).  ``constraint`` is
    the shared sum-to-zero reparameterisation.

    Attributes
    ----------
    design
        ``(n, k)`` basis design at the construction covariate (post-constraint).
    penalty
        ``(k, k)`` roughness penalty ``S`` (post-constraint).
    kind
        ``'bspline'`` (P-spline) or ``'tprs'`` (thin-plate regression spline).
    constraint
        ``(k0, k)`` sum-to-zero reparameterisation ``Z`` (``None`` if
        unconstrained); ``k = k0 - 1`` when present.
    n_basis, degree, penalty_order, lo, hi
        B-spline construction parameters.
    knots, radial_transform
        TPRS re-evaluation parameters: the knot positions and the radial
        eigen-truncation ``U_k`` (``None`` for B-splines).
    """

    design: Float[Array, 'n k']
    penalty: Float[Array, 'k k']
    kind: str
    constraint: Optional[Float[Array, 'k0 k']]
    n_basis: int
    degree: int
    penalty_order: int
    lo: float
    hi: float
    knots: Optional[Float[Array, ' nk']] = None
    radial_transform: Optional[Float[Array, 'nk kw']] = None

    @property
    def dim(self) -> int:
        """Number of (post-constraint) basis coefficients ``k``."""
        return self.design.shape[-1]

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Any, ...], Tuple[str, int, int, int, float, float]]:
        children = (
            self.design,
            self.penalty,
            self.constraint,
            self.knots,
            self.radial_transform,
        )
        aux = (
            self.kind,
            self.n_basis,
            self.degree,
            self.penalty_order,
            self.lo,
            self.hi,
        )
        return children, aux

    @classmethod
    def tree_unflatten(
        cls,
        aux: Tuple[str, int, int, int, float, float],
        children: Tuple[Any, ...],
    ) -> 'SplineBasis':
        design, penalty, constraint, knots, radial_transform = children
        kind, n_basis, degree, penalty_order, lo, hi = aux
        return cls(
            design=design,
            penalty=penalty,
            kind=kind,
            constraint=constraint,
            n_basis=n_basis,
            degree=degree,
            penalty_order=penalty_order,
            lo=lo,
            hi=hi,
            knots=knots,
            radial_transform=radial_transform,
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
    penalty = difference_penalty_1d(n_basis, penalty_order, x.dtype)

    constraint: Optional[Array] = None
    if center:
        col_sums = jnp.sum(design, axis=0)  # (k,)
        constraint = _householder_null(col_sums)  # (k, k-1)
        design = design @ constraint
        penalty = constraint.T @ penalty @ constraint

    return SplineBasis(
        design=design,
        penalty=penalty,
        kind='bspline',
        constraint=constraint,
        n_basis=n_basis,
        degree=degree,
        penalty_order=penalty_order,
        lo=lo,
        hi=hi,
    )


def _raw_features(
    basis: SplineBasis, x: Float[Array, ' m']
) -> Float[Array, 'm k0']:
    """The (nonlinear, pre-constraint) basis features at ``x`` for the kind."""
    x = jnp.asarray(x)
    if basis.kind == 'bspline':
        return _bspline_design(
            x, basis.n_basis, basis.degree, basis.lo, basis.hi
        )
    if basis.kind == 'tprs':
        radial = _tps_radial(x, cast(Array, basis.knots))
        poly = _tps_poly(x, basis.penalty_order, basis.lo, basis.hi)
        return jnp.concatenate(
            [radial @ cast(Array, basis.radial_transform), poly], axis=1
        )
    raise ValueError(f'unknown spline kind {basis.kind!r}.')


def spline_design(
    basis: SplineBasis, x: Float[Array, ' m']
) -> Float[Array, 'm k']:
    """Re-evaluate a ``SplineBasis`` design at new covariate values ``x``.

    Used to render a smooth's partial effect on a fresh grid.  Applies the same
    feature map and sum-to-zero constraint as the original construction.
    """
    design = _raw_features(basis, x)
    if basis.constraint is not None:
        design = design @ basis.constraint
    return design


# ---------------------------------------------------------------------------
# Thin-plate regression spline (mgcv bs='tp', the default smoother)
# ---------------------------------------------------------------------------


def _tps_radial(
    x: Float[Array, ' m'], knots: Float[Array, ' nk']
) -> Float[Array, 'm nk']:
    """Thin-plate radial basis ``eta(|x_i - knot_j|)`` (1-D, ``m = 2``).

    The 1-D order-2 thin-plate Green's function is ``|r|^3`` (the constant is
    absorbed into the smoothing parameter).
    """
    r = jnp.abs(x[:, None] - knots[None, :])
    return r**3


def _tps_poly(
    x: Float[Array, ' m'], order: int, lo: float, hi: float
) -> Float[Array, 'm order']:
    """Polynomial null-space basis of the thin-plate penalty: ``[1, t, ...,
    t^{order-1}]`` on a rescaled ``t in [0, 1]`` (the unpenalised part)."""
    t = (x - lo) / (hi - lo)
    return jnp.stack([t**j for j in range(order)], axis=-1)


def thinplate_regression_basis(
    x: Float[Array, ' n'],
    n_basis: int = 10,
    *,
    penalty_order: int = 2,
    max_knots: int = 100,
    bounds: Optional[Tuple[float, float]] = None,
    center: bool = True,
) -> SplineBasis:
    """Build a thin-plate regression spline (TPRS) basis for covariate ``x``.

    The ``mgcv`` default smoother (``bs='tp'``): isotropic and knot-free.  The
    full thin-plate basis (one function per knot) is truncated by an
    eigendecomposition of the radial penalty matrix ``E`` -- the ``k - M``
    leading (largest-eigenvalue) wiggly components, plus the ``M = penalty_order``
    unpenalised polynomial null-space terms (Wood 2003).  The truncation
    ``eigh`` is a one-off host (CPU/numpy) computation -- it is data-dependent
    but not per-trace, and avoids the cuSOLVER ``syevd`` on the broken GPU.

    Parameters
    ----------
    x
        ``(n,)`` covariate values.
    n_basis
        Total basis dimension ``k`` (wiggly + polynomial; ``mgcv``'s ``k``).
    penalty_order
        Thin-plate order ``m`` (default ``2`` -- penalise curvature); the
        polynomial null space has dimension ``M = m``.
    max_knots
        Cap on the number of knots (placed at quantiles of ``x`` when ``n``
        exceeds it), bounding the ``O(n_knots^3)`` eigendecomposition.
    bounds
        ``(lo, hi)`` for the polynomial rescaling (defaults to the data range).
    center
        Apply the sum-to-zero identifiability constraint (default ``True``).

    Returns
    -------
    ``SplineBasis`` (``kind='tprs'``).
    """
    x = jnp.asarray(x)
    n = x.shape[0]
    m = penalty_order
    if not 1 <= m < n_basis:
        raise ValueError(
            f'penalty_order={m} must satisfy 1 <= m < n_basis={n_basis}.'
        )
    if bounds is None:
        lo, hi = float(jnp.min(x)), float(jnp.max(x))
        lo, hi = lo - 1e-6, hi + 1e-6
    else:
        lo, hi = float(bounds[0]), float(bounds[1])

    # Knots: the data, or a quantile subsample when n exceeds max_knots.
    x_np = np.asarray(x)
    if n > max_knots:
        qs = np.linspace(0.0, 1.0, max_knots)
        knots_np = np.quantile(x_np, qs)
    else:
        knots_np = x_np
    n_knots = knots_np.shape[0]
    k_wiggly = n_basis - m
    if k_wiggly > n_knots:
        raise ValueError(
            f'n_basis - penalty_order = {k_wiggly} exceeds the {n_knots} '
            'available knots; raise max_knots or lower n_basis.'
        )

    # Radial penalty matrix on the knots; truncate to the k_wiggly leading
    # (largest, most positive) eigenpairs -> a PSD wiggly penalty diag(Dk).
    e_knots = np.abs(knots_np[:, None] - knots_np[None, :]) ** 3
    evals, evecs = np.linalg.eigh(e_knots)
    order = np.argsort(evals)[::-1][:k_wiggly]
    uk = evecs[:, order]  # (n_knots, k_wiggly)
    dk = evals[order]  # (k_wiggly,)
    knots = jnp.asarray(knots_np, dtype=x.dtype)
    radial_transform = jnp.asarray(uk, dtype=x.dtype)

    # Construction design: [E(x, knots) Uk | poly(x)] and the block penalty.
    wiggly = _tps_radial(x, knots) @ radial_transform  # (n, k_wiggly)
    poly = _tps_poly(x, m, lo, hi)  # (n, m)
    design = jnp.concatenate([wiggly, poly], axis=1)  # (n, n_basis)
    penalty = jnp.zeros((n_basis, n_basis), dtype=x.dtype)
    penalty = penalty.at[:k_wiggly, :k_wiggly].set(
        jnp.diag(jnp.asarray(dk, dtype=x.dtype))
    )

    constraint: Optional[Array] = None
    if center:
        col_sums = jnp.sum(design, axis=0)
        constraint = _householder_null(col_sums)
        design = design @ constraint
        penalty = constraint.T @ penalty @ constraint

    return SplineBasis(
        design=design,
        penalty=penalty,
        kind='tprs',
        constraint=constraint,
        n_basis=n_basis,
        degree=3,
        penalty_order=m,
        lo=lo,
        hi=hi,
        knots=knots,
        radial_transform=radial_transform,
    )
