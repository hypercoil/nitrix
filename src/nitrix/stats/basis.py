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

from dataclasses import dataclass, replace
from typing import Any, Optional, Tuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from ..graph import laplacian
from ..numerics._spline import difference_penalty_1d, uniform_bspline_weights

__all__ = [
    'SplineBasis',
    'TensorBasis',
    'REBasis',
    'bspline_basis',
    'cyclic_cubic_basis',
    'thinplate_regression_basis',
    'cr_basis',
    'gp_basis',
    'mrf_smooth',
    'tensor_product_basis',
    're_smooth',
    'by_factor_smooth',
    'varying_coefficient_smooth',
    'spline_design',
    'tensor_product_design',
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
    # Clamp the within-span coordinate to [0, 1]: for x outside [lo, hi] the span
    # is already clamped to the boundary, and clamping frac too gives constant
    # boundary extrapolation (a valid partition of unity) instead of evaluating
    # the boundary cubic at frac > 1 / < 0, which loses the partition and diverges.
    frac = jnp.clip(s - span.astype(x.dtype), 0.0, 1.0)
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
        ``'bspline'`` (P-spline), ``'cyclic'`` (cyclic P-spline), or ``'tprs'``
        (thin-plate regression spline).
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
    kernel_param: Optional[float] = None  # kernel range for kind='gp'

    @property
    def dim(self) -> int:
        """Number of (post-constraint) basis coefficients ``k``."""
        return self.design.shape[-1]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[Any, ...], Tuple[str, int, int, int, float, float, Optional[float]]
    ]:
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
            self.kernel_param,
        )
        return children, aux

    @classmethod
    def tree_unflatten(
        cls,
        aux: Tuple[str, int, int, int, float, float, Optional[float]],
        children: Tuple[Any, ...],
    ) -> 'SplineBasis':
        design, penalty, constraint, knots, radial_transform = children
        kind, n_basis, degree, penalty_order, lo, hi, kernel_param = aux
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
            kernel_param=kernel_param,
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
    if basis.kind == 'cyclic':
        return _cyclic_bspline_design(
            x, basis.n_basis, basis.degree, basis.lo, basis.hi
        )
    if basis.kind == 'tprs':
        radial = _tps_radial(x, cast(Array, basis.knots))
        poly = _tps_poly(x, basis.penalty_order, basis.lo, basis.hi)
        return jnp.concatenate(
            [radial @ cast(Array, basis.radial_transform), poly], axis=1
        )
    if basis.kind == 'cr':
        return _cr_design(
            x, cast(Array, basis.knots), cast(Array, basis.radial_transform)
        )
    if basis.kind == 'gp':
        knots = cast(Array, basis.knots)
        cxz = _matern32_kernel(
            jnp.abs(x[:, None] - knots[None, :]),
            cast(float, basis.kernel_param),
        )
        return cxz @ cast(Array, basis.radial_transform)
    if basis.kind == 'mrf':
        return jax.nn.one_hot(
            x.astype(jnp.int32), basis.n_basis, dtype=jnp.float32
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


# ---------------------------------------------------------------------------
# Cyclic cubic P-spline (mgcv bs='cc'/'cp', for periodic covariates)
# ---------------------------------------------------------------------------


def _cyclic_bspline_design(
    x: Float[Array, ' n'],
    n_basis: int,
    degree: int,
    lo: float,
    hi: float,
) -> Float[Array, 'n n_basis']:
    """Uniform *cyclic* B-spline design: the same weights as ``_bspline_design``
    but with the parametric coordinate taken modulo the period and the column
    indices wrapped, so the basis is periodic (``f(lo) == f(hi)``)."""
    s = jnp.mod((x - lo) / (hi - lo) * n_basis, n_basis)
    span = jnp.floor(s).astype(jnp.int32)
    frac = s - span.astype(x.dtype)
    w = uniform_bspline_weights(frac, degree)  # (n, degree + 1)
    n = x.shape[0]
    rows = jnp.arange(n)[:, None]
    cols = jnp.mod(span[:, None] + jnp.arange(degree + 1)[None, :], n_basis)
    design = jnp.zeros((n, n_basis), dtype=x.dtype)
    return design.at[rows, cols].add(w)


def _circular_difference_penalty(
    n_basis: int, order: int, dtype: Any
) -> Float[Array, 'n_basis n_basis']:
    """Circular ``order``-th difference penalty ``D^T D`` (wrap-around)."""
    d1 = -np.eye(n_basis) + np.roll(np.eye(n_basis), -1, axis=1)
    d = np.linalg.matrix_power(d1, order)
    return jnp.asarray(d.T @ d, dtype=dtype)


def cyclic_cubic_basis(
    x: Float[Array, ' n'],
    n_basis: int = 10,
    *,
    degree: int = 3,
    penalty_order: int = 2,
    bounds: Optional[Tuple[float, float]] = None,
    center: bool = True,
) -> SplineBasis:
    """Build a cyclic cubic P-spline basis for a periodic covariate ``x``.

    For periodic covariates (cortical angle, phase, time-of-day): a uniform
    cyclic B-spline design (the basis wraps at the period ends, matching ``f``,
    ``f'``, ``f''``) with a circular difference penalty (``mgcv``'s ``bs='cp'``).

    Parameters
    ----------
    x
        ``(n,)`` covariate values.
    n_basis
        Number of cyclic basis functions ``k``.
    degree, penalty_order
        B-spline degree (default ``3``) and circular difference order (``2``).
    bounds
        ``(lo, hi)`` = the period; ``hi`` is identified with ``lo``.  Defaults
        to the data range (pass the true period for a partial-cycle sample).
    center
        Sum-to-zero identifiability constraint (default ``True``).

    Returns
    -------
    ``SplineBasis`` (``kind='cyclic'``).
    """
    if not 1 <= penalty_order < n_basis:
        raise ValueError(
            f'penalty_order={penalty_order} must satisfy 1 <= order < '
            f'n_basis={n_basis}.'
        )
    x = jnp.asarray(x)
    if bounds is None:
        lo, hi = float(jnp.min(x)), float(jnp.max(x))
    else:
        lo, hi = float(bounds[0]), float(bounds[1])

    design = _cyclic_bspline_design(x, n_basis, degree, lo, hi)
    penalty = _circular_difference_penalty(n_basis, penalty_order, x.dtype)

    constraint: Optional[Array] = None
    if center:
        col_sums = jnp.sum(design, axis=0)
        constraint = _householder_null(col_sums)
        design = design @ constraint
        penalty = constraint.T @ penalty @ constraint

    return SplineBasis(
        design=design,
        penalty=penalty,
        kind='cyclic',
        constraint=constraint,
        n_basis=n_basis,
        degree=degree,
        penalty_order=penalty_order,
        lo=lo,
        hi=hi,
    )


# ---------------------------------------------------------------------------
# Cubic regression spline (mgcv bs='cr') -- knot-value parameterised
# ---------------------------------------------------------------------------


def _cr_construct(
    knots: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """The mgcv ``cr`` construction: the ``beta -> f''`` map ``F`` and the
    integrated-curvature penalty ``S = D^T B^{-1} D``.

    A natural cubic spline parameterised by its values ``beta`` at the knots.
    With ``h_i`` the knot spacings, ``B`` (tridiagonal) and ``D`` map the knot
    values to the interior second derivatives ``m = B^{-1} D beta`` (``f''`` is
    zero at the ends -- the *natural* boundary); ``F`` pads ``m`` with those
    zeros, and ``S`` is the exact ``integral f''(x)^2 dx``.  Host-side (the knots
    are static across the mass axis).
    """
    k = knots.shape[0]
    h = np.diff(knots)  # (k-1,)
    D = np.zeros((k - 2, k))
    B = np.zeros((k - 2, k - 2))
    for i in range(k - 2):
        D[i, i] = 1.0 / h[i]
        D[i, i + 1] = -1.0 / h[i] - 1.0 / h[i + 1]
        D[i, i + 2] = 1.0 / h[i + 1]
        B[i, i] = (h[i] + h[i + 1]) / 3.0
        if i < k - 3:
            B[i, i + 1] = h[i + 1] / 6.0
            B[i + 1, i] = h[i + 1] / 6.0
    binv_d = np.linalg.solve(B, D)  # (k-2, k)
    F = np.zeros((k, k))
    F[1:-1, :] = binv_d  # f''_1 = f''_k = 0 (natural)
    S = D.T @ binv_d  # (k, k) penalty
    return F, 0.5 * (S + S.T)


def _cr_design(
    x: Float[Array, ' m'],
    knots: Float[Array, ' k'],
    fmap: Float[Array, 'k k'],
) -> Float[Array, 'm k']:
    """Cardinal cubic-spline design at ``x`` (the knot-value interpolation)."""
    x = jnp.asarray(x)
    k = knots.shape[0]
    j = jnp.clip(jnp.searchsorted(knots, x) - 1, 0, k - 2)  # (m,) interval
    xl = knots[j]
    xr = knots[j + 1]
    h = xr - xl
    am = (xr - x) / h
    ap = (x - xl) / h
    cm = ((xr - x) ** 3 / h - h * (xr - x)) / 6.0
    cp = ((x - xl) ** 3 / h - h * (x - xl)) / 6.0
    base = am[:, None] * jax.nn.one_hot(j, k, dtype=x.dtype) + ap[
        :, None
    ] * jax.nn.one_hot(j + 1, k, dtype=x.dtype)
    return base + cm[:, None] * fmap[j] + cp[:, None] * fmap[j + 1]


def _strictly_increasing(v: np.ndarray) -> np.ndarray:
    """Nudge a sorted vector to be strictly increasing (de-duplicate knots)."""
    out = v.astype(np.float64).copy()
    span = max(float(out[-1] - out[0]), 1.0)
    for i in range(1, out.shape[0]):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + 1e-9 * span
    return out


def cr_basis(
    x: Float[Array, ' n'],
    n_basis: int = 10,
    *,
    bounds: Optional[Tuple[float, float]] = None,
    center: bool = True,
) -> SplineBasis:
    """Build a cubic regression spline (mgcv ``bs='cr'``) for covariate ``x``.

    A natural cubic spline parameterised by its values at ``n_basis`` knots
    (placed at quantiles of ``x``), with the exact integrated-curvature penalty.
    The cheap knot-based 1-D smoother: ``O(k)`` design, no eigen-truncation.
    Equivalent to ``scipy``'s natural ``CubicSpline`` interpolation at the knots.

    Parameters
    ----------
    x
        ``(n,)`` covariate values.
    n_basis
        Number of knots / basis coefficients ``k`` (>= 4).
    bounds
        Unused for ``cr`` (knots come from the data quantiles); accepted for a
        uniform smooth-constructor signature.
    center
        Apply the sum-to-zero identifiability constraint (default ``True``).

    Returns
    -------
    ``SplineBasis`` (``kind='cr'``).
    """
    if n_basis < 4:
        raise ValueError(f'cr_basis: n_basis={n_basis} must be >= 4.')
    x = jnp.asarray(x)
    x_np = np.asarray(x)
    knots_np = _strictly_increasing(
        np.quantile(x_np, np.linspace(0.0, 1.0, n_basis))
    )
    fmap_np, s_np = _cr_construct(knots_np)
    knots = jnp.asarray(knots_np, dtype=x.dtype)
    fmap = jnp.asarray(fmap_np, dtype=x.dtype)
    design = _cr_design(x, knots, fmap)
    penalty = jnp.asarray(s_np, dtype=x.dtype)

    constraint: Optional[Array] = None
    if center:
        col_sums = jnp.sum(design, axis=0)
        constraint = _householder_null(col_sums)
        design = design @ constraint
        penalty = constraint.T @ penalty @ constraint

    return SplineBasis(
        design=design,
        penalty=penalty,
        kind='cr',
        constraint=constraint,
        n_basis=n_basis,
        degree=3,
        penalty_order=2,
        lo=float(knots_np[0]),
        hi=float(knots_np[-1]),
        knots=knots,
        radial_transform=fmap,
    )


# ---------------------------------------------------------------------------
# Gaussian-process smooth (mgcv bs='gp') -- a covariance-kernel basis
# ---------------------------------------------------------------------------


def _matern32_kernel(
    r: Float[Array, '...'], rho: float
) -> Float[Array, '...']:
    """Matern-3/2 correlation ``(1 + sqrt(3) r / rho) exp(-sqrt(3) r / rho)``."""
    s = jnp.sqrt(3.0) * r / rho
    return (1.0 + s) * jnp.exp(-s)


def gp_basis(
    x: Float[Array, ' n'],
    n_basis: int = 10,
    *,
    rho: Optional[float] = None,
    max_knots: int = 100,
    bounds: Optional[Tuple[float, float]] = None,
    center: bool = True,
) -> SplineBasis:
    """Build a Gaussian-process smooth (mgcv ``bs='gp'``) for covariate ``x``.

    A low-rank kriging smooth with a **Matern-3/2** covariance kernel of range
    ``rho``: the smooth is ``f(x) = sum_j C(|x - z_j|) delta_j`` over knots ``z``
    with the RKHS penalty ``delta^T C_zz delta`` (``C_zz`` the kernel Gram at the
    knots).  Constructed like the thin-plate basis -- the radial features are
    eigen-reparameterised so the penalty is diagonal and truncated to the
    ``n_basis`` leading components (a one-off host ``eigh``, cuSOLVER-free on the
    broken GPU).

    Parameters
    ----------
    x
        ``(n,)`` covariate values.
    n_basis
        Retained basis dimension ``k`` (the leading kriging components).
    rho
        Kernel range (correlation length).  Defaults to the data range / 2.
    max_knots
        Cap on the number of knots (quantile-placed) bounding the ``eigh`` cost.
    bounds
        ``(lo, hi)`` recorded for re-evaluation (defaults to the data range).
    center
        Apply the sum-to-zero identifiability constraint (default ``True``).

    Returns
    -------
    ``SplineBasis`` (``kind='gp'``).
    """
    x = jnp.asarray(x)
    n = x.shape[0]
    x_np = np.asarray(x, dtype=np.float64)
    lo = float(np.min(x_np)) if bounds is None else float(bounds[0])
    hi = float(np.max(x_np)) if bounds is None else float(bounds[1])
    rho_v = float(rho) if rho is not None else max((hi - lo) / 2.0, 1e-6)
    if not 1 <= n_basis:
        raise ValueError(f'gp_basis: n_basis={n_basis} must be >= 1.')

    knots_np = _strictly_increasing(
        np.quantile(x_np, np.linspace(0.0, 1.0, min(max_knots, n)))
    )
    nk = knots_np.shape[0]
    if n_basis > nk:
        raise ValueError(
            f'gp_basis: n_basis={n_basis} exceeds the {nk} available knots.'
        )
    # Kernel Gram at the knots; eigen-reparameterise (host) so the penalty is
    # diagonal, then keep the n_basis leading kriging components.
    rk = np.abs(knots_np[:, None] - knots_np[None, :])
    czz = np.asarray(_matern32_kernel(jnp.asarray(rk), rho_v))
    evals, evecs = np.linalg.eigh(0.5 * (czz + czz.T))
    order = np.argsort(evals)[::-1][:n_basis]
    evals_k = np.clip(evals[order], 1e-10, None)
    uk = evecs[:, order]  # (nk, k)
    # f(x) = C_xz U_k diag(1/lambda) alpha; penalty alpha^T alpha on the
    # whitened coefficients -> identity penalty in the eigenbasis.
    radial_transform = jnp.asarray(uk / evals_k[None, :], dtype=x.dtype)
    knots = jnp.asarray(knots_np, dtype=x.dtype)

    def _gp_raw(xx: Float[Array, ' m']) -> Float[Array, 'm k']:
        cxz = _matern32_kernel(jnp.abs(xx[:, None] - knots[None, :]), rho_v)
        return cxz @ radial_transform

    design = _gp_raw(x)
    penalty = jnp.eye(n_basis, dtype=x.dtype)

    constraint: Optional[Array] = None
    if center:
        col_sums = jnp.sum(design, axis=0)
        constraint = _householder_null(col_sums)
        design = design @ constraint
        penalty = constraint.T @ penalty @ constraint

    return SplineBasis(
        design=design,
        penalty=penalty,
        kind='gp',
        constraint=constraint,
        n_basis=n_basis,
        degree=0,
        penalty_order=2,
        lo=lo,
        hi=hi,
        knots=knots,
        radial_transform=radial_transform,
        kernel_param=rho_v,
    )


# ---------------------------------------------------------------------------
# Markov-random-field smooth (mgcv bs='mrf') -- graph-Laplacian penalty
# ---------------------------------------------------------------------------


def mrf_smooth(
    labels: Int[Array, ' n'],
    neighbours: Float[Array, 'R R'],
    *,
    center: bool = True,
) -> SplineBasis:
    """Build a Markov-random-field smooth (mgcv ``bs='mrf'``) over a region
    adjacency -- the natural smoother on a parcel / vertex graph.

    Each observation carries an integer region ``label`` in ``0 .. R-1``; the
    smooth is a per-region effect (design = region indicator) penalised by the
    **combinatorial graph Laplacian** ``L = D - A`` of the region adjacency
    (``beta^T L beta = sum_{i~j} (beta_i - beta_j)^2`` -- neighbouring regions are
    shrunk together).  The penalty is ``nitrix.graph.laplacian`` -- a direct
    substrate reuse.

    Parameters
    ----------
    labels
        ``(n,)`` integer region index per observation (``0 .. R-1``).
    neighbours
        ``(R, R)`` region adjacency matrix (non-negative, symmetric; ``1`` for
        adjacent regions).
    center
        Apply the (count-weighted) sum-to-zero constraint (default ``True``).

    Returns
    -------
    ``SplineBasis`` (``kind='mrf'``); ``n_basis = R``.
    """
    labels = jnp.asarray(labels)
    a = jnp.asarray(neighbours)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(
            f'mrf_smooth: neighbours must be a square (R, R) adjacency; got '
            f'{a.shape}.'
        )
    r = a.shape[0]
    af = a.astype(jnp.result_type(a.dtype, jnp.float32))
    design = jax.nn.one_hot(labels.astype(jnp.int32), r, dtype=af.dtype)
    penalty = laplacian(0.5 * (af + af.T), normalisation='combinatorial')

    constraint: Optional[Array] = None
    if center:
        col_sums = jnp.sum(design, axis=0)
        constraint = _householder_null(col_sums)
        design = design @ constraint
        penalty = constraint.T @ penalty @ constraint

    return SplineBasis(
        design=design,
        penalty=penalty,
        kind='mrf',
        constraint=constraint,
        n_basis=r,
        degree=0,
        penalty_order=1,
        lo=0.0,
        hi=float(r - 1),
    )


# ---------------------------------------------------------------------------
# Tensor-product interaction smooth (mgcv te()/ti(), anisotropic interactions)
# ---------------------------------------------------------------------------


def _row_kron(designs: Tuple[np.ndarray, ...]) -> np.ndarray:
    """Row-wise Kronecker product of marginal designs.

    For designs ``[A (n, k1), B (n, k2), ...]`` the result is ``(n, k1*k2*...)``
    whose row ``i`` is the flattened outer product of the marginal rows -- the
    tensor-product basis evaluated at the matched covariate tuple of row ``i``.
    """
    out = designs[0]
    for d in designs[1:]:
        n, ka = out.shape
        kb = d.shape[1]
        out = (out[:, :, None] * d[:, None, :]).reshape(n, ka * kb)
    return out


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TensorBasis:
    """An anisotropic tensor-product interaction smooth ``f(x_1, ..., x_d)``.

    Built from ``d`` marginal :class:`SplineBasis` (already sum-to-zero
    constrained) by the **row-wise tensor product** of their designs and the
    Kronecker-sum of per-margin penalties ``S_j = I (x) ... P_j ... (x) I`` --
    so each margin keeps its own smoothing parameter (selected jointly by the
    GAM Fellner-Schall loop, which handles ``> 1`` penalty per smooth).  This is
    mgcv's tensor **interaction** (``ti``): the marginal centring removes the
    main effects, so the term models the *interaction* and stays identifiable
    against the intercept; add marginal ``s(x_j)`` smooths for a full ``te``.

    The marginal penalties commute (Kronecker structure), so they are
    **simultaneously diagonalisable**: ``pen_eig[j]`` is penalty ``j`` in the
    joint eigenbasis (``U_1 (x) ... (x) U_d``), which the GAM engine uses to
    evaluate the smoothing-parameter trace ``tr(S_lambda^+ S_j)`` as an
    elementwise sum -- cuSOLVER-free, no per-iteration pseudo-inverse -- while
    the fit itself keeps the original (non-rotated) tensor basis.

    Attributes
    ----------
    design
        ``(n, K)`` row-wise tensor design, ``K = prod_j k_j``.
    penalties
        ``(d, K, K)`` the ``d`` Kronecker-embedded marginal penalties ``S_j``.
    pen_eig
        ``(d, K)`` penalty ``j`` in the joint eigenbasis (its Kronecker
        eigenvalues), tiny values floored to exact zero for the rank.
    margins
        The constituent marginal bases, for re-evaluation on a fresh grid.
    dims
        ``(k_1, ..., k_d)`` per-margin basis dimensions.
    """

    design: Float[Array, 'n K']
    penalties: Float[Array, 'd K K']
    pen_eig: Float[Array, 'd K']
    margins: Tuple[SplineBasis, ...]
    dims: Tuple[int, ...]

    @property
    def dim(self) -> int:
        """Total tensor-basis dimension ``K = prod_j k_j``."""
        return self.design.shape[-1]

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Any, ...], Tuple[int, ...]]:
        children = (self.design, self.penalties, self.pen_eig, self.margins)
        return children, self.dims

    @classmethod
    def tree_unflatten(
        cls, aux: Tuple[int, ...], children: Tuple[Any, ...]
    ) -> 'TensorBasis':
        design, penalties, pen_eig, margins = children
        return cls(
            design=design,
            penalties=penalties,
            pen_eig=pen_eig,
            margins=tuple(margins),
            dims=aux,
        )


def tensor_product_basis(
    margins: Tuple[SplineBasis, ...],
    *,
    eig_rtol: float = 1e-10,
) -> TensorBasis:
    """Build an anisotropic tensor-product interaction smooth from marginals.

    Parameters
    ----------
    margins
        Two or more :class:`SplineBasis` on the **matched** covariates (same
        number of rows ``n``), each typically sum-to-zero constrained
        (``center=True``).  Mixed kinds are allowed (e.g. a P-spline crossed
        with a cyclic margin).
    eig_rtol
        Relative tolerance below which a marginal-penalty eigenvalue is floored
        to exact zero (its unpenalised null space), so the smoothing-parameter
        trace counts the correct penalty rank.

    Returns
    -------
    ``TensorBasis`` (design ``(n, prod k_j)`` and the per-margin penalties).
    """
    if len(margins) < 2:
        raise ValueError(
            'tensor_product_basis: need at least two marginal bases.'
        )
    n = margins[0].design.shape[0]
    for m in margins:
        if m.design.shape[0] != n:
            raise ValueError(
                'tensor_product_basis: all margins must share the row count '
                f'n={n}; got {m.design.shape[0]}.'
            )
    dtype = margins[0].design.dtype
    designs = tuple(np.asarray(m.design) for m in margins)
    dims = tuple(d.shape[1] for d in designs)
    pens = [np.asarray(m.penalty) for m in margins]

    design = _row_kron(designs)
    big_k = design.shape[1]

    # Per-margin eigendecomposition (the joint eigenbasis is the Kronecker
    # product; each S_j is diagonal there with the margin's eigenvalues tiled).
    eig_per_margin = []
    for w_src in pens:
        w, _ = np.linalg.eigh(w_src)
        floor = eig_rtol * max(float(w.max()), np.finfo(w.dtype).tiny)
        eig_per_margin.append(np.where(w > floor, w, 0.0))

    penalties = np.zeros((len(margins), big_k, big_k))
    pen_eig = np.zeros((len(margins), big_k))
    eye = [np.eye(k) for k in dims]
    for j in range(len(margins)):
        # S_j = I (x) ... P_j ... (x) I  (full, original tensor basis).
        mats = [eye[i] if i != j else pens[i] for i in range(len(margins))]
        s_full = mats[0]
        for mat in mats[1:]:
            s_full = np.kron(s_full, mat)
        penalties[j] = s_full
        # pen_eig[j] = 1 (x) ... eig_j ... (x) 1  (the Kronecker eigenvalues).
        vecs = [
            np.ones(dims[i]) if i != j else eig_per_margin[i]
            for i in range(len(margins))
        ]
        e = vecs[0]
        for vec in vecs[1:]:
            e = np.kron(e, vec)
        pen_eig[j] = e

    return TensorBasis(
        design=jnp.asarray(design, dtype=dtype),
        penalties=jnp.asarray(penalties, dtype=dtype),
        pen_eig=jnp.asarray(pen_eig, dtype=dtype),
        margins=tuple(margins),
        dims=dims,
    )


def tensor_product_design(
    basis: TensorBasis, xs: Tuple[Float[Array, ' g'], ...]
) -> Float[Array, 'g K']:
    """Re-evaluate a ``TensorBasis`` on a matched covariate grid.

    ``xs`` is one grid per margin (all length ``g``); point ``t`` has covariate
    tuple ``(xs[0][t], ..., xs[d-1][t])``.  Returns the ``(g, K)`` tensor design
    -- ``@`` the fitted tensor coefficients renders the interaction surface
    along that path.
    """
    if len(xs) != len(basis.margins):
        raise ValueError(
            f'tensor_product_design: got {len(xs)} grids for '
            f'{len(basis.margins)} margins.'
        )
    designs = tuple(
        np.asarray(spline_design(m, x)) for m, x in zip(basis.margins, xs)
    )
    return jnp.asarray(_row_kron(designs), dtype=basis.design.dtype)


# ---------------------------------------------------------------------------
# Random-effect smooth (mgcv bs='re') -- the GAMM block
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class REBasis:
    """A random-effect smooth block (mgcv ``bs="re"``) for a GAMM.

    A random intercept / slope per level of a grouping factor, expressed as the
    ``(design, penalty)`` block the GAM penalty machinery already consumes: the
    design is the one-hot indicator (intercept) or one-hot times a covariate
    (slope) over the factor levels, and the penalty is the **identity** (a
    ridge).  The single smoothing parameter the Fellner-Schall loop selects
    **is** the random-effect precision ``lambda = 1 / sigma_b^2`` -- the v1
    "penalised GLM == variance-components REML" identity, now reachable.  It is a
    third :data:`Smooth` variant alongside :class:`SplineBasis` /
    :class:`TensorBasis`, so it slots straight into ``gam_fit`` with **no new
    solver work** (a random effect is just one more penalty block).

    Unlike a spline, an RE has no continuous covariate to re-evaluate, so it
    carries only ``(design, penalty, levels)`` -- not the spline construction
    parameters.

    Attributes
    ----------
    design
        ``(n, q)`` random-effect design: ``one_hot(g)`` (intercept) or
        ``one_hot(g) * by`` (slope); ``q`` = number of factor levels.
    penalty
        ``(q, q)`` identity ridge penalty (``lambda I``).
    levels
        Number of factor levels ``q``.
    """

    design: Float[Array, 'n q']
    penalty: Float[Array, 'q q']
    levels: int

    @property
    def dim(self) -> int:
        """Number of random-effect coefficients ``q`` (the factor levels)."""
        return self.design.shape[-1]

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Array, ...], Tuple[int]]:
        return (self.design, self.penalty), (self.levels,)

    @classmethod
    def tree_unflatten(
        cls, aux: Tuple[int], children: Tuple[Any, ...]
    ) -> 'REBasis':
        design, penalty = children
        (levels,) = aux
        return cls(design=design, penalty=penalty, levels=levels)


def re_smooth(
    g: Int[Array, ' n'],
    *,
    by: Optional[Float[Array, ' n']] = None,
    n_levels: Optional[int] = None,
) -> REBasis:
    """Build a random-effect smooth block (mgcv ``bs="re"``).

    Parameters
    ----------
    g
        ``(n,)`` integer group labels in ``0 .. q-1`` (the factor levels) -- one
        per observation.
    by
        Optional ``(n,)`` covariate for a random **slope** ``(by | g)``; ``None``
        gives a random **intercept** ``(1 | g)``.  The design is then
        ``one_hot(g) * by[:, None]``.
    n_levels
        Number of levels ``q`` (defaults to ``int(g.max()) + 1``).  Pass it when
        some levels are absent from this sample so the block width is stable.

    Returns
    -------
    ``REBasis`` (a :data:`Smooth` variant) for ``gam_fit``.  The Fellner-Schall
    smoothing parameter on this block is the random-effect precision
    ``1 / sigma_b^2``; the fitted block coefficients are the BLUPs.
    """
    g = jnp.asarray(g)
    q = int(n_levels) if n_levels is not None else int(jnp.max(g)) + 1
    design = jax.nn.one_hot(g, q)  # (n, q)
    if by is not None:
        design = design * jnp.asarray(by, dtype=design.dtype)[:, None]
    penalty = jnp.eye(q, dtype=design.dtype)
    return REBasis(design=design, penalty=penalty, levels=q)


def by_factor_smooth(
    x: Float[Array, ' n'],
    by: Int[Array, ' n'],
    n_basis: int = 10,
    *,
    degree: int = 3,
    penalty_order: int = 2,
    bounds: Optional[Tuple[float, float]] = None,
    center: bool = True,
    n_levels: Optional[int] = None,
) -> Tuple[SplineBasis, ...]:
    """By-variable factor smooths ``s(x, by=f)`` -- one smooth of ``x`` per level.

    mgcv's ``s(x, by=f)`` for a **factor** ``f``: a *separate* penalised smooth
    of ``x`` for each level of ``f``, each with its **own** smoothing parameter.
    Returns a tuple of ``len == n_levels`` :class:`SplineBasis` blocks; splat it
    into ``gam_fit``'s ``smooths=`` (each block is an independent smooth, so the
    Fellner-Schall loop selects a per-level ``lambda``).

    Construction: a single marginal P-spline of ``x`` (shared knots / penalty /
    centering), then level ``l``'s design is that marginal design with the rows
    where ``f != l`` zeroed -- so level ``l``'s coefficients load only on its own
    observations.  Because the per-level designs have disjoint nonzero rows, the
    blocks are orthogonal in ``X^T W X`` (each level is fit on its own data).
    Pair with a parametric factor main effect to carry the per-level mean (the
    smooths are sum-to-zero when ``center=True``, as in mgcv).

    Parameters
    ----------
    x
        ``(n,)`` covariate.
    by
        ``(n,)`` integer factor labels in ``0 .. L-1``.
    n_basis, degree, penalty_order, bounds, center
        Marginal P-spline parameters (see :func:`bspline_basis`).
    n_levels
        Number of levels ``L`` (defaults to ``int(by.max()) + 1``); pass it when
        a level is absent from this sample so the returned tuple length is stable.

    Returns
    -------
    ``tuple`` of ``L`` :class:`SplineBasis`, one per factor level.
    """
    by = jnp.asarray(by)
    n_lev = int(n_levels) if n_levels is not None else int(jnp.max(by)) + 1
    base = bspline_basis(
        x,
        n_basis,
        degree=degree,
        penalty_order=penalty_order,
        bounds=bounds,
        center=center,
    )
    out = []
    for level in range(n_lev):
        mask = (by == level).astype(base.design.dtype)[:, None]  # (n, 1)
        out.append(replace(base, design=base.design * mask))
    return tuple(out)


def varying_coefficient_smooth(
    x: Float[Array, ' n'],
    by: Float[Array, ' n'],
    n_basis: int = 10,
    *,
    degree: int = 3,
    penalty_order: int = 2,
    bounds: Optional[Tuple[float, float]] = None,
    center: bool = True,
) -> SplineBasis:
    """Varying-coefficient smooth ``s(x, by=z)`` for a **continuous** ``z``.

    mgcv's ``s(x, by=z)`` with a numeric ``by``: the smooth term is
    ``z * f(x)`` -- a coefficient on ``z`` that varies smoothly with ``x``.  The
    design is the marginal P-spline of ``x`` scaled row-wise by ``z``; the
    penalty is unchanged.  A single :class:`SplineBasis` for ``gam_fit``.

    Parameters
    ----------
    x
        ``(n,)`` covariate that the coefficient varies over.
    by
        ``(n,)`` continuous covariate the smooth multiplies.
    n_basis, degree, penalty_order, bounds, center
        Marginal P-spline parameters (see :func:`bspline_basis`).
    """
    base = bspline_basis(
        x,
        n_basis,
        degree=degree,
        penalty_order=penalty_order,
        bounds=bounds,
        center=center,
    )
    z = jnp.asarray(by, dtype=base.design.dtype)[:, None]  # (n, 1)
    return replace(base, design=base.design * z)
