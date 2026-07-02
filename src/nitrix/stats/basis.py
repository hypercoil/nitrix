# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Penalised spline bases for additive models.

A GAM smooth term :math:`s(x)` is a spline basis expansion
:math:`f(x) = B(x) \\beta` with a roughness penalty
:math:`\\beta^{\\top} S \\beta`; the smoothing parameter trades the penalty
off against fit (chosen by REML in :func:`~nitrix.stats.gam_fit`).  This module
builds the ``(design, penalty)`` pair for a covariate vector and re-evaluates
the design at new covariate values (for plotting partial effects).

We ship the **P-spline** (Eilers & Marx): a uniform cubic B-spline design with
a discrete difference penalty -- ``mgcv``'s ``bs='ps'`` -- because it reuses
the uniform-B-spline machinery already validated for N4 and gives an exact
reference to test against.

Identifiability
---------------

A smooth competes with the model intercept unless it is constrained to sum to
zero over the data (:math:`\\sum_i f(x_i) = 0`, i.e.
:math:`(1^{\\top} B) \\beta = 0`).  We absorb that single constraint by
reparameterising :math:`\\beta = Z \\alpha` with :math:`Z` an orthonormal basis
of the constraint's null space, computed by a **single Householder reflection**
-- no ``qr`` (cuSOLVER) -- so the construction runs on the broken-cuSOLVER GPU
and stays differentiable.

Everything is value -> value and cuSOLVER-free.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Any, List, Optional, Protocol, Tuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from ..graph import laplacian
from ..linalg.kernel import spectral_density
from ..numerics._spline import difference_penalty_1d, uniform_bspline_weights
from ._hsgp import _hsgp_domain, _hsgp_features

__all__ = [
    'SmoothBasis',
    'SplineBasis',
    'TensorBasis',
    'REBasis',
    'bspline_basis',
    'cyclic_cubic_basis',
    'thinplate_regression_basis',
    'cr_basis',
    'gp_basis',
    'hsgp_basis',
    'hsgp_basis_nd',
    'gp_factor_smooth',
    'mrf_smooth',
    'tensor_product_basis',
    're_smooth',
    'by_factor_smooth',
    'varying_coefficient_smooth',
    'spline_design',
    'tensor_product_design',
]


class SmoothBasis(Protocol):
    """Structural interface of a GAM smooth term.

    A smooth carries a ``design`` matrix and a coefficient count ``dim``, knows
    its own penalty blocks, and can re-evaluate its design on a covariate grid.
    :func:`~nitrix.stats.gam_fit` dispatches through these members instead of an
    ``isinstance`` chain, so a **new** basis type (implementing this Protocol)
    needs no edit to the fitting routine.

    The concrete bases (:class:`SplineBasis` / :class:`TensorBasis` /
    :class:`REBasis`) all conform.
    """

    design: Float[Array, 'n k']

    @property
    def dim(self) -> int:
        """Number of basis coefficients ``k``."""
        ...

    def penalty_blocks(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return the penalty blocks of this smooth as host constants.

        Returns
        -------
        list of tuple of numpy.ndarray
            One ``(S_block, eig_block)`` pair per penalty: the roughness
            penalty matrix :math:`S` and its eigenvalues (with the unpenalised
            null space floored to zero).
        """
        ...

    def eval_design(self, x: Any) -> Float[Array, 'g k']:
        """Re-evaluate the design on a fresh covariate grid.

        Parameters
        ----------
        x
            The grid of covariate values at which to re-evaluate the smooth.
            The concrete shape and interpretation depend on the basis kind.

        Returns
        -------
        Float[Array, 'g k']
            The ``(g, k)`` design matrix at ``x``, with the same feature map
            and identifiability constraint as the original construction.
        """
        ...


def _bspline_design(
    x: Float[Array, ' n'],
    n_basis: int,
    degree: int,
    lo: float,
    hi: float,
) -> Float[Array, 'n n_basis']:
    """Banded uniform-B-spline design ``B[i, a]`` at arbitrary positions ``x``.

    ``x`` is mapped to the parametric coordinate :math:`s \\in [0, n_{spans}]`
    with :math:`n_{spans} = n_{basis} - degree`; each row has ``degree + 1``
    non-zeros at columns ``span .. span + degree``.

    Parameters
    ----------
    x
        ``(n,)`` positions at which to evaluate the basis.
    n_basis
        Number of B-spline basis functions (columns of the design).
    degree
        B-spline degree (e.g. ``3`` for cubic).
    lo, hi
        Lower and upper bounds of the knot range that ``x`` is mapped into.

    Returns
    -------
    Float[Array, 'n n_basis']
        The ``(n, n_basis)`` banded design matrix, each row summing to one
        (partition of unity), with constant boundary extrapolation outside
        ``[lo, hi]``.
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

    A single Householder reflection :math:`H` with :math:`H c \\propto e_1`;
    ``Z = H[:, 1:]`` then satisfies ``c @ Z = 0`` with orthonormal columns.
    Pure elementwise algebra -- no ``qr`` / cuSOLVER -- and differentiable.

    Parameters
    ----------
    c
        ``(k,)`` row vector whose null space is sought (e.g. the column sums
        of a design for the sum-to-zero constraint).

    Returns
    -------
    Float[Array, 'k k_minus_1']
        The ``(k, k-1)`` orthonormal null-space basis ``Z``.
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
        eigen-truncation :math:`U_k` (``None`` for B-splines).
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

    def penalty_blocks(self) -> list:
        """Return the single roughness-penalty block of this spline smooth.

        The block is the ``(k, k)`` roughness penalty and its eigenvalues, with
        the unpenalised null space floored to zero.  The eigenvalues drive only
        the basis-invariant trace :math:`\\operatorname{tr}(S_\\lambda^{+} S_k)`
        used to select the smoothing parameter.

        Returns
        -------
        list of tuple of numpy.ndarray
            A single ``(S, eig)`` pair: the penalty matrix and its floored
            eigenvalues.
        """
        s = np.asarray(self.penalty)
        w, _ = np.linalg.eigh(s)
        floor = 1e-10 * max(float(w.max()), float(np.finfo(w.dtype).tiny))
        return [(s, np.where(w > floor, w, 0.0))]

    def eval_design(self, x: Float[Array, ' g']) -> Float[Array, 'g k']:
        """Re-evaluate the spline design on a covariate grid.

        Parameters
        ----------
        x
            ``(g,)`` covariate values at which to re-evaluate the smooth.

        Returns
        -------
        Float[Array, 'g k']
            The ``(g, k)`` design matrix at ``x`` (post-constraint).
        """
        return spline_design(self, x)

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[Any, ...],
        Tuple[str, int, int, int, float, float, Optional[float]],
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
    SplineBasis
        The :class:`SplineBasis` holding the design, penalty, and
        re-evaluation parameters (``kind='bspline'``).
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
    """The (nonlinear, pre-constraint) basis features at ``x`` for the kind.

    Parameters
    ----------
    basis
        The :class:`SplineBasis` whose ``kind`` selects the feature map
        (B-spline, cyclic, thin-plate, cubic-regression, GP, HSGP, or MRF).
    x
        ``(m,)`` covariate values at which to evaluate the raw features.

    Returns
    -------
    Float[Array, 'm k0']
        The ``(m, k0)`` pre-constraint feature matrix, where ``k0`` is the
        unconstrained basis dimension.
    """
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
    if basis.kind == 'hsgp':
        u = cast(Array, basis.knots)  # sqrt-lambda frequencies (m,)
        rt = cast(Array, basis.radial_transform)  # (m, 2) = [sqrt-s, phase]
        L = cast(float, basis.kernel_param)
        phi = jnp.sqrt(1.0 / L) * jnp.sin(
            u[None, :] * x[:, None] + rt[None, :, 1]
        )
        return phi * rt[None, :, 0]
    if basis.kind == 'mrf':
        # ER6: re-evaluate in the stored design's dtype (not a hardcoded f32).
        return jax.nn.one_hot(
            x.astype(jnp.int32), basis.n_basis, dtype=basis.design.dtype
        )
    raise ValueError(f'unknown spline kind {basis.kind!r}.')


def spline_design(
    basis: SplineBasis, x: Float[Array, ' m']
) -> Float[Array, 'm k']:
    """Re-evaluate a :class:`SplineBasis` design at new covariate values.

    Used to render a smooth's partial effect on a fresh grid.  Applies the same
    feature map and sum-to-zero constraint as the original construction.

    Parameters
    ----------
    basis
        The :class:`SplineBasis` to re-evaluate.
    x
        ``(m,)`` covariate values.

    Returns
    -------
    Float[Array, 'm k']
        The ``(m, k)`` design matrix at ``x`` (post-constraint).
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
    """Thin-plate radial basis :math:`\\eta(|x_i - knot_j|)` (1-D, order 2).

    The 1-D order-2 thin-plate Green's function is :math:`|r|^3` (the constant
    is absorbed into the smoothing parameter).

    Parameters
    ----------
    x
        ``(m,)`` evaluation positions.
    knots
        ``(nk,)`` knot positions.

    Returns
    -------
    Float[Array, 'm nk']
        The ``(m, nk)`` radial-basis matrix :math:`|x_i - knot_j|^3`.
    """
    r = jnp.abs(x[:, None] - knots[None, :])
    return r**3


def _tps_poly(
    x: Float[Array, ' m'], order: int, lo: float, hi: float
) -> Float[Array, 'm order']:
    """Polynomial null-space basis of the thin-plate penalty.

    Builds the unpenalised polynomial columns
    :math:`[1, t, \\ldots, t^{order-1}]` on a rescaled coordinate
    :math:`t \\in [0, 1]`.

    Parameters
    ----------
    x
        ``(m,)`` evaluation positions.
    order
        Number of polynomial columns (the thin-plate order :math:`M`); columns
        run from the constant term up to :math:`t^{order-1}`.
    lo, hi
        Bounds used to rescale ``x`` into :math:`t \\in [0, 1]`.

    Returns
    -------
    Float[Array, 'm order']
        The ``(m, order)`` polynomial design.
    """
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
    unpenalised polynomial null-space terms.  The truncation ``eigh`` is a
    one-off host (CPU/numpy) computation -- it is data-dependent but not
    per-trace, and avoids the cuSOLVER ``syevd`` on the broken GPU.

    References
    ----------
    Wood, S. N. (2003). Thin plate regression splines. *Journal of the Royal
    Statistical Society: Series B (Statistical Methodology)*, 65(1), 95-114.
    https://doi.org/10.1111/1467-9868.00374

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
        exceeds it), bounding the :math:`O(n_{knots}^3)` eigendecomposition.
    bounds
        ``(lo, hi)`` for the polynomial rescaling (defaults to the data range).
    center
        Apply the sum-to-zero identifiability constraint (default ``True``).

    Returns
    -------
    SplineBasis
        The :class:`SplineBasis` for the thin-plate smooth (``kind='tprs'``).
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
    """Uniform *cyclic* B-spline design matrix at positions ``x``.

    Uses the same B-spline weights as :func:`_bspline_design` but takes the
    parametric coordinate modulo the period and wraps the column indices, so
    the basis is periodic (``f(lo) == f(hi)``).

    Parameters
    ----------
    x
        ``(n,)`` positions at which to evaluate the basis.
    n_basis
        Number of cyclic basis functions (columns).
    degree
        B-spline degree.
    lo, hi
        Bounds of the period; ``hi`` is identified with ``lo``.

    Returns
    -------
    Float[Array, 'n n_basis']
        The ``(n, n_basis)`` periodic design matrix.
    """
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
    """Circular ``order``-th difference penalty :math:`D^{\\top} D`.

    The difference operator ``D`` wraps around the ends so the penalty is
    periodic.

    Parameters
    ----------
    n_basis
        Number of cyclic basis functions (the penalty is
        ``(n_basis, n_basis)``).
    order
        Difference order (e.g. ``2`` to penalise curvature).
    dtype
        Output dtype of the returned penalty.

    Returns
    -------
    Float[Array, 'n_basis n_basis']
        The ``(n_basis, n_basis)`` circular difference penalty.
    """
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
    SplineBasis
        The :class:`SplineBasis` for the cyclic smooth (``kind='cyclic'``).
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
    """Build the mgcv ``cr`` construction matrices.

    Produces the coefficient-to-second-derivative map ``F`` and the
    integrated-curvature penalty :math:`S = D^{\\top} B^{-1} D` for a natural
    cubic spline parameterised by its values :math:`\\beta` at the knots.  With
    :math:`h_i` the knot spacings, ``B`` (tridiagonal) and ``D`` map the knot
    values to the interior second derivatives :math:`m = B^{-1} D \\beta`
    (:math:`f''` is zero at the ends -- the *natural* boundary); ``F`` pads
    :math:`m` with those zeros, and ``S`` is the exact
    :math:`\\int f''(x)^2 \\, dx`.  Computed host-side (the knots are static
    across the mass axis).

    Parameters
    ----------
    knots
        ``(k,)`` strictly increasing knot positions.

    Returns
    -------
    F : numpy.ndarray
        The ``(k, k)`` map from knot values to interior second derivatives
        (padded with the natural-boundary zeros).
    S : numpy.ndarray
        The symmetric ``(k, k)`` integrated-curvature penalty.
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
    """Cardinal cubic-spline design at ``x`` (the knot-value interpolation).

    Parameters
    ----------
    x
        ``(m,)`` evaluation positions.
    knots
        ``(k,)`` knot positions.
    fmap
        ``(k, k)`` coefficient-to-second-derivative map ``F`` from
        :func:`_cr_construct`.

    Returns
    -------
    Float[Array, 'm k']
        The ``(m, k)`` design matrix interpolating the knot values.
    """
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
    """Nudge a sorted vector to be strictly increasing (de-duplicate knots).

    Parameters
    ----------
    v
        A sorted (non-decreasing) 1-D array, e.g. quantile knot positions.

    Returns
    -------
    numpy.ndarray
        A strictly increasing copy of ``v`` (as ``float64``), each repeated
        entry bumped by a tiny fraction of the vector's span.
    """
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
    SplineBasis
        The :class:`SplineBasis` for the cubic-regression smooth
        (``kind='cr'``).
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
    """Matern-3/2 correlation
    :math:`(1 + \\sqrt{3}\\, r / \\rho) \\exp(-\\sqrt{3}\\, r / \\rho)`.

    Parameters
    ----------
    r
        Non-negative distances (any shape).
    rho
        Correlation length (kernel range).

    Returns
    -------
    Float[Array, '...']
        The Matern-3/2 correlation at each distance, matching the shape of
        ``r``.
    """
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
    ``rho``: the smooth is :math:`f(x) = \\sum_j C(|x - z_j|) \\delta_j` over
    knots :math:`z` with the RKHS penalty
    :math:`\\delta^{\\top} C_{zz} \\delta` (:math:`C_{zz}` the kernel Gram at
    the knots).  Constructed like the thin-plate basis -- the radial features
    are eigen-reparameterised so the penalty is diagonal and truncated to the
    ``n_basis`` leading components (a one-off host eigendecomposition,
    cuSOLVER-free on the broken GPU).

    Parameters
    ----------
    x
        ``(n,)`` covariate values.
    n_basis
        Retained basis dimension ``k`` (the leading kriging components).
    rho
        Kernel range (correlation length).  Defaults to the data range / 2.
    max_knots
        Cap on the number of knots (quantile-placed) bounding the
        eigendecomposition cost.
    bounds
        ``(lo, hi)`` recorded for re-evaluation (defaults to the data range).
    center
        Apply the sum-to-zero identifiability constraint (default ``True``).

    Returns
    -------
    SplineBasis
        The :class:`SplineBasis` for the kriging GP smooth (``kind='gp'``).
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
    # f(x) = C_xz U_k diag(1/lambda) alpha.  The kriging RKHS penalty on the
    # knot weights delta is delta^T C_zz delta; with delta = U_k diag(1/lambda)
    # alpha and C_zz U_k = U_k diag(lambda) this is alpha^T diag(1/lambda) alpha
    # -- NOT the identity.  Using diag(1/lambda) makes the implied prior
    # covariance D diag(lambda) D^T = C_xz U_k diag(1/lambda) U_k^T C_zx reproduce
    # the (Nystrom-truncated) Matern Gram C_xx, so the lambda<->sigma_f variance-
    # component identity holds (an identity penalty is off by ~91%).
    radial_transform = jnp.asarray(uk / evals_k[None, :], dtype=x.dtype)
    knots = jnp.asarray(knots_np, dtype=x.dtype)

    def _gp_raw(xx: Float[Array, ' m']) -> Float[Array, 'm k']:
        cxz = _matern32_kernel(jnp.abs(xx[:, None] - knots[None, :]), rho_v)
        return cxz @ radial_transform

    design = _gp_raw(x)
    penalty = jnp.diag(jnp.asarray(1.0 / evals_k, dtype=x.dtype))

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
# Hilbert-space approximate GP smooth (HSGP; Solin-Sarkka 2020,
# Riutort-Mayol/Burkner 2023) -- the primary GP smoother
# ---------------------------------------------------------------------------

# Below this, the top eigen-frequency sqrt(lambda_m) * rho is too small for the
# reduced-rank basis to resolve a kernel of lengthscale rho -- the (m, L, rho)
# coupling of Riutort-Mayol et al. (2023): a short rho needs a larger rank (or a
# larger boundary).  A warning, not an error.
_HSGP_NYQUIST_MIN = 2.4


def _check_hsgp_resolution(
    sqrt_lambda_top: float, rho: Optional[float], n_basis: int, where: str
) -> None:
    """Warn when the rank under-resolves the kernel for a *given* ``rho``.

    Parameters
    ----------
    sqrt_lambda_top
        The largest eigen-frequency :math:`\\sqrt{\\lambda_{top}}` of the
        reduced-rank basis.
    rho
        Kernel lengthscale; ``None`` or non-positive disables the check.
    n_basis
        Reduced rank (number of retained eigenfunctions), used in the warning
        message.
    where
        Caller label prepended to the warning message.
    """
    if rho is None or not rho > 0:
        return
    reach = float(sqrt_lambda_top) * float(rho)
    if reach < _HSGP_NYQUIST_MIN:
        warnings.warn(
            f'{where}: with n_basis={n_basis} the top eigen-frequency reaches '
            f'only {reach:.2f}/rho (rho={float(rho):.3g}); the basis may '
            'under-resolve the kernel. Increase n_basis or `boundary` for a '
            'short lengthscale (Riutort-Mayol et al. 2023).',
            stacklevel=3,
        )


def hsgp_basis(
    x: Float[Array, ' n'],
    n_basis: int = 20,
    *,
    kernel: str = 'matern52',
    rho: Optional[float] = None,
    amplitude: float = 1.0,
    boundary: float = 1.5,
    bounds: Optional[Tuple[float, float]] = None,
    center: bool = True,
) -> SplineBasis:
    """Build a Hilbert-space approximate GP smooth (HSGP) for covariate ``x``.

    A reduced-rank stationary GP on the bounded domain :math:`[c - L, c + L]`
    (:math:`c` the data midrange, :math:`L = boundary \\cdot \\text{half-range}`),
    expanded in the Dirichlet-Laplacian eigenfunctions
    :math:`\\phi_j(x) = \\sqrt{1/L} \\sin(\\sqrt{\\lambda_j}\\,(x - c + L))` with
    :math:`\\sqrt{\\lambda_j} = j \\pi / (2 L)`.  The kernel enters **only**
    through the spectral density weights
    :math:`s_j = S_\\theta(\\sqrt{\\lambda_j})`
    (:func:`~nitrix.linalg.spectral_density`): the whitened design is
    :math:`\\Psi = [\\sqrt{s_j}\\, \\phi_j(x)]` with an **identity penalty**, so
    the basis is a drop-in smooth for :func:`~nitrix.stats.gam_fit` and the
    Fellner-Schall smoothing parameter :math:`\\lambda = 1 / \\sigma_f^2` is the
    GP amplitude.

    Unlike the kriging :func:`gp_basis`, the eigenfunctions are independent of
    the kernel hyperparameters, so the lengthscale ``rho`` enters as a diagonal
    reweighting of a fixed design -- the basis for eigendecomposition-free
    lengthscale estimation (the standalone :func:`~nitrix.stats.gp_fit`; not
    this constructor, which fixes ``rho``).

    Parameters
    ----------
    x
        ``(n,)`` covariate values.
    n_basis
        Number of retained eigenfunctions ``m`` (the reduced rank).
    kernel
        Stationary kernel: ``'matern12'`` / ``'matern32'`` / ``'matern52'`` /
        ``'rbf'`` (squared-exponential).  See
        :func:`~nitrix.linalg.spectral_density`.
    rho
        Kernel lengthscale.  Defaults to the data half-range (a sane fixed
        value; lengthscale *estimation* is the standalone
        :func:`~nitrix.stats.gp_fit`).
    amplitude
        Kernel amplitude folded into the weights; keep ``1.0`` when fitting via
        :func:`~nitrix.stats.gam_fit` (the smoothing parameter carries the
        amplitude).
    boundary
        Domain-extension factor ``L / half-range`` (``>= 1``; default ``1.5``).
        Larger values reduce boundary bias; small ``rho`` needs larger
        ``n_basis``.
    bounds
        ``(lo, hi)`` data range override (defaults to the data min/max).
    center
        Apply the sum-to-zero identifiability constraint (default ``True``).

    Returns
    -------
    SplineBasis
        The :class:`SplineBasis` for the HSGP smooth (``kind='hsgp'``).
    """
    if not 1 <= n_basis:
        raise ValueError(f'hsgp_basis: n_basis={n_basis} must be >= 1.')
    if center and n_basis < 2:
        raise ValueError(
            f'hsgp_basis: n_basis={n_basis} with center=True leaves an empty '
            '(n, 0) design -- the sum-to-zero constraint drops one column. Use '
            'n_basis >= 2, or center=False.'
        )
    if not boundary >= 1.0:
        raise ValueError(f'hsgp_basis: boundary={boundary} must be >= 1.0.')
    x = jnp.asarray(x)
    x_np = np.asarray(x, dtype=np.float64)
    lo = float(np.min(x_np)) if bounds is None else float(bounds[0])
    hi = float(np.max(x_np)) if bounds is None else float(bounds[1])
    c, L = _hsgp_domain(lo, hi, boundary)  # DS3: shared HSGP domain
    rho_v = float(rho) if rho is not None else max(0.5 * (hi - lo), 1e-6)

    # Laplace eigen-frequencies, spectral weights, and the per-mode phase that
    # folds the centring constant c into the stored re-evaluation (so c need not
    # be a SplineBasis field): sin(sqrt(lam_j)(x - c + L)) = sin(sqrt(lam_j) x +
    # phase_j), phase_j = sqrt(lam_j)(L - c).
    j = np.arange(1, n_basis + 1, dtype=np.float64)
    sqrt_lambda = j * np.pi / (2.0 * L)  # (m,)
    _check_hsgp_resolution(float(sqrt_lambda[-1]), rho, n_basis, 'hsgp_basis')
    s = np.asarray(
        spectral_density(
            jnp.asarray(sqrt_lambda),
            kernel=kernel,
            rho=rho_v,
            amplitude=amplitude,
        ),
        dtype=np.float64,
    )
    sqrt_s = np.sqrt(np.clip(s, 1e-30, None))
    phase = sqrt_lambda * (L - c)

    knots = jnp.asarray(sqrt_lambda, dtype=x.dtype)
    radial_transform = jnp.asarray(
        np.stack([sqrt_s, phase], axis=1), dtype=x.dtype
    )  # (m, 2) = [sqrt-s, phase]
    inv_sqrt_L = float(np.sqrt(1.0 / L))

    def _hsgp_raw(xx: Float[Array, ' m']) -> Float[Array, 'm k']:
        # DS3: shared eigenfunction design, then the sqrt-spectral amplitude.
        phi = _hsgp_features(xx, knots, radial_transform[:, 1], inv_sqrt_L)
        return phi * radial_transform[None, :, 0]

    design = _hsgp_raw(x)
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
        kind='hsgp',
        constraint=constraint,
        n_basis=n_basis,
        degree=0,
        penalty_order=2,
        lo=lo,
        hi=hi,
        knots=knots,
        radial_transform=radial_transform,
        kernel_param=L,
    )


# ---------------------------------------------------------------------------
# Factor-smooth GP interaction (mgcv bs='fs' with a GP marginal) -- the
# fixed-rho hierarchical-GP building block
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _FactorGPBasis:
    """A factor-smooth GP interaction sharing one smoothing parameter.

    A separate GP curve per factor level, all sharing **one** smoothing
    parameter (mgcv ``bs="fs"`` with a GP marginal).  The whitened HSGP design
    :math:`\\Psi` (at a fixed ``rho``) is replicated per group -- level
    :math:`g`'s columns are :math:`\\Psi` on its own rows, zero elsewhere --
    and the penalty is the **identity**, so the single Fellner-Schall smoothing
    parameter on this block **is** the shared group precision
    :math:`1 / \\sigma_{grp}^2` (the penalty-versus-variance-component identity,
    exactly as :class:`REBasis` but with the GP basis in place of the one-hot).
    Pair it with a population :func:`hsgp_basis` of the same ``rho`` (plus the
    intercept) in :func:`~nitrix.stats.gam_fit` for the "GS" hierarchical GP at
    fixed ``rho`` -- the basis counterpart of :func:`~nitrix.stats.hgp_fit`
    (which additionally estimates ``rho``).

    Attributes
    ----------
    design
        ``(n, L*m)`` factor-smooth design.
    penalty
        ``(L*m, L*m)`` identity ridge (the whitened-GP prior is i.i.d.).
    base
        The whitened HSGP marginal :class:`SplineBasis` (for re-evaluation).
    n_levels
        Number of factor levels ``L``.
    """

    design: Float[Array, 'n Lm']
    penalty: Float[Array, 'Lm Lm']
    base: 'SplineBasis'
    n_levels: int

    @property
    def dim(self) -> int:
        """Number of factor-smooth coefficients ``L*m``."""
        return self.design.shape[-1]

    def penalty_blocks(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return the single identity-ridge penalty block.

        The block is the identity ridge (full rank, all eigenvalues one), so a
        single shared smoothing parameter is applied across all groups.

        Returns
        -------
        list of tuple of numpy.ndarray
            A single ``(I, 1)`` pair: the identity penalty and its all-ones
            eigenvalues.
        """
        return [(np.asarray(self.penalty), np.ones(self.dim))]

    def eval_design(self, x: Any) -> Float[Array, 'g Lm']:
        """Re-evaluate the factor-smooth design at new points.

        The whitened GP design at ``x_vals`` is placed in each observation's
        group block, leaving the other blocks zero.

        Parameters
        ----------
        x
            A ``(x_vals, group)`` tuple (the by-factor convention): ``x_vals``
            the ``(g,)`` covariate grid and ``group`` the ``(g,)`` integer
            level labels.

        Returns
        -------
        Float[Array, 'g Lm']
            The ``(g, L*m)`` factor-smooth design at the requested points.
        """
        x_vals, group = x
        psi = spline_design(self.base, jnp.asarray(x_vals))  # (g, m)
        group = jnp.asarray(group).astype(jnp.int32)
        onehot = jax.nn.one_hot(group, self.n_levels, dtype=psi.dtype)
        inter = onehot[:, :, None] * psi[:, None, :]  # (g, L, m)
        return inter.reshape(psi.shape[0], self.n_levels * self.base.dim)

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Any, ...], Tuple[int]]:
        return (self.design, self.penalty, self.base), (self.n_levels,)

    @classmethod
    def tree_unflatten(
        cls, aux: Tuple[int], children: Tuple[Any, ...]
    ) -> '_FactorGPBasis':
        design, penalty, base = children
        (n_levels,) = aux
        return cls(
            design=design, penalty=penalty, base=base, n_levels=n_levels
        )


def gp_factor_smooth(
    x: Float[Array, ' n'],
    group: Int[Array, ' n'],
    n_basis: int = 10,
    *,
    kernel: str = 'matern52',
    rho: Optional[float] = None,
    amplitude: float = 1.0,
    boundary: float = 1.5,
    bounds: Optional[Tuple[float, float]] = None,
    center: bool = False,
    n_levels: Optional[int] = None,
) -> _FactorGPBasis:
    """Build a factor-smooth GP interaction (mgcv ``bs="fs"`` with a GP marginal).

    A separate Hilbert-space GP curve of ``x`` for each level of ``group``, all
    sharing **one** smoothing parameter (so the per-group curves are random
    deviations with a common amplitude -- partial pooling).  This is the
    **fixed-``rho``** building block of the hierarchical GP: drop it into
    :func:`~nitrix.stats.gam_fit` alongside a population :func:`hsgp_basis` of
    the same ``rho`` (and the intercept) for the "GS" hierarchical GP.  For
    *estimated* ``rho`` and the full variance-component report, use
    :func:`~nitrix.stats.hgp_fit`.

    Parameters
    ----------
    x
        ``(n,)`` covariate.
    group
        ``(n,)`` integer factor labels ``0 .. L-1``.
    n_basis
        Per-group HSGP rank ``m`` (default ``10``; the design is ``L`` times this
        wide, so a smaller rank than :func:`hsgp_basis` is usual).
    kernel, rho, amplitude, boundary, bounds
        HSGP marginal parameters (see :func:`hsgp_basis`); ``rho`` defaults to the
        data half-range (a fixed value -- lengthscale estimation is
        :func:`~nitrix.stats.hgp_fit`).
    center
        Sum-to-zero each group's smooth (default ``False`` -- the group curves
        carry their own level, the usual factor-smooth; set ``True`` to pair with
        an explicit per-group mean).
    n_levels
        Number of levels ``L`` (defaults to ``int(group.max()) + 1``); pass it when
        a level is absent so the block width is stable.

    Returns
    -------
    _FactorGPBasis
        A factor-smooth GP basis (a :class:`SmoothBasis`) for
        :func:`~nitrix.stats.gam_fit`; the single Fellner-Schall smoothing
        parameter on this block is the shared group precision
        :math:`1 / \\sigma_{grp}^2`.
    """
    base = hsgp_basis(
        x,
        n_basis,
        kernel=kernel,
        rho=rho,
        amplitude=amplitude,
        boundary=boundary,
        bounds=bounds,
        center=center,
    )
    group = jnp.asarray(group)
    L = int(n_levels) if n_levels is not None else int(jnp.max(group)) + 1
    m = base.dim
    onehot = jax.nn.one_hot(group, L, dtype=base.design.dtype)  # (n, L)
    inter = onehot[:, :, None] * base.design[:, None, :]  # (n, L, m)
    design = inter.reshape(base.design.shape[0], L * m)
    penalty = np.eye(L * m, dtype=base.design.dtype)
    return _FactorGPBasis(
        design=design,
        penalty=penalty,  # type: ignore[arg-type]
        base=base,
        n_levels=L,
    )


# ---------------------------------------------------------------------------
# Multi-dimensional Hilbert-space GP smooth (tensor-product HSGP) -- a spatial /
# multi-covariate GP (Riutort-Mayol/Burkner 2023 sec.3)
# ---------------------------------------------------------------------------


def _hsgp_nd_raw(
    X: Float[Array, 'g D'],
    freqs: Float[Array, 'M D'],
    phase: Float[Array, 'M D'],
    inv_sqrt_L: Float[Array, ' D'],
    sqrt_s: Float[Array, ' M'],
) -> Float[Array, 'g M']:
    """Whitened tensor-product eigenfunction design (pre-constraint).

    For mode :math:`m` and point :math:`x`, the design entry is
    :math:`\\sqrt{s_m} \\prod_d \\sqrt{1/L_d}
    \\sin(w_{m,d} x_d + phase_{m,d})`.

    Parameters
    ----------
    X
        ``(g, D)`` evaluation points.
    freqs
        ``(M, D)`` per-mode per-dimension eigen-frequencies.
    phase
        ``(M, D)`` per-mode per-dimension phases (folding in the domain centre).
    inv_sqrt_L
        ``(D,)`` per-dimension :math:`\\sqrt{1/L_d}` amplitudes.
    sqrt_s
        ``(M,)`` per-mode spectral-weight square roots.

    Returns
    -------
    Float[Array, 'g M']
        The ``(g, M)`` whitened tensor-product design.
    """
    X = jnp.asarray(X)
    arg = freqs[None, :, :] * X[:, None, :] + phase[None, :, :]  # (g, M, D)
    terms = inv_sqrt_L[None, None, :] * jnp.sin(arg)
    phi = jnp.prod(terms, axis=2)  # (g, M)
    return phi * sqrt_s[None, :]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _HSGPndBasis:
    """A multi-dimensional Hilbert-space approximate GP smooth (tensor-product).

    The ``D``-dimensional analogue of :func:`hsgp_basis`: a reduced-rank GP on a
    box domain, expanded in the **tensor-product** Laplace-Dirichlet
    eigenfunctions
    :math:`\\phi_{j_1 \\ldots j_D}(x) = \\prod_d \\phi_{j_d}(x_d)` (eigenvalue
    :math:`\\sum_d \\lambda_{j_d}`), with the kernel entering only through the
    spectral-density weights of the mode frequency magnitude.  Whitened design,
    identity penalty -- a drop-in :func:`~nitrix.stats.gam_fit` smooth (the
    Fellner-Schall parameter is the GP amplitude).

    Attributes
    ----------
    design
        ``(n, M)`` whitened design (:math:`M = \\prod_d m_d` modes,
        post-constraint).
    penalty
        ``(M, M)`` identity penalty (post-constraint).
    constraint
        Sum-to-zero reparameterisation ``Z`` (``None`` if unconstrained).
    freqs, phase
        ``(M0, D)`` per-mode per-dimension eigen-frequency and phase (pre-
        constraint ``M0``).
    inv_sqrt_L
        ``(D,)`` per-dimension :math:`\\sqrt{1/L_d}` amplitude.
    sqrt_s
        ``(M0,)`` per-mode spectral-weight square roots.
    n_dim
        Input dimension ``D``.
    """

    design: Float[Array, 'n M']
    penalty: Float[Array, 'M M']
    constraint: Optional[Float[Array, 'M0 M']]
    freqs: Float[Array, 'M0 D']
    phase: Float[Array, 'M0 D']
    inv_sqrt_L: Float[Array, ' D']
    sqrt_s: Float[Array, ' M0']
    n_dim: int

    @property
    def dim(self) -> int:
        """Number of (post-constraint) basis coefficients ``M``."""
        return self.design.shape[-1]

    def penalty_blocks(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return the single identity-ridge penalty block.

        The block is the identity ridge (one smoothing parameter across all
        modes) together with its eigenvalues, the null space floored to zero.

        Returns
        -------
        list of tuple of numpy.ndarray
            A single ``(S, eig)`` pair: the penalty matrix and its floored
            eigenvalues.
        """
        s = np.asarray(self.penalty)
        w, _ = np.linalg.eigh(s)
        floor = 1e-10 * max(float(w.max()), float(np.finfo(w.dtype).tiny))
        return [(s, np.where(w > floor, w, 0.0))]

    def eval_design(self, x: Float[Array, 'g D']) -> Float[Array, 'g M']:
        """Re-evaluate the tensor-product design at new points.

        Parameters
        ----------
        x
            ``(g, D)`` evaluation points.

        Returns
        -------
        Float[Array, 'g M']
            The ``(g, M)`` design matrix at ``x`` (post-constraint).
        """
        raw = _hsgp_nd_raw(
            jnp.asarray(x),
            self.freqs,
            self.phase,
            self.inv_sqrt_L,
            self.sqrt_s,
        )
        if self.constraint is not None:
            raw = raw @ self.constraint
        return raw

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Any, ...], Tuple[int]]:
        return (
            (
                self.design,
                self.penalty,
                self.constraint,
                self.freqs,
                self.phase,
                self.inv_sqrt_L,
                self.sqrt_s,
            ),
            (self.n_dim,),
        )

    @classmethod
    def tree_unflatten(
        cls, aux: Tuple[int], children: Tuple[Any, ...]
    ) -> '_HSGPndBasis':
        (design, penalty, constraint, freqs, phase, inv_sqrt_L, sqrt_s) = (
            children
        )
        (n_dim,) = aux
        return cls(
            design=design,
            penalty=penalty,
            constraint=constraint,
            freqs=freqs,
            phase=phase,
            inv_sqrt_L=inv_sqrt_L,
            sqrt_s=sqrt_s,
            n_dim=n_dim,
        )


def hsgp_basis_nd(
    X: Float[Array, 'n D'],
    n_basis: Any = 8,
    *,
    kernel: str = 'matern52',
    rho: Any = None,
    amplitude: float = 1.0,
    boundary: float = 1.5,
    center: bool = True,
) -> _HSGPndBasis:
    """Build a multi-dimensional Hilbert-space GP smooth (tensor-product HSGP).

    The ``D``-D GP smoother (a spatial smooth, or a smooth interaction of ``D``
    continuous covariates): the reduced-rank GP on a box domain expanded in the
    tensor-product Laplace eigenfunctions.  Either **isotropic** (one lengthscale,
    a radial kernel of ``||x - x'||``) or **separable / ARD** (a per-dimension
    lengthscale, a product kernel) -- selected by ``rho``.

    Parameters
    ----------
    X
        ``(n, D)`` covariates (``D >= 1`` columns).
    n_basis
        Per-dimension rank: an int (the same ``m`` for every dimension) or a
        length-``D`` sequence :math:`[m_1, \\ldots, m_D]`.  The total rank is
        the product :math:`M = \\prod_d m_d` (it grows fast with ``D`` -- keep
        ``m`` modest for ``D >= 2``).
    kernel
        Stationary kernel: ``'matern52'`` (default) / ``'matern32'`` /
        ``'matern12'`` / ``'rbf'``.
    rho
        Lengthscale.  A scalar gives an **isotropic** GP (the ``D``-dimensional
        spectral density of ``||w||``); a length-``D`` sequence gives a
        **separable / ARD** GP (the product of per-dimension 1-D densities).
        ``None`` (default) is isotropic with ``rho`` the mean per-dimension
        half-range.
    amplitude
        Kernel amplitude folded into the weights; keep ``1.0`` for
        :func:`~nitrix.stats.gam_fit`.
    boundary
        Domain-extension factor :math:`L_d / \\text{half-range}_d` (``>= 1``;
        default ``1.5``).
    center
        Apply the sum-to-zero identifiability constraint (default ``True``).

    Returns
    -------
    _HSGPndBasis
        A multi-dimensional HSGP basis (a :class:`SmoothBasis`) for
        :func:`~nitrix.stats.gam_fit`; its ``eval_design`` takes new
        ``(g, D)`` points.
    """
    X = jnp.asarray(X)
    if X.ndim != 2:
        raise ValueError(
            f'hsgp_basis_nd: X must be (n, D); got shape {X.shape}.'
        )
    n, d_in = X.shape
    if not boundary >= 1.0:
        raise ValueError(f'hsgp_basis_nd: boundary={boundary} must be >= 1.0.')
    x_np = np.asarray(X, dtype=np.float64)

    if isinstance(n_basis, (int, np.integer)):
        m_per = [int(n_basis)] * d_in
    else:
        m_per = [int(v) for v in n_basis]
        if len(m_per) != d_in:
            raise ValueError(
                f'hsgp_basis_nd: n_basis has {len(m_per)} entries; expected '
                f'D={d_in}.'
            )
    if any(mm < 1 for mm in m_per):
        raise ValueError(
            'hsgp_basis_nd: every per-dimension rank must be >= 1.'
        )

    lo = x_np.min(axis=0)
    hi = x_np.max(axis=0)
    c_mid = 0.5 * (lo + hi)
    big_l = float(boundary) * np.maximum(0.5 * (hi - lo), 1e-6)  # (D,)

    # Per-dimension eigen-frequencies, then the tensor (cartesian) mode grid.
    sqrt_lams = [
        np.arange(1, m_per[d] + 1, dtype=np.float64) * np.pi / (2.0 * big_l[d])
        for d in range(d_in)
    ]
    grids = np.meshgrid(*sqrt_lams, indexing='ij')
    freqs_np = np.stack([g.ravel() for g in grids], axis=1)  # (M, D)
    phase_np = freqs_np * (big_l - c_mid)[None, :]
    inv_sqrt_L = np.sqrt(1.0 / big_l)  # (D,)

    # Per-axis (m, L, rho) resolution check (when a lengthscale is given).
    if rho is not None:
        rho_per = (
            [float(v) for v in rho]
            if not isinstance(rho, (int, float, np.floating, np.integer))
            else [float(rho)] * d_in
        )
        if len(rho_per) == d_in:
            for d in range(d_in):
                _check_hsgp_resolution(
                    float(sqrt_lams[d][-1]),
                    rho_per[d],
                    m_per[d],
                    f'hsgp_basis_nd[axis {d}]',
                )

    # Spectral weights: isotropic (||w||, D-dim density) or separable (product of
    # per-dimension 1-D densities).
    if rho is None:
        rho_iso = float(np.mean(0.5 * (hi - lo)))
        s = np.asarray(
            spectral_density(
                jnp.asarray(np.sqrt((freqs_np**2).sum(axis=1))),
                kernel=kernel,
                rho=rho_iso,
                amplitude=amplitude,
                dim=d_in,
            )
        )
    elif isinstance(rho, (int, float, np.floating, np.integer)):
        s = np.asarray(
            spectral_density(
                jnp.asarray(np.sqrt((freqs_np**2).sum(axis=1))),
                kernel=kernel,
                rho=float(rho),
                amplitude=amplitude,
                dim=d_in,
            )
        )
    else:
        rho_seq = [float(v) for v in rho]
        if len(rho_seq) != d_in:
            raise ValueError(
                f'hsgp_basis_nd: rho has {len(rho_seq)} entries; expected '
                f'D={d_in}.'
            )
        s = np.ones(freqs_np.shape[0], dtype=np.float64)
        for d in range(d_in):
            s = s * np.asarray(
                spectral_density(
                    jnp.asarray(freqs_np[:, d]),
                    kernel=kernel,
                    rho=rho_seq[d],
                    amplitude=amplitude,
                    dim=1,
                )
            )
    sqrt_s = np.sqrt(np.clip(s, 1e-30, None))

    freqs = jnp.asarray(freqs_np, dtype=X.dtype)
    phase = jnp.asarray(phase_np, dtype=X.dtype)
    inv_sqrt_L_j = jnp.asarray(inv_sqrt_L, dtype=X.dtype)
    sqrt_s_j = jnp.asarray(sqrt_s, dtype=X.dtype)

    design = _hsgp_nd_raw(X, freqs, phase, inv_sqrt_L_j, sqrt_s_j)
    m_total = design.shape[1]
    penalty = jnp.eye(m_total, dtype=X.dtype)

    constraint: Optional[Array] = None
    if center:
        col_sums = jnp.sum(design, axis=0)
        constraint = _householder_null(col_sums)
        design = design @ constraint
        penalty = constraint.T @ penalty @ constraint

    return _HSGPndBasis(
        design=design,
        penalty=penalty,
        constraint=constraint,
        freqs=freqs,
        phase=phase,
        inv_sqrt_L=inv_sqrt_L_j,
        sqrt_s=sqrt_s_j,
        n_dim=int(d_in),
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
    adjacency.

    The natural smoother on a parcel / vertex graph.  Each observation carries
    an integer region ``label`` in ``0 .. R-1``; the smooth is a per-region
    effect (design = region indicator) penalised by the **combinatorial graph
    Laplacian** :math:`L = D - A` of the region adjacency
    (:math:`\\beta^{\\top} L \\beta = \\sum_{i \\sim j} (\\beta_i - \\beta_j)^2`
    -- neighbouring regions are shrunk together).  The penalty is built with
    :func:`~nitrix.graph.laplacian` -- a direct substrate reuse.

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
    SplineBasis
        The :class:`SplineBasis` for the MRF smooth (``kind='mrf'``, with
        ``n_basis = R``).
    """
    labels = jnp.asarray(labels)
    a = jnp.asarray(neighbours)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(
            f'mrf_smooth: neighbours must be a square (R, R) adjacency; got '
            f'{a.shape}.'
        )
    r = a.shape[0]
    # ER6: promote an integer adjacency to the canonical float (f64 under x64),
    # not result_type(int, float32) which is float32 under JAX's promotion and
    # would leak float32 into the mrf design / penalty under the x64 invariant.
    af = a if jnp.issubdtype(a.dtype, jnp.floating) else a.astype(float)
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

    Parameters
    ----------
    designs
        A tuple of marginal designs, each ``(n, k_j)`` and sharing the row
        count ``n``.

    Returns
    -------
    numpy.ndarray
        The ``(n, k1*k2*...)`` row-wise Kronecker product.
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
    """An anisotropic tensor-product interaction smooth
    :math:`f(x_1, \\ldots, x_d)`.

    Built from ``d`` marginal :class:`SplineBasis` (already sum-to-zero
    constrained) by the **row-wise tensor product** of their designs and the
    Kronecker-sum of per-margin penalties
    :math:`S_j = I \\otimes \\cdots P_j \\cdots \\otimes I` -- so each margin
    keeps its own smoothing parameter (selected jointly by the GAM
    Fellner-Schall loop, which handles more than one penalty per smooth).  This
    is mgcv's tensor **interaction** (``ti``): the marginal centring removes the
    main effects, so the term models the *interaction* and stays identifiable
    against the intercept; add marginal :math:`s(x_j)` smooths for a full
    ``te``.

    The marginal penalties commute (Kronecker structure), so they are
    **simultaneously diagonalisable**: ``pen_eig[j]`` is penalty :math:`j` in
    the joint eigenbasis (:math:`U_1 \\otimes \\cdots \\otimes U_d`), which the
    GAM engine uses to evaluate the smoothing-parameter trace
    :math:`\\operatorname{tr}(S_\\lambda^{+} S_j)` as an elementwise sum --
    cuSOLVER-free, no per-iteration pseudo-inverse -- while the fit itself keeps
    the original (non-rotated) tensor basis.

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

    def penalty_blocks(self) -> list:
        """Return the per-margin penalty blocks.

        One block per margin: the Kronecker-embedded penalty and its
        joint-eigenbasis eigenvalues.

        Returns
        -------
        list of tuple of numpy.ndarray
            One ``(S_j, eig_j)`` pair per margin.
        """
        pens = np.asarray(self.penalties)
        eigs = np.asarray(self.pen_eig)
        return [(pens[j], eigs[j]) for j in range(pens.shape[0])]

    def eval_design(
        self, x: Tuple[Float[Array, ' g'], ...]
    ) -> Float[Array, 'g K']:
        """Re-evaluate the tensor design on matched per-margin grids.

        Parameters
        ----------
        x
            A tuple of one length-``g`` grid per margin; point ``t`` has
            covariate tuple ``(x[0][t], ..., x[d-1][t])``.

        Returns
        -------
        Float[Array, 'g K']
            The ``(g, K)`` tensor design along the matched path.
        """
        return tensor_product_design(self, tuple(x))

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
    TensorBasis
        The :class:`TensorBasis` with the ``(n, prod_j k_j)`` design and the
        per-margin penalties.
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
    """Re-evaluate a :class:`TensorBasis` on a matched covariate grid.

    ``xs`` is one grid per margin (all length ``g``); point ``t`` has covariate
    tuple ``(xs[0][t], ..., xs[d-1][t])``.  Matrix-multiplying the result by the
    fitted tensor coefficients renders the interaction surface along that path.

    Parameters
    ----------
    basis
        The :class:`TensorBasis` to re-evaluate.
    xs
        A tuple of one length-``g`` grid per margin.

    Returns
    -------
    Float[Array, 'g K']
        The ``(g, K)`` tensor design at the matched grid points.
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
    **is** the random-effect precision :math:`\\lambda = 1 / \\sigma_b^2` -- the
    "penalised GLM equals variance-components REML" identity.  It is a third
    smooth-term variant alongside :class:`SplineBasis` / :class:`TensorBasis`,
    so it slots straight into :func:`~nitrix.stats.gam_fit` with **no new solver
    work** (a random effect is just one more penalty block).

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

    def penalty_blocks(self) -> list:
        """Return the single identity-ridge penalty block.

        The block is the identity ridge (full rank ``q``, all eigenvalues
        exactly one -- no host eigendecomposition needed).

        Returns
        -------
        list of tuple of numpy.ndarray
            A single ``(I, 1)`` pair: the identity penalty and its all-ones
            eigenvalues.
        """
        return [(np.asarray(self.penalty), np.ones(self.dim))]

    def eval_design(self, x: Float[Array, ' g']) -> Float[Array, 'g q']:
        """Re-evaluate as the one-hot of the requested level indices.

        Parameters
        ----------
        x
            ``(g,)`` integer level indices at which to read off the per-level
            random effect.

        Returns
        -------
        Float[Array, 'g q']
            The ``(g, q)`` one-hot design of the requested levels.
        """
        levels = jnp.asarray(x).astype(jnp.int32)
        return jax.nn.one_hot(levels, self.dim, dtype=self.design.dtype)

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
    REBasis
        The :class:`REBasis` smooth-term block for
        :func:`~nitrix.stats.gam_fit`.  The Fellner-Schall smoothing parameter
        on this block is the random-effect precision
        :math:`1 / \\sigma_b^2`; the fitted block coefficients are the BLUPs.
    """
    g = jnp.asarray(g)
    q = int(n_levels) if n_levels is not None else int(jnp.max(g)) + 1
    design = jax.nn.one_hot(g, q)  # (n, q)
    if by is not None:
        design = design * jnp.asarray(by, dtype=design.dtype)[:, None]
    # The identity ridge depends only on the static level count q, so build it as
    # a host (numpy) constant -- like the spline difference penalties.  A jnp.eye
    # would be a *tracer* when re_smooth runs under jax.jit (e.g. a jitted
    # glmm_fit few-level fit), which gam_fit's penalty machinery np.asarray's at
    # trace time -- so the penalty must be concrete, not traced.
    penalty = np.eye(q, dtype=design.dtype)
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
    into the ``smooths=`` argument of :func:`~nitrix.stats.gam_fit` (each block
    is an independent smooth, so the Fellner-Schall loop selects a per-level
    smoothing parameter).

    Construction: a single marginal P-spline of ``x`` (shared knots / penalty /
    centering), then level ``l``'s design is that marginal design with the rows
    where ``f != l`` zeroed -- so level ``l``'s coefficients load only on its own
    observations.  Because the per-level designs have disjoint nonzero rows, the
    blocks are orthogonal in :math:`X^{\\top} W X` (each level is fit on its own
    data).  Pair with a parametric factor main effect to carry the per-level
    mean (the smooths are sum-to-zero when ``center=True``, as in mgcv).

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
    :math:`z\\, f(x)` -- a coefficient on ``z`` that varies smoothly with ``x``.
    The design is the marginal P-spline of ``x`` scaled row-wise by ``z``; the
    penalty is unchanged.  A single :class:`SplineBasis` for
    :func:`~nitrix.stats.gam_fit`.

    Parameters
    ----------
    x
        ``(n,)`` covariate that the coefficient varies over.
    by
        ``(n,)`` continuous covariate the smooth multiplies.
    n_basis, degree, penalty_order, bounds, center
        Marginal P-spline parameters (see :func:`bspline_basis`).

    Returns
    -------
    SplineBasis
        The :class:`SplineBasis` whose design is the marginal P-spline scaled
        row-wise by ``by``.
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
