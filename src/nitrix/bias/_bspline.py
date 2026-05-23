# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Separable cubic B-spline scattered-data approximation on a regular grid.

This is the field-smoothing engine of N4 (SPEC charter: "software in the
class of ANTs").  N4 (Tustison 2010) replaced N3's smoothing spline with
the Lee--Wolberg--Shin (1997) *multilevel B-spline approximation* (MBA),
which is what ANTs / ITK use via
``BSplineScatteredDataPointSetToImageFilter``.

The general MBA is a scatter over arbitrary point positions.  But N4's
data points are *image voxels*, which lie on a **regular grid**.  On a
regular grid the cubic B-spline weight from a voxel to a control point
depends only on the voxel's position relative to the control lattice, so
the fit and the reconstruction both factor into a product of per-axis
banded matrices.  We exploit this:

- **reconstruct** (control lattice -> voxel grid) is a sequence of small
  dense matrix contractions, one per spatial axis -- a separable
  transposed interpolation.  No gather, no scatter.
- **fit** (voxel grid -> control lattice) is the exact adjoint of the
  reconstruction *with the Lee--Wolberg--Shin* ``w^2`` *weighting*,
  likewise a sequence of per-axis contractions.

Both are dense, XLA-friendly (the contractions lower to ``dot`` /
tensor cores), differentiable, and free of the ``jax.experimental.sparse``
/ scatter friction the SPEC §3.2 warns against.  See
``docs/design/bias-field.md`` for the derivation and the equivalence to
ITK's control-point-lattice accumulation.

Mathematics (per spatial axis, control lattice value :math:`\\phi`):

For each data point :math:`c` with value :math:`z_c`, confidence weight
:math:`m_c`, and tensor-product B-spline weights :math:`w_{c\\to k}` onto
the surrounding control points :math:`k`, the MBA control value is

.. math::
    \\phi_k = \\frac{\\sum_c m_c\\, w_{c\\to k}^2\\, \\psi_{c,k}}
                   {\\sum_c m_c\\, w_{c\\to k}^2},
    \\qquad
    \\psi_{c,k} = \\frac{w_{c\\to k}\\, z_c}{\\sum_j w_{c\\to j}^2}.

Because the weights are separable, :math:`w_{c\\to k} = \\prod_d R_d[i_d, k_d]`
and :math:`\\sum_j w_{c\\to j}^2 = \\prod_d (\\sum_a R_d[i_d, a]^2)`, so both
sums become per-axis contractions with :math:`R_d^2` (denominator) and
:math:`R_d^3` (numerator).
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

__all__ = ['bspline_approximate']

# A tiny SPD jitter (relative to the Gram diagonal) so the regularised
# normal-equation matrix is always Cholesky/inverse-safe, including for
# control points whose support is entirely outside the mask (their fitted
# value is driven to ~0, mirroring the MBA "no support -> 0" behaviour).
_JITTER = 1e-6

FitMethod = Literal['mba', 'least_squares', 'psplines']


# ---------------------------------------------------------------------------
# Uniform B-spline basis
# ---------------------------------------------------------------------------


def _uniform_bspline_weights(
    t: Float[Array, ' n'],
    order: int,
) -> Float[Array, 'n order_plus_1']:
    '''Uniform B-spline basis weights for fractional positions ``t``.

    Returns the ``order + 1`` non-zero basis values for each fractional
    coordinate ``t`` in ``[0, 1]``.  These are the weights on the
    ``order + 1`` consecutive control points spanning the local knot
    interval.  The basis is a partition of unity (rows sum to 1).

    Closed forms for the orders that matter in practice:

    - order 1 (linear):    hat function, 2 control points.
    - order 2 (quadratic): 3 control points.
    - order 3 (cubic):     4 control points -- the N4 / ANTs default.
    '''
    if order == 1:
        return jnp.stack([1.0 - t, t], axis=-1)
    if order == 2:
        return jnp.stack(
            [
                0.5 * (1.0 - t) ** 2,
                0.5 * (1.0 + 2.0 * t - 2.0 * t**2),
                0.5 * t**2,
            ],
            axis=-1,
        )
    if order == 3:
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
        f'spline_order={order!r} is not supported; nitrix.bias ships the '
        'uniform B-spline basis for orders 1, 2, 3 (3 = cubic is the N4 / '
        'ANTs default and the parity-validated path). Higher orders are a '
        'documented extension point: add the uniform-knot Cox--de Boor '
        'recursion here (the closed forms above are its order-1/2/3 '
        'specialisations). Note we use *uniform* (non-clamped) knots to '
        'match ITK/ANTs N4 -- not the endpoint-interpolating clamped knots '
        'used by curve-fitting libraries.'
    )


def _reconstruction_matrix(
    n_vox: int,
    n_control: int,
    order: int,
    dtype: jnp.dtype,
) -> Float[Array, 'n_vox n_control']:
    '''Banded control-lattice -> voxel-grid interpolation matrix ``R``.

    ``R[i, a]`` is the tensor-product-axis B-spline weight of control
    point ``a`` at voxel ``i``.  Each row has ``order + 1`` non-zeros.

    The parametric mapping matches ITK's ``BSplineScatteredData``
    convention: voxel ``i`` of ``n_vox`` maps to parametric coordinate
    ``i / (n_vox - 1)`` in ``[0, 1]``, scaled by the number of B-spline
    spans (``n_control - order``).  The span index is clamped to the last
    valid span at the closed-domain endpoint.
    '''
    n_spans = n_control - order
    if n_spans < 1:
        raise ValueError(
            f'control_points={n_control} too small for spline_order={order}: '
            f'need at least order + 1 = {order + 1} control points per axis '
            '(one B-spline span).'
        )
    if n_vox < 2:
        raise ValueError(f'each spatial axis needs >= 2 voxels; got {n_vox}.')

    # Continuous knot coordinate s in [0, n_spans]; span = floor(s) clamped
    # to [0, n_spans - 1]; frac = s - span in [0, 1].
    param = jnp.arange(n_vox, dtype=dtype) / (n_vox - 1)
    s = param * n_spans
    span = jnp.clip(jnp.floor(s).astype(jnp.int32), 0, n_spans - 1)
    frac = s - span.astype(dtype)

    weights = _uniform_bspline_weights(frac, order)  # (n_vox, order+1)

    # Scatter the (order + 1) banded weights into a dense (n_vox, n_control)
    # matrix.  Column index for tap j at voxel i is span[i] + j.
    rows = jnp.arange(n_vox)[:, None]  # (n_vox, 1)
    cols = span[:, None] + jnp.arange(order + 1)[None, :]  # (n_vox, order+1)
    R = jnp.zeros((n_vox, n_control), dtype=dtype)
    R = R.at[rows, cols].add(weights)
    return R


# ---------------------------------------------------------------------------
# Separable contraction helpers
# ---------------------------------------------------------------------------


def _contract_axis(
    x: Array,
    matrix: Array,
    axis: int,
) -> Array:
    '''Contract ``matrix`` (out, in) against ``x`` along ``axis``.

    The named ``axis`` of ``x`` (size == ``matrix.shape[1]``) is replaced
    by an axis of size ``matrix.shape[0]``.  All other dims -- including
    arbitrary leading batch dims -- broadcast through unchanged.
    '''
    out = jnp.tensordot(x, matrix, axes=([axis], [1]))  # new axis last
    return jnp.moveaxis(out, -1, axis)


def _reconstruct(
    phi: Array,
    matrices: Sequence[Array],
    spatial_axes: Sequence[int],
) -> Array:
    '''Reconstruct the voxel-grid field from the control lattice ``phi``.'''
    out = phi
    for axis, R in zip(spatial_axes, matrices):
        out = _contract_axis(out, R, axis)
    return out


def _fit(
    values: Array,
    weight: Array,
    matrices: Sequence[Array],
    spatial_axes: Sequence[int],
    eps: float,
) -> Array:
    '''Fit the control lattice (Lee--Wolberg--Shin MBA) from grid data.

    ``matrices`` are the per-axis reconstruction matrices ``R_d``; the fit
    is their adjoint with the ``w^2`` weighting.  ``values`` and ``weight``
    share the (broadcastable) voxel-grid shape.
    '''
    # Per-axis sum-of-squared-weights profile: sumsq_d[i] = sum_a R_d[i, a]^2.
    # The per-point normaliser sum_j w_j^2 factorises as the product of these.
    normaliser = jnp.ones_like(values)
    for axis, R in zip(spatial_axes, matrices):
        sumsq = jnp.sum(R**2, axis=1)  # (n_vox_d,)
        shape = [1] * values.ndim
        shape[axis] = sumsq.shape[0]
        normaliser = normaliser * sumsq.reshape(shape)

    g = weight * values / (normaliser + eps)

    # numerator[ctrl] = sum over voxels of (prod_d R_d^3) * g
    numerator = g
    for axis, R in zip(spatial_axes, matrices):
        numerator = _contract_axis(numerator, (R**3).T, axis)

    # denominator[ctrl] = sum over voxels of (prod_d R_d^2) * weight
    denominator = weight
    for axis, R in zip(spatial_axes, matrices):
        denominator = _contract_axis(denominator, (R**2).T, axis)

    return numerator / (denominator + eps)


# ---------------------------------------------------------------------------
# Argument normalisation
# ---------------------------------------------------------------------------


def _normalise_control_points(
    control_points: Union[int, Sequence[int]],
    spatial_rank: int,
) -> tuple[int, ...]:
    if isinstance(control_points, int):
        return (int(control_points),) * spatial_rank
    out = tuple(int(c) for c in control_points)
    if len(out) != spatial_rank:
        raise ValueError(
            f'control_points must be an int or a length-{spatial_rank} '
            f'sequence; got {control_points!r}.'
        )
    return out


def _resolve_spatial_rank(
    control_points: Union[int, Sequence[int]],
    spatial_rank: Optional[int],
    ndim: int,
) -> int:
    inferred = (
        len(control_points)
        if isinstance(control_points, (tuple, list))
        else None
    )
    if spatial_rank is None:
        spatial_rank = inferred if inferred is not None else ndim
    elif inferred is not None and inferred != spatial_rank:
        raise ValueError(
            f'control_points has {inferred} elements but spatial_rank='
            f'{spatial_rank}.'
        )
    if spatial_rank < 1:
        raise ValueError('spatial_rank must be >= 1.')
    if ndim < spatial_rank:
        raise ValueError(
            f'input ndim={ndim} too small for spatial_rank={spatial_rank}.'
        )
    return spatial_rank


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def bspline_approximate(
    values: Float[Array, '... *spatial'],
    *,
    control_points: Union[int, Sequence[int]],
    weight: Optional[Float[Array, '... *spatial']] = None,
    spline_order: int = 3,
    spatial_rank: Optional[int] = None,
    eps: float = 1e-8,
) -> Float[Array, '... *spatial']:
    '''Smooth, separable cubic B-spline approximation of regular-grid data.

    Fits a uniform tensor-product B-spline control lattice to ``values``
    (the Lee--Wolberg--Shin scattered-data approximation specialised to the
    image grid) and reconstructs the smooth field at every voxel.  This is
    the field-smoothing primitive N4 uses to regularise its per-iteration
    bias-field residual, but it stands alone as a fast, differentiable
    scattered-data smoother (registration fields, surface fitting, any
    "approximate this noisy grid with a smooth low-DOF surface" task).

    Both the fit and the reconstruction are separable: a sequence of small
    dense per-axis matrix contractions (no gather / scatter), so the op is
    pure JAX, XLA-friendly, and differentiable end-to-end.  The number of
    control points sets the smoothness -- fewer control points -> stiffer
    (smoother) field.

    Parameters
    ----------
    values
        Data to approximate, ``(..., *spatial)``.  Leading dims are batch
        and broadcast through.
    control_points
        Number of B-spline control points per spatial axis (ITK's
        ``NumberOfControlPoints``).  ``int`` -- same count on every axis;
        sequence -- per-axis.  Must be at least ``spline_order + 1`` (one
        B-spline span).  More control points -> finer (less smooth) fit.
    weight
        Optional per-voxel confidence in ``[0, 1]`` (e.g. a brain mask),
        same broadcastable shape as ``values``.  ``None`` (default) treats
        every voxel as a unit-confidence data point.  Zero-weight voxels do
        not influence the fit; control points with no nearby support
        reconstruct to zero.
    spline_order
        B-spline order.  Default ``3`` (cubic, the N4 / ANTs default and
        the parity-validated path).  Orders 1 and 2 are also available.
    spatial_rank
        Number of trailing dims treated as spatial.  ``None`` infers it
        from ``control_points`` (if a sequence) or assumes all dims are
        spatial.
    eps
        Stabiliser for control points with little or no nearby support.

    Returns
    -------
    The smooth B-spline field sampled on the voxel grid, same shape as
    ``values``.

    Notes
    -----
    On a regular grid the cubic-B-spline weights are periodic in the
    control-point spacing, so the general (scattered) MBA collapses to
    per-axis banded matrices ``R_d`` of shape ``(n_vox_d, n_control_d)``.
    Reconstruction applies ``R_d`` along each axis; the fit applies the
    adjoint with the ``R_d^2`` (denominator) and ``R_d^3`` (numerator)
    weighting of Lee--Wolberg--Shin.  See ``docs/design/bias-field.md``.
    '''
    x = jnp.asarray(values)
    rank = _resolve_spatial_rank(control_points, spatial_rank, x.ndim)
    n_control = _normalise_control_points(control_points, rank)
    spatial_axes = tuple(range(x.ndim - rank, x.ndim))
    spatial_shape = tuple(x.shape[a] for a in spatial_axes)

    dtype = jnp.result_type(x.dtype, jnp.float32)
    x = x.astype(dtype)
    if weight is None:
        w = jnp.ones_like(x)
    else:
        w = jnp.broadcast_to(jnp.asarray(weight).astype(dtype), x.shape)

    matrices = [
        _reconstruction_matrix(n_vox, n_ctrl, spline_order, dtype)
        for n_vox, n_ctrl in zip(spatial_shape, n_control)
    ]

    phi = _fit(x, w, matrices, spatial_axes, eps)
    return _reconstruct(phi, matrices, spatial_axes)
