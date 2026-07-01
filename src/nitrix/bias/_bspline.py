# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Separable cubic B-spline scattered-data approximation on a regular grid.

This is the field-smoothing engine of N4 bias correction.  N4 replaced
N3's smoothing spline with the *multilevel B-spline approximation* (MBA)
of Lee, Wolberg and Shin, which is what ANTs / ITK use via
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
  reconstruction with the Lee--Wolberg--Shin :math:`w^2` weighting,
  likewise a sequence of per-axis contractions.

Both are dense, XLA-friendly (the contractions lower to ``dot`` /
tensor cores), differentiable, and free of the sparse-gather / scatter
friction of general scattered-data interpolation.

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

References
----------
.. [1] Tustison NJ, Avants BB, Cook PA, Zheng Y, Egan A, Yushkevich PA,
       Gee JC (2010). N4ITK: improved N3 bias correction. IEEE
       Transactions on Medical Imaging, 29(6), 1310--1320.
       :doi:`10.1109/TMI.2010.2046908`
.. [2] Lee S, Wolberg G, Shin SY (1997). Scattered data interpolation
       with multilevel B-splines. IEEE Transactions on Visualization and
       Computer Graphics, 3(3), 228--244. :doi:`10.1109/2945.620490`
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from nitrix.linalg._solver import safe_inv
from nitrix.numerics._spline import (
    difference_penalty_1d,
    uniform_bspline_weights,
)

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


# The uniform B-spline weight evaluator is shared with stats.basis -- see
# ``nitrix.numerics._spline.uniform_bspline_weights`` (imported above).  N4 uses
# *uniform* (non-clamped) knots to match ITK/ANTs, which is what that evaluator
# provides.


def _reconstruction_matrix(
    n_vox: int,
    n_control: int,
    order: int,
    dtype: jnp.dtype,
) -> Float[Array, 'n_vox n_control']:
    """Banded control-lattice -> voxel-grid interpolation matrix :math:`R`.

    Entry :math:`R_{ia}` is the single-axis B-spline weight of control
    point ``a`` at voxel ``i``.  Each row has ``order + 1`` non-zeros.

    The parametric mapping matches ITK's ``BSplineScatteredData``
    convention: voxel ``i`` of ``n_vox`` maps to parametric coordinate
    :math:`i / (\\mathrm{n\\_vox} - 1)` in :math:`[0, 1]`, scaled by the
    number of B-spline spans (``n_control - order``).  The span index is
    clamped to the last valid span at the closed-domain endpoint.

    Parameters
    ----------
    n_vox : int
        Number of voxels along this spatial axis.  Must be at least 2.
    n_control : int
        Number of B-spline control points along this axis.  Must be at
        least ``order + 1`` (one B-spline span).
    order : int
        B-spline order (3 for cubic).
    dtype : jnp.dtype
        Floating dtype of the returned matrix.

    Returns
    -------
    Float[Array, 'n_vox n_control']
        The banded interpolation matrix :math:`R`; row ``i`` holds the
        ``order + 1`` non-zero B-spline weights of the control points
        supporting voxel ``i``, scattered into their lattice columns.
    """
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

    weights = uniform_bspline_weights(frac, order)  # (n_vox, order+1)

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
    """Contract ``matrix`` (out, in) against ``x`` along ``axis``.

    The named ``axis`` of ``x`` (size ``matrix.shape[1]``) is replaced by
    an axis of size ``matrix.shape[0]``.  All other dims -- including
    arbitrary leading batch dims -- broadcast through unchanged.

    Parameters
    ----------
    x : Array
        Input tensor; its ``axis`` dimension must match the input size
        ``matrix.shape[1]``.
    matrix : Array
        A ``(out, in)`` contraction matrix.
    axis : int
        The axis of ``x`` to contract and replace.

    Returns
    -------
    Array
        ``x`` with its ``axis`` dimension replaced by one of size
        ``matrix.shape[0]``; all other dimensions unchanged.
    """
    out = jnp.tensordot(x, matrix, axes=([axis], [1]))  # new axis last
    return jnp.moveaxis(out, -1, axis)


def _reconstruct(
    phi: Array,
    matrices: Sequence[Array],
    spatial_axes: Sequence[int],
) -> Array:
    """Reconstruct the voxel-grid field from the control lattice ``phi``.

    Applies the per-axis interpolation matrices in turn, expanding each
    control-lattice axis up to its voxel-grid size (a separable transposed
    interpolation).

    Parameters
    ----------
    phi : Array
        Control-lattice coefficients; the ``spatial_axes`` dimensions size
        the control lattice, any other dimensions broadcast through.
    matrices : Sequence[Array]
        Per-axis interpolation matrices :math:`R_d` of shape
        ``(n_vox_d, n_control_d)``, one per entry of ``spatial_axes``.
    spatial_axes : Sequence[int]
        Axes of ``phi`` to expand, paired positionally with ``matrices``.

    Returns
    -------
    Array
        The reconstructed field with each spatial axis grown to its
        voxel-grid size.
    """
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
    """Fit the control lattice (Lee--Wolberg--Shin MBA) from grid data.

    ``matrices`` are the per-axis reconstruction matrices :math:`R_d`; the
    fit is their adjoint with the :math:`w^2` weighting.  ``values`` and
    ``weight`` share the (broadcastable) voxel-grid shape.

    Parameters
    ----------
    values : Array
        Voxel-grid data to approximate.
    weight : Array
        Per-voxel confidence weights, broadcastable to ``values``.
    matrices : Sequence[Array]
        Per-axis interpolation matrices :math:`R_d` of shape
        ``(n_vox_d, n_control_d)``, one per entry of ``spatial_axes``.
    spatial_axes : Sequence[int]
        Axes of ``values`` that are spatial, paired with ``matrices``.
    eps : float
        Small stabiliser added to the per-point normaliser and to the
        control-point denominator to guard against division by zero for
        control points with negligible support.

    Returns
    -------
    Array
        The fitted control-lattice coefficients; each spatial axis is
        reduced to its control-lattice size.
    """
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
# Least-squares and penalised (P-spline) fits
#
# These are the higher-accuracy alternatives to the MBA fit above (which is
# ITK-parity but biased on dense data).  The weighted least-squares fit
# minimises ``||R phi - z||^2_W`` and reproduces smooth fields exactly; the
# P-spline fit adds a difference penalty ``lambda ||D phi||^2`` for a single
# smooth fit without the multi-resolution hierarchy.  Both solve the normal
# equations ``(R^T W R + P) phi = R^T W z`` -- and crucially stay separable:
# the Gram ``R^T W R`` is assembled WITHOUT materialising ``R`` via a
# per-axis multilinear contraction of the weights, and the control lattice
# is tiny so the dense solve is cheap.  See ``docs/design/bias-field.md``.
# ---------------------------------------------------------------------------


def _adjoint_to_control(
    grid: Array,
    matrices: Sequence[Array],
    spatial_axes: Sequence[int],
) -> Array:
    """Plain separable adjoint :math:`R^{\\top}` of grid data onto the lattice.

    Unlike ``_fit`` (which carries the MBA :math:`w^2` weighting), this is
    the bare transpose of the reconstruction -- the right-hand side
    :math:`R^{\\top}(W z)` of the least-squares normal equations.

    Parameters
    ----------
    grid : Array
        Voxel-grid data (typically the weighted values :math:`W z`).
    matrices : Sequence[Array]
        Per-axis interpolation matrices :math:`R_d` of shape
        ``(n_vox_d, n_control_d)``, one per entry of ``spatial_axes``.
    spatial_axes : Sequence[int]
        Axes of ``grid`` that are spatial, paired with ``matrices``.

    Returns
    -------
    Array
        The adjoint result on the control lattice; each spatial axis is
        reduced to its control-lattice size.
    """
    out = grid
    for axis, R in zip(spatial_axes, matrices):
        out = _contract_axis(out, R.T, axis)
    return out


def _weighted_gram(
    weight: Array,
    matrices: Sequence[Array],
    spatial_axes: Sequence[int],
) -> Array:
    """Dense weighted Gram :math:`R^{\\top}\\operatorname{diag}(w)\\,R`.

    The Gram is over the flattened control lattice.  It is assembled
    without materialising the (huge) tensor-product :math:`R`: the entry
    :math:`G_{kl} = \\sum_v w_v R_{vk} R_{vl}` factorises over axes, so it
    is a per-axis contraction of ``weight`` with the basis outer-products
    :math:`Q_d[i, k, l] = R_d[i, k]\\, R_d[i, l]`.  ``weight`` must be
    exactly the spatial grid (no batch dims).

    Parameters
    ----------
    weight : Array
        Per-voxel confidence weights on the spatial grid (no batch dims).
    matrices : Sequence[Array]
        Per-axis interpolation matrices :math:`R_d` of shape
        ``(n_vox_d, n_control_d)``, one per entry of ``spatial_axes``.
    spatial_axes : Sequence[int]
        Axes of ``weight`` that are spatial, paired with ``matrices``.

    Returns
    -------
    Array
        The dense ``(n_control, n_control)`` weighted Gram matrix over the
        flattened control lattice, where ``n_control`` is the product of
        the per-axis control sizes.
    """
    g = weight
    csizes = []
    for axis, R in zip(spatial_axes, matrices):
        n_vox, c = R.shape
        q = (R[:, :, None] * R[:, None, :]).reshape(n_vox, c * c)
        g = _contract_axis(g, q.T, axis)  # n_vox -> c^2
        csizes.append(c)
    r = len(csizes)
    # g now has each spatial axis = c_d^2, interleaved (k_d, l_d) per axis.
    g = g.reshape(tuple(s for c in csizes for s in (c, c)))
    perm = tuple(range(0, 2 * r, 2)) + tuple(range(1, 2 * r, 2))
    g = jnp.transpose(g, perm)
    n = int(np.prod(csizes))
    return g.reshape(n, n)


def _difference_penalty(
    control_sizes: Sequence[int],
    order: int,
    dtype: jnp.dtype,
) -> Array:
    """Tensor-product difference (P-spline) penalty over the control grid.

    Builds :math:`P = \\sum_d (I \\otimes \\cdots \\otimes D_d^{\\top} D_d
    \\otimes \\cdots \\otimes I)`, where :math:`D_d` is the ``order``-th
    finite-difference operator on axis :math:`d`'s control points.
    Penalises roughness along each axis independently (the Eilers--Marx
    multidimensional P-spline penalty).

    Parameters
    ----------
    control_sizes : Sequence[int]
        Number of control points along each spatial axis.
    order : int
        Order of the finite-difference operator :math:`D_d` (2 penalises
        curvature).  Axes with fewer than ``order + 1`` control points
        contribute no penalty term.
    dtype : jnp.dtype
        Floating dtype of the returned matrix.

    Returns
    -------
    Array
        The dense ``(n_control, n_control)`` penalty matrix over the
        flattened control lattice, where ``n_control`` is the product of
        ``control_sizes``.
    """
    n = int(np.prod(control_sizes))
    P = jnp.zeros((n, n), dtype=dtype)
    eyes = [jnp.eye(c, dtype=dtype) for c in control_sizes]
    for d, c in enumerate(control_sizes):
        if c <= order:
            continue  # too few control points to take this difference
        dtd = difference_penalty_1d(c, order, dtype)  # shared 1-D D^T D
        term: Optional[Array] = None
        for j in range(len(control_sizes)):
            block = dtd if j == d else eyes[j]
            term = block if term is None else jnp.kron(term, block)
        if term is not None:
            P = P + term
    return P


def _control_inverse_gram(
    weight: Array,
    matrices: Sequence[Array],
    spatial_axes: Sequence[int],
    control_sizes: Sequence[int],
    *,
    ridge: float,
    penalty: float,
    penalty_order: int,
    dtype: jnp.dtype,
) -> Array:
    """Regularised inverse Gram :math:`(R^{\\top} W R + \\mathrm{reg})^{-1}`.

    The Gram is over the control lattice.  It depends only on ``weight``
    and the level's control grid -- not on the data -- so N4 computes this
    once per fitting level and reuses it across every sharpening iteration.
    The control lattice is small, so the explicit (well-conditioned,
    regularised) inverse is cheap.  The regulariser is a Tikhonov ridge
    plus an optional difference penalty, each scaled by the mean Gram
    diagonal.

    Parameters
    ----------
    weight : Array
        Per-voxel confidence weights on the spatial grid (no batch dims).
    matrices : Sequence[Array]
        Per-axis interpolation matrices :math:`R_d` of shape
        ``(n_vox_d, n_control_d)``, one per entry of ``spatial_axes``.
    spatial_axes : Sequence[int]
        Axes of ``weight`` that are spatial, paired with ``matrices``.
    control_sizes : Sequence[int]
        Number of control points along each spatial axis.
    ridge : float
        Tikhonov ridge strength relative to the mean Gram diagonal (a
        small SPD jitter is always added on top).
    penalty : float
        Weight of the difference penalty relative to the mean Gram
        diagonal.  A non-positive value omits the penalty term.
    penalty_order : int
        Order of the difference penalty operator.
    dtype : jnp.dtype
        Floating dtype for the assembled matrices and the inverse.

    Returns
    -------
    Array
        The dense ``(n_control, n_control)`` inverse of the regularised
        Gram over the flattened control lattice.
    """
    G = _weighted_gram(weight, matrices, spatial_axes)
    n = G.shape[0]
    mean_diag = jnp.mean(jnp.diagonal(G))
    G = G + (ridge + _JITTER) * mean_diag * jnp.eye(n, dtype=dtype)
    if penalty > 0.0:
        P = _difference_penalty(control_sizes, penalty_order, dtype)
        G = G + penalty * mean_diag * P
    return safe_inv(G)


def _solve_field(
    values: Array,
    weight: Array,
    inv_gram: Array,
    matrices: Sequence[Array],
    spatial_axes: Sequence[int],
    control_sizes: Sequence[int],
) -> Array:
    """Least-squares control lattice (via the precomputed inverse) -> field.

    Forms the right-hand side :math:`R^{\\top}(W z)`, applies the
    precomputed regularised inverse Gram to obtain the control-lattice
    coefficients, and reconstructs the smooth field on the voxel grid.

    Parameters
    ----------
    values : Array
        Voxel-grid data to approximate (spatial grid, no batch dims).
    weight : Array
        Per-voxel confidence weights, same shape as ``values``.
    inv_gram : Array
        Precomputed ``(n_control, n_control)`` inverse regularised Gram
        over the flattened control lattice.
    matrices : Sequence[Array]
        Per-axis interpolation matrices :math:`R_d` of shape
        ``(n_vox_d, n_control_d)``, one per entry of ``spatial_axes``.
    spatial_axes : Sequence[int]
        Axes of ``values`` that are spatial, paired with ``matrices``.
    control_sizes : Sequence[int]
        Number of control points along each spatial axis; used to reshape
        the solved coefficients back into the control lattice.

    Returns
    -------
    Array
        The reconstructed smooth field on the voxel grid, same shape as
        ``values``.
    """
    rhs = _adjoint_to_control(weight * values, matrices, spatial_axes)
    n = inv_gram.shape[0]
    phi = (inv_gram @ rhs.reshape(n)).reshape(tuple(control_sizes))
    return _reconstruct(phi, matrices, spatial_axes)


def _fit_regularised(
    values: Array,
    weight: Array,
    matrices: Sequence[Array],
    spatial_axes: Sequence[int],
    control_sizes: Sequence[int],
    *,
    ridge: float,
    penalty: float,
    penalty_order: int,
    dtype: jnp.dtype,
) -> Array:
    """One-shot least-squares / P-spline fit-and-reconstruct.

    Convenience wrapper that assembles the regularised inverse Gram and
    then solves for and reconstructs the smooth field in a single call
    (used when the inverse is not reused across iterations).

    Parameters
    ----------
    values : Array
        Voxel-grid data to approximate (spatial grid, no batch dims).
    weight : Array
        Per-voxel confidence weights, same shape as ``values``.
    matrices : Sequence[Array]
        Per-axis interpolation matrices :math:`R_d` of shape
        ``(n_vox_d, n_control_d)``, one per entry of ``spatial_axes``.
    spatial_axes : Sequence[int]
        Axes of ``values`` that are spatial, paired with ``matrices``.
    control_sizes : Sequence[int]
        Number of control points along each spatial axis.
    ridge : float
        Tikhonov ridge strength relative to the mean Gram diagonal.
    penalty : float
        Weight of the difference penalty relative to the mean Gram
        diagonal (non-positive omits it).
    penalty_order : int
        Order of the difference penalty operator.
    dtype : jnp.dtype
        Floating dtype for the assembled matrices and the solve.

    Returns
    -------
    Array
        The reconstructed smooth field on the voxel grid, same shape as
        ``values``.
    """
    inv_gram = _control_inverse_gram(
        weight,
        matrices,
        spatial_axes,
        control_sizes,
        ridge=ridge,
        penalty=penalty,
        penalty_order=penalty_order,
        dtype=dtype,
    )
    return _solve_field(
        values, weight, inv_gram, matrices, spatial_axes, control_sizes
    )


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
    method: FitMethod = 'mba',
    ridge: float = 1e-4,
    penalty: float = 1.0,
    penalty_order: int = 2,
    spatial_rank: Optional[int] = None,
    eps: float = 1e-8,
) -> Float[Array, '... *spatial']:
    """Smooth, separable cubic B-spline approximation of regular-grid data.

    Fits a uniform tensor-product B-spline control lattice to ``values`` and
    reconstructs the smooth field at every voxel.  This is the field-
    smoothing primitive N4 uses to regularise its per-iteration bias-field
    residual, but it stands alone as a fast, differentiable scattered-data
    smoother (registration fields, surface fitting, any "approximate this
    noisy grid with a smooth low-DOF surface" task).

    Both the fit and the reconstruction are separable -- a sequence of small
    dense per-axis matrix contractions (no gather / scatter) -- so every
    method is pure JAX, XLA-friendly, and differentiable.  The number of
    control points sets the model resolution (fewer -> smoother).

    Parameters
    ----------
    values
        Data to approximate, ``(..., *spatial)``.  Leading dims are batch.
    control_points
        Number of B-spline control points per spatial axis (ITK's
        ``NumberOfControlPoints``).  ``int`` -- same count on every axis;
        sequence -- per-axis.  Must be at least ``spline_order + 1``.
    weight
        Optional per-voxel confidence in ``[0, 1]`` (e.g. a brain mask),
        same broadcastable shape as ``values``.  ``None`` (default) treats
        every voxel as a unit-confidence data point.  Zero-weight voxels do
        not influence the fit.
    spline_order
        B-spline order.  Default ``3`` (cubic; the N4 / ANTs default).
    method
        Fitting estimator:

        - ``'mba'`` (default) -- Lee--Wolberg--Shin multilevel B-spline
          approximation, the :math:`w^2`-weighted scatter ITK / ANTs N4
          uses.  Biased on dense data at a single level (the bias is
          removed by *iterated* refitting on a doubling grid); the
          **parity** path.
        - ``'least_squares'`` -- weighted least-squares fit
          (:math:`\\min \\lVert R\\phi - z \\rVert_W^2`); reproduces smooth
          fields exactly, no MBA bias, converges in fewer levels.  Uses
          ``ridge``.
        - ``'psplines'`` -- penalised least-squares (Eilers--Marx
          P-splines): adds a difference penalty
          :math:`\\lambda \\lVert D\\phi \\rVert^2` for a single smooth fit
          on a *fine* control grid without the multi-resolution hierarchy.
          Uses ``penalty`` + ``penalty_order`` (and ``ridge``).
    ridge
        Tikhonov regularisation for the LS / P-spline normal equations,
        relative to the mean Gram diagonal.  Default ``1e-4`` -- near-exact,
        appropriate for *clean* data; it also stabilises control points
        poorly supported by the mask.  For **noisy** data, ``ridge`` is the
        denoising strength (the bias-variance knob): increase it (e.g.
        ``1e-1``) to trade exactness for noise rejection.  This is why the
        :func:`bias_field_correction` loop defaults ``ridge`` higher.
        Ignored by ``'mba'``.
    penalty
        P-spline roughness penalty weight (relative to the mean Gram
        diagonal); larger -> smoother.  Only used by ``'psplines'``.
    penalty_order
        Order of the difference penalty for ``'psplines'`` (default ``2``,
        penalising curvature).
    spatial_rank
        Number of trailing dims treated as spatial.  ``None`` infers it
        from ``control_points`` (if a sequence) or assumes all dims are
        spatial.
    eps
        Stabiliser for the MBA fit's near-empty control points.

    Returns
    -------
    Float[Array, '... *spatial']
        The smooth B-spline field sampled on the voxel grid, same shape as
        ``values``.

    Notes
    -----
    On a regular grid the B-spline weights factorise across axes, so the
    scattered-data fit collapses to per-axis banded matrices :math:`R_d`
    of shape ``(n_vox_d, n_control_d)``.  ``'mba'`` applies the adjoint
    with the Lee--Wolberg--Shin :math:`w^2` weighting; ``'least_squares'``
    / ``'psplines'`` solve the normal equations
    :math:`(R^{\\top} W R + P)\\,\\phi = R^{\\top} W z` with the Gram
    assembled without materialising :math:`R`.
    """
    if method not in ('mba', 'least_squares', 'psplines'):
        raise ValueError(
            f"method={method!r}; expected 'mba', 'least_squares', or "
            "'psplines'."
        )
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

    if method == 'mba':
        phi = _fit(x, w, matrices, spatial_axes, eps)
        return _reconstruct(phi, matrices, spatial_axes)

    # Least-squares / P-spline solve operates per data tensor (the Gram
    # depends on the weights), so vmap over any leading batch dims.
    use_penalty = penalty if method == 'psplines' else 0.0
    core_axes = tuple(range(rank))

    def core(v: Array, ww: Array) -> Array:
        return _fit_regularised(
            v,
            ww,
            matrices,
            core_axes,
            n_control,
            ridge=ridge,
            penalty=use_penalty,
            penalty_order=penalty_order,
            dtype=dtype,
        )

    batch_shape = x.shape[: x.ndim - rank]
    if batch_shape:
        n_batch = int(np.prod(batch_shape))
        xb = x.reshape((n_batch, *spatial_shape))
        wb = w.reshape((n_batch, *spatial_shape))
        out = jax.vmap(core)(xb, wb)
        return out.reshape(x.shape)
    return core(x, w)
