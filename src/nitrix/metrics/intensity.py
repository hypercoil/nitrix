# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Intensity-based image-similarity metrics.

- :func:`ssd` -- sum / mean of squared differences (within-modality;
  motion correction).
- :func:`ncc` -- global normalised cross-correlation (Pearson over all
  voxels; within-modality, robust to linear intensity change).
- :func:`lncc` -- local (windowed) normalised cross-correlation, the
  ANTs squared form -- the diffeomorphic workhorse, robust to smooth
  intensity inhomogeneity.

All are differentiable with respect to both arguments (they sit inside a
registration loss).  Convention: :func:`ssd` is a *cost* (minimise);
:func:`ncc` / :func:`lncc` are *similarities* in (roughly)
:math:`[0, 1]` (maximise, or minimise ``1 - metric``).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import jax.numpy as jnp
from jaxtyping import Array, Float

from ._common import BoundaryMode, Reduction, _box_sum, _reduce

__all__ = ['ssd', 'ncc', 'lncc', 'lncc_grad', 'lncc_grad_center']


def _normalise_radius(
    radius: Union[int, Sequence[int]],
    spatial_rank: int,
) -> tuple[int, ...]:
    if isinstance(radius, int):
        return (radius,) * spatial_rank
    out = tuple(int(r) for r in radius)
    if len(out) != spatial_rank:
        raise ValueError(
            f'radius must be an int or a length-{spatial_rank} '
            f'sequence; got {radius!r}.'
        )
    return out


def ssd(
    moving: Float[Array, '...'],
    fixed: Float[Array, '...'],
    *,
    mask: Optional[Float[Array, '...']] = None,
    reduction: Reduction = 'mean',
) -> Float[Array, '...']:
    """Sum / mean of squared differences.

    Parameters
    ----------
    moving, fixed
        Images of identical shape.
    mask
        Optional non-negative weight of the same shape; squared
        differences are weighted by it before reduction.
    reduction
        ``"mean"`` (default), ``"sum"``, or ``"none"`` (per-voxel
        squared-difference map).

    Returns
    -------
    Scalar cost (``"mean"`` / ``"sum"``) or the per-voxel map
    (``"none"``).

    Notes
    -----
    With the default ``reduction='mean'`` this is the ITK
    ``MeanSquares`` registration metric (verified bit-equal in fp64).
    """
    diff = moving - fixed
    return _reduce(diff * diff, mask, reduction)


def ncc(
    x: Float[Array, '...'],
    y: Float[Array, '...'],
    *,
    mask: Optional[Float[Array, '...']] = None,
    eps: float = 1e-8,
) -> Float[Array, '']:
    """Global normalised cross-correlation (Pearson correlation).

    Computed over *all* elements (flattened).  Returns a scalar in
    :math:`[-1, 1]`: :math:`1` for identical-up-to-positive-affine
    images, :math:`-1` for negated.  Use ``-ncc`` (or ``1 - ncc``) as a
    cost.

    Parameters
    ----------
    x, y
        Images of identical shape.  All elements are flattened together
        before the correlation is formed.
    mask
        Optional non-negative weight of the same shape.  Both means and
        the correlation are computed as weighted quantities under these
        weights, so an out-of-mask element (zero weight) does not
        contribute.  ``None`` (default) weights every element equally.
    eps
        Small positive floor guarding the zero-variance denominator (and
        the sum of weights), avoiding division by zero on a flat image.

    Returns
    -------
    Float[Array, '']
        The scalar signed Pearson correlation coefficient in
        :math:`[-1, 1]`.

    Notes
    -----
    This is the *signed* Pearson correlation.  ITK's ``Correlation``
    metric (``CorrelationImageToImageMetricv4``) is the same
    mean-subtracted correlation returned as ``-r**2`` (squared and
    negated for minimisation), so it is sign-insensitive where this is
    not.  Recover it as ``-ncc(...)**2``; the sign-agnostic cost is
    ``1 - ncc(...)**2`` (not ``1 - ncc``).
    """
    xf = x.reshape(-1)
    yf = y.reshape(-1)
    w = jnp.ones_like(xf) if mask is None else mask.reshape(-1)
    sw = jnp.maximum(w.sum(), eps)
    mx = (w * xf).sum() / sw
    my = (w * yf).sum() / sw
    xd = xf - mx
    yd = yf - my
    num = (w * xd * yd).sum()
    den = jnp.sqrt((w * xd * xd).sum() * (w * yd * yd).sum())
    return num / (den + eps)


def lncc(
    moving: Float[Array, '... *spatial'],
    fixed: Float[Array, '... *spatial'],
    *,
    radius: Union[int, Sequence[int]] = 4,
    spatial_rank: Optional[int] = None,
    mode: BoundaryMode = 'reflect',
    eps: float = 1e-5,
    reduction: Reduction = 'mean',
    mask: Optional[Float[Array, '... *spatial']] = None,
) -> Float[Array, '...']:
    r"""Local (windowed) normalised cross-correlation -- the ANTs form.

    Over a box window of radius :math:`r` (size :math:`2r + 1` per axis)
    the per-voxel local correlation is
    :math:`cc = (\sum \tilde{m}\,\tilde{f})^2 / (\sum \tilde{m}^2 \cdot \sum \tilde{f}^2)`,
    with :math:`\tilde{m}` / :math:`\tilde{f}` the window-mean-subtracted
    intensities, computed from windowed sums (separable box filter).
    Values lie in :math:`[0, 1]`; :math:`1` is a locally-perfect
    (affine) intensity match.  Robust to smooth intensity
    inhomogeneity, which is why it is the default for the diffeomorphic
    recipe.  Use ``1 - lncc`` as a cost.

    Parameters
    ----------
    moving, fixed
        Images of identical shape, ``(..., *spatial)``.  Leading axes
        are treated as batch (folded into the box-filter batch).
    radius
        Window half-width :math:`r`, giving a box of size
        :math:`2r + 1` per axis.  An ``int`` gives an isotropic window;
        a sequence gives one radius per spatial axis.
    spatial_rank
        Number of trailing axes to treat as spatial.  ``None`` (default)
        infers from ``radius`` (if a sequence) else assumes all of
        ``moving.ndim``.
    mode
        Boundary mode for the windowed sums.  Default ``"reflect"`` --
        which keeps the window count uniform, so the per-window
        normalisation is exact at the boundary.
    eps
        Denominator guard for flat (zero-variance) windows.
    reduction
        ``"mean"`` (default), ``"sum"``, or ``"none"`` (the per-voxel
        local-CC map).
    mask
        Optional non-negative per-voxel weight (image shape) restricting /
        weighting the **reduction** of the local-CC map -- an out-of-mask voxel
        does not count toward the mean.  Note this masks at reduction time only:
        the windowed sums at an in-mask voxel still draw on out-of-mask
        neighbours within its box (a deliberate, documented approximation --
        full window-gated sums would be a separate, costlier kernel).

    Returns
    -------
    Scalar similarity (``"mean"`` / ``"sum"``) or the per-voxel local-CC
    map (``"none"``).

    Notes
    -----
    The per-voxel value is interior-identical (fp64) to the ANTs
    ``ANTSNeighborhoodCorrelation`` local CC (the squared local
    correlation with window-local means).  ANTs returns it as a cost
    (the negated voxel mean); this returns the similarity, so its
    registration cost is ``1 - lncc``.  Only the boundary differs: the
    box sums use ``mode`` (default ``"reflect"``, a uniform window
    count) rather than ANTs' valid-neighbourhood trim.
    """
    if isinstance(radius, (tuple, list)):
        inferred_rank: Optional[int] = len(radius)
    else:
        inferred_rank = None
    if spatial_rank is None:
        spatial_rank = (
            inferred_rank if inferred_rank is not None else moving.ndim
        )
    elif inferred_rank is not None and inferred_rank != spatial_rank:
        raise ValueError(
            f'radius has {inferred_rank} elements but spatial_rank='
            f'{spatial_rank}.'
        )
    radii = _normalise_radius(radius, spatial_rank)
    sizes = tuple(2 * r + 1 for r in radii)
    spatial_axes = tuple(range(moving.ndim - spatial_rank, moving.ndim))
    n = 1.0
    for s in sizes:
        n *= float(s)

    sum_m = _box_sum(moving, sizes, spatial_axes, mode)
    sum_f = _box_sum(fixed, sizes, spatial_axes, mode)
    sum_mm = _box_sum(moving * moving, sizes, spatial_axes, mode)
    sum_ff = _box_sum(fixed * fixed, sizes, spatial_axes, mode)
    sum_mf = _box_sum(moving * fixed, sizes, spatial_axes, mode)

    cross = sum_mf - sum_m * sum_f / n
    var_m = sum_mm - sum_m * sum_m / n
    var_f = sum_ff - sum_f * sum_f / n
    cc = (cross * cross) / (var_m * var_f + eps)
    return _reduce(cc, mask, reduction)


def lncc_grad(
    moving: Float[Array, '... *spatial'],
    fixed: Float[Array, '... *spatial'],
    *,
    radius: Union[int, Sequence[int]] = 4,
    spatial_rank: Optional[int] = None,
    mode: BoundaryMode = 'reflect',
    eps: float = 1e-5,
) -> Float[Array, '... *spatial']:
    r"""Analytic gradient of the (summed) :func:`lncc` with respect to ``moving``.

    The closed-form :math:`\partial(\sum cc)/\partial\,\mathrm{moving}` --
    the local cross-correlation *force* that drives a greedy-SyN / Demons
    velocity update (multiply by the spatial gradient of ``moving`` for
    the deformation gradient).  Derived from the same windowed box sums
    :func:`lncc` already forms, plus a second box-sum pass over the
    per-window coefficients

    .. code-block:: text

        P = 2·cross / D,   Q = -2·cross²·var_f / D²,   D = var_m·var_f + eps
        grad = fixed·B(P) - B(P·f_bar) + moving·B(Q) - B(Q·m_bar)

    with :math:`B` the box sum and :math:`\bar{m} = \sum m / n` /
    :math:`\bar{f} = \sum f / n` the window means.  Interior-identical
    (fp64) to :func:`jax.grad` of ``lncc(..., reduction="sum")`` -- but
    without differentiating through the convolutions, so it is cheaper
    and materialises no autodiff tape.  (The boundary differs by the box
    filter's non-self-adjoint reflect pad, as ANTs' valid-neighbourhood
    trim does; the interior is what drives the registration.)

    Parameters
    ----------
    moving, fixed
        Images of identical shape, ``(..., *spatial)``.  Leading axes are
        treated as batch; the gradient is taken with respect to
        ``moving``.
    radius
        Window half-width :math:`r`, giving a box of size
        :math:`2r + 1` per axis.  An ``int`` gives an isotropic window;
        a sequence gives one radius per spatial axis.
    spatial_rank
        Number of trailing axes to treat as spatial.  ``None`` (default)
        infers from ``radius`` (if a sequence) else assumes all of
        ``moving.ndim``.
    mode
        Boundary mode for the windowed sums.  Default ``"reflect"``,
        which keeps the window count uniform.
    eps
        Denominator guard for flat (zero-variance) windows.

    Returns
    -------
    Float[Array, '... *spatial']
        The per-voxel local cross-correlation force, of the same shape as
        ``moving``.  This is the gradient of the ``"sum"`` reduction of
        :func:`lncc` (the natural per-voxel force magnitude); there is no
        ``reduction`` argument.
    """
    if isinstance(radius, (tuple, list)):
        inferred_rank: Optional[int] = len(radius)
    else:
        inferred_rank = None
    if spatial_rank is None:
        spatial_rank = (
            inferred_rank if inferred_rank is not None else moving.ndim
        )
    elif inferred_rank is not None and inferred_rank != spatial_rank:
        raise ValueError(
            f'radius has {inferred_rank} elements but spatial_rank='
            f'{spatial_rank}.'
        )
    radii = _normalise_radius(radius, spatial_rank)
    sizes = tuple(2 * r + 1 for r in radii)
    spatial_axes = tuple(range(moving.ndim - spatial_rank, moving.ndim))
    n = 1.0
    for s in sizes:
        n *= float(s)

    sum_m = _box_sum(moving, sizes, spatial_axes, mode)
    sum_f = _box_sum(fixed, sizes, spatial_axes, mode)
    sum_mm = _box_sum(moving * moving, sizes, spatial_axes, mode)
    sum_ff = _box_sum(fixed * fixed, sizes, spatial_axes, mode)
    sum_mf = _box_sum(moving * fixed, sizes, spatial_axes, mode)

    cross = sum_mf - sum_m * sum_f / n
    var_m = sum_mm - sum_m * sum_m / n
    var_f = sum_ff - sum_f * sum_f / n
    d = var_m * var_f + eps
    p = 2.0 * cross / d
    q = -2.0 * cross * cross * var_f / (d * d)
    mbar = sum_m / n
    fbar = sum_f / n
    return (
        fixed * _box_sum(p, sizes, spatial_axes, mode)
        - _box_sum(p * fbar, sizes, spatial_axes, mode)
        + moving * _box_sum(q, sizes, spatial_axes, mode)
        - _box_sum(q * mbar, sizes, spatial_axes, mode)
    )


def lncc_grad_center(
    moving: Float[Array, '... *spatial'],
    fixed: Float[Array, '... *spatial'],
    *,
    radius: Union[int, Sequence[int]] = 4,
    spatial_rank: Optional[int] = None,
    mode: BoundaryMode = 'reflect',
    eps: float = 1e-5,
) -> Float[Array, '... *spatial']:
    r"""Center-only local-cross-correlation force (the ANTs / ITK convention).

    The per-voxel ascent direction ITK's ``ANTSNeighborhoodCorrelation`` metric
    uses: each window's local CC derivative is attributed to its **window centre
    only**, so -- unlike the exact :func:`lncc_grad` (which back-propagates every
    window's CC to all of its members, the extra four box sums of its second
    pass) -- it needs only the **five** windowed sums

    .. code-block:: text

        sFF = Σf² - (Σf)²/n,   sMM = Σm² - (Σm)²/n,   sFM = Σfm - ΣfΣm/n
        fA  = f - Σf/n,        mA  = m - Σm/n        (centre minus window mean)
        grad = 2·sFM/(sFF·sMM) · (fA - (sFM/sMM)·mA)

    zeroed where :math:`s_{FF}` or :math:`s_{MM}` is :math:`\le` ``eps``
    (a flat window -- ITK's guard).  Multiply by the spatial gradient of
    ``moving`` for the deformation force (as
    :class:`~nitrix.register.LNCCForce` does).  This is a *different*,
    cheaper force than :func:`lncc_grad` (not its gradient): the
    centre-only approximation ANTs ships, ~5/9 the box-sum work and a
    single windowed pass (so it admits the sliding-window kernel the
    exact two-pass form cannot).  The *value* :func:`lncc` is unchanged
    -- only the derivative convention differs.

    Parameters
    ----------
    moving, fixed
        Images of identical shape, ``(..., *spatial)``.  Leading axes are
        treated as batch; the force is attributed to ``moving``.
    radius
        Window half-width :math:`r`, giving a box of size
        :math:`2r + 1` per axis.  An ``int`` gives an isotropic window;
        a sequence gives one radius per spatial axis.
    spatial_rank
        Number of trailing axes to treat as spatial.  ``None`` (default)
        infers from ``radius`` (if a sequence) else assumes all of
        ``moving.ndim``.
    mode
        Boundary mode for the windowed sums.  Default ``"reflect"``,
        which keeps the window count uniform.
    eps
        Flat-window threshold: the force is zeroed wherever the fixed- or
        moving-image window variance does not exceed this value.

    Returns
    -------
    Float[Array, '... *spatial']
        The per-voxel centre-only local cross-correlation force, of the
        same shape as ``moving``, zero on flat windows.
    """
    if isinstance(radius, (tuple, list)):
        inferred_rank: Optional[int] = len(radius)
    else:
        inferred_rank = None
    if spatial_rank is None:
        spatial_rank = (
            inferred_rank if inferred_rank is not None else moving.ndim
        )
    elif inferred_rank is not None and inferred_rank != spatial_rank:
        raise ValueError(
            f'radius has {inferred_rank} elements but spatial_rank='
            f'{spatial_rank}.'
        )
    radii = _normalise_radius(radius, spatial_rank)
    sizes = tuple(2 * r + 1 for r in radii)
    spatial_axes = tuple(range(moving.ndim - spatial_rank, moving.ndim))
    n = 1.0
    for s in sizes:
        n *= float(s)

    sum_m = _box_sum(moving, sizes, spatial_axes, mode)
    sum_f = _box_sum(fixed, sizes, spatial_axes, mode)
    sum_mm = _box_sum(moving * moving, sizes, spatial_axes, mode)
    sum_ff = _box_sum(fixed * fixed, sizes, spatial_axes, mode)
    sum_mf = _box_sum(moving * fixed, sizes, spatial_axes, mode)

    s_ff = sum_ff - sum_f * sum_f / n
    s_mm = sum_mm - sum_m * sum_m / n
    s_fm = sum_mf - sum_f * sum_m / n
    f_a = fixed - sum_f / n
    m_a = moving - sum_m / n
    # ITK guard: zero force on a flat window; the double-``where`` keeps the
    # gradient finite (a bare divide NaNs the backward even where masked).
    safe = (s_ff > eps) & (s_mm > eps)
    denom = jnp.where(safe, s_ff * s_mm, 1.0)
    s_mm_safe = jnp.where(safe, s_mm, 1.0)
    grad = 2.0 * s_fm / denom * (f_a - s_fm / s_mm_safe * m_a)
    return jnp.where(safe, grad, 0.0)
