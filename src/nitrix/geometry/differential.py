# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image spatial-gradient operators.

The directional derivatives :math:`\\partial I / \\partial x_d` of a
scalar field along each of its spatial axes -- the gradient
:math:`\\nabla M` that drives both registration recipes (the rigid
Gauss-Newton steepest-descent images :math:`\\nabla M \\cdot (G_j x)`
and the diffeomorphic-Demons symmetric force
:math:`\\tfrac{1}{2}(\\nabla F + \\nabla(M \\circ \\varphi))`).

Three separable schemes, all differentiable and all reusing one
cross-correlation engine:

- ``"central"`` (default) -- the bare central difference
  :math:`(I[+1] - I[-1]) / (2 h)`.  Generalises the private
  central-difference helper used by :func:`jacobian_displacement`.
- ``"sobel"`` -- central difference along the derivative axis,
  :math:`[1, 2, 1] / 4` smoothing along every other spatial axis
  (noise-robust).
- ``"scharr"`` -- as Sobel with :math:`[3, 10, 3] / 16` smoothing
  (better rotational symmetry).

Layout follows the Gaussian smoothing operators: ``(..., *spatial)``
with the trailing ``spatial_rank`` axes spatial.  The result appends a
trailing derivative-direction axis: ``(..., *spatial, ndim)``.
"""

from __future__ import annotations

from typing import Literal, Sequence, Union

import jax.numpy as jnp
from jaxtyping import Array, Float

from .._internal.separable import SeparableBoundaryMode, correlate1d

__all__ = ['spatial_gradient']

GradientScheme = Literal['central', 'sobel', 'scharr']
# The boundary-mode vocabulary and the cross-correlation engine are
# shared with the other separable spatial operators (the LNCC box sums
# in ``nitrix.metrics``); both live in ``nitrix._internal.separable``.
BoundaryMode = SeparableBoundaryMode

# Per-axis smoothing kernels for the smoothed schemes (normalised to
# sum 1 so the operator estimates a true derivative, not a scaled one).
_SMOOTH_KERNEL: dict[str, tuple[float, ...]] = {
    'sobel': (1.0, 2.0, 1.0),
    'scharr': (3.0, 10.0, 3.0),
}


def _normalise_spacing(
    spacing: Union[float, Sequence[float]],
    spatial_rank: int,
) -> tuple[float, ...]:
    if isinstance(spacing, (int, float)):
        return (float(spacing),) * spatial_rank
    out = tuple(float(s) for s in spacing)
    if len(out) != spatial_rank:
        raise ValueError(
            f'spacing must be a scalar or a length-{spatial_rank} '
            f'sequence; got {spacing!r}.'
        )
    return out


def spatial_gradient(
    image: Float[Array, '... *spatial'],
    *,
    spatial_rank: int | None = None,
    scheme: GradientScheme = 'central',
    mode: BoundaryMode = 'nearest',
    spacing: Union[float, Sequence[float]] = 1.0,
) -> Float[Array, '... *spatial ndim']:
    """Spatial gradient of a scalar field along each spatial axis.

    Parameters
    ----------
    image
        Scalar field, ``(..., *spatial)``.  All trailing axes are
        spatial unless ``spatial_rank`` (or a sequence ``spacing``)
        pins the rank; any leading axes are batch.  For a multi-channel
        image move the channel axis to the front (a leading batch axis)
        before calling.
    spatial_rank
        Number of trailing axes to treat as spatial.  ``None``
        (default) infers from ``spacing`` (if a sequence) else assumes
        ``image.ndim``.
    scheme
        Derivative scheme: ``"central"`` (default), ``"sobel"``, or
        ``"scharr"``.  ``"central"`` is the bare central difference;
        the others smooth along the off-axes for noise robustness.
    mode
        Boundary handling: ``"nearest"`` (default, edge-replicate --
        the voxelmorph / QA convention, matching
        :func:`jacobian_displacement`), ``"reflect"``, ``"mirror"``,
        ``"wrap"``, or ``"constant"`` (zero-pad).
    spacing
        Voxel spacing per spatial axis (physical units).  ``float`` ->
        isotropic; sequence -> per-axis (length ``spatial_rank``).  The
        central-difference denominator is ``2 * spacing[axis]``.

    Returns
    -------
    Gradient field, ``(..., *spatial, ndim)`` with ``ndim ==
    spatial_rank``.  The trailing axis indexes the derivative
    direction; ``out[..., d]`` is
    :math:`\\partial\\,\\mathrm{image} / \\partial x_d`.

    Notes
    -----
    Exact on affine intensity ramps (central differences reproduce the
    analytic slope on a linear field, interior and -- under
    ``"nearest"`` -- boundary).  Differentiable w.r.t. ``image`` (the
    operator is linear), which is what lets it sit inside a
    registration loss.
    """
    if isinstance(spacing, (tuple, list)):
        inferred_rank: int | None = len(spacing)
    else:
        inferred_rank = None
    if spatial_rank is None:
        spatial_rank = (
            inferred_rank if inferred_rank is not None else image.ndim
        )
    elif inferred_rank is not None and inferred_rank != spatial_rank:
        raise ValueError(
            f'spacing has {inferred_rank} elements but spatial_rank='
            f'{spatial_rank}.'
        )
    if spatial_rank < 1:
        raise ValueError('spatial_rank must be >= 1.')
    if image.ndim < spatial_rank:
        raise ValueError(
            f'image.ndim={image.ndim} too small for spatial_rank='
            f'{spatial_rank}.'
        )
    if scheme not in ('central', 'sobel', 'scharr'):
        raise ValueError(
            f'scheme={scheme!r}; expected "central", "sobel", or "scharr".'
        )

    spacings = _normalise_spacing(spacing, spatial_rank)
    spatial_axes = tuple(range(image.ndim - spatial_rank, image.ndim))
    smooth_taps = _SMOOTH_KERNEL.get(scheme)

    cols = []
    for i, ax_i in enumerate(spatial_axes):
        deriv = jnp.array([-1.0, 0.0, 1.0], dtype=image.dtype) / (
            2.0 * spacings[i]
        )
        g = correlate1d(image, deriv, axis=ax_i, mode=mode)
        if smooth_taps is not None:
            smooth = jnp.asarray(smooth_taps, dtype=image.dtype)
            smooth = smooth / smooth.sum()
            for ax_j in spatial_axes:
                if ax_j == ax_i:
                    continue
                g = correlate1d(g, smooth, axis=ax_j, mode=mode)
        cols.append(g)
    return jnp.stack(cols, axis=-1)
