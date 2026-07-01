# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Multi-resolution Gaussian pyramid.

Coarse-to-fine registration warm-starts each resolution from the
solution of the coarser one above it.  :func:`gaussian_pyramid` builds the
image stack; :func:`downsample` and :func:`upsample` are the single-step
anti-aliased resize and the (interpolating) prolongation.

Composition only -- no new kernel: :func:`downsample` is
:func:`gaussian` (anti-alias) followed by :func:`resample`
(align-corners resize); :func:`upsample` is :func:`resample` to a
larger shape.  Channel-last image layout ``(*spatial, c)`` matches
:func:`resample` and :func:`spatial_transform`.
"""

from __future__ import annotations

from typing import Sequence, Union

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..smoothing.gaussian import gaussian
from ._interpolate import Interpolator, Linear
from .grid import resample

__all__ = ['downsample', 'upsample', 'gaussian_pyramid']


def _smooth_spatial(
    image: Float[Array, '*spatial c'],
    sigma: Union[float, Sequence[float]],
    mode: str,
) -> Float[Array, '*spatial c']:
    """Separable Gaussian over the spatial axes of a channel-last image.

    :func:`gaussian` treats trailing axes as spatial, so the
    channel axis is moved to the front (a leading batch axis), smoothed,
    and moved back.

    Parameters
    ----------
    image
        Channel-last image, ``(*spatial, c)``.
    sigma
        Standard deviation of the smoothing Gaussian, in voxels. A scalar
        applies isotropically; a sequence gives one value per spatial axis.
    mode
        Boundary mode passed through to :func:`gaussian`.

    Returns
    -------
    Float[Array, '*spatial c']
        The smoothed image, same shape as ``image``.
    """
    spatial_rank = image.ndim - 1
    moved = jnp.moveaxis(image, -1, 0)
    smoothed = gaussian(
        moved,
        sigma=sigma,
        mode=mode,
        spatial_rank=spatial_rank,
    )
    return jnp.moveaxis(smoothed, 0, -1)


def downsample(
    image: Float[Array, '*spatial c'],
    *,
    factor: float = 2.0,
    sigma: Union[None, float, Sequence[float]] = None,
    mode: str = 'reflect',
    method: Interpolator = Linear(),
) -> Float[Array, '*target c']:
    """Anti-aliased downsample of a channel-last image by ``factor``.

    Smooths with a Gaussian (anti-alias) then resamples to
    ``round(spatial / factor)`` via the align-corners :func:`resample`
    grid.

    Parameters
    ----------
    image
        Channel-last image, ``(*spatial, c)``.
    factor
        Per-axis downscale factor (> 1 shrinks).  Default ``2``.
    sigma
        Anti-alias Gaussian standard deviation.  ``None`` (default) uses
        :math:`factor / 2` per axis (a standard heuristic: the cutoff that
        suppresses the frequencies the coarser grid cannot represent).
        Pass a float or sequence to override.
    mode
        Boundary mode for the anti-alias Gaussian (:func:`gaussian`
        vocabulary); default ``'reflect'``.
    method
        Interpolation kernel for the resize (an :class:`Interpolator`).
        Default :class:`Linear`.

    Returns
    -------
    Float[Array, '*target c']
        Downsampled image, ``(*target, c)`` with
        ``target[d] = max(1, round(spatial[d] / factor))``.
    """
    if factor <= 0:
        raise ValueError(f'factor must be positive; got {factor!r}.')
    spatial = image.shape[:-1]
    sig: Union[float, Sequence[float]] = (
        factor / 2.0 if sigma is None else sigma
    )
    smoothed = _smooth_spatial(image, sig, mode)
    target = tuple(max(1, int(round(s / factor))) for s in spatial)
    return resample(smoothed, target, method=method)


def upsample(
    image: Float[Array, '*spatial c'],
    target_shape: Sequence[int],
    *,
    method: Interpolator = Linear(),
    mode: str = 'constant',
    cval: float = 0.0,
) -> Float[Array, '*target c']:
    """Resample a channel-last image up to ``target_shape``.

    A thin wrapper over :func:`resample` (align-corners) for the
    coarse-to-fine prolongation step.  No anti-alias is needed going up.

    Parameters
    ----------
    image
        Channel-last image, ``(*spatial, c)``.
    target_shape
        Desired spatial shape ``(*target,)`` -- one size per spatial axis.
        The channel axis is left unchanged.
    method
        Interpolation kernel for the resize (an :class:`Interpolator`).
        Default :class:`Linear`.
    mode
        Boundary mode for out-of-bounds samples, passed through to
        :func:`resample`; default ``'constant'``.
    cval
        Fill value used where ``mode='constant'`` samples outside the
        image.  Default ``0.0``.

    Returns
    -------
    Float[Array, '*target c']
        The resized image, ``(*target, c)``.

    Notes
    -----
    When the image being upsampled is a *displacement / velocity field*
    (channels are spatial offsets in voxel units), the resize changes
    the voxel scale, so the field **values** must additionally be scaled
    by the per-axis size ratio ``target[d] / spatial[d]``.  ``upsample``
    only resizes the array; the registration driver applies that scaling
    explicitly (it is a property of the field semantics, not of the
    resampler).
    """
    return resample(
        image,
        target_shape,
        method=method,
        mode=mode,  # type: ignore[arg-type]
        cval=cval,
    )


def gaussian_pyramid(
    image: Float[Array, '*spatial c'],
    *,
    levels: int,
    factor: float = 2.0,
    sigma: Union[None, float, Sequence[float]] = None,
    mode: str = 'reflect',
    method: Interpolator = Linear(),
) -> tuple[Float[Array, '*spatial c'], ...]:
    """Build a ``levels``-deep anti-aliased Gaussian pyramid.

    Parameters
    ----------
    image
        Channel-last image, ``(*spatial, c)`` -- the finest level.
    levels
        Number of levels including the original (``>= 1``).
    factor, sigma, mode, method
        Forwarded to :func:`downsample` for each coarsening step.

    Returns
    -------
    A tuple of ``levels`` images, **finest first**: ``out[0]`` is
    ``image`` at full resolution and ``out[levels - 1]`` is the
    coarsest.  Coarse-to-fine registration iterates ``reversed(out)``.
    """
    if levels < 1:
        raise ValueError(f'levels must be >= 1; got {levels!r}.')
    pyramid = [image]
    current = image
    for _ in range(levels - 1):
        current = downsample(
            current,
            factor=factor,
            sigma=sigma,
            mode=mode,
            method=method,
        )
        pyramid.append(current)
    return tuple(pyramid)
