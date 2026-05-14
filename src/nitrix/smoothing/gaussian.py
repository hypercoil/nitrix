# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Separable n-D Gaussian filter.

Per SPEC_UPDATE §3.3, the unconditional baseline smoother.  Pure
JAX -- uses ``lax.conv_general_dilated`` for the underlying 1D
convolutions (since the algebra is REAL and we want the tensor-core
fast path for free).  The n-D Gaussian factors exactly as a product
of 1D Gaussians, so we convolve along each spatial axis serially.

Padding modes match ``scipy.ndimage.gaussian_filter``: default
``"reflect"`` (the medical-imaging convention -- preserves boundary
intensities), with ``"constant"`` and ``"edge"`` also supported.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float


__all__ = ['gaussian']


def _gaussian_1d_kernel(sigma: float, truncate: float, dtype) -> Array:
    '''Build a 1D Gaussian kernel of standard width.

    Kernel half-width is ``ceil(truncate * sigma)``; total width is
    ``2 * half + 1``.  Values are normalised so they sum to 1, so a
    constant input is preserved.
    '''
    if sigma <= 0:
        raise ValueError(f'sigma must be positive; got {sigma!r}.')
    # Match ``scipy.ndimage.gaussian_filter1d``: half-width is
    # ``int(truncate * sigma + 0.5)``, not ``ceil(truncate * sigma)``.
    # This rounds 4 * 0.8 = 3.2 down to 3, where ceil would give 4.
    half = int(truncate * sigma + 0.5)
    x = jnp.arange(-half, half + 1, dtype=dtype)
    kernel = jnp.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def _conv_1d_along_axis(
    x: Array, kernel: Array, axis: int, mode: str,
) -> Array:
    '''Convolve ``x`` with a 1D ``kernel`` along ``axis``.

    Boundary handling: ``mode`` controls how the input is padded
    before the (otherwise VALID) convolution.  Supports
    ``"reflect"``, ``"constant"`` (cval = 0), and ``"edge"``.
    '''
    half = (kernel.size - 1) // 2
    pad_widths = [(0, 0)] * x.ndim
    pad_widths[axis] = (half, half)
    # ``scipy.ndimage`` and numpy disagree on what "reflect" means:
    #   scipy: include the boundary in the mirror -- (b, a, | a, b, c, ...)
    #   numpy: exclude it                          -- (c, b, | a, b, c, ...)
    # We follow scipy convention (medical-imaging usage), which is
    # numpy's "symmetric" mode.  ``mirror`` is the alias for numpy's
    # "reflect" semantic if a user really wants it.
    if mode == 'reflect':
        x_padded = jnp.pad(x, pad_widths, mode='symmetric')
    elif mode == 'mirror':
        x_padded = jnp.pad(x, pad_widths, mode='reflect')
    elif mode == 'constant':
        x_padded = jnp.pad(x, pad_widths, mode='constant', constant_values=0)
    elif mode == 'edge':
        x_padded = jnp.pad(x, pad_widths, mode='edge')
    else:
        raise ValueError(
            f'mode={mode!r}; expected one of "reflect", "mirror", '
            '"constant", "edge".'
        )
    # Reshape the 1D kernel to a conv kernel: move ``axis`` to last
    # via ``moveaxis``, flatten the rest as batch, conv1d, reshape back.
    x_moved = jnp.moveaxis(x_padded, axis, -1)
    batch_shape = x_moved.shape[:-1]
    L = x_moved.shape[-1]
    x_flat = x_moved.reshape(-1, 1, L)                        # (B*, 1, L)
    k = kernel.reshape(1, 1, kernel.size)                     # (C_out=1, C_in=1, K)
    out = lax.conv_general_dilated(
        x_flat, k,
        window_strides=(1,),
        padding='VALID',
    )
    out = out.reshape(*batch_shape, out.shape[-1])
    return jnp.moveaxis(out, -1, axis)


def _normalise_sigma(
    sigma: Union[float, Sequence[float]], spatial_rank: int,
) -> tuple[float, ...]:
    if isinstance(sigma, (int, float)):
        return (float(sigma),) * spatial_rank
    out = tuple(float(s) for s in sigma)
    if len(out) != spatial_rank:
        raise ValueError(
            f'sigma must be a scalar or a length-{spatial_rank} '
            f'sequence; got {sigma!r}.'
        )
    return out


def gaussian(
    x: Float[Array, '... *spatial'],
    *,
    sigma: Union[float, Sequence[float]],
    truncate: float = 4.0,
    mode: str = 'reflect',
    spatial_rank: Optional[int] = None,
) -> Float[Array, '... *spatial']:
    '''Separable n-D Gaussian smoothing.

    Parameters
    ----------
    x
        Single-channel input, ``(..., *spatial)``.  *All* dims are
        treated as spatial unless ``spatial_rank`` is given or
        ``sigma`` is a sequence pinning the rank.
    sigma
        Standard deviation of the Gaussian.  ``float`` -- same
        sigma along every spatial axis.  Sequence -- per-axis
        sigma, for anisotropic Gaussians (e.g. fMRI with
        anisotropic voxel spacing).
    truncate
        Kernel half-width factor: kernel is ``2 * ceil(truncate *
        sigma) + 1`` taps.  Default ``4`` (matches
        ``scipy.ndimage.gaussian_filter``).
    mode
        Boundary handling: ``"reflect"`` (default), ``"constant"``
        (zero-pad), or ``"edge"`` (replicate boundary).
    spatial_rank
        Explicit number of trailing dims to treat as spatial.  If
        ``None`` (default), inferred from ``sigma`` (if sequence)
        or assumed to be ``x.ndim`` (if scalar).

    Returns
    -------
    Smoothed array of the same shape as ``x``.

    Notes
    -----
    The Gaussian kernel is normalised to sum to 1 per axis, so the
    overall n-D kernel sums to 1 and a constant input is preserved.
    For per-axis sigma = 0 the spec is undefined; we raise.  If you
    want to skip smoothing along an axis, omit it from ``spatial_rank``
    or use a tiny sigma plus large ``truncate``.

    The implementation uses ``lax.conv_general_dilated`` for each 1D
    pass, so on Ampere+ tensor cores accelerate the REAL convolution.
    No tensor-core gap to recover here -- gaussian is not a semiring
    op.
    '''
    if isinstance(sigma, (tuple, list)):
        inferred_rank = len(sigma)
    else:
        inferred_rank = None
    if spatial_rank is None:
        spatial_rank = inferred_rank if inferred_rank is not None else x.ndim
    elif inferred_rank is not None and inferred_rank != spatial_rank:
        raise ValueError(
            f'sigma has {inferred_rank} elements but spatial_rank='
            f'{spatial_rank}.'
        )
    if spatial_rank < 1:
        raise ValueError('spatial_rank must be >= 1.')
    if x.ndim < spatial_rank:
        raise ValueError(
            f'x.ndim={x.ndim} too small for spatial_rank={spatial_rank}.'
        )

    sigmas = _normalise_sigma(sigma, spatial_rank)
    spatial_axes = tuple(range(x.ndim - spatial_rank, x.ndim))

    out = x
    for axis, s in zip(spatial_axes, sigmas):
        kernel = _gaussian_1d_kernel(s, truncate, dtype=out.dtype)
        out = _conv_1d_along_axis(out, kernel, axis=axis, mode=mode)
    return out
