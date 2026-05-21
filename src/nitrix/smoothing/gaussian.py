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
from jax.typing import DTypeLike
from jaxtyping import Array, Float


__all__ = ['gaussian']


def _gaussian_1d_kernel(
    sigma: float,
    truncate: float,
    dtype: DTypeLike,
    *,
    kernel_size: Optional[int] = None,
) -> Array:
    '''Build a 1D Gaussian kernel.

    If ``kernel_size`` is ``None`` (default), the kernel half-width
    is ``int(truncate * sigma + 0.5)`` (scipy convention) and the
    total width is ``2 * half + 1`` (always odd).

    If ``kernel_size`` is an integer, the kernel has exactly that
    many taps -- overriding the ``truncate`` heuristic.  Odd
    ``kernel_size`` gives a symmetric kernel centred on the
    middle tap (taps at ``-half ... 0 ... +half``).  Even
    ``kernel_size`` gives an off-centre kernel with taps at
    half-integer offsets relative to the original pixel grid
    (e.g. ``kernel_size=2`` -> taps at ``-0.5, 0.5``), which
    shifts the output by half a pixel.

    Values are normalised so the kernel sums to 1 -- a constant
    input is preserved.
    '''
    if sigma <= 0:
        raise ValueError(f'sigma must be positive; got {sigma!r}.')
    if kernel_size is None:
        # Scipy convention.
        half = int(truncate * sigma + 0.5)
        x = jnp.arange(-half, half + 1, dtype=dtype)
    else:
        if kernel_size < 1:
            raise ValueError(
                f'kernel_size must be >= 1; got {kernel_size!r}.'
            )
        # For odd kernel_size N=2h+1, taps are at -h..+h.
        # For even kernel_size N=2h, taps are at -h+0.5..+h-0.5
        # (so the kernel is offset by half a pixel; the output is
        # half-pixel-shifted).
        if kernel_size % 2 == 1:
            half = (kernel_size - 1) // 2
            x = jnp.arange(-half, half + 1, dtype=dtype)
        else:
            half = kernel_size // 2
            x = jnp.arange(-half + 0.5, half, dtype=dtype)
            # length should equal kernel_size
            assert x.shape[0] == kernel_size
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

    For an **odd** ``kernel.size``, the pad is symmetric and the
    output is centred on the same pixel grid as the input.

    For an **even** ``kernel.size``, the pad is asymmetric
    (``half_left = K // 2 - 1``, ``half_right = K // 2``) -- the
    minimum same-size pad.  The output is half-pixel-shifted
    relative to a centred-kernel convolution; this is the price
    of an even-tap Gaussian and is documented at the public API.
    '''
    K = kernel.size
    if K % 2 == 1:
        half = (K - 1) // 2
        half_l = half_r = half
    else:
        half_l = K // 2 - 1
        half_r = K // 2
    pad_widths = [(0, 0)] * x.ndim
    pad_widths[axis] = (half_l, half_r)
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


def _normalise_kernel_size(
    kernel_size: Union[None, int, Sequence[Optional[int]]],
    spatial_rank: int,
) -> tuple[Optional[int], ...]:
    if kernel_size is None:
        return (None,) * spatial_rank
    if isinstance(kernel_size, int):
        return (int(kernel_size),) * spatial_rank
    out = tuple(int(k) if k is not None else None for k in kernel_size)
    if len(out) != spatial_rank:
        raise ValueError(
            f'kernel_size must be int, None, or a length-'
            f'{spatial_rank} sequence; got {kernel_size!r}.'
        )
    return out


def gaussian(
    x: Float[Array, '... *spatial'],
    *,
    sigma: Union[float, Sequence[float]],
    truncate: float = 4.0,
    kernel_size: Union[None, int, Sequence[Optional[int]]] = None,
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
        Kernel half-width factor when ``kernel_size`` is ``None``:
        kernel is ``2 * int(truncate * sigma + 0.5) + 1`` taps
        (scipy convention).  Default ``4``.  Ignored if
        ``kernel_size`` is given.

        **Framework cross-references**: ``truncate=4`` matches
        ``scipy.ndimage.gaussian_filter`` (the scientific-computing
        default).  The **neurite / SynthMorph / SynthSeg /
        FreeSurfer-augmentation lineage** uses ``windowsize =
        round(sigma * 3) * 2 + 1`` taps, which corresponds to
        ``truncate=3``.  Pass ``truncate=3`` when porting any model
        from that lineage; otherwise the kernel bandwidth differs
        by 1 tap per axis at typical sigmas and the boundary
        cells silently disagree.
    kernel_size
        Explicit per-axis kernel size override.  ``None`` (default)
        uses the ``truncate * sigma`` heuristic.  ``int`` --
        same size on every spatial axis.  Sequence -- per-axis;
        ``None`` entries fall back to the heuristic.

        **Odd** kernel size: symmetric, centred on the pixel grid.
        Output is on the same pixel grid as the input.

        **Even** kernel size: off-centre Gaussian with taps at
        half-integer offsets (e.g. ``kernel_size=2`` -> taps at
        ``-0.5, 0.5``); **output is half-pixel-shifted along that
        axis**.  Use this only when you specifically want an
        even-tap Gaussian-weighted average (e.g. the
        spheremorph / JOSA NegativeJacobianFiltering 2×2 kernel
        at ``sigma=0.7``); otherwise prefer the default heuristic.
    mode
        Boundary handling: ``"reflect"`` (default), ``"constant"``
        (zero-pad), or ``"edge"`` (replicate boundary).

        **Framework cross-references**: ``mode='reflect'`` matches
        ``scipy.ndimage.gaussian_filter``.  Use ``mode='constant'``
        to match **TF Conv2D / Conv3D ``padding='SAME'``** semantics
        (zero-pad at the boundary).  Without this override,
        neurite-lineage models silently disagree at the boundary
        cells -- typically by ~0.7 logit-space units per pixel,
        which becomes ~6x magnitude differences after a downstream
        ``exp()`` (e.g., ExVivoNorm's multiplicative bias field).
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
    kernel_sizes = _normalise_kernel_size(kernel_size, spatial_rank)
    spatial_axes = tuple(range(x.ndim - spatial_rank, x.ndim))

    out = x
    for axis, s, ks in zip(spatial_axes, sigmas, kernel_sizes):
        kernel = _gaussian_1d_kernel(
            s, truncate, dtype=out.dtype, kernel_size=ks,
        )
        out = _conv_1d_along_axis(out, kernel, axis=axis, mode=mode)
    return out
