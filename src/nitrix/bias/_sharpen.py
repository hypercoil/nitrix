# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
N3 / N4 intensity-histogram sharpening (Wiener log-histogram deconvolution).

This is the deconvolution half of N4 (Tustison 2010), inherited essentially
unchanged from N3 (Sled 1998).  The generative model is that a bias field
acts, in the log domain, as additive low-frequency noise -- which blurs the
*intensity histogram* by an (approximately Gaussian) point-spread.  The
sharpening step undoes that blur:

1. Build a 1-D histogram of the (log) intensities over the masked region,
   with triangular (Parzen) windowing between adjacent bins.
2. Treat the histogram ``V`` as the true intensity density ``U`` convolved
   with a Gaussian ``G`` of the given full-width-at-half-maximum.  Recover
   ``U`` by **Wiener deconvolution**: ``Uf = Vf * conj(Gf) / (|Gf|^2 + Z)``.
3. Form the conditional expectation map ``E[u | v] = ((c U) * G)(v) /
   ((U) * G)(v)`` -- the expected true intensity given an observed one --
   where ``c`` is the bin-centre intensity.
4. Remap every voxel through ``E`` by linear interpolation.

The result is an image whose intensity histogram is sharper (tissue peaks
de-blurred); the *difference* between the input and this sharpened image is
the bias-field residual that N4 then smooths with a B-spline (see
``_bspline.bspline_approximate`` and ``n4.n4_bias_field_correction``).

The FFT sizes are static (a power of two derived from ``n_bins``), so the
op JITs cleanly.  The histogram binning is piecewise-constant, so this op
is **not** differentiable through the bin assignment (the bias-field
*smoothing* is differentiable; the sharpening is not -- see
``docs/design/bias-field.md``).  Conventions (bin count, FWHM, Wiener
noise, the power-of-two zero-pad, the Parzen split) mirror ITK's
``N4BiasFieldCorrectionImageFilter::SharpenImage`` for reference parity.
"""

from __future__ import annotations

import math
from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = ['sharpen_histogram']


def _padded_fft_size(n_bins: int) -> tuple[int, int]:
    """Power-of-two FFT length and the centring offset (ITK convention).

    ``exponent = ceil(log2(n_bins)) + 1``; the histogram is centred in the
    padded buffer with ``offset = floor((padded - n_bins) / 2)``.
    """
    exponent = math.ceil(math.log2(n_bins)) + 1
    padded = int(2**exponent)
    offset = (padded - n_bins) // 2
    return padded, offset


def sharpen_histogram(
    image: Float[Array, '... *spatial'],
    *,
    weight: Optional[Float[Array, '... *spatial']] = None,
    n_bins: int = 200,
    fwhm: float = 0.15,
    wiener_noise: float = 0.01,
) -> Float[Array, '... *spatial']:
    """Sharpen an intensity image by Wiener deconvolution of its histogram.

    The N3 / N4 histogram-deconvolution step.  In N4 this is applied to the
    *log* of the (partially corrected) image; it is exposed standalone
    because the operation -- "deblur an intensity distribution under a
    Gaussian-spread model and remap" -- is a reusable primitive.

    The entire input is treated as a single intensity population (one
    histogram).  For independent per-volume sharpening of a batch,
    ``jax.vmap`` over the batch axis.

    Parameters
    ----------
    image
        Intensities to sharpen, ``(..., *spatial)``.  In N4 this is the log
        of the current bias-corrected image.
    weight
        Optional confidence / mask, same broadcastable shape as ``image``.
        Only voxels with non-zero weight contribute to the histogram (and
        define its intensity range).  ``None`` uses every voxel.
    n_bins
        Number of histogram bins.  Default ``200`` (ITK default).
    fwhm
        Full-width-at-half-maximum of the assumed Gaussian intensity spread,
        in the same units as ``image`` (log-intensity units in N4).  Default
        ``0.15`` (ITK default).  Larger -> more aggressive sharpening.
    wiener_noise
        Wiener-filter regularisation (the noise-to-signal floor that keeps
        the deconvolution stable).  Default ``0.01`` (ITK default).

    Returns
    -------
    The sharpened image, same shape as ``image``.  Voxels outside the mask
    are remapped through the same lookup (their values are immaterial in N4,
    where the subsequent fit is mask-weighted).

    Notes
    -----
    Not differentiable through the histogram binning (piecewise constant).
    The FFT length is a static power of two, so the op JITs cleanly.
    """
    x = jnp.asarray(image)
    dtype = jnp.result_type(x.dtype, jnp.float32)
    x = x.astype(dtype)
    if weight is None:
        w = jnp.ones_like(x)
    else:
        w = jnp.broadcast_to(jnp.asarray(weight).astype(dtype), x.shape)

    flat = x.reshape(-1)
    wflat = w.reshape(-1)
    masked = wflat > 0

    # Intensity range over the masked region.
    bin_min = jnp.min(jnp.where(masked, flat, jnp.inf))
    bin_max = jnp.max(jnp.where(masked, flat, -jnp.inf))
    slope = (bin_max - bin_min) / (n_bins - 1)
    slope = jnp.where(slope > 0, slope, 1.0)  # guard constant image

    # Continuous bin index, with triangular (Parzen) split into idx, idx+1.
    cidx = (flat - bin_min) / slope
    idx = jnp.floor(cidx).astype(jnp.int32)
    frac = cidx - idx.astype(dtype)
    idx0 = jnp.clip(idx, 0, n_bins - 1)
    idx1 = jnp.clip(idx + 1, 0, n_bins - 1)

    hist = jnp.zeros((n_bins,), dtype=dtype)
    hist = hist.at[idx0].add(wflat * (1.0 - frac))
    hist = hist.at[idx1].add(wflat * frac)

    padded, pad_offset = _padded_fft_size(n_bins)

    V = jnp.zeros((padded,), dtype=jnp.complex64)
    V = V.at[pad_offset : pad_offset + n_bins].set(hist.astype(jnp.complex64))
    Vf = jnp.fft.fft(V)

    # scaledFWHM = fwhm / slope; the kernel FFT is built with the *traced*
    # slope (data-dependent), so it cannot be precomputed at a static width.
    ln2 = math.log(2.0)
    scaled_fwhm_arr = fwhm / slope
    exp_factor = 4.0 * ln2 / (scaled_fwhm_arr**2)
    scale_factor = 2.0 * jnp.sqrt(ln2 / math.pi) / scaled_fwhm_arr
    i = jnp.arange(padded, dtype=dtype)
    d = jnp.minimum(i, padded - i)
    kernel = scale_factor * jnp.exp(-(d**2) * exp_factor)
    Gf = jnp.real(jnp.fft.fft(kernel.astype(jnp.complex64)))

    # Wiener deconvolution: Gf is real (kernel is symmetric under wrap).
    Uf = Vf * Gf / (Gf**2 + wiener_noise)
    U = jnp.real(jnp.fft.ifft(Uf))
    U = jnp.clip(U, 0.0, None)

    # Conditional-expectation map E[u | v] = ((c U) * G) / ((U) * G).
    bin_centres = (
        bin_min + (jnp.arange(padded, dtype=dtype) - pad_offset) * slope
    )
    num = jnp.real(jnp.fft.ifft(jnp.fft.fft(bin_centres * U) * Gf))
    den = jnp.real(jnp.fft.ifft(jnp.fft.fft(U) * Gf))
    E = jnp.where(jnp.abs(den) > 1e-10, num / den, 0.0)

    # Remap every voxel through E by linear interpolation (Parzen inverse).
    e0 = E[jnp.clip(idx0 + pad_offset, 0, padded - 1)]
    e1 = E[jnp.clip(idx1 + pad_offset, 0, padded - 1)]
    sharp = e0 * (1.0 - frac) + e1 * frac
    return sharp.reshape(x.shape)
