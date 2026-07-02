# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Intensity-domain augmentation transforms.

Image-only perturbations of the intensity histogram -- the building
blocks of contrast / appearance augmentation:

- :func:`gamma_contrast` -- a gamma (power-law) tone curve applied inside
  a normalised intensity bracket.  Deterministic in the gamma (draw the
  exponent in the caller); ``gamma < 1`` raises contrast, ``gamma > 1``
  lowers it.
- :func:`random_histogram_shift` -- a random, monotone, piecewise-linear
  remap of the intensity range through perturbed control points.
- :func:`gibbs_ringing` -- Gibbs (truncation) ringing from high-frequency
  k-space truncation, the artefact of a finitely-sampled acquisition.
  Deterministic in the truncation strength (draw it in the caller).
- :func:`gaussian_noise` / :func:`rician_noise` -- additive Gaussian
  noise and the Rician magnitude-noise model.

The noise generators take an explicit ``sigma`` (and a PRNG key, which
the draw is intrinsic to); a randomly-drawn ``sigma`` is one
``jax.random.uniform`` in the caller.  Everything is a pure function of
its inputs (and key), shape-agnostic, and differentiable in the image.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = [
    'gamma_contrast',
    'random_histogram_shift',
    'gibbs_ringing',
    'gaussian_noise',
    'rician_noise',
]

_EPS = 1e-8


def gamma_contrast(
    x: Float[Array, '...'],
    gamma: Float[Array, '...'],
    *,
    value_range: Optional[Tuple[float, float]] = None,
) -> Float[Array, '...']:
    """Apply a gamma tone curve within a normalised intensity bracket.

    Normalises ``x`` to ``[0, 1]`` over the bracket, raises it to
    ``gamma``, then maps back to the bracket::

        normed = clip((x - lo) / (hi - lo), 0, 1)
        out    = normed ** gamma * (hi - lo) + lo

    ``gamma < 1`` expands the dynamic range of dark values (higher
    contrast); ``gamma > 1`` compresses it.

    Parameters
    ----------
    x
        Intensity tensor.
    gamma
        Exponent (scalar or broadcastable to ``x``).  Drawing it
        randomly is the caller's concern (a single
        ``jax.random.uniform``).
    value_range
        The ``(lo, hi)`` bracket to normalise within.  ``None``
        (default) uses the per-tensor ``min`` / ``max``.

    Returns
    -------
    Tensor of the same shape as ``x``.
    """
    if value_range is None:
        lo = jnp.min(x)
        hi = jnp.max(x)
    else:
        lo = jnp.asarray(value_range[0], dtype=x.dtype)
        hi = jnp.asarray(value_range[1], dtype=x.dtype)
    span = jnp.maximum(hi - lo, jnp.asarray(_EPS, dtype=x.dtype))
    normed = jnp.clip((x - lo) / span, 0.0, 1.0)
    return normed**gamma * span + lo


def random_histogram_shift(
    x: Float[Array, '...'],
    key: Array,
    *,
    n_control_points: int = 10,
    shift_range: Tuple[float, float] = (-0.1, 0.1),
) -> Float[Array, '...']:
    """Random monotone piecewise-linear remap of the intensity range.

    Places ``n_control_points`` equally-spaced reference levels across
    ``[min(x), max(x)]``, perturbs each by a random offset drawn from
    ``shift_range`` (as a fraction of the intensity span), pins the two
    endpoints (so the global range is preserved), enforces monotonicity
    by a running maximum, then remaps every value through the resulting
    table by linear interpolation.

    Parameters
    ----------
    x
        Intensity tensor.
    key
        PRNG key.
    n_control_points
        Number of reference levels, including the two endpoints
        (``>= 2``).
    shift_range
        Per-control-point offset range, as a fraction of the intensity
        span.

    Returns
    -------
    Remapped tensor of the same shape as ``x``.
    """
    if n_control_points < 2:
        raise ValueError('n_control_points must be >= 2')
    lo = jnp.min(x)
    hi = jnp.max(x)
    span = jnp.maximum(hi - lo, jnp.asarray(_EPS, dtype=x.dtype))
    refs = jnp.linspace(lo, hi, n_control_points)
    shifts = jax.random.uniform(
        key,
        (n_control_points,),
        dtype=x.dtype,
        minval=shift_range[0] * span,
        maxval=shift_range[1] * span,
    )
    shifts = shifts.at[0].set(0.0).at[-1].set(0.0)
    shifted = jnp.maximum.accumulate(refs + shifts)
    return jnp.interp(x.ravel(), refs, shifted).reshape(x.shape)


def gibbs_ringing(
    x: Float[Array, '...'],
    alpha: float,
    *,
    axes: Optional[Sequence[int]] = None,
) -> Float[Array, '...']:
    """Inject Gibbs (truncation) ringing by truncating high-frequency k-space.

    Gibbs ringing is the oscillation that appears near sharp edges when a
    signal's spectrum is sharply truncated -- the artefact of a
    finitely-sampled (band-limited) acquisition.  This models it directly:
    transform to k-space, zero every frequency whose normalised radius
    exceeds ``1 - alpha`` (a hard spherical truncation -- the sharp cutoff
    *is* what produces the ringing), and transform back.

    Parameters
    ----------
    x
        Real-valued input.
    alpha
        Truncation strength in ``[0, 1]``.  ``0`` keeps the full spectrum
        (identity); larger values remove more high-frequency content and
        ring more strongly; ``1`` keeps only the DC component.  Deterministic
        in ``alpha`` -- draw it from a range in the caller for a random
        augmentation.
    axes
        Axes to transform over.  ``None`` (default) uses all axes; for a
        channels-last volume pass the spatial axes so the channel axis is
        not mixed into the transform.

    Returns
    -------
    Real array of the same shape as ``x``.
    """
    fft_axes = tuple(range(x.ndim)) if axes is None else tuple(axes)
    # k-space coordinate grid in (at least) the input's float precision.
    grid_dtype = jnp.result_type(x.dtype, jnp.float32)
    r2 = jnp.zeros((1,) * x.ndim, dtype=grid_dtype)
    for ax in fft_axes:
        n = x.shape[ax]
        coord = (jnp.arange(n, dtype=grid_dtype) - n // 2) / max(n / 2.0, 1.0)
        shape = [1] * x.ndim
        shape[ax] = n
        r2 = r2 + coord.reshape(shape) ** 2
    radius = jnp.sqrt(r2)
    mask = (radius <= (1.0 - alpha) * jnp.max(radius)).astype(x.dtype)
    k = jnp.fft.fftshift(jnp.fft.fftn(x, axes=fft_axes), axes=fft_axes)
    k = k * mask
    out = jnp.fft.ifftn(jnp.fft.ifftshift(k, axes=fft_axes), axes=fft_axes)
    return out.real.astype(x.dtype)


def gaussian_noise(
    x: Float[Array, '...'],
    key: Array,
    *,
    sigma: Float[Array, '...'],
) -> Float[Array, '...']:
    """Add i.i.d. Gaussian noise per element.

    Draws independent zero-mean Gaussian noise with standard deviation
    ``sigma`` for every element of ``x`` and adds it to the input, so each
    perturbation is distributed as :math:`\\mathcal{N}(0, \\sigma^2)`.

    Parameters
    ----------
    x
        Intensity tensor.
    key
        PRNG key.
    sigma
        Standard deviation of the additive noise (scalar or broadcastable
        to ``x``).  Drawing it randomly is the caller's concern.

    Returns
    -------
    Noised tensor of the same shape as ``x``.
    """
    noise = jax.random.normal(key, x.shape, dtype=x.dtype)
    return x + sigma * noise


def rician_noise(
    x: Float[Array, '...'],
    key: Array,
    *,
    sigma: Float[Array, '...'],
) -> Float[Array, '...']:
    """Add Rician noise -- the magnitude-image noise model.

    Returns :math:`\\sqrt{(x + n_r)^2 + n_i^2}` with independent noise
    terms :math:`n_r, n_i \\sim \\mathcal{N}(0, \\sigma^2)`.  Reduces to
    :math:`|x|` at :math:`\\sigma = 0`.  This is the noise distribution of
    the magnitude of a complex signal whose real and imaginary parts carry
    independent Gaussian perturbations.

    Parameters
    ----------
    x
        Intensity tensor, interpreted as the true magnitude signal.
    key
        PRNG key.  It is split into two independent streams for the real
        and imaginary noise components.
    sigma
        Standard deviation of the Gaussian perturbation on each of the
        real and imaginary parts (scalar or broadcastable to ``x``).
        Drawing it randomly is the caller's concern.

    Returns
    -------
    Noised magnitude tensor of the same shape as ``x``.
    """
    k_r, k_i = jax.random.split(key, 2)
    n_r = jax.random.normal(k_r, x.shape, dtype=x.dtype) * sigma
    n_i = jax.random.normal(k_i, x.shape, dtype=x.dtype) * sigma
    return jnp.sqrt((x + n_r) ** 2 + n_i**2)
