# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Generative augmentation: synthesise images from label maps and bias fields.

- :func:`gmm_label_to_image` renders an intensity image from an integer
  label map by a per-label Gaussian mixture: each voxel takes its label's
  mean plus that label's standard deviation times unit Gaussian noise.
  This is the domain-randomisation synthesis behind contrast-agnostic
  training (synthesise an image from anatomy rather than perturbing a real
  one).
- :func:`simulate_bias_field` produces a smooth, low-frequency,
  multiplicative intensity non-uniformity (INU / "bias") field, generated
  by exponentiating an upsampled low-resolution Gaussian field.  This is
  the *forward* (simulation) counterpart to the *corrective* bias-field
  estimators in :mod:`nitrix.bias`.

Both are pure functions of an explicit PRNG key.  :func:`gmm_label_to_image`
takes explicit per-label statistics so it serves both fixed intensities
and randomised contrast (draw the per-label means and standard deviations
in the caller); :func:`simulate_bias_field` is a generator parameterised by
a magnitude bound.
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int

from ._common import _coarse_random_field, _default_float

__all__ = ['gmm_label_to_image', 'simulate_bias_field']


def gmm_label_to_image(
    label_map: Int[Array, '*spatial'],
    means: Float[Array, 'n_labels'],
    stds: Float[Array, 'n_labels'],
    key: Array,
    *,
    nonneg: bool = True,
) -> Float[Array, '*spatial']:
    """Render an image from a label map by per-label Gaussian sampling.

    For each voxel, ``image = means[label] + stds[label] * N(0, 1)``,
    a draw from the per-label Gaussian.  Passing per-label statistics
    explicitly (rather than drawing them inside) keeps this a pure render:
    supply fixed ``means`` / ``stds`` for a deterministic intensity map,
    or draw them per call (e.g. ``means ~ U(...)``, ``stds ~ U(...)``) for
    the ever-changing-contrast domain-randomisation augmentation.

    Parameters
    ----------
    label_map
        Integer labels in ``[0, n_labels)``.
    means, stds
        Per-label mean and within-label standard deviation, ``(n_labels,)``.
    key
        PRNG key for the per-voxel unit-Gaussian draw.
    nonneg
        Clamp the output at 0 (default ``True``).

    Returns
    -------
    Float[Array, '*spatial']
        Intensity image with the shape of ``label_map``.
    """
    idx = label_map.astype(jnp.int32)
    noise = jax.random.normal(key, label_map.shape, dtype=means.dtype)
    image = means[idx] + stds[idx] * noise
    if nonneg:
        image = jnp.maximum(image, 0.0)
    return image


def simulate_bias_field(
    spatial_shape: Sequence[int],
    key: Array,
    *,
    max_std: float = 0.5,
    grid_fraction: float = 0.04,
    dtype: Optional[DTypeLike] = None,
) -> Float[Array, '*spatial']:
    """Simulate a smooth multiplicative intensity non-uniformity field.

    Draws a low-resolution Gaussian field, whose standard deviation is
    itself drawn as :math:`\\sigma \\sim U(0, \\texttt{max\\_std})`, on a
    coarse grid of ``round(shape * grid_fraction)`` cells per axis (at
    least 2), linearly upsamples it to ``spatial_shape``, and
    exponentiates the result.  The output is a smooth, strictly-positive
    field with mean near 1: multiply an image by it to impose a realistic
    low-frequency bias.  A ``max_std`` of 0 gives an all-ones (no-op)
    field.

    Parameters
    ----------
    spatial_shape
        Target shape of the simulated field, one entry per spatial axis.
    key
        PRNG key; split internally to draw both the field's standard
        deviation and the coarse Gaussian field.
    max_std
        Upper bound of the uniform distribution from which the coarse
        field's standard deviation is drawn (in log-intensity units,
        prior to exponentiation).  Larger values yield stronger bias.
    grid_fraction
        Fraction of each spatial extent used to size the coarse grid;
        the coarse grid has ``round(shape * grid_fraction)`` cells per
        axis (clamped to at least 2).  Smaller values give a smoother,
        lower-frequency field.
    dtype
        Floating dtype of the output.  Defaults to the x64-aware default
        float (``float64`` when x64 is enabled, otherwise ``float32``).

    Returns
    -------
    Float[Array, '*spatial']
        Strictly-positive multiplicative field with shape
        ``spatial_shape``.
    """
    dt = _default_float() if dtype is None else dtype
    k_std, k_field = jax.random.split(key, 2)
    std = jax.random.uniform(k_std, (), dtype=dt, minval=0.0, maxval=max_std)
    full = _coarse_random_field(
        spatial_shape, k_field, std=std, grid_fraction=grid_fraction, dtype=dt
    )
    return jnp.exp(full)
