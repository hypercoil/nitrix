# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
N4 bias-field correction (Tustison 2010), GPU-accelerated in pure JAX.

N4 estimates and removes the smooth, low-frequency multiplicative
intensity inhomogeneity ("bias field") that corrupts MRI.  It is the
improved ("N4ITK") successor to N3 (Sled 1998); the improvements over N3
are (a) a B-spline *approximation* of the field instead of N3's smoothing
spline and (b) a multi-resolution fitting hierarchy.  This module follows
the ITK / ANTs ``N4BiasFieldCorrectionImageFilter`` algorithm and default
parameters, reformulated for the accelerator:

- The per-iteration histogram **sharpening** is the N3 Wiener
  deconvolution (``_sharpen.sharpen_histogram``) -- 1-D FFTs, cheap.
- The field **smoothing** is a separable cubic B-spline scattered-data
  approximation (``_bspline``) -- dense per-axis contractions that lower
  to XLA ``dot`` / tensor cores; no gather, no scatter, differentiable.
- The **iteration** is a ``lax.while_loop`` with ITK's coefficient-of-
  variation convergence test; the **multi-resolution** hierarchy is the
  outer (static) Python loop doubling the B-spline mesh per level.

Why no Pallas: the heavy ops are dense small-matrix contractions and 1-D
FFTs, both of which XLA already schedules near-optimally on GPU/TPU.  Per
the house "benchmark-first, don't build Pallas speculatively" policy
(BACKLOG B6/B7), N4 ships pure-JAX; a Pallas kernel would only be
considered if a consumer benchmark showed a wall.  See
``docs/design/bias-field.md``.

Equivalence to ITK's lattice accumulation: ITK accumulates a B-spline
*control-point lattice* and refines it (B-spline subdivision) between
levels.  Because reconstruction is linear in the control points, that is
equivalent to accumulating the reconstructed *field* at full resolution
and fitting each level's residual on a finer grid -- which is what we do.
The reconstructed fields agree up to float; see the design doc.

Differentiability: the B-spline smoothing is differentiable; the
histogram sharpening is not (piecewise-constant binning).  Per the
consumer ask, efficient end-to-end differentiability is a plus, not a
requirement, at this time.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Sequence, Tuple, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._bspline import (
    _fit,
    _normalise_control_points,
    _reconstruct,
    _reconstruction_matrix,
    _resolve_spatial_rank,
)
from ._sharpen import sharpen_histogram

__all__ = ['n4_bias_field_correction']


def _coefficient_of_variation(
    ratio: Float[Array, '...'],
    mask: Float[Array, '...'],
) -> Float[Array, '']:
    '''ITK's convergence measurement: CV of the field ratio over the mask.

    ``ratio = exp(field_new - field_prev)``; the measurement is
    ``std(ratio) / mean(ratio)`` over masked voxels with the sample
    (``ddof = 1``) standard deviation, matching
    ``N4...::CalculateConvergenceMeasurement``.
    '''
    n = jnp.sum(mask)
    mu = jnp.sum(mask * ratio) / n
    var = jnp.sum(mask * (ratio - mu) ** 2) / jnp.maximum(n - 1.0, 1.0)
    return jnp.sqrt(var) / mu


def _n4_single(
    image: Float[Array, '*spatial'],
    mask: Float[Array, '*spatial'],
    *,
    spline_order: int,
    n_control_points: Tuple[int, ...],
    n_fitting_levels: int,
    max_iterations: Tuple[int, ...],
    convergence_threshold: float,
    n_histogram_bins: int,
    fwhm: float,
    wiener_noise: float,
    eps: float,
) -> Tuple[Float[Array, '*spatial'], Float[Array, '*spatial']]:
    '''Core N4 on a single volume; returns (corrected, multiplicative bias).'''
    rank = image.ndim
    spatial_axes = tuple(range(rank))
    spatial_shape = image.shape

    # Work in the log domain over the masked region.  Masked-out voxels are
    # set to log(1) = 0 so the array stays finite; the mask zeroes their
    # influence on every histogram / fit.
    safe = jnp.where(mask > 0, image, 1.0)
    log_input = jnp.log(jnp.clip(safe, eps, None))

    log_bias = jnp.zeros_like(log_input)

    mesh_initial = tuple(c - spline_order for c in n_control_points)

    for level in range(n_fitting_levels):
        ncp_level = tuple(m * (2**level) + spline_order for m in mesh_initial)
        matrices = [
            _reconstruction_matrix(
                n_vox, n_ctrl, spline_order, log_input.dtype
            )
            for n_vox, n_ctrl in zip(spatial_shape, ncp_level)
        ]
        max_iter = int(max_iterations[level])

        def cond(
            state: Tuple[Array, Array, Array],
            max_iter: int = max_iter,
        ) -> Array:
            _, it, cv = state
            return jnp.logical_and(it < max_iter, cv > convergence_threshold)

        def body(
            state: Tuple[Array, Array, Array],
            matrices: Sequence[Array] = matrices,
        ) -> Tuple[Array, Array, Array]:
            field, it, _ = state
            log_uncorrected = log_input - field
            sharpened = sharpen_histogram(
                log_uncorrected,
                weight=mask,
                n_bins=n_histogram_bins,
                fwhm=fwhm,
                wiener_noise=wiener_noise,
            )
            residual = log_uncorrected - sharpened
            phi = _fit(residual, mask, matrices, spatial_axes, eps)
            increment = _reconstruct(phi, matrices, spatial_axes)
            new_field = field + increment
            cv = _coefficient_of_variation(jnp.exp(increment), mask)
            return new_field, it + 1, cv

        init = (
            log_bias,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(jnp.inf, dtype=log_input.dtype),
        )
        log_bias, _, _ = lax.while_loop(cond, body, init)

    bias_field = jnp.exp(log_bias)
    corrected = image / bias_field
    return corrected, bias_field


def n4_bias_field_correction(
    image: Float[Array, '... *spatial'],
    *,
    mask: Optional[Float[Array, '... *spatial']] = None,
    spline_order: int = 3,
    n_control_points: Union[int, Sequence[int]] = 4,
    n_fitting_levels: int = 4,
    max_iterations: Union[int, Sequence[int]] = 50,
    convergence_threshold: float = 1e-3,
    n_histogram_bins: int = 200,
    fwhm: float = 0.15,
    wiener_noise: float = 0.01,
    spatial_rank: Optional[int] = None,
    return_bias_field: bool = False,
    eps: float = 1e-8,
) -> Union[
    Float[Array, '... *spatial'],
    Tuple[Float[Array, '... *spatial'], Float[Array, '... *spatial']],
]:
    '''N4 (Tustison) bias-field correction.

    Estimates the smooth multiplicative bias field corrupting ``image`` and
    returns the corrected image (``image / bias``).  Follows the ITK / ANTs
    ``N4BiasFieldCorrectionImageFilter`` algorithm and defaults; see the
    module docstring and ``docs/design/bias-field.md`` for the accelerator
    reformulation.

    The entire input (modulo leading batch dims) is treated as one volume.
    For a batch of volumes -- distinct leading dims with ``spatial_rank``
    set -- each volume is corrected independently (its own histogram,
    field, and convergence) via an internal ``vmap``.

    Parameters
    ----------
    image
        Positive intensities, ``(..., *spatial)``.  Non-positive voxels are
        clamped for the log transform; keep them out of the mask.
    mask
        Region over which the field is estimated (the histogram and B-spline
        fit are confined to it; the field is still reconstructed -- and the
        correction applied -- everywhere).  ``None`` (default) uses
        ``image > 0``.  Float masks act as per-voxel confidence weights.
    spline_order
        B-spline order for the field.  Default ``3`` (cubic; the N4 default
        and the parity-validated path).
    n_control_points
        Control points per spatial axis at the **coarsest** fitting level
        (ITK ``NumberOfControlPoints``).  ``int`` -- isotropic; sequence --
        per-axis.  Must be ``>= spline_order + 1``.  The mesh doubles each
        level.  Default ``4`` (-> mesh size 1 at the coarsest level).
    n_fitting_levels
        Number of multi-resolution levels; the B-spline mesh doubles per
        level.  Default ``4``.
    max_iterations
        Maximum sharpen/fit iterations per level.  ``int`` -- same cap every
        level; sequence -- per-level (length ``n_fitting_levels``).  Default
        ``50`` (ITK default is ``(50, 50, 50, 50)``).
    convergence_threshold
        Stop a level early when the coefficient of variation of the field
        update drops below this.  Default ``1e-3`` (ITK default).
    n_histogram_bins, fwhm, wiener_noise
        Histogram-sharpening parameters; see ``sharpen_histogram``.  Defaults
        ``200`` / ``0.15`` / ``0.01`` (ITK defaults).
    spatial_rank
        Number of trailing dims that are spatial.  ``None`` infers it from
        ``n_control_points`` (if a sequence) or treats every dim as spatial
        (single volume).  Set it to mark leading batch dims.
    return_bias_field
        If ``True``, also return the estimated multiplicative bias field
        (same shape as ``image``), e.g. for QC or to apply to another image.
    eps
        Numerical floor for the log transform and the B-spline denominators.

    Returns
    -------
    ``corrected`` (default), or ``(corrected, bias_field)`` if
    ``return_bias_field`` -- both with the same shape as ``image``.

    Notes
    -----
    The B-spline field smoothing is differentiable; the histogram sharpening
    is not (piecewise-constant binning).  The op JITs cleanly (static FFT
    and control-lattice shapes; ``lax.while_loop`` iteration).
    '''
    x = jnp.asarray(image)
    dtype = jnp.result_type(x.dtype, jnp.float32)
    x = x.astype(dtype)

    rank = _resolve_spatial_rank(n_control_points, spatial_rank, x.ndim)
    ncp = _normalise_control_points(n_control_points, rank)
    for c in ncp:
        if c < spline_order + 1:
            raise ValueError(
                f'n_control_points={n_control_points!r} too small for '
                f'spline_order={spline_order}: each axis needs at least '
                f'{spline_order + 1} control points.'
            )

    if isinstance(max_iterations, int):
        max_iters = (int(max_iterations),) * n_fitting_levels
    else:
        max_iters = tuple(int(m) for m in max_iterations)
        if len(max_iters) != n_fitting_levels:
            raise ValueError(
                f'max_iterations sequence has {len(max_iters)} entries but '
                f'n_fitting_levels={n_fitting_levels}.'
            )

    if mask is None:
        m = (x > 0).astype(dtype)
    else:
        m = jnp.broadcast_to(jnp.asarray(mask).astype(dtype), x.shape)

    core = partial(
        _n4_single,
        spline_order=spline_order,
        n_control_points=ncp,
        n_fitting_levels=n_fitting_levels,
        max_iterations=max_iters,
        convergence_threshold=convergence_threshold,
        n_histogram_bins=n_histogram_bins,
        fwhm=fwhm,
        wiener_noise=wiener_noise,
        eps=eps,
    )

    # Collapse any leading batch dims into a single axis and vmap so each
    # volume gets an independent histogram, field, and convergence test.
    batch_shape = x.shape[: x.ndim - rank]
    if batch_shape:
        n_batch = 1
        for d in batch_shape:
            n_batch *= d
        xb = x.reshape((n_batch, *x.shape[x.ndim - rank :]))
        mb = m.reshape((n_batch, *m.shape[m.ndim - rank :]))
        corrected, bias = jax.vmap(core)(xb, mb)
        corrected = corrected.reshape(x.shape)
        bias = bias.reshape(x.shape)
    else:
        corrected, bias = core(x, m)

    if return_bias_field:
        return corrected, bias
    return corrected
