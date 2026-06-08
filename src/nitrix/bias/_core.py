# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Shared iterative driver for log-domain bias-field correction.

The N3/N4 *iteration* -- log-transform, then per fitting level iterate
(sharpen the intensity histogram -> take the residual -> smooth it with a
B-spline -> accumulate the field) until a coefficient-of-variation
convergence test trips -- is independent of *which* B-spline estimator
smooths the residual.  This module is that shared driver, parameterised by
a ``fit_method``:

- ``'mba'``      -- Lee--Wolberg--Shin MBA; this *is* N4 (ITK parity).
- ``'least_squares'`` / ``'psplines'`` -- the higher-accuracy estimators
  (``nitrix.bias._bspline``); same iteration, different (and unbiased)
  field fit.  These are deliberately **not** N4: they are exposed through
  the ``bias_field_correction`` dispatcher, not through
  ``n4_bias_field_correction`` (which stays a faithful N4).

See ``docs/design/bias-field.md`` for the parity-vs-correctness rationale.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._bspline import (
    FitMethod,
    _control_inverse_gram,
    _fit,
    _normalise_control_points,
    _reconstruct,
    _reconstruction_matrix,
    _resolve_spatial_rank,
    _solve_field,
)
from ._sharpen import sharpen_histogram


def _coefficient_of_variation(
    ratio: Float[Array, '...'],
    mask: Float[Array, '...'],
) -> Float[Array, '']:
    """ITK's convergence measurement: CV of the field ratio over the mask.

    ``ratio = exp(field_new - field_prev)``; the measurement is
    ``std(ratio) / mean(ratio)`` over masked voxels with the sample
    (``ddof = 1``) standard deviation, matching
    ``N4...::CalculateConvergenceMeasurement``.
    """
    n = jnp.sum(mask)
    mu = jnp.sum(mask * ratio) / n
    var = jnp.sum(mask * (ratio - mu) ** 2) / jnp.maximum(n - 1.0, 1.0)
    return jnp.sqrt(var) / mu


def _bfc_single(
    image: Float[Array, '*spatial'],
    mask: Float[Array, '*spatial'],
    *,
    spline_order: int,
    level_control_points: Sequence[Tuple[int, ...]],
    max_iterations: Tuple[int, ...],
    convergence_threshold: float,
    n_histogram_bins: int,
    fwhm: float,
    wiener_noise: float,
    fit_method: FitMethod,
    ridge: float,
    penalty: float,
    penalty_order: int,
    eps: float,
) -> Tuple[Float[Array, '*spatial'], Float[Array, '*spatial']]:
    """Iterative bias correction on a single volume.

    Returns ``(corrected, multiplicative_bias_field)``.  The fitting
    estimator is selected by ``fit_method``; everything else (log domain,
    sharpening, residual, field accumulation, convergence) is the shared
    N3/N4 loop.
    """
    axes = tuple(range(image.ndim))
    spatial_shape = image.shape
    dtype = jnp.result_type(image.dtype, jnp.float32)

    # Log domain over the mask; masked-out voxels -> log(1) = 0 (finite).
    safe = jnp.where(mask > 0, image, 1.0)
    log_input = jnp.log(jnp.clip(safe, eps, None)).astype(dtype)
    log_bias = jnp.zeros_like(log_input)

    use_penalty = penalty if fit_method == 'psplines' else 0.0

    for level, ncp_level in enumerate(level_control_points):
        matrices = [
            _reconstruction_matrix(n_vox, n_ctrl, spline_order, dtype)
            for n_vox, n_ctrl in zip(spatial_shape, ncp_level)
        ]

        # For least-squares / P-splines the Gram depends only on the mask
        # and this level's grid, not on the data, so factor (invert) it
        # once per level and reuse it across every sharpening iteration.
        inv_gram: Optional[Array] = None
        if fit_method != 'mba':
            inv_gram = _control_inverse_gram(
                mask,
                matrices,
                axes,
                ncp_level,
                ridge=ridge,
                penalty=use_penalty,
                penalty_order=penalty_order,
                dtype=dtype,
            )

        def increment_fn(
            residual: Array,
            matrices: Sequence[Array] = matrices,
            inv_gram: Optional[Array] = inv_gram,
            ncp_level: Tuple[int, ...] = ncp_level,
        ) -> Array:
            if inv_gram is None:  # MBA
                phi = _fit(residual, mask, matrices, axes, eps)
                return _reconstruct(phi, matrices, axes)
            return _solve_field(
                residual, mask, inv_gram, matrices, axes, ncp_level
            )

        max_iter = int(max_iterations[level])

        def cond(
            state: Tuple[Array, Array, Array],
            max_iter: int = max_iter,
        ) -> Array:
            _, it, cv = state
            return jnp.logical_and(it < max_iter, cv > convergence_threshold)

        def body(
            state: Tuple[Array, Array, Array],
            increment_fn: Callable[[Array], Array] = increment_fn,
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
            increment = increment_fn(residual)
            new_field = field + increment
            cv = _coefficient_of_variation(jnp.exp(increment), mask)
            return new_field, it + 1, cv

        init = (
            log_bias,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(jnp.inf, dtype=dtype),
        )
        log_bias = lax.while_loop(cond, body, init)[0]

    bias_field = jnp.exp(log_bias)
    corrected = image / bias_field
    return corrected, bias_field


def _level_control_points(
    n_control_points: Tuple[int, ...],
    spline_order: int,
    n_fitting_levels: int,
) -> list[Tuple[int, ...]]:
    """Per-level control-point counts: the B-spline mesh doubles each level.

    ``mesh_initial = n_control_points - spline_order``; at level ``l`` the
    mesh is ``mesh_initial * 2**l`` and the control-point count is
    ``mesh + spline_order`` (matching ITK's lattice doubling).
    """
    mesh_initial = tuple(c - spline_order for c in n_control_points)
    return [
        tuple(m * (2**level) + spline_order for m in mesh_initial)
        for level in range(n_fitting_levels)
    ]


def apply_bias_field_correction(
    image: Float[Array, '... *spatial'],
    *,
    mask: Optional[Float[Array, '... *spatial']],
    fit_method: FitMethod,
    spline_order: int,
    n_control_points: Union[int, Sequence[int]],
    n_fitting_levels: int,
    max_iterations: Union[int, Sequence[int]],
    convergence_threshold: float,
    n_histogram_bins: int,
    fwhm: float,
    wiener_noise: float,
    ridge: float,
    penalty: float,
    penalty_order: int,
    spatial_rank: Optional[int],
    return_bias_field: bool,
    eps: float,
) -> Union[
    Float[Array, '... *spatial'],
    Tuple[Float[Array, '... *spatial'], Float[Array, '... *spatial']],
]:
    """Shared entry point behind ``n4_bias_field_correction`` and
    ``bias_field_correction``: validate, default the mask, then map the
    per-volume core over any leading batch dims."""
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

    levels = _level_control_points(ncp, spline_order, n_fitting_levels)

    core = partial(
        _bfc_single,
        spline_order=spline_order,
        level_control_points=levels,
        max_iterations=max_iters,
        convergence_threshold=convergence_threshold,
        n_histogram_bins=n_histogram_bins,
        fwhm=fwhm,
        wiener_noise=wiener_noise,
        fit_method=fit_method,
        ridge=ridge,
        penalty=penalty,
        penalty_order=penalty_order,
        eps=eps,
    )

    # Collapse leading batch dims into one axis and vmap so each volume gets
    # an independent histogram, field, and convergence test.
    batch_shape = x.shape[: x.ndim - rank]
    if batch_shape:
        n_batch = 1
        for d in batch_shape:
            n_batch *= d
        spatial = x.shape[x.ndim - rank :]
        corrected, bias = jax.vmap(core)(
            x.reshape((n_batch, *spatial)), m.reshape((n_batch, *spatial))
        )
        corrected = corrected.reshape(x.shape)
        bias = bias.reshape(x.shape)
    else:
        corrected, bias = core(x, m)

    if return_bias_field:
        return corrected, bias
    return corrected
