# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Shared iterative driver for log-domain bias-field correction.

The N3/N4 iteration -- log-transform, then per fitting level iterate
(sharpen the intensity histogram, take the residual, smooth it with a
B-spline, accumulate the field) until a coefficient-of-variation
convergence test trips -- is independent of *which* B-spline estimator
smooths the residual. This module is that shared driver, parameterised by
a ``fit_method``:

- ``'mba'`` -- Lee--Wolberg--Shin multilevel B-spline approximation; this
  *is* N4 (parity with the reference ITK implementation).
- ``'least_squares'`` / ``'psplines'`` -- the higher-accuracy estimators;
  same iteration, but a different (and unbiased) field fit. These are
  deliberately not N4: they are exposed through the
  :func:`bias_field_correction` dispatcher, not through
  :func:`n4_bias_field_correction` (which stays a faithful N4).
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
    """Convergence measurement: coefficient of variation of the field ratio.

    Computes the coefficient of variation :math:`\\sigma / \\mu` of the
    per-iteration multiplicative field ratio over the masked voxels, using
    the sample (``ddof = 1``) standard deviation. This matches the
    convergence measurement of the reference ITK N4 implementation, where
    the ratio is :math:`\\exp(\\text{field}_{\\text{new}} -
    \\text{field}_{\\text{prev}})`.

    Parameters
    ----------
    ratio : Float[Array, '...']
        Per-voxel multiplicative field ratio (typically the exponential of
        the field increment) over the spatial grid.
    mask : Float[Array, '...']
        Nonnegative per-voxel weights over the same spatial grid; only
        voxels with positive weight contribute to the mean and variance.

    Returns
    -------
    Float[Array, '']
        Scalar coefficient of variation :math:`\\sigma / \\mu` of ``ratio``
        over the masked voxels.
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

    Runs the shared N3/N4 loop on one spatial volume: the image is moved to
    the log domain over the mask, and for each fitting level the histogram
    is sharpened, the residual smoothed by a B-spline field increment, and
    the field accumulated until the coefficient-of-variation convergence
    test trips or the level's iteration budget is exhausted. The fitting
    estimator is selected by ``fit_method``; everything else is shared.

    Parameters
    ----------
    image : Float[Array, '*spatial']
        Single input volume over the spatial grid.
    mask : Float[Array, '*spatial']
        Nonnegative per-voxel weights over the same grid; voxels with
        positive weight are included in the fit and convergence test.
    spline_order : int
        Polynomial order of the B-spline basis used to represent the field.
    level_control_points : Sequence[Tuple[int, ...]]
        Per-level control-point counts, one tuple per fitting level, giving
        the number of control points along each spatial axis at that level.
    max_iterations : Tuple[int, ...]
        Per-level cap on the number of sharpening iterations, one entry per
        fitting level.
    convergence_threshold : float
        Iteration stops for a level once the coefficient of variation of
        the field ratio falls to or below this value.
    n_histogram_bins : int
        Number of bins used when sharpening the intensity histogram.
    fwhm : float
        Full width at half maximum of the Gaussian used to model the
        intensity histogram during sharpening.
    wiener_noise : float
        Noise level of the Wiener deconvolution applied when sharpening the
        histogram.
    fit_method : FitMethod
        Estimator selecting the B-spline field fit: ``'mba'`` for the
        multilevel B-spline approximation (N4), else the least-squares or
        P-spline estimators.
    ridge : float
        Ridge regularisation strength for the least-squares / P-spline
        estimators; unused for ``'mba'``.
    penalty : float
        Roughness-penalty strength; applied only for the ``'psplines'``
        estimator.
    penalty_order : int
        Order of the finite-difference roughness penalty for P-splines.
    eps : float
        Small positive floor applied before taking the logarithm and used
        as a numerical guard in the fit.

    Returns
    -------
    corrected : Float[Array, '*spatial']
        Bias-corrected volume, equal to ``image`` divided by the estimated
        multiplicative bias field.
    bias_field : Float[Array, '*spatial']
        Estimated multiplicative bias field over the spatial grid.
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

    The initial mesh size is :math:`\\text{n\\_control\\_points} -
    \\text{spline\\_order}` along each axis. At level :math:`l` the mesh is
    scaled by :math:`2^{l}` and the control-point count is :math:`\\text{mesh}
    + \\text{spline\\_order}`, matching the lattice doubling of the reference
    ITK implementation.

    Parameters
    ----------
    n_control_points : Tuple[int, ...]
        Number of control points along each spatial axis at the coarsest
        (first) fitting level.
    spline_order : int
        Polynomial order of the B-spline basis.
    n_fitting_levels : int
        Number of fitting levels; the mesh doubles between consecutive
        levels.

    Returns
    -------
    list[Tuple[int, ...]]
        Control-point counts per axis for each fitting level, ordered from
        coarsest to finest.
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
    """Shared entry point behind the two public bias-correction dispatchers.

    Backs both :func:`n4_bias_field_correction` and
    :func:`bias_field_correction`. Validates and normalises the arguments,
    defaults the mask when none is given, resolves the spatial rank, and
    then maps the per-volume correction core over any leading batch
    dimensions so that each volume gets an independent histogram, field, and
    convergence test.

    Parameters
    ----------
    image : Float[Array, '... *spatial']
        Input image, with optional leading batch dimensions preceding the
        spatial grid.
    mask : Float[Array, '... *spatial'] or None
        Nonnegative per-voxel weights broadcastable to ``image``. When
        ``None``, a mask of the strictly positive voxels of ``image`` is
        used.
    fit_method : FitMethod
        Estimator selecting the B-spline field fit (``'mba'``,
        ``'least_squares'``, or ``'psplines'``).
    spline_order : int
        Polynomial order of the B-spline basis.
    n_control_points : int or Sequence[int]
        Number of control points at the coarsest level, either shared
        across axes (int) or given per spatial axis (sequence).
    n_fitting_levels : int
        Number of fitting levels; the control-point mesh doubles between
        consecutive levels.
    max_iterations : int or Sequence[int]
        Maximum sharpening iterations, either shared across levels (int) or
        one entry per fitting level (sequence).
    convergence_threshold : float
        Coefficient-of-variation threshold below which a level's iteration
        stops.
    n_histogram_bins : int
        Number of bins used when sharpening the intensity histogram.
    fwhm : float
        Full width at half maximum of the Gaussian modelling the intensity
        histogram during sharpening.
    wiener_noise : float
        Noise level of the Wiener deconvolution applied when sharpening.
    ridge : float
        Ridge regularisation strength for the least-squares / P-spline
        estimators; unused for ``'mba'``.
    penalty : float
        Roughness-penalty strength; applied only for ``'psplines'``.
    penalty_order : int
        Order of the finite-difference roughness penalty for P-splines.
    spatial_rank : int or None
        Number of trailing spatial dimensions. When ``None`` it is inferred
        from the length of ``n_control_points`` if that is a sequence, else
        from the rank of ``image``.
    return_bias_field : bool
        If ``True``, also return the estimated multiplicative bias field.
    eps : float
        Small positive floor applied before taking the logarithm and used
        as a numerical guard in the fit.

    Returns
    -------
    Float[Array, '... *spatial'] or tuple of Float[Array, '... *spatial']
        The bias-corrected image. When ``return_bias_field`` is ``True``, a
        pair ``(corrected, bias_field)`` where ``bias_field`` is the
        estimated multiplicative bias field over the same shape.
    """
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
