# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Core machinery for the registration recipes.

The pieces shared by ``rigid_register`` / ``affine_register``: the
``RegistrationSpec`` static-config record (the ``SolverSpec`` /
``Interpolator`` ADT precedent -- a frozen, hashable dataclass that rides
``jit`` static args), the ``RegistrationResult`` ``NamedTuple`` output,
and the coarse-to-fine driver.

Per pyramid level the driver minimises a similarity cost over the
transform's Lie parameters ``θ`` and warm-starts the next finer level
from the result (translations rescaled to the finer voxel grid).  Two
optimisation paths:

- **SSD** -> Gauss-Newton / Levenberg-Marquardt on the *vector* residual
  ``warp(θ) - fixed`` (``linalg.optimize``).  Matrix-free, GPU-native,
  second-order -- the 3dvolreg / Lucas-Kanade lineage.
- **LNCC / MI / CR** -> BFGS on the *scalar* cost
  (``jax.scipy.optimize.minimize``), which handles the non-least-squares
  metric and the rotation/translation scaling via its Hessian estimate.
  Also matrix-free, so it too survives the cuSolver wedge.

Images are single-channel ``(*spatial,)`` (the registration norm); the
channel axis is added internally for ``spatial_transform`` / pyramids.
Coordinates are index-space (``identity_grid`` convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional

import jax.numpy as jnp
from jax.scipy.optimize import minimize
from jaxtyping import Array, Float

from ..geometry import (
    affine_grid,
    gaussian_pyramid,
    spatial_transform,
)
from ..geometry._interpolate import BoundaryMode, Interpolator, Linear
from ..linalg import gauss_newton, levenberg_marquardt
from ..metrics import correlation_ratio, lncc, mutual_information, ssd
from ._model import TransformModel


@dataclass(frozen=True)
class RegistrationSpec:
    """Static configuration for a registration recipe.

    Frozen and hashable so it rides ``jit`` static arguments.

    Attributes
    ----------
    levels
        Number of Gaussian-pyramid resolutions (coarse-to-fine).
    iterations
        Optimiser iterations per level.
    metric
        Similarity: ``"ssd"`` (within-modality; the GN/LM least-squares
        path), ``"lncc"`` (local cross-correlation; intensity-robust),
        ``"mi"`` / ``"cr"`` (cross-modal).
    optimizer
        ``"auto"`` (SSD -> ``"lm"``, else -> ``"bfgs"``), or force
        ``"lm"`` / ``"gn"`` (SSD only) / ``"bfgs"``.
    interpolation
        Sampling kernel for the warp (an ``Interpolator``).
    boundary_mode, cval
        Out-of-bounds handling for the warp (default zero-fill).
    pyramid_factor, pyramid_sigma
        Downsample factor and anti-alias sigma for the pyramid.
    lncc_radius
        Window radius for the ``"lncc"`` metric.
    bins
        Histogram bins for the ``"mi"`` / ``"cr"`` metrics.
    cg_tol
        Inner-CG tolerance for the GN/LM path.
    """

    levels: int = 3
    iterations: int = 30
    metric: str = 'ssd'
    optimizer: str = 'auto'
    interpolation: Interpolator = field(default_factory=Linear)
    boundary_mode: BoundaryMode = 'constant'
    cval: float = 0.0
    pyramid_factor: float = 2.0
    pyramid_sigma: Optional[float] = None
    lncc_radius: int = 4
    bins: int = 32
    cg_tol: float = 1e-6


class RegistrationResult(NamedTuple):
    """Output of a registration recipe.

    Attributes
    ----------
    matrix
        The estimated homogeneous transform, ``(ndim + 1, ndim + 1)``
        (full-resolution, index-space), mapping fixed coordinates to
        moving coordinates.
    params
        The transform's Lie parameters at full resolution.
    warped
        ``moving`` resampled by ``matrix`` onto the ``fixed`` grid.
    cost_history
        Concatenated per-level optimiser cost traces.
    """

    matrix: Float[Array, 'd1 d1']
    params: Float[Array, ' p']
    warped: Float[Array, '*spatial']
    cost_history: Float[Array, ' h']


def _center(shape: tuple[int, ...], dtype: Any) -> Array:
    return (jnp.asarray(shape, dtype=dtype) - 1.0) / 2.0


def _warp(
    moving: Array,
    params: Array,
    *,
    model: TransformModel,
    ndim: int,
    center: Array,
    spec: RegistrationSpec,
) -> Array:
    """Warp a single-channel ``moving`` image by ``model.exp(params)``."""
    matrix = model.exp(params, ndim=ndim)
    grid = affine_grid(matrix, moving.shape, center=center)
    warped = spatial_transform(
        moving[..., None],
        grid,
        mode=spec.boundary_mode,
        cval=spec.cval,
        method=spec.interpolation,
    )
    return warped[..., 0]


def _metric_cost(warped: Array, fixed: Array, spec: RegistrationSpec) -> Array:
    """Scalar similarity cost (lower is better)."""
    if spec.metric == 'ssd':
        return ssd(warped, fixed)
    if spec.metric == 'lncc':
        return 1.0 - lncc(warped, fixed, radius=spec.lncc_radius)
    if spec.metric == 'mi':
        return -mutual_information(warped, fixed, bins=spec.bins)
    if spec.metric == 'cr':
        return 1.0 - correlation_ratio(warped, fixed, bins=spec.bins)
    raise ValueError(
        f'spec.metric={spec.metric!r}; expected "ssd", "lncc", "mi", or "cr".'
    )


def _optimize_level(
    moving: Array,
    fixed: Array,
    params: Array,
    *,
    model: TransformModel,
    ndim: int,
    center: Array,
    spec: RegistrationSpec,
) -> tuple[Array, Array]:
    """Optimise ``params`` on one resolution; return ``(params, history)``."""
    use_lsq = spec.metric == 'ssd' and spec.optimizer in ('auto', 'lm', 'gn')
    if use_lsq:

        def residual(p: Array) -> Array:
            warped = _warp(
                moving, p, model=model, ndim=ndim, center=center, spec=spec
            )
            return (warped - fixed).ravel()

        if spec.optimizer == 'gn':
            res = gauss_newton(
                residual, params, n_iters=spec.iterations, cg_tol=spec.cg_tol
            )
        else:
            res = levenberg_marquardt(
                residual, params, n_iters=spec.iterations, cg_tol=spec.cg_tol
            )
        return res.params, res.cost_history

    def cost(p: Array) -> Array:
        warped = _warp(
            moving, p, model=model, ndim=ndim, center=center, spec=spec
        )
        return _metric_cost(warped, fixed, spec)

    init_cost = cost(params)
    out = minimize(
        cost, params, method='BFGS', options={'maxiter': spec.iterations}
    )
    history = jnp.stack([init_cost, out.fun])
    return out.x, history


def multi_resolution_register(
    moving: Float[Array, '*spatial'],
    fixed: Float[Array, '*spatial'],
    *,
    model: TransformModel,
    ndim: int,
    spec: RegistrationSpec,
    init_params: Optional[Float[Array, ' p']] = None,
) -> RegistrationResult:
    """Coarse-to-fine registration driver shared by the recipes."""
    if moving.shape != fixed.shape:
        raise ValueError(
            f'moving and fixed must share shape; got {moving.shape} '
            f'vs {fixed.shape}.'
        )
    if len(moving.shape) != ndim:
        raise ValueError(
            f'expected a {ndim}-D single-channel image; got shape '
            f'{moving.shape}.'
        )
    dtype = moving.dtype
    pyr_m = gaussian_pyramid(
        moving[..., None],
        levels=spec.levels,
        factor=spec.pyramid_factor,
        sigma=spec.pyramid_sigma,
    )
    pyr_f = gaussian_pyramid(
        fixed[..., None],
        levels=spec.levels,
        factor=spec.pyramid_factor,
        sigma=spec.pyramid_sigma,
    )

    params = (
        jnp.zeros(model.n_params(ndim), dtype=dtype)
        if init_params is None
        else init_params
    )
    histories = []
    prev_shape: Optional[tuple[int, ...]] = None
    # Coarsest (highest index) to finest (0).
    for level in range(spec.levels - 1, -1, -1):
        m_l = pyr_m[level][..., 0]
        f_l = pyr_f[level][..., 0]
        shape_l = m_l.shape
        if prev_shape is not None:
            # Translations are in voxel units; rescale to this grid.
            ratio = jnp.asarray(shape_l, dtype=dtype) / jnp.asarray(
                prev_shape, dtype=dtype
            )
            params = model.rescale_to_grid(params, ratio)
        center = _center(shape_l, dtype)
        params, hist = _optimize_level(
            m_l,
            f_l,
            params,
            model=model,
            ndim=ndim,
            center=center,
            spec=spec,
        )
        histories.append(hist)
        prev_shape = shape_l

    center = _center(moving.shape, dtype)
    matrix = model.exp(params, ndim=ndim)
    warped = _warp(
        moving, params, model=model, ndim=ndim, center=center, spec=spec
    )
    return RegistrationResult(
        matrix=matrix,
        params=params,
        warped=warped,
        cost_history=jnp.concatenate(histories),
    )
