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

The driver is written **once** against a ``CoordinateSpace`` (``_space``):
the only axis that separates an index-space, shared-grid registration from
a physically correct one is *how* the parameter transform relates the two
voxel grids, and that axis is the space's responsibility.  ``IndexSpace``
(default) is the leaner, fully-on-device, voxel-unit path; ``WorldSpace``
is anisotropy- and different-grid-correct.  Everything else -- the
pyramid, the level loop, the optimise, the result -- is identical.

Per pyramid level the driver minimises a similarity cost over the
transform's Lie parameters ``θ`` and warm-starts the next finer level
from the result (in ``IndexSpace`` the voxel-unit translations are
rescaled to the finer grid; in ``WorldSpace`` the physical parameters are
grid-independent and carry over unchanged).  Two optimisation paths:

- **SSD** -> Gauss-Newton / Levenberg-Marquardt on the *vector* residual
  ``warp(θ) - fixed`` (``linalg.optimize``).  Matrix-free, GPU-native,
  second-order -- the 3dvolreg / Lucas-Kanade lineage.
- **LNCC / MI / CR** -> BFGS on the *scalar* cost
  (``jax.scipy.optimize.minimize``), which handles the non-least-squares
  metric and the rotation/translation scaling via its Hessian estimate.
  Also matrix-free, so it too survives the cuSolver wedge.

Images are single-channel ``(*spatial,)`` (the registration norm); the
channel axis is added internally for ``spatial_transform`` / pyramids.
``moving`` and ``fixed`` need not share a shape (the warp is always built
on the ``fixed`` grid); ``IndexSpace`` additionally assumes they share a
voxel grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Optional, Sequence, Union

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
from ._metric import SSD, Metric
from ._model import TransformModel
from ._objective import MetricObjective, Objective
from ._space import CoordinateSpace, IndexSpace, _Sampler


@dataclass(frozen=True)
class RegistrationSpec:
    """Static configuration for a registration recipe.

    Frozen and hashable so it rides ``jit`` static arguments.  The
    per-image voxel->world geometry is **not** here (it is array data, not
    static config): it travels on the ``space`` argument of the recipes
    (``IndexSpace`` / ``WorldSpace``).

    Attributes
    ----------
    levels
        Number of Gaussian-pyramid resolutions (coarse-to-fine).
    iterations
        Optimiser iterations per level: an ``int`` (the same count at every
        level) or a length-``levels`` **coarse-to-fine** tuple (front-load the
        cheap coarse levels, starve the expensive finest one -- a second-order
        solver converges in a handful of finest-level steps).
    metric
        Similarity objective, a ``Metric`` record carrying its own
        hyper-parameters: ``SSD()`` (within-modality; the GN/LM
        least-squares path), ``LNCC(radius=...)`` (local cross-correlation;
        intensity-robust), ``MI(bins=...)`` / ``CorrelationRatio(bins=...)``
        (cross-modal).
    optimizer
        ``"auto"`` (least-squares metric -> ``"lm"``, else -> ``"bfgs"``),
        or force ``"lm"`` / ``"gn"`` (least-squares only) / ``"bfgs"``.
    interpolation
        Sampling kernel for the warp (an ``Interpolator``).
    boundary_mode, cval
        Out-of-bounds handling for the warp (default zero-fill).
    pyramid_factor, pyramid_sigma
        Downsample factor and anti-alias sigma for the pyramid.
    cg_tol
        Inner-CG tolerance for the GN/LM path.
    """

    levels: int = 3
    iterations: Union[int, tuple[int, ...]] = 30
    metric: Metric = field(default_factory=SSD)
    optimizer: str = 'auto'
    interpolation: Interpolator = field(default_factory=Linear)
    boundary_mode: BoundaryMode = 'constant'
    cval: float = 0.0
    pyramid_factor: float = 2.0
    pyramid_sigma: Optional[float] = None
    cg_tol: float = 1e-6


class RegistrationResult(NamedTuple):
    """Output of a registration recipe.

    Attributes
    ----------
    matrix
        The estimated homogeneous transform, ``(ndim + 1, ndim + 1)``.  In
        ``IndexSpace`` this is the full-resolution **index-space** map from
        fixed-voxel to moving-voxel coordinates (``== model.exp(params)``);
        in ``WorldSpace`` it is the **world->world** transform (fixed-world
        to moving-world, mm).
    params
        The transform's Lie parameters at full resolution (voxel units in
        ``IndexSpace``; physical units in ``WorldSpace``).
    warped
        ``moving`` resampled onto the ``fixed`` grid by the recovered
        transform.
    cost_history
        Concatenated per-level optimiser cost traces.
    """

    matrix: Float[Array, 'd1 d1']
    params: Float[Array, ' p']
    warped: Float[Array, '*spatial']
    cost_history: Float[Array, ' h']


def resolve_iterations(
    iterations: Union[int, Sequence[int]], levels: int
) -> list[int]:
    """Per-pyramid-level iteration counts (finest first).

    ``int`` -> the same count at every level; a length-``levels`` sequence
    (**coarse-to-fine**, the natural schedule order) -> one count per level,
    reversed to the finest-first pyramid indexing the coarse-to-fine drivers
    use.  Shared by the forward (``register_core``) and inverse-compositional
    (``ic_register_core``) paths.
    """
    if isinstance(iterations, int):
        return [iterations] * levels
    seq = list(iterations)
    if len(seq) != levels:
        raise ValueError(
            f'iterations schedule must have length levels={levels}; '
            f'got {len(seq)}.'
        )
    return list(reversed(seq))


def _warp(
    sampler: _Sampler,
    moving: Array,
    transform: Array,
    *,
    fixed_shape: tuple[int, ...],
    moving_shape: tuple[int, ...],
    spec: RegistrationSpec,
) -> Array:
    """Warp a single-channel ``moving`` onto ``fixed_shape``.

    Shared by both coordinate spaces: the space resolves the parameter
    ``transform`` into the ``fixed-voxel -> moving-voxel`` sampling matrix
    and grid centre; the sampling itself (boundary, interpolation) is the
    same.
    """
    matrix, center = sampler.index_sampling(
        transform, fixed_shape=fixed_shape, moving_shape=moving_shape
    )
    grid = affine_grid(matrix, fixed_shape, center=center)
    warped = spatial_transform(
        moving[..., None],
        grid,
        mode=spec.boundary_mode,
        cval=spec.cval,
        method=spec.interpolation,
    )
    return warped[..., 0]


def optimize_objective(
    objective: Objective,
    params: Array,
    *,
    optimizer: str,
    iterations: int,
    cg_tol: float,
) -> tuple[Array, Array]:
    """Minimise an ``Objective`` over ``params``; return ``(params, history)``.

    The optimiser dispatch shared by every recipe and coordinate space: a
    least-squares objective routes to the matrix-free Gauss-Newton /
    Levenberg-Marquardt path; any other to BFGS on the scalar cost.  The
    objective closes over its own data (an image pair + warp, boundary
    samples, ...), so this function is objective-agnostic.
    """
    use_lsq = objective.is_least_squares and optimizer in ('auto', 'lm', 'gn')
    if use_lsq:
        if optimizer == 'gn':
            res = gauss_newton(
                objective.residual, params, n_iters=iterations, cg_tol=cg_tol
            )
        else:
            res = levenberg_marquardt(
                objective.residual, params, n_iters=iterations, cg_tol=cg_tol
            )
        return res.params, res.cost_history

    init_cost = objective.cost(params)
    out = minimize(
        objective.cost, params, method='BFGS', options={'maxiter': iterations}
    )
    return out.x, jnp.stack([init_cost, out.fun])


def register_core(
    moving: Float[Array, '*mspatial'],
    pyr_f: tuple[Float[Array, '*fspatial 1'], ...],
    *,
    model: TransformModel,
    ndim: int,
    spec: RegistrationSpec,
    space: CoordinateSpace,
    sampler: _Sampler,
    init_params: Float[Array, ' p'],
) -> RegistrationResult:
    """Coarse-to-fine register ``moving`` against a precomputed reference.

    The per-image core of the driver: the **reference** pyramid ``pyr_f``
    and the ``sampler`` are built once by the caller and passed in, so a
    batched recipe (``volreg``) can compute the shared reference work once
    and ``vmap`` only this core over a series of moving images.  Builds
    the moving pyramid, runs the coarse-to-fine optimise, and finalises.
    """
    dtype = moving.dtype
    pyr_m = gaussian_pyramid(
        moving[..., None],
        levels=spec.levels,
        factor=spec.pyramid_factor,
        sigma=spec.pyramid_sigma,
    )
    full_fixed_shape = pyr_f[0].shape[:-1]
    params = init_params
    histories = []
    iters_per_level = resolve_iterations(spec.iterations, spec.levels)
    prev_fixed_shape: Optional[tuple[int, ...]] = None
    # Coarsest (highest index) to finest (0).
    for level in range(spec.levels - 1, -1, -1):
        m_l = pyr_m[level][..., 0]
        f_l = pyr_f[level][..., 0]
        f_shape = f_l.shape
        m_shape = m_l.shape
        if space.requires_grid_rescale and prev_fixed_shape is not None:
            # Voxel-unit translations: rescale to this (finer) grid.
            ratio = jnp.asarray(f_shape, dtype=dtype) / jnp.asarray(
                prev_fixed_shape, dtype=dtype
            )
            params = model.rescale_to_grid(params, ratio)

        def warp_fn(
            p: Array,
            m_l: Array = m_l,
            f_shape: tuple[int, ...] = f_shape,
            m_shape: tuple[int, ...] = m_shape,
        ) -> Array:
            return _warp(
                sampler,
                m_l,
                model.exp(p, ndim=ndim),
                fixed_shape=f_shape,
                moving_shape=m_shape,
                spec=spec,
            )

        objective = MetricObjective(
            metric=spec.metric, warp=warp_fn, fixed=f_l
        )
        params, hist = optimize_objective(
            objective,
            params,
            optimizer=spec.optimizer,
            iterations=iters_per_level[level],
            cg_tol=spec.cg_tol,
        )
        histories.append(hist)
        prev_fixed_shape = f_shape

    transform = model.exp(params, ndim=ndim)
    matrix = sampler.result_transform(transform)
    warped = _warp(
        sampler,
        moving,
        transform,
        fixed_shape=full_fixed_shape,
        moving_shape=moving.shape,
        spec=spec,
    )
    return RegistrationResult(
        matrix=matrix,
        params=params,
        warped=warped,
        cost_history=jnp.concatenate(histories),
    )


def multi_resolution_register(
    moving: Float[Array, '*mspatial'],
    fixed: Float[Array, '*fspatial'],
    *,
    model: TransformModel,
    ndim: int,
    spec: RegistrationSpec,
    init_params: Optional[Float[Array, ' p']] = None,
    space: CoordinateSpace = IndexSpace(),
) -> RegistrationResult:
    """Coarse-to-fine registration driver shared by the recipes."""
    if moving.ndim != ndim or fixed.ndim != ndim:
        raise ValueError(
            f'expected {ndim}-D single-channel images; got moving '
            f'{moving.shape}, fixed {fixed.shape}.'
        )
    dtype = moving.dtype
    pyr_f = gaussian_pyramid(
        fixed[..., None],
        levels=spec.levels,
        factor=spec.pyramid_factor,
        sigma=spec.pyramid_sigma,
    )
    sampler = space.sampler(
        ndim=ndim,
        full_fixed_shape=fixed.shape,
        full_moving_shape=moving.shape,
        dtype=dtype,
    )
    params = (
        jnp.zeros(model.n_params(ndim), dtype=dtype)
        if init_params is None
        else init_params
    )
    return register_core(
        moving,
        pyr_f,
        model=model,
        ndim=ndim,
        spec=spec,
        space=space,
        sampler=sampler,
        init_params=params,
    )
