# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Diffeomorphic registration recipe: log-domain Demons.

The representative diffeomorphic algorithm (Vercauteren et al. 2009),
parametrised by a **stationary velocity field** ``v`` whose exponential
``φ = exp(v)`` (scaling-and-squaring, ``integrate_velocity_field``) is a
diffeomorphism by construction.  Each iteration is operator splitting:

1. **ESM demons force** -- a closed-form per-voxel velocity update
   ``u = (F − M∘φ) · J / (|J|² + α²(F − M∘φ)²)`` with the symmetric
   gradient ``J = ½(∇F + ∇(M∘φ))`` (efficient symmetric forces).  No
   inner solve -- this is what makes the diffeomorphic recipe as
   GPU-clean as it is.
2. **Fluid regularisation** -- Gaussian-smooth the update ``u``.
3. **Log update** -- ``v ← v + u`` (additive; BCH ``+ ½[v,u]`` optional).
4. **Diffusion regularisation** -- Gaussian-smooth the velocity ``v``.

Coarse-to-fine over a Gaussian pyramid.  The Gaussian smoothings are the
Green's functions of the fluid/diffusion regularisers; ``α`` normalises
the force.  Pure composition of the substrate (SVF exp, warp, gradient,
Gaussian, velocity algebra) -- the only metric-specific piece is the
closed-form force.  An LNCC-driven (SyN-style) force and a symmetric
forward+inverse variant are the documented upgrade paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..geometry import (
    compose_velocity,
    gaussian_pyramid,
    identity_grid,
    integrate_velocity_field,
    jacobian_det_displacement,
    spatial_gradient,
    spatial_transform,
    upsample,
)
from ..geometry._interpolate import BoundaryMode
from ..smoothing import gaussian

__all__ = [
    'DemonsSpec',
    'DiffeomorphicResult',
    'diffeomorphic_demons_register',
]


@dataclass(frozen=True)
class DemonsSpec:
    """Static configuration for the log-Demons recipe (jit-static).

    Attributes
    ----------
    levels
        Gaussian-pyramid resolutions (coarse-to-fine).
    iterations
        Demons iterations per level.
    sigma_fluid
        Gaussian sigma for the fluid (per-update) regularisation.
    sigma_diffusion
        Gaussian sigma for the diffusion (accumulated-velocity)
        regularisation.
    n_steps
        Scaling-and-squaring steps for ``exp(v)``.
    alpha
        Force normalisation: ``denom = |J|² + α²(F − M∘φ)²``.  Larger
        ``α`` damps the step where the intensity difference is large.
    bch_order
        Log-update order: ``1`` additive (default), ``2`` adds ½[v,u].
    pyramid_factor, pyramid_sigma
        Pyramid downsample factor / anti-alias sigma.
    boundary_mode
        Out-of-bounds handling for the warps.
    """

    levels: int = 3
    iterations: int = 80
    sigma_fluid: float = 1.0
    sigma_diffusion: float = 1.5
    n_steps: int = 6
    alpha: float = 0.4
    bch_order: int = 1
    pyramid_factor: float = 2.0
    pyramid_sigma: Optional[float] = None
    boundary_mode: BoundaryMode = 'nearest'


class DiffeomorphicResult(NamedTuple):
    """Output of the diffeomorphic Demons recipe.

    Attributes
    ----------
    velocity
        The stationary velocity field, ``(*spatial, ndim)``.
    displacement
        ``exp(velocity)`` as a displacement field (add ``identity_grid``
        for the absolute deformation).
    warped
        ``moving`` resampled by the deformation onto the ``fixed`` grid.
    jacobian_det
        ``det J`` of the displacement -- the diffeomorphism QA map (all
        positive ⇒ no folding).
    cost_history
        Concatenated per-iteration SSD trace.
    """

    velocity: Float[Array, '*spatial ndim']
    displacement: Float[Array, '*spatial ndim']
    warped: Float[Array, '*spatial']
    jacobian_det: Float[Array, '*spatial']
    cost_history: Float[Array, ' h']


def _smooth_vector(field: Array, sigma: float, ndim: int) -> Array:
    """Separable Gaussian over the spatial axes of a channel-last field."""
    moved = jnp.moveaxis(field, -1, 0)
    smoothed = gaussian(moved, sigma=sigma, spatial_rank=ndim)
    return jnp.moveaxis(smoothed, 0, -1)


def _demons_level(
    moving: Array,
    fixed: Array,
    v: Array,
    *,
    spec: DemonsSpec,
    ndim: int,
) -> tuple[Array, Array]:
    """Run the Demons iterations on one resolution; return ``(v, costs)``.

    The iteration is rolled with ``lax.scan`` (carry = the velocity
    field, per-step output = the SSD), so the level compiles one
    iteration rather than ``spec.iterations`` unrolled copies; the loop
    stays differentiable for the unrolled gradient path.
    """
    id_grid = identity_grid(fixed.shape, dtype=fixed.dtype)
    grad_fixed = spatial_gradient(fixed)
    alpha2 = spec.alpha * spec.alpha

    def step(v: Array, _: Any) -> tuple[Array, Array]:
        s = integrate_velocity_field(
            v, n_steps=spec.n_steps, mode=spec.boundary_mode
        )
        warped = spatial_transform(
            moving[..., None], id_grid + s, mode=spec.boundary_mode
        )[..., 0]
        diff = fixed - warped
        j = 0.5 * (grad_fixed + spatial_gradient(warped))
        denom = jnp.sum(j * j, axis=-1) + alpha2 * diff * diff
        u = (diff / denom)[..., None] * j
        u = _smooth_vector(u, spec.sigma_fluid, ndim)
        v = (
            v + u
            if spec.bch_order == 1
            else compose_velocity(v, u, order=spec.bch_order)
        )
        v = _smooth_vector(v, spec.sigma_diffusion, ndim)
        return v, 0.5 * jnp.sum(diff * diff)

    v, costs = jax.lax.scan(step, v, xs=None, length=spec.iterations)
    return v, costs


def diffeomorphic_demons_register(
    moving: Float[Array, '*spatial'],
    fixed: Float[Array, '*spatial'],
    *,
    spec: DemonsSpec = DemonsSpec(),
) -> DiffeomorphicResult:
    """Diffeomorphic registration of ``moving`` to ``fixed`` (log-Demons).

    Estimates a stationary velocity field ``v`` (coarse-to-fine) whose
    exponential warps ``moving`` onto ``fixed``.  The result is a
    diffeomorphism by construction; ``jacobian_det`` lets the caller
    assert no folding (all positive).

    Parameters
    ----------
    moving, fixed
        Single-channel images of identical shape (2-D or 3-D).
    spec
        ``DemonsSpec`` controlling the schedule and regularisation.

    Returns
    -------
    ``DiffeomorphicResult`` (``velocity``, ``displacement``, ``warped``,
    ``jacobian_det``, ``cost_history``).
    """
    if moving.shape != fixed.shape:
        raise ValueError(
            f'moving and fixed must share shape; got {moving.shape} '
            f'vs {fixed.shape}.'
        )
    ndim = moving.ndim
    if ndim not in (2, 3):
        raise ValueError(
            f'diffeomorphic registration supports 2-D / 3-D '
            f'single-channel images; got shape {moving.shape}.'
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

    velocity: Optional[Array] = None
    prev_shape: Optional[tuple[int, ...]] = None
    histories = []
    for level in range(spec.levels - 1, -1, -1):
        m_l = pyr_m[level][..., 0]
        f_l = pyr_f[level][..., 0]
        shape_l = f_l.shape
        if velocity is None:
            velocity = jnp.zeros(shape_l + (ndim,), dtype=dtype)
        else:
            velocity = upsample(velocity, shape_l)
            ratio = jnp.asarray(shape_l, dtype=dtype) / jnp.asarray(
                prev_shape, dtype=dtype
            )
            velocity = velocity * ratio
        velocity, hist = _demons_level(
            m_l, f_l, velocity, spec=spec, ndim=ndim
        )
        histories.append(hist)
        prev_shape = shape_l

    assert velocity is not None
    id_grid = identity_grid(moving.shape, dtype=dtype)
    s = integrate_velocity_field(
        velocity, n_steps=spec.n_steps, mode=spec.boundary_mode
    )
    warped = spatial_transform(
        moving[..., None], id_grid + s, mode=spec.boundary_mode
    )[..., 0]
    det = jacobian_det_displacement(s)
    return DiffeomorphicResult(
        velocity=velocity,
        displacement=s,
        warped=warped,
        jacobian_det=det,
        cost_history=jnp.concatenate(histories),
    )
