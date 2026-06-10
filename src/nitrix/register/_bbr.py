# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Volumetric boundary-based registration (BBR).

Greve & Fischl (2009): align a volume to a tissue boundary (the WM/GM
surface) by maximising the image contrast *across* that boundary.  Where
the intensity recipes match an image pair, BBR has **no fixed image** --
its cost is over boundary-point samples, so it is a *sibling* objective
(:class:`._objective.Objective`), not a ``Metric``, reusing the rigid
``TransformModel``, the point sampler, and the shared optimiser dispatch.

For each boundary point the moving intensity is sampled a short distance
``±step`` along the (transformed) surface normal -- one sample each side of
the boundary -- and the normalised cross-boundary contrast ``Q = (I_out −
I_in) / (½(I_in + I_out))`` is formed.  The cost ``mean(1 + tanh(slope·(Q −
q0)))`` is low where the boundary sits on a strong, consistently-oriented
intensity edge.  Optimised over the rigid parameters (BFGS, via
``optimize_objective``); differentiable w.r.t. the moving image through the
sampler (``implicit_minimize`` for the exact-at-the-optimum layer).

Surfaces as a *data structure* are out of scope (a ``thrux`` / surface-
features concern): BBR consumes the boundary **points** and **normals** as
arrays, in the moving image's world frame (or voxel frame when no affine is
given).  ``moving_affine`` (voxel->world) makes ``step`` a physical (mm)
distance and the normals physical directions -- correct under anisotropy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, NamedTuple, Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..geometry import apply_affine, sample_at_points
from ..geometry._interpolate import BoundaryMode, Interpolator, Linear
from ..linalg._solver import safe_inv
from ._core import optimize_objective
from ._model import Rigid, TransformModel

__all__ = [
    'BBRSpec',
    'BBRResult',
    'BoundaryObjective',
    'bbr_cost',
    'bbr_register',
]


def bbr_cost(
    moving: Float[Array, '*spatial'],
    points: Float[Array, ' n d'],
    normals: Float[Array, ' n d'],
    params: Float[Array, ' p'],
    *,
    model: TransformModel,
    ndim: int,
    moving_affine_inv: Float[Array, ' d1 d1'],
    step: float,
    slope: float,
    q0: float,
    method: Interpolator,
    mode: BoundaryMode,
    cval: float,
    eps: float,
) -> Float[Array, '']:
    """Greve-Fischl boundary cost at the rigid parameters ``params``.

    Pure function (differentiable w.r.t. ``moving`` and ``params``): the
    core both :class:`BoundaryObjective` and the implicit-diff layer call.
    """
    transform = model.exp(params, ndim=ndim)
    # Rotate about the boundary centroid (not the origin), so the rotation
    # parameter's leverage is the structure radius -- not the distance to
    # the voxel origin, which would dominate the conditioning.  Surface
    # point -> moving world -> moving voxel.
    center = jnp.mean(points, axis=0)
    p_world = apply_affine(points, transform, center=center)
    q = apply_affine(p_world, moving_affine_inv)
    # Physical normal -> moving voxel direction (its magnitude carries the
    # mm->voxel scale, so ``step`` stays a physical distance).
    direction = moving_affine_inv[:ndim, :ndim] @ transform[:ndim, :ndim]
    n_vox = normals @ direction.T
    inside = q - step * n_vox
    outside = q + step * n_vox
    i_in = sample_at_points(
        moving, inside, method=method, mode=mode, cval=cval
    )
    i_out = sample_at_points(
        moving, outside, method=method, mode=mode, cval=cval
    )
    contrast = (i_out - i_in) / (0.5 * (i_in + i_out) + eps)
    return jnp.mean(1.0 + jnp.tanh(slope * (contrast - q0)))


@dataclass(frozen=True)
class BoundaryObjective:
    """The BBR cost as an :class:`._objective.Objective` (no ``fixed``).

    Closes over the moving image, the boundary points / normals, and the
    sampling configuration; ``cost(params)`` is :func:`bbr_cost`.  Not a
    least-squares objective (the tanh cost is not a sum of squares), so the
    shared dispatch routes it to BFGS.
    """

    moving: Float[Array, '*spatial']
    points: Float[Array, ' n d']
    normals: Float[Array, ' n d']
    moving_affine_inv: Float[Array, ' d1 d1']
    model: TransformModel
    ndim: int
    step: float
    slope: float
    q0: float
    method: Interpolator
    mode: BoundaryMode
    cval: float
    eps: float

    is_least_squares: ClassVar[bool] = False

    def cost(self, params: Float[Array, ' p']) -> Float[Array, '']:
        return bbr_cost(
            self.moving,
            self.points,
            self.normals,
            params,
            model=self.model,
            ndim=self.ndim,
            moving_affine_inv=self.moving_affine_inv,
            step=self.step,
            slope=self.slope,
            q0=self.q0,
            method=self.method,
            mode=self.mode,
            cval=self.cval,
            eps=self.eps,
        )

    def residual(self, params: Float[Array, ' p']) -> Float[Array, ' m']:
        raise NotImplementedError('BBR is not a least-squares objective.')


@dataclass(frozen=True)
class BBRSpec:
    """Static configuration for :func:`bbr_register`.

    Attributes
    ----------
    step
        Half-distance sampled across the boundary along the normal, in the
        units of ``moving_affine`` (mm when an affine is given, else
        voxels).
    slope
        Sharpness of the ``tanh`` contrast cost.
    q0
        Contrast offset (the cost's inflection).
    iterations
        BFGS iterations.
    interpolation
        Sampling kernel (must be differentiable in the coordinate --
        ``Linear`` default).
    boundary_mode, cval
        Out-of-bounds handling for the samples (``"nearest"`` default, so
        an off-edge sample clamps rather than reads spurious zero
        contrast).
    eps
        Guard on the contrast denominator.
    """

    step: float = 1.0
    slope: float = 0.5
    q0: float = 0.0
    iterations: int = 100
    interpolation: Interpolator = Linear()
    boundary_mode: BoundaryMode = 'nearest'
    cval: float = 0.0
    eps: float = 1e-3


class BBRResult(NamedTuple):
    """Output of :func:`bbr_register`.

    Attributes
    ----------
    matrix
        The recovered homogeneous rigid transform mapping the surface frame
        to the moving frame -- world->world when ``moving_affine`` is given,
        else index-space.
    params
        The rigid Lie parameters.
    cost
        The final boundary cost.
    cost_history
        The ``[initial, final]`` cost.
    """

    matrix: Float[Array, ' d1 d1']
    params: Float[Array, ' p']
    cost: Float[Array, '']
    cost_history: Float[Array, ' h']


def bbr_register(
    moving: Float[Array, '*spatial'],
    points: Float[Array, ' n d'],
    normals: Float[Array, ' n d'],
    *,
    moving_affine: Optional[Float[Array, ' d1 d1']] = None,
    init_params: Optional[Float[Array, ' p']] = None,
    spec: BBRSpec = BBRSpec(),
) -> BBRResult:
    """Boundary-based rigid registration of ``moving`` to a surface.

    Estimates the rigid transform that places the boundary ``points`` (with
    outward ``normals``) on a strong, consistently-oriented intensity edge
    of ``moving`` (the Greve-Fischl cost).  BBR is a *local refinement*
    (its basin is narrow) -- pass ``init_params`` from a prior affine for a
    large initial misalignment.

    Parameters
    ----------
    moving
        Single-channel volume (2-D or 3-D) being registered.
    points, normals
        ``(N, ndim)`` boundary sample coordinates and unit outward normals,
        in the moving world frame (or voxel frame when ``moving_affine`` is
        ``None``).  Normals oriented so the interior (``−normal``) side is
        the brighter tissue.
    moving_affine
        Voxel->world affine of ``moving`` ``(ndim+1, ndim+1)``; ``None``
        -> identity (voxel frame).  Makes ``step`` physical and the normals
        physical directions.
    init_params
        Rigid Lie-parameter warm start (default identity).
    spec
        ``BBRSpec``.

    Returns
    -------
    ``BBRResult`` (``matrix``, ``params``, ``cost``, ``cost_history``).
    """
    ndim = points.shape[-1]
    if ndim not in (2, 3):
        raise ValueError(f'BBR supports 2-D / 3-D; got points {points.shape}.')
    if normals.shape != points.shape:
        raise ValueError(
            f'normals {normals.shape} must match points {points.shape}.'
        )
    if moving.ndim != ndim:
        raise ValueError(
            f'moving must be a {ndim}-D single-channel volume; got '
            f'{moving.shape}.'
        )
    dtype = moving.dtype
    model = Rigid()
    if moving_affine is None:
        a_m_inv = jnp.eye(ndim + 1, dtype=dtype)
    else:
        a_m_inv = safe_inv(jnp.asarray(moving_affine, dtype=dtype))
    init = (
        jnp.zeros(model.n_params(ndim), dtype=dtype)
        if init_params is None
        else init_params
    )
    objective = BoundaryObjective(
        moving=moving,
        points=jnp.asarray(points, dtype=dtype),
        normals=jnp.asarray(normals, dtype=dtype),
        moving_affine_inv=a_m_inv,
        model=model,
        ndim=ndim,
        step=spec.step,
        slope=spec.slope,
        q0=spec.q0,
        method=spec.interpolation,
        mode=spec.boundary_mode,
        cval=spec.cval,
        eps=spec.eps,
    )
    params, history = optimize_objective(
        objective,
        init,
        optimizer='auto',
        iterations=spec.iterations,
        cg_tol=1e-6,
    )
    # The applied transform rotates about the boundary centroid; return that
    # centred world->moving transform (self-contained).
    center = jnp.mean(objective.points, axis=0)
    eye = jnp.eye(ndim + 1, dtype=dtype)
    t_pos = eye.at[:ndim, ndim].set(center)
    t_neg = eye.at[:ndim, ndim].set(-center)
    matrix = t_pos @ model.exp(params, ndim=ndim) @ t_neg
    return BBRResult(
        matrix=matrix,
        params=params,
        cost=objective.cost(params),
        cost_history=history,
    )
