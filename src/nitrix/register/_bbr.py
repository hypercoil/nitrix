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
``TransformModel`` and the point sampler.

For each boundary point the moving intensity is sampled a short distance
``±step`` along the (transformed) surface normal -- one sample each side of
the boundary -- and the normalised cross-boundary contrast ``Q = (I_out −
I_in) / (½(I_in + I_out))`` is formed.  The cost ``mean(1 + tanh(slope·(Q −
q0)))`` is low where the boundary sits on a strong, consistently-oriented
intensity edge.  ``Q`` is a *ratio* (invariant to a global intensity scale),
so the recovery is image-amplitude independent.

**Recovery (the BBR basin is narrow and non-convex).**  Two ingredients,
mirroring FSL ``flirt --bbr`` (``gridmeasurecost`` + annealed ``bbrstep``),
make the recovery robust without a fixed-image pyramid:

* **Step annealing** (``BBRSpec.schedule``) -- the sampling distance ``step``
  is annealed coarse->fine.  A *large* ``step`` samples far either side of the
  boundary, which smooths the cost and widens its basin (it captures the gross
  translation a fine ``step`` is blind to); a small ``step`` then refines.
* **Grid multistart** (``BBRSpec.search``) -- before refining, the cost is
  evaluated on a coarse grid of rigid seeds (at the coarsest ``step``) and the
  lowest-cost seed initialises the solve, so a single local descent cannot
  latch a wrong boundary alignment.

**Optimiser + the early-exit / fixed toggle (aligned with the other recipes).**
The per-stage solve is a *normalised-block gradient descent*: the rotation and
translation gradient blocks are unit-normalised and stepped by physical (rad /
mm) step lengths with a geometric decay -- this decouples the step from the
image amplitude and from the rad-vs-mm unit mismatch (which defeats a naive
Newton / single-rate step on this non-convex cost).  It runs through the shared
:func:`._converge.run_iterations`, so ``BBRSpec.mode`` gives the same scan/while
toggle as every other path: ``'fixed'`` (the default) is the fixed ``lax.scan``
(reproducible, reverse-differentiable); ``'early_exit'`` is the windowed-slope
``lax.while_loop`` (opt in, a little faster, **not** reverse-differentiable),
parameterised by ``BBRSpec.convergence``.

A reverse-mode gradient of the *optimum* w.r.t. the moving image is best taken
through the implicit-function layer (``linalg.implicit_minimize`` on
:func:`bbr_cost`), whose adjoint is solved at the optimum and is independent of
the unrolled trajectory -- preferred over differentiating the fixed-scan
trajectory.

Surfaces as a *data structure* are out of scope (a ``thrux`` / surface-
features concern): BBR consumes the boundary **points** and **normals** as
arrays, in the moving image's world frame (or voxel frame when no affine is
given).  ``moving_affine`` (voxel->world) makes ``step`` a physical (mm)
distance and the normals physical directions -- correct under anisotropy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from ..geometry import apply_affine, sample_at_points
from ..geometry._interpolate import BoundaryMode, Interpolator, Linear
from ..linalg._solver import safe_inv
from ._converge import run_iterations
from ._core import Convergence, ConvergenceMode, resolve_convergence_mode
from ._model import Rigid, TransformModel

__all__ = [
    'BBRSpec',
    'BBRSearch',
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
    least-squares objective (the tanh cost is not a sum of squares).
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
class BBRSearch:
    """Coarse grid multistart for the BBR basin (FSL ``gridmeasurecost``).

    BBR's cost is markedly non-convex; a local solve from a single start can
    latch a *wrong* boundary alignment (and, with the wide-basin coarse
    ``step``, descend confidently into it).  Before the local refine, the cost
    is evaluated on a grid of rigid seeds around ``init_params`` -- a
    Cartesian product of ``steps`` offsets per degree of freedom over
    ``±rotation`` (rad) on each rotation DOF and ``±translation`` (mm /
    voxels) on each translation DOF -- and the lowest-cost seed initialises the
    optimise.  The capture range widens to roughly ``rotation`` / ``translation``
    per axis (a few mm / degrees -- BBR remains a *local* refinement of a prior
    affine, not a global search).

    Attributes
    ----------
    rotation
        Half-extent of the grid per rotation DOF (radians).
    translation
        Half-extent per translation DOF (``moving_affine`` units -- mm with an
        affine, else voxels).
    steps
        Grid points per axis (odd; ``1`` puts the only seed at ``init_params``,
        i.e. disables that axis).  The grid is ``steps ** n_params`` seeds
        evaluated under one ``vmap`` (``3 ** 6 = 729`` for a 3-D rigid at the
        default) -- cheap on a GPU; raise ``steps`` with care.
    step
        Sampling distance the grid cost uses (``None`` -> the coarsest
        ``schedule`` step, the widest basin).
    """

    rotation: float = 0.07
    translation: float = 4.0
    steps: int = 3
    step: Optional[float] = None


@dataclass(frozen=True)
class BBRSpec:
    """Static configuration for :func:`bbr_register`.

    Attributes
    ----------
    step
        Half-distance sampled across the boundary along the normal, in the
        units of ``moving_affine`` (mm when an affine is given, else voxels).
        Used only as the single stage when ``schedule`` is ``None``.
    slope
        Sharpness of the ``tanh`` contrast cost.
    q0
        Contrast offset (the cost's inflection).
    iterations
        Gradient-descent iterations **per annealing stage**.
    schedule
        Coarse->fine sampling-distance (``step``) annealing, e.g.
        ``(4.0, 2.0, 1.0)`` (the default).  A large coarse ``step`` smooths the
        cost and widens its basin (capturing gross translation); finer stages
        refine, warm-started from the previous.  ``None`` runs a single stage at
        ``step`` (the legacy behaviour).
    search
        :class:`BBRSearch` grid multistart seed (default on), or ``None`` for a
        pure local refine from ``init_params`` (use when a prior affine has
        already brought the boundary within the fine basin).
    mode
        Iteration strategy (B2), the same toggle as the other recipes.
        ``'fixed'`` (default) runs the fixed ``lax.scan`` (reproducible and
        reverse-differentiable); ``'early_exit'`` runs the windowed-slope
        ``lax.while_loop`` (opt in, a little faster on easy alignments, **not**
        reverse-differentiable).  ``'fixed'`` is the BBR default deliberately:
        the GD early-exit only beats the (already fast) scan at a loosened
        ``convergence.threshold``.
    convergence
        The :class:`Convergence` (threshold / window) for ``mode='early_exit'``;
        inert under ``mode='fixed'``.
    rotation_step, translation_step
        Per-iteration GD step lengths -- the rotation block (rad) and
        translation block (mm / voxels), applied to the unit-normalised
        gradient blocks.  Defaults suit a seed within ~one grid cell.
    step_decay
        Geometric per-iteration decay of the step lengths (``< 1``); shrinks the
        step as the stage converges.
    interpolation
        Sampling kernel (differentiable in the coordinate -- ``Linear`` default).
    boundary_mode, cval
        Out-of-bounds handling for the samples (``"nearest"`` default, so an
        off-edge sample clamps rather than reads spurious zero contrast).
    eps
        Guard on the contrast denominator.
    """

    step: float = 1.0
    slope: float = 0.5
    q0: float = 0.0
    iterations: int = 50
    schedule: Optional[tuple[float, ...]] = (4.0, 2.0, 1.0)
    search: Optional[BBRSearch] = BBRSearch()
    mode: ConvergenceMode = 'fixed'
    convergence: Convergence = Convergence()
    rotation_step: float = 0.03
    translation_step: float = 2.0
    step_decay: float = 0.92
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
        The final boundary cost (at the finest ``schedule`` step).
    cost_history
        The concatenated per-stage optimiser cost traces.  Note the absolute
        cost scale shifts between annealing stages (a finer ``step`` yields a
        higher cost), as with the per-level traces of the pyramid recipes -- it
        is monotone *within* a stage, not necessarily across the concatenation.
    """

    matrix: Float[Array, ' d1 d1']
    params: Float[Array, ' p']
    cost: Float[Array, '']
    cost_history: Float[Array, ' h']


def _grid_axis(extent: float, steps: int, dtype: Any) -> Array:
    """``steps`` offsets in ``[-extent, extent]`` (``[0]`` when ``steps<=1``)."""
    if steps <= 1:
        return jnp.zeros((1,), dtype=dtype)
    return jnp.linspace(-extent, extent, steps, dtype=dtype)


def _grid_seed(
    objective: BoundaryObjective,
    init: Array,
    *,
    n_rot: int,
    ndim: int,
    search: BBRSearch,
) -> Array:
    """Lowest-cost rigid seed on a coarse grid around ``init`` (vmap-ed)."""
    rot = _grid_axis(search.rotation, search.steps, init.dtype)
    tr = _grid_axis(search.translation, search.steps, init.dtype)
    axes = [rot] * n_rot + [tr] * ndim
    mesh = jnp.meshgrid(*axes, indexing='ij')
    grid = jnp.stack([m.reshape(-1) for m in mesh], axis=-1)
    candidates = init[None, :] + grid
    costs = jax.vmap(objective.cost)(candidates)
    # The seed only initialises the local solve; never back-propagate the
    # discrete argmin (the optimum is seed-independent up to the basin).  Guard
    # NaN candidate costs (an off-grid seed can sample all-cval) so a NaN never
    # wins the argmin.
    costs = jnp.where(jnp.isnan(costs), jnp.inf, costs)
    return lax.stop_gradient(candidates[jnp.argmin(costs)])


def _gd_stage(
    objective: BoundaryObjective,
    init_params: Array,
    *,
    n_rot: int,
    iterations: int,
    convergence: Optional[Convergence],
    rotation_step: float,
    translation_step: float,
    decay: float,
    dtype: Any,
) -> tuple[Array, Array]:
    """One annealing stage: normalised-block GD via :func:`run_iterations`.

    Each step unit-normalises the rotation / translation gradient blocks and
    moves by ``rotation_step`` / ``translation_step`` (physical rad / mm),
    geometrically decayed -- amplitude- and unit-robust.  ``convergence``
    selects the fixed scan (``None``) or the windowed early-exit while-loop.
    """
    cost = objective.cost
    grad_cost = jax.grad(cost)

    def step_fn(
        state: tuple[Array, Array], _: None
    ) -> tuple[tuple[Array, Array], Array]:
        params, i = state
        g = grad_cost(params)
        g_rot = g[:n_rot]
        g_tr = g[n_rot:]
        g_rot = g_rot / (jnp.linalg.norm(g_rot) + 1e-8)
        g_tr = g_tr / (jnp.linalg.norm(g_tr) + 1e-8)
        fac = decay ** i.astype(dtype)
        delta = jnp.concatenate(
            [rotation_step * fac * g_rot, translation_step * fac * g_tr]
        )
        params = params - delta
        return (params, i + 1), cost(params)

    (params, _), history = run_iterations(
        step_fn,
        (init_params, jnp.asarray(0)),
        iterations=iterations,
        convergence=convergence,
        dtype=dtype,
    )
    return params, history


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
    of ``moving`` (the Greve-Fischl cost).  BBR is a *local refinement* of a
    prior affine; the default grid multistart + step annealing widen the
    capture range to a few mm / degrees (see :class:`BBRSpec`).  Pass
    ``init_params`` from a prior affine for a larger initial misalignment.

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
        Rigid Lie-parameter warm start (default identity); the grid search,
        when enabled, seeds *around* it.
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
    n_rot = 1 if ndim == 2 else 3
    if moving_affine is None:
        a_m_inv = jnp.eye(ndim + 1, dtype=dtype)
    else:
        a_m_inv = safe_inv(jnp.asarray(moving_affine, dtype=dtype))
    init = (
        jnp.zeros(model.n_params(ndim), dtype=dtype)
        if init_params is None
        else jnp.asarray(init_params, dtype=dtype)
    )
    # BBR GD always supports the windowed while_loop; mode='fixed' (the default)
    # -> scan, mode='early_exit' -> the loop (opt-in: it needs a loosened
    # threshold to beat the already-fast scan).
    convergence = resolve_convergence_mode(
        spec.mode,
        spec.convergence,
        supports_early_exit=True,
        path='BBR',
    )

    points_a = jnp.asarray(points, dtype=dtype)
    normals_a = jnp.asarray(normals, dtype=dtype)

    def objective_at(step: float) -> BoundaryObjective:
        return BoundaryObjective(
            moving=moving,
            points=points_a,
            normals=normals_a,
            moving_affine_inv=a_m_inv,
            model=model,
            ndim=ndim,
            step=float(step),
            slope=spec.slope,
            q0=spec.q0,
            method=spec.interpolation,
            mode=spec.boundary_mode,
            cval=spec.cval,
            eps=spec.eps,
        )

    stages = spec.schedule if spec.schedule is not None else (spec.step,)

    params = init
    if spec.search is not None:
        grid_step = (
            spec.search.step if spec.search.step is not None else stages[0]
        )
        params = _grid_seed(
            objective_at(grid_step),
            params,
            n_rot=n_rot,
            ndim=ndim,
            search=spec.search,
        )

    histories = []
    for stage_step in stages:
        params, history = _gd_stage(
            objective_at(stage_step),
            params,
            n_rot=n_rot,
            iterations=spec.iterations,
            convergence=convergence,
            rotation_step=spec.rotation_step,
            translation_step=spec.translation_step,
            decay=spec.step_decay,
            dtype=dtype,
        )
        histories.append(history)
    cost_history = (
        jnp.concatenate(histories) if len(histories) > 1 else histories[0]
    )

    fine = objective_at(stages[-1])
    # The applied transform rotates about the boundary centroid; return that
    # centred world->moving transform (self-contained).
    center = jnp.mean(points_a, axis=0)
    eye = jnp.eye(ndim + 1, dtype=dtype)
    t_pos = eye.at[:ndim, ndim].set(center)
    t_neg = eye.at[:ndim, ndim].set(-center)
    matrix = t_pos @ model.exp(params, ndim=ndim) @ t_neg
    return BBRResult(
        matrix=matrix,
        params=params,
        cost=fine.cost(params),
        cost_history=cost_history,
    )
