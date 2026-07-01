# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Volumetric boundary-based registration (BBR).

Align a volume to a tissue boundary (the white-matter / grey-matter
surface) by maximising the image contrast *across* that boundary.  Where
the intensity recipes match an image pair, BBR has **no fixed image** --
its cost is over boundary-point samples, so it is a *sibling* objective
(:class:`Objective`), not a :class:`Metric`, reusing the rigid
:class:`TransformModel` and the point sampler.

For each boundary point the moving intensity is sampled a short distance
:math:`\pm\text{step}` along the (transformed) surface normal -- one sample
each side of the boundary -- and the normalised cross-boundary contrast
:math:`Q = (I_{\text{out}} - I_{\text{in}}) / (\tfrac{1}{2}(I_{\text{in}} +
I_{\text{out}}))` is formed.  The cost :math:`\operatorname{mean}(1 +
\tanh(\text{slope} \cdot (Q - q_0)))` is low where the boundary sits on a
strong, consistently-oriented intensity edge.  :math:`Q` is a *ratio*
(invariant to a global intensity scale), so the recovery is image-amplitude
independent.

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

A reverse-mode gradient of the *optimum* with respect to the moving image is
best taken through the implicit-function layer
(:func:`~nitrix.linalg.implicit_minimize` on :func:`bbr_cost`), whose adjoint
is solved at the optimum and is independent of the unrolled trajectory --
preferred over differentiating the fixed-scan trajectory.

Surfaces as a *data structure* are out of scope: BBR consumes the boundary
**points** and **normals** as arrays, in the moving image's world frame (or
voxel frame when no affine is given).  ``moving_affine`` (voxel-to-world) makes
``step`` a physical (mm) distance and the normals physical directions --
correct under anisotropy.

References
----------
Greve, D. N., & Fischl, B. (2009). Accurate and robust brain image alignment
using boundary-based registration. *NeuroImage*, 48(1), 63-72.
https://doi.org/10.1016/j.neuroimage.2009.06.060
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
from ._converge import (
    Convergence,
    ConvergenceMode,
    resolve_convergence_mode,
    run_iterations,
)
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
    r"""Boundary-based (Greve-Fischl) contrast cost at rigid parameters.

    Evaluate the boundary cost for the rigid transform encoded by ``params``.
    Each boundary point is mapped through the transform into the moving voxel
    frame, sampled a distance ``step`` either side of the boundary along the
    transformed normal, and the normalised cross-boundary contrast is formed;
    the cost is low where the boundary sits on a strong, consistently-oriented
    intensity edge.  This is a pure function, differentiable with respect to
    both ``moving`` and ``params``, and is the core that both
    :class:`BoundaryObjective` and the implicit-differentiation layer call.

    Parameters
    ----------
    moving
        Single-channel volume being registered, of spatial shape ``*spatial``
        (2-D or 3-D).
    points
        ``(n, d)`` boundary sample coordinates, in the moving world frame (or
        voxel frame when the affine is the identity).
    normals
        ``(n, d)`` outward boundary normals, one per point.
    params
        ``(p,)`` rigid Lie parameters passed to ``model.exp``.
    model
        Transform model producing a homogeneous matrix from ``params`` via
        ``model.exp(params, ndim=ndim)``.
    ndim
        Spatial dimensionality (2 or 3).
    moving_affine_inv
        ``(d1, d1)`` world-to-voxel affine of the moving image (the inverse of
        its voxel-to-world affine, with ``d1 = ndim + 1``).
    step
        Half-distance sampled either side of the boundary along the normal, in
        the units of the moving affine (mm with a physical affine, else voxels).
    slope
        Sharpness of the :math:`\tanh` contrast cost.
    q0
        Contrast offset (the cost's inflection point).
    method
        Interpolator used to sample the moving volume at the offset points.
    mode
        Out-of-bounds boundary handling for the samples.
    cval
        Constant fill value used with a constant boundary mode.
    eps
        Guard added to the contrast denominator to avoid division by zero.

    Returns
    -------
    Float[Array, '']
        The scalar mean boundary cost over all points.
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
    r"""The BBR cost packaged as an :class:`Objective` (with no fixed image).

    Closes over the moving image, the boundary points and normals, and the
    sampling configuration; :meth:`cost` evaluates :func:`bbr_cost` at the
    given parameters.  This is not a least-squares objective (the
    :math:`\tanh` cost is not a sum of squares), so :attr:`is_least_squares`
    is ``False`` and :meth:`residual` raises.
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
        """Boundary cost at the rigid parameters ``params``.

        Parameters
        ----------
        params
            ``(p,)`` rigid Lie parameters.

        Returns
        -------
        Float[Array, '']
            The scalar boundary cost, as computed by :func:`bbr_cost` with
            this objective's closed-over image and configuration.
        """
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
        """Not defined for the BBR objective.

        The boundary cost is not a sum of squares, so it has no residual
        vector.  This method always raises :class:`NotImplementedError`.

        Parameters
        ----------
        params
            ``(p,)`` rigid Lie parameters (unused).

        Raises
        ------
        NotImplementedError
            Always, since BBR is not a least-squares objective.
        """
        raise NotImplementedError('BBR is not a least-squares objective.')


@dataclass(frozen=True)
class BBRSearch:
    r"""Coarse grid multistart for the BBR basin (FSL ``gridmeasurecost``).

    BBR's cost is markedly non-convex; a local solve from a single start can
    latch a *wrong* boundary alignment (and, with the wide-basin coarse
    ``step``, descend confidently into it).  Before the local refine, the cost
    is evaluated on a grid of rigid seeds around ``init_params`` -- a
    Cartesian product of ``steps`` offsets per degree of freedom over
    :math:`\pm` ``rotation`` (rad) on each rotation DOF and :math:`\pm`
    ``translation`` (mm / voxels) on each translation DOF -- and the
    lowest-cost seed initialises the
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
        Iteration strategy, the same toggle as the other recipes.
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
        Sampling kernel (differentiable in the coordinate -- :class:`Linear`
        default).
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
    """Evenly-spaced grid offsets along a single degree of freedom.

    Parameters
    ----------
    extent
        Half-extent of the axis: offsets span ``[-extent, extent]``.
    steps
        Number of offsets. ``steps <= 1`` collapses the axis to a single
        zero offset (disabling it).
    dtype
        Dtype of the returned offsets.

    Returns
    -------
    Array
        ``(steps,)`` offsets linearly spaced in ``[-extent, extent]``, or a
        single ``(1,)`` zero offset when ``steps <= 1``.
    """
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
    """Lowest-cost rigid seed on a coarse grid around ``init``.

    Builds a Cartesian grid of parameter offsets around ``init`` (rotation
    offsets on the rotation degrees of freedom, translation offsets on the
    translation ones), evaluates the objective cost at every seed under a
    single ``vmap``, and returns the seed with the lowest (non-NaN) cost. The
    returned seed is detached from the gradient, since it only initialises the
    subsequent local solve.

    Parameters
    ----------
    objective
        Boundary objective whose ``cost`` is evaluated at each seed.
    init
        ``(p,)`` centre of the grid (the warm-start rigid parameters).
    n_rot
        Number of rotation degrees of freedom (1 in 2-D, 3 in 3-D).
    ndim
        Spatial dimensionality (2 or 3); the number of translation degrees of
        freedom.
    search
        Grid configuration (extents and number of steps per axis).

    Returns
    -------
    Array
        ``(p,)`` lowest-cost rigid seed, with the gradient stopped.
    """
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
    """One annealing stage: normalised-block gradient descent.

    Runs a single sampling-distance stage of the BBR solve through
    :func:`run_iterations`.  Each step unit-normalises the rotation and
    translation gradient blocks separately and moves by ``rotation_step`` /
    ``translation_step`` (physical rad / mm), geometrically decayed by
    ``decay`` -- decoupling the step from the image amplitude and from the
    rad-vs-mm unit mismatch.

    Parameters
    ----------
    objective
        Boundary objective for this stage (fixes the sampling distance).
    init_params
        ``(p,)`` warm-start rigid parameters for the stage.
    n_rot
        Number of rotation degrees of freedom, used to split the gradient into
        its rotation and translation blocks.
    iterations
        Maximum gradient-descent iterations for the stage.
    convergence
        Early-exit criterion (threshold / window) for the windowed
        ``lax.while_loop``, or ``None`` to run the fixed ``lax.scan``.
    rotation_step
        Per-iteration step length for the rotation block (radians).
    translation_step
        Per-iteration step length for the translation block (mm / voxels).
    decay
        Geometric per-iteration decay factor applied to both step lengths.
    dtype
        Working dtype of the parameters and cost history.

    Returns
    -------
    params : Array
        ``(p,)`` refined rigid parameters after the stage.
    history : Array
        The per-iteration cost trace for the stage.
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
    r"""Boundary-based rigid registration of ``moving`` to a surface.

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
        ``(n, d)`` boundary sample coordinates and unit outward normals,
        in the moving world frame (or voxel frame when ``moving_affine`` is
        ``None``).  Normals oriented so the interior (the :math:`-`\ ``normal``
        side) is the brighter tissue.
    moving_affine
        ``(d1, d1)`` voxel-to-world affine of ``moving`` (with
        ``d1 = ndim + 1``); ``None`` uses the identity (voxel frame).  Makes
        ``step`` physical and the normals physical directions.
    init_params
        ``(p,)`` rigid Lie-parameter warm start (default identity); the grid
        search, when enabled, seeds *around* it.
    spec
        The :class:`BBRSpec` static configuration (sampling distance, cost
        shape, annealing schedule, grid multistart, and optimiser knobs).

    Returns
    -------
    BBRResult
        The recovered transform ``matrix``, its rigid ``params``, the final
        ``cost``, and the concatenated per-stage ``cost_history``.
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
