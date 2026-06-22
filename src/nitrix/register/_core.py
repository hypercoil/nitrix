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

from dataclasses import dataclass, field, replace
from typing import (
    Callable,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Union,
    cast,
)

import jax
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
from ._metric import SSD, Metric, pin_metric_ranges
from ._model import TransformModel
from ._objective import MetricObjective, Objective
from ._space import CoordinateSpace, IndexSpace, _Sampler


@dataclass(frozen=True)
class Convergence:
    """Windowed cost-slope criterion for the ``mode='early_exit'`` iteration.

    The threshold/window pair that parameterises the early-exit ``lax.while_loop``
    (see :data:`ConvergenceMode`): per level, the loop runs until the cost has
    plateaued -- a least-squares-line fit over the last ``window`` per-iteration
    costs whose normalised slope falls below ``threshold`` -- or the level's
    iteration count (the hard cap) is reached.  It is **inert** under
    ``mode='fixed'`` (the fixed ``lax.scan``); the two are orthogonal fields
    (B2), so a spec always carries a concrete :class:`Convergence`, used only
    when ``mode='early_exit'``.

    Attributes
    ----------
    threshold
        Stop when the windowed normalised cost slope drops below this.
    window
        Number of recent costs the slope is fit over (the convergence window).

    Notes
    -----
    An early-exit forward is **not** reverse-differentiable (``lax.while_loop``
    has no reverse rule); for a differentiable registration layer use
    ``mode='fixed'`` (the ``scan``) or the implicit-function path
    (``linalg.implicit_least_squares`` / ``implicit_minimize``), whose adjoint is
    solved at the optimum and is trajectory-independent.
    """

    threshold: float = 1e-6
    window: int = 10


# The iteration strategy (B2), orthogonal to the :class:`Convergence`
# parameters: ``'fixed'`` is the reproducible, reverse-differentiable,
# ``vmap``-batchable ``lax.scan``; ``'early_exit'`` is the windowed-slope
# ``lax.while_loop`` (single-pair; not reverse-differentiable).  Eligibility is
# checked in the one ``resolve_convergence_mode`` gate below.
ConvergenceMode = Literal['fixed', 'early_exit']


def resolve_convergence_mode(
    mode: ConvergenceMode,
    convergence: Convergence,
    *,
    supports_early_exit: bool,
    path: str,
) -> Optional[Convergence]:
    """The single B2 eligibility gate: ``(mode, convergence)`` -> run-time control.

    Returns the concrete :class:`Convergence` to drive an early-exit
    ``lax.while_loop`` (``mode='early_exit'``), or ``None`` for the fixed
    ``lax.scan`` (``mode='fixed'`` -- the default, reproducible and
    reverse-differentiable on every path).  ``mode='early_exit'`` on a path that
    cannot honour a data-dependent trip count (``supports_early_exit=False`` --
    the scalar/BFGS forward optimiser is monolithic) raises a loud, actionable
    error naming ``path`` rather than failing obscurely downstream.  This is the
    one place the polysemy used to live (three sentinel meanings at three sites);
    every recipe now routes its mode decision through here.
    """
    if mode not in ('fixed', 'early_exit'):
        raise ValueError(
            f"mode must be 'fixed' or 'early_exit'; got {mode!r}."
        )
    if mode == 'fixed':
        return None
    if not supports_early_exit:
        raise ValueError(
            f"mode='early_exit' is not supported on {path} -- it cannot honour "
            "a data-dependent trip count.  Use mode='fixed' (the reproducible "
            'fixed scan), or the inverse-compositional recipes (IndexSpace + a '
            'least-squares metric such as SSD), which do.'
        )
    return convergence


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
    mode
        Iteration strategy (B2), orthogonal to ``convergence``.  ``'fixed'``
        (the spec default) runs the fixed ``lax.scan`` -- reproducible,
        reverse-differentiable, and ``vmap``-batchable (``volreg``).
        ``'early_exit'`` runs the windowed-slope ``lax.while_loop`` with
        ``iterations`` as the hard cap; it is single-pair (a ``vmap``-ed
        ``while_loop`` exits only when *all* lanes converge) and **not**
        reverse-differentiable (``jax.grad`` through it raises a loud, actionable
        error -- use ``mode='fixed'`` or the implicit-function path).  Rejected
        on the scalar/BFGS forward path (a non-least-squares metric), which is
        monolithic.  The spec default is ``'fixed'`` on **every** recipe
        (reproducible + differentiable out of the box); ``'early_exit'`` is the
        recommended opt-in for the single-pair inverse-compositional recipes
        (``rigid_register`` / ``affine_register``), which converge in a few of
        their iterations.
    convergence
        The :class:`Convergence` (threshold / window) parameterising
        ``mode='early_exit'``; **inert** under ``mode='fixed'``.  **cost_history
        (C6):** on the early-exit path the per-level trace is padded to the
        ``iterations`` cap with the final cost (so the shape is path-independent;
        the value is constant past the stop iteration).
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
    mode: ConvergenceMode = 'fixed'
    convergence: Convergence = field(default_factory=Convergence)


class RegistrationResult(NamedTuple):
    """Output of a registration recipe.

    Attributes
    ----------
    matrix
        The estimated homogeneous transform, ``(ndim + 1, ndim + 1)``,
        **self-contained**: the centre the warp applies it about is baked in, so
        ``apply_affine(coords, matrix)`` reproduces the warp and it composes
        with another result.  In ``IndexSpace`` it is the full-resolution
        fixed-voxel -> moving-voxel index map; in ``WorldSpace`` the
        fixed-world -> moving-world transform (mm).  (The raw about-origin
        ``model.exp(params)`` is recovered from ``params``.)
    params
        The transform's raw **about-origin** Lie parameters at full resolution
        (voxel units in ``IndexSpace``; physical units in ``WorldSpace``).
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


def _mask_jacobian(
    base: Callable[[Array], Array], sqrt_w: Array
) -> Callable[[Array], Array]:
    """Row-scale a residual Jacobian by ``sqrt(mask)`` (A3, the masked LSQ path).

    The masked SSD residual is ``sqrt(mask) * (warped - fixed)`` (see
    ``SSD.residual``); its Jacobian is therefore the unmasked Jacobian with each
    voxel row multiplied by ``sqrt(mask)``, so an out-of-mask voxel (weight 0)
    contributes a zero row and drops out of the Gauss-Newton normal equations.
    """
    rows = sqrt_w[:, None]

    def jacobian(params: Array) -> Array:
        return base(params) * rows

    return jacobian


def _warp_jacobian(
    sampler: _Sampler,
    moving: Array,
    *,
    model: TransformModel,
    ndim: int,
    fixed_shape: tuple[int, ...],
    moving_shape: tuple[int, ...],
    spec: RegistrationSpec,
) -> Callable[[Array], Array]:
    """Closed-form Jacobian of the warp residual w.r.t. the transform params.

    ``J[x, j] = ∂warp(x)/∂grid · ∂grid(x;θ)/∂θ_j`` by the chain rule, factored so
    the expensive **gather** runs ``ndim`` times (the interpolation derivative
    ``∂warp/∂grid``, one JVP per grid axis) rather than ``jax.jacfwd``'s ``P``
    times (one warp-tangent gather per parameter); ``∂grid/∂θ`` is taken one
    parameter-column at a time (a JVP of the grid *construction* -- matmul only,
    no gather) and contracted against ``∂warp/∂grid`` immediately, so the dense
    ``(*spatial, ndim, P)`` grid tangent that ``jax.jacfwd`` would materialise
    (``ndim×`` the warp's memory) never exists -- peak memory is the ``(M, P)``
    Jacobian itself plus one ``(*spatial, ndim)`` column at a time.  This is
    **exact** -- equal to ``jax.jacfwd`` of the SSD residual (the interpolation
    derivative, not a central-difference approximation), so the forward path is
    byte-unchanged, just faster, with the gather cut from ``O(P)`` to
    ``O(ndim)``.  It speeds the cases the inverse-compositional kernel cannot
    cover (the WorldSpace / forced-forward forward path).  Returns
    ``params -> (M, P)``.
    """

    def grid_of(params: Array) -> Array:
        matrix, center = sampler.index_sampling(
            model.exp(params, ndim=ndim),
            fixed_shape=fixed_shape,
            moving_shape=moving_shape,
        )
        return affine_grid(matrix, fixed_shape, center=center)

    def warp_at(grid: Array) -> Array:
        return spatial_transform(
            moving[..., None],
            grid,
            mode=spec.boundary_mode,
            cval=spec.cval,
            method=spec.interpolation,
        )[..., 0]

    def jacobian(params: Array) -> Array:
        grid = grid_of(params)

        # ∂warp/∂grid: the exact interpolation derivative, one JVP per axis.
        def dwarp_axis(axis: int) -> Array:
            tangent = (
                jnp.zeros(grid.shape, dtype=grid.dtype).at[..., axis].set(1.0)
            )
            return cast(Array, jax.jvp(warp_at, (grid,), (tangent,))[1])

        dwarp = jnp.stack([dwarp_axis(d) for d in range(ndim)], axis=-1)
        # ∂grid/∂θ one column at a time, contracted into ∂warp/∂grid on the
        # spot -- the dense (*spatial, ndim, P) grid tangent never materialises.
        p = params.shape[-1]

        def jac_column(j: int) -> Array:
            e_j = jnp.zeros((p,), dtype=params.dtype).at[j].set(1.0)
            dgrid_j = cast(Array, jax.jvp(grid_of, (params,), (e_j,))[1])
            return jnp.sum(dwarp * dgrid_j, axis=-1).reshape(-1)

        return jnp.stack([jac_column(j) for j in range(p)], axis=-1)

    return jacobian


def optimize_objective(
    objective: Objective,
    params: Array,
    *,
    optimizer: str,
    iterations: int,
    cg_tol: float,
    jacobian_fn: Optional[Callable[[Array], Array]] = None,
    convergence: Optional[Convergence] = None,
) -> tuple[Array, Array]:
    """Minimise an ``Objective`` over ``params``; return ``(params, history)``.

    The optimiser dispatch shared by every recipe and coordinate space: a
    least-squares objective routes to the matrix-free Gauss-Newton /
    Levenberg-Marquardt path; any other to BFGS on the scalar cost.  The
    objective closes over its own data (an image pair + warp, boundary
    samples, ...), so this function is objective-agnostic.  ``jacobian_fn``
    (optional) supplies the residual's ``M×P`` Jacobian in closed form for the
    least-squares path (the analytic warp Jacobian -- far fewer gathers than
    ``jax.jacfwd``); ``None`` falls back to ``jacfwd`` (the parity oracle).

    ``convergence`` (the resolved :class:`Convergence`, or ``None`` for the fixed
    scan) early-exits the least-squares loop once the windowed cost slope flattens
    (the GN/LM ``early_stop``).  The caller's ``resolve_convergence_mode`` gate
    already rejects ``mode='early_exit'`` on the scalar/BFGS path (monolithic --
    no single-step), so ``convergence`` is always ``None`` there; the guard below
    is a defensive internal invariant.
    """
    use_lsq = objective.is_least_squares and optimizer in ('auto', 'lm', 'gn')
    early_stop = (
        (convergence.threshold, convergence.window)
        if convergence is not None
        else None
    )
    if use_lsq:
        if optimizer == 'gn':
            res = gauss_newton(
                objective.residual,
                params,
                n_iters=iterations,
                cg_tol=cg_tol,
                jacobian_fn=jacobian_fn,
                early_stop=early_stop,
            )
        else:
            res = levenberg_marquardt(
                objective.residual,
                params,
                n_iters=iterations,
                cg_tol=cg_tol,
                jacobian_fn=jacobian_fn,
                early_stop=early_stop,
            )
        return res.params, res.cost_history

    if convergence is not None:  # defensive: the upstream B2 gate ensures None
        raise ValueError(
            "mode='early_exit' is not supported on the scalar/BFGS forward path "
            '(non-least-squares metric: MI / correlation-ratio); use mode='
            "'fixed'."
        )
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
    convergence: Optional[Convergence] = None,
    pyr_mask: Optional[tuple[Float[Array, '*fspatial 1'], ...]] = None,
) -> RegistrationResult:
    """Coarse-to-fine register ``moving`` against a precomputed reference.

    The per-image core of the driver: the **reference** pyramid ``pyr_f``
    and the ``sampler`` are built once by the caller and passed in, so a
    batched recipe (``volreg``) can compute the shared reference work once
    and ``vmap`` only this core over a series of moving images.  Builds
    the moving pyramid, runs the coarse-to-fine optimise, and finalises.

    ``pyr_mask`` (optional, A3) is the ``fixed``-grid weight pyramidised to
    match ``pyr_f`` -- threaded into the metric cost (every level) and, on the
    least-squares path, into the residual Jacobian (``sqrt(mask)`` row-scaling,
    so out-of-mask voxels drop from the Gauss-Newton normal equations).
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
        mask_l = None if pyr_mask is None else pyr_mask[level][..., 0]
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
            metric=spec.metric, warp=warp_fn, fixed=f_l, mask=mask_l
        )
        # Closed-form warp Jacobian for the least-squares (SSD) forward path
        # -- far fewer gathers than jacfwd (the parity oracle).  A mask scales
        # the residual rows by sqrt(mask) (matching ``SSD.residual``), so the
        # Jacobian rows scale the same way -- out-of-mask voxels contribute a
        # zero row to the Gauss-Newton normal equations.
        jac_fn: Optional[Callable[[Array], Array]] = None
        if spec.metric.is_least_squares:
            jac_fn = _warp_jacobian(
                sampler,
                m_l,
                model=model,
                ndim=ndim,
                fixed_shape=f_shape,
                moving_shape=m_shape,
                spec=spec,
            )
            if mask_l is not None:
                jac_fn = _mask_jacobian(jac_fn, jnp.sqrt(mask_l).reshape(-1))
        params, hist = optimize_objective(
            objective,
            params,
            optimizer=spec.optimizer,
            iterations=iters_per_level[level],
            cg_tol=spec.cg_tol,
            jacobian_fn=jac_fn,
            convergence=convergence,
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
    mask: Optional[Float[Array, '*fspatial']] = None,
) -> RegistrationResult:
    """Coarse-to-fine registration driver shared by the recipes.

    ``mask`` (optional, A3) is a non-negative per-voxel weight on the ``fixed``
    grid; it is pyramidised alongside ``fixed`` and threaded into the per-level
    metric cost (and the least-squares Jacobian) so the registration ignores
    out-of-mask voxels.
    """
    if moving.ndim != ndim or fixed.ndim != ndim:
        raise ValueError(
            f'expected {ndim}-D single-channel images; got moving '
            f'{moving.shape}, fixed {fixed.shape}.'
        )
    if mask is not None and mask.shape != fixed.shape:
        raise ValueError(
            f'mask shape {mask.shape} must match the fixed grid {fixed.shape}.'
        )
    # Pin a histogram metric's ranges once from the full-res images (A6;
    # stationary objective).  Here, not in register_core, because volreg vmaps
    # register_core -- where an eager float(moving.min()) cannot run -- and that
    # path is SSD-only (a no-op pin) anyway.
    spec = replace(spec, metric=pin_metric_ranges(spec.metric, moving, fixed))
    dtype = moving.dtype
    pyr_f = gaussian_pyramid(
        fixed[..., None],
        levels=spec.levels,
        factor=spec.pyramid_factor,
        sigma=spec.pyramid_sigma,
    )
    # Pyramidise the mask on the fixed grid (same anti-aliased downsample as the
    # image); a hard mask softens to fractional weights at coarse boundaries.
    pyr_mask = (
        None
        if mask is None
        else gaussian_pyramid(
            mask[..., None],
            levels=spec.levels,
            factor=spec.pyramid_factor,
            sigma=spec.pyramid_sigma,
        )
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
        pyr_mask=pyr_mask,
        # Forward early-exit is honoured only on the least-squares (GN/LM)
        # optimiser's windowed ``early_stop``; the scalar/BFGS path (a
        # non-least-squares metric) is monolithic and rejects it (B2 gate).
        # Single-pair only -- ``volreg`` vmaps ``register_core`` and threads the
        # fixed scan (its forward branch passes no ``convergence``).
        convergence=resolve_convergence_mode(
            spec.mode,
            spec.convergence,
            supports_early_exit=spec.metric.is_least_squares
            and spec.optimizer in ('auto', 'lm', 'gn'),
            path='the scalar/BFGS forward path (a non-least-squares metric)',
        ),
    )
