# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Core machinery for the registration recipes.

The pieces shared by :func:`rigid_register` / :func:`affine_register`: the
:class:`RegistrationSpec` static-config record (following the
:class:`SolverSpec` / :class:`Interpolator` precedent -- a frozen, hashable
dataclass that rides ``jit`` static args), the :class:`RegistrationResult`
``NamedTuple`` output, and the coarse-to-fine driver.

The driver is written **once** against a :class:`CoordinateSpace`: the only
axis that separates an index-space, shared-grid registration from a
physically correct one is *how* the parameter transform relates the two
voxel grids, and that axis is the space's responsibility.
:class:`IndexSpace` (default) is the leaner, fully-on-device, voxel-unit
path; :class:`WorldSpace` is anisotropy- and different-grid-correct.
Everything else -- the pyramid, the level loop, the optimise, the result
-- is identical.

Per pyramid level the driver minimises a similarity cost over the
transform's Lie parameters ``θ`` and warm-starts the next finer level
from the result (in ``IndexSpace`` the voxel-unit translations are
rescaled to the finer grid; in ``WorldSpace`` the physical parameters are
grid-independent and carry over unchanged).  Two optimisation paths:

- **SSD** -> Gauss-Newton / Levenberg-Marquardt on the *vector* residual
  :math:`\\mathrm{warp}(\\theta) - \\mathrm{fixed}`.  Matrix-free, GPU-native,
  second-order -- the 3dvolreg / Lucas-Kanade lineage.
- **LNCC / MI / CR** -> BFGS on the *scalar* cost
  (``jax.scipy.optimize.minimize``), which handles the non-least-squares
  metric and the rotation/translation scaling via its Hessian estimate.
  Also matrix-free, so it too avoids cuSolver.

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
    NamedTuple,
    Optional,
    Protocol,
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
from ..linalg import (
    gauss_newton,
    implicit_least_squares,
    implicit_minimize,
    levenberg_marquardt,
)
from ._converge import (
    Convergence,
    ConvergenceMode,
    resolve_convergence_mode,
)
from ._metric import SSD, Metric, pin_metric_ranges
from ._model import TransformModel
from ._objective import MetricObjective, Objective
from ._space import CoordinateSpace, IndexSpace, _Sampler

# ``Convergence`` / ``ConvergenceMode`` / ``resolve_convergence_mode`` live in
# ``._converge`` (the iteration-driver module that owns the early-exit machinery,
# G1); imported above so ``RegistrationSpec`` and the drivers can reference them.


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
        Similarity objective, a :class:`Metric` record carrying its own
        hyper-parameters: ``SSD()`` (within-modality; the GN/LM
        least-squares path), ``LNCC(radius=...)`` (local cross-correlation;
        intensity-robust), ``MI(bins=...)`` / ``CorrelationRatio(bins=...)``
        (cross-modal).
    optimizer
        ``"auto"`` (least-squares metric -> ``"lm"``, else -> ``"bfgs"``),
        or force ``"lm"`` / ``"gn"`` (least-squares only) / ``"bfgs"``.
    interpolation
        Sampling kernel for the warp (an :class:`Interpolator`).
    boundary_mode, cval
        Out-of-bounds handling for the warp (default zero-fill).
    pyramid_factor, pyramid_sigma
        Downsample factor and anti-alias sigma for the pyramid.
    cg_tol
        Inner-CG tolerance for the GN/LM path.
    mode
        Iteration strategy, orthogonal to ``convergence``.  ``'fixed'``
        (the default) runs the fixed ``lax.scan`` -- reproducible,
        reverse-differentiable, and ``vmap``-batchable (:func:`volreg`).
        ``'early_exit'`` runs the windowed-slope ``lax.while_loop`` with
        ``iterations`` as the hard cap; it is single-pair (a ``vmap``-ed
        ``while_loop`` exits only when *all* lanes converge) and **not**
        reverse-differentiable (``jax.grad`` through it raises a loud, actionable
        error -- use ``mode='fixed'`` or the implicit-function path).  Rejected
        on the scalar/BFGS forward path (a non-least-squares metric), which is
        monolithic.  The default is ``'fixed'`` on **every** recipe
        (reproducible + differentiable out of the box); ``'early_exit'`` is the
        recommended opt-in for the single-pair inverse-compositional recipes
        (:func:`rigid_register` / :func:`affine_register`), which converge in a
        few of their iterations.
    convergence
        The :class:`Convergence` (threshold / window) parameterising
        ``mode='early_exit'``; **inert** under ``mode='fixed'``.  On the
        early-exit path the per-level cost trace is padded to the ``iterations``
        cap with the final cost (so the shape is path-independent; the value is
        constant past the stop iteration).
    ic_line_search
        Opt-in cost-decrease guard for the inverse-compositional step.
        ``False`` (default) takes the trust-region-clamped Gauss-Newton step
        directly -- the fast path (its single warp/iter is the IC speed win).
        ``True`` backtracks along the clamped direction and accepts the largest
        fraction that decreases the SSD, leaving the iterate unmoved if none does
        -- so the per-level cost is **monotone non-increasing** even when the
        constant-template Hessian proposes an ascent step on a hard case.  It
        costs extra warps/iter (the candidate evaluations -- ~3x the IC step on
        GPU), so it is off by default and enabled for robustness on pathological
        data; a step that already decreases at full length is taken
        byte-unchanged either way.  No effect off the inverse-compositional path.
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
    ic_line_search: bool = False


class RegistrationResult(NamedTuple):
    """Output of a registration recipe.

    Attributes
    ----------
    matrix
        The estimated homogeneous transform, ``(ndim + 1, ndim + 1)``,
        **self-contained**: the centre the warp applies it about is baked in, so
        ``apply_affine(coords, matrix)`` reproduces the warp and it composes
        with another result.  In :class:`IndexSpace` it is the full-resolution
        fixed-voxel -> moving-voxel index map; in :class:`WorldSpace` the
        fixed-world -> moving-world transform (mm).  (The raw about-origin
        ``model.exp(params)`` is recovered from ``params``.)
    params
        The transform's raw **about-origin** Lie parameters at full resolution
        (voxel units in :class:`IndexSpace`; physical units in
        :class:`WorldSpace`).
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
    """Expand an iteration schedule into per-pyramid-level counts (finest first).

    Shared by the forward (:func:`register_core`) and inverse-compositional
    (:func:`ic_register_core`) paths, which both index the pyramid from finest
    (level 0) to coarsest.

    Parameters
    ----------
    iterations
        Either a single ``int`` (the same count at every level) or a
        length-``levels`` sequence in **coarse-to-fine** order (the natural
        schedule order), one count per level.
    levels
        Number of pyramid resolutions.

    Returns
    -------
    list[int]
        Length-``levels`` list of iteration counts in **finest-first** order
        (the ordering the drivers index the pyramid with).  A sequence input is
        reversed from coarse-to-fine to finest-first; an ``int`` is broadcast.

    Raises
    ------
    ValueError
        If a sequence is passed whose length differs from ``levels``.
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
    """Warp a single-channel ``moving`` image onto ``fixed_shape``.

    Shared by both coordinate spaces: the sampler resolves the parameter
    ``transform`` into the fixed-voxel -> moving-voxel sampling matrix and grid
    centre; the sampling itself (boundary, interpolation) is the same.

    Parameters
    ----------
    sampler
        The space-bound sampler that maps the parameter transform into the
        fixed-voxel -> moving-voxel sampling matrix and grid centre.
    moving
        Single-channel moving image, shape ``(*moving_shape)``.
    transform
        The homogeneous transform matrix produced by ``model.exp``, shape
        ``(ndim + 1, ndim + 1)``.
    fixed_shape
        Spatial shape of the fixed grid the warp is sampled onto.
    moving_shape
        Spatial shape of the moving image.
    spec
        Registration configuration; supplies the boundary mode, fill value, and
        interpolation kernel for the resample.

    Returns
    -------
    Array
        The moving image resampled onto the fixed grid, shape ``(*fixed_shape)``
        (single-channel, the internal channel axis removed).
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
    """Row-scale a residual Jacobian by :math:`\\sqrt{\\mathrm{mask}}`.

    Wraps a base residual Jacobian for the masked least-squares path.  The
    masked SSD residual is :math:`\\sqrt{\\mathrm{mask}} \\cdot
    (\\mathrm{warped} - \\mathrm{fixed})` (see :meth:`SSD.residual`); its
    Jacobian is therefore the unmasked Jacobian with each voxel row multiplied
    by :math:`\\sqrt{\\mathrm{mask}}`, so an out-of-mask voxel (weight 0)
    contributes a zero row and drops out of the Gauss-Newton normal equations.

    Parameters
    ----------
    base
        The unmasked residual Jacobian, mapping ``params -> (M, P)``.
    sqrt_w
        Per-voxel square-root weights, shape ``(M,)`` (one entry per residual
        row).

    Returns
    -------
    Callable[[Array], Array]
        A Jacobian closure ``params -> (M, P)`` whose rows are the base
        Jacobian's rows scaled by ``sqrt_w``.
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
    """Closed-form Jacobian of the warp residual w.r.t. the transform parameters.

    By the chain rule,
    :math:`J[x, j] = \\partial\\,\\mathrm{warp}(x)/\\partial\\,\\mathrm{grid}
    \\cdot \\partial\\,\\mathrm{grid}(x;\\theta)/\\partial\\,\\theta_j`, factored
    so the expensive **gather** runs ``ndim`` times (the interpolation
    derivative :math:`\\partial\\,\\mathrm{warp}/\\partial\\,\\mathrm{grid}`, one
    JVP per grid axis) rather than ``jax.jacfwd``'s ``P`` times (one warp-tangent
    gather per parameter);
    :math:`\\partial\\,\\mathrm{grid}/\\partial\\,\\theta` is taken one
    parameter-column at a time (a JVP of the grid *construction* -- matmul only,
    no gather) and contracted against
    :math:`\\partial\\,\\mathrm{warp}/\\partial\\,\\mathrm{grid}` immediately, so
    the dense ``(*spatial, ndim, P)`` grid tangent that ``jax.jacfwd`` would
    materialise (``ndim`` times the warp's memory) never exists -- peak memory is
    the ``(M, P)`` Jacobian itself plus one ``(*spatial, ndim)`` column at a
    time.  This is **exact** -- equal to ``jax.jacfwd`` of the SSD residual (the
    interpolation derivative, not a central-difference approximation), so the
    forward path is byte-unchanged, just faster, with the gather cut from
    :math:`O(P)` to :math:`O(\\mathrm{ndim})`.  It speeds the cases the
    inverse-compositional kernel cannot cover (the :class:`WorldSpace` /
    forced-forward forward path).

    Parameters
    ----------
    sampler
        The space-bound sampler that maps the parameter transform into the
        fixed-voxel -> moving-voxel sampling matrix and grid centre.
    moving
        Single-channel moving image, shape ``(*moving_shape)``.
    model
        The transform model; ``model.exp(params, ndim=ndim)`` maps the raw Lie
        parameters to the homogeneous transform matrix.
    ndim
        Spatial dimensionality of the images (number of grid axes).
    fixed_shape
        Spatial shape of the fixed grid the warp is sampled onto.
    moving_shape
        Spatial shape of the moving image.
    spec
        Registration configuration; supplies the boundary mode, fill value, and
        interpolation kernel for the warp.

    Returns
    -------
    Callable[[Array], Array]
        A closure mapping the length-``P`` parameter vector to the ``(M, P)``
        residual Jacobian, where ``M`` is the number of fixed-grid voxels and
        ``P`` the number of transform parameters.
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
    """Minimise an :class:`Objective` over ``params``; return ``(params, history)``.

    The optimiser dispatch shared by every recipe and coordinate space: a
    least-squares objective routes to the matrix-free Gauss-Newton /
    Levenberg-Marquardt path; any other to BFGS on the scalar cost.  The
    objective closes over its own data (an image pair + warp, boundary
    samples, ...), so this function is objective-agnostic.

    Parameters
    ----------
    objective
        The similarity objective to minimise; carries the ``residual`` (for the
        least-squares path), ``cost`` (for the scalar path), and the
        ``is_least_squares`` flag.
    params
        Initial parameter vector, shape ``(P,)``.
    optimizer
        Optimiser selector: ``'auto'`` / ``'lm'`` / ``'gn'`` route to the
        least-squares path (only when the objective is least-squares); anything
        else, or a non-least-squares objective, routes to BFGS.
    iterations
        Maximum optimiser iterations.
    cg_tol
        Inner conjugate-gradient tolerance for the Gauss-Newton / LM path.
    jacobian_fn
        Optional closure supplying the residual's :math:`M \\times P` Jacobian in
        closed form for the least-squares path (the analytic warp Jacobian --
        far fewer gathers than ``jax.jacfwd``); ``None`` falls back to
        ``jacfwd`` (the parity oracle).
    convergence
        The resolved :class:`Convergence`, or ``None`` for the fixed scan.  When
        set, it early-exits the least-squares loop once the windowed cost slope
        flattens.  The caller's :func:`resolve_convergence_mode` gate already
        rejects ``mode='early_exit'`` on the scalar/BFGS path (monolithic -- no
        single-step), so ``convergence`` is always ``None`` there; the guard in
        the body is a defensive internal invariant.

    Returns
    -------
    params : Array
        The optimised parameter vector, shape ``(P,)``.
    history : Array
        The per-iteration cost trace for this solve.

    Raises
    ------
    ValueError
        If ``convergence`` is non-``None`` on the scalar/BFGS path (the
        defensive guard for the upstream gate).
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


class LevelSolver(Protocol):
    """Callback that minimises a single pyramid level.

    This is the one axis on which the forward and implicit drivers differ.  The
    coarse-to-fine driver (:func:`register_core`) owns the pyramid, the
    inter-level warm-start rescale, and the result assembly identically either
    way; *how a single level is minimised* is this callback -- the forward
    Gauss-Newton / LM / BFGS optimise (:func:`_forward_level_solve`) or the
    implicit-function solve (:func:`_implicit_level_solve`).  Both take the
    bound ``sampler`` plus the level images plus the current parameters and
    return ``(params, cost_trace)``.
    """

    def __call__(
        self,
        sampler: _Sampler,
        moving_level: Array,
        fixed_level: Array,
        mask_level: Optional[Array],
        params: Array,
        *,
        model: TransformModel,
        ndim: int,
        spec: RegistrationSpec,
        fixed_shape: tuple[int, ...],
        moving_shape: tuple[int, ...],
        iterations: int,
        convergence: Optional[Convergence],
    ) -> tuple[Array, Array]: ...


def _forward_level_solve(
    sampler: _Sampler,
    moving_level: Array,
    fixed_level: Array,
    mask_level: Optional[Array],
    params: Array,
    *,
    model: TransformModel,
    ndim: int,
    spec: RegistrationSpec,
    fixed_shape: tuple[int, ...],
    moving_shape: tuple[int, ...],
    iterations: int,
    convergence: Optional[Convergence],
) -> tuple[Array, Array]:
    """Solve one pyramid level by the forward GN / LM / BFGS optimise.

    The default :class:`LevelSolver`: the per-level body the coarse-to-fine
    driver has always run, lifted behind the protocol so the implicit path can
    swap in :func:`_implicit_level_solve` without duplicating the orchestration.
    Builds the level's :class:`MetricObjective` (and, on the least-squares path,
    the closed-form warp Jacobian, mask-scaled when a mask is present) and hands
    it to :func:`optimize_objective`.

    Parameters
    ----------
    sampler
        The space-bound sampler mapping the parameter transform to the
        fixed-voxel -> moving-voxel sampling matrix and grid centre.
    moving_level
        Single-channel moving image at this pyramid level, shape
        ``(*moving_shape)``.
    fixed_level
        Single-channel fixed image at this pyramid level, shape
        ``(*fixed_shape)``.
    mask_level
        Optional per-voxel weight on the fixed grid at this level, shape
        ``(*fixed_shape)``; ``None`` for an unmasked solve.
    params
        Current parameter vector (warm-started from the coarser level), shape
        ``(P,)``.
    model
        The transform model; ``model.exp(params, ndim=ndim)`` maps the raw Lie
        parameters to the homogeneous transform matrix.
    ndim
        Spatial dimensionality of the images.
    spec
        Registration configuration (metric, optimiser, interpolation, tolerances).
    fixed_shape
        Spatial shape of the fixed grid at this level.
    moving_shape
        Spatial shape of the moving image at this level.
    iterations
        Maximum optimiser iterations for this level.
    convergence
        Resolved :class:`Convergence` for the early-exit least-squares loop, or
        ``None`` for the fixed scan.

    Returns
    -------
    params : Array
        The optimised parameter vector for this level, shape ``(P,)``.
    cost_trace : Array
        The per-iteration cost trace for this level.
    """

    def warp_fn(p: Array) -> Array:
        return _warp(
            sampler,
            moving_level,
            model.exp(p, ndim=ndim),
            fixed_shape=fixed_shape,
            moving_shape=moving_shape,
            spec=spec,
        )

    objective = MetricObjective(
        metric=spec.metric, warp=warp_fn, fixed=fixed_level, mask=mask_level
    )
    # Closed-form warp Jacobian for the least-squares (SSD) forward path -- far
    # fewer gathers than jacfwd (the parity oracle).  A mask scales the residual
    # rows by sqrt(mask) (matching ``SSD.residual``), so the Jacobian rows scale
    # the same way -- out-of-mask voxels contribute a zero row to the
    # Gauss-Newton normal equations.
    jac_fn: Optional[Callable[[Array], Array]] = None
    if spec.metric.is_least_squares:
        jac_fn = _warp_jacobian(
            sampler,
            moving_level,
            model=model,
            ndim=ndim,
            fixed_shape=fixed_shape,
            moving_shape=moving_shape,
            spec=spec,
        )
        if mask_level is not None:
            jac_fn = _mask_jacobian(jac_fn, jnp.sqrt(mask_level).reshape(-1))
    return optimize_objective(
        objective,
        params,
        optimizer=spec.optimizer,
        iterations=iterations,
        cg_tol=spec.cg_tol,
        jacobian_fn=jac_fn,
        convergence=convergence,
    )


def _implicit_level_solve(
    sampler: _Sampler,
    moving_level: Array,
    fixed_level: Array,
    mask_level: Optional[Array],
    params: Array,
    *,
    model: TransformModel,
    ndim: int,
    spec: RegistrationSpec,
    fixed_shape: tuple[int, ...],
    moving_shape: tuple[int, ...],
    iterations: int,
    convergence: Optional[Convergence],
) -> tuple[Array, Array]:
    """Solve one pyramid level by the implicit-function theorem.

    The differentiable :class:`LevelSolver`.  Differentiates the level's optimum
    w.r.t. the (level) images directly -- :func:`implicit_least_squares` for a
    least-squares (SSD) metric (Gauss-Newton Hessian),
    :func:`implicit_minimize` for a general metric (LNCC / MI / CR -- exact
    Hessian via BFGS forward).  ``data = (moving_level, fixed_level)`` is the
    differentiable argument, so the gradient flows through the pyramid back to
    the originals; the ``mask`` is closed over as a constant.  ``convergence``
    is inert here (the implicit solve runs its own fixed forward iteration).
    Because the optimum's derivative w.r.t. the incoming parameters is zero
    under the implicit-function theorem, on a multi-level run the coarse levels
    act as a gradient-stopped initialiser and the finest level carries the
    exact implicit-function gradient.

    Parameters
    ----------
    sampler
        The space-bound sampler mapping the parameter transform to the
        fixed-voxel -> moving-voxel sampling matrix and grid centre.
    moving_level
        Single-channel moving image at this pyramid level, shape
        ``(*moving_shape)``.
    fixed_level
        Single-channel fixed image at this pyramid level, shape
        ``(*fixed_shape)``.
    mask_level
        Optional per-voxel weight on the fixed grid at this level, shape
        ``(*fixed_shape)``; closed over as a constant, so it does not carry a
        gradient.
    params
        Current parameter vector (warm-started from the coarser level), shape
        ``(P,)``.
    model
        The transform model; ``model.exp(params, ndim=ndim)`` maps the raw Lie
        parameters to the homogeneous transform matrix.
    ndim
        Spatial dimensionality of the images.
    spec
        Registration configuration (metric, tolerances).
    fixed_shape
        Spatial shape of the fixed grid at this level.
    moving_shape
        Spatial shape of the moving image at this level.
    iterations
        Number of forward solver iterations for the implicit solve.
    convergence
        Accepted for protocol compatibility but ignored; the implicit solve runs
        its own fixed forward iteration.

    Returns
    -------
    theta : Array
        The optimised parameter vector for this level, shape ``(P,)``.
    hist : Array
        Length-2 cost trace: the cost at the incoming parameters and at the
        converged optimum.
    """
    del convergence  # the implicit solve has its own (fixed) forward iteration
    metric = spec.metric

    def warp_of(moving_img: Array, p: Array) -> Array:
        return _warp(
            sampler,
            moving_img,
            model.exp(p, ndim=ndim),
            fixed_shape=fixed_shape,
            moving_shape=moving_shape,
            spec=spec,
        )

    data = (moving_level, fixed_level)
    if metric.is_least_squares:

        def residual_fn(d: tuple[Array, Array], p: Array) -> Array:
            mv, fx = d
            return metric.residual(warp_of(mv, p), fx, mask=mask_level)

        theta = implicit_least_squares(
            residual_fn, data, params, n_iters=iterations, cg_tol=spec.cg_tol
        )
        r0 = residual_fn(data, params)
        r1 = residual_fn(data, theta)
        hist = jnp.stack(
            [0.5 * jnp.vdot(r0, r0).real, 0.5 * jnp.vdot(r1, r1).real]
        )
        return theta, hist

    def objective_fn(d: tuple[Array, Array], p: Array) -> Array:
        mv, fx = d
        return metric.cost(warp_of(mv, p), fx, mask=mask_level)

    theta = implicit_minimize(
        objective_fn, data, params, maxiter=iterations, cg_tol=spec.cg_tol
    )
    hist = jnp.stack([objective_fn(data, params), objective_fn(data, theta)])
    return theta, hist


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
    solve_level: LevelSolver = _forward_level_solve,
) -> RegistrationResult:
    """Coarse-to-fine register ``moving`` against a precomputed reference.

    The per-image core of the driver: the **reference** pyramid ``pyr_f``
    and the ``sampler`` are built once by the caller and passed in, so a
    batched recipe (:func:`volreg`) can compute the shared reference work once
    and ``vmap`` only this core over a series of moving images.  Builds the
    moving pyramid, runs the coarse-to-fine optimise, and finalises.

    Parameters
    ----------
    moving
        Single-channel moving image, shape ``(*mspatial)``.
    pyr_f
        The precomputed **fixed** (reference) Gaussian pyramid, a tuple of
        single-channel levels ``(*fspatial, 1)`` finest first.
    model
        The transform model; ``model.exp`` maps raw Lie parameters to the
        homogeneous transform, ``model.rescale_to_grid`` warm-starts between
        levels.
    ndim
        Spatial dimensionality of the images.
    spec
        Registration configuration (levels, iterations, metric, optimiser, ...).
    space
        The :class:`CoordinateSpace` governing how the parameter transform
        relates the two voxel grids (and whether translations rescale between
        levels).
    sampler
        The space-bound sampler, built once by the caller.
    init_params
        Initial parameter vector, shape ``(p,)``.
    convergence
        Resolved :class:`Convergence` for the early-exit least-squares loop, or
        ``None`` for the fixed scan.
    pyr_mask
        Optional weight pyramid on the ``fixed`` grid, matching ``pyr_f`` -- a
        tuple of levels ``(*fspatial, 1)``.  Threaded into the metric cost at
        every level and, on the least-squares path, into the residual Jacobian
        (:math:`\\sqrt{\\mathrm{mask}}` row-scaling, so out-of-mask voxels drop
        from the Gauss-Newton normal equations).
    solve_level
        The per-level minimiser (:class:`LevelSolver`); defaults to
        :func:`_forward_level_solve`.

    Returns
    -------
    RegistrationResult
        The estimated transform (``matrix`` and raw ``params``), the ``moving``
        image ``warped`` onto the full-resolution fixed grid, and the
        concatenated per-level ``cost_history``.
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

        params, hist = solve_level(
            sampler,
            m_l,
            f_l,
            mask_l,
            params,
            model=model,
            ndim=ndim,
            spec=spec,
            fixed_shape=f_shape,
            moving_shape=m_shape,
            iterations=iters_per_level[level],
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
    solve_level: LevelSolver = _forward_level_solve,
) -> RegistrationResult:
    """Coarse-to-fine registration driver shared by the recipes.

    Builds both image pyramids (and the mask pyramid, if any), pins any
    histogram-metric ranges from the full-resolution images, constructs the
    space-bound sampler, and hands off to :func:`register_core`.

    Parameters
    ----------
    moving
        Single-channel moving image, shape ``(*mspatial)`` (``ndim`` axes).
    fixed
        Single-channel fixed (reference) image, shape ``(*fspatial)`` (``ndim``
        axes).
    model
        The transform model whose Lie parameters are optimised.
    ndim
        Spatial dimensionality; both images must be ``ndim``-D.
    spec
        Registration configuration (levels, iterations, metric, optimiser, ...).
    init_params
        Optional initial parameter vector, shape ``(p,)``; defaults to zeros
        (the identity transform).
    space
        The :class:`CoordinateSpace` relating the two voxel grids; defaults to
        :class:`IndexSpace`.
    mask
        Optional non-negative per-voxel weight on the ``fixed`` grid, shape
        ``(*fspatial)``; pyramidised alongside ``fixed`` (a hard mask softens to
        fractional weights at coarse boundaries) and threaded into the per-level
        metric cost (and the least-squares Jacobian) so the registration ignores
        out-of-mask voxels.
    solve_level
        The per-level minimiser (:class:`LevelSolver`):
        :func:`_forward_level_solve` (default; GN / LM / BFGS) or
        :func:`_implicit_level_solve` (the implicit-function differentiable
        layer).  The driver is otherwise identical -- the pyramid, the
        warm-start rescale, and the result assembly do not depend on which one
        runs.

    Returns
    -------
    RegistrationResult
        The estimated transform, the warped ``moving`` image, and the
        concatenated per-level cost history (see :class:`RegistrationResult`).

    Raises
    ------
    ValueError
        If ``moving`` or ``fixed`` is not ``ndim``-D, or if ``mask`` is given
        and its shape does not match ``fixed``.
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
        solve_level=solve_level,
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
