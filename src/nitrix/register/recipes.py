# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pairwise rigid / affine registration recipes.

Pure functions ``(moving, fixed) -> RegistrationResult`` -- the ``reml_fit``
precedent (a ``NamedTuple`` of arrays, no PyTree module, no atlas / I/O).
They compose the R0/R1 substrate (pyramid + metric + transform
parametrisation + matrix-free optimiser); the orchestration lives here so
``entense`` can wrap it (or re-implement) without re-deriving it.

The representative algorithm is intensity-based Gauss-Newton /
Levenberg-Marquardt on SE(3)/affine (the ``3dvolreg`` / AIR lineage):
coarse-to-fine, second-order, differentiable.  The metric (SSD by
default; LNCC / MI / CR for intensity-robust or cross-modal cases) and
the schedule are set on the ``RegistrationSpec``.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Literal, NamedTuple, Optional, Sequence, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..geometry import affine_grid, identity_grid, spatial_transform
from ..geometry._interpolate import (
    BoundaryMode,
    Interpolator,
    Linear,
    NearestNeighbour,
)
from ._core import (
    Convergence,
    RegistrationResult,
    RegistrationSpec,
    multi_resolution_register,
)
from ._force import Force
from ._inverse_compositional import (
    _affine_params_from_matrix,
    _rigid_params_from_matrix,
    ic_affine_register,
    ic_rigid_register,
)
from ._metric import MI, pin_metric_ranges
from ._model import Affine, Rigid, TransformModel
from ._preprocess import preprocess_images
from ._space import CoordinateSpace, IndexSpace, _conjugate_about
from ._syn import SyNResult, SyNSpec, greedy_syn_register

# C3: an explicit Convergence on a path that cannot honour it.  The forward
# least-squares (GN/LM) path now early-exits (Lever A); only the scalar/BFGS
# path (a non-least-squares metric: MI / correlation-ratio) cannot -- its
# optimiser is monolithic and stops on its own gradient/line-search criterion.
_CONVERGENCE_FORWARD_MSG = (
    'spec.convergence (windowed early-exit) is honoured on the inverse-'
    'compositional and forward *least-squares* (GN/LM) paths, but not on the '
    'scalar/BFGS forward path this run resolves to (a non-least-squares metric '
    "-- MI / correlation-ratio).  Use convergence='auto' (the default) or None, "
    'or an SSD metric.'
)


def _check_forward_convergence(spec: RegistrationSpec) -> None:
    """Raise (C3) if an explicit ``Convergence`` reaches a path that can't honour
    it -- now only the scalar/BFGS (non-least-squares) forward path; the forward
    least-squares path early-exits via the optimiser ``early_stop`` (Lever A)."""
    if (
        isinstance(spec.convergence, Convergence)
        and not spec.metric.is_least_squares
    ):
        raise ValueError(_CONVERGENCE_FORWARD_MSG)


def _image_moments(img: Array, ndim: int) -> tuple[Array, Array]:
    """Intensity-weighted centroid and per-axis std (voxel coordinates)."""
    grid = identity_grid(img.shape, dtype=img.dtype)  # (*spatial, ndim)
    w = jnp.clip(img, 0.0, None)  # non-negative weights (image as a density)
    total = jnp.sum(w) + 1e-12
    spatial = tuple(range(ndim))
    centroid = jnp.sum(w[..., None] * grid, axis=spatial) / total
    var = jnp.sum(w[..., None] * (grid - centroid) ** 2, axis=spatial) / total
    return centroid, jnp.sqrt(var + 1e-12)


def _moment_init_matrix(
    moving: Array, fixed: Array, *, ndim: int, scale: bool
) -> Array:
    """Centre-of-mass / moment initialisation (A8), an index-space affine.

    Aligns the intensity-weighted centroids (translation) and -- for ``scale``
    (affine) -- the per-axis spread (a **diagonal** scale, no rotation: a
    moment match cannot fix the principal-axis sign, so a rotation init risks a
    *worse* start).  In the ``affine_grid`` centring convention (sample voxel
    ``i`` at ``M(i − c) + t + c``): ``M = diag(std_m / std_f)`` (or ``I``), and
    ``t`` maps the fixed centroid onto the moving centroid.  The cheap robust
    start when a single zero init sits outside the narrow affine basin.
    """
    dtype = moving.dtype
    c = (jnp.asarray(fixed.shape, dtype=dtype) - 1.0) / 2.0
    cen_m, std_m = _image_moments(moving, ndim)
    cen_f, std_f = _image_moments(fixed, ndim)
    lin = (
        jnp.diag(std_m / (std_f + 1e-6))
        if scale
        else jnp.eye(ndim, dtype=dtype)
    )
    t = cen_m - c - lin @ (cen_f - c)
    matrix = jnp.eye(ndim + 1, dtype=dtype).at[:ndim, :ndim].set(lin)
    return matrix.at[:ndim, ndim].set(t)


def _resolve_init_matrix(
    init: str,
    moving: Array,
    fixed: Array,
    *,
    ndim: int,
    scale: bool,
    space: CoordinateSpace,
    spec: RegistrationSpec,
) -> Optional[Array]:
    """Resolve the ``init`` argument to a **coarsest-level** init matrix (or None).

    The driver starts the coarse-to-fine loop at the coarsest level and upscales
    the translation column to each finer grid, so the init's translation must be
    expressed at the coarsest scale -- the full-resolution moment translation
    divided by the pyramid's coarse-to-fine factor.  The linear (scale) block is
    resolution-independent and passes through unchanged.
    """
    if init == 'identity':
        return None
    if init != 'moment':
        raise ValueError(f"init must be 'identity' or 'moment'; got {init!r}.")
    if not isinstance(space, IndexSpace):
        raise ValueError(
            "init='moment' is an index-space initialisation; it is not defined "
            'for WorldSpace (pass an explicit world init or use IndexSpace).'
        )
    matrix = _moment_init_matrix(moving, fixed, ndim=ndim, scale=scale)
    coarse_factor = spec.pyramid_factor ** (spec.levels - 1)
    return matrix.at[:ndim, ndim].set(matrix[:ndim, ndim] / coarse_factor)


def _init_from_transform(
    transform: Array,
    fixed: Array,
    *,
    ndim: int,
    space: CoordinateSpace,
    spec: RegistrationSpec,
) -> Array:
    """A self-contained (centred) recipe matrix -> the internal coarsest-level
    about-origin init that ``_resolve_init_matrix`` returns.

    The warm-start hook: ``rigid_register`` / ``affine_register`` return the
    self-contained matrix (centre baked in, B1), so to *start* an optimise from
    a prior stage's result we de-centre it to about-origin and rescale the
    translation to the coarsest pyramid level (the driver upscales it back).
    Index-space only -- the staged-pipeline frame.
    """
    if not isinstance(space, IndexSpace):
        raise ValueError(
            'init_transform is an index-space warm-start; it is not defined for '
            'WorldSpace (pass a world init via the SyN/Demons init_affine, or '
            'use IndexSpace).'
        )
    dtype = fixed.dtype
    c = (jnp.asarray(fixed.shape, dtype=dtype) - 1.0) / 2.0
    origin = _conjugate_about(
        jnp.asarray(transform, dtype=dtype), -c, ndim, dtype
    )
    coarse_factor = spec.pyramid_factor ** (spec.levels - 1)
    return origin.at[:ndim, ndim].set(origin[:ndim, ndim] / coarse_factor)


__all__ = [
    'rigid_register',
    'affine_register',
    'apply_transform',
    'syn_pipeline',
    'PipelineResult',
]

# Coarsest pyramid axis (voxels) below which the matrix Hessian -- especially the
# 12-DOF affine one -- is too few-voxel to estimate reliably, so the constant-
# template step converges to a wrong (anti-correlated) minimum
# (`register-affine-small-grid-divergence`: a 24³/28³ image at the default 3
# levels drops its coarsest level to ≤14³ and diverges; single-level recovers).
# Below this, the affine pyramid is shortened so the coarsest level stays reliable
# -- a targeted fix that needs no damping, so the well-conditioned path is
# untouched.  Affine-only: rigid (6-DOF) is robust at small coarse grids.
_MIN_COARSE_AXIS = 16


class AffinePyramidDepthWarning(UserWarning):
    """The affine pyramid was shortened so its coarsest level stays reliable."""


def _cap_levels(
    spec: RegistrationSpec, shape: tuple[int, ...]
) -> RegistrationSpec:
    """Shorten the (affine) pyramid so its coarsest level keeps ``>=
    _MIN_COARSE_AXIS`` voxels per axis; **loud** (a warning, per the loud-fallback
    tenet) and a no-op when the requested depth already satisfies it.  Honours an
    explicit per-level ``iterations`` tuple by keeping its finest entries."""
    smallest = float(min(shape))
    effective = 1
    while (
        effective < spec.levels
        and smallest / (spec.pyramid_factor**effective) >= _MIN_COARSE_AXIS
    ):
        effective += 1
    if effective >= spec.levels:
        return spec
    warnings.warn(
        f'affine_register: shortening the pyramid from {spec.levels} to '
        f'{effective} level(s) -- a {int(smallest)}-voxel image at '
        f'{spec.levels} levels drops the coarsest grid below '
        f'{_MIN_COARSE_AXIS} vox/axis, where the affine Hessian is unreliable '
        f'and the fit diverges (register-affine-small-grid-divergence).  Pass a '
        f'smaller spec.levels to silence this.',
        category=AffinePyramidDepthWarning,
        stacklevel=3,
    )
    iterations = spec.iterations
    if isinstance(iterations, tuple):
        iterations = iterations[-effective:]  # keep the finest levels' counts
    return replace(spec, levels=effective, iterations=iterations)


def _spatial_ndim(moving: Array, fixed: Array) -> int:
    ndim = moving.ndim
    if ndim not in (2, 3):
        raise ValueError(
            f'registration supports 2-D / 3-D single-channel images; '
            f'got shape {moving.shape}.'
        )
    return ndim


def _use_inverse_compositional(
    method: str,
    space: CoordinateSpace,
    spec: RegistrationSpec,
    model: TransformModel,
) -> bool:
    """Resolve the ``method`` argument against the IC fast-path preconditions.

    The inverse-compositional kernel (constant-template Hessian, ~4-7x the
    forward GN/LM throughput) applies to a **rigid or affine** least-squares
    (SSD) registration in **IndexSpace** (the template is linearised in voxel
    coordinates).  ``"auto"`` takes it when those hold and falls back to the
    forward path otherwise (the parity oracle); ``"inverse_compositional"``
    forces it (and validates); ``"forward"`` always takes the forward path.
    """
    supported = (
        isinstance(space, IndexSpace)
        and spec.metric.is_least_squares
        and spec.optimizer in ('auto', 'lm', 'gn')
        and isinstance(model, (Rigid, Affine))
    )
    if method == 'auto':
        return supported
    if method == 'inverse_compositional':
        if not supported:
            raise ValueError(
                'method="inverse_compositional" requires IndexSpace + a '
                'least-squares (SSD) metric + a Rigid/Affine model.'
            )
        return True
    if method == 'forward':
        return False
    raise ValueError(
        f'method must be "auto", "forward", or "inverse_compositional"; '
        f'got {method!r}.'
    )


def rigid_register(
    moving: Float[Array, '*mspatial'],
    fixed: Float[Array, '*fspatial'],
    *,
    spec: RegistrationSpec = RegistrationSpec(),
    space: CoordinateSpace = IndexSpace(),
    method: str = 'auto',
    init: Literal['identity', 'moment'] = 'identity',
    init_transform: Optional[Float[Array, ' d1 d1']] = None,
    winsorize: Optional[tuple[float, float]] = None,
    histogram_match: bool = False,
) -> RegistrationResult:
    """Estimate the rigid transform aligning ``moving`` to ``fixed``.

    Optimises the 6-DOF (3-D) / 3-DOF (2-D) rigid Lie parameters
    (``geometry.rigid_exp``) coarse-to-fine so that ``moving`` resampled
    by the result matches ``fixed`` under ``spec.metric``.

    Parameters
    ----------
    moving, fixed
        Single-channel images (2-D or 3-D).  Shapes need not match (the
        warp is built on the ``fixed`` grid); the default ``IndexSpace``
        additionally assumes a shared voxel grid.
    spec
        ``RegistrationSpec`` (pyramid depth, iterations, metric, ...).
    space
        Coordinate space to optimise in (``_space``): ``IndexSpace()``
        (default; voxel-space, shared-grid, on-device) or
        ``WorldSpace(fixed_affine=..., moving_affine=...)`` (physical
        space -- correct under anisotropic voxels and different grids).
    method
        Solver: ``"auto"`` (default; the inverse-compositional fast path --
        ~4-7x the forward throughput -- when its preconditions hold:
        ``IndexSpace`` + a least-squares / SSD metric; the forward
        Gauss-Newton / LM path otherwise), ``"inverse_compositional"`` (force
        it; validates), or ``"forward"``.  The forward path is the parity
        oracle the fast path is tested against.
    init
        Starting transform (A8): ``"identity"`` (default) or ``"moment"`` -- a
        centre-of-mass start (intensity-weighted centroids, plus a per-axis
        diagonal scale for affine) that lands inside the optimiser's basin on a
        large misalignment a single zero start would miss.  ``IndexSpace`` only.
    init_transform
        A self-contained warm-start matrix (a prior recipe's ``result.matrix``),
        taking precedence over ``init`` -- the staged-pipeline hook (e.g. affine
        warm-started from rigid; see :func:`syn_pipeline`).  ``IndexSpace`` only.
    winsorize, histogram_match
        Intensity conditioning before registration (the fMRIPrep front-end):
        ``winsorize=(0.005, 0.995)`` clips each image to those percentiles
        (outlier-robust); ``histogram_match=True`` remaps ``moving`` onto
        ``fixed``'s distribution (within-modality).  Both default off
        (byte-unchanged).  They condition the *estimate*, so ``warped`` is the
        conditioned moving -- apply ``matrix`` to the original for the original
        intensities.

    Returns
    -------
    ``RegistrationResult`` (``matrix``, ``params``, ``warped``,
    ``cost_history``).  ``matrix`` maps ``fixed`` to ``moving`` (index
    coordinates in ``IndexSpace``, world coordinates in ``WorldSpace``);
    ``warped`` is ``moving`` on the ``fixed`` grid.
    """
    moving, fixed = preprocess_images(
        moving,
        fixed,
        winsorize_range=winsorize,
        histogram_match=histogram_match,
    )
    ndim = _spatial_ndim(moving, fixed)
    model = Rigid()
    if init_transform is not None:
        init_matrix = _init_from_transform(
            init_transform, fixed, ndim=ndim, space=space, spec=spec
        )
    else:
        init_matrix = _resolve_init_matrix(
            init, moving, fixed, ndim=ndim, scale=False, space=space, spec=spec
        )
    if _use_inverse_compositional(method, space, spec, model):
        return ic_rigid_register(
            moving, fixed, ndim=ndim, spec=spec, init_matrix=init_matrix
        )
    _check_forward_convergence(spec)
    init_params = (
        None
        if init_matrix is None
        else _rigid_params_from_matrix(init_matrix, ndim)
    )
    return multi_resolution_register(
        moving,
        fixed,
        model=model,
        ndim=ndim,
        spec=spec,
        space=space,
        init_params=init_params,
    )


# Per-restart deterministic init-perturbation scales (affine Lie params): the
# linear gl(n) block and the coarsest-level translation.  Small -- the restarts
# diversify *around* the moment/identity start to dodge the rare GPU MI / CR
# divergence, not to widen the capture range.
_RESTART_LINEAR_JITTER = 0.06
_RESTART_TRANS_JITTER = 1.5


def _affine_multistart(
    moving: Float[Array, '*mspatial'],
    fixed: Float[Array, '*fspatial'],
    *,
    ndim: int,
    spec: RegistrationSpec,
    space: CoordinateSpace,
    base_init: Optional[Array],
    restarts: int,
) -> RegistrationResult:
    """Run ``restarts`` forward affine solves from diversified inits; keep the
    lowest-cost result.

    The GPU joint-histogram scatter-add (``MI`` / ``CorrelationRatio``) is
    non-deterministic, and the 12-DOF affine BFGS occasionally amplifies that
    gradient noise into a catastrophic divergence (the dense-field force and
    6-DOF rigid paths absorb it; ``SSD`` is exact).  Re-solving from small
    deterministic init perturbations and keeping the best metric cost drives the
    failure rate to ~0 -- a diverged run scores far worse, so it is never kept.
    Deterministic (fixed PRNG seed), so the result is reproducible.
    """
    model = Affine()
    base = (
        jnp.zeros(model.n_params(ndim), dtype=moving.dtype)
        if base_init is None
        else base_init
    )
    n_lin = ndim * ndim
    # Score every restart with one metric pinned on the full-res images: a data
    # range drifts as the warp moves, so unpinned costs would not be comparable.
    scorer = pin_metric_ranges(spec.metric, moving, fixed)
    key = jax.random.PRNGKey(0)
    best: Optional[RegistrationResult] = None
    best_cost: Optional[Array] = None
    for k in range(restarts):
        if k == 0:
            init_k = base  # the unperturbed moment / identity start
        else:
            key, k_lin, k_tr = jax.random.split(key, 3)
            init_k = base + jnp.concatenate(
                [
                    _RESTART_LINEAR_JITTER
                    * jax.random.normal(k_lin, (n_lin,), dtype=base.dtype),
                    _RESTART_TRANS_JITTER
                    * jax.random.normal(k_tr, (ndim,), dtype=base.dtype),
                ]
            )
        res = multi_resolution_register(
            moving,
            fixed,
            model=model,
            ndim=ndim,
            spec=spec,
            space=space,
            init_params=init_k,
        )
        cost = scorer.cost(res.warped, fixed)
        if best is None or best_cost is None:
            best, best_cost = res, cost
        else:
            take = cost < best_cost
            best = jax.tree_util.tree_map(
                lambda b, n, t=take: jnp.where(t, n, b), best, res
            )
            best_cost = jnp.where(take, cost, best_cost)
    assert best is not None
    return best


def affine_register(
    moving: Float[Array, '*mspatial'],
    fixed: Float[Array, '*fspatial'],
    *,
    spec: RegistrationSpec = RegistrationSpec(),
    space: CoordinateSpace = IndexSpace(),
    method: str = 'auto',
    init: Literal['identity', 'moment'] = 'identity',
    init_transform: Optional[Float[Array, ' d1 d1']] = None,
    winsorize: Optional[tuple[float, float]] = None,
    histogram_match: bool = False,
    restarts: int = 1,
) -> RegistrationResult:
    """Estimate the affine transform aligning ``moving`` to ``fixed``.

    Optimises the 12-DOF (3-D) / 6-DOF (2-D) affine Lie parameters
    (``geometry.affine_exp`` -- linear block via ``matrix_exp``,
    guaranteeing an invertible map) coarse-to-fine.  For a robust result
    on a large initial misalignment, run ``rigid_register`` first and
    pass its parameters (extended with a zero linear-generator block) as
    a warm start, or compose the two transforms.

    Parameters / returns as ``rigid_register`` (including the ``space`` and
    ``method`` arguments; the inverse-compositional fast path -- where affine's
    large parameter count makes the forward ``jacfwd`` costliest -- engages
    under ``method="auto"`` for ``IndexSpace`` + an SSD metric).  ``init`` adds
    the **moment** start (centroid + per-axis scale), worth more here than for
    rigid -- the affine basin is narrow and a single zero start fails silently
    on a large misalignment.  ``init_transform`` (a prior recipe's
    ``result.matrix``) is the self-contained warm-start hook, taking precedence
    over ``init`` -- e.g. affine warm-started from rigid (see
    :func:`syn_pipeline`); ``IndexSpace`` only.  ``winsorize`` /
    ``histogram_match`` condition the intensities before registration (see
    ``rigid_register``).

    ``restarts`` (default ``1``) runs the **forward** solve from that many
    diversified inits and keeps the lowest-cost result.  Its purpose is the
    histogram metrics on GPU: the joint-histogram scatter-add (``MI`` /
    ``CorrelationRatio``) is non-deterministic, and the 12-DOF affine BFGS
    occasionally amplifies that gradient noise into a catastrophic divergence
    (rigid's 6-DOF and the dense-field force absorb it; ``SSD`` is exact and
    takes the deterministic inverse-compositional path, which ignores
    ``restarts``).  A diverged solve scores far worse, so ``restarts=4``--``6``
    drives the GPU affine-MI failure rate to ~0 at ``restarts``x the forward
    cost; ``1`` is the single-solve default (unchanged, and all that CPU --
    deterministic -- ever needs).
    """
    moving, fixed = preprocess_images(
        moving,
        fixed,
        winsorize_range=winsorize,
        histogram_match=histogram_match,
    )
    ndim = _spatial_ndim(moving, fixed)
    model = Affine()
    spec = _cap_levels(spec, fixed.shape)
    if init_transform is not None:
        init_matrix = _init_from_transform(
            init_transform, fixed, ndim=ndim, space=space, spec=spec
        )
    else:
        init_matrix = _resolve_init_matrix(
            init, moving, fixed, ndim=ndim, scale=True, space=space, spec=spec
        )
    if restarts < 1:
        raise ValueError(f'restarts must be >= 1; got {restarts}.')
    if _use_inverse_compositional(method, space, spec, model):
        # The inverse-compositional SSD path is deterministic; restarts (a GPU
        # MI / CR nondeterminism mitigation) is a no-op here.
        return ic_affine_register(
            moving, fixed, ndim=ndim, spec=spec, init_matrix=init_matrix
        )
    _check_forward_convergence(spec)
    init_params = (
        None
        if init_matrix is None
        else _affine_params_from_matrix(init_matrix, ndim)
    )
    if restarts > 1:
        return _affine_multistart(
            moving,
            fixed,
            ndim=ndim,
            spec=spec,
            space=space,
            base_init=init_params,
            restarts=restarts,
        )
    return multi_resolution_register(
        moving,
        fixed,
        model=model,
        ndim=ndim,
        spec=spec,
        space=space,
        init_params=init_params,
    )


def apply_transform(
    image: Float[Array, '*spatial'],
    result: object,
    *,
    reference_shape: Optional[tuple[int, ...]] = None,
    method: Optional[Interpolator] = None,
    mode: BoundaryMode = 'constant',
    cval: float = 0.0,
) -> Float[Array, '*rspatial']:
    """Apply a recovered transform to an in-memory image / label map.

    Resamples ``image`` -- an array in the **moving** frame of the registration
    that produced ``result`` (a second contrast, a label map, an atlas) -- onto
    the reference (fixed) grid by the recovered transform.  Dispatches on the
    result: a homogeneous matrix (``rigid_register`` / ``affine_register`` --
    the self-contained ``fixed -> moving`` voxel map) or a dense displacement
    field (``greedy_syn_register`` / ``diffeomorphic_demons_register``).
    In-memory resampling only -- image **file** I/O is out of scope (``thrux``).

    Parameters
    ----------
    image
        Single-channel array to resample, on the registration's moving grid.
    result
        A ``RegistrationResult`` (uses ``.matrix``) or a diffeomorphic result
        (uses ``.displacement``).
    reference_shape
        Output grid for a **matrix** result; ``None`` -> ``image.shape``
        (correct when moving and fixed share a grid, the ``IndexSpace`` norm --
        pass the fixed shape for ``WorldSpace`` / differing grids).  A
        displacement result defines its own output grid.
    method
        Interpolation kernel; ``None`` -> ``NearestNeighbour`` for an **integer**
        ``image`` (label-preserving, no interpolation bleed) and ``Linear``
        otherwise.  For anti-aliased multi-label resampling pass
        ``MultiLabel(labels=...)``; for overlap of a warped label map see
        ``metrics.dice`` / ``metrics.jaccard``.
    mode, cval
        Out-of-bounds handling (default ``'constant'`` 0 -- background).

    Returns
    -------
    ``image`` resampled onto the reference grid (same dtype as ``image``).
    """
    image = jnp.asarray(image)
    is_int = jnp.issubdtype(image.dtype, jnp.integer)
    if method is None:
        method = NearestNeighbour() if is_int else Linear()
    disp = getattr(result, 'displacement', None)
    if disp is not None:
        coords = identity_grid(disp.shape[:-1], dtype=disp.dtype) + disp
    else:
        matrix = getattr(result, 'matrix', None)
        if matrix is None:
            raise ValueError(
                'result must carry a .matrix (rigid / affine) or a '
                '.displacement field (SyN / Demons).'
            )
        ndim = matrix.shape[-1] - 1
        shape = (
            image.shape if reference_shape is None else tuple(reference_shape)
        )
        # B1: .matrix is the self-contained fixed -> moving voxel map (the grid
        # centre is baked in), so apply it directly about the origin.
        coords = affine_grid(
            matrix, shape, center=jnp.zeros(ndim, dtype=matrix.dtype)
        )
    # NearestNeighbour returns exact stored values, so the float round-trip is
    # label-exact; out-of-bounds resolve to ``cval`` (0 -> background).
    src = image.astype(coords.dtype) if is_int else image
    warped = spatial_transform(
        src[..., None], coords, mode=mode, cval=cval, method=method
    )[..., 0]
    if is_int:
        warped = jnp.round(warped).astype(image.dtype)
    return warped


class PipelineResult(NamedTuple):
    """Output of :func:`syn_pipeline` -- the composed staged transform.

    Attributes
    ----------
    matrix
        The linear composite -- the affine result (or the rigid result if the
        pipeline stopped at ``'rigid'``), self-contained fixed -> moving.
    displacement
        The **full** fixed -> moving displacement field (``transform='syn'``),
        already including the linear stages; ``None`` for ``'rigid'`` /
        ``'affine'``.  :func:`apply_transform` uses this when present, else
        ``matrix`` -- so ``apply_transform(image, pipeline_result)`` applies the
        whole composite.
    warped
        ``moving`` resampled onto the ``fixed`` grid by the full transform.
    rigid, affine, syn
        The per-stage results (``affine`` / ``syn`` are ``None`` if that stage
        did not run) -- their cost histories, params, Jacobian, etc.
    """

    matrix: Float[Array, ' d1 d1']
    displacement: Optional[Float[Array, '*spatial ndim']]
    warped: Float[Array, '*spatial']
    rigid: RegistrationResult
    affine: Optional[RegistrationResult]
    syn: Optional[SyNResult]


def syn_pipeline(
    moving: Float[Array, '*spatial'],
    fixed: Float[Array, '*spatial'],
    *,
    transform: Literal['rigid', 'affine', 'syn'] = 'syn',
    rigid_spec: Optional[RegistrationSpec] = None,
    affine_spec: Optional[RegistrationSpec] = None,
    syn_spec: Optional[SyNSpec] = None,
    force: Optional[Union[Force, Sequence[Force]]] = None,
    affine_restarts: Optional[int] = None,
) -> PipelineResult:
    """Staged rigid -> affine -> SyN registration in one call.

    The ``antsRegistrationSyN`` workhorse, composed from the leaf recipes: each
    stage **warm-starts** the next -- rigid (moment init) seeds affine
    (``init_transform``), the affine result seeds SyN (``init_affine``) -- so
    the SyN stage's returned ``displacement`` is the **full** fixed -> moving
    map (linear + non-linear).  ``apply_transform(image, result)`` then applies
    the whole composite.  Numerics only -- no image I/O (``thrux``).

    Parameters
    ----------
    moving, fixed
        Single-channel images sharing a grid (the ``IndexSpace`` norm).
    transform
        How far to stage: ``'rigid'``, ``'affine'``, or ``'syn'`` (default).
    rigid_spec, affine_spec, syn_spec, force
        Per-stage configuration; ``None`` -> a cross-modal-robust default
        (``MI`` linear stages, the default ``LNCCForce`` SyN -- the
        antsRegistrationSyN convention).  For a faster **within-modality** run
        pass ``SSD`` linear specs (the inverse-compositional fast path).
    affine_restarts
        Multi-start count for the affine stage; ``None`` -> ``4`` for a
        histogram (``MI`` / ``CR``) affine metric (the non-deterministic GPU
        scatter), else ``1``.

    Returns
    -------
    ``PipelineResult`` (``matrix``, ``displacement``, ``warped``, and the
    per-stage ``rigid`` / ``affine`` / ``syn`` results).
    """
    if transform not in ('rigid', 'affine', 'syn'):
        raise ValueError(
            f"transform must be 'rigid' | 'affine' | 'syn'; got {transform!r}."
        )
    if rigid_spec is None:
        rigid_spec = RegistrationSpec(metric=MI(bins=32))
    if affine_spec is None:
        affine_spec = RegistrationSpec(metric=MI(bins=32))
    if syn_spec is None:
        syn_spec = SyNSpec()
    if affine_restarts is None:
        affine_restarts = 1 if affine_spec.metric.is_least_squares else 4

    rigid = rigid_register(moving, fixed, spec=rigid_spec, init='moment')
    if transform == 'rigid':
        return PipelineResult(
            matrix=rigid.matrix,
            displacement=None,
            warped=rigid.warped,
            rigid=rigid,
            affine=None,
            syn=None,
        )
    affine = affine_register(
        moving,
        fixed,
        spec=affine_spec,
        init_transform=rigid.matrix,
        restarts=affine_restarts,
    )
    if transform == 'affine':
        return PipelineResult(
            matrix=affine.matrix,
            displacement=None,
            warped=affine.warped,
            rigid=rigid,
            affine=affine,
            syn=None,
        )
    syn = greedy_syn_register(
        moving, fixed, spec=syn_spec, force=force, init_affine=affine.matrix
    )
    return PipelineResult(
        matrix=affine.matrix,
        displacement=syn.displacement,
        warped=syn.warped,
        rigid=rigid,
        affine=affine,
        syn=syn,
    )
