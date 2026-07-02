# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pairwise rigid / affine registration recipes.

Pure functions ``(moving, fixed) -> RegistrationResult``, following the same
convention as :func:`nitrix.stats.reml_fit` (a :class:`~typing.NamedTuple` of
arrays, no PyTree module, no atlas or file I/O).  They compose the shared
registration substrate -- the resolution pyramid, the similarity metric, the
transform parametrisation, and the matrix-free optimiser -- with the
orchestration living here so that ``entense`` can wrap it (or re-implement it)
without re-deriving it.

The representative algorithm is intensity-based Gauss-Newton /
Levenberg-Marquardt on the rigid / affine Lie groups (the ``3dvolreg`` / AIR
lineage): coarse-to-fine, second-order, and differentiable.  The similarity
metric (:class:`~nitrix.register.SSD` by default;
:class:`~nitrix.register.LNCC` / :class:`~nitrix.register.MI` /
:class:`~nitrix.register.CorrelationRatio` for intensity-robust or cross-modal
cases) and the schedule are set on the :class:`~nitrix.register.RegistrationSpec`.
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

# C3 / B2: rejecting an early-exit request on a path that cannot honour it (the
# scalar/BFGS forward optimiser is monolithic) is now the single
# ``resolve_convergence_mode`` gate, applied where the forward driver resolves
# the mode (``multi_resolution_register``) -- no separate recipe-level check.


def _image_moments(img: Array, ndim: int) -> tuple[Array, Array]:
    """Intensity-weighted centroid and per-axis standard deviation.

    Treats the image as a non-negative density (negative intensities are
    clipped to zero) and computes the first two moments in voxel coordinates.

    Parameters
    ----------
    img
        Single-channel image, shape ``(*spatial,)`` with ``ndim`` spatial axes.
    ndim
        Number of spatial dimensions (2 or 3).

    Returns
    -------
    centroid : Array
        Intensity-weighted centroid, shape ``(ndim,)``, in voxel coordinates.
    std : Array
        Per-axis intensity-weighted standard deviation, shape ``(ndim,)``.
    """
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
    """Centre-of-mass / moment initialisation, an index-space affine.

    Aligns the intensity-weighted centroids (translation) and -- when ``scale``
    is set (the affine case) -- the per-axis spread (a **diagonal** scale, with
    no rotation: a moment match cannot fix the principal-axis sign, so a rotation
    init risks a *worse* start).  In the :func:`~nitrix.geometry.affine_grid`
    centring convention (sample voxel :math:`i` at :math:`M(i - c) + t + c`),
    :math:`M = \\operatorname{diag}(\\sigma_m / \\sigma_f)` (or the identity),
    and :math:`t` maps the fixed centroid onto the moving centroid.  This is the
    cheap robust start for when a single zero init sits outside the narrow affine
    basin.

    Parameters
    ----------
    moving, fixed
        Single-channel images, each with ``ndim`` spatial axes.
    ndim
        Number of spatial dimensions (2 or 3).
    scale
        If True, include the diagonal per-axis scale block (the affine case);
        if False, the linear block is the identity (the rigid / translation-only
        case).

    Returns
    -------
    Array
        Homogeneous ``(ndim + 1, ndim + 1)`` index-space affine matrix mapping
        the fixed grid onto the moving grid.
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
    """Resolve the ``init`` argument to a coarsest-level init matrix (or None).

    The driver starts the coarse-to-fine loop at the coarsest level and upscales
    the translation column to each finer grid, so the init's translation must be
    expressed at the coarsest scale -- the full-resolution moment translation
    divided by the pyramid's coarse-to-fine factor.  The linear (scale) block is
    resolution-independent and passes through unchanged.

    Parameters
    ----------
    init
        Either ``'identity'`` (no init; returns None) or ``'moment'`` (a
        centre-of-mass start).  Any other value raises.
    moving, fixed
        Single-channel images, each with ``ndim`` spatial axes.
    ndim
        Number of spatial dimensions (2 or 3).
    scale
        Whether the moment init includes the diagonal per-axis scale block
        (affine) or only the translation (rigid).
    space
        Coordinate space; ``'moment'`` init is defined for
        :class:`~nitrix.register.IndexSpace` only and raises otherwise.
    spec
        Registration spec supplying the pyramid depth and factor used to rescale
        the translation to the coarsest level.

    Returns
    -------
    Optional[Array]
        The coarsest-level homogeneous ``(ndim + 1, ndim + 1)`` init matrix, or
        None for identity init (the driver then starts from zero parameters).
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
    """Convert a self-contained recipe matrix into a coarsest-level init.

    Maps a centred (self-contained) recipe matrix into the internal
    coarsest-level about-origin init, matching what ``_resolve_init_matrix``
    returns.  This is the warm-start hook: :func:`rigid_register` and
    :func:`affine_register` return the self-contained matrix (with the grid
    centre baked in), so to *start* an optimisation from a prior stage's result
    we de-centre it to be about the origin and rescale the translation to the
    coarsest pyramid level (the driver upscales it back).  Index-space only --
    this is the staged-pipeline frame.

    Parameters
    ----------
    transform
        A self-contained homogeneous ``(ndim + 1, ndim + 1)`` matrix (a prior
        recipe's ``result.matrix``), with the grid centre baked in.
    fixed
        The fixed image, whose shape fixes the grid centre used for de-centring.
    ndim
        Number of spatial dimensions (2 or 3).
    space
        Coordinate space; this warm-start is defined for
        :class:`~nitrix.register.IndexSpace` only and raises otherwise.
    spec
        Registration spec supplying the pyramid depth and factor used to rescale
        the translation to the coarsest level.

    Returns
    -------
    Array
        The coarsest-level about-origin homogeneous init matrix.
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
    """Shorten the affine pyramid so its coarsest level stays reliable.

    Reduces the pyramid depth so the coarsest level keeps at least
    ``_MIN_COARSE_AXIS`` voxels per axis, below which the affine Hessian is too
    poorly determined and the fit can diverge.  It is loud (it warns when it
    shortens) and a no-op when the requested depth already satisfies the bound.
    An explicit per-level ``iterations`` tuple is honoured by keeping its finest
    entries.

    Parameters
    ----------
    spec
        The requested registration spec (pyramid depth, factor, iterations).
    shape
        Spatial shape of the fixed image; its smallest axis fixes the coarsest
        grid size.

    Returns
    -------
    RegistrationSpec
        The original ``spec`` if the depth is already safe, otherwise a copy
        with ``levels`` (and any per-level ``iterations`` tuple) shortened.
    """
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


def _resolve_ic_dispatch(
    method: str,
    space: CoordinateSpace,
    spec: RegistrationSpec,
    model: TransformModel,
    *,
    mask: Optional[Array] = None,
) -> bool:
    """Resolve the inverse-compositional-versus-forward dispatch.

    Returns whether to take the inverse-compositional fast path (a
    constant-template Hessian, roughly 4-7x the forward Gauss-Newton / LM
    throughput).  It applies to a **rigid or affine** least-squares (SSD)
    registration in :class:`~nitrix.register.IndexSpace` (where the template is
    linearised in voxel coordinates) with an unmasked cost.  A fixed-grid cost
    ``mask`` routes to the forward path regardless: the constant-template Hessian
    is a full-grid linearisation built once from the whole reference, so it
    cannot honour a per-voxel mask, and an explicit
    ``method='inverse_compositional'`` combined with a mask is a loud error.

    Parameters
    ----------
    method
        Dispatch selector: ``'auto'`` (fast path when its preconditions hold,
        otherwise the forward path), ``'inverse_compositional'`` (force the fast
        path and validate its preconditions), or ``'forward'`` (always forward).
    space
        Coordinate space; the fast path requires
        :class:`~nitrix.register.IndexSpace`.
    spec
        Registration spec; the fast path requires a least-squares metric and a
        Gauss-Newton / LM (or ``'auto'``) optimiser.
    model
        Transform model; the fast path requires a rigid or affine model.
    mask
        Optional per-voxel cost weight; when present, the fast path is disabled.

    Returns
    -------
    bool
        True to take the inverse-compositional fast path, False for the forward
        Gauss-Newton / LM path.
    """
    if mask is not None:
        if method == 'inverse_compositional':
            raise ValueError(
                "method='inverse_compositional' cannot honour a cost mask (its "
                'constant-template Hessian is a full-grid linearisation); use '
                "method='auto' (routes to the forward path) or 'forward'."
            )
        if method not in ('auto', 'forward'):
            raise ValueError(
                f'method must be "auto", "forward", or '
                f'"inverse_compositional"; got {method!r}.'
            )
        return False
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
    mask: Optional[Float[Array, '*fspatial']] = None,
    winsorize: Optional[tuple[float, float]] = None,
    histogram_match: bool = False,
) -> RegistrationResult:
    """Estimate the rigid transform aligning ``moving`` to ``fixed``.

    Optimises the 6-DOF (3-D) / 3-DOF (2-D) rigid Lie parameters
    (:func:`~nitrix.geometry.rigid_exp`) coarse-to-fine so that ``moving``
    resampled by the result matches ``fixed`` under ``spec.metric``.

    Parameters
    ----------
    moving, fixed
        Single-channel images (2-D or 3-D).  Shapes need not match (the
        warp is built on the ``fixed`` grid); the default
        :class:`~nitrix.register.IndexSpace` additionally assumes a shared voxel
        grid.
    spec
        The :class:`~nitrix.register.RegistrationSpec` (pyramid depth,
        iterations, metric, ...).  **Metric choice:**
        :class:`~nitrix.register.SSD` (default) within-modality -- the fast
        inverse-compositional path; :class:`~nitrix.register.MI` /
        :class:`~nitrix.register.CorrelationRatio` cross-modal;
        :class:`~nitrix.register.LNCC` under a smooth intensity bias.
        **Grid:** the default :class:`~nitrix.register.IndexSpace` assumes
        ``moving`` and ``fixed`` share a (roughly isotropic) voxel grid -- use
        :class:`~nitrix.register.WorldSpace` for anisotropic voxels or differing
        grids (a rigid in index space shears in physical space).
    space
        Coordinate space to optimise in: :class:`~nitrix.register.IndexSpace`
        (default; voxel-space, shared-grid, on-device) or
        ``WorldSpace(fixed_affine=..., moving_affine=...)`` (physical
        space -- correct under anisotropic voxels and different grids).
    method
        Solver: ``"auto"`` (default; the inverse-compositional fast path --
        roughly 4-7x the forward throughput -- when its preconditions hold:
        :class:`~nitrix.register.IndexSpace` and a least-squares / SSD metric;
        the forward Gauss-Newton / LM path otherwise), ``"inverse_compositional"``
        (force it; validates), or ``"forward"``.  The forward path is the parity
        oracle the fast path is tested against.
    init
        Starting transform: ``"identity"`` (default) or ``"moment"`` -- a
        centre-of-mass start (intensity-weighted centroids, plus a per-axis
        diagonal scale for affine) that lands inside the optimiser's basin on a
        large misalignment a single zero start would miss.
        :class:`~nitrix.register.IndexSpace` only.
    init_transform
        A self-contained warm-start matrix (a prior recipe's ``result.matrix``),
        taking precedence over ``init`` -- the staged-pipeline hook (e.g. affine
        warm-started from rigid; see :func:`syn_pipeline`).
        :class:`~nitrix.register.IndexSpace` only.
    mask
        Optional non-negative per-voxel weight on the ``fixed`` grid: the
        cost is restricted to / weighted by it (a brain mask, lesion exclusion,
        FoV), with out-of-mask voxels ignored -- for
        :class:`~nitrix.register.MI` / :class:`~nitrix.register.CorrelationRatio`
        it gates the joint-histogram scatter, not just the reduction.  A mask
        routes to the **forward** path (the inverse-compositional
        constant-template Hessian assumes the full grid);
        ``method='inverse_compositional'`` with a mask raises.
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
    RegistrationResult
        A :class:`~nitrix.register.RegistrationResult` carrying ``matrix``,
        ``params``, ``warped``, and ``cost_history``.  ``matrix`` maps ``fixed``
        to ``moving`` (index coordinates in
        :class:`~nitrix.register.IndexSpace`, world coordinates in
        :class:`~nitrix.register.WorldSpace`); ``warped`` is ``moving`` resampled
        on the ``fixed`` grid.
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
    if _resolve_ic_dispatch(method, space, spec, model, mask=mask):
        return ic_rigid_register(
            moving, fixed, ndim=ndim, spec=spec, init_matrix=init_matrix
        )
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
        mask=mask,
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
    mask: Optional[Array] = None,
) -> RegistrationResult:
    """Run ``restarts`` forward affine solves and keep the lowest-cost result.

    Runs the forward affine solve from that many diversified inits and keeps the
    lowest-cost result.  The GPU joint-histogram scatter-add
    (:class:`~nitrix.register.MI` / :class:`~nitrix.register.CorrelationRatio`)
    is non-deterministic, and the 12-DOF affine BFGS occasionally amplifies that
    gradient noise into a catastrophic divergence (the dense-field force and
    6-DOF rigid paths absorb it; :class:`~nitrix.register.SSD` is exact).
    Re-solving from small deterministic init perturbations and keeping the best
    metric cost drives the failure rate to nearly zero -- a diverged run scores
    far worse, so it is never kept.  A fixed PRNG seed makes the result
    reproducible.

    Parameters
    ----------
    moving, fixed
        Single-channel images, each with ``ndim`` spatial axes.
    ndim
        Number of spatial dimensions (2 or 3).
    spec
        Registration spec (pyramid depth, iterations, metric, ...).
    space
        Coordinate space to optimise in.
    base_init
        Optional base init parameter vector (the unperturbed moment / identity
        start); None starts from zero parameters.
    restarts
        Number of forward solves to run from diversified inits.
    mask
        Optional per-voxel cost weight on the ``fixed`` grid.

    Returns
    -------
    RegistrationResult
        The lowest-cost result across the restarts, scored under a metric pinned
        on the full-resolution images so the costs are comparable.
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
            mask=mask,
        )
        cost = scorer.cost(res.warped, fixed, mask=mask)
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
    mask: Optional[Float[Array, '*fspatial']] = None,
    winsorize: Optional[tuple[float, float]] = None,
    histogram_match: bool = False,
    restarts: int = 1,
) -> RegistrationResult:
    """Estimate the affine transform aligning ``moving`` to ``fixed``.

    Optimises the 12-DOF (3-D) / 6-DOF (2-D) affine Lie parameters
    (:func:`~nitrix.geometry.affine_exp` -- linear block via
    :func:`~nitrix.linalg.matrix_exp`, guaranteeing an invertible map)
    coarse-to-fine.  For a robust result on a large initial misalignment, run
    :func:`rigid_register` first and pass its parameters (extended with a zero
    linear-generator block) as a warm start, or compose the two transforms.

    The ``space`` and ``method`` arguments behave as in :func:`rigid_register`;
    the inverse-compositional fast path -- where affine's large parameter count
    makes the forward ``jacfwd`` costliest -- engages under ``method="auto"`` for
    :class:`~nitrix.register.IndexSpace` and an SSD metric.

    Parameters
    ----------
    moving, fixed
        Single-channel images (2-D or 3-D); see :func:`rigid_register`.
    spec
        Registration spec (pyramid, iterations, metric, ...); see
        :func:`rigid_register`.
    space
        Coordinate space to optimise in; see :func:`rigid_register`.
    method
        Solver dispatch (``"auto"`` / ``"inverse_compositional"`` /
        ``"forward"``); see :func:`rigid_register`.
    init
        Starting transform: ``"identity"`` (default) or the **moment** start
        (centroid plus per-axis scale), worth more here than for rigid -- the
        affine basin is narrow and a single zero start fails silently on a large
        misalignment.  :class:`~nitrix.register.IndexSpace` only.
    init_transform
        A self-contained warm-start matrix (a prior recipe's ``result.matrix``),
        taking precedence over ``init`` -- e.g. affine warm-started from rigid
        (see :func:`syn_pipeline`).  :class:`~nitrix.register.IndexSpace` only.
    mask
        Optional per-voxel cost weight on the ``fixed`` grid (see
        :func:`rigid_register`); restricts the cost to a region and routes to the
        forward path.
    winsorize, histogram_match
        Intensity conditioning before registration (see :func:`rigid_register`).
    restarts
        Number of diversified forward solves (default ``1``), keeping the
        lowest-cost result -- a pure basin-of-attraction lever.  It **no longer**
        doubles as the GPU MI determinism fix: the joint histogram now takes a
        deterministic one-hot-matmul path on GPU at affine pyramid sizes (at most
        roughly 200k voxels per level), so :class:`~nitrix.register.MI` /
        :class:`~nitrix.register.CorrelationRatio` recover reproducibly at
        ``restarts=1``.  (Historically the GPU joint-histogram scatter-add was
        non-deterministic and the 12-DOF affine BFGS occasionally amplified that
        noise into a catastrophic divergence, so ``restarts`` of 4-6 was the
        mitigation; that is retired for sizes under the gate.
        :class:`~nitrix.register.SSD` is exact and takes the
        inverse-compositional path, which ignores ``restarts``.)  Above the gate
        (the finest full-resolution level), or for a genuinely multi-basin start,
        ``restarts > 1`` still helps.

    Returns
    -------
    RegistrationResult
        A :class:`~nitrix.register.RegistrationResult`; see
        :func:`rigid_register` for the fields and conventions.
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
    if _resolve_ic_dispatch(method, space, spec, model, mask=mask):
        # The inverse-compositional SSD path is deterministic; restarts (a GPU
        # MI / CR nondeterminism mitigation) is a no-op here.
        return ic_affine_register(
            moving, fixed, ndim=ndim, spec=spec, init_matrix=init_matrix
        )
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
            mask=mask,
        )
    return multi_resolution_register(
        moving,
        fixed,
        model=model,
        ndim=ndim,
        spec=spec,
        space=space,
        init_params=init_params,
        mask=mask,
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
    result: a homogeneous matrix (:func:`rigid_register` / :func:`affine_register`
    -- the self-contained ``fixed -> moving`` voxel map) or a dense displacement
    field (:func:`greedy_syn_register` /
    :func:`diffeomorphic_demons_register`).  In-memory resampling only -- image
    **file** I/O is out of scope.

    Parameters
    ----------
    image
        Single-channel array to resample, on the registration's moving grid.
    result
        A :class:`~nitrix.register.RegistrationResult` (uses ``.matrix``) or a
        diffeomorphic result (uses ``.displacement``).
    reference_shape
        Output grid for a **matrix** result; ``None`` -> ``image.shape``
        (correct when moving and fixed share a grid, the
        :class:`~nitrix.register.IndexSpace` norm -- pass the fixed shape for
        :class:`~nitrix.register.WorldSpace` / differing grids).  A displacement
        result defines its own output grid.
    method
        Interpolation kernel; ``None`` -> ``NearestNeighbour`` for an **integer**
        ``image`` (label-preserving, no interpolation bleed) and ``Linear``
        otherwise.  For anti-aliased multi-label resampling pass
        ``MultiLabel(labels=...)``; for overlap of a warped label map see
        :func:`nitrix.metrics.dice` / :func:`nitrix.metrics.jaccard`.
    mode, cval
        Out-of-bounds handling (default ``'constant'`` 0 -- background).

    Returns
    -------
    Float[Array, '*rspatial']
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
        Single-channel images sharing a grid (the
        :class:`~nitrix.register.IndexSpace` norm).
    transform
        How far to stage: ``'rigid'``, ``'affine'``, or ``'syn'`` (default).
    rigid_spec, affine_spec, syn_spec, force
        Per-stage configuration; ``None`` -> a cross-modal-robust default
        (:class:`~nitrix.register.MI` linear stages, the default
        :class:`~nitrix.register.LNCCForce` SyN -- the antsRegistrationSyN
        convention).  For a faster **within-modality** run pass
        :class:`~nitrix.register.SSD` linear specs (the inverse-compositional
        fast path).
    affine_restarts
        Multi-start count for the affine stage; ``None`` -> ``4`` for a
        histogram (:class:`~nitrix.register.MI` /
        :class:`~nitrix.register.CorrelationRatio`) affine metric (the
        non-deterministic GPU scatter), else ``1``.

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
