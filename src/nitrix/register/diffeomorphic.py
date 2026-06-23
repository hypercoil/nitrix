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

The above is the **algebra** (log-domain SVF) representation -- the exact
oracle.  The default ``representation='group'`` (``DemonsSpec``) instead carries
the *displacement* and uses the greedy compositive update (warp directly,
compose the regularised increment -- no per-iteration ``exp``, ~2 gathers/iter
vs ~7), recovering the velocity once at finalisation via ``geometry.field_log``.
Greedy is not the SVF fixed point, so the two agree on synthetic recovery to
tolerance, not field-wise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence, Union

from jaxtyping import Array, Float

from ..geometry import (
    field_log,
    gaussian_pyramid,
    integrate_velocity_field,
)
from ._converge import (
    resolve_convergence_mode,
)
from ._core import resolve_iterations
from ._force import DemonsForce, Force, resolve_force_schedule
from ._preprocess import preprocess_images
from ._svf import (
    SVFSpec,
    _relative_spacing,
    finalize_with_init,
    group_single_sided_level,
    pin_force_ranges,
    prewarp_moving,
    resolve_init_displacement,
    resolve_smoothing,
    single_sided_level,
    smooth_pyramid,
    svf_coarse_to_fine,
)

__all__ = [
    'DemonsSpec',
    'DiffeomorphicResult',
    'diffeomorphic_demons_register',
]


@dataclass(frozen=True)
class DemonsSpec(SVFSpec):
    """Static configuration for the log-Demons recipe (jit-static).

    Embeds :class:`._svf.SVFSpec` (G1) for the shared schedule / regularisation /
    convergence fields (``levels``, ``iterations``, ``sigma_fluid`` /
    ``sigma_diffusion``, ``spacing``, ``pyramid_factor`` / ``pyramid_sigma``,
    ``boundary_mode``, ``representation``, ``mode`` / ``convergence``,
    ``compute_velocity``); Demons carries a single displacement / velocity, so
    its ``representation`` / ``compute_velocity`` recover the one ``velocity``
    field (``DiffeomorphicResult.velocity``).  The Demons-specific force knobs:

    Attributes
    ----------
    n_steps
        Scaling-and-squaring steps for ``exp(v)`` (the algebra path; ``'auto'``
        is not used here -- the count is jit-static).
    alpha
        Force normalisation: ``denom = |J|² + α²(F − M∘φ)²``.  Larger ``α`` damps
        the step where the intensity difference is large.
    bch_order
        Log-update order: ``1`` additive (default), ``2`` adds ½[v,u].
    """

    n_steps: int = 6
    alpha: float = 0.4
    bch_order: int = 1


class DiffeomorphicResult(NamedTuple):
    """Output of the diffeomorphic Demons recipe.

    Attributes
    ----------
    velocity
        The stationary velocity field, ``(*spatial, ndim)``, or ``None`` when
        ``DemonsSpec.compute_velocity`` is ``False`` (the default -- see that
        flag; the velocity recovery is skipped to save compile + runtime).
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

    velocity: Optional[Float[Array, '*spatial ndim']]
    displacement: Float[Array, '*spatial ndim']
    warped: Float[Array, '*spatial']
    jacobian_det: Float[Array, '*spatial']
    cost_history: Float[Array, ' h']


def _demons_level(
    moving: Array,
    fixed: Array,
    v: Array,
    *,
    force: Force,
    spec: DemonsSpec,
    ndim: int,
    iterations: int,
    rel_spacing: Optional[tuple[float, ...]],
    mask: Optional[Array] = None,
    restrict: Optional[tuple[float, ...]] = None,
) -> tuple[Array, Array]:
    """Run the Demons iterations on one resolution; return ``(v, costs)``.

    Thin wrapper over the metric-generic single-sided SVF driver
    (``_svf.single_sided_level``) with the level's ``force`` (the closed-form
    ESM :class:`DemonsForce` by default, or any :class:`Force` the caller
    supplies).  The driver hoists ``∇fixed`` out of the iteration, rolls it
    with ``lax.scan`` (so the level compiles one iteration), and stays
    differentiable for the unrolled gradient path.

    ``rel_spacing`` (anisotropy-only; ``None`` for isotropic) makes the physics
    axis-correct: the gradient is taken in physical units, the ESM force is
    converted back to the voxel-native field, and the fluid/diffusion sigmas
    become per-axis -- all reducing to the voxel behaviour when ``None``.
    """
    return single_sided_level(
        moving,
        fixed,
        v,
        force=force,
        ndim=ndim,
        iterations=iterations,
        n_steps=spec.n_steps,
        boundary_mode=spec.boundary_mode,
        sigma_fluid=spec.sigma_fluid,
        sigma_diffusion=spec.sigma_diffusion,
        bch_order=spec.bch_order,
        step=None,
        rel_spacing=rel_spacing,
        mask=mask,
        restrict=restrict,
        convergence=resolve_convergence_mode(
            spec.mode,
            spec.convergence,
            supports_early_exit=True,
            path='Demons',
        ),
    )


def diffeomorphic_demons_register(
    moving: Float[Array, '*spatial'],
    fixed: Float[Array, '*spatial'],
    *,
    spec: DemonsSpec = DemonsSpec(),
    force: Optional[Union[Force, Sequence[Force]]] = None,
    init_affine: Optional[Float[Array, ' d1 d1']] = None,
    init_displacement: Optional[Float[Array, '*spatial ndim']] = None,
    mask: Optional[Float[Array, '*spatial']] = None,
    restrict: Optional[tuple[float, ...]] = None,
    winsorize: Optional[tuple[float, float]] = None,
    histogram_match: bool = False,
    smoothing_sigma: Optional[Union[float, Sequence[float]]] = None,
) -> DiffeomorphicResult:
    """Diffeomorphic registration of ``moving`` to ``fixed`` (log-Demons).

    Estimates a stationary velocity field ``v`` (coarse-to-fine) whose
    exponential warps ``moving`` onto ``fixed``.  The result is a
    diffeomorphism by construction; ``jacobian_det`` lets the caller
    assert no folding (all positive).

    Parameters
    ----------
    moving, fixed
        Single-channel images (2-D or 3-D).  Identical shape unless an init
        (below) resamples ``moving`` onto the ``fixed`` grid.
    spec
        ``DemonsSpec`` controlling the schedule and regularisation.
    force
        The driving :class:`Force` (``_force``).  ``None`` (default) uses the
        closed-form ESM ``DemonsForce(spec.alpha)``; a single ``Force``
        overrides it at every level (e.g. ``MetricForce(MI())`` for cross-modal
        deformable registration); a length-``spec.levels`` **coarse-to-fine**
        sequence sets a per-level schedule (a cheap force coarse, a high-signal
        one at the finest level).
    init_affine, init_displacement
        Optional warm-start / multi-stage init (at most one).  ``init_affine``
        is a fixed->moving index matrix (as ``rigid_register`` /
        ``affine_register`` return); ``init_displacement`` is a fixed-grid
        displacement field (e.g. a SynthMorph network output).  ``moving`` is
        pre-warped by it onto the ``fixed`` grid and the **residual** is
        registered; the returned ``displacement`` / ``warped`` /
        ``jacobian_det`` are the **total** (init then residual) map, while
        ``velocity`` is the residual SVF.
    mask
        Optional fixed-grid weight field (``(*spatial,)``, e.g. a brain mask)
        gating the force to a region: the masked area drives the deformation,
        the rest follows by regularisation.
    restrict
        Optional length-``ndim`` per-axis weight on the deformation (ANTs
        ``--restrict-deformation``): a ``0`` suppresses deformation along that
        axis (e.g. ``(1, 1, 0)`` for in-plane-only).
    winsorize, histogram_match
        Intensity conditioning before registration (the fMRIPrep front-end; see
        ``register.rigid_register``).  Both default off.
    smoothing_sigma
        Optional per-level smoothing applied to each pyramid level **independent
        of the shrink** (ANTs ``-s``, decoupled from ``-f``): a scalar (all
        levels) or a length-``levels`` **coarse-to-fine** sequence (e.g.
        ``2x1x0`` -> ``(2, 1, 0)``).  ``None`` (default) leaves the pyramid's own
        anti-alias as the only smoothing (byte-unchanged).

    Returns
    -------
    ``DiffeomorphicResult`` (``velocity``, ``displacement``, ``warped``,
    ``jacobian_det``, ``cost_history``).  ``velocity`` is ``None`` unless
    ``spec.compute_velocity`` (the default skips its ``field_log`` recovery).

    Notes
    -----
    **Cohort registration (D4).**  This is a pure ``(moving, fixed) -> result``
    function, so register a *cohort* to a shared reference with ``jax.vmap``::

        jax.vmap(lambda m: diffeomorphic_demons_register(m, fixed, spec=spec))(moving_stack)

    The batch-aggregate early-exit comes for free: under ``mode='early_exit'`` the
    per-subject ``lax.while_loop`` runs (via ``vmap``) to the **all-lanes** exit,
    the slowest subject setting the trip count -- the same pattern ``volreg`` uses
    for its frames.  No dedicated cohort driver is needed.

    Warning
    -------
    The Gaussian regulariser dispatches engines per backend (parallel FIR on
    GPU, recursive Young-van Vliet on CPU), which differ by ~1-2 % near the
    edges -- so a pair can yield slightly different deformations on GPU vs CPU
    (recovery accuracy is unaffected).  For a reproducible study, register every
    subject of a cohort on the **same backend**; do not mix CPU- and GPU-computed
    warps in one analysis.
    """
    ndim = fixed.ndim
    if ndim not in (2, 3) or moving.ndim != ndim:
        raise ValueError(
            f'diffeomorphic registration supports 2-D / 3-D single-channel '
            f'images; got moving {moving.shape}, fixed {fixed.shape}.'
        )
    moving, fixed = preprocess_images(
        moving,
        fixed,
        winsorize_range=winsorize,
        histogram_match=histogram_match,
    )
    if restrict is not None and len(restrict) != ndim:
        raise ValueError(
            f'restrict must have length ndim={ndim}; got {len(restrict)}.'
        )
    dtype = moving.dtype
    init_disp = resolve_init_displacement(
        init_affine, init_displacement, fixed.shape, dtype
    )
    if init_disp is None and moving.shape != fixed.shape:
        raise ValueError(
            f'moving and fixed must share shape ({moving.shape} vs '
            f'{fixed.shape}); pass init_affine / init_displacement to resample '
            f'a different grid.'
        )
    moving_reg = prewarp_moving(
        moving, init_disp, fixed.shape, dtype, spec.boundary_mode
    )
    smoothing = resolve_smoothing(smoothing_sigma, spec.levels)
    pyr_m = smooth_pyramid(
        gaussian_pyramid(
            moving_reg[..., None],
            levels=spec.levels,
            factor=spec.pyramid_factor,
            sigma=spec.pyramid_sigma,
        ),
        smoothing,
        ndim,
    )
    pyr_f = smooth_pyramid(
        gaussian_pyramid(
            fixed[..., None],
            levels=spec.levels,
            factor=spec.pyramid_factor,
            sigma=spec.pyramid_sigma,
        ),
        smoothing,
        ndim,
    )

    # Anisotropy-only spacing -- level-independent, so computed once.
    rel_spacing = _relative_spacing(spec.spacing, ndim)
    forces = resolve_force_schedule(
        force, default=DemonsForce(spec.alpha), levels=spec.levels
    )
    # Pin any histogram-force ranges once from the full-res images (stationary
    # objective; see pin_force_ranges) before the pyramid.
    forces = [pin_force_ranges(f, moving_reg, fixed) for f in forces]
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

    iters_per_level = resolve_iterations(spec.iterations, spec.levels)

    def level_solve(
        level: int, m_l: Array, f_l: Array, state: tuple[Array, ...]
    ) -> tuple[tuple[Array, ...], Array]:
        (field,) = state
        mask_l = None if pyr_mask is None else pyr_mask[level][..., 0]
        if spec.representation == 'algebra':
            field, hist = _demons_level(
                m_l,
                f_l,
                field,
                force=forces[level],
                spec=spec,
                ndim=ndim,
                iterations=iters_per_level[level],
                rel_spacing=rel_spacing,
                mask=mask_l,
                restrict=restrict,
            )
        else:
            field, hist = group_single_sided_level(
                m_l,
                f_l,
                field,
                force=forces[level],
                ndim=ndim,
                iterations=iters_per_level[level],
                boundary_mode=spec.boundary_mode,
                sigma_fluid=spec.sigma_fluid,
                sigma_diffusion=spec.sigma_diffusion,
                step=None,
                rel_spacing=rel_spacing,
                mask=mask_l,
                restrict=restrict,
                convergence=resolve_convergence_mode(
                    spec.mode,
                    spec.convergence,
                    supports_early_exit=True,
                    path='Demons',
                ),
            )
        return (field,), hist

    (state,), cost_history = svf_coarse_to_fine(
        pyr_m,
        pyr_f,
        ndim=ndim,
        dtype=dtype,
        n_fields=1,
        level_solve=level_solve,
    )

    # Algebra mode carries the velocity (exp it for the residual); group mode
    # carries the displacement directly and recovers the velocity via field_log.
    # The velocity feeds neither warped/displacement/jacobian_det, so it is only
    # recovered when explicitly requested (spec.compute_velocity) -- under jit the
    # field_log loop nest is then never traced (no compile, no runtime).
    velocity: Optional[Array]
    if spec.representation == 'algebra':
        residual = integrate_velocity_field(
            state, n_steps=spec.n_steps, mode=spec.boundary_mode
        )
        velocity = state if spec.compute_velocity else None
    else:
        residual = state
        velocity = (
            field_log(residual, n_sqrt=spec.n_steps, mode=spec.boundary_mode)
            if spec.compute_velocity
            else None
        )
    total, warped, det = finalize_with_init(
        moving,
        residual,
        init_disp,
        shape=fixed.shape,
        dtype=dtype,
        boundary_mode=spec.boundary_mode,
    )
    return DiffeomorphicResult(
        velocity=velocity,
        displacement=total,
        warped=warped,
        jacobian_det=det,
        cost_history=cost_history,
    )
