# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Greedy symmetric diffeomorphic registration (SyN-style, LNCC-driven).

The diffeomorphic-quality jump over the sum-of-squared-differences log-Demons
recipe: a **symmetric** formulation under the **local cross-correlation**
metric -- the greedy variant of symmetric normalisation (gradient flow plus
Gaussian regularisation, no geodesic shooting).

Two stationary velocity fields are maintained, :math:`v_{\text{fwd}}` (moving
to midpoint) and :math:`v_{\text{inv}}` (fixed to midpoint); each iteration
warps both images to the shared midpoint and ascends the local
cross-correlation there, driving them together symmetrically:

1. :math:`A = M \circ \exp(v_{\text{fwd}})`,
   :math:`B = F \circ \exp(v_{\text{inv}})` (both at the midpoint).
2. Local cross-correlation forces
   :math:`u = \operatorname{lncc\_grad}(\cdot) \cdot \nabla(\cdot)` -- the
   analytic local-CC gradient (:func:`lncc_grad`) times the warped-image
   gradient -- normalised to a bounded per-step displacement.
3. **Fluid** smooth :math:`u`; :math:`v \mathrel{+}= u` (ascent);
   **diffusion** smooth :math:`v`.

The final moving-to-fixed deformation is
:math:`\varphi = (\mathrm{id} + s_{\text{fwd}}) \circ (\mathrm{id} + s_{\text{inv}})^{-1}`
(midpoint composition, via :func:`invert_displacement` and
:func:`compose_displacement`), a diffeomorphism by construction (its Jacobian
determinant serves for the folding quality assurance).

Reuses the stationary-velocity-field substrate (scaling-and-squaring
exponential, warp, gradient, Gaussian, velocity composition) exactly as the
Demons recipe -- the only new kernel is the local cross-correlation force.
The ``spacing`` field makes the force and regularisation anisotropy-correct,
identically to the Demons specification (the relative-spacing treatment).

The above is the **algebra** (log-domain stationary velocity field)
representation -- the exact oracle.  The default ``representation='group'``
(see :class:`SyNSpec`) instead carries the two *displacements* and uses the
greedy compositive update (warp directly, compose the regularised increment --
no per-iteration exponential, roughly 4 gathers per iteration versus 12; the
single inversion stays at finalisation), recovering the velocities via
:func:`field_log`.

References
----------
Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2008). Symmetric
diffeomorphic image registration with cross-correlation: evaluating automated
labeling of elderly and neurodegenerative brain. *Medical Image Analysis*,
12(1), 26-41. https://doi.org/10.1016/j.media.2007.06.004
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, Optional, Sequence, Union

from jaxtyping import Array, Float

from ..geometry import (
    compose_displacement,
    field_log,
    gaussian_pyramid,
    integrate_velocity_field,
    invert_displacement,
)
from ._converge import (
    resolve_convergence_mode,
)
from ._core import resolve_iterations
from ._force import Force, LNCCForce, resolve_force_schedule
from ._preprocess import preprocess_images
from ._svf import (
    SVFSpec,
    _relative_spacing,
    finalize_with_init,
    group_symmetric_level,
    pin_force_ranges,
    prewarp_moving,
    resolve_init_displacement,
    resolve_smoothing,
    smooth_pyramid,
    svf_coarse_to_fine,
    symmetric_level,
)

__all__ = [
    'SyNSpec',
    'SyNResult',
    'greedy_syn_register',
]


@dataclass(frozen=True)
class SyNSpec(SVFSpec):
    r"""Static configuration for the greedy-SyN recipe (jit-static).

    Embeds :class:`._svf.SVFSpec` for the shared schedule / regularisation /
    convergence fields (``levels``, ``iterations``, ``sigma_fluid`` /
    ``sigma_diffusion``, ``spacing``, ``pyramid_factor`` / ``pyramid_sigma``,
    ``boundary_mode``, ``representation``, ``mode`` / ``convergence``,
    ``compute_velocity``); SyN is *symmetric*, so it carries a pair of
    displacements / velocities -- ``representation`` / ``compute_velocity``
    recover both ``forward_velocity`` and ``inverse_velocity``, and ``spacing``
    additionally makes the LNCC window physically isotropic.  The SyN-specific
    force / step knobs:

    Attributes
    ----------
    radius
        Local cross-correlation window radius (size :math:`2 \cdot r + 1` per
        axis, where :math:`r` is ``radius``).
    step
        Maximum per-iteration voxel displacement (the force field is normalised
        to this bound, the greedy-SyN gradient-step convention).
    step_mode
        How ``step`` bounds the update (group driver).  ``'clamp'`` (default) is
        the trust-region clamp (:math:`\min(1, \text{step} / \lVert u \rVert)`)
        plus a per-step Jacobian backtracking guard.  ``'normalize'`` is the
        ANTs recipe: a magnitude-invariant **scale-to**
        (:math:`\text{step} / \lVert u \rVert`, so a small-magnitude force such
        as ``LNCCForce(derivative='center')`` is not under-stepped) and no
        Jacobian backtracking (the bounded smoothed step is diffeomorphic by
        construction).  Default ``'clamp'`` is byte-identical.
    n_steps
        Scaling-and-squaring steps for ``exp(v)`` (the algebra path).
    """

    radius: int = 2
    step: float = 0.25
    step_mode: Literal['clamp', 'normalize'] = 'clamp'
    n_steps: int = 5


class SyNResult(NamedTuple):
    r"""Output of the greedy-SyN recipe.

    Attributes
    ----------
    forward_velocity
        Velocity field :math:`v_{\text{fwd}}` (moving to midpoint), shape
        ``(*spatial, ndim)``, or ``None`` when ``SyNSpec.compute_velocity`` is
        ``False`` (the default -- the :func:`field_log` recovery is skipped to
        save compile and runtime).
    inverse_velocity
        Velocity field :math:`v_{\text{inv}}` (fixed to midpoint), shape
        ``(*spatial, ndim)``, or ``None`` (see above).
    displacement
        The moving-to-fixed deformation :math:`\varphi` as a displacement
        field, shape ``(*spatial, ndim)`` (add the output of
        :func:`identity_grid` for the absolute deformation).
    warped
        ``moving`` resampled by :math:`\varphi` onto the ``fixed`` grid, shape
        ``(*spatial,)``.
    jacobian_det
        :math:`\det J` of :math:`\varphi`, shape ``(*spatial,)`` -- the
        diffeomorphism quality-assurance map (all positive implies no folding).
    cost_history
        Concatenated per-iteration :math:`1 - \operatorname{lncc}` trace, shape
        ``(h,)``.
    """

    forward_velocity: Optional[Float[Array, '*spatial ndim']]
    inverse_velocity: Optional[Float[Array, '*spatial ndim']]
    displacement: Float[Array, '*spatial ndim']
    warped: Float[Array, '*spatial']
    jacobian_det: Float[Array, '*spatial']
    cost_history: Float[Array, ' h']


def _syn_level(
    moving: Array,
    fixed: Array,
    v_fwd: Array,
    v_inv: Array,
    *,
    force: Force,
    spec: SyNSpec,
    ndim: int,
    iterations: int,
    rel_spacing: Optional[tuple[float, ...]],
    mask: Optional[Array] = None,
    restrict: Optional[tuple[float, ...]] = None,
) -> tuple[Array, Array, Array]:
    """Run the symmetric SyN iterations on one resolution.

    Thin wrapper over the metric-generic symmetric-midpoint stationary-velocity
    driver (:func:`._svf.symmetric_level`) with the level's ``force`` (the
    analytic :class:`LNCCForce` by default) -- the recipe's symmetric structure
    (warp both images to the midpoint, ascend the similarity in each direction)
    plus its trust-region step clamp and anisotropy-aware regularisation.
    Rolled with ``lax.scan``; differentiable for the unrolled path.

    Parameters
    ----------
    moving, fixed
        The two single-channel images at this resolution level, identical
        spatial shape.
    v_fwd, v_inv
        Current forward (moving to midpoint) and inverse (fixed to midpoint)
        velocity fields to update, each of shape ``(*spatial, ndim)``.
    force
        The driving :class:`Force` for this level (the analytic
        :class:`LNCCForce` by default), bound per step against the other image
        at the midpoint.
    spec
        The :class:`SyNSpec` supplying the regularisation, step, boundary, and
        scaling-and-squaring settings shared with this level.
    ndim
        Spatial dimensionality of the images (2 or 3).
    iterations
        Number of ascent iterations to run at this level.
    rel_spacing
        Optional per-axis relative voxel spacing (length ``ndim``) making the
        force and regularisation anisotropy-correct, or ``None`` for isotropic.
    mask
        Optional region mask at this level's resolution gating both
        half-forces, or ``None`` for no masking.
    restrict
        Optional length-``ndim`` per-axis weight on the deformation (a ``0``
        suppresses deformation along that axis), applied to both half-forces,
        or ``None``.

    Returns
    -------
    tuple of Array
        ``(v_fwd, v_inv, cost_history)``: the updated forward and inverse
        velocity fields (each ``(*spatial, ndim)``) and the per-iteration cost
        trace.
    """
    return symmetric_level(
        moving,
        fixed,
        v_fwd,
        v_inv,
        force=force,
        ndim=ndim,
        iterations=iterations,
        n_steps=spec.n_steps,
        boundary_mode=spec.boundary_mode,
        sigma_fluid=spec.sigma_fluid,
        sigma_diffusion=spec.sigma_diffusion,
        step=spec.step,
        rel_spacing=rel_spacing,
        mask=mask,
        restrict=restrict,
        convergence=resolve_convergence_mode(
            spec.mode,
            spec.convergence,
            supports_early_exit=True,
            path='SyN',
        ),
    )


def greedy_syn_register(
    moving: Float[Array, '*spatial'],
    fixed: Float[Array, '*spatial'],
    *,
    spec: SyNSpec = SyNSpec(),
    force: Optional[Union[Force, Sequence[Force]]] = None,
    init_affine: Optional[Float[Array, ' d1 d1']] = None,
    init_displacement: Optional[Float[Array, '*spatial ndim']] = None,
    mask: Optional[Float[Array, '*spatial']] = None,
    restrict: Optional[tuple[float, ...]] = None,
    winsorize: Optional[tuple[float, float]] = None,
    histogram_match: bool = False,
    smoothing_sigma: Optional[Union[float, Sequence[float]]] = None,
) -> SyNResult:
    """Greedy symmetric diffeomorphic registration (LNCC-driven by default).

    Estimates symmetric forward / inverse velocity fields (coarse-to-fine)
    whose midpoint composition warps ``moving`` onto ``fixed``.  The result
    is a diffeomorphism by construction; ``jacobian_det`` lets the caller
    assert no folding (all positive).  The local cross-correlation metric
    is robust to smooth intensity inhomogeneity (the SSD Demons recipe is
    not), and the symmetric formulation removes the moving/fixed asymmetry.

    Parameters
    ----------
    moving, fixed
        Single-channel images (2-D or 3-D).  Identical shape unless an init
        (below) resamples ``moving`` onto the ``fixed`` grid.
    spec
        The :class:`SyNSpec` controlling the schedule, the local
        cross-correlation window, and the regularisation.
    force
        The driving :class:`Force`.  ``None`` (default) uses the analytic
        ``LNCCForce(spec.radius)``; a single :class:`Force` overrides it at
        every level (e.g. ``MetricForce(MI())`` for cross-modal / multimodal
        symmetric registration); a length-``spec.levels`` **coarse-to-fine**
        sequence sets a per-level schedule.
    init_affine, init_displacement
        Optional warm-start / multi-stage initialisation (at most one).
        ``init_affine`` is a fixed-to-moving index matrix, shape ``(d1, d1)``
        (as :func:`rigid_register` and :func:`affine_register` return);
        ``init_displacement`` is a fixed-grid displacement field of shape
        ``(*spatial, ndim)`` (e.g. a SynthMorph network output).  ``moving`` is
        pre-warped by it onto the ``fixed`` grid and the **residual** is
        registered; the returned ``displacement`` / ``warped`` /
        ``jacobian_det`` are the **total** (init then residual) map, while the
        velocity fields are the residual stationary velocity fields.
    mask
        Optional single-channel weight / region map on the ``fixed`` grid,
        shape ``(*spatial,)``, gating the force to a region.  It is downsampled
        onto each pyramid level.  ``None`` (default) applies no mask.
    restrict
        Optional length-``ndim`` per-axis weight on the deformation (ANTs
        ``--restrict-deformation``): a ``0`` suppresses deformation along that
        axis (applied to both half-forces, e.g. ``(1, 1, 0)`` for in-plane-only).
    winsorize, histogram_match
        Intensity conditioning before registration (the fMRIPrep front-end; see
        :func:`rigid_register`).  ``winsorize`` is a ``(low, high)`` quantile
        pair clamping intensity outliers; ``histogram_match`` matches the moving
        histogram to the fixed one.  Both default off.
    smoothing_sigma
        Optional per-level smoothing applied to each pyramid level **independent
        of the shrink** (ANTs ``-s``): a scalar or a length-``levels``
        coarse-to-fine sequence.  ``None`` (default) is byte-unchanged.

    Returns
    -------
    SyNResult
        A :class:`SyNResult` with fields ``forward_velocity``,
        ``inverse_velocity``, ``displacement``, ``warped``, ``jacobian_det``,
        and ``cost_history``.  The velocities are ``None`` unless
        ``spec.compute_velocity`` (the default skips their :func:`field_log`
        recovery).

    Notes
    -----
    **Cohort registration.**  This is a pure ``(moving, fixed) -> result``
    function, so register a *cohort* to a shared reference with ``jax.vmap``::

        jax.vmap(lambda m: greedy_syn_register(m, fixed, spec=spec))(moving_stack)

    The batch-aggregate early-exit comes for free: under ``mode='early_exit'`` the
    per-subject ``lax.while_loop`` runs (via ``vmap``) to the **all-lanes** exit,
    the slowest subject setting the trip count -- the same pattern ``volreg`` uses
    for its frames.  No dedicated cohort driver is needed.

    Warnings
    --------
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
            f'greedy SyN supports 2-D / 3-D single-channel images; got '
            f'moving {moving.shape}, fixed {fixed.shape}.'
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
    rel_spacing = _relative_spacing(spec.spacing, ndim)
    forces = resolve_force_schedule(
        force, default=LNCCForce(spec.radius), levels=spec.levels
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
        f_fwd, f_inv = state
        mask_l = None if pyr_mask is None else pyr_mask[level][..., 0]
        if spec.representation == 'algebra':
            f_fwd, f_inv, hist = _syn_level(
                m_l,
                f_l,
                f_fwd,
                f_inv,
                force=forces[level],
                spec=spec,
                ndim=ndim,
                iterations=iters_per_level[level],
                rel_spacing=rel_spacing,
                mask=mask_l,
                restrict=restrict,
            )
        else:
            f_fwd, f_inv, hist = group_symmetric_level(
                m_l,
                f_l,
                f_fwd,
                f_inv,
                force=forces[level],
                ndim=ndim,
                iterations=iters_per_level[level],
                boundary_mode=spec.boundary_mode,
                sigma_fluid=spec.sigma_fluid,
                sigma_diffusion=spec.sigma_diffusion,
                step=spec.step,
                step_mode=spec.step_mode,
                rel_spacing=rel_spacing,
                mask=mask_l,
                restrict=restrict,
                convergence=resolve_convergence_mode(
                    spec.mode,
                    spec.convergence,
                    supports_early_exit=True,
                    path='SyN',
                ),
            )
        return (f_fwd, f_inv), hist

    (state_fwd, state_inv), cost_history = svf_coarse_to_fine(
        pyr_m,
        pyr_f,
        ndim=ndim,
        dtype=dtype,
        n_fields=2,
        level_solve=level_solve,
    )

    # Algebra carries the half-velocities (exp each); group carries the half-
    # displacements directly (recover the velocities via field_log).  The final
    # composition + single inversion is identical in both modes.  The velocities
    # feed none of displacement/warped/jacobian_det, so they are recovered only
    # when requested (spec.compute_velocity) -- under jit the two field_log loop
    # nests are then never traced (no compile, no runtime).
    v_fwd: Optional[Array]
    v_inv: Optional[Array]
    if spec.representation == 'algebra':
        s_fwd = integrate_velocity_field(
            state_fwd, n_steps=spec.n_steps, mode=spec.boundary_mode
        )
        s_inv = integrate_velocity_field(
            state_inv, n_steps=spec.n_steps, mode=spec.boundary_mode
        )
        v_fwd = state_fwd if spec.compute_velocity else None
        v_inv = state_inv if spec.compute_velocity else None
    else:
        s_fwd, s_inv = state_fwd, state_inv
        if spec.compute_velocity:
            v_fwd = field_log(
                s_fwd, n_sqrt=spec.n_steps, mode=spec.boundary_mode
            )
            v_inv = field_log(
                s_inv, n_sqrt=spec.n_steps, mode=spec.boundary_mode
            )
        else:
            v_fwd = v_inv = None
    # moving -> fixed residual: fixed --(id+s_inv)⁻¹--> midpoint --(id+s_fwd)-->
    s_inv_inverse = invert_displacement(s_inv, mode=spec.boundary_mode)
    residual = compose_displacement(
        s_fwd, s_inv_inverse, mode=spec.boundary_mode
    )
    total, warped, det = finalize_with_init(
        moving,
        residual,
        init_disp,
        shape=fixed.shape,
        dtype=dtype,
        boundary_mode=spec.boundary_mode,
    )
    return SyNResult(
        forward_velocity=v_fwd,
        inverse_velocity=v_inv,
        displacement=total,
        warped=warped,
        jacobian_det=det,
        cost_history=cost_history,
    )
