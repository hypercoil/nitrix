# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Greedy symmetric diffeomorphic registration (SyN-style, LNCC-driven).

The diffeomorphic-quality jump over the SSD log-Demons recipe: a
**symmetric** formulation under the **local cross-correlation** metric --
the design doc's stated greedy-SyN representative (Avants 2008, the greedy
variant: gradient flow + Gaussian regularisation, no geodesic shooting).

Two stationary velocity fields are maintained, ``v_fwd`` (moving -> midpoint)
and ``v_inv`` (fixed -> midpoint); each iteration warps both images to the
shared midpoint and ascends the local CC there, driving them together
symmetrically:

1. ``A = M∘exp(v_fwd)``, ``B = F∘exp(v_inv)`` (both at the midpoint).
2. LNCC forces ``u = lncc_grad(·)·∇(·)`` -- the analytic local-CC gradient
   (``metrics.lncc_grad``) times the warped-image gradient -- normalised to
   a bounded per-step displacement.
3. **fluid** smooth ``u``; ``v += u`` (ascent); **diffusion** smooth ``v``.

The final moving->fixed deformation is ``φ = (id+s_fwd) ∘ (id+s_inv)⁻¹``
(midpoint composition, via ``invert_displacement`` + ``compose_displacement``),
a diffeomorphism by construction (``jacobian_det`` for the folding QA).

Reuses the SVF substrate (scaling-and-squaring exp, warp, gradient,
Gaussian, velocity composition) exactly as the Demons recipe -- the only
new kernel is the LNCC force.  ``spacing`` makes the force / regularisation
anisotropy-correct, identically to ``DemonsSpec`` (the relative-spacing
treatment).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence, Union

from jaxtyping import Array, Float

from ..geometry import (
    compose_displacement,
    gaussian_pyramid,
    integrate_velocity_field,
    invert_displacement,
)
from ..geometry._interpolate import BoundaryMode
from ._force import Force, LNCCForce, resolve_force_schedule
from ._svf import (
    _relative_spacing,
    finalize_with_init,
    prewarp_moving,
    resolve_init_displacement,
    svf_coarse_to_fine,
    symmetric_level,
)

__all__ = [
    'SyNSpec',
    'SyNResult',
    'greedy_syn_register',
]


@dataclass(frozen=True)
class SyNSpec:
    """Static configuration for the greedy-SyN recipe (jit-static).

    Attributes
    ----------
    levels
        Gaussian-pyramid resolutions (coarse-to-fine).
    iterations
        SyN iterations per level.
    radius
        LNCC window radius (size ``2·radius + 1`` per axis).
    step
        Maximum per-iteration voxel displacement (the force field is
        normalised to this bound, the greedy-SyN gradient-step convention).
    sigma_fluid
        Gaussian sigma for the fluid (per-update) regularisation.
    sigma_diffusion
        Gaussian sigma for the diffusion (accumulated-velocity)
        regularisation.
    n_steps
        Scaling-and-squaring steps for ``exp(v)``.
    spacing
        Per-axis voxel spacing; only the anisotropy is used (the relative
        spacing, ``None`` -> isotropic), identically to ``DemonsSpec``.
    pyramid_factor, pyramid_sigma
        Pyramid downsample factor / anti-alias sigma.
    boundary_mode
        Out-of-bounds handling for the warps.
    """

    levels: int = 3
    iterations: int = 80
    radius: int = 2
    step: float = 0.25
    sigma_fluid: float = 1.0
    sigma_diffusion: float = 1.5
    n_steps: int = 5
    spacing: Optional[Union[float, tuple[float, ...]]] = None
    pyramid_factor: float = 2.0
    pyramid_sigma: Optional[float] = None
    boundary_mode: BoundaryMode = 'nearest'


class SyNResult(NamedTuple):
    """Output of the greedy-SyN recipe.

    Attributes
    ----------
    forward_velocity
        Velocity field ``v_fwd`` (moving -> midpoint), ``(*spatial, ndim)``.
    inverse_velocity
        Velocity field ``v_inv`` (fixed -> midpoint).
    displacement
        The moving -> fixed deformation ``φ`` as a displacement field (add
        ``identity_grid`` for the absolute deformation).
    warped
        ``moving`` resampled by ``φ`` onto the ``fixed`` grid.
    jacobian_det
        ``det J`` of ``φ`` -- the diffeomorphism QA map (all positive ⇒ no
        folding).
    cost_history
        Concatenated per-iteration ``1 − lncc`` trace.
    """

    forward_velocity: Float[Array, '*spatial ndim']
    inverse_velocity: Float[Array, '*spatial ndim']
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
    rel_spacing: Optional[tuple[float, ...]],
) -> tuple[Array, Array, Array]:
    """Run the symmetric SyN iterations on one resolution.

    Thin wrapper over the metric-generic symmetric-midpoint SVF driver
    (``_svf.symmetric_level``) with the level's ``force`` (the analytic
    :class:`LNCCForce` by default) -- the recipe's symmetric structure (warp
    both images to the midpoint, ascend the similarity in each direction) plus
    its trust-region step clamp and anisotropy-aware regularisation.  Rolled
    with ``lax.scan``; differentiable for the unrolled path.
    """
    return symmetric_level(
        moving,
        fixed,
        v_fwd,
        v_inv,
        force=force,
        ndim=ndim,
        iterations=spec.iterations,
        n_steps=spec.n_steps,
        boundary_mode=spec.boundary_mode,
        sigma_fluid=spec.sigma_fluid,
        sigma_diffusion=spec.sigma_diffusion,
        step=spec.step,
        rel_spacing=rel_spacing,
    )


def greedy_syn_register(
    moving: Float[Array, '*spatial'],
    fixed: Float[Array, '*spatial'],
    *,
    spec: SyNSpec = SyNSpec(),
    force: Optional[Union[Force, Sequence[Force]]] = None,
    init_affine: Optional[Float[Array, ' d1 d1']] = None,
    init_displacement: Optional[Float[Array, '*spatial ndim']] = None,
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
        ``SyNSpec`` controlling the schedule, the LNCC window, and the
        regularisation.
    force
        The driving :class:`Force` (``_force``).  ``None`` (default) uses the
        analytic ``LNCCForce(spec.radius)``; a single ``Force`` overrides it at
        every level (e.g. ``MetricForce(MI())`` for cross-modal / multimodal
        symmetric registration); a length-``spec.levels`` **coarse-to-fine**
        sequence sets a per-level schedule.
    init_affine, init_displacement
        Optional warm-start / multi-stage init (at most one).  ``init_affine``
        is a fixed->moving index matrix (as ``rigid_register`` /
        ``affine_register`` return); ``init_displacement`` is a fixed-grid
        displacement field (e.g. a SynthMorph network output).  ``moving`` is
        pre-warped by it onto the ``fixed`` grid and the **residual** is
        registered; the returned ``displacement`` / ``warped`` /
        ``jacobian_det`` are the **total** (init then residual) map, while the
        velocity fields are the residual SVFs.

    Returns
    -------
    ``SyNResult`` (``forward_velocity``, ``inverse_velocity``,
    ``displacement``, ``warped``, ``jacobian_det``, ``cost_history``).
    """
    ndim = fixed.ndim
    if ndim not in (2, 3) or moving.ndim != ndim:
        raise ValueError(
            f'greedy SyN supports 2-D / 3-D single-channel images; got '
            f'moving {moving.shape}, fixed {fixed.shape}.'
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
    pyr_m = gaussian_pyramid(
        moving_reg[..., None],
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
    rel_spacing = _relative_spacing(spec.spacing, ndim)
    forces = resolve_force_schedule(
        force, default=LNCCForce(spec.radius), levels=spec.levels
    )

    def level_solve(
        level: int, m_l: Array, f_l: Array, state: tuple[Array, ...]
    ) -> tuple[tuple[Array, ...], Array]:
        v_fwd, v_inv = state
        v_fwd, v_inv, hist = _syn_level(
            m_l,
            f_l,
            v_fwd,
            v_inv,
            force=forces[level],
            spec=spec,
            ndim=ndim,
            rel_spacing=rel_spacing,
        )
        return (v_fwd, v_inv), hist

    (v_fwd, v_inv), cost_history = svf_coarse_to_fine(
        pyr_m,
        pyr_f,
        ndim=ndim,
        dtype=dtype,
        n_fields=2,
        level_solve=level_solve,
    )

    s_fwd = integrate_velocity_field(
        v_fwd, n_steps=spec.n_steps, mode=spec.boundary_mode
    )
    s_inv = integrate_velocity_field(
        v_inv, n_steps=spec.n_steps, mode=spec.boundary_mode
    )
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
