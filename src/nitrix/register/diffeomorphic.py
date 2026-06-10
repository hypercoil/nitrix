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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence, Union

from jaxtyping import Array, Float

from ..geometry import (
    gaussian_pyramid,
    integrate_velocity_field,
)
from ..geometry._interpolate import BoundaryMode
from ._force import DemonsForce, Force, resolve_force_schedule
from ._svf import (
    _relative_spacing,
    finalize_with_init,
    prewarp_moving,
    resolve_init_displacement,
    single_sided_level,
    svf_coarse_to_fine,
)

__all__ = [
    'DemonsSpec',
    'DiffeomorphicResult',
    'diffeomorphic_demons_register',
]


@dataclass(frozen=True)
class DemonsSpec:
    """Static configuration for the log-Demons recipe (jit-static).

    Attributes
    ----------
    levels
        Gaussian-pyramid resolutions (coarse-to-fine).
    iterations
        Demons iterations per level.
    sigma_fluid
        Gaussian sigma for the fluid (per-update) regularisation.
    sigma_diffusion
        Gaussian sigma for the diffusion (accumulated-velocity)
        regularisation.
    n_steps
        Scaling-and-squaring steps for ``exp(v)``.
    alpha
        Force normalisation: ``denom = |J|² + α²(F − M∘φ)²``.  Larger
        ``α`` damps the step where the intensity difference is large.
    bch_order
        Log-update order: ``1`` additive (default), ``2`` adds ½[v,u].
    spacing
        Per-axis voxel spacing (physical units); ``float`` or length-``ndim``
        tuple.  ``None`` (default) registers in voxel units.  Only the
        **anisotropy** is used -- the regularisation and the ESM force are
        made physically isotropic by the *relative* spacing
        ``spacing / geomean(spacing)`` (level-independent: the
        coarse-to-fine align-corners scale cancels in the ratio).  So
        isotropic spacing reduces exactly to ``None``; an anisotropic grid
        is corrected for the bias where a voxel-isotropic Gaussian / force
        is physically anisotropic.  The velocity field stays voxel-native
        (the ESM force is converted mm -> voxel before it updates ``v``).
    pyramid_factor, pyramid_sigma
        Pyramid downsample factor / anti-alias sigma.
    boundary_mode
        Out-of-bounds handling for the warps.
    """

    levels: int = 3
    iterations: int = 80
    sigma_fluid: float = 1.0
    sigma_diffusion: float = 1.5
    n_steps: int = 6
    alpha: float = 0.4
    bch_order: int = 1
    spacing: Optional[Union[float, tuple[float, ...]]] = None
    pyramid_factor: float = 2.0
    pyramid_sigma: Optional[float] = None
    boundary_mode: BoundaryMode = 'nearest'


class DiffeomorphicResult(NamedTuple):
    """Output of the diffeomorphic Demons recipe.

    Attributes
    ----------
    velocity
        The stationary velocity field, ``(*spatial, ndim)``.
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

    velocity: Float[Array, '*spatial ndim']
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
    rel_spacing: Optional[tuple[float, ...]],
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
        iterations=spec.iterations,
        n_steps=spec.n_steps,
        boundary_mode=spec.boundary_mode,
        sigma_fluid=spec.sigma_fluid,
        sigma_diffusion=spec.sigma_diffusion,
        bch_order=spec.bch_order,
        step=None,
        rel_spacing=rel_spacing,
    )


def diffeomorphic_demons_register(
    moving: Float[Array, '*spatial'],
    fixed: Float[Array, '*spatial'],
    *,
    spec: DemonsSpec = DemonsSpec(),
    force: Optional[Union[Force, Sequence[Force]]] = None,
    init_affine: Optional[Float[Array, ' d1 d1']] = None,
    init_displacement: Optional[Float[Array, '*spatial ndim']] = None,
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

    Returns
    -------
    ``DiffeomorphicResult`` (``velocity``, ``displacement``, ``warped``,
    ``jacobian_det``, ``cost_history``).
    """
    ndim = fixed.ndim
    if ndim not in (2, 3) or moving.ndim != ndim:
        raise ValueError(
            f'diffeomorphic registration supports 2-D / 3-D single-channel '
            f'images; got moving {moving.shape}, fixed {fixed.shape}.'
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

    # Anisotropy-only spacing -- level-independent, so computed once.
    rel_spacing = _relative_spacing(spec.spacing, ndim)
    forces = resolve_force_schedule(
        force, default=DemonsForce(spec.alpha), levels=spec.levels
    )

    def level_solve(
        level: int, m_l: Array, f_l: Array, state: tuple[Array, ...]
    ) -> tuple[tuple[Array, ...], Array]:
        (v,) = state
        v, hist = _demons_level(
            m_l,
            f_l,
            v,
            force=forces[level],
            spec=spec,
            ndim=ndim,
            rel_spacing=rel_spacing,
        )
        return (v,), hist

    (velocity,), cost_history = svf_coarse_to_fine(
        pyr_m,
        pyr_f,
        ndim=ndim,
        dtype=dtype,
        n_fields=1,
        level_solve=level_solve,
    )

    residual = integrate_velocity_field(
        velocity, n_steps=spec.n_steps, mode=spec.boundary_mode
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
