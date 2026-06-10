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
from typing import Any, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..geometry import (
    compose_displacement,
    gaussian_pyramid,
    identity_grid,
    integrate_velocity_field,
    invert_displacement,
    jacobian_det_displacement,
    spatial_gradient,
    spatial_transform,
    upsample,
)
from ..geometry._interpolate import BoundaryMode
from ..metrics import lncc, lncc_grad
from .diffeomorphic import _relative_spacing, _smooth_vector

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


def _normalise_step(u: Array, step: float) -> Array:
    """Cap the force field's largest voxel displacement at ``step``.

    A trust-region clamp (``min(1, step/‖u‖_max)``), not a scale-to.  The
    two coincide when the force exceeds the cap (both divide by ``‖u‖_max``);
    they differ only *below* it, where the clamp keeps the raw gradient
    magnitude rather than amplifying it to a full step.  Under a fixed
    iteration budget (no convergence gate) that is the safer discipline: a
    shrinking force keeps shrinking, and a small / low-signal update is not
    inflated -- scale-to-step would force a full ``step`` in whatever
    direction the (often spurious, flat-region) maximum points.

    **Contingent on the fixed-budget scheme.**  This choice is *because* the
    forward is a fixed-length ``lax.scan`` (no convergence gate).  If the
    ``while_loop`` early-exit (``docs/feature-requests/
    registration-early-stopping-while-loop.md``) is adopted, a convergence
    gate would bound the constant-step dithering that motivates the clamp,
    making scale-to-step (the ANTS choice) viable again -- so revisit
    clamp-vs-scale here if that lands.

    Note the LNCC force does **not** vanish at a perfect match -- the
    metric's ``eps`` guard leaves ``cc < 1`` in low-variance windows -- so
    it is the symmetric forward/inverse cancellation, not a vanishing force
    or this clamp, that zeroes the *net* deformation there.
    """
    norm = jnp.sqrt(jnp.sum(u * u, axis=-1))
    scale = jnp.minimum(1.0, step / (jnp.max(norm) + 1e-12))
    return u * scale


def _syn_level(
    moving: Array,
    fixed: Array,
    v_fwd: Array,
    v_inv: Array,
    *,
    spec: SyNSpec,
    ndim: int,
    rel_spacing: Optional[tuple[float, ...]],
) -> tuple[Array, Array, Array]:
    """Run the symmetric SyN iterations on one resolution.

    Rolled with ``lax.scan`` (carry = the two velocity fields), so the
    level compiles one iteration; differentiable for the unrolled path.
    """
    id_grid = identity_grid(fixed.shape, dtype=fixed.dtype)
    grad_spacing: Union[float, tuple[float, ...]] = (
        1.0 if rel_spacing is None else rel_spacing
    )
    rel_arr = (
        None
        if rel_spacing is None
        else jnp.asarray(rel_spacing, dtype=fixed.dtype)
    )
    sigma_fluid: Union[float, tuple[float, ...]] = (
        spec.sigma_fluid
        if rel_spacing is None
        else tuple(spec.sigma_fluid / r for r in rel_spacing)
    )
    sigma_diffusion: Union[float, tuple[float, ...]] = (
        spec.sigma_diffusion
        if rel_spacing is None
        else tuple(spec.sigma_diffusion / r for r in rel_spacing)
    )

    def warp_to_mid(image: Array, v: Array) -> Array:
        s = integrate_velocity_field(
            v, n_steps=spec.n_steps, mode=spec.boundary_mode
        )
        return spatial_transform(
            image[..., None], id_grid + s, mode=spec.boundary_mode
        )[..., 0]

    def force(scalar: Array, warped: Array) -> Array:
        u = scalar[..., None] * spatial_gradient(warped, spacing=grad_spacing)
        if rel_arr is not None:
            u = u / rel_arr
        return _smooth_vector(_normalise_step(u, spec.step), sigma_fluid, ndim)

    def step(carry: tuple[Array, Array], _: Any) -> tuple[Any, Array]:
        v_fwd, v_inv = carry
        a = warp_to_mid(moving, v_fwd)
        b = warp_to_mid(fixed, v_inv)
        # Symmetric local-CC ascent: each image flows toward the other.
        v_fwd = v_fwd + force(lncc_grad(a, b, radius=spec.radius), a)
        v_inv = v_inv + force(lncc_grad(b, a, radius=spec.radius), b)
        v_fwd = _smooth_vector(v_fwd, sigma_diffusion, ndim)
        v_inv = _smooth_vector(v_inv, sigma_diffusion, ndim)
        cost = 1.0 - lncc(a, b, radius=spec.radius)
        return (v_fwd, v_inv), cost

    (v_fwd, v_inv), costs = jax.lax.scan(
        step, (v_fwd, v_inv), xs=None, length=spec.iterations
    )
    return v_fwd, v_inv, costs


def greedy_syn_register(
    moving: Float[Array, '*spatial'],
    fixed: Float[Array, '*spatial'],
    *,
    spec: SyNSpec = SyNSpec(),
) -> SyNResult:
    """Greedy symmetric diffeomorphic registration (LNCC-driven).

    Estimates symmetric forward / inverse velocity fields (coarse-to-fine)
    whose midpoint composition warps ``moving`` onto ``fixed``.  The result
    is a diffeomorphism by construction; ``jacobian_det`` lets the caller
    assert no folding (all positive).  The local cross-correlation metric
    is robust to smooth intensity inhomogeneity (the SSD Demons recipe is
    not), and the symmetric formulation removes the moving/fixed asymmetry.

    Parameters
    ----------
    moving, fixed
        Single-channel images of identical shape (2-D or 3-D).
    spec
        ``SyNSpec`` controlling the schedule, the LNCC window, and the
        regularisation.

    Returns
    -------
    ``SyNResult`` (``forward_velocity``, ``inverse_velocity``,
    ``displacement``, ``warped``, ``jacobian_det``, ``cost_history``).
    """
    if moving.shape != fixed.shape:
        raise ValueError(
            f'moving and fixed must share shape; got {moving.shape} '
            f'vs {fixed.shape}.'
        )
    ndim = moving.ndim
    if ndim not in (2, 3):
        raise ValueError(
            f'greedy SyN supports 2-D / 3-D single-channel images; got '
            f'shape {moving.shape}.'
        )
    dtype = moving.dtype
    pyr_m = gaussian_pyramid(
        moving[..., None],
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

    v_fwd: Optional[Array] = None
    v_inv: Optional[Array] = None
    prev_shape: Optional[tuple[int, ...]] = None
    histories = []
    for level in range(spec.levels - 1, -1, -1):
        m_l = pyr_m[level][..., 0]
        f_l = pyr_f[level][..., 0]
        shape_l = f_l.shape
        if v_fwd is None:
            v_fwd = jnp.zeros(shape_l + (ndim,), dtype=dtype)
            v_inv = jnp.zeros(shape_l + (ndim,), dtype=dtype)
        else:
            assert v_inv is not None
            ratio = jnp.asarray(shape_l, dtype=dtype) / jnp.asarray(
                prev_shape, dtype=dtype
            )
            v_fwd = upsample(v_fwd, shape_l) * ratio
            v_inv = upsample(v_inv, shape_l) * ratio
        v_fwd, v_inv, hist = _syn_level(
            m_l,
            f_l,
            v_fwd,
            v_inv,
            spec=spec,
            ndim=ndim,
            rel_spacing=rel_spacing,
        )
        histories.append(hist)
        prev_shape = shape_l

    assert v_fwd is not None and v_inv is not None
    s_fwd = integrate_velocity_field(
        v_fwd, n_steps=spec.n_steps, mode=spec.boundary_mode
    )
    s_inv = integrate_velocity_field(
        v_inv, n_steps=spec.n_steps, mode=spec.boundary_mode
    )
    # moving -> fixed: fixed-grid --(id+s_inv)⁻¹--> midpoint --(id+s_fwd)--> moving
    s_inv_inverse = invert_displacement(s_inv, mode=spec.boundary_mode)
    displacement = compose_displacement(
        s_fwd, s_inv_inverse, mode=spec.boundary_mode
    )
    id_grid = identity_grid(moving.shape, dtype=dtype)
    warped = spatial_transform(
        moving[..., None], id_grid + displacement, mode=spec.boundary_mode
    )[..., 0]
    det = jacobian_det_displacement(displacement)
    return SyNResult(
        forward_velocity=v_fwd,
        inverse_velocity=v_inv,
        displacement=displacement,
        warped=warped,
        jacobian_det=det,
        cost_history=jnp.concatenate(histories),
    )
