# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Shared substrate for the stationary-velocity-field (SVF) recipes.

The log-Demons (``diffeomorphic``) and greedy-SyN (``_syn``) recipes are
the same family: a coarse-to-fine flow of one or two voxel-unit velocity
fields, regularised by separable Gaussian smoothing, with the only
difference being the per-level update.  This module holds what they share
-- the anisotropy-aware smoothing helpers and the coarse-to-fine scaffold
(``svf_coarse_to_fine``) -- so each recipe is just its per-level update
plus its finalisation, not a re-derived multiresolution loop.

(The matrix-transform driver, by contrast, carries a small parameter
*vector* and a coordinate-space sampler, not a velocity *field*, so it is
a genuinely different state machine and stays in ``_core`` -- the SVF
unification is the warranted one, not a forced matrix+SVF merge.)
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..geometry import (
    compose_velocity,
    identity_grid,
    integrate_velocity_field,
    spatial_transform,
    upsample,
)
from ..geometry._interpolate import BoundaryMode
from ..smoothing import gaussian
from ._force import Force

__all__ = [
    'svf_coarse_to_fine',
    'single_sided_level',
    'symmetric_level',
]


def _smooth_vector(
    field: Array, sigma: Union[float, Sequence[float]], ndim: int
) -> Array:
    """Separable Gaussian over the spatial axes of a channel-last field.

    ``sigma`` is a scalar (isotropic) or a length-``ndim`` per-axis
    sequence (anisotropic regularisation).
    """
    moved = jnp.moveaxis(field, -1, 0)
    smoothed = gaussian(moved, sigma=sigma, spatial_rank=ndim)
    return jnp.moveaxis(smoothed, 0, -1)


def _spacing_tuple(
    spacing: Union[float, Sequence[float]], ndim: int
) -> tuple[float, ...]:
    if isinstance(spacing, (int, float)):
        return (float(spacing),) * ndim
    out = tuple(float(s) for s in spacing)
    if len(out) != ndim:
        raise ValueError(
            f'spacing must be a scalar or a length-{ndim} sequence; '
            f'got {spacing!r}.'
        )
    return out


def _relative_spacing(
    spacing: Optional[Union[float, Sequence[float]]], ndim: int
) -> Optional[tuple[float, ...]]:
    """Anisotropy-only spacing ``spacing / geomean(spacing)``.

    Level-independent (the coarse-to-fine align-corners scale cancels in
    the ratio) and ``1`` for isotropic spacing, so the regularisation /
    force only see the per-axis *anisotropy*, not an absolute scale --
    isotropic data is unchanged.
    """
    if spacing is None:
        return None
    sp = _spacing_tuple(spacing, ndim)
    geomean = math.prod(sp) ** (1.0 / ndim)
    rel = tuple(s / geomean for s in sp)
    if all(r == 1.0 for r in rel):
        return None
    return rel


# Per-level update: ``(moving_level, fixed_level, state) -> (state, cost)``
# where ``state`` is a tuple of ``n_fields`` velocity fields.
LevelSolve = Callable[
    [Array, Array, tuple[Array, ...]], tuple[tuple[Array, ...], Array]
]


def svf_coarse_to_fine(
    pyr_m: tuple[Array, ...],
    pyr_f: tuple[Array, ...],
    *,
    ndim: int,
    dtype: jnp.dtype,
    n_fields: int,
    level_solve: LevelSolve,
) -> tuple[tuple[Array, ...], Array]:
    """Coarse-to-fine driver for the SVF recipes.

    Carries a tuple of ``n_fields`` voxel-unit velocity fields from the
    coarsest pyramid level to the finest, prolonging between levels
    (interpolating upsample + the per-axis voxel-scale that keeps the field
    in voxel units), and delegating the per-level update to ``level_solve``.

    Parameters
    ----------
    pyr_m, pyr_f
        Channel-last Gaussian pyramids (finest first) of the moving / fixed
        images.
    ndim, dtype
        Spatial rank and dtype of the velocity fields.
    n_fields
        Number of velocity fields carried (1 for Demons, 2 for symmetric
        SyN).
    level_solve
        ``(moving_level, fixed_level, state) -> (state, cost_trace)`` for
        one resolution; ``state`` is the ``n_fields``-tuple of velocities.

    Returns
    -------
    ``(state, cost_history)`` -- the finest-level velocity tuple and the
    concatenated per-level cost traces.
    """
    state: Optional[tuple[Array, ...]] = None
    prev_shape: Optional[tuple[int, ...]] = None
    histories = []
    for level in range(len(pyr_m) - 1, -1, -1):
        m_l = pyr_m[level][..., 0]
        f_l = pyr_f[level][..., 0]
        shape_l = f_l.shape
        if state is None:
            state = tuple(
                jnp.zeros(shape_l + (ndim,), dtype=dtype)
                for _ in range(n_fields)
            )
        else:
            ratio = jnp.asarray(shape_l, dtype=dtype) / jnp.asarray(
                prev_shape, dtype=dtype
            )
            state = tuple(upsample(s, shape_l) * ratio for s in state)
        state, hist = level_solve(m_l, f_l, state)
        histories.append(hist)
        prev_shape = shape_l

    assert state is not None
    return state, jnp.concatenate(histories)


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

    Note an LNCC force does **not** vanish at a perfect match -- the metric's
    ``eps`` guard leaves ``cc < 1`` in low-variance windows -- so it is the
    symmetric forward/inverse cancellation, not a vanishing force or this
    clamp, that zeroes the *net* deformation there.
    """
    norm = jnp.sqrt(jnp.sum(u * u, axis=-1))
    scale = jnp.minimum(1.0, step / (jnp.max(norm) + 1e-12))
    return u * scale


def _per_axis_sigma(
    sigma: float, rel_spacing: Optional[tuple[float, ...]]
) -> Union[float, tuple[float, ...]]:
    """Anisotropy-correct a regularisation sigma (per-axis when anisotropic)."""
    if rel_spacing is None:
        return sigma
    return tuple(sigma / r for r in rel_spacing)


def _regularise(
    v: Array,
    u: Array,
    *,
    step: Optional[float],
    sigma_fluid: Union[float, tuple[float, ...]],
    sigma_diffusion: Union[float, tuple[float, ...]],
    bch_order: int,
    ndim: int,
) -> Array:
    """Apply the shared per-update regularisation to a raw force ``u``.

    The fluid+diffusion operator splitting both SVF drivers share: an optional
    trust-region step clamp (``step is not None``; the SyN convention), fluid
    Gaussian smoothing of the update, the log-domain accumulation
    (``v + u`` additive, or BCH ``compose_velocity`` for ``bch_order > 1``),
    and diffusion Gaussian smoothing of the accumulated velocity.
    """
    if step is not None:
        u = _normalise_step(u, step)
    u = _smooth_vector(u, sigma_fluid, ndim)
    v = v + u if bch_order == 1 else compose_velocity(v, u, order=bch_order)
    return _smooth_vector(v, sigma_diffusion, ndim)


def single_sided_level(
    moving: Array,
    fixed: Array,
    v: Array,
    *,
    force: Force,
    ndim: int,
    iterations: int,
    n_steps: int,
    boundary_mode: BoundaryMode,
    sigma_fluid: float,
    sigma_diffusion: float,
    bch_order: int,
    step: Optional[float],
    rel_spacing: Optional[tuple[float, ...]],
) -> tuple[Array, Array]:
    """Single-sided SVF iterations on one resolution (the Demons structure).

    Warps ``moving`` by ``exp(v)`` and drives ``v`` up the similarity under
    ``force`` -- metric-generic: any :class:`Force` plugs in.  The force is
    bound to ``fixed`` **once** (its fixed-state, e.g. ``∇fixed``, is hoisted
    out of the iteration).  Rolled with ``lax.scan``; returns ``(v, costs)``.
    """
    id_grid = identity_grid(fixed.shape, dtype=fixed.dtype)
    bound = force.bind(fixed, ndim=ndim, rel_spacing=rel_spacing)
    sf = _per_axis_sigma(sigma_fluid, rel_spacing)
    sd = _per_axis_sigma(sigma_diffusion, rel_spacing)

    def step_fn(v: Array, _: object) -> tuple[Array, Array]:
        s = integrate_velocity_field(v, n_steps=n_steps, mode=boundary_mode)
        warped = spatial_transform(
            moving[..., None], id_grid + s, mode=boundary_mode
        )[..., 0]
        v = _regularise(
            v,
            bound.update(warped),
            step=step,
            sigma_fluid=sf,
            sigma_diffusion=sd,
            bch_order=bch_order,
            ndim=ndim,
        )
        return v, bound.cost(warped)

    return jax.lax.scan(step_fn, v, xs=None, length=iterations)


def symmetric_level(
    moving: Array,
    fixed: Array,
    v_fwd: Array,
    v_inv: Array,
    *,
    force: Force,
    ndim: int,
    iterations: int,
    n_steps: int,
    boundary_mode: BoundaryMode,
    sigma_fluid: float,
    sigma_diffusion: float,
    step: Optional[float],
    rel_spacing: Optional[tuple[float, ...]],
) -> tuple[Array, Array, Array]:
    """Symmetric-midpoint SVF iterations on one resolution (the SyN structure).

    Warps both images to the shared midpoint and ascends the similarity under
    ``force`` in each direction -- metric-generic.  The force is bound **per
    step** (its "fixed" is the other image at the midpoint, which changes every
    iteration).  Rolled with ``lax.scan``; returns ``(v_fwd, v_inv, costs)``.
    """
    id_grid = identity_grid(fixed.shape, dtype=fixed.dtype)
    sf = _per_axis_sigma(sigma_fluid, rel_spacing)
    sd = _per_axis_sigma(sigma_diffusion, rel_spacing)

    def warp_to_mid(image: Array, v: Array) -> Array:
        s = integrate_velocity_field(v, n_steps=n_steps, mode=boundary_mode)
        return spatial_transform(
            image[..., None], id_grid + s, mode=boundary_mode
        )[..., 0]

    def step_fn(
        carry: tuple[Array, Array], _: object
    ) -> tuple[tuple[Array, Array], Array]:
        v_fwd, v_inv = carry
        a = warp_to_mid(moving, v_fwd)
        b = warp_to_mid(fixed, v_inv)
        bound_fwd = force.bind(b, ndim=ndim, rel_spacing=rel_spacing)
        bound_inv = force.bind(a, ndim=ndim, rel_spacing=rel_spacing)
        v_fwd = _regularise(
            v_fwd,
            bound_fwd.update(a),
            step=step,
            sigma_fluid=sf,
            sigma_diffusion=sd,
            bch_order=1,
            ndim=ndim,
        )
        v_inv = _regularise(
            v_inv,
            bound_inv.update(b),
            step=step,
            sigma_fluid=sf,
            sigma_diffusion=sd,
            bch_order=1,
            ndim=ndim,
        )
        return (v_fwd, v_inv), bound_fwd.cost(a)

    (v_fwd, v_inv), costs = jax.lax.scan(
        step_fn, (v_fwd, v_inv), xs=None, length=iterations
    )
    return v_fwd, v_inv, costs
