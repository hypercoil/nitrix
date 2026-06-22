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
from dataclasses import replace
from typing import Callable, Literal, Optional, Sequence, Union, cast

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from .._internal.backend import default_backend_is_gpu
from ..geometry import (
    affine_grid,
    compose_displacement,
    compose_velocity,
    identity_grid,
    integrate_velocity_field,
    jacobian_det_displacement,
    spatial_transform,
    upsample,
)
from ..geometry._interpolate import BoundaryMode
from ..smoothing import gaussian
from ._converge import Convergence, run_iterations
from ._force import Force, MIForce

__all__ = [
    'svf_coarse_to_fine',
    'single_sided_level',
    'symmetric_level',
    'group_single_sided_level',
    'group_symmetric_level',
    'resolve_init_displacement',
    'prewarp_moving',
    'finalize_with_init',
    'pin_force_ranges',
    'resolve_smoothing',
    'smooth_pyramid',
]


def resolve_smoothing(
    smoothing_sigma: Optional[Union[float, Sequence[float]]], levels: int
) -> Optional[tuple[float, ...]]:
    """Per-level smoothing sigmas in **finest-first** pyramid order (or None).

    A scalar -> the same sigma at every level; a length-``levels``
    **coarse-to-fine** sequence (the ANTs ``-s`` order, e.g. ``2x1x0``) ->
    reversed to the finest-first pyramid indexing.  ``None`` -> no extra
    smoothing.
    """
    if smoothing_sigma is None:
        return None
    if isinstance(smoothing_sigma, (int, float)):
        return (float(smoothing_sigma),) * levels
    seq = [float(s) for s in smoothing_sigma]
    if len(seq) != levels:
        raise ValueError(
            f'smoothing_sigma must be a scalar or a length-{levels} '
            f'(coarse-to-fine) sequence; got {len(seq)}.'
        )
    return tuple(reversed(seq))


def smooth_pyramid(
    pyr: tuple[Array, ...],
    sigmas: Optional[tuple[float, ...]],
    ndim: int,
) -> tuple[Array, ...]:
    """Independent per-level Gaussian smoothing of a (channel-last) pyramid (A4).

    Decouples the multi-resolution smoothing (ANTs ``-s``) from the shrink (the
    pyramid's anti-alias): ``sigmas`` (finest-first, from ``resolve_smoothing``)
    smooths each level on top of the shrink.  ``None`` / a ``0`` sigma leaves the
    level unchanged.
    """
    if sigmas is None:
        return pyr
    return tuple(
        lvl
        if s <= 0.0
        else gaussian(lvl[..., 0], sigma=s, spatial_rank=ndim)[..., None]
        for lvl, s in zip(pyr, sigmas)
    )


def _pin_range(x: Array) -> tuple[float, float]:
    """A stationary ``(lo, hi)`` from the full-res data, jit-safe.

    A ``stop_gradient``-ed ``jnp.min``/``jnp.max`` (not ``float()``): under
    ``jax.jit`` the images are tracers, so an eager ``float(tracer)`` cannot run
    -- a traced reduction can.  ``stop_gradient`` keeps the bin edges *constant*
    (the Mattes piecewise-constant-edge assumption; no gradient flows through
    the range derivation).  Eager value is identical to the old ``float`` path.
    """
    lo = lax.stop_gradient(jnp.min(x))
    hi = lax.stop_gradient(jnp.max(x))
    return cast('tuple[float, float]', (lo, hi))


def pin_force_ranges(force: Force, moving: Array, fixed: Array) -> Force:
    """Pin a histogram force's intensity ranges from the full-res images.

    A data ``min/max`` range drifts as the moving image deforms across the
    optimisation (a non-stationary objective) and truncates the force at the clip
    boundary, so each SVF recipe resolves any ``None`` range on an
    :class:`MIForce` **once**, before the pyramid, from the full-resolution
    images -- a ``stop_gradient``-ed reduction (``_pin_range``), so the range
    rides the frozen force as a *constant* (the piecewise-constant-edge Mattes
    gradient) and the pin is **jit-safe**: ``MIForce(bins=...)`` needs no
    explicit range even under ``jax.jit`` (the eager value is unchanged).  A
    no-op for any other force (and for an ``MIForce`` already pinned).
    """
    if isinstance(force, MIForce) and (
        force.range_moving is None or force.range_fixed is None
    ):
        rm = force.range_moving or _pin_range(moving)
        rf = force.range_fixed or _pin_range(fixed)
        return replace(force, range_moving=rm, range_fixed=rf)
    return force


def _smooth_method(
    sigma: Union[float, Sequence[float]],
) -> Literal['fir', 'recursive']:
    """Backend-aware Gaussian engine for the vector-field regulariser.

    The fluid/diffusion smooths run on the multi-channel velocity/displacement
    field on *every* iteration (2x Demons, 4x SyN), so the engine matters.  On
    GPU the shifted-slice FIR path is far cheaper than the sequential recursive
    scan (~0.3 ms vs ~12 ms at the registration sigmas, measured) and stays the
    engine -- byte-identical to the prior behaviour.  On CPU the FIR shift-sum
    is the dominant per-iteration cost and the O(N) Young-van Vliet recursion is
    ~1.1-1.5x cheaper, so ``'recursive'`` is selected -- but only when every
    per-axis sigma clears the YvV validity floor (>= 0.5), else FIR.  The CPU
    recursion differs from the FIR truncation by ~1-2% within a few sigma of the
    edge (a *regulariser*, not the objective), so CPU results diverge slightly
    from GPU; the recovery is unaffected.  Mirrors the ``signal._iir`` auto split
    (parallel on GPU, sequential recursion on CPU).
    """
    if default_backend_is_gpu():
        return 'fir'
    sigmas = (sigma,) if isinstance(sigma, (int, float)) else tuple(sigma)
    return 'recursive' if all(s >= 0.5 for s in sigmas) else 'fir'


def _smooth_vector(
    field: Array, sigma: Union[float, Sequence[float]], ndim: int
) -> Array:
    """Separable Gaussian over the spatial axes of a channel-last field.

    ``sigma`` is a scalar (isotropic) or a length-``ndim`` per-axis
    sequence (anisotropic regularisation).  The engine is backend-aware
    (``_smooth_method``): FIR on GPU, recursive on CPU.
    """
    moved = jnp.moveaxis(field, -1, 0)
    smoothed = gaussian(
        moved, sigma=sigma, spatial_rank=ndim, method=_smooth_method(sigma)
    )
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


# Per-level update: ``(level, moving_level, fixed_level, state) -> (state,
# cost)`` where ``level`` is the pyramid index (finest = 0; lets a recipe pick
# a per-level force / metric) and ``state`` is a tuple of ``n_fields`` velocity
# fields.
LevelSolve = Callable[
    [int, Array, Array, tuple[Array, ...]], tuple[tuple[Array, ...], Array]
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
        ``(level, moving_level, fixed_level, state) -> (state, cost_trace)``
        for one resolution (``level`` = pyramid index, finest = 0); ``state``
        is the ``n_fields``-tuple of velocities.

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
        state, hist = level_solve(level, m_l, f_l, state)
        histories.append(hist)
        prev_shape = shape_l

    assert state is not None
    return state, jnp.concatenate(histories)


# Robust-max percentile for the trust-region clamp: the cap is this percentile
# of the per-voxel displacement, so the top ~1% (outlier edge / hot voxels) do
# not single-handedly throttle the step (B4).  On the smallest coarse grids
# (~16^2) this is still the ~2nd-3rd largest -- robust to a lone outlier.
_STEP_ROBUST_PCTL = 99.0


def _normalise_step(u: Array, step: float, *, scale_to: bool = False) -> Array:
    """Cap (or scale to) the force field's (robust) largest voxel displacement.

    ``scale_to=False`` (default) is the trust-region **clamp**; ``scale_to=
    True`` is the ANTs **scale-to** (``step/cap``, magnitude-invariant) -- the
    branch the centre-only LNCC force needs (see the body).

    **The clamp (``scale_to=False``).**
    A trust-region clamp (``min(1, step/‖u‖_cap)``), not a scale-to.  The
    two coincide when the force exceeds the cap (both divide by ``‖u‖_cap``);
    they differ only *below* it, where the clamp keeps the raw gradient
    magnitude rather than amplifying it to a full step.  Under a fixed
    iteration budget (no convergence gate) that is the safer discipline: a
    shrinking force keeps shrinking, and a small / low-signal update is not
    inflated -- scale-to-step would force a full ``step`` in whatever
    direction the (often spurious, flat-region) maximum points.

    **Robust cap (B4).**  ``‖u‖_cap`` is the ``_STEP_ROBUST_PCTL``-th
    *percentile* of the per-voxel displacement, not the global ``max``: a single
    outlier voxel (edge / boundary / hot voxel) would otherwise set the cap for
    the whole field and starve the real signal.  A high *percentile* (not the
    RMS / mean) is the right robustification for a *clamp* -- it preserves the
    trust-region intent (bound essentially every voxel's step, keeping the warp
    diffeomorphic), ignoring only the top ~1% tail, whereas an RMS cap would let
    a large fraction of voxels exceed ``step`` (folding risk).  (Cost: one
    per-iteration sort; ``lax.top_k`` is the O(``N log k``) alternative if a
    brain-scale profile flags it.)

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
    cap = jnp.percentile(norm, _STEP_ROBUST_PCTL)
    if scale_to:
        # ANTs ``ScaleUpdateField``: scale the *whole* field so the robust-max
        # displacement is **exactly** ``step`` (``scale = step / max``), not the
        # ``min(1, ·)`` clamp.  This makes the step **magnitude-invariant** -- it
        # no longer matters that one force (e.g. the centre-only LNCC) is ~100x
        # smaller than another (the exact lncc_grad); both take a ``step``-sized
        # move.  The clamp instead *under-steps* a small-magnitude force (it
        # never amplifies), which starves the centre force.  Robust ``cap`` (a
        # high percentile, not the true max) keeps ITK's intent without letting
        # one hot voxel shrink the whole step.  Safe to scale-up only because a
        # convergence gate (the SVF early-exit) bounds the constant-step
        # dithering this would otherwise invite.
        safe = cap > 1e-12
        scale = jnp.where(safe, step / jnp.where(safe, cap, 1.0), 0.0)
    else:
        scale = jnp.minimum(1.0, step / (cap + 1e-12))
    return u * scale


def _per_axis_sigma(
    sigma: float, rel_spacing: Optional[tuple[float, ...]]
) -> Union[float, tuple[float, ...]]:
    """Anisotropy-correct a regularisation sigma (per-axis when anisotropic)."""
    if rel_spacing is None:
        return sigma
    return tuple(sigma / r for r in rel_spacing)


def _mask_force(u: Array, mask: Optional[Array]) -> Array:
    """Gate a force field by a (channel-less) mask; ``None`` -> unchanged."""
    return u if mask is None else u * mask[..., None]


def _restrict_force(u: Array, restrict: Optional[tuple[float, ...]]) -> Array:
    """Weight the force per spatial axis (ANTs ``--restrict-deformation``).

    ``restrict`` is a length-``ndim`` per-axis weight on the force's vector
    components, so a ``0`` suppresses deformation along that axis -- e.g.
    ``(1, 1, 0)`` for in-plane-only registration of 2-D-acquired data.  The
    suppression *persists*: the zeroed component never enters the additive
    log-update, and the spatial-only (per-channel) Gaussian smoothing cannot
    reintroduce it.  ``None`` -> unchanged.
    """
    if restrict is None:
        return u
    return u * jnp.asarray(restrict, dtype=u.dtype)


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


def _step_clamp_diffeo(
    delta: Array, *, det_floor: float = 0.1, max_halvings: int = 3
) -> Array:
    """Per-step diffeomorphism guard for the group (compositive) update.

    Halve the increment ``δ`` until ``det(I + ∇δ) > det_floor`` everywhere -- a
    clamp on the *Jacobian* (a magnitude clamp can still fold).  ``max_halvings``
    is a small static bound (the halving rarely fires under a sane
    ``_normalise_step`` + fluid smoothing); the ``jnp.where`` makes it jit-safe
    (a satisfied step is a no-op, so the remaining iterations do nothing).
    Mirrors the 0b inverse-compositional backtracking discipline.  The *total*-
    field ``det > 0`` QA is necessary but not sufficient -- an intermediate
    compose can fold while the total stays positive -- so the guard is per step.
    """
    for _ in range(max_halvings):
        det = jacobian_det_displacement(delta)
        delta = jnp.where(jnp.min(det) <= det_floor, delta * 0.5, delta)
    return delta


def _group_regularise(
    u: Array,
    *,
    step: Optional[float],
    sigma_fluid: Union[float, tuple[float, ...]],
    ndim: int,
    step_mode: Literal['clamp', 'normalize'] = 'clamp',
) -> Array:
    """Per-update regularisation for the group (greedy) driver.

    Produces the increment ``δ`` the level fn composes onto the total
    displacement: the step normalisation (``_normalise_step``), fluid (update)
    Gaussian smoothing, and -- in ``'clamp'`` mode -- the per-step
    diffeomorphism guard (:func:`_step_clamp_diffeo`).  The *total*-field
    (diffusion) smoothing is applied to ``s`` **after** the composition, in the
    level fn -- the same fluid/diffusion split the algebra :func:`_regularise`
    keeps.

    ``step_mode`` (L3): ``'clamp'`` (default) is the trust-region clamp + the
    Jacobian-backtracking guard.  ``'normalize'`` is the ANTs recipe -- a
    magnitude-invariant **scale-to** step (so a small-magnitude force such as
    the centre-only LNCC is not under-stepped) and **no** Jacobian backtracking
    (a bounded ``step``-sized smoothed increment is diffeomorphic by
    construction, so the per-step det guard is dropped -- the ANTs choice).
    """
    if step is None:
        delta = u
    else:
        delta = _normalise_step(u, step, scale_to=(step_mode == 'normalize'))
    delta = _smooth_vector(delta, sigma_fluid, ndim)
    if step_mode == 'normalize':
        return delta
    return _step_clamp_diffeo(delta)


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
    mask: Optional[Array] = None,
    restrict: Optional[tuple[float, ...]] = None,
    convergence: Optional[Convergence] = None,
) -> tuple[Array, Array]:
    """Single-sided SVF iterations on one resolution (the Demons structure).

    Warps ``moving`` by ``exp(v)`` and drives ``v`` up the similarity under
    ``force`` -- metric-generic: any :class:`Force` plugs in.  The force is
    bound to ``fixed`` **once** (its fixed-state, e.g. ``∇fixed``, is hoisted
    out of the iteration).  ``mask`` (this level's, channel-less) gates the
    force to a region -- the masked area drives the deformation, the rest
    follows by regularisation.  ``restrict`` (length-``ndim``) weights the force
    per axis (deformation-axis masking).  Rolled with ``lax.scan``; returns
    ``(v, costs)``.
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
            _restrict_force(_mask_force(bound.update(warped), mask), restrict),
            step=step,
            sigma_fluid=sf,
            sigma_diffusion=sd,
            bch_order=bch_order,
            ndim=ndim,
        )
        return v, bound.cost(warped)

    return run_iterations(
        step_fn,
        v,
        iterations=iterations,
        convergence=convergence,
        dtype=fixed.dtype,
    )


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
    mask: Optional[Array] = None,
    restrict: Optional[tuple[float, ...]] = None,
    convergence: Optional[Convergence] = None,
) -> tuple[Array, Array, Array]:
    """Symmetric-midpoint SVF iterations on one resolution (the SyN structure).

    Warps both images to the shared midpoint and ascends the similarity under
    ``force`` in each direction -- metric-generic.  The force is bound **per
    step** (its "fixed" is the other image at the midpoint, which changes every
    iteration).  ``mask`` (this level's) gates both half-forces to a region;
    ``restrict`` (length-``ndim``) weights both per axis (deformation-axis
    masking).  Rolled with ``lax.scan``; returns ``(v_fwd, v_inv, costs)``.
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
            _restrict_force(_mask_force(bound_fwd.update(a), mask), restrict),
            step=step,
            sigma_fluid=sf,
            sigma_diffusion=sd,
            bch_order=1,
            ndim=ndim,
        )
        v_inv = _regularise(
            v_inv,
            _restrict_force(_mask_force(bound_inv.update(b), mask), restrict),
            step=step,
            sigma_fluid=sf,
            sigma_diffusion=sd,
            bch_order=1,
            ndim=ndim,
        )
        return (v_fwd, v_inv), bound_fwd.cost(a)

    (v_fwd, v_inv), costs = run_iterations(
        step_fn,
        (v_fwd, v_inv),
        iterations=iterations,
        convergence=convergence,
        dtype=fixed.dtype,
    )
    return v_fwd, v_inv, costs


def group_single_sided_level(
    moving: Array,
    fixed: Array,
    s: Array,
    *,
    force: Force,
    ndim: int,
    iterations: int,
    boundary_mode: BoundaryMode,
    sigma_fluid: float,
    sigma_diffusion: float,
    step: Optional[float],
    rel_spacing: Optional[tuple[float, ...]],
    mask: Optional[Array] = None,
    restrict: Optional[tuple[float, ...]] = None,
    convergence: Optional[Convergence] = None,
    step_mode: Literal['clamp', 'normalize'] = 'clamp',
) -> tuple[Array, Array]:
    """Single-sided **group (greedy)** iterations on one resolution.

    The group-domain sibling of :func:`single_sided_level`: the state is the
    *displacement* ``s`` (``φ = id + s``), warped **directly** (one gather, no
    ``exp``), and the regularised increment is **composed** onto ``s`` (the
    compositive demons update ``φ ← φ ∘ (id+δ)``) rather than added in the log
    domain.  ~2 gathers/iter (warp + compose) vs the algebra driver's ~7.  No
    per-iteration ``exp``; the velocity is recovered once at finalisation via
    ``field_log`` (the recipe).  Dropping ``n_steps`` / ``bch_order`` (no
    integration, no log-domain BCH) marks it a different driver, not a
    re-parametrised one.
    """
    id_grid = identity_grid(fixed.shape, dtype=fixed.dtype)
    bound = force.bind(fixed, ndim=ndim, rel_spacing=rel_spacing)
    sf = _per_axis_sigma(sigma_fluid, rel_spacing)
    sd = _per_axis_sigma(sigma_diffusion, rel_spacing)

    def step_fn(s: Array, _: object) -> tuple[Array, Array]:
        warped = spatial_transform(
            moving[..., None], id_grid + s, mode=boundary_mode
        )[..., 0]
        u = _restrict_force(_mask_force(bound.update(warped), mask), restrict)
        delta = _group_regularise(
            u, step=step, sigma_fluid=sf, ndim=ndim, step_mode=step_mode
        )
        s = compose_displacement(s, delta, mode=boundary_mode)
        s = _smooth_vector(s, sd, ndim)  # total-field (diffusion) regulariser
        return s, bound.cost(warped)

    return run_iterations(
        step_fn,
        s,
        iterations=iterations,
        convergence=convergence,
        dtype=fixed.dtype,
    )


def group_symmetric_level(
    moving: Array,
    fixed: Array,
    s_fwd: Array,
    s_inv: Array,
    *,
    force: Force,
    ndim: int,
    iterations: int,
    boundary_mode: BoundaryMode,
    sigma_fluid: float,
    sigma_diffusion: float,
    step: Optional[float],
    rel_spacing: Optional[tuple[float, ...]],
    mask: Optional[Array] = None,
    restrict: Optional[tuple[float, ...]] = None,
    convergence: Optional[Convergence] = None,
    step_mode: Literal['clamp', 'normalize'] = 'clamp',
) -> tuple[Array, Array, Array]:
    """Symmetric-midpoint **group (greedy)** iterations on one resolution.

    The group-domain sibling of :func:`symmetric_level`: carry the two
    *displacements* ``s_fwd`` / ``s_inv``, warp both images to the midpoint by
    ``id + s`` directly, and **compose** each regularised half-increment onto its
    displacement.  ~4 gathers/iter (2 warps + 2 composes) vs the algebra
    driver's ~12.  **No per-iteration inversion** -- inverse-consistency is
    realised once at finalisation (``compose(s_fwd, invert(s_inv))``), exactly as
    the algebra SyN, not per step.
    """
    id_grid = identity_grid(fixed.shape, dtype=fixed.dtype)
    sf = _per_axis_sigma(sigma_fluid, rel_spacing)
    sd = _per_axis_sigma(sigma_diffusion, rel_spacing)

    def warp_to_mid(image: Array, s: Array) -> Array:
        return spatial_transform(
            image[..., None], id_grid + s, mode=boundary_mode
        )[..., 0]

    def step_fn(
        carry: tuple[Array, Array], _: object
    ) -> tuple[tuple[Array, Array], Array]:
        s_fwd, s_inv = carry
        a = warp_to_mid(moving, s_fwd)
        b = warp_to_mid(fixed, s_inv)
        bound_fwd = force.bind(b, ndim=ndim, rel_spacing=rel_spacing)
        bound_inv = force.bind(a, ndim=ndim, rel_spacing=rel_spacing)
        d_fwd = _group_regularise(
            _restrict_force(_mask_force(bound_fwd.update(a), mask), restrict),
            step=step,
            sigma_fluid=sf,
            ndim=ndim,
            step_mode=step_mode,
        )
        d_inv = _group_regularise(
            _restrict_force(_mask_force(bound_inv.update(b), mask), restrict),
            step=step,
            sigma_fluid=sf,
            ndim=ndim,
            step_mode=step_mode,
        )
        s_fwd = _smooth_vector(
            compose_displacement(s_fwd, d_fwd, mode=boundary_mode), sd, ndim
        )
        s_inv = _smooth_vector(
            compose_displacement(s_inv, d_inv, mode=boundary_mode), sd, ndim
        )
        return (s_fwd, s_inv), bound_fwd.cost(a)

    (s_fwd, s_inv), costs = run_iterations(
        step_fn,
        (s_fwd, s_inv),
        iterations=iterations,
        convergence=convergence,
        dtype=fixed.dtype,
    )
    return s_fwd, s_inv, costs


def resolve_init_displacement(
    init_affine: Optional[Array],
    init_displacement: Optional[Array],
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> Optional[Array]:
    """Resolve a recipe's init arguments to a fixed-grid displacement (or None).

    At most one of ``init_affine`` (a fixed-voxel -> moving-voxel homogeneous
    matrix, as ``rigid_register`` / ``affine_register`` return in ``IndexSpace``)
    or ``init_displacement`` (a displacement field on the fixed grid, e.g. a
    SynthMorph network output) may be given; the diffeomorphic recipe pre-warps
    ``moving`` by the result and registers the residual (warm-start /
    multi-stage).  An affine is expanded to its displacement field by applying
    the **self-contained** recipe matrix about the origin (its grid centre is
    already baked in -- B1).
    """
    if init_affine is not None and init_displacement is not None:
        raise ValueError(
            'pass at most one of init_affine / init_displacement.'
        )
    if init_affine is not None:
        # The recipes return a SELF-CONTAINED matrix (the grid centre baked in,
        # B1), so apply it about the origin -- re-centring it here (the old
        # affine_grid default) would double-centre and silently defeat the
        # warm-start.
        center = jnp.zeros(len(shape), dtype=dtype)
        return affine_grid(init_affine, shape, center=center) - identity_grid(
            shape, dtype=dtype
        )
    return init_displacement


def prewarp_moving(
    moving: Array,
    init_disp: Optional[Array],
    shape: tuple[int, ...],
    dtype: jnp.dtype,
    boundary_mode: BoundaryMode,
) -> Array:
    """Resample ``moving`` onto the fixed grid by the init displacement.

    ``None`` -> ``moving`` unchanged (the no-init path).  Otherwise ``moving`` is
    sampled at ``id + init_disp`` so it lands on the fixed grid even when its own
    grid differs -- the residual deformable then registers this pre-warped image
    (this is what retires the matching-grid constraint for a multi-stage run).
    """
    if init_disp is None:
        return moving
    id_grid = identity_grid(shape, dtype=dtype)
    return spatial_transform(
        moving[..., None], id_grid + init_disp, mode=boundary_mode
    )[..., 0]


def finalize_with_init(
    moving: Array,
    residual_disp: Array,
    init_disp: Optional[Array],
    *,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
    boundary_mode: BoundaryMode,
) -> tuple[Array, Array, Array]:
    """Compose the residual deformable with the init; warp the original moving.

    Returns ``(total_displacement, warped, jacobian_det)`` -- the full
    ``moving -> fixed`` map (init applied, then the residual), the *original*
    ``moving`` resampled by it, and the Jacobian determinant of the **total**
    map (``init_disp is None`` reduces exactly to the residual).
    """
    id_grid = identity_grid(shape, dtype=dtype)
    total = (
        residual_disp
        if init_disp is None
        else compose_displacement(init_disp, residual_disp, mode=boundary_mode)
    )
    warped = spatial_transform(
        moving[..., None], id_grid + total, mode=boundary_mode
    )[..., 0]
    return total, warped, jacobian_det_displacement(total)
