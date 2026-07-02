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
(:func:`svf_coarse_to_fine`) -- so each recipe is just its per-level update
plus its finalisation, not a re-derived multiresolution loop.

(The matrix-transform driver, by contrast, carries a small parameter
*vector* and a coordinate-space sampler, not a velocity *field*, so it is
a genuinely different state machine and stays in ``_core`` -- the SVF
unification is the warranted one, not a forced matrix+SVF merge.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Literal, Optional, Sequence, Union, cast

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from .._internal.backend import default_backend_is_gpu
from .._internal.config import resolve_driver
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
from ._converge import Convergence, ConvergenceMode, run_iterations
from ._force import Force, MIForce

__all__ = [
    'SVFSpec',
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


@dataclass(frozen=True)
class SVFSpec:
    """Shared stationary-velocity-field schedule embedded by both SVF recipe specs.

    The base configuration that :class:`DemonsSpec` and :class:`SyNSpec` both
    embed.  Holds the multi-resolution + regularisation + convergence fields the
    log-Demons and greedy-SyN recipes have in common, so a recipe spec adds only
    its own force/update knobs (Demons: ``alpha`` / ``bch_order``; SyN:
    ``radius`` / ``step`` / ``step_mode``) and the field's type, default, and
    docstring live in exactly one place.

    Attributes
    ----------
    levels
        Gaussian-pyramid resolutions (coarse-to-fine).
    iterations
        Iterations per level: an ``int`` (the same count at every level) or a
        length-``levels`` **coarse-to-fine** tuple (front-load the cheap coarse
        levels, cap the expensive finest one -- the ANTs schedule discipline,
        e.g. ``(100, 70, 50, 20)`` over a 4-level pyramid).
    sigma_fluid
        Gaussian sigma for the fluid (per-update) regularisation.
    sigma_diffusion
        Gaussian sigma for the diffusion (accumulated-field) regularisation.
    spacing
        Per-axis voxel spacing (physical units); ``float`` or length-``ndim``
        tuple.  ``None`` (default) registers in voxel units.  Only the
        **anisotropy** is used -- the regularisation (and the LNCC window /
        ESM force) are made physically isotropic by the *relative* spacing
        ``spacing / geomean(spacing)`` (level-independent), so isotropic spacing
        reduces exactly to ``None``; the velocity field stays voxel-native.
    pyramid_factor, pyramid_sigma
        Pyramid downsample factor / anti-alias sigma.
    boundary_mode
        Out-of-bounds handling for the warps.
    representation
        Dense-deformation domain: ``'group'`` (default) carries the
        *displacement(s)* and uses the greedy compositive update (warp directly,
        compose the increment -- the perf path); ``'algebra'`` carries the
        *stationary velocity(ies)* and re-exponentiates every iteration (the
        exact-SVF path, byte-identical to the pre-v4 recipe, the parity oracle).
        The velocity output is recovered from the final displacement(s) via
        :func:`nitrix.geometry.field_log` in group mode.  (The recipe sets how many
        displacements / velocities -- one for Demons, a symmetric pair for SyN.)
    mode
        Iteration strategy.  ``'fixed'`` (the SVF default) runs the full
        fixed schedule (a ``lax.scan``); ``'early_exit'`` runs the windowed-slope
        ``lax.while_loop`` -- a level stops once the normalised cost slope drops
        below ``convergence.threshold`` (or ``iterations`` is hit).
        **Single-pair** (a ``vmap``-ed ``while_loop`` runs to the all-lanes exit,
        the slowest pair governing).  A *tapered* per-level ``iterations``
        schedule already removes the over-iteration the early-exit targets, so
        the strict ``threshold=1e-6`` then *costs* time on the fast GPU path --
        hence ``'fixed'`` is the SVF default; early-exit pays off for an untuned
        / flat schedule or the CPU path with a looser threshold.
    convergence
        The :class:`Convergence` (threshold / window) for ``mode='early_exit'``;
        inert under ``mode='fixed'``.
    compute_velocity
        Whether to recover and return the stationary velocity output (the
        :func:`nitrix.geometry.field_log` of the final displacement(s)).
        ``False`` (default) leaves it ``None`` and skips the
        :func:`nitrix.geometry.field_log` finalisation entirely -- under ``jit``
        that loop nest is never traced, so it costs neither compile nor runtime;
        the deformation outputs (``warped`` / ``displacement`` / ``jacobian_det``)
        never depend on it.  Set ``True`` to recover it (e.g. to feed
        :func:`nitrix.geometry.velocity_mean` or the transform-algebra path).
    """

    levels: int = 3
    iterations: Union[int, tuple[int, ...]] = 80
    sigma_fluid: float = 1.0
    sigma_diffusion: float = 1.5
    spacing: Optional[Union[float, tuple[float, ...]]] = None
    pyramid_factor: float = 2.0
    pyramid_sigma: Optional[float] = None
    boundary_mode: BoundaryMode = 'nearest'
    representation: Literal['group', 'algebra'] = 'group'
    mode: ConvergenceMode = 'fixed'
    convergence: Convergence = Convergence()
    compute_velocity: bool = False


def resolve_smoothing(
    smoothing_sigma: Optional[Union[float, Sequence[float]]], levels: int
) -> Optional[tuple[float, ...]]:
    """Resolve per-level smoothing sigmas to finest-first pyramid order.

    A scalar -> the same sigma at every level; a length-``levels``
    **coarse-to-fine** sequence (the ANTs ``-s`` order, e.g. ``2x1x0``) ->
    reversed to the finest-first pyramid indexing.  ``None`` -> no extra
    smoothing.

    Parameters
    ----------
    smoothing_sigma
        Additional per-level Gaussian smoothing.  A scalar applies the same
        sigma at every level; a length-``levels`` sequence is given in
        coarse-to-fine order; ``None`` requests no extra smoothing.
    levels
        Number of pyramid levels.

    Returns
    -------
    tuple of float or None
        The per-level sigmas in finest-first pyramid order (length ``levels``),
        or ``None`` when no extra smoothing was requested.

    Raises
    ------
    ValueError
        If ``smoothing_sigma`` is a sequence whose length is not ``levels``.
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
    """Independent per-level Gaussian smoothing of a channel-last pyramid.

    Decouples the multi-resolution smoothing (ANTs ``-s``) from the shrink (the
    pyramid's anti-alias): ``sigmas`` (finest-first, from
    :func:`resolve_smoothing`) smooths each level on top of the shrink.  ``None``
    / a ``0`` sigma leaves the level unchanged.

    Parameters
    ----------
    pyr
        Gaussian pyramid as a tuple of channel-last arrays (finest first), each
        of shape ``(*spatial, 1)``.
    sigmas
        Per-level smoothing sigmas in finest-first order (as returned by
        :func:`resolve_smoothing`), or ``None`` for no extra smoothing.
    ndim
        Spatial rank of each pyramid level.

    Returns
    -------
    tuple of Array
        The pyramid with each level additionally Gaussian-smoothed by its sigma;
        levels with ``None`` or non-positive sigma are returned unchanged.
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

    Parameters
    ----------
    x
        Image array whose extent defines the range.

    Returns
    -------
    tuple of float
        The ``(lo, hi)`` minimum and maximum of ``x`` as gradient-stopped scalar
        arrays.
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
    images -- a ``stop_gradient``-ed reduction (:func:`_pin_range`), so the range
    rides the frozen force as a *constant* (the piecewise-constant-edge Mattes
    gradient) and the pin is **jit-safe**: ``MIForce(bins=...)`` needs no
    explicit range even under ``jax.jit`` (the eager value is unchanged).  A
    no-op for any other force (and for an :class:`MIForce` already pinned).

    Parameters
    ----------
    force
        The similarity force.  Only an unpinned :class:`MIForce` is modified;
        any other force (or an already-pinned one) is returned unchanged.
    moving
        Full-resolution moving image, used to derive the moving-side range.
    fixed
        Full-resolution fixed image, used to derive the fixed-side range.

    Returns
    -------
    Force
        The input force with any ``None`` intensity range filled in from the
        image extents; the same object for a non-:class:`MIForce` or an
        already-pinned :class:`MIForce`.
    """
    if isinstance(force, MIForce) and (
        force.range_moving is None or force.range_fixed is None
    ):
        rm = force.range_moving or _pin_range(moving)
        rf = force.range_fixed or _pin_range(fixed)
        return replace(force, range_moving=rm, range_fixed=rf)
    return force


def _smooth_fast(
    sigma: Union[float, Sequence[float]],
) -> Literal['fir', 'recursive']:
    """The hardware-/sigma-aware Gaussian engine pick (the ``driver`` fast path).

    On GPU the shifted-slice FIR path is far cheaper than the sequential
    recursive scan (~0.3 ms vs ~12 ms at the registration sigmas, measured), so
    FIR.  On CPU the FIR shift-sum dominates and the O(N) Young-van Vliet
    recursion is ~1.1-1.5x cheaper, so ``'recursive'`` -- but only when every
    per-axis sigma clears the YvV validity floor (>= 0.5), else FIR.

    Parameters
    ----------
    sigma
        Gaussian standard deviation: a scalar (isotropic) or a per-axis
        sequence.

    Returns
    -------
    {'fir', 'recursive'}
        ``'fir'`` on GPU, or on CPU when any per-axis sigma is below the
        Young-van Vliet validity floor (0.5); ``'recursive'`` on CPU otherwise.
    """
    if default_backend_is_gpu():
        return 'fir'
    sigmas = (sigma,) if isinstance(sigma, (int, float)) else tuple(sigma)
    return 'recursive' if all(s >= 0.5 for s in sigmas) else 'fir'


def _smooth_method(
    sigma: Union[float, Sequence[float]],
    driver: str = 'auto',
) -> Literal['fir', 'recursive']:
    """Resolve the Gaussian engine for the vector-field regulariser.

    The fluid/diffusion smooths run on the multi-channel velocity/displacement
    field on *every* iteration (2x Demons, 4x SyN), so the engine matters --
    hence the hardware-aware default (:func:`_smooth_fast`).  But FIR and the
    recursive Young-van Vliet path differ by ~1-2% within a few sigma of the
    edge, so an ``'auto'`` choice diverges CPU-vs-GPU.  This routes through the
    ``driver`` axis (the divergent op ``register.field_smooth``): ``'auto'`` ->
    the fast pick; ``nitrix.reproducible()`` forces the canonical ``'fir'`` on
    every platform (the regulariser is then bit-stable cross-platform up to the
    FIR/recursive tolerance); an explicit ``driver`` overrides.

    Parameters
    ----------
    sigma
        Gaussian standard deviation: a scalar (isotropic) or a per-axis
        sequence.  Passed through to :func:`_smooth_fast` for the fast pick.
    driver
        Driver selector on the ``register.field_smooth`` axis.  ``'auto'``
        (default) takes the hardware-aware fast pick; ``'fir'`` / ``'recursive'``
        force that engine; the reproducible mode forces canonical ``'fir'``.

    Returns
    -------
    {'fir', 'recursive'}
        The resolved Gaussian engine.
    """
    return cast(
        Literal['fir', 'recursive'],
        resolve_driver(
            driver,
            op='register.field_smooth',
            fast=lambda: _smooth_fast(sigma),
        ),
    )


def _smooth_vector(
    field: Array,
    sigma: Union[float, Sequence[float]],
    ndim: int,
    driver: str = 'auto',
) -> Array:
    """Separable Gaussian over the spatial axes of a channel-last field.

    ``sigma`` is a scalar (isotropic) or a length-``ndim`` per-axis
    sequence (anisotropic regularisation).  The engine is the ``driver`` axis
    (:func:`_smooth_method`): hardware-aware by default (FIR on GPU, recursive on
    CPU), forced to canonical FIR under ``nitrix.reproducible()``.

    Parameters
    ----------
    field
        Channel-last field of shape ``(*spatial, c)``; each channel is smoothed
        independently over the spatial axes.
    sigma
        Gaussian standard deviation: a scalar (isotropic) or a length-``ndim``
        per-axis sequence (anisotropic).
    ndim
        Spatial rank (number of leading spatial axes).
    driver
        Driver selector passed to :func:`_smooth_method`; ``'auto'`` by default.

    Returns
    -------
    Array
        The smoothed field, same shape as ``field``.
    """
    moved = jnp.moveaxis(field, -1, 0)
    smoothed = gaussian(
        moved,
        sigma=sigma,
        spatial_rank=ndim,
        driver=_smooth_method(sigma, driver),
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

    Parameters
    ----------
    spacing
        Per-axis voxel spacing (physical units): a scalar, a length-``ndim``
        sequence, or ``None`` (voxel-unit registration).
    ndim
        Spatial rank.

    Returns
    -------
    tuple of float or None
        The per-axis spacing divided by its geometric mean, or ``None`` when the
        spacing is absent or isotropic (all ratios equal to one).

    Raises
    ------
    ValueError
        If ``spacing`` is a sequence whose length is not ``ndim``.
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
    r"""Cap (or scale to) the force field's (robust) largest voxel displacement.

    ``scale_to=False`` (default) is the trust-region **clamp**; ``scale_to=
    True`` is the ANTs **scale-to** (``step/cap``, magnitude-invariant) -- the
    branch the centre-only LNCC force needs (see the body).

    **The clamp (``scale_to=False``).**
    A trust-region clamp (:math:`\min(1, step / \|u\|_{cap})`), not a scale-to.
    The two coincide when the force exceeds the cap (both divide by
    :math:`\|u\|_{cap}`);
    they differ only *below* it, where the clamp keeps the raw gradient
    magnitude rather than amplifying it to a full step.  Under a fixed
    iteration budget (no convergence gate) that is the safer discipline: a
    shrinking force keeps shrinking, and a small / low-signal update is not
    inflated -- scale-to-step would force a full ``step`` in whatever
    direction the (often spurious, flat-region) maximum points.

    **Robust cap.**  :math:`\|u\|_{cap}` is the ``_STEP_ROBUST_PCTL``-th
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
    forward is a fixed-length ``lax.scan`` (no convergence gate).  With a
    ``while_loop`` early-exit, a convergence gate would bound the constant-step
    dithering that motivates the clamp, making scale-to-step (the ANTs choice)
    viable again -- so revisit clamp-vs-scale here if that lands.

    Note an LNCC force does **not** vanish at a perfect match -- the metric's
    ``eps`` guard leaves :math:`cc < 1` in low-variance windows -- so it is the
    symmetric forward/inverse cancellation, not a vanishing force or this
    clamp, that zeroes the *net* deformation there.

    Parameters
    ----------
    u
        Force / displacement field, channel-last of shape ``(*spatial, ndim)``.
    step
        Trust-region step size: the target cap (clamp mode) or exact
        robust-max magnitude (scale-to mode).
    scale_to
        ``False`` (default) applies the trust-region clamp
        :math:`\min(1, step / \|u\|_{cap})`; ``True`` applies the ANTs
        magnitude-invariant scale-to (:math:`step / \|u\|_{cap}`) with a
        dtype-derived zero-cap guard.

    Returns
    -------
    Array
        The field scaled by the (per-field scalar) clamp or scale-to factor,
        same shape as ``u``.
    """
    norm = jnp.sqrt(jnp.sum(u * u, axis=-1))
    # Robust cap as the top-(100-pctl)% order statistic via lax.top_k
    # (O(N log k)) rather than a full O(N log N) percentile sort every
    # iteration: the k-th largest is the pctl-th percentile up to a sub-rank
    # difference -- immaterial for a trust-region cap.  ``k`` is static (the
    # field shape is).
    k = max(1, math.ceil((100.0 - _STEP_ROBUST_PCTL) / 100.0 * norm.size))
    cap = jnp.min(lax.top_k(norm.reshape(-1), k)[0])
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
        # Dtype-derived zero-cap guard (E1): finfo(dtype).eps, not a fixed
        # 1e-12 that float32 (~1.2e-7 precision) cannot represent meaningfully.
        eps = jnp.finfo(u.dtype).eps
        safe = cap > eps
        scale = jnp.where(safe, step / jnp.where(safe, cap, 1.0), 0.0)
    else:
        scale = jnp.minimum(1.0, step / (cap + jnp.finfo(u.dtype).eps))
    return u * scale


def _per_axis_sigma(
    sigma: float, rel_spacing: Optional[tuple[float, ...]]
) -> Union[float, tuple[float, ...]]:
    """Anisotropy-correct a regularisation sigma (per-axis when anisotropic).

    Parameters
    ----------
    sigma
        The isotropic regularisation sigma (in voxel units).
    rel_spacing
        Relative (anisotropy-only) per-axis spacing from
        :func:`_relative_spacing`, or ``None`` for isotropic data.

    Returns
    -------
    float or tuple of float
        ``sigma`` unchanged when ``rel_spacing`` is ``None``; otherwise the
        per-axis sigma ``sigma / rel_spacing`` (larger sigma along the
        finer-spaced axes).
    """
    if rel_spacing is None:
        return sigma
    return tuple(sigma / r for r in rel_spacing)


def _mask_force(u: Array, mask: Optional[Array]) -> Array:
    """Gate a force field by a (channel-less) mask; ``None`` -> unchanged.

    Parameters
    ----------
    u
        Force field, channel-last of shape ``(*spatial, ndim)``.
    mask
        Channel-less mask broadcast over the vector components (shape
        ``(*spatial,)``), or ``None`` to leave the force unchanged.

    Returns
    -------
    Array
        The force with the mask applied per voxel, same shape as ``u``.
    """
    return u if mask is None else u * mask[..., None]


def _restrict_force(u: Array, restrict: Optional[tuple[float, ...]]) -> Array:
    """Weight the force per spatial axis (ANTs ``--restrict-deformation``).

    ``restrict`` is a length-``ndim`` per-axis weight on the force's vector
    components, so a ``0`` suppresses deformation along that axis -- e.g.
    ``(1, 1, 0)`` for in-plane-only registration of 2-D-acquired data.  The
    suppression *persists*: the zeroed component never enters the additive
    log-update, and the spatial-only (per-channel) Gaussian smoothing cannot
    reintroduce it.  ``None`` -> unchanged.

    Parameters
    ----------
    u
        Force field, channel-last of shape ``(*spatial, ndim)``.
    restrict
        Length-``ndim`` per-axis weight on the force's vector components (``0``
        suppresses deformation along that axis), or ``None`` to leave the force
        unchanged.

    Returns
    -------
    Array
        The force with each vector component scaled by its axis weight, same
        shape as ``u``.
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
    (``v + u`` additive, or the Baker-Campbell-Hausdorff
    :func:`nitrix.geometry.compose_velocity` for ``bch_order > 1``), and
    diffusion Gaussian smoothing of the accumulated velocity.

    Parameters
    ----------
    v
        Accumulated velocity field, channel-last of shape ``(*spatial, ndim)``.
    u
        Raw per-update force field, same shape as ``v``.
    step
        Trust-region step cap passed to :func:`_normalise_step`, or ``None`` to
        skip the clamp.
    sigma_fluid
        Gaussian sigma for the fluid (per-update) smoothing of ``u``; scalar or
        per-axis.
    sigma_diffusion
        Gaussian sigma for the diffusion (accumulated-field) smoothing of ``v``;
        scalar or per-axis.
    bch_order
        Order of the log-domain accumulation: ``1`` is the additive update
        ``v + u``; higher orders use the Baker-Campbell-Hausdorff composition.
    ndim
        Spatial rank.

    Returns
    -------
    Array
        The updated, diffusion-smoothed velocity field, same shape as ``v``.
    """
    if step is not None:
        u = _normalise_step(u, step)
    u = _smooth_vector(u, sigma_fluid, ndim)
    v = v + u if bch_order == 1 else compose_velocity(v, u, order=bch_order)
    return _smooth_vector(v, sigma_diffusion, ndim)


def _step_clamp_diffeo(
    delta: Array, *, det_floor: float = 0.1, max_halvings: int = 3
) -> Array:
    r"""Per-step diffeomorphism guard for the group (compositive) update.

    Halve the increment :math:`\delta` until
    :math:`\det(I + \nabla\delta) > det\_floor` everywhere -- a clamp on the
    *Jacobian* (a magnitude clamp can still fold).  ``max_halvings`` is a small
    static bound (the halving rarely fires under a sane :func:`_normalise_step`
    + fluid smoothing); the ``jnp.where`` makes it jit-safe (a satisfied step is
    a no-op, so the remaining iterations do nothing).  Mirrors the
    inverse-compositional backtracking discipline.  The *total*-field
    :math:`\det > 0` check is necessary but not sufficient -- an intermediate
    compose can fold while the total stays positive -- so the guard is per step.

    Parameters
    ----------
    delta
        The compositive increment, channel-last of shape ``(*spatial, ndim)``.
    det_floor
        Lower bound on the per-voxel Jacobian determinant of ``id + delta``;
        the increment is halved while any voxel falls at or below it.
    max_halvings
        Static upper bound on the number of halvings attempted.

    Returns
    -------
    Array
        The (possibly halved) increment, same shape as ``delta``.
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
    r"""Per-update regularisation for the group (greedy) driver.

    Produces the increment :math:`\delta` the level fn composes onto the total
    displacement: the step normalisation (:func:`_normalise_step`), fluid
    (update) Gaussian smoothing, and -- in ``'clamp'`` mode -- the per-step
    diffeomorphism guard (:func:`_step_clamp_diffeo`).  The *total*-field
    (diffusion) smoothing is applied to ``s`` **after** the composition, in the
    level fn -- the same fluid/diffusion split the algebra :func:`_regularise`
    keeps.

    Parameters
    ----------
    u
        Raw per-update force field, channel-last of shape ``(*spatial, ndim)``.
    step
        Trust-region step size passed to :func:`_normalise_step`, or ``None`` to
        skip the step normalisation.
    sigma_fluid
        Gaussian sigma for the fluid (per-update) smoothing; scalar or per-axis.
    ndim
        Spatial rank.
    step_mode
        ``'clamp'`` (default) applies the trust-region clamp followed by the
        Jacobian-backtracking guard :func:`_step_clamp_diffeo`.  ``'normalize'``
        applies the ANTs magnitude-invariant scale-to step (so a small-magnitude
        force such as the centre-only LNCC is not under-stepped) and drops the
        per-step Jacobian guard, since a bounded ``step``-sized smoothed
        increment is diffeomorphic by construction.

    Returns
    -------
    Array
        The regularised compositive increment, same shape as ``u``.
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
    r"""Single-sided SVF iterations on one resolution (the Demons structure).

    Warps ``moving`` by :math:`\exp(v)` and drives ``v`` up the similarity under
    ``force`` -- metric-generic: any :class:`Force` plugs in.  The force is
    bound to ``fixed`` **once** (its fixed-state, e.g. :math:`\nabla fixed`, is
    hoisted out of the iteration).  ``mask`` (this level's, channel-less) gates
    the force to a region -- the masked area drives the deformation, the rest
    follows by regularisation.  ``restrict`` (length-``ndim``) weights the force
    per axis (deformation-axis masking).  Rolled with ``lax.scan``.

    Parameters
    ----------
    moving
        Moving image at this resolution, shape ``(*spatial,)``.
    fixed
        Fixed (target) image at this resolution, shape ``(*spatial,)``.
    v
        Initial stationary velocity field, channel-last of shape
        ``(*spatial, ndim)``.
    force
        The similarity :class:`Force` driving the update.
    ndim
        Spatial rank.
    iterations
        Number of iterations at this resolution.
    n_steps
        Number of scaling-and-squaring steps used to integrate ``v`` to a
        displacement each iteration.
    boundary_mode
        Out-of-bounds handling for the warp and integration.
    sigma_fluid
        Gaussian sigma for the fluid (per-update) regularisation.
    sigma_diffusion
        Gaussian sigma for the diffusion (accumulated-field) regularisation.
    bch_order
        Order of the log-domain accumulation (``1`` for the additive update,
        higher for the Baker-Campbell-Hausdorff composition).
    step
        Trust-region step cap, or ``None`` to skip the step clamp.
    rel_spacing
        Relative (anisotropy-only) per-axis spacing, or ``None`` for isotropic
        data.
    mask
        Channel-less region mask gating the force, or ``None``.
    restrict
        Length-``ndim`` per-axis force weight (deformation-axis masking), or
        ``None``.
    convergence
        Early-exit convergence criterion, or ``None`` for the fixed schedule.

    Returns
    -------
    v : Array
        The updated velocity field, same shape as the input ``v``.
    costs : Array
        The per-iteration cost trace.
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
    masking).  Rolled with ``lax.scan``.

    Parameters
    ----------
    moving
        Moving image at this resolution, shape ``(*spatial,)``.
    fixed
        Fixed (target) image at this resolution, shape ``(*spatial,)``.
    v_fwd
        Initial forward velocity field (moving -> midpoint), channel-last of
        shape ``(*spatial, ndim)``.
    v_inv
        Initial inverse velocity field (fixed -> midpoint), same shape as
        ``v_fwd``.
    force
        The similarity :class:`Force` driving both half-updates.
    ndim
        Spatial rank.
    iterations
        Number of iterations at this resolution.
    n_steps
        Number of scaling-and-squaring steps used to integrate each velocity to
        a displacement.
    boundary_mode
        Out-of-bounds handling for the warps and integration.
    sigma_fluid
        Gaussian sigma for the fluid (per-update) regularisation.
    sigma_diffusion
        Gaussian sigma for the diffusion (accumulated-field) regularisation.
    step
        Trust-region step cap, or ``None`` to skip the step clamp.
    rel_spacing
        Relative (anisotropy-only) per-axis spacing, or ``None`` for isotropic
        data.
    mask
        Channel-less region mask gating both half-forces, or ``None``.
    restrict
        Length-``ndim`` per-axis force weight applied to both halves, or
        ``None``.
    convergence
        Early-exit convergence criterion, or ``None`` for the fixed schedule.
        The early-exit cost is the symmetric mean of both half-costs.

    Returns
    -------
    v_fwd : Array
        The updated forward velocity field.
    v_inv : Array
        The updated inverse velocity field.
    costs : Array
        The per-iteration symmetric-mean cost trace.
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
        # The convergence cost is the SYMMETRIC mean of both half-costs; keying
        # the early-exit on the forward half alone lets an asymmetric metric
        # (MI / CR) stop while the inverse half is still improving.  A no-op for
        # a symmetric metric (SSD / LNCC: fwd == inv).
        cost = 0.5 * (bound_fwd.cost(a) + bound_inv.cost(b))
        return (v_fwd, v_inv), cost

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
    r"""Single-sided **group (greedy)** iterations on one resolution.

    The group-domain sibling of :func:`single_sided_level`: the state is the
    *displacement* ``s`` (:math:`\varphi = id + s`), warped **directly** (one
    gather, no exponential), and the regularised increment is **composed** onto
    ``s`` (the compositive demons update
    :math:`\varphi \leftarrow \varphi \circ (id + \delta)`) rather than added in
    the log domain.  ~2 gathers/iter (warp + compose) vs the algebra driver's
    ~7.  No per-iteration exponential; the velocity is recovered once at
    finalisation via :func:`nitrix.geometry.field_log` (the recipe).  Dropping
    ``n_steps`` / ``bch_order`` (no integration, no log-domain BCH) marks it a
    different driver, not a re-parametrised one.

    Parameters
    ----------
    moving
        Moving image at this resolution, shape ``(*spatial,)``.
    fixed
        Fixed (target) image at this resolution, shape ``(*spatial,)``.
    s
        Initial displacement field, channel-last of shape ``(*spatial, ndim)``.
    force
        The similarity :class:`Force` driving the update.
    ndim
        Spatial rank.
    iterations
        Number of iterations at this resolution.
    boundary_mode
        Out-of-bounds handling for the warp and composition.
    sigma_fluid
        Gaussian sigma for the fluid (per-update) regularisation.
    sigma_diffusion
        Gaussian sigma for the diffusion (total-field) regularisation applied
        after each composition.
    step
        Trust-region step size, or ``None`` to skip the step normalisation.
    rel_spacing
        Relative (anisotropy-only) per-axis spacing, or ``None`` for isotropic
        data.
    mask
        Channel-less region mask gating the force, or ``None``.
    restrict
        Length-``ndim`` per-axis force weight (deformation-axis masking), or
        ``None``.
    convergence
        Early-exit convergence criterion, or ``None`` for the fixed schedule.
    step_mode
        Step discipline passed to :func:`_group_regularise`: ``'clamp'``
        (default) or ``'normalize'``.

    Returns
    -------
    s : Array
        The updated displacement field, same shape as the input ``s``.
    costs : Array
        The per-iteration cost trace.
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

    Parameters
    ----------
    moving
        Moving image at this resolution, shape ``(*spatial,)``.
    fixed
        Fixed (target) image at this resolution, shape ``(*spatial,)``.
    s_fwd
        Initial forward displacement field (moving -> midpoint), channel-last of
        shape ``(*spatial, ndim)``.
    s_inv
        Initial inverse displacement field (fixed -> midpoint), same shape as
        ``s_fwd``.
    force
        The similarity :class:`Force` driving both half-updates.
    ndim
        Spatial rank.
    iterations
        Number of iterations at this resolution.
    boundary_mode
        Out-of-bounds handling for the warps and compositions.
    sigma_fluid
        Gaussian sigma for the fluid (per-update) regularisation.
    sigma_diffusion
        Gaussian sigma for the diffusion (total-field) regularisation applied
        after each composition.
    step
        Trust-region step size, or ``None`` to skip the step normalisation.
    rel_spacing
        Relative (anisotropy-only) per-axis spacing, or ``None`` for isotropic
        data.
    mask
        Channel-less region mask gating both half-forces, or ``None``.
    restrict
        Length-``ndim`` per-axis force weight applied to both halves, or
        ``None``.
    convergence
        Early-exit convergence criterion, or ``None`` for the fixed schedule.
        The early-exit cost is the symmetric mean of both half-costs.
    step_mode
        Step discipline passed to :func:`_group_regularise`: ``'clamp'``
        (default) or ``'normalize'``.

    Returns
    -------
    s_fwd : Array
        The updated forward displacement field.
    s_inv : Array
        The updated inverse displacement field.
    costs : Array
        The per-iteration symmetric-mean cost trace.
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
        # Symmetric mean of both half-costs (see symmetric_level).
        return (s_fwd, s_inv), 0.5 * (bound_fwd.cost(a) + bound_inv.cost(b))

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
    matrix, as :func:`rigid_register` / :func:`affine_register` return in
    :class:`IndexSpace`) or ``init_displacement`` (a displacement field on the
    fixed grid, e.g. a SynthMorph network output) may be given; the
    diffeomorphic recipe pre-warps ``moving`` by the result and registers the
    residual (warm-start / multi-stage).  An affine is expanded to its
    displacement field by applying the **self-contained** recipe matrix about
    the origin (its grid centre is already baked in).

    Parameters
    ----------
    init_affine
        Homogeneous ``(ndim + 1, ndim + 1)`` fixed-voxel -> moving-voxel matrix,
        or ``None``.  Mutually exclusive with ``init_displacement``.
    init_displacement
        Displacement field on the fixed grid, channel-last of shape
        ``(*shape, ndim)``, or ``None``.  Mutually exclusive with
        ``init_affine``.
    shape
        Spatial shape of the fixed grid.
    dtype
        Dtype of the constructed identity / displacement grid.

    Returns
    -------
    Array or None
        The initialisation as a fixed-grid displacement field of shape
        ``(*shape, ndim)``, or ``None`` when neither initialisation was given.

    Raises
    ------
    ValueError
        If both ``init_affine`` and ``init_displacement`` are provided.
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

    Parameters
    ----------
    moving
        Moving image, shape ``(*shape,)``.
    init_disp
        Fixed-grid initialisation displacement, channel-last of shape
        ``(*shape, ndim)``, or ``None`` for the no-init path.
    shape
        Spatial shape of the fixed grid.
    dtype
        Dtype of the constructed identity grid.
    boundary_mode
        Out-of-bounds handling for the resample.

    Returns
    -------
    Array
        ``moving`` unchanged when ``init_disp`` is ``None``; otherwise ``moving``
        resampled onto the fixed grid at ``id + init_disp``, shape ``(*shape,)``.
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

    The full ``moving -> fixed`` map applies the init first, then the residual
    deformable; ``init_disp is None`` reduces exactly to the residual.

    Parameters
    ----------
    moving
        The *original* moving image, shape ``(*shape,)``.
    residual_disp
        The residual deformable displacement on the fixed grid, channel-last of
        shape ``(*shape, ndim)``.
    init_disp
        The initialisation displacement composed before the residual, same shape
        as ``residual_disp``, or ``None``.
    shape
        Spatial shape of the fixed grid.
    dtype
        Dtype of the constructed identity grid.
    boundary_mode
        Out-of-bounds handling for the composition and warp.

    Returns
    -------
    total : Array
        The total ``moving -> fixed`` displacement, shape ``(*shape, ndim)``.
    warped : Array
        The original ``moving`` resampled by the total map, shape ``(*shape,)``.
    jacobian_det : Array
        The Jacobian determinant of the total map, shape ``(*shape,)``.
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
