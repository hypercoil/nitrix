# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Dense-field driving forces -- the velocity-update analogue of ``Objective``.

Where ``Objective`` is what the *matrix*-transform optimiser minimises over a
parameter vector, a ``Force`` is what the *dense-field* (SVF) drivers
(log-Demons, greedy SyN) ascend over a velocity field: per iteration it turns
``(warped, fixed)`` into a raw per-voxel velocity update.  Holding it behind a
protocol decouples the metric from the algorithm -- the same single-sided /
symmetric driver runs under any force -- exactly as ``Objective`` decoupled the
metric from the matrix driver.

Two tiers, the suite-wide perf-vs-composability discipline:

- **Tier 1 (performant, the named recipes' defaults):** :class:`LNCCForce`
  (the analytic local-CC force, ``metrics.lncc_grad``), :class:`DemonsForce`
  (the closed-form ESM force), and :class:`MIForce` (the closed-form Mattes
  mutual-information force, ``metrics.mi_grad`` -- the fast cross-modal path
  replacing the ``MetricForce(MI())`` autodiff tape) -- closed-form forces
  behind the protocol.
- **Tier 2 (escape hatch, *no* performance guarantee):** :class:`MetricForce`
  -- ``jax.grad`` of *any* :class:`Metric`'s cost, so MI / correlation-ratio /
  a custom descriptor can drive a diffeomorphic recipe (multimodal deformable
  registration) without a hand-written force.

:class:`SumForce` is the **combinator** -- a weighted sum of other forces
(multi-metric within one stage, ANTs ``-m … -m …``), itself a :class:`Force`, so
it composes any of the above with no driver change.

**The force direction.**  With a minimisation cost ``c(warped, fixed)`` the
gradient-descent direction on the moving deformation is, by the chain rule
``∂c/∂(deformation) = ∂c/∂warped · ∇warped``, the field
``u = −(∂c/∂warped) · ∇warped``.  The closed forms are this up to the
metric-specific structure: LNCC's ``∂(sim)/∂warped`` is ``lncc_grad``; Demons'
is the symmetric ESM variant ``(F−warped)·J/denom``.  So
``MetricForce(LNCC)`` and :class:`LNCCForce` agree in *direction*
(machine-precision; the magnitude is fixed by the driver's step-normalisation),
which is the parity oracle for the closed form.

**The ``bind`` hoist.**  ``Force.bind(fixed)`` returns a :class:`BoundForce`
that has precomputed everything fixed-dependent (Demons' ``∇fixed``), so the
single-sided driver binds **once per level** and reuses it across iterations;
the symmetric driver binds **per step** (its "fixed" is the other image warped
to the midpoint, which changes every iteration) -- the same hoisting discipline
as ``ic_reference`` / ``MetricObjective``, paid only where it is recoverable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Literal,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .._internal.backend import Backend, fallback, resolve_backend
from ..geometry import spatial_gradient
from ..metrics import (
    lncc,
    lncc_grad,
    lncc_grad_center,
    mi_grad,
    mutual_information,
    nmi_grad,
)
from ._metric import Metric

__all__ = [
    'Force',
    'BoundForce',
    'LNCCForce',
    'DemonsForce',
    'MIForce',
    'MetricForce',
    'SumForce',
    'resolve_force_schedule',
]

# A recipe's ``force`` argument: a single force everywhere, a coarse-to-fine
# per-level schedule, or ``None`` for the recipe's closed-form default.
ForceArg = Optional[Union['Force', Sequence['Force']]]


def resolve_force_schedule(
    force: ForceArg,
    *,
    default: Force,
    levels: int,
) -> list[Force]:
    """Resolve the recipe ``force`` argument to a per-pyramid-level list.

    ``None`` -> the ``default`` closed-form force at every level; a single
    :class:`Force` -> that force at every level; a length-``levels`` sequence
    (**coarse-to-fine**, the natural schedule order -- e.g. a cheap force at
    the coarse levels, a high-signal one at the finest) -> one force per level,
    reversed to the finest-first pyramid indexing the driver uses.  The result
    is indexed by pyramid level (``[0]`` = finest).
    """
    if force is None:
        return [default] * levels
    if isinstance(force, Force):
        return [force] * levels
    seq = list(force)
    if len(seq) != levels:
        raise ValueError(
            f'force schedule must have length levels={levels}; got {len(seq)}.'
        )
    return list(reversed(seq))


# Anisotropy-only spacing (``None`` -> isotropic); the per-axis ratio
# ``spacing / geomean(spacing)`` the SVF substrate threads (see ``_svf``).
RelSpacing = Optional[tuple[float, ...]]

# Below this the ESM denominator ``|j|² + α²·diff²`` is treated as a matched
# uniform region (both the gradient and the mismatch vanish): the force is
# zeroed there rather than dividing 0/0.  Absolute (not relative) -- any real
# gradient or mismatch puts the denominator orders of magnitude above it across
# the intensity scales registration sees.
_DEMONS_DENOM_EPS = 1e-8

# Below this RMS the force field is treated as no-signal and left at zero rather
# than amplified to the target magnitude (``_normalise_rms``).  Absolute on the
# already metric-scale-arbitrary cost gradient: only the genuinely-zero field
# (a degenerate all-flat / perfectly-matched histogram) falls under it.
_RMS_EPS = 1e-12


def _grad_spacing(
    rel_spacing: RelSpacing,
) -> Union[float, tuple[float, ...]]:
    """Spacing argument for ``spatial_gradient`` (1.0 when isotropic)."""
    return 1.0 if rel_spacing is None else rel_spacing


def _to_voxel(u: Array, rel_spacing: RelSpacing) -> Array:
    """Convert a physical (mm) force to the voxel-native velocity field."""
    if rel_spacing is None:
        return u
    return u / jnp.asarray(rel_spacing, dtype=u.dtype)


def _normalise_rms(u: Float[Array, '*spatial ndim'], target: float) -> Array:
    """Scale a force field to a target per-voxel RMS magnitude.

    RMS over the spatial voxels of the per-voxel vector norm ``‖u_x‖`` (not the
    global max -- robust to a single outlier voxel setting the scale, the same
    discipline ``_svf._normalise_step`` adopts).  Scaling *to* a fixed target
    makes the step **metric-scale-invariant**: independent of the metric's
    intrinsic units (MI in nats, CR in ``[0,1]``) and of the voxel count, so an
    unclamped driver (Demons) gets a controlled magnitude instead of the
    arbitrary ``·size`` constant.  A genuinely-zero field (no signal) is left at
    zero, with the double-``where`` keeping the gradient finite there too.
    """
    rms = jnp.sqrt(jnp.mean(jnp.sum(u * u, axis=-1)))
    safe = rms > _RMS_EPS
    scale = jnp.where(safe, target / jnp.where(safe, rms, 1.0), 0.0)
    return u * scale


@runtime_checkable
class BoundForce(Protocol):
    """A :class:`Force` bound to a ``fixed`` image (fixed-state hoisted)."""

    def update(
        self, warped: Float[Array, '*spatial']
    ) -> Float[Array, '*spatial ndim']:
        """Raw per-voxel velocity update (voxel units, pre-regularisation)."""
        ...

    def cost(self, warped: Float[Array, '*spatial']) -> Float[Array, '']:
        """Scalar similarity cost (lower is a better match)."""
        ...


@runtime_checkable
class Force(Protocol):
    """Driving force for a dense-field (SVF) registration driver."""

    def bind(
        self,
        fixed: Float[Array, '*spatial'],
        *,
        ndim: int,
        rel_spacing: RelSpacing = None,
    ) -> BoundForce:
        """Bind to ``fixed``, hoisting fixed-dependent state once."""
        ...


# The lncc_grad / lncc_grad_center flat-window guard epsilon (their default).
_LNCC_EPS = 1e-5

# Size gate (L2b): the sliding-window kernel only beats the JAX integral image
# above ~128^3 -- below, its per-program x-scan overhead dominates (measured
# crossover).  Critically, the ANTs schedule front-loads iterations at the
# *coarse* (small) levels, so firing the kernel there would slow the recipe; the
# gate defers those to JAX so the kernel only runs at the large fine levels
# (where it wins, e.g. the 256^3 full-resolution stage).
_MIN_KERNEL_VOXELS = 2_000_000  # ~128^3


def _lncc_center_dispatch(
    warped: Array,
    fixed: Array,
    *,
    radius: int,
    backend: Backend,
) -> Optional[Array]:
    """The fused sliding-window centre force where eligible, else ``None``.

    Returns the Pallas-kernel force (3-D isotropic single-pair GPU, large
    enough to win) or ``None`` to signal the caller to run the JAX path.  A
    too-small volume / non-3-D input defers to JAX (the size gate is silent;
    a non-tileable shape is a loud ``NitrixBackendFallback``).  ``backend='jax'``
    (the default) resolves straight to ``None`` and never warns.
    """
    if resolve_backend(backend) != 'pallas-cuda':
        return None
    if warped.size < _MIN_KERNEL_VOXELS:
        return None  # size gate: silent defer to JAX at the coarse levels
    from .._kernels.cuda import lncc_force as _k

    try:
        return _k.lncc_center_force_pallas(
            warped, fixed, radius=int(radius), eps=_LNCC_EPS
        )
    except _k.PallasNotTileable as exc:
        fallback(
            function='LNCCForce',
            requested='pallas-cuda',
            resolved='jax',
            reason=str(exc),
            shapes=(tuple(warped.shape),),
            dtype=warped.dtype,
        )
        return None


@dataclass(frozen=True)
class LNCCForce:
    """Analytic local-cross-correlation force (the greedy-SyN default).

    The closed-form ``lncc_grad(warped, fixed) · ∇warped`` -- ascends the
    windowed CC; robust to smooth intensity inhomogeneity.

    Attributes
    ----------
    radius
        LNCC box-window radius (size ``2·radius + 1`` per axis).  On an
        anisotropic grid (``rel_spacing`` passed by the recipe) the window is
        made **physically isotropic**: the per-axis voxel radius is scaled by
        ``1 / rel_spacing`` so every axis spans the same mm extent (the same
        convention the regularisation sigmas follow).
    derivative
        Which local-CC force to use.  ``'exact'`` (default) is
        :func:`metrics.lncc_grad` -- the exact gradient of the summed CC (nine
        box sums).  ``'center'`` is :func:`metrics.lncc_grad_center`, the
        ANTs / ITK centre-only convention (five box sums, ~1.5-1.75x faster on
        both GPU and CPU): a *different*, cheaper force, not ``lncc_grad``'s
        gradient.  Default is ``'exact'`` so recipe output is byte-identical.
    backend
        Compute backend for the ``'center'`` force (v4 L2b).  ``'jax'``
        (default) is the pure-JAX path; ``'pallas-cuda'`` / ``'auto'`` use the
        fused **sliding-window** Triton kernel on a supported GPU (3-D isotropic
        single-pair -- otherwise it falls back to JAX, tolerance-equal).  The
        kernel beats the JAX integral image and the win grows with volume
        (~1.2x @128^3, ~4x @256^3).  ``'exact'`` always uses JAX (its two-pass
        form has no winning kernel).  Default ``'jax'`` keeps output unchanged.
    """

    radius: int = 2
    derivative: Literal['exact', 'center'] = 'exact'
    backend: Backend = 'jax'

    def bind(
        self,
        fixed: Float[Array, '*spatial'],
        *,
        ndim: int,
        rel_spacing: RelSpacing = None,
    ) -> _BoundLNCC:
        return _BoundLNCC(
            fixed=fixed,
            radius=self.radius,
            ndim=ndim,
            rel_spacing=rel_spacing,
            derivative=self.derivative,
            backend=self.backend,
        )


@dataclass(frozen=True)
class _BoundLNCC:
    fixed: Array
    radius: int
    ndim: int
    rel_spacing: RelSpacing
    derivative: Literal['exact', 'center'] = 'exact'
    backend: Backend = 'jax'

    def _radii(self) -> Union[int, tuple[int, ...]]:
        """Per-axis voxel radii of a *physically isotropic* window.

        On an anisotropic grid a voxel-isotropic window spans different mm per
        axis; scaling the radius by ``1 / rel_spacing`` (the same physical
        convention the regularisation sigmas already follow) makes the window
        the same mm extent on every axis.  Isotropic ``rel_spacing is None``
        leaves the plain voxel radius unchanged.
        """
        if self.rel_spacing is None:
            return self.radius
        return tuple(max(1, round(self.radius / r)) for r in self.rel_spacing)

    def update(
        self, warped: Float[Array, '*spatial']
    ) -> Float[Array, '*spatial ndim']:
        # L2b: the fused sliding-window kernel computes the whole centre force.
        if self.derivative == 'center' and self.rel_spacing is None:
            kernel_u = _lncc_center_dispatch(
                warped, self.fixed, radius=self.radius, backend=self.backend
            )
            if kernel_u is not None:
                return kernel_u
        grad_fn = (
            lncc_grad_center if self.derivative == 'center' else lncc_grad
        )
        scalar = grad_fn(warped, self.fixed, radius=self._radii())
        grad = spatial_gradient(
            warped, spacing=_grad_spacing(self.rel_spacing)
        )
        return _to_voxel(scalar[..., None] * grad, self.rel_spacing)

    def cost(self, warped: Float[Array, '*spatial']) -> Float[Array, '']:
        return 1.0 - lncc(warped, self.fixed, radius=self._radii())


def _demons_update_jax(
    warped: Array,
    fixed: Array,
    grad_fixed: Array,
    alpha: float,
    rel_spacing: RelSpacing,
) -> Array:
    """The pure-JAX ESM force -- the parity oracle and the fallback path."""
    diff = fixed - warped
    grad = spatial_gradient(warped, spacing=_grad_spacing(rel_spacing))
    j = 0.5 * (grad_fixed + grad)
    denom = jnp.sum(j * j, axis=-1) + (alpha**2) * diff * diff
    # Guard the 0/0 on a matched uniform region (denom = |j|² + α²·diff² = 0
    # iff both ∇ and the mismatch vanish): zero force there -- the correct
    # demons update (no gradient, no mismatch -> no information).  The
    # double-``where`` keeps the *gradient* finite too (a bare ``diff/denom``
    # NaNs the backward even where the forward is masked).
    safe = denom > _DEMONS_DENOM_EPS
    scale = jnp.where(safe, diff / jnp.where(safe, denom, 1.0), 0.0)
    return _to_voxel(scale[..., None] * j, rel_spacing)


def _demons_force_dispatch(
    warped: Array,
    fixed: Array,
    grad_fixed: Array,
    *,
    alpha: float,
    rel_spacing: RelSpacing,
    backend: Backend,
) -> Array:
    """Pallas-fused ESM force where eligible, else the JAX path (5a).

    The fused kernel (``∇warped`` stencil + the force in one tiled pass) is
    isotropic single-pair GPU only; an anisotropic spacing or an untileable
    shape falls back to :func:`_demons_update_jax` with a loud, deduped
    ``NitrixBackendFallback`` (``backend='jax'``, the default, never warns).
    """
    resolved = resolve_backend(backend)
    if resolved == 'pallas-cuda':
        if rel_spacing is not None:
            resolved = fallback(
                function='DemonsForce',
                requested='pallas-cuda',
                resolved='jax',
                reason='anisotropic spacing is unsupported by the fused '
                'kernel; the voxel-native path runs instead',
                shapes=(tuple(warped.shape),),
                dtype=warped.dtype,
            )
        else:
            # Lazy import: only the pallas-cuda path pulls the Triton kernel.
            from .._kernels.cuda import demons_force as _k

            try:
                return _k.demons_esm_force_pallas(
                    warped,
                    fixed,
                    grad_fixed,
                    alpha=alpha,
                    eps=_DEMONS_DENOM_EPS,
                )
            except _k.PallasNotTileable as exc:
                resolved = fallback(
                    function='DemonsForce',
                    requested='pallas-cuda',
                    resolved='jax',
                    reason=str(exc),
                    shapes=(tuple(warped.shape),),
                    dtype=warped.dtype,
                )
    return _demons_update_jax(warped, fixed, grad_fixed, alpha, rel_spacing)


@dataclass(frozen=True)
class DemonsForce:
    """Closed-form efficient-symmetric (ESM) Demons force (the Demons default).

    ``u = (F − warped) · J / (|J|² + α²(F − warped)²)`` with the symmetric
    gradient ``J = ½(∇F + ∇warped)``.  The fixed gradient ``∇F`` is hoisted by
    :meth:`bind` (constant across the single-sided iterations).

    Attributes
    ----------
    alpha
        Force normalisation: larger ``α`` damps the step where the intensity
        difference is large.
    backend
        Compute backend (v4 Phase 5a).  ``'jax'`` (default) is the pure-JAX
        path; ``'pallas-cuda'`` / ``'auto'`` use the fused ESM Triton kernel on
        a supported GPU (isotropic single-pair only -- otherwise it falls back
        to JAX, ULP-equal either way).  The default is ``'jax'`` so recipe
        output is byte-identical until a profile justifies the kernel.
    """

    alpha: float = 0.4
    backend: Backend = 'jax'

    def bind(
        self,
        fixed: Float[Array, '*spatial'],
        *,
        ndim: int,
        rel_spacing: RelSpacing = None,
    ) -> _BoundDemons:
        grad_fixed = spatial_gradient(
            fixed, spacing=_grad_spacing(rel_spacing)
        )
        return _BoundDemons(
            fixed=fixed,
            grad_fixed=grad_fixed,
            alpha=self.alpha,
            ndim=ndim,
            rel_spacing=rel_spacing,
            backend=self.backend,
        )


@dataclass(frozen=True)
class _BoundDemons:
    fixed: Array
    grad_fixed: Array
    alpha: float
    ndim: int
    rel_spacing: RelSpacing
    backend: Backend = 'jax'

    def update(
        self, warped: Float[Array, '*spatial']
    ) -> Float[Array, '*spatial ndim']:
        return _demons_force_dispatch(
            warped,
            self.fixed,
            self.grad_fixed,
            alpha=self.alpha,
            rel_spacing=self.rel_spacing,
            backend=self.backend,
        )

    def cost(self, warped: Float[Array, '*spatial']) -> Float[Array, '']:
        diff = self.fixed - warped
        return 0.5 * jnp.sum(diff * diff)


@dataclass(frozen=True)
class MIForce:
    """Closed-form Mattes mutual-information force (the cross-modal fast path).

    The Tier-1 analogue of ``MetricForce(MI())``: drives a dense-field recipe by
    ``mi_grad · ∇warped`` (the closed-form ``metrics.mi_grad``, ``∂MI/∂warped``)
    instead of a full ``jax.grad`` of the soft-histogram cost every iteration --
    the fast cross-modal deformable path (the fMRIPrep metric), with the
    histogram-scatter autodiff tape removed.

    Like ``MetricForce`` for a histogram metric (and for the same reason -- MI is
    not a spatial mean, so its raw ``(1/N)`` gradient is an arbitrary,
    metric-scale-dependent magnitude), the force is normalised to a controlled
    per-voxel RMS ``magnitude``: this is what makes it a true *drop-in* fast
    replacement (same controlled step the clamped SyN / unclamped Demons drivers
    need), and it agrees with ``MetricForce(MI())`` in **direction** to the
    empty-bin tolerance and in **magnitude** by construction (the §3 parity
    oracle).

    Attributes
    ----------
    bins
        Joint-histogram bins per axis (must match the cost).
    range_moving, range_fixed
        Pinned ``(lo, hi)`` intensity ranges.  ``None`` is resolved **once** from
        the full-resolution images by the recipe (``_svf.pin_force_ranges``)
        before the pyramid -- a data ``min/max`` range drifts as the moving image
        deforms (a non-stationary objective).  The pin is a ``stop_gradient``-ed
        reduction, so it is **jit-safe**: ``MIForce(bins=...)`` traces with no
        explicit range (the range is auto-derived once and held constant).  Pass
        an explicit range only to fix a binning other than the data min/max.
    magnitude
        Target per-voxel RMS magnitude (voxels); ``0.5`` matches
        ``MetricForce``.
    normalized
        Use Studholme's normalised MI (``NMI = (H_m + H_f) / H_mf``, the ANTs
        cross-modal default), routing the force through the closed-form
        ``metrics.nmi_grad`` (the quotient-rule ``∂NMI/∂warped``, C1) instead of
        ``mi_grad``.  ``False`` (default) is unnormalised Mattes MI.  The cost
        (``BoundForce.cost``) follows the same flag.  The RMS-``magnitude``
        normalisation makes NMI and MI interchangeable as drop-in forces (the
        scale difference between the two raw gradients is absorbed).
    sample_stride
        Estimate the joint histogram from every ``sample_stride``-th voxel (ITK
        "Regular" sampling -- the histogram scatter is the MI bottleneck, and
        the PDF is a smooth global statistic well estimated from a sample); the
        force is still applied **densely**.  ``1`` (default) is the exact full
        histogram.  ``4`` (~25 %, ITK's default sampling) keeps the force
        ~0.98 cos-aligned with the full one on real cross-modal data for a ~3x
        cheaper ``mi_grad`` -- the deformable-MI speed lever (fMRIPrep's metric).
        The regular grid is a *deterministic* sampler, so in principle it could
        alias with stride-frequency image periodicity (a heavy Gibbs artifact);
        on real anatomy it does not (the gradient is offset-insensitive), and
        the grid's spatial uniformity gives it *lower* variance than random
        sampling -- but rotate the offset per level if a pathological texture
        ever shows offset-dependence.
    """

    bins: int = 32
    range_moving: Optional[tuple[float, float]] = None
    range_fixed: Optional[tuple[float, float]] = None
    magnitude: float = 0.5
    sample_stride: int = 1
    normalized: bool = False

    def bind(
        self,
        fixed: Float[Array, '*spatial'],
        *,
        ndim: int,
        rel_spacing: RelSpacing = None,
    ) -> _BoundMI:
        return _BoundMI(
            bins=self.bins,
            range_moving=self.range_moving,
            range_fixed=self.range_fixed,
            magnitude=self.magnitude,
            sample_stride=self.sample_stride,
            normalized=self.normalized,
            fixed=fixed,
            ndim=ndim,
            rel_spacing=rel_spacing,
        )


@dataclass(frozen=True)
class _BoundMI:
    bins: int
    range_moving: Optional[tuple[float, float]]
    range_fixed: Optional[tuple[float, float]]
    magnitude: float
    fixed: Array
    ndim: int
    rel_spacing: RelSpacing
    sample_stride: int = 1
    normalized: bool = False

    def update(
        self, warped: Float[Array, '*spatial']
    ) -> Float[Array, '*spatial ndim']:
        # The joint histogram depends on BOTH images at the current warp, so --
        # unlike DemonsForce's ∇fixed -- there is nothing image-dependent to
        # hoist in ``bind``; the gradient recomputes the histogram every
        # iteration.  ``normalized`` routes to the closed-form NMI gradient (C1).
        grad_fn = nmi_grad if self.normalized else mi_grad
        g = grad_fn(
            warped,
            self.fixed,
            bins=self.bins,
            range_moving=self.range_moving,
            range_fixed=self.range_fixed,
            sample_stride=self.sample_stride,
        )
        grad = spatial_gradient(
            warped, spacing=_grad_spacing(self.rel_spacing)
        )
        # Force convention u = −∂cost/∂warped·∇warped with cost = −MI/−NMI gives
        # u = +grad·∇warped (ascend MI / NMI), then the controlled-magnitude RMS
        # normalisation (a histogram metric is not a spatial mean -- 0c / B2;
        # this also absorbs the scale gap between the MI and NMI gradients).
        force = _normalise_rms(g[..., None] * grad, self.magnitude)
        return _to_voxel(force, self.rel_spacing)

    def cost(self, warped: Float[Array, '*spatial']) -> Float[Array, '']:
        return -mutual_information(
            warped,
            self.fixed,
            bins=self.bins,
            range_moving=self.range_moving,
            range_fixed=self.range_fixed,
            normalized=self.normalized,
        )


@dataclass(frozen=True)
class MetricForce:
    """Generic force from any :class:`Metric` -- the escape hatch.

    Drives a dense-field recipe by ``−∂(metric.cost)/∂warped · ∇warped`` (the
    cost gradient via ``jax.grad``), so a cross-modal metric (MI, correlation
    ratio) or a custom descriptor can register diffeomorphically with **no**
    hand-written force.  *No performance guarantee* -- the autodiff cost is the
    price of generality; ship a closed-form :class:`Force` for a metric that
    earns one.

    **Two magnitude regimes, keyed on ``metric.is_spatial_mean``:**

    - **Spatial-mean cost** (``SSD`` / ``LNCC``): the cost reduces by a
      per-voxel mean, so rescaling the gradient by the voxel count undoes the
      ``1/N`` and recovers the *sum-convention* gradient the closed forms use --
      ``MetricForce(LNCC(r))`` is then numerically identical to ``LNCCForce(r)``
      (the parity oracle, in magnitude not only direction), and the magnitude
      shrinks naturally as the gradient does.  ``SSD`` has **no** normalised
      closed form here -- ``MetricForce(SSD)`` is the *raw* optical-flow
      gradient ``(warped − fixed)·∇warped``, not the Thirion-normalised demons
      force, and recovers markedly worse on a monomodal warp (it is unscaled by
      ``|∇|² + (m−f)²``, so the regulariser / trust-region clamp dominates and
      the field under-recovers); for monomodal SSD-type registration prefer
      :class:`DemonsForce` (the ``diffeomorphic_demons_register`` default).
    - **Global histogram scalar** (``MI`` / ``CorrelationRatio``, or any metric
      that does not declare ``is_spatial_mean``): there is no ``1/N`` to undo,
      so ``·N`` would be an arbitrary, metric-scale-dependent constant -- under
      the **unclamped** Demons driver (``step=None``) that feeds a magnitude no
      one has tuned straight into the velocity.  Instead the force is normalised
      to a target per-voxel RMS ``magnitude`` (``_normalise_rms``), a
      controlled, metric-scale-invariant step.  Under the clamped SyN driver the
      subsequent trust-region clamp only sharpens this; the direction (the parity
      that matters for a metric with no closed form) is untouched either way.

    Attributes
    ----------
    metric
        The :class:`Metric` whose cost is descended (``MI()``,
        ``CorrelationRatio()``, ``LNCC()``, ``SSD()``, ...).
    magnitude
        Target per-voxel RMS magnitude (voxels) for a **non**-spatial-mean
        metric (ignored for ``SSD`` / ``LNCC``, which take the parity rescale).
        The controlled step the unclamped driver applies per iteration; the
        ``0.5`` default is conservative -- larger over-steps the gateless
        Demons accumulation (weaker recovery, Jacobian toward folding), while
        the clamped SyN driver is insensitive to it (an active trust-region
        clamp cancels the pre-scale).
    """

    metric: Metric
    magnitude: float = 0.5

    def bind(
        self,
        fixed: Float[Array, '*spatial'],
        *,
        ndim: int,
        rel_spacing: RelSpacing = None,
    ) -> _BoundMetric:
        return _BoundMetric(
            metric=self.metric,
            fixed=fixed,
            ndim=ndim,
            rel_spacing=rel_spacing,
            magnitude=self.magnitude,
        )


@dataclass(frozen=True)
class _BoundMetric:
    metric: Metric
    fixed: Array
    ndim: int
    rel_spacing: RelSpacing
    magnitude: float

    def update(
        self, warped: Float[Array, '*spatial']
    ) -> Float[Array, '*spatial ndim']:
        grad_cost = jax.grad(lambda w: self.metric.cost(w, self.fixed))(warped)
        grad = spatial_gradient(
            warped, spacing=_grad_spacing(self.rel_spacing)
        )
        force = -grad_cost[..., None] * grad
        if getattr(self.metric, 'is_spatial_mean', False):
            # Spatial-mean cost (SSD, LNCC): the closed forms (and the driver's
            # step-normalisation) use the SUM-convention gradient
            # (``lncc_grad = ∂(Σcc)/∂warped``); rescale by the voxel count to
            # undo the ``1/N`` mean.  This makes MetricForce(LNCC) *numerically
            # identical* to the closed-form ``LNCCForce``.  It does **not**
            # reproduce ``DemonsForce`` (the symmetric ESM update carries its own
            # per-voxel denominator), so MetricForce(SSD) matches only the
            # Thirion *direction*, not the ESM step.
            force = force * warped.size
        else:
            # Global histogram scalar (MI, CR) or an undeclared user metric:
            # there is no ``1/N`` to undo, so ``·N`` would be an arbitrary
            # metric-scale-dependent constant -- fed straight into the velocity
            # by the unclamped Demons driver.  Normalise to a controlled,
            # metric-scale-invariant RMS instead (B2; the clamped SyN driver
            # then only sharpens it, leaving the direction intact).
            force = _normalise_rms(force, self.magnitude)
        return _to_voxel(force, self.rel_spacing)

    def cost(self, warped: Float[Array, '*spatial']) -> Float[Array, '']:
        return self.metric.cost(warped, self.fixed)


@dataclass(frozen=True)
class SumForce:
    """Weighted sum of forces -- multi-metric within one stage (A5).

    The dense-field analogue of ANTs ``-m MI[...] -m CC[...]`` in a single stage:
    :meth:`update` is ``Σ wᵢ · forceᵢ.update`` and :meth:`cost` the matching
    weighted sum, so a *combination* of metrics drives one recipe with **no**
    driver change -- a ``SumForce`` is itself just another :class:`Force`.  Each
    term is bound once (its fixed-state hoisted) by :meth:`bind`; the terms may
    mix tiers freely (e.g. a closed-form :class:`LNCCForce` plus a
    :class:`MIForce` for a cross-modal-plus-structural drive).

    Attributes
    ----------
    terms
        A non-empty tuple of ``(weight, force)`` pairs.
    """

    terms: tuple[tuple[float, Force], ...]

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError(
                'SumForce requires at least one (weight, force) term.'
            )

    def bind(
        self,
        fixed: Float[Array, '*spatial'],
        *,
        ndim: int,
        rel_spacing: RelSpacing = None,
    ) -> _BoundSum:
        return _BoundSum(
            terms=tuple(
                (float(w), f.bind(fixed, ndim=ndim, rel_spacing=rel_spacing))
                for w, f in self.terms
            )
        )


@dataclass(frozen=True)
class _BoundSum:
    terms: tuple[tuple[float, BoundForce], ...]

    def update(
        self, warped: Float[Array, '*spatial']
    ) -> Float[Array, '*spatial ndim']:
        w0, b0 = self.terms[0]
        total = w0 * b0.update(warped)
        for w, b in self.terms[1:]:
            total = total + w * b.update(warped)
        return total

    def cost(self, warped: Float[Array, '*spatial']) -> Float[Array, '']:
        w0, b0 = self.terms[0]
        total = w0 * b0.cost(warped)
        for w, b in self.terms[1:]:
            total = total + w * b.cost(warped)
        return total
