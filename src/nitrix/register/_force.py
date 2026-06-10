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
  (the analytic local-CC force, ``metrics.lncc_grad``) and :class:`DemonsForce`
  (the closed-form ESM force) -- the forces we already ship, now behind the
  protocol.
- **Tier 2 (escape hatch, *no* performance guarantee):** :class:`MetricForce`
  -- ``jax.grad`` of *any* :class:`Metric`'s cost, so MI / correlation-ratio /
  a custom descriptor can drive a diffeomorphic recipe (multimodal deformable
  registration) without a hand-written force.

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
from typing import Optional, Protocol, Sequence, Union, runtime_checkable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..geometry import spatial_gradient
from ..metrics import lncc, lncc_grad
from ._metric import Metric

__all__ = [
    'Force',
    'BoundForce',
    'LNCCForce',
    'DemonsForce',
    'MetricForce',
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


@dataclass(frozen=True)
class LNCCForce:
    """Analytic local-cross-correlation force (the greedy-SyN default).

    The closed-form ``lncc_grad(warped, fixed) · ∇warped`` -- ascends the
    windowed CC; robust to smooth intensity inhomogeneity.

    Attributes
    ----------
    radius
        LNCC box-window radius (size ``2·radius + 1`` per axis).
    """

    radius: int = 2

    def bind(
        self,
        fixed: Float[Array, '*spatial'],
        *,
        ndim: int,
        rel_spacing: RelSpacing = None,
    ) -> _BoundLNCC:
        return _BoundLNCC(
            fixed=fixed, radius=self.radius, ndim=ndim, rel_spacing=rel_spacing
        )


@dataclass(frozen=True)
class _BoundLNCC:
    fixed: Array
    radius: int
    ndim: int
    rel_spacing: RelSpacing

    def update(
        self, warped: Float[Array, '*spatial']
    ) -> Float[Array, '*spatial ndim']:
        scalar = lncc_grad(warped, self.fixed, radius=self.radius)
        grad = spatial_gradient(warped, spacing=_grad_spacing(self.rel_spacing))
        return _to_voxel(scalar[..., None] * grad, self.rel_spacing)

    def cost(self, warped: Float[Array, '*spatial']) -> Float[Array, '']:
        return 1.0 - lncc(warped, self.fixed, radius=self.radius)


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
    """

    alpha: float = 0.4

    def bind(
        self,
        fixed: Float[Array, '*spatial'],
        *,
        ndim: int,
        rel_spacing: RelSpacing = None,
    ) -> _BoundDemons:
        grad_fixed = spatial_gradient(fixed, spacing=_grad_spacing(rel_spacing))
        return _BoundDemons(
            fixed=fixed,
            grad_fixed=grad_fixed,
            alpha=self.alpha,
            ndim=ndim,
            rel_spacing=rel_spacing,
        )


@dataclass(frozen=True)
class _BoundDemons:
    fixed: Array
    grad_fixed: Array
    alpha: float
    ndim: int
    rel_spacing: RelSpacing

    def update(
        self, warped: Float[Array, '*spatial']
    ) -> Float[Array, '*spatial ndim']:
        diff = self.fixed - warped
        grad = spatial_gradient(warped, spacing=_grad_spacing(self.rel_spacing))
        j = 0.5 * (self.grad_fixed + grad)
        denom = jnp.sum(j * j, axis=-1) + (self.alpha**2) * diff * diff
        return _to_voxel((diff / denom)[..., None] * j, self.rel_spacing)

    def cost(self, warped: Float[Array, '*spatial']) -> Float[Array, '']:
        diff = self.fixed - warped
        return 0.5 * jnp.sum(diff * diff)


@dataclass(frozen=True)
class MetricForce:
    """Generic force from any :class:`Metric` -- the escape hatch.

    Drives a dense-field recipe by ``−∂(metric.cost)/∂warped · ∇warped`` (the
    cost gradient via ``jax.grad``), so a cross-modal metric (MI, correlation
    ratio) or a custom descriptor can register diffeomorphically with **no**
    hand-written force.  *No performance guarantee* -- the autodiff cost is the
    price of generality; ship a closed-form :class:`Force` for a metric that
    earns one.

    The cost gradient is rescaled by the voxel count to match the
    sum-convention the closed forms use (the metric costs reduce by spatial
    mean), so ``MetricForce(LNCC(r))`` is numerically identical to
    ``LNCCForce(r)`` -- the parity oracle holds in magnitude, not only
    direction.

    Attributes
    ----------
    metric
        The :class:`Metric` whose cost is descended (``MI()``,
        ``CorrelationRatio()``, ``LNCC()``, ``SSD()``, ...).
    """

    metric: Metric

    def bind(
        self,
        fixed: Float[Array, '*spatial'],
        *,
        ndim: int,
        rel_spacing: RelSpacing = None,
    ) -> _BoundMetric:
        return _BoundMetric(
            metric=self.metric, fixed=fixed, ndim=ndim, rel_spacing=rel_spacing
        )


@dataclass(frozen=True)
class _BoundMetric:
    metric: Metric
    fixed: Array
    ndim: int
    rel_spacing: RelSpacing

    def update(
        self, warped: Float[Array, '*spatial']
    ) -> Float[Array, '*spatial ndim']:
        # The metric costs reduce by spatial MEAN; the closed-form forces (and
        # the driver's step-normalisation) use the SUM-convention gradient
        # (``lncc_grad = ∂(Σcc)/∂warped``).  Rescale by the voxel count to undo
        # the mean: for a spatial-mean metric (LNCC, SSD) this makes the escape
        # hatch *numerically identical* to the closed form; for a histogram
        # metric (MI, CR) it is a constant the clamped driver absorbs.
        grad_cost = jax.grad(lambda w: self.metric.cost(w, self.fixed))(warped)
        grad_cost = grad_cost * warped.size
        grad = spatial_gradient(warped, spacing=_grad_spacing(self.rel_spacing))
        return _to_voxel(-grad_cost[..., None] * grad, self.rel_spacing)

    def cost(self, warped: Float[Array, '*spatial']) -> Float[Array, '']:
        return self.metric.cost(warped, self.fixed)
