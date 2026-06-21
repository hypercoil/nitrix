# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Registration similarity metrics as an ADT.

The objective a matrix-transform recipe minimises over the transform
parameters.  Each metric is an immutable ``@dataclass(frozen=True)``
record carrying its own hyper-parameters (window radius, histogram bins)
-- the ``Interpolator`` precedent -- so the ``RegistrationSpec`` config no
longer flattens every metric's knobs into one god-record, and the driver
dispatches on a method rather than a string ``if``-chain.

A metric exposes:

- ``is_least_squares`` (``ClassVar``) -- whether it admits a vector
  residual whose half-sum-of-squares is the cost.  ``True`` routes the
  driver to the Gauss-Newton / Levenberg-Marquardt least-squares path;
  ``False`` to the scalar BFGS path.
- ``is_spatial_mean`` (``ClassVar``) -- whether the scalar cost is a
  per-voxel spatial *mean* (``SSD`` / ``LNCC``) rather than a global
  histogram statistic (``MI`` / ``CorrelationRatio``).  Consumed by
  ``register._force.MetricForce``: the voxel-count rescale that makes the
  autodiff force match the sum-convention closed forms is meaningful only
  for a spatial mean (see there).
- ``cost(warped, fixed)`` -- the scalar minimisation objective (lower is
  better; e.g. ``1 - lncc``, ``-mutual_information``).
- ``residual(warped, fixed)`` -- the least-squares residual vector, for
  the ``is_least_squares`` members only (others raise).

The metric *math* stays in :mod:`nitrix.metrics`; these records only wrap
the kernels with a minimisation-cost orientation.  ``SSD`` / ``LNCC`` /
``MI`` / ``CorrelationRatio`` are public ``nitrix.register`` exports so a
consumer constructs a spec directly (``RegistrationSpec(metric=LNCC(
radius=2))``).

Boundary-based (BBR) registration is a *sibling objective*, not a
``Metric``: its cost is over boundary-point samples (``cost(T, moving,
surface)``, no ``fixed`` image), so it does not fit
``cost(warped, fixed)`` and will compose the optimiser + transform model
through its own recipe.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Optional, Protocol, cast

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from ..metrics import correlation_ratio, lncc, mutual_information, ssd

__all__ = [
    'Metric',
    'SSD',
    'LNCC',
    'MI',
    'CorrelationRatio',
    'pin_metric_ranges',
]


class Metric(Protocol):
    """Image-pair similarity objective for a registration recipe.

    Attributes
    ----------
    is_least_squares
        Whether the metric admits a residual vector (``residual``) whose
        ``½‖·‖²`` is the cost -- the property that routes the driver to
        the Gauss-Newton / Levenberg-Marquardt path rather than BFGS.
    is_spatial_mean
        Whether the cost reduces by a per-voxel spatial mean (so the
        ``MetricForce`` voxel-count rescale recovers the sum-convention
        closed-form gradient) rather than being a global histogram scalar.
    """

    is_least_squares: ClassVar[bool]
    is_spatial_mean: ClassVar[bool]

    def cost(
        self,
        warped: Float[Array, '*spatial'],
        fixed: Float[Array, '*spatial'],
    ) -> Float[Array, '']:
        """Scalar minimisation cost (lower is a better match)."""
        ...

    def residual(
        self,
        warped: Float[Array, '*spatial'],
        fixed: Float[Array, '*spatial'],
    ) -> Float[Array, ' m']:
        """Least-squares residual vector (``is_least_squares`` only)."""
        ...


@dataclass(frozen=True)
class SSD:
    """Sum-of-squared-differences (within-modality; least-squares).

    The Gauss-Newton / Levenberg-Marquardt path's metric: the residual is
    the raw intensity difference, so the recipe is the ``3dvolreg`` /
    Lucas-Kanade lineage.
    """

    is_least_squares: ClassVar[bool] = True
    is_spatial_mean: ClassVar[bool] = True  # mean-squared-difference

    def cost(
        self,
        warped: Float[Array, '*spatial'],
        fixed: Float[Array, '*spatial'],
    ) -> Float[Array, '']:
        return ssd(warped, fixed)

    def residual(
        self,
        warped: Float[Array, '*spatial'],
        fixed: Float[Array, '*spatial'],
    ) -> Float[Array, ' m']:
        return (warped - fixed).ravel()


@dataclass(frozen=True)
class LNCC:
    """Local (windowed) normalised cross-correlation cost ``1 - lncc``.

    Robust to smooth intensity inhomogeneity; the diffeomorphic-class
    workhorse.  Scalar (BFGS) path.

    Attributes
    ----------
    radius
        Box-window radius (size ``2 * radius + 1`` per axis).
    """

    radius: int = 4
    is_least_squares: ClassVar[bool] = False
    is_spatial_mean: ClassVar[bool] = True  # mean of the local CC over voxels

    def cost(
        self,
        warped: Float[Array, '*spatial'],
        fixed: Float[Array, '*spatial'],
    ) -> Float[Array, '']:
        return 1.0 - lncc(warped, fixed, radius=self.radius)

    def residual(
        self,
        warped: Float[Array, '*spatial'],
        fixed: Float[Array, '*spatial'],
    ) -> Float[Array, ' m']:
        raise NotImplementedError('LNCC is not a least-squares metric.')


@dataclass(frozen=True)
class MI:
    """Mutual-information cost ``-mutual_information`` (cross-modal).

    Scalar (BFGS) path.

    Attributes
    ----------
    bins
        Histogram bins per axis.
    normalized
        Use Studholme's normalised MI.
    range_moving, range_fixed
        Pinned ``(lo, hi)`` intensity ranges.  ``None`` is resolved **once**
        from the full-resolution images by the driver (``pin_metric_ranges``):
        a data ``min/max`` range drifts as the moving image deforms across the
        optimisation (a non-stationary objective), making the soft-histogram
        bin assignment -- and the gradient -- piecewise-unstable (A6).
    """

    bins: int = 32
    normalized: bool = False
    range_moving: Optional[tuple[float, float]] = None
    range_fixed: Optional[tuple[float, float]] = None
    is_least_squares: ClassVar[bool] = False
    is_spatial_mean: ClassVar[bool] = False  # global joint-histogram scalar

    def cost(
        self,
        warped: Float[Array, '*spatial'],
        fixed: Float[Array, '*spatial'],
    ) -> Float[Array, '']:
        return -mutual_information(
            warped,
            fixed,
            bins=self.bins,
            normalized=self.normalized,
            range_moving=self.range_moving,
            range_fixed=self.range_fixed,
        )

    def residual(
        self,
        warped: Float[Array, '*spatial'],
        fixed: Float[Array, '*spatial'],
    ) -> Float[Array, ' m']:
        raise NotImplementedError('MI is not a least-squares metric.')


@dataclass(frozen=True)
class CorrelationRatio:
    """Correlation-ratio cost ``1 - eta**2`` (cross-modal; FSL lineage).

    Scalar (BFGS) path.

    Attributes
    ----------
    bins
        Soft-binning groups for the explanatory (``fixed``) image.
    range_fixed
        Pinned ``(lo, hi)`` range for the explanatory (``fixed``) image's
        binning; ``None`` resolved once by the driver (``pin_metric_ranges``;
        A6).  (CR bins only ``fixed``, so there is no ``range_moving``.)
    """

    bins: int = 32
    range_fixed: Optional[tuple[float, float]] = None
    is_least_squares: ClassVar[bool] = False
    is_spatial_mean: ClassVar[bool] = False  # global histogram statistic

    def cost(
        self,
        warped: Float[Array, '*spatial'],
        fixed: Float[Array, '*spatial'],
    ) -> Float[Array, '']:
        return 1.0 - correlation_ratio(
            warped, fixed, bins=self.bins, range_fixed=self.range_fixed
        )

    def residual(
        self,
        warped: Float[Array, '*spatial'],
        fixed: Float[Array, '*spatial'],
    ) -> Float[Array, ' m']:
        raise NotImplementedError(
            'CorrelationRatio is not a least-squares metric.'
        )


def _pin_range(x: Float[Array, '*spatial']) -> tuple[float, float]:
    """A stationary ``(lo, hi)`` from the full-res data, jit-safe.

    A ``stop_gradient``-ed ``jnp.min``/``jnp.max`` (not ``float()``): under
    ``jax.jit`` the images are tracers, so an eager ``float(tracer)`` cannot run
    -- a traced reduction can.  ``stop_gradient`` keeps the bin edges *constant*
    (the histogram-gradient assumes fixed edges).  Mirrors ``_svf._pin_range``
    (kept local -- ``_svf`` imports ``_force`` which imports this module).
    """
    lo = lax.stop_gradient(jnp.min(x))
    hi = lax.stop_gradient(jnp.max(x))
    return cast('tuple[float, float]', (lo, hi))


def pin_metric_ranges(
    metric: Metric,
    moving: Float[Array, '*spatial'],
    fixed: Float[Array, '*spatial'],
) -> Metric:
    """Pin a histogram metric's intensity ranges from the full-res images.

    The matrix-driver analogue of ``register._svf.pin_force_ranges`` (A6, both
    halves): resolve any ``None`` range on an :class:`MI` / :class:`CorrelationRatio`
    **once** from the full-resolution ``moving`` / ``fixed`` before the pyramid --
    a ``stop_gradient``-ed reduction (``_pin_range``), so the range rides the
    frozen spec as a *constant* (stationary binning -- a data ``min/max`` range
    otherwise drifts as the moving image deforms) and the pin is **jit-safe**:
    ``MI(bins=...)`` / ``CorrelationRatio(bins=...)`` trace with no explicit range
    even under ``jax.jit`` (the eager value is unchanged).  A no-op for a
    least-squares / already-pinned metric.
    """
    if isinstance(metric, MI) and (
        metric.range_moving is None or metric.range_fixed is None
    ):
        rm = metric.range_moving or _pin_range(moving)
        rf = metric.range_fixed or _pin_range(fixed)
        return replace(metric, range_moving=rm, range_fixed=rf)
    if isinstance(metric, CorrelationRatio) and metric.range_fixed is None:
        return replace(metric, range_fixed=_pin_range(fixed))
    return metric
