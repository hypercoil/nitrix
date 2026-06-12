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

from dataclasses import dataclass
from typing import ClassVar, Protocol

from jaxtyping import Array, Float

from ..metrics import correlation_ratio, lncc, mutual_information, ssd

__all__ = [
    'Metric',
    'SSD',
    'LNCC',
    'MI',
    'CorrelationRatio',
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
    """

    bins: int = 32
    normalized: bool = False
    is_least_squares: ClassVar[bool] = False
    is_spatial_mean: ClassVar[bool] = False  # global joint-histogram scalar

    def cost(
        self,
        warped: Float[Array, '*spatial'],
        fixed: Float[Array, '*spatial'],
    ) -> Float[Array, '']:
        return -mutual_information(
            warped, fixed, bins=self.bins, normalized=self.normalized
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
    """

    bins: int = 32
    is_least_squares: ClassVar[bool] = False
    is_spatial_mean: ClassVar[bool] = False  # global histogram statistic

    def cost(
        self,
        warped: Float[Array, '*spatial'],
        fixed: Float[Array, '*spatial'],
    ) -> Float[Array, '']:
        return 1.0 - correlation_ratio(warped, fixed, bins=self.bins)

    def residual(
        self,
        warped: Float[Array, '*spatial'],
        fixed: Float[Array, '*spatial'],
    ) -> Float[Array, ' m']:
        raise NotImplementedError(
            'CorrelationRatio is not a least-squares metric.'
        )
