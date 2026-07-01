# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Registration similarity metrics as an algebraic data type.

Each metric is the objective a matrix-transform recipe minimises over the
transform parameters.  A metric is an immutable ``@dataclass(frozen=True)``
record carrying its own hyper-parameters (window radius, histogram bins),
following the :class:`~nitrix.geometry.Interpolator` precedent, so that the
:class:`~nitrix.register.RegistrationSpec` config need not flatten every
metric's knobs into a single record and the driver dispatches on a method
rather than a string ``if``-chain.

A metric exposes:

- ``is_least_squares`` (a ``ClassVar``) -- whether it admits a vector
  residual whose half-sum-of-squares is the cost.  ``True`` routes the
  driver to the Gauss-Newton / Levenberg-Marquardt least-squares path;
  ``False`` to the scalar BFGS path.
- ``is_spatial_mean`` (a ``ClassVar``) -- whether the scalar cost is a
  per-voxel spatial *mean* (:class:`SSD` / :class:`LNCC`) rather than a
  global histogram statistic (:class:`MI` / :class:`CorrelationRatio`).
  Consumed by :class:`~nitrix.register.MetricForce`: the voxel-count
  rescale that makes the autodiff force match the sum-convention closed
  forms is meaningful only for a spatial mean.
- ``cost(warped, fixed)`` -- the scalar minimisation objective (lower is
  better; for example :math:`1 - \\mathrm{lncc}`, :math:`-I` for mutual
  information :math:`I`).
- ``residual(warped, fixed)`` -- the least-squares residual vector, for
  the ``is_least_squares`` members only (others raise).

The metric *math* stays in :mod:`nitrix.metrics`; these records only wrap
the kernels with a minimisation-cost orientation.  :class:`SSD` /
:class:`LNCC` / :class:`MI` / :class:`CorrelationRatio` are public
``nitrix.register`` exports so a consumer constructs a spec directly
(``RegistrationSpec(metric=LNCC(radius=2))``).

Boundary-based (BBR) registration is a *sibling objective*, not a
:class:`Metric`: its cost is over boundary-point samples
(``cost(T, moving, surface)``, no ``fixed`` image), so it does not fit
``cost(warped, fixed)`` and composes the optimiser and transform model
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

    A metric bundles the scalar minimisation cost with an optional
    least-squares residual and the two class flags a coarse-to-fine driver
    keys its dispatch on.

    Attributes
    ----------
    is_least_squares
        Whether the metric admits a residual vector (:meth:`residual`)
        whose half-squared-norm :math:`\\tfrac{1}{2}\\lVert\\cdot\\rVert^2`
        is the cost -- the property that routes the driver to the
        Gauss-Newton / Levenberg-Marquardt path rather than BFGS.
    is_spatial_mean
        Whether the cost reduces by a per-voxel spatial mean (so the
        :class:`~nitrix.register.MetricForce` voxel-count rescale recovers
        the sum-convention closed-form gradient) rather than being a
        global histogram scalar.
    """

    is_least_squares: ClassVar[bool]
    is_spatial_mean: ClassVar[bool]

    def cost(
        self,
        warped: Float[Array, '*spatial'],
        fixed: Float[Array, '*spatial'],
        *,
        mask: Optional[Float[Array, '*spatial']] = None,
    ) -> Float[Array, '']:
        """Scalar minimisation cost (lower is a better match).

        Parameters
        ----------
        warped : Float[Array, '*spatial']
            The moving image after it has been warped onto the fixed grid.
        fixed : Float[Array, '*spatial']
            The reference (fixed) image, on the same grid as ``warped``.
        mask : Float[Array, '*spatial'], optional
            Non-negative per-voxel weight on the ``fixed`` grid restricting
            the cost to a region: out-of-mask voxels are ignored.  For a
            histogram metric (:class:`MI` / :class:`CorrelationRatio`) it
            gates the scatter (excludes the voxel from the joint
            distribution), not merely the reduction.

        Returns
        -------
        Float[Array, '']
            The scalar cost; lower values indicate a better match.
        """
        ...

    def residual(
        self,
        warped: Float[Array, '*spatial'],
        fixed: Float[Array, '*spatial'],
        *,
        mask: Optional[Float[Array, '*spatial']] = None,
    ) -> Float[Array, ' m']:
        """Least-squares residual vector (``is_least_squares`` metrics only).

        Non-least-squares metrics raise ``NotImplementedError``.

        Parameters
        ----------
        warped : Float[Array, '*spatial']
            The moving image after it has been warped onto the fixed grid.
        fixed : Float[Array, '*spatial']
            The reference (fixed) image, on the same grid as ``warped``.
        mask : Float[Array, '*spatial'], optional
            Non-negative per-voxel weight on the ``fixed`` grid.  When
            given, each residual entry is weighted by :math:`\\sqrt{mask}`
            so that the half-squared-norm
            :math:`\\tfrac{1}{2}\\lVert r\\rVert^2` of the residual
            :math:`r` is the masked sum-of-squares (an out-of-mask voxel
            contributes a zero row, dropping out of the Gauss-Newton normal
            equations).

        Returns
        -------
        Float[Array, ' m']
            The flattened residual vector of length ``m`` (the number of
            voxels), whose half-squared-norm equals the cost.
        """
        ...


@dataclass(frozen=True)
class SSD:
    """Squared-difference metric, SSD (within-modality; least-squares).

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
        *,
        mask: Optional[Float[Array, '*spatial']] = None,
    ) -> Float[Array, '']:
        """Mean squared-difference cost between ``warped`` and ``fixed``.

        Parameters
        ----------
        warped : Float[Array, '*spatial']
            The moving image after it has been warped onto the fixed grid.
        fixed : Float[Array, '*spatial']
            The reference (fixed) image, on the same grid as ``warped``.
        mask : Float[Array, '*spatial'], optional
            Non-negative per-voxel weight restricting the cost to a region.

        Returns
        -------
        Float[Array, '']
            The scalar mean squared-difference cost.
        """
        return ssd(warped, fixed, mask=mask)

    def residual(
        self,
        warped: Float[Array, '*spatial'],
        fixed: Float[Array, '*spatial'],
        *,
        mask: Optional[Float[Array, '*spatial']] = None,
    ) -> Float[Array, ' m']:
        """Raw intensity-difference residual for the least-squares path.

        Parameters
        ----------
        warped : Float[Array, '*spatial']
            The moving image after it has been warped onto the fixed grid.
        fixed : Float[Array, '*spatial']
            The reference (fixed) image, on the same grid as ``warped``.
        mask : Float[Array, '*spatial'], optional
            Non-negative per-voxel weight; each residual entry is scaled by
            :math:`\\sqrt{mask}` so its half-squared-norm is the masked
            sum-of-squares.

        Returns
        -------
        Float[Array, ' m']
            The flattened per-voxel difference ``warped - fixed`` (weighted
            by :math:`\\sqrt{mask}` when a mask is given).
        """
        diff = warped - fixed
        if mask is not None:
            diff = diff * jnp.sqrt(mask)
        return diff.ravel()


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
        *,
        mask: Optional[Float[Array, '*spatial']] = None,
    ) -> Float[Array, '']:
        """Local normalised cross-correlation cost :math:`1 - \\mathrm{lncc}`.

        Parameters
        ----------
        warped : Float[Array, '*spatial']
            The moving image after it has been warped onto the fixed grid.
        fixed : Float[Array, '*spatial']
            The reference (fixed) image, on the same grid as ``warped``.
        mask : Float[Array, '*spatial'], optional
            Non-negative per-voxel weight restricting the cost to a region.

        Returns
        -------
        Float[Array, '']
            The scalar cost :math:`1 - \\mathrm{lncc}` over the box window
            of radius ``self.radius``; lower is a better match.
        """
        return 1.0 - lncc(warped, fixed, radius=self.radius, mask=mask)

    def residual(
        self,
        warped: Float[Array, '*spatial'],
        fixed: Float[Array, '*spatial'],
        *,
        mask: Optional[Float[Array, '*spatial']] = None,
    ) -> Float[Array, ' m']:
        """Not defined: :class:`LNCC` is not a least-squares metric.

        Raises
        ------
        NotImplementedError
            Always; local normalised cross-correlation has no
            half-sum-of-squares residual form and uses the scalar path.
        """
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
        from the full-resolution images by the driver (see
        :func:`pin_metric_ranges`): a data ``min``/``max`` range drifts as
        the moving image deforms across the optimisation (a non-stationary
        objective), making the soft-histogram bin assignment -- and the
        gradient -- piecewise-unstable.
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
        *,
        mask: Optional[Float[Array, '*spatial']] = None,
    ) -> Float[Array, '']:
        """Mutual-information cost :math:`-I` (negated for minimisation).

        Parameters
        ----------
        warped : Float[Array, '*spatial']
            The moving image after it has been warped onto the fixed grid.
        fixed : Float[Array, '*spatial']
            The reference (fixed) image, on the same grid as ``warped``.
        mask : Float[Array, '*spatial'], optional
            Non-negative per-voxel weight; for this histogram metric it
            gates the joint-histogram scatter (excludes the voxel from the
            joint distribution), not merely the reduction.

        Returns
        -------
        Float[Array, '']
            The negated mutual information :math:`-I` (Studholme's
            normalised variant when ``self.normalized``), computed with
            ``self.bins`` bins per axis over the pinned intensity ranges;
            lower is a better match.
        """
        return -mutual_information(
            warped,
            fixed,
            bins=self.bins,
            normalized=self.normalized,
            range_moving=self.range_moving,
            range_fixed=self.range_fixed,
            mask=mask,
        )

    def residual(
        self,
        warped: Float[Array, '*spatial'],
        fixed: Float[Array, '*spatial'],
        *,
        mask: Optional[Float[Array, '*spatial']] = None,
    ) -> Float[Array, ' m']:
        """Not defined: :class:`MI` is not a least-squares metric.

        Raises
        ------
        NotImplementedError
            Always; mutual information is a global histogram statistic with
            no half-sum-of-squares residual form and uses the scalar path.
        """
        raise NotImplementedError('MI is not a least-squares metric.')


@dataclass(frozen=True)
class CorrelationRatio:
    """Correlation-ratio cost :math:`1 - \\eta^2` (cross-modal; FSL lineage).

    Scalar (BFGS) path.

    Attributes
    ----------
    bins
        Soft-binning groups for the explanatory (``fixed``) image.
    range_fixed
        Pinned ``(lo, hi)`` range for the explanatory (``fixed``) image's
        binning; ``None`` is resolved once by the driver (see
        :func:`pin_metric_ranges`).  The correlation ratio bins only
        ``fixed``, so there is no ``range_moving``.
    """

    bins: int = 32
    range_fixed: Optional[tuple[float, float]] = None
    is_least_squares: ClassVar[bool] = False
    is_spatial_mean: ClassVar[bool] = False  # global histogram statistic

    def cost(
        self,
        warped: Float[Array, '*spatial'],
        fixed: Float[Array, '*spatial'],
        *,
        mask: Optional[Float[Array, '*spatial']] = None,
    ) -> Float[Array, '']:
        """Correlation-ratio cost :math:`1 - \\eta^2`.

        Parameters
        ----------
        warped : Float[Array, '*spatial']
            The moving image after it has been warped onto the fixed grid.
        fixed : Float[Array, '*spatial']
            The reference (fixed, explanatory) image, on the same grid as
            ``warped``; only this image is binned.
        mask : Float[Array, '*spatial'], optional
            Non-negative per-voxel weight; for this histogram metric it
            gates which voxels enter the binned statistic.

        Returns
        -------
        Float[Array, '']
            The scalar cost :math:`1 - \\eta^2`, with ``self.bins``
            soft-binning groups over the pinned ``fixed`` range; lower is a
            better match.
        """
        return 1.0 - correlation_ratio(
            warped,
            fixed,
            bins=self.bins,
            range_fixed=self.range_fixed,
            mask=mask,
        )

    def residual(
        self,
        warped: Float[Array, '*spatial'],
        fixed: Float[Array, '*spatial'],
        *,
        mask: Optional[Float[Array, '*spatial']] = None,
    ) -> Float[Array, ' m']:
        """Not defined: :class:`CorrelationRatio` is not least-squares.

        Raises
        ------
        NotImplementedError
            Always; the correlation ratio is a global histogram statistic
            with no half-sum-of-squares residual form and uses the scalar
            path.
        """
        raise NotImplementedError(
            'CorrelationRatio is not a least-squares metric.'
        )


def _pin_range(x: Float[Array, '*spatial']) -> tuple[float, float]:
    """A stationary ``(lo, hi)`` intensity range from the data, jit-safe.

    Reduces ``x`` to a ``stop_gradient``-ed ``jnp.min``/``jnp.max`` pair
    rather than an eager ``float()``: under ``jax.jit`` the images are
    tracers, so ``float(tracer)`` cannot run whereas a traced reduction
    can.  The ``stop_gradient`` keeps the bin edges *constant* (the
    histogram gradient assumes fixed edges).  Mirrors ``_svf._pin_range``
    (kept local because ``_svf`` imports ``_force``, which imports this
    module).

    Parameters
    ----------
    x : Float[Array, '*spatial']
        The image whose intensity extremes define the range.

    Returns
    -------
    tuple of float
        The ``(lo, hi)`` pair -- the (gradient-stopped) minimum and maximum
        of ``x`` -- carried as constant bin-edge bounds.
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

    The matrix-driver analogue of ``register._svf.pin_force_ranges``:
    resolve any ``None`` range on an :class:`MI` / :class:`CorrelationRatio`
    **once** from the full-resolution ``moving`` / ``fixed`` before the
    pyramid, via a ``stop_gradient``-ed reduction (:func:`_pin_range`), so
    the range rides the frozen spec as a *constant* (stationary binning --
    a data ``min``/``max`` range otherwise drifts as the moving image
    deforms) and the pin is **jit-safe**: ``MI(bins=...)`` /
    ``CorrelationRatio(bins=...)`` trace with no explicit range even under
    ``jax.jit`` (the eager value is unchanged).  This is a no-op for a
    least-squares or already-pinned metric.

    Parameters
    ----------
    metric : Metric
        The metric to pin.  Only :class:`MI` (with either range unset) and
        :class:`CorrelationRatio` (with ``range_fixed`` unset) are affected.
    moving : Float[Array, '*spatial']
        The full-resolution moving image, source of ``range_moving`` for
        :class:`MI`.
    fixed : Float[Array, '*spatial']
        The full-resolution fixed image, source of ``range_fixed`` for both
        :class:`MI` and :class:`CorrelationRatio`.

    Returns
    -------
    Metric
        The metric with its previously-``None`` ranges filled in from the
        images; the input metric unchanged if nothing needed pinning.
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
