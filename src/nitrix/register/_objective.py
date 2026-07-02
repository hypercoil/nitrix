# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Registration objective over the transform parameters.

The thing a recipe's optimiser minimises: a scalar (or least-squares
vector) function of the transform's Lie parameters :math:`\theta`.  This is
the generalisation the :class:`Metric <._metric.Metric>` abstraction
foreshadowed -- a metric scores an *image pair* (``cost(warped, fixed)``)
and only becomes an objective once the driver pairs it with a warp, whereas
a boundary-based objective (BBR) scores :math:`\theta` directly over
surface-point samples with no ``fixed`` image.  Both are objectives, mapping
:math:`\theta \mapsto \mathrm{cost}` while closing over their own data.

Two implementers:

- :class:`MetricObjective` -- an image-pair metric plus a warp closure plus
  the ``fixed`` image, so that ``cost(θ) = metric.cost(warp(θ), fixed)``.
  This is what the coarse-to-fine matrix driver builds per level, so the
  optimiser dispatch (Gauss-Newton / Levenberg-Marquardt for a least-squares
  metric, BFGS otherwise) is written once against the objective protocol.
- :class:`BoundaryObjective <._bbr.BoundaryObjective>` -- the Greve-Fischl
  boundary cost (a *sibling* objective: no ``fixed`` image), which reuses the
  same optimiser dispatch.

Holding the objective behind a protocol keeps a new objective (a MIND /
self-similarity descriptor, BBR, ...) a single record rather than another
branch in the driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, runtime_checkable

from jaxtyping import Array, Float

from ._metric import Metric

__all__ = ['Objective', 'MetricObjective']


@runtime_checkable
class Objective(Protocol):
    r"""Scalar / least-squares objective over the transform parameters.

    Attributes
    ----------
    is_least_squares
        Whether the objective admits a residual vector whose
        :math:`\tfrac{1}{2}\lVert \cdot \rVert^{2}` is the cost -- the
        property that routes the optimiser to the Gauss-Newton /
        Levenberg-Marquardt path rather than BFGS.
    """

    @property
    def is_least_squares(self) -> bool: ...

    def cost(self, params: Float[Array, ' p']) -> Float[Array, '']:
        """Scalar minimisation cost at ``params`` (lower is better).

        Parameters
        ----------
        params : Float[Array, 'p']
            The transform's Lie parameter vector at which the objective is
            evaluated.

        Returns
        -------
        Float[Array, '']
            The scalar cost; the optimiser drives this downwards.
        """
        ...

    def residual(self, params: Float[Array, ' p']) -> Float[Array, ' m']:
        """Least-squares residual at ``params``.

        Defined only when :attr:`is_least_squares` is ``True``, in which case
        the cost equals half the squared Euclidean norm of this residual.

        Parameters
        ----------
        params : Float[Array, 'p']
            The transform's Lie parameter vector at which the residual is
            evaluated.

        Returns
        -------
        Float[Array, 'm']
            The residual vector whose half squared norm is the cost.
        """
        ...


@dataclass(frozen=True)
class MetricObjective:
    """An image-pair :class:`Metric <._metric.Metric>` as an :class:`Objective`.

    Pairs a metric with the level's warp closure (``params -> warped``)
    and the ``fixed`` image, so ``cost(params) = metric.cost(warp(params),
    fixed)`` -- the form the coarse-to-fine driver optimises.

    Attributes
    ----------
    metric : Metric
        The image-pair similarity metric scored between the warped moving
        image and the ``fixed`` image.
    warp : Callable[[Float[Array, 'p']], Float[Array, '*spatial']]
        Closure mapping a parameter vector to the warped moving image on the
        level's grid.
    fixed : Float[Array, '*spatial']
        The reference image on the level's grid.
    mask : Float[Array, '*spatial'] or None, optional
        Optional per-voxel weight on the ``fixed`` grid (the level's
        resolution) threaded into the metric so the cost is computed over a
        region; ``None`` gives the full-image cost.
    """

    metric: Metric
    warp: Callable[[Float[Array, ' p']], Float[Array, '*spatial']]
    fixed: Float[Array, '*spatial']
    mask: Optional[Float[Array, '*spatial']] = None

    @property
    def is_least_squares(self) -> bool:
        return self.metric.is_least_squares

    def cost(self, params: Float[Array, ' p']) -> Float[Array, '']:
        """Scalar cost of the warped moving image against ``fixed``.

        Warps the moving image with ``warp(params)`` and scores it against the
        ``fixed`` image under ``metric``, restricted by ``mask`` where given.

        Parameters
        ----------
        params : Float[Array, 'p']
            The transform's Lie parameter vector.

        Returns
        -------
        Float[Array, '']
            The scalar similarity cost at ``params`` (lower is better).
        """
        return self.metric.cost(self.warp(params), self.fixed, mask=self.mask)

    def residual(self, params: Float[Array, ' p']) -> Float[Array, ' m']:
        """Least-squares residual of the warped image against ``fixed``.

        Warps the moving image with ``warp(params)`` and returns the metric's
        residual vector against the ``fixed`` image, restricted by ``mask``
        where given. Defined only when the metric is least-squares.

        Parameters
        ----------
        params : Float[Array, 'p']
            The transform's Lie parameter vector.

        Returns
        -------
        Float[Array, 'm']
            The residual vector whose half squared norm is the cost.
        """
        return self.metric.residual(
            self.warp(params), self.fixed, mask=self.mask
        )
