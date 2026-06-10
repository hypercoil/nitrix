# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Registration objective over the transform parameters.

The thing a recipe's optimiser minimises: a scalar (or least-squares
vector) function of the transform's Lie parameters ``θ``.  This is the
generalisation the ``Metric`` ADT foreshadowed -- a ``Metric`` scores an
*image pair* (``cost(warped, fixed)``) and only becomes an objective once
the driver pairs it with a warp, whereas a **boundary-based** objective
(BBR) scores ``θ`` directly over surface-point samples with no ``fixed``
image.  Both are ``Objective``s: ``θ ↦ cost``, closing over their own data.

Two implementers:

- :class:`MetricObjective` -- an image-pair ``Metric`` + a warp closure +
  the ``fixed`` image; ``cost(θ) = metric.cost(warp(θ), fixed)``.  This is
  what the coarse-to-fine matrix driver builds per level, so the optimiser
  dispatch (Gauss-Newton / Levenberg-Marquardt for a least-squares metric,
  BFGS otherwise) is written once against ``Objective``.
- :class:`._bbr.BoundaryObjective` -- the Greve-Fischl boundary cost (a
  *sibling* objective: no ``fixed`` image), which reuses the same optimiser
  dispatch.

Holding the objective behind a protocol keeps a new objective (a MIND /
self-similarity descriptor, BBR, ...) a single record rather than another
branch in the driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from jaxtyping import Array, Float

from ._metric import Metric

__all__ = ['Objective', 'MetricObjective']


@runtime_checkable
class Objective(Protocol):
    """Scalar / least-squares objective over the transform parameters.

    Attributes
    ----------
    is_least_squares
        Whether the objective admits a residual vector whose ``½‖·‖²`` is
        the cost -- the property that routes the optimiser to the
        Gauss-Newton / Levenberg-Marquardt path rather than BFGS.
    """

    @property
    def is_least_squares(self) -> bool: ...

    def cost(self, params: Float[Array, ' p']) -> Float[Array, '']:
        """Scalar minimisation cost at ``params`` (lower is better)."""
        ...

    def residual(self, params: Float[Array, ' p']) -> Float[Array, ' m']:
        """Least-squares residual at ``params`` (``is_least_squares`` only)."""
        ...


@dataclass(frozen=True)
class MetricObjective:
    """An image-pair ``Metric`` as an :class:`Objective`.

    Pairs a metric with the level's warp closure (``params -> warped``)
    and the ``fixed`` image, so ``cost(params) = metric.cost(warp(params),
    fixed)`` -- the form the coarse-to-fine driver optimises.
    """

    metric: Metric
    warp: Callable[[Float[Array, ' p']], Float[Array, '*spatial']]
    fixed: Float[Array, '*spatial']

    @property
    def is_least_squares(self) -> bool:
        return self.metric.is_least_squares

    def cost(self, params: Float[Array, ' p']) -> Float[Array, '']:
        return self.metric.cost(self.warp(params), self.fixed)

    def residual(self, params: Float[Array, ' p']) -> Float[Array, ' m']:
        return self.metric.residual(self.warp(params), self.fixed)
