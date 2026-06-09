# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Region-overlap segmentation metrics.

- ``dice`` -- the soft Sørensen--Dice overlap coefficient: twice the
  intersection over the sum of magnitudes, ``2 |A ∩ B| / (|A| + |B|)``.
- ``jaccard`` -- the soft Jaccard index / intersection-over-union
  (IoU): the intersection over the union,
  ``|A ∩ B| / |A ∪ B|``.

Both are evaluated on soft (probabilistic) masks rather than hard
labels, so they are differentiable and double as losses
(``1 - metric``).  They are monotonically related --
``jaccard = dice / (2 - dice)`` -- so they rank a set of predictions
identically, but report different operating points (Jaccard is the
stricter of the two for any imperfect overlap); pipelines commonly
track both.

Convention (shared with ``intensity`` / ``information``): the function
returns the *similarity* (the coefficient, in ``[0, 1]``); form the
cost as ``1 - metric``.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array, Float

from ._common import Reduction, _reduce

__all__ = ['dice', 'jaccard']

_AxisArg = Union[int, Tuple[int, ...]]


def _overlap_sums(
    pred: Float[Array, '...'],
    target: Float[Array, '...'],
    axis: Optional[_AxisArg],
) -> Tuple[Float[Array, '...'], Float[Array, '...']]:
    """Per-region ``(intersection, magnitude_sum)`` for the overlap metrics.

    ``intersection = Σ p t`` and ``magnitude_sum = Σ p + Σ t`` over
    ``axis``.  The union is ``magnitude_sum - intersection``.
    """
    intersection = jnp.sum(pred * target, axis=axis)
    magnitude_sum = jnp.sum(pred, axis=axis) + jnp.sum(target, axis=axis)
    return intersection, magnitude_sum


def dice(
    pred: Float[Array, '...'],
    target: Float[Array, '...'],
    *,
    axis: Optional[_AxisArg] = None,
    smooth: float = 1e-7,
    reduction: Reduction = 'mean',
) -> Float[Array, '...']:
    """Soft Sørensen--Dice overlap coefficient.

    For soft masks ``p`` (``pred``) and ``t`` (``target``) the
    coefficient over the summed region is

    ``(2 Σ p t + s) / (Σ p + Σ t + s)``

    with ``s = smooth``.  Values lie in ``[0, 1]``; ``1`` is a perfect
    overlap.  Use ``1 - dice`` as a loss.

    The inputs are treated as soft masks and are *not* activated here:
    pass class probabilities (e.g. a softmax) against a one-hot target
    for the multi-class case, or a sigmoid against a ``{0, 1}`` target
    for the binary case.  Keeping the activation in the caller leaves
    this a pure overlap measure.

    Parameters
    ----------
    pred, target
        Soft masks of identical shape.
    axis
        Axes that span a single overlap region (summed in the
        numerator and denominator).  ``None`` (default) sums over the
        whole tensor for one global coefficient; pass the spatial axes
        (e.g. ``axis=(-3, -2, -1)`` on a ``(batch, class, *spatial)``
        volume) for a per-``(batch, class)`` coefficient.
    smooth
        Laplace smoothing added to both numerator and denominator.
        Besides guarding the empty-mask denominator, adding it to the
        numerator makes an empty prediction against an empty target
        score exactly ``1`` (vacuously perfect) rather than ``0/0``.
    reduction
        ``"mean"`` (default), ``"sum"``, or ``"none"`` applied to the
        per-region coefficient array.

    Returns
    -------
    Scalar coefficient (``"mean"`` / ``"sum"``) or the per-region
    coefficient array (``"none"``).
    """
    intersection, magnitude_sum = _overlap_sums(pred, target, axis)
    coeff = (2.0 * intersection + smooth) / (magnitude_sum + smooth)
    return _reduce(coeff, None, reduction)


def jaccard(
    pred: Float[Array, '...'],
    target: Float[Array, '...'],
    *,
    axis: Optional[_AxisArg] = None,
    smooth: float = 1e-7,
    reduction: Reduction = 'mean',
) -> Float[Array, '...']:
    """Soft Jaccard index / intersection-over-union (IoU).

    For soft masks ``p`` (``pred``) and ``t`` (``target``) the index
    over the summed region is

    ``(Σ p t + s) / (Σ p + Σ t - Σ p t + s)``

    -- the intersection over the union, with ``s = smooth``.  Values
    lie in ``[0, 1]``; ``1`` is a perfect overlap.  Use ``1 - jaccard``
    as a loss (the "soft IoU" / Lovász-free IoU loss).

    Like :func:`dice`, the inputs are soft masks and are not activated
    here.  See :func:`dice` for the ``axis`` / ``smooth`` / ``reduction``
    semantics, which are identical; the two differ only in the
    denominator (Dice sums the magnitudes, Jaccard takes their union).

    Returns
    -------
    Scalar coefficient (``"mean"`` / ``"sum"``) or the per-region
    coefficient array (``"none"``).
    """
    intersection, magnitude_sum = _overlap_sums(pred, target, axis)
    union = magnitude_sum - intersection
    coeff = (intersection + smooth) / (union + smooth)
    return _reduce(coeff, None, reduction)
