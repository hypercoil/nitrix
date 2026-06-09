# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Cross-entropy classification / segmentation losses.

- ``bce_with_logits`` -- binary cross-entropy from logits, in the
  numerically stable ``max(x, 0) - x t + log1p(exp(-|x|))`` form (no
  intermediate ``sigmoid`` / ``log`` that overflows for large ``|x|``).
- ``cross_entropy_with_logits`` -- categorical cross-entropy from
  logits against integer class targets (``log_softmax`` + gather).
- ``focal_loss`` -- the Lin et al. focal loss: binary cross-entropy
  down-weighted by ``(1 - p_t) ** gamma`` so easy, well-classified
  examples contribute little, with an optional class-balancing
  ``alpha``.

These are *costs* (minimise), per the loss convention; ``reduction``
selects ``"mean"`` / ``"sum"`` / ``"none"``.  ``focal_loss`` shares the
stable binary-cross-entropy core with ``bce_with_logits``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ._common import Reduction, _reduce

__all__ = [
    'bce_with_logits',
    'cross_entropy_with_logits',
    'focal_loss',
]


def _bce_with_logits(
    logits: Float[Array, '...'],
    targets: Float[Array, '...'],
) -> Float[Array, '...']:
    """Stable elementwise binary cross-entropy from logits.

    ``max(x, 0) - x t + log1p(exp(-|x|))`` is the overflow-free
    rewrite of ``-(t log σ(x) + (1 - t) log(1 - σ(x)))``: the
    ``max`` / ``-|x|`` split keeps every term finite for large
    ``|x|`` (the naive form computes ``log σ(x) -> log 0``).
    """
    return (
        jnp.maximum(logits, 0.0)
        - logits * targets
        + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    )


def bce_with_logits(
    logits: Float[Array, '...'],
    targets: Float[Array, '...'],
    *,
    reduction: Reduction = 'mean',
) -> Float[Array, '...']:
    """Binary cross-entropy from logits (numerically stable).

    Parameters
    ----------
    logits
        Pre-sigmoid scores.
    targets
        Soft or hard targets in ``[0, 1]``, same shape as ``logits``.
    reduction
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Scalar loss (``"mean"`` / ``"sum"``) or the per-element loss
    (``"none"``).
    """
    return _reduce(_bce_with_logits(logits, targets), None, reduction)


def cross_entropy_with_logits(
    logits: Float[Array, '...'],
    target: Int[Array, '...'],
    *,
    axis: int = 1,
    reduction: Reduction = 'mean',
) -> Float[Array, '...']:
    """Categorical cross-entropy from logits against integer targets.

    ``-log softmax(logits)[target]`` along the class ``axis``, via a
    ``log_softmax`` and a gather (so no explicit one-hot is
    materialised).

    Parameters
    ----------
    logits
        Class scores with the class axis at ``axis``.
    target
        Integer class indices, the shape of ``logits`` with the class
        axis removed (e.g. ``(batch, *spatial)`` for
        ``logits`` of ``(batch, class, *spatial)``).
    axis
        The class axis of ``logits``.  Default ``1`` (the
        channel-first ``(batch, class, *spatial)`` convention).
    reduction
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Scalar loss (``"mean"`` / ``"sum"``) or the per-sample loss with
    the class axis removed (``"none"``).
    """
    log_probs = jax.nn.log_softmax(logits, axis=axis)
    picked = jnp.take_along_axis(
        log_probs, jnp.expand_dims(target, axis), axis=axis
    )
    nll = -jnp.squeeze(picked, axis=axis)
    return _reduce(nll, None, reduction)


def focal_loss(
    logits: Float[Array, '...'],
    targets: Float[Array, '...'],
    *,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: Reduction = 'mean',
) -> Float[Array, '...']:
    """Binary focal loss (Lin et al., 2017).

    The stable binary cross-entropy scaled by the focusing factor
    ``(1 - p_t) ** gamma``, where ``p_t`` is the probability assigned
    to the true class.  ``gamma = 0`` recovers (alpha-weighted) BCE;
    larger ``gamma`` more aggressively suppresses the loss of easy,
    confidently-correct examples.

    Parameters
    ----------
    logits
        Pre-sigmoid scores.
    targets
        Targets in ``[0, 1]``, same shape as ``logits``.
    gamma
        Focusing exponent (``>= 0``).  Default ``2``.
    alpha
        Class-balancing weight in ``[0, 1]`` applied to the positive
        class (``1 - alpha`` to the negative).  A negative value
        disables alpha-weighting (the unbalanced focal loss).  Default
        ``0.25``.
    reduction
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Scalar loss (``"mean"`` / ``"sum"``) or the per-element loss
    (``"none"``).
    """
    bce = _bce_with_logits(logits, targets)
    p = jax.nn.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = bce * (1.0 - p_t) ** gamma
    if alpha >= 0.0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    return _reduce(loss, None, reduction)
