# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Classification / segmentation comparison kernels.

*Cross-entropy losses* (costs, minimise; ``reduction`` selects
``"mean"`` / ``"sum"`` / ``"none"``):

- :func:`bce_with_logits` -- binary cross-entropy from logits, in the
  numerically stable :math:`\\max(x, 0) - x t + \\log(1 + e^{-|x|})` form
  (no intermediate ``sigmoid`` / ``log`` that overflows for large
  :math:`|x|`).
- :func:`cross_entropy_with_logits` -- categorical cross-entropy from
  logits against integer class targets (``log_softmax`` + gather).
- :func:`focal_loss` -- the focal loss of Lin et al. (2017): binary
  cross-entropy down-weighted by :math:`(1 - p_t)^{\\gamma}` so easy,
  well-classified examples contribute little, with an optional
  class-balancing :math:`\\alpha`.  Shares the stable
  binary-cross-entropy core with :func:`bce_with_logits`.

*Evaluation metrics* (reporting, not training objectives; **not
differentiable** -- they ride argsort / bincount / top-``k`` operations,
which are piecewise-constant and so carry no useful gradient):

- :func:`roc_auc` -- area under the ROC curve via the rank
  (Mann-Whitney :math:`U`) form, so ties are tie-corrected by average
  ranks.  A **global** statistic: score it on the *gathered*
  ``(scores, labels)``, never as a per-batch mean (averaging AUCs is not
  the AUC).
- :func:`confusion_matrix` -- integer count matrix, ``row = true`` /
  ``col = predicted`` (the sklearn orientation).
- :func:`topk_accuracy` -- fraction of rows whose top-``k`` predictions
  contain the true label.

References
----------
.. [1] Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
   Focal loss for dense object detection. *Proceedings of the IEEE
   International Conference on Computer Vision (ICCV)*, 2980-2988.
   :doi:`10.1109/ICCV.2017.324`
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ._common import Reduction, _reduce

__all__ = [
    'bce_with_logits',
    'cross_entropy_with_logits',
    'focal_loss',
    'roc_auc',
    'confusion_matrix',
    'topk_accuracy',
]

Average = Literal['macro', 'micro', 'none']


def _bce_with_logits(
    logits: Float[Array, '...'],
    targets: Float[Array, '...'],
) -> Float[Array, '...']:
    """Stable elementwise binary cross-entropy from logits.

    :math:`\\max(x, 0) - x t + \\log(1 + e^{-|x|})` is the overflow-free
    rewrite of :math:`-(t \\log \\sigma(x) + (1 - t) \\log(1 - \\sigma(x)))`:
    the :math:`\\max` / :math:`-|x|` split keeps every term finite for
    large :math:`|x|` (the naive form computes
    :math:`\\log \\sigma(x) \\to \\log 0`).

    Parameters
    ----------
    logits : Float[Array, '...']
        Pre-sigmoid scores :math:`x`, of any shape.
    targets : Float[Array, '...']
        Soft or hard targets :math:`t` in :math:`[0, 1]`, broadcastable to
        the shape of ``logits``.

    Returns
    -------
    Float[Array, '...']
        The elementwise binary cross-entropy, of the broadcast shape of
        ``logits`` and ``targets``.
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
    """Binary focal loss of Lin et al. (2017) [1]_.

    The stable binary cross-entropy scaled by the focusing factor
    :math:`(1 - p_t)^{\\gamma}`, where :math:`p_t` is the probability
    assigned to the true class.  ``gamma = 0`` recovers the
    (alpha-weighted) binary cross-entropy; larger ``gamma`` more
    aggressively suppresses the loss of easy, confidently-correct
    examples.

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
    Float[Array, '...']
        Scalar loss (``"mean"`` / ``"sum"``) or the per-element loss with
        the shape of ``logits`` (``"none"``).

    References
    ----------
    .. [1] Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P.
       (2017). Focal loss for dense object detection. *Proceedings of the
       IEEE International Conference on Computer Vision (ICCV)*, 2980-2988.
       :doi:`10.1109/ICCV.2017.324`
    """
    bce = _bce_with_logits(logits, targets)
    p = jax.nn.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = bce * (1.0 - p_t) ** gamma
    if alpha >= 0.0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    return _reduce(loss, None, reduction)


# --- evaluation metrics (reporting; non-differentiable) ------------------


def _binary_auc(
    scores: Float[Array, 'N'],
    positive: Array,
) -> Float[Array, '']:
    """Tie-corrected binary AUROC via the Mann-Whitney :math:`U` rank form.

    :math:`\\mathrm{auc} = (\\sum \\mathrm{rank}_{\\mathrm{pos}} -
    n_{\\mathrm{pos}} (n_{\\mathrm{pos}} + 1) / 2) /
    (n_{\\mathrm{pos}} \\cdot n_{\\mathrm{neg}})` with **average ranks**
    for ties.  The average (1-based) rank of each score is
    :math:`(L + R + 1) / 2`, where :math:`L` / :math:`R` are the counts
    strictly-less / at-most (``searchsorted`` left / right) -- an
    :math:`O(N \\log N)` computation with no threshold sweep.  A degenerate
    input (all of one class) yields ``nan``.

    Parameters
    ----------
    scores : Float[Array, 'N']
        Decision scores for the ``N`` samples.
    positive : Array
        Boolean-coercible mask ``(N,)`` marking the positive-class samples.

    Returns
    -------
    Float[Array, '']
        The scalar area under the ROC curve, or ``nan`` when either class
        is absent (so the denominator vanishes).
    """
    scores = scores.astype(jnp.result_type(scores, jnp.float32))
    positive = positive.astype(jnp.bool_)
    n = scores.shape[0]
    ordered = jnp.sort(scores)
    left = jnp.searchsorted(ordered, scores, side='left')
    right = jnp.searchsorted(ordered, scores, side='right')
    ranks = (left + right + 1).astype(scores.dtype) * 0.5
    n_pos = jnp.count_nonzero(positive).astype(scores.dtype)
    n_neg = jnp.asarray(n, scores.dtype) - n_pos
    sum_rank_pos = jnp.sum(jnp.where(positive, ranks, 0.0))
    denom = n_pos * n_neg
    auc = (sum_rank_pos - n_pos * (n_pos + 1.0) * 0.5) / denom
    return jnp.where(denom > 0, auc, jnp.nan)


def roc_auc(
    scores: Float[Array, 'N'] | Float[Array, 'N C'],
    labels: Int[Array, 'N'],
    *,
    average: Average = 'macro',
) -> Float[Array, ''] | Float[Array, 'C']:
    """Area under the ROC curve (rank / Mann-Whitney form, tie-corrected).

    1-D ``scores`` is the binary case (positive class = label ``1``).  2-D
    ``scores`` ``(N, C)`` is multiclass one-vs-rest: column ``c`` scores
    "class ``c`` vs the rest" against ``labels == c``.

    Parameters
    ----------
    scores
        Decision scores: ``(N,)`` (binary) or ``(N, C)`` (per-class, OvR).
    labels
        Integer labels ``(N,)``: ``{0, 1}`` binary, ``{0, ..., C-1}`` multiclass.
    average
        Multiclass pooling of the per-class OvR AUCs (ignored for 1-D scores):
        ``'macro'`` (unweighted mean, default), ``'micro'`` (one AUC over the
        pooled one-hot scores/labels), or ``'none'`` (the per-class ``(C,)``
        vector).

    Returns
    -------
    Scalar AUROC (binary, or ``'macro'`` / ``'micro'``) or the per-class
    ``(C,)`` vector (``'none'``).  A class with no positives or no negatives
    yields ``nan`` (``'macro'`` then propagates ``nan`` -- the degeneracy is
    surfaced, not silently skipped).

    Notes
    -----
    **Global statistic, not batch-meanable.**  Score on the gathered
    ``(scores, labels)`` accumulated across all batches; a mean of
    per-batch AUCs is not the AUC.  Non-differentiable (rank-based).
    """
    scores = jnp.asarray(scores)
    labels = jnp.asarray(labels)
    if scores.ndim == 1:
        return _binary_auc(scores, labels == 1)
    if scores.ndim != 2:
        raise ValueError(
            f'roc_auc: scores must be 1-D (binary) or 2-D (N, C); got '
            f'ndim {scores.ndim}.'
        )
    num_classes = scores.shape[1]
    onehot = labels[:, None] == jnp.arange(num_classes)
    if average == 'micro':
        return _binary_auc(scores.reshape(-1), onehot.reshape(-1))
    per_class = jax.vmap(_binary_auc, in_axes=(1, 1))(scores, onehot)
    if average == 'none':
        return per_class
    if average == 'macro':
        return jnp.mean(per_class)
    raise ValueError(
        f"roc_auc: average must be 'macro' / 'micro' / 'none'; got {average!r}."
    )


def confusion_matrix(
    pred: Int[Array, 'N'],
    target: Int[Array, 'N'],
    *,
    num_classes: int,
) -> Int[Array, 'C C']:
    """Confusion matrix, ``row = true`` / ``col = predicted`` (sklearn form).

    ``cm[i, j]`` counts samples of true class ``i`` predicted as class ``j``,
    via ``bincount(target * C + pred)``.  ``num_classes`` is static (it fixes
    the ``(C, C)`` output shape, so the op is ``jit``-clean).  Non-differentiable
    (integer counts).

    Parameters
    ----------
    pred, target
        Integer predicted / true class indices ``(N,)`` in ``[0, num_classes)``.
    num_classes
        Number of classes ``C`` (static).

    Returns
    -------
    Int[Array, 'C C']
        The ``(C, C)`` integer count matrix, where entry ``[i, j]`` is the
        number of samples of true class ``i`` predicted as class ``j``.
    """
    pred = jnp.asarray(pred).reshape(-1)
    target = jnp.asarray(target).reshape(-1)
    flat = jnp.bincount(
        target * num_classes + pred, length=num_classes * num_classes
    )
    return flat.reshape(num_classes, num_classes)


def topk_accuracy(
    logits: Float[Array, 'N C'],
    target: Int[Array, 'N'],
    *,
    k: int = 1,
    reduction: Reduction = 'mean',
) -> Float[Array, ''] | Float[Array, 'N']:
    """Top-``k`` accuracy: fraction of rows whose top-``k`` logits hit the label.

    Parameters
    ----------
    logits
        Class scores ``(N, C)`` (the top-``k`` are taken along the class axis).
    target
        True class indices ``(N,)`` in ``[0, C)``.
    k
        Number of top predictions to consider (static).  Default ``1``.
    reduction
        ``"mean"`` (default; the accuracy), ``"sum"`` (the hit count), or
        ``"none"`` (the per-row ``0/1`` hit).

    Returns
    -------
    Scalar accuracy / count, or the per-row hit vector.  Non-differentiable
    (``top_k`` is an arg-selection).
    """
    logits = jnp.asarray(logits)
    target = jnp.asarray(target)
    _, topk_idx = jax.lax.top_k(logits, k)  # (N, k) class indices
    hit = jnp.any(topk_idx == target[:, None], axis=-1)
    return _reduce(
        hit.astype(jnp.result_type(logits, jnp.float32)), None, reduction
    )
