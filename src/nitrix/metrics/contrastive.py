# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Contrastive / self-supervised representation kernels.

- ``info_nce`` -- the InfoNCE / NT-Xent objective between two aligned view
  batches: pull the matched pair together and push every other sample apart
  on the unit sphere.
- ``dino_cross_entropy`` -- the self-distillation cross-entropy between a
  sharpened, centred teacher distribution (stop-gradient) and a student.
- ``ibot_cross_entropy`` -- the masked-token variant of the above, averaged
  over the masked positions of each sample.
- ``koleo`` -- the Kozachenko--Leonenko differential-entropy regulariser:
  spread features out by penalising small nearest-neighbour distances.

Per ``SPEC_UPDATE_v0.5 §1`` these are *score kernels*: they take the
**objective structure** (the view pairing, the masked-token selection, the
teacher ``center`` and its EMA) as explicit arguments rather than baking a
recipe in, and they reduce through the shared leaf reduction. The EMA /
centre bookkeeping and the view-stacking layout are recipes and stay with
the nimox / ilex caller.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from .._internal.reductions import Reduction, reduce
from ..numerics import l2_normalize

__all__ = [
    'info_nce',
    'dino_cross_entropy',
    'ibot_cross_entropy',
    'koleo',
]


def info_nce(
    za: Float[Array, 'n d'],
    zb: Float[Array, 'n d'],
    *,
    temperature: float = 0.5,
    reduction: Reduction = 'mean',
) -> Float[Array, '...']:
    """InfoNCE / NT-Xent loss between two aligned view batches.

    ``za[i]`` and ``zb[i]`` are the two views of sample ``i`` (the positive
    pair); all other samples are negatives.  Both batches are L2-normalised,
    and the scaled cross-view cosine-similarity ``za zbᵀ / τ`` is the logits:
    row ``i`` is classified against the ``n`` keys in ``zb`` with the true
    class ``i`` (and symmetrically for ``zb`` against ``za``).  The loss is
    the average of the two directional cross-entropies.

    This is the **layout-agnostic** core: the caller supplies the two view
    batches, so there is no baked pair-index convention.  Because the
    similarity is *cross-view*, a sample is never compared with itself --
    there are no self-pairs to mask, so the loss has no self-similarity bias
    (unlike a single stacked ``2N`` matrix).  A SimCLR-style stacked layout,
    or one with within-view negatives, is recovered by the caller arranging
    the views.

    Parameters
    ----------
    za, zb
        ``(n, d)`` embeddings; ``za[i]``/``zb[i]`` are a positive pair.
    temperature
        Softmax temperature ``τ``.
    reduction
        ``"mean"`` (default), ``"sum"``, or ``"none"`` (per-sample loss).
    """
    za = l2_normalize(za, axis=-1)
    zb = l2_normalize(zb, axis=-1)
    logits = (za @ zb.T) / temperature  # (n, n)
    n = za.shape[0]
    idx = jnp.arange(n)
    log_ab = jax.nn.log_softmax(logits, axis=-1)
    log_ba = jax.nn.log_softmax(logits.T, axis=-1)
    loss = -0.5 * (log_ab[idx, idx] + log_ba[idx, idx])
    return reduce(loss, reduction=reduction)


def _distill_ce(
    student_logits: Float[Array, '... k'],
    teacher_logits: Float[Array, '... k'],
    center: Float[Array, 'k'],
    *,
    student_temp: float,
    teacher_temp: float,
) -> Float[Array, '...']:
    """Centred / sharpened teacher cross-entropy, per sample (unreduced).

    ``H(softmax((teacher - center)/τ_t)  [stop-grad],  log_softmax(student/τ_s))``
    summed over the ``k`` prototypes.  The shared core of
    :func:`dino_cross_entropy` and :func:`ibot_cross_entropy`.
    """
    teacher = jax.nn.softmax((teacher_logits - center) / teacher_temp, axis=-1)
    teacher = jax.lax.stop_gradient(teacher)
    student = jax.nn.log_softmax(student_logits / student_temp, axis=-1)
    return -jnp.sum(teacher * student, axis=-1)


def dino_cross_entropy(
    student_logits: Float[Array, '... k'],
    teacher_logits: Float[Array, '... k'],
    center: Float[Array, 'k'],
    *,
    student_temp: float = 0.1,
    teacher_temp: float = 0.04,
    reduction: Reduction = 'mean',
) -> Float[Array, '...']:
    """Self-distillation cross-entropy (DINO).

    The teacher distribution is centred and sharpened
    (``softmax((teacher - center) / teacher_temp)``) and detached; the loss
    is its cross-entropy against the student's
    ``log_softmax(student / student_temp)``, summed over the ``k`` prototypes
    and reduced over the batch.  Centring (subtracting a running mean) plus a
    low teacher temperature is what prevents collapse.

    Parameters
    ----------
    student_logits, teacher_logits
        Prototype scores ``(..., k)``.
    center
        Teacher centring vector ``(k,)`` (maintained upstream).
    student_temp, teacher_temp
        Softmax temperatures (teacher sharper, i.e. smaller).
    reduction
        ``"mean"`` (default), ``"sum"``, or ``"none"`` (per-sample CE).
    """
    ce = _distill_ce(
        student_logits,
        teacher_logits,
        center,
        student_temp=student_temp,
        teacher_temp=teacher_temp,
    )
    return reduce(ce, reduction=reduction)


def ibot_cross_entropy(
    student_logits: Float[Array, 'b t k'],
    teacher_logits: Float[Array, 'b t k'],
    center: Float[Array, 'k'],
    mask: Bool[Array, 'b t'],
    *,
    student_temp: float = 0.1,
    teacher_temp: float = 0.04,
    reduction: Reduction = 'mean',
) -> Float[Array, '...']:
    """Masked-token self-distillation cross-entropy (iBOT).

    Exactly :func:`dino_cross_entropy`'s centred/sharpened cross-entropy
    evaluated per patch token, then **domain-mask reduced** over the masked
    tokens of each sample: the per-sample score is the
    ``Σ(mask·ce)/Σ(mask)`` weighted mean over the token axis (the masked
    token is the measurement domain), and the result is reduced over the
    batch.  An all-unmasked sample contributes 0.

    Parameters
    ----------
    student_logits, teacher_logits
        Per-token prototype scores ``(batch, tokens, k)``.
    center
        Teacher centring vector ``(k,)``.
    mask
        Boolean ``(batch, tokens)``; ``True`` marks the masked tokens the
        loss is computed on.
    student_temp, teacher_temp
        Softmax temperatures.
    reduction
        ``"mean"`` (default), ``"sum"``, or ``"none"`` (per-sample CE).
    """
    ce = _distill_ce(
        student_logits,
        teacher_logits,
        center,
        student_temp=student_temp,
        teacher_temp=teacher_temp,
    )  # (batch, tokens)
    per_sample = reduce(
        ce, axis=-1, weight=mask.astype(ce.dtype), reduction='mean'
    )
    return reduce(per_sample, reduction=reduction)


def koleo(
    z: Float[Array, 'n d'],
    *,
    eps: float = 1e-8,
    reduction: Reduction = 'mean',
) -> Float[Array, '...']:
    """Kozachenko--Leonenko entropy (feature-spread) regulariser.

    Penalises features clumping together: for each (L2-normalised) point it
    finds the nearest neighbour via the maximum cosine similarity (self
    excluded by a ``-2`` diagonal, below the ``[-1, 1]`` range), converts to
    the Euclidean distance ``sqrt(2 - 2 cos)``, and returns
    ``-log(distance + eps)`` -- large when points collapse together, small
    when they are well spread.

    The dense self-excluded cosine nearest-neighbour is exact and right at
    embedding-batch sizes; the EUCLIDEAN-semiring ELL k-NN is the at-scale
    path if this is ever applied to a large point set.

    Parameters
    ----------
    z
        ``(n, d)`` embeddings.
    eps
        Guards ``log`` of a zero nearest-neighbour distance.
    reduction
        ``"mean"`` (default), ``"sum"``, or ``"none"`` (per-point value).
    """
    n = z.shape[0]
    zn = l2_normalize(z, axis=-1)
    sim = zn @ zn.T
    sim = sim - jnp.eye(n, dtype=sim.dtype) * 2.0
    nn_sim = jnp.max(sim, axis=-1)
    distance = jnp.sqrt(jnp.maximum(2.0 - 2.0 * nn_sim, 0.0))
    return reduce(-jnp.log(distance + eps), reduction=reduction)
