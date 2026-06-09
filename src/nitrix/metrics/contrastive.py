# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Contrastive / self-supervised representation losses.

- ``nt_xent`` -- the normalised temperature-scaled cross-entropy
  (InfoNCE / SimCLR): pull paired views together and push all other
  samples apart on the unit sphere.
- ``dino_cross_entropy`` -- the self-distillation cross-entropy between a
  sharpened, centred teacher distribution (stop-gradient) and a student.
- ``ibot_cross_entropy`` -- the masked-token variant of the above (a
  masked-image-modelling objective), averaged over masked positions.
- ``koleo`` -- the Kozachenko--Leonenko differential-entropy regulariser:
  spread features out by penalising small nearest-neighbour distances.

These are *costs* (minimise).  ``dino`` / ``ibot`` take the teacher
``center`` as an argument -- maintaining it (an EMA over batches) is a
training-loop concern, not part of the loss numeric.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from ..numerics import l2_normalize
from ._common import Reduction, _reduce

__all__ = [
    'nt_xent',
    'dino_cross_entropy',
    'ibot_cross_entropy',
    'koleo',
]


def nt_xent(
    z: Float[Array, 'two_n d'],
    *,
    temperature: float = 0.5,
    reduction: Reduction = 'mean',
) -> Float[Array, '...']:
    """Normalised temperature-scaled cross-entropy (InfoNCE / SimCLR).

    ``z`` stacks the ``2N`` view embeddings with **adjacent pairs**: rows
    ``2i`` and ``2i+1`` are the two augmented views of sample ``i`` (each
    other's positive).  Embeddings are L2-normalised; the scaled
    cosine-similarity matrix ``z zᵀ / τ`` is the logits, the row's
    positive (index ``i XOR 1``) is the target, and every other row is a
    negative.

    The self-similarity on the diagonal is masked with a **finite**
    ``-2/τ`` subtraction rather than ``-inf`` -- the diagonal cosine is
    ``1/τ``, so this drops it below every off-diagonal entry without
    risking an ``inf - inf`` in the ``log_softmax``.

    Parameters
    ----------
    z
        ``(2N, d)`` view embeddings (adjacent-pair layout).
    temperature
        Softmax temperature ``τ``.
    reduction
        ``"mean"`` (default), ``"sum"``, or ``"none"`` (per-view loss).
    """
    n2 = z.shape[0]
    zn = l2_normalize(z, axis=-1)
    sim = (zn @ zn.T) / temperature
    sim = sim - jnp.eye(n2, dtype=sim.dtype) * (2.0 / temperature)
    log_probs = jax.nn.log_softmax(sim, axis=-1)
    positive = jnp.arange(n2) ^ 1
    loss = -log_probs[jnp.arange(n2), positive]
    return _reduce(loss, None, reduction)


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
    (``softmax((teacher - center) / teacher_temp)``) and detached; the
    loss is its cross-entropy against the student's
    ``log_softmax(student / student_temp)``, summed over the ``k``
    prototypes and reduced over the batch.  Centring (subtracting a
    running mean) plus a low teacher temperature is what prevents
    collapse.

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
    teacher = jax.nn.softmax((teacher_logits - center) / teacher_temp, axis=-1)
    teacher = jax.lax.stop_gradient(teacher)
    student = jax.nn.log_softmax(student_logits / student_temp, axis=-1)
    ce = -jnp.sum(teacher * student, axis=-1)
    return _reduce(ce, None, reduction)


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

    The same centred / sharpened teacher cross-entropy as
    :func:`dino_cross_entropy`, but evaluated per patch token and
    averaged over the **masked** positions of each sample (the denominator
    is clipped at 1 so an all-unmasked sample contributes 0 rather than
    ``0/0``).

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
    teacher = jax.nn.softmax((teacher_logits - center) / teacher_temp, axis=-1)
    teacher = jax.lax.stop_gradient(teacher)
    student = jax.nn.log_softmax(student_logits / student_temp, axis=-1)
    ce = -jnp.sum(teacher * student, axis=-1)  # (batch, tokens)
    m = mask.astype(ce.dtype)
    denom = jnp.maximum(jnp.sum(m, axis=-1), 1.0)
    per_sample = jnp.sum(ce * m, axis=-1) / denom
    return _reduce(per_sample, None, reduction)


def koleo(
    z: Float[Array, 'n d'],
    *,
    eps: float = 1e-8,
    reduction: Reduction = 'mean',
) -> Float[Array, '...']:
    """Kozachenko--Leonenko entropy (feature-spread) regulariser.

    Penalises features clumping together: for each (L2-normalised) point
    it finds the nearest neighbour via the maximum cosine similarity
    (self excluded by a ``-2`` diagonal, below the ``[-1, 1]`` range),
    converts to the Euclidean distance ``sqrt(2 - 2 cos)``, and returns
    ``-log(distance + eps)`` -- large when points collapse together, small
    when they are well spread.

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
    return _reduce(-jnp.log(distance + eps), None, reduction)
