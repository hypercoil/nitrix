# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Multiple-comparison corrections + multi-contrast combination for mass-univariate
p-value / statistic maps.

Companions to the permutation engine: the Benjamini-Hochberg FDR and the
Bonferroni family-wise correction (over a flat per-element p-value vector), and
the **valid conjunction** (minimum-statistic) combination across contrasts
(audit N7).  All are pure array ops.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

__all__ = ['fdr_bh', 'bonferroni', 'conjunction', 'conjunction_pvalue']


def conjunction(
    stats: Float[Array, 'k *spatial'],
) -> Float[Array, '*spatial']:
    """Minimum-statistic conjunction across ``k`` contrasts (Nichols et al. 2005).

    ``stats`` stacks the ``k`` per-contrast statistic maps on the leading axis.
    Returns the per-element **minimum** over contrasts -- the valid statistic for
    the conjunction null *"at least one of the ``k`` effects is null"*: a voxel
    survives a threshold only where **every** contrast clears it (the minimum
    does).  This is the correct conjunction; the common
    "all-effects-null" intersection of separately-thresholded maps tests a
    *different* (global) null and does not control the conjunction error.
    """
    return jnp.min(jnp.asarray(stats), axis=0)


def conjunction_pvalue(
    p_values: Float[Array, 'k *spatial'],
) -> Float[Array, '*spatial']:
    """Conjunction p-value: the **maximum** per-contrast p-value over ``k``
    contrasts (the p-scale dual of :func:`conjunction`).

    The minimum-statistic conjunction is significant at ``alpha`` iff every
    contrast's p-value is below ``alpha`` iff their maximum is -- so ``max p`` is
    the valid conjunction p-value (Nichols et al. 2005).
    """
    return jnp.max(jnp.asarray(p_values), axis=0)


def fdr_bh(
    p_values: Float[Array, 'n'],
    *,
    alpha: float = 0.05,
) -> Tuple[Bool[Array, 'n'], Float[Array, 'n']]:
    """Benjamini-Hochberg FDR correction.

    Returns ``(rejected, p_adjusted)``: the BH-adjusted p-values (q-values,
    enforced monotone non-decreasing in rank) and the rejection mask
    ``p_adjusted <= alpha``.
    """
    p = jnp.asarray(p_values)
    m = p.shape[0]
    order = jnp.argsort(p)
    ranks = jnp.arange(1, m + 1, dtype=p.dtype)
    p_sorted = p[order]
    # BH adjusted (step-up): q_i = min_{k>=i} (m/k) p_(k), enforced monotone.
    raw = p_sorted * m / ranks
    q_sorted = jnp.minimum.accumulate(raw[::-1])[::-1]
    q_sorted = jnp.clip(q_sorted, None, 1.0)
    # unsort
    p_adj = jnp.zeros_like(p).at[order].set(q_sorted)
    return p_adj <= alpha, p_adj


def bonferroni(
    p_values: Float[Array, 'n'],
    *,
    alpha: float = 0.05,
) -> Tuple[Bool[Array, 'n'], Float[Array, 'n']]:
    """Bonferroni family-wise correction.

    Returns ``(rejected, p_adjusted)`` with ``p_adjusted = min(p * n, 1)``.
    """
    p = jnp.asarray(p_values)
    m = p.shape[0]
    p_adj = jnp.clip(p * m, None, 1.0)
    return p_adj <= alpha, p_adj
