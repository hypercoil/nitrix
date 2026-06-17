# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Multiple-comparison corrections for mass-univariate p-value maps.

Companions to the permutation engine for the parametric route: the
Benjamini-Hochberg FDR and the Bonferroni family-wise correction.  Both are
pure array ops over a flat (per-element) p-value vector and return the adjusted
p-values plus a rejection mask at a chosen level.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

__all__ = ['fdr_bh', 'bonferroni']


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
