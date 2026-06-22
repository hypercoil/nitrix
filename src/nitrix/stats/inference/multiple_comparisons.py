# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Multiple-comparison corrections + multi-contrast combination for mass-univariate
p-value / statistic maps.

Companions to the permutation engine: false-discovery-rate control --
Benjamini-Hochberg (``fdr_bh``; valid under independence / PRDS),
Benjamini-Yekutieli (``fdr_by``; valid under *arbitrary* dependence), and
Storey's pi0-adaptive q-values (``fdr_storey``; higher power), behind a unified
``fdr(method=...)`` dispatcher -- the Bonferroni family-wise correction, and the
**valid conjunction** (minimum-statistic) combination across contrasts (audit
N7).  All are pure array ops over a flat per-element p-value vector.
"""

from __future__ import annotations

from typing import Literal, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

__all__ = [
    'fdr',
    'fdr_bh',
    'fdr_by',
    'fdr_storey',
    'storey_pi0',
    'bonferroni',
    'conjunction',
    'conjunction_pvalue',
]


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


FDRMethod = Literal['bh', 'by', 'storey']


def _bh_stepup(
    p: Float[Array, 'n'], scale: Union[int, Float[Array, '']],
) -> Float[Array, 'n']:
    """Step-up adjusted p-values ``q_(i) = min_{k>=i} (scale / k) p_(k)`` (enforced
    monotone non-decreasing in rank, clipped to ``1``).

    ``scale`` replaces the count ``m`` of the Benjamini-Hochberg procedure:
    ``scale = m`` is BH; ``scale = m * sum_k 1/k`` is Benjamini-Yekutieli;
    ``scale = m * pi0`` is Storey's pi0-adaptive FDR.
    """
    m = p.shape[0]
    order = jnp.argsort(p)
    ranks = jnp.arange(1, m + 1, dtype=p.dtype)
    p_sorted = p[order]
    raw = p_sorted * scale / ranks
    q_sorted = jnp.clip(jnp.minimum.accumulate(raw[::-1])[::-1], None, 1.0)
    return jnp.zeros_like(p).at[order].set(q_sorted)


def fdr_bh(
    p_values: Float[Array, 'n'],
    *,
    alpha: float = 0.05,
) -> Tuple[Bool[Array, 'n'], Float[Array, 'n']]:
    """Benjamini-Hochberg FDR correction (valid under independence / PRDS).

    Returns ``(rejected, p_adjusted)``: the BH-adjusted p-values (q-values,
    enforced monotone non-decreasing in rank) and the rejection mask
    ``p_adjusted <= alpha``.
    """
    p = jnp.asarray(p_values)
    p_adj = _bh_stepup(p, p.shape[0])
    return p_adj <= alpha, p_adj


def fdr_by(
    p_values: Float[Array, 'n'],
    *,
    alpha: float = 0.05,
) -> Tuple[Bool[Array, 'n'], Float[Array, 'n']]:
    """Benjamini-Yekutieli FDR correction -- valid under **arbitrary** dependence.

    The BH step-up at level ``alpha / c(m)`` with the harmonic number
    ``c(m) = sum_{k=1}^m 1/k`` (equivalently, the BH q-values scaled by ``c(m)``).
    Use when the p-values may be arbitrarily (e.g. negatively) dependent and BH's
    positive-dependence assumption is unsafe -- the price is a ``~log(m)`` factor
    more conservative.  Returns ``(rejected, p_adjusted)``.
    """
    p = jnp.asarray(p_values)
    m = p.shape[0]
    c_m = jnp.sum(1.0 / jnp.arange(1, m + 1, dtype=p.dtype))
    p_adj = _bh_stepup(p, m * c_m)
    return p_adj <= alpha, p_adj


def storey_pi0(
    p_values: Float[Array, 'n'],
    *,
    lam: float = 0.5,
) -> Float[Array, '']:
    """Storey's estimate of the true-null proportion ``pi0`` (Storey 2002).

    ``pi0(lambda) = #{p_i > lambda} / (m (1 - lambda))``, clipped to
    ``[1/m, 1]``.  The tuning ``lambda in (0, 1)`` (default ``0.5``) trades bias
    for variance; ``pi0 = 1`` recovers Benjamini-Hochberg, and a smaller ``pi0``
    (more alternatives present) yields a less conservative, higher-power adaptive
    FDR.  The ``1/m`` floor guards the degenerate "no null" estimate.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError(f'storey_pi0: lam={lam} must lie in (0, 1).')
    p = jnp.asarray(p_values)
    m = p.shape[0]
    pi0 = jnp.sum(p > lam) / (m * (1.0 - lam))
    return jnp.clip(pi0, 1.0 / m, 1.0)


def fdr_storey(
    p_values: Float[Array, 'n'],
    *,
    alpha: float = 0.05,
    lam: float = 0.5,
) -> Tuple[Bool[Array, 'n'], Float[Array, 'n']]:
    """Storey's pi0-adaptive FDR / q-values (higher power; independence / PRDS).

    Benjamini-Hochberg with the count ``m`` replaced by the estimated number of
    true nulls ``m * pi0`` (see :func:`storey_pi0`): uniformly less conservative
    than BH when a non-trivial fraction of effects are real, and identical to BH
    when ``pi0 = 1``.  Returns ``(rejected, p_adjusted)``.
    """
    p = jnp.asarray(p_values)
    p_adj = _bh_stepup(p, p.shape[0] * storey_pi0(p, lam=lam))
    return p_adj <= alpha, p_adj


def fdr(
    p_values: Float[Array, 'n'],
    *,
    method: FDRMethod = 'bh',
    alpha: float = 0.05,
    lam: float = 0.5,
) -> Tuple[Bool[Array, 'n'], Float[Array, 'n']]:
    """Unified false-discovery-rate correction.

    ``method``: ``'bh'`` (default -- Benjamini-Hochberg, valid under independence
    / PRDS), ``'by'`` (Benjamini-Yekutieli, valid under arbitrary dependence), or
    ``'storey'`` (pi0-adaptive, higher power; ``lam`` tunes the ``pi0`` estimate).
    Returns ``(rejected, p_adjusted)``.
    """
    if method == 'bh':
        return fdr_bh(p_values, alpha=alpha)
    if method == 'by':
        return fdr_by(p_values, alpha=alpha)
    if method == 'storey':
        return fdr_storey(p_values, alpha=alpha, lam=lam)
    raise ValueError(
        f"fdr: method={method!r}; expected 'bh', 'by', or 'storey'."
    )


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
