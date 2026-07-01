# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Multiple-comparison corrections and multi-contrast combination for
mass-univariate p-value and statistic maps.

These are companions to the permutation engine. They provide false-discovery-rate
control -- Benjamini-Hochberg (:func:`fdr_bh`, valid under independence or
positive regression dependence), Benjamini-Yekutieli (:func:`fdr_by`, valid under
arbitrary dependence), and Storey's :math:`\\pi_0`-adaptive q-values
(:func:`fdr_storey`, higher power) behind a unified :func:`fdr` dispatcher -- the
Bonferroni family-wise correction (:func:`bonferroni`), and the valid conjunction
(minimum-statistic) combination across contrasts (:func:`conjunction`,
:func:`conjunction_pvalue`). All are pure array operations over a flat
per-element p-value vector.
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
    """Minimum-statistic conjunction across ``k`` contrasts.

    Returns the per-element minimum statistic over the ``k`` contrasts, the valid
    statistic for the conjunction null *"at least one of the ``k`` effects is
    null"*: an element survives a threshold only where every contrast clears it,
    which is exactly where the minimum clears it. This is the correct conjunction;
    the common "all-effects-null" intersection of separately-thresholded maps
    tests a different (global) null and does not control the conjunction error.

    Parameters
    ----------
    stats : Float[Array, 'k *spatial']
        The ``k`` per-contrast statistic maps stacked on the leading axis, each of
        shape ``*spatial``.

    Returns
    -------
    Float[Array, '*spatial']
        The per-element minimum statistic over the ``k`` contrasts.

    References
    ----------
    Nichols, T., Brett, M., Andersson, J., Wager, T., & Poline, J.-B. (2005).
    Valid conjunction inference with the minimum statistic. NeuroImage, 25(3),
    653-660. https://doi.org/10.1016/j.neuroimage.2004.12.005
    """
    return jnp.min(jnp.asarray(stats), axis=0)


def conjunction_pvalue(
    p_values: Float[Array, 'k *spatial'],
) -> Float[Array, '*spatial']:
    """Conjunction p-value across ``k`` contrasts.

    Returns the per-element maximum of the per-contrast p-values, the p-scale
    dual of :func:`conjunction`. The minimum-statistic conjunction is significant
    at level :math:`\\alpha` if and only if every contrast's p-value is below
    :math:`\\alpha`, equivalently if and only if their maximum is, so the maximum
    p-value is the valid conjunction p-value.

    Parameters
    ----------
    p_values : Float[Array, 'k *spatial']
        The ``k`` per-contrast p-value maps stacked on the leading axis, each of
        shape ``*spatial``.

    Returns
    -------
    Float[Array, '*spatial']
        The per-element maximum p-value over the ``k`` contrasts.

    References
    ----------
    Nichols, T., Brett, M., Andersson, J., Wager, T., & Poline, J.-B. (2005).
    Valid conjunction inference with the minimum statistic. NeuroImage, 25(3),
    653-660. https://doi.org/10.1016/j.neuroimage.2004.12.005
    """
    return jnp.max(jnp.asarray(p_values), axis=0)


FDRMethod = Literal['bh', 'by', 'storey']


def _bh_stepup(
    p: Float[Array, 'n'],
    scale: Union[int, Float[Array, '']],
) -> Float[Array, 'n']:
    """Step-up adjusted p-values for a generalised Benjamini-Hochberg procedure.

    Computes the adjusted p-values
    :math:`q_{(i)} = \\min_{k \\geq i} (\\mathrm{scale} / k)\\, p_{(k)}`, enforced
    monotone non-decreasing in rank and clipped to ``1``, where :math:`p_{(k)}` is
    the ``k``-th smallest p-value.

    The ``scale`` argument replaces the count :math:`m` in the standard
    Benjamini-Hochberg procedure: ``scale = m`` gives Benjamini-Hochberg,
    :math:`\\mathrm{scale} = m \\sum_k 1/k` gives Benjamini-Yekutieli, and
    :math:`\\mathrm{scale} = m\\,\\pi_0` gives Storey's :math:`\\pi_0`-adaptive
    FDR.

    Parameters
    ----------
    p : Float[Array, 'n']
        Flat vector of ``n`` per-element p-values.
    scale : int or Float[Array, '']
        The step-up scaling factor that determines the procedure variant (see
        above). A scalar.

    Returns
    -------
    Float[Array, 'n']
        The adjusted p-values (q-values), in the original element order.
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
    """Benjamini-Hochberg false-discovery-rate correction.

    Valid under independence or positive regression dependence of the p-values.

    Parameters
    ----------
    p_values : Float[Array, 'n']
        Flat vector of ``n`` per-element p-values.
    alpha : float, optional
        Target false-discovery rate defining the rejection mask. Default ``0.05``.

    Returns
    -------
    rejected : Bool[Array, 'n']
        The rejection mask ``p_adjusted <= alpha``.
    p_adjusted : Float[Array, 'n']
        The Benjamini-Hochberg adjusted p-values (q-values), enforced monotone
        non-decreasing in rank.
    """
    p = jnp.asarray(p_values)
    p_adj = _bh_stepup(p, p.shape[0])
    return p_adj <= alpha, p_adj


def fdr_by(
    p_values: Float[Array, 'n'],
    *,
    alpha: float = 0.05,
) -> Tuple[Bool[Array, 'n'], Float[Array, 'n']]:
    """Benjamini-Yekutieli false-discovery-rate correction.

    Valid under arbitrary dependence of the p-values. This is the
    Benjamini-Hochberg step-up applied at level :math:`\\alpha / c(m)` with the
    harmonic number :math:`c(m) = \\sum_{k=1}^{m} 1/k`, equivalently the
    Benjamini-Hochberg q-values scaled by :math:`c(m)`. Use it when the p-values
    may be arbitrarily (for example, negatively) dependent and the
    positive-dependence assumption of Benjamini-Hochberg is unsafe; the price is
    being a factor of about :math:`\\log(m)` more conservative.

    Parameters
    ----------
    p_values : Float[Array, 'n']
        Flat vector of ``n`` per-element p-values.
    alpha : float, optional
        Target false-discovery rate defining the rejection mask. Default ``0.05``.

    Returns
    -------
    rejected : Bool[Array, 'n']
        The rejection mask ``p_adjusted <= alpha``.
    p_adjusted : Float[Array, 'n']
        The Benjamini-Yekutieli adjusted p-values (q-values), enforced monotone
        non-decreasing in rank.
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
    """Storey's estimate of the true-null proportion :math:`\\pi_0`.

    Computes
    :math:`\\hat{\\pi}_0(\\lambda) = \\#\\{p_i > \\lambda\\} / (m (1 - \\lambda))`,
    clipped to :math:`[1/m, 1]`, where :math:`m` is the number of p-values. The
    tuning parameter :math:`\\lambda \\in (0, 1)` (default ``0.5``) trades bias
    for variance. A value :math:`\\pi_0 = 1` recovers Benjamini-Hochberg, and a
    smaller :math:`\\pi_0` (more alternatives present) yields a less conservative,
    higher-power adaptive FDR. The :math:`1/m` floor guards against the degenerate
    "no null" estimate.

    Parameters
    ----------
    p_values : Float[Array, 'n']
        Flat vector of ``n`` per-element p-values.
    lam : float, optional
        Tuning parameter :math:`\\lambda \\in (0, 1)` at which the null proportion
        is estimated. Must lie strictly within ``(0, 1)``. Default ``0.5``.

    Returns
    -------
    Float[Array, '']
        The estimated true-null proportion :math:`\\pi_0`, a scalar clipped to
        :math:`[1/m, 1]`.

    Raises
    ------
    ValueError
        If ``lam`` does not lie strictly within ``(0, 1)``.

    References
    ----------
    Storey, J. D. (2002). A direct approach to false discovery rates. Journal of
    the Royal Statistical Society: Series B (Statistical Methodology), 64(3),
    479-498. https://doi.org/10.1111/1467-9868.00346
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
    """Storey's :math:`\\pi_0`-adaptive false-discovery-rate correction.

    Yields higher power, valid under independence or positive regression
    dependence. This is Benjamini-Hochberg with the count :math:`m` replaced by
    the estimated number of true nulls :math:`m\\,\\pi_0` (see
    :func:`storey_pi0`): uniformly less conservative than Benjamini-Hochberg when
    a non-trivial fraction of effects are real, and identical to it when
    :math:`\\pi_0 = 1`.

    Parameters
    ----------
    p_values : Float[Array, 'n']
        Flat vector of ``n`` per-element p-values.
    alpha : float, optional
        Target false-discovery rate defining the rejection mask. Default ``0.05``.
    lam : float, optional
        Tuning parameter :math:`\\lambda \\in (0, 1)` passed to :func:`storey_pi0`
        for the :math:`\\pi_0` estimate. Default ``0.5``.

    Returns
    -------
    rejected : Bool[Array, 'n']
        The rejection mask ``p_adjusted <= alpha``.
    p_adjusted : Float[Array, 'n']
        The adaptive adjusted p-values (q-values), enforced monotone
        non-decreasing in rank.
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
    """Unified false-discovery-rate correction dispatcher.

    Dispatches to one of the false-discovery-rate procedures according to
    ``method``.

    Parameters
    ----------
    p_values : Float[Array, 'n']
        Flat vector of ``n`` per-element p-values.
    method : {'bh', 'by', 'storey'}, optional
        The correction procedure: ``'bh'`` (default, Benjamini-Hochberg via
        :func:`fdr_bh`, valid under independence or positive regression
        dependence), ``'by'`` (Benjamini-Yekutieli via :func:`fdr_by`, valid
        under arbitrary dependence), or ``'storey'`` (:math:`\\pi_0`-adaptive via
        :func:`fdr_storey`, higher power).
    alpha : float, optional
        Target false-discovery rate defining the rejection mask. Default ``0.05``.
    lam : float, optional
        Tuning parameter :math:`\\lambda \\in (0, 1)` for the :math:`\\pi_0`
        estimate; used only when ``method='storey'``. Default ``0.5``.

    Returns
    -------
    rejected : Bool[Array, 'n']
        The rejection mask ``p_adjusted <= alpha``.
    p_adjusted : Float[Array, 'n']
        The adjusted p-values (q-values) of the selected procedure.

    Raises
    ------
    ValueError
        If ``method`` is not one of ``'bh'``, ``'by'``, or ``'storey'``.
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
    """Bonferroni family-wise error-rate correction.

    Scales each p-value by the number of tests ``n`` and clips to ``1``.

    Parameters
    ----------
    p_values : Float[Array, 'n']
        Flat vector of ``n`` per-element p-values.
    alpha : float, optional
        Target family-wise error rate defining the rejection mask. Default
        ``0.05``.

    Returns
    -------
    rejected : Bool[Array, 'n']
        The rejection mask ``p_adjusted <= alpha``.
    p_adjusted : Float[Array, 'n']
        The Bonferroni-adjusted p-values :math:`\\min(n\\, p, 1)`.
    """
    p = jnp.asarray(p_values)
    m = p.shape[0]
    p_adj = jnp.clip(p * m, None, 1.0)
    return p_adj <= alpha, p_adj
