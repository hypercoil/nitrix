# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Effect-size readouts on a fitted contrast.

The contrast fitters return ``(effect, se, dof)`` for a contrast, so a
confidence interval and a standardised effect are each one transform away --
but they are the readouts a neuroimaging report actually quotes.  These are
thin, fitter-agnostic helpers: feed them the ``effect`` / ``se`` (and ``dof``)
from :func:`t_contrast` or :func:`lme_t_contrast`, or any per-element estimate.

The Student's t quantile is computed per element (so a per-voxel Satterthwaite
``dof`` is fine) by a few Newton steps on the :math:`\\beta`-incomplete t-CDF,
seeded from the normal quantile -- jit-friendly and differentiable, with no
``scipy`` at trace time.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
from jax.scipy.special import betainc, gammaln, ndtri
from jaxtyping import Array, Float

__all__ = ['confidence_interval', 'standardized_effect']

_EPS = 1e-30


def _t_ppf(
    p: Float[Array, '...'],
    dof: Float[Array, '...'],
    n_newton: int = 8,
) -> Float[Array, '...']:
    """Per-element quantile of Student's t (the inverse CDF).

    Newton iteration on the incomplete-beta t-CDF, seeded from the normal
    quantile ``ndtri(p)``.  Broadcasts and vmaps over a per-element ``dof``.

    Parameters
    ----------
    p : Float[Array, '...']
        Cumulative probability at which to evaluate the quantile, in
        :math:`(0, 1)`.
    dof : Float[Array, '...']
        Degrees of freedom of the Student's t distribution, evaluated per
        element and broadcast against ``p``.
    n_newton : int, optional
        Number of Newton refinement steps taken from the normal-quantile
        seed (default ``8``).

    Returns
    -------
    Float[Array, '...']
        The Student's t quantile :math:`t` such that the CDF equals ``p``,
        with the broadcast shape of ``p`` and ``dof``.
    """
    t = ndtri(p)  # normal-approximation seed
    half_dof = 0.5 * dof
    log_const = (
        gammaln(0.5 * (dof + 1.0))
        - gammaln(half_dof)
        - 0.5 * jnp.log(dof * jnp.pi)
    )
    for _ in range(n_newton):
        x = dof / (dof + t * t)
        upper = 0.5 * betainc(half_dof, 0.5, x)  # P(T > |t|), t >= 0
        cdf = jnp.where(t >= 0.0, 1.0 - upper, upper)
        log_pdf = log_const - 0.5 * (dof + 1.0) * jnp.log1p(t * t / dof)
        pdf = jnp.exp(log_pdf)
        t = t - (cdf - p) / jnp.clip(pdf, _EPS, None)
    return t


def confidence_interval(
    effect: Float[Array, '...'],
    se: Float[Array, '...'],
    dof: Float[Array, '...'],
    *,
    level: float = 0.95,
) -> Tuple[Float[Array, '...'], Float[Array, '...']]:
    """Two-sided confidence interval :math:`[\\mathrm{lo}, \\mathrm{hi}]`.

    Forms the interval :math:`[\\text{effect} - t_{\\mathrm{crit}}\\,\\text{se},
    \\ \\text{effect} + t_{\\mathrm{crit}}\\,\\text{se}]` with
    :math:`t_{\\mathrm{crit}}` the :math:`(1 + \\text{level}) / 2` quantile of
    Student's t on ``dof`` degrees of freedom.  Evaluated per element and
    broadcasting -- ``dof`` may vary per element (e.g. a per-voxel Satterthwaite
    degrees-of-freedom estimate).

    Parameters
    ----------
    effect : Float[Array, '...']
        The contrast estimate, e.g. from :func:`t_contrast` or
        :func:`lme_t_contrast`.
    se : Float[Array, '...']
        The standard error of ``effect``, broadcast against it.
    dof : Float[Array, '...']
        Denominator degrees of freedom, broadcast against ``effect`` -- e.g.
        the residual degrees of freedom from :func:`t_contrast` or the
        Satterthwaite degrees of freedom from :func:`lme_t_contrast`.
    level : float, optional
        Two-sided coverage in :math:`(0, 1)` (default ``0.95``).

    Returns
    -------
    lo : Float[Array, '...']
        Lower interval bound, with the same shape as ``effect``.
    hi : Float[Array, '...']
        Upper interval bound, with the same shape as ``effect``.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(
            f'confidence_interval: level={level!r} must be in (0, 1).'
        )
    effect = jnp.asarray(effect)
    p = jnp.asarray(0.5 * (1.0 + level), dtype=effect.dtype)
    t_crit = _t_ppf(p, jnp.asarray(dof, dtype=effect.dtype))
    half = t_crit * jnp.asarray(se)
    return effect - half, effect + half


def standardized_effect(
    effect: Float[Array, '...'],
    scale: Float[Array, '...'],
) -> Float[Array, '...']:
    """Standardised effect size :math:`\\text{effect} / \\text{scale}`.

    A Cohen's-d-style ratio that expresses the contrast on the residual
    standard-deviation scale: pass ``scale = sqrt(result.dispersion)`` (the
    residual standard deviation) for a GLM or LME fit, so the result is the
    effect in standard-deviation units.  The denominator is clipped away from
    zero for numerical safety.

    Parameters
    ----------
    effect : Float[Array, '...']
        The contrast estimate to standardise.
    scale : Float[Array, '...']
        The scale (typically a residual standard deviation) by which to
        divide ``effect``, broadcast against it.

    Returns
    -------
    Float[Array, '...']
        The standardised effect ``effect / scale``, with the broadcast shape
        of the two inputs.
    """
    return jnp.asarray(effect) / jnp.clip(jnp.asarray(scale), _EPS, None)
