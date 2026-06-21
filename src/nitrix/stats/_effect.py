# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Effect-size readouts on a fitted contrast (audit N5).

The fitters return ``(effect, se, dof)`` for a contrast, so a confidence
interval and a standardized effect are each one transform away -- but they are
the readouts a neuroimaging report actually quotes.  These are thin, fitter-
agnostic helpers: feed them the ``effect`` / ``se`` (and ``dof``) from
``glm.t_contrast`` or ``lme_t_contrast``, or any per-element estimate.

The Student-t quantile is computed per element (so a per-voxel Satterthwaite
``dof`` is fine) by a few Newton steps on the ``betainc`` t-CDF, seeded from the
normal quantile -- jit-friendly and differentiable, no ``scipy`` at trace time.
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

    Newton iteration on the ``betainc``-based t-CDF, seeded from the normal
    quantile ``ndtri(p)``.  Vectorises / vmaps over a per-element ``dof``.
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
    """Two-sided confidence interval ``[lo, hi]`` for a contrast estimate.

    ``[effect - t_crit se, effect + t_crit se]`` with ``t_crit`` the
    ``(1 + level) / 2`` quantile of Student's t on ``dof`` degrees of freedom.
    Per-element and broadcasting -- ``dof`` may vary per element (e.g. a
    per-voxel Satterthwaite df).

    Parameters
    ----------
    effect, se, dof
        The contrast estimate, its standard error, and the denominator degrees
        of freedom -- e.g. from ``glm.t_contrast`` (``dof = result.dof_resid``)
        or ``lme_t_contrast`` (``dof = result.df``).
    level
        Coverage in ``(0, 1)`` (default ``0.95``).

    Returns
    -------
    ``(lo, hi)`` -- the lower / upper interval bounds (same shape as ``effect``).
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
    """Standardized effect size ``effect / scale`` (a Cohen's-d-style ratio).

    Expresses the contrast on the residual-SD scale: pass ``scale =
    sqrt(result.dispersion)`` (the residual standard deviation) for a GLM /
    LME fit, so the result is the effect in standard-deviation units.
    """
    return jnp.asarray(effect) / jnp.clip(jnp.asarray(scale), _EPS, None)
