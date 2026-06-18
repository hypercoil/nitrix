# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Exponential families for the GLM / GAM IRLS core.

A ``Family`` is a frozen record of the pure GLM/IRLS primitives (link,
variance, deviance, log-likelihood, an IRLS initialiser).  It is kept frozen --
**not** a bare ``Protocol`` -- because hashability is load-bearing: a family
rides as a static ``vmap`` / ``custom_vjp`` non-differentiable argument, exactly
like ``VarCompSpec`` / ``SolverSpec``.  The *open set* is served by the
``_FAMILIES`` registry + ``resolve_family``: callers pass ``str`` (a built-in)
or any ``Family`` instance.

This lives in its own module (not ``glm``) so the IRLS core (``_irls``), the
GLM surface (``glm``), and the GAM surface (``gam``) can all import the family
type without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Union

import jax.numpy as jnp
from jax.scipy.special import gammaln, xlogy
from jaxtyping import Array

__all__ = [
    'Family',
    'GAUSSIAN',
    'BINOMIAL',
    'POISSON',
    'GAMMA',
    'NEGBINOMIAL',
    'negbinomial',
    'resolve_family',
]

_EPS = 1e-10


@dataclass(frozen=True)
class Family:
    """An exponential-family + link, as a record of pure functions.

    Frozen and hashable so it rides as a static config (and a future
    ``custom_vjp`` nondiff arg).  The fields are the GLM/IRLS primitives:

    - ``link`` / ``linkinv`` -- ``g(mu)`` and ``g^{-1}(eta)``.
    - ``mu_eta`` -- ``d mu / d eta`` as a function of ``eta`` (IRLS weight).
    - ``variance`` -- the variance function ``V(mu)``.
    - ``unit_deviance`` -- per-observation deviance contribution ``d(y, mu)``.
    - ``init_mu`` -- a fitted-mean initialiser from ``y`` (IRLS start).
    - ``loglik`` -- per-observation log-likelihood ``l(y, mu, dispersion)``
      (the dispersion argument is ignored by the fixed-dispersion families).
    - ``has_fixed_dispersion`` -- ``True`` when the dispersion is 1 (binomial /
      Poisson; negative-binomial with a *known* ``alpha``); ``False`` when it is
      estimated (Gaussian, Gamma).
    """

    name: str
    has_fixed_dispersion: bool
    link: Callable[[Array], Array]
    linkinv: Callable[[Array], Array]
    mu_eta: Callable[[Array], Array]
    variance: Callable[[Array], Array]
    unit_deviance: Callable[[Array, Array], Array]
    init_mu: Callable[[Array], Array]
    loglik: Callable[[Array, Array, Array], Array]


def _identity(x: Array) -> Array:
    return x


def _ones_like(x: Array) -> Array:
    return jnp.ones_like(x)


def _gaussian_loglik(y: Array, mu: Array, dispersion: Array) -> Array:
    phi = jnp.clip(dispersion, _EPS, None)
    return -0.5 * (jnp.log(2.0 * jnp.pi * phi) + (y - mu) ** 2 / phi)


GAUSSIAN = Family(
    name='gaussian',
    has_fixed_dispersion=False,
    link=_identity,
    linkinv=_identity,
    mu_eta=_ones_like,
    variance=_ones_like,
    unit_deviance=lambda y, mu: (y - mu) ** 2,
    init_mu=_identity,
    loglik=_gaussian_loglik,
)


def _logit(mu: Array) -> Array:
    m = jnp.clip(mu, _EPS, 1.0 - _EPS)
    return jnp.log(m / (1.0 - m))


def _expit(eta: Array) -> Array:
    return 0.5 * (1.0 + jnp.tanh(0.5 * eta))


def _binomial_mu_eta(eta: Array) -> Array:
    mu = _expit(eta)
    return mu * (1.0 - mu)


def _binomial_deviance(y: Array, mu: Array) -> Array:
    m = jnp.clip(mu, _EPS, 1.0 - _EPS)
    return 2.0 * (xlogy(y, y / m) + xlogy(1.0 - y, (1.0 - y) / (1.0 - m)))


def _binomial_loglik(y: Array, mu: Array, dispersion: Array) -> Array:
    m = jnp.clip(mu, _EPS, 1.0 - _EPS)
    return xlogy(y, m) + xlogy(1.0 - y, 1.0 - m)


BINOMIAL = Family(
    name='binomial',
    has_fixed_dispersion=True,
    link=_logit,
    linkinv=_expit,
    mu_eta=_binomial_mu_eta,
    variance=lambda mu: mu * (1.0 - mu),
    unit_deviance=_binomial_deviance,
    init_mu=lambda y: (y + 0.5) / 2.0,
    loglik=_binomial_loglik,
)


def _poisson_deviance(y: Array, mu: Array) -> Array:
    m = jnp.clip(mu, _EPS, None)
    return 2.0 * (xlogy(y, y / m) - (y - m))


def _poisson_loglik(y: Array, mu: Array, dispersion: Array) -> Array:
    m = jnp.clip(mu, _EPS, None)
    return xlogy(y, m) - m - gammaln(y + 1.0)


POISSON = Family(
    name='poisson',
    has_fixed_dispersion=True,
    link=lambda mu: jnp.log(jnp.clip(mu, _EPS, None)),
    linkinv=jnp.exp,
    mu_eta=jnp.exp,
    variance=_identity,
    unit_deviance=_poisson_deviance,
    init_mu=lambda y: y + 0.1,
    loglik=_poisson_loglik,
)


def _log_link(mu: Array) -> Array:
    return jnp.log(jnp.clip(mu, _EPS, None))


def _gamma_deviance(y: Array, mu: Array) -> Array:
    ym = jnp.clip(y, _EPS, None)
    m = jnp.clip(mu, _EPS, None)
    return 2.0 * ((ym - m) / m - jnp.log(ym / m))


def _gamma_loglik(y: Array, mu: Array, dispersion: Array) -> Array:
    # Gamma(mean=mu, dispersion=phi): shape k = 1/phi, scale = phi * mu.
    phi = jnp.clip(dispersion, _EPS, None)
    m = jnp.clip(mu, _EPS, None)
    ym = jnp.clip(y, _EPS, None)
    k = 1.0 / phi
    return (
        (k - 1.0) * jnp.log(ym)
        - ym / (phi * m)
        - k * jnp.log(phi * m)
        - gammaln(k)
    )


# Gamma uses a **log** link (positive mean by construction, the practical
# default for RTs / volumes); the canonical inverse link is available by
# constructing a custom ``Family``.  Dispersion (the gamma shape's reciprocal)
# is estimated, as for Gaussian.
GAMMA = Family(
    name='gamma',
    has_fixed_dispersion=False,
    link=_log_link,
    linkinv=jnp.exp,
    mu_eta=jnp.exp,
    variance=lambda mu: mu * mu,
    unit_deviance=_gamma_deviance,
    init_mu=lambda y: jnp.clip(y, _EPS, None),
    loglik=_gamma_loglik,
)


def negbinomial(alpha: float = 1.0) -> Family:
    """A negative-binomial (NB2) family with **known** dispersion ``alpha``.

    NB2 parameterisation: ``V(mu) = mu + alpha * mu^2`` (overdispersed counts),
    log link.  ``alpha`` is fixed at construction -- with it known, NB2 is a
    proper one-parameter family (GLM dispersion 1), so the shared IRLS core fits
    it directly.  Jointly *estimating* ``alpha`` (the profile-MLE outer loop) is
    a documented follow-up.  ``alpha -> 0`` recovers Poisson.
    """
    if alpha <= 0.0:
        raise ValueError(f'negbinomial: alpha must be > 0, got {alpha}.')
    a = float(alpha)
    inv_a = 1.0 / a

    def variance(mu: Array) -> Array:
        return mu + a * mu * mu

    def unit_deviance(y: Array, mu: Array) -> Array:
        ym = jnp.clip(y, 0.0, None)
        m = jnp.clip(mu, _EPS, None)
        return 2.0 * (
            xlogy(ym, ym / m)
            - (ym + inv_a) * jnp.log((1.0 + a * ym) / (1.0 + a * m))
        )

    def loglik(y: Array, mu: Array, dispersion: Array) -> Array:
        ym = jnp.clip(y, 0.0, None)
        m = jnp.clip(mu, _EPS, None)
        return (
            gammaln(ym + inv_a)
            - gammaln(inv_a)
            - gammaln(ym + 1.0)
            + xlogy(ym, a * m)
            - (ym + inv_a) * jnp.log(1.0 + a * m)
        )

    return Family(
        name='negbinomial',
        has_fixed_dispersion=True,
        link=_log_link,
        linkinv=jnp.exp,
        mu_eta=jnp.exp,
        variance=variance,
        unit_deviance=unit_deviance,
        init_mu=lambda y: jnp.clip(y, 0.0, None) + 0.1,
        loglik=loglik,
    )


# Default NB2 (alpha = 1); for another dispersion pass ``negbinomial(alpha)``.
NEGBINOMIAL = negbinomial(1.0)


# The registry of built-ins -- the open-set extension point (callers may also
# pass any ``Family`` instance directly).
_FAMILIES: Mapping[str, Family] = {
    'gaussian': GAUSSIAN,
    'binomial': BINOMIAL,
    'poisson': POISSON,
    'gamma': GAMMA,
    'negbinomial': NEGBINOMIAL,
}


def resolve_family(family: Union[str, Family]) -> Family:
    """Resolve a ``str`` name (a built-in) or a ``Family`` instance to a family.

    ``glm_fit`` / ``gam_fit`` accept either; a string is looked up in the
    built-in registry (``'gaussian'`` / ``'binomial'`` / ``'poisson'`` /
    ``'gamma'`` / ``'negbinomial'``).  ``'negbinomial'`` resolves to the default
    ``alpha = 1``; for another dispersion pass ``negbinomial(alpha)`` directly.
    """
    if isinstance(family, Family):
        return family
    try:
        return _FAMILIES[family]
    except KeyError:
        raise ValueError(
            f'unknown family {family!r}; built-ins are '
            f'{sorted(_FAMILIES)}, or pass a Family instance.'
        ) from None
