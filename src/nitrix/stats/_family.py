# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Exponential families for the GLM / GAM IRLS core.

A :class:`Family` is a frozen record of the pure GLM/IRLS primitives (link,
variance, deviance, log-likelihood, an IRLS initialiser).  It is kept frozen --
**not** a bare ``Protocol`` -- because hashability is load-bearing: a family
rides as a static ``vmap`` / ``custom_vjp`` non-differentiable argument, exactly
like a variance-component or solver specification.  The *open set* of families is
served by a built-in registry together with :func:`resolve_family`: callers pass
a ``str`` (a built-in name) or any :class:`Family` instance.

This lives in its own module (not ``glm``) so the IRLS core, the GLM surface, and
the GAM surface can all import the family type without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping, Union

import jax.numpy as jnp
from jax.scipy.special import gammaln, xlogy
from jax.scipy.stats import norm as _norm
from jaxtyping import Array

__all__ = [
    'Family',
    'Link',
    'IDENTITY_LINK',
    'LOG_LINK',
    'LOGIT_LINK',
    'PROBIT_LINK',
    'CLOGLOG_LINK',
    'SQRT_LINK',
    'INVERSE_LINK',
    'resolve_link',
    'GAUSSIAN',
    'BINOMIAL',
    'POISSON',
    'GAMMA',
    'NEGBINOMIAL',
    'TWEEDIE',
    'negbinomial',
    'tweedie',
    'resolve_family',
]

_EPS = 1e-10

# Linear-predictor clamp for the unbounded (``exp``) inverse links.  ``mu =
# exp(eta)`` overflows for ``eta`` beyond a few hundred and -- well before that --
# produces astronomically large IRLS weights that let a single transient-overshoot
# observation dominate the normal equations (garbage / NaN).  This is the textbook
# fragility of PQL for a Poisson / Gamma random *slope* (a large ``b_slope * x``
# blows up ``exp(eta)``) and of any log-link GAM whose smooth swings wide during
# iteration.  ``mu = exp(20) ~ 5e8`` is already far past any realistic count /
# rate, so clamping ``eta`` to ``+-20`` is harmless for a sane fit while breaking
# the runaway feedback (empirically it is also the difference between the
# random-slope PQL landing in the right REML basin vs a degenerate one).  The
# bounded links (identity / logit) never overflow, so they keep ``eta_bound = inf``
# (no clamp -- a Gaussian fit's ``eta = mu`` is legitimately unbounded).
_ETA_MAX = 20.0


def _safe_exp(eta: Array) -> Array:
    """Overflow-safe exponential inverse link shared by the log-link families.

    Computes :math:`\\exp(\\eta)` after clamping the linear predictor to the
    numerically-sane range :math:`\\pm` ``_ETA_MAX``, so the exponential cannot
    overflow and the IRLS weights cannot run away.

    Parameters
    ----------
    eta : Array
        Linear predictor :math:`\\eta`, of arbitrary shape.

    Returns
    -------
    Array
        The clamped inverse link :math:`\\exp(\\operatorname{clip}(\\eta))`, of
        the same shape as ``eta``.
    """
    return jnp.exp(jnp.clip(eta, -_ETA_MAX, _ETA_MAX))


@dataclass(frozen=True)
class Family:
    """An exponential family together with a link, as a record of pure functions.

    Frozen and hashable so it rides as a static configuration argument (and a
    future ``custom_vjp`` non-differentiable argument).  The fields are the
    GLM/IRLS primitives:

    - ``link`` / ``linkinv`` -- the link :math:`g(\\mu)` and inverse link
      :math:`g^{-1}(\\eta)`.
    - ``mu_eta`` -- the derivative :math:`\\mathrm{d}\\mu / \\mathrm{d}\\eta` as
      a function of :math:`\\eta` (an IRLS weight).
    - ``variance`` -- the variance function :math:`V(\\mu)`.
    - ``unit_deviance`` -- the per-observation deviance contribution
      :math:`d(y, \\mu)`.
    - ``init_mu`` -- a fitted-mean initialiser from :math:`y` (the IRLS start).
    - ``loglik`` -- the per-observation log-likelihood
      :math:`\\ell(y, \\mu, \\phi)` (the dispersion argument :math:`\\phi` is
      ignored by the fixed-dispersion families).
    - ``has_fixed_dispersion`` -- ``True`` when the dispersion is 1 (binomial /
      Poisson; negative-binomial with a *known* ``alpha``); ``False`` when it is
      estimated (Gaussian, Gamma).
    - ``eta_bound`` -- the IRLS clamp on the linear predictor
      (:math:`\\infty` for the bounded identity / logit links; ``_ETA_MAX`` for
      the unbounded exponential links, where it stabilises the working
      response).  The IRLS core clamps :math:`\\eta` to
      :math:`\\pm` ``eta_bound``.
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
    eta_bound: float = float('inf')

    def clip_eta(self, eta: Array) -> Array:
        """Clamp the linear predictor to the family's numerically-sane range.

        Clamps :math:`\\eta` to :math:`\\pm` ``eta_bound`` -- :math:`\\infty`
        for the bounded identity / logit links (a no-op), ``_ETA_MAX`` for the
        unbounded exponential links (where an un-clamped :math:`\\eta` blows up
        :math:`\\exp(\\eta)` and the IRLS weights).  This is the single source
        of truth for the clamp across the IRLS, PQL, and Laplace
        working-response sites.

        Parameters
        ----------
        eta : Array
            Linear predictor :math:`\\eta`, of arbitrary shape.

        Returns
        -------
        Array
            The clamped linear predictor, of the same shape as ``eta``.
        """
        return jnp.clip(eta, -self.eta_bound, self.eta_bound)

    def with_link(self, link: Union[str, 'Link']) -> 'Family':
        """A copy of this family with a **non-canonical** link substituted.

        Only the link triple (``link`` / ``linkinv`` / ``mu_eta``) and the IRLS
        ``eta_bound`` change; the distribution -- ``variance`` / ``unit_deviance``
        / ``loglik`` / ``init_mu`` (all functions of :math:`\\mu`, not
        :math:`\\eta`) -- is untouched, so e.g. ``BINOMIAL.with_link('probit')``
        is the probit model and ``POISSON.with_link('sqrt')`` the
        square-root-link Poisson, exactly as in ``statsmodels``'
        ``Family(link=...)`` or ``mgcv``.  The shared IRLS core is Fisher scoring
        (weight :math:`w = (\\mathrm{d}\\mu/\\mathrm{d}\\eta)^2 / V` and working
        response :math:`z = \\eta + (y - \\mu) / (\\mathrm{d}\\mu/\\mathrm{d}\\eta)`),
        so any link composes without a solver change.

        Parameters
        ----------
        link : str or Link
            The replacement link, as a built-in name or a :class:`Link`
            instance; resolved via :func:`resolve_link`.

        Returns
        -------
        Family
            A copy of this family carrying the substituted link (and its
            ``eta_bound``), with the name suffixed by the link's name.
        """
        lk = resolve_link(link)
        return replace(
            self,
            name=f'{self.name}[{lk.name}]',
            link=lk.link,
            linkinv=lk.linkinv,
            mu_eta=lk.mu_eta,
            eta_bound=lk.eta_bound,
        )


@dataclass(frozen=True)
class Link:
    """A link function as a record of its three IRLS primitives.

    Here ``link`` is :math:`g(\\mu)`, ``linkinv`` is :math:`g^{-1}(\\eta)`, and
    ``mu_eta`` is :math:`\\mathrm{d}\\mu / \\mathrm{d}\\eta` (as a function of
    :math:`\\eta`); ``eta_bound`` is the IRLS linear-predictor clamp
    (:math:`\\infty` unless the inverse link can overflow).  Compose onto a
    distribution with :meth:`Family.with_link`.
    """

    name: str
    link: Callable[[Array], Array]
    linkinv: Callable[[Array], Array]
    mu_eta: Callable[[Array], Array]
    eta_bound: float = float('inf')


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
    linkinv=_safe_exp,
    mu_eta=_safe_exp,
    variance=_identity,
    unit_deviance=_poisson_deviance,
    init_mu=lambda y: y + 0.1,
    loglik=_poisson_loglik,
    eta_bound=_ETA_MAX,
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
    linkinv=_safe_exp,
    mu_eta=_safe_exp,
    variance=lambda mu: mu * mu,
    unit_deviance=_gamma_deviance,
    init_mu=lambda y: jnp.clip(y, _EPS, None),
    loglik=_gamma_loglik,
    eta_bound=_ETA_MAX,
)


def negbinomial(alpha: float = 1.0) -> Family:
    """A negative-binomial (NB2) family with **known** dispersion ``alpha``.

    NB2 parameterisation: variance :math:`V(\\mu) = \\mu + \\alpha \\mu^2`
    (overdispersed counts), log link.  ``alpha`` is fixed at construction --
    with it known, NB2 is a proper one-parameter family (GLM dispersion 1), so
    the shared IRLS core fits it directly.  Jointly *estimating* ``alpha`` (the
    profile-MLE outer loop) is a follow-up.  The limit
    :math:`\\alpha \\to 0` recovers the Poisson family.

    Parameters
    ----------
    alpha : float, default=1.0
        The known NB2 dispersion :math:`\\alpha`, entering the variance as
        :math:`\\mu + \\alpha \\mu^2`.  Must be strictly positive.

    Returns
    -------
    Family
        A negative-binomial family with the given fixed dispersion and a log
        link.
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
        linkinv=_safe_exp,
        mu_eta=_safe_exp,
        variance=variance,
        unit_deviance=unit_deviance,
        init_mu=lambda y: jnp.clip(y, 0.0, None) + 0.1,
        loglik=loglik,
        eta_bound=_ETA_MAX,
    )


# Default NB2 (alpha = 1); for another dispersion pass ``negbinomial(alpha)``.
NEGBINOMIAL = negbinomial(1.0)


def tweedie(p: float = 1.5) -> Family:
    """A Tweedie family with **fixed** power ``p`` (compound Poisson-Gamma).

    The regime :math:`1 < p < 2` is the semicontinuous one (an exact zero with
    probability mass plus a continuous positive part: rainfall, insurance claims,
    some imaging measures).  Variance :math:`V(\\mu) = \\mu^p`, log link,
    dispersion estimated (``has_fixed_dispersion=False``, like Gamma).  With
    ``p`` fixed the *mean* coefficients fit the shared IRLS core directly
    (working weight :math:`\\mu^{2-p}`); jointly profiling ``p`` is a follow-up
    (:math:`p \\to 1` recovers Poisson, :math:`p \\to 2` Gamma).

    The unit deviance is the closed form

    :math:`d(y, \\mu) = 2\\left[ y^{2-p}/((1-p)(2-p)) - y\\,\\mu^{1-p}/(1-p) + \\mu^{2-p}/(2-p) \\right]`;

    the log-likelihood uses the **saddlepoint approximation** (with the exact
    compound-Poisson zero mass at :math:`y = 0`) -- enough for :func:`aic` /
    :func:`bic` model comparison; the exact Dunn-Smyth series is a follow-up.

    Parameters
    ----------
    p : float, default=1.5
        The fixed Tweedie power :math:`p`, entering the variance as
        :math:`\\mu^p`.  Must satisfy :math:`1 < p < 2` (the compound
        Poisson-Gamma regime).

    Returns
    -------
    Family
        A Tweedie family with the given fixed power, a log link, and an
        estimated dispersion.
    """
    if not 1.0 < p < 2.0:
        raise ValueError(
            f'tweedie: power p must satisfy 1 < p < 2 (compound '
            f'Poisson-Gamma), got {p}.'
        )
    pw = float(p)
    one_m_p = 1.0 - pw  # in (-1, 0)
    two_m_p = 2.0 - pw  # in (0, 1)

    def variance(mu: Array) -> Array:
        return jnp.clip(mu, _EPS, None) ** pw

    def unit_deviance(y: Array, mu: Array) -> Array:
        ym = jnp.clip(y, 0.0, None)
        m = jnp.clip(mu, _EPS, None)
        return 2.0 * (
            ym**two_m_p / (one_m_p * two_m_p)
            - ym * m**one_m_p / one_m_p
            + m**two_m_p / two_m_p
        )

    def loglik(y: Array, mu: Array, dispersion: Array) -> Array:
        phi = jnp.clip(dispersion, _EPS, None)
        m = jnp.clip(mu, _EPS, None)
        ym = jnp.clip(y, 0.0, None)
        # Saddlepoint density for y > 0; exact compound-Poisson mass at y = 0.
        dev = unit_deviance(ym, m)
        ll_pos = -0.5 * jnp.log(2.0 * jnp.pi * phi * ym**pw + _EPS) - dev / (
            2.0 * phi
        )
        ll_zero = -(m**two_m_p) / (phi * two_m_p)  # log P(Y = 0)
        return jnp.where(y > 0.0, ll_pos, ll_zero)

    return Family(
        name='tweedie',
        has_fixed_dispersion=False,
        link=_log_link,
        linkinv=_safe_exp,
        mu_eta=_safe_exp,
        variance=variance,
        unit_deviance=unit_deviance,
        init_mu=lambda y: jnp.clip(y, 0.0, None) + 0.1,
        loglik=loglik,
        eta_bound=_ETA_MAX,
    )


# Default Tweedie (p = 1.5); for another power pass ``tweedie(p)``.
TWEEDIE = tweedie(1.5)


# ---------------------------------------------------------------------------
# Non-canonical links (compose onto a family via ``Family.with_link``)
# ---------------------------------------------------------------------------

IDENTITY_LINK = Link(
    name='identity', link=_identity, linkinv=_identity, mu_eta=_ones_like
)

LOG_LINK = Link(
    name='log',
    link=_log_link,
    linkinv=_safe_exp,
    mu_eta=_safe_exp,
    eta_bound=_ETA_MAX,
)

LOGIT_LINK = Link(
    name='logit', link=_logit, linkinv=_expit, mu_eta=_binomial_mu_eta
)


def _probit_linkinv(eta: Array) -> Array:
    return _norm.cdf(eta)


def _probit_mu_eta(eta: Array) -> Array:
    return jnp.clip(_norm.pdf(eta), _EPS, None)


PROBIT_LINK = Link(
    name='probit',
    link=lambda mu: _norm.ppf(jnp.clip(mu, _EPS, 1.0 - _EPS)),
    linkinv=_probit_linkinv,
    mu_eta=_probit_mu_eta,
)


def _cloglog_link(mu: Array) -> Array:
    m = jnp.clip(mu, _EPS, 1.0 - _EPS)
    return jnp.log(-jnp.log1p(-m))


def _cloglog_linkinv(eta: Array) -> Array:
    # 1 - exp(-exp(eta)); clamp eta so the inner exp cannot overflow.
    return -jnp.expm1(-jnp.exp(jnp.clip(eta, -_ETA_MAX, _ETA_MAX)))


def _cloglog_mu_eta(eta: Array) -> Array:
    e = jnp.clip(eta, -_ETA_MAX, _ETA_MAX)
    return jnp.clip(jnp.exp(e - jnp.exp(e)), _EPS, None)


CLOGLOG_LINK = Link(
    name='cloglog',
    link=_cloglog_link,
    linkinv=_cloglog_linkinv,
    mu_eta=_cloglog_mu_eta,
    eta_bound=_ETA_MAX,
)

SQRT_LINK = Link(
    name='sqrt',
    link=lambda mu: jnp.sqrt(jnp.clip(mu, 0.0, None)),
    linkinv=lambda eta: eta * eta,
    mu_eta=lambda eta: 2.0 * eta,
)


def _inverse_mu_eta(eta: Array) -> Array:
    e = jnp.clip(eta, _EPS, None)
    return -1.0 / (e * e)


# Reciprocal (the canonical Gamma link).  ``mu = 1 / eta`` requires ``eta > 0``;
# the IRLS clamps ``eta`` to a small positive floor (a sane init -- ``init_mu``
# is positive, so ``eta_0 = 1 / mu_0 > 0`` -- keeps it there in practice).
INVERSE_LINK = Link(
    name='inverse',
    link=lambda mu: 1.0 / jnp.clip(mu, _EPS, None),
    linkinv=lambda eta: 1.0 / jnp.clip(eta, _EPS, None),
    mu_eta=_inverse_mu_eta,
    eta_bound=float('inf'),
)


_LINKS: Mapping[str, Link] = {
    'identity': IDENTITY_LINK,
    'log': LOG_LINK,
    'logit': LOGIT_LINK,
    'probit': PROBIT_LINK,
    'cloglog': CLOGLOG_LINK,
    'sqrt': SQRT_LINK,
    'inverse': INVERSE_LINK,
}


def resolve_link(link: Union[str, Link]) -> Link:
    """Resolve a ``str`` name or a :class:`Link` instance to a :class:`Link`.

    Built-ins: ``'identity'`` / ``'log'`` / ``'logit'`` / ``'probit'`` /
    ``'cloglog'`` / ``'sqrt'`` / ``'inverse'``; or pass a :class:`Link` directly.

    Parameters
    ----------
    link : str or Link
        A built-in link name or a :class:`Link` instance.  A :class:`Link` is
        returned unchanged.

    Returns
    -------
    Link
        The resolved link.

    Raises
    ------
    ValueError
        If ``link`` is a string that names no built-in link.
    """
    if isinstance(link, Link):
        return link
    try:
        return _LINKS[link]
    except KeyError:
        raise ValueError(
            f'unknown link {link!r}; built-ins are {sorted(_LINKS)}, or pass '
            f'a Link instance.'
        ) from None


# The registry of built-ins -- the open-set extension point (callers may also
# pass any ``Family`` instance directly).
_FAMILIES: Mapping[str, Family] = {
    'gaussian': GAUSSIAN,
    'binomial': BINOMIAL,
    'poisson': POISSON,
    'gamma': GAMMA,
    'negbinomial': NEGBINOMIAL,
    'tweedie': TWEEDIE,
}


def resolve_family(family: Union[str, Family]) -> Family:
    """Resolve a ``str`` name (a built-in) or a :class:`Family` instance to a family.

    :func:`glm_fit` / :func:`gam_fit` accept either; a string is looked up in
    the built-in registry (``'gaussian'`` / ``'binomial'`` / ``'poisson'`` /
    ``'gamma'`` / ``'negbinomial'`` / ``'tweedie'``).  ``'negbinomial'`` resolves
    to the default :math:`\\alpha = 1`; for another dispersion pass
    :func:`negbinomial` directly.

    Parameters
    ----------
    family : str or Family
        A built-in family name or a :class:`Family` instance.  A :class:`Family`
        is returned unchanged.

    Returns
    -------
    Family
        The resolved family.

    Raises
    ------
    ValueError
        If ``family`` is a string that names no built-in family.
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
