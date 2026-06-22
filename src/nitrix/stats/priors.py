# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Lengthscale priors as REML regularisers (the ``map_rho=`` penalties).

A small, curated set of priors on the GP lengthscale ``rho``, each returned as a
callable ``rho -> -log p(rho)`` suitable for the ``map_rho=`` argument of
:func:`nitrix.stats.gp_fit` and :func:`nitrix.stats.hgp_fit`.  This is the
lightweight **penalised / MAP** lengthscale within scope (a): the REML objective
``-2 l_R(rho)`` gains ``-2 log p(rho)``, so the selected ``rho`` is the *maximum a
posteriori* value rather than the pure REML maximiser.  (A full *posterior* over
``rho`` is scope (b) -- Stan / ``brms`` territory.)

Each builder returns ``-log p(rho)`` **up to an additive constant** (the
normaliser is irrelevant to the argmin), so they compose directly::

    from nitrix.stats import gp_fit, invgamma_prior
    res = gp_fit(Y, x, map_rho=invgamma_prior(a=3.0, b=0.5))

The penalties are pure JAX (they ride the jitted ``rho``-search), elementwise on a
scalar ``rho``.

Which prior?
------------
- :func:`halfnormal_prior` -- shrinks **large** ``rho`` (keeps the GP from going
  near-linear); a soft upper scale.
- :func:`invgamma_prior` -- penalises **small** ``rho`` (the usual
  penalised-complexity-style regulariser that stops the lengthscale collapsing
  into noise-fitting); the practical default when ``rho`` is weakly identified.
- :func:`lognormal_prior` -- centres ``rho`` on a prior **median** ``exp(mu)`` with
  multiplicative spread; symmetric in ``log rho``.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = [
    'PriorFn',
    'halfnormal_prior',
    'invgamma_prior',
    'lognormal_prior',
]

#: A lengthscale penalty: ``rho -> -log p(rho)`` (the ``map_rho=`` contract).
PriorFn = Callable[[Float[Array, '']], Float[Array, '']]


def halfnormal_prior(sd: float) -> PriorFn:
    """Half-normal prior on the lengthscale ``rho`` (penalises large ``rho``).

    ``p(rho) ~ exp(-rho**2 / (2 sd**2))`` on ``rho > 0``, so
    ``-log p(rho) = rho**2 / (2 sd**2)`` (up to a constant): a soft cap that pulls
    the estimate **down**, away from over-smooth (near-linear) fits.

    Parameters
    ----------
    sd
        The half-normal scale.  Smaller ``sd`` is a stronger pull toward small
        ``rho``; a sensible value is on the order of the covariate range.
    """
    if not sd > 0:
        raise ValueError(f'halfnormal_prior: sd={sd} must be > 0.')

    def neg_log_p(rho: Float[Array, '']) -> Float[Array, '']:
        return 0.5 * (rho / sd) ** 2

    return neg_log_p


def invgamma_prior(a: float, b: float) -> PriorFn:
    """Inverse-gamma prior on the lengthscale ``rho`` (penalises small ``rho``).

    ``p(rho) ~ rho**(-a-1) exp(-b / rho)`` on ``rho > 0``, so
    ``-log p(rho) = (a + 1) log rho + b / rho`` (up to a constant).  The ``b/rho``
    term diverges as ``rho -> 0``, so this pulls the estimate **up**, away from a
    lengthscale that collapses into noise-fitting -- the usual
    penalised-complexity-style regulariser for a weakly-identified ``rho`` (e.g.
    sparse or short series).

    Parameters
    ----------
    a, b
        Shape and scale (``a, b > 0``).  The prior mode is ``b / (a + 1)``; pick
        ``a, b`` so that is a plausibly-small lengthscale.
    """
    if not a > 0 or not b > 0:
        raise ValueError(f'invgamma_prior: a={a}, b={b} must both be > 0.')

    def neg_log_p(rho: Float[Array, '']) -> Float[Array, '']:
        return (a + 1.0) * jnp.log(rho) + b / rho

    return neg_log_p


def lognormal_prior(mu: float, sd: float) -> PriorFn:
    """Log-normal prior on the lengthscale ``rho`` (median ``exp(mu)``).

    ``p(rho) ~ (1/rho) exp(-(log rho - mu)**2 / (2 sd**2))`` on ``rho > 0``, so
    ``-log p(rho) = (log rho - mu)**2 / (2 sd**2) + log rho`` (up to a constant):
    a symmetric-in-``log rho`` pull toward the prior median ``exp(mu)`` with
    multiplicative spread ``sd``.

    Parameters
    ----------
    mu
        Prior mean of ``log rho`` (so ``exp(mu)`` is the prior median lengthscale).
    sd
        Prior standard deviation of ``log rho`` (``> 0``).
    """
    if not sd > 0:
        raise ValueError(f'lognormal_prior: sd={sd} must be > 0.')

    def neg_log_p(rho: Float[Array, '']) -> Float[Array, '']:
        log_rho = jnp.log(rho)
        return 0.5 * ((log_rho - mu) / sd) ** 2 + log_rho

    return neg_log_p
