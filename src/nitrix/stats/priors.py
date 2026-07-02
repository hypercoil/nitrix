# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Lengthscale priors as REML regularisers (the ``map_rho=`` penalties).

A small, curated set of priors on the Gaussian-process lengthscale
:math:`\\rho`, each returned as a callable :math:`\\rho \\mapsto -\\log p(\\rho)`
suitable for the ``map_rho=`` argument of :func:`nitrix.stats.gp_fit` and
:func:`nitrix.stats.hgp_fit`.  This is the lightweight *penalised / maximum a
posteriori* lengthscale: the REML objective :math:`-2 l_R(\\rho)` gains
:math:`-2 \\log p(\\rho)`, so the selected :math:`\\rho` is the *maximum a
posteriori* value rather than the pure REML maximiser.  (A full posterior over
:math:`\\rho` is a separate concern, addressed by dedicated probabilistic
programming tools such as Stan or ``brms``.)

Each builder returns :math:`-\\log p(\\rho)` up to an additive constant (the
normaliser is irrelevant to the argmin), so they compose directly::

    from nitrix.stats import gp_fit, invgamma_prior
    res = gp_fit(Y, x, map_rho=invgamma_prior(a=3.0, b=0.5))

The penalties are pure JAX (they ride the jitted :math:`\\rho`-search),
elementwise on a scalar :math:`\\rho`.

Which prior?
------------
- :func:`halfnormal_prior` -- shrinks large :math:`\\rho` (keeps the GP from
  going near-linear); a soft upper scale.
- :func:`invgamma_prior` -- penalises small :math:`\\rho` (the usual
  penalised-complexity-style regulariser that stops the lengthscale collapsing
  into noise-fitting); the practical default when :math:`\\rho` is weakly
  identified.
- :func:`lognormal_prior` -- centres :math:`\\rho` on a prior median
  :math:`\\exp(\\mu)` with multiplicative spread; symmetric in
  :math:`\\log \\rho`.
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
    """Build a half-normal lengthscale prior that penalises large ``rho``.

    The prior density is
    :math:`p(\\rho) \\propto \\exp(-\\rho^2 / (2\\,\\sigma^2))` on :math:`\\rho > 0`,
    so the returned penalty is
    :math:`-\\log p(\\rho) = \\rho^2 / (2\\,\\sigma^2)` (up to an additive
    constant): a soft cap that pulls the estimate down, away from over-smooth
    (near-linear) fits.

    Parameters
    ----------
    sd : float
        The half-normal scale :math:`\\sigma` (must be positive).  Smaller
        ``sd`` is a stronger pull toward small :math:`\\rho`; a sensible value
        is on the order of the covariate range.

    Returns
    -------
    PriorFn
        A callable mapping a scalar lengthscale :math:`\\rho` (shape ``()``) to
        the scalar negative log-density :math:`-\\log p(\\rho)` (shape ``()``),
        suitable for the ``map_rho=`` argument of :func:`nitrix.stats.gp_fit`
        and :func:`nitrix.stats.hgp_fit`.
    """
    if not sd > 0:
        raise ValueError(f'halfnormal_prior: sd={sd} must be > 0.')

    def neg_log_p(rho: Float[Array, '']) -> Float[Array, '']:
        return 0.5 * (rho / sd) ** 2

    return neg_log_p


def invgamma_prior(a: float, b: float) -> PriorFn:
    """Build an inverse-gamma lengthscale prior that penalises small ``rho``.

    The prior density is
    :math:`p(\\rho) \\propto \\rho^{-a-1} \\exp(-b / \\rho)` on :math:`\\rho > 0`,
    so the returned penalty is
    :math:`-\\log p(\\rho) = (a + 1) \\log \\rho + b / \\rho` (up to an additive
    constant).  The :math:`b / \\rho` term diverges as :math:`\\rho \\to 0`, so
    this pulls the estimate up, away from a lengthscale that collapses into
    noise-fitting -- the usual penalised-complexity-style regulariser for a
    weakly-identified :math:`\\rho` (e.g. sparse or short series).

    Parameters
    ----------
    a : float
        Shape parameter :math:`a` (must be positive).
    b : float
        Scale parameter :math:`b` (must be positive).  Together with ``a`` it
        sets the prior mode :math:`b / (a + 1)`; pick ``a`` and ``b`` so that
        is a plausibly-small lengthscale.

    Returns
    -------
    PriorFn
        A callable mapping a scalar lengthscale :math:`\\rho` (shape ``()``) to
        the scalar negative log-density :math:`-\\log p(\\rho)` (shape ``()``),
        suitable for the ``map_rho=`` argument of :func:`nitrix.stats.gp_fit`
        and :func:`nitrix.stats.hgp_fit`.
    """
    if not a > 0 or not b > 0:
        raise ValueError(f'invgamma_prior: a={a}, b={b} must both be > 0.')

    def neg_log_p(rho: Float[Array, '']) -> Float[Array, '']:
        return (a + 1.0) * jnp.log(rho) + b / rho

    return neg_log_p


def lognormal_prior(mu: float, sd: float) -> PriorFn:
    """Build a log-normal lengthscale prior with median ``exp(mu)``.

    The prior density is
    :math:`p(\\rho) \\propto (1/\\rho) \\exp(-(\\log \\rho - \\mu)^2 / (2\\,\\sigma^2))`
    on :math:`\\rho > 0`, so the returned penalty is
    :math:`-\\log p(\\rho) = (\\log \\rho - \\mu)^2 / (2\\,\\sigma^2) + \\log \\rho`
    (up to an additive constant): a pull that is symmetric in
    :math:`\\log \\rho`, toward the prior median :math:`\\exp(\\mu)` with
    multiplicative spread :math:`\\sigma`.

    Parameters
    ----------
    mu : float
        Prior mean :math:`\\mu` of :math:`\\log \\rho` (so :math:`\\exp(\\mu)` is
        the prior median lengthscale).
    sd : float
        Prior standard deviation :math:`\\sigma` of :math:`\\log \\rho` (must be
        positive).

    Returns
    -------
    PriorFn
        A callable mapping a scalar lengthscale :math:`\\rho` (shape ``()``) to
        the scalar negative log-density :math:`-\\log p(\\rho)` (shape ``()``),
        suitable for the ``map_rho=`` argument of :func:`nitrix.stats.gp_fit`
        and :func:`nitrix.stats.hgp_fit`.
    """
    if not sd > 0:
        raise ValueError(f'lognormal_prior: sd={sd} must be > 0.')

    def neg_log_p(rho: Float[Array, '']) -> Float[Array, '']:
        log_rho = jnp.log(rho)
        return 0.5 * ((log_rho - mu) / sd) ** 2 + log_rho

    return neg_log_p
