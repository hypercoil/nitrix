# -*- coding: utf-8 -*-
"""Tests for the lengthscale-prior regularisers (``map_rho=`` penalties)."""

from __future__ import annotations

import jax
import numpy as np

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp

from nitrix.stats import (
    gp_fit,
    halfnormal_prior,
    hgp_fit,
    invgamma_prior,
    lognormal_prior,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gp_draw(rng, x, rho, noise=0.1, nu=2.5):
    from sklearn.gaussian_process.kernels import Matern

    K = Matern(length_scale=rho, nu=nu)(x[:, None]) + 1e-9 * np.eye(len(x))
    f = rng.multivariate_normal(np.zeros(len(x)), K)
    return f + noise * rng.standard_normal(len(x))


def _hier_data(rng, L=8, per=18, dev_amp=0.4, noise=0.1):
    t = np.linspace(0.0, 1.0, per)
    x = np.tile(t, L)
    group = np.repeat(np.arange(L), per)
    pop = np.sin(2 * np.pi * t)
    devs = [dev_amp * np.sin(2 * np.pi * t + ph)
            for ph in np.linspace(0.0, 2.0, L)]
    y = np.concatenate([pop + devs[g] + noise * rng.standard_normal(per)
                        for g in range(L)])
    return x, group, y


# ---------------------------------------------------------------------------
# 1. The -log p(rho) formulae (up to a constant)
# ---------------------------------------------------------------------------


def test_prior_formulae():
    rho = jnp.asarray([0.1, 0.5, 1.0, 2.0])
    r = np.asarray(rho)
    # half-normal
    hn = halfnormal_prior(sd=0.7)
    assert np.allclose(np.asarray(hn(rho)), 0.5 * (r / 0.7) ** 2)
    # inverse-gamma
    ig = invgamma_prior(a=3.0, b=0.5)
    assert np.allclose(np.asarray(ig(rho)), 4.0 * np.log(r) + 0.5 / r)
    # log-normal
    ln = lognormal_prior(mu=np.log(0.3), sd=0.5)
    assert np.allclose(
        np.asarray(ln(rho)),
        0.5 * ((np.log(r) - np.log(0.3)) / 0.5) ** 2 + np.log(r),
    )


def test_prior_directions():
    """The penalties pull in the documented directions."""
    hn = halfnormal_prior(sd=0.5)
    assert float(hn(jnp.asarray(2.0))) > float(hn(jnp.asarray(0.2)))  # caps large
    ig = invgamma_prior(a=2.0, b=0.3)
    assert float(ig(jnp.asarray(0.02))) > float(ig(jnp.asarray(0.5)))  # caps small
    ln = lognormal_prior(mu=np.log(0.3), sd=0.4)
    # minimised near the prior median exp(mu - sd^2) (close to 0.3)
    grid = np.linspace(0.05, 1.5, 200)
    vals = np.asarray(ln(jnp.asarray(grid)))
    assert 0.15 < grid[int(np.argmin(vals))] < 0.4


def test_prior_jittable():
    ig = invgamma_prior(a=3.0, b=0.5)
    out = jax.jit(ig)(jnp.asarray(0.4))
    assert np.isfinite(float(out))


# ---------------------------------------------------------------------------
# 2. Integration with gp_fit / hgp_fit
# ---------------------------------------------------------------------------


def test_invgamma_raises_lengthscale_in_gp_fit():
    """An inverse-gamma prior (penalising small rho) pulls rho-hat up vs ML."""
    rng = np.random.default_rng(0)
    n = 120
    x = np.sort(rng.uniform(0.0, 1.0, n))
    y = _gp_draw(rng, x, rho=0.1, noise=0.15)
    ml = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), rank=20, n_rho=32)
    mapf = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), rank=20, n_rho=32,
                  map_rho=invgamma_prior(a=8.0, b=8.0))  # mode ~ 0.9
    assert float(np.exp(mapf.theta[0, 2])) > float(np.exp(ml.theta[0, 2]))


def test_halfnormal_lowers_lengthscale_in_gp_fit():
    """A tight half-normal (penalising large rho) pulls rho-hat down vs ML."""
    rng = np.random.default_rng(1)
    n = 120
    x = np.sort(rng.uniform(0.0, 1.0, n))
    y = _gp_draw(rng, x, rho=0.5, noise=0.1)
    ml = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), rank=20, n_rho=32)
    mapf = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), rank=20, n_rho=32,
                  map_rho=halfnormal_prior(sd=0.05))
    assert float(np.exp(mapf.theta[0, 2])) < float(np.exp(ml.theta[0, 2]))


def test_prior_composes_with_hgp_fit():
    """A lengthscale prior composes with the hierarchical fit."""
    rng = np.random.default_rng(2)
    x, group, y = _hier_data(rng, L=8, per=16)
    ml = hgp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), jnp.asarray(group),
                 rank=8, n_rho=16)
    mapf = hgp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), jnp.asarray(group),
                   rank=8, n_rho=16, map_rho=invgamma_prior(a=10.0, b=10.0))
    assert float(np.exp(mapf.theta[0, 3])) > float(np.exp(ml.theta[0, 3]))
