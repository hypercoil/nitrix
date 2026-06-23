# -*- coding: utf-8 -*-
"""Cross-library parity of the nitrix GP smooths against R ``mgcv``.

``mgcv``'s ``s(x, bs="gp", m=c(3, rho))`` is Matern-3/2 *kriging* (a low-rank
eigen-truncation) -- the same construction as :func:`nitrix.stats.basis.gp_basis`
-- so it anchors the kriging basis directly.  For the Hilbert-space
:func:`~nitrix.stats.basis.hsgp_basis` it is a *cross-basis* check: a different
reduced-rank expansion of the same Matern GP, so the REML-fitted smooths agree to
sub-percent RMSE on smooth data without being bit-identical.

These run only where ``Rscript`` + ``mgcv`` are available (skipped in their
absence, e.g. a Python-only CI); the unconditional HSGP correctness anchor is the
scikit-learn exact-GP comparison in ``test_hsgp.py``.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import tempfile

import jax
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp

from nitrix.stats.basis import gp_basis, gp_factor_smooth, hsgp_basis
from nitrix.stats.gam import gam_fit, smooth_partial_effect
from nitrix.stats.gp import gp_fit, gp_predict


@functools.lru_cache(maxsize=1)
def _rscript() -> str | None:
    """Path to an ``Rscript`` that can load ``mgcv``; ``None`` if unavailable."""
    cand = shutil.which('Rscript') or '/scratch/nperf/renv/bin/Rscript'
    if not (cand and os.path.exists(cand)):
        return None
    try:
        r = subprocess.run(
            [cand, '-e', 'suppressMessages(library(mgcv)); cat("ok")'],
            capture_output=True, text=True, timeout=90,
        )
        return cand if 'ok' in r.stdout else None
    except Exception:
        return None


requires_mgcv = pytest.mark.skipif(
    _rscript() is None, reason='Rscript + mgcv not available'
)


def _mgcv_gp_fitted(x, y, *, k, rho, model=3):
    """Fitted values of ``gam(y ~ s(x, bs='gp', k=k, m=c(model, rho)))`` (REML).

    ``model=3`` selects mgcv's Matern-3/2 correlation (``4`` -> Matern-5/2).
    """
    rs = _rscript()
    assert rs is not None
    with tempfile.TemporaryDirectory() as d:
        np.savetxt(
            os.path.join(d, 'xy.csv'), np.c_[x, y],
            delimiter=',', header='x,y', comments='',
        )
        script = os.path.join(d, 'fit.R')
        out = os.path.join(d, 'out.csv')
        with open(script, 'w') as fh:
            fh.write(
                'suppressMessages(library(mgcv))\n'
                f'df <- read.csv("{d}/xy.csv")\n'
                f'b <- gam(y ~ s(x, bs="gp", k={k}, m=c({model}, {rho})),'
                ' data=df, method="REML")\n'
                f'write.csv(data.frame(fit=fitted(b)), "{out}", row.names=FALSE)\n'
            )
        p = subprocess.run(
            [rs, script], capture_output=True, text=True, timeout=180
        )
        if not os.path.exists(out):
            raise RuntimeError(f'mgcv fit failed:\n{p.stderr}')
        return np.loadtxt(out, delimiter=',', skiprows=1)


def _nitrix_gp_fitted(basis, x, y):
    """Fitted values of a single-smooth gam_fit (intercept + smooth)."""
    res = gam_fit(jnp.asarray(y[None, :]), [basis])
    eff, _ = smooth_partial_effect(res, 0, basis, jnp.asarray(x))
    fitted = np.asarray(res.coef[:, 0:1]) + np.asarray(eff)
    return fitted[0]


def _data(seed=7, n=150):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, n))
    truth = np.sin(2 * np.pi * x) + 0.3 * np.sin(6 * np.pi * x)
    y = truth + 0.1 * rng.standard_normal(n)
    return x, y, truth


@requires_mgcv
def test_gp_basis_matches_mgcv_kriging():
    """Kriging gp_basis (Matern-3/2) vs mgcv s(x, bs='gp', m=c(3, rho)) -- the
    same construction; REML-fitted smooths agree to sub-percent RMSE."""
    x, y, truth = _data()
    k, rho = 20, 0.15
    f_nitrix = _nitrix_gp_fitted(gp_basis(jnp.asarray(x), k, rho=rho), x, y)
    f_mgcv = _mgcv_gp_fitted(x, y, k=k, rho=rho, model=3)
    assert np.corrcoef(f_nitrix, f_mgcv)[0, 1] > 0.998
    assert np.sqrt(np.mean((f_nitrix - f_mgcv) ** 2)) < 0.025
    # both genuinely recover the smooth (the agreement is not joint failure)
    assert np.sqrt(np.mean((f_nitrix - truth) ** 2)) < 0.06


@requires_mgcv
def test_hsgp_cross_checks_mgcv():
    """HSGP (Matern-3/2) vs mgcv kriging GP -- different reduced-rank bases for
    the same Matern GP; fitted smooths agree to sub-percent RMSE."""
    x, y, truth = _data()
    k, rho = 20, 0.15
    f_hsgp = _nitrix_gp_fitted(
        hsgp_basis(jnp.asarray(x), k, kernel='matern32', rho=rho), x, y
    )
    f_mgcv = _mgcv_gp_fitted(x, y, k=k, rho=rho, model=3)
    assert np.corrcoef(f_hsgp, f_mgcv)[0, 1] > 0.998
    assert np.sqrt(np.mean((f_hsgp - f_mgcv) ** 2)) < 0.025
    assert np.sqrt(np.mean((f_hsgp - truth) ** 2)) < 0.06


def _mgcv_fs_fitted(x, y, fac, *, k, rho):
    """Fitted values of mgcv's factor-smooth GAM ``y ~ s(x, bs='gp') + s(x, fac,
    bs='fs')`` (REML) -- a population GP smooth plus a shared-wiggliness
    factor-smooth (the GS hierarchical structure)."""
    rs = _rscript()
    assert rs is not None
    with tempfile.TemporaryDirectory() as d:
        np.savetxt(
            os.path.join(d, 'xy.csv'), np.c_[x, y, fac],
            delimiter=',', header='x,y,fac', comments='',
        )
        script = os.path.join(d, 'fit.R')
        out = os.path.join(d, 'out.csv')
        with open(script, 'w') as fh:
            fh.write(
                'suppressMessages(library(mgcv))\n'
                f'df <- read.csv("{d}/xy.csv")\n'
                'df$fac <- factor(df$fac)\n'
                f'b <- gam(y ~ s(x, bs="gp", k={k}, m=c(3, {rho})) + '
                f's(x, fac, bs="fs", k={k}, m=1), data=df, method="REML")\n'
                f'write.csv(data.frame(fit=fitted(b)), "{out}", row.names=FALSE)\n'
            )
        p = subprocess.run(
            [rs, script], capture_output=True, text=True, timeout=240
        )
        if not os.path.exists(out):
            raise RuntimeError(f'mgcv fs fit failed:\n{p.stderr}')
        return np.loadtxt(out, delimiter=',', skiprows=1)


def _hier_fs_data(seed=11, n_groups=6, per=22):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, per)
    pop = np.sin(2 * np.pi * t)
    devs = [0.4 * np.sin(2 * np.pi * t + ph)
            for ph in np.linspace(0.0, 2.0, n_groups)]
    x = np.tile(t, n_groups)
    fac = np.repeat(np.arange(n_groups), per)
    y = np.concatenate([pop + devs[g] + 0.1 * rng.standard_normal(per)
                        for g in range(n_groups)])
    return x, y, fac


@requires_mgcv
def test_gp_factor_smooth_cross_checks_mgcv_fs():
    """The GS hierarchical GP (``hsgp_basis`` population + ``gp_factor_smooth``)
    cross-checks mgcv's factor-smooth ``s(x, fac, bs='fs')``.

    The marginal bases differ (HSGP-gp vs mgcv's fs marginal), so this is a
    structural check that the shared-wiggliness per-group factor-smooth recovers
    the same fitted surface -- the per-observation fitted values track closely."""
    rho, k = 0.2, 12
    x, y, fac = _hier_fs_data()
    pop_b = hsgp_basis(jnp.asarray(x), k, kernel='matern52', rho=rho)
    fac_b = gp_factor_smooth(jnp.asarray(x), jnp.asarray(fac), 10,
                             kernel='matern52', rho=rho)
    res = gam_fit(jnp.asarray(y[None, :]), [pop_b, fac_b])
    eff_pop, _ = smooth_partial_effect(res, 0, pop_b, jnp.asarray(x))
    eff_fac, _ = smooth_partial_effect(
        res, 1, fac_b, (jnp.asarray(x), jnp.asarray(fac))
    )
    f_nitrix = (
        np.asarray(res.coef[:, 0:1]) + np.asarray(eff_pop) + np.asarray(eff_fac)
    )[0]
    f_mgcv = _mgcv_fs_fitted(x, y, fac, k=k, rho=rho)
    assert np.corrcoef(f_nitrix, f_mgcv)[0, 1] > 0.95


def _mgcv_gp_poisson_fitted(x, y, *, k, rho):
    """Response-scale fitted rates of a Poisson GP smooth at a fixed ``rho``
    (``gam(y ~ s(x, bs='gp', m=c(3, rho)), family=poisson(), method='REML')``)."""
    rs = _rscript()
    assert rs is not None
    with tempfile.TemporaryDirectory() as d:
        np.savetxt(
            os.path.join(d, 'xy.csv'), np.c_[x, y],
            delimiter=',', header='x,y', comments='',
        )
        script = os.path.join(d, 'fit.R')
        out = os.path.join(d, 'out.csv')
        with open(script, 'w') as fh:
            fh.write(
                'suppressMessages(library(mgcv))\n'
                f'df <- read.csv("{d}/xy.csv")\n'
                f'b <- gam(y ~ s(x, bs="gp", k={k}, m=c(3, {rho})), '
                'family=poisson(), data=df, method="REML")\n'
                f'write.csv(data.frame(fit=fitted(b)), "{out}", row.names=FALSE)\n'
            )
        p = subprocess.run(
            [rs, script], capture_output=True, text=True, timeout=180
        )
        if not os.path.exists(out):
            raise RuntimeError(f'mgcv poisson gp fit failed:\n{p.stderr}')
        return np.loadtxt(out, delimiter=',', skiprows=1)


@requires_mgcv
def test_gp_glm_poisson_cross_checks_mgcv():
    """CV2 Phase 1: the nitrix Poisson GP (PQL-REML ``rho``) cross-checks mgcv's
    Poisson GP smooth evaluated at the nitrix-estimated ``rho`` -- the
    response-scale fitted rates track closely, and both recover the true rate.
    (A structural check: nitrix's PQL ``rho``-selection differs from mgcv's LAML,
    so ``rho`` is fed in rather than compared.)"""
    rng = np.random.default_rng(13)
    n = 160
    x = np.sort(rng.uniform(0.0, 1.0, n))
    eta = 0.8 + np.sin(2 * np.pi * x)
    y = rng.poisson(np.exp(eta)).astype(float)
    res = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), family='poisson',
                 n_rho=16, n_pql=8)
    rho_hat = float(np.exp(res.theta[0, 2]))
    mu_nitrix, _ = gp_predict(res, jnp.asarray(x), type='response')
    f_nitrix = np.asarray(mu_nitrix[0])
    f_mgcv = _mgcv_gp_poisson_fitted(x, y, k=20, rho=rho_hat)
    assert np.corrcoef(f_nitrix, f_mgcv)[0, 1] > 0.95
    assert np.corrcoef(f_nitrix, np.exp(eta))[0, 1] > 0.9
