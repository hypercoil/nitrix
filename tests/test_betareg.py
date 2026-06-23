# -*- coding: utf-8 -*-
"""Tests for beta regression (``nitrix.stats.beta_fit``, v3 §4).

Proportions in ``(0, 1)`` modelled as ``Beta(mu phi, (1-mu) phi)`` with a logit
mean and an estimated precision.  Anchored against ``statsmodels`` ``BetaModel``
(the same Ferrari-Cribari-Neto MLE): coefficients, precision, standard errors
(the joint ``(beta, phi)`` information), and the log-likelihood must all match.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.stats import BetaResult, beta_fit

sm_betareg = pytest.importorskip('statsmodels.othermod.betareg')


def _sim(seed, n=300, phi=8.0):
    rng = np.random.default_rng(seed)
    X = np.c_[np.ones(n), rng.standard_normal(n), rng.standard_normal(n)]
    mu = 1.0 / (1.0 + np.exp(-(0.4 + 0.7 * X[:, 1] - 0.5 * X[:, 2])))
    y = rng.beta(mu * phi, (1.0 - mu) * phi)
    return X, y


@pytest.mark.parametrize('seed', range(4))
def test_beta_matches_statsmodels(seed):
    X, y = _sim(seed)
    res = beta_fit(jnp.asarray(y[None, :]), jnp.asarray(X))
    smf = sm_betareg.BetaModel(y, X).fit()
    np.testing.assert_allclose(np.asarray(res.coef[0]), smf.params[:3], atol=1e-5)
    np.testing.assert_allclose(
        float(res.precision[0]), np.exp(smf.params[3]), rtol=1e-4
    )
    se = np.sqrt(np.diag(np.asarray(res.cov_unscaled[0])))
    np.testing.assert_allclose(se, smf.bse[:3], rtol=2e-2)
    np.testing.assert_allclose(float(res.log_lik[0]), smf.llf, atol=1e-3)


def test_beta_recovers_precision():
    """High precision -> tight beta -> recovered phi tracks the truth."""
    X, y = _sim(0, n=600, phi=20.0)
    res = beta_fit(jnp.asarray(y[None, :]), jnp.asarray(X))
    assert abs(float(res.precision[0]) - 20.0) < 6.0  # noisy but in range
    assert float(res.precision[0]) > 0


def test_beta_shapes_and_pytree():
    X, y = _sim(1)
    V = 5
    res = beta_fit(jnp.asarray(np.tile(y, (V, 1))), jnp.asarray(X))
    assert res.coef.shape == (V, 3)
    assert res.precision.shape == (V,)
    assert res.cov_unscaled.shape == (V, 3, 3)
    assert res.log_lik.shape == (V,)
    assert res.n_obs == X.shape[0]
    leaves, treedef = jax.tree_util.tree_flatten(res)
    res2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(res2, BetaResult) and res2.n_obs == res.n_obs


def test_beta_shape_validation():
    X, y = _sim(0, n=100)
    with pytest.raises(ValueError, match='must match N'):
        beta_fit(jnp.asarray(y[None, :-3]), jnp.asarray(X))


def test_beta_out_of_unit_interval_warns():
    """Round 4: responses outside (0, 1) are clipped to the boundary; warn rather
    than silently corrupt the data."""
    import warnings

    X, y = _sim(0, n=80)
    y_in = np.clip(y, 1e-3, 1 - 1e-3)
    y_bad = y_in.copy()
    y_bad[0] = 1.5  # out of (0, 1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        beta_fit(jnp.asarray(y_in[None, :]), jnp.asarray(X), n_iter=5)
        assert not any('outside the open interval' in str(m.message) for m in w)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        beta_fit(jnp.asarray(y_bad[None, :]), jnp.asarray(X), n_iter=5)
        assert any('outside the open interval' in str(m.message) for m in w)
