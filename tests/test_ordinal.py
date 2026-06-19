# -*- coding: utf-8 -*-
"""Tests for ordinal regression (``ordinal_fit``, v3 §4).

The cumulative-link / proportional-odds model ``P(y <= k) = F(theta_k - X beta)``
for an ordered categorical ``y``.  Anchored against ``statsmodels``'
``OrderedModel`` (the same MLE): coefficients, ordered thresholds, standard
errors, and the log-likelihood.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import expit

jax.config.update('jax_enable_x64', True)

from nitrix.stats import OrdinalResult, ordinal_fit

OrderedModel = pytest.importorskip(
    'statsmodels.miscmodels.ordinal_model'
).OrderedModel


def _sim(seed, n=600, K=4, beta=(0.8, -0.5), theta=(-1.0, 0.3, 1.5)):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(beta)))
    cum = expit(np.asarray(theta)[None, :] - (X @ np.asarray(beta))[:, None])
    P = np.diff(np.c_[np.zeros(n), cum, np.ones(n)], axis=1)
    y = np.array([rng.choice(K, p=P[i]) for i in range(n)])
    return X, y


def _sm_thresholds(sm, p):
    return np.r_[sm.params[p], sm.params[p] + np.cumsum(np.exp(sm.params[p + 1 :]))]


@pytest.mark.parametrize('seed', range(3))
def test_ordinal_matches_statsmodels(seed):
    X, y = _sim(seed)
    res = ordinal_fit(jnp.asarray(y[None, :]), jnp.asarray(X), n_classes=4, n_iter=60)
    sm = OrderedModel(y, X, distr='logit').fit(method='bfgs', disp=False)
    np.testing.assert_allclose(np.asarray(res.coef[0]), sm.params[:2], atol=2e-3)
    np.testing.assert_allclose(
        np.asarray(res.thresholds[0]), _sm_thresholds(sm, 2), atol=2e-3
    )
    assert abs(float(res.log_lik[0]) - sm.llf) < 1e-2
    se = np.sqrt(np.diag(np.asarray(res.cov_coef[0])))
    np.testing.assert_allclose(se, sm.bse[:2], rtol=0.05)


def test_ordinal_thresholds_are_ordered():
    X, y = _sim(1, K=5, beta=(0.6,), theta=(-1.2, -0.2, 0.5, 1.4))
    res = ordinal_fit(jnp.asarray(y[None, :]), jnp.asarray(X), n_classes=5, n_iter=60)
    th = np.asarray(res.thresholds[0])
    assert np.all(np.diff(th) > 0)  # strictly increasing cutpoints


def test_ordinal_probit_link_runs_and_differs():
    X, y = _sim(2)
    logit = ordinal_fit(jnp.asarray(y[None, :]), jnp.asarray(X), n_classes=4, n_iter=60)
    probit = ordinal_fit(
        jnp.asarray(y[None, :]), jnp.asarray(X), n_classes=4, link='probit', n_iter=60
    )
    # probit coefficients are on a tighter scale (~ logit / 1.7) but same sign
    assert np.all(np.sign(np.asarray(probit.coef[0])) == np.sign(np.asarray(logit.coef[0])))
    assert np.abs(probit.coef[0, 0]) < np.abs(logit.coef[0, 0])


def test_ordinal_shapes_and_pytree():
    X, y = _sim(0)
    V = 4
    res = ordinal_fit(
        jnp.asarray(np.tile(y, (V, 1))), jnp.asarray(X), n_classes=4
    )
    assert res.coef.shape == (V, 2)
    assert res.thresholds.shape == (V, 3)
    assert res.cov_coef.shape == (V, 2, 2)
    assert res.log_lik.shape == (V,)
    assert res.n_obs == X.shape[0] and res.n_classes == 4
    leaves, treedef = jax.tree_util.tree_flatten(res)
    res2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(res2, OrdinalResult) and res2.n_classes == 4


def test_ordinal_validation():
    X, y = _sim(0, n=120)
    with pytest.raises(ValueError, match='n_classes'):
        ordinal_fit(jnp.asarray(y[None, :]), jnp.asarray(X), n_classes=1)
    with pytest.raises(ValueError, match='must match N'):
        ordinal_fit(jnp.asarray(y[None, :-2]), jnp.asarray(X), n_classes=4)
    with pytest.raises(ValueError, match='logit'):
        ordinal_fit(jnp.asarray(y[None, :]), jnp.asarray(X), n_classes=4, link='cauchit')
