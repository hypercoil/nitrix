# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for covariance, correlation, and derived measures
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import pingouin # required for pcorr test
import pandas as pd
from nitrix.functional.covariance import (
    cov,
    corr,
    partialcorr,
    pairedcov,
    pairedcorr,
    precision,
    conditionalcov,
    conditionalcorr,
    covariance,
    correlation,
    corrcoef,
    pcorr,
    ccov,
    ccorr,
    _prepare_weight_and_avg,
    _prepare_denomfact,
)


#TODO: Unit tests still needed for:
# - correctness of off-diagonal weighted covariance

@pytest.fixture(scope='module')
def x():
    return np.random.rand(100)

@pytest.fixture(scope='module')
def X():
    return np.random.rand(7, 100)

@pytest.fixture(scope='module')
def XM():
    return np.random.rand(200, 7, 100)

@pytest.fixture(scope='module')
def w():
    return np.random.rand(100)

@pytest.fixture(scope='module')
def wM():
    return np.random.rand(3, 100)

@pytest.fixture(scope='module')
def wMe(wM):
    return jax.vmap(jnp.diagflat, 0, 0)(wM)

@pytest.fixture(scope='module')
def W():
    return np.random.rand(100, 100)

@pytest.fixture(scope='module')
def Y():
    return np.random.rand(3, 100)

def approx(out, ref, tol=5e-7):
    return np.allclose(out, ref, atol=tol)

ofunc = cov
rfunc = np.cov


def test_normalisations():
    test_obs = 2000
    test_dim = 10
    offset = 1000
    offset2 = 1500
    bias = False
    ddof = None
    # variances should all be close to 1
    X = np.random.randn(test_dim, test_obs)
    X = (X - X.mean(-1, keepdims=True)) / X.std(-1, keepdims=True)

    # No weights
    w = None
    weight, w_type, w_sum, (avg,) = _prepare_weight_and_avg((X,), w)
    fact = _prepare_denomfact(w_sum, w_type, ddof, bias, weight)
    assert(
        np.allclose(np.diag(X @ X.T / fact).mean(), 1, atol=0.1))

    # Vector weights
    w = np.random.rand(test_obs)
    weight, w_type, w_sum, (avg,) = _prepare_weight_and_avg((X,), w)
    fact = _prepare_denomfact(w_sum, w_type, ddof, bias, weight)
    assert(
        np.allclose(np.diag(X * weight @ X.T / fact).mean(), 1, atol=0.1))

    # Diagonal weights
    w = np.diag(np.random.rand(test_obs))
    weight, w_type, w_sum, (avg,) = _prepare_weight_and_avg((X,), w)
    fact = _prepare_denomfact(w_sum, w_type, ddof, bias, weight)
    assert(
        np.allclose(np.diag(X @ weight @ X.T / fact).mean(), 1, atol=0.1))

    # Nondiagonal weights
    #TODO: This is not working yet / actually not tested
    w = np.zeros((test_obs, test_obs))
    rows, cols = np.diag_indices_from(w)
    w[(rows, cols)] = np.random.rand(test_obs)
    w[(rows[:-offset], cols[offset:])] = (
        3 * np.random.rand(test_obs - offset))
    w[(rows[:-offset2], cols[offset2:])] = (
        np.random.rand(test_obs - offset2))
    weight, w_type, w_sum, (avg,) = _prepare_weight_and_avg((X,), w)
    with pytest.raises(NotImplementedError):
        fact = _prepare_denomfact(w_sum, w_type, ddof, bias, weight)
    # out = np.diag(X @ weight @ X.T / fact).mean()


def covpattern(X, **args):
    out = ofunc(X, **args)
    ref = rfunc(X, **args)
    assert approx(out, ref)


@pytest.mark.parametrize('args', [
    {},
    {'rowvar': False},
    {'bias': True},
    {'ddof': 17},
])
def test_cov(X, args):
    covpattern(X, **args)


@pytest.mark.parametrize('args', [
    {},
    {'rowvar': False},
    {'bias': True},
    {'ddof': 17},
])
def test_cov_var(x, args):
    out = ofunc(x, **args)
    ref = rfunc(x, **args)
    assert approx(out, ref)


def test_cov_weighted(X, w):
    out = ofunc(X, weight=w)
    ref = rfunc(X, aweights=w)
    assert approx(out, ref)

    out = ofunc(X, weight=w, bias=True)
    ref = rfunc(X, aweights=w, bias=True)
    assert approx(out, ref)


def test_cov_multiweighted_1d(X, wMe, wM):
    out = ofunc(X, weight=wMe)
    ref = np.stack([
        rfunc(X, aweights=wM[i, :])
        for i in range(wM.shape[0])
    ])
    assert approx(out, ref)


def test_cov_multiweighted(X, wMe):
    out = cov(X, weight=wMe)
    ref = cov(X, weight=wMe[0, :])
    assert approx(out[0, :], ref)
    assert out.shape == (3, 7, 7)


def test_cov_Weighted(X, W):
    with pytest.raises(NotImplementedError):
        out = ofunc(X, weight=W)


def test_cov_multidim(XM, w):
    out = ofunc(XM, weight=w)
    ref = np.stack([
        rfunc(XM[i, :, :].squeeze(), aweights=w)
        for i in range(XM.shape[0])
    ])
    assert approx(out, ref)


@pytest.mark.parametrize('args', [
    {},
    {'bias': True},
    {'ddof': 17},
])
def test_paired(X, Y, args):
    out = pairedcov(X, Y, **args)
    ref = np.cov(np.concatenate([X, Y], -2), **args)[:7, -3:]
    assert approx(out, ref)


def test_corr(X):
    out = corr(X)
    ref = np.corrcoef(X)
    assert approx(out, ref)


@pytest.mark.parametrize('args', [
    {},
    {'bias': True},
    {'ddof': 17},
])
def test_pairedcorr(X, Y, args):
    out = pairedcorr(X, Y, **args)
    ref = corr(jnp.concatenate((X, Y)), **args)[:7, 7:]
    assert approx(out, ref)


def test_precision(X):
    out = precision(X)
    ref = np.linalg.inv(cov(X))
    assert approx(out, ref)

    out = precision(X, l2=0.1)
    ref = np.linalg.inv(cov(X) + 0.1 * np.eye(7))
    assert approx(out, ref)

    assert jnp.any(jnp.isnan(
        precision(jnp.ones((10, 10)))
    ))
    assert not jnp.any(jnp.isnan(precision(X.T, l2=0.1)))
    assert not jnp.any(jnp.isnan(
        precision(X.T, require_nonsingular=False)
    ))


def test_pcorr(X):
    out = partialcorr(X)
    ref = pd.DataFrame(X.T).pcorr().values
    assert approx(out, ref)


def test_ccov(X, Y):
    """
    Verify equivalence of the Schur complement approach and fit-based
    confound regression.
    """
    out = conditionalcov(X, Y)
    ref = jnp.linalg.pinv(
        precision(jnp.concatenate((X, Y), -2))[:7, :7])
    assert approx(out, ref)
    Y_intercept = np.concatenate([Y, np.ones((1, 100))])
    ref = np.cov(
        X - np.linalg.lstsq(Y_intercept.T, X.T, rcond=None)[0].T @ Y_intercept
    )
    assert approx(out, ref)


def test_ccorr(X, Y):
    """
    Verify equivalence of the Schur complement approach and fit-based
    confound regression.
    """
    Y_intercept = np.concatenate([Y, np.ones((1, 100))])
    out = conditionalcorr(X, Y)
    ref = np.corrcoef(
        X - np.linalg.lstsq(Y_intercept.T, X.T, rcond=None)[0].T @ Y_intercept
    )
    assert approx(out, ref)
    out = jax.jit(conditionalcorr)(X, Y)
    assert approx(out, ref)


def test_aliases(X, Y):
    out = covariance(X)
    ref = cov(X)
    assert approx(out, ref)

    out = correlation(X)
    ref = corr(X)
    assert approx(out, ref)

    out = corrcoef(X)
    ref = corr(X)
    assert approx(out, ref)

    out = pcorr(X)
    ref = partialcorr(X)
    assert approx(out, ref)

    out = ccov(X, Y)
    ref = conditionalcov(X, Y)
    assert approx(out, ref)

    out = ccorr(X, Y)
    ref = conditionalcorr(X, Y)
    assert approx(out, ref)
