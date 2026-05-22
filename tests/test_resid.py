# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for residualisation
"""
from functools import partial

import pytest
import hypothesis
from hypothesis import given, note, strategies as st
from hypothesis.extra import numpy as npst
import jax
import jax.numpy as jnp
from nitrix.linalg import residualise
from nitrix._internal.testutil import cfg_variants_test


run_variants = cfg_variants_test(
    residualise,
    jit_params={'static_argnames': ('l2', 'return_mode', 'rowvar')},
)

# The SVD path (``method='svd'``) is the robust solver: it stays exact for
# rank-deficient / ill-conditioned designs (``p -> obs`` and ``p > obs``)
# where the default Cholesky path returns NaN.  ``method`` is baked into the
# partial so it is a trace-time constant under the jitted variant.
run_variants_svd = cfg_variants_test(
    partial(residualise, method='svd'),
    jit_params={'static_argnames': ('l2', 'return_mode', 'rowvar')},
)


#TODO: This is a temporary implementation of the cosine kernel. We should
# replace it with the nitrix implementation once it is available.
def cosine_kernel(x, y, rowvar=True):
    if not rowvar:
        x = x.swapaxes(-1, -2)
        y = y.swapaxes(-1, -2)
    return x @ y.swapaxes(-1, -2) / (
        jnp.linalg.norm(x, axis=-1, keepdims=True) *
        jnp.linalg.norm(y, axis=-1, keepdims=True).swapaxes(-1, -2)
    )


@st.composite
def generate_valid_arrays(
    draw,
    allow_p_greater_n: bool = False,
    allow_p_eq_n: bool = True,
    well_conditioned: bool = False,
):
    batch_rank = draw(st.integers(min_value=0, max_value=3))
    batch_shape = draw(npst.array_shapes(min_dims=batch_rank, max_dims=batch_rank))
    obs_dim = draw(st.integers(min_value=2, max_value=100))
    if well_conditioned:
        # Cap the number of regressors at obs/2 so the random-normal design
        # Gram is well-conditioned (Marchenko-Pastur: cond(X) <~ 6 for p/n <=
        # 1/2, so cond(X X^T) <~ 40 << 1/eps_float32).  This is the regime
        # where the exact `residual + projection == Y` decomposition holds at
        # 1e-5 for the *Cholesky* path.  As p -> obs the Gram becomes
        # ill-conditioned and the Cholesky path loses that property (and NaNs
        # outright for p > obs) -- so these default-method properties stay in
        # the well-conditioned regime.  The `method='svd'` robust path is
        # exercised across the full p -> obs (and p > obs) regime by the
        # `*_svd_robust` tests below.
        p_max = max(1, obs_dim // 2)
    elif not allow_p_eq_n:
        p_max = obs_dim - 1
    elif not allow_p_greater_n:
        p_max = obs_dim
    else:
        p_max = 100
    ind_features = draw(st.integers(min_value=1, max_value=p_max))
    dep_features = draw(st.integers(
        min_value=1,
        max_value=100,
    ))
    rowvar = draw(st.booleans())
    if rowvar:
        X_shape = (*batch_shape, ind_features, obs_dim)
        Y_shape = (*batch_shape, dep_features, obs_dim)
    else:
        X_shape = (*batch_shape, obs_dim, ind_features)
        Y_shape = (*batch_shape, obs_dim, dep_features)
    #TODO
    # We're not using hypothesis's built-in strategy for generating arrays
    # (https://hypothesis.readthedocs.io/en/latest/numpy.html) because it
    # is *too good* at coming up with adversarial examples. We should return to
    # it once we have a better understanding of the problem (will likely need to
    # have a strong background in error analysis and numerical linear algebra).
    rng_key = draw(st.integers(min_value=0, max_value=10 ** 8))
    rng_key = jax.random.PRNGKey(rng_key)
    rng_key_X, rng_key_Y = jax.random.split(rng_key)
    X = jax.random.normal(key=rng_key_X, shape=X_shape)
    Y = jax.random.normal(key=rng_key_Y, shape=Y_shape)
    return X, Y, (rowvar,)


@run_variants
@given(arrays=generate_valid_arrays())
@hypothesis.settings(deadline=None)
def test_output_shape(arrays, fn):
    X, Y, (rowvar,) = arrays
    out = fn(Y, X, rowvar=rowvar)
    note((Y.shape, out.shape, rowvar))
    assert out.shape == Y.shape


@run_variants
@given(arrays=generate_valid_arrays(well_conditioned=True))
@hypothesis.settings(deadline=None, max_examples=20)
def test_residual_mean_zero_with_intercept(arrays, fn):
    X, Y, (rowvar,) = arrays
    if rowvar:
        X = jnp.concatenate(
            (X, jnp.ones((*X.shape[:-2], 1, X.shape[-1]))), axis=-2
        )
        axis = -1
    else:
        X = jnp.concatenate(
            (X, jnp.ones((*X.shape[:-1], 1))), axis=-1
        )
        axis = -2
    out = fn(Y, X, rowvar=rowvar)
    note(out.mean(axis))
    # Here we're covering for a poor understanding of error analysis by
    # just using a multiple of the condition number of the design matrix
    # (the multiple selected in a wholely unprincipled manner) as a proxy for
    # the expected tolerance.
    assert jnp.allclose(
        out.mean(axis, keepdims=True),
        0,
        atol=jnp.linalg.cond(X)[..., None, None] * 1e-6,
    )


@run_variants
@given(arrays=generate_valid_arrays(well_conditioned=True))
@hypothesis.settings(deadline=None)
def test_residual_decomposition(arrays, fn):
    X, Y, (rowvar,) = arrays
    rej = fn(Y, X, rowvar=rowvar)
    proj = fn(Y, X, rowvar=rowvar, return_mode='projection')
    assert jnp.allclose(rej + proj, Y, atol=1e-5)


@run_variants_svd
@given(arrays=generate_valid_arrays(allow_p_greater_n=True))
@hypothesis.settings(deadline=None, max_examples=20)
def test_residual_decomposition_svd_robust(arrays, fn):
    # B9: the SVD path holds the exact decomposition across the *full*
    # regime -- including p == obs and p > obs (rank-deficient), where the
    # default Cholesky path returns NaN.  ``proj = X @ beta`` is the unique
    # orthogonal projection onto col(X) regardless of how the (non-unique)
    # min-norm coefficients are chosen, so rej + proj == Y stays exact.
    X, Y, (rowvar,) = arrays
    rej = fn(Y, X, rowvar=rowvar)
    proj = fn(Y, X, rowvar=rowvar, return_mode='projection')
    assert jnp.allclose(rej + proj, Y, atol=1e-5)


def test_svd_robust_where_cholesky_degenerates():
    # p > obs: the Gram is rank-deficient, so the Cholesky path has no factor
    # and returns NaN (the documented method='cholesky' limitation); the SVD
    # path stays finite and exact, and the projection is non-expansive
    # (orthogonal projection onto col(X), ||proj|| <= ||Y||).
    key = jax.random.PRNGKey(0)
    xkey, ykey = jax.random.split(key)
    X = jax.random.normal(key=xkey, shape=(60, 40))  # p=60 > obs=40
    Y = jax.random.normal(key=ykey, shape=(8, 40))
    chol = residualise(Y, X, method='cholesky')
    assert not jnp.all(jnp.isfinite(chol))
    rej = residualise(Y, X, method='svd')
    proj = residualise(Y, X, method='svd', return_mode='projection')
    assert jnp.all(jnp.isfinite(rej))
    assert jnp.allclose(rej + proj, Y, atol=1e-5)
    assert jnp.linalg.norm(proj) <= jnp.linalg.norm(Y) + 1e-4


#Note: The commented out example represents a currently known failure mode
#      resulting from an ill-conditioned design matrix.
@run_variants
# @hypothesis.example(arrays=(
#     jnp.asarray([[1.001, 1], [1, 1.001]]),
#     jnp.asarray([[1.], [0.]]),
#     (False,),
# ))
@given(arrays=generate_valid_arrays(allow_p_eq_n=False))
@hypothesis.settings(deadline=None, max_examples=20)
def test_residual_varshared_zero(arrays, fn):
    X, Y, (rowvar,) = arrays
    rej = fn(Y, X, rowvar=rowvar)
    rho_r = cosine_kernel(X, rej, rowvar=rowvar)
    chk_r = jnp.sum(rho_r ** 2, axis=-2, keepdims=True)
    cond = jnp.linalg.cond(X)[..., None, None]
    assert jnp.allclose(
        jnp.where(jnp.isnan(chk_r), 0, chk_r),
        0,
        atol=cond * 1e-3, # This could still be flaky.
    )


@run_variants
@given(arrays=generate_valid_arrays(allow_p_eq_n=False, well_conditioned=True))
@hypothesis.settings(deadline=None, max_examples=20)
def test_total_shrinkage(arrays, fn):
    X, Y, (rowvar,) = arrays
    out = fn(Y, X, rowvar=rowvar, l2=1e8)
    assert jnp.allclose(out, Y, atol=1e-5)
    out = fn(Y, X, rowvar=rowvar, l2=1e8, return_mode='projection')
    assert jnp.allclose(out, 0, atol=1e-5)


def test_residualisation_exceptions():
    key = jax.random.PRNGKey(0)
    xkey, ykey = jax.random.split(key)
    X = jax.random.normal(key=xkey, shape=(3, 30, 100))
    Y = jax.random.normal(key=ykey, shape=(3, 1000, 100))
    with pytest.raises(ValueError):
        residualise(Y, X, l2=10000., return_mode='invalid_mode')
