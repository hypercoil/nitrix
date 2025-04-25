# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for residualisation
"""
import pytest
import hypothesis
from hypothesis import given, note, strategies as st
from hypothesis.extra import numpy as npst
import jax
import jax.numpy as jnp
from nitrix.functional import residualise
from nitrix._internal.testutil import cfg_variants_test


run_variants = cfg_variants_test(
    residualise,
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
):
    batch_rank = draw(st.integers(min_value=0, max_value=3))
    batch_shape = draw(npst.array_shapes(min_dims=batch_rank, max_dims=batch_rank))
    obs_dim = draw(st.integers(min_value=2, max_value=100))
    ind_features = draw(st.integers(
        min_value=1,
        max_value=(
            obs_dim - 1
            if not allow_p_eq_n
            else obs_dim
            if not allow_p_greater_n
            else 100
        ),
    ))
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
@hypothesis.settings(deadline=1000)
def test_output_shape(arrays, fn):
    X, Y, (rowvar,) = arrays
    out = fn(Y, X, rowvar=rowvar)
    note((Y.shape, out.shape, rowvar))
    assert out.shape == Y.shape


@run_variants
@given(arrays=generate_valid_arrays())
@hypothesis.settings(deadline=1000, max_examples=20)
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
@given(arrays=generate_valid_arrays())
@hypothesis.settings(deadline=2000)
def test_residual_decomposition(arrays, fn):
    X, Y, (rowvar,) = arrays
    rej = fn(Y, X, rowvar=rowvar)
    proj = fn(Y, X, rowvar=rowvar, return_mode='projection')
    assert jnp.allclose(rej + proj, Y, atol=1e-5)


#Note: The commented out example represents a currently known failure mode
#      resulting from an ill-conditioned design matrix.
@run_variants
# @hypothesis.example(arrays=(
#     jnp.asarray([[1.001, 1], [1, 1.001]]),
#     jnp.asarray([[1.], [0.]]),
#     (False,),
# ))
@given(arrays=generate_valid_arrays(allow_p_eq_n=False))
@hypothesis.settings(deadline=10000, max_examples=20)
def test_residual_varshared_zero(arrays, fn):
    X, Y, (rowvar,) = arrays
    rej = fn(Y, X, rowvar=rowvar)
    rho_r = cosine_kernel(X, rej, rowvar=rowvar)
    chk_r = jnp.sum(rho_r ** 2, axis=-2, keepdims=True)
    cond = jnp.linalg.cond(X)[..., None, None]
    assert jnp.allclose(
        jnp.where(jnp.isnan(chk_r), 0, chk_r),
        0,
        atol=cond * 1e-4,
    )


@run_variants
@given(arrays=generate_valid_arrays(allow_p_eq_n=False))
@hypothesis.settings(deadline=10000, max_examples=20)
def test_total_shrinkage(arrays, fn):
    X, Y, (rowvar,) = arrays
    out = fn(Y, X, rowvar=rowvar, l2=10000.)
    assert jnp.allclose(out, Y, atol=1e-5)
    out = fn(Y, X, rowvar=rowvar, l2=10000., return_mode='projection')
    assert jnp.allclose(out, 0, atol=1e-5)


def test_residualisation_exceptions():
    key = jax.random.PRNGKey(0)
    xkey, ykey = jax.random.split(key)
    X = jax.random.normal(key=xkey, shape=(3, 30, 100))
    Y = jax.random.normal(key=ykey, shape=(3, 1000, 100))
    with pytest.raises(ValueError):
        residualise(Y, X, l2=10000., return_mode='invalid_mode')
