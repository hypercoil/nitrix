# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for residualisation
"""
import pytest
import jax
import jax.numpy as jnp
import hypothesis
from hypothesis import given, note, strategies as st
from hypothesis.extra import numpy as npst
from nitrix.functional import pairedcorr, residualise


# def promote_shape(struc):
#     (x, y), z = struc
#     if len(y) > 1 and y[-2] > y[-1]:
#         y = (*y[:-2], y[-1], y[-1])
#     return (x, y), z

from nitrix._internal.util import Tensor
from typing import Literal, Tuple
def fwd_err(
    f: callable,
    mode: Literal['forward', 'backward'] | None = 'forward',
    *jpparams,
    **jparams,
) -> callable:
    match mode:
        case 'forward':
            jacfn = jax.jacfwd
        case 'backward':
            jacfn = jax.jacrev
        case None:
            jacfn = jax.jacobian

    def _f(x_dx: Tuple[Tensor, Tensor], *pparams, **params) -> float:
        x, dx = x_dx
        jcb = jacfn(f, *jpparams, **jparams)(x, *pparams, **params)
        return jnp.linalg.norm(jcb * dx)

    return _f


def cfg_variants_test(base_fn: callable, jit_params = None):
    if jit_params is None:
        jit_params = {}
    def test_variants(test: callable):
        return pytest.mark.parametrize(
            'fn', [base_fn, jax.jit(base_fn, **jit_params)]
        )(test)
    return test_variants


# def cfg_variants_test(base_fn: callable):
#     def vargiven(*pparams, **params):
#         return given(
#             *pparams,
#             **params,
#             fn=st.one_of([st.just(base_fn), st.just(jax.jit(base_fn))])
#         )


run_variants = cfg_variants_test(
    residualise,
    jit_params={'static_argnames': ('l2', 'return_mode', 'rowvar')},
)


@st.composite
def generate_valid_arrays(
    draw,
    allow_p_greater_n: bool = False,
    allow_singularity: bool = False,
    allow_zerovar_target: bool = True,
):
    batch_rank = draw(st.integers(min_value=0, max_value=3))
    batch_shape = draw(npst.array_shapes(min_dims=batch_rank, max_dims=batch_rank))
    obs_dim = draw(st.integers(min_value=2, max_value=100))
    ind_features = draw(st.integers(
        min_value=1,
        max_value=obs_dim if not allow_p_greater_n else 100,
    ))
    dep_features = draw(st.integers(
        min_value=1,
        max_value=100,
    ))
    rowvar = draw(st.booleans())
    if rowvar:
        ax = -1
        X_shape = (*batch_shape, ind_features, obs_dim)
        Y_shape = (*batch_shape, dep_features, obs_dim)
    else:
        ax = -2
        X_shape = (*batch_shape, obs_dim, ind_features)
        Y_shape = (*batch_shape, obs_dim, dep_features)
    X = draw(npst.arrays(
        dtype=jnp.float32,
        shape=X_shape,
        elements=st.floats(-100, 100, width=32),
    ).map(jnp.asarray))
    Y_src = npst.arrays(
        dtype=jnp.float32,
        shape=Y_shape,
        elements=(
            st.floats(-100, 100, width=32)
        ),
    )
    # Target array must have non-zero variance in all channels.
    # Otherwise, the projection matrix will be singular.
    if not allow_zerovar_target:
        Y_src = Y_src.filter(lambda x: not jnp.isclose(x.var(ax), 0).any())
    Y = draw(Y_src.map(jnp.asarray))
    if not allow_singularity:
        ptb = jnp.eye(*X.shape[-2:]) * 1e-3
        X = X + ptb
    return X, Y, (rowvar,)


# @run_variants
# @given(arrays=generate_valid_arrays(allow_singularity=True))
# @hypothesis.settings(deadline=1000)
# def test_output_shape(arrays, fn):
#     X, Y, (rowvar,) = arrays
#     out = fn(Y, X, rowvar=rowvar)
#     note((Y.shape, out.shape, rowvar))
#     assert out.shape == Y.shape


# @run_variants
# @given(arrays=generate_valid_arrays())
# @hypothesis.settings(deadline=1000, max_examples=20)
# def test_residual_mean_zero_with_intercept(arrays, fn):
#     X, Y, (rowvar,) = arrays
#     if rowvar:
#         X = jnp.concatenate(
#             (X, jnp.ones((*X.shape[:-2], 1, X.shape[-1]))), axis=-2
#         )
#         axis = -1
#     else:
#         X = jnp.concatenate(
#             (X, jnp.ones((*X.shape[:-1], 1))), axis=-1
#         )
#         axis = -2
#     out = fn(Y, X, rowvar=rowvar)
#     note(out.mean(axis))
#     assert jnp.allclose(out.mean(axis), 0, atol=1e-5)


# @run_variants
# @given(arrays=generate_valid_arrays())
# @hypothesis.settings(deadline=1000)
# def test_residual_decomposition(arrays, fn):
#     X, Y, (rowvar,) = arrays
#     rej = fn(Y, X, rowvar=rowvar)
#     proj = fn(Y, X, rowvar=rowvar, return_mode='projection')
#     assert jnp.allclose(rej + proj, Y, atol=1e-5)


def cosine_kernel(x, y, rowvar=True):
    if not rowvar:
        x = x.swapaxes(-1, -2)
        y = y.swapaxes(-1, -2)
    return x @ y.swapaxes(-1, -2) / (
        jnp.linalg.norm(x, axis=-1, keepdims=True) *
        jnp.linalg.norm(y, axis=-1, keepdims=True).swapaxes(-1, -2)
    )


@run_variants
@hypothesis.example(arrays=(
    jnp.asarray([[1.001, 1], [1, 1.001]]),
    jnp.asarray([[1.], [0.]]),
    (False,),
))
@given(arrays=generate_valid_arrays(allow_zerovar_target=False))
@hypothesis.settings(deadline=10000, max_examples=20)
def test_residual_varshared_zero(arrays, fn):
    def nf(X, Y, rowvar=True):
        return fn(Y, X, rowvar=rowvar)
    X, Y, (rowvar,) = arrays
    rej = fn(Y, X, rowvar=rowvar)
    proj = fn(Y, X, rowvar=rowvar, return_mode='projection')
    rho_i = cosine_kernel(X, Y, rowvar=rowvar)
    rho_r = cosine_kernel(X, rej, rowvar=rowvar)
    rho_p = cosine_kernel(X, proj, rowvar=rowvar)
    chk_r = jnp.sum(rho_r ** 2, axis=-2)
    chk_p = jnp.sum(rho_p ** 2, axis=-2)
    cond = jnp.linalg.cond(X)
    assert jnp.allclose(
        chk_r[~jnp.isnan(chk_r)],
        0,
        atol=cond * 1e-5,
    )
    assert jnp.allclose(
        chk_p[~jnp.isnan(chk_p)],
        1.,
        atol=cond * 1e-5,
    )

def test_total_shrinkage():
    assert 0


def test_residualisation():
    key = jax.random.PRNGKey(0)
    xkey, ykey = jax.random.split(key)
    X = jax.random.normal(key=xkey, shape=(3, 30, 100))
    Y = jax.random.normal(key=ykey, shape=(3, 1000, 100))
    out = residualise(Y, X)
    assert out.shape == (3, 1000, 100)

    X = jnp.ones((3, 100, 1))
    Y = jax.random.normal(key=ykey, shape=(3, 100, 1000))
    out = residualise(Y, X, rowvar=False)
    assert out.shape == (3, 100, 1000)
    assert jnp.allclose(out.mean(-2), 0, atol=1e-5)

    X = jax.random.normal(key=xkey, shape=(3, 30, 100))
    Y = jax.random.normal(key=ykey, shape=(3, 1000, 100))
    out = residualise(Y, X, l2=0.01)
    assert not jnp.allclose(out, Y)
    out = residualise(Y, X, l2=10000.)
    assert jnp.allclose(out, Y, atol=1e-5)
    out = residualise(Y, X, l2=10000., return_mode='projection')
    assert jnp.allclose(out, 0, atol=1e-5)
    with pytest.raises(ValueError):
        residualise(Y, X, l2=10000., return_mode='invalid_mode')
