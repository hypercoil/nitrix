# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for tensor utility functions.
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from nitrix.util import (
    _dim_or_none, _compose, _seq_pad, atleast_4d, apply_vmap_over_outer,
    vmap_over_outer, broadcast_ignoring, orient_and_conform,
    promote_axis, demote_axis, fold_axis, unfold_axes,
    axis_complement, standard_axis_number, negative_axis_number,
    fold_and_promote, demote_and_unfold, promote_to_rank,
    extend_to_size, extend_to_max_size, argsort,
)


def test_dim_or_none():
    assert _dim_or_none(3, True, 1, 4) == 1
    assert _dim_or_none(3, True, 0, 4) == 0
    assert _dim_or_none(3, False, 1, 4) == 0
    assert _dim_or_none(3, False, 0, 4) is None


def test_compose():
    def f(g):
        def h(x):
            return g(x) + 1
        return h
    h = _compose(lambda x: x, f)
    assert h(0) == 1


def test_seq_pad():
    assert _seq_pad((1, 2, 3), 5) == (1, 2, 3, None, None)
    assert _seq_pad((1, 2, 3), 3) == (1, 2, 3)
    assert _seq_pad((1, 2, 3), 5, pad_value=0) == (1, 2, 3, 0, 0)
    assert _seq_pad((1, 2, 3), 5, pad='first') == (None, None, 1, 2, 3)
    assert _seq_pad((1, 2, 3), 2) == (1, 2, 3)
    with pytest.raises(ValueError):
        _seq_pad((1, 2, 3), 2, pad='invalid')


def test_atleast_4d():
    x = jnp.asarray(0)
    assert atleast_4d(x).shape == (1, 1, 1, 1)
    x = jnp.zeros(3)
    assert atleast_4d(x).shape == (1, 1, 1, 3)
    x = jnp.zeros((2, 3))
    assert atleast_4d(x).shape == (1, 1, 2, 3)
    x = jnp.zeros((2, 3, 4))
    assert atleast_4d(x).shape == (1, 2, 3, 4)
    x = jnp.zeros((2, 3, 4, 5))
    assert atleast_4d(x).shape == (2, 3, 4, 5)
    assert (
        atleast_4d(jnp.asarray(0), jnp.asarray(0)) ==
        (jnp.zeros((1, 1, 1, 1)), jnp.zeros((1, 1, 1, 1)))
    )

def test_axis_ops():
    shape = (2, 3, 5, 7, 11)
    X = np.empty(shape)
    ndim = X.ndim
    assert axis_complement(ndim, -2) == (0, 1, 2, 4)
    assert axis_complement(ndim, (0, 1, 4)) == (2, 3)
    assert axis_complement(ndim, (0, 1, 2, 3, -1)) == ()

    assert standard_axis_number(-2, ndim) == 3
    assert standard_axis_number(1, ndim) == 1
    assert standard_axis_number(7, ndim) is None

    assert negative_axis_number(-2, ndim) == -2
    assert negative_axis_number(0, ndim) == -5
    assert negative_axis_number(6, ndim) is None

    assert promote_axis(ndim, -2) == (3, 0, 1, 2, 4)
    assert promote_axis(ndim, 1) == (1, 0, 2, 3, 4)
    assert promote_axis(ndim, (-2, 1)) == (3, 1, 0, 2, 4)

    assert demote_axis(7, (5, 2)) == (2, 3, 0, 4, 5, 1, 6)
    assert demote_axis(ndim, 2) == (1, 2, 0, 3, 4)

    assert unfold_axes(X, 0).shape == X.shape
    assert unfold_axes(X, (-3, -2)).shape == (2, 3, 35, 11)
    assert unfold_axes(X, (1, 2, 3)).shape == (2, 105, 11)

    assert fold_axis(X, -3, 1).shape == (2, 3, 5, 1, 7, 11)
    assert fold_axis(X, -3, 5).shape == (2, 3, 1, 5, 7, 11)

    assert fold_and_promote(X, -2, 7).shape == (7, 2, 3, 5, 1, 11)
    assert fold_and_promote(X, -4, 3).shape == (3, 2, 1, 5, 7, 11)

    assert demote_and_unfold(X, -2, (3, 4)).shape == (3, 5, 7, 22)
    assert demote_and_unfold(X, 1, (1, 2, 3)).shape == (3, 70, 11)

    X2 = np.random.rand(4, 3, 100, 7)
    Y = fold_and_promote(X2, -2, 5)
    assert Y.shape == (5, 4, 3, 20, 7)
    X2_hat = demote_and_unfold(Y, -2, (-3, -2))
    assert np.all(X2 == X2_hat)

    Y = demote_and_unfold(X2, -2, (-3, -2))
    assert Y.shape == (3, 400, 7)
    X2_hat = fold_and_promote(Y, -2, 4)
    assert np.all(X2 == X2_hat)

def test_broadcast_ignoring():
    shapes = (
        (
            ((2, 3, 2), (4, 2)),
            ((2, 3, 2), (2, 4, 2))
        ),
        (
            ((2, 3, 2), (2,)),
            ((2, 3, 2), (2, 1, 2))
        ),
    )
    for ((x_in, y_in), (x_out, y_out)) in shapes:
        X, Y = broadcast_ignoring(
            jnp.empty(x_in),
            jnp.empty(y_in),
            axis=-2,
        )
        assert X.shape == x_out
        assert Y.shape == y_out
    shapes = (
        (
            ((2, 3, 2), (4, 2)),
            ((2, 3, 2), (1, 4, 2))
        ),
        (
            ((2, 3, 2), (2,)),
            ((2, 3, 2), (1, 1, 2))
        ),
    )
    for ((x_in, y_in), (x_out, y_out)) in shapes:
        X, Y = broadcast_ignoring(
            jnp.empty(x_in),
            jnp.empty(y_in),
            axis=(-3, -2),
        )
        assert X.shape == x_out
        assert Y.shape == y_out


def vmap_over_outer_test_arg():
    test_obs = 100
    offset = 10
    offset2 = 50
    w = np.zeros((test_obs, test_obs))
    rows, cols = np.diag_indices_from(w)
    w[(rows, cols)] = np.random.rand(test_obs)
    w[(rows[:-offset], cols[offset:])] = (
        3 * np.random.rand(test_obs - offset))
    w[(rows[:-offset2], cols[offset2:])] = (
        np.random.rand(test_obs - offset2))
    w = jnp.stack([w] * 20)
    w = w.reshape(2, 2, 5, test_obs, test_obs)
    return w


def test_apply_vmap_over_outer():
    w = vmap_over_outer_test_arg()
    out = apply_vmap_over_outer((w,), f=jnp.diagonal, f_dim=(2,), align_outer=(False,))
    ref = jax.vmap(jax.vmap(jax.vmap(jnp.diagonal, 0, 0), 1, 1), 2, 2)(w)
    assert np.allclose(out, ref)


def test_vmap_over_outer():
    w = vmap_over_outer_test_arg()

    jaxpr_test = jax.make_jaxpr(vmap_over_outer(jnp.diagonal, 2))((w,))
    jaxpr_ref = jax.make_jaxpr(
        jax.vmap(jax.vmap(jax.vmap(jnp.diagonal, 0, 0), 1, 1), 2, 2))(w)
    assert jaxpr_test.jaxpr.pretty_print() == jaxpr_ref.jaxpr.pretty_print()

    out = vmap_over_outer(jnp.diagonal, 2)((w,))
    ref = jax.vmap(jax.vmap(jax.vmap(jnp.diagonal, 0, 0), 1, 1), 2, 2)(w)
    assert np.allclose(out, ref)

    out = jax.jit(vmap_over_outer(jnp.diagonal, 2))((w,))
    ref = jax.jit(jax.vmap(jax.vmap(jax.vmap(jnp.diagonal, 0, 0), 1, 1), 2, 2))(w)
    assert np.allclose(out, ref)

    L = np.random.rand(5, 13)
    R = np.random.rand(2, 5, 4)
    jvouter = jax.jit(vmap_over_outer(jnp.outer, 1))
    out = jvouter((L, R))
    ref = jax.vmap(jax.vmap(jnp.outer, (None, 0), 0), (0, 1), 1)(L, R)
    assert out.shape == (2, 5, 13, 4)
    assert np.allclose(out, ref)

def test_argsort():
    assert (
        argsort([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]) == 
        [1, 3, 6, 0, 9, 2, 4, 8, 10, 7, 5]
    )

def test_orient_and_conform():
    X = np.random.rand(3, 7)
    R = np.random.rand(2, 7, 11, 1, 3)
    out = orient_and_conform(X.swapaxes(-1, 0), (1, -1), reference=R)
    ref = X.swapaxes(-1, -2)[None, :, None, None, :]
    assert(out.shape == ref.shape)
    assert np.all(out == ref)

    X = np.random.rand(7)
    R = np.random.rand(2, 7, 11, 1, 3)
    out = orient_and_conform(X, 1, reference=R)
    ref = X[None, :, None, None, None]
    assert(out.shape == ref.shape)
    assert np.all(out == ref)

    # test with jit compilation
    jorient = jax.jit(orient_and_conform, static_argnames=('axis', 'dim'))
    out = jorient(X, 1, dim=R.ndim)
    ref = X[None, :, None, None, None]
    assert(out.shape == ref.shape)
    assert np.all(out == ref)

    with pytest.raises(ValueError):
        orient_and_conform(X, 1, reference=None, dim=None)

def test_promote():
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (2, 3))
    out = promote_to_rank(X, 3)
    assert out.shape == (1, 2, 3)

    out = promote_to_rank(X, 2)
    assert out.shape == (2, 3)

def test_extend():
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (2, 3))
    out = extend_to_size(X, (5, 5))
    assert out.shape == (5, 5)
    assert jnp.isnan(out).sum() == 19
    assert out[~jnp.isnan(out)].sum() == X.sum()

    out = extend_to_size(X, (2, 3))
    assert out.shape == (2, 3)
    assert np.all(out == X)

    Xs = tuple(
        jax.random.normal(jax.random.fold_in(key, i), (5 - i, i + 1))
        for i in range(5)
    )
    out = extend_to_max_size(Xs)
    targets = {0: 20, 1: 17, 2: 16, 3: 17, 4: 20}
    for i, o in enumerate(out):
        assert o.shape == (5, 5)
        assert jnp.isnan(o).sum() == targets[i]
