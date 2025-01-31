# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for tensor utility functions.
"""
import pytest
import hypothesis
from hypothesis import assume, given
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst
import jax
import jax.numpy as jnp
import numpy as np
from nitrix._internal import (
    _conform_bform_weight,
    _dim_or_none,
    _compose,
    _seq_pad,
    atleast_4d,
    apply_vmap_over_outer,
    vmap_over_outer,
    broadcast_ignoring,
    orient_and_conform,
    promote_axis,
    demote_axis,
    fold_axis,
    unfold_axes,
    axis_complement,
    standard_axis_number,
    standard_axis_number_strict,
    negative_axis_number,
    negative_axis_number_strict,
    fold_and_promote,
    demote_and_unfold,
    promote_to_rank,
    extend_to_size,
    extend_to_max_size,
    argsort,
    complex_decompose,
    complex_recompose,
    amplitude_apply,
    apply_mask,
    conform_mask,
    mask_tensor,
)
from nitrix._internal.testutil import cfg_variants_test


VMOO_TEST_FUNCTIONS = {
    0: ('sum', jnp.sum, 1),
    1: ('corr', jnp.corrcoef, 2),
    2: ('diag', jnp.diagonal, 2),
    3: ('trace', jnp.trace, 2),
}


@st.composite
def generate_array_for_folding(
    draw,
    max_ndim: int = 5,
):
    rank = draw(st.integers(min_value=2, max_value=max_ndim))
    axis_to_fold = draw(st.integers(
        min_value=-rank,
        max_value=rank - 1,
    ))
    ax_std = standard_axis_number(axis_to_fold, rank)
    folded_shape = draw(npst.array_shapes(
        min_dims=rank + 1,
        max_dims=rank + 1,
    ))
    folded_size = folded_shape[ax_std]
    num_folds = folded_shape[ax_std + 1]
    shape = (
        *folded_shape[:ax_std],
        folded_size * num_folds,
        *folded_shape[ax_std + 2:],
    )
    arr = jnp.arange(np.prod(shape)).reshape(shape)
    return arr, axis_to_fold, folded_size, num_folds


@st.composite
def generate_arrays_for_broadcast_ignoring(
    draw,
    max_ndim: int = 5,
):
    rank = draw(st.integers(min_value=2, max_value=max_ndim))
    shape_1 = draw(npst.array_shapes(min_dims=rank, max_dims=rank))
    shape_2_bc = draw(npst.broadcastable_shapes(shape_1))
    rank_min = min(rank, len(shape_2_bc))
    rank_max = max(rank, len(shape_2_bc))
    shape_2_src = draw(npst.array_shapes(min_dims=rank_min, max_dims=rank_min))
    num_ignore = draw(st.integers(min_value=0, max_value=rank_min))
    if num_ignore > 0:
        ignore = set(draw(st.lists(
            st.integers(min_value=-rank_min, max_value=-1),
            min_size=num_ignore,
            max_size=num_ignore,
            unique=True,
        )))
    else:
        ignore = set()
    shape_2_rev = tuple(
        shape_2_src[-(i + 1)]
        if -(i + 1) in ignore
        else shape
        for i, shape in enumerate(shape_2_bc[::-1])
    )
    shape_2 = shape_2_rev[::-1]
    switch = draw(st.booleans())
    if switch:
        shape_1, shape_2 = shape_2, shape_1
    arr_1 = jnp.arange(np.prod(shape_1)).reshape(shape_1)
    arr_2 = jnp.arange(np.prod(shape_2)).reshape(shape_2)
    shape_1_padded_rev = _seq_pad(shape_1, rank_max, pad='first', pad_value=1)[::-1]
    shape_2_padded_rev = _seq_pad(shape_2, rank_max, pad='first', pad_value=1)[::-1]
    shape_1_out = tuple(
        shape
        if -(i + 1) in ignore
        else max(shape, shape_2_padded_rev[i])
        for i, shape in enumerate(shape_1_padded_rev)
    )[::-1]
    shape_2_out = tuple(
        shape
        if -(i + 1) in ignore
        else max(shape, shape_1_padded_rev[i])
        for i, shape in enumerate(shape_2_padded_rev)
    )[::-1]
    return arr_1, arr_2, tuple(ignore), shape_1_out, shape_2_out


@st.composite
def generate_array_for_vmoo(
    draw,
    max_batch_ndim: int = 5,
):
    vmoo_fn_idx = draw(
        st.integers(min_value=0, max_value=len(VMOO_TEST_FUNCTIONS) - 1)
    )
    vmoo_name, vmoo_fn, vmoo_rank = VMOO_TEST_FUNCTIONS[vmoo_fn_idx]
    batch_rank = draw(st.integers(min_value=0, max_value=max_batch_ndim))
    batch_shape = draw(npst.array_shapes(min_dims=batch_rank, max_dims=batch_rank))
    obs_dim = draw(st.integers(min_value=2, max_value=20))
    if vmoo_name == 'sum':
        shape = (*batch_shape, obs_dim)
    else:
        shape = (*batch_shape, obs_dim, obs_dim)
    arr = jnp.arange(np.prod(shape)).reshape(shape)
    return arr, vmoo_fn, vmoo_rank, batch_rank


@st.composite
def generate_arrays_for_orient_conform(
    draw,
    max_ndim: int = 5,
):
    ref_ndim = draw(st.integers(min_value=2, max_value=max_ndim))
    src_ndim = draw(st.integers(min_value=2, max_value=ref_ndim))
    src_axes = draw(st.lists(
        st.integers(min_value=-ref_ndim, max_value=ref_ndim - 1),
        min_size=src_ndim,
        max_size=src_ndim,
        unique=True,
    ))
    src_axes_std = set(standard_axis_number(ax, ref_ndim) for ax in src_axes)
    assume(len(src_axes_std) == src_ndim)
    ref_shape = draw(npst.array_shapes(min_dims=ref_ndim, max_dims=ref_ndim))
    src_shape = tuple(ref_shape[ax] for ax in src_axes)
    src_arr = jnp.arange(np.prod(src_shape)).reshape(src_shape)
    return src_arr, tuple(src_axes), ref_shape


@st.composite
def generate_arrays_for_max_size_extend(
    draw,
    max_ndim: int = 5,
    num_arrays: int = 5,
):
    ref_ndim = draw(st.integers(min_value=2, max_value=max_ndim))
    ref_shape = draw(npst.array_shapes(min_dims=ref_ndim, max_dims=ref_ndim))
    other_ranks = draw(st.lists(
        st.integers(min_value=2, max_value=ref_ndim),
        min_size=num_arrays - 1,
        max_size=num_arrays - 1,
    ))
    other_shapes = [
        tuple(
            draw(st.integers(min_value=1, max_value=ax))
            for i, ax in enumerate(ref_shape)
            if i >= ref_ndim - rank
        )
        for rank in other_ranks
    ]
    ref_arr = jnp.arange(np.prod(ref_shape)).reshape(ref_shape)
    other_arrs = [
        jnp.arange(np.prod(shape)).reshape(shape) for shape in other_shapes
    ]
    ref_idx = draw(st.integers(min_value=0, max_value=num_arrays - 1))
    arr_list = other_arrs[:ref_idx] + [ref_arr] + other_arrs[ref_idx:]
    return arr_list, ref_shape


@pytest.mark.parametrize("weight, expected_shape, expected_values", [
    (jnp.array([1, 2, 3]), (3,), [1, 2, 3]),
    (jnp.array([[1, 2, 3], [4, 5, 6]]), (2, 1, 3), [1, 2, 3, 4, 5, 6]),
    (jnp.array([[1, 2, 3], [4, 5, 6]])[:, None, :], (2, 1, 3), [1, 2, 3, 4, 5, 6])
])
def test_conform_bform_weight(weight, expected_shape, expected_values):
    result = _conform_bform_weight(weight)
    assert result.shape == expected_shape
    assert jnp.all(result.ravel() == jnp.asarray(expected_values))


@pytest.mark.parametrize("x, align, i, ndmax, expected", [
    (3, True, 1, 4, 1),
    (3, True, 0, 4, 0),
    (3, False, 1, 4, 0),
    (3, False, 0, 4, None)
])
def test_dim_or_none(x, align, i, ndmax, expected):
    assert _dim_or_none(x, align, i, ndmax) == expected


def test_compose():
    def f(g):
        def h(x):
            return g(x) + 1
        return h
    h = _compose(lambda x: x, f)
    assert h(0) == 1


@pytest.mark.parametrize("seq, length, pad_value, pad, expected", [
    ((1, 2, 3), 5, None, 'last', (1, 2, 3, None, None)),
    ((1, 2, 3), 3, None, 'last', (1, 2, 3)),
    ((1, 2, 3), 5, 0, 'last', (1, 2, 3, 0, 0)),
    ((1, 2, 3), 5, None, 'first', (None, None, 1, 2, 3)),
    ((1, 2, 3), 2, None, 'last', (1, 2, 3)),
])
def test_seq_pad(seq, length, pad_value, pad, expected):
    assert _seq_pad(seq, length, pad_value=pad_value, pad=pad) == expected


@pytest.mark.parametrize("seq, length, pad", [
    ((1, 2, 3), 2, 'invalid'),
])
def test_seq_pad_raises(seq, length, pad):
    with pytest.raises(ValueError):
        _seq_pad(seq, length, pad=pad)


@pytest.mark.parametrize("x, expected_shape", [
    (jnp.asarray(0), (1, 1, 1, 1)),
    (jnp.zeros(3), (1, 1, 1, 3)),
    (jnp.zeros((2, 3)), (1, 1, 2, 3)),
    (jnp.zeros((2, 3, 4)), (1, 2, 3, 4)),
    (jnp.zeros((2, 3, 4, 5)), (2, 3, 4, 5)),
])
def test_atleast_4d(x, expected_shape):
    assert atleast_4d(x).shape == expected_shape


def test_atleast_4d_multiple():
    assert (
        atleast_4d(jnp.asarray(0), jnp.asarray(0)) ==
        (jnp.zeros((1, 1, 1, 1)), jnp.zeros((1, 1, 1, 1)))
    )


@given(array=generate_array_for_folding())
def test_fold_and_unfold(array):
    X, axis, folded_size, num_folds = array
    Y = fold_axis(X, axis, num_folds)
    ax_std = standard_axis_number(axis, X.ndim)
    assert Y.shape[ax_std] == folded_size
    assert Y.shape[ax_std + 1] == num_folds
    Z = unfold_axes(Y, (ax_std, ax_std + 1))
    assert np.all(Z == X)


@given(array=generate_array_for_folding())
def test_fold_and_promote(array):
    X, axis, folded_size, num_folds = array
    Y = fold_and_promote(X, axis, num_folds)
    ax_std = standard_axis_number(axis, X.ndim)
    assert Y.shape[0] == num_folds
    assert Y.shape[ax_std + 1] == folded_size
    Z = demote_and_unfold(Y, ax_std + 1)
    assert np.all(Z == X)


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

    assert standard_axis_number_strict(-2, ndim) == 3
    assert standard_axis_number_strict(1, ndim) == 1
    with pytest.raises(ValueError):
        standard_axis_number_strict(7, ndim)

    assert negative_axis_number(-2, ndim) == -2
    assert negative_axis_number(0, ndim) == -5
    assert negative_axis_number(6, ndim) is None

    assert negative_axis_number_strict(-2, ndim) == -2
    assert negative_axis_number_strict(0, ndim) == -5
    with pytest.raises(ValueError):
        negative_axis_number_strict(6, ndim)

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


@cfg_variants_test(
    broadcast_ignoring,
    jit_params={'static_argnames': ('axis',)},
)
@given(arrays=generate_arrays_for_broadcast_ignoring())
def test_broadcast_ignoring_pbt(arrays, fn):
    arr_1, arr_2, ignore, shape_1_out, shape_2_out = arrays
    X, Y = fn(arr_1, arr_2, ignore)
    assert X.shape == shape_1_out
    assert Y.shape == shape_2_out


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


@hypothesis.settings(deadline=500)
@given(array=generate_array_for_vmoo())
def test_vmap_over_outer_pbt(array):
    arr, vmoo_fn, vmoo_rank, batch_rank = array
    ref_fn = vmoo_fn
    for i in range(0, -batch_rank, -1):
        ref_fn = jax.vmap(ref_fn, -i, -i)

    jaxpr_test = jax.make_jaxpr(vmap_over_outer(vmoo_fn, vmoo_rank))((arr,))
    jaxpr_ref = jax.make_jaxpr(ref_fn)(arr)
    assert jaxpr_test.jaxpr.pretty_print() == jaxpr_ref.jaxpr.pretty_print()

    out = vmap_over_outer(vmoo_fn, vmoo_rank)((arr,))
    ref = ref_fn(arr)
    assert np.allclose(out, ref)

    out = jax.jit(vmap_over_outer(vmoo_fn, vmoo_rank))((arr,))
    ref = jax.jit(ref_fn)(arr)
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


@cfg_variants_test(
    orient_and_conform,
    jit_params={'static_argnames': ('axis', 'dim')},
)
@given(array=generate_arrays_for_orient_conform())
def test_orient_and_conform_pbt(array, fn):
    X, axes, ref_shape = array
    out0 = fn(X, axes, reference=jnp.empty(ref_shape))
    out1 = fn(X, axes, dim=len(ref_shape))
    out0 + jnp.empty(ref_shape) # implicit check for shape compatibility
    out1 + jnp.empty(ref_shape) # implicit check for shape compatibility
    assert out0.shape == out1.shape


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


@cfg_variants_test(
    extend_to_max_size,
    jit_params={'static_argnames': ('fill',)},
)
@hypothesis.settings(deadline=500)
@given(arrays=generate_arrays_for_max_size_extend())
def test_extend_pbt(arrays, fn):
    arr_list, ref_shape = arrays
    out = fn(arr_list, fill=-1)
    for o in out:
        assert o.shape == ref_shape


def test_complex_views():
    key = jax.random.PRNGKey(0)
    key_X, key_Y = jax.random.split(key)
    X = jnp.abs(jax.random.normal(key_X, (2, 3)))
    Y = jnp.clip(jax.random.normal(key_Y, (2, 3)), -jnp.pi, jnp.pi)
    Z = X * jnp.exp(Y * 1j)
    out = complex_decompose(Z)
    assert out[0].shape == (2, 3)
    assert out[1].shape == (2, 3)
    assert np.all(out[0] == X)
    assert np.all(out[1] == Y)

    out = complex_recompose(out[0], out[1])
    assert np.all(out == Z)

    out = amplitude_apply(jnp.log)(Z)
    assert np.all(out == jnp.log(X) * jnp.exp(Y * 1j))


def test_mask():
    msk = jnp.array([1, 1, 0, 0, 0], dtype=bool)
    tsr = np.random.rand(5, 5, 5)
    tsr = jnp.asarray(tsr)
    mskd = apply_mask(tsr, msk, axis=0)
    assert mskd.shape == (2, 5, 5)
    assert np.all(mskd == tsr[:2])
    mskd = apply_mask(tsr, msk, axis=1)
    assert mskd.shape == (5, 2, 5)
    assert np.all(mskd == tsr[:, :2])
    mskd = apply_mask(tsr, msk, axis=2)
    assert np.all(mskd == tsr[:, :, :2])
    assert mskd.shape == (5, 5, 2)
    mskd = apply_mask(tsr, msk, axis=-1)
    assert np.all(mskd == tsr[:, :, :2])
    assert mskd.shape == (5, 5, 2)
    mskd = apply_mask(tsr, msk, axis=-2)
    assert np.all(mskd == tsr[:, :2])
    assert mskd.shape == (5, 2, 5)
    mskd = apply_mask(tsr, msk, axis=-3)
    assert np.all(mskd == tsr[:2])
    assert mskd.shape == (2, 5, 5)

    mask0 = conform_mask(tsr, msk, axis=-1)
    mask1 = conform_mask(tsr, msk, axis=(-1,))
    assert mask0.shape == (5, 5, 5)
    assert tsr[mask0].size == 50
    assert mask0.shape == mask1.shape
    assert jnp.all(mask0 == mask1)

    jconform = jax.jit(conform_mask, static_argnames=('axis',))
    mask = jconform(tsr, msk, axis=-1)
    assert mask.shape == (5, 5, 5)
    assert tsr[mask].size == 50

    jtsrmsk = jax.jit(mask_tensor, static_argnames=('axis',))
    mskd = jtsrmsk(tsr, msk, axis=-1)
    assert (mskd != 0).sum() == 50
    mskd = jtsrmsk(tsr, msk, axis=-1, fill_value=float('nan'))
    assert np.isnan(mskd).sum() == 75
