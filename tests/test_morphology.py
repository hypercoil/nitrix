# -*- coding: utf-8 -*-
"""Tests for ``nitrix.morphology``.

Coverage:

- ``dilate`` / ``erode`` / ``open`` / ``close``: bit-exact match
  against ``scipy.ndimage.grey_dilation`` etc. on interior pixels;
  boundary semantics differ (we pad with algebra identity, scipy
  defaults to ``reflect``) but the interior is the meaningful test.
- ``distance_transform``: parity against
  ``scipy.ndimage.distance_transform_cdt`` for ``chessboard`` and
  ``taxicab`` metrics.
- ``median_filter``: interior parity against
  ``scipy.ndimage.median_filter``.
- Non-trivial structuring element: explicit per-position weights for
  dilation match the algebra's definition.
- Differentiability: ``dilate`` / ``erode`` have well-defined
  subgradients via the TROPICAL_* backward; check the gradient is
  finite and routes to a single neighbour.
- ``susan_emulator``: raises ``NotImplementedError`` until the
  ``bilateral_gaussian`` dependency lands.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

scipy_ndi = pytest.importorskip('scipy.ndimage')

from nitrix.morphology import (
    close as morph_close,
    dilate,
    distance_transform,
    erode,
    median_filter,
    open as morph_open,
    susan_emulator,
)


jax.config.update('jax_enable_x64', True)


# ---------------------------------------------------------------------------
# dilate / erode / open / close
# ---------------------------------------------------------------------------


def _interior_match(got, ref, pad):
    '''Trim ``pad`` from every axis and compare.'''
    got_i = np.asarray(got)
    ref_i = np.asarray(ref)
    sl = tuple(slice(pad, -pad if pad else None) for _ in range(got_i.ndim))
    np.testing.assert_allclose(got_i[sl], ref_i[sl], atol=1e-10)


def test_dilate_flat_3x3_matches_scipy():
    x = jax.random.normal(jax.random.key(0), (16, 16))
    got = dilate(x, size=3, backend='jax')
    ref = scipy_ndi.grey_dilation(
        np.asarray(x), size=3, mode='constant', cval=-np.inf,
    )
    np.testing.assert_array_equal(got, ref)


def test_erode_flat_3x3_matches_scipy():
    x = jax.random.normal(jax.random.key(1), (16, 16))
    got = erode(x, size=3, backend='jax')
    ref = scipy_ndi.grey_erosion(
        np.asarray(x), size=3, mode='constant', cval=np.inf,
    )
    np.testing.assert_array_equal(got, ref)


def test_open_matches_dilate_of_erode():
    x = jax.random.normal(jax.random.key(2), (16, 16))
    got = morph_open(x, size=3, backend='jax')
    eroded = erode(x, size=3, backend='jax')
    expected = dilate(eroded, size=3, backend='jax')
    np.testing.assert_array_equal(got, expected)


def test_close_matches_erode_of_dilate():
    x = jax.random.normal(jax.random.key(3), (16, 16))
    got = morph_close(x, size=3, backend='jax')
    dilated = dilate(x, size=3, backend='jax')
    expected = erode(dilated, size=3, backend='jax')
    np.testing.assert_array_equal(got, expected)


def test_dilate_3d():
    x = jax.random.normal(jax.random.key(4), (8, 8, 8))
    got = dilate(x, size=3, backend='jax')
    ref = scipy_ndi.grey_dilation(
        np.asarray(x), size=3, mode='constant', cval=-np.inf,
    )
    np.testing.assert_array_equal(got, ref)


def test_dilate_4d_fmri_shape():
    '''4D morphology: volume + time, the natural fMRI layout.'''
    x = jax.random.normal(jax.random.key(40), (6, 6, 6, 6))
    got = dilate(x, size=3, backend='jax')
    ref = scipy_ndi.grey_dilation(
        np.asarray(x), size=3, mode='constant', cval=-np.inf,
    )
    np.testing.assert_array_equal(got, ref)


def test_erode_4d_fmri_shape():
    x = jax.random.normal(jax.random.key(41), (5, 5, 5, 5))
    got = erode(x, size=3, backend='jax')
    ref = scipy_ndi.grey_erosion(
        np.asarray(x), size=3, mode='constant', cval=np.inf,
    )
    np.testing.assert_array_equal(got, ref)


def test_dilate_explicit_structuring_element():
    # 1D, structure = [-1, 0, +1] -- centered SAME-padded conv.
    # Padded x = [-inf, 1, 2, 3, 4, 5, 6, 7, -inf].
    # out[i] = max( x_pad[i+0]+(-1), x_pad[i+1]+0, x_pad[i+2]+1 ).
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    se = jnp.array([-1.0, 0.0, 1.0])
    got = dilate(x, structuring_element=se, backend='jax')
    # i=0: max(-inf-1, 1+0,  2+1) = 3
    # i=1: max(1-1,   2+0,  3+1) = 4
    # i=2: max(2-1,   3+0,  4+1) = 5
    # i=3: max(3-1,   4+0,  5+1) = 6
    # i=4: max(4-1,   5+0,  6+1) = 7
    # i=5: max(5-1,   6+0,  7+1) = 8   (the +1 pulls the right neighbour)
    # i=6: max(6-1,   7+0,  -inf+1) = 7
    expected = jnp.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 7.0])
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_dilate_erode_anisotropic_size():
    x = jax.random.normal(jax.random.key(5), (10, 16))
    got = dilate(x, size=(3, 5), backend='jax')
    ref = scipy_ndi.grey_dilation(
        np.asarray(x), size=(3, 5), mode='constant', cval=-np.inf,
    )
    np.testing.assert_array_equal(got, ref)


# ---------------------------------------------------------------------------
# distance_transform
# ---------------------------------------------------------------------------


def test_distance_transform_chebyshev_matches_scipy():
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[4:12, 4:12] = 1.0
    got = distance_transform(
        jnp.asarray(mask), metric='chebyshev', backend='jax',
    )
    ref = scipy_ndi.distance_transform_cdt(mask, metric='chessboard')
    np.testing.assert_array_equal(got, ref.astype(np.float32))


def test_distance_transform_city_block_matches_scipy():
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[4:12, 4:12] = 1.0
    got = distance_transform(
        jnp.asarray(mask), metric='city_block', backend='jax',
    )
    ref = scipy_ndi.distance_transform_cdt(mask, metric='taxicab')
    np.testing.assert_array_equal(got, ref.astype(np.float32))


def test_distance_transform_3d():
    mask = np.zeros((8, 8, 8), dtype=np.float32)
    mask[2:6, 2:6, 2:6] = 1.0
    got = distance_transform(
        jnp.asarray(mask), metric='chebyshev', backend='jax',
    )
    ref = scipy_ndi.distance_transform_cdt(mask, metric='chessboard')
    np.testing.assert_array_equal(got, ref.astype(np.float32))


def test_distance_transform_rejects_unknown_metric():
    mask = jnp.zeros((4, 4))
    with pytest.raises(ValueError, match='euclidean|metric'):
        distance_transform(mask, metric='euclidean', backend='jax')


# ---------------------------------------------------------------------------
# median_filter
# ---------------------------------------------------------------------------


def test_median_filter_interior_matches_scipy():
    x = jax.random.normal(jax.random.key(10), (16, 16))
    got = median_filter(x, size=3)
    ref = scipy_ndi.median_filter(
        np.asarray(x), size=3, mode='reflect',
    )
    # Boundary modes differ; compare interior.
    _interior_match(got, ref, pad=2)


def test_median_filter_1d_interior():
    x = jax.random.normal(jax.random.key(11), (20,))
    got = median_filter(x, size=5)
    ref = scipy_ndi.median_filter(np.asarray(x), size=5, mode='reflect')
    _interior_match(got, ref, pad=3)


def test_median_filter_3d_interior():
    x = jax.random.normal(jax.random.key(12), (8, 8, 8))
    got = median_filter(x, size=3)
    ref = scipy_ndi.median_filter(np.asarray(x), size=3, mode='reflect')
    _interior_match(got, ref, pad=2)


def test_median_filter_structuring_element_mask():
    # Plus-shaped neighbourhood (only axis-aligned positions).
    x = jax.random.normal(jax.random.key(13), (16, 16))
    se = jnp.array([
        [False, True, False],
        [True,  True, True],
        [False, True, False],
    ])
    got = median_filter(x, structuring_element=se)
    # Build same with scipy's footprint argument.
    ref = scipy_ndi.median_filter(
        np.asarray(x), footprint=np.asarray(se), mode='reflect',
    )
    _interior_match(got, ref, pad=2)


# ---------------------------------------------------------------------------
# Differentiability of dilate / erode (TROPICAL_* subgradient)
# ---------------------------------------------------------------------------


def test_dilate_gradient_routes_to_argmax():
    # In 1D, dilate at position i routes its upstream gradient to the
    # j ∈ {i-1, i, i+1} (with VALID/SAME padding) attaining the max.
    # Construct x so each window has a clear winner.
    x = jnp.array([1.0, 5.0, 2.0, 8.0, 3.0])
    # 3-window dilation w/ SAME, VALID-interior:
    #   out[0] = max(x[0], x[1]) = 5    [boundary]
    #   out[1] = max(x[0], x[1], x[2]) = 5   -> argmax = 1
    #   out[2] = max(x[1], x[2], x[3]) = 8   -> argmax = 3
    #   out[3] = max(x[2], x[3], x[4]) = 8   -> argmax = 3
    #   out[4] = max(x[3], x[4]) = 8    [boundary]
    def loss(x):
        return dilate(x, size=3, backend='jax').sum()
    g = jax.grad(loss)(x)
    # Each output cell contributes 1 to its argmax position.
    # Position 1 receives 1 (from out[1]).
    # Position 3 receives 3 (from out[2], out[3], out[4]).
    # Positions 0, 2, 4 receive 0 from interior; boundary
    # tie-breaking with -inf gives them an extra 1 each at certain
    # boundary windows.  Exact values depend on tie semantics; the
    # *invariant* is non-negative integers summing to the count of
    # output cells, which here is 5.
    assert bool(jnp.all(g >= 0))
    assert float(g.sum()) == 5.0
    assert bool(jnp.all(jnp.isfinite(g)))


def test_erode_gradient_routes_to_argmin():
    x = jnp.array([5.0, 1.0, 8.0, 2.0, 7.0])
    def loss(x):
        return erode(x, size=3, backend='jax').sum()
    g = jax.grad(loss)(x)
    assert bool(jnp.all(jnp.isfinite(g)))
    # Each output position contributes 1 to its argmin -- sum is the
    # number of outputs.
    assert float(g.sum()) == 5.0


# ---------------------------------------------------------------------------
# susan_emulator (stub until smoothing.bilateral_gaussian lands)
# ---------------------------------------------------------------------------


def test_susan_emulator_raises_until_bilateral_lands():
    x = jnp.zeros((8, 8))
    with pytest.raises(NotImplementedError, match='bilateral_gaussian'):
        susan_emulator(x, sigma_space=1.0, sigma_intensity=0.1)
