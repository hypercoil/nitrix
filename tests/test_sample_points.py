# -*- coding: utf-8 -*-
"""Tests for ``nitrix.geometry.sample_at_points`` (arbitrary-point sampler)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.geometry import NearestNeighbour, sample_at_points


def test_sample_exact_at_integer_points():
    vol = jnp.arange(16.0).reshape(4, 4)
    pts = jnp.asarray([[0.0, 0.0], [1.0, 2.0], [3.0, 3.0]])
    out = sample_at_points(vol, pts)
    np.testing.assert_allclose(
        np.asarray(out), [0.0, vol[1, 2], vol[3, 3]], atol=1e-6
    )


def test_sample_linear_midpoint():
    vol = jnp.arange(16.0).reshape(4, 4)
    pts = jnp.asarray([[0.5, 0.0]])  # midway between (0,0) and (1,0)
    out = sample_at_points(vol, pts)
    np.testing.assert_allclose(
        float(out[0]), float((vol[0, 0] + vol[1, 0]) / 2), atol=1e-6
    )


def test_sample_zero_fill_out_of_bounds():
    vol = jnp.ones((4, 4))
    pts = jnp.asarray([[-1.0, -1.0], [10.0, 10.0]])
    out = sample_at_points(vol, pts, mode='constant', cval=0.0)
    np.testing.assert_allclose(np.asarray(out), 0.0, atol=1e-6)


def test_sample_border_clamp():
    vol = jnp.arange(16.0).reshape(4, 4)
    pts = jnp.asarray([[-5.0, -5.0], [99.0, 99.0]])
    out = sample_at_points(vol, pts, mode='nearest')
    # Clamps to the nearest in-bounds corner.
    np.testing.assert_allclose(
        np.asarray(out), [float(vol[0, 0]), float(vol[3, 3])], atol=1e-6
    )


def test_sample_multichannel():
    vol = jnp.stack(
        [jnp.arange(16.0).reshape(4, 4), jnp.ones((4, 4)) * 7.0], axis=-1
    )  # (4, 4, 2)
    pts = jnp.asarray([[1.0, 1.0], [2.0, 3.0]])
    out = sample_at_points(vol, pts)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(np.asarray(out)[:, 1], 7.0, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(out)[:, 0], [float(vol[1, 1, 0]), float(vol[2, 3, 0])],
        atol=1e-6,
    )


def test_sample_nearest_method_3d():
    vol = jnp.arange(27.0).reshape(3, 3, 3)
    pts = jnp.asarray([[0.4, 0.0, 0.0], [1.6, 2.0, 2.0]])
    out = sample_at_points(vol, pts, method=NearestNeighbour())
    np.testing.assert_allclose(
        np.asarray(out), [float(vol[0, 0, 0]), float(vol[2, 2, 2])], atol=1e-6
    )


def test_sample_differentiable():
    vol = jnp.asarray(np.random.default_rng(0).standard_normal((6, 6)))
    pts = jnp.asarray([[1.5, 2.5], [3.1, 0.7]])
    # Gradient w.r.t. the sample coordinates is well-defined for linear.
    g = jax.grad(lambda p: jnp.sum(sample_at_points(vol, p)))(pts)
    assert g.shape == pts.shape
    assert bool(jnp.all(jnp.isfinite(g)))
