# -*- coding: utf-8 -*-
"""Tests for ``nitrix.signal`` -- windowing (and later filter/tsconv/interpolate)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.signal import sample_windows


# ---------------------------------------------------------------------------
# sample_windows
# ---------------------------------------------------------------------------


def test_sample_windows_nonoverlapping_concatenates():
    x = jnp.arange(100, dtype=jnp.float32).reshape(10, 10)
    w = sample_windows(
        x, window_size=3, num_windows=3,
        allow_overlap=False, create_new_axis=False,
        windowing_axis=-1, multiplying_axis=0,
        key=jax.random.key(0),
    )
    # 3 windows of (10, 3) concatenated along axis 0 -> (30, 3)
    assert w.shape == (30, 3)


def test_sample_windows_nonoverlapping_new_axis():
    x = jnp.arange(100, dtype=jnp.float32).reshape(10, 10)
    w = sample_windows(
        x, window_size=3, num_windows=3,
        allow_overlap=False, create_new_axis=True,
        windowing_axis=-1, multiplying_axis=0,
        key=jax.random.key(0),
    )
    # 3 windows of (10, 3) stacked into new axis 0 -> (3, 10, 3)
    assert w.shape == (3, 10, 3)


def test_sample_windows_overlapping_new_axis():
    x = jnp.arange(100, dtype=jnp.float32).reshape(10, 10)
    w = sample_windows(
        x, window_size=4, num_windows=5,
        allow_overlap=True, create_new_axis=True,
        windowing_axis=-1, multiplying_axis=0,
        key=jax.random.key(0),
    )
    assert w.shape == (5, 10, 4)


def test_sample_windows_raises_on_oversize_request():
    x = jnp.arange(20, dtype=jnp.float32).reshape(2, 10)
    # Can't fit 5 non-overlapping windows of size 3 in 10 values.
    with pytest.raises(ValueError, match='cannot fit'):
        sample_windows(
            x, window_size=3, num_windows=5,
            allow_overlap=False, create_new_axis=True,
            windowing_axis=-1, multiplying_axis=0,
            key=jax.random.key(0),
        )


def test_sample_windows_deterministic_under_same_key():
    x = jnp.arange(50, dtype=jnp.float32)
    key = jax.random.key(42)
    w1 = sample_windows(
        x, window_size=5, num_windows=3,
        allow_overlap=True, create_new_axis=True,
        windowing_axis=-1, multiplying_axis=0, key=key,
    )
    w2 = sample_windows(
        x, window_size=5, num_windows=3,
        allow_overlap=True, create_new_axis=True,
        windowing_axis=-1, multiplying_axis=0, key=key,
    )
    np.testing.assert_array_equal(w1, w2)


def test_sample_windows_differentiable():
    '''Windows are dynamic_slice; gradients flow through the slice
    values (not through the random key).
    '''
    x = jnp.arange(50, dtype=jnp.float32)
    def loss(x):
        w = sample_windows(
            x, window_size=5, num_windows=2,
            allow_overlap=True, create_new_axis=True,
            windowing_axis=-1, multiplying_axis=0,
            key=jax.random.key(0),
        )
        return jnp.sum(w ** 2)
    g = jax.grad(loss)(x)
    assert g.shape == x.shape
    assert bool(jnp.all(jnp.isfinite(g)))
