# -*- coding: utf-8 -*-
"""Tests for ``nitrix.augment`` -- pure-numeric augmentation primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.augment import (
    gamma_contrast,
    gaussian_noise,
    random_histogram_shift,
    rician_noise,
)

# ---------------------------------------------------------------------------
# gamma_contrast
# ---------------------------------------------------------------------------


def test_gamma_identity_at_gamma_one():
    x = jnp.asarray(np.random.default_rng(0).standard_normal((4, 8)))
    out = gamma_contrast(x, jnp.asarray(1.0))
    np.testing.assert_allclose(np.asarray(out), np.asarray(x), atol=1e-6)


def test_gamma_preserves_bracket_endpoints():
    x = jnp.linspace(3.0, 7.0, 50)
    out = gamma_contrast(x, jnp.asarray(2.5))
    # min/max are fixed points of the tone curve.
    np.testing.assert_allclose(float(out.min()), 3.0, atol=1e-5)
    np.testing.assert_allclose(float(out.max()), 7.0, atol=1e-5)


def test_gamma_matches_formula_with_explicit_range():
    x = jnp.asarray(np.random.default_rng(1).uniform(0, 1, size=200))
    g = 0.5
    out = gamma_contrast(x, jnp.asarray(g), value_range=(0.0, 1.0))
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(x) ** g, atol=1e-6
    )


def test_gamma_differentiable():
    x = jnp.asarray(np.random.default_rng(2).uniform(0.1, 0.9, size=(3, 5)))
    g = jax.grad(lambda z: jnp.sum(gamma_contrast(z, jnp.asarray(1.7))))(x)
    assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# random_histogram_shift
# ---------------------------------------------------------------------------


def test_histogram_shift_preserves_range_and_monotone():
    x = jnp.asarray(np.random.default_rng(3).standard_normal(2000))
    out = random_histogram_shift(x, jax.random.PRNGKey(0), n_control_points=8)
    # Endpoints pinned -> global min/max preserved.
    np.testing.assert_allclose(float(out.min()), float(x.min()), atol=1e-4)
    np.testing.assert_allclose(float(out.max()), float(x.max()), atol=1e-4)
    # Monotone remap: order is preserved.
    order_in = np.argsort(np.asarray(x))
    remapped = np.asarray(out)[order_in]
    assert bool(np.all(np.diff(remapped) >= -1e-6))


def test_histogram_shift_reproducible_and_shape():
    x = jnp.asarray(np.random.default_rng(4).standard_normal((6, 7, 8)))
    a = random_histogram_shift(x, jax.random.PRNGKey(1))
    b = random_histogram_shift(x, jax.random.PRNGKey(1))
    assert a.shape == x.shape
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------------------
# gaussian_noise / rician_noise
# ---------------------------------------------------------------------------


def test_gaussian_noise_zero_sigma_is_identity():
    x = jnp.asarray(np.random.default_rng(5).standard_normal((4, 4)))
    out = gaussian_noise(x, jax.random.PRNGKey(0), sigma=jnp.asarray(0.0))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x))


def test_gaussian_noise_residual_has_expected_std():
    x = jnp.zeros(100000)
    out = gaussian_noise(x, jax.random.PRNGKey(2), sigma=jnp.asarray(0.3))
    np.testing.assert_allclose(float(out.std()), 0.3, rtol=5e-2)


def test_rician_noise_zero_sigma_is_abs():
    x = jnp.asarray([-2.0, -0.5, 0.0, 1.0, 3.0])
    out = rician_noise(x, jax.random.PRNGKey(0), sigma=jnp.asarray(0.0))
    np.testing.assert_allclose(np.asarray(out), np.abs(np.asarray(x)), atol=1e-6)


def test_rician_noise_nonnegative():
    x = jnp.asarray(np.random.default_rng(6).standard_normal((20, 20)))
    out = rician_noise(x, jax.random.PRNGKey(3), sigma=jnp.asarray(0.2))
    assert float(out.min()) >= 0.0
