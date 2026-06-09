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
    gibbs_ringing,
    gmm_label_to_image,
    random_affine_matrix,
    random_crop,
    random_flip,
    random_histogram_shift,
    random_resized_crop,
    random_svf_displacement,
    rician_noise,
    simulate_bias_field,
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


# ---------------------------------------------------------------------------
# geometric: flip / crop / resized-crop / affine / svf
# ---------------------------------------------------------------------------


def test_random_flip_p1_single_axis_flips():
    x = jnp.asarray(np.random.default_rng(0).standard_normal((5, 6)))
    out = random_flip(x, jax.random.PRNGKey(0), axes=[0], p=1.0)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(jnp.flip(x, 0)))


def test_random_flip_p0_is_identity():
    x = jnp.asarray(np.random.default_rng(1).standard_normal((4, 4, 3)))
    out = random_flip(x, jax.random.PRNGKey(1), p=0.0)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x))


def test_random_crop_shape_and_membership():
    x = jnp.arange(10 * 12).reshape(10, 12).astype(jnp.float32)
    out = random_crop(x, jax.random.PRNGKey(2), size=(4, 5))
    assert out.shape == (4, 5)
    # The crop is a contiguous block of x: top-left value determines it.
    tl = int(out[0, 0])
    r, c = divmod(tl, 12)
    np.testing.assert_array_equal(
        np.asarray(out), np.asarray(x)[r : r + 4, c : c + 5]
    )


def test_random_crop_full_size_is_identity():
    x = jnp.asarray(np.random.default_rng(3).standard_normal((3, 4)))
    out = random_crop(x, jax.random.PRNGKey(3), size=(3, 4))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x))


def test_random_resized_crop_full_scale_is_identity():
    x = jnp.asarray(np.random.default_rng(4).standard_normal((8, 8, 2)))
    out = random_resized_crop(
        x, jax.random.PRNGKey(4), size=(8, 8), scale_range=(1.0, 1.0)
    )
    assert out.shape == (8, 8, 2)
    np.testing.assert_allclose(np.asarray(out), np.asarray(x), atol=1e-4)


def test_random_resized_crop_output_shape():
    x = jnp.asarray(np.random.default_rng(5).standard_normal((16, 16, 3)))
    out = random_resized_crop(x, jax.random.PRNGKey(5), size=(8, 8))
    assert out.shape == (8, 8, 3)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_random_affine_matrix_identity_at_zero_bounds():
    mat = random_affine_matrix(
        jax.random.PRNGKey(6),
        max_rotation=0.0,
        max_scale=0.0,
        max_shear=0.0,
        max_translation=0.0,
    )
    expected = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=-1)
    assert mat.shape == (3, 4)
    np.testing.assert_allclose(np.asarray(mat), expected, atol=1e-6)


def test_random_affine_matrix_reproducible():
    a = random_affine_matrix(jax.random.PRNGKey(7))
    b = random_affine_matrix(jax.random.PRNGKey(7))
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_random_svf_zero_std_is_zero_field():
    disp = random_svf_displacement(
        (16, 16, 16), jax.random.PRNGKey(8), max_std=0.0
    )
    assert disp.shape == (16, 16, 16, 3)
    np.testing.assert_allclose(np.asarray(disp), 0.0, atol=1e-6)


def test_random_svf_shape_and_finite():
    disp = random_svf_displacement((24, 24, 24), jax.random.PRNGKey(9))
    assert disp.shape == (24, 24, 24, 3)
    assert bool(jnp.all(jnp.isfinite(disp)))


# ---------------------------------------------------------------------------
# synthesis: gmm_label_to_image / simulate_bias_field
# ---------------------------------------------------------------------------


def test_gmm_zero_std_is_mean_gather():
    labels = jnp.asarray([[0, 1, 2], [2, 1, 0]])
    means = jnp.asarray([10.0, 20.0, 30.0])
    stds = jnp.zeros(3)
    out = gmm_label_to_image(labels, means, stds, jax.random.PRNGKey(0))
    np.testing.assert_array_equal(
        np.asarray(out), np.asarray(means)[np.asarray(labels)]
    )


def test_gmm_matches_explicit_formula():
    rng = np.random.default_rng(0)
    labels = jnp.asarray(rng.integers(0, 4, size=(8, 8)))
    means = jnp.asarray([1.0, -3.0, 5.0, 2.0])
    stds = jnp.asarray([0.5, 1.0, 0.2, 0.8])
    key = jax.random.PRNGKey(1)
    out = gmm_label_to_image(labels, means, stds, key, nonneg=False)
    noise = jax.random.normal(key, labels.shape, dtype=means.dtype)
    ref = means[labels] + stds[labels] * noise
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), atol=1e-6)


def test_gmm_nonneg_clamps():
    labels = jnp.zeros((50,), dtype=jnp.int32)
    means = jnp.asarray([-5.0])
    stds = jnp.asarray([1.0])
    out = gmm_label_to_image(labels, means, stds, jax.random.PRNGKey(2))
    assert float(out.min()) >= 0.0


def test_simulate_bias_field_zero_std_is_ones():
    field = simulate_bias_field((16, 16, 16), jax.random.PRNGKey(0), max_std=0.0)
    assert field.shape == (16, 16, 16)
    np.testing.assert_allclose(np.asarray(field), 1.0, atol=1e-6)


def test_simulate_bias_field_positive_and_reproducible():
    a = simulate_bias_field((20, 20), jax.random.PRNGKey(3), max_std=0.5)
    b = simulate_bias_field((20, 20), jax.random.PRNGKey(3), max_std=0.5)
    assert a.shape == (20, 20)
    assert float(a.min()) > 0.0  # exp is strictly positive
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------------------
# gibbs_ringing
# ---------------------------------------------------------------------------


def test_gibbs_alpha_zero_is_identity():
    x = jnp.asarray(np.random.default_rng(0).standard_normal((16, 16)))
    out = gibbs_ringing(x, 0.0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(x), atol=1e-6)


def test_gibbs_rings_at_a_step_edge():
    # A sharp step is the canonical Gibbs trigger: truncation overshoots
    # beyond the original [0, 1] range near the discontinuity.
    x = jnp.concatenate([jnp.zeros(64), jnp.ones(64)])
    out = np.asarray(gibbs_ringing(x, 0.5))
    assert out.min() < -1e-3 or out.max() > 1.0 + 1e-3


def test_gibbs_shape_and_finite_3d():
    x = jnp.asarray(np.random.default_rng(1).standard_normal((12, 12, 12)))
    out = gibbs_ringing(x, 0.3)
    assert out.shape == x.shape
    assert bool(jnp.all(jnp.isfinite(out)))


def test_gibbs_axes_subset_channels_last():
    # Apply over spatial axes only; output keeps shape and is real/finite.
    x = jnp.asarray(np.random.default_rng(2).standard_normal((16, 16, 3)))
    out = gibbs_ringing(x, 0.4, axes=(0, 1))
    assert out.shape == x.shape
    assert bool(jnp.all(jnp.isfinite(out)))


def test_gibbs_differentiable():
    x = jnp.asarray(np.random.default_rng(3).standard_normal((8, 8)))
    g = jax.grad(lambda z: jnp.sum(gibbs_ringing(z, 0.5) ** 2))(x)
    assert g.shape == x.shape
    assert bool(jnp.all(jnp.isfinite(g)))
