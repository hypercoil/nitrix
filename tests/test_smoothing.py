# -*- coding: utf-8 -*-
"""Tests for ``nitrix.smoothing``.

Coverage:

- ``gaussian``: bit-exact parity with ``scipy.ndimage.gaussian_filter``
  across 1D / 2D / 3D / 4D, isotropic and anisotropic sigma, both
  ``reflect`` and ``constant`` boundary modes.
- ``bilateral_gaussian``: matches a hand-written direct-N-body
  reference on small inputs; preserves edges (step-edge contrast
  retained); approaches ``gaussian`` in the large-``sigma_intensity``
  limit.
- ``susan_emulator``: shape preservation; edge preservation vs.
  gaussian on a step image; ``use_median=True`` path produces
  finite outputs.

Sectioned ELL has its own coverage in ``test_ell_sectioned.py``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

scipy_ndi = pytest.importorskip('scipy.ndimage')

from nitrix.smoothing import (
    bilateral_gaussian,
    brute_force_knn,
    gaussian,
    susan_emulator,
)


jax.config.update('jax_enable_x64', True)


# ---------------------------------------------------------------------------
# gaussian
# ---------------------------------------------------------------------------


def test_gaussian_1d_matches_scipy():
    x = jax.random.normal(jax.random.key(0), (32,))
    got = gaussian(x, sigma=1.5, mode='reflect')
    ref = scipy_ndi.gaussian_filter(
        np.asarray(x), sigma=1.5, mode='reflect', truncate=4.0,
    )
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_gaussian_2d_matches_scipy_isotropic():
    x = jax.random.normal(jax.random.key(1), (32, 32))
    got = gaussian(x, sigma=1.5, mode='reflect')
    ref = scipy_ndi.gaussian_filter(
        np.asarray(x), sigma=1.5, mode='reflect', truncate=4.0,
    )
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_gaussian_2d_matches_scipy_anisotropic():
    x = jax.random.normal(jax.random.key(2), (32, 32))
    got = gaussian(x, sigma=(1.0, 2.0), mode='reflect')
    ref = scipy_ndi.gaussian_filter(
        np.asarray(x), sigma=(1.0, 2.0), mode='reflect', truncate=4.0,
    )
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_gaussian_3d_matches_scipy():
    x = jax.random.normal(jax.random.key(3), (16, 16, 16))
    got = gaussian(x, sigma=1.0)
    ref = scipy_ndi.gaussian_filter(
        np.asarray(x), sigma=1.0, mode='reflect', truncate=4.0,
    )
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_gaussian_4d_fmri_shape_matches_scipy():
    x = jax.random.normal(jax.random.key(4), (8, 8, 8, 8))
    got = gaussian(x, sigma=0.8)
    ref = scipy_ndi.gaussian_filter(
        np.asarray(x), sigma=0.8, mode='reflect', truncate=4.0,
    )
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_gaussian_constant_mode_matches_scipy():
    x = jax.random.normal(jax.random.key(5), (16,))
    got = gaussian(x, sigma=1.0, mode='constant')
    ref = scipy_ndi.gaussian_filter(
        np.asarray(x), sigma=1.0, mode='constant', truncate=4.0,
    )
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_gaussian_constant_input_preserved():
    const = jnp.ones((8, 8)) * 3.14
    got = gaussian(const, sigma=2.0)
    np.testing.assert_allclose(got, 3.14, atol=1e-10)


def test_gaussian_differentiable():
    x = jax.random.normal(jax.random.key(6), (16, 16))
    def loss(x):
        return gaussian(x, sigma=1.5).sum()
    g = jax.grad(loss)(x)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert g.shape == x.shape


def test_gaussian_rejects_zero_sigma():
    x = jnp.ones((4, 4))
    with pytest.raises(ValueError, match='sigma'):
        gaussian(x, sigma=0.0)


def test_gaussian_rejects_unknown_mode():
    x = jnp.ones((4,))
    with pytest.raises(ValueError, match='mode'):
        gaussian(x, sigma=1.0, mode='wrap')


# ---------------------------------------------------------------------------
# kernel_size override (JOSA J.2b)
# ---------------------------------------------------------------------------


def test_gaussian_kernel_size_odd_centered():
    '''Odd kernel_size produces a symmetric, on-grid kernel.'''
    x = jax.random.normal(jax.random.key(0), (8, 8))
    out = gaussian(x, sigma=1.0, kernel_size=5)
    assert out.shape == x.shape
    # 5-tap with sigma=1 is similar to default (truncate=4 -> half=4 -> 9-tap)
    # but truncated.  Constant preservation should still hold.
    const = jnp.ones((8, 8))
    out_const = gaussian(const, sigma=1.0, kernel_size=5)
    np.testing.assert_allclose(out_const, 1.0, atol=1e-6)


def test_gaussian_kernel_size_even_half_pixel_shift():
    '''Even kernel_size produces a half-pixel-shifted output.

    For input [0, 1, 2, 3, 4] with kernel_size=2 and a near-flat
    Gaussian (large sigma -> weights ~ [0.5, 0.5]), the output at
    index i is approximately (x[i] + x[i+1]) / 2 -- shifted by
    half a pixel.  At the right boundary, edge replication kicks
    in: result[-1] ~= (x[-1] + x[-1]) / 2 = x[-1].
    '''
    x = jnp.arange(5, dtype=jnp.float64)
    out = gaussian(x, sigma=1000.0, kernel_size=2)  # uniform weights
    expected = jnp.array([0.5, 1.5, 2.5, 3.5, 4.0])  # last is replicated
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_gaussian_kernel_size_2_sigma_0p7_josa_kernel():
    '''Specific JOSA NJF case: kernel_size=2, sigma=0.7.

    With taps at -0.5 / +0.5, the Gaussian weights at sigma=0.7
    are equal: exp(-0.5 * (0.5 / 0.7)**2) for both taps.  After
    normalisation, kernel = [0.5, 0.5].
    '''
    from nitrix.smoothing.gaussian import _gaussian_1d_kernel
    k = _gaussian_1d_kernel(0.7, 4.0, jnp.float64, kernel_size=2)
    np.testing.assert_allclose(np.asarray(k), [0.5, 0.5], atol=1e-12)


def test_gaussian_per_axis_kernel_size():
    '''Per-axis kernel_size sequence; None entries use the heuristic.'''
    x = jax.random.normal(jax.random.key(0), (8, 8, 8))
    out = gaussian(x, sigma=(1.0, 0.7, 1.5), kernel_size=(3, 2, None))
    assert out.shape == x.shape


def test_gaussian_kernel_size_default_matches_truncate():
    '''kernel_size=None reproduces the prior truncate-based behaviour
    exactly -- regression check that we didn't change the default.
    '''
    x = jax.random.normal(jax.random.key(0), (16,))
    out_default = gaussian(x, sigma=1.5, truncate=4.0)
    out_explicit_none = gaussian(
        x, sigma=1.5, truncate=4.0, kernel_size=None,
    )
    np.testing.assert_array_equal(out_default, out_explicit_none)


def test_gaussian_kernel_size_rejects_zero():
    x = jnp.ones((4,))
    with pytest.raises(ValueError, match='kernel_size must be >= 1'):
        gaussian(x, sigma=1.0, kernel_size=0)


# ---------------------------------------------------------------------------
# bilateral_gaussian
# ---------------------------------------------------------------------------


def _bilateral_reference(values, features, sigma_features, k):
    '''Direct hand-coded bilateral over feature-space k-NN.

    Used as ground truth for the JAX implementation.
    '''
    n = values.shape[0]
    out = np.zeros_like(np.asarray(values))
    feats = np.asarray(features)
    vals = np.asarray(values)
    sf = np.asarray(sigma_features)
    for i in range(n):
        # Distances to all other points in rescaled feature space.
        diff = (feats[i] - feats) / sf
        d2 = (diff ** 2).sum(axis=-1)
        # k nearest
        idx = np.argpartition(d2, k - 1)[:k]
        w = np.exp(-0.5 * d2[idx])
        w = w / w.sum()
        out[i] = (w[:, None] * vals[idx]).sum(axis=0)
    return out


def test_bilateral_matches_handwritten_reference():
    n = 16
    values = jax.random.normal(jax.random.key(20), (n, 2))
    features = jax.random.normal(jax.random.key(21), (n, 3))
    sigma = jnp.array([1.0, 1.0, 1.5])
    got = bilateral_gaussian(
        values, features,
        sigma_features=sigma, neighbourhood=5,
        backend='jax',
    )
    ref = _bilateral_reference(values, features, sigma, k=5)
    np.testing.assert_allclose(got, ref, atol=1e-12, rtol=1e-12)


def test_bilateral_preserves_step_edge():
    '''A step image should retain its contrast under bilateral with
    small sigma_intensity; gaussian with the same sigma_space would
    not.
    '''
    step = jnp.where(
        jnp.arange(16)[None, :] < 8, 0.0, 5.0,
    ).astype(jnp.float64)
    step_2d = jnp.broadcast_to(step, (16, 16))
    out = susan_emulator(
        step_2d, sigma_space=2.0, sigma_intensity=0.5, use_median=False,
    )
    out_gauss = gaussian(step_2d, sigma=2.0)
    # At the edge boundary (columns 7-8), bilateral should preserve
    # more contrast than gaussian.
    edge_bilat = float(out[8, 8] - out[8, 7])
    edge_gauss = float(out_gauss[8, 8] - out_gauss[8, 7])
    assert edge_bilat > edge_gauss
    # Bilateral with small sigma_intensity should preserve the step
    # almost exactly.
    assert edge_bilat > 4.9   # original step was 5.0


def test_bilateral_approaches_gaussian_at_large_sigma_intensity():
    '''In the limit ``sigma_intensity -> infinity``, the intensity
    contribution to the weights vanishes and bilateral reduces to
    a spatial Gaussian.
    '''
    n = 32
    spatial = jnp.arange(n, dtype=jnp.float64)[:, None]
    intensity = jnp.sin(spatial * 0.5)
    features = jnp.concatenate([spatial, intensity], axis=-1)
    values = intensity   # smooth intensity itself
    sigma_huge = jnp.array([1.0, 1e6])
    out_bilateral = bilateral_gaussian(
        values, features,
        sigma_features=sigma_huge,
        neighbourhood=7,
        backend='jax',
    )
    # Compare interior to a 7-tap normalised spatial Gaussian.
    half = 3
    xs = jnp.arange(-half, half + 1, dtype=jnp.float64)
    g_kernel = jnp.exp(-0.5 * xs ** 2)
    g_kernel = g_kernel / g_kernel.sum()
    ref = jnp.array([
        sum(
            g_kernel[k] * values[max(0, min(n - 1, i + k - half))][0]
            for k in range(2 * half + 1)
        )
        for i in range(n)
    ])
    interior = slice(half + 1, -(half + 1))
    np.testing.assert_allclose(
        out_bilateral[interior, 0], ref[interior], atol=1e-12,
    )


def test_bilateral_explicit_adjacency():
    '''Pre-computed adjacency: the user supplies (n, k_max) indices.'''
    n = 8
    values = jax.random.normal(jax.random.key(30), (n, 1))
    features = jnp.arange(n, dtype=jnp.float64)[:, None]
    sigma = jnp.array([1.0])
    # Adjacency: each point's two nearest neighbours, hand-built.
    adj = jnp.array([
        [0, 1], [0, 1], [1, 2], [2, 3],
        [3, 4], [4, 5], [5, 6], [6, 7],
    ])
    got = bilateral_gaussian(
        values, features,
        sigma_features=sigma, neighbourhood=adj,
        backend='jax',
    )
    assert got.shape == (n, 1)
    assert bool(jnp.all(jnp.isfinite(got)))


def test_bilateral_differentiable():
    n = 16
    values = jax.random.normal(jax.random.key(40), (n, 1))
    features = jax.random.normal(jax.random.key(41), (n, 2))
    sigma = jnp.array([1.0, 1.0])
    def loss(values):
        return bilateral_gaussian(
            values, features,
            sigma_features=sigma, neighbourhood=4,
            backend='jax',
        ).sum()
    g = jax.grad(loss)(values)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_bilateral_rejects_mismatched_n():
    values = jnp.zeros((4, 2))
    features = jnp.zeros((5, 3))
    sigma = jnp.ones(3)
    with pytest.raises(ValueError, match='n='):
        bilateral_gaussian(
            values, features,
            sigma_features=sigma, neighbourhood=2,
            backend='jax',
        )


def test_brute_force_knn_self_first():
    '''A point should always be its own nearest neighbour.'''
    features = jax.random.normal(jax.random.key(50), (10, 3))
    idx = brute_force_knn(features, k=3)
    # Each row's first index (sorted by ascending distance, distance
    # to self = 0) should be the row index itself.
    np.testing.assert_array_equal(idx[:, 0], jnp.arange(10))


# ---------------------------------------------------------------------------
# susan_emulator
# ---------------------------------------------------------------------------


def test_susan_emulator_preserves_shape():
    img = jax.random.normal(jax.random.key(60), (8, 8))
    out = susan_emulator(img, sigma_space=1.0, sigma_intensity=1.0)
    assert out.shape == img.shape
    assert bool(jnp.all(jnp.isfinite(out)))


def test_susan_emulator_3d_volume():
    vol = jax.random.normal(jax.random.key(61), (6, 6, 6))
    out = susan_emulator(vol, sigma_space=1.5, sigma_intensity=2.0)
    assert out.shape == vol.shape
    assert bool(jnp.all(jnp.isfinite(out)))


def test_susan_emulator_with_median_prepass():
    img = jax.random.normal(jax.random.key(62), (8, 8))
    # Inject impulse noise
    img = img.at[3, 3].set(100.0)
    img = img.at[5, 5].set(-100.0)
    out_no_med = susan_emulator(
        img, sigma_space=1.0, sigma_intensity=10.0, use_median=False,
    )
    out_med = susan_emulator(
        img, sigma_space=1.0, sigma_intensity=10.0, use_median=True,
    )
    # The impulse-noise positions should be more attenuated under
    # use_median than without.
    assert abs(float(out_med[3, 3])) < abs(float(out_no_med[3, 3]))
    assert abs(float(out_med[5, 5])) < abs(float(out_no_med[5, 5]))


def test_permutohedral_lattice_raises_with_pointer():
    '''Per SPEC_UPDATE §3.3 the symbol raises NotImplementedError
    at first GA, with a clear pointer at ``bilateral_gaussian``
    and the tripwire rationale.  See docs/design/permutohedral-g2.md.
    '''
    from nitrix.smoothing import permutohedral_lattice
    with pytest.raises(NotImplementedError, match='bilateral_gaussian'):
        permutohedral_lattice(
            jnp.zeros((4, 1)),
            jnp.zeros((4, 2)),
            sigma_features=jnp.array([1.0, 1.0]),
        )


def test_susan_emulator_canonical_home_is_smoothing():
    '''SUSAN lives in ``nitrix.smoothing`` per SPEC_UPDATE §3.3.

    It is *not* re-exported from ``nitrix.morphology`` to avoid a
    circular import with ``median_filter``; users should import it
    from ``nitrix.smoothing``.
    '''
    from nitrix import morphology, smoothing
    assert hasattr(smoothing, 'susan_emulator')
    assert not hasattr(morphology, 'susan_emulator')
