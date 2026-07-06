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
    DiagonalMetric,
    FactorMetric,
    bilateral_gaussian,
    block_diagonal_metric,
    brute_force_knn,
    gaussian,
    metric_from_spd,
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
        np.asarray(x),
        sigma=1.5,
        mode='reflect',
        truncate=4.0,
    )
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_gaussian_2d_matches_scipy_isotropic():
    x = jax.random.normal(jax.random.key(1), (32, 32))
    got = gaussian(x, sigma=1.5, mode='reflect')
    ref = scipy_ndi.gaussian_filter(
        np.asarray(x),
        sigma=1.5,
        mode='reflect',
        truncate=4.0,
    )
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_gaussian_2d_matches_scipy_anisotropic():
    x = jax.random.normal(jax.random.key(2), (32, 32))
    got = gaussian(x, sigma=(1.0, 2.0), mode='reflect')
    ref = scipy_ndi.gaussian_filter(
        np.asarray(x),
        sigma=(1.0, 2.0),
        mode='reflect',
        truncate=4.0,
    )
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_gaussian_3d_matches_scipy():
    x = jax.random.normal(jax.random.key(3), (16, 16, 16))
    got = gaussian(x, sigma=1.0)
    ref = scipy_ndi.gaussian_filter(
        np.asarray(x),
        sigma=1.0,
        mode='reflect',
        truncate=4.0,
    )
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_gaussian_4d_fmri_shape_matches_scipy():
    x = jax.random.normal(jax.random.key(4), (8, 8, 8, 8))
    got = gaussian(x, sigma=0.8)
    ref = scipy_ndi.gaussian_filter(
        np.asarray(x),
        sigma=0.8,
        mode='reflect',
        truncate=4.0,
    )
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_gaussian_constant_mode_matches_scipy():
    x = jax.random.normal(jax.random.key(5), (16,))
    got = gaussian(x, sigma=1.0, mode='constant')
    ref = scipy_ndi.gaussian_filter(
        np.asarray(x),
        sigma=1.0,
        mode='constant',
        truncate=4.0,
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
# traced sigma (jit-safe FIR path)
# ---------------------------------------------------------------------------


def test_gaussian_traced_sigma_matches_static():
    """A jit-traced sigma with explicit kernel_size matches the static path.

    Only the tap weights vary with sigma; the kernel shape is fixed by
    kernel_size, so the op traces under jit and reproduces the host-static
    result bit-for-bit at the same sigma.
    """
    x = jax.random.normal(jax.random.key(0), (16, 16, 16))
    static = gaussian(x, sigma=1.3, kernel_size=11)
    traced = jax.jit(lambda x, s: gaussian(x, sigma=s, kernel_size=11))(
        x, jnp.asarray(1.3)
    )
    np.testing.assert_allclose(traced, static, atol=1e-12, rtol=0)


def test_gaussian_traced_sigma_vmap_per_view():
    """vmap over a per-view sampled sigma (the DINO augmentation use case)."""
    x = jax.random.normal(jax.random.key(1), (12, 12))
    sig = jnp.array([0.5, 1.0, 1.5])
    xs = jnp.broadcast_to(x, (3,) + x.shape)
    out = jax.vmap(
        lambda xi, si: gaussian(xi, sigma=si, kernel_size=9)
    )(xs, sig)
    assert out.shape == (3,) + x.shape
    for i, s in enumerate([0.5, 1.0, 1.5]):
        np.testing.assert_allclose(
            out[i], gaussian(x, sigma=s, kernel_size=9), atol=1e-12
        )


def test_gaussian_traced_sigma_1d_anisotropic():
    """A 1-D traced sigma pins the rank and matches the static sequence."""
    x = jax.random.normal(jax.random.key(2), (10, 10, 10))
    sig = jnp.array([0.7, 1.0, 1.3])
    traced = jax.jit(lambda x, s: gaussian(x, sigma=s, kernel_size=9))(x, sig)
    static = gaussian(x, sigma=(0.7, 1.0, 1.3), kernel_size=9)
    np.testing.assert_allclose(traced, static, atol=1e-12)


def test_gaussian_traced_sigma_requires_kernel_size():
    """A traced sigma without kernel_size raises: the shape must be static."""
    x = jnp.ones((8, 8))
    with pytest.raises(ValueError, match='kernel_size'):
        jax.jit(lambda x, s: gaussian(x, sigma=s))(x, jnp.asarray(1.0))


def test_gaussian_traced_sigma_rejects_recursive():
    """The recursive driver needs a host-static sigma; a traced sigma raises."""
    x = jnp.ones((8, 8))
    with pytest.raises(ValueError, match='recursive'):
        gaussian(x, sigma=jnp.asarray(1.0), driver='recursive')


# ---------------------------------------------------------------------------
# kernel_size override (JOSA J.2b)
# ---------------------------------------------------------------------------


def test_gaussian_kernel_size_odd_centered():
    """Odd kernel_size produces a symmetric, on-grid kernel."""
    x = jax.random.normal(jax.random.key(0), (8, 8))
    out = gaussian(x, sigma=1.0, kernel_size=5)
    assert out.shape == x.shape
    # 5-tap with sigma=1 is similar to default (truncate=4 -> half=4 -> 9-tap)
    # but truncated.  Constant preservation should still hold.
    const = jnp.ones((8, 8))
    out_const = gaussian(const, sigma=1.0, kernel_size=5)
    np.testing.assert_allclose(out_const, 1.0, atol=1e-6)


def test_gaussian_kernel_size_even_half_pixel_shift():
    """Even kernel_size produces a half-pixel-shifted output.

    For input [0, 1, 2, 3, 4] with kernel_size=2 and a near-flat
    Gaussian (large sigma -> weights ~ [0.5, 0.5]), the output at
    index i is approximately (x[i] + x[i+1]) / 2 -- shifted by
    half a pixel.  At the right boundary, edge replication kicks
    in: result[-1] ~= (x[-1] + x[-1]) / 2 = x[-1].
    """
    x = jnp.arange(5, dtype=jnp.float64)
    out = gaussian(x, sigma=1000.0, kernel_size=2)  # uniform weights
    expected = jnp.array([0.5, 1.5, 2.5, 3.5, 4.0])  # last is replicated
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_gaussian_kernel_size_2_sigma_0p7_josa_kernel():
    """Specific JOSA NJF case: kernel_size=2, sigma=0.7.

    With taps at -0.5 / +0.5, the Gaussian weights at sigma=0.7
    are equal: exp(-0.5 * (0.5 / 0.7)**2) for both taps.  After
    normalisation, kernel = [0.5, 0.5].
    """
    from nitrix.smoothing.gaussian import _gaussian_1d_kernel

    k = _gaussian_1d_kernel(0.7, 4.0, jnp.float64, kernel_size=2)
    np.testing.assert_allclose(np.asarray(k), [0.5, 0.5], atol=1e-12)


def test_gaussian_per_axis_kernel_size():
    """Per-axis kernel_size sequence; None entries use the heuristic."""
    x = jax.random.normal(jax.random.key(0), (8, 8, 8))
    out = gaussian(x, sigma=(1.0, 0.7, 1.5), kernel_size=(3, 2, None))
    assert out.shape == x.shape


def test_gaussian_kernel_size_default_matches_truncate():
    """kernel_size=None reproduces the prior truncate-based behaviour
    exactly -- regression check that we didn't change the default.
    """
    x = jax.random.normal(jax.random.key(0), (16,))
    out_default = gaussian(x, sigma=1.5, truncate=4.0)
    out_explicit_none = gaussian(
        x,
        sigma=1.5,
        truncate=4.0,
        kernel_size=None,
    )
    np.testing.assert_array_equal(out_default, out_explicit_none)


def test_gaussian_kernel_size_rejects_zero():
    x = jnp.ones((4,))
    with pytest.raises(ValueError, match='kernel_size must be >= 1'):
        gaussian(x, sigma=1.0, kernel_size=0)


# ---------------------------------------------------------------------------
# recursive (Young-van Vliet) Gaussian -- 1d
# ---------------------------------------------------------------------------


def _blobs(n, seed):
    yy, xx = np.mgrid[0:n, 0:n].astype('float64')
    rng = np.random.RandomState(seed)
    img = np.zeros((n, n))
    for _ in range(8):
        cy, cx = rng.uniform(0.15, 0.85, 2) * n
        s = rng.uniform(0.06, 0.14) * n
        img += rng.uniform(0.4, 1.0) * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s)
        )
    return jnp.asarray(img)


def test_recursive_gaussian_preserves_constant():
    out = gaussian(jnp.full((64, 64), 3.7), sigma=4.0, driver='recursive')
    assert np.allclose(np.asarray(out), 3.7, atol=1e-9)


def test_recursive_gaussian_impulse_approximates_true_gaussian():
    # The YvV recursive filter is a Gaussian *approximation*: its impulse
    # response integrates to 1 and matches the true Gaussian profile to the
    # documented ~few-% 3rd-order bound.
    for sg in (2.0, 4.0):
        d = np.zeros(201)
        d[100] = 1.0
        rec = np.asarray(
            gaussian(
                jnp.asarray(d), sigma=sg, driver='recursive', spatial_rank=1
            )
        )
        xs = np.arange(201) - 100
        true = np.exp(-(xs**2) / (2 * sg * sg))
        true /= true.sum()
        assert abs(rec.sum() - 1.0) < 1e-6  # normalised (DC gain 1)
        assert np.linalg.norm(rec - true) / np.linalg.norm(true) < 0.1


def test_recursive_gaussian_interior_parity_with_fir():
    # On a realistic (low-frequency) signal the recursive Gaussian matches the
    # FIR Gaussian in the interior to ~1-2% -- the parity gate (it is radius-free
    # and O(N), the win at large sigma; the FIR conv stays the reference).
    img = _blobs(128, 0)
    for sg in (2.0, 4.0, 8.0):
        fir = np.asarray(gaussian(img, sigma=sg, driver='fir', truncate=5.0))
        rec = np.asarray(gaussian(img, sigma=sg, driver='recursive'))
        m = int(np.ceil(4 * sg))
        fi, ri = fir[m:-m, m:-m], rec[m:-m, m:-m]
        assert np.linalg.norm(fi - ri) / np.linalg.norm(fi) < 0.02


def test_recursive_gaussian_anisotropic_and_differentiable():
    rng = np.random.RandomState(1)
    x3 = jnp.asarray(rng.standard_normal((20, 20, 20)))
    out = gaussian(
        x3, sigma=(2.0, 3.0, 4.0), driver='recursive', spatial_rank=3
    )
    assert out.shape == (20, 20, 20)
    assert bool(jnp.all(jnp.isfinite(out)))
    img = _blobs(48, 2)
    g = jax.grad(
        lambda z: (gaussian(z, sigma=3.0, driver='recursive') ** 2).sum()
    )(img)
    fd_plus = (gaussian(img + 1e-4, sigma=3.0, driver='recursive') ** 2).sum()
    fd_minus = (gaussian(img - 1e-4, sigma=3.0, driver='recursive') ** 2).sum()
    # the directional FD along the all-ones perturbation matches grad.sum()
    assert np.isclose(
        float(g.sum()), float((fd_plus - fd_minus) / 2e-4), rtol=1e-4
    )


def test_recursive_gaussian_rejects_small_sigma_and_kernel_size():
    with pytest.raises(ValueError, match='sigma >='):
        gaussian(
            jnp.ones((16,)), sigma=0.3, driver='recursive', spatial_rank=1
        )
    with pytest.raises(ValueError, match='no kernel'):
        gaussian(
            jnp.ones((16,)),
            sigma=2.0,
            driver='recursive',
            kernel_size=5,
            spatial_rank=1,
        )


# ---------------------------------------------------------------------------
# bilateral_gaussian
# ---------------------------------------------------------------------------


def _bilateral_reference(values, features, sigma_features, k):
    """Direct hand-coded bilateral over feature-space k-NN.

    Used as ground truth for the JAX implementation.
    """
    n = values.shape[0]
    out = np.zeros_like(np.asarray(values))
    feats = np.asarray(features)
    vals = np.asarray(values)
    sf = np.asarray(sigma_features)
    for i in range(n):
        # Distances to all other points in rescaled feature space.
        diff = (feats[i] - feats) / sf
        d2 = (diff**2).sum(axis=-1)
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
        values,
        features,
        metric=DiagonalMetric(sigma),
        neighbourhood=5,
        backend='jax',
    )
    ref = _bilateral_reference(values, features, sigma, k=5)
    np.testing.assert_allclose(got, ref, atol=1e-12, rtol=1e-12)


def test_bilateral_preserves_step_edge():
    """A step image should retain its contrast under bilateral with
    small sigma_intensity; gaussian with the same sigma_space would
    not.
    """
    step = jnp.where(
        jnp.arange(16)[None, :] < 8,
        0.0,
        5.0,
    ).astype(jnp.float64)
    step_2d = jnp.broadcast_to(step, (16, 16))
    out = susan_emulator(
        step_2d,
        sigma_space=2.0,
        sigma_intensity=0.5,
        use_median=False,
    )
    out_gauss = gaussian(step_2d, sigma=2.0)
    # At the edge boundary (columns 7-8), bilateral should preserve
    # more contrast than gaussian.
    edge_bilat = float(out[8, 8] - out[8, 7])
    edge_gauss = float(out_gauss[8, 8] - out_gauss[8, 7])
    assert edge_bilat > edge_gauss
    # Bilateral with small sigma_intensity should preserve the step
    # almost exactly.
    assert edge_bilat > 4.9  # original step was 5.0


def test_bilateral_approaches_gaussian_at_large_sigma_intensity():
    """In the limit ``sigma_intensity -> infinity``, the intensity
    contribution to the weights vanishes and bilateral reduces to
    a spatial Gaussian.
    """
    n = 32
    spatial = jnp.arange(n, dtype=jnp.float64)[:, None]
    intensity = jnp.sin(spatial * 0.5)
    features = jnp.concatenate([spatial, intensity], axis=-1)
    values = intensity  # smooth intensity itself
    sigma_huge = jnp.array([1.0, 1e6])
    out_bilateral = bilateral_gaussian(
        values,
        features,
        metric=DiagonalMetric(sigma_huge),
        neighbourhood=7,
        backend='jax',
    )
    # Compare interior to a 7-tap normalised spatial Gaussian.
    half = 3
    xs = jnp.arange(-half, half + 1, dtype=jnp.float64)
    g_kernel = jnp.exp(-0.5 * xs**2)
    g_kernel = g_kernel / g_kernel.sum()
    ref = jnp.array(
        [
            sum(
                g_kernel[k] * values[max(0, min(n - 1, i + k - half))][0]
                for k in range(2 * half + 1)
            )
            for i in range(n)
        ]
    )
    interior = slice(half + 1, -(half + 1))
    np.testing.assert_allclose(
        out_bilateral[interior, 0],
        ref[interior],
        atol=1e-12,
    )


def test_bilateral_explicit_adjacency():
    """Pre-computed adjacency: the user supplies (n, k_max) indices."""
    n = 8
    values = jax.random.normal(jax.random.key(30), (n, 1))
    features = jnp.arange(n, dtype=jnp.float64)[:, None]
    sigma = jnp.array([1.0])
    # Adjacency: each point's two nearest neighbours, hand-built.
    adj = jnp.array(
        [
            [0, 1],
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
        ]
    )
    got = bilateral_gaussian(
        values,
        features,
        metric=DiagonalMetric(sigma),
        neighbourhood=adj,
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
            values,
            features,
            metric=DiagonalMetric(sigma),
            neighbourhood=4,
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
            values,
            features,
            metric=DiagonalMetric(sigma),
            neighbourhood=2,
            backend='jax',
        )


def test_brute_force_knn_self_first():
    """A point should always be its own nearest neighbour."""
    features = jax.random.normal(jax.random.key(50), (10, 3))
    idx = brute_force_knn(features, k=3)
    # Each row's first index (sorted by ascending distance, distance
    # to self = 0) should be the row index itself.
    np.testing.assert_array_equal(idx[:, 0], jnp.arange(10))


# ---------------------------------------------------------------------------
# bounded bilateral: factored metric, mask, iteration
# ---------------------------------------------------------------------------


def test_bilateral_diagonal_equals_factor_diag():
    """DiagonalMetric(sigma) == FactorMetric(diag(1/sigma))."""
    n = 16
    values = jax.random.normal(jax.random.key(70), (n, 2))
    features = jax.random.normal(jax.random.key(71), (n, 3))
    sigma = jnp.array([1.0, 2.0, 0.5])
    out_diag = bilateral_gaussian(
        values,
        features,
        metric=DiagonalMetric(sigma),
        neighbourhood=5,
        backend='jax',
    )
    out_factor = bilateral_gaussian(
        values,
        features,
        metric=FactorMetric(jnp.diag(1.0 / sigma)),
        neighbourhood=5,
        backend='jax',
    )
    np.testing.assert_allclose(out_diag, out_factor, atol=1e-12)


def test_bilateral_full_factor_matches_dense_quadratic():
    """A full FactorMetric weights by exp(-1/2 d^T M d), M = L L^T."""
    n = 12
    values = jax.random.normal(jax.random.key(72), (n, 1))
    features = jax.random.normal(jax.random.key(73), (n, 3))
    a = jax.random.normal(jax.random.key(74), (3, 3))
    L = jnp.linalg.cholesky(a @ a.T + 3.0 * jnp.eye(3))
    M = L @ L.T
    # Explicit (n, k_max) adjacency so the reference shares it.
    adj = brute_force_knn(features, 4)
    got = bilateral_gaussian(
        values,
        features,
        metric=FactorMetric(L),
        neighbourhood=adj,
        backend='jax',
    )
    # Direct reference using the dense quadratic form.
    f = np.asarray(features)
    v = np.asarray(values)
    idx = np.asarray(adj)
    Mnp = np.asarray(M)
    ref = np.zeros_like(v)
    for i in range(n):
        d = f[i] - f[idx[i]]
        q = np.einsum('kd,de,ke->k', d, Mnp, d)
        w = np.exp(-0.5 * q)
        w = w / w.sum()
        ref[i] = (w[:, None] * v[idx[i]]).sum(0)
    np.testing.assert_allclose(got, ref, atol=1e-10)


def test_bilateral_low_rank_factor_projects():
    """A low-rank factor (k < d_f) costs like its rank: q uses L only."""
    n = 10
    values = jax.random.normal(jax.random.key(75), (n, 1))
    features = jax.random.normal(jax.random.key(76), (n, 4))
    L = jax.random.normal(jax.random.key(77), (4, 2))  # rank-2 metric
    adj = brute_force_knn(features, 4)
    got = bilateral_gaussian(
        values,
        features,
        metric=FactorMetric(L),
        neighbourhood=adj,
        backend='jax',
    )
    f = np.asarray(features)
    v = np.asarray(values)
    idx = np.asarray(adj)
    Lnp = np.asarray(L)
    ref = np.zeros_like(v)
    for i in range(n):
        z = (f[i] - f[idx[i]]) @ Lnp  # (k_max, 2)
        w = np.exp(-0.5 * (z**2).sum(-1))
        w = w / w.sum()
        ref[i] = (w[:, None] * v[idx[i]]).sum(0)
    np.testing.assert_allclose(got, ref, atol=1e-10)


def test_bilateral_mask_nulls_padding_no_double_count():
    """A validity mask removes the double-count of repeated/padded
    neighbour indices.  Without the mask, a duplicated index is
    weighted twice; with it, the duplicate contributes nothing.
    """
    n = 5
    values = jnp.arange(n, dtype=jnp.float64)[:, None]
    features = jnp.arange(n, dtype=jnp.float64)[:, None]
    metric = DiagonalMetric(jnp.array([1.0]))
    # Row 0 lists neighbour 1 twice (a padded duplicate) and 0 once.
    adj = jnp.array([[0, 1, 1], [1, 2, 2], [2, 3, 3], [3, 4, 4], [4, 4, 4]])
    mask = jnp.array(
        [
            [True, True, False],
            [True, True, False],
            [True, True, False],
            [True, True, False],
            [True, False, False],
        ]
    )
    masked = bilateral_gaussian(
        values,
        features,
        metric=metric,
        neighbourhood=adj,
        mask=mask,
        backend='jax',
    )
    unmasked = bilateral_gaussian(
        values,
        features,
        metric=metric,
        neighbourhood=adj,
        backend='jax',
    )
    # Row 0 masked: neighbours {0, 1}, weights exp(0), exp(-0.5).
    w0, w1 = np.exp(0.0), np.exp(-0.5)
    expect0 = (w0 * 0.0 + w1 * 1.0) / (w0 + w1)
    np.testing.assert_allclose(float(masked[0, 0]), expect0, atol=1e-12)
    # Unmasked counts neighbour 1 twice -> different (larger) value.
    assert float(unmasked[0, 0]) > float(masked[0, 0]) + 1e-6


def test_bilateral_ell_neighbourhood_derives_mask():
    """An ELL neighbourhood derives its validity mask from padding,
    so a ragged mesh k-ring (pentagons have lower degree) is handled
    correctly with no caller bookkeeping.
    """
    from nitrix.sparse.mesh import icosphere, mesh_k_ring_adjacency

    mesh = icosphere(1)  # 42 vertices, 12 pentagons
    n = mesh.n_vertices
    ell = mesh_k_ring_adjacency(mesh, k=1, include_self=True)
    features = mesh.vertices.astype(jnp.float64)
    values = jax.random.normal(jax.random.key(78), (n, 1))
    metric = DiagonalMetric(jnp.full((3,), 0.5))
    got = bilateral_gaussian(
        values,
        features,
        metric=metric,
        neighbourhood=ell,
        backend='jax',
    )
    # Reference honouring the ELL's validity mask.
    idx = np.asarray(ell.indices)
    valid = np.asarray(ell.values) != ell.identity
    f = np.asarray(features)
    v = np.asarray(values)
    ref = np.zeros_like(v)
    for i in range(n):
        d2 = (((f[i] - f[idx[i]]) / 0.5) ** 2).sum(-1)
        w = np.exp(-0.5 * d2) * valid[i]
        w = w / max(w.sum(), np.finfo(np.float64).tiny)
        ref[i] = (w[:, None] * v[idx[i]]).sum(0)
    np.testing.assert_allclose(got, ref, atol=1e-10)


def test_bilateral_n_iters_matches_manual_reapply():
    """n_iters=t equals t manual single-pass re-applications (fixed
    affinity: features and weights are held constant).
    """
    n = 20
    values = jax.random.normal(jax.random.key(79), (n, 2))
    features = jax.random.normal(jax.random.key(80), (n, 2))
    metric = DiagonalMetric(jnp.array([1.0, 1.0]))
    adj = brute_force_knn(features, 5)

    def step(x):
        return bilateral_gaussian(
            x,
            features,
            metric=metric,
            neighbourhood=adj,
            backend='jax',
        )

    iterated = bilateral_gaussian(
        values,
        features,
        metric=metric,
        neighbourhood=adj,
        n_iters=3,
        backend='jax',
    )
    manual = step(step(step(values)))
    np.testing.assert_allclose(iterated, manual, atol=1e-12)


def test_bilateral_n_iters_rejects_zero():
    values = jnp.zeros((4, 1))
    features = jnp.zeros((4, 1))
    with pytest.raises(ValueError, match='n_iters'):
        bilateral_gaussian(
            values,
            features,
            metric=DiagonalMetric(jnp.array([1.0])),
            neighbourhood=2,
            n_iters=0,
            backend='jax',
        )


def test_bilateral_grad_wrt_metric_factor():
    """The smoothed output is differentiable w.r.t. the metric factor
    L (so the metric is learnable end-to-end).
    """
    n = 12
    values = jax.random.normal(jax.random.key(81), (n, 1))
    features = jax.random.normal(jax.random.key(82), (n, 3))
    adj = brute_force_knn(features, 5)

    def loss(L):
        out = bilateral_gaussian(
            values,
            features,
            metric=FactorMetric(L),
            neighbourhood=adj,
            backend='jax',
        )
        return (out**2).sum()

    L0 = jnp.eye(3) * 0.8
    g = jax.grad(loss)(L0)
    assert g.shape == (3, 3)
    assert bool(jnp.all(jnp.isfinite(g)))
    # Finite-difference check on one entry.
    eps = 1e-4
    dL = jnp.zeros((3, 3)).at[0, 1].set(eps)
    num = (loss(L0 + dL) - loss(L0 - dL)) / (2 * eps)
    np.testing.assert_allclose(
        float(g[0, 1]), float(num), rtol=1e-3, atol=1e-5
    )


def test_metric_from_spd_and_block_diagonal():
    """Constructor helpers: Cholesky factor and block-diagonal assembly."""
    a = jax.random.normal(jax.random.key(83), (3, 3))
    M = a @ a.T + 2.0 * jnp.eye(3)
    fm = metric_from_spd(M)
    d = jnp.array([[1.0, -2.0, 0.5]])
    q = (fm.project(d) ** 2).sum(-1)
    q_ref = (d @ M * d).sum(-1)
    np.testing.assert_allclose(q, q_ref, atol=1e-9)

    bd = block_diagonal_metric(
        [jnp.diag(1.0 / jnp.array([1.0, 2.0])), jnp.array([[0.5]])]
    )
    np.testing.assert_allclose(
        bd.factor,
        jnp.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]),
        atol=1e-12,
    )

    with pytest.raises(ValueError, match='non-empty'):
        block_diagonal_metric([])
    with pytest.raises(ValueError, match='square'):
        metric_from_spd(jnp.zeros((3, 2)))


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
        img,
        sigma_space=1.0,
        sigma_intensity=10.0,
        use_median=False,
    )
    out_med = susan_emulator(
        img,
        sigma_space=1.0,
        sigma_intensity=10.0,
        use_median=True,
    )
    # The impulse-noise positions should be more attenuated under
    # use_median than without.
    assert abs(float(out_med[3, 3])) < abs(float(out_no_med[3, 3]))
    assert abs(float(out_med[5, 5])) < abs(float(out_no_med[5, 5]))


def test_permutohedral_lattice_symbol_retired():
    """Permutohedral was retired in SPEC §4.4; the bounded
    bilateral (``bilateral_gaussian`` + factored metric / mask /
    ``n_iters``) supersedes it.  The symbol no longer exists.
    See docs/design/bounded-bilateral.md.
    """
    import nitrix.smoothing as smoothing

    assert not hasattr(smoothing, 'permutohedral_lattice')
    with pytest.raises(ImportError):
        from nitrix.smoothing import permutohedral_lattice  # noqa: F401


def test_susan_emulator_canonical_home_is_smoothing():
    """SUSAN lives in ``nitrix.smoothing`` per SPEC §4.4.

    It is *not* re-exported from ``nitrix.morphology`` to avoid a
    circular import with ``median_filter``; users should import it
    from ``nitrix.smoothing``.
    """
    from nitrix import morphology, smoothing

    assert hasattr(smoothing, 'susan_emulator')
    assert not hasattr(morphology, 'susan_emulator')
