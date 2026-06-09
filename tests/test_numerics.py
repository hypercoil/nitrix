# -*- coding: utf-8 -*-
"""Tests for ``nitrix.numerics`` -- tensor_ops + normalize."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.numerics import (
    broadcast_ignoring,
    complex_decompose,
    complex_recompose,
    crop_to_multiple,
    demean,
    fold_axis,
    gaussian_window,
    instance_norm,
    intensity_normalize,
    l2_normalize,
    lp_normalize,
    nonzero_bounding_box,
    orient_and_conform,
    overlap_add,
    pad_to_multiple,
    percentile_rescale,
    promote_to_rank,
    psc_normalize,
    robust_zscore_normalize,
    unfold_axes,
    zscore_normalize,
)

# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


def test_zscore_normalize_has_zero_mean_unit_std():
    x = jnp.asarray(np.random.default_rng(0).standard_normal(1000))
    z = zscore_normalize(x)
    assert abs(float(z.mean())) < 1e-10
    assert abs(float(z.std()) - 1.0) < 1e-10


def test_zscore_normalize_per_axis():
    x = jnp.asarray(np.random.default_rng(0).standard_normal((5, 100)))
    z = zscore_normalize(x, axis=-1)
    np.testing.assert_allclose(z.mean(axis=-1), 0.0, atol=1e-10)
    np.testing.assert_allclose(z.std(axis=-1), 1.0, atol=1e-10)


def test_zscore_normalize_weighted_matches_unweighted_uniform():
    x = jnp.asarray(np.random.default_rng(0).standard_normal(100))
    w = jnp.ones(100)
    np.testing.assert_allclose(
        zscore_normalize(x, weights=w),
        zscore_normalize(x),
        atol=1e-13,
    )


def test_psc_normalize_zero_mean():
    x = jnp.asarray(np.abs(np.random.default_rng(0).standard_normal(100)) + 10)
    p = psc_normalize(x)
    assert abs(float(p.mean())) < 1e-9


def test_robust_zscore_resistant_to_outliers():
    x = jnp.asarray(np.random.default_rng(0).standard_normal(100))
    x_outlier = x.at[0].set(1000.0)
    # Robust z-score: interior std should remain ~1
    z_robust = robust_zscore_normalize(x_outlier)
    interior_std = float(z_robust[1:].std())
    assert 0.7 < interior_std < 1.4, (
        f'interior std after robust z-score = {interior_std}, expected ~1'
    )
    # Standard z-score: interior std is dominated by the outlier.
    z_normal = zscore_normalize(x_outlier)
    # The outlier dominates so interior std should be tiny.
    interior_std_normal = float(z_normal[1:].std())
    assert interior_std_normal < 0.1


def test_intensity_normalize_compresses_to_unit_interval():
    x = jnp.asarray(np.random.default_rng(0).standard_normal(100) * 10 + 50)
    x = x.at[0].set(1000)  # outlier
    n = intensity_normalize(x, low_percentile=1, high_percentile=99)
    assert float(n.min()) >= 0.0
    assert float(n.max()) <= 1.0


def test_percentile_rescale_min_p99_clip_matches_reference():
    """Defaults are the Synth* min--p99--clip recipe."""
    x_np = np.random.default_rng(0).standard_normal(1000) * 10 + 50
    out = percentile_rescale(jnp.asarray(x_np), lo=0.0, hi=99.0, clip=True)
    p_lo = np.percentile(x_np, 0.0)  # = min
    p_hi = np.percentile(x_np, 99.0)
    ref = np.clip((x_np - p_lo) / p_hi, 0.0, 1.0)
    np.testing.assert_allclose(np.asarray(out), ref, atol=1e-7)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_percentile_rescale_clip_flag_saturates_top():
    x = jnp.asarray([0.0, 1.0, 2.0, 100.0])
    out_clip = percentile_rescale(x, clip=True)
    out_noclip = percentile_rescale(x, clip=False)
    # With clip the top saturates at 1; without, it exceeds 1.
    assert float(out_clip.max()) <= 1.0 + 1e-6
    assert float(out_noclip.max()) > 1.0
    # Below the ceiling the two paths agree.
    np.testing.assert_allclose(
        np.asarray(out_clip)[:3],
        np.asarray(out_noclip)[:3],
        atol=1e-7,
    )


def test_percentile_rescale_differs_from_intensity_normalize():
    """The two recipes diverge when the minimum is far from zero:
    percentile_rescale divides by p_hi, intensity_normalize by the
    inter-percentile width.
    """
    x = jnp.asarray(np.random.default_rng(0).standard_normal(500) * 5 + 100)
    a = percentile_rescale(x, lo=1.0, hi=99.0, clip=True)
    b = intensity_normalize(x, low_percentile=1.0, high_percentile=99.0)
    assert float(jnp.abs(a - b).max()) > 1e-2


def test_zscore_nonzero_mask_leaves_background_zero():
    rng = np.random.default_rng(0)
    fg = rng.standard_normal(800) * 3 + 10
    x_np = np.concatenate([fg, np.zeros(200)])
    rng.shuffle(x_np)
    z = np.asarray(zscore_normalize(jnp.asarray(x_np), nonzero_mask=True))
    bg = x_np == 0
    # Background stays exactly zero (not -mean/std).
    np.testing.assert_array_equal(z[bg], 0.0)
    # Foreground standardised against foreground-only statistics.
    mean = x_np[~bg].mean()
    std = x_np[~bg].std()  # population (ddof=0)
    np.testing.assert_allclose(z[~bg], (x_np[~bg] - mean) / std, atol=1e-6)
    assert abs(float(z[~bg].mean())) < 1e-6
    np.testing.assert_allclose(z[~bg].std(), 1.0, atol=1e-6)


def test_zscore_nonzero_mask_per_channel():
    rng = np.random.default_rng(1)
    x_np = np.stack(
        [
            np.where(
                rng.random(1000) > 0.3, rng.standard_normal(1000) * 2 + 5, 0.0
            ),
            np.where(
                rng.random(1000) > 0.5, rng.standard_normal(1000) * 4 - 3, 0.0
            ),
        ]
    )
    z = np.asarray(
        zscore_normalize(jnp.asarray(x_np), axis=-1, nonzero_mask=True)
    )
    for c in range(2):
        fg = x_np[c] != 0
        np.testing.assert_array_equal(z[c][~fg], 0.0)
        assert abs(float(z[c][fg].mean())) < 1e-6
        np.testing.assert_allclose(z[c][fg].std(), 1.0, atol=1e-6)


def test_zscore_nonzero_mask_rejects_explicit_weights():
    x = jnp.asarray(np.random.default_rng(0).standard_normal(100))
    with pytest.raises(ValueError, match='not both'):
        zscore_normalize(x, weights=jnp.ones(100), nonzero_mask=True)


def test_demean_preserves_unit_variance():
    x = jnp.asarray(np.random.default_rng(0).standard_normal(100))
    d = demean(x)
    assert abs(float(d.mean())) < 1e-10
    np.testing.assert_allclose(d.std(), x.std(), atol=1e-13)


# ---------------------------------------------------------------------------
# tensor_ops
# ---------------------------------------------------------------------------


def test_orient_and_conform_broadcasts():
    h = jnp.arange(5)
    ref = jnp.zeros((3, 5, 4))
    o = orient_and_conform(h, axis=1, reference=ref)
    assert o.shape == (1, 5, 1)
    # The values along axis 1 match h
    np.testing.assert_array_equal(o[0, :, 0], h)


def test_fold_unfold_roundtrip():
    a = jnp.arange(24).reshape(4, 6)
    folded = fold_axis(a, axis=1, n_folds=3)
    assert folded.shape == (4, 2, 3)
    unfolded = unfold_axes(folded, axes=(1, 2))
    assert unfolded.shape == (4, 6)
    np.testing.assert_array_equal(unfolded, a)


def test_complex_decompose_recompose():
    x = jnp.asarray(
        np.random.default_rng(0).standard_normal(20)
        + 1j * np.random.default_rng(1).standard_normal(20)
    )
    ampl, phase = complex_decompose(x)
    assert ampl.shape == phase.shape == x.shape
    np.testing.assert_allclose(ampl, jnp.abs(x), atol=1e-13)
    x_rec = complex_recompose(ampl, phase)
    np.testing.assert_allclose(x_rec, x, atol=1e-13)


def test_promote_to_rank():
    x = jnp.arange(5)
    y = promote_to_rank(x, 3)
    assert y.shape == (1, 1, 5)
    # Already-larger rank: no-op.
    z = jnp.zeros((2, 3, 4))
    assert promote_to_rank(z, 2).shape == z.shape


def test_broadcast_ignoring_axes():
    a = jnp.zeros((4, 3))
    b = jnp.zeros((4, 5))
    # Broadcast all axes except -1 (which has differing sizes 3 and 5).
    aa, bb = broadcast_ignoring(a, b, axis=-1)
    assert aa.shape[:-1] == bb.shape[:-1]


# ---------------------------------------------------------------------------
# l2_normalize / lp_normalize / instance_norm
# ---------------------------------------------------------------------------


def test_l2_normalize_unit_norm():
    x = jnp.asarray(np.random.default_rng(0).standard_normal((7, 5)))
    y = l2_normalize(x, axis=-1)
    np.testing.assert_allclose(
        jnp.linalg.norm(y, axis=-1), 1.0, atol=1e-12
    )


def test_l2_normalize_matches_lp_p2():
    x = jnp.asarray(np.random.default_rng(1).standard_normal((4, 6)))
    np.testing.assert_allclose(
        l2_normalize(x, axis=-1), lp_normalize(x, p=2.0, axis=-1), atol=1e-12
    )


def test_l2_normalize_zero_vector_is_finite():
    # Clamp-denominator (not add-eps): a zero row stays zero, no NaN/Inf.
    x = jnp.zeros((3, 4))
    y = l2_normalize(x, axis=-1)
    assert bool(jnp.all(jnp.isfinite(y)))
    np.testing.assert_allclose(y, 0.0, atol=0.0)


def test_lp_normalize_l1():
    x = jnp.asarray(
        np.abs(np.random.default_rng(2).standard_normal((5, 8)))
    )
    y = lp_normalize(x, p=1.0, axis=-1)
    np.testing.assert_allclose(jnp.sum(jnp.abs(y), axis=-1), 1.0, atol=1e-12)


def test_instance_norm_zero_mean_unit_var():
    x = jnp.asarray(
        np.random.default_rng(3).standard_normal((2, 3, 9, 9, 9))
    )
    y = instance_norm(x, axes=(-3, -2, -1))
    # Per-(sample, channel) zero mean and (population) unit variance.
    np.testing.assert_allclose(y.mean(axis=(-3, -2, -1)), 0.0, atol=1e-6)
    np.testing.assert_allclose(y.var(axis=(-3, -2, -1)), 1.0, atol=1e-4)


def test_instance_norm_matches_zscore_single_axis():
    # With one reduction axis instance_norm == zscore (population std).
    x = jnp.asarray(np.random.default_rng(4).standard_normal((5, 50)))
    np.testing.assert_allclose(
        instance_norm(x, axes=-1, eps=1e-12),
        zscore_normalize(x, axis=-1, eps=0.0),
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# spatial: pad/crop to multiple, bbox, window, overlap-add
# ---------------------------------------------------------------------------


def test_pad_to_multiple_shape_and_unpad():
    x = jnp.asarray(np.random.default_rng(0).standard_normal((13, 18, 2)))
    padded, widths = pad_to_multiple(x, 8, spatial_rank=2)
    assert padded.shape == (16, 24, 2)
    assert widths == ((1, 2), (3, 3))
    # Unpad recovers the original.
    (l0, h0), (l1, h1) = widths
    back = padded[l0 : padded.shape[0] - h0, l1 : padded.shape[1] - h1, :]
    np.testing.assert_array_equal(np.asarray(back), np.asarray(x))


def test_pad_to_multiple_per_axis_and_cval():
    x = jnp.ones((5, 6))
    padded, widths = pad_to_multiple(
        x, (4, 4), spatial_rank=2, mode='constant', cval=-1.0
    )
    assert padded.shape == (8, 8)
    assert float(padded[0, 0]) == -1.0  # padded region uses cval


def test_crop_to_multiple_shape():
    x = jnp.asarray(np.random.default_rng(1).standard_normal((17, 20, 3)))
    cropped, widths = crop_to_multiple(x, 8, spatial_rank=2)
    assert cropped.shape == (16, 16, 3)
    assert widths == ((0, 1), (2, 2))


def test_nonzero_bounding_box_known():
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:5, 3:9] = True
    lo, hi = nonzero_bounding_box(jnp.asarray(mask))
    np.testing.assert_array_equal(np.asarray(lo), [2, 3])
    np.testing.assert_array_equal(np.asarray(hi), [5, 9])


def test_nonzero_bounding_box_threshold_and_3d():
    rng = np.random.default_rng(2)
    x = rng.random((8, 9, 10))
    x[x < 0.99] = 0.0  # sparse foreground
    lo, hi = nonzero_bounding_box(jnp.asarray(x), threshold=0.0)
    idx = np.argwhere(np.asarray(x) > 0)
    np.testing.assert_array_equal(np.asarray(lo), idx.min(0))
    np.testing.assert_array_equal(np.asarray(hi), idx.max(0) + 1)


def test_nonzero_bounding_box_empty():
    lo, hi = nonzero_bounding_box(jnp.zeros((4, 4)))
    np.testing.assert_array_equal(np.asarray(lo), [0, 0])
    np.testing.assert_array_equal(np.asarray(hi), [0, 0])


def test_gaussian_window_peak_centre_and_symmetry():
    w = np.asarray(gaussian_window((9, 9)))
    assert w.shape == (9, 9)
    # Peak 1 at the centre; symmetric.
    assert abs(w[4, 4] - 1.0) < 1e-6
    np.testing.assert_allclose(w, w[::-1, :], atol=1e-6)
    np.testing.assert_allclose(w, w[:, ::-1], atol=1e-6)
    assert w.min() > 0.0


def test_overlap_add_normalises():
    ws = jnp.asarray([2.0, 6.0, 0.0])
    w = jnp.asarray([2.0, 3.0, 0.0])
    out = overlap_add(ws, w)
    # 2/2=1, 6/3=2, 0/eps=0 (uncovered voxel safe).
    np.testing.assert_allclose(np.asarray(out)[:2], [1.0, 2.0], atol=1e-6)
    assert np.isfinite(np.asarray(out)[2])


def test_percentile_rescale_mask_ignores_background():
    rng = np.random.default_rng(3)
    fg = rng.uniform(10, 20, size=400)
    x_np = np.concatenate([fg, np.zeros(600)])
    x = jnp.asarray(x_np)
    masked = percentile_rescale(x, lo=0.0, hi=100.0, mask=(x != 0))
    # Reference percentiles over foreground only.
    p_lo = np.percentile(fg, 0.0)
    p_hi = np.percentile(fg, 100.0)
    ref = np.clip((x_np - p_lo) / (p_hi + 1e-12), 0.0, 1.0)
    np.testing.assert_allclose(np.asarray(masked), ref, atol=1e-6)
    # Differs from the unmasked version (zeros drag the low percentile).
    unmasked = percentile_rescale(x, lo=0.0, hi=100.0)
    assert float(jnp.abs(masked - unmasked).max()) > 1e-3
