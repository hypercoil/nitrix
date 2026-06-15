# -*- coding: utf-8 -*-
"""Tests for ``nitrix.metrics`` overlap / loss numerics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.metrics import (
    bce_with_logits,
    cross_entropy_with_logits,
    dice,
    dino_cross_entropy,
    focal_loss,
    ibot_cross_entropy,
    info_nce,
    jaccard,
    koleo,
    lncc,
    lncc_grad,
    lncc_grad_center,
    match_histogram,
    mi_grad,
    mutual_information,
    winsorize,
)


def test_lncc_grad_matches_autodiff():
    # The analytic LNCC force is the exact gradient of lncc(reduction='sum')
    # w.r.t. moving -- machine precision vs autodiff, with no autodiff tape.
    rng = np.random.RandomState(0)
    for shape, radius in [((40, 44), 4), ((18, 16, 20), 3)]:
        moving = jnp.asarray(rng.standard_normal(shape))
        fixed = jnp.asarray(rng.standard_normal(shape) + 0.5 * np.asarray(moving))
        analytic = lncc_grad(moving, fixed, radius=radius)
        auto = jax.grad(
            lambda m: lncc(m, fixed, radius=radius, reduction='sum')
        )(moving)
        assert analytic.shape == moving.shape
        assert np.allclose(np.asarray(analytic), np.asarray(auto), atol=1e-10)


def test_lncc_grad_center_matches_itk_formula():
    # The centre-only force reproduces ITK's ANTSNeighborhoodCorrelation
    # derivative (a direct windowed reference), to machine precision interior.
    from scipy.ndimage import uniform_filter

    def ref(m, f, r, eps=1e-5):
        sz = 2 * r + 1
        n = sz ** m.ndim

        def bs(x):
            return uniform_filter(x, size=sz, mode='reflect') * n

        sm, sf = bs(m), bs(f)
        s_ff = bs(f * f) - sf * sf / n
        s_mm = bs(m * m) - sm * sm / n
        s_fm = bs(m * f) - sf * sm / n
        f_a, m_a = f - sf / n, m - sm / n
        safe = (s_ff > eps) & (s_mm > eps)
        return np.where(
            safe,
            2 * s_fm / np.where(safe, s_ff * s_mm, 1.0)
            * (f_a - s_fm / np.where(safe, s_mm, 1.0) * m_a),
            0.0,
        )

    rng = np.random.RandomState(0)
    for shape, r in [((40, 44), 2), ((22, 18, 20), 2)]:
        m = rng.standard_normal(shape)
        f = rng.standard_normal(shape) + 0.5 * m
        got = np.asarray(lncc_grad_center(jnp.asarray(m), jnp.asarray(f), radius=r))
        want = ref(m, f, r)
        sl = tuple(slice(r + 1, s - r - 1) for s in shape)  # interior
        assert np.allclose(got[sl], want[sl], atol=1e-10)


def test_lncc_grad_center_zeroes_flat_window():
    # A constant (flat) image has sFF=sMM=0 -> the ITK guard zeroes the force,
    # and the gradient stays finite (the double-where).
    flat = jnp.ones((24, 24, 24))
    moving = jnp.asarray(np.random.RandomState(1).standard_normal((24, 24, 24)))
    g = lncc_grad_center(moving, flat, radius=2)
    assert np.all(np.isfinite(np.asarray(g)))
    # where fixed is flat (sFF=0) the force is exactly zero
    assert float(jnp.abs(g).max()) == 0.0


# ---------------------------------------------------------------------------
# preprocessing (fMRIPrep front-end, 4a/4b)
# ---------------------------------------------------------------------------


def test_winsorize_clips_outliers_to_percentiles():
    rng = np.random.RandomState(0)
    x = rng.standard_normal((64, 64))
    x[0, 0], x[1, 1] = 100.0, -100.0  # hot / cold outliers
    xw = np.asarray(winsorize(jnp.asarray(x), lower=0.005, upper=0.995))
    lo, hi = np.quantile(x, [0.005, 0.995])
    assert np.isclose(xw.max(), hi, atol=1e-6)  # hot voxel clipped to the pctile
    assert np.isclose(xw.min(), lo, atol=1e-6)
    # interior (non-tail) voxels are untouched
    interior = (x > lo) & (x < hi)
    assert np.allclose(xw[interior], x[interior])


def test_winsorize_validation():
    with pytest.raises(ValueError):
        winsorize(jnp.ones((8,)), lower=0.9, upper=0.1)


def test_match_histogram_matches_reference_distribution():
    rng = np.random.RandomState(1)
    reference = jnp.asarray(rng.standard_normal((80, 80)) * 2.0 + 5.0)
    moving = reference * 0.4 - 3.0  # a global gain/offset of the same structure
    matched = match_histogram(moving, reference)
    qs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    assert np.allclose(
        np.quantile(np.asarray(matched), qs),
        np.quantile(np.asarray(reference), qs),
        atol=1e-2,
    )
    # the remap is monotone, so structure (correlation with reference) is kept
    corr = np.corrcoef(np.asarray(matched).ravel(), np.asarray(reference).ravel())
    assert corr[0, 1] > 0.999
    g = jax.grad(lambda z: (match_histogram(z, reference) ** 2).sum())(moving)
    assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# _box_sum integral image (1c)
# ---------------------------------------------------------------------------


def _ref_box_sum_2d(x, r, np_mode):
    xp = np.pad(np.asarray(x), r, mode=np_mode)
    n0, n1 = x.shape
    out = np.zeros((n0, n1))
    for i in range(n0):
        for j in range(n1):
            out[i, j] = xp[i : i + 2 * r + 1, j : j + 2 * r + 1].sum()
    return out


def test_box_sum_integral_image_matches_windowed_reference():
    # The integral-image box sum equals the direct padded windowed sum across
    # boundary modes (only the O(N) computation changed, not the operator).
    from nitrix.metrics._common import _box_sum

    rng = np.random.RandomState(1)
    x = jnp.asarray(rng.standard_normal((24, 26)))
    r = 3
    for mode, np_mode in [
        ('reflect', 'symmetric'),
        ('mirror', 'reflect'),
        ('nearest', 'edge'),
        ('constant', 'constant'),
    ]:
        got = np.asarray(_box_sum(x, (2 * r + 1, 2 * r + 1), (0, 1), mode))
        ref = _ref_box_sum_2d(x, r, np_mode)
        assert np.allclose(got, ref, atol=1e-10)


def test_box_sum_fp32_within_tolerance_of_fp64():
    # 1c gate: the integral image trades a cancellation in fp32 (the prefix-sum
    # magnitude ~ axis_length·max) for O(N), radius-free cost.  At a realistic
    # intensity range it stays well within fp32 tolerance of fp64 -- the gate
    # certifying the cancellation is acceptable (NOT an assertion of fp32 safety
    # at any scale: a huge grid at a wide range should use fp64 / winsorise).
    from nitrix.metrics._common import _box_sum

    rng = np.random.RandomState(0)
    x = rng.uniform(0.0, 500.0, (160, 160))  # scaled-MRI-like range, not [0,1]
    bs64 = np.asarray(_box_sum(jnp.asarray(x, jnp.float64), (9, 9), (0, 1), 'reflect'))
    bs32 = np.asarray(_box_sum(jnp.asarray(x, jnp.float32), (9, 9), (0, 1), 'reflect'))
    assert (np.abs(bs32 - bs64) / (np.abs(bs64) + 1e-6)).max() < 1e-4
    a = rng.uniform(0.0, 500.0, (160, 160))
    b = 0.6 * a + rng.uniform(0.0, 200.0, (160, 160))
    l32 = float(lncc(jnp.asarray(a, jnp.float32), jnp.asarray(b, jnp.float32), radius=4))
    l64 = float(lncc(jnp.asarray(a, jnp.float64), jnp.asarray(b, jnp.float64), radius=4))
    assert abs(l32 - l64) < 1e-4


# ---------------------------------------------------------------------------
# mi_grad (closed-form Mattes MI gradient)
# ---------------------------------------------------------------------------


def _mi_autograd(moving, fixed, *, bins, rm, rf):
    return jax.grad(
        lambda m: mutual_information(
            m, fixed, bins=bins, range_moving=rm, range_fixed=rf
        )
    )(moving)


def test_mi_grad_matches_autodiff_on_populated_bins():
    # The parity oracle: on a fully-populated joint histogram (no empty bins)
    # the closed-form Mattes gradient is the exact derivative of the cost --
    # machine precision vs autodiff, with no histogram-scatter tape.
    rng = np.random.RandomState(0)
    for shape, bins in [((64, 64), 16), ((40, 40), 12), ((16, 18, 20), 8)]:
        moving = jnp.asarray(rng.uniform(0.0, 1.0, shape))  # noise fills bins
        fixed = jnp.asarray(rng.uniform(0.0, 1.0, shape))
        rm = rf = (0.0, 1.0)
        analytic = np.asarray(mi_grad(moving, fixed, bins=bins, range_moving=rm, range_fixed=rf))
        auto = np.asarray(_mi_autograd(moving, fixed, bins=bins, rm=rm, rf=rf))
        assert analytic.shape == tuple(shape)
        rel = np.linalg.norm(analytic - auto) / (np.linalg.norm(auto) + 1e-30)
        assert rel < 1e-6


def test_mi_grad_direction_aligns_cross_modal():
    # On a SPARSE cross-modal histogram (many empty bins) the magnitude diverges
    # from autodiff at the empty-bin boundaries (the where(hist>0) mask is
    # non-smooth there -- the documented analogue of lncc_grad's boundary
    # divergence), but the *direction* stays tightly aligned, which is what
    # drives the registration force.
    rng = np.random.RandomState(1)
    yy, xx = np.mgrid[0:56, 0:56].astype('float64')
    fixed = np.zeros((56, 56))
    for _ in range(6):
        cy, cx = rng.uniform(0.2, 0.8, 2) * 56
        fixed += rng.uniform(0.4, 1.0) * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (0.12 * 56) ** 2)
        )
    fixed = jnp.asarray(fixed)
    moving = jnp.sqrt(fixed - fixed.min() + 0.05)  # "different modality"
    rm = (float(moving.min()), float(moving.max()))
    rf = (float(fixed.min()), float(fixed.max()))
    a = np.asarray(mi_grad(moving, fixed, bins=32, range_moving=rm, range_fixed=rf)).ravel()
    b = np.asarray(_mi_autograd(moving, fixed, bins=32, rm=rm, rf=rf)).ravel()
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    assert cos > 0.99


def test_mi_grad_zero_outside_pinned_range():
    # A voxel outside the pinned moving range has a clipped soft bin, so its
    # derivative is exactly zero (the force never pushes an over-range voxel).
    rng = np.random.RandomState(2)
    fixed = jnp.asarray(rng.uniform(0.0, 1.0, (40, 40)))
    moving = jnp.asarray(rng.uniform(0.0, 1.0, (40, 40))).at[0, 0].set(5.0)
    g = mi_grad(moving, fixed, bins=16, range_moving=(0.0, 1.0), range_fixed=(0.0, 1.0))
    assert float(g[0, 0]) == 0.0
    assert bool(jnp.all(jnp.isfinite(g)))


def test_mi_grad_finite_on_degenerate_uniform():
    # A fully-uniform (zero-range) moving image with an unpinned range: the span
    # floor keeps s_m bounded so the force is finite, not NaN/inf (robustness).
    rng = np.random.RandomState(3)
    fixed = jnp.asarray(rng.uniform(0.0, 1.0, (40, 40)))
    moving = jnp.ones((40, 40))
    g = mi_grad(moving, fixed, bins=16)  # range_moving=None -> data range = 0
    assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# dice / jaccard
# ---------------------------------------------------------------------------


def test_dice_perfect_overlap_is_one():
    x = jnp.asarray([1.0, 0.0, 1.0, 1.0, 0.0])
    assert abs(float(dice(x, x)) - 1.0) < 1e-6


def test_dice_disjoint_is_zero():
    p = jnp.asarray([1.0, 1.0, 0.0, 0.0])
    t = jnp.asarray([0.0, 0.0, 1.0, 1.0])
    assert float(dice(p, t)) < 1e-5


def test_dice_empty_vs_empty_is_one():
    z = jnp.zeros(8)
    # 0/0 -> smooth/smooth == 1 (vacuously perfect).
    assert abs(float(dice(z, z)) - 1.0) < 1e-6


def test_dice_half_overlap_known_value():
    p = jnp.asarray([1.0, 1.0, 0.0, 0.0])
    t = jnp.asarray([1.0, 0.0, 1.0, 0.0])
    # 2*1 / (2 + 2) = 0.5
    np.testing.assert_allclose(float(dice(p, t, smooth=0.0)), 0.5, atol=1e-7)


def test_jaccard_perfect_and_disjoint():
    x = jnp.asarray([1.0, 0.0, 1.0, 1.0])
    assert abs(float(jaccard(x, x)) - 1.0) < 1e-6
    p = jnp.asarray([1.0, 1.0, 0.0, 0.0])
    t = jnp.asarray([0.0, 0.0, 1.0, 1.0])
    assert float(jaccard(p, t)) < 1e-5


def test_jaccard_half_overlap_known_value():
    p = jnp.asarray([1.0, 1.0, 0.0, 0.0])
    t = jnp.asarray([1.0, 0.0, 1.0, 0.0])
    # intersection 1, union 3 -> 1/3
    np.testing.assert_allclose(
        float(jaccard(p, t, smooth=0.0)), 1.0 / 3.0, atol=1e-7
    )


def test_jaccard_dice_relationship():
    rng = np.random.default_rng(0)
    p = jnp.asarray(rng.random((4, 3, 16)))
    t = jnp.asarray((rng.random((4, 3, 16)) > 0.5).astype(float))
    d = dice(p, t, axis=-1, smooth=0.0, reduction='none')
    j = jaccard(p, t, axis=-1, smooth=0.0, reduction='none')
    # jaccard == dice / (2 - dice), elementwise.
    np.testing.assert_allclose(
        np.asarray(j), np.asarray(d / (2.0 - d)), atol=1e-9
    )


def test_dice_per_region_shape_and_reduction():
    rng = np.random.default_rng(1)
    p = jnp.asarray(rng.random((2, 4, 8, 8)))
    t = jnp.asarray((rng.random((2, 4, 8, 8)) > 0.5).astype(float))
    per = dice(p, t, axis=(-2, -1), reduction='none')
    assert per.shape == (2, 4)
    m = dice(p, t, axis=(-2, -1), reduction='mean')
    np.testing.assert_allclose(float(m), float(per.mean()), atol=1e-9)
    s = dice(p, t, axis=(-2, -1), reduction='sum')
    np.testing.assert_allclose(float(s), float(per.sum()), atol=1e-9)


def test_dice_multiclass_one_hot():
    # Argmax-consistent prediction onehot vs target onehot, per class.
    target = jnp.asarray([0, 1, 2, 1, 0])
    n_cls = 3
    t_oh = jax.nn.one_hot(target, n_cls).T  # (class, n)
    p_oh = t_oh  # perfect
    per = dice(p_oh, t_oh, axis=-1, reduction='none')
    np.testing.assert_allclose(np.asarray(per), 1.0, atol=1e-6)


def test_dice_jaccard_differentiable():
    rng = np.random.default_rng(2)
    logits = jnp.asarray(rng.standard_normal((3, 10)))
    t = jnp.asarray((rng.random((3, 10)) > 0.5).astype(float))

    def loss(z):
        p = jax.nn.sigmoid(z)
        return 1.0 - dice(p, t, axis=-1) + 1.0 - jaccard(p, t, axis=-1)

    g = jax.grad(loss)(logits)
    assert g.shape == logits.shape
    assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# bce_with_logits / cross_entropy_with_logits / focal_loss
# ---------------------------------------------------------------------------


def test_bce_with_logits_matches_naive_reference():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((6, 5)))
    t = jnp.asarray((rng.random((6, 5)) > 0.5).astype(float))
    sig = jax.nn.sigmoid(x)
    ref = -(t * jnp.log(sig) + (1 - t) * jnp.log(1 - sig))
    out = bce_with_logits(x, t, reduction='none')
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), atol=1e-9)


def test_bce_with_logits_stable_at_large_magnitude():
    # Naive log(sigmoid) overflows here; the stable form stays finite.
    x = jnp.asarray([-1e3, 1e3, 1e3, -1e3])
    t = jnp.asarray([0.0, 1.0, 0.0, 1.0])
    out = bce_with_logits(x, t, reduction='none')
    assert bool(jnp.all(jnp.isfinite(out)))
    # Correct predictions -> ~0 loss; wrong -> ~|x|.
    np.testing.assert_allclose(np.asarray(out)[:2], 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(out)[2:], 1e3, rtol=1e-6)


def test_cross_entropy_uniform_logits_is_log_C():
    logits = jnp.zeros((4, 7, 3))  # (batch, class, spatial)
    target = jnp.zeros((4, 3), dtype=jnp.int32)
    ce = cross_entropy_with_logits(logits, target, axis=1)
    np.testing.assert_allclose(float(ce), np.log(7.0), atol=1e-6)


def test_cross_entropy_matches_gather_reference():
    rng = np.random.default_rng(1)
    logits = jnp.asarray(rng.standard_normal((5, 4)))  # (batch, class)
    target = jnp.asarray(rng.integers(0, 4, size=(5,)))
    out = cross_entropy_with_logits(logits, target, axis=1, reduction='none')
    logp = np.asarray(jax.nn.log_softmax(logits, axis=1))
    ref = -logp[np.arange(5), np.asarray(target)]
    np.testing.assert_allclose(np.asarray(out), ref, atol=1e-9)


def test_focal_gamma0_no_alpha_equals_bce():
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.standard_normal((4, 6)))
    t = jnp.asarray((rng.random((4, 6)) > 0.5).astype(float))
    f = focal_loss(x, t, gamma=0.0, alpha=-1.0, reduction='none')
    b = bce_with_logits(x, t, reduction='none')
    np.testing.assert_allclose(np.asarray(f), np.asarray(b), atol=1e-9)


def test_focal_downweights_easy_examples():
    # A confidently-correct example contributes far less under focal.
    x = jnp.asarray([6.0])
    t = jnp.asarray([1.0])
    f = float(focal_loss(x, t, gamma=2.0, alpha=-1.0, reduction='sum'))
    b = float(bce_with_logits(x, t, reduction='sum'))
    assert f < 0.05 * b


def test_focal_loss_differentiable():
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.standard_normal((3, 8)))
    t = jnp.asarray((rng.random((3, 8)) > 0.5).astype(float))
    g = jax.grad(lambda z: focal_loss(z, t))(x)
    assert g.shape == x.shape
    assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# contrastive / self-supervised
# ---------------------------------------------------------------------------


def test_info_nce_matches_manual_reference():
    rng = np.random.default_rng(0)
    za = jnp.asarray(rng.standard_normal((6, 8)))
    zb = jnp.asarray(rng.standard_normal((6, 8)))
    tau = 0.3
    out = info_nce(za, zb, temperature=tau, reduction='none')
    # Manual reference: symmetric cross-view InfoNCE, diagonal positives.
    a = np.asarray(za) / np.linalg.norm(np.asarray(za), axis=-1, keepdims=True)
    b = np.asarray(zb) / np.linalg.norm(np.asarray(zb), axis=-1, keepdims=True)
    logits = a @ b.T / tau
    lab = logits - np.log(np.sum(np.exp(logits), axis=-1, keepdims=True))
    lba = logits.T - np.log(np.sum(np.exp(logits.T), axis=-1, keepdims=True))
    d = np.arange(6)
    ref = -0.5 * (lab[d, d] + lba[d, d])
    np.testing.assert_allclose(np.asarray(out), ref, atol=1e-6)


def test_info_nce_lower_for_aligned_pairs():
    # Matched views well separated -> near-zero loss; random pairing -> high.
    base = jnp.asarray([[5.0, 0.0], [-5.0, 0.0], [0.0, 5.0], [0.0, -5.0]])
    rng = np.random.default_rng(1)
    noise = jnp.asarray(rng.standard_normal((4, 2)) * 0.01)
    assert float(info_nce(base, base + noise, temperature=0.1)) < float(
        info_nce(base, jnp.asarray(rng.standard_normal((4, 2))), temperature=0.1)
    )


def test_info_nce_self_pairs_have_no_self_similarity_bias():
    # Cross-view formulation has no self-pairs -> identical-view inputs give a
    # well-defined, finite loss with the diagonal as the (perfect) positive.
    z = jnp.asarray(np.random.default_rng(7).standard_normal((5, 4)))
    out = info_nce(z, z, temperature=0.2)
    assert bool(jnp.isfinite(out))


def test_dino_cross_entropy_matches_manual_and_stops_teacher_grad():
    rng = np.random.default_rng(2)
    s = jnp.asarray(rng.standard_normal((4, 10)))
    t = jnp.asarray(rng.standard_normal((4, 10)))
    c = jnp.asarray(rng.standard_normal(10))
    st, tt = 0.1, 0.04
    out = dino_cross_entropy(s, t, c, student_temp=st, teacher_temp=tt)
    tp = jax.nn.softmax((t - c) / tt, axis=-1)
    slp = jax.nn.log_softmax(s / st, axis=-1)
    ref = float((-jnp.sum(tp * slp, axis=-1)).mean())
    np.testing.assert_allclose(float(out), ref, atol=1e-6)
    # Teacher is detached: no gradient flows into teacher_logits.
    g = jax.grad(lambda tl: dino_cross_entropy(s, tl, c))(t)
    np.testing.assert_array_equal(np.asarray(g), 0.0)


def test_ibot_masked_mean_and_empty_mask():
    rng = np.random.default_rng(3)
    s = jnp.asarray(rng.standard_normal((2, 5, 7)))
    t = jnp.asarray(rng.standard_normal((2, 5, 7)))
    c = jnp.zeros(7)
    mask = jnp.asarray([[True, False, True, False, False],
                        [False, False, False, False, False]])
    out = ibot_cross_entropy(s, t, c, mask, reduction='none')
    # Sample 1: mean CE over its 2 masked tokens; sample 2: all-unmasked -> 0.
    tp = jax.nn.softmax((t - c) / 0.04, axis=-1)  # default teacher_temp
    slp = jax.nn.log_softmax(s / 0.1, axis=-1)  # default student_temp
    ce = -jnp.sum(tp * slp, axis=-1)
    ref0 = float((ce[0, 0] + ce[0, 2]) / 2)
    np.testing.assert_allclose(float(out[0]), ref0, atol=1e-5)
    assert float(out[1]) == 0.0


def test_koleo_larger_for_collapsed_features():
    rng = np.random.default_rng(4)
    spread = jnp.asarray(rng.standard_normal((16, 8)) * 5.0)
    collapsed = jnp.asarray(
        np.ones((16, 8)) + rng.standard_normal((16, 8)) * 1e-3
    )
    assert float(koleo(collapsed)) > float(koleo(spread))


def test_contrastive_losses_differentiable():
    rng = np.random.default_rng(5)
    z = jnp.asarray(rng.standard_normal((6, 8)))
    assert bool(jnp.all(jnp.isfinite(jax.grad(lambda x: info_nce(x, x + 0.1))(z))))
    assert bool(jnp.all(jnp.isfinite(jax.grad(lambda x: koleo(x))(z))))
    s = jnp.asarray(rng.standard_normal((4, 10)))
    t = jnp.asarray(rng.standard_normal((4, 10)))
    c = jnp.zeros(10)
    g = jax.grad(lambda sl: dino_cross_entropy(sl, t, c))(s)
    assert bool(jnp.all(jnp.isfinite(g)))
