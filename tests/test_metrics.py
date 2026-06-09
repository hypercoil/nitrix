# -*- coding: utf-8 -*-
"""Tests for ``nitrix.metrics`` overlap / loss numerics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.metrics import (
    bce_with_logits,
    cross_entropy_with_logits,
    dice,
    dino_cross_entropy,
    focal_loss,
    ibot_cross_entropy,
    jaccard,
    koleo,
    nt_xent,
)

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


def test_nt_xent_matches_manual_reference():
    rng = np.random.default_rng(0)
    z = jnp.asarray(rng.standard_normal((6, 8)))
    tau = 0.3
    out = nt_xent(z, temperature=tau, reduction='none')
    # Manual reference.
    zn = np.asarray(z) / np.linalg.norm(np.asarray(z), axis=-1, keepdims=True)
    sim = zn @ zn.T / tau
    sim = sim - np.eye(6) * (2.0 / tau)
    logp = sim - np.log(np.sum(np.exp(sim), axis=-1, keepdims=True))
    pos = np.arange(6) ^ 1
    ref = -logp[np.arange(6), pos]
    np.testing.assert_allclose(np.asarray(out), ref, atol=1e-6)


def test_nt_xent_lower_for_aligned_pairs():
    # Pairs identical and well separated -> near-zero loss.
    base = np.array([[5.0, 0.0], [-5.0, 0.0]])
    aligned = jnp.asarray(np.repeat(base, 2, axis=0))  # [a,a,b,b]
    misaligned = jnp.asarray(np.random.default_rng(1).standard_normal((4, 2)))
    assert float(nt_xent(aligned, temperature=0.1)) < float(
        nt_xent(misaligned, temperature=0.1)
    )


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
    assert bool(jnp.all(jnp.isfinite(jax.grad(lambda x: nt_xent(x))(z))))
    assert bool(jnp.all(jnp.isfinite(jax.grad(lambda x: koleo(x))(z))))
    s = jnp.asarray(rng.standard_normal((4, 10)))
    t = jnp.asarray(rng.standard_normal((4, 10)))
    c = jnp.zeros(10)
    g = jax.grad(lambda sl: dino_cross_entropy(sl, t, c))(s)
    assert bool(jnp.all(jnp.isfinite(g)))
