# -*- coding: utf-8 -*-
"""Tests for the ``nitrix.metrics.classification`` evaluation metrics:
``roc_auc`` / ``confusion_matrix`` / ``topk_accuracy``.

Parity oracles: ``sklearn`` for AUROC + the confusion matrix (incl. ties and
multiclass one-vs-rest macro/micro/none); a numpy reference for top-k.  Also
pins the degenerate cases, the ``row = true`` confusion orientation, the
reductions, and jit-cleanliness.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.metrics import (  # noqa: E402
    confusion_matrix,
    roc_auc,
    topk_accuracy,
)

sk = pytest.importorskip('sklearn.metrics')


# --- roc_auc -------------------------------------------------------------


def test_roc_auc_binary_matches_sklearn():
    rng = np.random.RandomState(0)
    s = rng.rand(500)
    y = (rng.rand(500) < 0.4).astype(int)
    got = float(roc_auc(jnp.asarray(s), jnp.asarray(y)))
    assert np.isclose(got, sk.roc_auc_score(y, s), atol=1e-12)


def test_roc_auc_binary_ties_average_ranks_match_sklearn():
    # Heavily-tied (rounded) scores exercise the average-rank tie correction.
    rng = np.random.RandomState(1)
    s = np.round(rng.rand(800), 1)
    y = (rng.rand(800) < 0.5).astype(int)
    got = float(roc_auc(jnp.asarray(s), jnp.asarray(y)))
    assert np.isclose(got, sk.roc_auc_score(y, s), atol=1e-12)


def test_roc_auc_separable_and_chance():
    rng = np.random.RandomState(2)
    s = np.r_[rng.rand(60), rng.rand(60) + 2.0]
    y = np.r_[np.zeros(60), np.ones(60)].astype(int)
    assert float(roc_auc(jnp.asarray(s), jnp.asarray(y))) == 1.0
    # reversed scores -> perfectly wrong -> 0.0
    assert float(roc_auc(jnp.asarray(-s), jnp.asarray(y))) == 0.0
    # shuffled labels -> ~0.5
    chance = float(
        roc_auc(
            jnp.asarray(rng.rand(4000)),
            jnp.asarray((rng.rand(4000) < 0.5).astype(int)),
        )
    )
    assert abs(chance - 0.5) < 0.05


@pytest.mark.parametrize('average', ['macro', 'micro', 'none'])
def test_roc_auc_multiclass_ovr_matches_sklearn(average):
    rng = np.random.RandomState(3)
    n, c = 400, 4
    scores = rng.rand(n, c)
    scores = scores / scores.sum(1, keepdims=True)
    labels = rng.randint(0, c, n)
    got = np.asarray(
        roc_auc(jnp.asarray(scores), jnp.asarray(labels), average=average)
    )
    if average == 'none':
        ref = np.array(
            [
                sk.roc_auc_score((labels == k).astype(int), scores[:, k])
                for k in range(c)
            ]
        )
    else:
        ref = sk.roc_auc_score(
            labels, scores, multi_class='ovr', average=average
        )
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_roc_auc_degenerate_is_nan():
    rng = np.random.RandomState(4)
    assert np.isnan(
        float(roc_auc(jnp.asarray(rng.rand(10)), jnp.ones(10, int)))
    )
    assert np.isnan(
        float(roc_auc(jnp.asarray(rng.rand(10)), jnp.zeros(10, int)))
    )


def test_roc_auc_rejects_bad_average_and_ndim():
    with pytest.raises(ValueError):
        roc_auc(jnp.zeros((4, 3)), jnp.zeros(4, int), average='weird')
    with pytest.raises(ValueError):
        roc_auc(jnp.zeros((2, 2, 2)), jnp.zeros(2, int))


# --- confusion_matrix ----------------------------------------------------


def test_confusion_matrix_matches_sklearn():
    rng = np.random.RandomState(5)
    pred = rng.randint(0, 5, 600)
    target = rng.randint(0, 5, 600)
    got = np.asarray(
        confusion_matrix(jnp.asarray(pred), jnp.asarray(target), num_classes=5)
    )
    ref = sk.confusion_matrix(target, pred, labels=list(range(5)))
    assert np.array_equal(got, ref)


def test_confusion_matrix_orientation_row_is_true():
    # All true-class-0 predicted as class 1 -> a single count at [0, 1].
    target = jnp.zeros(7, int)
    pred = jnp.ones(7, int)
    cm = np.asarray(confusion_matrix(pred, target, num_classes=3))
    assert cm[0, 1] == 7 and cm.sum() == 7


# --- topk_accuracy -------------------------------------------------------


@pytest.mark.parametrize('k', [1, 3, 5])
def test_topk_accuracy_matches_numpy(k):
    rng = np.random.RandomState(6)
    logits = rng.rand(300, 10)
    target = rng.randint(0, 10, 300)
    topk = np.argsort(-logits, axis=1)[:, :k]
    ref = np.mean([target[i] in topk[i] for i in range(300)])
    got = float(topk_accuracy(jnp.asarray(logits), jnp.asarray(target), k=k))
    assert np.isclose(got, ref, atol=1e-12)


def test_topk_accuracy_k1_equals_argmax_accuracy():
    rng = np.random.RandomState(7)
    logits = rng.rand(200, 8)
    target = rng.randint(0, 8, 200)
    ref = float(np.mean(np.argmax(logits, 1) == target))
    got = float(topk_accuracy(jnp.asarray(logits), jnp.asarray(target), k=1))
    assert np.isclose(got, ref, atol=1e-12)


def test_topk_accuracy_reductions():
    rng = np.random.RandomState(8)
    logits = jnp.asarray(rng.rand(50, 6))
    target = jnp.asarray(rng.randint(0, 6, 50))
    per_row = topk_accuracy(logits, target, k=2, reduction='none')
    assert per_row.shape == (50,)
    assert set(np.unique(np.asarray(per_row)).tolist()) <= {0.0, 1.0}
    total = float(topk_accuracy(logits, target, k=2, reduction='sum'))
    mean = float(topk_accuracy(logits, target, k=2, reduction='mean'))
    assert np.isclose(total, float(np.sum(np.asarray(per_row))))
    assert np.isclose(mean, total / 50)


# --- jit-cleanliness -----------------------------------------------------


def test_jit_clean():
    rng = np.random.RandomState(9)
    s = jnp.asarray(rng.rand(120))
    y = jnp.asarray((rng.rand(120) < 0.5).astype(int))
    auc_jit = jax.jit(roc_auc)(s, y)
    assert np.isclose(float(auc_jit), float(roc_auc(s, y)), atol=1e-12)

    pred = jnp.asarray(rng.randint(0, 4, 120))
    tgt = jnp.asarray(rng.randint(0, 4, 120))
    cm_jit = jax.jit(lambda p, t: confusion_matrix(p, t, num_classes=4))(
        pred, tgt
    )
    assert np.array_equal(
        np.asarray(cm_jit),
        np.asarray(confusion_matrix(pred, tgt, num_classes=4)),
    )

    logits = jnp.asarray(rng.rand(120, 7))
    acc_jit = jax.jit(lambda lo, t: topk_accuracy(lo, t, k=3))(logits, tgt)
    assert np.isclose(
        float(acc_jit), float(topk_accuracy(logits, tgt, k=3)), atol=1e-12
    )
