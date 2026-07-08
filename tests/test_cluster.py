# -*- coding: utf-8 -*-
"""Tests for ``nitrix.numerics.cluster`` (k-means) and its spectral
composition ``nitrix.graph.normalized_cut``.

Coverage: cluster recovery on separated blobs; the fit/apply seam identity
(``kmeans == predict(fit)``); the ``KMeansState`` pytree with ``similarity``
as static aux; cosine (magnitude-invariant) assignment; ``n_init`` restarts;
empty-cluster stability; ``jit`` / ``vmap``; and normalised-cut recovery on a
planted two-community graph.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest  # noqa: F401

jax.config.update('jax_enable_x64', True)

from nitrix.graph import normalized_cut  # noqa: E402
from nitrix.numerics import (  # noqa: E402
    kmeans,
    kmeans_fit,
    kmeans_predict,
    nmf,
    ward_linkage,
)


def _blobs(centres, per=40, sd=0.3, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack(
        [c + sd * rng.standard_normal((per, len(c))) for c in centres]
    )
    return jnp.asarray(X), per


def _is_pure_recovery(labels, per, k):
    """Each of the ``k`` contiguous blocks maps to one distinct label."""
    blocks = [np.asarray(labels[i * per : (i + 1) * per]) for i in range(k)]
    pure = all(len(np.unique(b)) == 1 for b in blocks)
    distinct = len({int(b[0]) for b in blocks}) == k
    return pure and distinct


def test_kmeans_recovers_separated_blobs():
    X, per = _blobs([[0, 0], [10, 10], [0, 10]])
    labels = kmeans(X, 3, key=jax.random.key(0), n_init=5)
    assert _is_pure_recovery(labels, per, 3)


def test_kmeans_seam_identity():
    """kmeans(X, k) == kmeans_predict(X, kmeans_fit(X, k)) exactly."""
    X, _ = _blobs([[0, 0], [8, 8]])
    key = jax.random.key(1)
    a = kmeans(X, 2, key=key, n_init=3)
    b = kmeans_predict(X, kmeans_fit(X, 2, key=key, n_init=3))
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_kmeans_state_similarity_is_static_aux():
    """Only centroids + inertia are traced leaves; similarity is static aux."""
    X, _ = _blobs([[0, 0], [8, 8]])
    st = kmeans_fit(X, 2, key=jax.random.key(2), similarity='cosine')
    leaves = jax.tree_util.tree_leaves(st)
    assert len(leaves) == 2
    assert all(isinstance(x, jax.Array) for x in leaves)
    st2 = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(st), leaves
    )
    assert st2.similarity == 'cosine'
    assert st2.centroids.shape == st.centroids.shape


def test_kmeans_predict_is_pure_function_of_state_under_jit():
    X, per = _blobs([[0, 0], [10, 10], [0, 10]])
    st = kmeans_fit(X, 3, key=jax.random.key(0), n_init=5)
    labels = jax.jit(kmeans_predict)(X, st)
    assert _is_pure_recovery(labels, per, 3)


def test_kmeans_cosine_groups_by_direction():
    """Cosine ignores magnitude: same-direction rays cluster together."""
    dirs = np.array([[1.0, 0.0], [0.0, 1.0]])
    rng = np.random.default_rng(3)
    X = np.vstack(
        [
            (1 + 5 * rng.random((30, 1))) * d
            + 0.02 * rng.standard_normal((30, 2))
            for d in dirs
        ]
    )
    labels = kmeans(
        jnp.asarray(X), 2, key=jax.random.key(0), similarity='cosine', n_init=5
    )
    assert _is_pure_recovery(labels, 30, 2)


def test_kmeans_restarts_reach_global_optimum():
    X, per = _blobs([[0, 0], [10, 10], [0, 10], [10, 0]], seed=7)
    labels = kmeans(X, 4, key=jax.random.key(0), n_init=12)
    assert _is_pure_recovery(labels, per, 4)


def test_kmeans_empty_cluster_stays_finite():
    """k larger than the tight groups leaves empty clusters -> no NaN."""
    X, _ = _blobs([[0, 0], [10, 10]], per=5)
    st = kmeans_fit(X, 6, key=jax.random.key(0), n_init=3)
    assert bool(jnp.all(jnp.isfinite(st.centroids)))


def test_kmeans_vmap_over_batch():
    X, _ = _blobs([[0, 0], [10, 10], [0, 10]])
    batch = jnp.stack([X, X + 1.0])
    labels = jax.vmap(
        lambda Xi: kmeans(Xi, 3, key=jax.random.key(0), n_init=5)
    )(batch)
    assert labels.shape == (2, X.shape[0])


def _planted_two_cluster(npc=12, seed=1):
    rng = np.random.default_rng(seed)
    n = 2 * npc
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = 0.9 if (i // npc) == (j // npc) else 0.05
            if rng.random() < p:
                A[i, j] = A[j, i] = 1.0
    return A, npc


def test_normalized_cut_recovers_planted_clusters():
    A, npc = _planted_two_cluster()
    labels = normalized_cut(jnp.asarray(A), 2, key=jax.random.key(0), n_init=5)
    assert _is_pure_recovery(labels, npc, 2)


# --- Ward agglomerative clustering -------------------------------------------


def _ward_blobs(per=25, k=4, spread=3.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack(
        [rng.standard_normal((per, 4)) + spread * i for i in range(k)]
    )


def test_ward_recovers_separated_blobs():
    X = _ward_blobs(spread=10.0)  # well-separated -> each block pure
    labels = np.asarray(ward_linkage(jnp.asarray(X), k=4))
    assert labels.shape == (100,)
    # each planted block of 25 points lands in a single cluster
    for i in range(4):
        block = labels[i * 25 : (i + 1) * 25]
        assert len(np.unique(block)) == 1
    assert len(np.unique(labels)) == 4


def test_ward_matches_scipy():
    sp = pytest.importorskip('scipy.cluster.hierarchy')
    opt = pytest.importorskip('scipy.optimize')
    X = _ward_blobs(seed=2)
    for k in (4, 6):
        mine = np.asarray(ward_linkage(jnp.asarray(X), k=k))
        ref = sp.fcluster(sp.linkage(X, method='ward'), k, 'maxclust') - 1
        conf = np.zeros((k, k))
        for a, b in zip(mine, ref):
            conf[a, b] += 1
        r, c = opt.linear_sum_assignment(-conf)
        assert conf[r, c].sum() / len(mine) == 1.0  # exact partition match


def test_ward_k_bounds():
    X = _ward_blobs()
    with pytest.raises(ValueError):
        ward_linkage(jnp.asarray(X), k=0)
    with pytest.raises(ValueError):
        ward_linkage(jnp.asarray(X), k=101)


# --- non-negative matrix factorisation ---------------------------------------


def test_nmf_reconstructs_low_rank_nonnegative():
    rng = np.random.default_rng(0)
    w_true = np.abs(rng.standard_normal((40, 3)))
    h_true = np.abs(rng.standard_normal((3, 15)))
    v = jnp.asarray(w_true @ h_true)  # a genuine non-negative rank-3 matrix
    res = nmf(v, 3, key=jax.random.key(0), max_iter=800)
    recon = res.basis @ res.coefficients
    rel = float(jnp.linalg.norm(recon - v) / jnp.linalg.norm(v))
    assert rel < 0.02


def test_nmf_factors_are_nonnegative():
    rng = np.random.default_rng(1)
    v = jnp.asarray(np.abs(rng.standard_normal((20, 8))))
    res = nmf(v, 4, key=jax.random.key(1))
    assert res.basis.shape == (20, 4) and res.coefficients.shape == (4, 8)
    assert bool((res.basis >= 0).all()) and bool((res.coefficients >= 0).all())


def test_nmf_jit_and_grad():
    rng = np.random.default_rng(2)
    v = jnp.asarray(np.abs(rng.standard_normal((20, 8))))
    out = jax.jit(lambda v: nmf(v, 4, key=jax.random.key(0), max_iter=50).basis)(v)
    assert bool(jnp.all(jnp.isfinite(out)))
    g = jax.grad(
        lambda v: jnp.sum(nmf(v, 4, key=jax.random.key(0), max_iter=50).basis)
    )(v)
    assert bool(jnp.all(jnp.isfinite(g)))
