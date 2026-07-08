# -*- coding: utf-8 -*-
"""Tests for graph / mesh connected components (graph.components).

The label-propagation components are anchored against
``scipy.sparse.csgraph.connected_components`` -- two nodes share a nitrix label
iff scipy places them in the same component -- across random graphs, plus the
structural guarantees: background nodes label 0, labels are contiguous 1..K,
isolated foreground nodes are singletons, and the labelling is orientation- and
jit-invariant.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.graph import connected_components  # noqa: E402


def _edges_from_dense(adj):
    ii, jj = np.nonzero(np.triu(adj, 1))
    return jnp.asarray(np.stack([ii, jj], axis=1))


def _same_partition(got, ref, foreground):
    fg = np.where(foreground)[0]
    for a in range(len(fg)):
        for b in range(a + 1, len(fg)):
            u, v = fg[a], fg[b]
            if (got[u] == got[v]) != (ref[u] == ref[v]):
                return False
    return True


def test_matches_scipy_partition_random_graphs():
    csg = pytest.importorskip('scipy.sparse.csgraph')
    sp = pytest.importorskip('scipy.sparse')
    rng = np.random.default_rng(0)
    for _ in range(15):
        n = int(rng.integers(12, 50))
        adj = rng.uniform(size=(n, n)) < rng.uniform(0.03, 0.14)
        adj = np.triu(adj, 1)
        adj = adj | adj.T
        mask = rng.uniform(size=n) < 0.7
        masked = adj.copy()
        masked[~mask, :] = False
        masked[:, ~mask] = False
        _, ref = csg.connected_components(
            sp.csr_matrix(masked), directed=False
        )
        got = np.asarray(
            connected_components(jnp.asarray(mask), _edges_from_dense(adj))
        )
        assert np.all(got[~mask] == 0)
        assert _same_partition(got, ref, mask)


def test_background_is_zero_and_labels_contiguous():
    edges = jnp.asarray([[0, 1], [1, 2], [4, 5]])
    mask = jnp.asarray([True, True, True, False, True, True])
    labels = np.asarray(connected_components(mask, edges))
    assert labels[3] == 0  # background node
    positive = labels[labels > 0]
    assert set(positive.tolist()) == {1, 2}  # contiguous 1..K
    assert labels[0] == labels[1] == labels[2]  # one component
    assert labels[4] == labels[5]  # the other
    assert labels[0] != labels[4]


def test_isolated_foreground_node_is_a_singleton():
    edges = jnp.asarray([[0, 1]])
    mask = jnp.asarray([True, True, True])  # node 2 has no edges
    labels = np.asarray(connected_components(mask, edges))
    assert labels[2] != 0
    assert labels[2] != labels[0]
    assert len(np.unique(labels[labels > 0])) == 2


def test_edge_orientation_is_ignored():
    mask = jnp.asarray([True, True, True])
    fwd = connected_components(mask, jnp.asarray([[0, 1], [1, 2]]))
    rev = connected_components(mask, jnp.asarray([[1, 0], [2, 1]]))
    np.testing.assert_array_equal(np.asarray(fwd), np.asarray(rev))


def test_jit_invariance():
    edges = jnp.asarray([[0, 1], [2, 3]])
    mask = jnp.asarray([True, True, True, True])
    eager = connected_components(mask, edges)
    jitted = jax.jit(connected_components)(mask, edges)
    np.testing.assert_array_equal(np.asarray(eager), np.asarray(jitted))
