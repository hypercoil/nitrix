# -*- coding: utf-8 -*-
"""Tests for ``nitrix.semiring.semiring_ell_edge_aggregate``.

Covers:

- **GCN forward** matches brute-force reference at machine eps.
- **DGCNN forward** runs end-to-end with a learned MLP edge_fn.
- **TROPICAL_MAX / MIN_PLUS** semiring reductions.
- **Padding** is correctly absorbed when edge_fn multiplies by ``w``.
- **Differentiability** through edge_fn parameters.
- **Batched** call gives per-batch results matching unbatched.
- **Unsupported semiring** raises clearly (LOG / EUCLIDEAN deferred).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import dataclasses

from nitrix.semiring import (
    LOG,
    REAL,
    TROPICAL_MAX_PLUS,
    TROPICAL_MIN_PLUS,
    ell_row_softmax,
    semiring_ell_edge_aggregate,
)
from nitrix.sparse import ELL


def _toy_graph(n=4, k_max=3, d_in=2, seed=0):
    rng = np.random.default_rng(seed)
    values = jnp.asarray(rng.standard_normal((n, k_max)))
    indices = jnp.asarray(rng.integers(0, n, (n, k_max)).astype(np.int32))
    x = jnp.asarray(rng.standard_normal((n, d_in)))
    ell = ELL(values=values, indices=indices, n_cols=n, identity=0.0)
    return ell, x


def test_gcn_matches_brute_force():
    '''GCN-style edge_fn (linear projection of neighbour) matches
    a Python brute-force reference at machine eps.
    '''
    ell, x = _toy_graph()
    n, k_max = ell.values.shape
    d_in = x.shape[-1]
    d_out = 4
    W = jnp.asarray(np.random.default_rng(1).standard_normal((d_out, d_in)))

    def edge_fn(h_i, h_j, w, ij):
        return w * (W @ h_j)

    out = semiring_ell_edge_aggregate(edge_fn, ell, x, semiring=REAL)

    # Brute force
    ref = jnp.zeros((n, d_out))
    for i in range(n):
        for p in range(k_max):
            j = int(ell.indices[i, p])
            w = float(ell.values[i, p])
            ref = ref.at[i].add(w * (W @ x[j]))

    np.testing.assert_allclose(out, ref, atol=1e-13)


def test_dgcnn_forward_runs():
    '''DGCNN/EdgeConv-style edge_fn with concat + MLP runs end-
    to-end and produces finite output.
    '''
    ell, x = _toy_graph(n=8, d_in=4)
    rng = np.random.default_rng(2)
    W1 = jnp.asarray(rng.standard_normal((16, 8)))
    W2 = jnp.asarray(rng.standard_normal((8, 16)))

    def edge_fn(h_i, h_j, w, ij):
        msg = jnp.concatenate([h_i, h_j - h_i])
        h = jax.nn.leaky_relu(W1 @ msg, 0.3)
        return w * (W2 @ h)

    out = semiring_ell_edge_aggregate(edge_fn, ell, x, semiring=REAL)
    assert out.shape == (8, 8)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_tropical_max_plus_semiring():
    '''MAX_PLUS reduction returns elementwise max over the neighbours.'''
    ell, x = _toy_graph(d_in=1)
    def edge_fn(h_i, h_j, w, ij):
        return w + h_j  # additive in the tropical sense

    out = semiring_ell_edge_aggregate(
        edge_fn, ell, x, semiring=TROPICAL_MAX_PLUS,
    )
    # Hand-roll: per row, max over the 3 (value + neighbour) sums
    ref = jnp.zeros_like(out)
    n, k_max = ell.values.shape
    for i in range(n):
        per_row = []
        for p in range(k_max):
            j = int(ell.indices[i, p])
            w = float(ell.values[i, p])
            per_row.append(w + x[j])
        ref = ref.at[i].set(jnp.max(jnp.stack(per_row), axis=0))
    np.testing.assert_allclose(out, ref, atol=1e-13)


def test_tropical_min_plus_semiring():
    '''MIN_PLUS reduction returns elementwise min over the neighbours.'''
    ell, x = _toy_graph(d_in=1)
    def edge_fn(h_i, h_j, w, ij):
        return w + h_j

    out = semiring_ell_edge_aggregate(
        edge_fn, ell, x, semiring=TROPICAL_MIN_PLUS,
    )
    n, k_max = ell.values.shape
    ref = jnp.zeros_like(out)
    for i in range(n):
        per_row = []
        for p in range(k_max):
            j = int(ell.indices[i, p])
            w = float(ell.values[i, p])
            per_row.append(w + x[j])
        ref = ref.at[i].set(jnp.min(jnp.stack(per_row), axis=0))
    np.testing.assert_allclose(out, ref, atol=1e-13)


def test_padding_absorbed_when_user_multiplies_by_w():
    '''When edge_fn multiplies by ``w``, ELL pad rows (w=0)
    contribute 0 to the REAL aggregate -- matching the documented
    user contract.
    '''
    n, k_max, d_in = 5, 4, 2
    # Build an ELL where the LAST column of every row is padding:
    # indices point to row 0, values = 0.
    rng = np.random.default_rng(0)
    values_full = rng.standard_normal((n, k_max - 1))
    values = jnp.asarray(np.concatenate(
        [values_full, np.zeros((n, 1))], axis=1,
    ))
    indices_full = rng.integers(0, n, (n, k_max - 1))
    indices = jnp.asarray(
        np.concatenate([indices_full, np.zeros((n, 1))], axis=1).astype(np.int32),
    )
    x = jnp.asarray(rng.standard_normal((n, d_in)))
    ell = ELL(values=values, indices=indices, n_cols=n, identity=0.0)

    W = jnp.asarray(rng.standard_normal((3, d_in)))

    def edge_fn(h_i, h_j, w, ij):
        return w * (W @ h_j)

    out = semiring_ell_edge_aggregate(edge_fn, ell, x, semiring=REAL)

    # Reference: same fn with the padding column dropped.
    ref = jnp.zeros((n, 3))
    for i in range(n):
        for p in range(k_max - 1):  # exclude pad
            j = int(indices[i, p])
            w = float(values[i, p])
            ref = ref.at[i].add(w * (W @ x[j]))
    np.testing.assert_allclose(out, ref, atol=1e-13)


def test_differentiable_through_edge_fn_params():
    '''Gradients flow through parameters captured in the edge_fn closure.'''
    ell, x = _toy_graph()
    d_out = 4
    d_in = x.shape[-1]
    W = jnp.asarray(np.random.default_rng(1).standard_normal((d_out, d_in)))

    def loss(W):
        def edge_fn(h_i, h_j, w, ij):
            return w * (W @ h_j)
        out = semiring_ell_edge_aggregate(edge_fn, ell, x, semiring=REAL)
        return jnp.sum(out ** 2)

    g = jax.grad(loss)(W)
    assert g.shape == W.shape
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.abs(g).max()) > 0  # nontrivial gradient


def test_differentiable_through_x():
    '''Gradients flow back to the input features.'''
    ell, x = _toy_graph()
    d_out = 4
    W = jnp.asarray(np.random.default_rng(1).standard_normal((d_out, x.shape[-1])))

    def loss(x):
        def edge_fn(h_i, h_j, w, ij):
            return w * (W @ h_j)
        return jnp.sum(
            semiring_ell_edge_aggregate(edge_fn, ell, x, semiring=REAL) ** 2,
        )

    g = jax.grad(loss)(x)
    assert g.shape == x.shape
    assert bool(jnp.all(jnp.isfinite(g)))


def test_batched_matches_unbatched():
    '''Leading batch dims on ``x`` produce per-batch results that
    match the unbatched fit.
    '''
    ell, x = _toy_graph(n=4, d_in=3)
    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.standard_normal((5, 3)))

    def edge_fn(h_i, h_j, w, ij):
        return w * (W @ h_j)

    B = 4
    x_batched = jnp.stack([x + 0.1 * b for b in range(B)])  # (B, n, d_in)
    out_batched = semiring_ell_edge_aggregate(
        edge_fn, ell, x_batched, semiring=REAL,
    )
    assert out_batched.shape == (B, 4, 5)
    for b in range(B):
        out_single = semiring_ell_edge_aggregate(
            edge_fn, ell, x_batched[b], semiring=REAL,
        )
        np.testing.assert_allclose(
            out_batched[b], out_single, atol=1e-13,
        )


def test_unsupported_semiring_raises():
    '''LOG is not yet supported -- should raise NotImplementedError.'''
    ell, x = _toy_graph()
    def edge_fn(h_i, h_j, w, ij):
        return w * h_j

    with pytest.raises(NotImplementedError, match='LOG'):
        semiring_ell_edge_aggregate(edge_fn, ell, x, semiring=LOG)


# ---------------------------------------------------------------------------
# edge_attr: per-edge attribute tensor (SUGAR / GATv2 delta 1)
# ---------------------------------------------------------------------------


def _toy_graph_with_attr(n=5, k_max=3, d_in=2, d_e=4, seed=0):
    rng = np.random.default_rng(seed)
    values = jnp.asarray(rng.standard_normal((n, k_max)))
    indices = jnp.asarray(rng.integers(0, n, (n, k_max)).astype(np.int32))
    x = jnp.asarray(rng.standard_normal((n, d_in)))
    edge_attr = jnp.asarray(rng.standard_normal((n, k_max, d_e)))
    ell = ELL(values=values, indices=indices, n_cols=n, identity=0.0)
    return ell, x, edge_attr


def test_edge_attr_gatv2_shaped_matches_brute_force():
    '''A GATv2-style edge_fn that folds a per-edge attribute vector
    into the message matches a Python brute-force reference.
    '''
    ell, x, edge_attr = _toy_graph_with_attr(d_in=3, d_e=4)
    n, k_max = ell.values.shape
    d_out = 6
    rng = np.random.default_rng(7)
    W = jnp.asarray(rng.standard_normal((d_out, x.shape[-1])))
    W_e = jnp.asarray(rng.standard_normal((d_out, edge_attr.shape[-1])))

    def edge_fn(h_i, h_j, w, ij, a):
        return w * (W @ h_j + W_e @ a)

    out = semiring_ell_edge_aggregate(
        edge_fn, ell, x, semiring=REAL, edge_attr=edge_attr,
    )
    assert out.shape == (n, d_out)

    ref = jnp.zeros((n, d_out))
    for i in range(n):
        for p in range(k_max):
            j = int(ell.indices[i, p])
            w = float(ell.values[i, p])
            a = edge_attr[i, p]
            ref = ref.at[i].add(w * (W @ x[j] + W_e @ a))
    np.testing.assert_allclose(out, ref, atol=1e-13)


def test_edge_attr_none_keeps_four_arg_contract():
    '''With edge_attr=None (default) the four-argument edge_fn path is
    unchanged -- the scalar w is the third argument.
    '''
    ell, x = _toy_graph(d_in=2)
    n, k_max = ell.values.shape
    W = jnp.asarray(np.random.default_rng(1).standard_normal((3, 2)))

    def edge_fn(h_i, h_j, w, ij):
        return w * (W @ h_j)

    out_default = semiring_ell_edge_aggregate(edge_fn, ell, x, semiring=REAL)
    out_explicit = semiring_ell_edge_aggregate(
        edge_fn, ell, x, semiring=REAL, edge_attr=None,
    )
    np.testing.assert_array_equal(out_default, out_explicit)


def test_edge_attr_preserves_scalar_w_for_padding():
    '''edge_attr does not displace the scalar w: pad rows (w=0) are
    still absorbed when the message multiplies by w, even though a
    per-edge attribute vector is also supplied.
    '''
    n, k_max, d_in, d_e = 5, 4, 2, 3
    rng = np.random.default_rng(0)
    values_full = rng.standard_normal((n, k_max - 1))
    values = jnp.asarray(np.concatenate([values_full, np.zeros((n, 1))], axis=1))
    indices_full = rng.integers(0, n, (n, k_max - 1))
    indices = jnp.asarray(
        np.concatenate([indices_full, np.zeros((n, 1))], axis=1).astype(np.int32),
    )
    x = jnp.asarray(rng.standard_normal((n, d_in)))
    edge_attr = jnp.asarray(rng.standard_normal((n, k_max, d_e)))
    ell = ELL(values=values, indices=indices, n_cols=n, identity=0.0)

    W = jnp.asarray(rng.standard_normal((4, d_in)))
    W_e = jnp.asarray(rng.standard_normal((4, d_e)))

    def edge_fn(h_i, h_j, w, ij, a):
        return w * (W @ h_j + W_e @ a)

    out = semiring_ell_edge_aggregate(
        edge_fn, ell, x, semiring=REAL, edge_attr=edge_attr,
    )
    ref = jnp.zeros((n, 4))
    for i in range(n):
        for p in range(k_max - 1):  # exclude the pad column
            j = int(indices[i, p])
            w = float(values[i, p])
            ref = ref.at[i].add(w * (W @ x[j] + W_e @ edge_attr[i, p]))
    np.testing.assert_allclose(out, ref, atol=1e-13)


def test_edge_attr_wrong_shape_raises():
    '''edge_attr that does not lead with (n, k_max) raises a clear error.'''
    ell, x, edge_attr = _toy_graph_with_attr(n=5, k_max=3, d_e=4)

    def edge_fn(h_i, h_j, w, ij, a):
        return w * a

    bad = edge_attr[:, :2]  # wrong k_max
    with pytest.raises(ValueError, match='edge_attr'):
        semiring_ell_edge_aggregate(
            edge_fn, ell, x, semiring=REAL, edge_attr=bad,
        )


def test_edge_attr_batched_shares_across_batch():
    '''edge_attr is shared across leading batch axes of x, like values.'''
    ell, x, edge_attr = _toy_graph_with_attr(n=4, k_max=3, d_in=3, d_e=2)
    rng = np.random.default_rng(3)
    W = jnp.asarray(rng.standard_normal((5, 3)))
    W_e = jnp.asarray(rng.standard_normal((5, 2)))

    def edge_fn(h_i, h_j, w, ij, a):
        return w * (W @ h_j + W_e @ a)

    B = 3
    x_batched = jnp.stack([x + 0.1 * b for b in range(B)])
    out_batched = semiring_ell_edge_aggregate(
        edge_fn, ell, x_batched, semiring=REAL, edge_attr=edge_attr,
    )
    assert out_batched.shape == (B, 4, 5)
    for b in range(B):
        out_single = semiring_ell_edge_aggregate(
            edge_fn, ell, x_batched[b], semiring=REAL, edge_attr=edge_attr,
        )
        np.testing.assert_allclose(out_batched[b], out_single, atol=1e-13)


def test_edge_attr_differentiable_through_edge_params():
    '''Gradients flow through the edge-attribute projection matrix.'''
    ell, x, edge_attr = _toy_graph_with_attr(d_in=3, d_e=4)
    d_out = 5
    rng = np.random.default_rng(9)
    W = jnp.asarray(rng.standard_normal((d_out, x.shape[-1])))

    def loss(W_e):
        def edge_fn(h_i, h_j, w, ij, a):
            return w * (W @ h_j + W_e @ a)
        out = semiring_ell_edge_aggregate(
            edge_fn, ell, x, semiring=REAL, edge_attr=edge_attr,
        )
        return jnp.sum(out ** 2)

    W_e0 = jnp.asarray(rng.standard_normal((d_out, edge_attr.shape[-1])))
    g = jax.grad(loss)(W_e0)
    assert g.shape == W_e0.shape
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.abs(g).max()) > 0


# ---------------------------------------------------------------------------
# ell_row_softmax (GAT attention pre-pass; SUGAR delta 1b)
# ---------------------------------------------------------------------------


def test_ell_row_softmax_matches_masked_dense_softmax():
    '''Row softmax over the ELL slots equals a masked dense softmax,
    with padding (values == identity) excluded.
    '''
    n, k_max = 6, 4
    rng = np.random.default_rng(0)
    # Binary adjacency: 1 at valid edges, 0 at pad (identity 0.0).
    valid = rng.integers(0, 2, (n, k_max)).astype(bool)
    valid[:, 0] = True  # ensure each row has at least one valid edge
    values = jnp.asarray(valid.astype(np.float64))
    indices = jnp.asarray(rng.integers(0, n, (n, k_max)).astype(np.int32))
    ell = ELL(values=values, indices=indices, n_cols=n, identity=0.0)

    scores = jnp.asarray(rng.standard_normal((n, k_max)))
    out = ell_row_softmax(scores, ell)

    # Reference: -inf at invalid, then softmax.
    masked = np.where(valid, np.asarray(scores), -np.inf)
    m = masked.max(axis=-1, keepdims=True)
    e = np.where(valid, np.exp(masked - m), 0.0)
    ref = e / e.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(out, ref, atol=1e-13)
    # Valid rows sum to 1.
    np.testing.assert_allclose(np.asarray(out).sum(axis=-1), np.ones(n), atol=1e-13)


def test_ell_row_softmax_all_pad_row_is_zero():
    '''An isolated vertex (all-pad row) returns zeros, not NaN.'''
    n, k_max = 3, 3
    values = jnp.asarray(np.array([
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],  # all padding -> isolated
        [1.0, 0.0, 0.0],
    ]))
    indices = jnp.zeros((n, k_max), dtype=jnp.int32)
    ell = ELL(values=values, indices=indices, n_cols=n, identity=0.0)
    scores = jnp.asarray(np.random.default_rng(0).standard_normal((n, k_max)))
    out = ell_row_softmax(scores, ell)
    assert bool(jnp.all(jnp.isfinite(out)))
    np.testing.assert_allclose(out[1], np.zeros(k_max), atol=1e-13)
    np.testing.assert_allclose(float(out[0].sum()), 1.0, atol=1e-13)


def test_ell_row_softmax_then_aggregate_is_convex_combination():
    '''Using ell_row_softmax weights as the message scale yields a
    per-row convex combination of neighbour features (GAT readout).
    '''
    n, k_max, d = 5, 4, 3
    rng = np.random.default_rng(1)
    valid = np.ones((n, k_max), dtype=bool)
    valid[:, -1] = False  # last column padding everywhere
    values = jnp.asarray(valid.astype(np.float64))
    indices = jnp.asarray(rng.integers(0, n, (n, k_max)).astype(np.int32))
    ell = ELL(values=values, indices=indices, n_cols=n, identity=0.0)
    x = jnp.asarray(rng.standard_normal((n, d)))
    scores = jnp.asarray(rng.standard_normal((n, k_max)))

    alpha = ell_row_softmax(scores, ell)
    ell_alpha = dataclasses.replace(ell, values=alpha)

    def edge_fn(h_i, h_j, w, ij):
        return w * h_j  # convex combination since sum_p w = 1

    out = semiring_ell_edge_aggregate(edge_fn, ell_alpha, x, semiring=REAL)
    # Each output row is within the convex hull of its neighbour feats.
    for i in range(n):
        neigh = np.asarray(x[np.asarray(indices[i])[:k_max - 1]])
        lo, hi = neigh.min(axis=0), neigh.max(axis=0)
        assert bool(np.all(np.asarray(out[i]) >= lo - 1e-9))
        assert bool(np.all(np.asarray(out[i]) <= hi + 1e-9))


def test_index_pair_seen_by_edge_fn():
    '''edge_fn receives the (i, j) pair correctly.'''
    ell, x = _toy_graph(n=4, k_max=3, d_in=1)

    # Build an edge_fn that uses ij; the output should reflect
    # the correct indices.
    def edge_fn(h_i, h_j, w, ij):
        i, j = ij[0], ij[1]
        # Return a one-element vector: i * j (just a probe).
        return (i.astype(jnp.float64) * j.astype(jnp.float64) * w)[None]

    out = semiring_ell_edge_aggregate(edge_fn, ell, x, semiring=REAL)
    # Reference
    ref = jnp.zeros((4, 1))
    for i in range(4):
        for p in range(3):
            j = int(ell.indices[i, p])
            w = float(ell.values[i, p])
            ref = ref.at[i].add(jnp.array([i * j * w]))
    np.testing.assert_allclose(out, ref, atol=1e-13)
