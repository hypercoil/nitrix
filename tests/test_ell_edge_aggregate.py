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

from nitrix.semiring import (
    LOG,
    REAL,
    TROPICAL_MAX_PLUS,
    TROPICAL_MIN_PLUS,
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
