# -*- coding: utf-8 -*-
"""Tests for the differentiable dense distance kernels (mesh-loss substrate).

``geometry.segment_segment_sq_dist`` (Ericson clamped segment-segment) and
``geometry.point_set_nearest_sq_dist`` (the chamfer core).  Oracles: a dense
``(s, t)`` grid scan for the segment kernel, and a brute pairwise ``cdist``
for the point-set kernel.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.geometry import (
    point_set_nearest_sq_dist,
    segment_segment_sq_dist,
)

# ---------------------------------------------------------------------------
# segment_segment_sq_dist
# ---------------------------------------------------------------------------


def _seg_seg_grid_oracle(p1, q1, p2, q2, n=400):
    s = np.linspace(0.0, 1.0, n)[:, None]
    t = np.linspace(0.0, 1.0, n)[None, :]
    # c1(s) - c2(t) over the (s, t) grid, min squared norm.
    c1 = p1[None, None] + s[..., None] * (q1 - p1)[None, None]
    c2 = p2[None, None] + t[..., None] * (q2 - p2)[None, None]
    d2 = np.sum((c1 - c2) ** 2, axis=-1)
    return float(d2.min())


def _call(p1, q1, p2, q2):
    return float(
        segment_segment_sq_dist(
            jnp.asarray(p1), jnp.asarray(q1), jnp.asarray(p2), jnp.asarray(q2)
        )
    )


def test_parallel_offset_segments():
    # Two unit segments along x, offset by 0.5 in y -> distance^2 = 0.25.
    p1, q1 = np.array([0.0, 0, 0]), np.array([1.0, 0, 0])
    p2, q2 = np.array([0.0, 0.5, 0]), np.array([1.0, 0.5, 0])
    assert abs(_call(p1, q1, p2, q2) - 0.25) < 1e-9


def test_crossing_segments_zero():
    # Two segments that cross at the origin -> distance 0.
    p1, q1 = np.array([-1.0, 0, 0]), np.array([1.0, 0, 0])
    p2, q2 = np.array([0.0, -1, 0]), np.array([0.0, 1, 0])
    assert _call(p1, q1, p2, q2) < 1e-12


def test_skew_segments_match_grid_oracle():
    rng = np.random.default_rng(3)
    for _ in range(20):
        p1, q1, p2, q2 = (rng.normal(size=3) for _ in range(4))
        got = _call(p1, q1, p2, q2)
        ref = _seg_seg_grid_oracle(p1, q1, p2, q2)
        # The grid oracle is an upper bound that converges from above; allow a
        # tolerance proportional to the grid step.
        assert abs(got - ref) < 1e-3, (got, ref)


def test_endpoint_separated_segments():
    # Collinear, non-overlapping segments on the x-axis: nearest points are
    # the inner endpoints, gap 1.0 -> distance^2 = 1.0.
    p1, q1 = np.array([0.0, 0, 0]), np.array([1.0, 0, 0])
    p2, q2 = np.array([2.0, 0, 0]), np.array([3.0, 0, 0])
    assert abs(_call(p1, q1, p2, q2) - 1.0) < 1e-9


def test_degenerate_both_points():
    p = np.array([0.0, 0, 0])
    q = np.array([3.0, 4.0, 0.0])
    d2 = _call(p, p, q, q)  # both segments are points
    assert abs(d2 - 25.0) < 1e-9


def test_degenerate_one_point_to_segment():
    # A point at (0.5, 1, 0) to the unit x-segment -> nearest is (0.5,0,0).
    pt = np.array([0.5, 1.0, 0.0])
    p2, q2 = np.array([0.0, 0, 0]), np.array([1.0, 0, 0])
    assert abs(_call(pt, pt, p2, q2) - 1.0) < 1e-9


def test_segment_segment_batched():
    rng = np.random.default_rng(5)
    p1, q1, p2, q2 = (jnp.asarray(rng.normal(size=(17, 3))) for _ in range(4))
    out = segment_segment_sq_dist(p1, q1, p2, q2)
    assert out.shape == (17,)
    # Cross-check a couple of lanes against the scalar oracle.
    for i in (0, 9, 16):
        ref = _seg_seg_grid_oracle(
            np.asarray(p1[i]),
            np.asarray(q1[i]),
            np.asarray(p2[i]),
            np.asarray(q2[i]),
        )
        assert abs(float(out[i]) - ref) < 2e-3


def test_segment_segment_differentiable():
    rng = np.random.default_rng(7)
    p1, q1, p2, q2 = (jnp.asarray(rng.normal(size=3)) for _ in range(4))
    g = jax.grad(lambda a: segment_segment_sq_dist(a, q1, p2, q2))(p1)
    assert g.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_segment_segment_jit():
    p1, q1 = jnp.zeros(3), jnp.array([1.0, 0, 0])
    p2, q2 = jnp.array([0.0, 0.5, 0]), jnp.array([1.0, 0.5, 0])
    f = jax.jit(segment_segment_sq_dist)
    assert abs(float(f(p1, q1, p2, q2)) - 0.25) < 1e-9


# ---------------------------------------------------------------------------
# point_set_nearest_sq_dist
# ---------------------------------------------------------------------------


def _nn_oracle(queries, refs):
    d2 = np.sum((queries[:, None, :] - refs[None, :, :]) ** 2, axis=-1)
    return d2.min(axis=1)


def test_nearest_matches_brute_oracle():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(40, 3))
    r = rng.normal(size=(55, 3))
    got = np.asarray(point_set_nearest_sq_dist(jnp.asarray(q), jnp.asarray(r)))
    np.testing.assert_allclose(got, _nn_oracle(q, r), atol=1e-9)


def test_nearest_coincident_points_zero():
    q = jnp.asarray(np.eye(3))
    out = point_set_nearest_sq_dist(q, q)  # each query is its own ref
    np.testing.assert_allclose(np.asarray(out), 0.0, atol=1e-12)


def test_nearest_chunked_equals_dense():
    rng = np.random.default_rng(1)
    q = jnp.asarray(rng.normal(size=(43, 4)))  # 43 not divisible by chunk
    r = jnp.asarray(rng.normal(size=(31, 4)))
    dense = point_set_nearest_sq_dist(q, r)
    chunked = point_set_nearest_sq_dist(q, r, chunk_size=8)
    np.testing.assert_allclose(
        np.asarray(dense), np.asarray(chunked), atol=1e-9
    )


def test_nearest_differentiable():
    rng = np.random.default_rng(2)
    q = jnp.asarray(rng.normal(size=(10, 3)))
    r = jnp.asarray(rng.normal(size=(12, 3)))
    # gradient of the (one-directional) chamfer mean w.r.t. the query set.
    g = jax.grad(lambda x: jnp.mean(point_set_nearest_sq_dist(x, r)))(q)
    assert g.shape == q.shape
    assert bool(jnp.all(jnp.isfinite(g)))


def test_nearest_empty_queries():
    r = jnp.asarray(np.random.default_rng(0).normal(size=(5, 3)))
    out = point_set_nearest_sq_dist(jnp.zeros((0, 3)), r)
    assert out.shape == (0,)


def test_nearest_empty_refs_raises():
    import pytest

    q = jnp.asarray(np.random.default_rng(0).normal(size=(5, 3)))
    with pytest.raises(ValueError):
        point_set_nearest_sq_dist(q, jnp.zeros((0, 3)))
