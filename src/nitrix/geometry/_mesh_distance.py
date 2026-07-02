# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Differentiable dense geometric distance kernels for mesh losses.

The pure-JAX, ``grad``-through-vertices distance primitives a mesh
training loss is built from -- the geometry substrate underneath a
chamfer / self-intersection / proximity penalty, with the loss *reduction*
(the symmetric mean, the hinge) left to the consumer:

- :func:`segment_segment_sq_dist` -- the clamped segment-to-segment squared
  distance (Ericson), the proximity core of an edge-edge self-intersection
  penalty.
- :func:`point_set_nearest_sq_dist` -- per-query nearest squared distance to
  a reference point set, the chamfer core; chunked to bound the
  :math:`n \times m` intermediate.

Distinct from :func:`nearest_surface_distance` -- that is the **host-side,
non-differentiable** point-to-*triangle-mesh* distance (with a uniform-grid
broad phase) used by :func:`cortical_thickness` / :func:`mesh_to_sdf`.  These
are **traced, differentiable** dense kernels over point / segment sets; the
exact spatial index that would accelerate them is tracked separately
(``docs/feature-requests/mesh-spatial-acceleration.md``), so the broad phase
here is dense (:math:`O(n m)` / pairwise).
"""

from __future__ import annotations

from typing import Optional, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = [
    'segment_segment_sq_dist',
    'point_set_nearest_sq_dist',
]

# Degeneracy guard for a segment that has collapsed to a point (squared
# length below this is treated as zero-length).  Absolute, so it only fires
# on a genuinely degenerate segment; a real mesh edge never reaches it.
_DEGENERATE_SQ_LEN = 1e-12


def segment_segment_sq_dist(
    p1: Float[Array, '*batch 3'],
    q1: Float[Array, '*batch 3'],
    p2: Float[Array, '*batch 3'],
    q2: Float[Array, '*batch 3'],
) -> Float[Array, '*batch']:
    r"""Clamped segment-to-segment squared distance (Ericson).

    The squared Euclidean distance between the closest points of segment
    ``[p1, q1]`` and segment ``[p2, q2]`` -- the branchless, vectorised form
    of Ericson's ``ClosestPtSegmentSegment`` (*Real-Time Collision
    Detection*, section 5.1.9): minimise
    :math:`\|(p_1 + s d_1) - (p_2 + t d_2)\|^2` over
    :math:`(s, t) \in [0, 1]^2` with the parameters clamped to the unit
    square.

    The proximity core of an edge-edge mesh self-intersection penalty: two
    mesh edges intersect (or nearly so) iff this distance is ~0.  Inputs
    broadcast over any leading ``*batch`` (e.g. a set of candidate edge
    pairs), so the per-pair distance is one vectorised call.

    Parameters
    ----------
    p1, q1
        Endpoints of the first segment, ``(*batch, 3)``.
    p2, q2
        Endpoints of the second segment, ``(*batch, 3)``.

    Returns
    -------
    ``(*batch,)`` closest-point squared distances.  Pure JAX; differentiable
    w.r.t. all four endpoints (the clamps are sub-differentiable).  A
    degenerate (point-like) segment is handled without ``NaN``.

    Notes
    -----
    Squared distance (no ``sqrt``) -- it is the cheaper, smoother quantity a
    penalty thresholds, and the caller takes the root only if a true length
    is needed.
    """
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = jnp.sum(d1 * d1, axis=-1)  # squared length of segment 1
    e = jnp.sum(d2 * d2, axis=-1)  # squared length of segment 2
    f = jnp.sum(d2 * r, axis=-1)
    c = jnp.sum(d1 * r, axis=-1)
    b = jnp.sum(d1 * d2, axis=-1)

    a_safe = jnp.where(a > _DEGENERATE_SQ_LEN, a, 1.0)
    e_safe = jnp.where(e > _DEGENERATE_SQ_LEN, e, 1.0)
    denom = a * e - b * b

    # General (both non-degenerate) solve: s from the joint stationary point
    # (denom > 0), then t = (b s + f) / e clamped, recomputing s when t is
    # clamped (the four edge regions collapse to s = clamp((b t - c) / a)).
    s = jnp.where(denom > _DEGENERATE_SQ_LEN, (b * f - c * e) / denom, 0.0)
    s = jnp.clip(s, 0.0, 1.0)
    t = (b * s + f) / e_safe
    t_lo, t_hi = t < 0.0, t > 1.0
    t = jnp.clip(t, 0.0, 1.0)
    s = jnp.where(t_lo, jnp.clip(-c / a_safe, 0.0, 1.0), s)
    s = jnp.where(t_hi, jnp.clip((b - c) / a_safe, 0.0, 1.0), s)

    # Degenerate overrides (a segment collapsed to a point).
    a_deg = a <= _DEGENERATE_SQ_LEN
    e_deg = e <= _DEGENERATE_SQ_LEN
    # Segment 2 a point: t = 0, s = clamp(-c / a).
    t = jnp.where(e_deg, 0.0, t)
    s = jnp.where(e_deg, jnp.clip(-c / a_safe, 0.0, 1.0), s)
    # Segment 1 a point: s = 0, t = clamp(f / e).
    s = jnp.where(a_deg, 0.0, s)
    t = jnp.where(a_deg, jnp.clip(f / e_safe, 0.0, 1.0), t)
    # Both points: s = t = 0 (a already set s = 0; pin t).
    t = jnp.where(a_deg & e_deg, 0.0, t)

    closest1 = p1 + s[..., None] * d1
    closest2 = p2 + t[..., None] * d2
    diff = closest1 - closest2
    return jnp.sum(diff * diff, axis=-1)


def _block_min_sq_dist(
    block: Float[Array, 'c d'], refs: Float[Array, 'm d']
) -> Float[Array, ' c']:
    """Per-row nearest squared distance from ``block`` rows to ``refs``.

    Forms the full pairwise squared-distance matrix between the two point
    sets and reduces it along the reference axis.

    Parameters
    ----------
    block
        ``(c, d)`` block of query points.
    refs
        ``(m, d)`` reference points.

    Returns
    -------
    Float[Array, ' c']
        ``(c,)`` array whose ``i``-th entry is the minimum squared Euclidean
        distance from ``block[i]`` to any row of ``refs``.
    """
    d2 = jnp.sum((block[:, None, :] - refs[None, :, :]) ** 2, axis=-1)
    return jnp.min(d2, axis=-1)


def point_set_nearest_sq_dist(
    queries: Float[Array, 'n d'],
    refs: Float[Array, 'm d'],
    *,
    chunk_size: Optional[int] = None,
) -> Float[Array, ' n']:
    r"""Per-query nearest squared distance to a reference point set.

    For each row of ``queries``, the minimum squared Euclidean distance to
    any row of ``refs`` -- the (one-directional) **chamfer core**.  The
    chamfer *distance* is the symmetric, reduced quantity
    :math:`\operatorname{mean}(\text{nearest}(A \to B)) +
    \operatorname{mean}(\text{nearest}(B \to A))`; that reduction is left to
    the consumer, so this returns the **unreduced** per-query distances.

    Parameters
    ----------
    queries
        ``(n, d)`` query points.
    refs
        ``(m, d)`` reference points.
    chunk_size
        Optional query-block size bounding the ``(chunk_size, m)`` pairwise
        intermediate.  ``None`` (default) forms the full ``(n, m)`` block in
        one shot (fine for moderate sizes); a positive value evaluates the
        queries in ``lax.map``-ed blocks (queries are padded to a multiple of
        ``chunk_size`` and sliced back), trading the dense intermediate for a
        bounded one.

    Returns
    -------
    ``(n,)`` nearest squared distances.  Pure JAX; differentiable w.r.t. both
    point sets (the ``min`` is sub-differentiable -- the gradient flows to the
    nearest reference).

    Notes
    -----
    Squared distance (no ``sqrt``), the chamfer convention.  The broad phase
    is dense (:math:`O(n m)`); an exact spatial index is the separate, gated
    ``mesh-spatial-acceleration`` request.
    """
    if refs.shape[0] == 0:
        raise ValueError('point_set_nearest_sq_dist: refs is empty.')
    n = queries.shape[0]
    if n == 0:
        return jnp.zeros((0,), dtype=queries.dtype)
    if chunk_size is None or chunk_size >= n:
        return _block_min_sq_dist(queries, refs)
    if chunk_size < 1:
        raise ValueError(f'chunk_size must be >= 1; got {chunk_size}.')
    pad = (-n) % chunk_size
    if pad:
        queries = jnp.concatenate(
            [queries, jnp.zeros((pad, queries.shape[1]), queries.dtype)],
            axis=0,
        )
    blocks = queries.reshape(-1, chunk_size, queries.shape[1])
    out = jax.lax.map(lambda blk: _block_min_sq_dist(blk, refs), blocks)
    return cast(Float[Array, ' n'], out.reshape(-1)[:n])
