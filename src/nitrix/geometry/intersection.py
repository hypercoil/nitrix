# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Triangle-mesh self-intersection detection and removal.

A **host-side QA / cleanup** tool, not a differentiable kernel: a uniform-grid
broad phase finds candidate face pairs, a segment-triangle (Moller-Trumbore)
narrow phase confirms genuine crossings, and ``remove_self_intersections``
relaxes the offending vertices with Laplacian smoothing.

**Use as a post-hoc cleanup, never inside a jitted loop.**  The broad/narrow
phase is host-side NumPy and cannot run under ``lax.fori_loop``; surface movers
like ``deform_to_sdf`` regularise in-loop with the jittable Laplacian fraction
instead, and call this afterwards if a guard is wanted.

Limitation: the narrow phase detects transversal (edge-pierces-face) crossings
-- the self-intersections a deformed surface actually produces -- but not
exactly-coplanar overlaps (a documented, rare case).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple, cast

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int
from numpy.typing import NDArray

from ..sparse import Mesh, mesh_laplacian_smooth

__all__ = [
    'find_self_intersections',
    'remove_self_intersections',
]

_EPS = 1e-9


def _candidate_pairs(tri: NDArray[Any]) -> NDArray[Any]:
    """Uniform-grid broad phase: candidate face-pair indices (C, 2).

    Faces are binned into a grid of cells ~ the median edge length; faces whose
    AABBs share a cell are candidates.
    """
    n_faces = tri.shape[0]
    fmin = tri.min(axis=1)
    fmax = tri.max(axis=1)
    edge = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    cell = max(float(np.median(edge)), 1e-6)
    lo = np.floor(fmin / cell).astype(np.int64)
    hi = np.floor(fmax / cell).astype(np.int64)

    cellmap: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for fi in range(n_faces):
        for ix in range(lo[fi, 0], hi[fi, 0] + 1):
            for iy in range(lo[fi, 1], hi[fi, 1] + 1):
                for iz in range(lo[fi, 2], hi[fi, 2] + 1):
                    cellmap[(ix, iy, iz)].append(fi)

    pairs: set[Tuple[int, int]] = set()
    for members in cellmap.values():
        m = len(members)
        if m < 2:
            continue
        for i in range(m):
            for j in range(i + 1, m):
                pairs.add((members[i], members[j]))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(sorted(pairs), dtype=np.int64)


def _seg_tri_hit(
    p0: NDArray[Any],
    p1: NDArray[Any],
    a: NDArray[Any],
    b: NDArray[Any],
    c: NDArray[Any],
) -> NDArray[Any]:
    """Vectorised segment(p0->p1)-triangle(a,b,c) intersection (Moller-Trumbore)."""
    e1, e2 = b - a, c - a
    d = p1 - p0
    pv = np.cross(d, e2)
    det = (e1 * pv).sum(-1)
    parallel = np.abs(det) < _EPS
    inv = 1.0 / np.where(parallel, 1.0, det)
    tv = p0 - a
    u = (tv * pv).sum(-1) * inv
    qv = np.cross(tv, e1)
    v = (d * qv).sum(-1) * inv
    t = (e2 * qv).sum(-1) * inv
    return cast(
        NDArray[Any],
        (~parallel)
        & (u >= -_EPS)
        & (v >= -_EPS)
        & (u + v <= 1.0 + _EPS)
        & (t >= -_EPS)
        & (t <= 1.0 + _EPS),
    )


def _pairs_intersect(
    verts: NDArray[Any], faces: NDArray[Any], pairs: NDArray[Any]
) -> NDArray[Any]:
    """Boolean (C,): does each candidate face pair genuinely cross?"""
    t1 = verts[faces[pairs[:, 0]]]  # (C, 3, 3)
    t2 = verts[faces[pairs[:, 1]]]
    a1, b1, c1 = t1[:, 0], t1[:, 1], t1[:, 2]
    a2, b2, c2 = t2[:, 0], t2[:, 1], t2[:, 2]
    hit = np.zeros(pairs.shape[0], dtype=bool)
    for p0, p1 in ((a1, b1), (b1, c1), (c1, a1)):
        hit |= _seg_tri_hit(p0, p1, a2, b2, c2)
    for p0, p1 in ((a2, b2), (b2, c2), (c2, a2)):
        hit |= _seg_tri_hit(p0, p1, a1, b1, c1)
    return hit


def find_self_intersections(mesh: Mesh) -> Int[Array, 'n_pairs 2']:
    """Find genuinely intersecting (non-adjacent) triangle pairs.

    Uniform-grid broad phase + segment-triangle narrow phase.  Face pairs that
    share a vertex (adjacent faces, which always meet at the shared feature)
    are excluded -- only true self-intersections are reported.

    Parameters
    ----------
    mesh
        Triangle mesh.

    Returns
    -------
    ``(n_pairs, 2)`` integer array of intersecting face-index pairs
    (``i < j``); empty if the mesh is intersection-free.  Host-side; not
    differentiable.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    tri = verts[faces]  # (F, 3, 3)
    pairs = _candidate_pairs(tri)
    if pairs.shape[0] == 0:
        return jnp.zeros((0, 2), dtype=jnp.int32)
    # Drop pairs that share a vertex (adjacent faces are not self-intersections).
    f1 = faces[pairs[:, 0]]
    f2 = faces[pairs[:, 1]]
    shares = (f1[:, :, None] == f2[:, None, :]).any(axis=(1, 2))
    pairs = pairs[~shares]
    if pairs.shape[0] == 0:
        return jnp.zeros((0, 2), dtype=jnp.int32)
    hit = _pairs_intersect(verts, faces, pairs)
    return jnp.asarray(pairs[hit].astype(np.int32))


def remove_self_intersections(
    mesh: Mesh,
    *,
    n_iterations: int = 10,
    lam: float = 0.5,
) -> Mesh:
    """Reduce self-intersections by Laplacian-relaxing the offending vertices.

    Each iteration detects intersecting face pairs and applies one Laplacian
    smoothing pass *restricted to the vertices of those faces*, until the mesh
    is intersection-free or ``n_iterations`` is reached.  The topology
    (``faces``) is preserved.

    Parameters
    ----------
    mesh
        Triangle mesh.
    n_iterations
        Maximum relaxation passes.
    lam
        Laplacian smoothing step for the offending vertices.

    Returns
    -------
    ``Mesh`` with the same ``faces`` and relaxed ``vertices``.  Host-side
    (the detection loop is host-side); not differentiable.
    """
    verts = mesh.vertices
    faces = mesh.faces
    faces_np = np.asarray(faces)
    n_vertices = mesh.n_vertices
    for _ in range(n_iterations):
        pairs = np.asarray(find_self_intersections(Mesh(verts, faces)))
        if pairs.shape[0] == 0:
            break
        bad_faces = np.unique(pairs.reshape(-1))
        bad_verts = np.unique(faces_np[bad_faces].reshape(-1))
        smoothed = mesh_laplacian_smooth(verts, faces, lam=lam, iterations=1)
        mask = np.zeros(n_vertices, dtype=bool)
        mask[bad_verts] = True
        verts = jnp.where(jnp.asarray(mask)[:, None], smoothed, verts)
    return Mesh(vertices=verts, faces=faces)
