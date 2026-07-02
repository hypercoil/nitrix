# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Triangle-mesh self-intersection detection and removal.

A **host-side QA / cleanup** tool, not a differentiable kernel: a uniform-grid
broad phase finds candidate face pairs, a segment-triangle (Moller-Trumbore)
narrow phase confirms genuine crossings, and :func:`remove_self_intersections`
relaxes the offending vertices with Laplacian smoothing.

**Use as a post-hoc cleanup, never inside a jitted loop.**  The broad/narrow
phase is host-side NumPy and cannot run under ``lax.fori_loop``; surface movers
like :func:`deform_to_sdf` regularise in-loop with the jittable Laplacian
fraction instead, and call this afterwards if a guard is wanted.

Limitation: the narrow phase detects transversal (edge-pierces-face) crossings
-- the self-intersections a deformed surface actually produces -- but not
exactly-coplanar overlaps (a documented, rare case).
"""

from __future__ import annotations

from typing import Any, cast

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
    """Uniform-grid broad phase over triangles, yielding candidate face pairs.

    Faces are binned into a uniform grid whose cell size is roughly the median
    triangle edge length; two faces whose axis-aligned bounding boxes share at
    least one grid cell form a candidate pair for the narrow phase.  The
    per-(face, cell) membership enumeration is fully vectorised over NumPy (no
    quad-nested Python loop); only the within-cell pairing loops over occupied
    multi-face cells (far fewer than the number of faces), using a vectorised
    ``np.triu_indices`` per cell.

    Parameters
    ----------
    tri : NDArray, shape (F, 3, 3)
        Per-face triangle vertex coordinates: ``F`` faces, three vertices each,
        three spatial coordinates per vertex.

    Returns
    -------
    NDArray, shape (C, 2)
        Integer array of ``C`` candidate face-index pairs, deduplicated and
        ordered so that ``i < j`` in each row.  Empty (shape ``(0, 2)``) when
        fewer than two faces are supplied or no boxes overlap.
    """
    n_faces = tri.shape[0]
    if n_faces < 2:
        return np.zeros((0, 2), dtype=np.int64)
    fmin = tri.min(axis=1)
    fmax = tri.max(axis=1)
    edge = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    cell = max(float(np.median(edge)), 1e-6)
    lo = np.floor(fmin / cell).astype(np.int64)
    hi = np.floor(fmax / cell).astype(np.int64)

    # Enumerate (face, cell) memberships: face fi spans dims[fi] cells.
    dims = hi - lo + 1  # (n_faces, 3)
    counts = dims.prod(axis=1)  # cells per face
    total = int(counts.sum())
    face_id = np.repeat(np.arange(n_faces, dtype=np.int64), counts)
    start = np.zeros(n_faces + 1, dtype=np.int64)
    np.cumsum(counts, out=start[1:])
    local = (
        np.arange(total, dtype=np.int64) - start[face_id]
    )  # idx within face
    dpm = dims[face_id]
    dz = local % dpm[:, 2]
    dy = (local // dpm[:, 2]) % dpm[:, 1]
    dx = local // (dpm[:, 2] * dpm[:, 1])
    cx = lo[face_id, 0] + dx
    cy = lo[face_id, 1] + dy
    cz = lo[face_id, 2] + dz
    # Combined non-negative cell key.
    cxm, cym, czm = int(cx.min()), int(cy.min()), int(cz.min())
    ny = int(cy.max()) - cym + 1
    nz = int(cz.max()) - czm + 1
    cellkey = ((cx - cxm) * ny + (cy - cym)) * nz + (cz - czm)

    # Group by cell (faces ascending within a group), pair within each.
    order = np.lexsort((face_id, cellkey))
    ck = cellkey[order]
    fids = face_id[order]
    _, grp_start, grp_cnt = np.unique(
        ck, return_index=True, return_counts=True
    )
    blocks = []
    for gs, gc in zip(grp_start.tolist(), grp_cnt.tolist()):
        if gc < 2:
            continue
        members = fids[gs : gs + gc]  # ascending face ids
        ii, jj = np.triu_indices(gc, k=1)
        blocks.append(np.stack([members[ii], members[jj]], axis=1))
    if not blocks:
        return np.zeros((0, 2), dtype=np.int64)
    allp = np.concatenate(blocks, axis=0)  # (a < b) per row
    # Dedup + sort (a pair may share several cells) via a combined key.
    pkey = np.unique(allp[:, 0] * np.int64(n_faces) + allp[:, 1])
    return np.stack(
        [pkey // np.int64(n_faces), pkey % np.int64(n_faces)], axis=1
    ).astype(np.int64)


def _seg_tri_hit(
    p0: NDArray[Any],
    p1: NDArray[Any],
    a: NDArray[Any],
    b: NDArray[Any],
    c: NDArray[Any],
) -> NDArray[Any]:
    """Vectorised segment-triangle intersection test (Moller-Trumbore).

    Tests, for each entry of the batched inputs, whether the line segment from
    ``p0`` to ``p1`` crosses the triangle with vertices ``a``, ``b``, ``c``.
    The barycentric coordinates and ray parameter are computed via the
    Moller-Trumbore algorithm; segments parallel to the triangle plane
    (vanishing determinant, within :math:`\\varepsilon`) are treated as
    non-intersecting.

    Parameters
    ----------
    p0, p1 : NDArray, shape (..., 3)
        Endpoints of the segment; each intersection is tested along
        :math:`p_0 \\to p_1`.
    a, b, c : NDArray, shape (..., 3)
        The three triangle vertices, broadcast against the segment batch.

    Returns
    -------
    NDArray of bool, shape (...)
        ``True`` where the segment genuinely crosses the triangle interior (or
        boundary, within :math:`\\varepsilon`), ``False`` otherwise.
    """
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
    """Narrow phase: confirm which candidate face pairs genuinely cross.

    For each candidate pair the six directed triangle edges (three from each
    face) are tested against the opposing triangle with :func:`_seg_tri_hit`; a
    pair is reported as crossing if any of the six edge-triangle tests hits.

    Parameters
    ----------
    verts : NDArray, shape (V, 3)
        Vertex coordinates for the ``V`` mesh vertices.
    faces : NDArray, shape (F, 3)
        Triangle connectivity: three vertex indices per face.
    pairs : NDArray, shape (C, 2)
        Candidate face-index pairs from the broad phase.

    Returns
    -------
    NDArray of bool, shape (C,)
        ``True`` for each candidate pair whose triangles transversally
        intersect, ``False`` otherwise.
    """
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
    mesh : Mesh
        Triangle mesh to inspect.

    Returns
    -------
    Int[Array, 'n_pairs 2']
        Integer array of intersecting face-index pairs (``i < j`` per row);
        empty (shape ``(0, 2)``) if the mesh is intersection-free.  Host-side;
        not differentiable.
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
    mesh : Mesh
        Triangle mesh to clean up.
    n_iterations : int, optional
        Maximum number of relaxation passes.
    lam : float, optional
        Laplacian smoothing step applied to the offending vertices.

    Returns
    -------
    Mesh
        A :class:`Mesh` with the same ``faces`` and relaxed ``vertices``.
        Host-side (the detection loop is host-side); not differentiable.
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
