# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Point-to-triangle distance (host-side, vectorised).

The shared clean-room primitive behind ``isosurface.mesh_to_sdf`` (the
unsigned distance for a signed-distance volume) and
``surface.cortical_thickness`` (symmetric nearest-surface distance).  Exact
closest-point-on-triangle (Ericson, the seven Voronoi regions), vectorised over
``(query, face)`` and chunked over queries to bound memory.  ``scipy.spatial``
is a banned runtime dependency, so the broad phase is brute force (O(n.F)) --
fine for moderate sizes; a uniform-grid / BVH broad phase is a future
acceleration.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ..sparse import Mesh

__all__ = ['nearest_surface_distance']


def _closest_point_dist2(
    p: NDArray[Any], a: NDArray[Any], b: NDArray[Any], c: NDArray[Any]
) -> NDArray[Any]:
    """Squared distance from points ``p`` (m,1,3) to triangles (a,b,c) (1,F,3).

    Vectorised Ericson closest-point-on-triangle (the seven Voronoi regions:
    the interior, three edges, three vertices).
    """
    ab, ac = b - a, c - a
    ap = p - a
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    bp = p - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    cp = p - c
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    def _safe(x: NDArray[Any]) -> NDArray[Any]:
        return np.where(np.abs(x) < 1e-30, 1e-30, x)

    denom = _safe(va + vb + vc)
    v = (vb / denom)[..., None]
    w = (vc / denom)[..., None]
    c_int = a + v * ab + w * ac
    c_ab = a + (d1 / _safe(d1 - d3))[..., None] * ab
    c_ac = a + (d2 / _safe(d2 - d6))[..., None] * ac
    c_bc = b + ((d4 - d3) / _safe((d4 - d3) + (d5 - d6)))[..., None] * (c - b)
    a_b = np.broadcast_to(a, c_int.shape)
    b_b = np.broadcast_to(b, c_int.shape)
    c_b = np.broadcast_to(c, c_int.shape)
    conds = [
        ((d1 <= 0) & (d2 <= 0))[..., None],
        ((d3 >= 0) & (d4 <= d3))[..., None],
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None],
        ((d6 >= 0) & (d5 <= d6))[..., None],
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None],
        ((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0))[..., None],
    ]
    closest = np.select(
        conds, [a_b, b_b, c_ab, c_b, c_ac, c_bc], default=c_int
    )
    diff = p - closest
    return cast(NDArray[Any], (diff * diff).sum(-1))


def _brute_nearest_dist2(
    q: NDArray[Any], tri: NDArray[Any], chunk_target: int
) -> NDArray[Any]:
    """Exact squared nearest distance by testing every triangle (chunked)."""
    a = tri[:, 0][None]  # (1, F, 3)
    b = tri[:, 1][None]
    c = tri[:, 2][None]
    n_q = q.shape[0]
    n_faces = tri.shape[0]
    out = np.empty(n_q, dtype=np.float64)
    chunk = max(1, chunk_target // max(n_faces, 1))
    for start in range(0, n_q, chunk):
        p = q[start : start + chunk][:, None, :]
        out[start : start + chunk] = _closest_point_dist2(p, a, b, c).min(
            axis=1
        )
    return out


def _shell_offsets(r: int) -> NDArray[Any]:
    """Integer cell offsets at Chebyshev (L-infinity) distance exactly ``r``."""
    if r == 0:
        return np.zeros((1, 3), dtype=np.int64)
    rng = np.arange(-r, r + 1)
    gx, gy, gz = np.meshgrid(rng, rng, rng, indexing='ij')
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    return cast(NDArray[Any], pts[np.abs(pts).max(axis=1) == r].astype(np.int64))


def _grid_nearest_dist2(
    q: NDArray[Any], tri: NDArray[Any], cell: float, r_max: int
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Squared nearest distance via a uniform grid + expanding-shell search.

    Returns ``(d2, finalised)``.  A query is **finalised** (its ``d2`` is the
    *exact* nearest squared distance) once the running minimum is ``<= (R*cell)^2``
    after searching all cells within Chebyshev radius ``R`` -- because any
    triangle binned only into cells at Chebyshev ``> R`` has every point at
    ``L-infinity`` (hence Euclidean) distance ``>= R*cell`` from the query.
    Queries not finalised within ``r_max`` are left for the brute fallback.
    """
    n = q.shape[0]
    n_faces = tri.shape[0]
    fmin = tri.min(axis=1)
    fmax = tri.max(axis=1)
    origin = fmin.min(axis=0)
    # Triangle -> cell memberships (every cell its AABB overlaps).
    lo = np.floor((fmin - origin) / cell).astype(np.int64)
    hi = np.floor((fmax - origin) / cell).astype(np.int64)
    dims = hi - lo + 1
    counts = dims.prod(axis=1)
    total = int(counts.sum())
    fid = np.repeat(np.arange(n_faces, dtype=np.int64), counts)
    cstart = np.zeros(n_faces + 1, dtype=np.int64)
    np.cumsum(counts, out=cstart[1:])
    loc = np.arange(total, dtype=np.int64) - cstart[fid]
    dpm = dims[fid]
    tcz = lo[fid, 2] + loc % dpm[:, 2]
    tcy = lo[fid, 1] + (loc // dpm[:, 2]) % dpm[:, 1]
    tcx = lo[fid, 0] + loc // (dpm[:, 2] * dpm[:, 1])

    qc = np.floor((q - origin) / cell).astype(np.int64)  # query cells
    # Injective cell encoding over the union of triangle cells and the queries'
    # +/- r_max neighbourhoods.
    gmin = np.array(
        [
            min(int(tcx.min()), int(qc[:, 0].min()) - r_max),
            min(int(tcy.min()), int(qc[:, 1].min()) - r_max),
            min(int(tcz.min()), int(qc[:, 2].min()) - r_max),
        ],
        dtype=np.int64,
    )
    gmax = np.array(
        [
            max(int(tcx.max()), int(qc[:, 0].max()) + r_max),
            max(int(tcy.max()), int(qc[:, 1].max()) + r_max),
            max(int(tcz.max()), int(qc[:, 2].max()) + r_max),
        ],
        dtype=np.int64,
    )
    ny = int(gmax[1] - gmin[1] + 1)
    nz = int(gmax[2] - gmin[2] + 1)

    def enc(cx: NDArray[Any], cy: NDArray[Any], cz: NDArray[Any]) -> NDArray[Any]:
        out: NDArray[Any] = ((cx - gmin[0]) * ny + (cy - gmin[1])) * nz + (
            cz - gmin[2]
        )
        return out

    tkey = enc(tcx, tcy, tcz)
    order = np.argsort(tkey, kind='stable')
    fid_s = fid[order]
    ukey, ustart, ucount = np.unique(
        tkey[order], return_index=True, return_counts=True
    )

    d2 = np.full(n, np.inf, dtype=np.float64)
    done = np.zeros(n, dtype=bool)
    for r in range(r_max + 1):
        live = ~done
        if not live.any():
            break
        qidx = np.where(live)[0]
        qcl = qc[qidx]
        for off in _shell_offsets(r):
            keys = enc(qcl[:, 0] + off[0], qcl[:, 1] + off[1], qcl[:, 2] + off[2])
            pos = np.searchsorted(ukey, keys)
            hit = (pos < ukey.shape[0]) & (ukey[np.minimum(pos, ukey.shape[0] - 1)] == keys)
            if not hit.any():
                continue
            hq = qidx[hit]
            hpos = pos[hit]
            cnt = ucount[hpos]
            tot = int(cnt.sum())
            if tot == 0:
                continue
            qrep = np.repeat(hq, cnt)
            seg = np.repeat(ustart[hpos], cnt) + (
                np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            )
            fc = fid_s[seg]
            dd = _closest_point_dist2(
                q[qrep][:, None, :],
                tri[fc, 0][:, None, :],
                tri[fc, 1][:, None, :],
                tri[fc, 2][:, None, :],
            )[:, 0]
            np.minimum.at(d2, qrep, dd)
        done |= d2 <= (r * cell) ** 2
    return d2, done


def nearest_surface_distance(
    query: NDArray[Any],
    mesh: Mesh,
    *,
    method: str = 'auto',
    cell: float | None = None,
    r_max: int = 8,
    chunk_target: int = 1_000_000,
) -> NDArray[Any]:
    """Per-query nearest (unsigned) distance to a triangle mesh's surface.

    Parameters
    ----------
    query
        ``(n, 3)`` query points.
    mesh
        Triangle mesh to measure distance to.
    method
        ``'auto'`` (default; uniform-grid broad phase when the brute work
        ``n_query * n_faces`` is large enough to amortise, else brute force),
        ``'grid'``, or ``'brute'``.  All three return the **identical exact**
        nearest distance -- the grid path finalises a query only once no
        unsearched cell can hold a closer triangle, and any query not resolved
        within ``r_max`` cells falls back to the exact brute scan (Tier C /
        audit AI-C5).
    cell
        Grid cell size (``'grid'``/``'auto'``).  Defaults to the median edge
        length -- roughly one triangle per cell.
    r_max
        Maximum Chebyshev shell radius searched before a query falls back to
        brute force.  Surface-proximal queries (e.g. cortical thickness)
        finalise within a few cells; far queries (e.g. SDF-grid corners) fall
        back -- still exact, just not accelerated.
    chunk_target
        Approximate ``(chunk * n_faces)`` work per block, to bound memory.

    Returns
    -------
    ``(n,)`` NumPy array of nearest point-to-surface distances (host-side).
    """
    q = np.asarray(query, dtype=np.float64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    tri = verts[faces]  # (F, 3, 3)
    n_q, n_faces = q.shape[0], faces.shape[0]
    if n_q == 0:
        return np.zeros(0, dtype=np.float64)

    use_grid = method == 'grid' or (
        method == 'auto'
        and n_faces >= 64
        and n_q * n_faces > 5_000_000
    )
    if not use_grid or n_faces == 0:
        if method == 'grid' and n_faces == 0:
            return np.full(n_q, np.inf, dtype=np.float64)
        return cast(
            NDArray[Any], np.sqrt(_brute_nearest_dist2(q, tri, chunk_target))
        )

    if cell is None:
        e0 = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
        e1 = np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1)
        e2 = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
        cell = float(np.median(np.concatenate([e0, e1, e2])))
    cell = max(cell, 1e-9)

    d2, done = _grid_nearest_dist2(q, tri, cell, r_max)
    if not done.all():  # exact brute fallback for unfinalised queries
        miss = ~done
        d2[miss] = _brute_nearest_dist2(q[miss], tri, chunk_target)
    return cast(NDArray[Any], np.sqrt(d2))
