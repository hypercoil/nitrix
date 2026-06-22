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


def nearest_surface_distance(
    query: NDArray[Any],
    mesh: Mesh,
    *,
    chunk_target: int = 1_000_000,
) -> NDArray[Any]:
    """Per-query nearest (unsigned) distance to a triangle mesh's surface.

    Parameters
    ----------
    query
        ``(n, 3)`` query points.
    mesh
        Triangle mesh to measure distance to.
    chunk_target
        Approximate ``(chunk * n_faces)`` work per block, to bound memory.

    Returns
    -------
    ``(n,)`` NumPy array of nearest point-to-surface distances (host-side).
    """
    q = np.asarray(query, dtype=np.float64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    a = verts[faces[:, 0]][None]  # (1, F, 3)
    b = verts[faces[:, 1]][None]
    c = verts[faces[:, 2]][None]
    n_q = q.shape[0]
    n_faces = faces.shape[0]
    out = np.empty(n_q, dtype=np.float64)
    chunk = max(1, chunk_target // max(n_faces, 1))
    for start in range(0, n_q, chunk):
        p = q[start : start + chunk][:, None, :]
        out[start : start + chunk] = np.sqrt(
            _closest_point_dist2(p, a, b, c).min(axis=1)
        )
    return out
