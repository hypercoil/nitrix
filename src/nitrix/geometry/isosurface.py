# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Volume <-> surface conversion.

:func:`marching_cubes` extracts the level-set of a scalar volume as a
:class:`Mesh`; :func:`mesh_to_sdf` is the inverse -- it rasterises a triangle
mesh to a signed-distance volume.  Both are host-side NumPy builders
(data-dependent work) co-located here as the field<->mesh pair.

**Engine: marching tetrahedra.**  Each grid cube is split into six tetrahedra
by the Freudenthal-Kuhn decomposition (all sharing the cube's main diagonal),
and the isosurface is extracted per tetrahedron.  Unlike the classic
Lorensen-Cline cube algorithm, the tetrahedral decomposition has **no face /
internal ambiguity**, so the surface is **watertight and manifold by
construction** -- no holes that would fight the genus-0 goal of the
field->mesh route (the alternative, a full asymptotic-decider 33-case cube
table, achieves the same guarantee at much higher implementation risk).
Vertices on shared grid edges are deduplicated, so the mesh is closed.

The output triangle count is data-dependent, so -- like the icosphere
construction -- this is a **host-side NumPy** builder that emits a
:class:`Mesh` of JAX arrays; it is not a fixed-shape JAX kernel and is not
differentiable.

Faces are oriented so per-vertex normals point toward *increasing* scalar value
(outward for a signed-distance field, which is negative inside).
"""

from __future__ import annotations

from typing import Any, List, Tuple, cast

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from numpy.typing import NDArray

from ..sparse import Mesh
from ._triangle_distance import _closest_point_dist2

__all__ = ['marching_cubes', 'mesh_to_sdf']

# Cube corners c=0..7 as (dx, dy, dz), dx=bit0, dy=bit1, dz=bit2.
_CUBE_CORNERS = np.array(
    [(c & 1, (c >> 1) & 1, (c >> 2) & 1) for c in range(8)], dtype=np.int64
)
# Freudenthal-Kuhn split into 6 tetrahedra (cube-corner indices), all sharing
# the main diagonal 0=(0,0,0) -> 7=(1,1,1); crack-free across adjacent cubes.
_TETS = np.array(
    [
        [0, 1, 3, 7],
        [0, 1, 5, 7],
        [0, 2, 3, 7],
        [0, 2, 6, 7],
        [0, 4, 5, 7],
        [0, 4, 6, 7],
    ],
    dtype=np.int64,
)
# The 6 edges of a tetrahedron, as local corner-index pairs.
_TET_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def _edge_index(a: int, b: int) -> int:
    key = (min(a, b), max(a, b))
    return _TET_EDGES.index(key)


def _build_tet_table() -> List[List[Tuple[int, int, int]]]:
    """Build the marching-tetrahedra triangulation lookup table.

    For each of the 16 possible inside/outside configurations of a
    tetrahedron's four corners, this returns the triangles that tessellate the
    isosurface within that tetrahedron.  The configuration is encoded as an
    inside-bitmask ``code`` over the four corners (bit ``i`` set means corner
    ``i`` is inside).  Each triangle is a triple of tetrahedron-edge indices
    (into :data:`_TET_EDGES`) naming the edges the surface crosses.  Winding
    is fixed at runtime, so the table only needs to list the correct crossed
    edges without bowties.

    Returns
    -------
    list of list of tuple of int
        A 16-element list indexed by ``code``.  Each entry is the list of
        triangles for that configuration; an empty list for the fully-inside
        (``code == 15``) and fully-outside (``code == 0``) cases, one triangle
        when exactly one corner is on the minority side, and two triangles
        (a split quad) when the corners are evenly divided.  Every triangle is
        a ``(int, int, int)`` triple of edge indices.
    """
    table: List[List[Tuple[int, int, int]]] = []
    for code in range(16):
        inside = [(code >> i) & 1 for i in range(4)]
        n_in = sum(inside)
        ins = [i for i in range(4) if inside[i]]
        out = [i for i in range(4) if not inside[i]]
        if n_in in (0, 4):
            table.append([])
        elif n_in in (1, 3):
            lone = ins[0] if n_in == 1 else out[0]
            others = [i for i in range(4) if i != lone]
            e = [_edge_index(lone, o) for o in others]
            table.append([(e[0], e[1], e[2])])
        else:  # n_in == 2 -> a quad split into two triangles
            i0, i1 = ins
            o0, o1 = out
            e = [
                _edge_index(i0, o0),
                _edge_index(i0, o1),
                _edge_index(i1, o1),
                _edge_index(i1, o0),
            ]
            table.append([(e[0], e[1], e[2]), (e[0], e[2], e[3])])
    return table


_TET_TABLE = _build_tet_table()


def marching_cubes(
    volume: NDArray[Any],
    *,
    level: float = 0.0,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Mesh:
    """Extract the ``level`` isosurface of a scalar volume as a triangle mesh.

    Parameters
    ----------
    volume
        Scalar field, shape ``(X, Y, Z)``.  For a signed-distance field pass
        the default ``level=0`` (negative inside).
    level
        Isovalue.  A grid corner is "inside" when ``volume < level``.
    spacing
        Per-axis voxel size; vertex coordinates are in physical units
        (``index * spacing``).

    Returns
    -------
    Mesh
        A watertight, manifold :class:`Mesh` with outward-oriented faces
        (normals pointing toward increasing scalar value).  Empty (0 vertices,
        0 faces) if the level set does not intersect the volume.

    Notes
    -----
    Host-side NumPy construction (data-dependent size); not differentiable.
    Marching-tetrahedra engine (see module docstring).
    """
    vol = np.asarray(volume, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(
            f'marching_cubes: volume must be 3-D; got {vol.shape}.'
        )
    nx, ny, nz = vol.shape
    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError(
            f'marching_cubes: each axis must have length >= 2; got {vol.shape}.'
        )
    sx, sy, sz = spacing

    # Per-tet inverse edge matrix, for the exact linear-interpolant gradient
    # (the isosurface normal within a tet).  The tet shape is identical across
    # all cubes, so M^{-1} is a per-tet constant.
    corner_phys = _CUBE_CORNERS.astype(np.float64) * np.array([sx, sy, sz])
    tet_minv = []
    for tet in _TETS:
        p = corner_phys[tet]
        mat = np.stack([p[1] - p[0], p[2] - p[0], p[3] - p[0]], axis=0)
        tet_minv.append(np.linalg.inv(mat))

    val = vol.reshape(-1)  # global index g = i*ny*nz + j*nz + k (C-order)
    # Break exact level-on-corner ties: when the isosurface passes through a
    # grid vertex, crossings collapse onto it and produce a locally
    # non-manifold pinch.  Nudge the level by a tiny amount relative to the
    # value scale so no corner sits on it (the surface shifts by a negligible
    # epsilon).  Only triggered by an exact tie -- generic SDFs are untouched.
    if np.any(val == level):
        level = level + 1e-5 * max(1.0, float(np.max(np.abs(val - level))))
    # Cube base global indices.
    ii, jj, kk = np.meshgrid(
        np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing='ij'
    )
    base_g = (ii * ny * nz + jj * nz + kk).reshape(-1)  # (ncubes,)
    corner_off = (
        _CUBE_CORNERS[:, 0] * ny * nz
        + _CUBE_CORNERS[:, 1] * nz
        + _CUBE_CORNERS[:, 2]
    )  # (8,)
    cube_g = (
        base_g[:, None] + corner_off[None, :]
    )  # (ncubes, 8) global indices

    # Accumulate triangle corner edges as (global_a, global_b) pairs, plus the
    # per-triangle outward direction (the tet gradient).
    v_ga: List[List[NDArray[Any]]] = [
        [],
        [],
        [],
    ]  # per triangle-vertex (0,1,2)
    v_gb: List[List[NDArray[Any]]] = [[], [], []]
    d_list: List[NDArray[Any]] = []

    for ti, tet in enumerate(_TETS):
        g_tet = cube_g[:, tet]  # (ncubes, 4) global corner indices
        v_tet = val[g_tet]  # (ncubes, 4) corner values
        inside = v_tet < level  # (ncubes, 4)
        codes = (
            inside[:, 0] * 1
            + inside[:, 1] * 2
            + inside[:, 2] * 4
            + inside[:, 3] * 8
        )
        minv = tet_minv[ti]
        for code in range(1, 15):
            tris = _TET_TABLE[code]
            if not tris:
                continue
            mask = codes == code
            if not mask.any():
                continue
            sub = g_tet[mask]  # (m, 4)
            sub_v = v_tet[mask]  # (m, 4)
            # Exact gradient of the linear interpolant -> outward normal.
            grad_rhs = np.stack(
                [
                    sub_v[:, 1] - sub_v[:, 0],
                    sub_v[:, 2] - sub_v[:, 0],
                    sub_v[:, 3] - sub_v[:, 0],
                ],
                axis=1,
            )  # (m, 3)
            grad = grad_rhs @ minv.T  # (m, 3) toward increasing value
            for tri in tris:
                for slot, e in enumerate(tri):
                    la, lb = _TET_EDGES[e]
                    v_ga[slot].append(sub[:, la])
                    v_gb[slot].append(sub[:, lb])
                d_list.append(grad)

    if not v_ga[0]:
        empty_v = jnp.zeros((0, 3), dtype=jnp.float32)
        empty_f = jnp.zeros((0, 3), dtype=jnp.int32)
        return Mesh(vertices=empty_v, faces=empty_f)

    # (ntri,) global-corner arrays for each of the 3 triangle vertices.
    ga = np.stack(
        [np.concatenate(v_ga[s]) for s in range(3)], axis=1
    )  # (ntri,3)
    gb = np.stack(
        [np.concatenate(v_gb[s]) for s in range(3)], axis=1
    )  # (ntri,3)
    d = np.concatenate(d_list, axis=0)  # (ntri, 3) outward direction per face
    lo = np.minimum(ga, gb)
    hi = np.maximum(ga, gb)

    # Deduplicate crossing vertices by grid-edge key (lo, hi): a shared edge
    # interpolates to the same point regardless of which tet/cube referenced it.
    # Combined 1-D integer key (lo * N + hi, hi < N) instead of a lexicographic
    # np.unique(axis=0) -- same ordering, far faster on large isosurfaces
    # (Tier C / audit AI-C7).
    n_corners = nx * ny * nz
    key = lo.reshape(-1).astype(np.int64) * np.int64(n_corners) + hi.reshape(
        -1
    ).astype(np.int64)
    ukey, inverse = np.unique(key, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int64)  # (ntri, 3)

    # Physical positions of every grid corner.
    g_all = np.arange(nx * ny * nz)
    gi = g_all // (ny * nz)
    gj = (g_all % (ny * nz)) // nz
    gk = g_all % nz
    pos = np.stack([gi * sx, gj * sy, gk * sz], axis=1).astype(np.float64)

    # Linear edge interpolation at the crossing for each unique edge.
    ea = (ukey // np.int64(n_corners)).astype(np.int64)
    eb = (ukey % np.int64(n_corners)).astype(np.int64)
    va, vb = val[ea], val[eb]
    denom = vb - va
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    t = (level - va) / denom
    verts = pos[ea] + t[:, None] * (pos[eb] - pos[ea])  # (n_unique, 3)

    # Orient each triangle so its normal points toward increasing value: align
    # the face normal with the (exact) tet gradient.
    tri_v = verts[faces]  # (ntri, 3, 3)
    nrm = np.cross(tri_v[:, 1] - tri_v[:, 0], tri_v[:, 2] - tri_v[:, 0])
    flip = (nrm * d).sum(axis=1) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]

    # Drop exactly-degenerate (zero-area) faces -- they arise when a crossing
    # lands on a grid corner whose value equals ``level`` (two crossings
    # coincide), contribute nothing geometrically, and would inject spurious
    # huge cotangent weights downstream.  Then drop now-unreferenced vertices
    # and reindex so the Euler characteristic counts correctly.
    keep = 0.5 * np.linalg.norm(nrm, axis=1) > 1e-9
    faces = faces[keep]
    used = np.unique(faces)
    remap = np.zeros(verts.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    verts = verts[used]
    faces = remap[faces]

    return Mesh(
        vertices=jnp.asarray(verts, dtype=jnp.float32),
        faces=jnp.asarray(faces, dtype=jnp.int32),
    )


def _solid_angle(
    p: NDArray[Any], a: NDArray[Any], b: NDArray[Any], c: NDArray[Any]
) -> NDArray[Any]:
    """Signed solid angle subtended by triangles at a set of query points.

    Evaluates the Van Oosterom-Strackee formula for the signed solid angle
    that each triangle ``(a, b, c)`` subtends at each point ``p``.  Summed over
    all faces of a mesh and divided by :math:`4\\pi`, this yields the
    generalised winding number, which is used to determine inside/outside sign.

    Parameters
    ----------
    p : NDArray
        Query points, shape ``(m, 1, 3)`` (broadcast against the faces).
    a, b, c : NDArray
        The three vertices of each triangle, shape ``(1, F, 3)``.

    Returns
    -------
    NDArray
        Signed solid angles, shape ``(m, F)``, one per (point, triangle) pair.
    """
    pa, pb, pc = a - p, b - p, c - p
    la = np.linalg.norm(pa, axis=-1)
    lb = np.linalg.norm(pb, axis=-1)
    lc = np.linalg.norm(pc, axis=-1)
    num = (pa * np.cross(pb, pc)).sum(-1)
    den = (
        la * lb * lc
        + (pa * pb).sum(-1) * lc
        + (pb * pc).sum(-1) * la
        + (pc * pa).sum(-1) * lb
    )
    return cast(NDArray[Any], 2.0 * np.arctan2(num, den))


def mesh_to_sdf(
    mesh: Mesh,
    shape: Tuple[int, int, int],
    *,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Float[Array, 'x y z']:
    """Rasterise a triangle mesh to a signed-distance volume (negative inside).

    The inverse of :func:`marching_cubes`: the training-target generator behind
    learned level-set surface models and a field<->mesh round-trip validator.

    Unsigned distance is the exact point-to-triangle distance (Ericson's
    closest-point, the seven Voronoi regions), minimised over faces.  The sign
    is from the **generalised winding number** (the summed solid angle divided
    by :math:`4\\pi`): robust on a watertight mesh regardless of global face
    orientation (:math:`|w| \\approx 1` inside, :math:`\\approx 0` outside).

    Cost is ``O(n_voxels * n_faces)`` (host-side NumPy, chunked over voxels to
    bound memory) -- practical for moderate volumes / meshes; a spatial-hash
    broad phase is a future acceleration.  Not differentiable.

    Execution class: host-by-implementation.  The output shape is the static
    ``shape`` argument -- there is no data-dependent shape -- so, unlike
    :func:`marching_cubes`, a device (``jnp``) port is feasible; it currently
    runs host-side and is not jittable through ``mesh``.

    Parameters
    ----------
    mesh
        Triangle mesh (watertight, for a meaningful sign).
    shape
        Output volume shape ``(X, Y, Z)``.
    spacing
        Per-axis voxel size; grid point ``(i, j, k)`` is at
        ``(i, j, k) * spacing`` in the mesh's coordinate frame.

    Returns
    -------
    ``(X, Y, Z)`` signed-distance volume (negative inside), as a JAX array.

    Raises
    ------
    ValueError
        If ``shape`` is not 3-D or the mesh has no faces.
    """
    if len(shape) != 3:
        raise ValueError(f'mesh_to_sdf: shape must be 3-D; got {shape}.')
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if faces.shape[0] == 0:
        raise ValueError('mesh_to_sdf: mesh has no faces.')
    a = verts[faces[:, 0]][None]  # (1, F, 3)
    b = verts[faces[:, 1]][None]
    c = verts[faces[:, 2]][None]

    nx, ny, nz = shape
    sx, sy, sz = spacing
    gi, gj, gk = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij'
    )
    pts = np.stack(
        [gi.reshape(-1) * sx, gj.reshape(-1) * sy, gk.reshape(-1) * sz], axis=1
    )  # (N, 3)

    n_pts = pts.shape[0]
    n_faces = faces.shape[0]
    out = np.empty(n_pts, dtype=np.float64)
    chunk = max(1, 1_000_000 // n_faces)  # bound (chunk, F, 3) memory
    four_pi = 4.0 * np.pi
    for start in range(0, n_pts, chunk):
        p = pts[start : start + chunk][:, None, :]  # (m, 1, 3)
        dist = np.sqrt(_closest_point_dist2(p, a, b, c).min(axis=1))  # (m,)
        winding = _solid_angle(p, a, b, c).sum(axis=1) / four_pi  # (m,)
        inside = np.abs(winding) > 0.5
        out[start : start + chunk] = np.where(inside, -dist, dist)

    return jnp.asarray(out.reshape(nx, ny, nz), dtype=jnp.float32)
