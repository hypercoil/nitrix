# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Triangle-mesh sparse adjacency constructors.

Supports two mesh families with the same ``Mesh`` representation:

- **Arbitrary triangulations**: ``Mesh(vertices, faces)`` from
  user-supplied tensors.  The k-ring adjacency, edge incidence,
  and cotangent Laplacian are all derived combinatorially from
  ``faces``.
- **Icosphere**: ``icosphere(n_iterations)`` builds the canonical
  spherical mesh via recursive subdivision of the icosahedron.
  Vertex count ``10 * 4^n + 2``: ``n=0`` -> 12 (icosahedron),
  ``n=7`` -> 163842 (FreeSurfer's "ico7").

Operators returned as ``ELL`` for ``semiring_ell_matmul``
compatibility.

The construction is host-side NumPy -- the BFS for k-ring
adjacency and the icosphere midpoint table are not natural in
pure JAX (they need dynamic-shape intermediates).  The resulting
``ELL`` is plain JAX from there.

Use cases:

- **Cortical surface analysis**: FreeSurfer-style icospheres for
  cross-subject registration.  ``mesh_k_ring_adjacency`` gives
  the neighbourhood structure for smoothing / atlas alignment.
- **Surface-based convolutional networks**: the k-ring adjacency
  is the natural support for ``spherical_conv`` (already in
  ``nitrix.geometry.sphere``); this module gives the underlying
  ELL constructor.
- **Cotangent Laplacian**: the discrete Laplace-Beltrami operator
  for surface processing.  Used in spectral surface analysis,
  smoothing, harmonic-coordinate maps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from .ell import ELL


__all__ = [
    'Mesh',
    'icosphere',
    'mesh_k_ring_adjacency',
    'mesh_cotangent_laplacian',
    'mesh_pool_max',
    'mesh_unpool_max',
    'mesh_bary_upsample',
]


@dataclass(frozen=True)
class Mesh:
    '''Triangle mesh: vertex coordinates plus face indices.

    Attributes
    ----------
    vertices : (n_vertices, 3) array
        Per-vertex 3-D coordinates.  Float, on any device.
    faces : (n_faces, 3) array
        Per-face vertex indices.  ``faces[f] = [i, j, k]`` means
        face ``f`` connects vertices ``i``, ``j``, ``k``.  Integer.
    '''

    vertices: Float[Array, 'n_vertices 3']
    faces: Int[Array, 'n_faces 3']

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])


# ---------------------------------------------------------------------------
# Icosphere construction (host-side NumPy)
# ---------------------------------------------------------------------------


def _icosahedron() -> Tuple[np.ndarray, np.ndarray]:
    '''Base icosahedron: 12 vertices, 20 faces.

    Coordinates from the standard ``(±1, ±φ, 0)``-permutation
    construction with ``φ = (1 + √5) / 2``.  All vertices lie on
    the unit sphere after normalisation.
    '''
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts_unnorm = np.array([
        [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1],
    ], dtype=np.float64)
    verts = verts_unnorm / np.linalg.norm(verts_unnorm, axis=-1, keepdims=True)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)
    return verts, faces


def _subdivide(
    verts: np.ndarray,
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    '''One step of icosphere subdivision.

    Each triangle is split into 4 smaller triangles by inserting
    midpoints on each edge and projecting back to the unit sphere.
    Vertex / face counts grow by 4x faces and approximately 4x
    vertices (the exact growth is ``new_v = old_v + new_edges``).
    '''
    midpoint_cache: dict[Tuple[int, int], int] = {}
    new_verts = verts.tolist()

    def get_midpoint(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key in midpoint_cache:
            return midpoint_cache[key]
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        idx = len(new_verts)
        new_verts.append(mid.tolist())
        midpoint_cache[key] = idx
        return idx

    new_faces = []
    for tri in faces:
        a, b, c = tri
        ab = get_midpoint(a, b)
        bc = get_midpoint(b, c)
        ca = get_midpoint(c, a)
        new_faces.append([a, ab, ca])
        new_faces.append([b, bc, ab])
        new_faces.append([c, ca, bc])
        new_faces.append([ab, bc, ca])
    return np.asarray(new_verts, dtype=np.float64), np.asarray(new_faces, dtype=np.int64)


def icosphere(n_iterations: int = 0) -> Mesh:
    '''Construct an icosphere via recursive subdivision.

    Parameters
    ----------
    n_iterations
        Subdivision count.  ``0`` -> 12 vertices (base icosahedron).
        ``n`` -> ``10 * 4^n + 2`` vertices:
        ``n=1`` -> 42, ``n=2`` -> 162, ..., ``n=7`` -> 163842
        (matches FreeSurfer's ``ico7``).

    Returns
    -------
    ``Mesh`` with unit-sphere vertices and oriented faces.
    '''
    if n_iterations < 0:
        raise ValueError(f'n_iterations must be >= 0; got {n_iterations}.')
    verts, faces = _icosahedron()
    for _ in range(n_iterations):
        verts, faces = _subdivide(verts, faces)
    return Mesh(
        vertices=jnp.asarray(verts, dtype=jnp.float32),
        faces=jnp.asarray(faces, dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# Adjacency / k-ring
# ---------------------------------------------------------------------------


def _edges_from_faces(faces_np: np.ndarray) -> np.ndarray:
    '''Return the unique undirected edge list from a face list.

    ``edges[e] = [i, j]`` with ``i < j``; one row per undirected edge.
    '''
    e = np.concatenate([
        faces_np[:, [0, 1]],
        faces_np[:, [1, 2]],
        faces_np[:, [2, 0]],
    ], axis=0)
    e = np.sort(e, axis=-1)
    return np.unique(e, axis=0)


def _onering_adj_list(
    faces_np: np.ndarray, n_vertices: int,
) -> list[list[int]]:
    '''Per-vertex 1-ring neighbour lists (NumPy/host construction).'''
    edges = _edges_from_faces(faces_np)
    adj = [[] for _ in range(n_vertices)]
    for i, j in edges:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    return adj


def _kring_adj_list(
    onering: list[list[int]], k: int,
) -> list[list[int]]:
    '''BFS expansion of 1-ring adjacency to k-ring.'''
    n = len(onering)
    out = [set([v]) for v in range(n)]  # include self
    frontiers = [set([v]) for v in range(n)]
    for _step in range(k):
        new_frontiers = []
        for v in range(n):
            nf = set()
            for u in frontiers[v]:
                for w in onering[u]:
                    if w not in out[v]:
                        nf.add(w)
                        out[v].add(w)
            new_frontiers.append(nf)
        frontiers = new_frontiers
    # Drop self from neighbour list (the matvec already handles
    # self via the centre weight; for adjacency we want strict
    # neighbours).  Return sorted lists for determinism.
    return [sorted(out[v] - {v}) for v in range(n)]


def mesh_k_ring_adjacency(
    mesh: Mesh,
    *,
    k: int = 1,
    binary: bool = True,
    include_self: bool = False,
) -> ELL:
    '''Build the k-ring adjacency of a mesh as an ELL.

    Parameters
    ----------
    mesh
        Triangle mesh.
    k
        Ring depth.  ``k=1`` is the immediate neighbours; ``k=2`` adds
        their neighbours; etc.
    binary
        ``True`` (default) -- adjacency entries are ``1`` for
        neighbours and ``0`` (the ELL identity) for the pad.
        ``False`` -- entries are ``1 / |neighbour_set|``, normalised
        per row so the adjacency is row-stochastic (useful for
        smoothing / random-walk operators).
    include_self
        If ``True``, every vertex's neighbour list includes itself.
        Default ``False`` (strict adjacency).

    Returns
    -------
    ELL of shape ``(n_vertices, n_vertices)`` with ``k_max = max
    neighbour-set size`` (pad rows below k_max use the ELL identity).

    Notes
    -----
    Construction is host-side NumPy (BFS per vertex); the resulting
    ELL is plain JAX.  For repeated k-ring queries on the same
    mesh, build the ELL once and reuse.
    '''
    if k < 1:
        raise ValueError(f'k must be >= 1; got {k}.')
    faces_np = np.asarray(mesh.faces)
    n = mesh.n_vertices
    onering = _onering_adj_list(faces_np, n)
    kring = _kring_adj_list(onering, k)
    if include_self:
        kring = [[v] + lst for v, lst in enumerate(kring)]

    k_max = max((len(lst) for lst in kring), default=1)
    # k_max could be 0 if all vertices are isolated (degenerate mesh).
    k_max = max(k_max, 1)

    indices_np = np.zeros((n, k_max), dtype=np.int32)
    values_np = np.zeros((n, k_max), dtype=np.float32)
    for v, neigh in enumerate(kring):
        if not neigh:
            # Isolated vertex: pad with self-index (in-range for gather).
            indices_np[v, :] = v
            continue
        indices_np[v, : len(neigh)] = neigh
        if len(neigh) < k_max:
            # Pad indices with the first neighbour (in-range, values 0).
            indices_np[v, len(neigh):] = neigh[0]
        if binary:
            values_np[v, : len(neigh)] = 1.0
        else:
            values_np[v, : len(neigh)] = 1.0 / len(neigh)

    return ELL(
        values=jnp.asarray(values_np),
        indices=jnp.asarray(indices_np),
        n_cols=n,
        identity=0.0,
    )


# ---------------------------------------------------------------------------
# Cotangent Laplacian (discrete Laplace-Beltrami)
# ---------------------------------------------------------------------------


def _cotangent_weights(
    vertices_np: np.ndarray, faces_np: np.ndarray,
) -> np.ndarray:
    '''Per-(face, edge) cotangent weight for the discrete L-B operator.

    For a triangle with vertices ``(a, b, c)`` and the edge opposite
    vertex ``c`` (i.e. the edge ``a-b``), the cotangent weight at
    that edge contribution is ``cot(angle at c) / 2``.  The total
    edge weight in the assembled Laplacian is the sum of cot/2 over
    the (one or two) triangles incident to the edge.

    Returns
    -------
    ``(n_faces, 3)`` array where entry ``[f, k]`` is the cotangent
    weight at face ``f`` for the edge **opposite** local vertex
    ``k``.  ``k = 0`` -> edge (b, c); ``k = 1`` -> edge (c, a);
    ``k = 2`` -> edge (a, b).
    '''
    a = vertices_np[faces_np[:, 0]]
    b = vertices_np[faces_np[:, 1]]
    c = vertices_np[faces_np[:, 2]]

    def cot(u, v):
        '''``cot(angle between u and v) = (u.v) / |u x v|``.'''
        dot = np.sum(u * v, axis=-1)
        cross = np.cross(u, v)
        cross_norm = np.linalg.norm(cross, axis=-1)
        cross_norm = np.where(cross_norm > 1e-30, cross_norm, 1e-30)
        return dot / cross_norm

    # Angle at vertex a is the angle between (b - a) and (c - a),
    # so the cot weight at the edge opposite a (= edge b - c) is
    # 0.5 * cot(angle at a).
    cot_a = 0.5 * cot(b - a, c - a)
    cot_b = 0.5 * cot(c - b, a - b)
    cot_c = 0.5 * cot(a - c, b - c)
    # Entry [f, k] = cot weight at edge opposite local vertex k.
    return np.stack([cot_a, cot_b, cot_c], axis=-1)


def mesh_cotangent_laplacian(
    mesh: Mesh,
) -> ELL:
    '''Discrete Laplace-Beltrami operator (cotangent Laplacian) as ELL.

    The standard finite-element / discrete-exterior-calculus
    Laplacian on a triangle mesh:

    ``(L u)[i] = sum_{j ∈ N(i)} w_ij (u[j] - u[i])``

    where ``w_ij = (cot α_ij + cot β_ij) / 2`` with ``α_ij`` and
    ``β_ij`` the angles opposite edge ``(i, j)`` in the two
    triangles sharing that edge (or only one cot for boundary
    edges in an open mesh).

    The diagonal entry is ``L_ii = -sum_j w_ij``, so each row sums
    to zero (the Laplacian sends constants to zero).

    Parameters
    ----------
    mesh
        Triangle mesh.

    Returns
    -------
    ELL of shape ``(n_vertices, n_vertices)``.  ``k_max`` is the
    1-ring max degree plus 1 (the diagonal entry).

    Notes
    -----
    Construction is host-side; the matvec via
    ``semiring_ell_matmul`` is pure JAX.

    Sign convention: returns the **positive-semidefinite** form
    ``-Δ`` (eigenvalues ``≥ 0``).  For the negative-semidefinite
    form (matching the continuous Laplacian sign), negate the
    output's ``values``.
    '''
    verts_np = np.asarray(mesh.vertices, dtype=np.float64)
    faces_np = np.asarray(mesh.faces)
    n = mesh.n_vertices

    cot = _cotangent_weights(verts_np, faces_np)  # (n_faces, 3)

    # Assemble per-edge weights by summing contributions from incident
    # triangles.  edge (i, j) gets cot from face f where (i, j) is the
    # edge opposite local vertex k of f.  Local opposite vertex of
    # edge (i, j) in face [a, b, c]:
    #   if (i, j) = (b, c) or (c, b): k = 0 -> cot_a.
    #   if (i, j) = (c, a) or (a, c): k = 1 -> cot_b.
    #   if (i, j) = (a, b) or (b, a): k = 2 -> cot_c.
    edge_pairs = []   # (i, j, w)
    for k_local, (k_src, k_dst) in enumerate([(1, 2), (2, 0), (0, 1)]):
        for f, tri in enumerate(faces_np):
            i = int(tri[k_src])
            j = int(tri[k_dst])
            w = float(cot[f, k_local])
            edge_pairs.append((i, j, w))
            edge_pairs.append((j, i, w))

    # Aggregate to sparse adjacency: w_ij = sum over (i, j) entries.
    edge_weight: dict[Tuple[int, int], float] = {}
    for i, j, w in edge_pairs:
        key = (i, j)
        edge_weight[key] = edge_weight.get(key, 0.0) + w

    # Per-row neighbour lists and weights.  Include the diagonal as
    # the LAST column for ELL.
    onering_w: list[list[Tuple[int, float]]] = [[] for _ in range(n)]
    for (i, j), w in edge_weight.items():
        onering_w[i].append((j, w))

    # Max degree + 1 for the diagonal.
    k_max = max((len(r) for r in onering_w), default=0) + 1

    indices_np = np.zeros((n, k_max), dtype=np.int32)
    values_np = np.zeros((n, k_max), dtype=np.float64)
    for v in range(n):
        # Diagonal entry FIRST -- predictable layout for downstream
        # consumers that may want to extract it.
        indices_np[v, 0] = v
        if not onering_w[v]:
            indices_np[v, :] = v
            continue
        values_np[v, 0] = sum(w for _, w in onering_w[v])  # diagonal
        for k_idx, (j, w) in enumerate(onering_w[v]):
            indices_np[v, 1 + k_idx] = j
            values_np[v, 1 + k_idx] = -w  # off-diagonal: -w_ij
        # Pad remaining slots with the first neighbour idx, value 0.
        if 1 + len(onering_w[v]) < k_max:
            indices_np[v, 1 + len(onering_w[v]):] = onering_w[v][0][0]

    # Return values matching the mesh.vertices dtype so downstream
    # autodiff doesn't trip on a dtype mismatch.
    out_dtype = mesh.vertices.dtype
    return ELL(
        values=jnp.asarray(values_np.astype(out_dtype)),
        indices=jnp.asarray(indices_np),
        n_cols=n,
        identity=jnp.asarray(0.0, dtype=out_dtype).item(),
    )


# ---------------------------------------------------------------------------
# Cross-level convenience wrappers (compose existing primitives)
# ---------------------------------------------------------------------------
#
# These are documented sugar over ``semiring_ell_matmul``: the math
# is already in the substrate; these wrappers exist for
# discoverability.  Topofit-style mesh hierarchies expose exactly
# three cross-level operations (max-pool, max-unpool, barycentric
# upsample); we ship them as named helpers so the consumer doesn't
# have to know the substrate idiom by heart.


def mesh_pool_max(
    cross_level_adjacency,
    fine_features: Float[Array, '... n_fine d'],
):
    '''Max-pool fine-level features to a coarse level via an ELL adjacency.

    For each coarse vertex ``i``, returns the max of fine-vertex
    features at the positions ``cross_level_adjacency.indices[i, :]``.
    Equivalent to a ``semiring_ell_matmul`` under
    ``TROPICAL_MAX_PLUS`` with the ELL's ``values`` ignored (the
    max-plus reduction degenerates to ``max`` when all values are 0).

    Used in mesh hierarchies (e.g. icosphere-cascaded networks) for
    coarse-to-fine downsampling: ``cross_level_adjacency`` is the
    ELL whose row ``i`` (coarse vertex) lists the fine vertices
    that map to it under the subdivision rule.

    Parameters
    ----------
    cross_level_adjacency
        ``ELL`` of shape ``(n_coarse, n_fine)``; row ``i`` indexes
        the fine vertices belonging to coarse vertex ``i``.  Stored
        ``values`` are ignored (replaced internally by zeros for
        the max-plus identity).
    fine_features
        ``(..., n_fine, d)`` per-vertex fine-level features.

    Returns
    -------
    ``(..., n_coarse, d)`` per-vertex coarse-level features.
    '''
    from ..semiring import TROPICAL_MAX_PLUS, semiring_ell_matmul

    zeros = jnp.zeros_like(cross_level_adjacency.values)
    return semiring_ell_matmul(
        zeros, cross_level_adjacency.indices, fine_features,
        semiring=TROPICAL_MAX_PLUS,
        n_cols=cross_level_adjacency.n_cols,
        backend='jax',
    )


def mesh_unpool_max(
    cross_level_adjacency,
    coarse_features: Float[Array, '... n_coarse d'],
):
    '''Max-unpool coarse-level features back to a fine level.

    Symmetric to ``mesh_pool_max``: the cross-level adjacency is
    interpreted as fine-to-coarse (each row is a fine vertex,
    listing the coarse vertex / vertices it maps from), and the
    max-plus reduction produces a per-fine-vertex output.  In
    practice the cross-level adjacency is usually single-source
    per fine vertex (each fine vertex has one parent coarse
    vertex), in which case "max" reduces to the trivial gather.

    Parameters / Returns mirror ``mesh_pool_max`` with the levels
    swapped.
    '''
    from ..semiring import TROPICAL_MAX_PLUS, semiring_ell_matmul

    zeros = jnp.zeros_like(cross_level_adjacency.values)
    return semiring_ell_matmul(
        zeros, cross_level_adjacency.indices, coarse_features,
        semiring=TROPICAL_MAX_PLUS,
        n_cols=cross_level_adjacency.n_cols,
        backend='jax',
    )


def mesh_bary_upsample(
    bary_ell,
    coarse_coords: Float[Array, '... n_coarse d'],
):
    '''Barycentric upsample from coarse to fine.

    For each fine vertex ``i``::

        out[i, :] = sum_k bary_ell.values[i, k]
                    * coarse_coords[bary_ell.indices[i, k], :]

    Standard ``semiring_ell_matmul`` under ``REAL``; provided as a
    named wrapper so mesh-hierarchy code reads at the right level
    of abstraction.

    The ``bary_ell`` is the "barycentric upsampler" structure:
    row ``i`` (fine vertex) lists the ``k`` coarse vertices that
    define the barycentric coordinates of ``i``, with
    ``bary_ell.values[i, :]`` storing the per-source weight.

    Parameters
    ----------
    bary_ell
        ELL whose row ``i`` (fine vertex) carries the barycentric
        weights and source indices into the coarse level.
    coarse_coords
        ``(..., n_coarse, d)`` per-vertex coarse-level coordinates
        (or any per-vertex feature).

    Returns
    -------
    ``(..., n_fine, d)`` upsampled values.
    '''
    from ..semiring import REAL, semiring_ell_matmul

    return semiring_ell_matmul(
        bary_ell.values, bary_ell.indices, coarse_coords,
        semiring=REAL, n_cols=bary_ell.n_cols, backend='jax',
    )
