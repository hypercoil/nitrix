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
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, Num
from numpy.typing import NDArray

from .ell import ELL

if TYPE_CHECKING:
    # Annotation-only; a runtime import would risk a cycle (the matmul
    # itself is imported lazily inside the functions below).
    from ..semiring._types import Semiring


__all__ = [
    'IcosphereHierarchy',
    'Mesh',
    'compute_vertex_normals',
    'icosphere',
    'icosphere_bary_upsampler',
    'icosphere_cross_level_adjacency',
    'icosphere_hierarchy',
    'icosphere_hierarchy_from_levels',
    'mesh_bary_upsample',
    'mesh_coarsen_meanpool',
    'mesh_cotangent_laplacian',
    'mesh_k_ring_adjacency',
    'mesh_laplacian_smooth',
    'mesh_pool_max',
    'mesh_unpool_max',
]


@dataclass(frozen=True)
class Mesh:
    """Triangle mesh: vertex coordinates plus face indices.

    Attributes
    ----------
    vertices : (n_vertices, 3) array
        Per-vertex 3-D coordinates.  Float, on any device.
    faces : (n_faces, 3) array
        Per-face vertex indices.  ``faces[f] = [i, j, k]`` means
        face ``f`` connects vertices ``i``, ``j``, ``k``.  Integer.
    """

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


def _icosahedron() -> Tuple[NDArray[Any], NDArray[Any]]:
    """Base icosahedron: 12 vertices, 20 faces.

    Coordinates from the standard ``(±1, ±φ, 0)``-permutation
    construction with ``φ = (1 + √5) / 2``.  All vertices lie on
    the unit sphere after normalisation.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts_unnorm = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float64,
    )
    verts = verts_unnorm / np.linalg.norm(verts_unnorm, axis=-1, keepdims=True)
    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int64,
    )
    return verts, faces


def _subdivide(
    verts: NDArray[Any],
    faces: NDArray[Any],
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """One step of icosphere subdivision.

    Each triangle is split into 4 smaller triangles by inserting
    midpoints on each edge and projecting back to the unit sphere.
    Vertex / face counts grow by 4x faces and approximately 4x
    vertices (the exact growth is ``new_v = old_v + new_edges``).

    Returns
    -------
    ``(new_verts, new_faces, parents)`` where ``parents`` is an
    ``(n_new_verts, 2)`` ``int64`` array giving the parentage of
    each new vertex relative to the **coarse** vertex set:

    - For a coarse-original vertex ``v`` (preserved in place at the
      same index), ``parents[v] = (v, v)``.
    - For an edge-midpoint vertex with parent coarse edge
      ``(a, b)`` (sorted ``a < b``), ``parents[v] = (a, b)``.
    """
    midpoint_cache: dict[Tuple[int, int], int] = {}
    new_verts = verts.tolist()
    parents: list[Tuple[int, int]] = [(v, v) for v in range(len(verts))]

    def get_midpoint(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key in midpoint_cache:
            return midpoint_cache[key]
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        idx = len(new_verts)
        new_verts.append(mid.tolist())
        midpoint_cache[key] = idx
        parents.append(key)
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
    return (
        np.asarray(new_verts, dtype=np.float64),
        np.asarray(new_faces, dtype=np.int64),
        np.asarray(parents, dtype=np.int64),
    )


def icosphere(n_iterations: int = 0) -> Mesh:
    """Construct an icosphere via recursive subdivision.

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
    """
    if n_iterations < 0:
        raise ValueError(f'n_iterations must be >= 0; got {n_iterations}.')
    verts, faces = _icosahedron()
    for _ in range(n_iterations):
        verts, faces, _ = _subdivide(verts, faces)
    return Mesh(
        vertices=jnp.asarray(verts, dtype=jnp.float32),
        faces=jnp.asarray(faces, dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# Icosphere hierarchy (parent-child bookkeeping across subdivision levels)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IcosphereHierarchy:
    """Sequence of icosphere meshes at subdivision levels ``0..max_level``.

    Built by ``icosphere_hierarchy(max_level)``.  Holds the per-level
    meshes plus the parent-child bookkeeping needed by
    ``icosphere_cross_level_adjacency`` and ``icosphere_bary_upsampler``
    to construct the cross-level ELLs in O(n_vertices) without
    re-running subdivision.

    Attributes
    ----------
    meshes
        Tuple of length ``max_level + 1``: ``meshes[L]`` is the
        icosphere at subdivision level ``L``.  ``meshes[0]`` is the
        base icosahedron (12 vertices, 20 faces).
    parents
        Tuple of length ``max_level + 1``.  ``parents[0] is None``
        (the base level has no parent).  For ``L >= 1``,
        ``parents[L]`` is an ``(n_vertices_at_L, 2)`` ``int64`` array
        giving the parentage of each fine vertex in level ``L``
        relative to the coarse vertex set at level ``L - 1``:

        - For a fine vertex that is a coarse-original (preserved in
          place at the same index), ``parents[L][v] == (v, v)``.
        - For a fine vertex that is a midpoint of the coarse edge
          ``(a, b)`` (sorted ``a < b``), ``parents[L][v] == (a, b)``.

    Notes
    -----
    The hierarchy is host-side (NumPy parent tables + JAX-array
    meshes).  Cross-level ELLs returned from the constructors are
    plain JAX, so the downstream matmul / pool / upsample paths
    stay GPU-native.
    """

    meshes: Tuple[Mesh, ...]
    parents: Tuple[Optional[NDArray[Any]], ...]

    @property
    def max_level(self) -> int:
        return len(self.meshes) - 1

    def __len__(self) -> int:
        return len(self.meshes)


def icosphere_hierarchy(max_level: int) -> IcosphereHierarchy:
    """Construct a sequence of icospheres from level 0 to ``max_level``.

    Parameters
    ----------
    max_level
        Inclusive upper bound on the subdivision level.  The
        hierarchy contains ``max_level + 1`` meshes,
        ``[ico_0, ico_1, ..., ico_max_level]``.  ``max_level = 0``
        returns the base icosahedron alone.

    Returns
    -------
    ``IcosphereHierarchy`` with ``meshes`` and ``parents`` tables
    populated (``parents[0] is None``).

    Notes
    -----
    Subdivision is deterministic (the midpoint cache assigns indices
    by a fixed traversal order over faces), so the cross-level
    helpers below depend only on the stored parent tables.
    """
    if max_level < 0:
        raise ValueError(f'max_level must be >= 0; got {max_level}.')

    meshes: list[Mesh] = []
    parents_per_level: List[Optional[NDArray[Any]]] = [None]

    verts, faces = _icosahedron()
    meshes.append(
        Mesh(
            vertices=jnp.asarray(verts, dtype=jnp.float32),
            faces=jnp.asarray(faces, dtype=jnp.int32),
        )
    )
    for _ in range(max_level):
        verts, faces, parents = _subdivide(verts, faces)
        meshes.append(
            Mesh(
                vertices=jnp.asarray(verts, dtype=jnp.float32),
                faces=jnp.asarray(faces, dtype=jnp.int32),
            )
        )
        parents_per_level.append(parents)

    return IcosphereHierarchy(
        meshes=tuple(meshes),
        parents=tuple(parents_per_level),
    )


def icosphere_hierarchy_from_levels(
    meshes: Sequence[Mesh],
    parents: Sequence[Optional[Int[Array, 'n_fine 2']]],
) -> IcosphereHierarchy:
    """Package caller-supplied per-level meshes + parent tables into a hierarchy.

    Use this when the subdivision hierarchy does **not** come from
    nitrix's own math-canonical ``icosphere`` -- most importantly when
    the per-level meshes are an external icosphere-subdivision family
    with a different vertex ordering and coordinates, e.g. FreeSurfer's
    ``fsaverage{0..6}.sphere``.  Pre-trained surface models (SUGAR,
    SphereMorph, ...) are tied to the *exact* topology they were trained
    on, so inference must run on that topology, not the math icosphere.

    **nitrix does not source the topology.**  Reading FreeSurfer
    ``.sphere`` binaries (``nibabel.freesurfer.read_geometry``),
    resolving ``$SUBJECTS_DIR``, or loading any neuroimaging container
    is outside nitrix's dependency contract (SPEC §5.2: no ``nibabel``,
    no filesystem / container concerns -- those live in ``thrux`` or the
    consuming port).  The caller reads the files and hands nitrix plain
    arrays; this constructor validates them and produces the same
    ``IcosphereHierarchy`` that ``icosphere_hierarchy`` builds.  Every
    cross-level operator -- ``icosphere_cross_level_adjacency``,
    ``icosphere_bary_upsampler``, ``mesh_pool_max``,
    ``mesh_coarsen_meanpool`` -- then works unchanged and with no
    topology-source branching, because they depend only on the parent
    tables and per-level vertex counts, not on the coordinates being
    math-canonical.

    Parameters
    ----------
    meshes
        Per-level ``Mesh`` objects at consecutive subdivision levels,
        coarse first.  ``meshes[0]`` is the base level.
    parents
        Same length as ``meshes``.  ``parents[0]`` must be ``None`` (the
        base level has no parent).  For ``L >= 1``, ``parents[L]`` is an
        ``(n_vertices_at_L, 2)`` integer array giving each fine vertex's
        parentage relative to level ``L - 1``, following the
        ``IcosphereHierarchy.parents`` convention:

        - a coarse-original fine vertex (preserved in place at the same
          index ``v``) has ``parents[L][v] == (v, v)``;
        - a midpoint of coarse edge ``(a, b)`` has
          ``parents[L][v] == (a, b)``.

        For a FreeSurfer hierarchy the consumer derives these triples
        from the precomputed mid-edge tables (e.g. SUGAR's
        ``fsaverage{i}_upsample_neighbors.npz``).

    Returns
    -------
    ``IcosphereHierarchy``.

    Raises
    ------
    ValueError
        If the lengths disagree, ``parents[0]`` is not ``None``, a
        non-base parent table has the wrong shape, its row count does
        not match the corresponding mesh, or any parent index is out of
        range for the coarser level.
    """
    meshes = tuple(meshes)
    parents = tuple(parents)
    if len(meshes) == 0:
        raise ValueError('icosphere_hierarchy_from_levels: meshes is empty.')
    if len(meshes) != len(parents):
        raise ValueError(
            f'icosphere_hierarchy_from_levels: len(meshes)={len(meshes)} '
            f'!= len(parents)={len(parents)}.'
        )
    if parents[0] is not None:
        raise ValueError(
            'icosphere_hierarchy_from_levels: parents[0] must be None '
            '(the base level has no parent).'
        )
    parents_norm: List[Optional[NDArray[Any]]] = [None]
    for L in range(1, len(meshes)):
        p = parents[L]
        if p is None:
            raise ValueError(
                f'icosphere_hierarchy_from_levels: parents[{L}] must not '
                'be None for a non-base level.'
            )
        p_np = np.asarray(p)
        if p_np.ndim != 2 or p_np.shape[1] != 2:
            raise ValueError(
                f'icosphere_hierarchy_from_levels: parents[{L}].shape='
                f'{p_np.shape} must be (n_fine, 2).'
            )
        n_fine = meshes[L].n_vertices
        if p_np.shape[0] != n_fine:
            raise ValueError(
                f'icosphere_hierarchy_from_levels: parents[{L}] has '
                f'{p_np.shape[0]} rows but meshes[{L}] has {n_fine} '
                'vertices.'
            )
        n_coarse = meshes[L - 1].n_vertices
        if p_np.size and (int(p_np.min()) < 0 or int(p_np.max()) >= n_coarse):
            raise ValueError(
                f'icosphere_hierarchy_from_levels: parents[{L}] indices '
                f'must lie in [0, {n_coarse}) (the level-{L - 1} vertex '
                'range).'
            )
        parents_norm.append(p_np.astype(np.int64))

    return IcosphereHierarchy(
        meshes=meshes,
        parents=tuple(parents_norm),
    )


def _validate_consecutive(
    hier: IcosphereHierarchy, coarse: int, fine: int
) -> None:
    if fine != coarse + 1:
        raise ValueError(
            f'cross-level helpers require consecutive levels; got '
            f'coarse={coarse}, fine={fine}.  For multi-level pooling, '
            f'compose mesh_pool_max / mesh_bary_upsample calls.'
        )
    if coarse < 0 or fine > hier.max_level:
        raise ValueError(
            f'level pair ({coarse}, {fine}) out of range for hierarchy '
            f'with max_level={hier.max_level}.'
        )


def icosphere_cross_level_adjacency(
    hierarchy: IcosphereHierarchy,
    coarse_level: int,
    fine_level: int,
) -> ELL:
    """Coarse-to-fine adjacency ELL for icosphere pooling.

    Row ``i`` (coarse vertex at level ``coarse_level``) carries the
    fine vertices at level ``fine_level`` that descend from it under
    the subdivision rule:

    - The coarse-original fine vertex (same index ``i``), and
    - Every fine vertex that is a midpoint of an edge ``(i, j)``
      incident to ``i`` in the coarse mesh.

    The stored ``values`` are a ``1.0 / 0.0`` validity indicator: ``1``
    at real (vertex, child) entries and ``0`` at padding.  This serves
    both downstream consumers without a per-consumer rebuild:

    - ``mesh_pool_max(adj, fine_features)`` ignores the values (it
      substitutes the TROPICAL_MAX_PLUS identity internally), so the
      indicator is harmless there.
    - ``mesh_coarsen_meanpool(adj, fine_features)`` reads the indicator
      directly as the per-edge weight and as the per-row count, so a
      row-mean over the real children falls out for free.

    Parameters
    ----------
    hierarchy
        ``IcosphereHierarchy`` produced by ``icosphere_hierarchy``.
    coarse_level, fine_level
        Must be consecutive: ``fine_level == coarse_level + 1``.
        Multi-level pooling is composed by the consumer.

    Returns
    -------
    ``ELL`` of shape ``(n_coarse, n_fine)`` with
    ``k_max = 1 + max_coarse_degree`` (6 for the base icosahedron,
    7 for every subdivided level).  ``values`` carry a ``1.0 / 0.0``
    validity indicator (real entries / padding); ``identity`` is the
    REAL identity ``0.0``.
    """
    _validate_consecutive(hierarchy, coarse_level, fine_level)
    n_coarse = hierarchy.meshes[coarse_level].n_vertices
    n_fine = hierarchy.meshes[fine_level].n_vertices
    parents = hierarchy.parents[fine_level]  # (n_fine, 2)
    # A fine (non-base) level always has a parent table; only parents[0]
    # is None, and ``_validate_consecutive`` guarantees fine_level >= 1.
    assert parents is not None

    # For each coarse vertex i, accumulate the fine vertices whose
    # parent pair contains i.
    rows: list[list[int]] = [[] for _ in range(n_coarse)]
    for v_fine in range(n_fine):
        a, b = int(parents[v_fine, 0]), int(parents[v_fine, 1])
        if a == b:
            # Coarse-original fine vertex.
            rows[a].append(v_fine)
        else:
            rows[a].append(v_fine)
            rows[b].append(v_fine)

    k_max = max((len(r) for r in rows), default=1)
    k_max = max(k_max, 1)

    indices_np = np.zeros((n_coarse, k_max), dtype=np.int32)
    values_np = np.zeros((n_coarse, k_max), dtype=np.float32)
    for i, row in enumerate(rows):
        if not row:
            indices_np[i, :] = i
            continue
        indices_np[i, : len(row)] = row
        values_np[i, : len(row)] = 1.0  # validity indicator
        if len(row) < k_max:
            indices_np[i, len(row) :] = row[0]

    return ELL(
        values=jnp.asarray(values_np),
        indices=jnp.asarray(indices_np),
        n_cols=n_fine,
        identity=0.0,
    )


def icosphere_bary_upsampler(
    hierarchy: IcosphereHierarchy,
    coarse_level: int,
    fine_level: int,
) -> ELL:
    """Fine-from-coarse barycentric upsampler ELL.

    Row ``i`` (fine vertex at level ``fine_level``) carries the
    coarse-vertex sources at level ``coarse_level`` and the
    barycentric weights:

    - Coarse-original fine vertex ``i``: one source (``i`` itself)
      with weight ``1`` (and a padded second slot, weight ``0``).
    - Midpoint fine vertex of coarse edge ``(a, b)``: two sources
      (``a``, ``b``) with weights ``(0.5, 0.5)``.

    Feeds ``mesh_bary_upsample(bary_ell, coarse_features)`` to
    interpolate per-vertex coordinates or features from the coarse
    level to the fine level.

    Parameters
    ----------
    hierarchy
        ``IcosphereHierarchy`` produced by ``icosphere_hierarchy``.
    coarse_level, fine_level
        Must be consecutive: ``fine_level == coarse_level + 1``.

    Returns
    -------
    ``ELL`` of shape ``(n_fine, n_coarse)`` with ``k_max = 2``.
    """
    _validate_consecutive(hierarchy, coarse_level, fine_level)
    n_coarse = hierarchy.meshes[coarse_level].n_vertices
    n_fine = hierarchy.meshes[fine_level].n_vertices
    parents = hierarchy.parents[fine_level]  # (n_fine, 2)
    # A fine (non-base) level always has a parent table; only parents[0]
    # is None, and ``_validate_consecutive`` guarantees fine_level >= 1.
    assert parents is not None

    indices_np = np.zeros((n_fine, 2), dtype=np.int32)
    values_np = np.zeros((n_fine, 2), dtype=np.float32)
    for v_fine in range(n_fine):
        a, b = int(parents[v_fine, 0]), int(parents[v_fine, 1])
        if a == b:
            # Coincident with coarse vertex; weight 1 on a, pad slot.
            indices_np[v_fine, 0] = a
            indices_np[v_fine, 1] = a
            values_np[v_fine, 0] = 1.0
            values_np[v_fine, 1] = 0.0
        else:
            indices_np[v_fine, 0] = a
            indices_np[v_fine, 1] = b
            values_np[v_fine, 0] = 0.5
            values_np[v_fine, 1] = 0.5

    return ELL(
        values=jnp.asarray(values_np),
        indices=jnp.asarray(indices_np),
        n_cols=n_coarse,
        identity=0.0,
    )


# ---------------------------------------------------------------------------
# Adjacency / k-ring
# ---------------------------------------------------------------------------


def _edges_from_faces(faces_np: NDArray[Any]) -> NDArray[Any]:
    """Return the unique undirected edge list from a face list.

    ``edges[e] = [i, j]`` with ``i < j``; one row per undirected edge.
    """
    e = np.concatenate(
        [
            faces_np[:, [0, 1]],
            faces_np[:, [1, 2]],
            faces_np[:, [2, 0]],
        ],
        axis=0,
    )
    e = np.sort(e, axis=-1)
    return np.unique(e, axis=0)


def _onering_adj_list(
    faces_np: NDArray[Any],
    n_vertices: int,
) -> list[list[int]]:
    """Per-vertex 1-ring neighbour lists (NumPy/host construction)."""
    edges = _edges_from_faces(faces_np)
    adj: List[List[int]] = [[] for _ in range(n_vertices)]
    for i, j in edges:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    return adj


def _kring_adj_list(
    onering: list[list[int]],
    k: int,
) -> list[list[int]]:
    """BFS expansion of 1-ring adjacency to k-ring."""
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
    """Build the k-ring adjacency of a mesh as an ELL.

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
    """
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
            indices_np[v, len(neigh) :] = neigh[0]
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
    vertices_np: NDArray[Any],
    faces_np: NDArray[Any],
) -> NDArray[Any]:
    """Per-(face, edge) cotangent weight for the discrete L-B operator.

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
    """
    a = vertices_np[faces_np[:, 0]]
    b = vertices_np[faces_np[:, 1]]
    c = vertices_np[faces_np[:, 2]]

    def cot(u: NDArray[Any], v: NDArray[Any]) -> NDArray[Any]:
        """``cot(angle between u and v) = (u.v) / |u x v|``."""
        dot = np.sum(u * v, axis=-1)
        cross = np.cross(u, v)
        cross_norm = np.linalg.norm(cross, axis=-1)
        cross_norm = np.where(cross_norm > 1e-30, cross_norm, 1e-30)
        # NumPy reductions resolve to Any here; restore the array type.
        return cast(NDArray[Any], dot / cross_norm)

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
    """Discrete Laplace-Beltrami operator (cotangent Laplacian) as ELL.

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
    """
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
    edge_pairs = []  # (i, j, w)
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
            indices_np[v, 1 + len(onering_w[v]) :] = onering_w[v][0][0]

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


def _apply_shared_ell(
    values: Num[Array, 'm k_max'],
    indices: Int[Array, 'm k_max'],
    B: Num[Array, '... n d'],
    *,
    semiring: 'Semiring[Any]',
    n_cols: int,
) -> Num[Array, '... m d']:
    """Apply a *shared* (un-batched) ELL to possibly-batched features.

    ``semiring_ell_matmul`` vmaps over leading dims that are present on
    *all* of ``values`` / ``indices`` / ``B``.  The cross-level wrappers
    hold a single 2-D ELL (one per (coarse, fine) level pair) and apply
    it to features ``(..., n, d)`` whose leading axes are batch / channel
    groupings.  We vmap the 2-D core over those leading axes so the same
    operator broadcasts across the batch without materialising a batched
    copy of the ELL.
    """
    from ..semiring import semiring_ell_matmul

    def core(b2d: Num[Array, 'n d']) -> Num[Array, 'm d']:
        return semiring_ell_matmul(
            values,
            indices,
            b2d,
            semiring=semiring,
            n_cols=n_cols,
            backend='jax',
        )

    fn: Callable[..., Any] = core
    for _ in range(B.ndim - 2):
        fn = jax.vmap(fn)
    # ``jax.vmap`` erases the return type to Any; restore it.
    return cast(Num[Array, '... m d'], fn(B))


def mesh_pool_max(
    cross_level_adjacency: ELL,
    fine_features: Float[Array, '... n_fine d'],
) -> Float[Array, '... n_coarse d']:
    """Max-pool fine-level features to a coarse level via an ELL adjacency.

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
    """
    from ..semiring import TROPICAL_MAX_PLUS

    zeros = jnp.zeros_like(cross_level_adjacency.values)
    return _apply_shared_ell(
        zeros,
        cross_level_adjacency.indices,
        fine_features,
        semiring=TROPICAL_MAX_PLUS,
        n_cols=cross_level_adjacency.n_cols,
    )


def mesh_coarsen_meanpool(
    coarsen_ell: ELL,
    fine_features: Float[Array, '... n_fine d'],
) -> Float[Array, '... n_coarse d']:
    """Mean-pool fine-level features to a coarse level via an ELL adjacency.

    For each coarse vertex ``i``, returns the ``values``-weighted mean
    of the fine-vertex features it gathers::

        out[i, :] = (sum_p values[i, p] * fine[indices[i, p], :])
                    / (sum_p values[i, p])

    With the ``1.0 / 0.0`` validity indicator that
    ``icosphere_cross_level_adjacency`` stores, this is a plain mean
    over the real children of coarse vertex ``i`` (the padding columns,
    weight ``0``, drop out of both sum and count).  A coarse vertex
    whose row is entirely padding returns zeros rather than ``NaN``.

    This is the mean-pool sibling of ``mesh_pool_max`` (max over the
    same children).  It matches the "scatter-mean with self-loop"
    coarsening used by surface-domain encoders (e.g. SUGAR's
    ``IcosahedronPooling``): the coarse vertex's own previous-level
    feature is included because the coarse-original fine vertex is one
    of the children in the cross-level adjacency row.

    The op is documented sugar over ``semiring_ell_matmul`` under
    ``REAL``: the numerator is one matmul; the denominator is the
    per-row sum of ``values`` (no second matmul needed).

    Parameters
    ----------
    coarsen_ell
        ``ELL`` of shape ``(n_coarse, n_fine)``; row ``i`` (coarse
        vertex) lists the fine vertices belonging to coarse vertex
        ``i``, with ``values`` the per-edge weight (use the ``1.0 /
        0.0`` indicator from ``icosphere_cross_level_adjacency`` for an
        unweighted mean).
    fine_features
        ``(..., n_fine, d)`` per-vertex fine-level features.

    Returns
    -------
    ``(..., n_coarse, d)`` per-vertex coarse-level features.
    """
    from ..semiring import REAL

    num = _apply_shared_ell(
        coarsen_ell.values,
        coarsen_ell.indices,
        fine_features,
        semiring=REAL,
        n_cols=coarsen_ell.n_cols,
    )
    denom = jnp.sum(coarsen_ell.values, axis=-1)  # (n_coarse,)
    denom = jnp.where(denom > 0, denom, 1.0)
    return num / denom[..., None]


def mesh_unpool_max(
    cross_level_adjacency: ELL,
    coarse_features: Float[Array, '... n_coarse d'],
) -> Float[Array, '... n_fine d']:
    """Max-unpool coarse-level features back to a fine level.

    Symmetric to ``mesh_pool_max``: the cross-level adjacency is
    interpreted as fine-to-coarse (each row is a fine vertex,
    listing the coarse vertex / vertices it maps from), and the
    max-plus reduction produces a per-fine-vertex output.  In
    practice the cross-level adjacency is usually single-source
    per fine vertex (each fine vertex has one parent coarse
    vertex), in which case "max" reduces to the trivial gather.

    Parameters / Returns mirror ``mesh_pool_max`` with the levels
    swapped.
    """
    from ..semiring import TROPICAL_MAX_PLUS

    zeros = jnp.zeros_like(cross_level_adjacency.values)
    return _apply_shared_ell(
        zeros,
        cross_level_adjacency.indices,
        coarse_features,
        semiring=TROPICAL_MAX_PLUS,
        n_cols=cross_level_adjacency.n_cols,
    )


def mesh_bary_upsample(
    bary_ell: ELL,
    coarse_coords: Float[Array, '... n_coarse d'],
) -> Float[Array, '... n_fine d']:
    """Barycentric upsample from coarse to fine.

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
    """
    from ..semiring import REAL

    return _apply_shared_ell(
        bary_ell.values,
        bary_ell.indices,
        coarse_coords,
        semiring=REAL,
        n_cols=bary_ell.n_cols,
    )


# ---------------------------------------------------------------------------
# Per-vertex differential geometry (normals, uniform smoothing)
# ---------------------------------------------------------------------------


def compute_vertex_normals(
    vertices: Float[Array, 'n_vertices 3'],
    faces: Int[Array, 'n_faces 3'],
) -> Float[Array, 'n_vertices 3']:
    """Unit per-vertex normals from a triangle mesh.

    Each face contributes its (un-normalised) cross-product normal
    ``(v1 - v0) x (v2 - v0)`` -- whose magnitude is twice the triangle
    area, so larger faces weight more -- scattered onto its three
    vertices; the per-vertex sum is then L2-normalised.

    Parameters
    ----------
    vertices
        ``(n_vertices, 3)`` coordinates.
    faces
        ``(n_faces, 3)`` vertex indices.

    Returns
    -------
    ``(n_vertices, 3)`` unit normals (a zero-area vertex maps to a zero
    vector rather than ``NaN``).
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = jnp.cross(v1 - v0, v2 - v0)
    normals = jnp.zeros_like(vertices)
    for k in range(3):
        normals = normals.at[faces[:, k]].add(face_normals)
    norm = jnp.sqrt(jnp.sum(normals**2, axis=-1, keepdims=True))
    return normals / jnp.maximum(norm, 1e-12)


def mesh_laplacian_smooth(
    vertices: Float[Array, 'n_vertices 3'],
    faces: Int[Array, 'n_faces 3'],
    *,
    lam: float = 0.5,
    iterations: int = 1,
) -> Float[Array, 'n_vertices 3']:
    """Uniform (combinatorial) Laplacian smoothing of vertex positions.

    Each iteration moves every vertex a fraction ``lam`` toward the
    average of its 1-ring neighbours:
    ``v <- (1 - lam) v + lam * mean(neighbours(v))``.  Neighbours come
    from the triangle edges; on a closed manifold every neighbour is
    counted by an equal number of incident faces, so the unweighted mean
    is exact without de-duplicating shared edges.

    Parameters
    ----------
    vertices
        ``(n_vertices, 3)`` coordinates.
    faces
        ``(n_faces, 3)`` vertex indices.
    lam
        Step size in ``[0, 1]`` (``0`` = no-op).
    iterations
        Number of smoothing passes (static).

    Returns
    -------
    ``(n_vertices, 3)`` smoothed coordinates.
    """
    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    src = jnp.concatenate([i0, i1, i2, i1, i2, i0])
    dst = jnp.concatenate([i1, i2, i0, i0, i1, i2])
    n = vertices.shape[0]
    degree = jnp.zeros((n,), dtype=vertices.dtype).at[src].add(1.0)
    inv_degree = (1.0 / jnp.maximum(degree, 1.0))[:, None]

    def body(_: int, v: Array) -> Array:
        neighbour_sum = jnp.zeros_like(v).at[src].add(v[dst])
        mean = neighbour_sum * inv_degree
        return (1.0 - lam) * v + lam * mean

    out = jax.lax.fori_loop(0, iterations, body, vertices)
    return cast(Float[Array, 'n_vertices 3'], out)
