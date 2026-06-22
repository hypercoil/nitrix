# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Surface functional parcellation: boundary mapping + watershed.

The Cohen / Wig / Gordon / Schaefer boundary-detection lineage (Cohen 2008;
Wig 2014; Gordon 2016; Schaefer 2018), in two composable steps:

- ``surface_boundary_map`` -- per-vertex connectivity profiles -> adjacent-
  profile dissimilarity -> a per-vertex boundary map.  A named wrapper on
  ``semiring_ell_edge_aggregate`` (no new kernel): the edge function is the
  profile dissimilarity ``1 - similarity(h_i, h_j)``, aggregated per row by
  mean (``REAL``) or max (``TROPICAL_MAX_PLUS``).  ``eta_squared`` is the
  Cohen-2008 companion similarity.
- ``mesh_watershed`` -- priority-flood watershed (Barnes 2014) on a vertex-
  valued field over an arbitrary mesh adjacency: every vertex is labelled by
  its catchment basin (the basins of the boundary map's minima are the
  parcels).  Host-side (the flood is serial), returning a JAX int array --
  the ``mesh_k_ring_adjacency`` host-construct -> JAX-array pattern.

``geometry`` / ``graph`` depend on ``sparse`` and ``semiring``, never the
reverse.  See ``SPEC_UPDATE_v0.3 §12.16 / §12.17``.
"""

from __future__ import annotations

import heapq
from typing import Any, Dict, List

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from numpy.typing import NDArray

from ..semiring import REAL, TROPICAL_MAX_PLUS, semiring_ell_edge_aggregate
from ..sparse import ELL

__all__ = [
    'eta_squared',
    'surface_boundary_map',
    'mesh_watershed',
]

_TINY = 1e-12


# ---------------------------------------------------------------------------
# Profile similarity
# ---------------------------------------------------------------------------


def eta_squared(
    x: Float[Array, '... d'],
    y: Float[Array, '... d'],
) -> Float[Array, '...']:
    """Cohen-2008 eta-squared similarity between two profiles.

    ``eta^2 = 1 - SS_within / SS_total`` where, over the profile axis,
    ``SS_within = sum_i (x_i - m_i)^2 + (y_i - m_i)^2`` with
    ``m_i = (x_i + y_i) / 2`` the per-element pair mean, and
    ``SS_total = sum_i (x_i - M)^2 + (y_i - M)^2`` with ``M`` the grand mean of
    both profiles.  Ranges ``[0, 1]`` (``1`` iff ``x == y``); the canonical
    functional-connectivity profile-similarity measure (more robust than
    Pearson to global offset / scale between profiles).  Differentiable.

    Parameters
    ----------
    x, y
        Profiles with the comparison along the last axis; leading axes
        broadcast.

    Returns
    -------
    eta-squared similarity, the broadcast leading shape.
    """
    m = 0.5 * (x + y)
    grand = 0.5 * (
        jnp.mean(x, axis=-1, keepdims=True) + jnp.mean(y, axis=-1, keepdims=True)
    )
    ss_within = jnp.sum((x - m) ** 2 + (y - m) ** 2, axis=-1)
    ss_total = jnp.sum((x - grand) ** 2 + (y - grand) ** 2, axis=-1)
    return 1.0 - ss_within / jnp.maximum(ss_total, _TINY)


def _pearson(
    x: Float[Array, '... d'],
    y: Float[Array, '... d'],
) -> Float[Array, '...']:
    """Pearson correlation along the last axis (leading axes broadcast)."""
    xc = x - jnp.mean(x, axis=-1, keepdims=True)
    yc = y - jnp.mean(y, axis=-1, keepdims=True)
    num = jnp.sum(xc * yc, axis=-1)
    den = jnp.sqrt(jnp.sum(xc * xc, axis=-1) * jnp.sum(yc * yc, axis=-1))
    return num / jnp.maximum(den, _TINY)


# ---------------------------------------------------------------------------
# Boundary map
# ---------------------------------------------------------------------------


def surface_boundary_map(
    connectivity_profiles: Float[Array, 'n_vertices d_profile'],
    adjacency: ELL,
    *,
    similarity: str = 'eta_squared',
    aggregate: str = 'mean',
) -> Float[Array, 'n_vertices']:
    """Per-vertex functional-connectivity boundary map.

    For each vertex, the dissimilarity ``1 - similarity(profile_i, profile_j)``
    to each mesh neighbour ``j`` is aggregated into a scalar boundary value:
    high where adjacent connectivity profiles differ (a putative areal border),
    low inside a homogeneous region.  Composes ``semiring_ell_edge_aggregate``
    over the ``adjacency`` ELL -- no new kernel -- so it inherits the
    format-agnostic ELL substrate and is **differentiable** w.r.t. the
    profiles.  The minima of the resulting map seed ``mesh_watershed``.

    Connectivity profiles are typically ``stats.corr`` of the
    ``(n_vertices, T)`` vertex time-series (so ``d_profile = n_vertices``); at
    ico7 that profile is 163 842-dim per vertex -- materialise / tile it via
    ``jax.lax.map`` over the row axis at the call site (a memory caveat, not an
    API surface here).

    Parameters
    ----------
    connectivity_profiles
        ``(n_vertices, d_profile)`` per-vertex profile (e.g. a connectivity
        fingerprint).
    adjacency
        Mesh adjacency in ELL format (e.g. ``mesh_k_ring_adjacency``); real
        edges carry nonzero ``values`` and padding carries ``0`` (the standard
        ELL convention), which is how padding is masked out.
    similarity
        ``'eta_squared'`` (default, Cohen 2008) or ``'pearson'``.
    aggregate
        ``'mean'`` (default; mean dissimilarity over neighbours) or ``'max'``
        (the strongest neighbour border).

    Returns
    -------
    ``(n_vertices,)`` boundary map.

    Raises
    ------
    ValueError
        If ``similarity`` or ``aggregate`` is unknown.
    """
    if similarity == 'eta_squared':
        sim = eta_squared
    elif similarity == 'pearson':
        sim = _pearson
    else:
        raise ValueError(
            f"surface_boundary_map: similarity must be 'eta_squared' or "
            f"'pearson'; got {similarity!r}."
        )
    if aggregate == 'mean':
        semiring = REAL
        pad_val = 0.0
    elif aggregate == 'max':
        semiring = TROPICAL_MAX_PLUS
        pad_val = -jnp.inf
    else:
        raise ValueError(
            f"surface_boundary_map: aggregate must be 'mean' or 'max'; got "
            f'{aggregate!r}.'
        )

    def edge_fn(
        h_i: Array, h_j: Array, w: Array, ij: Any
    ) -> Float[Array, '1']:
        diss = 1.0 - sim(h_i, h_j)
        return jnp.where(w == 0, pad_val, diss)[None]

    out = semiring_ell_edge_aggregate(
        edge_fn, adjacency, connectivity_profiles, semiring=semiring
    )[..., 0]
    if aggregate == 'mean':
        degree = jnp.sum(adjacency.values != 0, axis=1)
        out = out / jnp.maximum(degree, 1).astype(out.dtype)
    return out


# ---------------------------------------------------------------------------
# Watershed (host-side priority flood)
# ---------------------------------------------------------------------------


def _neighbour_lists(adjacency: ELL) -> List[List[int]]:
    """Per-vertex neighbour lists (host-side) from an ELL adjacency.

    A neighbour is an entry whose stored value is nonzero (the ELL padding
    convention); self-loops and duplicates are dropped.
    """
    idx = np.asarray(adjacency.indices)
    val = np.asarray(adjacency.values)
    n = idx.shape[0]
    out: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        seen = set()
        for p in range(idx.shape[1]):
            if val[i, p] != 0:
                j = int(idx[i, p])
                if j != i and j not in seen:
                    seen.add(j)
                    out[i].append(j)
    return out


def _merge_basins(
    labels: NDArray[Any],
    neighbours: List[List[int]],
    field: NDArray[Any],
    min_basin_size: int,
    h_min: float,
) -> NDArray[Any]:
    """Merge basins below a size or depth (prominence) criterion via union-find.

    The inter-basin saddle between two basins is the lowest ``max(field[u],
    field[v])`` over edges crossing them; a basin's depth is its lowest saddle
    minus its minimum.  A basin is merged into its lowest-saddle neighbour when
    it is smaller than ``min_basin_size`` or shallower than ``h_min``.
    """
    n = len(labels)
    roots = sorted(set(int(b) for b in labels.tolist()))
    parent: Dict[int, int] = {b: b for b in roots}

    def find(b: int) -> int:
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        return b

    size: Dict[int, float] = {b: 0 for b in roots}
    bmin: Dict[int, float] = {b: np.inf for b in roots}
    for i in range(n):
        b = int(labels[i])
        size[b] += 1
        bmin[b] = min(bmin[b], float(field[i]))

    saddle: Dict[Any, float] = {}
    for i in range(n):
        a = int(labels[i])
        for j in neighbours[i]:
            b = int(labels[j])
            if a == b:
                continue
            key = frozenset((a, b))
            s = max(float(field[i]), float(field[j]))
            if key not in saddle or s < saddle[key]:
                saddle[key] = s
    if not saddle:
        return labels

    def lowest_neighbour(r: int) -> Any:
        best_other = None
        best_s = np.inf
        for key, s in saddle.items():
            a, b = tuple(key)
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            if ra == r:
                other = rb
            elif rb == r:
                other = ra
            else:
                continue
            if s < best_s:
                best_s = s
                best_other = other
        return best_other, best_s

    changed = True
    while changed:
        changed = False
        for r in [b for b in roots if find(b) == b]:
            other, s = lowest_neighbour(r)
            if other is None:
                continue
            too_small = size[r] < min_basin_size
            too_shallow = (s - bmin[r]) < h_min
            if too_small or too_shallow:
                parent[r] = other
                size[other] += size[r]
                bmin[other] = min(bmin[other], bmin[r])
                changed = True

    out = np.empty(n, dtype=np.int64)
    remap: Dict[int, int] = {}
    nxt = 0
    for i in range(n):
        r = find(int(labels[i]))
        if r not in remap:
            remap[r] = nxt
            nxt += 1
        out[i] = remap[r]
    return out


def mesh_watershed(
    field: Float[Array, 'n_vertices'],
    adjacency: ELL,
    *,
    min_basin_size: int = 1,
    h_min: float = 0.0,
) -> Int[Array, 'n_vertices']:
    """Priority-flood watershed labelling of a vertex field on a mesh.

    Labels every vertex by its catchment basin: water rises from each local
    minimum of ``field`` and each vertex joins the basin whose flood reaches it
    first (Barnes 2014 priority-flood, the monotone ``max(level, field[j])``
    key).  On a boundary map (``surface_boundary_map``) the minima are region
    interiors and the basins are the parcels; pass ``field`` directly to flood
    its minima (negate to flood maxima).

    Connected equal-valued minima are coalesced into a single seed (no plateau
    over-segmentation).  Two optional post-merges clean spurious basins:
    ``min_basin_size`` absorbs basins smaller than the threshold, and ``h_min``
    absorbs basins shallower than that depth (saddle-to-minimum prominence),
    each merging into the lowest-saddle neighbour.  The defaults
    (``min_basin_size=1``, ``h_min=0``) apply no merge -- a pure watershed.

    Host-side (the flood is inherently serial), returning a JAX int array --
    the ``mesh_k_ring_adjacency`` host-construct -> JAX-array pattern.  Works on
    any mesh adjacency, not just the icosphere.

    Parameters
    ----------
    field
        ``(n_vertices,)`` scalar field to flood (e.g. a boundary map).
    adjacency
        Mesh adjacency in ELL format; real edges carry nonzero ``values``.
    min_basin_size
        Basins with fewer vertices are merged away (``1`` = no size merge).
    h_min
        Basins shallower than this depth are merged away (``0`` = no depth
        merge).

    Returns
    -------
    ``(n_vertices,)`` int32 basin labels in ``[0, n_basins)``.
    """
    field_np = np.asarray(field, dtype=np.float64).reshape(-1)
    n = field_np.shape[0]
    neighbours = _neighbour_lists(adjacency)

    # Local minima: value <= every neighbour's value.
    is_min = np.ones(n, dtype=bool)
    for i in range(n):
        fi = field_np[i]
        for j in neighbours[i]:
            if field_np[j] < fi:
                is_min[i] = False
                break

    # Coalesce connected equal-valued minima (plateau) into single seeds.
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        if not is_min[i]:
            continue
        for j in neighbours[i]:
            if is_min[j] and field_np[j] == field_np[i]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    labels = np.full(n, -1, dtype=np.int64)
    seed_label: Dict[int, int] = {}
    next_label = 0
    heap: List[Any] = []
    for i in range(n):
        if is_min[i]:
            r = find(i)
            if r not in seed_label:
                seed_label[r] = next_label
                next_label += 1
            labels[i] = seed_label[r]
            heapq.heappush(heap, (field_np[i], i))

    # Priority flood: pop the lowest water level, claim unlabelled neighbours.
    while heap:
        level, i = heapq.heappop(heap)
        for j in neighbours[i]:
            if labels[j] == -1:
                labels[j] = labels[i]
                heapq.heappush(heap, (max(level, field_np[j]), j))

    # Any vertex unreached (disconnected component with no interior min) gets
    # its own basin.
    for i in range(n):
        if labels[i] == -1:
            labels[i] = next_label
            next_label += 1

    if min_basin_size > 1 or h_min > 0.0:
        labels = _merge_basins(
            labels, neighbours, field_np, min_basin_size, h_min
        )
    return jnp.asarray(labels, dtype=jnp.int32)
