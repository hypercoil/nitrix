# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Surface functional parcellation: boundary mapping and watershed.

The Cohen / Wig / Gordon / Schaefer boundary-detection lineage, in two
composable steps:

- :func:`surface_boundary_map` -- per-vertex connectivity profiles ->
  adjacent-profile dissimilarity -> a per-vertex boundary map.  A named
  wrapper on :func:`semiring_ell_edge_aggregate` (no new kernel): the edge
  function is the profile dissimilarity :math:`1 - \\operatorname{sim}(h_i,
  h_j)`, aggregated per row by mean (over the :data:`REAL` semiring) or max
  (over the :data:`TROPICAL_MAX_PLUS` semiring).  :func:`eta_squared` is the
  companion similarity of Cohen et al. (2008).
- :func:`mesh_watershed` -- priority-flood watershed on a vertex-valued field
  over an arbitrary mesh adjacency: every vertex is labelled by its catchment
  basin (the basins of the boundary map's minima are the parcels).  Runs
  host-side (the flood is serial), returning a JAX integer array.

References
----------
Cohen, A. L., Fair, D. A., Dosenbach, N. U. F., Miezin, F. M., Dierker, D.,
Van Essen, D. C., Schlaggar, B. L., & Petersen, S. E. (2008). Defining
functional areas in individual human brains using resting functional
connectivity MRI. NeuroImage, 41(1), 45-57.
https://doi.org/10.1016/j.neuroimage.2008.01.066

Wig, G. S., Laumann, T. O., & Petersen, S. E. (2014). An approach for
parcellating human cortical areas using resting-state correlations.
NeuroImage, 93, 276-291.
https://doi.org/10.1016/j.neuroimage.2013.07.035

Gordon, E. M., Laumann, T. O., Adeyemo, B., Huckins, J. F., Kelley, W. M., &
Petersen, S. E. (2016). Generation and evaluation of a cortical area
parcellation from resting-state correlations. Cerebral Cortex, 26(1),
288-303. https://doi.org/10.1093/cercor/bhu239

Schaefer, A., Kong, R., Gordon, E. M., Laumann, T. O., Zuo, X.-N., Holmes,
A. J., Eickhoff, S. B., & Yeo, B. T. T. (2018). Local-global parcellation of
the human cerebral cortex from intrinsic functional connectivity MRI.
Cerebral Cortex, 28(9), 3095-3114. https://doi.org/10.1093/cercor/bhx179
"""

from __future__ import annotations

import heapq
from typing import Any, Dict, List, Literal, Optional

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int
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
    """Eta-squared similarity between two profiles (Cohen et al., 2008).

    Computes :math:`\\eta^2 = 1 - \\mathrm{SS}_{\\mathrm{within}} /
    \\mathrm{SS}_{\\mathrm{total}}` where, over the profile axis,
    :math:`\\mathrm{SS}_{\\mathrm{within}} = \\sum_i (x_i - m_i)^2 + (y_i -
    m_i)^2` with :math:`m_i = (x_i + y_i) / 2` the per-element pair mean, and
    :math:`\\mathrm{SS}_{\\mathrm{total}} = \\sum_i (x_i - M)^2 + (y_i - M)^2`
    with :math:`M` the grand mean of both profiles.  The result ranges over
    :math:`[0, 1]` (attaining :math:`1` iff ``x == y``); it is the canonical
    functional-connectivity profile-similarity measure, more robust than the
    Pearson correlation to a global offset or scale between profiles, and is
    differentiable.

    Parameters
    ----------
    x, y
        Profiles with the comparison along the last axis; leading axes
        broadcast.

    Returns
    -------
    Float[Array, '...']
        Eta-squared similarity, with the broadcast leading shape of ``x`` and
        ``y`` (the comparison axis reduced away).
    """
    m = 0.5 * (x + y)
    grand = 0.5 * (
        jnp.mean(x, axis=-1, keepdims=True)
        + jnp.mean(y, axis=-1, keepdims=True)
    )
    ss_within = jnp.sum((x - m) ** 2 + (y - m) ** 2, axis=-1)
    ss_total = jnp.sum((x - grand) ** 2 + (y - grand) ** 2, axis=-1)
    return 1.0 - ss_within / jnp.maximum(ss_total, _TINY)


def _pearson(
    x: Float[Array, '... d'],
    y: Float[Array, '... d'],
) -> Float[Array, '...']:
    """Pearson correlation along the last axis (leading axes broadcast).

    Parameters
    ----------
    x, y
        Profiles with the comparison along the last axis; leading axes
        broadcast.

    Returns
    -------
    Float[Array, '...']
        Pearson correlation, with the broadcast leading shape of ``x`` and
        ``y`` (the comparison axis reduced away).
    """
    xc = x - jnp.mean(x, axis=-1, keepdims=True)
    yc = y - jnp.mean(y, axis=-1, keepdims=True)
    num = jnp.sum(xc * yc, axis=-1)
    den = jnp.sqrt(jnp.sum(xc * xc, axis=-1) * jnp.sum(yc * yc, axis=-1))
    return num / jnp.maximum(den, _TINY)


# ---------------------------------------------------------------------------
# Boundary map
# ---------------------------------------------------------------------------


def _roi_masked_adjacency(
    adjacency: ELL, roi: Bool[Array, 'n_vertices']
) -> ELL:
    """Drop edges pointing at out-of-ROI vertices (set their value to 0).

    A neighbour outside the region of interest (e.g. the medial wall) is
    turned into ELL padding, so it contributes the semiring identity and is
    excluded from the degree count -- the boundary map and flooding never
    cross the ROI boundary.

    Parameters
    ----------
    adjacency
        Mesh adjacency in ELL format; real edges carry nonzero ``values`` and
        padding carries ``0``.
    roi
        ``(n_vertices,)`` boolean mask; ``True`` marks in-ROI vertices.

    Returns
    -------
    ELL
        A copy of ``adjacency`` with the same ``indices``, ``n_cols`` and
        ``identity`` but with edges to out-of-ROI neighbours zeroed out.
    """
    keep = roi[adjacency.indices]  # (n, k) -- is each neighbour in the ROI?
    return ELL(
        values=jnp.where(keep, adjacency.values, 0.0),
        indices=adjacency.indices,
        n_cols=adjacency.n_cols,
        identity=adjacency.identity,
    )


def surface_boundary_map(
    connectivity_profiles: Float[Array, 'n_vertices d_profile'],
    adjacency: ELL,
    *,
    similarity: Literal['eta_squared', 'pearson'] = 'eta_squared',
    aggregate: Literal['mean', 'max'] = 'mean',
    roi: Optional[Bool[Array, 'n_vertices']] = None,
) -> Float[Array, 'n_vertices']:
    """Per-vertex functional-connectivity boundary map.

    For each vertex, the dissimilarity :math:`1 -
    \\operatorname{sim}(\\mathrm{profile}_i, \\mathrm{profile}_j)` to each mesh
    neighbour :math:`j` is aggregated into a scalar boundary value: high where
    adjacent connectivity profiles differ (a putative areal border), low inside
    a homogeneous region.  Composes :func:`semiring_ell_edge_aggregate` over
    the ``adjacency`` ELL -- no new kernel -- so it inherits the
    format-agnostic ELL substrate and is differentiable with respect to the
    profiles.  The minima of the resulting map seed :func:`mesh_watershed`.

    Connectivity profiles are typically the vertex-wise correlation of the
    ``(n_vertices, T)`` vertex time-series (so ``d_profile = n_vertices``); at
    ico7 that profile is 163 842-dimensional per vertex, so it is generally
    materialised or tiled via a map over the row axis at the call site (a
    memory caveat, not part of the API here).

    Parameters
    ----------
    connectivity_profiles
        ``(n_vertices, d_profile)`` per-vertex profile (e.g. a connectivity
        fingerprint).
    adjacency
        Mesh adjacency in ELL format (e.g. from :func:`mesh_k_ring_adjacency`);
        real edges carry nonzero ``values`` and padding carries ``0`` (the
        standard ELL convention), which is how padding is masked out.
    similarity
        ``'eta_squared'`` (default, the measure of Cohen et al., 2008; see
        :func:`eta_squared`) or ``'pearson'``.
    aggregate
        ``'mean'`` (default; mean dissimilarity over neighbours) or ``'max'``
        (the strongest neighbour border).
    roi
        Optional ``(n_vertices,)`` boolean cortex mask.  Edges to out-of-ROI
        neighbours (e.g. the medial wall, whose profiles are meaningless) are
        dropped so they raise no spurious boundary, and out-of-ROI vertices are
        returned as ``0``.

    Returns
    -------
    Float[Array, 'n_vertices']
        Per-vertex boundary map: the aggregated neighbour dissimilarity at each
        vertex, or ``0`` at out-of-ROI vertices when ``roi`` is given.

    Raises
    ------
    ValueError
        If ``similarity`` or ``aggregate`` is unknown.
    """
    if roi is not None:
        adjacency = _roi_masked_adjacency(adjacency, jnp.asarray(roi))
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
    if roi is not None:
        out = jnp.where(jnp.asarray(roi), out, 0.0)
    return out


# ---------------------------------------------------------------------------
# Watershed (host-side priority flood)
# ---------------------------------------------------------------------------


def _neighbour_lists(adjacency: ELL) -> List[List[int]]:
    """Per-vertex neighbour lists (host-side) from an ELL adjacency.

    A neighbour is an entry whose stored value is nonzero (the ELL padding
    convention); self-loops and duplicates are dropped.  The unique
    ``(row, col)`` edge set is built with numpy and split into per-row lists,
    rather than a per-(vertex, slot) Python loop and set.  Neighbour order
    within a row is irrelevant to the serial flood -- a vertex's basin is
    fixed by the heap pop order, not the order in which a popped vertex
    enumerates its neighbours.

    Parameters
    ----------
    adjacency
        Mesh adjacency in ELL format; real edges carry nonzero ``values`` and
        padding carries ``0``.

    Returns
    -------
    list of list of int
        For each vertex (in row order), the sorted list of its unique
        neighbour vertex indices, excluding self-loops.
    """
    idx = np.asarray(adjacency.indices)
    val = np.asarray(adjacency.values)
    n, k = idx.shape
    self_idx = np.arange(n)[:, None]
    valid = (val != 0) & (idx != self_idx)
    rows = np.repeat(np.arange(n, dtype=np.int64), k)[valid.reshape(-1)]
    cols = idx.reshape(-1)[valid.reshape(-1)].astype(np.int64)
    if rows.size == 0:
        return [[] for _ in range(n)]
    key = np.unique(rows * np.int64(n) + cols)  # dedup (row, col), sorted
    urows = key // np.int64(n)
    ucols = (key % np.int64(n)).astype(np.int64)
    deg = np.bincount(urows, minlength=n)
    groups = np.split(ucols, np.cumsum(deg)[:-1])
    return [g.tolist() for g in groups]


def _merge_basins(
    labels: NDArray[Any],
    neighbours: List[List[int]],
    field: NDArray[Any],
    min_basin_size: int,
    h_min: float,
) -> NDArray[Any]:
    """Merge basins below a size or depth (prominence) criterion via union-find.

    The inter-basin saddle between two basins is the lowest :math:`\\max(f_u,
    f_v)` over edges :math:`(u, v)` crossing them; a basin's depth is its
    lowest saddle minus its minimum.  A basin is merged into its lowest-saddle
    neighbour when it is smaller than ``min_basin_size`` or shallower than
    ``h_min``, iterating to a fixed point, after which the surviving labels are
    compacted to a contiguous range.

    Parameters
    ----------
    labels
        ``(n,)`` integer basin labels, with ``-1`` marking out-of-ROI
        background (never a basin, never merged).
    neighbours
        Per-vertex neighbour lists (see :func:`_neighbour_lists`), in vertex
        order.
    field
        ``(n,)`` scalar field values that were flooded.
    min_basin_size
        Basins with fewer vertices than this are merged into their
        lowest-saddle neighbour.
    h_min
        Basins whose depth (saddle minus minimum) is below this are merged
        into their lowest-saddle neighbour.

    Returns
    -------
    NDArray
        ``(n,)`` integer labels after merging, compacted to a contiguous
        ``[0, n_basins)`` range, with ``-1`` preserved for background
        vertices.  If there are no inter-basin edges the input ``labels`` are
        returned unchanged.
    """
    n = len(labels)
    # -1 is the out-of-ROI background label: never a basin, never merged.
    roots = sorted(set(int(b) for b in labels.tolist() if b >= 0))
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
        if b < 0:
            continue
        size[b] += 1
        bmin[b] = min(bmin[b], float(field[i]))

    saddle: Dict[Any, float] = {}
    for i in range(n):
        a = int(labels[i])
        if a < 0:
            continue
        for j in neighbours[i]:
            b = int(labels[j])
            if b < 0 or a == b:
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
        lab = int(labels[i])
        if lab < 0:  # preserve background
            out[i] = -1
            continue
        r = find(lab)
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
    roi: Optional[Bool[Array, 'n_vertices']] = None,
) -> Int[Array, 'n_vertices']:
    """Priority-flood watershed labelling of a vertex field on a mesh.

    Labels every vertex by its catchment basin: water rises from each local
    minimum of ``field`` and each vertex joins the basin whose flood reaches it
    first (the priority-flood scheme of Barnes et al., 2014, with the monotone
    :math:`\\max(\\mathrm{level}, f_j)` key).  On a boundary map (see
    :func:`surface_boundary_map`) the minima are region interiors and the
    basins are the parcels; pass ``field`` directly to flood its minima (negate
    to flood maxima).

    Connected equal-valued minima are coalesced into a single seed (no plateau
    over-segmentation).  Two optional post-merges clean spurious basins:
    ``min_basin_size`` absorbs basins smaller than the threshold, and ``h_min``
    absorbs basins shallower than that depth (saddle-to-minimum prominence),
    each merging into the lowest-saddle neighbour.  The defaults
    (``min_basin_size=1``, ``h_min=0``) apply no merge -- a pure watershed.

    Runs host-side (the flood is inherently serial), returning a JAX integer
    array.  Works on any mesh adjacency, not just the icosphere.

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
    roi
        Optional ``(n_vertices,)`` boolean cortex mask.  Out-of-ROI vertices
        (e.g. the medial wall) are excluded from flooding and labelled ``-1``
        (background); basins never cross the ROI boundary.

    Returns
    -------
    Int[Array, 'n_vertices']
        ``int32`` basin labels in ``[0, n_basins)``, with ``-1`` for
        out-of-ROI vertices when ``roi`` is given.
    """
    field_np = np.asarray(field, dtype=np.float64).reshape(-1)
    n = field_np.shape[0]
    neighbours = _neighbour_lists(adjacency)
    active = (
        np.ones(n, dtype=bool)
        if roi is None
        else np.asarray(roi, dtype=bool).reshape(-1)
    )
    if roi is not None:  # drop edges to out-of-ROI vertices
        neighbours = [
            [j for j in nb if active[j]] if active[i] else []
            for i, nb in enumerate(neighbours)
        ]

    # Local minima: an active vertex whose value <= every active neighbour's.
    # Vectorised (Tier C / AI-C3) via the ELL adjacency + a segment min.
    idx = np.asarray(adjacency.indices)
    val = np.asarray(adjacency.values)
    valid = (val != 0) & (idx != np.arange(n)[:, None])
    if roi is not None:
        valid = valid & active[idx] & active[:, None]
    nbr_field = np.where(valid, field_np[idx], np.inf)
    min_nbr = nbr_field.min(
        axis=1
    )  # +inf for a vertex with no active neighbour
    is_min = active & (field_np <= min_nbr)

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

    # Any active vertex unreached (disconnected component with no interior min)
    # gets its own basin; out-of-ROI vertices stay -1 (background).
    for i in range(n):
        if labels[i] == -1 and active[i]:
            labels[i] = next_label
            next_label += 1

    if min_basin_size > 1 or h_min > 0.0:
        labels = _merge_basins(
            labels, neighbours, field_np, min_basin_size, h_min
        )
    return jnp.asarray(labels, dtype=jnp.int32)
