# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Surface-distance metrics: ``hausdorff95`` and ``surface_dice``.

The boundary-overlap reporting set for segmentation (BraTS / KiTS / MSD score
Dice **+ HD95 + surface-Dice**).  Both are built from one pipeline over the
``morphology`` substrate:

1. **surface** voxels = ``mask & ~erode(mask)`` with a connectivity-1 (face)
   structuring element -- the boundary voxels (those with at least one
   background face-neighbour).  The mask is padded with background first, so a
   foreground voxel on the volume border counts as surface (scipy
   ``binary_erosion`` ``border_value=0`` semantics).
2. **distance field** to the other surface = exact (anisotropic) Euclidean
   distance transform of the surface's complement (``morphology.
   distance_transform_edt``), so distances are in ``spacing`` units.
3. **reduce**: HD95 = the max over directions of the 95th-percentile directed
   surface distance; surface-Dice = the fraction of both surfaces within a
   tolerance of the other.

Convention is pinned to **MONAI** (the lineage of the consuming ports):
``compute_hausdorff_distance(percentile=95)`` and ``compute_surface_dice``
(``use_subvoxels=False``, the count -- not area -- form).  These are
**reporting metrics, not training objectives**: non-differentiable (boundary
extraction / EDT / percentile / counting are piecewise-constant), per SPEC §2
tenet 2's hard-output carve-out.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from .._internal.backend import Backend
from ..morphology import distance_transform_edt, erode

__all__ = ['hausdorff95', 'surface_dice']

_Spacing = Optional[Union[float, Sequence[float]]]


def _cross_se(ndim: int) -> Float[Array, '...']:
    """Connectivity-1 (face) flat erosion support: 0 at centre + axis
    neighbours, ``-inf`` at the corners.

    ``erode`` computes ``min_p (x[i+p] - se[p])``, so an off-support corner needs
    ``se = -inf`` (then ``-se = +inf`` keeps it out of the min); the support uses
    ``se = 0`` (a flat structuring element).
    """
    se = jnp.full((3,) * ndim, -jnp.inf, dtype=jnp.float32)
    centre = (1,) * ndim
    se = se.at[centre].set(0.0)
    for axis in range(ndim):
        for off in (0, 2):
            idx = list(centre)
            idx[axis] = off
            se = se.at[tuple(idx)].set(0.0)
    return se


def _surface(mask: Array, *, backend: Backend) -> Bool[Array, '...']:
    """Connectivity-1 boundary voxels of a binary ``mask``.

    Padded with background before the erosion so volume-border foreground is a
    surface (scipy ``binary_erosion`` ``border_value=0``); the min-plus
    ``erode`` then matches ``binary_erosion(mask) ^ mask``.
    """
    m = jnp.asarray(mask) != 0
    ndim = m.ndim
    se = _cross_se(ndim)
    padded = jnp.pad(
        m.astype(jnp.float32), [(1, 1)] * ndim, constant_values=0.0
    )
    eroded = erode(padded, structuring_element=se, backend=backend)
    interior = tuple(slice(1, -1) for _ in range(ndim))
    return m & (eroded[interior] <= 0.5)


def _distance_field(
    edges: Array, spacing: _Spacing, *, backend: Backend
) -> Float[Array, '...']:
    """Exact (anisotropic) Euclidean distance to the ``edges`` surface; ``+inf``
    everywhere if ``edges`` is empty (no surface reachable)."""
    seed = jnp.where(edges, 0.0, 1.0)
    return distance_transform_edt(seed, sampling=spacing, backend=backend)


def hausdorff95(
    pred: Bool[Array, '*spatial'],
    target: Bool[Array, '*spatial'],
    *,
    spacing: _Spacing = None,
    backend: Backend = 'auto',
) -> Float[Array, '']:
    """95th-percentile symmetric surface distance (MONAI ``percentile=95``).

    ``max`` over the two directions of the 95th-percentile directed surface
    distance: ``max(Q95(d(S_pred -> S_target)), Q95(d(S_target -> S_pred)))``,
    where ``S_*`` are the connectivity-1 surfaces and the percentile uses linear
    interpolation (``numpy`` / ``torch.quantile`` default).

    Parameters
    ----------
    pred, target
        Binary masks (any non-zero is foreground), same spatial shape.
    spacing
        Per-axis voxel size (scalar or one per axis); ``None`` is unit.  The
        result is in these units.
    backend
        Dispatch for the underlying ``erode`` / EDT (``"auto"`` / ``"jax"`` /
        ``"pallas-cuda"``).

    Returns
    -------
    Scalar HD95 in ``spacing`` units; ``+inf`` if either mask has no surface
    (empty), matching MONAI.  Non-differentiable (reporting metric).
    """
    ep = _surface(pred, backend=backend)
    eg = _surface(target, backend=backend)
    d_to_g = _distance_field(eg, spacing, backend=backend)
    d_to_p = _distance_field(ep, spacing, backend=backend)
    pg = jnp.nanquantile(jnp.where(ep, d_to_g, jnp.nan), 0.95)
    gp = jnp.nanquantile(jnp.where(eg, d_to_p, jnp.nan), 0.95)
    empty = (jnp.sum(ep) == 0) | (jnp.sum(eg) == 0)
    return jnp.where(empty, jnp.inf, jnp.maximum(pg, gp))


def surface_dice(
    pred: Bool[Array, '*spatial'],
    target: Bool[Array, '*spatial'],
    *,
    tolerance: float,
    spacing: _Spacing = None,
    backend: Backend = 'auto',
) -> Float[Array, '']:
    """Normalised surface Dice at ``tolerance`` (Nikolov et al. 2018; MONAI
    ``compute_surface_dice``, count form).

    The fraction of both surfaces lying within ``tolerance`` of the other:
    ``(|{S_pred : d<=tol}| + |{S_target : d<=tol}|) / (|S_pred| + |S_target|)``.

    Parameters
    ----------
    pred, target
        Binary masks (any non-zero is foreground), same spatial shape.
    tolerance
        Boundary tolerance ``tau`` in ``spacing`` units.
    spacing
        Per-axis voxel size (scalar or one per axis); ``None`` is unit.
    backend
        Dispatch for the underlying ``erode`` / EDT.

    Returns
    -------
    Scalar NSD in ``[0, 1]``; ``1`` for identical masks, ``0`` when a class is
    present in only one mask, ``nan`` when both surfaces are empty (matching
    MONAI).  Non-differentiable (reporting metric).
    """
    ep = _surface(pred, backend=backend)
    eg = _surface(target, backend=backend)
    d_to_g = _distance_field(eg, spacing, backend=backend)
    d_to_p = _distance_field(ep, spacing, backend=backend)
    correct = jnp.sum((d_to_g <= tolerance) & ep) + jnp.sum(
        (d_to_p <= tolerance) & eg
    )
    complete = jnp.sum(ep) + jnp.sum(eg)
    correct = correct.astype(jnp.float32)
    complete_safe = jnp.maximum(complete, 1).astype(jnp.float32)
    return jnp.where(complete == 0, jnp.nan, correct / complete_safe)
