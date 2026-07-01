# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Surface-distance metrics: :func:`hausdorff95` and :func:`surface_dice`.

The boundary-overlap reporting set for segmentation (BraTS / KiTS / MSD score
Dice **+ HD95 + surface-Dice**).  Both are built from one pipeline over the
morphology substrate:

1. **surface** voxels = :math:`\\text{mask} \\wedge \\neg\\,\\text{erode(mask)}`
   with a connectivity-1 (face) structuring element -- the boundary voxels
   (those with at least one background face-neighbour).  The mask is padded with
   background first, so a foreground voxel on the volume border counts as
   surface (scipy ``binary_erosion`` ``border_value=0`` semantics).
2. **distance field** to the other surface = exact (anisotropic) Euclidean
   distance transform of the surface's complement
   (:func:`~nitrix.morphology.distance_transform_edt`), so distances are in
   ``spacing`` units.
3. **reduce**: HD95 = the max over directions of the 95th-percentile directed
   surface distance; surface-Dice = the fraction of both surfaces within a
   tolerance of the other.

Convention is pinned to **MONAI** (the lineage of the consuming ports):
``compute_hausdorff_distance(percentile=95)`` and ``compute_surface_dice``
(``use_subvoxels=False``, the count -- not area -- form).  These are
**reporting metrics, not training objectives**: they are non-differentiable
(boundary extraction, the Euclidean distance transform, the percentile, and the
counting are all piecewise-constant), and so are admitted as hard-output
diagnostics rather than as gradients.
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
    """Connectivity-1 (face) flat structuring element for erosion.

    Returns the ``3 x ... x 3`` support that is ``0`` at the centre and at each
    axis-aligned face neighbour, and :math:`-\\infty` at every corner.

    :func:`~nitrix.morphology.erode` computes
    :math:`\\min_p (x_{i+p} - \\text{se}_p)`, so an off-support corner needs
    :math:`\\text{se} = -\\infty` (then :math:`-\\text{se} = +\\infty` keeps it
    out of the minimum); the support uses :math:`\\text{se} = 0` (a flat
    structuring element).

    Parameters
    ----------
    ndim
        Number of spatial dimensions of the volume the element applies to.  The
        returned element has shape ``(3,) * ndim``.

    Returns
    -------
    Float[Array, '...']
        Flat structuring element of shape ``(3,) * ndim`` and dtype
        ``float32``, holding ``0`` on the connectivity-1 support (centre plus
        axis neighbours) and :math:`-\\infty` off it.
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

    A voxel is a surface voxel when it is foreground and has at least one
    background face-neighbour.  The mask is padded with background before the
    erosion so that volume-border foreground counts as surface (scipy
    ``binary_erosion`` ``border_value=0`` semantics); the min-plus
    :func:`~nitrix.morphology.erode` then matches
    :math:`\\text{binary\\_erosion(mask)} \\oplus \\text{mask}` (the logical
    difference between the mask and its erosion).

    Parameters
    ----------
    mask
        Binary volume; any non-zero entry is treated as foreground.  Its number
        of dimensions sets the connectivity-1 structuring element.
    backend
        Execution engine forwarded to :func:`~nitrix.morphology.erode`
        (``"auto"`` / ``"jax"`` / ``"pallas-cuda"``).

    Returns
    -------
    Bool[Array, '...']
        Boolean array with the spatial shape of ``mask``, true exactly at the
        connectivity-1 boundary voxels.
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
    """Exact (anisotropic) Euclidean distance to the ``edges`` surface.

    Every voxel is assigned its distance to the nearest true entry of ``edges``,
    measured in ``spacing`` units.  When ``edges`` is empty (no surface
    reachable) the field is :math:`+\\infty` everywhere.

    Parameters
    ----------
    edges
        Boolean surface mask; true entries are the surface voxels distances are
        measured to.
    spacing
        Per-axis voxel size (scalar or one per axis); ``None`` uses unit
        spacing.  Sets the units of the returned distances.
    backend
        Execution engine forwarded to
        :func:`~nitrix.morphology.distance_transform_edt`.

    Returns
    -------
    Float[Array, '...']
        Distance field with the spatial shape of ``edges``, giving the Euclidean
        distance from each voxel to the nearest surface voxel in ``spacing``
        units.
    """
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

    Takes the maximum over the two directions of the 95th-percentile directed
    surface distance:
    :math:`\\max(Q_{95}(d(S_{\\text{pred}} \\to S_{\\text{target}})),
    Q_{95}(d(S_{\\text{target}} \\to S_{\\text{pred}})))`, where
    :math:`S_{\\ast}` are the connectivity-1 surfaces and the percentile uses
    linear interpolation (the ``numpy`` / ``torch.quantile`` default).

    Parameters
    ----------
    pred, target
        Binary masks (any non-zero is foreground), of the same spatial shape.
    spacing
        Per-axis voxel size (scalar or one per axis); ``None`` uses unit
        spacing.  The result is in these units.
    backend
        Execution engine for the underlying
        :func:`~nitrix.morphology.erode` and Euclidean distance transform
        (``"auto"`` / ``"jax"`` / ``"pallas-cuda"``).

    Returns
    -------
    Float[Array, '']
        Scalar HD95 in ``spacing`` units; :math:`+\\infty` if either mask has no
        surface (is empty), matching MONAI.  Non-differentiable (a reporting
        metric).
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
    """Normalised surface Dice at a boundary ``tolerance``.

    The fraction of both surfaces lying within ``tolerance`` of the other:
    :math:`(|\\{S_{\\text{pred}} : d \\le \\tau\\}| +
    |\\{S_{\\text{target}} : d \\le \\tau\\}|) /
    (|S_{\\text{pred}}| + |S_{\\text{target}}|)`.  This is the count form of the
    metric (surface voxels counted, not areas), following MONAI's
    ``compute_surface_dice`` with ``use_subvoxels=False``.

    Parameters
    ----------
    pred, target
        Binary masks (any non-zero is foreground), of the same spatial shape.
    tolerance
        Boundary tolerance :math:`\\tau` in ``spacing`` units.
    spacing
        Per-axis voxel size (scalar or one per axis); ``None`` uses unit
        spacing.
    backend
        Execution engine for the underlying
        :func:`~nitrix.morphology.erode` and Euclidean distance transform.

    Returns
    -------
    Float[Array, '']
        Scalar normalised surface Dice in :math:`[0, 1]`; ``1`` for identical
        masks, ``0`` when a class is present in only one mask, and ``nan`` when
        both surfaces are empty (matching MONAI).  Non-differentiable (a
        reporting metric).

    References
    ----------
    Nikolov S, Blackwell S, Zverovitch A, et al. (2018). Deep learning to
    achieve clinically applicable segmentation of head and neck anatomy for
    radiotherapy. arXiv:1809.04430. https://doi.org/10.48550/arXiv.1809.04430
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
