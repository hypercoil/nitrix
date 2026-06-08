# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Coordinate utilities: centre of mass on point clouds, displacements
from reference points, and the weight-compactness regulariser.

Shape convention -- different from the legacy code, more natural:

- ``weight``: ``(..., n_regions, n_points)`` -- one row per region,
  one column per point.
- ``coords``: ``(..., n_points, ndim)`` -- one row per point.
- ``center_of_mass_points`` returns ``(..., n_regions, ndim)``.

The legacy ``(*, D, L)`` coordinate layout (channel-leading) is
gone.  The new ``(*, L, D)`` matches the rest of the library
(``bilateral_gaussian``, ``brute_force_knn``, etc.) and lets the
einsum read naturally: ``weight @ coords``.

Renamings from legacy code:

- ``cmass_coor``                       -> ``center_of_mass_points``
- ``cmass_reference_displacement_grid`` -> ``displacement_from_reference_grid``
- ``cmass_reference_displacement_coor`` -> ``displacement_from_reference_points``
- ``diffuse``                           -> ``compactness_penalty``
"""

from __future__ import annotations

from typing import Optional, Sequence, Union, cast

import jax.numpy as jnp
from jaxtyping import Array, Float

from .grid import center_of_mass_grid
from .sphere import spherical_geodesic_distance

__all__ = [
    'center_of_mass_points',
    'displacement_from_reference_grid',
    'displacement_from_reference_points',
    'compactness_penalty',
    # legacy aliases (removed at v0.1 cleanup)
    'cmass_coor',
    'cmass_reference_displacement_grid',
    'cmass_reference_displacement_coor',
    'diffuse',
]


def center_of_mass_points(
    weight: Float[Array, '... n_regions n_points'],
    coords: Float[Array, '... n_points ndim'],
    *,
    radius: Optional[float] = None,
) -> Float[Array, '... n_regions ndim']:
    """Weighted centre of mass for a set of regions over a point cloud.

    For each region ``r`` and dim ``d`` the result is::

        cm[r, d] = sum_p weight[r, p] * coords[p, d] / sum_p weight[r, p]

    If ``radius`` is given, the result is projected back onto a
    sphere of that radius -- useful when the underlying coordinates
    live on a sphere (the centre-of-mass off the sphere has no
    geometric meaning).

    Parameters
    ----------
    weight
        Non-negative weights, ``(..., n_regions, n_points)``.
    coords
        Per-point coordinates, ``(..., n_points, ndim)``.
    radius
        Optional sphere radius for projection.

    Returns
    -------
    Per-region centre-of-mass, ``(..., n_regions, ndim)``.
    """
    # Plain matrix multiplication once shapes are batched.
    num = jnp.matmul(weight, coords)  # (..., n_regions, ndim)
    denom = weight.sum(axis=-1, keepdims=True)  # (..., n_regions, 1)
    cm = num / denom
    if radius is not None:
        norm = jnp.linalg.norm(cm, ord=2, axis=-1, keepdims=True)
        cm = radius * cm / norm
    return cm


def displacement_from_reference_grid(
    weight: Float[Array, '...'],
    reference: Float[Array, '... ndim'],
    *,
    axes: Optional[Sequence[int]] = None,
    na_value: Optional[float] = None,
) -> Float[Array, '... ndim']:
    """Displacement of a centre-of-mass (regular grid) from a reference.

    Equivalent to ``center_of_mass_grid(weight, axes=...) - reference``;
    a convenience wrapper for the common regularisation pattern
    where a weight tensor is encouraged to have its centre of mass
    near a fixed location.

    Parameters
    ----------
    weight, axes, na_value
        Forwarded to ``center_of_mass_grid``.
    reference
        Reference coordinates, broadcastable to the centre-of-mass
        result.

    Returns
    -------
    Displacement vector ``cm - reference``.
    """
    cm = center_of_mass_grid(weight, axes=axes, na_value=na_value)
    return cm - reference


def displacement_from_reference_points(
    weight: Float[Array, '... n_regions n_points'],
    reference: Float[Array, '... n_regions ndim'],
    coords: Float[Array, '... n_points ndim'],
    *,
    radius: Optional[float] = None,
) -> Float[Array, '... n_regions ndim']:
    """Displacement of a centre-of-mass (point cloud) from a reference.

    See ``center_of_mass_points``; subtracts a per-region reference
    coordinate.
    """
    cm = center_of_mass_points(weight, coords, radius=radius)
    return cm - reference


def compactness_penalty(
    weight: Float[Array, '... n_regions n_points'],
    coords: Float[Array, '... n_points ndim'],
    *,
    norm: Union[int, float] = 2,
    floor: float = 0.0,
    radius: Optional[float] = None,
) -> Float[Array, '... n_regions']:
    """How spread-out is each region's weight from its own centre of mass?

    Returns a scalar per region: large values mean the weight is
    dispersed (incoherent region); small values mean it concentrates
    near a single locus.  The standard use is as a regulariser
    pulling region weights toward compact, contiguous supports.

    Parameters
    ----------
    weight
        Non-negative weights, ``(..., n_regions, n_points)``.
    coords
        Per-point coordinates, ``(..., n_points, ndim)``.
    norm
        ``p`` of the ``L_p`` norm used as the distance from each
        point to the region's centre of mass.  Default ``2``
        (Euclidean).
    floor
        Distances below ``floor`` are clamped to zero.  Useful
        when the centre-of-mass lies inside a coherent support
        and we want the penalty to only "punish" points truly
        outside.
    radius
        If given, the centre of mass is projected onto a sphere of
        that radius and distances are spherical geodesics.  Use
        this for region weights defined over a 2-sphere.

    Returns
    -------
    Per-region penalty, ``(..., n_regions)``.
    """
    cm = center_of_mass_points(weight, coords, radius=radius)
    if radius is None:
        # ``cm[..., r, :] - coords[..., p, :]`` -> ``(..., n_regions, n_points, ndim)``
        diff = cm[..., :, None, :] - coords[..., None, :, :]
        dist = jnp.linalg.norm(diff, ord=norm, axis=-1)
    else:
        # Spherical: pairwise geodesic between (cm, coords).
        # cm: (..., n_regions, ndim); coords: (..., n_points, ndim)
        dist = spherical_geodesic_distance(cm, coords, r=radius)
    dist = jnp.maximum(dist - floor, 0.0)
    # Penalty = mean over points of weight * dist.  ``dist`` flows from
    # ``jnp.linalg.norm`` (typed Any) on one branch; restore the type.
    return cast(Float[Array, '...'], (weight * dist).mean(axis=-1))


# ---------------------------------------------------------------------------
# Legacy-name aliases (removed at v0.1 cleanup)
# ---------------------------------------------------------------------------


cmass_coor = center_of_mass_points
cmass_reference_displacement_grid = displacement_from_reference_grid
cmass_reference_displacement_coor = displacement_from_reference_points
diffuse = compactness_penalty
