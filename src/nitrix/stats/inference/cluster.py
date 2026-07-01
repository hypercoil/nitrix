# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Cluster maps for threshold-based enhancement and inference.

Given a statistic image and a height threshold, the supra-threshold voxels
form spatial clusters (via :func:`connected_components`).
:func:`cluster_size_map` labels every voxel with the *size* (voxel count) of
its cluster; :func:`cluster_mass_map` with the integrated supra-threshold
statistic of its cluster.  These are the per-threshold ingredients of
threshold-free cluster enhancement (:func:`tfce`) and the cluster-extent /
cluster-mass thresholding modes of the ``randomise`` permutation driver.
"""

from __future__ import annotations

from math import prod

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from ...morphology import connected_components

__all__ = ['cluster_size_map', 'cluster_mass_map']


def _component_totals(
    labels: Int[Array, '*spatial'],
    weights: Float[Array, '*spatial'],
) -> Float[Array, '*spatial']:
    """Map each voxel to the sum of ``weights`` over its component.

    A scatter-add accumulates ``weights`` into a per-label total, then gathers
    that total back to every voxel of the component.  The background label
    (``0``) always maps to ``0``, regardless of the weights it carries.

    Parameters
    ----------
    labels
        Integer label image of shape ``(*spatial)``; ``0`` denotes background
        and ``1 .. K`` the connected components.
    weights
        Per-voxel weights of shape ``(*spatial)`` to accumulate within each
        component.

    Returns
    -------
    Float[Array, '*spatial']
        Array of shape ``(*spatial)`` in which each voxel carries the summed
        ``weights`` of its component; background voxels carry ``0``.
    """
    n = prod(labels.shape)
    flat_labels = labels.reshape(-1)
    flat_w = weights.reshape(-1)
    totals = (
        jnp.zeros((n + 1,), dtype=weights.dtype).at[flat_labels].add(flat_w)
    )
    totals = totals.at[0].set(0.0)  # background carries no mass
    return totals[labels]


def cluster_size_map(
    labels: Int[Array, '*spatial'],
) -> Float[Array, '*spatial']:
    """Per-voxel size (voxel count) of the connected component it belongs to.

    Every voxel of a component is assigned the number of voxels in that
    component; background voxels (label ``0``) map to ``0``.  The count is
    accumulated in the canonical floating dtype (``float64`` under x64) so that
    cluster sizes beyond ``2**24`` remain exactly representable.

    Parameters
    ----------
    labels
        Integer label image of shape ``(*spatial)``; ``0`` denotes background
        and ``1 .. K`` the connected components, as produced by
        :func:`connected_components`.

    Returns
    -------
    Float[Array, '*spatial']
        Array of shape ``(*spatial)`` giving the voxel count of each voxel's
        component; background voxels carry ``0``.
    """
    # ER6: count in the canonical float (f64 under x64), not a hardcoded f32 --
    # cluster sizes past 2**24 are not exactly representable in float32, eroding
    # the TFCE / cluster-extent fp64 exactness.
    return _component_totals(labels, jnp.ones(labels.shape, dtype=float))


def cluster_mass_map(
    labels: Int[Array, '*spatial'],
    stat: Float[Array, '*spatial'],
    threshold: float,
) -> Float[Array, '*spatial']:
    """Per-voxel cluster *mass*: the component's summed excess statistic.

    The excess statistic :math:`\\max(\\mathrm{stat} - \\mathrm{threshold}, 0)`
    is summed over each supra-threshold component and assigned to every voxel
    of that component; background voxels map to ``0``.

    Parameters
    ----------
    labels
        Integer label image of shape ``(*spatial)``; ``0`` denotes background
        and ``1 .. K`` the connected components, as produced by
        :func:`connected_components`.
    stat
        Statistic image of shape ``(*spatial)`` from which the excess above
        ``threshold`` is computed.
    threshold
        Height threshold subtracted from ``stat`` before clipping at zero.

    Returns
    -------
    Float[Array, '*spatial']
        Array of shape ``(*spatial)`` giving the cluster mass of each voxel's
        component; background voxels carry ``0``.
    """
    excess = jnp.clip(stat - threshold, 0.0, None)
    return _component_totals(labels, excess)


def supra_threshold_clusters(
    stat: Float[Array, '*spatial'],
    threshold: float,
    *,
    connectivity: int = 1,
) -> Int[Array, '*spatial']:
    """Label the connected components of the supra-threshold voxels.

    Thresholds the statistic image at ``threshold`` and labels the connected
    components of the resulting ``stat > threshold`` mask via
    :func:`connected_components`.

    Parameters
    ----------
    stat
        Statistic image of shape ``(*spatial)``.
    threshold
        Height threshold; voxels with ``stat > threshold`` are foreground.
    connectivity
        Neighbourhood order passed to :func:`connected_components`: ``1`` means
        face neighbours only, up to the array's dimensionality for full
        (diagonal-inclusive) connectivity.

    Returns
    -------
    Int[Array, '*spatial']
        Integer label image of shape ``(*spatial)``: ``0`` = background,
        ``1 .. K`` = the supra-threshold connected components.
    """
    mask: Bool[Array, '*spatial'] = stat > threshold
    return connected_components(mask, connectivity=connectivity)
