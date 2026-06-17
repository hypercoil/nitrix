# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Cluster maps for threshold-based enhancement and inference.

Given a statistic image and a height threshold, the supra-threshold voxels form
spatial clusters (via ``morphology.connected_components``).  ``cluster_size_map``
labels every voxel with the *size* (voxel count) of its cluster;
``cluster_mass_map`` with the integrated supra-threshold statistic of its
cluster.  These are the per-threshold ingredients of TFCE (``tfce``) and the
cluster-extent / cluster-mass thresholding modes of ``randomise``.
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
    """Map each voxel to the sum of ``weights`` over its component (0 for bg)."""
    n = prod(labels.shape)
    flat_labels = labels.reshape(-1)
    flat_w = weights.reshape(-1)
    totals = jnp.zeros((n + 1,), dtype=weights.dtype).at[flat_labels].add(flat_w)
    totals = totals.at[0].set(0.0)  # background carries no mass
    return totals[labels]


def cluster_size_map(
    labels: Int[Array, '*spatial'],
) -> Float[Array, '*spatial']:
    """Per-voxel size (voxel count) of the connected component it belongs to.

    Background voxels (label ``0``) map to ``0``.
    """
    return _component_totals(labels, jnp.ones(labels.shape, dtype=jnp.float32))


def cluster_mass_map(
    labels: Int[Array, '*spatial'],
    stat: Float[Array, '*spatial'],
    threshold: float,
) -> Float[Array, '*spatial']:
    """Per-voxel cluster *mass*: the component's summed ``stat - threshold``.

    The excess statistic ``max(stat - threshold, 0)`` is summed over each
    supra-threshold component; background voxels map to ``0``.
    """
    excess = jnp.clip(stat - threshold, 0.0, None)
    return _component_totals(labels, excess)


def supra_threshold_clusters(
    stat: Float[Array, '*spatial'],
    threshold: float,
    *,
    connectivity: int = 1,
) -> Int[Array, '*spatial']:
    """Label the connected components of ``stat > threshold``."""
    mask: Bool[Array, '*spatial'] = stat > threshold
    return connected_components(mask, connectivity=connectivity)
