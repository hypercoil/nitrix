# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Threshold-free cluster enhancement.

Threshold-free cluster enhancement (TFCE) replaces an arbitrary cluster-forming
threshold with an integral over all thresholds, weighting each voxel by its
supra-threshold cluster extent:

.. math::

    \\operatorname{TFCE}(v)
        = \\int_{h=0}^{\\operatorname{stat}(v)}
          \\operatorname{extent}(v, h)^{E}\\, h^{H}\\, \\mathrm{d}h

where :math:`\\operatorname{extent}(v, h)` is the size of the connected
component containing voxel :math:`v` when the image is thresholded at height
:math:`h`. The defaults :math:`E = 0.5`, :math:`H = 2.0` are the 3-D volume
settings; surface and 1-D maps use different exponents and ``connectivity``.

The integral is a fixed ``n_steps`` Riemann sum over :math:`h \\in (0, \\max]`,
so the step adapts to each map's range (as in FSL) while the *shape* stays
fixed, making the kernel friendly to :func:`jax.vmap` / :func:`jax.lax.scan`
across permutations. Each step is a single
:func:`~nitrix.morphology.connected_components` call (jit-able, fixed-shape).
Two-sided enhancement adds the TFCE of the negative part on its disjoint
support, giving a non-negative magnitude map suitable for two-sided
max-statistic family-wise error control.

Built on :func:`~nitrix.morphology.connected_components` and
:func:`~nitrix.stats.inference.cluster.cluster_size_map`. It is *not*
differentiable through the discrete cluster forming; like
:func:`~nitrix.morphology.connected_components` itself it is an inference
kernel rather than a smooth score.

References
----------
Smith, S. M., & Nichols, T. E. (2009). Threshold-free cluster enhancement:
addressing problems of smoothing, threshold dependence and localisation in
cluster inference. *NeuroImage*, 44(1), 83-98.
https://doi.org/10.1016/j.neuroimage.2008.03.061
"""

from __future__ import annotations

from typing import Callable, Optional

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Bool, Float, Int

from ...graph import connected_components as graph_connected_components
from ...morphology import connected_components
from .cluster import cluster_size_map

__all__ = ['tfce']


def _tfce_one_sided(
    pos: Float[Array, '*spatial'],
    e: float,
    h_exp: float,
    n_steps: int,
    extent_fn: Callable[[Bool[Array, '*spatial']], Float[Array, '*spatial']],
) -> Float[Array, '*spatial']:
    """
    Threshold-free cluster enhancement of a single non-negative map.

    Accumulates the TFCE integral for one side of a statistic image by summing
    the extent- and height-weighted contributions over a fixed grid of
    threshold heights. The heights are ``n_steps`` equal steps spanning
    :math:`(0, \\max]`, where :math:`\\max` is the largest value of ``pos``, so
    the step size adapts to the map's range while the loop shape stays fixed.

    Parameters
    ----------
    pos : Float[Array, '*spatial']
        Non-negative statistic map (the positive side of a statistic image),
        of arbitrary spatial dimensionality.
    e : float
        Extent exponent :math:`E` applied to the cluster size at each height.
    h_exp : float
        Height exponent :math:`H` applied to the threshold at each step.
    n_steps : int
        Number of threshold steps in the Riemann sum.
    extent_fn : callable
        Maps a supra-threshold boolean mask to a per-element cluster-extent map
        (component size at that element) -- the lattice or graph cluster-forming
        rule.

    Returns
    -------
    Float[Array, '*spatial']
        The enhanced map, same shape as ``pos``: at each voxel, the accumulated
        integral of extent :sup:`E` times height :sup:`H` over all thresholds.
    """
    max_h = jnp.max(pos)
    dh = max_h / n_steps
    heights = jnp.arange(1, n_steps + 1, dtype=pos.dtype) * dh

    def body(
        enhanced: Float[Array, '*spatial'], h: Float[Array, '']
    ) -> tuple[Float[Array, '*spatial'], None]:
        extent = extent_fn(pos > h)
        return enhanced + (extent**e) * (h**h_exp) * dh, None

    enhanced, _ = lax.scan(body, jnp.zeros_like(pos), heights)
    return enhanced


def tfce(
    stat: Float[Array, '*spatial'],
    *,
    E: float = 0.5,
    H: float = 2.0,
    n_steps: int = 100,
    connectivity: int = 1,
    two_sided: bool = True,
    mask: Optional[Bool[Array, '*spatial']] = None,
    adjacency: Optional[Int[Array, 'n_edges 2']] = None,
) -> Float[Array, '*spatial']:
    """Threshold-free cluster enhancement of a statistic image.

    Enhances either a regular-grid image (default) or -- when ``adjacency`` is
    given -- a ``(n_nodes,)`` map over an arbitrary graph / mesh (cortical
    surface, fixel connectivity), forming clusters over the supplied edge list
    instead of the voxel lattice.

    Parameters
    ----------
    stat
        Statistic image (e.g. a t-map): arbitrary spatial dimensionality for
        the grid path, or ``(n_nodes,)`` when ``adjacency`` is given.
    E, H
        Extent and height exponents (defaults ``0.5`` / ``2.0`` -- 3-D volume;
        surface maps typically use ``E = 1.0``).
    n_steps
        Number of threshold steps in the Riemann sum (FSL default ~100).
    connectivity
        Neighbourhood order for grid cluster forming (``1`` = faces; ``ndim`` =
        full, incl. diagonals -- the scipy convention). Ignored when
        ``adjacency`` is given.
    two_sided
        Enhance the negative side as well (default ``True``); the result is the
        non-negative magnitude map used for two-sided max-statistic FWE.
    mask
        Optional spatial / node mask; out-of-mask elements are zeroed before
        enhancement.
    adjacency
        Optional ``(n_edges, 2)`` undirected edge list. When given, clusters are
        the connected components of the induced sub-graph
        (:func:`~nitrix.graph.connected_components`) rather than lattice
        neighbourhoods, and ``stat`` is a ``(n_nodes,)`` node map.

    Returns
    -------
    The (non-negative) enhanced map, same shape as ``stat``.
    """
    if adjacency is None:

        def extent_fn(m: Bool[Array, '*spatial']) -> Float[Array, '*spatial']:
            return cluster_size_map(
                connected_components(m, connectivity=connectivity)
            )
    else:

        def extent_fn(m: Bool[Array, '*spatial']) -> Float[Array, '*spatial']:
            return cluster_size_map(graph_connected_components(m, adjacency))

    s = stat if mask is None else jnp.where(mask, stat, 0.0)
    enhanced = _tfce_one_sided(
        jnp.clip(s, 0.0, None), E, H, n_steps, extent_fn
    )
    if two_sided:
        enhanced = enhanced + _tfce_one_sided(
            jnp.clip(-s, 0.0, None), E, H, n_steps, extent_fn
        )
    return enhanced
