# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Threshold-Free Cluster Enhancement (Smith & Nichols 2009).

TFCE replaces an arbitrary cluster-forming threshold with an integral over all
thresholds, weighting each voxel by its supra-threshold cluster extent::

    TFCE(v) = integral_{h=0}^{stat(v)} extent(v, h)^E * h^H dh

where ``extent(v, h)`` is the size of the connected component containing ``v``
when the image is thresholded at ``h``.  Defaults ``E = 0.5``, ``H = 2.0`` are
the 3-D volume settings; surface / 1-D maps use different exponents and
``connectivity``.

The integral is a fixed ``n_steps`` Riemann sum over ``h in (0, max]`` (so the
step adapts to each map's range, as in FSL -- and the *shape* is fixed, hence
``vmap`` / ``scan`` friendly across permutations).  Each step is a
``connected_components`` call (jit-able, fixed-shape).  Two-sided enhancement
adds the TFCE of the negative part (disjoint support), giving a non-negative
magnitude map for the max-statistic FWE.

Built on ``morphology.connected_components`` and ``cluster.cluster_size_map``;
no cuSOLVER, no float-only restriction -- a pure spatial-statistics kernel.  It
is **not** differentiable through the discrete cluster forming (an inference
kernel, like ``connected_components`` itself).
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Bool, Float

from ...morphology import connected_components
from .cluster import cluster_size_map

__all__ = ['tfce']


def _tfce_one_sided(
    pos: Float[Array, '*spatial'],
    e: float,
    h_exp: float,
    n_steps: int,
    connectivity: int,
) -> Float[Array, '*spatial']:
    """TFCE of a non-negative map ``pos`` (the positive side)."""
    max_h = jnp.max(pos)
    dh = max_h / n_steps
    heights = jnp.arange(1, n_steps + 1, dtype=pos.dtype) * dh

    def body(
        enhanced: Float[Array, '*spatial'], h: Float[Array, '']
    ) -> tuple[Float[Array, '*spatial'], None]:
        labels = connected_components(pos > h, connectivity=connectivity)
        extent = cluster_size_map(labels)
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
) -> Float[Array, '*spatial']:
    """Threshold-free cluster enhancement of a statistic image.

    Parameters
    ----------
    stat
        Statistic image (e.g. a t-map), arbitrary spatial dimensionality.
    E, H
        Extent and height exponents (defaults ``0.5`` / ``2.0`` -- 3-D volume).
    n_steps
        Number of threshold steps in the Riemann sum (FSL default ~100).
    connectivity
        Neighbourhood order for cluster forming (``1`` = faces; ``ndim`` =
        full, incl. diagonals -- the scipy convention).
    two_sided
        Enhance the negative side as well (default ``True``); the result is the
        non-negative magnitude map used for two-sided max-statistic FWE.
    mask
        Optional spatial mask; out-of-mask voxels are zeroed before
        enhancement.

    Returns
    -------
    The (non-negative) enhanced image, same shape as ``stat``.
    """
    s = stat if mask is None else jnp.where(mask, stat, 0.0)
    enhanced = _tfce_one_sided(
        jnp.clip(s, 0.0, None), E, H, n_steps, connectivity
    )
    if two_sided:
        enhanced = enhanced + _tfce_one_sided(
            jnp.clip(-s, 0.0, None), E, H, n_steps, connectivity
        )
    return enhanced
