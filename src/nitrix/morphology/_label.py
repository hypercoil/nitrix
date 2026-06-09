# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Connected-components labelling (N-D, jit-able).

``connected_components`` labels the connected foreground regions of a
boolean mask; ``largest_connected_component`` returns the single biggest
one (the recurring "clean up the mask" post-processing step).

Method -- **label propagation with pointer jumping**.  Each foreground
voxel is seeded with a unique label (its flat index + 1).  Every iteration
does two things: (1) a neighbour-max hop (each voxel takes the maximum
label over itself and its neighbours), and (2) a **pointer-jumping** step
-- each voxel adopts the label currently held by the voxel its own label
points to (label ``ℓ`` → flat index ``ℓ - 1``).  Both the pointed-to voxel
and the neighbour are in the same component, so step (2) only accelerates
the flood: the reach **doubles** per pass, so a component of diameter ``d``
converges in ``O(log d)`` passes (a ``lax.while_loop`` to the fixed point)
rather than ``O(d)``.  The fixed point is identical to a pure neighbour-max
flood (every voxel ends at its component's maximum seed); pointer jumping
never merges distinct components.  A final pass renumbers the surviving
labels to a contiguous ``1 .. K``.  The label *image* has a fixed
(data-independent) shape throughout, so the whole thing is jit-able.

``connectivity`` follows the scipy convention: an offset is a neighbour
when ``sum(abs(offset)) <= connectivity``, so ``connectivity=1`` is
face-adjacency and ``connectivity=ndim`` includes every diagonal.
"""

from __future__ import annotations

import itertools
from math import prod

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int

__all__ = ['connected_components', 'largest_connected_component']


def _neighbour_offsets(ndim: int, connectivity: int) -> list[tuple[int, ...]]:
    """Non-zero ``{-1,0,1}^ndim`` offsets with ``sum|offset| <= connectivity``."""
    offsets = []
    for offset in itertools.product((-1, 0, 1), repeat=ndim):
        order = sum(abs(v) for v in offset)
        if 1 <= order <= connectivity:
            offsets.append(offset)
    return offsets


def connected_components(
    mask: Bool[Array, '*spatial'],
    *,
    connectivity: int = 1,
) -> Int[Array, '*spatial']:
    """Label connected foreground regions of ``mask``.

    Parameters
    ----------
    mask
        Boolean (or 0/1) array; non-zero entries are foreground.
    connectivity
        Neighbourhood order in ``[1, ndim]``: ``1`` = face neighbours
        only, ``ndim`` = full (including diagonals).  Follows scipy's
        ``sum(abs(offset)) <= connectivity`` rule.

    Returns
    -------
    Integer label image of the same shape: ``0`` = background,
    ``1 .. K`` = the connected components (contiguously numbered).
    """
    ndim = mask.ndim
    if not 1 <= connectivity <= ndim:
        raise ValueError(
            f'connectivity must be in [1, {ndim}]; got {connectivity}.'
        )
    shape = mask.shape
    n = prod(shape)
    mask_bool = mask.astype(bool)
    seeds = jnp.arange(1, n + 1, dtype=jnp.int32).reshape(shape)
    labels0 = jnp.where(mask_bool, seeds, 0)
    offsets = _neighbour_offsets(ndim, connectivity)

    def step(labels: Array) -> Array:
        padded = jnp.pad(labels, 1)
        best = labels
        for offset in offsets:
            sl = tuple(
                slice(1 + offset[a], 1 + offset[a] + shape[a])
                for a in range(ndim)
            )
            best = jnp.maximum(best, padded[sl])
        return jnp.where(mask_bool, best, 0)

    def cond(state: tuple[Array, Array]) -> Array:
        return state[1]

    def body(state: tuple[Array, Array]) -> tuple[Array, Array]:
        previous = state[0]
        labels = step(previous)
        # Pointer jumping: a voxel's label ``ℓ`` points to flat index
        # ``ℓ - 1`` (its seed voxel, in the same component); adopt the label
        # that voxel currently holds.  This doubles the flood's reach per
        # pass without ever crossing component boundaries.
        flat = labels.reshape(-1)
        pointer = jnp.maximum(labels - 1, 0)
        jumped = jnp.where(mask_bool, flat[pointer], 0)
        return jumped, jnp.any(jumped != previous)

    labels, _ = jax.lax.while_loop(cond, body, (labels0, jnp.asarray(True)))

    # Renumber the surviving labels to a contiguous 1..K (background 0).
    flat = labels.reshape(-1)
    used = jnp.zeros(n + 1, dtype=jnp.int32).at[flat].set(1)
    used = used.at[0].set(1)  # reserve 0 for background
    remap = jnp.cumsum(used) - 1
    return remap[flat].reshape(shape).astype(jnp.int32)


def largest_connected_component(
    mask: Bool[Array, '*spatial'],
    *,
    connectivity: int = 1,
) -> Bool[Array, '*spatial']:
    """Boolean mask of the single largest connected foreground region.

    Empty input (no foreground) yields an all-``False`` mask.
    """
    labels = connected_components(mask, connectivity=connectivity)
    n = labels.size
    counts = jnp.bincount(labels.reshape(-1), length=n + 1)
    counts = counts.at[0].set(0)  # ignore background
    has_foreground = counts.max() > 0
    largest = jnp.argmax(counts)
    return (labels == largest) & has_foreground
