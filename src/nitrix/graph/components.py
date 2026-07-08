# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Connected components of a graph / mesh.

Labels the connected components of the sub-graph induced by a boolean node mask,
over an arbitrary edge list -- the graph / mesh analogue of the regular-grid
:func:`~nitrix.morphology.connected_components`. This is the adjacency-general
cluster-forming primitive under surface / fixel threshold-free cluster
enhancement (:func:`~nitrix.stats.inference.tfce`), where the neighbourhood is a
cortical mesh or a connectivity graph rather than a voxel lattice.

Labelling is by iterated max-label propagation with pointer jumping (the scheme
:func:`~nitrix.morphology.connected_components` uses on the grid): each masked
node adopts the largest label among itself and its masked neighbours until the
labels stop changing, so the whole thing is a fixed-shape
:func:`jax.lax.while_loop` friendly to :func:`jax.vmap` across permutations.
Being a discrete labelling it is an inference kernel, not a differentiable score.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Bool, Int

__all__ = ['connected_components']


def connected_components(
    node_mask: Bool[Array, ' n_nodes'],
    edges: Int[Array, 'n_edges 2'],
) -> Int[Array, ' n_nodes']:
    """Label the connected components of a masked graph.

    Parameters
    ----------
    node_mask : Bool[Array, ' n_nodes']
        Foreground node mask; ``False`` nodes are background and are excluded
        from every component.
    edges : Int[Array, 'n_edges 2']
        Undirected edge list ``[i, j]`` (orientation ignored -- each edge is
        symmetrised internally). An edge to a background node carries no
        connectivity.

    Returns
    -------
    Int[Array, ' n_nodes']
        Per-node component label: ``0`` for background nodes, and ``1 .. K``
        contiguously for the ``K`` connected components of the foreground.
    """
    n_nodes = node_mask.shape[0]
    row = jnp.concatenate([edges[:, 0], edges[:, 1]])
    col = jnp.concatenate([edges[:, 1], edges[:, 0]])
    labels0 = jnp.where(
        node_mask, jnp.arange(1, n_nodes + 1, dtype=jnp.int32), 0
    )

    def step(labels: Int[Array, ' n_nodes']) -> Int[Array, ' n_nodes']:
        # Each node proposes to adopt the max label among its neighbours;
        # background neighbours carry label 0 and so never raise a foreground
        # node's label.
        prop = jnp.zeros((n_nodes,), labels.dtype).at[row].max(labels[col])
        return jnp.where(node_mask, jnp.maximum(labels, prop), 0)

    def cond(state: tuple[Array, Array]) -> Array:
        return state[1]

    def body(state: tuple[Array, Array]) -> tuple[Array, Array]:
        previous = state[0]
        labels = step(previous)
        # Pointer jumping: label ``l`` points at node ``l - 1`` (a node in the
        # same component); adopt the label it currently holds, doubling the
        # flood's reach per pass without crossing component boundaries.
        pointer = jnp.maximum(labels - 1, 0)
        jumped = jnp.where(node_mask, labels[pointer], 0)
        return jumped, jnp.any(jumped != previous)

    labels, _ = lax.while_loop(cond, body, (labels0, jnp.asarray(True)))

    # Renumber the surviving labels to a contiguous 1..K (background stays 0).
    used = jnp.zeros(n_nodes + 1, dtype=jnp.int32).at[labels].set(1)
    used = used.at[0].set(1)
    remap = jnp.cumsum(used) - 1
    return remap[labels].astype(jnp.int32)
