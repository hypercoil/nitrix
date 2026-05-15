# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Edge-functional aggregation over ELL adjacencies.

Where ``semiring_ell_matmul`` aggregates **stored** edge values
under a semiring -- ``out[i] = (+)_p values[i, p] (*) B[indices[i,
p]]`` -- this module's ``semiring_ell_edge_aggregate`` aggregates
**computed** edge messages under the same semiring:

::

    e[i, p] = edge_fn(x[i], x[indices[i, p]], values[i, p], (i, j))
    out[i]  = (+)_p e[i, p]

The crucial distinction: ``edge_fn`` is a user-supplied callable
that runs at forward time, producing a per-edge feature vector
from the source feature, the neighbour feature, and the stored
geometric weight.  This is the "edge-functional mesh conv" pattern
of DGCNN / EdgeConv (Wang et al. 2019), GAT (Velickovic et al.
2018), MoNet (Monti et al. 2017), GCN (Kipf-Welling 2017), and
more.

The aggregation half (the ``(+)_p`` reduction) is identical to
``semiring_ell_matmul`` and uses the same algebra; only the
"value-of-edge" step is functional rather than stored.

Padding semantics
-----------------

The ELL pattern pads rows below ``k_max`` with ``values[i, p] =
identity`` (REAL: 0; TROPICAL: ±inf).  The user's ``edge_fn``
sees these pad positions as ordinary edges -- the ``values``
argument is the canonical signal of padding.  **The user is
responsible** for incorporating ``values`` into the edge message
in a way that the pad positions contribute the semiring identity
to the aggregate (typically by multiplying the message by
``values``, which is 0 at REAL pads).  This module does not
auto-mask -- doing so would conflate padding with user-supplied
zero-weight edges, which are sometimes meaningful.

Memory & cost
-------------

For ``n`` vertices, ``k_max`` neighbours per row, ``d_in`` input
features, ``d_out`` output features:

- Gathered neighbour features: ``(n, k_max, d_in)``.
- Broadcast source features: ``(n, k_max, d_in)`` (via
  ``jnp.broadcast_to``, no real allocation).
- Per-edge message tensor: ``(n, k_max, d_out)``.
- Aggregated output: ``(n, d_out)``.

So the peak HBM is ``O(n * k_max * max(d_in, d_out))``.  No
``O(n^2)`` materialisation.  Comparable to a single
``semiring_ell_matmul`` but with the extra ``edge_fn`` evaluation
inside the inner loop.

Differentiability
-----------------

Fully differentiable end-to-end.  ``edge_fn`` is any JAX
function; gradients flow back through both the gathered features
and any parameters captured in ``edge_fn``'s closure.

Backend
-------

Pure JAX at first cut.  A Pallas / Triton path is possible for
fixed-shape ``edge_fn`` (e.g., a small MLP) but not currently
shipped -- the cost of inlining an arbitrary callable into Triton
is significant.  ``edge_fn`` runs in the JAX path on Ampere+ via
the standard XLA fusion.

References
----------
- Wang et al. 2019.  *Dynamic Graph CNN for Learning on Point
  Clouds.*  ACM Trans. Graph. (the EdgeConv / DGCNN family).
- Velickovic et al. 2018.  *Graph Attention Networks.*  ICLR.
- Monti et al. 2017.  *Geometric deep learning on graphs and
  manifolds using mixture model CNNs.*  CVPR.
- Kipf & Welling 2017.  *Semi-Supervised Classification with
  Graph Convolutional Networks.*  ICLR.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Num

from ..sparse.ell import ELL
from ._types import Semiring
from .algebras import REAL, TROPICAL_MAX_PLUS, TROPICAL_MIN_PLUS


__all__ = ['semiring_ell_edge_aggregate']


EdgeFn = Callable[
    [
        Num[Array, 'd_in'],  # h_i: source vertex feature
        Num[Array, 'd_in'],  # h_j: neighbour vertex feature
        Num[Array, ''],       # w:   stored edge value (= ell.values[i, p])
        Int[Array, '2'],      # ij:  (i, j) index pair
    ],
    Num[Array, 'd_out'],
]


def _reduce_by_semiring(
    e: Num[Array, '... n k_max d_out'],
    semiring: Semiring,
    axis: int,
):
    '''Aggregate an edge-message tensor along the ``k_max`` axis under ``semiring``.

    First-cut support is restricted to algebras whose monoid
    reduction is a single JAX axis-reduction primitive:

    - ``REAL`` -> ``jnp.sum``.
    - ``TROPICAL_MAX_PLUS`` -> ``jnp.max``.
    - ``TROPICAL_MIN_PLUS`` -> ``jnp.min``.

    Algebras with pytree-state accumulators (LOG, EUCLIDEAN) are
    deferred; they need scaffolding around the monoid's ``init``
    / ``update`` / ``merge`` / ``finalize`` to reduce along an
    arbitrary axis, which is feasible but not yet implemented.
    Users wanting those algebras can either reduce manually after
    materialising ``e``, or wait for the follow-up.
    '''
    if semiring is REAL:
        return jnp.sum(e, axis=axis)
    if semiring is TROPICAL_MAX_PLUS:
        return jnp.max(e, axis=axis)
    if semiring is TROPICAL_MIN_PLUS:
        return jnp.min(e, axis=axis)
    raise NotImplementedError(
        f'semiring_ell_edge_aggregate: semiring={semiring.name!r} '
        'is not yet supported.  First-cut supports REAL / '
        'TROPICAL_MAX_PLUS / TROPICAL_MIN_PLUS only.  LOG and '
        'EUCLIDEAN have pytree-state accumulators that need '
        'scaffolding; track as a follow-up.'
    )


def semiring_ell_edge_aggregate(
    edge_fn: EdgeFn,
    ell: ELL,
    x: Num[Array, '... n d_in'],
    *,
    semiring: Semiring = REAL,
) -> Num[Array, '... n d_out']:
    '''Per-edge functional aggregation over an ELL adjacency.

    For each ``(i, p)`` in the ELL pattern with ``j = ell.indices[i,
    p]``, build a per-edge message via the user-supplied callable
    ``edge_fn(x[i], x[j], ell.values[i, p], (i, j))``, then aggregate
    per row under ``semiring``:

    .. code::

        out[i] = (+)_p edge_fn(x[i], x[indices[i, p]], values[i, p], (i, j))

    The ``edge_fn`` signature is fixed: it always receives the
    source feature, the neighbour feature, the stored edge value,
    and the ``(i, j)`` index pair.  Use of any subset is up to the
    callable; e.g., a GCN-style fn ignores the index pair, a MoNet
    fn uses it to look up coordinates.

    Parameters
    ----------
    edge_fn
        Callable ``(h_i, h_j, w, ij) -> e``.  Operates on a single
        edge; vectorised internally via ``jax.vmap`` over the
        ``(i, p)`` index pair.  See the EdgeFn type alias for the
        argument shapes.
    ell
        Adjacency in ELL format.  ``ell.indices`` indexes ``x``'s
        ``n`` axis; ``ell.values`` carries the per-edge stored
        scalar (geometric weight, edge feature, etc.).
    x
        Per-vertex feature tensor, ``(..., n, d_in)``.  Leading
        batch axes broadcast through the vmap.
    semiring
        Aggregation algebra.  Default ``REAL`` (sum); also supports
        ``TROPICAL_MAX_PLUS`` and ``TROPICAL_MIN_PLUS``.  Other
        semirings raise (see the ``_reduce_by_semiring`` docstring).

    Returns
    -------
    Aggregated per-vertex output, ``(..., n, d_out)``.

    Notes
    -----
    **Padding.**  ELL rows below ``k_max`` carry ``values = identity``
    (REAL: 0).  ``edge_fn`` is called on padding positions too; the
    caller is responsible for ensuring pad messages contribute the
    semiring identity.  The standard idiom -- multiplying the edge
    message by ``w`` (the stored value) -- gives zero contribution
    at REAL padding for free.  Functions that don't multiply by
    ``w`` (e.g., pure-attention readouts) must mask explicitly via
    ``jnp.where(w == 0, identity, e)``.

    **Composability examples.**

    GCN (Kipf-Welling)::

        def edge_fn(h_i, h_j, w, ij):
            return w * (W @ h_j)   # W is a learned linear layer

    EdgeConv / DGCNN (Wang et al.)::

        def edge_fn(h_i, h_j, w, ij):
            return w * mlp(jnp.concatenate([h_i, h_j - h_i]))

    MoNet (Monti et al.)::

        def edge_fn(h_i, h_j, w, ij):
            i, j = ij[0], ij[1]
            return gaussian_kernel(coord[i] - coord[j]) * (W @ h_j)
    '''
    n = x.shape[-2]
    k_max = ell.k_max

    # Gather: x[indices] -> (..., n, k_max, d_in)
    x_neigh = x[..., ell.indices, :]
    # Broadcast source: (..., n, 1, d_in) -> (..., n, k_max, d_in)
    x_src = jnp.broadcast_to(
        x[..., :, None, :], x_neigh.shape,
    )
    # Build (i, j) pair grid.
    i_grid = jnp.broadcast_to(jnp.arange(n)[:, None], (n, k_max))
    j_grid = ell.indices
    ij_grid = jnp.stack([i_grid, j_grid], axis=-1)  # (n, k_max, 2)

    # Vectorise edge_fn over (i, p) -- inner vmap over p, outer over n.
    # The function sees a single edge per call.
    inner = jax.vmap(edge_fn, in_axes=(0, 0, 0, 0))   # over p
    outer = jax.vmap(inner, in_axes=(0, 0, 0, 0))     # over n

    # Handle batch dims via additional vmaps.  Only x_src and
    # x_neigh have a batch axis; values and ij_grid are shared.
    fn = outer
    for _ in range(x.ndim - 2):
        fn = jax.vmap(fn, in_axes=(0, 0, None, None))

    e = fn(x_src, x_neigh, ell.values, ij_grid)
    # e shape: (..., n, k_max, d_out)

    return _reduce_by_semiring(e, semiring, axis=-2)
