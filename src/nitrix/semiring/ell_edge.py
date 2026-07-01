# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Edge-functional aggregation over ELL adjacencies.

Where :func:`semiring_ell_matmul` aggregates **stored** edge values
under a semiring -- :math:`\\mathrm{out}_i = \\bigoplus_p
\\mathrm{values}_{i,p} \\otimes B_{\\mathrm{indices}_{i,p}}` -- this
module's :func:`semiring_ell_edge_aggregate` aggregates **computed**
edge messages under the same semiring:

.. code::

    e[i, p] = edge_fn(x[i], x[indices[i, p]], values[i, p], (i, j))
    out[i]  = (+)_p e[i, p]

The crucial distinction is that ``edge_fn`` is a user-supplied
callable that runs at forward time, producing a per-edge feature
vector from the source feature, the neighbour feature, and the stored
geometric weight.  This is the "edge-functional mesh convolution"
pattern of DGCNN / EdgeConv (Wang et al. 2019), the graph attention
network (Veličković et al. 2018), MoNet (Monti et al. 2017), the graph
convolutional network (Kipf & Welling 2017), and more.

The aggregation half (the :math:`\\bigoplus_p` reduction) is identical
to :func:`semiring_ell_matmul` and uses the same algebra; only the
"value-of-edge" step is functional rather than stored.

Padding semantics
-----------------

The ELL pattern pads rows below ``k_max`` with :math:`\\mathrm{values}_{i,p}
= \\mathrm{identity}` (:math:`0` under the real semiring; :math:`\\pm\\infty`
under the tropical semirings).  The user's ``edge_fn`` sees these pad
positions as ordinary edges -- the ``values`` argument is the canonical
signal of padding.  **The user is responsible** for incorporating
``values`` into the edge message in a way that the pad positions contribute
the semiring identity to the aggregate (typically by multiplying the message
by ``values``, which is :math:`0` at real-semiring pads).  This module does
not auto-mask -- doing so would conflate padding with user-supplied
zero-weight edges, which are sometimes meaningful.

Memory and cost
---------------

For ``n`` vertices, ``k_max`` neighbours per row, ``d_in`` input
features and ``d_out`` output features:

- Gathered neighbour features: ``(n, k_max, d_in)``.
- Broadcast source features: ``(n, k_max, d_in)`` (via
  ``jnp.broadcast_to``, no real allocation).
- Per-edge message tensor: ``(n, k_max, d_out)``.
- Aggregated output: ``(n, d_out)``.

So the peak device memory is :math:`O(n \\cdot k_{\\max} \\cdot
\\max(d_{\\mathrm{in}}, d_{\\mathrm{out}}))`.  There is no
:math:`O(n^2)` materialisation.  This is comparable to a single
:func:`semiring_ell_matmul`, but with the extra ``edge_fn`` evaluation
inside the inner loop.

Differentiability
-----------------

The operator is fully differentiable end-to-end.  ``edge_fn`` is any
JAX function; gradients flow back through both the gathered features
and any parameters captured in ``edge_fn``'s closure.

Backend
-------

The path is pure JAX at first cut.  A Pallas / Triton path is possible
for fixed-shape ``edge_fn`` (e.g. a small MLP) but is not currently
shipped -- the cost of inlining an arbitrary callable into Triton is
significant.  ``edge_fn`` runs in the JAX path on Ampere and newer
hardware via the standard XLA fusion.

References
----------
- Wang, Y., Sun, Y., Liu, Z., Sarma, S. E., Bronstein, M. M. and
  Solomon, J. M. (2019). Dynamic Graph CNN for Learning on Point Clouds.
  *ACM Transactions on Graphics*, 38(5), Article 146.
  https://doi.org/10.1145/3326362
- Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P. and
  Bengio, Y. (2018). Graph Attention Networks. *ICLR*.
  https://arxiv.org/abs/1710.10903
- Monti, F., Boscaini, D., Masci, J., Rodolà, E., Svoboda, J. and
  Bronstein, M. M. (2017). Geometric deep learning on graphs and
  manifolds using mixture model CNNs. *CVPR*.
  https://arxiv.org/abs/1611.08402
- Kipf, T. N. and Welling, M. (2017). Semi-Supervised Classification
  with Graph Convolutional Networks. *ICLR*.
  https://arxiv.org/abs/1609.02907
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Num

from ..sparse.ell import ELL
from ._types import Semiring
from .algebras import REAL, TROPICAL_MAX_PLUS, TROPICAL_MIN_PLUS

__all__ = ['semiring_ell_edge_aggregate', 'ell_row_softmax']


# ``edge_fn`` has two arities depending on whether ``edge_attr`` is
# supplied to ``semiring_ell_edge_aggregate``:
#
# - ``edge_attr is None`` (default): four positional args
#   ``(h_i, h_j, w, ij) -> e`` where ``w`` is the stored scalar edge
#   value ``ell.values[i, p]``.
# - ``edge_attr`` set: five positional args
#   ``(h_i, h_j, w, ij, a) -> e`` where ``a`` is the per-edge attribute
#   vector ``edge_attr[i, p, :]``.  ``w`` is still passed (it remains the
#   canonical padding signal); ``a`` carries the arbitrary per-edge
#   tensor (e.g. a GATv2 ``edge_dim`` Fourier embedding of the midpoint).
EdgeFn = Callable[..., Num[Array, 'd_out']]


def _reduce_by_semiring(
    e: Num[Array, '... n k_max d_out'],
    semiring: Semiring[Any],
    axis: int,
) -> Num[Array, '... n d_out']:
    """Aggregate an edge-message tensor along the ``k_max`` axis under a semiring.

    First-cut support is restricted to algebras whose monoid
    reduction is a single JAX axis-reduction primitive:

    - :data:`REAL` maps to ``jnp.sum``.
    - :data:`TROPICAL_MAX_PLUS` maps to ``jnp.max``.
    - :data:`TROPICAL_MIN_PLUS` maps to ``jnp.min``.

    Algebras with PyTree-state accumulators (such as the log and
    Euclidean semirings) are deferred; they need scaffolding around the
    monoid's initialisation / update / merge / finalise steps to reduce
    along an arbitrary axis, which is feasible but not yet implemented.
    Users wanting those algebras can either reduce manually after
    materialising ``e``, or wait for the follow-up.

    Parameters
    ----------
    e : Num[Array, '... n k_max d_out']
        The per-edge message tensor to reduce, with the neighbour slots
        laid out along the ``k_max`` axis.
    semiring : Semiring
        The aggregation algebra. Must be one of :data:`REAL`,
        :data:`TROPICAL_MAX_PLUS` or :data:`TROPICAL_MIN_PLUS`; any other
        algebra raises :class:`NotImplementedError`.
    axis : int
        The axis of ``e`` holding the ``k_max`` neighbour slots to reduce
        over (``-2`` for the standard ``(..., n, k_max, d_out)`` layout).

    Returns
    -------
    Num[Array, '... n d_out']
        The reduced tensor, with the ``k_max`` axis collapsed under the
        semiring's additive monoid.
    """
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
    semiring: Semiring[Any] = REAL,
    edge_attr: Optional[Num[Array, 'n k_max d_e']] = None,
) -> Num[Array, '... n d_out']:
    """Per-edge functional aggregation over an ELL adjacency.

    For each ``(i, p)`` in the ELL pattern with ``j = ell.indices[i,
    p]``, build a per-edge message via the user-supplied callable
    ``edge_fn``, then aggregate per row under the given semiring:

    .. code::

        out[i] = (+)_p edge_fn(x[i], x[indices[i, p]], values[i, p], (i, j)[, a])

    The ``edge_fn`` signature depends on ``edge_attr``: with the
    default ``edge_attr=None`` it receives the source feature, the
    neighbour feature, the stored scalar edge value, and the
    :math:`(i, j)` index pair.  When ``edge_attr`` is supplied it
    receives a fifth argument, the per-edge attribute vector
    ``edge_attr[i, p, :]``.  Use of any subset is up to the callable;
    e.g. a graph-convolution message function ignores the index pair, a
    MoNet function uses it to look up coordinates, and a GATv2 function
    folds the edge attribute into the score and the message.

    Parameters
    ----------
    edge_fn : callable
        Per-edge message function
        :math:`(h_i, h_j, w, ij) \\mapsto e` (or
        :math:`(h_i, h_j, w, ij, a) \\mapsto e` when ``edge_attr`` is
        set), where :math:`h_i` is the source feature ``(d_in,)``,
        :math:`h_j` the neighbour feature ``(d_in,)``, :math:`w` the
        stored scalar edge value, :math:`ij` the length-2 index pair,
        :math:`a` the per-edge attribute vector ``(d_e,)`` and :math:`e`
        the returned message ``(d_out,)``.  It operates on a single edge
        and is vectorised internally via ``jax.vmap`` over the
        ``(i, p)`` index pair.
    ell : ELL
        Adjacency in ELL format.  ``ell.indices`` indexes ``x``'s ``n``
        axis; ``ell.values`` carries the per-edge stored scalar
        (geometric weight, edge feature, etc.).
    x : Num[Array, '... n d_in']
        Per-vertex feature tensor, ``(..., n, d_in)``.  Leading batch
        axes broadcast through the vmap.
    semiring : Semiring, optional
        Aggregation algebra.  Default :data:`REAL` (sum); also supports
        :data:`TROPICAL_MAX_PLUS` and :data:`TROPICAL_MIN_PLUS`.  Other
        semirings raise :class:`NotImplementedError`.
    edge_attr : Num[Array, 'n k_max d_e'], optional
        Optional per-edge attribute tensor, ``(n, k_max, d_e)``,
        aligned with the ELL pattern: ``edge_attr[i, p, :]`` is the
        attribute of the edge from ``i`` to ``ell.indices[i, p]``.
        When given, ``edge_fn`` is called with a fifth positional
        argument ``a = edge_attr[i, p, :]`` in addition to the scalar
        ``w``.  This carries an arbitrary per-edge vector (e.g. the
        ``edge_dim`` Fourier embedding GATv2 mixes into the attention
        score and message) that the scalar ``w`` cannot hold.  The
        attribute is **shared across the leading batch axes of** ``x``,
        exactly like ``ell.values`` -- the common case is geometric
        edge data that does not vary per batch element.  ``None``
        (default) keeps the four-argument ``edge_fn`` contract.

    Returns
    -------
    Num[Array, '... n d_out']
        Aggregated per-vertex output, ``(..., n, d_out)``.

    Notes
    -----
    **Padding.** ELL rows below ``k_max`` carry ``values = identity``
    (:math:`0` under the real semiring).  ``edge_fn`` is called on
    padding positions too; the caller is responsible for ensuring pad
    messages contribute the semiring identity.  The standard idiom --
    multiplying the edge message by :math:`w` (the stored value) --
    gives zero contribution at real-semiring padding for free.
    Functions that do not multiply by :math:`w` (e.g. pure-attention
    readouts) must mask explicitly via ``jnp.where(w == 0, identity,
    e)``.  ``w`` is passed regardless of ``edge_attr`` for exactly this
    reason.

    **Composability examples.**

    Graph convolutional network (Kipf & Welling 2017)::

        def edge_fn(h_i, h_j, w, ij):
            return w * (W @ h_j)   # W is a learned linear layer

    EdgeConv / DGCNN (Wang et al. 2019)::

        def edge_fn(h_i, h_j, w, ij):
            return w * mlp(jnp.concatenate([h_i, h_j - h_i]))

    MoNet (Monti et al. 2017)::

        def edge_fn(h_i, h_j, w, ij):
            i, j = ij[0], ij[1]
            return gaussian_kernel(coord[i] - coord[j]) * (W @ h_j)

    GATv2 with edge features (Brody et al. 2022), using ``edge_attr`` and
    :func:`ell_row_softmax` for the attention pre-pass.  **Add self-loops
    first**: graph attention attends each vertex to itself -- the
    neighbourhood includes node ``i`` (Veličković et al. 2018) -- so
    augment the bare mesh adjacency with an ``(i, i)`` edge before the
    softmax, else every row's attention omits the self-term (a different
    operator).  :func:`~nitrix.sparse.ell_add_self_loops` adds the slot
    and gives the new edge a mean-of-incident-edges attribute::

        from nitrix.sparse import ell_add_self_loops
        ell_sl, edge_attr_sl = ell_add_self_loops(
            ell, edge_attr, fill='mean',
        )                                               # k_max -> k_max + 1
        # scores[i, p] = attention logit over (h_i, h_j, a); the per-edge
        # attribute ``a`` enters both the score and the message.
        alpha = ell_row_softmax(scores, ell_sl)        # (n, k_max + 1)
        def edge_fn(h_i, h_j, w, ij, a):
            return w * (W @ h_j + W_e @ a)              # message
        out = semiring_ell_edge_aggregate(
            edge_fn, dataclasses.replace(ell_sl, values=alpha), x,
            edge_attr=edge_attr_sl,
        )
    """
    n = x.shape[-2]
    k_max = ell.k_max

    # Gather: x[indices] -> (..., n, k_max, d_in)
    x_neigh = x[..., ell.indices, :]
    # Broadcast source: (..., n, 1, d_in) -> (..., n, k_max, d_in)
    x_src = jnp.broadcast_to(
        x[..., :, None, :],
        x_neigh.shape,
    )
    # Build (i, j) pair grid.
    i_grid = jnp.broadcast_to(jnp.arange(n)[:, None], (n, k_max))
    j_grid = ell.indices
    ij_grid = jnp.stack([i_grid, j_grid], axis=-1)  # (n, k_max, 2)

    if edge_attr is None:
        # Vectorise edge_fn over (i, p) -- inner vmap over p, outer
        # over n.  The function sees a single edge per call.
        inner = jax.vmap(edge_fn, in_axes=(0, 0, 0, 0))  # over p
        outer = jax.vmap(inner, in_axes=(0, 0, 0, 0))  # over n
        # Batch dims: x_src / x_neigh carry them; values and ij_grid
        # are shared.
        fn = outer
        for _ in range(x.ndim - 2):
            fn = jax.vmap(fn, in_axes=(0, 0, None, None))
        e = fn(x_src, x_neigh, ell.values, ij_grid)
    else:
        if edge_attr.shape[:2] != (n, k_max):
            raise ValueError(
                f'semiring_ell_edge_aggregate: edge_attr.shape='
                f'{tuple(edge_attr.shape)} must lead with (n, k_max)='
                f'{(n, k_max)} matching the ELL pattern.'
            )
        # Five-arg edge_fn: extra vmapped axis for the per-edge
        # attribute.  ``edge_attr`` is shared across batch dims, like
        # ``ell.values``.
        inner = jax.vmap(edge_fn, in_axes=(0, 0, 0, 0, 0))  # over p
        outer = jax.vmap(inner, in_axes=(0, 0, 0, 0, 0))  # over n
        fn = outer
        for _ in range(x.ndim - 2):
            fn = jax.vmap(fn, in_axes=(0, 0, None, None, None))
        e = fn(x_src, x_neigh, ell.values, ij_grid, edge_attr)
    # e shape: (..., n, k_max, d_out)

    return _reduce_by_semiring(e, semiring, axis=-2)


def ell_row_softmax(
    scores: Float[Array, '... n k_max'],
    ell: ELL,
) -> Float[Array, '... n k_max']:
    """Row-wise softmax over the ELL neighbour slots, masking padding.

    Normalises ``scores`` along the ``k_max`` (neighbour) axis so each
    row sums to 1, with the ELL's padding positions excluded from both
    the numerator and the denominator.  This is the attention
    pre-pass for GAT / GATv2-style aggregation: compute per-edge
    scores, :func:`ell_row_softmax` them into attention weights, then
    feed the weights as the ELL ``values`` (or multiply them into the
    message) of a :func:`semiring_ell_edge_aggregate` under the real
    semiring.

    Padding is identified from the ELL convention: pad slots carry
    ``ell.values == ell.identity``.  Those positions are forced to
    :math:`-\\infty` before the softmax (so they contribute :math:`0`
    after the exponential) and contribute :math:`0` to the denominator.
    A row that is entirely padding (an isolated vertex) returns all-zero
    weights rather than ``NaN``.

    Unlike the rest of this module, :func:`ell_row_softmax` takes the
    whole :class:`~nitrix.sparse.ELL` rather than a bare ``indices``
    array: it needs ``ell.values`` / ``ell.identity`` to build the
    padding mask.  When ``ell.identity`` is ``None`` (sentinels handled
    by the caller) no masking is applied and every slot participates.

    Parameters
    ----------
    scores
        Per-edge attention logits aligned with the ELL pattern,
        ``(..., n, k_max)``.  ``scores[..., i, p]`` is the score of the
        edge from ``i`` to ``ell.indices[i, p]``.
    ell
        The adjacency whose padding structure defines which slots are
        real edges.

    Returns
    -------
    Row-normalised attention weights, same shape as ``scores``; each
    real-edge row sums to 1 along the ``k_max`` axis.
    """
    if ell.identity is None:
        valid = jnp.ones(scores.shape, dtype=bool)
    else:
        valid = ell.values != ell.identity
    neg_inf = jnp.asarray(-jnp.inf, dtype=scores.dtype)
    masked = jnp.where(valid, scores, neg_inf)
    row_max = jnp.max(masked, axis=-1, keepdims=True)
    # Guard the all-pad row: row_max == -inf would make (masked - row_max)
    # nan; clamp to 0 there (the row's exp is forced to 0 by ``valid``).
    row_max = jnp.where(jnp.isfinite(row_max), row_max, 0.0)
    exp = jnp.where(valid, jnp.exp(masked - row_max), 0.0)
    denom = jnp.sum(exp, axis=-1, keepdims=True)
    denom = jnp.where(denom > 0, denom, 1.0)
    return exp / denom
