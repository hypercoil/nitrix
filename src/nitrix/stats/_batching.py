# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Memory-bounded mass-univariate batching.

:func:`blocked_vmap` is the shared chunked-``vmap`` spine for the
mass-univariate fits (LME, GLM-IRLS, GAM): it maps ``fn`` over the element
axis, but in blocks, so that a brain-scale element count ``V`` need not
materialise every element's intermediates at once.  It lives here, rather than
under the linear mixed-effects code, because it is a general utility with no
mixed-effects content -- the LME, GLM, and GAM routines all route their
per-element fit through it.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

__all__ = ['blocked_vmap']


def blocked_vmap(
    fn: Callable[..., Any],
    batched: Sequence[Array],
    *,
    block: Optional[int],
) -> Any:
    """Vectorise ``fn`` over axis 0 of every array in ``batched``, in blocks.

    This is a memory-bounded variant of ``jax.vmap``.  With ``block=None`` it
    is a single ``vmap`` over the full element axis of length ``V``, identical
    to the un-chunked behaviour and HLO.  With an integer ``block`` it caps the
    number of elements whose intermediates are live at once: the batch is
    padded up to a multiple of ``block`` (replicating the first row, which is
    later discarded), reshaped to ``(n_blocks, block, ...)``, run through a
    ``jax.lax.map`` of the ``vmap``-ped ``fn``, then flattened and trimmed back
    to the original ``V`` elements.  Peak memory therefore scales with
    ``block`` rather than with ``V``.

    Parameters
    ----------
    fn : callable
        Per-element function to vectorise.  It is applied via ``jax.vmap`` and
        receives one leading-axis slice of each array in ``batched`` as its
        positional arguments.  It may return any PyTree of arrays.
    batched : sequence of Array
        Arrays sharing a common leading axis of length ``V`` (the element
        axis) that are mapped in parallel.  The remaining, trailing dimensions
        of each array are passed through to ``fn`` unchanged and may differ
        between arrays.
    block : int or None
        Maximum number of elements processed in a single vectorised call. If
        ``None`` (or greater than or equal to ``V``), all ``V`` elements are
        mapped in one ``vmap``.  Otherwise the element axis is processed in
        chunks of at most ``block`` elements to bound peak memory.

    Returns
    -------
    Any
        The result of applying ``fn`` across all ``V`` elements: a PyTree whose
        leaves each carry a leading axis of length ``V`` in the original
        element order (any block padding is removed).  The structure and
        trailing shapes of the leaves are those produced by ``fn``.
    """
    v = batched[0].shape[0]
    vfn = jax.vmap(fn)
    if block is None or block >= v:
        return vfn(*batched)

    n_blocks = -(-v // block)  # ceil(v / block)
    pad = n_blocks * block - v

    def _pad(a: Array) -> Array:
        if pad == 0:
            return a
        filler = jnp.broadcast_to(a[:1], (pad,) + a.shape[1:])
        return jnp.concatenate([a, filler], axis=0)

    reshaped = tuple(
        _pad(a).reshape((n_blocks, block) + a.shape[1:]) for a in batched
    )

    def run_block(args: Tuple[Array, ...]) -> Any:
        return vfn(*args)

    out = jax.lax.map(run_block, reshaped)
    # out is a pytree whose leaves have leading (n_blocks, block); flatten+trim.
    return jax.tree_util.tree_map(
        lambda leaf: leaf.reshape((n_blocks * block,) + leaf.shape[2:])[:v],
        out,
    )
