# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Memory-bounded mass-univariate batching.

``blocked_vmap`` is the shared chunked-``vmap`` spine for the mass-univariate
fits (LME / GLM-IRLS / GAM): ``vmap`` ``fn`` over the element axis, but in
blocks so a brain-scale ``V`` need not materialise every element's
intermediates at once.  Lives here, not under ``lme``, because it is a general
utility with no LME content -- LME, GLM, and GAM all route their per-element
fit through it.
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
    """``vmap`` ``fn`` over axis 0 of every array in ``batched``, in blocks.

    ``block=None`` (default) is a single ``vmap`` over all ``V`` -- identical to
    the un-chunked behaviour and HLO.  An integer ``block`` caps the number of
    elements whose intermediates are live at once: the batch is padded to a
    multiple of ``block`` (replicating the first row -- discarded), reshaped to
    ``(n_blocks, block, ...)``, run through ``lax.map`` of the ``vmap``'d
    ``fn``, then flattened and trimmed back to ``V``.  Peak memory scales with
    ``block``, not ``V``.
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
