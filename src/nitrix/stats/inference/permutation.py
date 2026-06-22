# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Exchangeability operators for permutation inference.

Keyed pure generators (SPEC_UPDATE_v0.5 §2) that build the set of relabellings
used by ``randomise``-style permutation tests:

- ``sign_flips`` -- the ``+/-1`` sign-flip matrix for symmetric / one-sample
  tests (the null is invariant to flipping an exchangeable observation's sign).
- ``permutations`` -- the row-permutation index matrix for shuffle-based tests
  (two-sample, regression).

Both honour **exchangeability blocks**: with a per-observation ``blocks`` label,
sign flips act per *whole block* (a block flips together) and permutations act
*within* each block (an observation can only move to a same-block position).
The first row is always the identity (the unpermuted data), so the observed
statistic is one of the permutations -- the standard ``(count)/(n_perm)``
convention with ``p >= 1/n_perm``.

RNG *policy* (which key) stays the caller's; these are deterministic given
``(shape, key)``.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

__all__ = ['sign_flips', 'permutations']


def sign_flips(
    n: int,
    n_perm: int,
    key: Array,
    *,
    blocks: Optional[Int[Array, 'N']] = None,
) -> Float[Array, 'n_perm N']:
    """``(n_perm, n)`` matrix of ``+/-1`` sign flips (row 0 = identity).

    Without ``blocks`` each observation flips independently; with ``blocks``
    every observation in a block shares the block's sign (whole-block flip).
    """
    if blocks is None:
        flips = jnp.where(
            jax.random.bernoulli(key, 0.5, (n_perm, n)), 1.0, -1.0
        )
    else:
        # Canonicalise arbitrary (possibly negative / non-contiguous) labels to a
        # dense 0..K-1 encoding so each distinct label is one independent
        # whole-block flip. Without this, a negative label wraps the column index
        # and sparse labels alias distinct blocks into one column -- silently
        # coupling exchangeability blocks (fewer effective relabellings ->
        # invalid null). sign_flips is already eager-only (int(ids.max())), so
        # jnp.unique here costs no jit-traceability. (permutations() uses the
        # labels only as a sort key and is correct for any distinct integer
        # labels, so it is left untouched to stay jit-traceable.)
        ids = jnp.unique(jnp.asarray(blocks), return_inverse=True)[1].reshape(-1)
        n_blocks = int(ids.max()) + 1
        block_flips = jnp.where(
            jax.random.bernoulli(key, 0.5, (n_perm, n_blocks)), 1.0, -1.0
        )
        flips = block_flips[:, ids]  # broadcast block sign to its members
    return flips.at[0].set(1.0)


def permutations(
    n: int,
    n_perm: int,
    key: Array,
    *,
    blocks: Optional[Int[Array, 'N']] = None,
) -> Int[Array, 'n_perm N']:
    """``(n_perm, n)`` permutation-index matrix (row 0 = identity).

    Without ``blocks`` rows are free permutations of ``0..n-1``; with ``blocks``
    each row is a *within-block* permutation (observation ``i`` maps only to a
    position sharing its block label).
    """
    keys = jax.random.split(key, n_perm)
    # blocks=None -> a single block (free permutation of all observations).
    ids = (
        jnp.zeros((n,), jnp.float32)
        if blocks is None
        else jnp.asarray(blocks).astype(jnp.float32)
    )
    arange = jnp.arange(n)
    # Identity within-block order: sort by (block, original index).
    base = jnp.argsort(ids + arange / (n + 1.0))

    def one(k: Array) -> Int[Array, 'N']:
        u = jax.random.uniform(k, (n,))
        # Random within-block order: sort by (block, uniform noise).
        order = jnp.argsort(ids + u)
        # Map the j-th block-ordered slot to a random same-block slot.
        return jnp.zeros((n,), jnp.int32).at[base].set(order.astype(jnp.int32))

    perms = jax.vmap(one)(keys)
    return perms.at[0].set(arange.astype(jnp.int32))
