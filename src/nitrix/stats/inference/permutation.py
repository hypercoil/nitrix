# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Exchangeability operators for permutation inference.

Keyed pure generators that build the set of relabellings used by
``randomise``-style permutation tests:

- :func:`sign_flips` -- the :math:`\\pm 1` sign-flip matrix for symmetric /
  one-sample tests (the null is invariant to flipping an exchangeable
  observation's sign).
- :func:`permutations` -- the row-permutation index matrix for shuffle-based
  tests (two-sample, regression).

Both honour **exchangeability blocks**: with a per-observation ``blocks`` label,
sign flips act per *whole block* (a block flips together) and permutations act
*within* each block (an observation can only move to a same-block position).
The first row is always the identity (the unpermuted data), so the observed
statistic is one of the permutations -- the standard ``count / n_perm``
convention with :math:`p \\geq 1 / n_{\\mathrm{perm}}`.

The choice of random key is left to the caller; these generators are
deterministic given ``(shape, key)``.
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
    """Build a matrix of :math:`\\pm 1` sign flips for permutation inference.

    Each row of the returned ``(n_perm, n)`` matrix is one relabelling of the
    ``n`` observations under sign-flip exchangeability, suitable for symmetric
    or one-sample tests. Row 0 is always the all-ones identity (the unflipped
    data). Without ``blocks`` each observation flips independently; with
    ``blocks`` every observation sharing a block label shares that block's
    sign, so the block flips as a whole.

    Parameters
    ----------
    n : int
        Number of observations (columns of the output).
    n_perm : int
        Number of relabellings to generate, including the identity row.
    key : Array
        A ``jax.random`` key seeding the Bernoulli draws of the flip signs.
    blocks : Int[Array, 'N'], optional
        Per-observation integer exchangeability-block labels of shape ``(n,)``.
        Observations sharing a label flip together. Labels may be arbitrary
        (negative or non-contiguous) integers; they are canonicalised to a
        dense encoding so each distinct label is one independent whole-block
        flip. If ``None`` (the default), every observation flips independently.

    Returns
    -------
    Float[Array, 'n_perm N']
        A ``(n_perm, n)`` matrix whose entries are ``+1.0`` or ``-1.0``. Row 0
        is all ones (the identity); each remaining row is one sign-flip
        relabelling.
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
        ids = jnp.unique(jnp.asarray(blocks), return_inverse=True)[1].reshape(
            -1
        )
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
    """Build a permutation-index matrix for shuffle-based permutation tests.

    Each row of the returned ``(n_perm, n)`` matrix is a permutation of the
    observation indices, suitable for two-sample or regression tests. Row 0 is
    always the identity ``0, 1, ..., n - 1`` (the unpermuted data). Without
    ``blocks`` the rows are free permutations of ``0 .. n - 1``; with
    ``blocks`` each row is a *within-block* permutation, so observation ``i``
    maps only to a position sharing its block label.

    Parameters
    ----------
    n : int
        Number of observations (columns of the output).
    n_perm : int
        Number of permutations to generate, including the identity row.
    key : Array
        A ``jax.random`` key; it is split into ``n_perm`` subkeys, one per row,
        seeding the uniform noise that orders observations within each block.
    blocks : Int[Array, 'N'], optional
        Per-observation integer exchangeability-block labels of shape ``(n,)``.
        Permutations are confined within each block: an observation can only
        move to a position sharing its block label. Labels are used only as a
        sort key, so any distinct integer labels are valid. If ``None`` (the
        default), all observations form a single block (free permutation).

    Returns
    -------
    Int[Array, 'n_perm N']
        A ``(n_perm, n)`` matrix of ``int32`` indices. Row 0 is the identity;
        each remaining row is a within-block (or free) permutation of the
        observation indices.
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
