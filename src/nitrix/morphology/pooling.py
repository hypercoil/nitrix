# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Encoder-decoder pooling primitives with index-bookkeeping.

``max_pool_with_indices_nd`` records the per-window argmax position
during the pool, returning *both* the pooled values and the
per-output integer indices.  ``max_unpool_nd`` scatters values
back into a higher-resolution grid at those same positions.  The
pair is what an "indices-aware" U-Net / V-Net encoder-decoder uses
to preserve spatial localisation across the bottleneck.

Family
------

These belong with ``dilate`` and ``erode`` in
``nitrix.morphology``: both ``max_pool_with_indices_nd`` and
``dilate`` are max-window reductions; the only difference is that
pooling **strides** the window (one output per stride'd position)
while dilation keeps the spatial shape.

Channel-first layout
--------------------

PyTorch / FreeSurfer-Keras / Flax-LayerNorm convention; channel
axis precedes the spatial axes (``(*batch, C, *spatial)``).  The
caller specifies the spatial rank explicitly via ``spatial_rank``
to disambiguate batch vs channel; we don't infer.

Cross-framework parity caveat
-----------------------------

The PGlandsSeg port flagged this and the docstring repeats it
explicitly: argmax-based pooling is **fragile to cross-framework
float noise**.  When the encoder accumulates ~1e-3 max abs
difference between JAX and torch (typical for a 4-level Conv3D
cascade), the per-window argmax can flip at ~0.02-0.03% of voxels
where two neighbours are nearly equal.  The matching unpool then
scatters values to slightly different positions, producing O(10)
per-voxel raw-logit differences.  At the semantic level (per-voxel
class via argmax over channel axis) this is harmless -- the
inter-class ordering is preserved.

**The load-bearing parity check is argmax-of-output agreement,
not raw-logit allclose.**  When validating against a torch or TF
reference, compare ``argmax(out, channel_axis)``, not
``allclose(out, ref)``.

Reference
---------
Badrinarayanan, V., Kendall, A. & Cipolla, R. (2017).  *SegNet: A
deep convolutional encoder-decoder architecture for image
segmentation.*  IEEE TPAMI 39(12), 2481-2495.  The original
indices-based encoder-decoder pattern.
"""
from __future__ import annotations

from typing import Sequence, Tuple, Union, cast

import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Array, Float, Int, Num


__all__ = [
    'max_pool_with_indices_nd',
    'max_unpool_nd',
]


PoolSize = Union[int, Tuple[int, ...]]


def _resolve_pool_size(
    pool_size: PoolSize, spatial_rank: int,
) -> Tuple[int, ...]:
    if isinstance(pool_size, int):
        return (int(pool_size),) * spatial_rank
    pool_t = tuple(int(s) for s in pool_size)
    if len(pool_t) != spatial_rank:
        raise ValueError(
            f'pool_size length {len(pool_t)} must equal '
            f'spatial_rank={spatial_rank}.'
        )
    return pool_t


def max_pool_with_indices_nd(
    x: Num[Array, '*batch C *spatial'],
    *,
    pool_size: PoolSize,
    spatial_rank: int,
) -> Tuple[
    Num[Array, '*batch C *pooled'],
    Int[Array, '*batch C *pooled'],
]:
    '''N-D max pool returning ``(values, indices)``.

    For each output position, returns the max value in its
    corresponding window (per-channel) and the flattened index
    of the argmax within the original spatial grid (per-channel,
    per-output).

    Parameters
    ----------
    x
        Channel-first input, ``(..., C, *spatial)``.  Leading
        ``...`` axes are batch.
    pool_size
        Window size per spatial axis.  ``int`` = isotropic;
        ``tuple`` of length ``spatial_rank`` for per-axis.
        Spatial dimensions must be divisible by ``pool_size``
        (no padding logic; caller pads upstream if needed).
    spatial_rank
        Number of trailing spatial axes; explicit to disambiguate
        batch vs channel.

    Returns
    -------
    ``(pooled, indices)``:

    - ``pooled``: same shape as ``x`` except each spatial axis is
      divided by ``pool_size``.  Per-channel max within each window.
    - ``indices``: same shape as ``pooled``.  Each entry is the
      flattened C-order index (into the *unbatched, single-channel*
      spatial grid) of the argmax voxel for that output position.
      ``max_unpool_nd`` consumes this directly.

    Notes
    -----
    Implementation: reshape the spatial axes via "unfold" so each
    window becomes its own axis, then ``jnp.argmax`` along the
    unfolded axes (flattened) gives the within-window index.
    Combined with the window's starting voxel coordinates, this
    yields the global flat index.  Cost: ``O(N)`` in voxels with
    ``O(N)`` HBM (no per-window scan).

    Ties: when multiple voxels share the window max, returns the
    first argmax (lowest flat index within the window) -- matches
    ``jnp.argmax`` behaviour.  Cross-framework consumers should
    expect tie-breaking to differ; see the module docstring on
    parity.
    '''
    pool_t = _resolve_pool_size(pool_size, spatial_rank)
    spatial_shape = x.shape[-spatial_rank:]
    for d, p in zip(spatial_shape, pool_t):
        if d % p != 0:
            raise ValueError(
                f'spatial dim {d} not divisible by pool_size {p}; '
                'caller must pre-pad.'
            )

    # Unfold each spatial axis into (n_windows_d, window_size_d).
    # New shape: (..., C, n_w_1, w_1, n_w_2, w_2, ..., n_w_d, w_d).
    new_shape = x.shape[:-spatial_rank] + tuple(
        v for d, p in zip(spatial_shape, pool_t) for v in (d // p, p)
    )
    x_unfolded = x.reshape(new_shape)

    # Reorder axes to (..., C, n_w_1, ..., n_w_d, w_1, ..., w_d) so
    # the window-axes are trailing.
    # Source spatial axes start at position (x.ndim - spatial_rank).
    # The unfolded layout has them at strides of 2.
    n_lead = x.ndim - spatial_rank
    # current order of trailing axes: n_w_1, w_1, n_w_2, w_2, ...
    new_perm = list(range(n_lead))
    n_w_axes = [n_lead + 2 * d for d in range(spatial_rank)]
    w_axes = [n_lead + 2 * d + 1 for d in range(spatial_rank)]
    new_perm = list(range(n_lead)) + n_w_axes + w_axes
    x_reordered = jnp.transpose(x_unfolded, new_perm)
    # Shape: (..., C, *n_w, *w)

    # Flatten the trailing window axes to a single axis.
    win_total = 1
    for p in pool_t:
        win_total *= p
    pooled_shape = x_reordered.shape[: -spatial_rank]
    x_flat = x_reordered.reshape(pooled_shape + (win_total,))

    # Max + argmax within each window.
    pooled = jnp.max(x_flat, axis=-1)
    arg_within = jnp.argmax(x_flat, axis=-1).astype(jnp.int32)
    # arg_within has shape (..., C, *n_w), with values in [0, win_total).

    # Convert within-window flat index to within-window n-D coords.
    # The window flat index uses C-order: idx_d = (flat // prod(w_<d+1>)) % w_d.
    coord_within = []
    rem = arg_within
    pool_arr = jnp.asarray(pool_t, dtype=jnp.int32)
    # We need: coord_d = (arg // (prod p_{d+1:})) % p_d.
    suffix_prods = []
    cur = 1
    for p in reversed(pool_t):
        suffix_prods.append(cur)
        cur *= p
    suffix_prods.reverse()  # length spatial_rank
    for d in range(spatial_rank):
        coord_d = (arg_within // suffix_prods[d]) % pool_t[d]
        coord_within.append(coord_d)
    # Each coord_d has shape (..., C, *n_w).

    # Window-start coordinates: for each output position, the
    # window's anchor in the original spatial grid.
    # output position along axis d is n_w_d-shaped; the window
    # start is `pos_d * p_d`.
    starts = []
    for d in range(spatial_rank):
        n_w_d = spatial_shape[d] // pool_t[d]
        # Build a shape (1, ..., n_w_d, ..., 1) for broadcast.
        shape = [1] * len(pooled.shape)
        shape[n_lead + d] = n_w_d
        ax = jnp.arange(n_w_d, dtype=jnp.int32) * pool_t[d]
        starts.append(ax.reshape(shape))

    # Global per-axis coordinates = starts + coord_within.
    global_coords = [s + c for s, c in zip(starts, coord_within)]

    # Flat index in C-order over the original spatial grid.
    spatial_arr = jnp.asarray(spatial_shape, dtype=jnp.int32)
    suffix_prods_orig = []
    cur = 1
    for d in reversed(spatial_shape):
        suffix_prods_orig.append(cur)
        cur *= d
    suffix_prods_orig.reverse()
    # ``sum`` over a non-empty generator with the implicit ``0`` start
    # widens to ``Array | Literal[0]``; the spatial grid is always >=1-D
    # so the result is an Array.
    flat_indices = cast(
        Int[Array, '...'],
        sum(gc * sp for gc, sp in zip(global_coords, suffix_prods_orig)),
    )

    return pooled, flat_indices


def max_unpool_nd(
    x: Num[Array, '*batch C *pooled'],
    indices: Int[Array, '*batch C *pooled'],
    *,
    output_shape: Sequence[int],
    spatial_rank: int,
) -> Num[Array, '*batch C *spatial']:
    '''Scatter pooled values back into a higher-resolution grid.

    For each ``(batch, channel, output_position)``, write
    ``x[batch, channel, output_position]`` into the source grid
    at the flat index ``indices[batch, channel, output_position]``;
    all other positions get zero.

    Parameters
    ----------
    x
        Pooled values, ``(..., C, *pooled_spatial)``.
    indices
        Flat indices (per-channel, in C-order over the unbatched
        spatial grid) as produced by ``max_pool_with_indices_nd``.
        Same shape as ``x``.
    output_shape
        Target spatial shape (length ``spatial_rank``).  Determines
        the resolution we're unpooling into.
    spatial_rank
        Number of trailing spatial axes.

    Returns
    -------
    Unpooled tensor, ``(..., C, *output_shape)`` with the same
    leading shape as ``x``.

    Notes
    -----
    Implementation: build a flat ``(batch, C, prod(output_shape))``
    target buffer via ``.at[..., indices].set(x)``, then reshape.
    Differentiable via the standard scatter VJP; gradient flows
    back to ``x`` but not ``indices`` (a discrete index map).
    '''
    output_shape_t = tuple(int(s) for s in output_shape)
    if len(output_shape_t) != spatial_rank:
        raise ValueError(
            f'output_shape length {len(output_shape_t)} must equal '
            f'spatial_rank={spatial_rank}.'
        )

    if x.shape != indices.shape:
        raise ValueError(
            f'x.shape={x.shape} must equal indices.shape={indices.shape}.'
        )

    n_lead = x.ndim - spatial_rank   # batch + C axes
    pooled_shape = x.shape[-spatial_rank:]
    n_per_channel = 1
    for s in pooled_shape:
        n_per_channel *= s
    target_per_channel = 1
    for s in output_shape_t:
        target_per_channel *= s

    # Flatten the trailing pooled axes -> (..., C, n_per_channel).
    x_flat = x.reshape(x.shape[:-spatial_rank] + (n_per_channel,))
    idx_flat = indices.reshape(x_flat.shape)

    leading_shape = x.shape[:-spatial_rank]
    # Flatten leading axes into a single batch axis for vmap.
    n_lead_total = 1
    for s in leading_shape:
        n_lead_total *= s
    x_2d = x_flat.reshape(n_lead_total, n_per_channel)
    idx_2d = idx_flat.reshape(n_lead_total, n_per_channel)

    def _scatter_one(
        idx: Int[Array, 'n'], vals: Num[Array, 'n']
    ) -> Num[Array, 'm']:
        return jnp.zeros(
            (target_per_channel,), dtype=x.dtype
        ).at[idx].set(vals)

    out_2d = jax.vmap(_scatter_one)(idx_2d, x_2d)
    return out_2d.reshape(leading_shape + output_shape_t)
