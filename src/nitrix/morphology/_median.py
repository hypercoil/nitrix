# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Median filter -- a gather-based op, deliberately *not* a semiring op.

The true median requires materialising the full neighbourhood at each
output position, because the state size for a streaming reduction is
unbounded in the window size.  For the small neighbourhoods morphology
targets (3x3 = 9 voxels, 3x3x3 = 27 voxels, mesh k-rings of tens of
neighbours) the materialisation is fine: the op gathers each window and
applies :func:`jax.numpy.nanmedian`, with no streaming kernel.

NaN-safe boundary handling: the spatial dims are padded with NaN before
the gather and :func:`jax.numpy.nanmedian` is computed over the window
axes.  This is the morphological analogue of "ignore boundary positions"
without needing an algebra identity (the median has no identity in the
semiring sense).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array, Num

__all__ = ['median_filter']


def _normalise_size(
    size: Union[int, Sequence[int]], spatial_rank: int
) -> Tuple[int, ...]:
    if isinstance(size, int):
        return (size,) * spatial_rank
    out = tuple(size)
    if len(out) != spatial_rank:
        raise ValueError(
            f'size must be an int or a length-{spatial_rank} '
            f'sequence; got {size!r}.'
        )
    return out


def _pad_lo_hi(kspatial: Tuple[int, ...]) -> Tuple[Tuple[int, int], ...]:
    """Compute SAME-style ``(lo, hi)`` padding per spatial dim.

    For each window extent the low and high pad widths sum to the extent
    minus one, splitting evenly for odd-size kernels so that the output
    retains the input spatial shape.

    Parameters
    ----------
    kspatial
        Window extent along each spatial dimension, one entry per
        spatial axis.

    Returns
    -------
    tuple of tuple of int
        A ``(lo, hi)`` pad-width pair for each spatial dimension, in the
        same order as ``kspatial``.
    """
    return tuple(((d - 1) // 2, d - 1 - (d - 1) // 2) for d in kspatial)


def median_filter(
    x: Num[Array, '... *spatial'],
    *,
    size: Optional[Union[int, Sequence[int]]] = None,
    structuring_element: Optional[Num[Array, '*kspatial']] = None,
    padding: str = 'SAME',
) -> Num[Array, '... *spatial']:
    """Apply a median filter via a NaN-safe gather and :func:`jax.numpy.nanmedian`.

    Each output position takes the median over its local window.  The
    window is materialised by gathering neighbouring positions along the
    spatial axes; excluded and out-of-bounds positions are set to ``NaN``
    so that :func:`jax.numpy.nanmedian` ignores them.

    Parameters
    ----------
    x
        Single-channel input of shape ``(..., *spatial)``.  All dims of
        ``x`` are treated as spatial unless ``size`` or
        ``structuring_element`` pins a lower rank (see :func:`dilate`
        for the same convention).
    size
        Per-spatial-dim window size: an ``int`` (broadcast across all
        spatial dims) or a sequence matching the spatial rank.  Ignored
        when ``structuring_element`` is given, which sets the window from
        its own shape.  ``None`` (the default) resolves to ``3``.
    structuring_element
        Boolean or 0-1 mask of shape ``(*kspatial,)`` selecting which
        positions of the window contribute to the median.  Excluded
        positions are set to ``NaN`` so that :func:`jax.numpy.nanmedian`
        skips them.  ``None`` uses every position in the ``size``-cube.
    padding
        ``"SAME"`` (the default) pads the spatial dims with ``NaN`` so
        that boundary positions take the median over the available
        subset and the output keeps the input spatial shape.
        ``"VALID"`` computes only over full windows and returns the
        shrunken interior.

    Returns
    -------
    Num[Array, '... *spatial']
        The filtered array.  Its spatial shape matches ``x`` when
        ``padding="SAME"`` and is reduced by ``kspatial - 1`` along each
        spatial dim when ``padding="VALID"``.  The result is promoted to
        at least float32.

    Notes
    -----
    The cost is dominated by the median sort rather than the gather, so a
    manual sort with a valid-count index is no faster.  A hardcoded
    sorting network for the fixed small window could reduce the number of
    compare-exchanges, but NaN-padded borders make the median index depend
    on the per-window valid count, so such a network would be exact only
    on the all-valid interior and borders would need a separate path.
    """
    if structuring_element is not None:
        se = jnp.asarray(structuring_element)
        kspatial = tuple(se.shape)
        spatial_rank = se.ndim
        # Boolean mask: False -> NaN sentinel, True -> 1 multiplier.
        # We don't multiply (NaN-unsafe); we use the mask later by
        # ``jnp.where``.
    else:
        if size is None:
            size = 3
        if isinstance(size, (tuple, list)):
            spatial_rank = len(size)
        else:
            spatial_rank = x.ndim
        kspatial = _normalise_size(size, spatial_rank)
        se = None

    if x.ndim < spatial_rank:
        raise ValueError(
            f'x.ndim={x.ndim} too small for spatial_rank={spatial_rank}.'
        )

    # Pad spatial dims with NaN for SAME mode.
    if padding == 'SAME':
        pad_widths = [(0, 0)] * (x.ndim - spatial_rank) + list(
            _pad_lo_hi(kspatial)
        )
        x_padded = jnp.pad(
            x.astype(jnp.result_type(x.dtype, jnp.float32)),
            pad_widths,
            mode='constant',
            constant_values=jnp.nan,
        )
    elif padding == 'VALID':
        x_padded = x.astype(jnp.result_type(x.dtype, jnp.float32))
    else:
        raise ValueError(
            f'padding={padding!r}; only "SAME" or "VALID" supported.'
        )

    # Gather patches: one ``jnp.take`` per spatial dim.
    out_spatial = tuple(
        x_padded.shape[-spatial_rank + d] - kspatial[d] + 1
        if spatial_rank > 0
        else 0
        for d in range(spatial_rank)
    )
    patches = x_padded
    for d in range(spatial_rank):
        ax = (x.ndim - spatial_rank) + 2 * d
        idx = (
            jnp.arange(out_spatial[d])[:, None]
            + jnp.arange(kspatial[d])[None, :]
        )
        patches = jnp.take(patches, idx, axis=ax)
    # Layout now: (*batch, out_0, k_0, out_1, k_1, ..., out_r-1, k_r-1).
    # Permute to (*batch, *out, *kspatial), then median over the
    # kspatial axes.
    perm = list(range(x.ndim - spatial_rank))  # batch
    for d in range(spatial_rank):
        perm.append((x.ndim - spatial_rank) + 2 * d)
    for d in range(spatial_rank):
        perm.append((x.ndim - spatial_rank) + 2 * d + 1)
    patches = jnp.transpose(patches, perm)

    # If a structuring_element mask was given, zero out excluded entries
    # by setting them to NaN.  We don't multiply (NaN-safe) -- use where.
    if se is not None:
        mask = se.astype(jnp.bool_)
        # Broadcast mask over batch + out_spatial axes.
        bcast_shape = (1,) * (x.ndim) + tuple(mask.shape)
        mask_b = jnp.broadcast_to(
            mask.reshape(bcast_shape),
            patches.shape,
        )
        patches = jnp.where(mask_b, patches, jnp.nan)

    # Median over the kspatial axes (the last ``spatial_rank`` dims).
    kspatial_axes = tuple(range(patches.ndim - spatial_rank, patches.ndim))
    return jnp.nanmedian(patches, axis=kspatial_axes)
