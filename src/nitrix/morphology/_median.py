# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Median filter -- a gather-based op, deliberately *not* a semiring op.

Per SPEC_UPDATE §3.4, the true median requires materialising the
full neighbourhood at each output position because the state size
for a streaming reduction is unbounded in K.  For the small
neighbourhoods morphology targets (3×3 = 9 voxels, 3×3×3 = 27 voxels,
mesh k-rings of O(10s)) the materialisation is fine: implement as
``gather → jnp.nanmedian``, no streaming kernel.

NaN-safe boundary handling: we pad the spatial dims with NaN before
the gather and compute ``jnp.nanmedian`` over the kspatial axes.
This is the morphological analogue of "ignore boundary positions"
without needing an algebra identity (median has no identity in the
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
    """SAME-style ``(lo, hi)`` per spatial dim for an odd-size kernel."""
    return tuple(((d - 1) // 2, d - 1 - (d - 1) // 2) for d in kspatial)


def median_filter(
    x: Num[Array, '... *spatial'],
    *,
    size: Optional[Union[int, Sequence[int]]] = None,
    structuring_element: Optional[Num[Array, '*kspatial']] = None,
    padding: str = 'SAME',
) -> Num[Array, '... *spatial']:
    """Median filter via NaN-safe gather + ``jnp.nanmedian``.

    Parameters
    ----------
    x
        Single-channel input, ``(..., *spatial)``.  *All* dims of
        ``x`` are treated as spatial unless ``size`` / ``structuring_element``
        pin a lower rank (see ``dilate`` for the same convention).
    size
        Per-spatial-dim window size, ``int`` (broadcast) or a tuple
        matching the spatial rank.  Default ``3``.
    structuring_element
        Boolean / 0-1 mask selecting which positions of the window
        contribute to the median.  Excluded positions are filled
        with ``NaN`` so ``nanmedian`` skips them.  Shape
        ``(*kspatial,)``.  ``None`` means "use every position in the
        ``size``-cube".
    padding
        ``"SAME"`` (default) pads with ``NaN`` along the spatial
        dims so boundary positions median over the available subset.
        ``"VALID"`` returns the shrunken interior.

    Returns
    -------
    Array of the same spatial shape as ``x`` when ``padding="SAME"``.

    Notes
    -----
    Perf (profiled on the L4, 256² / size-3, ~5x behind cupy): the cost splits
    ~90 µs gather + ~210 µs ``nanmedian``, i.e. **the sort dominates** -- and a
    manual ``jnp.sort`` + valid-count index is no faster (the sort is the same).
    The only real lever is a *hardcoded sorting network* for the fixed small
    window (a median-of-K network is ~K compare-exchanges, far fewer than a
    general sort), but it is non-trivial here: NaN-padded borders make the
    median index depend on the per-window valid count, so the network is exact
    only on the all-valid interior and borders need a separate path.  Deferred
    -- and gated on first giving this case a fidelity oracle in
    ``nitrix-perf-bench`` (it is currently perf-only; see
    ``docs/feature-requests/perf-bench-case-hardening.md``), since there is no
    bench guard against a sorting-network correctness regression.
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
