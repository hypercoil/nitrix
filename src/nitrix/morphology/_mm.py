# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Grayscale morphology via the tropical semirings.

Implementations are thin wrappers over ``semiring_conv``.  The user-
facing API is single-channel: ``x: (..., *spatial)``.  We add a
trailing ``c_in = c_out = 1`` dim before calling ``semiring_conv``
and squeeze it back on return.

Convention -- matches ``scipy.ndimage.grey_dilation`` /
``grey_erosion``:

- dilation:  ``out[i] = max_p ( x[i + p] + se[p] )``
- erosion:   ``out[i] = min_p ( x[i + p] - se[p] )``
- opening:   ``dilate(erode(x))``
- closing:   ``erode(dilate(x))``

For a flat (all-zero) structuring element, dilation and erosion
reduce to local-max / local-min sliding-window reductions.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Union, cast

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Num

from .._internal.backend import Backend
from ..semiring import (
    TROPICAL_MAX_PLUS,
    TROPICAL_MIN_PLUS,
    Semiring,
    semiring_conv,
)


__all__ = [
    'dilate',
    'erode',
    'open',
    'close',
    'distance_transform',
    'distance_transform_edt',
]


# ---------------------------------------------------------------------------
# Structuring-element helpers
# ---------------------------------------------------------------------------


def _resolve_structuring_element(
    structuring_element: Optional[Num[Array, '*kspatial']],
    size: Optional[Union[int, Sequence[int]]],
    spatial_rank: int,
    dtype: DTypeLike,
) -> Num[Array, '*kspatial']:
    '''Return a structuring element of the right rank and dtype.

    If ``structuring_element`` is given, it is used directly (after a
    rank check).  Otherwise ``size`` is interpreted as the side
    length of a flat (all-zero) hyper-rectangular structuring
    element; an ``int`` becomes ``(size,) * spatial_rank``.  If
    neither is given, defaults to a ``3``-cube.
    '''
    if structuring_element is not None:
        se = jnp.asarray(structuring_element, dtype=dtype)
        if se.ndim != spatial_rank:
            raise ValueError(
                f'structuring_element.ndim={se.ndim} must equal '
                f'spatial_rank={spatial_rank}.'
            )
        return se
    if size is None:
        size = 3
    if isinstance(size, int):
        shape = (size,) * spatial_rank
    else:
        shape = tuple(size)
        if len(shape) != spatial_rank:
            raise ValueError(
                f'size must be an int or a length-{spatial_rank} '
                f'sequence; got {size!r}.'
            )
    return jnp.zeros(shape, dtype=dtype)


def _conv_wrap(
    x: Num[Array, '... *spatial'],
    se: Num[Array, '*kspatial'],
    *,
    semiring: Semiring[Any],
    padding: str,
    backend: Backend,
) -> Num[Array, '... *spatial']:
    '''Run ``semiring_conv`` on a single-channel input.

    Adds the trailing ``c_in = 1`` to ``x``, expands the structuring
    element ``se: (*kspatial,)`` to ``k: (*kspatial, 1, 1)``, calls
    ``semiring_conv``, and squeezes the trailing dim of the output.
    '''
    spatial_rank = se.ndim
    if x.ndim < spatial_rank:
        raise ValueError(
            f'x.ndim={x.ndim} too small for spatial_rank={spatial_rank}; '
            'morphology API expects (..., *spatial) inputs.'
        )
    x_c = x[..., None]                          # (..., *spatial, 1)
    k = se.reshape(se.shape + (1, 1))           # (*kspatial, 1, 1)
    out = semiring_conv(
        x_c, k,
        semiring=semiring,
        padding=padding,
        backend=backend,
    )
    return out[..., 0]


# ---------------------------------------------------------------------------
# Public ops
# ---------------------------------------------------------------------------


def dilate(
    x: Num[Array, '... *spatial'],
    *,
    structuring_element: Optional[Num[Array, '*kspatial']] = None,
    size: Optional[Union[int, Sequence[int]]] = None,
    padding: str = 'SAME',
    backend: Backend = 'auto',
) -> Num[Array, '... *spatial']:
    '''Grayscale dilation: ``out[i] = max_p ( x[i + p] + se[p] )``.

    For a flat (all-zero) structuring element this reduces to a
    sliding-window max -- the most common case.  For a non-trivial
    ``structuring_element``, the values are added per-position before
    taking the max.

    Parameters
    ----------
    x
        Single-channel input, ``(..., *spatial)``.
    structuring_element
        Per-position offsets; shape ``(*kspatial,)`` matching the
        spatial rank of ``x``.  ``None`` selects a flat box of
        ``size``.
    size
        Side length of a flat hyper-rectangular structuring element.
        ``int`` is broadcast to all spatial dims.  Ignored if
        ``structuring_element`` is provided.  Default ``3``.
    padding
        ``"SAME"`` (default) or ``"VALID"``.  ``"SAME"`` pads with
        the algebra identity (``-inf`` for max-plus), preserving the
        spatial shape.

        **For non-Euclidean boundaries** (periodic / spherical-grid /
        custom), do not pass a new ``padding`` mode here -- instead
        pre-pad the input via the appropriate topology helper and
        call ``dilate(..., padding="VALID")``.  The VALID dilation
        consumes exactly the pad you added when ``pad == se_radius``;
        the result has the original spatial shape, no explicit
        unpad needed::

            # Spherical-grid topology (parameterised sphere):
            from nitrix.geometry import sphere_grid_pad_2d
            se_radius = (size - 1) // 2
            padded = sphere_grid_pad_2d(mask, pad=se_radius)
            result = dilate(padded, size=size, padding='VALID')

            # Generic periodic (toroidal):
            p = (size - 1) // 2
            padded = jnp.pad(mask, ((p, p), (p, p)), mode='wrap')
            result = dilate(padded, size=size, padding='VALID')

        If you padded by more than the SE radius (e.g. to chain
        multiple kernel passes), strip the surplus via
        ``sphere_grid_unpad_2d`` (for the spherical case) or a plain
        slice (for the toroidal case).

        Threading ``padding='periodic'`` into ``dilate`` itself was
        considered and rejected: it would only solve the toroidal
        case, not the pole-flip case that the spherical-grid
        primitive handles.  Composition is more general and only
        costs 2 lines.
    backend
        ``"auto"``, ``"pallas-cuda"``, or ``"jax"``.

    Returns
    -------
    Array of the same shape as ``x`` (for ``padding="SAME"``).
    '''
    spatial_rank = len(x.shape) - (x.ndim - x.ndim)  # placeholder
    # The spatial rank is inferred from the structuring element or
    # ``size``; if neither pins it we treat all of x's dims as spatial.
    if structuring_element is not None:
        spatial_rank = jnp.asarray(structuring_element).ndim
    elif isinstance(size, (tuple, list)):
        spatial_rank = len(size)
    else:
        # Default: all of x's dims are spatial.
        spatial_rank = x.ndim
    se = _resolve_structuring_element(
        structuring_element, size, spatial_rank, x.dtype,
    )
    return _conv_wrap(
        x, se,
        semiring=TROPICAL_MAX_PLUS,
        padding=padding,
        backend=backend,
    )


def erode(
    x: Num[Array, '... *spatial'],
    *,
    structuring_element: Optional[Num[Array, '*kspatial']] = None,
    size: Optional[Union[int, Sequence[int]]] = None,
    padding: str = 'SAME',
    backend: Backend = 'auto',
) -> Num[Array, '... *spatial']:
    '''Grayscale erosion: ``out[i] = min_p ( x[i + p] - se[p] )``.

    Implemented as ``semiring_conv`` with ``TROPICAL_MIN_PLUS`` and
    the negated structuring element so the algebra's ``+`` produces
    the conventional ``-``.

    Args / Returns: see ``dilate``.
    '''
    if structuring_element is not None:
        spatial_rank = jnp.asarray(structuring_element).ndim
    elif isinstance(size, (tuple, list)):
        spatial_rank = len(size)
    else:
        spatial_rank = x.ndim
    se = _resolve_structuring_element(
        structuring_element, size, spatial_rank, x.dtype,
    )
    return _conv_wrap(
        x, -se,
        semiring=TROPICAL_MIN_PLUS,
        padding=padding,
        backend=backend,
    )


def open(
    x: Num[Array, '... *spatial'],
    *,
    structuring_element: Optional[Num[Array, '*kspatial']] = None,
    size: Optional[Union[int, Sequence[int]]] = None,
    padding: str = 'SAME',
    backend: Backend = 'auto',
) -> Num[Array, '... *spatial']:
    '''Morphological opening: ``dilate(erode(x))``.

    Removes small bright structures, preserves overall shape.
    '''
    eroded = erode(
        x,
        structuring_element=structuring_element,
        size=size, padding=padding, backend=backend,
    )
    return dilate(
        eroded,
        structuring_element=structuring_element,
        size=size, padding=padding, backend=backend,
    )


def close(
    x: Num[Array, '... *spatial'],
    *,
    structuring_element: Optional[Num[Array, '*kspatial']] = None,
    size: Optional[Union[int, Sequence[int]]] = None,
    padding: str = 'SAME',
    backend: Backend = 'auto',
) -> Num[Array, '... *spatial']:
    '''Morphological closing: ``erode(dilate(x))``.

    Fills small dark holes, preserves overall shape.
    '''
    dilated = dilate(
        x,
        structuring_element=structuring_element,
        size=size, padding=padding, backend=backend,
    )
    return erode(
        dilated,
        structuring_element=structuring_element,
        size=size, padding=padding, backend=backend,
    )


# ---------------------------------------------------------------------------
# Distance transform
# ---------------------------------------------------------------------------


def _chebyshev_3x3_offsets(spatial_rank: int, dtype: DTypeLike) -> Array:
    '''3x3x... structuring element with Chebyshev (chessboard) distances.

    The center is 0, every other position is 1.  Iterated min-plus
    convolution with this kernel propagates Chebyshev distances
    one step per iteration.
    '''
    shape = (3,) * spatial_rank
    se = jnp.ones(shape, dtype=dtype)
    center = (1,) * spatial_rank
    se = se.at[center].set(0)
    return se


# ---------------------------------------------------------------------------
# Exact Euclidean distance transform (Felzenszwalb-Huttenlocher, separable)
# ---------------------------------------------------------------------------

# Finite sentinel for "no seed here": large enough that no real squared
# distance approaches it (the max squared EDT of a grid is sum of (extent-1)^2,
# << this), but small enough that ``BIG + d^2`` and ``BIG - BIG`` stay exact /
# finite in fp32.  A finite sentinel (vs ``+inf``) keeps the parabola-
# intersection arithmetic free of ``inf - inf = nan``.
_EDT_BIG = 1e10
_EDT_BIG_THRESHOLD = 1e9  # accumulated cost >= this <=> no boundary reachable


def _edt_1d_squared(f: Float[Array, 'n']) -> Float[Array, 'n']:
    '''1D squared distance transform ``D[p] = min_q ((p - q)^2 + f[q])``.

    The lower envelope of the parabolas ``(p - q)^2 + f[q]`` (one rooted at
    each ``q``, height ``f[q]``), computed in ``O(n)`` by the Felzenszwalb-
    Huttenlocher two-pass scan: pass 1 builds the envelope as a stack of
    parabola indices ``v`` and their breakpoints ``z``; pass 2 walks the
    breakpoints assigning each position to its governing parabola.  The
    stack pops are data-dependent but amortise to ``O(n)``; the fixed-size
    ``v`` / ``z`` carries make it ``jit`` / ``vmap``-able.

    ``f`` is the per-position cost: ``0`` at seeds and ``_EDT_BIG`` elsewhere
    for the first axis, the accumulated squared distance for later axes.
    '''
    n = f.shape[0]
    dtype = f.dtype
    big = jnp.asarray(_EDT_BIG, dtype)
    neg = jnp.asarray(-_EDT_BIG, dtype)
    pos = jnp.arange(n, dtype=dtype)

    def intersect(q: Array, vk: Array) -> Array:
        # Abscissa where the parabolas rooted at ``q`` and ``vk`` meet
        # (``q > vk`` always in pass 1, so the denominator is >= 2).
        return (
            (f[q] + pos[q] * pos[q]) - (f[vk] + pos[vk] * pos[vk])
        ) / (2.0 * pos[q] - 2.0 * pos[vk])

    # Pass 1 -- build the lower envelope.
    v = jnp.zeros(n, jnp.int32)
    z = jnp.full(n + 1, big, dtype).at[0].set(neg)

    def build(q: Array, carry: tuple) -> tuple:
        k, v, z = carry
        s = intersect(q, v[k])
        # Pop envelope parabolas the new one undercuts.
        k, s = lax.while_loop(
            lambda st: st[1] <= z[st[0]],
            lambda st: (st[0] - 1, intersect(q, v[st[0] - 1])),
            (k, s),
        )
        k = k + 1
        v = v.at[k].set(q)
        z = z.at[k].set(s).at[k + 1].set(big)
        return k, v, z

    k, v, z = lax.fori_loop(1, n, build, (jnp.int32(0), v, z))

    # Pass 2 -- assign each position to the parabola governing it.
    def assign(q: Array, carry: tuple) -> tuple:
        k, out = carry
        k = lax.while_loop(lambda k: z[k + 1] < pos[q], lambda k: k + 1, k)
        d = (pos[q] - pos[v[k]]) ** 2 + f[v[k]]
        return k, out.at[q].set(d)

    _, out = lax.fori_loop(
        0, n, assign, (jnp.int32(0), jnp.zeros(n, dtype))
    )
    return out


def _edt_along_axis(
    g: Float[Array, '...'], axis: int,
) -> Float[Array, '...']:
    '''Apply the 1D squared-EDT along ``axis``, vmapped over every line.'''
    g = jnp.moveaxis(g, axis, -1)
    shape = g.shape
    flat = g.reshape(-1, shape[-1])
    out = jax.vmap(_edt_1d_squared)(flat).reshape(shape)
    return jnp.moveaxis(out, -1, axis)


def _distance_transform_edt(
    mask: Num[Array, '... *spatial'],
) -> Float[Array, '... *spatial']:
    '''Exact Euclidean DT via the separable Felzenszwalb-Huttenlocher engine.

    Squared Euclidean distance separates over axes, so the exact transform is
    the sequential composition of the 1D squared-EDT along each axis; ``sqrt``
    once at the end.  Seeds are the zero positions of ``mask``; every axis is
    treated as spatial (scipy ``distance_transform_edt`` convention).
    '''
    arr = jnp.asarray(mask)
    dtype = jnp.promote_types(arr.dtype, jnp.float32)
    g = jnp.where(
        arr == 0,
        jnp.zeros((), dtype),
        jnp.asarray(_EDT_BIG, dtype),
    )
    for axis in range(arr.ndim):
        g = _edt_along_axis(g, axis)
    return jnp.where(
        g >= _EDT_BIG_THRESHOLD,
        jnp.asarray(jnp.inf, dtype),
        jnp.sqrt(g),
    )


def distance_transform(
    mask: Num[Array, '... *spatial'],
    *,
    metric: str = 'euclidean',
    structuring_element: Optional[Num[Array, '*kspatial']] = None,
    max_iters: Optional[int] = None,
    backend: Backend = 'auto',
) -> Num[Array, '... *spatial']:
    '''Distance transform: distance from each interior voxel to the boundary.

    For each output position ``i``, returns the distance to the nearest
    position where ``mask`` is zero (the "boundary").  Non-zero ``mask``
    positions are interior; zero positions are distance zero.

    Two engines, selected by ``metric``:

    - ``metric="euclidean"`` (**default**) -- **exact** Euclidean distance
      transform via the separable Felzenszwalb-Huttenlocher lower-envelope
      algorithm (``O(N * d)`` for ``N`` voxels, spatial rank ``d``).  This is
      the metric ``scipy.ndimage.distance_transform_edt`` computes; the result
      matches it to floating-point round-off.  Forward-mode differentiable;
      **not** reverse-mode differentiable (the envelope's argmin control flow
      is piecewise constant) -- use a chamfer metric below when a sub-gradient
      is needed.
    - ``metric="chebyshev"`` / ``"city_block"`` / a custom
      ``structuring_element`` -- approximate **chamfer** DT via iterated
      ``TROPICAL_MIN_PLUS`` convolution, sharing the tropical-semiring
      substrate with ``erode`` / ``dilate`` and converging in
      ``max(spatial_shape)`` iterations.  Reverse-mode differentiable.

    Parameters
    ----------
    mask
        Input, ``(..., *spatial)``.  Non-zero positions are interior.  The
        Euclidean engine treats every axis as spatial (the
        ``scipy.ndimage.distance_transform_edt`` convention); ``vmap`` to
        batch.
    metric
        ``"euclidean"`` (default, exact EDT), ``"chebyshev"`` (chessboard
        chamfer), or ``"city_block"`` (Manhattan chamfer).
    structuring_element
        Explicit per-step cost kernel for the chamfer engine.  When given,
        **selects the chamfer path regardless of** ``metric`` and is used as
        the step-cost kernel.  Center should be 0; neighbours encode their
        per-step distance contribution (1 for Chebyshev, per-axis step for
        city-block).
    max_iters
        Chamfer engine only: cap on iterations (defaults to the longest
        spatial extent, sufficient for exact convergence).  Ignored by the
        Euclidean engine, which is non-iterative.
    backend
        Chamfer engine only: ``"auto"``, ``"pallas-cuda"``, or ``"jax"``.

    Returns
    -------
    Distance map of the same spatial shape; ``+inf`` where ``mask`` is
    everywhere non-zero (no boundary reachable).

    Notes
    -----
    The Euclidean default supersedes the historical chamfer default: a
    function named ``distance_transform`` is expected to compute the Euclidean
    metric (as scipy / cupy / ITK do), and the separable engine is both exact
    and ``O(N * d)`` versus the iterative chamfer's ``O(N * D * 3**d)``.  The
    chamfer engine is retained for differentiable distance and custom
    step-cost kernels.  See ``docs/design/perf-audit-2025-05.md``.
    '''
    if structuring_element is None and metric == 'euclidean':
        return _distance_transform_edt(mask)

    spatial_rank = jnp.asarray(mask).ndim
    if structuring_element is not None:
        se = jnp.asarray(structuring_element, dtype=mask.dtype)
        spatial_rank = se.ndim
    else:
        if metric not in ('chebyshev', 'city_block'):
            raise ValueError(
                f'metric={metric!r} not in '
                '{"euclidean", "chebyshev", "city_block"}; the chamfer '
                'engine accepts "chebyshev" / "city_block", and "euclidean" '
                '(the default) computes exact EDT via a separate path.'
            )
        # The structuring element's rank is the input's rank (single-
        # channel API; all dims of x are spatial).
        spatial_rank = mask.ndim
        if metric == 'chebyshev':
            se = _chebyshev_3x3_offsets(spatial_rank, mask.dtype)
        else:  # city_block
            # Center 0; only axis-aligned neighbours are 1; corners are +inf.
            shape = (3,) * spatial_rank
            se = jnp.full(shape, jnp.inf, dtype=mask.dtype)
            center = (1,) * spatial_rank
            se = se.at[center].set(0)
            for axis in range(spatial_rank):
                for offset in (0, 2):
                    idx = list(center)
                    idx[axis] = offset
                    se = se.at[tuple(idx)].set(1)

    # Initial distance: 0 on boundary (mask == 0), +inf elsewhere.
    boundary = mask == 0
    dist = jnp.where(
        boundary,
        jnp.zeros_like(mask, dtype=jnp.float32),
        jnp.full(mask.shape, jnp.inf, dtype=jnp.float32),
    )

    if max_iters is None:
        max_iters = max(mask.shape[-spatial_rank:])

    def body(_i: int, d: Float[Array, '...']) -> Float[Array, '...']:
        return _conv_wrap(
            d, se,
            semiring=TROPICAL_MIN_PLUS,
            padding='SAME',
            backend=backend,
        )

    # ``lax.fori_loop`` is typed as returning Any; restore the array type.
    return cast(Float[Array, '...'], lax.fori_loop(0, max_iters, body, dist))


def distance_transform_edt(
    mask: Num[Array, '... *spatial'],
) -> Float[Array, '... *spatial']:
    '''Exact Euclidean distance transform (scipy ``distance_transform_edt``).

    Thin alias for ``distance_transform(mask, metric="euclidean")`` -- the
    separable Felzenszwalb-Huttenlocher engine.  Distance from each non-zero
    voxel to the nearest zero voxel; ``+inf`` where ``mask`` is everywhere
    non-zero.  Forward-mode differentiable only (see ``distance_transform``).
    '''
    return _distance_transform_edt(mask)
