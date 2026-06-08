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

from typing import Any, Callable, Optional, Sequence, Union, cast

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Num

from .._internal.backend import Backend
from ..semiring import (
    TROPICAL_MAX_PLUS,
    TROPICAL_MIN_PLUS,
    Semiring,
    semiring_conv,
    semiring_matmul,
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
    """Return a structuring element of the right rank and dtype.

    If ``structuring_element`` is given, it is used directly (after a
    rank check).  Otherwise ``size`` is interpreted as the side
    length of a flat (all-zero) hyper-rectangular structuring
    element; an ``int`` becomes ``(size,) * spatial_rank``.  If
    neither is given, defaults to a ``3``-cube.
    """
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
    """Run ``semiring_conv`` on a single-channel input.

    Adds the trailing ``c_in = 1`` to ``x``, expands the structuring
    element ``se: (*kspatial,)`` to ``k: (*kspatial, 1, 1)``, calls
    ``semiring_conv``, and squeezes the trailing dim of the output.
    """
    spatial_rank = se.ndim
    if x.ndim < spatial_rank:
        raise ValueError(
            f'x.ndim={x.ndim} too small for spatial_rank={spatial_rank}; '
            'morphology API expects (..., *spatial) inputs.'
        )
    x_c = x[..., None]  # (..., *spatial, 1)
    k = se.reshape(se.shape + (1, 1))  # (*kspatial, 1, 1)
    out = semiring_conv(
        x_c,
        k,
        semiring=semiring,
        padding=padding,
        backend=backend,
    )
    return out[..., 0]


def _to_float(x: Num[Array, '...']) -> Float[Array, '...']:
    """Lift integer / boolean inputs to ``float32``; pass floating dtypes through.

    Grayscale morphology is real-valued: the flat-path window identity is the
    tropical ``-inf`` / ``+inf`` (``TROPICAL_MAX_PLUS`` / ``TROPICAL_MIN_PLUS``),
    representable only in a floating dtype.  Promoting at the op boundary gives a
    uniform ``float-in -> float-out`` contract across *both* the flat fast path
    and the explicit-structuring-element semiring path, keeps the subgradient
    well-defined, and avoids the ``-inf -> int`` overflow an integer input would
    otherwise hit.  Floating inputs (``float16`` / ``float32`` / ``float64``)
    are returned unchanged.
    """
    arr = jnp.asarray(x)
    if jnp.issubdtype(arr.dtype, jnp.floating):
        return arr
    return arr.astype(jnp.float32)


def _windowed_reduce(
    x: Float[Array, '... *spatial'],
    box: Sequence[int],
    spatial_rank: int,
    reducer: Callable[[Array, Array], Array],
    identity: float,
    padding: str,
) -> Float[Array, '... *spatial']:
    """Flat-structuring-element dilation / erosion as a fused reduce-window.

    For a flat (all-zero) structuring element, dilation reduces to a sliding-
    window max and erosion to a sliding-window min -- which ``lax.reduce_window``
    computes as a single fused kernel (XLA lowers it like a pooling op), instead
    of the ``semiring_conv`` im2col-patches + matmul path.  ``"SAME"`` padding
    pads with ``identity`` (``-inf`` for max, ``+inf`` for min) -- the tropical
    algebra identity sourced from the same ``Semiring`` the non-flat path uses,
    so the two paths' boundary handling matches bit-for-bit by construction.  The
    window is ``1`` on the leading batch dims and ``box`` on the trailing
    ``spatial_rank`` dims.

    The window init **must be a concrete scalar** (hence ``np.asarray``, not
    ``jnp.asarray``).  ``lax.reduce_window`` only routes a generic ``reducer``
    (``lax.max`` / ``lax.min``) to its differentiable specialised primitive
    (``reduce_window_max_p`` / ``reduce_window_min_p``) when JAX's monoid
    detection sees a *concrete* identity equal to the dtype's max/min identity.  A
    traced init -- which ``jnp.asarray(...)`` becomes under ``jit`` -- is not
    ``core.is_concrete``, so detection falls back to the generic
    ``reduce_window_p``, whose missing transpose rule breaks ``jit(grad(...))``.
    (B19.)
    """
    window = (1,) * (x.ndim - spatial_rank) + tuple(box)
    strides = (1,) * x.ndim
    # ``lax.reduce_window`` is typed as returning Any; restore the array type.
    return cast(
        Float[Array, '...'],
        lax.reduce_window(
            x,
            np.asarray(identity, x.dtype),
            reducer,
            window,
            strides,
            padding,
        ),
    )


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
    """Grayscale dilation: ``out[i] = max_p ( x[i + p] + se[p] )``.

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
        ``"auto"``, ``"pallas-cuda"``, or ``"jax"``.  Consulted only for the
        non-flat (explicit ``structuring_element``) path; the common flat-box
        case lowers to a fused ``lax.reduce_window`` regardless.

    Returns
    -------
    Array of the same shape as ``x`` (for ``padding="SAME"``).
    """
    # Grayscale morphology is real-valued; lift int/bool to float so the flat
    # path's ``-inf`` identity is representable and both paths share one dtype
    # contract (no-op for floating inputs).
    x = _to_float(x)
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
        structuring_element,
        size,
        spatial_rank,
        x.dtype,
    )
    if structuring_element is None:
        # Flat box (the common case): a fused sliding-window max.
        return _windowed_reduce(
            x,
            se.shape,
            spatial_rank,
            lax.max,
            TROPICAL_MAX_PLUS.identity,
            padding,
        )
    return _conv_wrap(
        x,
        se,
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
    """Grayscale erosion: ``out[i] = min_p ( x[i + p] - se[p] )``.

    Implemented as ``semiring_conv`` with ``TROPICAL_MIN_PLUS`` and
    the negated structuring element so the algebra's ``+`` produces
    the conventional ``-``.

    Args / Returns: see ``dilate``.
    """
    x = _to_float(x)
    if structuring_element is not None:
        spatial_rank = jnp.asarray(structuring_element).ndim
    elif isinstance(size, (tuple, list)):
        spatial_rank = len(size)
    else:
        spatial_rank = x.ndim
    se = _resolve_structuring_element(
        structuring_element,
        size,
        spatial_rank,
        x.dtype,
    )
    if structuring_element is None:
        # Flat box (the common case): a fused sliding-window min.
        return _windowed_reduce(
            x,
            se.shape,
            spatial_rank,
            lax.min,
            TROPICAL_MIN_PLUS.identity,
            padding,
        )
    return _conv_wrap(
        x,
        -se,
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
    """Morphological opening: ``dilate(erode(x))``.

    Removes small bright structures, preserves overall shape.
    """
    eroded = erode(
        x,
        structuring_element=structuring_element,
        size=size,
        padding=padding,
        backend=backend,
    )
    return dilate(
        eroded,
        structuring_element=structuring_element,
        size=size,
        padding=padding,
        backend=backend,
    )


def close(
    x: Num[Array, '... *spatial'],
    *,
    structuring_element: Optional[Num[Array, '*kspatial']] = None,
    size: Optional[Union[int, Sequence[int]]] = None,
    padding: str = 'SAME',
    backend: Backend = 'auto',
) -> Num[Array, '... *spatial']:
    """Morphological closing: ``erode(dilate(x))``.

    Fills small dark holes, preserves overall shape.
    """
    dilated = dilate(
        x,
        structuring_element=structuring_element,
        size=size,
        padding=padding,
        backend=backend,
    )
    return erode(
        dilated,
        structuring_element=structuring_element,
        size=size,
        padding=padding,
        backend=backend,
    )


# ---------------------------------------------------------------------------
# Distance transform
# ---------------------------------------------------------------------------


def _chebyshev_3x3_offsets(spatial_rank: int, dtype: DTypeLike) -> Array:
    """3x3x... structuring element with Chebyshev (chessboard) distances.

    The center is 0, every other position is 1.  Iterated min-plus
    convolution with this kernel propagates Chebyshev distances
    one step per iteration.
    """
    shape = (3,) * spatial_rank
    se = jnp.ones(shape, dtype=dtype)
    center = (1,) * spatial_rank
    se = se.at[center].set(0)
    return se


# ---------------------------------------------------------------------------
# Exact Euclidean distance transform (separable, via min-plus matmul)
# ---------------------------------------------------------------------------

# Finite sentinel for "no seed here": large enough that no real squared
# distance approaches it (the max squared EDT of a grid is sum of (extent-1)^2,
# << this), but small enough that ``BIG + d^2`` stays exact / finite in fp32.
# A finite sentinel (vs ``+inf``) keeps the min-plus accumulation free of
# ``inf`` propagation; positions still BIG after all axes had no reachable
# boundary and are mapped to ``+inf`` at the end.
_EDT_BIG = 1e10
_EDT_BIG_THRESHOLD = 1e9  # accumulated cost >= this <=> no boundary reachable


def _edt_along_axis(
    g: Float[Array, '...'],
    axis: int,
    *,
    backend: Backend,
) -> Float[Array, '...']:
    """Squared 1D EDT along ``axis`` as a tropical min-plus matmul.

    The 1D squared distance transform ``out[p] = min_q (g[q] + (q - p)^2)`` is
    exactly the tropical (min, +) contraction of the per-position cost ``g``
    against the squared-distance matrix ``D2[q, p] = (q - p)^2``.  Reshaping
    the off-axis dims into a batch of lines, the whole pass is one
    ``semiring_matmul`` (``(lines, n) @ (n, n)``) -- which carries the
    Pallas-CUDA streaming kernel and its JAX fallback, so the EDT inherits a
    backend-dispatched, differentiable, no-control-flow implementation instead
    of a per-line stack scan.
    """
    g = jnp.moveaxis(g, axis, -1)
    shape = g.shape
    n = shape[-1]
    pos = jnp.arange(n, dtype=g.dtype)
    d2 = (pos[:, None] - pos[None, :]) ** 2  # (n, n) squared-distance matrix
    out = semiring_matmul(
        g.reshape(-1, n),
        d2,
        semiring=TROPICAL_MIN_PLUS,
        backend=backend,
    )
    return jnp.moveaxis(out.reshape(shape), -1, axis)


def _distance_transform_edt(
    mask: Num[Array, '... *spatial'],
    *,
    backend: Backend = 'auto',
) -> Float[Array, '... *spatial']:
    """Exact Euclidean DT as a separable sequence of min-plus matmuls.

    Squared Euclidean distance separates over axes, so the exact transform is
    the sequential composition of the 1D squared-EDT (``_edt_along_axis``)
    along each axis; ``sqrt`` once at the end.  Seeds are the zero positions of
    ``mask``; every axis is treated as spatial (scipy
    ``distance_transform_edt`` convention).
    """
    arr = jnp.asarray(mask)
    dtype = jnp.promote_types(arr.dtype, jnp.float32)
    g = jnp.where(
        arr == 0,
        jnp.zeros((), dtype),
        jnp.asarray(_EDT_BIG, dtype),
    )
    for axis in range(arr.ndim):
        g = _edt_along_axis(g, axis, backend=backend)
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
    """Distance transform: distance from each interior voxel to the boundary.

    For each output position ``i``, returns the distance to the nearest
    position where ``mask`` is zero (the "boundary").  Non-zero ``mask``
    positions are interior; zero positions are distance zero.

    Two engines, selected by ``metric``:

    - ``metric="euclidean"`` (**default**) -- **exact** Euclidean distance
      transform.  Each axis pass is a tropical (min, +) matmul of the
      per-position cost against the squared-distance matrix
      ``out[p] = min_q (g[q] + (q - p)^2)`` (``semiring_matmul`` with
      ``TROPICAL_MIN_PLUS``), so the EDT runs on the semiring Pallas-CUDA
      streaming kernel (with the JAX fallback) and is reverse-mode
      differentiable through the min-plus matmul.  This is the metric
      ``scipy.ndimage.distance_transform_edt`` computes; the result matches it
      to floating-point round-off.
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
        ``"auto"``, ``"pallas-cuda"``, or ``"jax"``.  Selects the
        ``semiring_matmul`` backend for the Euclidean engine and the
        ``semiring_conv`` backend for the chamfer engine.

    Returns
    -------
    Distance map of the same spatial shape; ``+inf`` where ``mask`` is
    everywhere non-zero (no boundary reachable).

    Notes
    -----
    The Euclidean default supersedes the historical chamfer default: a
    function named ``distance_transform`` is expected to compute the Euclidean
    metric (as scipy / cupy / ITK do).  Expressing the separable EDT as a
    per-axis min-plus matmul reuses the semiring substrate's Pallas-CUDA kernel
    -- on the L4 it matches ``cupyx.scipy.ndimage.distance_transform_edt`` at
    64^3 and beats scipy on CPU, exact, where the iterative chamfer was ~80x
    slower on GPU.  The chamfer engine is retained for non-Euclidean metrics
    and custom step-cost kernels.  See ``docs/design/perf-audit-2025-05.md``.
    """
    if structuring_element is None and metric == 'euclidean':
        return _distance_transform_edt(mask, backend=backend)

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
            d,
            se,
            semiring=TROPICAL_MIN_PLUS,
            padding='SAME',
            backend=backend,
        )

    # ``lax.fori_loop`` is typed as returning Any; restore the array type.
    return cast(Float[Array, '...'], lax.fori_loop(0, max_iters, body, dist))


def distance_transform_edt(
    mask: Num[Array, '... *spatial'],
    *,
    backend: Backend = 'auto',
) -> Float[Array, '... *spatial']:
    """Exact Euclidean distance transform (scipy ``distance_transform_edt``).

    Thin alias for ``distance_transform(mask, metric="euclidean")`` -- the
    separable min-plus-matmul engine (semiring Pallas-CUDA kernel + JAX
    fallback).  Distance from each non-zero voxel to the nearest zero voxel;
    ``+inf`` where ``mask`` is everywhere non-zero.
    """
    return _distance_transform_edt(mask, backend=backend)
