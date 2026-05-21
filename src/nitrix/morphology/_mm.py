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


__all__ = ['dilate', 'erode', 'open', 'close', 'distance_transform']


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


def distance_transform(
    mask: Num[Array, '... *spatial'],
    *,
    metric: str = 'chebyshev',
    structuring_element: Optional[Num[Array, '*kspatial']] = None,
    max_iters: Optional[int] = None,
    backend: Backend = 'auto',
) -> Num[Array, '... *spatial']:
    '''Distance transform via iterative min-plus convolution.

    For each output position ``i``, returns the distance to the
    nearest position where ``mask`` is zero (the "boundary").
    Non-zero ``mask`` positions are interior; zero positions are
    distance zero.  Iterative ``TROPICAL_MIN_PLUS`` convolution with
    a step-cost structuring element converges to the exact distance
    transform in ``max(spatial_shape)`` iterations.

    Parameters
    ----------
    mask
        Input, ``(..., *spatial)``.  Non-zero positions are interior.
    metric
        Either ``"chebyshev"`` (chessboard distance, default) or
        ``"city_block"`` (Manhattan distance).  Approximate Euclidean
        is not currently implemented; the exact Felzenszwalb-Huttenlocher
        algorithm is a 1.x deferral.
    structuring_element
        Explicit per-step cost kernel.  When given, overrides
        ``metric``.  Center should be 0; neighbours should encode
        their per-step distance contribution (1 for Chebyshev,
        per-axis step for city-block).
    max_iters
        Cap on iterations.  Defaults to the longest spatial extent
        (sufficient for exact convergence).
    backend
        ``"auto"``, ``"pallas-cuda"``, or ``"jax"``.

    Returns
    -------
    Distance map of the same spatial shape; ``+inf`` where ``mask``
    is everywhere non-zero (no boundary reachable).

    Notes
    -----
    This is the **chamfer DT** (Chebyshev / Manhattan / arbitrary
    user-supplied step-cost kernel).  For **exact Euclidean DT**,
    no nitrix primitive is shipped at first GA; the standard
    reference is the Felzenszwalb-Huttenlocher 2012 parabolic
    algorithm, which scipy implements as
    ``scipy.ndimage.distance_transform_edt``.  Use scipy on host
    if you need exact EDT and don't need GPU placement /
    differentiability; a JAX-native EDT primitive
    (``distance_transform_edt``) is a planned follow-up -- see
    ``docs/design/perf-audit-2025-05.md``.

    Perf characteristics: per the 2025-05 audit, the iterative
    chamfer DT here is competitive with scipy's EDT in 3D
    (1.3-2.0× ratio depending on shape) and lags at small 2D
    (up to 15× at ``(64, 64)``).  The "lag" reflects an
    algorithmic mismatch (chamfer vs Euclidean), not a slow
    implementation; the two functions compute different metrics.
    '''
    spatial_rank = mask.ndim - (mask.ndim - jnp.asarray(mask).ndim)
    # Determine spatial rank from the metric / SE.
    if structuring_element is not None:
        se = jnp.asarray(structuring_element, dtype=mask.dtype)
        spatial_rank = se.ndim
    else:
        if metric not in ('chebyshev', 'city_block'):
            raise ValueError(
                f'metric={metric!r} not in '
                '{"chebyshev", "city_block"}; for Euclidean DT, use '
                'Felzenszwalb-Huttenlocher (not yet implemented).'
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
