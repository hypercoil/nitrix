# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Grayscale morphology via the tropical semirings.

Implementations are thin wrappers over :func:`~nitrix.semiring.semiring_conv`.
The user-facing API is single-channel: ``x: (..., *spatial)``.  A trailing
``c_in = c_out = 1`` dimension is added before calling
:func:`~nitrix.semiring.semiring_conv` and squeezed back on return.

The conventions match ``scipy.ndimage.grey_dilation`` / ``grey_erosion``:

- dilation:  :math:`\\mathrm{out}[i] = \\max_p ( x[i + p] + \\mathrm{se}[p] )`
- erosion:   :math:`\\mathrm{out}[i] = \\min_p ( x[i + p] - \\mathrm{se}[p] )`
- opening:   :func:`dilate` of :func:`erode`
- closing:   :func:`erode` of :func:`dilate`

For a flat (all-zero) structuring element, dilation and erosion reduce to
local-max / local-min sliding-window reductions.
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

    If ``structuring_element`` is given, it is used directly (after a rank
    check).  Otherwise ``size`` is interpreted as the side length of a flat
    (all-zero) hyper-rectangular structuring element; an ``int`` becomes
    ``(size,) * spatial_rank``.  If neither is given, defaults to a
    ``3``-cube.

    Parameters
    ----------
    structuring_element
        Explicit structuring element of shape ``(*kspatial,)``, or ``None``
        to construct a flat box.  When given, its rank must equal
        ``spatial_rank``.
    size
        Side length of the flat hyper-rectangular structuring element used
        when ``structuring_element`` is ``None``.  An ``int`` is broadcast to
        every spatial dimension; a sequence must have length ``spatial_rank``.
        ``None`` defaults to ``3``.
    spatial_rank
        Number of spatial dimensions the structuring element must span.
    dtype
        Target dtype of the returned structuring element.

    Returns
    -------
    Num[Array, '*kspatial']
        Structuring element of rank ``spatial_rank`` and dtype ``dtype`` --
        either the validated input or a freshly constructed flat box of
        zeros.
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
    """Run :func:`~nitrix.semiring.semiring_conv` on a single-channel input.

    Adds the trailing ``c_in = 1`` to ``x``, expands the structuring element
    ``se: (*kspatial,)`` to ``k: (*kspatial, 1, 1)``, calls
    :func:`~nitrix.semiring.semiring_conv`, and squeezes the trailing
    dimension of the output.

    Parameters
    ----------
    x
        Single-channel input of shape ``(..., *spatial)``.
    se
        Structuring element of shape ``(*kspatial,)`` whose rank sets the
        spatial rank of the convolution.
    semiring
        Algebra over which the convolution reduces (e.g.
        :data:`~nitrix.semiring.TROPICAL_MAX_PLUS` for dilation).
    padding
        Padding mode passed through to
        :func:`~nitrix.semiring.semiring_conv` (``"SAME"`` or ``"VALID"``).
    backend
        Execution backend forwarded to :func:`~nitrix.semiring.semiring_conv`
        (``"auto"``, ``"pallas-cuda"``, or ``"jax"``).

    Returns
    -------
    Num[Array, '... *spatial']
        Convolved single-channel output with the trailing channel dimension
        removed; spatial shape follows from ``padding``.
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
    tropical :math:`-\\infty` / :math:`+\\infty`
    (:data:`~nitrix.semiring.TROPICAL_MAX_PLUS` /
    :data:`~nitrix.semiring.TROPICAL_MIN_PLUS`), representable only in a
    floating dtype.  Promoting at the op boundary gives a uniform
    ``float-in -> float-out`` contract across *both* the flat fast path and the
    explicit-structuring-element semiring path, keeps the subgradient
    well-defined, and avoids the :math:`-\\infty \\to \\mathrm{int}` overflow an
    integer input would otherwise hit.  Floating inputs (``float16`` /
    ``float32`` / ``float64``) are returned unchanged.

    Parameters
    ----------
    x
        Input array of any numeric dtype.

    Returns
    -------
    Float[Array, '...']
        The same array unchanged if already floating, otherwise cast to
        ``float32``.
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
    of the :func:`~nitrix.semiring.semiring_conv` im2col-patches + matmul path.
    ``"SAME"`` padding pads with ``identity`` (:math:`-\\infty` for max,
    :math:`+\\infty` for min) -- the tropical algebra identity sourced from the
    same :class:`~nitrix.semiring.Semiring` the non-flat path uses, so the two
    paths' boundary handling matches bit-for-bit by construction.  The window is
    ``1`` on the leading batch dimensions and ``box`` on the trailing
    ``spatial_rank`` dimensions.

    The window init **must be a concrete scalar** (hence ``np.asarray``, not
    ``jnp.asarray``).  ``lax.reduce_window`` only routes a generic ``reducer``
    (``lax.max`` / ``lax.min``) to its differentiable specialised primitive
    (``reduce_window_max_p`` / ``reduce_window_min_p``) when JAX's monoid
    detection sees a *concrete* identity equal to the dtype's max/min identity.  A
    traced init -- which ``jnp.asarray(...)`` becomes under ``jit`` -- is not
    ``core.is_concrete``, so detection falls back to the generic
    ``reduce_window_p``, whose missing transpose rule breaks ``jit(grad(...))``.

    Parameters
    ----------
    x
        Floating single-channel input of shape ``(..., *spatial)``.
    box
        Per-axis window side lengths for the trailing ``spatial_rank``
        spatial dimensions.
    spatial_rank
        Number of trailing spatial dimensions the window spans; leading
        dimensions are treated as batch and windowed with size ``1``.
    reducer
        Binary reduction applied over each window (``lax.max`` for dilation,
        ``lax.min`` for erosion).
    identity
        Concrete scalar identity of ``reducer`` used both as the window
        initial value and as the ``"SAME"`` padding fill (:math:`-\\infty` for
        max, :math:`+\\infty` for min).
    padding
        Padding mode passed to ``lax.reduce_window`` (``"SAME"`` or
        ``"VALID"``).

    Returns
    -------
    Float[Array, '... *spatial']
        Windowed reduction of ``x``; spatial shape follows from ``padding``.
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
    """Grayscale dilation :math:`\\mathrm{out}[i] = \\max_p ( x[i + p] + \\mathrm{se}[p] )`.

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
        :func:`~nitrix.geometry.sphere_grid_unpad_2d` (for the spherical
        case) or a plain slice (for the toroidal case).

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
    """Grayscale erosion :math:`\\mathrm{out}[i] = \\min_p ( x[i + p] - \\mathrm{se}[p] )`.

    Implemented as :func:`~nitrix.semiring.semiring_conv` with
    :data:`~nitrix.semiring.TROPICAL_MIN_PLUS` and the negated structuring
    element so the algebra's ``+`` produces the conventional ``-``.  For a
    flat (all-zero) structuring element this reduces to a sliding-window min.

    Parameters
    ----------
    x
        Single-channel input, ``(..., *spatial)``.
    structuring_element
        Per-position offsets; shape ``(*kspatial,)`` matching the spatial rank
        of ``x``.  ``None`` selects a flat box of ``size``.
    size
        Side length of a flat hyper-rectangular structuring element.  An
        ``int`` is broadcast to all spatial dimensions.  Ignored if
        ``structuring_element`` is provided.  Default ``3``.
    padding
        ``"SAME"`` (default) or ``"VALID"``.  ``"SAME"`` pads with the algebra
        identity (:math:`+\\infty` for min-plus), preserving the spatial shape.
        See :func:`dilate` for the recipe on non-Euclidean (periodic /
        spherical-grid) boundaries.
    backend
        ``"auto"``, ``"pallas-cuda"``, or ``"jax"``.  Consulted only for the
        non-flat (explicit ``structuring_element``) path; the common flat-box
        case lowers to a fused ``lax.reduce_window`` regardless.

    Returns
    -------
    Num[Array, '... *spatial']
        Eroded array of the same shape as ``x`` (for ``padding="SAME"``).

    See Also
    --------
    dilate : Grayscale dilation (the dual operation).
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
    """Morphological opening: :func:`dilate` of :func:`erode` of ``x``.

    Erosion followed by dilation with the same structuring element.  Removes
    small bright structures while preserving the overall shape.

    Parameters
    ----------
    x
        Single-channel input, ``(..., *spatial)``.
    structuring_element
        Per-position offsets; shape ``(*kspatial,)`` matching the spatial rank
        of ``x``.  ``None`` selects a flat box of ``size``.
    size
        Side length of a flat hyper-rectangular structuring element.  An
        ``int`` is broadcast to all spatial dimensions.  Ignored if
        ``structuring_element`` is provided.  Default ``3``.
    padding
        ``"SAME"`` (default) or ``"VALID"``, applied to both the erosion and
        dilation passes.
    backend
        ``"auto"``, ``"pallas-cuda"``, or ``"jax"``; forwarded to both passes.

    Returns
    -------
    Num[Array, '... *spatial']
        Opened array of the same shape as ``x`` (for ``padding="SAME"``).
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
    """Morphological closing: :func:`erode` of :func:`dilate` of ``x``.

    Dilation followed by erosion with the same structuring element.  Fills
    small dark holes while preserving the overall shape.

    Parameters
    ----------
    x
        Single-channel input, ``(..., *spatial)``.
    structuring_element
        Per-position offsets; shape ``(*kspatial,)`` matching the spatial rank
        of ``x``.  ``None`` selects a flat box of ``size``.
    size
        Side length of a flat hyper-rectangular structuring element.  An
        ``int`` is broadcast to all spatial dimensions.  Ignored if
        ``structuring_element`` is provided.  Default ``3``.
    padding
        ``"SAME"`` (default) or ``"VALID"``, applied to both the dilation and
        erosion passes.
    backend
        ``"auto"``, ``"pallas-cuda"``, or ``"jax"``; forwarded to both passes.

    Returns
    -------
    Num[Array, '... *spatial']
        Closed array of the same shape as ``x`` (for ``padding="SAME"``).
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

    The centre is 0, every other position is 1.  Iterated min-plus
    convolution with this kernel propagates Chebyshev distances one step per
    iteration.

    Parameters
    ----------
    spatial_rank
        Number of spatial dimensions; the returned kernel has shape
        ``(3,) * spatial_rank``.
    dtype
        Target dtype of the returned kernel.

    Returns
    -------
    Array
        Structuring element of shape ``(3,) * spatial_rank`` with 0 at the
        centre and 1 everywhere else.
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


def _resolve_sampling(
    sampling: Optional[Union[float, Sequence[float]]],
    ndim: int,
) -> tuple[float, ...]:
    """Resolve ``sampling`` to a per-axis voxel-spacing tuple (scipy semantics).

    ``None`` -> unit spacing on every axis; a scalar -> that spacing on every
    axis; a sequence -> per-axis (must match ``ndim``).

    Parameters
    ----------
    sampling
        ``None`` (unit spacing), a scalar (isotropic spacing), or a per-axis
        sequence of spacings that must have length ``ndim``.
    ndim
        Number of spatial axes to produce a spacing for.

    Returns
    -------
    tuple of float
        Per-axis voxel spacing, one entry per axis, in axis order.
    """
    if sampling is None:
        return (1.0,) * ndim
    if isinstance(sampling, (int, float)):
        return (float(sampling),) * ndim
    spac = tuple(float(s) for s in sampling)
    if len(spac) != ndim:
        raise ValueError(
            f'distance_transform: sampling has {len(spac)} entries but the '
            f'input is {ndim}-D; give one spacing per axis (or a scalar).'
        )
    return spac


def _edt_along_axis(
    g: Float[Array, '...'],
    axis: int,
    *,
    spacing: float = 1.0,
    backend: Backend,
) -> Float[Array, '...']:
    """Squared 1D EDT along ``axis`` as a tropical min-plus matmul.

    The 1D squared distance transform
    :math:`\\mathrm{out}[p] = \\min_q ( g[q] + (\\mathrm{spacing} \\cdot (q - p))^2 )`
    is exactly the tropical (min, +) contraction of the per-position cost
    ``g`` against the squared-distance matrix
    :math:`\\mathrm{D2}[q, p] = (\\mathrm{spacing} \\cdot (q - p))^2`.
    Reshaping the off-axis dimensions into a batch of lines, the whole pass is
    one :func:`~nitrix.semiring.semiring_matmul` (``(lines, n) @ (n, n)``) --
    which carries the Pallas-CUDA streaming kernel and its JAX fallback, so the
    EDT inherits a backend-dispatched, differentiable, no-control-flow
    implementation instead of a per-line stack scan.  ``spacing`` is the
    per-axis voxel size (the anisotropic EDT); ``spacing == 1.0`` is
    byte-identical to the unit grid.

    Parameters
    ----------
    g
        Per-position cost array; the pass runs over ``axis``, treating the
        remaining dimensions as a batch of lines.
    axis
        Axis along which to compute the 1D squared distance transform.
    spacing
        Voxel spacing along ``axis``.  Default ``1.0`` (unit grid).
    backend
        Execution backend forwarded to
        :func:`~nitrix.semiring.semiring_matmul` (``"auto"``,
        ``"pallas-cuda"``, or ``"jax"``).

    Returns
    -------
    Float[Array, '...']
        Array of the same shape as ``g`` holding the squared distance
        transform along ``axis``.
    """
    g = jnp.moveaxis(g, axis, -1)
    shape = g.shape
    n = shape[-1]
    pos = jnp.arange(n, dtype=g.dtype)
    delta = jnp.asarray(spacing, g.dtype) * (pos[:, None] - pos[None, :])
    d2 = delta**2  # (n, n) squared (anisotropic) distance matrix
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
    sampling: Optional[Union[float, Sequence[float]]] = None,
    backend: Backend = 'auto',
) -> Float[Array, '... *spatial']:
    """Exact Euclidean DT as a separable sequence of min-plus matmuls.

    Squared Euclidean distance separates over axes, so the exact transform is
    the sequential composition of the 1D squared-EDT (:func:`_edt_along_axis`)
    along each axis; ``sqrt`` once at the end.  Seeds are the zero positions of
    ``mask``; every axis is treated as spatial (the scipy
    ``distance_transform_edt`` convention).  ``sampling`` is the per-axis voxel
    spacing (anisotropic grids); each axis pass scales by its own spacing, so
    the separable composition yields the exact anisotropic squared distance
    :math:`\\sum_{\\mathrm{axis}} (\\mathrm{spacing}_{\\mathrm{axis}} \\cdot \\Delta_{\\mathrm{axis}})^2`.

    Parameters
    ----------
    mask
        Input array, ``(..., *spatial)``.  Zero positions are seeds (distance
        zero); every axis is treated as spatial.
    sampling
        Per-axis voxel spacing: ``None`` (unit spacing), a scalar (isotropic),
        or one value per axis.
    backend
        Execution backend forwarded to the per-axis
        :func:`~nitrix.semiring.semiring_matmul` (``"auto"``,
        ``"pallas-cuda"``, or ``"jax"``).

    Returns
    -------
    Float[Array, '... *spatial']
        Euclidean distance map of the same spatial shape as ``mask``;
        :math:`+\\infty` where ``mask`` is everywhere non-zero (no seed
        reachable).
    """
    arr = jnp.asarray(mask)
    dtype = jnp.promote_types(arr.dtype, jnp.float32)
    spac = _resolve_sampling(sampling, arr.ndim)
    g = jnp.where(
        arr == 0,
        jnp.zeros((), dtype),
        jnp.asarray(_EDT_BIG, dtype),
    )
    for axis in range(arr.ndim):
        g = _edt_along_axis(g, axis, spacing=spac[axis], backend=backend)
    return jnp.where(
        g >= _EDT_BIG_THRESHOLD,
        jnp.asarray(jnp.inf, dtype),
        jnp.sqrt(g),
    )


def distance_transform(
    mask: Num[Array, '... *spatial'],
    *,
    metric: str = 'euclidean',
    sampling: Optional[Union[float, Sequence[float]]] = None,
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
      :math:`\\mathrm{out}[p] = \\min_q ( g[q] + (q - p)^2 )`
      (:func:`~nitrix.semiring.semiring_matmul` with
      :data:`~nitrix.semiring.TROPICAL_MIN_PLUS`), so the EDT runs on the
      semiring Pallas-CUDA streaming kernel (with the JAX fallback) and is
      reverse-mode differentiable through the min-plus matmul.  This is the
      metric ``scipy.ndimage.distance_transform_edt`` computes; the result
      matches it to floating-point round-off.
    - ``metric="chebyshev"`` / ``"city_block"`` / a custom
      ``structuring_element`` -- approximate **chamfer** DT via iterated
      :data:`~nitrix.semiring.TROPICAL_MIN_PLUS` convolution, sharing the
      tropical-semiring substrate with :func:`erode` / :func:`dilate` and
      converging in ``max(spatial_shape)`` iterations.  Reverse-mode
      differentiable.

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
    sampling
        Per-axis voxel spacing for the **euclidean** engine (anisotropic grids,
        e.g. a 1x1x3 mm MRI): a scalar (isotropic) or one value per spatial
        axis (``scipy.ndimage.distance_transform_edt`` ``sampling=`` semantics).
        ``None`` (default) is unit spacing.  Only the euclidean engine supports
        it (the chamfer engine encodes integer steps in its kernel); passing it
        with a chamfer ``metric`` / ``structuring_element`` raises.
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
    and custom step-cost kernels.
    """
    if structuring_element is None and metric == 'euclidean':
        return _distance_transform_edt(
            mask, sampling=sampling, backend=backend
        )

    if sampling is not None:
        raise ValueError(
            'distance_transform: sampling= (anisotropic voxel spacing) is only '
            'supported by the exact euclidean engine; the chamfer engine '
            "(metric='chebyshev'/'city_block' or a structuring_element) encodes "
            'integer steps in its step kernel. Use metric="euclidean".'
        )

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
    sampling: Optional[Union[float, Sequence[float]]] = None,
    backend: Backend = 'auto',
) -> Float[Array, '... *spatial']:
    """Exact Euclidean distance transform (scipy ``distance_transform_edt``).

    Thin alias for :func:`distance_transform` with ``metric="euclidean"`` --
    the separable min-plus-matmul engine (semiring Pallas-CUDA kernel + JAX
    fallback).  Computes the distance from each non-zero voxel to the nearest
    zero voxel.

    Parameters
    ----------
    mask
        Input array, ``(..., *spatial)``.  Zero positions are the boundary;
        every axis is treated as spatial.
    sampling
        Per-axis voxel spacing for anisotropic grids: ``None`` (unit spacing),
        a scalar (isotropic), or one value per axis (scipy semantics).
    backend
        Execution backend forwarded to the underlying
        :func:`~nitrix.semiring.semiring_matmul` (``"auto"``,
        ``"pallas-cuda"``, or ``"jax"``).

    Returns
    -------
    Float[Array, '... *spatial']
        Euclidean distance map of the same spatial shape as ``mask``;
        :math:`+\\infty` where ``mask`` is everywhere non-zero (no boundary
        reachable).
    """
    return _distance_transform_edt(mask, sampling=sampling, backend=backend)
