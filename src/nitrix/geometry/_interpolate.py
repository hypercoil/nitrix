# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interpolation-method dispatcher for grid resampling.

:func:`resample` and :func:`spatial_transform` both reduce to "sample an
image at a set of continuous coordinates"; they differ only in *where*
the coordinates come from (an align-corners resize grid vs an arbitrary
deformation field).  The *kernel* -- how neighbouring voxels are weighted
into each sampled value -- is an orthogonal axis.  This module is that
axis: a small algebraic data type of immutable :class:`Interpolator`
records, each a pure ``sample(image, coords)`` function, dispatched by
one shared :func:`_sample_at_coords` seam that both grid functions call.

Here the coordinate *source* (the task) is orthogonal to the sampling
*kernel* (the method), and -- once a Pallas gather lands -- to the
*backend* (execution engine).  Adding a method is adding one frozen
record; the grid functions are untouched.

Methods:

- :class:`Linear` -- (multi-)linear, ``order=1``.  The default; the prior
  :func:`resample` / :func:`spatial_transform` behaviour.
- :class:`NearestNeighbour` -- ``order=0`` round-to-nearest.  Exact under
  label-preserving resampling; the gradient with respect to *coordinates*
  is zero almost everywhere (the round is piecewise-constant), so it is
  not usable for coordinate-driven registration losses -- hence
  ``differentiable_in_coords = False``.  It remains differentiable with
  respect to the image *values* (a one-hot gather).
- :class:`Lanczos` -- windowed-sinc interpolation of configurable order
  :math:`a` (default 3): a :math:`2a`-tap separable kernel
  :math:`L_a(x) = \\operatorname{sinc}(x)\\,\\operatorname{sinc}(x / a)`,
  renormalised per axis to a partition of unity (so constants are
  preserved exactly).  High-fidelity and fully differentiable (smooth
  weights) in both values and coordinates; the ANTs
  ``LanczosWindowedSinc`` algorithm class (not bit-exact ITK).
- :class:`CubicBSpline` -- order-3 B-spline (``scipy.ndimage``
  ``order=3``): a recursive **prefilter** (samples to interpolating
  coefficients) plus a 4-tap cubic basis gather.  Differentiable in
  values and coordinates; bit-exact with ``scipy`` ``order=3,
  mode='mirror'``.  The path nnUNet / ``hd_bet`` style preprocessing
  needs.
- :class:`CatmullRomCubic` -- the cardinal (tension :math:`-1/2`)
  cubic-Hermite *interpolating* spline: a 4-tap separable gather with
  **no prefilter** (it interpolates the raw samples directly), so --
  unlike :class:`CubicBSpline` -- no backend-dependent scan.
  Differentiable in values and coordinates; :math:`C^1` (vs
  :class:`CubicBSpline`'s :math:`C^2`), a partition of unity (reproduces
  constants and linear ramps) but *not* a convex combination (small
  negative side lobes, so it can overshoot/ring near edges).  The OpenCV
  ``INTER_CUBIC`` / Keras-STN cardinal cubic a parity-locked consumer
  (ilex ``fd_net``) pins against.
- :class:`MultiLabel` -- ANTs-style label resampling for segmentations:
  each label's indicator mask is resampled with an ``inner`` kernel
  (default :class:`Linear`) and the per-voxel arg-max label is returned.
  Anti-aliases label boundaries while keeping the output inside the input
  label set.  The label set is **static** (a constructor argument --
  ``jnp.unique`` would break ``jit``), and the arg-max makes it
  **non-differentiable** (a hard label output).

An :class:`Interpolator`'s configuration (spline order, the static label
set) is **structural / static**: the records are plain frozen, hashable
dataclasses with no array leaves, so they ride ``jit`` static arguments
and ``custom_vjp`` non-diff slots directly, and are *not* registered as
pytrees.

The kernels lower onto one shared **explicit separable gather**
(:func:`_separable_gather`): each axis contributes a small set of integer
taps and per-tap weights, the boundary mode folds the taps to in-range
indices, and the :math:`T^{\\mathrm{ndim}}` corner values are gathered
and contracted against the outer product of the per-axis weights (the
"explicit 8-corner gather"), and the algorithmic shape a future Pallas
pointer-load kernel ports onto.

For the orders ``map_coordinates`` supports natively (:class:`Linear` /
:class:`NearestNeighbour`) the engine is :func:`_map_coordinates_sample`
on **both** platforms (:func:`_gather_sample`).  The two are parity-equal
to a ULP, so the choice is pure perf, and (re-measured on the pinned JAX
``cuda12==0.10.1``) the XLA ``map_coordinates`` gather is faster than the
explicit 8-corner gather on the **GPU** too -- 1.1-1.5x for any image
:math:`\\geq 48^3`, across coherent / random coords and 1 / 3 channels;
the explicit gather only ties it (~0.1 ms, noise) on tiny
:math:`\\leq 64^3` single-channel volumes.
:func:`_map_coordinates_sample` is thus both the engine *and* the parity
**oracle** the tests pin the gather against.  :class:`Lanczos` has no
``map_coordinates`` equivalent and always uses the explicit gather;
:func:`_separable_gather` is also the algorithmic shape the parked 5-D
Pallas pointer-load kernel ports onto.

This cross-backend bit-identity is an order-0/1 property only.
:class:`CubicBSpline` (order 3) runs a recursive **prefilter** whose
engine is the ``driver`` axis (parallel ``associative_scan`` on GPU /
sequential ``scan`` on CPU); those reassociate the long pole filter
differently, so a cubic resample can differ CPU-vs-GPU beyond a ULP.  It
is a registered divergent op: force ``CubicBSpline(driver='sequential')``
or ``nitrix.reproducible()`` for cross-platform stability.

Boundary handling (``mode`` / ``cval``) is a property of the *call*, not
the kernel, so it stays a keyword on ``sample`` / :func:`resample` /
:func:`spatial_transform` rather than a field on the record.  The five
modes match ``map_coordinates`` exactly at the integer-tap level:
``constant`` fills out-of-range taps with ``cval``, ``nearest`` clamps,
``wrap`` is periodic (:math:`i \\bmod n`), ``mirror`` folds with period
:math:`2(n - 1)` (no edge repeat), and ``reflect`` folds with period
:math:`2n` (edge repeat).
"""

from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass
from typing import (
    Callable,
    ClassVar,
    Literal,
    Protocol,
    Sequence,
    Tuple,
    cast,
    runtime_checkable,
)

import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jsp_ndi
from jaxtyping import Array, Float, Int

from .._internal.backend import default_backend_is_gpu
from .._internal.config import resolve_driver

__all__ = [
    'BoundaryMode',
    'Interpolator',
    'Linear',
    'NearestNeighbour',
    'Lanczos',
    'CubicBSpline',
    'CubicBSplineBoundaryWarning',
    'CatmullRomCubic',
    'MultiLabel',
]


# Boundary modes accepted by the samplers and the grid functions.  Names
# match ``jax.scipy.ndimage.map_coordinates``.  Defined here (not in
# ``grid``) so the kernel records can annotate against it without a
# circular import; ``grid`` re-imports it.
BoundaryMode = Literal['constant', 'nearest', 'wrap', 'mirror', 'reflect']


# ---------------------------------------------------------------------------
# The kernel axis: an ADT of immutable Interpolator records
# ---------------------------------------------------------------------------


# An ``Interpolator`` is anything exposing ``sample`` plus the two
# differentiability flags.  Structural conformance (a Protocol) rather
# than inheritance keeps the concrete records plain frozen dataclasses,
# matching the ``smoothing.metric.FeatureMetric`` precedent.
@runtime_checkable
class Interpolator(Protocol):
    """Sample an image at continuous coordinates with a fixed kernel.

    A method maps an unbatched channel-last image ``(*spatial, c)`` and
    a coordinate field ``(*out_spatial, ndim)`` to the sampled values
    ``(*out_spatial, c)``.  ``coords[..., d]`` is the floating-point
    position along input axis ``d``.  Leading batch axes are composed by
    callers (:func:`spatial_transform` vmaps the core); a method only
    sees the unbatched core shapes.

    Attributes
    ----------
    differentiable_in_values
        Whether the sampled output has a non-trivial gradient with
        respect to the input image values.
    differentiable_in_coords
        Whether the sampled output has a non-trivial gradient with
        respect to the sample coordinates -- the property a
        coordinate-driven registration loss needs.  ``False`` for
        piecewise-constant kernels (nearest neighbour) and hard label
        selection (multi-label).
    """

    # ``ClassVar`` (not a plain instance annotation): the flags are
    # fixed per kernel, so implementers expose them as class constants;
    # this lets the frozen records carry them without an ``__init__``
    # field, and a ``ClassVar`` Protocol member is what mypy requires a
    # class-variable implementation to satisfy.
    differentiable_in_values: ClassVar[bool]
    differentiable_in_coords: ClassVar[bool]

    def sample(
        self,
        image: Float[Array, '*spatial c'],
        coords: Float[Array, '*out_spatial ndim'],
        *,
        mode: BoundaryMode,
        cval: float,
    ) -> Float[Array, '*out_spatial c']:
        """Sample the image at continuous coordinates with this kernel.

        Parameters
        ----------
        image
            Unbatched channel-last image of shape ``(*spatial, c)``.
        coords
            Coordinate field of shape ``(*out_spatial, ndim)``;
            ``coords[..., d]`` is the floating-point sample position
            along input spatial axis ``d``.
        mode
            Boundary rule for out-of-range taps -- one of ``'constant'``,
            ``'nearest'``, ``'wrap'``, ``'mirror'`` or ``'reflect'``,
            matching ``jax.scipy.ndimage.map_coordinates``.
        cval
            Fill value for out-of-range taps under ``mode='constant'``.

        Returns
        -------
        Float[Array, '*out_spatial c']
            The sampled values, one per coordinate and channel.
        """
        ...


def _map_coordinates_sample(
    image: Float[Array, '*spatial c'],
    coords: Float[Array, '*out_spatial ndim'],
    *,
    order: int,
    mode: BoundaryMode,
    cval: float,
) -> Float[Array, '*out_spatial c']:
    """Channel-wise gather via ``jax.scipy.ndimage.map_coordinates``.

    The shared engine for the spline orders ``map_coordinates`` supports
    natively (``0`` nearest, ``1`` linear).  ``coords`` is reshaped to
    the ``(ndim, n_samples)`` layout the routine wants; the trailing
    channel axis is mapped over with ``jax.vmap``.

    Parameters
    ----------
    image
        Unbatched channel-last image of shape ``(*spatial, c)``.
    coords
        Coordinate field of shape ``(*out_spatial, ndim)``, with
        ``coords[..., d]`` the sample position along input axis ``d``.
    order
        Spline order passed through to ``map_coordinates``; ``0``
        (nearest) or ``1`` (linear).
    mode
        Boundary rule for out-of-range taps (see ``map_coordinates``).
    cval
        Fill value for out-of-range taps under ``mode='constant'``.

    Returns
    -------
    Float[Array, '*out_spatial c']
        The sampled values, one per coordinate and channel.
    """
    ndim = coords.shape[-1]
    # ``map_coordinates`` wants coords as ``(ndim, n_samples)`` -- one
    # row per input spatial axis.  Move the ndim axis to the front and
    # flatten the output-spatial dims.
    coords_t = jnp.moveaxis(coords, -1, 0)  # (ndim, *out_spatial)
    coords_flat = coords_t.reshape(ndim, -1)  # (ndim, N)

    def sample_one_channel(
        img_ch: Float[Array, '*spatial'],
    ) -> Float[Array, 'n']:
        # ``map_coordinates`` is typed to want a Sequence of per-axis
        # coordinate arrays (and to return Any); the stacked ``(ndim, N)``
        # array is accepted at runtime (jax iterates axis 0).  cast the
        # arg to satisfy the checker and restore the result type.
        return cast(
            Float[Array, 'n'],
            jsp_ndi.map_coordinates(
                img_ch,
                cast(Sequence[Array], coords_flat),
                order=order,
                mode=mode,
                cval=cval,
            ),
        )

    sample_v = jax.vmap(sample_one_channel, in_axes=-1, out_axes=-1)
    flat_out = sample_v(image)  # (N, c)
    out_spatial = coords.shape[:-1]
    return flat_out.reshape(out_spatial + (image.shape[-1],))


# ---------------------------------------------------------------------------
# The explicit separable gather (the runtime engine)
# ---------------------------------------------------------------------------


# A per-axis tap rule: given the sample coordinate along one axis
# ``(n_samples,)``, return the integer taps and their weights, each
# ``(n_samples, n_taps)``.  The kernel is isotropic, so one rule serves
# every axis.
AxisTapRule = Callable[
    [Float[Array, 'n']],
    Tuple[Int[Array, 'n t'], Float[Array, 'n t']],
]


def _boundary_index(
    taps: Int[Array, 'n t'],
    size: int,
    mode: BoundaryMode,
) -> Tuple[Int[Array, 'n t'], Float[Array, 'n t']]:
    """Fold integer taps to in-range gather indices for ``mode``.

    The folds reproduce ``jax.scipy.ndimage.map_coordinates`` exactly:

    - ``constant`` -- clamp for the (masked-out) gather; ``valid`` marks
      the in-range taps.
    - ``nearest``  -- clamp to :math:`[0, \\mathrm{size} - 1]`.
    - ``wrap``     -- :math:`\\mathrm{tap} \\bmod \\mathrm{size}` (period
      :math:`\\mathrm{size}`).
    - ``mirror``   -- fold with period :math:`2(\\mathrm{size} - 1)`, no
      edge repeat.
    - ``reflect``  -- fold with period :math:`2\\,\\mathrm{size}`, edge
      repeat.

    Parameters
    ----------
    taps
        Integer sample taps of shape ``(n, t)`` -- ``n`` samples, ``t``
        taps each -- possibly out of the valid index range.
    size
        Length of the input axis being indexed.
    mode
        Boundary rule -- one of ``'constant'``, ``'nearest'``, ``'wrap'``,
        ``'mirror'`` or ``'reflect'``.

    Returns
    -------
    idx : Int[Array, 'n t']
        Taps folded to legal indices into an axis of length ``size`` (so
        the gather never reads out of bounds).
    valid : Float[Array, 'n t']
        Per-tap mask of taps that were *originally* in range.  Only
        ``mode='constant'`` uses it (out-of-range taps there contribute
        ``cval`` rather than a folded neighbour); the other modes return
        an all-ones mask.
    """
    ones = jnp.ones(taps.shape, dtype=bool)
    if mode == 'constant':
        valid = (taps >= 0) & (taps < size)
        return jnp.clip(taps, 0, size - 1), valid
    if mode == 'nearest':
        return jnp.clip(taps, 0, size - 1), ones
    if mode == 'wrap':
        return jnp.mod(taps, size), ones
    if mode == 'mirror':
        if size == 1:
            return jnp.zeros_like(taps), ones
        period = 2 * (size - 1)
        folded = jnp.mod(taps, period)
        return jnp.where(folded >= size, period - folded, folded), ones
    if mode == 'reflect':
        period = 2 * size
        folded = jnp.mod(taps, period)
        return jnp.where(folded >= size, period - 1 - folded, folded), ones
    raise ValueError(
        f'mode={mode!r}; expected one of "constant", "nearest", "wrap", '
        '"mirror", "reflect".'
    )


def _separable_gather(
    image: Float[Array, '*spatial c'],
    coords: Float[Array, '*out_spatial ndim'],
    *,
    tap_rule: AxisTapRule,
    mode: BoundaryMode,
    cval: float,
) -> Float[Array, '*out_spatial c']:
    """Sample ``image`` at ``coords`` with an explicit separable gather.

    The runtime engine for the gather-based kernels.  Each of the
    ``ndim`` axes contributes ``T`` integer taps and weights via
    ``tap_rule``; the boundary mode folds the taps; the
    :math:`T^{\\mathrm{ndim}}` corner values are gathered and contracted
    against the outer product of the per-axis weights.  Differentiable in
    the image values (the gather) and -- whenever ``tap_rule`` makes the
    weights depend smoothly on the coordinate -- in ``coords``.

    Operates on the *unbatched* core shapes; leading batch axes are
    composed by the caller (:func:`spatial_transform` vmaps the core).

    Parameters
    ----------
    image
        Unbatched channel-last image of shape ``(*spatial, c)``.
    coords
        Coordinate field of shape ``(*out_spatial, ndim)``, with
        ``coords[..., d]`` the sample position along input axis ``d``.
    tap_rule
        Per-axis rule mapping a 1-D coordinate vector ``(n,)`` to integer
        taps and per-tap weights, each ``(n, t)``.  The kernel is
        isotropic, so one rule serves every axis.
    mode
        Boundary rule for out-of-range taps (see :func:`_boundary_index`).
    cval
        Fill value for out-of-range taps under ``mode='constant'``.

    Returns
    -------
    Float[Array, '*out_spatial c']
        The sampled values, one per coordinate and channel, cast back to
        the input image dtype.
    """
    ndim = coords.shape[-1]
    out_spatial = coords.shape[:-1]
    spatial = image.shape[:ndim]  # (n_0, ..., n_{ndim-1})
    n_channels = image.shape[-1]
    coords_flat = coords.reshape(-1, ndim)  # (N, ndim)

    # Per-axis taps / weights / validity.
    idx_axes: list[Int[Array, 'n t']] = []
    weight_axes: list[Float[Array, 'n t']] = []
    valid_axes: list[Int[Array, 'n t']] = []
    n_taps = 0
    for axis in range(ndim):
        taps, weights = tap_rule(coords_flat[:, axis])
        idx, valid = _boundary_index(taps, spatial[axis], mode)
        idx_axes.append(idx)
        weight_axes.append(weights)
        valid_axes.append(valid)
        n_taps = taps.shape[-1]

    # Accumulate over the T**ndim corners of the separable stencil.
    acc: Float[Array, 'n c'] | None = None
    for corner in itertools.product(range(n_taps), repeat=ndim):
        gather_idx = tuple(
            idx_axes[axis][:, corner[axis]] for axis in range(ndim)
        )
        values = image[gather_idx]  # (N, c)
        weight = weight_axes[0][:, corner[0]]
        for axis in range(1, ndim):
            weight = weight * weight_axes[axis][:, corner[axis]]
        if mode == 'constant':
            corner_valid = valid_axes[0][:, corner[0]]
            for axis in range(1, ndim):
                corner_valid = corner_valid & valid_axes[axis][:, corner[axis]]
            values = jnp.where(corner_valid[:, None], values, cval)
        term = weight[:, None] * values
        acc = term if acc is None else acc + term

    assert acc is not None  # ndim >= 1, so at least one corner ran
    return acc.astype(image.dtype).reshape(out_spatial + (n_channels,))


def _gather_sample(
    image: Float[Array, '*spatial c'],
    coords: Float[Array, '*out_spatial ndim'],
    *,
    order: int,
    tap_rule: AxisTapRule,
    mode: BoundaryMode,
    cval: float,
) -> Float[Array, '*out_spatial c']:
    """Sample via the faster engine for a ``map_coordinates`` order (0/1).

    The explicit separable gather and ``map_coordinates`` are
    parity-equal to a ULP for ``order`` 0/1, so this is pure perf.  It
    uses ``map_coordinates`` on both platforms: on the pinned JAX
    ``cuda12==0.10.1``, the XLA ``map_coordinates`` gather is faster than
    the explicit 8-corner gather on the GPU too, across coords (coherent
    id+disp and random), channels (1 and 3) and sizes -- 1.1-1.5x for any
    image :math:`\\geq 48^3`; the explicit gather only edges it
    (~1.04-1.11x, ~0.1 ms, noise) on tiny :math:`\\leq 64^3`
    single-channel volumes.  :func:`_separable_gather` is retained for
    :class:`Lanczos` (no ``map_coordinates`` equivalent) and as the
    algorithmic shape the parked 5-D Pallas gather ports onto.

    Parameters
    ----------
    image
        Unbatched channel-last image of shape ``(*spatial, c)``.
    coords
        Coordinate field of shape ``(*out_spatial, ndim)``, with
        ``coords[..., d]`` the sample position along input axis ``d``.
    order
        Spline order (``0`` nearest or ``1`` linear) forwarded to
        ``map_coordinates``.
    tap_rule
        Per-axis tap/weight rule.  Accepted for a uniform signature with
        the explicit-gather kernels but unused on this path (the
        ``map_coordinates`` engine computes its own weights).
    mode
        Boundary rule for out-of-range taps (see ``map_coordinates``).
    cval
        Fill value for out-of-range taps under ``mode='constant'``.

    Returns
    -------
    Float[Array, '*out_spatial c']
        The sampled values, one per coordinate and channel.
    """
    return _map_coordinates_sample(
        image,
        coords,
        order=order,
        mode=mode,
        cval=cval,
    )


@runtime_checkable
class _SeparableKernel(Protocol):
    """A gather kernel whose weights factor per axis (a tap rule).

    The internal marker that :func:`resample` uses to choose its engine.
    :class:`Linear` / :class:`NearestNeighbour` / :class:`Lanczos` /
    :class:`CubicBSpline` conform; :class:`MultiLabel` does not (its
    output is a per-voxel arg-max, not a separable weighted sum), so it
    falls back to the dense meshgrid gather.

    ``prefers_separable_resample`` decides the *resample* engine: ``True``
    routes the resize grid through the cheap per-axis 1-D passes
    (:func:`_separable_resample`, :math:`O(N \\cdot T \\cdot
    \\mathrm{ndim})`) -- the only tractable form for the wide
    :class:`Lanczos` stencil (:math:`T = 2\\,\\mathrm{order}` giving
    :math:`T^{\\mathrm{ndim}}` dense corners).  ``False`` keeps the
    low-tap kernels (:class:`Linear` / :class:`NearestNeighbour`) on the
    dense per-coordinate path, whose ``map_coordinates`` engine still wins
    on CPU (the 1-D passes there are a mild regression on the
    throughput-sensitive CPU interpolation path).

    ``_prepare`` is a per-image preprocessing hook applied once before the
    gather: the identity for the direct kernels, the recursive B-spline
    prefilter for :class:`CubicBSpline` (which gathers *coefficients*, not
    the raw samples).
    """

    prefers_separable_resample: bool

    def _axis_taps_weights(
        self,
        coord: Float[Array, 'n'],
    ) -> Tuple[Int[Array, 'n t'], Float[Array, 'n t']]: ...

    def _prepare(
        self,
        image: Float[Array, '*spatial c'],
        mode: BoundaryMode,
    ) -> Float[Array, '*spatial c']: ...


def _separable_resample(
    image: Float[Array, '*spatial c'],
    axes_coords: Sequence[Float[Array, 'out']],
    *,
    tap_rule: AxisTapRule,
    mode: BoundaryMode,
    cval: float,
) -> Float[Array, '*out_spatial c']:
    """Resample a separable resize grid via successive per-axis 1-D passes.

    ``axes_coords[d]`` is the 1-D sample-coordinate vector along input
    axis ``d``.  Because the resize grid is the outer product of these
    vectors, each axis is resampled independently -- one ``T``-tap
    weighted gather along that axis -- so the cost is
    :math:`O(N_{\\mathrm{out}} \\cdot T \\cdot \\mathrm{ndim})` rather
    than the dense :math:`O(N_{\\mathrm{out}} \\cdot T^{\\mathrm{ndim}})`
    corner gather of :func:`_separable_gather`.  The two agree to a ULP;
    this is the form that keeps high-order kernels (:class:`Lanczos`:
    :math:`T = 2\\,\\mathrm{order}`) tractable on 3-D volumes.

    Parameters
    ----------
    image
        Unbatched channel-last image of shape ``(*spatial, c)``.
    axes_coords
        Per-axis 1-D sample-coordinate vectors of the (outer-product)
        resize grid; ``axes_coords[d]`` targets input spatial axis ``d``.
    tap_rule
        Per-axis rule mapping a 1-D coordinate vector to integer taps and
        per-tap weights, each of shape ``(out_d, t)``.
    mode
        Boundary rule for out-of-range taps (see :func:`_boundary_index`).
    cval
        Fill value for out-of-range taps under ``mode='constant'``.

    Returns
    -------
    Float[Array, '*out_spatial c']
        The resampled image on the resize grid, cast back to the input
        image dtype.
    """
    out = image
    for axis, coord in enumerate(axes_coords):
        size = out.shape[axis]
        taps, weights = tap_rule(coord)  # (out_d, T)
        idx, valid = _boundary_index(taps, size, mode)
        out_d, n_taps = taps.shape
        # Gather the T taps for every output position along ``axis``;
        # ``take`` applies the same index vector across all other dims.
        gathered = jnp.take(out, idx.reshape(-1), axis=axis)
        split_shape = (
            out.shape[:axis] + (out_d, n_taps) + out.shape[axis + 1 :]
        )
        gathered = gathered.reshape(split_shape)
        broadcast = [1] * gathered.ndim
        broadcast[axis] = out_d
        broadcast[axis + 1] = n_taps
        if mode == 'constant':
            gathered = jnp.where(
                valid.reshape(broadcast),
                gathered,
                cval,
            )
        out = jnp.sum(gathered * weights.reshape(broadcast), axis=axis + 1)
    return out.astype(image.dtype)


# ---------------------------------------------------------------------------
# Cubic B-spline prefilter (Unser / Aldroubi / Eden; the order-3 coefficients)
# ---------------------------------------------------------------------------


# The single real pole of the cubic B-spline (z1 = sqrt(3) - 2 ~= -0.2679).
_BSPLINE_POLE = math.sqrt(3.0) - 2.0


def _bspline_initial_causal(
    coeffs: Float[Array, 'n'],
    pole: float,
) -> Float[Array, '']:
    """Exact whole-sample-mirror initial causal coefficient.

    The closed-form boundary condition for the causal recursion under a
    mirror (whole-sample-symmetric) extension -- the Thévenaz / ITK
    convention, equal to ``scipy.ndimage`` with ``mode='mirror'``.
    Vectorised (a single dot product), so it is parallel and
    differentiable -- no scan.

    Parameters
    ----------
    coeffs
        The (gain-scaled) 1-D signal of length ``n`` being prefiltered.
    pole
        The real pole of the recursive filter (for a cubic B-spline,
        :math:`\\sqrt{3} - 2`).

    Returns
    -------
    Float[Array, '']
        The scalar initial causal coefficient :math:`y[0]`.
    """
    n = coeffs.shape[0]
    k = jnp.arange(n)
    weights = jnp.power(pole, k) + jnp.power(pole, 2 * n - 2 - k)
    weights = weights.at[0].set(1.0).at[n - 1].set(jnp.power(pole, n - 1))
    return jnp.dot(weights, coeffs) / (1.0 - jnp.power(pole, 2 * n - 2))


def _first_order_causal(
    inputs: Float[Array, 'n'],
    pole: float,
    init: Float[Array, ''],
    *,
    associative: bool,
) -> Float[Array, 'n']:
    """Resolve :math:`y[k] = \\mathrm{pole}\\,y[k-1] + \\mathrm{inputs}[k]`.

    The recurrence is solved with the boundary condition
    :math:`y[0] = \\mathrm{init}`; ``inputs[0]`` is ignored (overridden by
    ``init``).  Two exact engines are offered (a first-order linear
    recurrence composes associatively):

    - sequential ``lax.scan`` (CPU; low overhead), and
    - ``lax.associative_scan`` (GPU; :math:`O(\\log n)` depth, the
      parallel form that removes the scan's sequential-depth cost).

    Both are exact (no truncation), but they **reassociate the recurrence
    differently**, so over a long pole filter they diverge beyond a ULP --
    a cross-platform numerical difference, not bit-equality.  This is
    governed by the ``driver`` axis
    (``CubicBSpline(driver=...)`` / ``nitrix.reproducible()``).

    Parameters
    ----------
    inputs
        The forcing sequence of length ``n``; ``inputs[0]`` is unused.
    pole
        The real pole of the recursion.
    init
        The scalar value of :math:`y[0]`.
    associative
        If ``True`` use the parallel ``lax.associative_scan``; if
        ``False`` use the sequential ``lax.scan``.

    Returns
    -------
    Float[Array, 'n']
        The resolved sequence :math:`y` of length ``n``.
    """
    n = inputs.shape[0]
    init = jnp.reshape(init, (1,))
    if associative:
        # Element k carries (a_k, v_k); the prefix combine resolves
        # y[k] = a_k * y[k-1] + v_k.  a_0 = 0 so y[0] = v_0 = init.
        a_seq = jnp.concatenate(
            [
                jnp.zeros((1,), inputs.dtype),
                jnp.full((n - 1,), pole, inputs.dtype),
            ],
        )
        v_seq = jnp.concatenate([init, inputs[1:]])

        def combine(
            left: Tuple[Array, Array],
            right: Tuple[Array, Array],
        ) -> Tuple[Array, Array]:
            a_l, v_l = left
            a_r, v_r = right
            return a_r * a_l, a_r * v_l + v_r

        _, values = jax.lax.associative_scan(combine, (a_seq, v_seq))
        return cast(Float[Array, 'n'], values)

    def step(prev: Array, x: Array) -> Tuple[Array, Array]:
        cur = x + pole * prev
        return cur, cur

    _, tail = jax.lax.scan(step, init[0], inputs[1:])
    return jnp.concatenate([init, tail])


def _bspline_prefilter_1d(
    samples: Float[Array, 'n'],
    pole: float,
    *,
    associative: bool,
) -> Float[Array, 'n']:
    """B-spline coefficients of a 1-D signal (one causal + one anti-causal).

    Applies the standard two-pass recursive prefilter: a gain scaling
    followed by a causal forward recursion and an anti-causal backward
    recursion, both initialised for the whole-sample mirror boundary.

    Parameters
    ----------
    samples
        The raw 1-D signal of length ``n``.
    pole
        The real pole of the B-spline recursion (for cubic,
        :math:`\\sqrt{3} - 2`).
    associative
        Whether the two recursions use the parallel associative scan
        (``True``) or the sequential scan (``False``).

    Returns
    -------
    Float[Array, 'n']
        The interpolating B-spline coefficients of the signal.
    """
    n = samples.shape[0]
    gain = (1.0 - pole) * (1.0 - 1.0 / pole)  # = 6 for cubic
    scaled = samples * gain
    causal0 = _bspline_initial_causal(scaled, pole)
    causal = _first_order_causal(
        scaled,
        pole,
        causal0,
        associative=associative,
    )
    # Anti-causal init (mirror) and recursion ``c[k] = pole*(c[k+1] - c[k])``,
    # run as a forward first-order recurrence on the reversed sequence.
    anti0 = (pole / (pole * pole - 1.0)) * (
        causal[n - 1] + pole * causal[n - 2]
    )
    reversed_input = -pole * causal[::-1]
    reversed_out = _first_order_causal(
        reversed_input,
        pole,
        anti0,
        associative=associative,
    )
    return reversed_out[::-1]


def _bspline_prefilter(
    image: Float[Array, '*spatial c'],
    mode: BoundaryMode,
    *,
    driver: str = 'auto',
) -> Float[Array, '*spatial c']:
    """Separable cubic B-spline prefilter over the spatial axes.

    Converts samples to interpolating B-spline coefficients, one
    recursive pass per spatial axis (the trailing channel axis is left
    alone).  Mirror boundary initialisation regardless of ``mode`` (the
    standard B-spline convention; ``mode`` then governs only the gather
    extrapolation -- exact ``scipy`` ``order=3`` parity holds for
    ``mode='mirror'``, interior parity otherwise).

    The recursion engine is the ``driver`` axis: ``'auto'`` runs the
    parallel ``associative_scan`` on GPU / sequential ``scan`` on CPU
    (which diverge beyond a ULP over a long filter);
    ``nitrix.reproducible()`` or ``driver='sequential'`` forces the
    canonical sequential recurrence on every platform.

    Parameters
    ----------
    image
        Unbatched channel-last image of shape ``(*spatial, c)``.
    mode
        Accepted for a uniform signature but unused: the prefilter always
        initialises with the mirror boundary.
    driver
        Recursion engine selector -- ``'auto'`` (platform-dependent),
        ``'associative'`` or ``'sequential'``.

    Returns
    -------
    Float[Array, '*spatial c']
        The interpolating B-spline coefficients, same shape as ``image``.
    """
    del mode  # prefilter boundary is always mirror (documented)
    ndim = image.ndim - 1
    resolved = resolve_driver(
        driver,
        op='geometry.cubic_bspline_prefilter',
        fast=lambda: (
            'associative' if default_backend_is_gpu() else 'sequential'
        ),
    )
    associative = resolved == 'associative'
    out = image
    for axis in range(ndim):
        if out.shape[axis] < 2:
            continue  # a length-1 axis has no recursion
        moved = jnp.moveaxis(out, axis, -1)  # (..., N)
        flat = moved.reshape(-1, moved.shape[-1])  # (M, N)
        filtered = jax.vmap(
            lambda row: _bspline_prefilter_1d(
                row,
                _BSPLINE_POLE,
                associative=associative,
            )
        )(flat)
        out = jnp.moveaxis(filtered.reshape(moved.shape), -1, axis)
    return out


@dataclass(frozen=True)
class Linear:
    """(Multi-)linear interpolation (``order=1``).

    The default kernel and the prior :func:`resample` /
    :func:`spatial_transform` behaviour: each sampled value is the
    multilinear blend of its :math:`2^{\\mathrm{ndim}}` surrounding
    voxels.  Smooth in both the image values and the sample coordinates,
    so it is the kernel for differentiable registration losses.
    """

    differentiable_in_values: ClassVar[bool] = True
    differentiable_in_coords: ClassVar[bool] = True
    prefers_separable_resample: ClassVar[bool] = False

    def _axis_taps_weights(
        self,
        coord: Float[Array, 'n'],
    ) -> Tuple[Int[Array, 'n 2'], Float[Array, 'n 2']]:
        base = jnp.floor(coord)
        frac = coord - base
        lower = base.astype(jnp.int32)
        taps = jnp.stack([lower, lower + 1], axis=-1)
        weights = jnp.stack([1.0 - frac, frac], axis=-1)
        return taps, weights

    def _prepare(
        self,
        image: Float[Array, '*spatial c'],
        mode: BoundaryMode,
    ) -> Float[Array, '*spatial c']:
        return image

    def sample(
        self,
        image: Float[Array, '*spatial c'],
        coords: Float[Array, '*out_spatial ndim'],
        *,
        mode: BoundaryMode,
        cval: float,
    ) -> Float[Array, '*out_spatial c']:
        """Sample the image at ``coords`` by multilinear interpolation.

        Parameters
        ----------
        image
            Unbatched channel-last image of shape ``(*spatial, c)``.
        coords
            Coordinate field of shape ``(*out_spatial, ndim)``, with
            ``coords[..., d]`` the sample position along input axis ``d``.
        mode
            Boundary rule for out-of-range taps -- one of ``'constant'``,
            ``'nearest'``, ``'wrap'``, ``'mirror'`` or ``'reflect'``.
        cval
            Fill value for out-of-range taps under ``mode='constant'``.

        Returns
        -------
        Float[Array, '*out_spatial c']
            The sampled values, one per coordinate and channel.
        """
        return _gather_sample(
            image,
            coords,
            order=1,
            tap_rule=self._axis_taps_weights,
            mode=mode,
            cval=cval,
        )


@dataclass(frozen=True)
class NearestNeighbour:
    """Nearest-neighbour interpolation (``order=0``).

    Each sampled value is the single voxel nearest the sample position.
    The output is restricted to values that occur in the input, so this
    is the label-preserving choice when an anti-aliased boundary is *not*
    wanted (cf. :class:`MultiLabel`).  The round-to-nearest is
    piecewise-constant in the coordinates, so the gradient w.r.t.
    coordinates is zero almost everywhere
    (``differentiable_in_coords = False``); the gather is still
    differentiable w.r.t. the image values.

    The rounding is ``floor(coord + 0.5)`` (round half up), matching
    ``jax.scipy.ndimage.map_coordinates(order=0)`` -- *not* round
    half-to-even.
    """

    differentiable_in_values: ClassVar[bool] = True
    differentiable_in_coords: ClassVar[bool] = False
    prefers_separable_resample: ClassVar[bool] = False

    def _axis_taps_weights(
        self,
        coord: Float[Array, 'n'],
    ) -> Tuple[Int[Array, 'n 1'], Float[Array, 'n 1']]:
        nearest = jnp.floor(coord + 0.5).astype(jnp.int32)
        taps = nearest[..., None]
        weights = jnp.ones(taps.shape, dtype=coord.dtype)
        return taps, weights

    def _prepare(
        self,
        image: Float[Array, '*spatial c'],
        mode: BoundaryMode,
    ) -> Float[Array, '*spatial c']:
        return image

    def sample(
        self,
        image: Float[Array, '*spatial c'],
        coords: Float[Array, '*out_spatial ndim'],
        *,
        mode: BoundaryMode,
        cval: float,
    ) -> Float[Array, '*out_spatial c']:
        """Sample the image at ``coords`` by nearest-neighbour selection.

        Parameters
        ----------
        image
            Unbatched channel-last image of shape ``(*spatial, c)``.
        coords
            Coordinate field of shape ``(*out_spatial, ndim)``, with
            ``coords[..., d]`` the sample position along input axis ``d``.
        mode
            Boundary rule for out-of-range taps -- one of ``'constant'``,
            ``'nearest'``, ``'wrap'``, ``'mirror'`` or ``'reflect'``.
        cval
            Fill value for out-of-range taps under ``mode='constant'``.

        Returns
        -------
        Float[Array, '*out_spatial c']
            The nearest input voxel value per coordinate and channel.
        """
        return _gather_sample(
            image,
            coords,
            order=0,
            tap_rule=self._axis_taps_weights,
            mode=mode,
            cval=cval,
        )


def _lanczos_kernel(
    x: Float[Array, '...'],
    order: int,
) -> Float[Array, '...']:
    """The Lanczos windowed-sinc kernel.

    Evaluates :math:`L_a(x) = \\operatorname{sinc}(x)\\,
    \\operatorname{sinc}(x / a)`, zero outside the support
    :math:`|x| < a` (:math:`a = \\mathrm{order}`).  ``jnp.sinc`` is the
    *normalised* sinc (:math:`\\sin(\\pi x) / (\\pi x)`, unit at 0), so
    the kernel is 1 at :math:`x = 0` and 0 at every nonzero integer
    offset -- i.e. it reproduces grid samples exactly.  Smooth within the
    support, so differentiable in ``x``.

    Parameters
    ----------
    x
        Signed tap distances at which to evaluate the kernel, any shape.
    order
        The Lanczos radius :math:`a` (number of sinc lobes per side).

    Returns
    -------
    Float[Array, '...']
        The kernel weights, same shape as ``x``.
    """
    a = float(order)
    kernel = jnp.sinc(x) * jnp.sinc(x / a)
    return jnp.where(jnp.abs(x) < a, kernel, 0.0)


@dataclass(frozen=True)
class Lanczos:
    """Lanczos windowed-sinc interpolation of order ``order`` (radius ``a``).

    A separable :math:`2a`-tap kernel
    :math:`L_a(x) = \\operatorname{sinc}(x)\\,\\operatorname{sinc}(x / a)`
    (:math:`a = \\mathrm{order}`), the high-fidelity resampler of the ANTs
    ``LanczosWindowedSinc`` class.  The :math:`2a` taps per axis are
    :math:`\\lfloor\\mathrm{coord}\\rfloor - a + 1, \\ldots,
    \\lfloor\\mathrm{coord}\\rfloor + a`; the per-axis weights are
    **renormalised to sum to 1**, so the separable outer product is a
    partition of unity and constants are preserved exactly (raw Lanczos
    weights sum to ~1 but not exactly).

    Smooth in both the image values and the sample coordinates, so it is
    differentiable end-to-end -- the high-quality differentiable
    alternative to :class:`Linear` for registration and anti-aliased
    resampling.  Unlike :class:`Linear` / :class:`NearestNeighbour` it
    has no ``map_coordinates`` equivalent, so it always uses the explicit
    separable gather (no platform branch).

    Attributes
    ----------
    order
        The radius ``a`` (number of sinc lobes per side); ``2 * order``
        taps per axis.  Default ``3`` (the common ANTs/ITK choice);
        larger is sharper and wider-support but more ringing-prone.
        Must be a positive integer.

    Notes
    -----
    This is the ANTs *algorithm class*, not bit-exact ITK parity (ITK's
    windowed-sinc radius / normalisation conventions differ in detail).
    """

    order: int = 3

    differentiable_in_values: ClassVar[bool] = True
    differentiable_in_coords: ClassVar[bool] = True
    prefers_separable_resample: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if not isinstance(self.order, int) or self.order < 1:
            raise ValueError(
                f'Lanczos: order must be a positive integer; got '
                f'{self.order!r}.'
            )

    def _axis_taps_weights(
        self,
        coord: Float[Array, 'n'],
    ) -> Tuple[Int[Array, 'n t'], Float[Array, 'n t']]:
        a = self.order
        base = jnp.floor(coord).astype(jnp.int32)  # (N,)
        offsets = jnp.arange(-a + 1, a + 1, dtype=jnp.int32)  # (2a,)
        taps = base[:, None] + offsets[None, :]  # (N, 2a)
        dist = coord[:, None] - taps.astype(coord.dtype)  # (N, 2a)
        weights = _lanczos_kernel(dist, a)  # (N, 2a)
        # Renormalise to a partition of unity (constant-preserving).
        weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
        return taps, weights

    def _prepare(
        self,
        image: Float[Array, '*spatial c'],
        mode: BoundaryMode,
    ) -> Float[Array, '*spatial c']:
        return image

    def sample(
        self,
        image: Float[Array, '*spatial c'],
        coords: Float[Array, '*out_spatial ndim'],
        *,
        mode: BoundaryMode,
        cval: float,
    ) -> Float[Array, '*out_spatial c']:
        """Sample the image at ``coords`` by windowed-sinc interpolation.

        Parameters
        ----------
        image
            Unbatched channel-last image of shape ``(*spatial, c)``.
        coords
            Coordinate field of shape ``(*out_spatial, ndim)``, with
            ``coords[..., d]`` the sample position along input axis ``d``.
        mode
            Boundary rule for out-of-range taps -- one of ``'constant'``,
            ``'nearest'``, ``'wrap'``, ``'mirror'`` or ``'reflect'``.
        cval
            Fill value for out-of-range taps under ``mode='constant'``.

        Returns
        -------
        Float[Array, '*out_spatial c']
            The sampled values, one per coordinate and channel.
        """
        return _separable_gather(
            image,
            coords,
            tap_rule=self._axis_taps_weights,
            mode=mode,
            cval=cval,
        )


class CubicBSplineBoundaryWarning(UserWarning):
    """Emitted when ``CubicBSpline`` is given a boundary it cannot honour.

    :class:`CubicBSpline` always uses the mirror boundary (the only one
    its prefilter implements), so an explicit non-mirror ``mode`` -- or a
    non-zero ``cval`` -- is ignored.  So that this override is not silent,
    it is announced with this warning.  Pass ``mode='mirror'`` (or leave
    ``mode`` at its default) to silence it, or filter this category.  The
    default ``mode='constant'`` with ``cval=0`` is treated as
    "unspecified" and does *not* warn.
    """


def _warn_cubic_boundary_ignored(mode: BoundaryMode, cval: float) -> None:
    """Announce a CubicBSpline boundary override (deduped by ``warnings``).

    Emits a :class:`CubicBSplineBoundaryWarning` when the requested
    boundary is not the mirror boundary the kernel actually uses; the
    bare default (``mode='constant'``, ``cval=0``) is treated as
    unspecified and stays silent.

    Parameters
    ----------
    mode
        The boundary mode requested on the call.
    cval
        The constant fill value requested on the call.
    """
    if mode not in ('mirror', 'constant') or cval != 0.0:
        warnings.warn(
            f'CubicBSpline always uses the mirror boundary; the supplied '
            f'mode={mode!r} / cval={cval!r} is ignored.  Pass mode="mirror" '
            f'to silence this (a mode-aware B-spline prefilter is a planned '
            f'extension).',
            category=CubicBSplineBoundaryWarning,
            stacklevel=3,
        )


@dataclass(frozen=True)
class CubicBSpline:
    """Cubic (order-3) B-spline interpolation -- ``scipy.ndimage`` ``order=3``.

    The order-3 spline path consumers like the nnUNet / ``hd_bet``
    preprocessing in some ilex pipelines use.  Unlike a plain cubic
    *convolution*, a B-spline interpolator is a two-step operation:

    1. a **prefilter** converts the image samples to interpolating
       B-spline coefficients (a separable recursive filter with the cubic
       pole :math:`\\sqrt{3} - 2`; :func:`_bspline_prefilter`), and
    2. a 4-tap separable **cubic B-spline basis** gathers those
       coefficients.

    Without the prefilter the cubic basis only *approximates* (blurs) the
    samples; with it the result passes through the samples exactly -- the
    interpolation property.  The basis is a partition of unity (weights
    sum to 1; no renormalisation), :math:`C^2`-smooth, so the kernel is
    differentiable in both the values (the prefilter and gather are
    linear) and the coordinates (the basis is smooth).

    Engine note: the prefilter is a first-order linear recurrence, run
    sequentially (``lax.scan``) on CPU and in parallel
    (``lax.associative_scan``, ``O(log n)`` depth) on GPU -- the same
    platform split as ``signal._iir``.  Both are exact.  (An FFT
    convolution -- the other ``signal._iir`` engine -- is *not* used: the
    cubic pole is mild, so the prefilter's impulse response is a ~25-tap
    short FIR, too short for the transform overhead to amortise.)

    Boundary: this kernel **always uses the mirror (whole-sample-
    symmetric) boundary** for *both* the prefilter and the gather, and
    **ignores the ``mode`` / ``cval`` call arguments** (as
    :class:`MultiLabel` ignores ``cval``).  This is deliberate: a B-spline
    is only self-consistent -- the interpolation property (reproducing the
    samples) only holds -- when the prefilter and the gather share a
    boundary, and only the mirror initialisation is implemented.  The
    result is bit-exact with ``scipy.ndimage.map_coordinates(order=3,
    mode='mirror')``.  A mode-aware prefilter (matching ``scipy`` for
    ``nearest`` / ``reflect`` / ... ) is a future extension (cf.
    ``docs/feature-requests/boundary-mode-parity.md``).

    Per the "loud fallbacks" tenet, the override is **announced**: an
    explicit non-mirror ``mode`` (or a non-zero ``cval``) raises a
    :class:`CubicBSplineBoundaryWarning` (the bare default
    ``mode='constant', cval=0`` is treated as unspecified and is silent).

    Reproducibility: the prefilter is a recursive IIR filter, so its engine is
    the ``driver`` axis (the divergent op ``geometry.cubic_bspline_prefilter``).
    ``driver='auto'`` (default) runs the parallel ``associative_scan`` on GPU /
    sequential ``scan`` on CPU, which **reassociate the long pole filter
    differently** -- so a cubic resample can differ CPU-vs-GPU beyond a ULP
    (unlike the ``order``-0/1 kernels, which are bit-identical across backends).
    Force ``driver='sequential'`` (or wrap in ``nitrix.reproducible()``) for a
    cross-platform-stable result; ``nitrix.divergent_ops()`` reports the budget.
    """

    driver: str = 'auto'

    differentiable_in_values: ClassVar[bool] = True
    differentiable_in_coords: ClassVar[bool] = True
    prefers_separable_resample: ClassVar[bool] = True

    def _axis_taps_weights(
        self,
        coord: Float[Array, 'n'],
    ) -> Tuple[Int[Array, 'n 4'], Float[Array, 'n 4']]:
        base = jnp.floor(coord)
        frac = coord - base
        lower = base.astype(jnp.int32)
        taps = jnp.stack(
            [lower - 1, lower, lower + 1, lower + 2],
            axis=-1,
        )
        # Cubic B-spline basis beta^3 at the four tap distances.
        comp = 1.0 - frac
        weights = jnp.stack(
            [
                comp**3 / 6.0,
                2.0 / 3.0 - frac**2 + frac**3 / 2.0,
                2.0 / 3.0 - comp**2 + comp**3 / 2.0,
                frac**3 / 6.0,
            ],
            axis=-1,
        )
        return taps, weights

    def _prepare(
        self,
        image: Float[Array, '*spatial c'],
        mode: BoundaryMode,
    ) -> Float[Array, '*spatial c']:
        return _bspline_prefilter(image, mode, driver=self.driver)

    def sample(
        self,
        image: Float[Array, '*spatial c'],
        coords: Float[Array, '*out_spatial ndim'],
        *,
        mode: BoundaryMode,
        cval: float,
    ) -> Float[Array, '*out_spatial c']:
        """Sample the image at ``coords`` by cubic B-spline interpolation.

        Prefilters the image to interpolating coefficients and gathers
        them with the 4-tap cubic basis.  The mirror boundary is forced
        on both steps, so a non-mirror ``mode`` (or non-zero ``cval``) is
        ignored and announced via :class:`CubicBSplineBoundaryWarning`.

        Parameters
        ----------
        image
            Unbatched channel-last image of shape ``(*spatial, c)``.
        coords
            Coordinate field of shape ``(*out_spatial, ndim)``, with
            ``coords[..., d]`` the sample position along input axis ``d``.
        mode
            Requested boundary rule; ignored (mirror is always used) and
            a warning is emitted for any non-mirror choice.
        cval
            Requested constant fill value; ignored (see ``mode``).

        Returns
        -------
        Float[Array, '*out_spatial c']
            The sampled values, one per coordinate and channel.
        """
        # mode / cval are ignored: a B-spline forces the mirror boundary on
        # both the prefilter and the gather (see the class docstring).  The
        # override is announced, not silent.
        _warn_cubic_boundary_ignored(mode, cval)
        coeffs = self._prepare(image, 'mirror')
        return _separable_gather(
            coeffs,
            coords,
            tap_rule=self._axis_taps_weights,
            mode='mirror',
            cval=0.0,
        )


@dataclass(frozen=True)
class CatmullRomCubic:
    """Catmull-Rom cubic-Hermite interpolation -- the cardinal cubic spline.

    The *interpolating* :math:`C^1` cubic Hermite spline whose tangents
    are centred finite differences (tension :math:`a = -1/2`): the curve
    passes through the samples, so -- unlike :class:`CubicBSpline` -- it
    needs **no prefilter**.  The gather is a direct 4-tap separable cubic
    applied to the raw samples; at integer positions it reproduces them
    exactly.

    Coefficients for the 4 taps
    :math:`\\{\\lfloor x\\rfloor - 1, \\lfloor x\\rfloor,
    \\lfloor x\\rfloor + 1, \\lfloor x\\rfloor + 2\\}` at fractional
    position :math:`t = x - \\lfloor x\\rfloor`::

        w_{-1}(t) = -0.5 t^3 +     t^2 - 0.5 t
        w_{ 0}(t) =  1.5 t^3 - 2.5 t^2       + 1
        w_{+1}(t) = -1.5 t^3 + 2.0 t^2 + 0.5 t
        w_{+2}(t) =  0.5 t^3 - 0.5 t^2

    At :math:`t = 0` the weights are ``(0, 1, 0, 0)`` (the sample at
    :math:`\\lfloor x\\rfloor`) and at :math:`t = 1` they are
    ``(0, 0, 1, 0)`` (:math:`\\lfloor x\\rfloor + 1`) -- the
    interpolation property.

    Distinct from the other cubics: :class:`CubicBSpline` is
    *approximating* (it needs the recursive prefilter to interpolate, and
    that prefilter is backend-dependent), and is :math:`C^2`-smooth with a
    partition-of-unity basis; :class:`Lanczos` is a windowed sinc.
    Catmull-Rom is the cardinal cubic of
    OpenCV ``INTER_CUBIC`` / Keras image ``cubic`` / the Keras
    "BilinearInterpolation" STN (a misnomer -- it evaluates a 4x4 Catmull-Rom
    stencil), so a consumer parity-locked to that family (e.g. ilex
    ``fd_net``'s spatial transformer) reproduces the upstream bit-for-bit.

    Smooth in both the image values and the sample coordinates (the basis
    is :math:`C^1` -- piecewise cubic, continuous with continuous first
    derivative across knots), so it is differentiable end-to-end and
    usable for coordinate-driven registration losses.  Having no
    ``map_coordinates`` equivalent, it always uses the explicit separable
    gather (no platform branch) -- and, with no prefilter, no
    backend-dependent scan either.

    Notes
    -----
    The cardinal-cubic basis **is** a partition of unity -- the four
    weights sum to exactly 1 for every :math:`t` (the
    :math:`t^3`/:math:`t^2`/:math:`t^1` coefficients cancel across the
    taps) -- so it reproduces constants *and* linear ramps
    exactly (the Keys ``a = -1/2`` kernel has third-order accuracy).  No
    renormalisation is applied (none is needed -- and the raw cardinal weights
    are what the parity-locked consumers pin against).  The kernel is *not*,
    however, a **convex** combination: the outer taps carry small negative
    weights (the ``w_{-1}`` / ``w_{+2}`` lobes reach ``~-0.07``), so the output
    can overshoot the local sample range (ringing near sharp edges) -- unlike
    :class:`Linear` (non-negative weights), whose output is bounded by its
    neighbours.  This overshoot is the price of the higher-order fit; where a
    strictly bounded, non-ringing resample is required, prefer :class:`Linear`.
    """

    differentiable_in_values: ClassVar[bool] = True
    differentiable_in_coords: ClassVar[bool] = True
    prefers_separable_resample: ClassVar[bool] = True

    def _axis_taps_weights(
        self,
        coord: Float[Array, 'n'],
    ) -> Tuple[Int[Array, 'n 4'], Float[Array, 'n 4']]:
        base = jnp.floor(coord)
        frac = coord - base
        lower = base.astype(jnp.int32)
        taps = jnp.stack(
            [lower - 1, lower, lower + 1, lower + 2],
            axis=-1,
        )
        # Cardinal cubic (tension a = -1/2) at the four tap distances; the
        # raw Hermite weights (no renormalisation -- see the class docstring).
        t = frac
        t2 = t * t
        t3 = t2 * t
        w_m1 = -0.5 * t3 + t2 - 0.5 * t
        w_0 = 1.5 * t3 - 2.5 * t2 + 1.0
        w_p1 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
        w_p2 = 0.5 * t3 - 0.5 * t2
        weights = jnp.stack([w_m1, w_0, w_p1, w_p2], axis=-1)
        return taps, weights

    def _prepare(
        self,
        image: Float[Array, '*spatial c'],
        mode: BoundaryMode,
    ) -> Float[Array, '*spatial c']:
        # Direct interpolator: it gathers the raw samples, no prefilter.
        del mode
        return image

    def sample(
        self,
        image: Float[Array, '*spatial c'],
        coords: Float[Array, '*out_spatial ndim'],
        *,
        mode: BoundaryMode,
        cval: float,
    ) -> Float[Array, '*out_spatial c']:
        """Sample the image at ``coords`` by cardinal-cubic interpolation.

        Parameters
        ----------
        image
            Unbatched channel-last image of shape ``(*spatial, c)``.
        coords
            Coordinate field of shape ``(*out_spatial, ndim)``, with
            ``coords[..., d]`` the sample position along input axis ``d``.
        mode
            Boundary rule for out-of-range taps -- one of ``'constant'``,
            ``'nearest'``, ``'wrap'``, ``'mirror'`` or ``'reflect'``.
        cval
            Fill value for out-of-range taps under ``mode='constant'``.

        Returns
        -------
        Float[Array, '*out_spatial c']
            The sampled values, one per coordinate and channel.
        """
        return _separable_gather(
            image,
            coords,
            tap_rule=self._axis_taps_weights,
            mode=mode,
            cval=cval,
        )


@dataclass(frozen=True)
class MultiLabel:
    """ANTs-style multi-label interpolation for label / segmentation maps.

    Naive nearest-neighbour resampling of a segmentation aliases label
    boundaries and drops thin structures; plain linear interpolation
    invents non-label values between labels.  ``MultiLabel`` (ANTs
    ``MultiLabel`` / ITK label interpolators) does neither: each label's
    binary indicator mask is resampled with a smooth ``inner`` kernel,
    and the output at each voxel is the **arg-max** label over those
    resampled scores.  Boundaries are anti-aliased (the inner kernel
    smooths each mask), yet the output is restricted to the input label
    set (no invented values).

    The implementation is a memory-bounded running arg-max fold over the
    static label set -- ``O(K)`` inner resamples carrying two
    output-shaped arrays, never a ``K``-channel stack.

    Attributes
    ----------
    labels
        The label values to consider, as a **static** tuple.  This must
        be supplied explicitly (and exactly match the values stored in
        ``image``): ``jnp.unique`` is dynamically shaped and would break
        ``jit``.  Contiguous labels are ``tuple(range(K))``.  Earlier
        labels win ties, and a sample with no label support (e.g. fully
        out of bounds under ``mode='constant'``) resolves to
        ``labels[0]`` -- so list the background label **first**.
    inner
        The kernel used to resample each indicator mask.  Default
        ``Linear()``.  The *support width* of the inner kernel sets how
        much anti-aliasing happens: :class:`Linear` (2-tap) is a cheap
        sub-voxel area-weighting that refines :class:`NearestNeighbour`
        at label junctions but, being narrow, still drops sub-pixel
        structures under heavy downsampling (it coincides with nearest
        neighbour on a 2x decimation).  A **wider** inner --
        :class:`Lanczos` -- area-weights over a larger neighbourhood, so
        it preserves thin structures and gives the stronger anti-aliasing
        the ANTs ``MultiLabel`` (a Gaussian) is known for, at higher
        cost.  :class:`NearestNeighbour` as the inner degenerates to
        plain nearest-neighbour label resampling.

    Notes
    -----
    Not differentiable: the arg-max is a hard selection, so the gradient
    with respect to both the image values and the coordinates is zero
    (the call does not error under ``jax.grad`` -- it returns zeros).
    ``cval`` is ignored; out-of-support samples resolve to ``labels[0]``
    as above.
    """

    labels: Tuple[int, ...]
    inner: Interpolator = Linear()

    differentiable_in_values: ClassVar[bool] = False
    differentiable_in_coords: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if len(self.labels) == 0:
            raise ValueError('MultiLabel: labels must be a non-empty tuple.')
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(
                f'MultiLabel: labels must be distinct; got {self.labels!r}.'
            )

    def _score(
        self,
        image: Float[Array, '*spatial c'],
        coords: Float[Array, '*out_spatial ndim'],
        label: int,
        mode: BoundaryMode,
    ) -> Float[Array, '*out_spatial c']:
        # Resample the binary indicator of ``label``; an out-of-bounds
        # tap contributes 0 membership (cval=0), not the image cval.
        indicator = (image == label).astype(image.dtype)
        return self.inner.sample(indicator, coords, mode=mode, cval=0.0)

    def sample(
        self,
        image: Float[Array, '*spatial c'],
        coords: Float[Array, '*out_spatial ndim'],
        *,
        mode: BoundaryMode,
        cval: float,
    ) -> Float[Array, '*out_spatial c']:
        """Sample the label map at ``coords`` by per-label arg-max.

        Resamples each label's indicator mask with the ``inner`` kernel
        and returns, per voxel, the label with the highest resampled
        score, keeping the output inside the input label set.

        Parameters
        ----------
        image
            Unbatched channel-last label map of shape ``(*spatial, c)``;
            entries are drawn from ``labels``.
        coords
            Coordinate field of shape ``(*out_spatial, ndim)``, with
            ``coords[..., d]`` the sample position along input axis ``d``.
        mode
            Boundary rule for out-of-range taps, forwarded to the inner
            kernel.
        cval
            Ignored: out-of-support samples resolve to ``labels[0]``, not
            a fill value.

        Returns
        -------
        Float[Array, '*out_spatial c']
            The per-voxel arg-max label, restricted to ``labels``.
        """
        # Running arg-max over the static label set: O(K) inner samples,
        # O(1) label channels resident.  ``cval`` is intentionally unused
        # (label outputs come from ``labels``; see the class docstring).
        del cval
        first = self.labels[0]
        best_score = self._score(image, coords, first, mode)
        best_label = jnp.full_like(best_score, first)
        for label in self.labels[1:]:
            score = self._score(image, coords, label, mode)
            take = score > best_score
            best_label = jnp.where(
                take,
                jnp.asarray(label, best_label.dtype),
                best_label,
            )
            best_score = jnp.where(take, score, best_score)
        return best_label


# ---------------------------------------------------------------------------
# The dispatcher seam
# ---------------------------------------------------------------------------


def _sample_at_coords(
    image: Float[Array, '*spatial c'],
    coords: Float[Array, '*out_spatial ndim'],
    *,
    method: Interpolator,
    mode: BoundaryMode,
    cval: float,
) -> Float[Array, '*out_spatial c']:
    """Sample ``image`` at continuous ``coords`` with the given method.

    The seam :func:`spatial_transform` calls (and :func:`resample` falls
    back to for non-separable methods) -- the one place a ``backend`` axis
    will branch once a Pallas gather kernel lands.  Operates on the
    *unbatched* core shapes; leading batch axes are composed by the
    caller.

    Parameters
    ----------
    image
        Unbatched channel-last image of shape ``(*spatial, c)``.
    coords
        Coordinate field of shape ``(*out_spatial, ndim)``, with
        ``coords[..., d]`` the sample position along input axis ``d``.
    method
        The interpolation kernel to sample with.
    mode
        Boundary rule for out-of-range taps.
    cval
        Fill value for out-of-range taps under ``mode='constant'``.

    Returns
    -------
    Float[Array, '*out_spatial c']
        The sampled values, one per coordinate and channel.
    """
    return method.sample(image, coords, mode=mode, cval=cval)


def _resample_on_grid(
    image: Float[Array, '*spatial c'],
    axes_coords: Sequence[Float[Array, 'out']],
    *,
    method: Interpolator,
    mode: BoundaryMode,
    cval: float,
) -> Float[Array, '*out_spatial c']:
    """Sample a separable resize grid with the given method.

    The seam :func:`resample` calls.  ``axes_coords`` are the per-axis
    1-D sample-coordinate vectors of the (outer-product) resize grid.  A
    separable gather kernel takes the cheap per-axis 1-D-pass form
    (:func:`_separable_resample`, :math:`O(N \\cdot T \\cdot
    \\mathrm{ndim})`); any other method (e.g. :class:`MultiLabel`) falls
    back to the dense meshgrid gather via :func:`_sample_at_coords`.

    Parameters
    ----------
    image
        Unbatched channel-last image of shape ``(*spatial, c)``.
    axes_coords
        Per-axis 1-D sample-coordinate vectors of the (outer-product)
        resize grid; ``axes_coords[d]`` targets input spatial axis ``d``.
    method
        The interpolation kernel to sample with.
    mode
        Boundary rule for out-of-range taps.
    cval
        Fill value for out-of-range taps under ``mode='constant'``.

    Returns
    -------
    Float[Array, '*out_spatial c']
        The resampled image on the resize grid.
    """
    if (
        isinstance(method, _SeparableKernel)
        and method.prefers_separable_resample
    ):
        # ``_prepare`` is the identity for the direct kernels and the
        # B-spline prefilter for ``CubicBSpline`` (applied once, on the
        # full image, before the separable per-axis passes).  The B-spline
        # forces the mirror boundary on both prefilter and gather (it
        # ignores ``mode`` / ``cval``; see ``CubicBSpline``).
        is_bspline = isinstance(method, CubicBSpline)
        if is_bspline:
            _warn_cubic_boundary_ignored(mode, cval)
        prepared = method._prepare(image, 'mirror' if is_bspline else mode)
        return _separable_resample(
            prepared,
            axes_coords,
            tap_rule=method._axis_taps_weights,
            mode='mirror' if is_bspline else mode,
            cval=0.0 if is_bspline else cval,
        )
    grids = jnp.meshgrid(*axes_coords, indexing='ij')
    coords = jnp.stack(grids, axis=-1)
    return _sample_at_coords(
        image,
        coords,
        method=method,
        mode=mode,
        cval=cval,
    )
