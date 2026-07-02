# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pallas-Triton fused efficient second-order minimisation (ESM) Demons force.

Fuses the symmetric warped-image central-difference stencil with the ESM force
elementwise combine into a single tiled kernel.  The pure-JAX oracle lets XLA
fuse the elementwise force chain (``diff``, ``j``, ``denom``, ``scale``, ``u``)
but breaks fusion at the :func:`correlate1d` stencil that builds the moving-image
gradient :math:`\\nabla\\mathrm{warped}` -- so that gradient (of size
:math:`M \\cdot \\mathrm{ndim}`) round-trips high-bandwidth memory before the
force consumes it.  This kernel reads ``warped`` (with a 1-voxel halo for the
central difference), ``fixed`` and the hoisted fixed-image gradient
:math:`\\nabla F` once, computes :math:`\\nabla\\mathrm{warped}` in-tile, and
writes only the velocity update ``u`` (of size :math:`M \\cdot \\mathrm{ndim}`),
collapsing the gradient round-trip.

The parity oracle is the pure-JAX Demons update path, to which this kernel is
ULP-equal, including the denominator guard ``denom > eps`` that zeroes the force
where both the gradient and the intensity mismatch vanish.  Scope: isotropic
spacing (``rel_spacing=None``), single moving/fixed pair, on GPU.  The public
dispatcher falls back to the JAX path for anisotropic spacing, CPU, the cohort
``vmap``, or an untileable shape, emitting a loud
:class:`NitrixBackendFallback`.

Notes
-----
This is a private implementation detail.  Never import from
``nitrix._kernels.cuda`` directly; use the public :class:`DemonsForce`
constructor with a chosen ``backend``, which handles dispatch and fallback
observability.
"""

from __future__ import annotations

from typing import Any, Callable, cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jaxtyping import Array, Float

__all__ = [
    'demons_esm_force_pallas',
    'PallasNotTileable',
]


class PallasNotTileable(RuntimeError):
    """The Pallas kernel rejected the requested shape.

    Raised when a spatial extent admits no friendly tile, or when the
    ``grad_fixed`` argument does not carry the trailing gradient axis.  Caught
    by the Demons force dispatcher and translated into a
    :class:`NitrixBackendFallback` warning, so that the JAX path runs instead.
    """


# A 1-voxel halo central difference; tiles must divide each spatial extent.
# The candidate ladder is **ndim-aware**: each Triton program holds its whole
# ``(T+1)^ndim`` haloed tile in registers, so the tile *volume* -- not the side
# -- is what must stay small.  A 64-wide tile is fine in 2-D (~4k elements) but
# a 64**3 tile (~290k) spills registers and wedges ``ptxas``; 3-D therefore caps
# the side at 16.  Largest-friendly-that-divides within the per-ndim ladder.
_TILE_LADDER: dict[int, tuple[int, ...]] = {
    1: (256, 128, 64, 32, 16, 8),
    2: (32, 16, 8),
    3: (16, 8),
}
_DEFAULT_LADDER: tuple[int, ...] = (8,)


def _pick_tile(extent: int, ndim: int) -> int | None:
    """Pick the largest friendly tile side that evenly divides an extent.

    Scans the ``ndim``-aware candidate ladder from the largest side downwards
    and returns the first side length that divides ``extent`` exactly.

    Parameters
    ----------
    extent : int
        Length of the spatial axis to be tiled.
    ndim : int
        Number of spatial dimensions, selecting which candidate ladder to use.

    Returns
    -------
    int or None
        The largest ladder side dividing ``extent`` exactly, or ``None`` if no
        candidate divides it.
    """
    for c in _TILE_LADDER.get(ndim, _DEFAULT_LADDER):
        if extent % c == 0:
            return c
    return None


def _build_kernel(
    ndim: int, tiles: tuple[int, ...], alpha: float, eps: float
) -> Callable[..., None]:
    """Build the per-tile ESM force kernel for a given dimensionality.

    Returns a Pallas kernel closure that, over one spatial tile, computes the
    in-tile warped-image central-difference stencil and the guarded ESM force
    combine.  The ``alpha`` and ``eps`` values are baked in as compile-time
    constants, mirroring the static-config build used by the semiring kernels.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions the kernel operates over.
    tiles : tuple of int
        Per-axis tile side lengths, one entry per spatial dimension.
    alpha : float
        Normalisation weight on the intensity-mismatch term of the ESM
        denominator; its square is precomputed and baked in.
    eps : float
        Denominator floor: the force is zeroed wherever the accumulated
        denominator does not exceed this value.

    Returns
    -------
    Callable
        A Pallas kernel taking the haloed warped, fixed, fixed-gradient, and
        output references and writing the per-tile velocity update in place.
    """
    a2 = alpha * alpha
    full = (slice(None),) * ndim

    def _win(offsets: tuple[int, ...]) -> tuple[Any, ...]:
        # Per-axis dynamic slice into the haloed warped tile: offset 1 is the
        # centre (padded coords), 0 / 2 the central-difference neighbours.
        return tuple(pl.ds(offsets[k], tiles[k]) for k in range(ndim))

    centre = _win((1,) * ndim)

    def kernel(w_ref: Any, f_ref: Any, gf_ref: Any, o_ref: Any) -> None:
        wc = w_ref[centre]
        diff = f_ref[...] - wc
        denom = a2 * diff * diff
        js = []
        for d in range(ndim):
            plus = _win(tuple(2 if k == d else 1 for k in range(ndim)))
            minus = _win(tuple(0 if k == d else 1 for k in range(ndim)))
            grad_d = (w_ref[plus] - w_ref[minus]) * 0.5
            j_d = 0.5 * (gf_ref[(*full, d)] + grad_d)
            js.append(j_d)
            denom = denom + j_d * j_d
        # The 0a double-``where``: zero force where (and only where) both the
        # gradient and the mismatch vanish, with a finite gradient too.
        safe = denom > eps
        scale = jnp.where(safe, diff / jnp.where(safe, denom, 1.0), 0.0)
        for d in range(ndim):
            o_ref[(*full, d)] = scale * js[d]

    return kernel


def demons_esm_force_pallas(
    warped: Float[Array, '*spatial'],
    fixed: Float[Array, '*spatial'],
    grad_fixed: Float[Array, '*spatial ndim'],
    *,
    alpha: float,
    eps: float,
) -> Float[Array, '*spatial ndim']:
    """Fused ESM Demons force for one isotropic single moving/fixed pair.

    Computes the efficient second-order minimisation (ESM) velocity update

    .. math::

        u = \\frac{(F - \\mathrm{warped})\\, J}
                  {|J|^{2} + \\alpha^{2}\\,(F - \\mathrm{warped})^{2}},
        \\qquad
        J = \\tfrac{1}{2}\\,(\\nabla F + \\nabla\\mathrm{warped}),

    where :math:`F` is the fixed image, ``warped`` is the moving image resampled
    into the fixed frame, and :math:`J` is the symmetric ESM gradient.  The
    warped-image gradient :math:`\\nabla\\mathrm{warped}` is formed in-tile by a
    1-voxel-halo central difference with edge padding, matching the pure-JAX
    oracle for isotropic spacing (``rel_spacing=None``), where the conversion to
    voxel units is the identity.  The result is ULP-equal to that oracle,
    including its denominator guard.

    Parameters
    ----------
    warped : Float[Array, '*spatial']
        Moving image resampled into the fixed frame, with ``ndim`` spatial axes.
    fixed : Float[Array, '*spatial']
        Fixed (reference) image, same shape as ``warped``.
    grad_fixed : Float[Array, '*spatial ndim']
        Precomputed spatial gradient of ``fixed``, carrying a trailing axis of
        length ``ndim`` for the per-dimension components.
    alpha : float
        Normalisation weight on the intensity-mismatch term of the ESM
        denominator.
    eps : float
        Denominator floor: the force is zeroed wherever the accumulated
        denominator does not exceed this value.

    Returns
    -------
    Float[Array, '*spatial ndim']
        The per-voxel velocity update, with the same spatial shape as ``warped``
        and a trailing axis of length ``ndim`` giving the update components.

    Raises
    ------
    PallasNotTileable
        If a spatial extent admits no friendly tile, or ``grad_fixed`` does not
        carry the trailing ``ndim`` axis.  The dispatcher catches this and runs
        the JAX path with a :class:`NitrixBackendFallback` warning.
    """
    ndim = warped.ndim
    if grad_fixed.shape != warped.shape + (ndim,):
        raise PallasNotTileable(
            f'grad_fixed shape {grad_fixed.shape} != warped.shape + (ndim,) '
            f'{warped.shape + (ndim,)}.'
        )
    tiles_list = []
    for ext in warped.shape:
        t = _pick_tile(int(ext), ndim)
        if t is None:
            raise PallasNotTileable(
                f'no tile in {_TILE_LADDER.get(ndim, _DEFAULT_LADDER)} divides '
                f'spatial extent {ext}; pad to a friendly shape or use '
                'backend="jax".'
            )
        tiles_list.append(t)
    tiles = tuple(tiles_list)
    grid = tuple(warped.shape[i] // tiles[i] for i in range(ndim))

    # Edge-pad the haloed input so the central difference replicates the border
    # (matches spatial_gradient's mode='nearest' exactly).
    warped_p = jnp.pad(warped, [(1, 1)] * ndim, mode='edge')

    def warped_index_map(*idx: int) -> tuple[int, ...]:
        # Element indexing: the overlapping (tile+2) window starts at element
        # i*tile in the padded array (its centre is the original tile).
        return tuple(idx[i] * tiles[i] for i in range(ndim))

    def blocked_index_map(*idx: int) -> tuple[int, ...]:
        return idx

    def trailing_index_map(*idx: int) -> tuple[int, ...]:
        return (*idx, 0)

    kernel = _build_kernel(ndim, tiles, float(alpha), float(eps))
    return cast(
        Float[Array, '*spatial ndim'],
        pl.pallas_call(
            kernel,
            grid=grid,
            in_specs=[
                pl.BlockSpec(
                    tuple(pl.Element(tiles[i] + 2) for i in range(ndim)),
                    warped_index_map,
                ),
                pl.BlockSpec(tiles, blocked_index_map),
                pl.BlockSpec((*tiles, ndim), trailing_index_map),
            ],
            out_specs=pl.BlockSpec((*tiles, ndim), trailing_index_map),
            out_shape=jax.ShapeDtypeStruct(
                warped.shape + (ndim,), warped.dtype
            ),
            compiler_params=plgpu.CompilerParams(),
            name='demons_esm_force',
        )(warped_p, fixed, grad_fixed),
    )
