# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pallas Triton fused ESM Demons force (v4 Phase 5a).

Fuses the symmetric ``∇warped`` central-difference stencil with the ESM force
elementwise combine into **one** tiled kernel.  The pure-JAX
:meth:`register._force._BoundDemons.update` lets XLA fuse the elementwise force
chain (``diff``, ``j``, ``denom``, ``scale``, ``u``) but **breaks fusion at the
``correlate1d`` stencil** that builds ``∇warped`` -- so ``∇warped`` (``M·ndim``)
round-trips HBM before the force consumes it.  This kernel reads ``warped`` (with
a 1-voxel halo for the central difference), ``fixed`` and the hoisted ``∇F``
once, computes ``∇warped`` in-tile, and writes only the velocity update ``u``
(``M·ndim``) -- collapsing the gradient round-trip.

Parity oracle = ``_BoundDemons.update`` (ULP-equal, **including** the 0a denom
guard ``denom > eps``).  Scope: **isotropic** (``rel_spacing=None``) single-pair
GPU.  The public dispatcher (``register._force``) falls back to the JAX path for
anisotropic spacing, CPU, the cohort ``vmap``, or an untileable shape (a loud
``NitrixBackendFallback``).

Implementation detail: never import from ``nitrix._kernels.cuda`` directly.  Use
``register.DemonsForce(backend=...)`` which handles dispatch and fallback
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

    Caught by the dispatcher in ``register._force`` and translated into a
    ``NitrixBackendFallback`` warning (the JAX path runs instead).
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
    """Largest friendly tile dividing ``extent`` (``None`` if none divides)."""
    for c in _TILE_LADDER.get(ndim, _DEFAULT_LADDER):
        if extent % c == 0:
            return c
    return None


def _build_kernel(
    ndim: int, tiles: tuple[int, ...], alpha: float, eps: float
) -> Callable[..., None]:
    """ESM force over one spatial tile (the stencil + the 0a-guarded combine).

    ``alpha``/``eps`` are baked in as compile-time constants (mirrors the
    semiring kernel's static-config build).
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
    """Fused ESM Demons force for one isotropic single-pair field.

    Returns ``u = (F − warped)·J / (|J|² + α²(F − warped)²)`` with
    ``J = ½(∇F + ∇warped)`` -- ULP-equal to ``_BoundDemons.update`` for
    ``rel_spacing=None`` (the conversion to voxel units is the identity there).

    Raises
    ------
    PallasNotTileable
        If a spatial extent admits no friendly tile, or ``grad_fixed`` does not
        carry the trailing ``ndim`` axis.  The dispatcher catches this and runs
        the JAX path with a ``NitrixBackendFallback`` warning.
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
            out_shape=jax.ShapeDtypeStruct(warped.shape + (ndim,), warped.dtype),
            compiler_params=plgpu.CompilerParams(),
            name='demons_esm_force',
        )(warped_p, fixed, grad_fixed),
    )
