# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pallas Triton sliding-window centre-only LNCC force (3-D).

The ANTs / ITK ``ANTSNeighborhoodCorrelation`` force is single-pass: the
local-correlation derivative is attributed to the window centre, so it needs
only the five windowed sums and no second box-sum pass.  This kernel evaluates
it with ITK's scanning window -- a program owns a :math:`(T_y, T_z)` column of
the volume and scans along :math:`x`, carrying the five window sums
(:math:`\\sum m`, :math:`\\sum f`, :math:`\\sum m^2`, :math:`\\sum f^2`,
:math:`\\sum m f`) as running state.  At each step it drops the trailing
:math:`x`-plane and adds the leading one (:math:`O(1)` in :math:`x`), each plane
being the :math:`(2r+1)^2` :math:`(y, z)` sum via ``pl.ds`` ref-slices.  This is
:math:`O(\\text{window area})` per voxel, not :math:`O(\\text{window volume})`:
unlike a direct :math:`(2r+1)^3` accumulation (which merely ties the JAX
integral image), it beats it, and the advantage grows with volume (a single
streamed pass versus the integral image's many cumulative-sum passes).

Per output voxel it then forms the centre-only scalar and multiplies by the
gradient of the warped image (central difference; the :math:`x`-derivative
reuses the scan neighbours, while :math:`y` and :math:`z` use ref-slices).  One
symmetric pad serves both the window and the gradient (a width-1 symmetric pad
equals the edge pad the warped gradient wants), so the result is tolerance-equal
to the centre-derivative LNCC force update, boundary included.

Scope: 3-D isotropic single-pair GPU, radius no greater than
:data:`_MAX_RADIUS`.  Anything else falls back to the JAX path with a loud
fallback warning.

Implementation detail: never import this module directly; use the public LNCC
force estimator with the centre derivative and an explicit backend selector.
"""

from __future__ import annotations

from typing import Any, Callable, cast

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jaxtyping import Array, Float

__all__ = [
    'lncc_center_force_pallas',
    'PallasNotTileable',
    '_MAX_RADIUS',
]


class PallasNotTileable(RuntimeError):
    """The Pallas kernel rejected the requested shape / radius (-> JAX path)."""


_MAX_RADIUS = 4
# (y, z) tile; the scanned x-axis is not tiled.  A column program holds a few
# (Ty, Tz) planes of running state -- keep the tile modest.
_TILE_CANDIDATES: tuple[int, ...] = (32, 16, 8)


def _pick_tile(extent: int) -> int:
    for c in _TILE_CANDIDATES:
        if extent % c == 0:
            return c
    return _TILE_CANDIDATES[
        -1
    ]  # a multiple of the smallest tile always divides


def _next_tileable(extent: int, r: int) -> int:
    """Round ``extent`` up to a tileable size with a boundary-safe pad.

    Any size then tiles (the scanned :math:`x`-axis is never tiled, so only
    :math:`y` and :math:`z` are padded -- a small symmetric pad, cropped back),
    so the kernel handles arbitrary volumes (e.g. :math:`220^3`) and the speed
    advantage still holds (the pad is a couple of percent on two axes).  The pad
    must be :math:`0` or at least :math:`r`: a smaller pad would let an original
    boundary voxel's window reach past the (correct, symmetric) tileability pad
    into the kernel's own :math:`r`-halo, which reflects the padded -- not the
    original -- array (a boundary error).  When the next multiple of the tile
    would pad by :math:`1` to :math:`r-1`, round up one more tile so the
    symmetric tileability pad fully covers the window's reach.

    Parameters
    ----------
    extent : int
        Size of the axis (:math:`y` or :math:`z`) to be rounded up.
    r : int
        Window radius, so the sliding window spans :math:`2r+1` voxels per axis.

    Returns
    -------
    int
        The smallest tileable extent that is at least ``extent`` and whose pad
        over ``extent`` is either zero or at least ``r``.
    """
    base = _TILE_CANDIDATES[-1]
    target = ((extent + base - 1) // base) * base
    pad = target - extent
    if 0 < pad < r:
        target += base
    return target


def _build_sliding_center_kernel(
    nx: int, ty: int, tz: int, r: int, eps: float, n: float
) -> Callable[..., None]:
    win = 2 * r + 1

    def planes(w_ref: Any, f_ref: Any, p: Any) -> tuple[Any, ...]:
        # The five (y, z)-windowed sums of the x-plane at padded depth p, each a
        # (1, Ty, Tz) tile.  The (dy, dz) window is ROLLED with nested
        # ``fori_loop``s (dynamic ``pl.ds`` loads) so the kernel stays tiny and
        # cold-compile is radius-independent (an unrolled (2r+1)^2 x 5-sum body
        # blows up ptxas -- 84 s even at 16^3).
        z = jnp.zeros((1, ty, tz), dtype=w_ref.dtype)

        def over_z(dy: Any, acc: Any) -> Any:
            def inner(dz: Any, a: Any) -> Any:
                sm, sf, smm, sff, smf = a
                sl = (pl.ds(p, 1), pl.ds(dy, ty), pl.ds(dz, tz))
                wv = w_ref[sl]
                fv = f_ref[sl]
                return (
                    sm + wv,
                    sf + fv,
                    smm + wv * wv,
                    sff + fv * fv,
                    smf + wv * fv,
                )

            return lax.fori_loop(0, win, inner, acc)

        return cast(
            'tuple[Any, ...]', lax.fori_loop(0, win, over_z, (z, z, z, z, z))
        )

    def emit(
        w_ref: Any, f_ref: Any, ox: Any, s: tuple[Any, ...], o_ref: Any
    ) -> None:
        sm, sf, smm, sff, smf = s
        cen = (pl.ds(ox + r, 1), pl.ds(r, ty), pl.ds(r, tz))
        wc = w_ref[cen]
        fc = f_ref[cen]
        s_ff = sff - sf * sf / n
        s_mm = smm - sm * sm / n
        s_fm = smf - sf * sm / n
        f_a = fc - sf / n
        m_a = wc - sm / n
        safe = (s_ff > eps) & (s_mm > eps)
        denom = jnp.where(safe, s_ff * s_mm, 1.0)
        s_mm_safe = jnp.where(safe, s_mm, 1.0)
        scalar = jnp.where(
            safe, 2.0 * s_fm / denom * (f_a - s_fm / s_mm_safe * m_a), 0.0
        )
        # ∇warped central difference (scan axis x = spatial axis 0, then y, z).
        gx = (
            w_ref[pl.ds(ox + r + 1, 1), pl.ds(r, ty), pl.ds(r, tz)]
            - w_ref[pl.ds(ox + r - 1, 1), pl.ds(r, ty), pl.ds(r, tz)]
        ) * 0.5
        gy = (
            w_ref[pl.ds(ox + r, 1), pl.ds(r + 1, ty), pl.ds(r, tz)]
            - w_ref[pl.ds(ox + r, 1), pl.ds(r - 1, ty), pl.ds(r, tz)]
        ) * 0.5
        gz = (
            w_ref[pl.ds(ox + r, 1), pl.ds(r, ty), pl.ds(r + 1, tz)]
            - w_ref[pl.ds(ox + r, 1), pl.ds(r, ty), pl.ds(r - 1, tz)]
        ) * 0.5
        o_ref[pl.ds(ox, 1), :, :, 0] = scalar * gx
        o_ref[pl.ds(ox, 1), :, :, 1] = scalar * gy
        o_ref[pl.ds(ox, 1), :, :, 2] = scalar * gz

    def kernel(w_ref: Any, f_ref: Any, o_ref: Any) -> None:
        # Initial window sum over padded x in [0, 2r], emit output x = 0.
        acc = planes(w_ref, f_ref, 0)
        for p in range(1, win):
            pp = planes(w_ref, f_ref, p)
            acc = tuple(a + b for a, b in zip(acc, pp))
        emit(w_ref, f_ref, 0, acc, o_ref)

        def body(ox: Any, s: tuple[Any, ...]) -> tuple[Any, ...]:
            # window padded[ox .. ox+2r]: drop plane ox-1, add plane ox+2r.
            old = planes(w_ref, f_ref, ox - 1)
            new = planes(w_ref, f_ref, ox + 2 * r)
            s = tuple(a - o + nw for a, o, nw in zip(s, old, new))
            emit(w_ref, f_ref, ox, s, o_ref)
            return s

        lax.fori_loop(1, nx, body, acc)

    return kernel


def lncc_center_force_pallas(
    warped: Float[Array, '*spatial'],
    fixed: Float[Array, '*spatial'],
    *,
    radius: int,
    eps: float,
) -> Float[Array, '*spatial ndim']:
    """Fused sliding-window centre-only LNCC force (3-D, isotropic).

    Computes the local-normalised-cross-correlation registration force with the
    derivative attributed to the window centre, over an isotropic cubic window
    of side :math:`2r+1`.  The volume is padded symmetrically (once for both the
    windowed sums and the central-difference gradient), scanned along :math:`x`
    with running window sums, and the result is tolerance-equal to the
    centre-derivative LNCC force update, boundary included.

    Parameters
    ----------
    warped : Float[Array, '*spatial']
        The moving image resampled into the fixed frame, a 3-D volume of shape
        ``(nx, ny, nz)``.
    fixed : Float[Array, '*spatial']
        The fixed (target) image, the same shape as ``warped``.
    radius : int
        Window radius; the sliding window spans :math:`2r+1` voxels per axis.
        Must lie in ``[1, _MAX_RADIUS]``.
    eps : float
        Variance floor; window voxels whose warped or fixed local variance does
        not exceed ``eps`` contribute a zero force.

    Returns
    -------
    Float[Array, '*spatial ndim']
        The per-voxel force, shape ``(nx, ny, nz, 3)``, the centre-only LNCC
        scalar times the central-difference gradient of ``warped`` along each of
        the three spatial axes.

    Raises
    ------
    PallasNotTileable
        Not 3-D, radius out of range, or the warped and fixed shapes differ.
    """
    if warped.ndim != 3:
        raise PallasNotTileable(
            f'sliding-window kernel is 3-D only; got ndim={warped.ndim}.'
        )
    r = int(radius)
    if r < 1 or r > _MAX_RADIUS:
        raise PallasNotTileable(
            f'radius {r} outside [1, {_MAX_RADIUS}]; JAX path runs instead.'
        )
    if fixed.shape != warped.shape:
        raise PallasNotTileable(
            f'warped/fixed shape mismatch {warped.shape} vs {fixed.shape}.'
        )
    nx, ny, nz = (int(s) for s in warped.shape)
    # Pad (y, z) up to the next tileable size (symmetric, so the windowed sums
    # of the original voxels see the correct boundary -> result is interior-
    # exact and boundary-matching); x is scanned, never tiled.  An already-
    # aligned size pads by 0.
    nyp, nzp = _next_tileable(ny, r), _next_tileable(nz, r)
    if (nyp, nzp) != (ny, nz):
        pad = [(0, 0), (0, nyp - ny), (0, nzp - nz)]
        warped = jnp.pad(warped, pad, mode='symmetric')
        fixed = jnp.pad(fixed, pad, mode='symmetric')
    ty, tz = _pick_tile(nyp), _pick_tile(nzp)
    n = float((2 * r + 1) ** 3)

    w_sym = jnp.pad(warped, [(r, r)] * 3, mode='symmetric')
    f_sym = jnp.pad(fixed, [(r, r)] * 3, mode='symmetric')

    def in_map(j: int, k: int) -> tuple[int, ...]:
        return (0, j * ty, k * tz)

    in_block = (nx + 2 * r, pl.Element(ty + 2 * r), pl.Element(tz + 2 * r))
    u = pl.pallas_call(
        _build_sliding_center_kernel(nx, ty, tz, r, float(eps), n),
        grid=(nyp // ty, nzp // tz),
        in_specs=[
            pl.BlockSpec(in_block, in_map),
            pl.BlockSpec(in_block, in_map),
        ],
        out_specs=pl.BlockSpec((nx, ty, tz, 3), lambda j, k: (0, j, k, 0)),
        out_shape=jax.ShapeDtypeStruct((nx, nyp, nzp, 3), warped.dtype),
        compiler_params=plgpu.CompilerParams(),
        name='lncc_center_sliding',
    )(w_sym, f_sym)
    # crop the (y, z) tileability pad back to the original extents
    return cast(Float[Array, '*spatial ndim'], u[:, :ny, :nz])
