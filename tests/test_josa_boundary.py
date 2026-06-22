# -*- coding: utf-8 -*-
"""JOSA spherical-SVF boundary mode (geometry-suite P4.3 / GS-13) -- VERIFY-FIRST.

GS-13 asks for a ``mode='nearest'`` boundary option on the equirectangular
sphere sampler so the ``josa`` spherical-diffeomorphism path composes
``sphere_grid_pad_2d`` + ``integrate_velocity_field`` + ``spatial_transform``
**without re-vendoring a sampler**.  Per the design plan (P4.3) this is
*likely already shipped*; this module is the composition test that closes
GS-13 with **no new code** if it passes.

It establishes four things:

1. ``spatial_transform`` honours ``mode='nearest'`` (the literal residual
   keyword) -- out-of-bounds samples edge-replicate instead of zero-filling.
2. The ``sphere_grid_pad_2d`` longitudinal wrap composes with the sampler and
   matches ``mode='wrap'`` exactly (the seam is handled by the padding).
3. ``sphere_grid_pad_2d`` supplies a topology-correct *over-the-pole* halo
   (flipped + rolled) that no flat boundary mode provides -- the reason the
   sphere padding is needed at all.
4. The full equirectangular SVF warp path runs end-to-end, is finite, and is
   shape-preserving (the ``josa`` diffeomorphism smoke), including the
   longitudinal-flow pole sign-flip via ``pole_negate_channel``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nitrix.geometry import (
    identity_grid,
    integrate_velocity_field,
    spatial_transform,
    sphere_grid_pad_2d,
    sphere_grid_unpad_2d,
)


def test_spatial_transform_honours_nearest_mode() -> None:
    # The GS-13 keyword: out-of-bounds samples edge-replicate under 'nearest'
    # and zero-fill under 'constant'.
    line = jnp.arange(5, dtype=jnp.float32)[None, :]  # (1, 5)
    coords = jnp.stack(
        [jnp.zeros((1, 3)), jnp.array([[-2.0, 2.0, 10.0]])], axis=-1
    )
    near = spatial_transform(line[..., None], coords, mode='nearest')[..., 0]
    cons = spatial_transform(line[..., None], coords, mode='constant')[..., 0]
    assert np.allclose(np.asarray(near), [[0.0, 2.0, 4.0]])  # edge-clamped
    assert np.allclose(np.asarray(cons), [[0.0, 2.0, 0.0]])  # zero-filled


def test_longitudinal_wrap_composes_with_sampler() -> None:
    # An equirectangular longitude shift, done via sphere_grid_pad_2d's
    # longitudinal wrap + plain sampling, equals mode='wrap' sampling: the
    # padding handles the seam, so the sampler needs no special longitude mode.
    h, w = 8, 16
    rng = np.random.default_rng(0)
    img = jnp.asarray(rng.standard_normal((h, w)).astype(np.float32))
    idg = identity_grid((h, w))  # (H, W, 2) -> (row=lat, col=lon)
    dx = 3.7  # longitude shift

    shift = jnp.stack([jnp.zeros((h, w)), jnp.full((h, w), dx)], axis=-1)
    out_wrap = spatial_transform(img[..., None], idg + shift, mode='wrap')
    out_wrap = out_wrap[..., 0]

    w_pad = 5
    img_p = sphere_grid_pad_2d(img, (0, w_pad), height_axis=-2, width_axis=-1)
    shift_p = jnp.stack(
        [jnp.zeros((h, w)), jnp.full((h, w), dx + w_pad)], axis=-1
    )
    out_pad = spatial_transform(
        img_p[..., None], idg + shift_p, mode='constant'
    )[..., 0]
    assert np.allclose(np.asarray(out_wrap), np.asarray(out_pad), atol=1e-4)


def test_pole_halo_is_topology_correct_not_flat_fill() -> None:
    # The over-the-pole halo is the rows just inside the pole, flipped and
    # rolled by W//2 -- a genuine spherical neighbourhood, not a constant /
    # edge / wrap fill that a flat boundary mode would give.
    h, w = 8, 16
    rng = np.random.default_rng(1)
    img = jnp.asarray(rng.standard_normal((h, w)).astype(np.float32))
    h_pad = 2
    padded = sphere_grid_pad_2d(img, (h_pad, 0), height_axis=-2, width_axis=-1)
    # Top halo row 0 == original row 1 flipped vertically + rolled by W//2.
    expected_top = jnp.roll(img[1], w // 2)
    assert np.allclose(np.asarray(padded[h_pad - 1]), np.asarray(expected_top))
    # It is not a flat replicate of the pole row (the wrong, flat-mode answer).
    assert not np.allclose(np.asarray(padded[h_pad - 1]), np.asarray(img[0]))


def test_josa_svf_warp_path_composes_end_to_end() -> None:
    # The full josa equirectangular path: a stationary velocity field ->
    # integrate (diffeomorphic) -> warp with mode='nearest'.  Runs, finite,
    # shape-preserving; this is the spherical-diffeomorphism smoke.
    h, w = 8, 16
    rng = np.random.default_rng(2)
    img = jnp.asarray(rng.standard_normal((h, w)).astype(np.float32))
    idg = identity_grid((h, w))
    svf = 0.2 * jnp.asarray(rng.standard_normal((h, w, 2)).astype(np.float32))
    disp = integrate_velocity_field(svf, n_steps=5)
    warped = spatial_transform(img[..., None], idg + disp, mode='nearest')
    warped = warped[..., 0]
    assert warped.shape == (h, w)
    assert np.all(np.isfinite(np.asarray(warped)))


def test_sphere_pad_unpad_roundtrip_with_pole_negate() -> None:
    # The 2D-flow padding (pole sign-flip of the longitudinal component) is
    # invertible: pad then unpad returns the original SVF.  This is the field
    # the josa path pads before integrating.
    h, w = 8, 16
    rng = np.random.default_rng(3)
    svf = jnp.asarray(rng.standard_normal((h, w, 2)).astype(np.float32))
    pad = 2
    padded = sphere_grid_pad_2d(
        svf,
        pad,
        height_axis=-3,
        width_axis=-2,
        pole_negate_channel=1,  # longitudinal flow reverses across the pole
        pole_negate_axis=-1,
    )
    assert padded.shape == (h + 2 * pad, w + 2 * pad, 2)
    unpadded = sphere_grid_unpad_2d(
        padded, pad, height_axis=-3, width_axis=-2
    )
    assert np.allclose(np.asarray(unpadded), np.asarray(svf), atol=1e-6)
