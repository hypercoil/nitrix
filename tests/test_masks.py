# -*- coding: utf-8 -*-
"""V5a: region masks for the diffeomorphic recipes (ANTs ``-x`` parity).

A fixed-grid ``mask`` gates the driving force to a region -- the masked area
drives the deformation, the rest follows by regularisation.  An all-ones mask
reduces exactly to the unmasked recipe; a partial mask still recovers (and stays
diffeomorphic).
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nitrix.geometry import (  # noqa: E402
    identity_grid,
    integrate_velocity_field,
    spatial_transform,
)
from nitrix.metrics import ncc  # noqa: E402
from nitrix.register import (  # noqa: E402
    DemonsSpec,
    SyNSpec,
    diffeomorphic_demons_register,
    greedy_syn_register,
)
from nitrix.smoothing import gaussian  # noqa: E402


def _blobs(n=64):
    yy, xx = np.mgrid[0:n, 0:n].astype('float64')
    img = np.zeros((n, n), dtype='float64')
    for cy, cx, s, a in [
        (0.31, 0.38, 0.13, 1.0),
        (0.62, 0.69, 0.16, 0.7),
        (0.72, 0.31, 0.11, 0.6),
        (0.44, 0.75, 0.14, 0.5),
    ]:
        img += a * np.exp(
            -((xx - cx * n) ** 2 + (yy - cy * n) ** 2) / (2 * (s * n) ** 2)
        )
    return jnp.asarray(img)


def _smooth_velocity(shape, sigma, scale, seed):
    rng = np.random.RandomState(seed)
    v = rng.standard_normal(shape + (2,))
    v = np.moveaxis(v, -1, 0)
    v = np.asarray(gaussian(jnp.asarray(v), sigma=sigma, spatial_rank=2))
    v = np.moveaxis(v, 0, -1)
    return jnp.asarray(scale * v)


def _warp(image, v):
    grid = identity_grid(image.shape, dtype=image.dtype)
    s = integrate_velocity_field(v)
    return spatial_transform(image[..., None], grid + s, mode='nearest')[..., 0]


def _disk_mask(n=64, r=24):
    yy, xx = np.mgrid[0:n, 0:n]
    return jnp.asarray(
        ((xx - n / 2) ** 2 + (yy - n / 2) ** 2 < r * r).astype('float64')
    )


def test_ones_mask_equals_no_mask_syn():
    fixed = _blobs(64)
    moving = _warp(fixed, _smooth_velocity((64, 64), 8.0, 30.0, 0))
    spec = SyNSpec(levels=2, iterations=30, radius=2, step=0.5)
    no_mask = greedy_syn_register(moving, fixed, spec=spec)
    ones = greedy_syn_register(
        moving, fixed, spec=spec, mask=jnp.ones((64, 64))
    )
    assert np.allclose(
        np.asarray(no_mask.warped), np.asarray(ones.warped), atol=1e-4
    )


def test_ones_mask_equals_no_mask_demons():
    fixed = _blobs(64)
    moving = _warp(fixed, _smooth_velocity((64, 64), 8.0, 25.0, 1))
    spec = DemonsSpec(levels=2, iterations=40)
    no_mask = diffeomorphic_demons_register(moving, fixed, spec=spec)
    ones = diffeomorphic_demons_register(
        moving, fixed, spec=spec, mask=jnp.ones((64, 64))
    )
    assert np.allclose(
        np.asarray(no_mask.warped), np.asarray(ones.warped), atol=1e-4
    )


def test_partial_mask_recovers_and_diffeomorphic():
    fixed = _blobs(64)
    moving = _warp(fixed, _smooth_velocity((64, 64), 8.0, 30.0, 0))
    res = greedy_syn_register(
        moving,
        fixed,
        spec=SyNSpec(levels=3, iterations=40, radius=2, step=0.5),
        mask=_disk_mask(64, 24),
    )
    assert float(ncc(res.warped, fixed)) > float(ncc(moving, fixed))
    assert float(res.jacobian_det.min()) > 0.0


# ---------------------------------------------------------------------------
# restrict-deformation (A7) -- per-axis deformation masking
# ---------------------------------------------------------------------------


def test_restrict_ones_equals_no_restrict_syn():
    # An all-ones restrict reduces exactly to the unrestricted recipe.
    fixed = _blobs(64)
    moving = _warp(fixed, _smooth_velocity((64, 64), 8.0, 30.0, 0))
    spec = SyNSpec(levels=2, iterations=30, radius=2, step=0.5)
    base = greedy_syn_register(moving, fixed, spec=spec)
    ones = greedy_syn_register(moving, fixed, spec=spec, restrict=(1.0, 1.0))
    assert np.allclose(
        np.asarray(base.warped), np.asarray(ones.warped), atol=1e-5
    )


def test_restrict_suppresses_axis_syn():
    # restrict=(0, 1) zeroes the force's axis-0 component, so the recovered
    # deformation has (essentially) no axis-0 displacement -- where the free run
    # uses it -- and stays diffeomorphic.
    fixed = _blobs(64)
    moving = _warp(fixed, _smooth_velocity((64, 64), 8.0, 30.0, 0))
    spec = SyNSpec(levels=3, iterations=40, radius=2, step=0.5)
    free = greedy_syn_register(moving, fixed, spec=spec)
    restricted = greedy_syn_register(
        moving, fixed, spec=spec, restrict=(0.0, 1.0)
    )
    assert float(jnp.abs(restricted.displacement[..., 0]).max()) < 1e-3
    assert float(jnp.abs(free.displacement[..., 0]).max()) > 0.5
    assert float(restricted.jacobian_det.min()) > 0.0


def test_restrict_suppresses_axis_demons():
    fixed = _blobs(64)
    moving = _warp(fixed, _smooth_velocity((64, 64), 8.0, 25.0, 1))
    spec = DemonsSpec(levels=2, iterations=40)
    restricted = diffeomorphic_demons_register(
        moving, fixed, spec=spec, restrict=(0.0, 1.0)
    )
    assert float(jnp.abs(restricted.displacement[..., 0]).max()) < 1e-3
    assert float(restricted.jacobian_det.min()) > 0.0


def test_restrict_length_validation():
    fixed, moving = _blobs(32), _blobs(32)
    with pytest.raises(ValueError):
        greedy_syn_register(
            moving, fixed, spec=SyNSpec(levels=2, iterations=5), restrict=(1.0,)
        )
