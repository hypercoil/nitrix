# -*- coding: utf-8 -*-
"""Lazy-velocity contract for the diffeomorphic recipes.

The stationary velocity is only recovered (via ``geometry.field_log`` in group
mode) when ``spec.compute_velocity`` is set -- it feeds none of
``warped``/``displacement``/``jacobian_det``, so skipping it (the default) must
leave those byte-identical while saving the field_log compile + runtime.  Gate:
default leaves the velocity ``None``; ``compute_velocity=True`` restores a finite
velocity; the deformation outputs are bit-for-bit unchanged either way -- for
Demons and SyN, in both representations.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from dataclasses import replace  # noqa: E402

from nitrix.geometry import (  # noqa: E402
    identity_grid,
    integrate_velocity_field,
    spatial_transform,
)
from nitrix.register import (  # noqa: E402
    DemonsSpec,
    SyNSpec,
    diffeomorphic_demons_register,
    greedy_syn_register,
)
from nitrix.smoothing import gaussian  # noqa: E402


def _blobs(n=64):
    yy, xx = np.mgrid[0:n, 0:n].astype('float64')

    def blob(cy, cx, s, a):
        return a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))

    return jnp.asarray(
        blob(0.31 * n, 0.38 * n, 0.13 * n, 1.0)
        + blob(0.62 * n, 0.69 * n, 0.16 * n, 0.7)
        + blob(0.72 * n, 0.31 * n, 0.11 * n, 0.6)
    )


def _warp(image, sigma, scale, seed):
    rng = np.random.RandomState(seed)
    v = rng.standard_normal(image.shape + (2,))
    v = np.moveaxis(v, -1, 0)
    v = np.asarray(gaussian(jnp.asarray(v), sigma=sigma, spatial_rank=2))
    v = jnp.asarray(scale * np.moveaxis(v, 0, -1))
    grid = identity_grid(image.shape, dtype=image.dtype)
    s = integrate_velocity_field(v)
    return spatial_transform(image[..., None], grid + s, mode='nearest')[
        ..., 0
    ]


@pytest.mark.parametrize('representation', ['group', 'algebra'])
def test_demons_lazy_velocity_contract(representation):
    fixed = _blobs(64)
    moving = _warp(fixed, 8.0, 30.0, 0)
    spec = DemonsSpec(levels=2, iterations=25, representation=representation)
    off = diffeomorphic_demons_register(moving, fixed, spec=spec)
    on = diffeomorphic_demons_register(
        moving, fixed, spec=replace(spec, compute_velocity=True)
    )
    # default skips the velocity ...
    assert off.velocity is None
    # ... while the deformation outputs are bit-for-bit identical.
    assert np.array_equal(np.asarray(off.warped), np.asarray(on.warped))
    assert np.array_equal(
        np.asarray(off.displacement), np.asarray(on.displacement)
    )
    assert np.array_equal(
        np.asarray(off.jacobian_det), np.asarray(on.jacobian_det)
    )
    # requested velocity is a finite field of the right shape.
    assert on.velocity is not None
    assert on.velocity.shape == (64, 64, 2)
    assert bool(jnp.all(jnp.isfinite(on.velocity)))


@pytest.mark.parametrize('representation', ['group', 'algebra'])
def test_syn_lazy_velocity_contract(representation):
    fixed = _blobs(64)
    moving = _warp(fixed, 8.0, 30.0, 1)
    spec = SyNSpec(
        levels=2, iterations=25, step=0.5, representation=representation
    )
    off = greedy_syn_register(moving, fixed, spec=spec)
    on = greedy_syn_register(
        moving, fixed, spec=replace(spec, compute_velocity=True)
    )
    assert off.forward_velocity is None and off.inverse_velocity is None
    assert np.array_equal(np.asarray(off.warped), np.asarray(on.warped))
    assert np.array_equal(
        np.asarray(off.displacement), np.asarray(on.displacement)
    )
    assert np.array_equal(
        np.asarray(off.jacobian_det), np.asarray(on.jacobian_det)
    )
    assert on.forward_velocity is not None
    assert on.inverse_velocity is not None
    assert bool(jnp.all(jnp.isfinite(on.forward_velocity)))
    assert bool(jnp.all(jnp.isfinite(on.inverse_velocity)))
