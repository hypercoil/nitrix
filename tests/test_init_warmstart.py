# -*- coding: utf-8 -*-
"""V2a: warm-start / multi-stage init for the diffeomorphic recipes.

``init_affine`` / ``init_displacement`` pre-warp the moving onto the fixed grid
and the recipe registers the residual, composing the result -- one mechanism
serving three use cases: SynthMorph-style seed-then-refine (``init_displacement``
from a network), affine-init multi-stage (``init_affine`` from a prior
``rigid_register`` / ``affine_register``), and registration across *different*
grids (the pre-warp resamples onto the common grid, retiring the
matching-shape constraint).
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nitrix.geometry import (  # noqa: E402
    affine_grid,
    identity_grid,
    integrate_velocity_field,
    rigid_exp,
    spatial_transform,
)
from nitrix.metrics import ncc  # noqa: E402
from nitrix.register import (  # noqa: E402
    RegistrationSpec,
    SyNSpec,
    diffeomorphic_demons_register,
    greedy_syn_register,
    rigid_register,
)
from nitrix.smoothing import gaussian  # noqa: E402


def _blobs(n=64, seed=0):
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


def test_syn_warmstart_refines():
    # Seed-then-refine: a first partial registration's displacement seeds a
    # second run, which refines past it (warm-start does not regress).
    fixed = _blobs(64)
    moving = _warp(fixed, _smooth_velocity((64, 64), 8.0, 45.0, 0))
    short = SyNSpec(levels=2, iterations=15, radius=2, step=0.5)
    res1 = greedy_syn_register(moving, fixed, spec=short)
    res2 = greedy_syn_register(
        moving, fixed, spec=short, init_displacement=res1.displacement
    )
    assert float(ncc(res2.warped, fixed)) >= float(ncc(res1.warped, fixed))
    assert float(ncc(res2.warped, fixed)) > 0.99
    assert float(res2.jacobian_det.min()) > 0.0
    assert res2.displacement.shape == (64, 64, 2)


def test_syn_init_affine_multistage():
    # rigid -> SyN: a large rigid offset + a deformation; rigid_register
    # supplies init_affine, SyN cleans up the residual deformation.
    fixed = _blobs(64)
    deformed = _warp(fixed, _smooth_velocity((64, 64), 8.0, 12.0, 1))
    rigid = rigid_exp(jnp.asarray([0.18, 5.0, -4.0]), ndim=2)
    moving = spatial_transform(
        deformed[..., None], affine_grid(rigid, (64, 64)), mode='nearest'
    )[..., 0]
    rres = rigid_register(
        moving, fixed, spec=RegistrationSpec(levels=3, iterations=20)
    )
    res = greedy_syn_register(
        moving,
        fixed,
        spec=SyNSpec(levels=3, iterations=40, radius=2, step=0.5),
        init_affine=rres.matrix,
    )
    assert float(ncc(res.warped, fixed)) > 0.97
    assert float(res.jacobian_det.min()) > 0.0


def test_demons_warmstart():
    fixed = _blobs(64)
    moving = _warp(fixed, _smooth_velocity((64, 64), 8.0, 30.0, 2))
    res1 = diffeomorphic_demons_register(moving, fixed)
    res2 = diffeomorphic_demons_register(
        moving, fixed, init_displacement=res1.displacement
    )
    assert float(ncc(res2.warped, fixed)) >= float(ncc(res1.warped, fixed))
    assert float(res2.jacobian_det.min()) > 0.0


def test_init_different_grid():
    # moving on a coarser (48x48) grid than fixed (64x64); init_displacement
    # resamples it onto the fixed grid (a proportional scale), then SyN refines.
    fixed = _blobs(64)
    moving = spatial_transform(
        fixed[..., None],
        identity_grid((48, 48), dtype=fixed.dtype) * (64.0 / 48.0),
        mode='nearest',
    )[..., 0]
    grid64 = identity_grid((64, 64), dtype=fixed.dtype)
    init_disp = grid64 * (48.0 / 64.0) - grid64
    res = greedy_syn_register(
        moving,
        fixed,
        spec=SyNSpec(levels=2, iterations=30, radius=2, step=0.5),
        init_displacement=init_disp,
    )
    assert res.displacement.shape == (64, 64, 2)
    assert res.warped.shape == (64, 64)
    assert float(res.jacobian_det.min()) > 0.0
    assert float(ncc(res.warped, fixed)) > 0.95


def test_init_mutual_exclusion():
    fixed = _blobs(32)
    moving = _blobs(32, seed=1)
    with pytest.raises(ValueError):
        greedy_syn_register(
            moving,
            fixed,
            spec=SyNSpec(levels=2, iterations=5),
            init_affine=jnp.eye(3),
            init_displacement=jnp.zeros((32, 32, 2)),
        )


def test_no_init_shape_mismatch_raises():
    fixed = _blobs(64)
    moving = _blobs(48)
    with pytest.raises(ValueError):
        greedy_syn_register(moving, fixed, spec=SyNSpec(levels=2, iterations=5))
