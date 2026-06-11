# -*- coding: utf-8 -*-
"""V4d (lever C): the closed-form forward warp Jacobian.

``_warp_jacobian`` factors the residual Jacobian as ``∂warp/∂grid · ∂grid/∂θ``
so the expensive gather runs ``ndim`` times (the interpolation derivative)
instead of ``jax.jacfwd``'s ``P`` times.  It is **exact** -- the parity oracle
is ``jax.jacfwd`` of the SSD residual -- so the forward path is byte-unchanged,
just faster.  These tests assert that parity across model x space.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from nitrix.register._core import (  # noqa: E402
    RegistrationSpec,
    _warp,
    _warp_jacobian,
)
from nitrix.register._model import Affine, Rigid  # noqa: E402
from nitrix.register._space import IndexSpace, WorldSpace  # noqa: E402


def _blobs(n, seed=0):
    grids = np.meshgrid(*([np.arange(n)] * 2), indexing='ij')
    rng = np.random.RandomState(seed)
    img = np.zeros((n, n), dtype='float64')
    for _ in range(5):
        c = rng.uniform(0.3, 0.7, 2) * n
        s = rng.uniform(0.1, 0.16) * n
        img += rng.uniform(0.4, 1.0) * np.exp(
            -sum((g - ci) ** 2 for g, ci in zip(grids, c)) / (2 * s * s)
        )
    return jnp.asarray(img)


def _check_parity(model, space, params, n=48):
    moving, fixed = _blobs(n, 0), _blobs(n, 1)
    shape = (n, n)
    sampler = space.sampler(
        ndim=2, full_fixed_shape=shape, full_moving_shape=shape, dtype=moving.dtype
    )
    spec = RegistrationSpec()

    def residual(p):
        warped = _warp(
            sampler,
            moving,
            model.exp(p, ndim=2),
            fixed_shape=shape,
            moving_shape=shape,
            spec=spec,
        )
        return (warped - fixed).ravel()

    jac_fn = _warp_jacobian(
        sampler,
        moving,
        model=model,
        ndim=2,
        fixed_shape=shape,
        moving_shape=shape,
        spec=spec,
    )
    j_closed = np.asarray(jac_fn(params))
    j_jacfwd = np.asarray(jax.jacfwd(residual)(params))
    assert j_closed.shape == j_jacfwd.shape
    assert np.allclose(j_closed, j_jacfwd, atol=1e-9)


def test_parity_rigid_index():
    _check_parity(Rigid(), IndexSpace(), jnp.asarray([0.1, 2.0, -1.5]))


def test_parity_affine_index():
    _check_parity(
        Affine(),
        IndexSpace(),
        jnp.asarray([0.05, -0.08, 0.06, -0.03, 2.0, -1.5]),
    )


def test_parity_rigid_world():
    _check_parity(Rigid(), WorldSpace(), jnp.asarray([0.08, 1.5, -2.0]))


def test_parity_affine_world():
    _check_parity(
        Affine(),
        WorldSpace(),
        jnp.asarray([0.04, -0.05, 0.03, -0.06, 1.0, -1.5]),
    )
