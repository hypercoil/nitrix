# -*- coding: utf-8 -*-
"""E1: recovery parametrized over dtype -- float32 is the production path.

Every other registration test enables x64 and builds float64 inputs, so a
float32 / GPU conditioning regression would pass CI silently.  These gates run
the IC-vs-forward parity and the stiff-affine + SVF recovery at **both**
float32 and float64 (dtype-appropriate tolerances), and assert the result dtype
is preserved -- the float32 conditioning guard.  The hot-path zero-guards
(``_trust_scale``, ``_normalise_rms``, ``_normalise_step``) are derived from
``jnp.finfo(dtype).eps`` so they are meaningful at float32 precision, not a
fixed 1e-12 that sits below it.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nitrix.geometry import (  # noqa: E402
    affine_exp,
    affine_grid,
    identity_grid,
    integrate_velocity_field,
    spatial_transform,
)
from nitrix.metrics import ncc  # noqa: E402
from nitrix.register import (  # noqa: E402
    RegistrationSpec,
    SyNSpec,
    affine_register,
    greedy_syn_register,
    rigid_register,
)
from nitrix.smoothing import gaussian  # noqa: E402

# Per-dtype tolerances: float32's ~1.2e-7 precision loosens the IC-vs-forward
# parity and the recovery floor relative to float64's ~2.2e-16.
_DTYPES = [
    pytest.param(jnp.float64, 3e-2, 0.99, id='float64'),
    pytest.param(jnp.float32, 6e-2, 0.985, id='float32'),
]


def _blobs(n, seed, dtype):
    rng = np.random.RandomState(seed)
    g = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    img = np.zeros((n, n))
    for _ in range(6):
        c = rng.uniform(0.25, 0.75, 2) * n
        s = rng.uniform(0.08, 0.14) * n
        img += rng.uniform(0.4, 1.0) * np.exp(
            -sum((gi - ci) ** 2 for gi, ci in zip(g, c)) / (2 * s * s)
        )
    return jnp.asarray((img / img.max()).astype(dtype))


def _warp_affine(img, params):
    c = (jnp.asarray(img.shape, dtype=img.dtype) - 1.0) / 2.0
    grid = affine_grid(
        affine_exp(jnp.asarray(params, dtype=img.dtype), ndim=2),
        img.shape,
        center=c,
    )
    return spatial_transform(img[..., None], grid, mode='nearest')[..., 0]


def _smooth_velocity(shape, sigma, scale, seed, dtype):
    rng = np.random.RandomState(seed)
    v = rng.standard_normal(shape + (2,))
    v = np.moveaxis(v, -1, 0)
    v = np.asarray(gaussian(jnp.asarray(v), sigma=sigma, spatial_rank=2))
    return jnp.asarray((scale * np.moveaxis(v, 0, -1)).astype(dtype))


@pytest.mark.parametrize('dtype, atol, ncc_min', _DTYPES)
def test_ic_matches_forward_dtype(dtype, atol, ncc_min):
    fixed = _blobs(64, 0, dtype)
    moving = _warp_affine(fixed, [0.04, -0.08, 0.06, -0.05, 3.0, -2.0])
    spec = RegistrationSpec(levels=3, iterations=25)
    ic = affine_register(
        moving, fixed, spec=spec, method='inverse_compositional'
    )
    fwd = affine_register(moving, fixed, spec=spec, method='forward')
    assert ic.warped.dtype == dtype  # dtype preserved (no silent upcast)
    assert float(ncc(ic.warped, fixed)) > ncc_min
    assert float(ncc(fwd.warped, fixed)) > ncc_min
    assert np.allclose(
        np.asarray(ic.warped), np.asarray(fwd.warped), atol=atol
    )


@pytest.mark.parametrize('dtype, atol, ncc_min', _DTYPES)
def test_hard_affine_ic_recovers_dtype(dtype, atol, ncc_min):
    # The stiff affine (anisotropic scale + shear + large translation) is the
    # conditioning-sensitive case: the constant-template Hessian spans orders of
    # magnitude.  The float32 instance is the conditioning guard.
    fixed = _blobs(64, 1, dtype)
    moving = _warp_affine(fixed, [0.18, -0.22, 0.20, -0.16, 6.0, -5.0])
    ic = affine_register(
        moving,
        fixed,
        spec=RegistrationSpec(levels=3, iterations=40),
        method='inverse_compositional',
    )
    assert ic.warped.dtype == dtype
    assert float(ncc(ic.warped, fixed)) > ncc_min


@pytest.mark.parametrize('dtype, atol, ncc_min', _DTYPES)
def test_rigid_recovers_dtype(dtype, atol, ncc_min):
    fixed = _blobs(64, 2, dtype)
    c = (jnp.asarray(fixed.shape, dtype=fixed.dtype) - 1.0) / 2.0
    from nitrix.geometry import rigid_exp

    grid = affine_grid(
        rigid_exp(jnp.asarray([0.12, 3.0, -2.0], dtype=dtype), ndim=2),
        fixed.shape,
        center=c,
    )
    moving = spatial_transform(fixed[..., None], grid, mode='nearest')[..., 0]
    res = rigid_register(
        moving, fixed, spec=RegistrationSpec(levels=3, iterations=30)
    )
    assert res.warped.dtype == dtype
    assert float(ncc(res.warped, fixed)) > ncc_min


@pytest.mark.parametrize('dtype, atol, ncc_min', _DTYPES)
def test_syn_recovers_dtype(dtype, atol, ncc_min):
    # exercises the SVF hot-path eps guards (_normalise_step / _normalise_rms).
    fixed = _blobs(64, 3, dtype)
    g = identity_grid(fixed.shape, dtype=fixed.dtype)
    s = integrate_velocity_field(
        _smooth_velocity((64, 64), 8.0, 30.0, 4, dtype)
    )
    moving = spatial_transform(fixed[..., None], g + s, mode='nearest')[..., 0]
    res = greedy_syn_register(
        moving,
        fixed,
        spec=SyNSpec(levels=2, iterations=40, radius=2, step=0.5),
    )
    assert res.warped.dtype == dtype
    assert float(ncc(res.warped, fixed)) > ncc_min
    assert float(res.jacobian_det.min()) > 0.0  # diffeomorphic at both dtypes
