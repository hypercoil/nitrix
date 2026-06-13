# -*- coding: utf-8 -*-
"""R7: greedy symmetric diffeomorphic registration (SyN, LNCC-driven).

Warp a structured image by a known smooth diffeomorphism, register it back
with ``greedy_syn_register``, and assert the recovered warp reproduces the
fixed image (high NCC) and is a diffeomorphism (no non-positive Jacobian
determinant).  Plus the LNCC payoff -- recovery survives a smooth
multiplicative bias field (which an SSD force does not) -- and the
symmetry / validation.
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
from nitrix.register import SyNSpec, greedy_syn_register  # noqa: E402
from nitrix.register._svf import (  # noqa: E402
    _normalise_step,
    resolve_smoothing,
)
from nitrix.smoothing import gaussian  # noqa: E402


def _blobs_2d(n=64):
    yy, xx = np.mgrid[0:n, 0:n].astype('float64')

    def blob(cy, cx, s, a):
        return a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))

    return jnp.asarray(
        blob(0.31 * n, 0.38 * n, 0.13 * n, 1.0)
        + blob(0.62 * n, 0.69 * n, 0.16 * n, 0.7)
        + blob(0.72 * n, 0.31 * n, 0.11 * n, 0.6)
        + blob(0.44 * n, 0.75 * n, 0.14 * n, 0.5)
    )


def _smooth_velocity(shape, ndim, sigma, scale, seed):
    rng = np.random.RandomState(seed)
    v = rng.standard_normal(shape + (ndim,))
    v = np.moveaxis(v, -1, 0)
    v = np.asarray(
        gaussian(jnp.asarray(v), sigma=sigma, spatial_rank=len(shape))
    )
    v = np.moveaxis(v, 0, -1)
    return jnp.asarray(scale * v)


def _warp_by_velocity(image, v):
    grid = identity_grid(image.shape, dtype=image.dtype)
    s = integrate_velocity_field(v)
    return spatial_transform(image[..., None], grid + s, mode='nearest')[..., 0]


def test_syn_2d_recovery_and_diffeomorphism():
    fixed = _blobs_2d(64)
    v_true = _smooth_velocity((64, 64), 2, 8.0, 45.0, 0)
    moving = _warp_by_velocity(fixed, v_true)
    init = float(ncc(moving, fixed))
    assert init < 0.99

    res = greedy_syn_register(
        moving, fixed, spec=SyNSpec(levels=3, iterations=60, step=0.5)
    )
    assert float(ncc(res.warped, fixed)) > 0.99
    assert float(ncc(res.warped, fixed)) > init + 0.02
    assert float(res.jacobian_det.min()) > 0.0  # no folding
    # Symmetric: both velocity fields are driven (neither stays ~zero).
    assert float(jnp.abs(res.forward_velocity).max()) > 1e-2
    assert float(jnp.abs(res.inverse_velocity).max()) > 1e-2
    assert res.displacement.shape == (64, 64, 2)


def test_syn_smoothing_default_off_byte_identical():
    # A scalar 0 sigma is a no-op smooth and must reproduce the default
    # (``smoothing_sigma=None``) path bit-for-bit.
    fixed = _blobs_2d(48)
    v_true = _smooth_velocity((48, 48), 2, 8.0, 30.0, 7)
    moving = _warp_by_velocity(fixed, v_true)
    spec = SyNSpec(levels=2, iterations=30, step=0.5)
    res_default = greedy_syn_register(moving, fixed, spec=spec)
    res_zero = greedy_syn_register(
        moving, fixed, spec=spec, smoothing_sigma=0.0
    )
    assert np.array_equal(
        np.asarray(res_default.forward_velocity),
        np.asarray(res_zero.forward_velocity),
    )


def test_syn_smoothing_schedule_recovers_and_diffeomorphic():
    # A coarse-to-fine smoothing schedule (decoupled from the shrink, ANTs
    # ``-s 2x1x0``) still recovers the planted warp and stays diffeomorphic.
    assert resolve_smoothing((2.0, 1.0, 0.0), 3) == (0.0, 1.0, 2.0)
    fixed = _blobs_2d(64)
    v_true = _smooth_velocity((64, 64), 2, 8.0, 55.0, 6)
    moving = _warp_by_velocity(fixed, v_true)
    init = float(ncc(moving, fixed))
    assert init < 0.99  # a genuine misalignment

    res = greedy_syn_register(
        moving,
        fixed,
        spec=SyNSpec(levels=3, iterations=60, step=0.5),
        smoothing_sigma=(2.0, 1.0, 0.0),
    )
    assert float(ncc(res.warped, fixed)) > 0.99
    assert float(ncc(res.warped, fixed)) > init + 0.02
    assert float(res.jacobian_det.min()) > 0.0


def test_syn_bias_robustness():
    # LNCC is robust to smooth intensity inhomogeneity: SyN recovers a warp
    # even when the moving image carries a smooth multiplicative bias field
    # (an SSD force would be pulled off by the bias).
    fixed = _blobs_2d(64)
    v_true = _smooth_velocity((64, 64), 2, 8.0, 45.0, 0)
    moving = _warp_by_velocity(fixed, v_true)
    bias = jnp.exp(0.6 * _smooth_velocity((64, 64), 2, 12.0, 1.0, 3)[..., 0])
    moving_biased = moving * bias

    res = greedy_syn_register(
        moving_biased, fixed, spec=SyNSpec(levels=3, iterations=60, step=0.5)
    )
    assert float(ncc(res.warped, fixed)) > 0.99
    assert float(res.jacobian_det.min()) > 0.0


def test_syn_3d_recovery():
    n = 22
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype('float64')

    def blob(c, s, a):
        return a * np.exp(
            -((xx - c[2]) ** 2 + (yy - c[1]) ** 2 + (zz - c[0]) ** 2)
            / (2 * s * s)
        )

    fixed = jnp.asarray(
        blob((0.4 * n, 0.4 * n, 0.5 * n), 0.18 * n, 1.0)
        + blob((0.6 * n, 0.62 * n, 0.4 * n), 0.22 * n, 0.7)
        + blob((0.5 * n, 0.3 * n, 0.63 * n), 0.14 * n, 0.6)
    )
    v_true = _smooth_velocity((n, n, n), 3, 4.0, 8.0, 1)
    moving = _warp_by_velocity(fixed, v_true)

    res = greedy_syn_register(
        moving, fixed, spec=SyNSpec(levels=2, iterations=40, radius=2, step=0.5)
    )
    assert float(ncc(res.warped, fixed)) > float(ncc(moving, fixed))
    assert float(ncc(res.warped, fixed)) > 0.96
    assert float(res.jacobian_det.min()) > 0.0


def test_syn_identity():
    fixed = _blobs_2d(48)
    res = greedy_syn_register(
        fixed, fixed, spec=SyNSpec(levels=2, iterations=20, step=0.5)
    )
    # The symmetric forward / inverse velocities receive identical updates
    # at a perfect match, so they cancel in the net deformation: warped ==
    # fixed, displacement ~ 0, no folding (the individual velocities may
    # carry a symmetric drift in low-contrast windows -- the composed map
    # is what is diffeomorphism-guaranteed).
    assert float(ncc(res.warped, fixed)) > 0.9999
    assert float(jnp.abs(res.displacement).max()) < 1e-3
    assert np.allclose(np.asarray(res.jacobian_det), 1.0, atol=1e-3)


def test_syn_validation():
    with pytest.raises(ValueError):
        greedy_syn_register(_blobs_2d(32), _blobs_2d(48))
    with pytest.raises(ValueError):
        greedy_syn_register(jnp.zeros((4, 4, 4, 4)), jnp.zeros((4, 4, 4, 4)))


def test_normalise_step_robust_to_outlier():
    # B4: the trust-region clamp caps at a high *percentile* of the per-voxel
    # displacement, so a single hot/edge voxel cannot throttle the whole field.
    n = 48
    step = 1.0
    # Bulk below `step`: the clamp is a no-op (scale == 1) -- the real signal is
    # preserved.  A global-max clamp would have scaled it by step/100 = 0.01.
    u = jnp.full((n, n, 2), 0.3 / np.sqrt(2.0))  # per-voxel norm 0.3 everywhere
    u = u.at[0, 0].set(jnp.asarray([100.0, 0.0]))  # one outlier voxel
    out = _normalise_step(u, step)
    bulk = float(jnp.linalg.norm(out[n // 2, n // 2]))
    assert np.isclose(bulk, 0.3, rtol=1e-6)  # preserved, not starved to ~0.003

    # Bulk genuinely above `step`: the clamp still bounds it -- but at the robust
    # cap (p99 = 2.0), so the bulk lands at ~step, not at step*(2/100).
    u2 = jnp.full((n, n, 2), 2.0 / np.sqrt(2.0))
    u2 = u2.at[0, 0].set(jnp.asarray([100.0, 0.0]))
    bulk2 = float(jnp.linalg.norm(_normalise_step(u2, step)[n // 2, n // 2]))
    assert np.isclose(bulk2, step, rtol=1e-3)
