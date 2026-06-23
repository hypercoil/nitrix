# -*- coding: utf-8 -*-
"""Forward least-squares early-exit (Lever A): opt-in, default-unchanged.

``mode='early_exit'`` (windowed cost-slope, B2) early-exits the *forward*
Gauss-Newton / Levenberg-Marquardt path too, not just the inverse-compositional
path -- via the optimiser ``early_stop`` (a ``while_loop``).  It stays
**opt-in**: ``mode='fixed'`` (the default) keeps the fixed ``lax.scan``
(reproducible, vmap-batchable, reverse-differentiable).  The scalar/BFGS path
(MI / correlation-ratio) still rejects it (monolithic optimiser).  Gates: forward
SSD + early_exit recovers like the fixed scan; the default forward is unchanged;
MI + early_exit raises.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.ndimage import affine_transform  # noqa: E402

from nitrix.metrics import ncc  # noqa: E402
from nitrix.register import (  # noqa: E402
    MI,
    SSD,
    RegistrationSpec,
    rigid_register,
)


def _rigid_pair(n=48):
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype('float64')

    def blob(cz, cy, cx, s, a):
        return a * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) / (2 * s * s)
        )

    fixed = (
        blob(0.4 * n, 0.42 * n, 0.5 * n, 0.16 * n, 1.0)
        + blob(0.6 * n, 0.62 * n, 0.4 * n, 0.2 * n, 0.7)
    ).astype('float32')
    c = (np.array(fixed.shape) - 1) / 2.0
    th = np.deg2rad(7.0)
    R = np.array(
        [[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]]
    )
    t = np.array([3.0, -2.0, 2.0])
    moving = affine_transform(
        fixed, R, offset=c - R @ c + t, order=1, mode='nearest'
    ).astype('float32')
    return jnp.asarray(moving), jnp.asarray(fixed)


def test_forward_ssd_early_exit_recovers_like_fixed():
    moving, fixed = _rigid_pair()
    base = RegistrationSpec(levels=3, iterations=(120, 80, 40), metric=SSD())
    fixed_res = rigid_register(
        moving,
        fixed,
        spec=base.__class__(**{**base.__dict__, 'mode': 'fixed'}),
        method='forward',
    )
    ee_res = rigid_register(
        moving,
        fixed,
        spec=base.__class__(**{**base.__dict__, 'mode': 'early_exit'}),
        method='forward',
    )
    # both recover; early-exit matches the fixed scan to tolerance (it stops once
    # the cost has plateaued, so the final transform is ~the converged one).
    assert float(ncc(fixed_res.warped, fixed)) > 0.99
    assert float(ncc(ee_res.warped, fixed)) > 0.99
    assert (
        abs(
            float(ncc(ee_res.warped, fixed))
            - float(ncc(fixed_res.warped, fixed))
        )
        < 5e-3
    )
    # cost_history keeps its (padded) shape either way.
    assert ee_res.cost_history.shape == fixed_res.cost_history.shape


def test_forward_default_is_fixed_scan():
    # mode='fixed' (the default) forward is deterministic / reproducible.
    moving, fixed = _rigid_pair()
    spec = RegistrationSpec(
        levels=2, iterations=(80, 40), metric=SSD(), mode='fixed'
    )
    a = rigid_register(moving, fixed, spec=spec, method='forward')
    b = rigid_register(moving, fixed, spec=spec, method='forward')
    assert np.array_equal(np.asarray(a.warped), np.asarray(b.warped))


def test_forward_mi_convergence_rejected():
    moving, fixed = _rigid_pair()
    spec = RegistrationSpec(
        levels=2,
        iterations=(40, 20),
        metric=MI(bins=32),
        mode='early_exit',
    )
    with pytest.raises(ValueError, match='scalar/BFGS'):
        rigid_register(moving, fixed, spec=spec, method='forward')
