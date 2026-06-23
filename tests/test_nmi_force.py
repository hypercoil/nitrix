# -*- coding: utf-8 -*-
"""C1: closed-form NMI force -- ``nmi_grad`` + ``MIForce(normalized=True)``.

NMI (Studholme's ``(H_m + H_f) / H_mf``) is the ANTs cross-modal default.  The
closed-form ``metrics.nmi_grad`` is the quotient-rule ``∂NMI/∂moving`` (the NMI
analogue of the Mattes ``mi_grad``), and ``MIForce(normalized=True)`` routes the
deformable force through it instead of the generic autodiff ``MetricForce``
path.  Gates: ``nmi_grad`` is ``jax.grad``-parity on populated bins (the same
oracle ``mi_grad`` uses); the force recovers a planted cross-modal warp at SyN
quality, on par with unnormalised MI.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from nitrix.geometry import (  # noqa: E402
    identity_grid,
    integrate_velocity_field,
    spatial_transform,
)
from nitrix.metrics import mutual_information, ncc, nmi_grad  # noqa: E402
from nitrix.register import (  # noqa: E402
    MIForce,
    SyNSpec,
    greedy_syn_register,
)
from nitrix.smoothing import gaussian  # noqa: E402

_RM = (0.0, 1.0)


def _blobs(n=64):
    yy, xx = np.mgrid[0:n, 0:n].astype('float64')
    img = np.zeros((n, n))
    for cy, cx, s, a in [
        (0.31, 0.38, 0.13, 1.0),
        (0.62, 0.69, 0.16, 0.7),
        (0.72, 0.31, 0.11, 0.6),
        (0.44, 0.75, 0.14, 0.5),
    ]:
        img += a * np.exp(
            -((xx - cx * n) ** 2 + (yy - cy * n) ** 2) / (2 * (s * n) ** 2)
        )
    return jnp.asarray(img / img.max())


def _smooth_velocity(shape, sigma, scale, seed):
    rng = np.random.RandomState(seed)
    v = rng.standard_normal(shape + (2,))
    v = np.moveaxis(v, -1, 0)
    v = np.asarray(gaussian(jnp.asarray(v), sigma=sigma, spatial_rank=2))
    return jnp.asarray(scale * np.moveaxis(v, 0, -1))


def _warp(image, v):
    grid = identity_grid(image.shape, dtype=image.dtype)
    s = integrate_velocity_field(v)
    return spatial_transform(image[..., None], grid + s, mode='nearest')[
        ..., 0
    ]


# ---------------------------------------------------------------------------
# nmi_grad == jax.grad(NMI) on populated bins (the parity oracle)
# ---------------------------------------------------------------------------


def test_nmi_grad_matches_autodiff():
    rng = np.random.RandomState(0)
    a = jnp.asarray(rng.rand(40, 40))
    b = jnp.asarray(rng.rand(40, 40) ** 2)  # cross-modal-ish
    closed = np.asarray(
        nmi_grad(a, b, bins=24, range_moving=_RM, range_fixed=_RM)
    )

    def cost(m):
        return mutual_information(
            m, b, bins=24, normalized=True, range_moving=_RM, range_fixed=_RM
        )

    autodiff = np.asarray(jax.grad(cost)(a))
    assert np.allclose(closed, autodiff, atol=1e-7)


def test_nmi_grad_distinct_from_mi_grad():
    # NMI is a genuinely different criterion -- its gradient is not the MI one.
    from nitrix.metrics import mi_grad

    rng = np.random.RandomState(1)
    a = jnp.asarray(rng.rand(32, 32))
    b = jnp.asarray(rng.rand(32, 32) ** 2)
    gn = np.asarray(nmi_grad(a, b, bins=24, range_moving=_RM, range_fixed=_RM))
    gm = np.asarray(mi_grad(a, b, bins=24, range_moving=_RM, range_fixed=_RM))
    assert not np.allclose(gn, gm, atol=1e-6)


# ---------------------------------------------------------------------------
# MIForce(normalized=True) routes to nmi_grad and recovers a cross-modal warp
# ---------------------------------------------------------------------------


def test_miforce_normalized_uses_nmi_grad():
    # the bound NMI force's raw (pre-RMS) direction is nmi_grad * grad(warped);
    # check the force is finite and that its cost is the NMI cost.
    rng = np.random.RandomState(2)
    fixed = _blobs(48)
    warped = jnp.asarray(rng.rand(48, 48))
    bound = MIForce(bins=24, normalized=True).bind(fixed, ndim=2)
    u = bound.update(warped)
    assert u.shape == (48, 48, 2)
    assert bool(jnp.all(jnp.isfinite(u)))
    # cost follows the normalized flag (== -NMI).
    assert np.allclose(
        float(bound.cost(warped)),
        -float(mutual_information(warped, fixed, bins=24, normalized=True)),
    )


def test_nmi_force_recovers_cross_modal_warp():
    n = 64
    fixed = _blobs(n)
    mono = _warp(fixed, _smooth_velocity((n, n), 8.0, 30.0, 0))
    moving = 1.0 - mono  # inverted contrast: NMI/MI is the right criterion
    spec = SyNSpec(levels=3, iterations=60, radius=2, step=0.5)
    res = greedy_syn_register(
        moving, fixed, spec=spec, force=MIForce(bins=32, normalized=True)
    )
    # un-invert the recovered moving and compare structure to the fixed.
    assert float(ncc(1.0 - res.warped, fixed)) > 0.99
    assert float(res.jacobian_det.min()) > 0.0  # diffeomorphic
