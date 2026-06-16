# -*- coding: utf-8 -*-
"""Synthetic-recovery tests for the diffeomorphic Demons recipe (R2b).

Warp a structured image by a *known smooth diffeomorphism* ``exp(v_true)``
to make ``moving``, register it back to ``fixed``, and assert the
recovered warp reproduces ``fixed`` (high NCC) **and is a diffeomorphism**
(no non-positive Jacobian determinant -- the R2 gate).  2-D and 3-D,
identity, and input validation.
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
    diffeomorphic_demons_register,
)
from nitrix.register._svf import resolve_smoothing  # noqa: E402
from nitrix.register.diffeomorphic import _relative_spacing  # noqa: E402
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
    v = np.asarray(gaussian(jnp.asarray(v), sigma=sigma, spatial_rank=len(shape)))
    v = np.moveaxis(v, 0, -1)
    return jnp.asarray(scale * v)


def _warp_by_velocity(image, v):
    grid = identity_grid(image.shape, dtype=image.dtype)
    s = integrate_velocity_field(v)
    return spatial_transform(image[..., None], grid + s, mode='nearest')[..., 0]


def test_demons_2d_recovery_and_diffeomorphism():
    fixed = _blobs_2d(64)
    v_true = _smooth_velocity((64, 64), 2, 8.0, 45.0, 0)
    moving = _warp_by_velocity(fixed, v_true)
    init = float(ncc(moving, fixed))
    assert init < 0.99  # a genuine misalignment

    res = diffeomorphic_demons_register(
        moving, fixed, spec=DemonsSpec(levels=3, iterations=60)
    )
    assert float(ncc(res.warped, fixed)) > 0.99
    assert float(ncc(res.warped, fixed)) > init + 0.02
    # diffeomorphism gate: no folding.
    assert float(res.jacobian_det.min()) > 0.0
    assert res.velocity.shape == (64, 64, 2)
    assert res.displacement.shape == (64, 64, 2)


def test_demons_3d_recovery_and_diffeomorphism():
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
    v_true = _smooth_velocity((n, n, n), 3, 4.0, 9.0, 1)
    moving = _warp_by_velocity(fixed, v_true)

    res = diffeomorphic_demons_register(
        moving, fixed, spec=DemonsSpec(levels=2, iterations=40)
    )
    assert float(ncc(res.warped, fixed)) > float(ncc(moving, fixed))
    assert float(ncc(res.warped, fixed)) > 0.97
    assert float(res.jacobian_det.min()) > 0.0


def test_demons_identity():
    fixed = _blobs_2d(48)
    res = diffeomorphic_demons_register(
        fixed, fixed, spec=DemonsSpec(levels=2, iterations=20)
    )
    assert float(jnp.abs(res.velocity).max()) < 1e-6
    assert float(ncc(res.warped, fixed)) > 0.999
    assert np.allclose(np.asarray(res.jacobian_det), 1.0, atol=1e-4)


def test_demons_shape_validation():
    with pytest.raises(ValueError):
        diffeomorphic_demons_register(_blobs_2d(32), _blobs_2d(48))
    with pytest.raises(ValueError):
        diffeomorphic_demons_register(
            jnp.zeros((4, 4, 4, 4)), jnp.zeros((4, 4, 4, 4))
        )


def test_relative_spacing_anisotropy_only():
    # Isotropic spacing collapses to None (the voxel-native path).
    assert _relative_spacing(None, 3) is None
    assert _relative_spacing(2.0, 3) is None
    # Anisotropy is normalised to unit geometric mean (absolute scale drops).
    rel = _relative_spacing((4.0, 1.0), 2)
    assert np.allclose(rel, (2.0, 0.5))
    gm = 4.0 ** (1.0 / 3.0)
    rel3 = _relative_spacing((1.0, 1.0, 4.0), 3)
    assert np.allclose(rel3, (1.0 / gm, 1.0 / gm, 4.0 / gm))
    assert np.isclose(float(np.prod(rel3)) ** (1.0 / 3.0), 1.0)


def test_resolve_smoothing_schedule():
    # None -> no extra smoothing.
    assert resolve_smoothing(None, 3) is None
    # A scalar broadcasts to every level.
    assert resolve_smoothing(2.0, 3) == (2.0, 2.0, 2.0)
    # A coarse-to-fine sequence (ANTs ``-s`` order) is reversed to the
    # finest-first pyramid indexing.
    assert resolve_smoothing((2.0, 1.0, 0.0), 3) == (0.0, 1.0, 2.0)
    # A length mismatch is rejected.
    with pytest.raises(ValueError):
        resolve_smoothing((2.0, 1.0), 3)


def test_demons_smoothing_default_off_byte_identical():
    # ``smoothing_sigma`` defaults to None; a scalar 0 sigma is a no-op
    # smooth (s <= 0 leaves each level unchanged), so it must reproduce the
    # default path bit-for-bit.
    fixed = _blobs_2d(48)
    v_true = _smooth_velocity((48, 48), 2, 8.0, 30.0, 5)
    moving = _warp_by_velocity(fixed, v_true)
    spec = DemonsSpec(levels=2, iterations=30)
    res_default = diffeomorphic_demons_register(moving, fixed, spec=spec)
    res_zero = diffeomorphic_demons_register(
        moving, fixed, spec=spec, smoothing_sigma=0.0
    )
    assert np.array_equal(
        np.asarray(res_default.velocity), np.asarray(res_zero.velocity)
    )


def test_demons_smoothing_schedule_recovers_and_diffeomorphic():
    # An explicit coarse-to-fine smoothing schedule (decoupled from the
    # shrink, ANTs ``-s 2x1x0``) still recovers the planted warp and stays
    # diffeomorphic.
    fixed = _blobs_2d(64)
    v_true = _smooth_velocity((64, 64), 2, 8.0, 55.0, 6)
    moving = _warp_by_velocity(fixed, v_true)
    init = float(ncc(moving, fixed))
    assert init < 0.99  # a genuine misalignment

    res = diffeomorphic_demons_register(
        moving,
        fixed,
        spec=DemonsSpec(levels=3, iterations=60),
        smoothing_sigma=(2.0, 1.0, 0.0),
    )
    assert float(ncc(res.warped, fixed)) > 0.98
    assert float(ncc(res.warped, fixed)) > init + 0.02
    assert float(res.jacobian_det.min()) > 0.0


def test_demons_per_level_iteration_schedule():
    # A coarse-to-fine iteration schedule (the ANTs taper) recovers and stays
    # diffeomorphic, matching the scalar-iteration behaviour it generalises.
    fixed = _blobs_2d(64)
    v_true = _smooth_velocity((64, 64), 2, 8.0, 55.0, 6)
    moving = _warp_by_velocity(fixed, v_true)
    init = float(ncc(moving, fixed))
    assert init < 0.99
    res = diffeomorphic_demons_register(
        moving, fixed, spec=DemonsSpec(levels=3, iterations=(60, 40, 20))
    )
    assert float(ncc(res.warped, fixed)) > 0.98
    assert float(res.jacobian_det.min()) > 0.0


def test_demons_anisotropic_spacing_recovers_and_differs():
    # On an anisotropic grid the spacing-aware physics still recovers a
    # planted warp and stays diffeomorphic, and it genuinely changes the
    # deformation relative to the voxel-isotropic path.
    fixed = _blobs_2d(64)
    v_true = _smooth_velocity((64, 64), 2, 8.0, 40.0, 3)
    moving = _warp_by_velocity(fixed, v_true)
    init = float(ncc(moving, fixed))

    res_iso = diffeomorphic_demons_register(
        moving, fixed, spec=DemonsSpec(levels=3, iterations=60)
    )
    res_aniso = diffeomorphic_demons_register(
        moving,
        fixed,
        spec=DemonsSpec(levels=3, iterations=60, spacing=(2.0, 1.0)),
    )

    assert float(ncc(res_aniso.warped, fixed)) > 0.98
    assert float(ncc(res_aniso.warped, fixed)) > init + 0.02
    assert float(res_aniso.jacobian_det.min()) > 0.0
    # The anisotropy correction actually changes the recovered deformation.
    assert not np.allclose(
        np.asarray(res_aniso.velocity),
        np.asarray(res_iso.velocity),
        atol=1e-3,
    )


def test_demons_finite_on_uniform_background():
    # register-demons-force-divide-by-zero: the ESM force denominator
    # ``|j|² + α²·diff²`` is ``0/0`` on a *matched uniform region* (no gradient
    # AND no intensity mismatch) -- which every real scan has (air /
    # skull-stripped zero background).  Pre-guard this returned an *all-NaN* warp
    # at a single iteration; the synthetic smooth-blob inputs hid it (they have
    # no exactly-uniform region).  The denominator guard must keep the warp (and
    # the velocity it integrates) finite.
    n = 32
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype('float64')
    c = n / 2
    bump = np.exp(
        -((xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2) / (2 * 3.0**2)
    )
    bump[bump < 1e-3] = 0.0  # an *exactly* zero (uniform) background
    fixed = jnp.asarray(bump)
    moving = jnp.asarray(np.roll(bump, 2, axis=0))  # background stays exact-0
    res = diffeomorphic_demons_register(
        moving, fixed, spec=DemonsSpec(levels=1, iterations=1)
    )
    assert bool(jnp.all(jnp.isfinite(res.warped)))
    assert bool(jnp.all(jnp.isfinite(res.velocity)))
