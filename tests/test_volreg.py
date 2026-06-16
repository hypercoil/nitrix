# -*- coding: utf-8 -*-
"""R5: batched volume registration (motion realignment).

Plant a *known* per-frame rigid motion on a base image to build a series,
realign it with ``volreg``, and assert each frame is recovered onto the
reference (high NCC, recovered transform ~ the inverse of the planted
one).  Covers the reference policies (frame index / mean / explicit), the
two-pass schedule, the physical-space (anisotropic) path, and validation.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nitrix.geometry import (  # noqa: E402
    affine_grid,
    rigid_exp,
    spatial_transform,
)
from nitrix.metrics import ncc  # noqa: E402
from nitrix.register import (  # noqa: E402
    LNCC,
    Convergence,
    RegistrationSpec,
    WorldSpace,
    volreg,
)


def _blobs_2d(n=48):
    yy, xx = np.mgrid[0:n, 0:n].astype('float64')

    def blob(cy, cx, s, a):
        return a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))

    return jnp.asarray(
        blob(0.30 * n, 0.38 * n, 0.12 * n, 1.0)
        + blob(0.62 * n, 0.69 * n, 0.15 * n, 0.7)
        + blob(0.74 * n, 0.29 * n, 0.10 * n, 0.6)
        + blob(0.46 * n, 0.80 * n, 0.13 * n, 0.5)
    )


def _warp_known(img, matrix, center=None):
    shape = img.shape
    if center is None:
        center = (jnp.asarray(shape, dtype=img.dtype) - 1.0) / 2.0
    grid = affine_grid(matrix, shape, center=center)
    return spatial_transform(img[..., None], grid, mode='constant')[..., 0]


def _series_index(base, thetas, ndim):
    frames = [
        _warp_known(base, rigid_exp(jnp.asarray(th), ndim=ndim))
        for th in thetas
    ]
    return jnp.stack(frames, axis=0)


_THETAS = [
    [0.0, 0.0, 0.0],
    [0.08, 1.5, -1.0],
    [-0.06, -1.2, 2.0],
    [0.05, 2.0, 1.2],
]


def test_volreg_2d_recovers_per_frame_motion():
    base = _blobs_2d(48)
    series = _series_index(base, _THETAS, 2)
    res = volreg(
        series, reference=0, spec=RegistrationSpec(levels=3, iterations=25)
    )
    assert res.matrices.shape == (4, 3, 3)
    assert res.params.shape == (4, 3)
    assert res.realigned.shape == (4, 48, 48)
    # Every frame is realigned onto the reference (frame 0 == base).
    for t in range(4):
        assert float(ncc(res.realigned[t], base)) > 0.97
    # The reference frame barely moves; the others recover the inverse rot.
    assert np.allclose(np.asarray(res.params[0]), 0.0, atol=1e-2)
    for t in range(1, 4):
        assert np.isclose(float(res.params[t, 0]), -_THETAS[t][0], atol=0.03)


def test_volreg_reference_policies():
    base = _blobs_2d(40)
    series = _series_index(base, _THETAS, 2)
    spec = RegistrationSpec(levels=2, iterations=15)
    # mean reference, explicit-array reference, and a frame index all run
    # and realign the series to a consistent target (low inter-frame
    # variance after realignment).
    for reference in ('mean', base, 0):
        res = volreg(series, reference=reference, spec=spec)
        var_before = float(jnp.var(series, axis=0).mean())
        var_after = float(jnp.var(res.realigned, axis=0).mean())
        assert var_after < var_before


def test_volreg_two_pass_runs_and_recovers():
    base = _blobs_2d(40)
    series = _series_index(base, _THETAS, 2)
    res = volreg(
        series,
        reference='mean',
        passes=2,
        spec=RegistrationSpec(levels=2, iterations=20),
    )
    # Two passes drive the realigned frames to a common image (the
    # final-pass mean reference).
    assert float(jnp.var(res.realigned, axis=0).mean()) < float(
        jnp.var(series, axis=0).mean()
    )


def test_volreg_3d_runs():
    n = 22
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype('float64')

    def blob(c, s, a):
        return a * np.exp(
            -((xx - c[2]) ** 2 + (yy - c[1]) ** 2 + (zz - c[0]) ** 2)
            / (2 * s * s)
        )

    base = jnp.asarray(
        blob((0.4 * n, 0.4 * n, 0.5 * n), 0.18 * n, 1.0)
        + blob((0.62 * n, 0.6 * n, 0.4 * n), 0.20 * n, 0.7)
        + blob((0.5 * n, 0.3 * n, 0.63 * n), 0.13 * n, 0.6)
    )
    thetas = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.05, -0.04, 0.03, 0.8, -0.6, 1.0],
        [-0.03, 0.05, -0.02, -1.0, 0.7, -0.5],
    ]
    series = _series_index(base, thetas, 3)
    res = volreg(
        series, reference=0, spec=RegistrationSpec(levels=2, iterations=25)
    )
    assert res.matrices.shape == (3, 4, 4)
    for t in range(3):
        assert float(ncc(res.realigned[t], base)) > 0.97


def test_volreg_worldspace_anisotropic():
    # Motion that is rigid in *world* space on an anisotropic grid.
    base = _blobs_2d(48)
    affine = np.diag([3.0, 1.0, 1.0])
    a_inv = np.linalg.inv(affine)
    c_vox = (np.asarray(base.shape, dtype='float64') - 1.0) / 2.0
    c_world = (affine @ np.concatenate([c_vox, [1.0]]))[:2]
    t_pos = np.eye(3)
    t_pos[:2, 2] = c_world
    t_neg = np.eye(3)
    t_neg[:2, 2] = -c_world

    thetas = [[0.0, 0.0, 0.0], [0.1, 1.0, -0.8], [-0.08, -1.0, 1.2]]
    frames = []
    for th in thetas:
        t_world = np.asarray(rigid_exp(jnp.asarray(th), ndim=2))
        m_index = a_inv @ (t_pos @ t_world @ t_neg) @ affine
        frames.append(
            _warp_known(base, jnp.asarray(m_index), center=jnp.zeros(2))
        )
    series = jnp.stack(frames, axis=0)

    space = WorldSpace(
        fixed_affine=jnp.asarray(affine), moving_affine=jnp.asarray(affine)
    )
    res = volreg(
        series,
        reference=0,
        spec=RegistrationSpec(levels=3, iterations=30),
        space=space,
    )
    for t in range(3):
        assert float(ncc(res.realigned[t], base)) > 0.95


def test_volreg_inverse_compositional_matches_forward():
    # The inverse-compositional path (constant-template Hessian, the
    # default for IndexSpace) converges to the same SSD minimiser as the
    # forward Gauss-Newton path.
    base = _blobs_2d(48)
    series = _series_index(base, _THETAS, 2)
    spec = RegistrationSpec(levels=3, iterations=30)
    res_ic = volreg(
        series, reference=0, spec=spec, method='inverse_compositional'
    )
    res_fw = volreg(series, reference=0, spec=spec, method='forward')
    for t in range(4):
        # Same alignment (the rigorous equivalence check), genuine recovery.
        assert float(ncc(res_ic.realigned[t], res_fw.realigned[t])) > 0.999
        assert float(ncc(res_ic.realigned[t], base)) > 0.98
    # Parameters agree to within sub-voxel slack (the SSD minimum is flat in
    # directions the data barely constrains; the alignment above is the
    # authoritative agreement).
    assert np.allclose(
        np.asarray(res_ic.params), np.asarray(res_fw.params), atol=0.2
    )


def test_volreg_validation():
    series = _series_index(_blobs_2d(24), _THETAS, 2)
    # Non-least-squares metric is rejected (the batched LM path is SSD-only).
    with pytest.raises(ValueError):
        volreg(series, spec=RegistrationSpec(metric=LNCC()))
    # A bare image (no frame axis) is not a series.
    with pytest.raises(ValueError):
        volreg(_blobs_2d(24))
    with pytest.raises(ValueError):
        volreg(series, passes=0)
    with pytest.raises(ValueError):
        volreg(series, method='nonsense')
    # Inverse-compositional is index-space only.
    affine = jnp.asarray(np.diag([2.0, 1.0, 1.0]))
    with pytest.raises(ValueError):
        volreg(
            series,
            method='inverse_compositional',
            space=WorldSpace(fixed_affine=affine, moving_affine=affine),
        )
    # C3: an explicit early-exit Convergence is rejected (the while_loop breaks
    # the per-frame vmap); 'auto'/None resolve to the fixed scan and are fine.
    with pytest.raises(ValueError, match='vmap'):
        volreg(series, spec=RegistrationSpec(convergence=Convergence()))
