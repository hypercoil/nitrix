# -*- coding: utf-8 -*-
"""R6: volumetric boundary-based registration (BBR).

A synthetic boundary oracle: a smoothed disk (bright inside, dark outside)
displaced by a known rigid.  Boundary points + outward normals on the
*nominal* circle drive ``bbr_register`` to recover the displacement (the
cost is minimal when the points sit on the moving edge).  Plus the
differentiable-layer check (grad w.r.t. the moving image via
``implicit_minimize``) and validation.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nitrix.linalg import implicit_minimize  # noqa: E402
from nitrix.register import (  # noqa: E402
    BBRSpec,
    Rigid,
    bbr_cost,
    bbr_register,
)


def _disk(n, center, radius, width=1.5):
    """Smoothed disk: ~1 inside ``radius`` of ``center``, ~0 outside."""
    yy, xx = np.mgrid[0:n, 0:n].astype('float64')
    dist = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    return jnp.asarray(1.0 / (1.0 + np.exp((dist - radius) / width)))


def _circle(center, radius, n_pts):
    """Boundary points (axis0, axis1) and outward unit normals on a circle."""
    phi = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    pts = np.stack(
        [center[0] + radius * np.cos(phi), center[1] + radius * np.sin(phi)],
        axis=-1,
    )
    normals = np.stack([np.cos(phi), np.sin(phi)], axis=-1)
    return jnp.asarray(pts), jnp.asarray(normals)


def test_bbr_recovers_translation_2d():
    n, center, radius = 64, (32.0, 32.0), 15.0
    true_t = np.array([3.0, -2.0])
    moving = _disk(n, (center[0] + true_t[0], center[1] + true_t[1]), radius)
    points, normals = _circle(center, radius, 72)

    res = bbr_register(
        moving, points, normals, spec=BBRSpec(step=2.0, iterations=80)
    )
    # The boundary cost drops, and the recovered translation is the offset.
    assert float(res.cost_history[-1]) < float(res.cost_history[0])
    assert np.allclose(np.asarray(res.params[1:]), true_t, atol=0.5)
    assert res.matrix.shape == (3, 3)


def test_bbr_cost_minimal_at_truth():
    # The cost is genuinely lower at the true offset than at identity --
    # i.e. the objective points the optimiser the right way.
    n, center, radius = 64, (32.0, 32.0), 15.0
    true_t = np.array([3.0, -2.0])
    moving = _disk(n, (center[0] + true_t[0], center[1] + true_t[1]), radius)
    points, normals = _circle(center, radius, 72)
    obj_args = dict(
        model=Rigid(),
        ndim=2,
        moving_affine_inv=jnp.eye(3),
        step=2.0,
        slope=0.5,
        q0=0.0,
        method=BBRSpec().interpolation,
        mode='nearest',
        cval=0.0,
        eps=1e-3,
    )
    c_id = float(
        bbr_cost(moving, points, normals, jnp.zeros(3), **obj_args)
    )
    c_true = float(
        bbr_cost(
            moving,
            points,
            normals,
            jnp.asarray([0.0, true_t[0], true_t[1]]),
            **obj_args,
        )
    )
    assert c_true < c_id


def test_bbr_differentiable_wrt_moving():
    # The BBR optimum is differentiable w.r.t. the moving image via the
    # implicit-function layer; check a couple of entries against FD.
    n, center, radius = 24, (12.0, 12.0), 6.0
    moving = _disk(n, (13.0, 11.0), radius)
    points, normals = _circle(center, radius, 48)
    cfg = dict(
        model=Rigid(),
        ndim=2,
        moving_affine_inv=jnp.eye(3),
        step=1.5,
        slope=0.5,
        q0=0.0,
        method=BBRSpec().interpolation,
        mode='nearest',
        cval=0.0,
        eps=1e-3,
    )

    def solve_sum(mv):
        params = implicit_minimize(
            lambda m, p: bbr_cost(m, points, normals, p, **cfg),
            mv,
            jnp.zeros(3),
            maxiter=60,
        )
        return params.sum()

    g = jax.grad(solve_sum)(moving)
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(jnp.abs(g).max()) > 0.0
    # FD on a high-gradient boundary pixel.
    idx = (13, 14)
    eps = 1e-4
    base = float(solve_sum(moving))
    bumped = moving.at[idx].add(eps)
    fd = (float(solve_sum(bumped)) - base) / eps
    assert np.isclose(float(g[idx]), fd, atol=2e-2)


def test_bbr_anisotropic_step_in_mm():
    # With a voxel->world affine, ``step`` is a physical (mm) distance and
    # the normals are physical directions.  Build a true *world* circle (a
    # voxel ellipse under the anisotropic affine) so points + unit
    # world-radial normals are geometrically consistent, and recover a
    # known world translation.
    n = 64
    spacing = np.array([2.0, 1.0])  # axis-0 voxels are 2 mm
    affine = jnp.asarray(np.diag([spacing[0], spacing[1], 1.0]))
    c_world = np.array([32.0, 32.0]) * spacing
    radius_mm = 16.0
    true_t = np.array([3.0, -2.0])  # world (mm) translation

    yy, xx = np.mgrid[0:n, 0:n].astype('float64')
    world = np.stack([spacing[0] * yy, spacing[1] * xx], axis=-1)
    cen = c_world + true_t
    dist = np.sqrt(np.sum((world - cen) ** 2, axis=-1))
    moving = jnp.asarray(1.0 / (1.0 + np.exp((dist - radius_mm) / 1.5)))

    phi = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    pts = np.stack(
        [c_world[0] + radius_mm * np.cos(phi),
         c_world[1] + radius_mm * np.sin(phi)],
        axis=-1,
    )
    nrm = np.stack([np.cos(phi), np.sin(phi)], axis=-1)

    res = bbr_register(
        moving,
        jnp.asarray(pts),
        jnp.asarray(nrm),
        moving_affine=affine,
        spec=BBRSpec(step=2.0, iterations=80),
    )
    assert float(res.cost_history[-1]) < float(res.cost_history[0])
    # Recovered to sub-voxel: the thick axis (2 mm voxels) constrains its
    # component only to ~half a voxel.
    assert np.allclose(np.asarray(res.params[1:]), true_t, atol=1.0)


def test_bbr_validation():
    moving = _disk(32, (16.0, 16.0), 8.0)
    points, normals = _circle((16.0, 16.0), 8.0, 24)
    # normals must match points shape.
    with pytest.raises(ValueError):
        bbr_register(moving, points, normals[:-1])
    # moving rank must match the point dimensionality.
    with pytest.raises(ValueError):
        bbr_register(jnp.zeros((8, 8, 8)), points, normals)
    # unsupported rank.
    p4 = jnp.zeros((10, 4))
    with pytest.raises(ValueError):
        bbr_register(jnp.zeros((8, 8, 8, 8)), p4, p4)
