# -*- coding: utf-8 -*-
"""R6: volumetric boundary-based registration (BBR).

Synthetic boundary oracles: a smoothed 2-D disk (translation) and a smoothed
3-D box (full rigid -- its faces/corners carry rotation signal) displaced by a
known rigid.  Boundary points + outward normals on the *nominal* shape drive
``bbr_register`` to recover the displacement (the cost is minimal when the
points sit on the moving edge).  Covers the grid-multistart + step-annealing
recovery recipe, the scan/while (fixed/early-exit) toggle and its reverse-mode
contract, intensity invariance, the legacy single-stage path, the
differentiable-layer check (grad w.r.t. the moving image via
``implicit_minimize``) and validation.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy import ndimage  # noqa: E402

from nitrix.linalg import implicit_minimize  # noqa: E402
from nitrix.register import (  # noqa: E402
    BBRSearch,
    BBRSpec,
    Convergence,
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


def _rot3(ax, ay, az):
    ax, ay, az = np.deg2rad([ax, ay, az])
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    return (
        np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    )


def _box(n, center, half, width=1.0):
    """Smoothed axis-aligned box: ~1 inside, ~0 outside."""
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype('float64')
    co = np.stack([zz, yy, xx], -1) - np.array(center)
    d = np.max(np.abs(co / np.array(half)), -1)
    return jnp.asarray(
        1.0 / (1.0 + np.exp((d - 1.0) / (width / np.mean(half))))
    )


def _box_surface(center, half, per_face=500, seed=0):
    """Boundary points + outward axis normals on the six faces of a box."""
    rng = np.random.default_rng(seed)
    center = np.array(center)
    half = np.array(half)
    pts, nrm = [], []
    for ax in range(3):
        for s in (-1, 1):
            uv = rng.uniform(-1.0, 1.0, size=(per_face, 2))
            pt = np.zeros((per_face, 3))
            nor = np.zeros((per_face, 3))
            others = [a for a in range(3) if a != ax]
            pt[:, ax] = s * half[ax]
            pt[:, others[0]] = uv[:, 0] * half[others[0]]
            pt[:, others[1]] = uv[:, 1] * half[others[1]]
            nor[:, ax] = s
            pts.append(center + pt)
            nrm.append(nor)
    return jnp.asarray(np.concatenate(pts)), jnp.asarray(np.concatenate(nrm))


def _planted_box():
    """A smoothed box displaced by a known rigid; returns (moving, pts, nrm,
    truth index-space transform)."""
    n, cen, half = 48, (24.0, 24.0, 24.0), (12.0, 9.0, 7.0)
    vol = np.asarray(_box(n, cen, half))
    pts, nrm = _box_surface(cen, half)
    c = (np.array(vol.shape) - 1) / 2.0
    rot = _rot3(5.0, -4.0, 3.0)
    tvox = np.array([2.5, -2.0, 1.5])
    moving = ndimage.affine_transform(
        vol, rot, offset=c - rot @ c + tvox, order=1, mode='nearest'
    )
    m_vox = np.eye(4)
    m_vox[:3, :3] = rot
    m_vox[:3, 3] = c - rot @ c + tvox
    truth = np.linalg.inv(m_vox)  # no affine -> index space
    return jnp.asarray(moving), pts, nrm, truth


def _rigid_error(matrix, truth):
    rel = np.linalg.inv(truth) @ np.asarray(matrix)
    ang = np.rad2deg(
        np.arccos(np.clip((np.trace(rel[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    )
    return ang, float(np.linalg.norm(rel[:3, 3]))


def test_bbr_recovers_translation_2d():
    n, center, radius = 64, (32.0, 32.0), 15.0
    true_t = np.array([3.0, -2.0])
    moving = _disk(n, (center[0] + true_t[0], center[1] + true_t[1]), radius)
    points, normals = _circle(center, radius, 72)

    res = bbr_register(moving, points, normals)
    # Recovered translation is the offset, to well under a voxel.
    assert np.allclose(np.asarray(res.params[1:]), true_t, atol=0.5)
    assert res.matrix.shape == (3, 3)
    # The optimiser lowered the cost relative to the identity start (compared
    # at the finest step, since the cost scale shifts between anneal stages).
    cfg = _obj_cfg(2, jnp.eye(3), step=1.0)
    c_id = float(bbr_cost(moving, points, normals, jnp.zeros(3), **cfg))
    assert float(res.cost) < c_id


def test_bbr_recovers_rigid_3d_box():
    # Full 3-D rigid (rotation + translation): the grid-multistart + step-
    # annealing recipe over six DOF recovers a 7 deg / 6 vox displacement.
    moving, pts, nrm, truth = _planted_box()
    a0, t0 = _rigid_error(np.eye(4), truth)
    res = bbr_register(moving, pts, nrm)
    ang, trans = _rigid_error(res.matrix, truth)
    assert res.matrix.shape == (4, 4)
    assert ang < 1.5 and trans < 1.0  # init ~7 deg / ~6 vox
    assert ang < a0 and trans < t0


def _obj_cfg(ndim, affine_inv, step):
    return dict(
        model=Rigid(),
        ndim=ndim,
        moving_affine_inv=affine_inv,
        step=step,
        slope=0.5,
        q0=0.0,
        method=BBRSpec().interpolation,
        mode='nearest',
        cval=0.0,
        eps=1e-3,
    )


def test_bbr_cost_minimal_at_truth():
    # The cost is genuinely lower at the true offset than at identity --
    # i.e. the objective points the optimiser the right way.
    n, center, radius = 64, (32.0, 32.0), 15.0
    true_t = np.array([3.0, -2.0])
    moving = _disk(n, (center[0] + true_t[0], center[1] + true_t[1]), radius)
    points, normals = _circle(center, radius, 72)
    cfg = _obj_cfg(2, jnp.eye(3), step=2.0)
    c_id = float(bbr_cost(moving, points, normals, jnp.zeros(3), **cfg))
    c_true = float(
        bbr_cost(
            moving,
            points,
            normals,
            jnp.asarray([0.0, true_t[0], true_t[1]]),
            **cfg,
        )
    )
    assert c_true < c_id


def test_bbr_scan_while_toggle():
    # Fixed scan (mode='fixed', the default) and the windowed early-exit
    # (mode='early_exit') both recover; results are close.
    moving, pts, nrm, truth = _planted_box()
    scan = bbr_register(moving, pts, nrm, spec=BBRSpec(mode='fixed'))
    while_ = bbr_register(
        moving,
        pts,
        nrm,
        spec=BBRSpec(
            mode='early_exit', convergence=Convergence(threshold=1e-4)
        ),
    )
    for res in (scan, while_):
        ang, trans = _rigid_error(res.matrix, truth)
        assert ang < 2.0 and trans < 1.5
    # mode='fixed' is the default: identical to the explicit scan.
    default = bbr_register(moving, pts, nrm, spec=BBRSpec())
    assert np.allclose(np.asarray(default.params), np.asarray(scan.params))


def test_bbr_reverse_diff_scan_and_barrier_while():
    # Scan mode is reverse-differentiable through the unrolled trajectory; the
    # while-loop early-exit raises the registration early-exit barrier instead.
    n, center, radius = 32, (16.0, 16.0), 8.0
    moving = _disk(n, (17.0, 15.0), radius)
    points, normals = _circle(center, radius, 48)
    small = BBRSpec(schedule=(2.0, 1.0), iterations=8, search=None)

    def cost_of(spec):
        def f(mv):
            return bbr_register(mv, points, normals, spec=spec).cost

        return f

    g = jax.grad(cost_of(small))(moving)
    assert np.all(np.isfinite(np.asarray(g)))

    early = BBRSpec(
        schedule=(2.0, 1.0),
        iterations=8,
        search=None,
        mode='early_exit',
    )
    with pytest.raises(RuntimeError):
        jax.grad(cost_of(early))(moving)


def test_bbr_intensity_invariance():
    # The contrast is a ratio, so recovery is invariant to a global intensity
    # scale (the normalised-block GD inherits the invariance) -- up to the eps
    # denominator guard, which is not itself scale-free, so a small residual.
    moving, pts, nrm, _ = _planted_box()
    base = bbr_register(moving, pts, nrm)
    scaled = bbr_register(moving * 50.0, pts, nrm)
    assert np.allclose(
        np.asarray(base.params), np.asarray(scaled.params), atol=2e-2
    )


def test_bbr_single_stage_and_no_search():
    # Legacy single-stage path (schedule=None -> one stage at ``step``) with no
    # grid search still recovers; its single-stage cost trace is monotone.
    n, center, radius = 64, (32.0, 32.0), 15.0
    true_t = np.array([3.0, -2.0])
    moving = _disk(n, (center[0] + true_t[0], center[1] + true_t[1]), radius)
    points, normals = _circle(center, radius, 72)
    res = bbr_register(
        moving,
        points,
        normals,
        spec=BBRSpec(schedule=None, step=2.0, search=None, iterations=80),
    )
    assert np.allclose(np.asarray(res.params[1:]), true_t, atol=0.6)
    assert float(res.cost_history[-1]) < float(res.cost_history[0])


def test_bbr_search_capture_range():
    # A wider grid extent captures a larger initial offset (still a local
    # refinement -- the grid widens the basin, it is not a global search).
    n, center, radius = 80, (40.0, 40.0), 18.0
    true_t = np.array([6.0, -5.0])
    moving = _disk(n, (center[0] + true_t[0], center[1] + true_t[1]), radius)
    points, normals = _circle(center, radius, 96)
    res = bbr_register(
        moving,
        points,
        normals,
        spec=BBRSpec(search=BBRSearch(translation=8.0, steps=3)),
    )
    assert np.allclose(np.asarray(res.params[1:]), true_t, atol=0.6)


def test_bbr_differentiable_wrt_moving():
    # The BBR optimum is differentiable w.r.t. the moving image via the
    # implicit-function layer; check a couple of entries against FD.
    n, center, radius = 24, (12.0, 12.0), 6.0
    moving = _disk(n, (13.0, 11.0), radius)
    points, normals = _circle(center, radius, 48)
    cfg = _obj_cfg(2, jnp.eye(3), step=1.5)

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
        [
            c_world[0] + radius_mm * np.cos(phi),
            c_world[1] + radius_mm * np.sin(phi),
        ],
        axis=-1,
    )
    nrm = np.stack([np.cos(phi), np.sin(phi)], axis=-1)

    res = bbr_register(
        moving,
        jnp.asarray(pts),
        jnp.asarray(nrm),
        moving_affine=affine,
    )
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


def test_bbr_jit_smoke():
    # jit-coverage: bbr_register compiles + runs under jax.jit and matches the
    # eager result.
    n, center, radius = 64, (32.0, 32.0), 15.0
    moving = _disk(n, (35.0, 30.0), radius)
    points, normals = _circle(center, radius, 48)
    eager = bbr_register(moving, points, normals)
    jitted = jax.jit(bbr_register)(moving, points, normals)
    assert bool(jnp.all(jnp.isfinite(jitted.matrix)))
    assert np.allclose(
        np.asarray(eager.matrix), np.asarray(jitted.matrix), atol=1e-4
    )
