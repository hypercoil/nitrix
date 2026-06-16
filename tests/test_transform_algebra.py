# -*- coding: utf-8 -*-
"""V3a: transform algebra -- Lie-group Fréchet mean + geodesic interpolation.

The barycentre substrate for groupwise / template construction and motion
summary: the Fréchet (Karcher) mean of homogeneous transforms (rigid SE(n) ⊂
affine), the SVF mean, and geodesic interpolation -- plus the general
``matrix_log`` that the affine mean warranted.

The mean / geodesic use ``matrix_log`` (hence ``safe_inv``), so they are
forward / eager ops on the wedged-cuSolver dev box (jit- and grad-clean only on
a healthy GPU); these tests exercise the forward pass.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from nitrix.geometry import (  # noqa: E402
    affine_exp,
    affine_grid,
    fuse_transforms,
    identity_grid,
    rigid_exp,
    spatial_transform,
    transform_geodesic,
    transform_mean,
    velocity_mean,
)
from nitrix.linalg import matrix_exp, matrix_log  # noqa: E402
from nitrix.metrics import ncc  # noqa: E402


def _blobs2d(n=48):
    yy, xx = np.mgrid[0:n, 0:n].astype('float64')
    img = np.zeros((n, n), dtype='float64')
    for cy, cx, s, a in [(0.3, 0.4, 0.13, 1.0), (0.65, 0.6, 0.15, 0.7)]:
        img += a * np.exp(
            -((xx - cx * n) ** 2 + (yy - cy * n) ** 2) / (2 * (s * n) ** 2)
        )
    return jnp.asarray(img)


def _np(x):
    return np.asarray(x)


def _se3_algebra(omega, trans):
    """A rigid (se(3)) algebra matrix: skew rotation block + translation."""
    ox, oy, oz = omega
    skew = np.array([[0, -oz, oy], [oz, 0, -ox], [-oy, ox, 0]])
    d = np.zeros((4, 4))
    d[:3, :3] = skew
    d[:3, 3] = trans
    return jnp.asarray(d)


def test_matrix_log_inverts_matrix_exp():
    rng = np.random.RandomState(0)
    # affine-algebra generator (last row zero): log(exp(X)) == X
    x = jnp.asarray(rng.standard_normal((4, 4)) * 0.2).at[3, :].set(0.0)
    assert np.allclose(_np(matrix_log(matrix_exp(x))), _np(x), atol=1e-10)
    # rigid with a large translation: exp(log(M)) == M
    m = rigid_exp(jnp.asarray([0.2, -0.1, 0.15, 8.0, -6.0, 7.0]), ndim=3)
    assert np.allclose(_np(matrix_exp(matrix_log(m))), _np(m), atol=1e-10)


def test_matrix_log_large_translation_illscaled_in_domain():
    # B3: a valid large-translation / anisotropic-scale / sheared affine (large
    # Frobenius ‖A−I‖, but a spectrum the square roots still drive to I) round-
    # trips to ~machine precision and is NOT spuriously NaN'd by the guard.
    a = jnp.asarray(
        [[2.5, 0.8, 180.0], [0.0, 0.4, -150.0], [0.0, 0.0, 1.0]]
    )
    log_a = matrix_log(a)
    assert bool(jnp.all(jnp.isfinite(log_a)))
    rel = float(jnp.linalg.norm(matrix_exp(log_a) - a) / jnp.linalg.norm(a))
    assert rel < 1e-8


def test_matrix_log_large_rotation_in_domain_not_nan():
    # A 150-deg rotation is in-domain (its eigenvalues are off the negative real
    # axis until 180 deg) -- the guard must not be so aggressive as to NaN it.
    th = np.deg2rad(150.0)
    rot = jnp.asarray(
        [
            [np.cos(th), -np.sin(th), 4.0],
            [np.sin(th), np.cos(th), -3.0],
            [0.0, 0.0, 1.0],
        ]
    )
    log_r = matrix_log(rot)
    assert bool(jnp.all(jnp.isfinite(log_r)))
    rel = float(jnp.linalg.norm(matrix_exp(log_r) - rot) / jnp.linalg.norm(rot))
    assert rel < 1e-8


def test_matrix_log_out_of_domain_returns_nan():
    # Outside the domain (an eigenvalue on the closed negative real axis) the
    # principal log does not exist: matrix_log returns a LOUD NaN, not the
    # finite garbage the bare series would yield.
    reflection = jnp.asarray(
        [[-1.0, 0.0, 5.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]]
    )
    assert bool(jnp.all(jnp.isnan(matrix_log(reflection))))
    th = np.pi  # 180-deg rotation: eigenvalue -1
    rot180 = jnp.asarray(
        [
            [np.cos(th), -np.sin(th), 0.0],
            [np.sin(th), np.cos(th), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert bool(jnp.all(jnp.isnan(matrix_log(rot180))))


def test_transform_mean_of_identical():
    t = rigid_exp(jnp.asarray([0.1, -0.05, 0.08, 3.0, -2.0, 1.5]), ndim=3)
    assert np.allclose(
        _np(transform_mean(jnp.stack([t, t, t]))), _np(t), atol=1e-6
    )


def test_transform_mean_rigid_symmetric_recovers_centre():
    centre = rigid_exp(jnp.asarray([0.12, 0.05, -0.1, 2.0, -3.0, 4.0]), ndim=3)
    # symmetric in the true matrix chart (matrix_exp of an se(3) algebra elt)
    d = _se3_algebra([0.04, -0.03, 0.05], [1.0, 0.5, -0.8])
    stack = jnp.stack([centre @ matrix_exp(d), centre @ matrix_exp(-d)])
    mean = transform_mean(stack)
    assert np.allclose(_np(mean), _np(centre), atol=1e-5)
    # the mean of rigids is rigid (orthogonal block, det +1)
    r = _np(mean)[:3, :3]
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-6)


def test_transform_mean_pure_translations_is_arithmetic():
    ts = np.array([[1.0, 2.0, -3.0], [4.0, -1.0, 0.5], [-2.0, 3.0, 1.0]])
    stack = jnp.stack(
        [rigid_exp(jnp.asarray([0, 0, 0, *t]), ndim=3) for t in ts]
    )
    mean = transform_mean(stack)
    assert np.allclose(_np(mean)[:3, :3], np.eye(3), atol=1e-6)
    assert np.allclose(_np(mean)[:3, 3], ts.mean(0), atol=1e-5)


def test_transform_mean_affine_symmetric_recovers_centre():
    rng = np.random.RandomState(1)
    centre = affine_exp(
        jnp.asarray([*(rng.standard_normal(9) * 0.15), 2.0, -1.0, 1.5]), ndim=3
    )
    d = jnp.asarray(rng.standard_normal((4, 4)) * 0.1).at[3, :].set(0.0)
    stack = jnp.stack([centre @ matrix_exp(d), centre @ matrix_exp(-d)])
    assert np.allclose(_np(transform_mean(stack)), _np(centre), atol=1e-4)


def test_transform_mean_residual_small_on_clustered_cohort():
    # B3: return_residual surfaces convergence -- a clustered cohort drives the
    # final Karcher update tangent to ~0 within the default cap.
    rng = np.random.RandomState(3)
    centre = rigid_exp(jnp.asarray([0.1, -0.05, 0.08, 3.0, -2.0, 1.5]), ndim=3)
    stack = jnp.stack(
        [
            centre
            @ matrix_exp(
                _se3_algebra(rng.standard_normal(3) * 0.05, rng.standard_normal(3) * 0.5)
            )
            for _ in range(5)
        ]
    )
    mean, residual = transform_mean(stack, return_residual=True)
    assert bool(jnp.all(jnp.isfinite(mean)))
    assert float(residual) < 1e-6  # converged: the final Karcher step ~ 0


def test_transform_mean_dispersed_cohort_needs_more_iters():
    # A widely-dispersed, NON-commuting (multi-axis, ~80-deg) rotation cohort:
    # the Karcher fixed point converges slowly (a single-axis cohort would
    # converge in one step).  The residual is honest about non-convergence at a
    # tight cap and shrinks as iters rises -- the fixed cap does not silently
    # return a non-converged mean.
    axes_angles = [
        ([1.0, 0.0, 0.0], 1.4),
        ([0.0, 1.0, 0.0], -1.3),
        ([0.0, 0.0, 1.0], 1.5),
        ([1.0, 1.0, 0.0], 1.2),
        ([0.0, 1.0, 1.0], -1.4),
    ]
    stack = jnp.stack(
        [
            matrix_exp(
                _se3_algebra(
                    np.asarray(ax) / np.linalg.norm(ax) * ang, [2.0, -1.0, 1.5]
                )
            )
            for ax, ang in axes_angles
        ]
    )
    _, r_few = transform_mean(stack, iters=2, return_residual=True)
    mean, r_many = transform_mean(stack, iters=30, return_residual=True)
    assert float(r_few) > 1e-2  # a tight cap leaves it visibly non-converged
    assert float(r_many) < float(r_few)
    assert float(r_many) < 1e-6  # raising the cap drives it to the barycentre
    assert bool(jnp.all(jnp.isfinite(mean)))


def test_transform_mean_antipodal_cohort_is_nan():
    # An antipodal pair (~π apart) sits on matrix_log's negative-real-axis
    # boundary: the barycentre is genuinely ill-defined and surfaces as NaN
    # (loud), not silent garbage.
    rot180 = matrix_exp(_se3_algebra([np.pi, 0.0, 0.0], [0.0, 0.0, 0.0]))
    mean = transform_mean(jnp.stack([jnp.eye(4), rot180]))
    assert bool(jnp.any(jnp.isnan(mean)))


def test_transform_geodesic_endpoints_and_halfway():
    t = rigid_exp(jnp.asarray([0.3, -0.2, 0.25, 5.0, -4.0, 3.0]), ndim=3)
    eye = jnp.eye(4)
    assert np.allclose(_np(transform_geodesic(t, 0.0)), _np(eye), atol=1e-9)
    assert np.allclose(_np(transform_geodesic(t, 1.0)), _np(t), atol=1e-8)
    half = transform_geodesic(t, 0.5)
    assert np.allclose(_np(half @ half), _np(t), atol=1e-8)


def test_fuse_inverse_chain_is_identity():
    # A followed by A⁻¹ fuses to ~zero displacement (matrices only -> exact).
    a = rigid_exp(jnp.asarray([0.2, 5.0, -3.0]), ndim=2)
    a_inv = jnp.asarray(np.linalg.inv(_np(a)))
    fused = fuse_transforms([a, a_inv], (48, 48))
    assert float(jnp.abs(fused).max()) < 1e-6


def test_fuse_matches_sequential_warp():
    # A mixed chain (affine then displacement) fused to one resampling
    # reproduces the sequential two-resample warp (one interpolation, not two).
    moving = _blobs2d(48)
    a = rigid_exp(jnp.asarray([0.12, 3.0, -2.0]), ndim=2)
    rng = np.random.RandomState(0)
    from nitrix.smoothing import gaussian

    disp = rng.standard_normal((48, 48, 2))
    disp = np.stack(
        [
            np.asarray(gaussian(jnp.asarray(disp[..., i]), sigma=6.0))
            for i in range(2)
        ],
        axis=-1,
    )
    disp = jnp.asarray(2.0 * disp)
    grid = identity_grid((48, 48), dtype=moving.dtype)
    # sequential: warp by A, then by the displacement (two interpolations)
    m1 = spatial_transform(moving[..., None], affine_grid(a, (48, 48)))[..., 0]
    m2 = spatial_transform(m1[..., None], grid + disp)[..., 0]
    # fused: one interpolation of the composed grid
    fused = fuse_transforms([a, disp], (48, 48))
    m_fused = spatial_transform(moving[..., None], grid + fused)[..., 0]
    assert float(ncc(m_fused, m2)) > 0.99


def test_fuse_batched_application():
    # Batched application over a cohort: vmap the fused warp across K transforms
    # (the cohort-throughput path; spatial_transform_batched is the primitive).
    moving = _blobs2d(48)
    grid = identity_grid((48, 48), dtype=moving.dtype)
    angles = jnp.asarray([0.05, 0.1, 0.15, -0.08])
    fused = jnp.stack(
        [
            fuse_transforms(
                [rigid_exp(jnp.asarray([a, 2.0, -1.0]), ndim=2)], (48, 48)
            )
            for a in angles
        ]
    )

    def warp(d):
        return spatial_transform(moving[..., None], grid + d)[..., 0]

    out = jax.vmap(warp)(fused)
    assert out.shape == (4, 48, 48)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_velocity_mean():
    rng = np.random.RandomState(2)
    v = jnp.asarray(rng.standard_normal((5, 16, 16, 2)))
    assert np.allclose(_np(velocity_mean(v)), _np(v).mean(0), atol=1e-10)
    w = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    expect = np.tensordot(_np(w) / _np(w).sum(), _np(v), axes=(0, 0))
    assert np.allclose(_np(velocity_mean(v, weights=w)), expect, atol=1e-10)
