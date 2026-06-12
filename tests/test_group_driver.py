# -*- coding: utf-8 -*-
"""v4 Phase 2: the dense-field algebra<->group drivers.

The group (greedy, compositive-displacement) driver is the perf path; the
algebra (log-domain SVF) driver is the exact oracle.  Greedy is NOT the SVF
fixed point, so the cross-driver gate is *synthetic-recovery-to-tolerance*, not
field-wise equality: group recovers a known warp as well as algebra, stays
diffeomorphic, and its displacement recovers a stationary velocity via
``field_log`` that re-exponentiates to it and feeds the barycentre machinery.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

from dataclasses import replace  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from nitrix.geometry import (  # noqa: E402
    identity_grid,
    integrate_velocity_field,
    jacobian_det_displacement,
    spatial_transform,
    velocity_mean,
)
from nitrix.metrics import ncc  # noqa: E402
from nitrix.register import (  # noqa: E402
    Convergence,
    DemonsSpec,
    SyNSpec,
    diffeomorphic_demons_register,
    greedy_syn_register,
)
from nitrix.register._svf import _step_clamp_diffeo  # noqa: E402
from nitrix.smoothing import gaussian  # noqa: E402


def _blobs(n=64):
    yy, xx = np.mgrid[0:n, 0:n].astype('float64')
    img = np.zeros((n, n), dtype='float64')
    for cy, cx, s, a in [
        (0.31, 0.38, 0.13, 1.0),
        (0.62, 0.69, 0.16, 0.7),
        (0.72, 0.31, 0.11, 0.6),
        (0.44, 0.75, 0.14, 0.5),
    ]:
        img += a * np.exp(
            -((xx - cx * n) ** 2 + (yy - cy * n) ** 2) / (2 * (s * n) ** 2)
        )
    return jnp.asarray(img)


def _smooth_velocity(shape, sigma, scale, seed):
    rng = np.random.RandomState(seed)
    v = rng.standard_normal(shape + (2,))
    v = np.moveaxis(v, -1, 0)
    v = np.asarray(gaussian(jnp.asarray(v), sigma=sigma, spatial_rank=2))
    return jnp.asarray(scale * np.moveaxis(v, 0, -1))


def _warp(image, v):
    grid = identity_grid(image.shape, dtype=image.dtype)
    s = integrate_velocity_field(v)
    return spatial_transform(image[..., None], grid + s, mode='nearest')[..., 0]


def _interior(a, m=6):
    return np.asarray(a)[m:-m, m:-m]


# ---------------------------------------------------------------------------
# Recovery gate (group recovers as well as algebra) + diffeomorphism
# ---------------------------------------------------------------------------


def test_group_syn_recovers_like_algebra():
    fixed = _blobs(64)
    moving = _warp(fixed, _smooth_velocity((64, 64), 8.0, 45.0, 0))
    spec = SyNSpec(levels=3, iterations=60, radius=2, step=0.5)
    alg = greedy_syn_register(
        moving, fixed, spec=replace(spec, representation='algebra')
    )
    grp = greedy_syn_register(
        moving, fixed, spec=replace(spec, representation='group')
    )
    assert float(ncc(grp.warped, fixed)) > 0.99
    # recovery agrees with the oracle to tolerance (NOT field-wise equality --
    # greedy is a different variational problem from the SVF fixed point)
    assert abs(float(ncc(grp.warped, fixed)) - float(ncc(alg.warped, fixed))) < 0.01
    assert float(grp.jacobian_det.min()) > 0.0  # diffeomorphic (total)
    assert bool(jnp.all(jnp.isfinite(grp.forward_velocity)))  # field_log ok


def test_group_demons_recovers_like_algebra():
    fixed = _blobs(64)
    moving = _warp(fixed, _smooth_velocity((64, 64), 8.0, 30.0, 1))
    spec = DemonsSpec(levels=3, iterations=80)
    alg = diffeomorphic_demons_register(
        moving, fixed, spec=replace(spec, representation='algebra')
    )
    grp = diffeomorphic_demons_register(
        moving, fixed, spec=replace(spec, representation='group')
    )
    assert float(ncc(grp.warped, fixed)) > float(ncc(moving, fixed))
    assert abs(float(ncc(grp.warped, fixed)) - float(ncc(alg.warped, fixed))) < 0.01
    assert float(grp.jacobian_det.min()) > 0.0
    assert bool(jnp.all(jnp.isfinite(grp.velocity)))


# ---------------------------------------------------------------------------
# Velocity recovery: field_log round-trip + feeds the barycentre
# ---------------------------------------------------------------------------


def test_group_velocity_roundtrips_and_feeds_barycentre():
    # In group mode the recipe recovers `velocity = field_log(displacement)`;
    # exp(velocity) reproduces the displacement (the round-trip is exact on the
    # SVF submanifold by construction), and the recovered velocity flows through
    # the v3 barycentre machinery (template construction).
    fixed = _blobs(64)
    moving = _warp(fixed, _smooth_velocity((64, 64), 8.0, 30.0, 2))
    spec = DemonsSpec(levels=3, iterations=80, representation='group')
    res = diffeomorphic_demons_register(moving, fixed, spec=spec)
    re_exp = integrate_velocity_field(
        res.velocity, n_steps=spec.n_steps, mode=spec.boundary_mode
    )
    assert np.abs(_interior(re_exp) - _interior(res.displacement)).max() < 1e-3
    vm = velocity_mean(jnp.stack([res.velocity, res.velocity]))
    assert bool(jnp.all(jnp.isfinite(vm)))


# ---------------------------------------------------------------------------
# Per-step diffeomorphism guard
# ---------------------------------------------------------------------------


def test_step_clamp_diffeo_prevents_fold():
    # A sinusoidal x-displacement steep enough that 1 + d(dx)/dx dips negative
    # (the warp folds); the guard halves it back to a diffeomorphism (det > 0
    # everywhere), where a magnitude-only clamp would not.
    n = 40
    yy, _ = np.mgrid[0:n, 0:n].astype('float64')
    # axis-0 displacement varying along axis 0 -> det = 1 + d(d0)/d(axis0) dips
    # negative (a true fold, not a det-preserving shear).
    d0 = 6.0 * np.sin(2 * np.pi * 2 * yy / n)
    delta = jnp.asarray(np.stack([d0, np.zeros_like(d0)], axis=-1))
    assert float(jacobian_det_displacement(delta).min()) < 0.0  # folds
    clamped = _step_clamp_diffeo(delta)
    assert float(jacobian_det_displacement(clamped).min()) > 0.0  # guard fixes


# ---------------------------------------------------------------------------
# SVF early-exit (A3 / Phase 2.5)
# ---------------------------------------------------------------------------


def test_demons_early_exit_recovers_and_pads_history():
    # The SSD log-Demons converges fast, so the windowed-slope early-exit stops
    # well before the cap with the same recovery; the cost_history keeps its
    # (levels x iterations) shape (the unrun tail padded with the final cost).
    fixed = _blobs(64)
    moving = _warp(fixed, _smooth_velocity((64, 64), 8.0, 30.0, 1))
    spec = DemonsSpec(levels=1, iterations=80)
    full = diffeomorphic_demons_register(moving, fixed, spec=spec)
    early = diffeomorphic_demons_register(
        moving,
        fixed,
        spec=replace(spec, convergence=Convergence(threshold=1e-4, window=8)),
    )
    assert early.cost_history.shape == full.cost_history.shape  # padded
    assert abs(
        float(ncc(early.warped, fixed)) - float(ncc(full.warped, fixed))
    ) < 0.005
    # the early-exit fired: a constant padded tail before the 80 cap
    ch = np.asarray(early.cost_history)
    changing = int(np.sum(np.abs(np.diff(ch)) > 1e-12)) + 1
    assert changing < 80
    assert float(early.jacobian_det.min()) > 0.0


def test_convergence_none_matches_default():
    fixed = _blobs(48)
    moving = _warp(fixed, _smooth_velocity((48, 48), 6.0, 18.0, 3))
    spec = DemonsSpec(levels=1, iterations=20)
    a = diffeomorphic_demons_register(moving, fixed, spec=spec)
    b = diffeomorphic_demons_register(
        moving, fixed, spec=replace(spec, convergence=None)
    )
    assert np.allclose(np.asarray(a.warped), np.asarray(b.warped))


def test_syn_early_exit_recovers_diffeomorphic():
    fixed = _blobs(64)
    moving = _warp(fixed, _smooth_velocity((64, 64), 8.0, 45.0, 0))
    spec = SyNSpec(
        levels=2,
        iterations=60,
        radius=2,
        step=0.5,
        convergence=Convergence(threshold=1e-4, window=10),
    )
    res = greedy_syn_register(moving, fixed, spec=spec)
    assert float(ncc(res.warped, fixed)) > 0.99
    assert float(res.jacobian_det.min()) > 0.0


def test_step_clamp_diffeo_noop_when_diffeomorphic():
    # A gentle (already-diffeomorphic) field is passed through untouched.
    delta = _smooth_velocity((40, 40), 4.0, 0.3, 8)
    assert float(jacobian_det_displacement(delta).min()) > 0.1
    clamped = _step_clamp_diffeo(delta)
    assert np.allclose(np.asarray(clamped), np.asarray(delta))
