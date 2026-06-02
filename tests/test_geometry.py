# -*- coding: utf-8 -*-
"""Tests for ``nitrix.geometry``.

Coverage:

- **grid**: ``identity_grid`` produces the correct coordinates;
  ``spatial_transform`` with the identity grid is the identity map;
  with a constant displacement is a translation; ``resample``
  preserves constants and smooth ramps; ``integrate_velocity_field``
  produces a near-identity displacement for zero / small velocity
  and recovers the linearised expectation for small steps.
- **sphere**: lat/long round-trip; ``spherical_geodesic_distance``
  matches the spherical law of cosines and gives ``pi`` for
  antipodal points; ``spherical_conv`` preserves constants, equals
  identity in the small-sigma limit, and produces the same answer
  as a hand-coded all-pairs reference on small inputs.
- **coords**: centre-of-mass on points and on a grid matches a
  hand-coded reference; ``compactness_penalty`` is zero for a
  delta-weight, positive for spread-out weight.

This is the third validation of the substrate bet (after morphology
and smoothing): ``spherical_conv`` lowers to ``semiring_ell_matmul``
rather than the legacy ``O(N²)`` all-pairs implementation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nitrix.geometry import (
    cartesian_to_latlong,
    center_of_mass_grid,
    center_of_mass_points,
    compactness_penalty,
    displacement_from_reference_grid,
    displacement_from_reference_points,
    identity_grid,
    integrate_velocity_field,
    latlong_to_cartesian,
    resample,
    spatial_transform,
    spatial_transform_batched,
    spherical_conv,
    spherical_geodesic_distance,
)


jax.config.update('jax_enable_x64', True)


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------


def test_identity_grid_2d():
    g = identity_grid((3, 4))
    assert g.shape == (3, 4, 2)
    np.testing.assert_array_equal(g[2, 1], jnp.array([2.0, 1.0]))
    np.testing.assert_array_equal(g[0, 0], jnp.array([0.0, 0.0]))


def test_identity_grid_3d():
    g = identity_grid((2, 3, 4))
    assert g.shape == (2, 3, 4, 3)
    np.testing.assert_array_equal(g[1, 2, 3], jnp.array([1.0, 2.0, 3.0]))


def test_spatial_transform_identity_is_identity():
    img = jax.random.normal(jax.random.key(0), (5, 5, 2))
    out = spatial_transform(img, identity_grid((5, 5)))
    np.testing.assert_array_equal(out, img)


def test_spatial_transform_constant_displacement_is_translation():
    '''Adding a constant 1-pixel shift in the x direction should
    translate the image by one pixel.
    '''
    img = jnp.arange(5 * 5, dtype=jnp.float64).reshape(5, 5)[..., None]
    delta = jnp.zeros((5, 5, 2)).at[..., 0].set(1.0)
    deform = identity_grid((5, 5)) + delta
    out = spatial_transform(img, deform)
    # Out[i, j] should equal img[i+1, j] for interior rows.
    np.testing.assert_allclose(out[:-1, :, 0], img[1:, :, 0], atol=1e-10)


def test_spatial_transform_out_of_bounds_uses_cval():
    img = jnp.ones((4, 4, 1))
    # Map every pixel to a far-away coordinate.
    far = jnp.full((4, 4, 2), 1e6)
    out = spatial_transform(img, far, cval=-1.0)
    np.testing.assert_array_equal(out, jnp.full((4, 4, 1), -1.0))


def test_spatial_transform_mode_nearest_edge_replicates():
    '''mode="nearest" clamps OOB coords to the input extent.

    Sampling at coordinate (-100, -100) on a 4x4 image with mode='nearest'
    must return image[0, 0]; sampling at (1e6, 1e6) returns image[-1, -1].
    Matches scipy.ndimage.map_coordinates(mode='nearest').
    '''
    img = jnp.arange(16, dtype=jnp.float32).reshape(4, 4, 1)
    # Sample at (negative-far, negative-far) -> should clamp to (0, 0)
    coords_neg = jnp.full((4, 4, 2), -1e6)
    out_neg = spatial_transform(img, coords_neg, mode='nearest')
    np.testing.assert_array_equal(
        out_neg, jnp.full((4, 4, 1), img[0, 0, 0]),
    )
    # Sample at (positive-far, positive-far) -> should clamp to (-1, -1)
    coords_pos = jnp.full((4, 4, 2), 1e6)
    out_pos = spatial_transform(img, coords_pos, mode='nearest')
    np.testing.assert_array_equal(
        out_pos, jnp.full((4, 4, 1), img[-1, -1, 0]),
    )


def test_spatial_transform_mode_wrap_is_periodic():
    '''mode="wrap" treats the image as toroidally periodic.'''
    img = jnp.arange(16, dtype=jnp.float32).reshape(4, 4, 1)
    # Coords at (4, 0) wrap to (0, 0).
    coords = jnp.zeros((1, 1, 2)).at[0, 0].set(jnp.array([4.0, 0.0]))
    out = spatial_transform(img, coords, mode='wrap')
    np.testing.assert_allclose(
        out[0, 0, 0], img[0, 0, 0], atol=1e-6,
    )


def test_spatial_transform_accepts_leading_batch_dims():
    '''spatial_transform now vmaps over leading batch dims natively.'''
    B = 3
    img = jnp.broadcast_to(
        jnp.arange(16, dtype=jnp.float32).reshape(1, 4, 4, 1),
        (B, 4, 4, 1),
    )
    # Each batch element gets the same identity deformation.
    grid_ = identity_grid((4, 4), dtype=jnp.float32)  # (4, 4, 2)
    deform = jnp.broadcast_to(grid_[None], (B, 4, 4, 2))
    out = spatial_transform(img, deform)
    assert out.shape == (B, 4, 4, 1)
    np.testing.assert_allclose(out, img, atol=1e-6)


def test_spatial_transform_batch_dim_mismatch_raises():
    img = jnp.zeros((3, 4, 4, 1))
    deform = jnp.zeros((4, 4, 4, 2))  # batch size differs
    with pytest.raises(ValueError, match='leading batch dims must match'):
        spatial_transform(img, deform)


def test_spatial_transform_batched_shared_deformation():
    '''A batch of images under one shared deformation equals a
    manual vmap with in_axes=(0, None).'''
    B = 3
    rng = np.random.default_rng(0)
    img = jnp.asarray(rng.standard_normal((B, 5, 5, 2)), dtype=jnp.float32)
    deform = identity_grid((5, 5), dtype=jnp.float32) + 0.3  # (5, 5, 2)
    out = spatial_transform_batched(img, deform)
    ref = jax.vmap(lambda im: spatial_transform(im, deform))(img)
    assert out.shape == (B, 5, 5, 2)
    np.testing.assert_allclose(out, ref, atol=1e-6)


def test_spatial_transform_batched_shared_image():
    '''One image under a batch of deformations equals a manual vmap
    with in_axes=(None, 0).'''
    B = 4
    rng = np.random.default_rng(1)
    img = jnp.asarray(rng.standard_normal((6, 6, 1)), dtype=jnp.float32)
    base = identity_grid((6, 6), dtype=jnp.float32)
    deforms = jnp.stack([base + float(s) * 0.1 for s in range(B)])  # (B,6,6,2)
    out = spatial_transform_batched(img, deforms)
    ref = jax.vmap(lambda df: spatial_transform(img, df))(deforms)
    assert out.shape == (B, 6, 6, 1)
    np.testing.assert_allclose(out, ref, atol=1e-6)


def test_spatial_transform_batched_both_batched_matches_native():
    '''When both operands are batched it agrees with the native
    multi-leading-dim spatial_transform.'''
    B = 3
    rng = np.random.default_rng(2)
    img = jnp.asarray(rng.standard_normal((B, 5, 5, 1)), dtype=jnp.float32)
    deform = jnp.broadcast_to(
        identity_grid((5, 5), dtype=jnp.float32)[None], (B, 5, 5, 2),
    )
    out = spatial_transform_batched(img, deform)
    ref = spatial_transform(img, deform)
    np.testing.assert_allclose(out, ref, atol=1e-6)


def test_spatial_transform_batched_unbatched_raises():
    img = jnp.zeros((5, 5, 1))
    deform = identity_grid((5, 5))
    with pytest.raises(ValueError, match='leading batch axis'):
        spatial_transform_batched(img, deform)


# ---------------------------------------------------------------------------
# sphere_grid: parameterised-sphere topology padding (JOSA J.1a)
# ---------------------------------------------------------------------------


def test_sphere_grid_pad_2d_longitudinal_wrap():
    '''The longitudinal axis pads as a circular wrap.'''
    from nitrix.geometry import sphere_grid_pad_2d
    img = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)
    padded = sphere_grid_pad_2d(
        img, pad=(0, 1), height_axis=-2, width_axis=-1,
    )
    assert padded.shape == (4, 6)
    # First column = last column of input; last column = first.
    np.testing.assert_array_equal(padded[:, 0], img[:, -1])
    np.testing.assert_array_equal(padded[:, -1], img[:, 0])
    np.testing.assert_array_equal(padded[:, 1:-1], img)


def test_sphere_grid_pad_2d_pole_flip_skips_pole_row():
    '''The pole pad uses row[1] (not row[0]) flipped + rolled by W//2.

    This is the load-bearing detail of the topology: row[0] is the
    compressed pole point and reflecting it would duplicate the pole.
    '''
    from nitrix.geometry import sphere_grid_pad_2d
    H, W = 4, 4
    img = jnp.arange(H * W, dtype=jnp.float32).reshape(H, W)
    padded = sphere_grid_pad_2d(
        img, pad=(1, 0), height_axis=-2, width_axis=-1,
    )
    assert padded.shape == (H + 2, W)
    # Top pad row = row[1] rolled by W // 2 = 2 (no longitudinal pad).
    np.testing.assert_array_equal(
        padded[0], jnp.roll(img[1], W // 2),
    )
    # Bottom pad row = row[H-2] rolled.
    np.testing.assert_array_equal(
        padded[-1], jnp.roll(img[H - 2], W // 2),
    )
    # Body unchanged.
    np.testing.assert_array_equal(padded[1:-1], img)


def test_sphere_grid_pad_2d_full_topology_roundtrip():
    '''pad + unpad recovers the original.'''
    from nitrix.geometry import sphere_grid_pad_2d, sphere_grid_unpad_2d
    img = jax.random.normal(jax.random.key(0), (8, 8))
    padded = sphere_grid_pad_2d(img, pad=2)
    assert padded.shape == (12, 12)
    out = sphere_grid_unpad_2d(padded, pad=2)
    np.testing.assert_array_equal(out, img)


def test_sphere_grid_pad_2d_pole_negate_channel():
    '''Negate the longitudinal-flow channel in pole-pad regions.

    A constant longitudinal flow flips sign across each pole;
    a constant latitudinal flow does not.  Verify the negate flag
    operates on the named index alone.
    '''
    from nitrix.geometry import sphere_grid_pad_2d
    # 2D flow field: (H, W, 2) with channel 0 = longitudinal, 1 = lat.
    flow = jnp.zeros((4, 4, 2))
    flow = flow.at[..., 0].set(1.0)   # longitudinal: 1
    flow = flow.at[..., 1].set(0.5)   # latitudinal: 0.5
    padded = sphere_grid_pad_2d(
        flow, pad=(1, 0),
        height_axis=-3, width_axis=-2,
        pole_negate_channel=0, pole_negate_axis=-1,
    )
    assert padded.shape == (6, 4, 2)
    # Top pad row: longitudinal channel should be -1; latitudinal still 0.5.
    np.testing.assert_array_equal(padded[0, :, 0], jnp.full((4,), -1.0))
    np.testing.assert_array_equal(padded[0, :, 1], jnp.full((4,), 0.5))
    # Body unchanged.
    np.testing.assert_array_equal(padded[1:-1, :, 0], jnp.full((4, 4), 1.0))


def test_sphere_grid_pad_2d_channel_last_axes():
    '''Works on channel-last (B, H, W, C) layout.'''
    from nitrix.geometry import sphere_grid_pad_2d
    img = jax.random.normal(jax.random.key(0), (2, 4, 6, 3))
    padded = sphere_grid_pad_2d(
        img, pad=1, height_axis=-3, width_axis=-2,
    )
    assert padded.shape == (2, 6, 8, 3)


def test_sphere_grid_pad_2d_odd_width_raises():
    '''Odd width is rejected (pole roll is by W // 2; odd W would shift).'''
    from nitrix.geometry import sphere_grid_pad_2d
    img = jnp.zeros((4, 5))
    with pytest.raises(ValueError, match='width must be even'):
        sphere_grid_pad_2d(img, pad=1)


def test_sphere_grid_pad_2d_differentiable():
    '''Pad is pure indexing/flip/roll -- gradient is identity-like.'''
    from nitrix.geometry import sphere_grid_pad_2d
    img = jax.random.normal(jax.random.key(0), (4, 4))
    def loss(img):
        return jnp.sum(sphere_grid_pad_2d(img, pad=1) ** 2)
    g = jax.grad(loss)(img)
    assert g.shape == img.shape
    assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# Jacobian primitives (JOSA J.1b)
# ---------------------------------------------------------------------------


def test_jacobian_displacement_zero_is_identity():
    '''Zero displacement -> J = I everywhere.'''
    from nitrix.geometry import jacobian_displacement
    u = jnp.zeros((6, 8, 2))
    J = jacobian_displacement(u)
    assert J.shape == (6, 8, 2, 2)
    np.testing.assert_array_equal(J, jnp.broadcast_to(jnp.eye(2), (6, 8, 2, 2)))


def test_jacobian_det_displacement_zero_is_one():
    '''Zero displacement -> det(J) = 1 everywhere.'''
    from nitrix.geometry import jacobian_det_displacement
    u = jnp.zeros((6, 8, 2))
    det = jacobian_det_displacement(u)
    np.testing.assert_allclose(det, jnp.ones((6, 8)), atol=1e-12)


def test_jacobian_det_displacement_bulk_compression():
    '''u(x) = -0.1 x -> J = 0.9 I -> det = 0.81 (2D), 0.729 (3D).'''
    from nitrix.geometry import (
        identity_grid, jacobian_det_displacement,
    )
    # 2D
    grid_2d = identity_grid((8, 8), dtype=jnp.float64)
    u_2d = -0.1 * grid_2d
    det_2d = jacobian_det_displacement(u_2d)
    np.testing.assert_allclose(det_2d[4, 4], 0.81, atol=1e-12)
    # 3D
    grid_3d = identity_grid((6, 6, 6), dtype=jnp.float64)
    u_3d = -0.1 * grid_3d
    det_3d = jacobian_det_displacement(u_3d)
    np.testing.assert_allclose(det_3d[3, 3, 3], 0.729, atol=1e-12)


def test_jacobian_det_displacement_folding():
    '''A folding deformation has det <= 0 where it folds.

    Construct u with ∂u_0/∂x_0 = -2 (i.e. u_0 = -2*x_0), which
    gives J = [[1 + (-2), 0], [0, 1]] = [[-1, 0], [0, 1]]; det = -1.
    '''
    from nitrix.geometry import jacobian_det_displacement
    H, W = 10, 10
    # u_0(i, j) = -2 * i; u_1 = 0.
    rows = jnp.arange(H, dtype=jnp.float64)[:, None]
    u = jnp.zeros((H, W, 2), dtype=jnp.float64)
    u = u.at[..., 0].set(jnp.broadcast_to(rows * -2.0, (H, W)))
    det = jacobian_det_displacement(u)
    # Interior should have det ~= -1.
    np.testing.assert_allclose(det[H // 2, W // 2], -1.0, atol=1e-12)


def test_jacobian_displacement_boundary_mode_nearest():
    '''Boundary mode "nearest" replicates the edge cell.

    For a linear ramp ``u_0 = i``, central diff in the interior is
    ``(u_0[i+1] - u_0[i-1]) / 2 = 1`` (exact).  At ``i = 0`` the
    "nearest" mode treats the previous neighbour as ``u_0[0]``
    itself, giving ``(u_0[1] - u_0[0]) / 2 = 0.5`` -- half the
    true gradient.  This matches scipy / voxelmorph convention.
    '''
    from nitrix.geometry import jacobian_displacement
    H, W = 6, 6
    rows = jnp.arange(H, dtype=jnp.float64)[:, None]
    u = jnp.zeros((H, W, 2), dtype=jnp.float64)
    u = u.at[..., 0].set(jnp.broadcast_to(rows, (H, W)))
    J = jacobian_displacement(u, boundary_mode='nearest')
    # Interior: ∂u_0/∂x_0 = 1 -> J[..., 0, 0] = 2.
    np.testing.assert_allclose(J[1:-1, :, 0, 0], 2.0, atol=1e-12)
    # Boundary (i=0 and i=H-1): the edge-replicated central diff
    # gives ∂u_0/∂x_0 = 0.5 -> J = 1.5.
    np.testing.assert_allclose(J[0, :, 0, 0], 1.5, atol=1e-12)
    np.testing.assert_allclose(J[-1, :, 0, 0], 1.5, atol=1e-12)
    # ∂u_1/∂x_1 = 0 everywhere -> J[..., 1, 1] = 1.
    np.testing.assert_allclose(J[..., 1, 1], 1.0, atol=1e-12)


def test_jacobian_displacement_anisotropic_spacing():
    '''Per-axis spacing scales the central difference correctly.'''
    from nitrix.geometry import jacobian_displacement
    # u_0(i, j) = i (ramp in x_0).  With spacing=(2, 1):
    #   ∂u_0/∂x_0 = 1/2 (we measure physical units, voxel spacing 2)
    # So J[..., 0, 0] = 1 + 0.5 = 1.5.
    H, W = 6, 6
    rows = jnp.arange(H, dtype=jnp.float64)[:, None]
    u = jnp.zeros((H, W, 2), dtype=jnp.float64)
    u = u.at[..., 0].set(jnp.broadcast_to(rows, (H, W)))
    J = jacobian_displacement(u, spacing=(2.0, 1.0))
    np.testing.assert_allclose(J[1:-1, 1:-1, 0, 0], 1.5, atol=1e-12)


def test_jacobian_det_displacement_differentiable():
    '''det is smooth in u; jax.grad must produce finite gradients.'''
    from nitrix.geometry import jacobian_det_displacement
    u = jax.random.normal(jax.random.key(0), (6, 8, 2)) * 0.1
    def loss(u):
        return jnp.sum(jacobian_det_displacement(u) ** 2)
    g = jax.grad(loss)(u)
    assert g.shape == u.shape
    assert bool(jnp.all(jnp.isfinite(g)))


def test_jacobian_det_displacement_4d_falls_back_to_linalg_det():
    '''d > 3 routes through jnp.linalg.det.  Sanity check shape only.'''
    from nitrix.geometry import jacobian_det_displacement
    u = jax.random.normal(jax.random.key(0), (4, 4, 4, 4, 4)) * 0.01
    det = jacobian_det_displacement(u)
    assert det.shape == (4, 4, 4, 4)
    assert bool(jnp.all(jnp.isfinite(det)))


def test_resample_preserves_constant():
    const = jnp.full((5, 5, 2), 3.0)
    out = resample(const, (8, 8))
    np.testing.assert_allclose(out, 3.0, atol=1e-10)


def test_resample_preserves_ramp_roundtrip():
    '''A linear ramp is exact under bilinear roundtrip resample.'''
    ramp = jnp.tile(
        jnp.arange(5, dtype=jnp.float64)[:, None, None], (1, 5, 1),
    )
    out = resample(resample(ramp, (8, 8)), (5, 5))
    np.testing.assert_allclose(out, ramp, atol=1e-10)


def test_resample_3d():
    vol = jax.random.normal(jax.random.key(1), (4, 4, 4, 1))
    out = resample(vol, (8, 8, 8))
    assert out.shape == (8, 8, 8, 1)


def test_integrate_velocity_field_zero_is_zero():
    v = jnp.zeros((6, 6, 2))
    phi = integrate_velocity_field(v)
    np.testing.assert_array_equal(phi, jnp.zeros_like(v))


def test_integrate_velocity_field_uniform_translation():
    '''A spatially-uniform velocity should integrate to roughly the
    same uniform displacement -- *including at the boundary* under
    the default mode='nearest'.  Under the old mode='constant'
    default the boundary cells diverged by O(n_steps) voxels.
    '''
    v = jnp.zeros((8, 8, 2)).at[..., 0].set(0.5)
    phi = integrate_velocity_field(v, n_steps=5)
    # Interior values close to 0.5
    np.testing.assert_allclose(
        phi[4, 4, 0], 0.5, atol=1e-3,
    )
    # Boundary values should also be close to 0.5 under edge-replicate.
    # This is the regression test for the JOSA-feedback fix: under the
    # prior mode='constant' default these would have been ~0.
    np.testing.assert_allclose(
        phi[0, 0, 0], 0.5, atol=1e-2,
    )
    np.testing.assert_allclose(
        phi[-1, -1, 0], 0.5, atol=1e-2,
    )


def test_integrate_velocity_field_mode_constant_vs_nearest_differs_at_boundary():
    '''Documenting the behaviour the consumer flagged: the two modes
    disagree at the boundary whenever the integrated flow samples
    OOB.  Construct a velocity with a negative x-component that
    pulls the i=0 boundary OOB during SS integration.
    '''
    H, W = 8, 8
    # v_x = -1 uniformly -> each SS step samples at i - phi_x, which
    # is negative for i=0.  mode='constant' fills with cval=0,
    # mode='nearest' replicates phi_x at i=0.
    v = jnp.zeros((H, W, 2)).at[..., 0].set(-1.0)
    phi_const = integrate_velocity_field(v, n_steps=4, mode='constant')
    phi_nearest = integrate_velocity_field(v, n_steps=4, mode='nearest')
    # Interior values agree.
    np.testing.assert_allclose(
        phi_const[H // 2, W // 2], phi_nearest[H // 2, W // 2], atol=1e-3,
    )
    # Boundary values diverge -- this is the JOSA-feedback bug surface.
    diff = float(jnp.abs(phi_const[0, 0] - phi_nearest[0, 0]).max())
    assert diff > 0.1, f'expected boundary divergence between modes; got {diff}'


def test_center_of_mass_grid_1d():
    w = jnp.array([[0.0, 0.0, 1.0, 0.0]])
    cm = center_of_mass_grid(w)
    np.testing.assert_allclose(cm.flatten(), [0.0, 2.0], atol=1e-10)


def test_center_of_mass_grid_uniform_at_centre():
    w = jnp.ones((5, 5))
    cm = center_of_mass_grid(w)
    np.testing.assert_allclose(cm, [2.0, 2.0], atol=1e-10)


def test_center_of_mass_grid_na_value():
    w = jnp.zeros((5,))
    cm = center_of_mass_grid(w, na_value=-1.0)
    np.testing.assert_array_equal(cm, jnp.array([-1.0]))


def test_displacement_from_reference_grid():
    w = jnp.ones((5,))
    ref = jnp.array([1.0])
    disp = displacement_from_reference_grid(w, ref)
    # centre of mass is 2.0; displacement from 1.0 is 1.0.
    np.testing.assert_allclose(disp, [1.0], atol=1e-10)


# ---------------------------------------------------------------------------
# sphere
# ---------------------------------------------------------------------------


def test_latlong_roundtrip():
    # Avoid the poles where longitude is ill-defined.
    ll = jnp.array([
        [0.0, 0.0],
        [jnp.pi / 4, jnp.pi / 3],
        [-jnp.pi / 4, -jnp.pi / 3],
        [jnp.pi / 6, jnp.pi],
    ])
    xyz = latlong_to_cartesian(ll)
    ll2 = cartesian_to_latlong(xyz)
    np.testing.assert_allclose(ll2, ll, atol=1e-10)


def test_latlong_unit_sphere():
    '''Cartesian coordinates from lat/long should lie on the unit sphere.'''
    ll = jax.random.uniform(
        jax.random.key(0), (16, 2),
        minval=-jnp.pi / 2 + 0.01, maxval=jnp.pi / 2 - 0.01,
    )
    xyz = latlong_to_cartesian(ll, r=1.0)
    norms = jnp.linalg.norm(xyz, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_spherical_geodesic_antipodal():
    north = jnp.array([[0.0, 0.0, 1.0]])
    south = jnp.array([[0.0, 0.0, -1.0]])
    d = spherical_geodesic_distance(north, south)
    np.testing.assert_allclose(d, [[jnp.pi]], atol=1e-10)


def test_spherical_geodesic_identical_points_are_zero():
    p = jnp.array([[1.0, 0.0, 0.0]])
    d = spherical_geodesic_distance(p, p)
    np.testing.assert_allclose(d, [[0.0]], atol=1e-10)


def test_spherical_geodesic_quarter_circle():
    '''Equator to north pole = pi/2 on unit sphere.'''
    equator = jnp.array([[1.0, 0.0, 0.0]])
    pole = jnp.array([[0.0, 0.0, 1.0]])
    d = spherical_geodesic_distance(equator, pole)
    np.testing.assert_allclose(d, [[jnp.pi / 2]], atol=1e-10)


def test_spherical_geodesic_scales_with_radius():
    p = jnp.array([[1.0, 0.0, 0.0]])
    q = jnp.array([[0.0, 1.0, 0.0]])
    d_unit = spherical_geodesic_distance(p, q, r=1.0)
    d_5 = spherical_geodesic_distance(p, q, r=5.0)
    np.testing.assert_allclose(d_5, 5.0 * d_unit, atol=1e-10)


def _spherical_conv_legacy_reference(data, coor, sigma, r=1.0):
    '''All-pairs O(N²) reference matching the *spec-correct* behaviour
    of the legacy ``spherical_conv``: per-row normalised Gaussian
    weights over geodesic distance.
    '''
    n = coor.shape[0]
    d = spherical_geodesic_distance(coor, coor, r=r)
    w = jnp.exp(-0.5 * (d / sigma) ** 2)
    Z = w.sum(axis=-1, keepdims=True)
    w = w / Z
    return w @ data


def test_spherical_conv_preserves_constant():
    pts = jax.random.normal(jax.random.key(1), (32, 3))
    pts = pts / jnp.linalg.norm(pts, axis=-1, keepdims=True)
    const_data = jnp.ones((32, 4)) * 2.5
    out = spherical_conv(const_data, pts, sigma=0.3, neighbourhood=5)
    np.testing.assert_allclose(out, 2.5, atol=1e-10)


def test_spherical_conv_matches_all_pairs_at_large_k():
    '''With k = n, the k-NN adjacency covers everything and the
    re-backed conv equals the all-pairs reference.
    '''
    n = 16
    pts = jax.random.normal(jax.random.key(2), (n, 3))
    pts = pts / jnp.linalg.norm(pts, axis=-1, keepdims=True)
    data = jax.random.normal(jax.random.key(3), (n, 2))
    sigma = 0.5
    got = spherical_conv(data, pts, sigma=sigma, neighbourhood=n)
    ref = _spherical_conv_legacy_reference(data, pts, sigma)
    np.testing.assert_allclose(got, ref, atol=1e-10, rtol=1e-10)


def test_spherical_conv_batched_data():
    n, batch = 12, 3
    pts = jax.random.normal(jax.random.key(4), (n, 3))
    pts = pts / jnp.linalg.norm(pts, axis=-1, keepdims=True)
    data = jax.random.normal(jax.random.key(5), (batch, n, 4))
    out = spherical_conv(data, pts, sigma=0.4, neighbourhood=6)
    assert out.shape == (batch, n, 4)
    # Each batch should equal the per-batch unbatched call.
    for b in range(batch):
        per = spherical_conv(data[b], pts, sigma=0.4, neighbourhood=6)
        np.testing.assert_allclose(out[b], per, atol=1e-10)


def test_spherical_conv_truncate_excludes_far_neighbours():
    '''With ``truncate`` smaller than every neighbour, the weights
    all become zero except the self-weight (which has distance 0),
    so the conv reduces to the identity.
    '''
    n = 16
    pts = jax.random.normal(jax.random.key(6), (n, 3))
    pts = pts / jnp.linalg.norm(pts, axis=-1, keepdims=True)
    data = jax.random.normal(jax.random.key(7), (n, 2))
    # k-NN includes the self at distance 0; truncate < any non-self
    # distance.
    out = spherical_conv(
        data, pts, sigma=0.1, neighbourhood=5, truncate=1e-6,
    )
    np.testing.assert_allclose(out, data, atol=1e-10)


def test_spherical_conv_differentiable():
    n = 12
    pts = jax.random.normal(jax.random.key(8), (n, 3))
    pts = pts / jnp.linalg.norm(pts, axis=-1, keepdims=True)
    data = jax.random.normal(jax.random.key(9), (n, 1))
    def loss(data):
        return spherical_conv(
            data, pts, sigma=0.3, neighbourhood=4,
        ).sum()
    g = jax.grad(loss)(data)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert g.shape == data.shape


# ---------------------------------------------------------------------------
# coords
# ---------------------------------------------------------------------------


def test_center_of_mass_points_1d():
    weight = jnp.array([[0.0, 0.0, 1.0, 0.0]])
    coords = jnp.array([[0.0], [1.0], [2.0], [3.0]])
    cm = center_of_mass_points(weight, coords)
    np.testing.assert_allclose(cm, [[2.0]], atol=1e-10)


def test_center_of_mass_points_uniform():
    weight = jnp.ones((1, 4))
    coords = jnp.array([[0.0], [1.0], [2.0], [3.0]])
    cm = center_of_mass_points(weight, coords)
    np.testing.assert_allclose(cm, [[1.5]], atol=1e-10)


def test_center_of_mass_points_multi_region():
    '''Two regions with mass at different points -> different CMs.'''
    weight = jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    coords = jnp.array([[0.0], [1.0], [2.0], [3.0]])
    cm = center_of_mass_points(weight, coords)
    np.testing.assert_allclose(cm, [[0.0], [3.0]], atol=1e-10)


def test_center_of_mass_points_radius_projection():
    '''With ``radius``, the CM is projected onto a sphere.'''
    weight = jnp.array([[1.0, 1.0]])
    coords = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cm = center_of_mass_points(weight, coords, radius=1.0)
    # CM is (0.5, 0.5, 0); norm = sqrt(0.5); projected -> (1/sqrt(2),
    # 1/sqrt(2), 0).
    expected = jnp.array([[1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2), 0.0]])
    np.testing.assert_allclose(cm, expected, atol=1e-10)


def test_compactness_penalty_zero_for_delta():
    '''A weight concentrated at one point has compactness = 0.'''
    weight = jnp.array([[0.0, 0.0, 1.0, 0.0]])
    coords = jnp.array([[0.0], [1.0], [2.0], [3.0]])
    pen = compactness_penalty(weight, coords)
    np.testing.assert_allclose(pen, [0.0], atol=1e-10)


def test_compactness_penalty_positive_for_spread():
    weight = jnp.array([[1.0, 0.0, 0.0, 1.0]])
    coords = jnp.array([[0.0], [1.0], [2.0], [3.0]])
    pen = compactness_penalty(weight, coords)
    assert float(pen[0]) > 0.0


def test_displacement_from_reference_points():
    weight = jnp.array([[0.0, 0.0, 1.0, 0.0]])
    coords = jnp.array([[0.0], [1.0], [2.0], [3.0]])
    ref = jnp.array([[0.5]])
    disp = displacement_from_reference_points(weight, ref, coords)
    np.testing.assert_allclose(disp, [[1.5]], atol=1e-10)


# ---------------------------------------------------------------------------
# Legacy aliases retained for migration
# ---------------------------------------------------------------------------


def test_legacy_aliases_present():
    '''The legacy names route to the new implementations.'''
    from nitrix.geometry import (
        cmass_coor,
        cmass_regular_grid,
        cmass_reference_displacement_coor,
        cmass_reference_displacement_grid,
        diffuse,
        rescale,
        vec_int,
    )
    # Identity test: each alias must equal the new function.
    assert cmass_coor is center_of_mass_points
    assert cmass_regular_grid is center_of_mass_grid
    assert rescale is resample
    assert vec_int is integrate_velocity_field
    assert diffuse is compactness_penalty
    assert (
        cmass_reference_displacement_coor
        is displacement_from_reference_points
    )
    assert (
        cmass_reference_displacement_grid
        is displacement_from_reference_grid
    )
