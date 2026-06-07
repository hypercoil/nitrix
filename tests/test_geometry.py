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
    Interpolator,
    Lanczos,
    Linear,
    MultiLabel,
    NearestNeighbour,
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


# ---------------------------------------------------------------------------
# interpolation-method dispatcher (Stage 1: Linear / NearestNeighbour)
# ---------------------------------------------------------------------------


def test_interpolator_records_satisfy_protocol():
    '''The concrete records structurally conform to ``Interpolator``.'''
    assert isinstance(Linear(), Interpolator)
    assert isinstance(NearestNeighbour(), Interpolator)


def test_interpolator_records_are_frozen_and_hashable():
    '''Records are static config: frozen, hashable, equality-by-value.'''
    assert Linear() == Linear()
    assert hash(Linear()) == hash(Linear())
    assert {Linear(), NearestNeighbour(), Linear()} == {
        Linear(), NearestNeighbour(),
    }
    with pytest.raises((AttributeError, TypeError)):
        Linear().differentiable_in_values = False  # type: ignore[misc]


def test_differentiability_flags():
    '''The per-method differentiability contract is self-described.'''
    assert Linear.differentiable_in_values
    assert Linear.differentiable_in_coords
    assert NearestNeighbour.differentiable_in_values
    assert not NearestNeighbour.differentiable_in_coords


def test_resample_default_method_is_linear():
    '''The default path equals an explicit ``method=Linear()``.'''
    vol = jax.random.normal(jax.random.key(3), (5, 5, 1))
    np.testing.assert_array_equal(
        resample(vol, (8, 8)), resample(vol, (8, 8), method=Linear()),
    )


def test_resample_nearest_preserves_label_values():
    '''Nearest-neighbour output is a subset of the input label set.'''
    seg = jnp.array(
        [[0, 0, 1, 1, 2],
         [0, 0, 1, 1, 2],
         [3, 3, 4, 4, 5],
         [3, 3, 4, 4, 5],
         [6, 6, 7, 7, 8]],
        dtype=jnp.float64,
    )[..., None]
    out = resample(seg, (9, 9), method=NearestNeighbour())
    allowed = set(int(v) for v in jnp.unique(seg))
    present = set(int(v) for v in jnp.unique(out))
    assert present.issubset(allowed)


def test_resample_nearest_matches_map_coordinates_order0():
    '''NN parity against the ``map_coordinates`` order-0 oracle.'''
    import jax.scipy.ndimage as jsp_ndi

    img = jax.random.normal(jax.random.key(4), (6, 7))
    axes = [
        jnp.linspace(0.0, s - 1, t, dtype=jnp.float64)
        for s, t in zip((6, 7), (9, 5))
    ]
    grids = jnp.meshgrid(*axes, indexing='ij')
    coords = jnp.stack([g.reshape(-1) for g in grids], axis=0)
    oracle = jsp_ndi.map_coordinates(
        img, coords, order=0, mode='constant', cval=0.0,
    ).reshape(9, 5)
    out = resample(img[..., None], (9, 5), method=NearestNeighbour())[..., 0]
    np.testing.assert_array_equal(out, oracle)


def test_spatial_transform_nearest_identity_is_identity():
    '''A nearest-neighbour identity warp is the identity map.'''
    seg = jax.random.randint(
        jax.random.key(5), (5, 5, 1), 0, 4,
    ).astype(jnp.float64)
    out = spatial_transform(
        seg, identity_grid((5, 5)), method=NearestNeighbour(),
    )
    np.testing.assert_array_equal(out, seg)


def test_nearest_coordinate_gradient_is_zero():
    '''NN is coordinate-flat: ``differentiable_in_coords = False`` in fact.

    The round-to-nearest is piecewise-constant in the coordinates, so the
    gradient of any output functional w.r.t. the deformation is zero
    almost everywhere.
    '''
    img = jax.random.normal(jax.random.key(6), (5, 5, 1))
    grid = identity_grid((5, 5))
    g = jax.grad(
        lambda d: spatial_transform(
            img, d, method=NearestNeighbour(),
        ).sum()
    )(grid)
    np.testing.assert_array_equal(g, jnp.zeros_like(grid))


def test_resample_nearest_jit_and_vmap():
    '''The NN path is jit- and vmap-clean.'''
    batch = jax.random.normal(jax.random.key(7), (3, 5, 5, 1))
    jitted = jax.jit(
        lambda x: resample(x, (8, 8), method=NearestNeighbour())
    )
    out = jax.vmap(jitted)(batch)
    assert out.shape == (3, 8, 8, 1)


# ---------------------------------------------------------------------------
# Stage 2: explicit separable-gather engine matches the map_coordinates oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'mode', ['constant', 'nearest', 'wrap', 'mirror', 'reflect'],
)
@pytest.mark.parametrize('ndim', [1, 2, 3])
def test_separable_gather_matches_map_coordinates(mode, ndim):
    '''The explicit gather reproduces ``map_coordinates`` to ~ULP.

    Exercises ``_separable_gather`` **directly** (not via ``.sample``,
    which on CPU routes to the ``map_coordinates`` engine) so the gather
    is validated on any platform: across every boundary mode and
    1-/2-/3-D, the order-1 (linear taps) and order-0 (nearest taps)
    gathers match the retained oracle, including out-of-bounds
    coordinates that exercise the boundary folds.  This is the B7
    engine-swap guard rail.
    '''
    from nitrix.geometry._interpolate import (
        Linear as _Linear,
        NearestNeighbour as _NN,
        _map_coordinates_sample,
        _separable_gather,
    )

    rng = np.random.default_rng(11 + ndim)
    spatial = tuple(int(s) for s in rng.integers(4, 7, size=ndim))
    img = jnp.asarray(rng.standard_normal(spatial + (2,)))
    hi = max(spatial) + 1.5
    coords = jnp.asarray(rng.uniform(-2.5, hi, size=(40, ndim)))

    ref_lin = _map_coordinates_sample(img, coords, order=1, mode=mode, cval=-7.0)
    got_lin = _separable_gather(
        img, coords, tap_rule=_Linear()._axis_taps_weights,
        mode=mode, cval=-7.0,
    )
    np.testing.assert_allclose(got_lin, ref_lin, atol=1e-12)

    ref_nn = _map_coordinates_sample(img, coords, order=0, mode=mode, cval=-7.0)
    got_nn = _separable_gather(
        img, coords, tap_rule=_NN()._axis_taps_weights,
        mode=mode, cval=-7.0,
    )
    np.testing.assert_array_equal(got_nn, ref_nn)


def test_separable_gather_constant_fills_cval_per_tap():
    '''``constant`` blends out-of-range taps as ``cval`` (per scipy).

    A sample at fractional position ``-0.3`` blends the (out-of-range)
    tap ``-1`` at ``cval`` with the in-range tap ``0`` -- it is *not* a
    whole-sample ``cval`` fill.
    '''
    from nitrix.geometry._interpolate import (
        Linear as _Linear,
        _separable_gather,
    )

    img = jnp.arange(5.0)[..., None]
    out = _separable_gather(
        img, jnp.array([[-0.3]]),
        tap_rule=_Linear()._axis_taps_weights, mode='constant', cval=-9.0,
    )
    # 0.3 * cval + 0.7 * img[0] = 0.3 * -9 + 0 = -2.7
    np.testing.assert_allclose(out[0, 0], -2.7, atol=1e-12)


def test_linear_gather_is_differentiable_in_coords():
    '''Linear keeps a non-trivial coordinate gradient (vs NN's zero).'''
    img = jax.random.normal(jax.random.key(8), (6, 6, 1))
    grid = identity_grid((6, 6)) + 0.3
    g = jax.grad(
        lambda d: spatial_transform(img, d, method=Linear()).sum()
    )(grid)
    assert bool(jnp.any(g != 0.0))
    assert bool(jnp.all(jnp.isfinite(g)))


@pytest.mark.parametrize('method', [Linear(), NearestNeighbour()])
def test_platform_engines_agree_through_sample(method, monkeypatch):
    '''Both platform engines yield the same ``.sample`` result.

    ``Linear`` / ``NearestNeighbour`` pick the explicit gather on GPU and
    ``map_coordinates`` on CPU; forcing each branch (via the platform
    probe) and comparing proves the public dispatch path is
    platform-invariant -- and covers the GPU branch on a CPU host.
    '''
    img = jax.random.normal(jax.random.key(9), (6, 7, 2))
    coords = identity_grid((6, 7)) + 0.4

    monkeypatch.setattr(
        'nitrix.geometry._interpolate.default_backend_is_gpu', lambda: False,
    )
    cpu_engine = method.sample(img, coords, mode='reflect', cval=0.0)
    monkeypatch.setattr(
        'nitrix.geometry._interpolate.default_backend_is_gpu', lambda: True,
    )
    gpu_engine = method.sample(img, coords, mode='reflect', cval=0.0)
    np.testing.assert_allclose(gpu_engine, cpu_engine, atol=1e-12)


# ---------------------------------------------------------------------------
# Stage 3: Lanczos windowed sinc
# ---------------------------------------------------------------------------


def test_lanczos_preserves_constant_edge_replicate():
    '''A renormalised Lanczos kernel is a partition of unity.

    Under edge-replicate (``mode='nearest'``) every tap is a real
    neighbour, so a constant resamples to the same constant everywhere
    (the renormalisation makes the per-axis weights sum to 1 exactly).
    '''
    const = jnp.full((10, 10, 1), 2.5)
    out = resample(const, (16, 16), method=Lanczos(), mode='nearest')
    np.testing.assert_allclose(out, 2.5, atol=1e-10)


def test_lanczos_constant_mode_interior_preserved():
    '''``constant`` mode preserves the interior; only the border fills.'''
    const = jnp.full((12, 12, 1), 2.5)
    out = resample(const, (20, 20), method=Lanczos(order=3), mode='constant')
    # The radius-3 stencil only reaches cval within ~3 voxels of the edge.
    np.testing.assert_allclose(out[5:-5, 5:-5], 2.5, atol=1e-10)


def test_lanczos_reproduces_grid_samples():
    '''Resampling onto the original grid is the identity (L_a(integer)=0).'''
    img = jax.random.normal(jax.random.key(20), (7, 7, 2))
    out = resample(img, (7, 7), method=Lanczos(order=3))
    np.testing.assert_allclose(out, img, atol=1e-10)


def test_lanczos_higher_fidelity_than_linear():
    '''On a band-limited signal Lanczos beats linear under a roundtrip.'''
    x = jnp.linspace(0.0, 2.0 * jnp.pi, 64)
    signal = jnp.sin(3.0 * x)[:, None]

    def roundtrip(method):
        up = resample(signal, (128,), method=method)
        return resample(up, (64,), method=method)

    err_linear = float(jnp.mean((roundtrip(Linear()) - signal) ** 2))
    err_lanczos = float(jnp.mean((roundtrip(Lanczos()) - signal) ** 2))
    assert err_lanczos < err_linear / 10.0


def test_lanczos_weights_match_hand_computed():
    '''Numerical oracle: a 1-D sample equals the normalised sinc sum.'''
    img = jnp.arange(10.0)[..., None]
    p = 4.3  # interior sample, full radius-3 stencil in bounds
    a = 3
    taps = np.arange(int(np.floor(p)) - a + 1, int(np.floor(p)) + a + 1)
    x = p - taps

    def sinc(t):
        return np.sinc(t)  # numpy sinc is the normalised sinc too

    w = sinc(x) * sinc(x / a)
    w = w / w.sum()
    expected = float((w * np.asarray(img)[taps, 0]).sum())
    got = spatial_transform(
        img, jnp.array([[p]]), method=Lanczos(order=3), mode='nearest',
    )
    np.testing.assert_allclose(float(got[0, 0]), expected, atol=1e-10)


def test_lanczos_resample_matches_scattered_gather():
    '''resample (1-D passes) == spatial_transform (dense gather) for Lanczos.

    Validates the separable resample optimisation against the general
    scattered-coordinate gather on the same align-corners grid.
    '''
    img = jax.random.normal(jax.random.key(21), (6, 7, 2))
    target = (9, 5)
    axes = [
        jnp.linspace(0.0, s - 1, t, dtype=img.dtype)
        for s, t in zip((6, 7), target)
    ]
    coords = jnp.stack(jnp.meshgrid(*axes, indexing='ij'), axis=-1)
    via_resample = resample(img, target, method=Lanczos(order=3))
    via_gather = spatial_transform(img, coords, method=Lanczos(order=3))
    np.testing.assert_allclose(via_resample, via_gather, atol=1e-10)


def test_lanczos_differentiable_values_and_coords():
    '''Lanczos is smooth: non-trivial, finite gradients both ways.'''
    img = jax.random.normal(jax.random.key(22), (7, 7, 1))
    gv = jax.grad(
        lambda im: resample(im, (10, 10), method=Lanczos()).sum()
    )(img)
    gc = jax.grad(
        lambda d: spatial_transform(img, d, method=Lanczos()).sum()
    )(identity_grid((7, 7)) + 0.3)
    for g in (gv, gc):
        assert bool(jnp.all(jnp.isfinite(g)))
        assert bool(jnp.any(g != 0.0))


def test_lanczos_jit_and_vmap():
    '''The Lanczos path is jit- and vmap-clean (order as a static field).'''
    batch = jax.random.normal(jax.random.key(23), (3, 6, 6, 1))
    f = jax.jit(lambda im: resample(im, (9, 9), method=Lanczos(order=4)))
    assert jax.vmap(f)(batch).shape == (3, 9, 9, 1)


def test_lanczos_order_validation():
    '''A non-positive or non-integer order is rejected at construction.'''
    with pytest.raises(ValueError):
        Lanczos(order=0)
    with pytest.raises(ValueError):
        Lanczos(order=-2)
    with pytest.raises(ValueError):
        Lanczos(order=2.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Stage 4: ANTs MultiLabel
# ---------------------------------------------------------------------------


_SEG = jnp.array(
    [[0, 0, 1, 1, 2],
     [0, 0, 1, 1, 2],
     [3, 3, 4, 4, 2],
     [3, 3, 4, 4, 2],
     [3, 3, 0, 0, 0]],
    dtype=jnp.float64,
)[..., None]
_LABELS = (0, 1, 2, 3, 4)


def test_multilabel_output_is_subset_of_labels():
    '''No invented values: every output is one of the input labels.'''
    out = resample(_SEG, (11, 9), method=MultiLabel(labels=_LABELS))
    present = set(int(v) for v in jnp.unique(out))
    assert present.issubset(set(_LABELS))


def test_multilabel_identity_warp_is_identity():
    out = spatial_transform(
        _SEG, identity_grid((5, 5)), method=MultiLabel(labels=_LABELS),
    )
    np.testing.assert_array_equal(out, _SEG)


def test_multilabel_is_renumber_invariant():
    '''Adding a constant offset to the labels offsets the output the same.'''
    out = resample(_SEG, (9, 9), method=MultiLabel(labels=_LABELS))
    shifted = resample(
        _SEG + 10.0, (9, 9),
        method=MultiLabel(labels=tuple(label + 10 for label in _LABELS)),
    )
    np.testing.assert_array_equal(shifted, out + 10.0)


def test_multilabel_wide_inner_preserves_thin_structure():
    '''A wider inner kernel anti-aliases: a thin label survives downsampling.

    A 1-voxel stripe of label 2 is dropped by nearest-neighbour and by a
    narrow ``Linear`` inner, but a ``Lanczos`` inner area-weights it
    enough to win the arg-max at some output voxels.
    '''
    yy, xx = np.mgrid[0:16, 0:16]
    seg = np.where(xx + yy < 16, 0, 1).astype(float)
    seg[7, :] = 2  # thin horizontal stripe
    seg = jnp.asarray(seg)[..., None]
    labels = (0, 1, 2)

    nn = resample(seg, (8, 8), method=NearestNeighbour())
    ml_lanczos = resample(
        seg, (8, 8), method=MultiLabel(labels=labels, inner=Lanczos(order=3)),
    )
    assert 2 not in set(int(v) for v in jnp.unique(nn))      # NN drops it
    assert 2 in set(int(v) for v in jnp.unique(ml_lanczos))  # MultiLabel keeps it


def test_multilabel_is_non_differentiable_without_error():
    '''The arg-max is a hard selection: gradients are zero, not an error.'''
    gv = jax.grad(
        lambda im: resample(im, (8, 8), method=MultiLabel(labels=_LABELS)).sum()
    )(_SEG)
    np.testing.assert_array_equal(gv, jnp.zeros_like(_SEG))
    assert not MultiLabel.differentiable_in_values
    assert not MultiLabel.differentiable_in_coords


def test_multilabel_out_of_bounds_resolves_to_first_label():
    '''A fully out-of-support sample resolves to ``labels[0]``.'''
    far = jnp.full((1, 1, 2), 999.0)  # way outside the image
    out = spatial_transform(
        _SEG, far, method=MultiLabel(labels=(7, 1, 2, 3, 4)), mode='constant',
    )
    np.testing.assert_array_equal(out, jnp.full_like(out, 7.0))


def test_multilabel_jit_with_static_labels():
    '''Static label set makes the op jit-clean.'''
    f = jax.jit(lambda im: resample(im, (7, 7), method=MultiLabel(labels=_LABELS)))
    assert f(_SEG).shape == (7, 7, 1)


def test_multilabel_validation():
    with pytest.raises(ValueError):
        MultiLabel(labels=())
    with pytest.raises(ValueError):
        MultiLabel(labels=(1, 1, 2))


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
