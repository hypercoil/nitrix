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
    same uniform displacement (with mild interior loss from
    repeated interpolation).
    '''
    v = jnp.zeros((8, 8, 2)).at[..., 0].set(0.5)
    phi = integrate_velocity_field(v, n_steps=5)
    # Interior values close to 0.5
    np.testing.assert_allclose(
        phi[4, 4, 0], 0.5, atol=1e-3,
    )


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
