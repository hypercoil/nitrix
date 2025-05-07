# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for geometric operations.
"""
import pytest
import numpy as np
import jax.numpy as jnp
from functools import partial
from numpyro.distributions import Normal
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics.pairwise import haversine_distances
from communities.utilities import (
    modularity_matrix as modularity_matrix_ref,
    modularity as modularity_ref
)
from nitrix.functional.geom import (
    cmass_regular_grid,
    cmass_coor,
    cmass_reference_displacement_coor,
    cmass_reference_displacement_grid,
    spherical_geodesic,
    spherical_conv,
    euclidean_conv,
    sphere_to_normals,
    sphere_to_latlong,
    kernel_gaussian,
    diffuse,
    modularity_matrix,
    relaxed_modularity,
)


#TODO: Unit tests still needed for
# - "centres of mass" in spherical coordinates
# - regularisers: `cmass_reference_displacement` and `diffuse`
# - case with positive and negative weights in the adjacency matrix
# - correctness of nonassociative block modularity
# - correctness of non-normalised modularity and coaffiliation
# - correctness of graph measures on directed or bipartite graphs


@pytest.fixture(scope='module')
def X():
    return np.array([
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])

@pytest.fixture(scope='module')
def X_all():
    return np.array([1.5, 1.5])

@pytest.fixture(scope='module')
def X_0():
    return np.array([1, 2, 1.5, 1.5])

@pytest.fixture(scope='module')
def X_1():
    return np.array([1.5, 1.5, 1, 2])

@pytest.fixture(scope='module')
def Y():
    Y = np.random.rand(5, 3, 4, 4)
    Y = (Y > 0.5).astype(float)
    return Y

@pytest.fixture(scope='module')
def Xcoor():
    coor = np.arange(4).reshape(1, 4)
    return np.stack([
        np.tile(coor, (4, 1)),
        np.tile(coor.reshape(4, 1), (1, 4))
    ])

@pytest.fixture(scope='module')
def Ycoor(Y):
    return np.stack([
        np.broadcast_to(np.arange(5).reshape(-1, 1, 1, 1), Y.shape),
        np.broadcast_to(np.arange(3).reshape(1, -1, 1, 1), Y.shape),
        np.broadcast_to(np.arange(4).reshape(1, 1, -1, 1), Y.shape),
        np.broadcast_to(np.arange(4).reshape(1, 1, 1, -1), Y.shape),
    ])

@pytest.fixture(scope='module')
def data():
    n = 5
    c = 6
    data = np.zeros((n, c))
    data[:n, :n] = np.eye(n)
    data[:, c:] = np.random.rand(n, c - n)
    return data

@pytest.fixture(scope='module')
def dist_euc():
    n = 5
    return np.linspace(0, 1, n)

@pytest.fixture(scope='module')
def coor_euc(dist_euc):
    n = dist_euc.shape[-1]
    coor = np.zeros((n, 3))
    coor[:, 1] = dist_euc
    return coor

@pytest.fixture(scope='module')
def coor_sph():
    return np.array([
        [np.pi / 2, 0],          # north pole
        [0, 0],                  # equator / prime meridian junction
        [0, -np.pi / 2],         # equator, 90 W
        [-np.pi / 4, np.pi],     # 45 S, 180th meridian
        [-np.pi / 2, 0]          # south pole
    ])

@pytest.fixture(scope='module')
def coor_sph_norm(coor_sph):
    return np.array([
        [0, 0, 1],                  # north pole
        [1, 0, 0],                  # equator / prime meridian junction
        [0, -1, 0],                 # equator, 90 W
        [-2 ** 0.5 / 2, 0, -2 ** 0.5 / 2],
        [0, 0, -1]                  # south pole
    ])

@pytest.fixture(scope='module')
def truncated():
    n = 5
    truncated = np.ones((n, n))
    truncated[0, 3] = 0
    truncated[0, 4] = 0
    truncated[1, 3] = 0
    return np.minimum(truncated, truncated.T)

@pytest.fixture(scope='module')
def coor_sph_rand():
    coor_sph_rand = np.random.rand(5, 3)
    coor_sph_rand /= np.linalg.norm(coor_sph_rand, axis=0, ord=2)
    return coor_sph_rand

@pytest.fixture(scope='module')
def A():
    A = np.random.rand(3, 20, 20)
    A += A.swapaxes(-1, -2)
    return A

@pytest.fixture(scope='module')
def aff():
    aff = np.random.randint(0, 4, 20)
    return aff

@pytest.fixture(scope='module')
def comms(aff):
    return [np.where(aff==c)[0] for c in np.unique(aff)]

@pytest.fixture(scope='module')
def C(aff):
    return np.eye(4)[aff]

@pytest.fixture(scope='module')
def L():
    return np.random.rand(4, 4)


def test_alias_cmass(X, Y):
    from nitrix.functional.geom import cmass
    assert np.allclose(cmass_regular_grid(X), cmass(X))
    assert np.allclose(cmass_regular_grid(Y), cmass(Y))


def test_cmass_negatives(X, Y):
    out = cmass_regular_grid(X, [0])
    ref = cmass_regular_grid(X, [-2])
    assert np.all(out == ref)
    out = cmass_regular_grid(Y, [-1, -3])
    ref = cmass_regular_grid(Y, [3, 1])
    assert np.allclose(out, ref)
    out = cmass_regular_grid(Y, [0, -1])
    ref = cmass_regular_grid(Y, [0, 3])
    assert np.allclose(out, ref)


def test_cmass_values(X, X_all, X_0, X_1):
    out = cmass_regular_grid(X)
    ref = X_all
    assert np.allclose(out, ref)
    out = cmass_regular_grid(X, [0]).squeeze()
    ref = X_0
    assert np.allclose(out, ref)
    out = cmass_regular_grid(X, [1]).squeeze()
    ref = X_1
    assert np.allclose(out, ref)


def test_cmass_dim(Y):
    out = cmass_regular_grid(Y, [-1, -3])
    assert out.shape == (5, 4, 2)
    out = cmass_regular_grid(Y, [-2])
    assert out.shape == (5, 3, 4, 1)
    out = cmass_regular_grid(Y, [0, -3, -2])
    assert out.shape == (4, 3)


def test_cmass_coor(X, Y, Xcoor, Ycoor):
    out = cmass_coor(X.reshape(1, -1), Xcoor.reshape(2, -1))
    ref = cmass_regular_grid(X)
    assert np.allclose(out, ref)

    out = cmass_coor(Y.reshape(1, -1), Ycoor.reshape(4, -1))
    ref = cmass_regular_grid(Y)
    assert np.allclose(out.squeeze(), ref.squeeze())

    out = cmass_coor(Y.reshape(1, -1), Ycoor.reshape(4, -1), radius=1)
    assert np.allclose(jnp.linalg.norm(out, axis=-2), 1)


def test_cmass_empty():
    X = np.zeros((30, 10))
    out = cmass_regular_grid(X)
    assert jnp.all(jnp.isnan(out))
    out = cmass_regular_grid(X, na_rm=0.)
    assert jnp.allclose(out, 0.)


def test_cmass_equivalence(X, Xcoor, Y, Ycoor):
    out = cmass_coor(X.reshape(1, -1), Xcoor.reshape(2, -1))
    ref = cmass_regular_grid(X)
    assert np.allclose(out, ref)

    out = cmass_coor(Y.reshape(1, -1), Ycoor.reshape(4, -1))
    ref = cmass_regular_grid(Y)
    assert np.allclose(out.squeeze(), ref.squeeze())


def test_cmass_displacements(X, Xcoor, Y, Ycoor):
    reference = cmass_regular_grid(X)
    out = cmass_reference_displacement_grid(X, reference)
    assert np.allclose(out, 0)
    X, Xcoor = X.reshape(1, -1), Xcoor.reshape(2, -1)
    reference = cmass_coor(X, Xcoor)
    out = cmass_reference_displacement_coor(X, reference, Xcoor)
    assert np.allclose(out, 0)

    reference = cmass_regular_grid(Y)
    out = cmass_reference_displacement_grid(Y, reference)
    assert np.allclose(out, 0)
    Y, Ycoor = Y.reshape(1, -1), Ycoor.reshape(4, -1)
    reference = cmass_coor(Y, Ycoor)
    out = cmass_reference_displacement_coor(Y, reference, Ycoor)
    assert np.allclose(out, 0)


def test_gauss_kernel(dist_euc):
    scale = 0.5
    n = Normal(loc=0, scale=scale)
    ker_ref = lambda x: jnp.exp(n.log_prob(x))
    ker = partial(kernel_gaussian, scale=scale)
    out = ker(dist_euc) / ker_ref(dist_euc)
    assert np.allclose(out, out.max())


def test_spherical_conversion(coor_sph, coor_sph_norm):
    normals = sphere_to_normals(coor_sph)
    assert np.allclose(normals, coor_sph_norm, atol=1e-6)
    restore = sphere_to_latlong(normals)
    # Discard meaningless longitudes at poles
    restore = restore.at[0, 1].set(0)
    restore = restore.at[4, 1].set(0)
    # Account for equivalence between pi and -pi longitude
    restore_equiv = restore
    restore_equiv = restore_equiv.at[3, 1].set(restore_equiv[3, 1] * -1)
    assert (np.allclose(restore, coor_sph, atol=1e-6) or
            np.allclose(restore_equiv, coor_sph, atol=1e-6))
    # And check that it's still the same forward after multiple
    # applications.
    normals = sphere_to_normals(
        sphere_to_latlong(sphere_to_normals(restore))
    )
    assert np.allclose(normals, coor_sph_norm, atol=1e-6)


def test_spherical_geodesic(coor_sph, coor_sph_rand):
    normals = sphere_to_normals(coor_sph)
    out = spherical_geodesic(normals)
    ref = haversine_distances(coor_sph)
    assert np.allclose(out, ref, atol=1e-6)
    latlong = sphere_to_latlong(coor_sph_rand)
    out = spherical_geodesic(coor_sph_rand)
    ref = haversine_distances(latlong)
    assert np.allclose(out, ref, atol=1e-6)

    invalid_input = np.eye(2)
    with pytest.raises(ValueError):
        spherical_geodesic(invalid_input)
    with pytest.raises(ValueError):
        spherical_geodesic(X=normals, Y=invalid_input)


def test_spatial_convolution(data, coor_euc):
    n = data.shape[-2]
    scale = 0.5
    out = euclidean_conv(
        data=data,
        coor=coor_euc,
        scale=scale
    )
    ref = gaussian_filter1d(
        input=data,
        sigma=scale * (n - 1),
        axis=0,
        mode='constant',
        truncate=16,
    )
    out = out / out.max()
    ref = ref / ref.max()
    assert np.allclose(out, ref, atol=1e-4)


def test_spherical_convolution(data, coor_sph, coor_sph_rand, truncated):
    """
    WARNING: Correctness is not tested.
    """
    scale = 3
    n = data.shape[-2]
    out = spherical_conv(
        data=data,
        coor=sphere_to_normals(coor_sph),
        scale=scale
    )
    out = spherical_conv(
        data=data,
        coor=coor_sph_rand,
        scale=scale
    )
    # truncation test
    out = spherical_conv(
        data=data,
        coor=sphere_to_normals(coor_sph),
        scale=scale,
        truncate=(np.pi / 2)
    )
    out = (out[:, :n] == 0).astype(float)
    assert np.allclose(
        out + truncated,
        np.ones((n, n))
    )


def test_compactness():
    coor = jnp.stack(
        jnp.meshgrid(jnp.arange(100), jnp.arange(100))
    )
    centroid = jnp.asarray((50, 50))[:, None, None]
    small = jnp.linalg.norm(coor - centroid, axis=0) < 10
    large = jnp.linalg.norm(coor - centroid, axis=0) < 50
    diff_small = diffuse(small.reshape(1, -1), coor.reshape(2, -1))
    diff_large = diffuse(large.reshape(1, -1), coor.reshape(2, -1))
    assert diff_small < diff_large

    # This is totally invalid (not on a sphere), but is here for
    # testing that the radius argument is properly handled.
    coor = jnp.stack(
        jnp.meshgrid(jnp.arange(100), jnp.arange(100), jnp.arange(100))
    )
    centroid = jnp.asarray((30, 30, 30))[:, None, None, None]
    small = jnp.linalg.norm(coor - centroid, axis=0) < 10
    large = jnp.linalg.norm(coor - centroid, axis=0) < 50
    diff_small = diffuse(small.reshape(1, -1), coor.reshape(3, -1), radius=1)
    diff_large = diffuse(large.reshape(1, -1), coor.reshape(3, -1), radius=1)
    assert diff_small < diff_large


def test_modularity_matrix(A):
    out = modularity_matrix(A, normalise_modularity=True)
    ref = np.stack([modularity_matrix_ref(x) for x in A])
    assert np.allclose(out, ref)


def test_modularity(A, C, comms):
    out = relaxed_modularity(
        A,
        C,
        exclude_diag=True,
        directed=False,
    )
    ref = np.stack([
        modularity_ref(modularity_matrix_ref(x), comms)
        for x in A
    ])
    assert np.allclose(out, ref)
    ref = relaxed_modularity(
        A,
        C,
        C,
        exclude_diag=True,
        directed=False,
    )
    assert np.allclose(out, ref)
    ref = relaxed_modularity(
        -A,
        C,
        exclude_diag=True,
        directed=False,
        sign='-',
    )
    assert np.allclose(out, ref)


def test_nonassociative_block(A, C, L):
    #TODO: this test only checks that the call does not crash
    out = relaxed_modularity(
        A,
        C,
        L=L,
        exclude_diag=False,
        normalise_modularity=False,
        normalise_coaffiliation=False,
        directed=True,
    ) / 2
