# -*- coding: utf-8 -*-
"""Tests for the spherical harmonic transform (geometry.harmonics).

The transform is exact on the Gauss--Legendre grid for band-limited fields, so it
is pinned by: forward of a pure ``Y_lm`` grid gives a unit coefficient (checked
against ``scipy.special.sph_harm_y``); the coefficient / field round-trips are
identities; and Parseval's theorem holds between spatial and spectral energy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.geometry import (  # noqa: E402
    real_sht_forward,
    real_sht_inverse,
    sht_forward,
    sht_grid,
    sht_inverse,
    sht_rotate,
    sht_rotation_matrix,
)


def _random_bandlimited_coeffs(L, seed=0):
    """Random complex coefficients respecting the |m| <= l triangular mask."""
    rng = np.random.default_rng(seed)
    c = np.zeros((L + 1, 2 * L + 1), complex)
    for ell in range(L + 1):
        for m in range(-ell, ell + 1):
            c[ell, m + L] = rng.standard_normal() + 1j * rng.standard_normal()
    return jnp.asarray(c)


def test_grid_shape():
    g = sht_grid(8)
    assert g.colatitude.shape == (9,)  # L + 1
    assert g.longitude.shape == (17,)  # 2L + 1
    assert 0.0 <= float(g.colatitude.min())
    assert float(g.colatitude.max()) <= np.pi


def test_forward_of_pure_harmonic_is_unit_coefficient():
    sp = pytest.importorskip('scipy.special')
    L = 8
    g = sht_grid(L)
    th, ph = np.meshgrid(
        np.asarray(g.colatitude), np.asarray(g.longitude), indexing='ij'
    )
    for ell, m in [(3, 2), (6, -4), (0, 0), (7, 7)]:
        f = jnp.asarray(sp.sph_harm_y(ell, m, th, ph))
        c = sht_forward(f, band_limit=L)
        off = np.asarray(c).copy()
        off[ell, m + L] = 0.0
        np.testing.assert_allclose(
            abs(complex(c[ell, m + L])), 1.0, atol=1e-10
        )
        assert np.max(np.abs(off)) < 1e-10


def test_coefficient_round_trip():
    L = 8
    c0 = _random_bandlimited_coeffs(L)
    c1 = sht_forward(sht_inverse(c0), band_limit=L)
    np.testing.assert_allclose(np.asarray(c1), np.asarray(c0), atol=1e-11)


def test_field_round_trip_is_band_limited_identity():
    L = 10
    field = sht_inverse(_random_bandlimited_coeffs(L, seed=1))  # band-limited
    recon = sht_inverse(sht_forward(field, band_limit=L))
    np.testing.assert_allclose(
        np.asarray(recon), np.asarray(field), atol=1e-10
    )


def test_coefficient_triangular_mask():
    L = 6
    c = sht_forward(
        sht_inverse(_random_bandlimited_coeffs(L, 2)), band_limit=L
    )
    c = np.asarray(c)
    for ell in range(L + 1):
        for m in range(-L, L + 1):
            if abs(m) > ell:
                assert abs(c[ell, m + L]) < 1e-12


def test_parseval():
    L = 8
    c0 = _random_bandlimited_coeffs(L, seed=3)
    field = sht_inverse(c0)
    weights = np.polynomial.legendre.leggauss(L + 1)[1]
    dphi = 2 * np.pi / (2 * L + 1)
    spatial = float(
        np.sum(weights[:, None] * np.abs(np.asarray(field)) ** 2) * dphi
    )
    spectral = float(jnp.sum(jnp.abs(c0) ** 2))
    np.testing.assert_allclose(spatial, spectral, rtol=1e-10)


def test_real_field_has_hermitian_coefficients():
    # A real field's coefficients satisfy c[l,-m] = (-1)^m conj(c[l,m]); an
    # inverse of such coefficients is real.
    L = 6
    rng = np.random.default_rng(5)
    field = jnp.asarray(rng.standard_normal((L + 1, 2 * L + 1)))
    c = np.asarray(sht_forward(field, band_limit=L))
    for ell in range(L + 1):
        for m in range(1, ell + 1):
            np.testing.assert_allclose(
                c[ell, -m + L], (-1) ** m * np.conj(c[ell, m + L]), atol=1e-10
            )
    recon = sht_inverse(jnp.asarray(c))
    assert float(jnp.max(jnp.abs(jnp.imag(recon)))) < 1e-10


def test_batching_over_leading_dims():
    L = 8
    c = _random_bandlimited_coeffs(L)
    fields = jnp.stack(
        [sht_inverse(c), sht_inverse(2.0 * c)]
    )  # (2, n_lat, n_lon)
    coeffs = sht_forward(fields, band_limit=L)
    assert coeffs.shape == (2, L + 1, 2 * L + 1)
    np.testing.assert_allclose(
        np.asarray(coeffs[0]),
        np.asarray(sht_forward(fields[0], band_limit=L)),
        atol=1e-11,
    )


def test_jit_and_grad():
    L = 8
    field = jnp.real(sht_inverse(_random_bandlimited_coeffs(L, seed=6)))
    c = jax.jit(lambda f: sht_forward(f, band_limit=L))(field)
    assert bool(jnp.all(jnp.isfinite(jnp.abs(c))))
    g = jax.grad(
        lambda f: jnp.sum(jnp.abs(sht_forward(f, band_limit=L)) ** 2)
    )(field)
    assert bool(jnp.all(jnp.isfinite(g)))


# --- real spherical harmonics ------------------------------------------------


def _real_bandlimited_coeffs(L, seed=0):
    rng = np.random.default_rng(seed)
    c = np.zeros((L + 1, 2 * L + 1))
    for ell in range(L + 1):
        for m in range(-ell, ell + 1):
            c[ell, m + L] = rng.standard_normal()
    return jnp.asarray(c)


def test_real_sht_forward_of_pure_real_harmonic():
    sp = pytest.importorskip('scipy.special')
    L = 8
    g = sht_grid(L)
    th, ph = np.meshgrid(
        np.asarray(g.colatitude), np.asarray(g.longitude), indexing='ij'
    )

    def real_y(ell, m):
        if m == 0:
            return np.real(sp.sph_harm_y(ell, 0, th, ph))
        if m > 0:
            return (
                np.sqrt(2) * (-1) ** m * np.real(sp.sph_harm_y(ell, m, th, ph))
            )
        return (
            np.sqrt(2)
            * (-1) ** abs(m)
            * np.imag(sp.sph_harm_y(ell, abs(m), th, ph))
        )

    for ell, m in [(3, 2), (5, -4), (4, 0), (7, 7)]:
        c = real_sht_forward(jnp.asarray(real_y(ell, m)), band_limit=L)
        assert not jnp.iscomplexobj(c)  # real coefficients
        off = np.asarray(c).copy()
        off[ell, m + L] = 0.0
        np.testing.assert_allclose(float(c[ell, m + L]), 1.0, atol=1e-10)
        assert np.max(np.abs(off)) < 1e-10


def test_real_sht_round_trip():
    L = 8
    c0 = _real_bandlimited_coeffs(L)
    c1 = real_sht_forward(real_sht_inverse(c0), band_limit=L)
    np.testing.assert_allclose(np.asarray(c1), np.asarray(c0), atol=1e-11)


def test_real_sht_inverse_is_real():
    L = 6
    field = real_sht_inverse(_real_bandlimited_coeffs(L, seed=2))
    assert not jnp.iscomplexobj(field)


# --- Driscoll-Healy equiangular grid -----------------------------------------


def test_dh_grid_shape():
    g = sht_grid(8, grid='driscoll_healy')
    assert g.colatitude.shape == (18,)  # 2(L+1)
    assert g.longitude.shape == (18,)


def test_dh_round_trip_matches_gauss_accuracy():
    L = 8
    c0 = _random_bandlimited_coeffs(L)
    field = sht_inverse(c0, grid='driscoll_healy')
    c1 = sht_forward(field, band_limit=L, grid='driscoll_healy')
    np.testing.assert_allclose(np.asarray(c1), np.asarray(c0), atol=1e-11)


def test_dh_real_round_trip():
    L = 8
    c0 = _real_bandlimited_coeffs(L)
    field = real_sht_inverse(c0, grid='driscoll_healy')
    assert not jnp.iscomplexobj(field)
    c1 = real_sht_forward(field, band_limit=L, grid='driscoll_healy')
    np.testing.assert_allclose(np.asarray(c1), np.asarray(c0), atol=1e-11)


def test_invalid_grid_raises():
    with pytest.raises(ValueError):
        sht_grid(4, grid='healpix')


# --- Wigner-D coefficient rotation -------------------------------------------


def _rotation(seed):
    # a rotation matrix via a QR of a random matrix (det +1)
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return jnp.asarray(q)


def test_rotate_by_identity_is_a_no_op():
    L = 8
    c = _random_bandlimited_coeffs(L)
    np.testing.assert_array_equal(
        np.asarray(sht_rotate(c, jnp.eye(3))), np.asarray(c)
    )


def test_rotation_preserves_per_degree_norm():
    L = 8
    c = _random_bandlimited_coeffs(L)
    cr = sht_rotate(c, _rotation(1))
    for ell in range(L + 1):
        got = float(jnp.sum(jnp.abs(cr[ell]) ** 2))
        want = float(jnp.sum(jnp.abs(c[ell]) ** 2))
        np.testing.assert_allclose(got, want, atol=1e-10)


def test_rotation_composition():
    L = 6
    c = _random_bandlimited_coeffs(L)
    r1, r2 = _rotation(1), _rotation(2)
    lhs = sht_rotate(sht_rotate(c, r1), r2)
    rhs = sht_rotate(c, r2 @ r1)
    np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), atol=1e-10)


def test_rotation_matches_direct_field_rotation():
    sp = pytest.importorskip('scipy.special')
    L = 8
    c = np.asarray(_random_bandlimited_coeffs(L))
    r = _rotation(3)
    c_rot = sht_rotate(jnp.asarray(c), r)
    field = np.asarray(sht_inverse(c_rot))
    # direct: f_rot(x) = f(R^{-1} x) = sum c_lm Y_lm(R^{-1} x) at the grid points
    g = sht_grid(L)
    th, ph = np.meshgrid(
        np.asarray(g.colatitude), np.asarray(g.longitude), indexing='ij'
    )
    xyz = np.stack(
        [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1
    )
    xr = xyz @ np.asarray(r)  # R^{-1} x  (rows)
    thr = np.arccos(np.clip(xr[..., 2], -1, 1))
    phr = np.arctan2(xr[..., 1], xr[..., 0])
    direct = np.zeros_like(th, complex)
    for ell in range(L + 1):
        for m in range(-ell, ell + 1):
            direct += c[ell, m + L] * sp.sph_harm_y(ell, m, thr, phr)
    np.testing.assert_allclose(field, direct, atol=1e-9)


def test_rotation_matrix_is_block_diagonal():
    L = 5
    d = sht_rotation_matrix(_rotation(4), band_limit=L)
    assert d.shape == (L + 1, 2 * L + 1, 2 * L + 1)
    for ell in range(L + 1):
        block = np.asarray(d[ell])
        # entries outside the [-ell, ell] block are zero
        mask = np.ones((2 * L + 1, 2 * L + 1), bool)
        mask[L - ell : L + ell + 1, L - ell : L + ell + 1] = False
        assert np.max(np.abs(block[mask]), initial=0.0) < 1e-12


def test_rotation_jit_and_grad():
    L = 6
    c = _random_bandlimited_coeffs(L)
    r = _rotation(5)
    d = jax.jit(lambda r: sht_rotation_matrix(r, band_limit=L))(r)
    assert bool(jnp.all(jnp.isfinite(jnp.abs(d))))
    g = jax.grad(lambda r: jnp.sum(jnp.abs(sht_rotate(c, r)) ** 2))(r)
    assert bool(jnp.all(jnp.isfinite(g)))
