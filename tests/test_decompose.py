# -*- coding: utf-8 -*-
"""Tests for ``nitrix.linalg.decompose`` (randomized sketch SVD).

Validated against a dense ``numpy`` SVD reference (computed on the CPU, so the
oracle itself never touches the cuSOLVER-affected GPU stack): leading singular
values, orthonormality of the factors, exact rank-``k`` reconstruction, exact
recovery of a genuinely low-rank matrix, key determinism, and shape/validation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.linalg import randomized_svd


def _decaying_spectrum(m, n, r, seed, noise=0.01):
    """An (m, n) matrix with a known geometric top-``r`` spectrum + noise."""
    rng = np.random.default_rng(seed)
    U = np.linalg.qr(rng.standard_normal((m, r)))[0]
    V = np.linalg.qr(rng.standard_normal((n, r)))[0]
    sv = np.geomspace(50.0, 0.5, r)
    A = U @ np.diag(sv) @ V.T + rng.standard_normal((m, n)) * noise
    return A


def test_singular_values_match_dense():
    A = jnp.asarray(_decaying_spectrum(600, 200, 12, seed=0))
    k = 8
    _, s, _ = randomized_svd(A, k, key=jax.random.PRNGKey(0), n_power=4)
    sv_ref = np.linalg.svd(np.asarray(A), compute_uv=False)[:k]
    np.testing.assert_allclose(np.asarray(s), sv_ref, rtol=1e-6)


def test_factors_orthonormal_and_reconstruct():
    A = jnp.asarray(_decaying_spectrum(400, 150, 10, seed=1))
    k = 6
    U, s, Vt = randomized_svd(A, k, key=jax.random.PRNGKey(1), n_power=4)
    assert U.shape == (400, k) and s.shape == (k,) and Vt.shape == (k, 150)
    np.testing.assert_allclose(np.asarray(U.T @ U), np.eye(k), atol=1e-9)
    np.testing.assert_allclose(np.asarray(Vt @ Vt.T), np.eye(k), atol=1e-9)
    # reconstruction equals the exact rank-k truncation
    Uf, Sf, Vtf = np.linalg.svd(np.asarray(A), full_matrices=False)
    Ak = Uf[:, :k] @ np.diag(Sf[:k]) @ Vtf[:k]
    rec = np.asarray(U) * np.asarray(s) @ np.asarray(Vt)
    assert float(np.linalg.norm(rec - Ak) / np.linalg.norm(Ak)) < 1e-6
    # singular values are non-negative and descending
    assert bool(np.all(np.diff(np.asarray(s)) <= 1e-9))
    assert bool(np.all(np.asarray(s) >= 0.0))


def test_exact_recovery_of_low_rank_matrix():
    """When ``A`` has exact rank ``r`` and ``k = r``, the sketch is exact."""
    rng = np.random.default_rng(2)
    m, n, r = 300, 120, 5
    A = rng.standard_normal((m, r)) @ rng.standard_normal((r, n))
    A = jnp.asarray(A)
    U, s, Vt = randomized_svd(A, r, key=jax.random.PRNGKey(2), n_power=2)
    rec = np.asarray(U) * np.asarray(s) @ np.asarray(Vt)
    assert (
        float(
            np.linalg.norm(rec - np.asarray(A)) / np.linalg.norm(np.asarray(A))
        )
        < 1e-8
    )


def test_deterministic_given_key():
    A = jnp.asarray(_decaying_spectrum(200, 80, 8, seed=3))
    a = randomized_svd(A, 5, key=jax.random.PRNGKey(7), n_power=2)
    b = randomized_svd(A, 5, key=jax.random.PRNGKey(7), n_power=2)
    np.testing.assert_array_equal(np.asarray(a[1]), np.asarray(b[1]))
    np.testing.assert_array_equal(np.asarray(a[0]), np.asarray(b[0]))
    # a different key gives a (slightly) different sketch realisation
    c = randomized_svd(A, 5, key=jax.random.PRNGKey(8), n_power=2)
    assert not bool(np.array_equal(np.asarray(a[0]), np.asarray(c[0])))


@pytest.mark.parametrize('shape', [(500, 120), (120, 500)])  # tall and wide
def test_tall_and_wide(shape):
    m, n = shape
    A = jnp.asarray(_decaying_spectrum(m, n, 10, seed=m + n))
    k = 7
    U, s, Vt = randomized_svd(A, k, key=jax.random.PRNGKey(0), n_power=4)
    assert U.shape == (m, k) and Vt.shape == (k, n)
    sv_ref = np.linalg.svd(np.asarray(A), compute_uv=False)[:k]
    np.testing.assert_allclose(np.asarray(s), sv_ref, rtol=1e-6)


def test_power_iteration_improves_slow_spectrum():
    """On a slowly decaying spectrum, power iterations sharpen accuracy."""
    rng = np.random.default_rng(9)
    m, n, r = 500, 200, 30
    U = np.linalg.qr(rng.standard_normal((m, r)))[0]
    V = np.linalg.qr(rng.standard_normal((n, r)))[0]
    sv = np.linspace(10.0, 1.0, r)  # slow linear decay
    A = jnp.asarray(U @ np.diag(sv) @ V.T)
    sv_ref = np.linalg.svd(np.asarray(A), compute_uv=False)[:5]

    _, s0, _ = randomized_svd(A, 5, key=jax.random.PRNGKey(0), n_power=0)
    _, s4, _ = randomized_svd(A, 5, key=jax.random.PRNGKey(0), n_power=4)
    err0 = float(np.max(np.abs(np.asarray(s0) - sv_ref) / sv_ref))
    err4 = float(np.max(np.abs(np.asarray(s4) - sv_ref) / sv_ref))
    assert err4 < err0
    assert err4 < 1e-3


def test_k_out_of_range_raises():
    A = jnp.asarray(_decaying_spectrum(100, 40, 5, seed=0))
    with pytest.raises(ValueError):
        randomized_svd(A, 41, key=jax.random.PRNGKey(0))
    with pytest.raises(ValueError):
        randomized_svd(A, 0, key=jax.random.PRNGKey(0))
