# -*- coding: utf-8 -*-
"""Tests for the Grassmann subspace-geometry primitives in ``nitrix.linalg``.

- ``image_basis`` matches ``scipy.linalg.orth`` (numerical rank + range, i.e.
  the projector) and is orthonormal; the static-``rank`` path is jit-clean.
- ``subspace_angles`` matches ``scipy.linalg.subspace_angles`` across the
  small-angle / large-angle regimes *and* the ``0`` / ``pi/2`` boundaries
  (the numerically-stable arcsin/arccos split), with **finite gradients at both
  boundaries** -- the property a naive ``arccos`` lacks.
- jit / vmap clean.

Run on the CPU correctness floor (the eigh solve is cuSOLVER-free but, like
``decompose`` / ``pca``, ``jit``-able only on a healthy-``eigh`` backend).
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
from scipy.linalg import orth as scipy_orth  # noqa: E402
from scipy.linalg import subspace_angles as scipy_subspace_angles  # noqa: E402

from nitrix.linalg import image_basis, subspace_angles  # noqa: E402

# --------------------------------------------------------------------------- #
# image_basis                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize('m,n', [(20, 5), (5, 20), (10, 10), (30, 3)])
def test_image_basis_matches_scipy_orth(m: int, n: int) -> None:
    rng = np.random.default_rng(m * 100 + n)
    a = rng.standard_normal((m, n))
    q = np.asarray(image_basis(jnp.asarray(a)))
    qs = scipy_orth(a)
    assert q.shape[1] == qs.shape[1]  # same numerical rank
    k = q.shape[1]
    np.testing.assert_allclose(q.T @ q, np.eye(k), atol=1e-10)  # orthonormal
    # Same range <=> same orthogonal projector (basis itself is gauge-dependent).
    np.testing.assert_allclose(q @ q.T, qs @ qs.T, atol=1e-10)


def test_image_basis_rank_deficient() -> None:
    rng = np.random.default_rng(1)
    a = rng.standard_normal((20, 6))
    a[:, 5] = a[:, 0] + a[:, 1]  # planted rank 5
    q = np.asarray(image_basis(jnp.asarray(a)))
    assert q.shape[1] == 5 == scipy_orth(a).shape[1]


def test_image_basis_static_rank_is_jit_clean() -> None:
    rng = np.random.default_rng(2)
    a = jnp.asarray(rng.standard_normal((20, 5)))
    q = jax.jit(lambda x: image_basis(x, rank=5))(a)
    assert q.shape == (20, 5)
    np.testing.assert_allclose(
        np.asarray(q).T @ np.asarray(q), np.eye(5), atol=1e-10
    )
    # Range recovered: projector matches the full-rank basis.
    qs = scipy_orth(np.asarray(a))
    np.testing.assert_allclose(
        np.asarray(q) @ np.asarray(q).T, qs @ qs.T, atol=1e-9
    )


def test_image_basis_recovers_planted_subspace() -> None:
    # Columns spanning a known 3-D subspace of R^10: the projector is exact.
    rng = np.random.default_rng(3)
    basis = np.linalg.qr(rng.standard_normal((10, 3)))[0]  # orthonormal (10,3)
    coeffs = rng.standard_normal((3, 7))
    a = basis @ coeffs  # (10, 7), range = span(basis)
    q = np.asarray(image_basis(jnp.asarray(a)))
    assert q.shape[1] == 3
    np.testing.assert_allclose(q @ q.T, basis @ basis.T, atol=1e-9)


# --------------------------------------------------------------------------- #
# subspace_angles                                                             #
# --------------------------------------------------------------------------- #


def _cmp_angles(a: np.ndarray, b: np.ndarray, atol: float = 1e-7) -> None:
    got = np.sort(np.asarray(subspace_angles(jnp.asarray(a), jnp.asarray(b))))
    ref = np.sort(scipy_subspace_angles(a, b))
    np.testing.assert_allclose(got, ref, atol=atol)


def test_subspace_angles_matches_scipy_random() -> None:
    rng = np.random.default_rng(10)
    a = rng.standard_normal((30, 4))
    b = rng.standard_normal((30, 3))
    _cmp_angles(a, b)


def test_subspace_angles_orthogonal_is_half_pi() -> None:
    rng = np.random.default_rng(11)
    h = np.linalg.qr(rng.standard_normal((30, 30)))[0]
    a, b = h[:, :4], h[:, 4:7]  # mutually orthogonal columns
    angles = np.asarray(subspace_angles(jnp.asarray(a), jnp.asarray(b)))
    np.testing.assert_allclose(angles, np.pi / 2, atol=1e-9)
    _cmp_angles(a, b)


def test_subspace_angles_nested_is_zero() -> None:
    rng = np.random.default_rng(12)
    a = rng.standard_normal((30, 4))
    b = a[:, :2]  # a subspace of span(a)
    angles = np.asarray(subspace_angles(jnp.asarray(a), jnp.asarray(b)))
    np.testing.assert_allclose(angles, 0.0, atol=1e-7)
    _cmp_angles(a, b)


def test_subspace_angles_small_angles_are_accurate() -> None:
    # The arcsin branch: small angles are resolved to machine precision, where a
    # naive arccos(cosine) would lose ~half the digits.
    rng = np.random.default_rng(13)
    qa = scipy_orth(rng.standard_normal((30, 4)))
    b = qa + 1e-3 * rng.standard_normal((30, 4))  # a tiny perturbation
    _cmp_angles(qa, b, atol=1e-10)


def test_subspace_angles_mixed_regimes() -> None:
    rng = np.random.default_rng(14)
    h = np.linalg.qr(rng.standard_normal((30, 30)))[0]
    a = h[:, :3]
    b = np.c_[
        h[:, 0], h[:, 3], h[:, 4]
    ]  # one shared direction, two orthogonal
    _cmp_angles(a, b, atol=1e-6)


def test_subspace_angles_gradients_finite_near_boundaries() -> None:
    # The stable split keeps gradients finite as theta -> 0 (cosine -> 1, the
    # arccos singular point) and theta -> pi/2 (cosine -> 0, the arcsin singular
    # point).  Generic (non-orthonormal) spanning matrices are used so the test
    # exercises the arccos/arcsin split, not the separate eigh degeneracy that a
    # perfectly-orthonormal input (Gram = I) would introduce.
    rng = np.random.default_rng(15)
    a = jnp.asarray(rng.standard_normal((20, 3)))
    b = jnp.asarray(rng.standard_normal((20, 3)))
    g = jax.grad(lambda x: subspace_angles(x, b).sum())(a)
    assert bool(jnp.all(jnp.isfinite(g)))  # generic, mid-range

    # Near theta = 0: b spans (almost) the same subspace as a.
    b0 = jnp.asarray(
        np.asarray(a) @ rng.standard_normal((3, 3))
        + 1e-4 * rng.standard_normal((20, 3))
    )
    g0 = jax.grad(lambda x: subspace_angles(x, b0).sum())(a)
    assert bool(jnp.all(jnp.isfinite(g0)))

    # Near theta = pi/2: b lies (almost) in the orthogonal complement of a.
    proj = np.asarray(a) @ np.linalg.pinv(np.asarray(a))
    b90 = jnp.asarray(
        (np.eye(20) - proj) @ rng.standard_normal((20, 3))
        + 1e-3 * rng.standard_normal((20, 3))
    )
    g90 = jax.grad(lambda x: subspace_angles(x, b90).sum())(a)
    assert bool(jnp.all(jnp.isfinite(g90)))


def test_subspace_angles_jit_and_vmap() -> None:
    rng = np.random.default_rng(16)
    a = jnp.asarray(rng.standard_normal((20, 3)))
    b = jnp.asarray(rng.standard_normal((20, 3)))
    eager = np.asarray(subspace_angles(a, b))
    jitted = np.asarray(jax.jit(subspace_angles)(a, b))
    np.testing.assert_allclose(jitted, eager, atol=1e-10)

    ab = jnp.asarray(rng.standard_normal((4, 20, 3)))
    bb = jnp.asarray(rng.standard_normal((4, 20, 3)))
    batched = jax.vmap(subspace_angles)(ab, bb)
    looped = jnp.stack([subspace_angles(ab[i], bb[i]) for i in range(4)])
    np.testing.assert_allclose(
        np.asarray(batched), np.asarray(looped), atol=1e-10
    )


def test_subspace_angles_grad_stable_at_orthonormal_inputs() -> None:
    # Exactly-orthonormal inputs give X^T X = I, a fully repeated spectrum that
    # makes an eigh-based orthonormalisation's VJP blow up.  The matmul-only
    # Loewdin orthonormalisation keeps the gradient finite -- including the
    # theta = pi/2 case that a naive path NaNs on.
    rng = np.random.default_rng(17)
    h = jnp.asarray(np.linalg.qr(rng.standard_normal((30, 30)))[0])
    a, b = h[:, :4], h[:, 4:8]  # orthonormal columns, mutually orthogonal
    g = jax.grad(lambda x: subspace_angles(x, b).sum())(a)
    assert bool(jnp.all(jnp.isfinite(g)))
    # theta = 0 (a compared with an orthonormal subset of itself).
    g0 = jax.grad(lambda x: subspace_angles(x, a[:, :2]).sum())(a)
    assert bool(jnp.all(jnp.isfinite(g0)))


def test_subspace_angles_grad_stable_at_repeated_angles() -> None:
    # Genuinely repeated principal angles (all equal to theta): only the
    # eigenvalues of the reduced Grams feed the angles, so the VJP stays finite.
    rng = np.random.default_rng(18)
    theta = 0.5
    c, s = np.cos(theta), np.sin(theta)
    eye6 = np.eye(6)
    # span{e0,e1,e2} vs its rotation by theta into {e3,e4,e5}: 3 equal angles.
    a_basis, b_basis = eye6[:, :3], c * eye6[:, :3] + s * eye6[:, 3:6]
    a = jnp.asarray(
        a_basis @ rng.standard_normal((3, 3))
    )  # non-orthonormal span
    b = jnp.asarray(b_basis @ rng.standard_normal((3, 3)))
    angles = np.asarray(subspace_angles(a, b))
    np.testing.assert_allclose(angles, theta, atol=1e-8)  # all three equal
    g = jax.grad(lambda x: subspace_angles(x, b).sum())(a)
    assert bool(jnp.all(jnp.isfinite(g)))
