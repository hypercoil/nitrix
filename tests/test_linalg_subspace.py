# -*- coding: utf-8 -*-
"""Tests for ``nitrix.linalg.subspace`` (orthogonal Procrustes alignment).

Clean-room from the orthogonal-Procrustes theory (Schoenemann 1966), validated
against the ``scipy.linalg.orthogonal_procrustes`` oracle.  Coverage:

- Oracle parity (the full-orthogonal ``O(p)`` solution == scipy).
- Planted-rotation recovery (``b = a Q`` recovers ``Q``).
- Orthogonality + Frobenius optimality of the solution.
- ``allow_reflection``: ``O(p)`` recovers a planted reflection; ``SO(p)``
  constrains ``det = +1`` (the Kabsch variant) while staying orthogonal.
- The matrix-vMF ``prior`` term: ``None`` == an explicit zero; a strong prior
  pulls the solution toward the prior orientation (the ProMises MAP).
- Batching, ``jit``, ``vmap``, ``grad`` (with ``psi`` reconditioning at a
  fully-degenerate spectrum), and the float32 path.

The eigh-based solve is cuSOLVER-free but, like ``decompose`` / ``pca``, is
``jit``-able only on a healthy-``eigh`` backend; these tests run on the CPU
correctness floor.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
from scipy.linalg import (
    orthogonal_procrustes as scipy_procrustes,  # noqa: E402
)

from nitrix.linalg import orthogonal_procrustes  # noqa: E402
from nitrix.linalg.subspace import _det_sign  # noqa: E402


def _random_orthogonal(
    p: int, *, seed: int, proper: bool = False
) -> np.ndarray:
    """A Haar-ish orthogonal matrix from the QR of a Gaussian (host-side)."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((p, p)))
    q = q * np.sign(np.diag(r))  # fix the QR sign gauge
    if proper and np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


# ---------------------------------------------------------------------------
# Oracle parity + recovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('n,p', [(40, 6), (12, 12), (100, 3)])
def test_matches_scipy_oracle(n: int, p: int) -> None:
    rng = np.random.default_rng(n + p)
    a = rng.standard_normal((n, p))
    b = rng.standard_normal((n, p))
    r_scipy, _ = scipy_procrustes(a, b)
    r = np.asarray(orthogonal_procrustes(jnp.asarray(a), jnp.asarray(b)))
    np.testing.assert_allclose(r, r_scipy, atol=1e-10)


def test_recovers_planted_rotation() -> None:
    rng = np.random.default_rng(1)
    q = _random_orthogonal(5, seed=7, proper=True)
    a = rng.standard_normal((60, 5))
    b = a @ q
    r = np.asarray(orthogonal_procrustes(jnp.asarray(a), jnp.asarray(b)))
    np.testing.assert_allclose(r, q, atol=1e-9)
    assert np.linalg.det(r) > 0


def test_orthogonality_and_optimality() -> None:
    rng = np.random.default_rng(2)
    a = rng.standard_normal((50, 6))
    b = rng.standard_normal((50, 6))
    r = np.asarray(orthogonal_procrustes(jnp.asarray(a), jnp.asarray(b)))
    np.testing.assert_allclose(r.T @ r, np.eye(6), atol=1e-10)
    # No other orthogonal matrix beats it in Frobenius residual.
    best = np.linalg.norm(a @ r - b)
    for s in range(5):
        q = _random_orthogonal(6, seed=100 + s)
        assert np.linalg.norm(a @ q - b) >= best - 1e-9


# ---------------------------------------------------------------------------
# Reflection control
# ---------------------------------------------------------------------------


def test_allow_reflection_recovers_reflection() -> None:
    rng = np.random.default_rng(3)
    a = rng.standard_normal((40, 4))
    ref = np.eye(4)
    ref[0, 0] = -1.0  # a planted improper transform (det = -1)
    b = a @ ref
    r_o = np.asarray(orthogonal_procrustes(jnp.asarray(a), jnp.asarray(b)))
    np.testing.assert_allclose(r_o, ref, atol=1e-9)
    assert np.linalg.det(r_o) < 0


def test_so_constrains_to_proper_rotation() -> None:
    rng = np.random.default_rng(4)
    a = rng.standard_normal((40, 4))
    ref = np.eye(4)
    ref[0, 0] = -1.0
    b = a @ ref
    r_so = np.asarray(
        orthogonal_procrustes(
            jnp.asarray(a), jnp.asarray(b), allow_reflection=False
        )
    )
    np.testing.assert_allclose(r_so.T @ r_so, np.eye(4), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(r_so), 1.0, atol=1e-9)


@pytest.mark.parametrize('p', [3, 5, 8])
def test_det_sign_matches_numpy(p: int) -> None:
    rng = np.random.default_rng(p)
    m = rng.standard_normal((p, p))
    assert float(_det_sign(jnp.asarray(m))) == np.sign(np.linalg.det(m))
    batch = rng.standard_normal((4, p, p))
    np.testing.assert_array_equal(
        np.asarray(_det_sign(jnp.asarray(batch))),
        np.sign(np.linalg.det(batch)),
    )


# ---------------------------------------------------------------------------
# The matrix-vMF prior term (ProMises MAP)
# ---------------------------------------------------------------------------


def test_prior_none_equals_zero() -> None:
    rng = np.random.default_rng(5)
    a = rng.standard_normal((30, 4))
    b = rng.standard_normal((30, 4))
    r_none = np.asarray(orthogonal_procrustes(jnp.asarray(a), jnp.asarray(b)))
    r_zero = np.asarray(
        orthogonal_procrustes(
            jnp.asarray(a), jnp.asarray(b), prior=jnp.zeros((4, 4))
        )
    )
    np.testing.assert_allclose(r_none, r_zero, atol=1e-12)


def test_prior_pulls_toward_orientation() -> None:
    rng = np.random.default_rng(6)
    q = _random_orthogonal(4, seed=9, proper=True)
    a = rng.standard_normal((50, 4))
    b = a @ q  # data want R ~= q
    # A strong prior toward the identity should pull R away from q toward I.
    prior = 200.0 * np.eye(4)
    r0 = np.asarray(orthogonal_procrustes(jnp.asarray(a), jnp.asarray(b)))
    r1 = np.asarray(
        orthogonal_procrustes(
            jnp.asarray(a), jnp.asarray(b), prior=jnp.asarray(prior)
        )
    )
    eye = np.eye(4)
    assert np.linalg.norm(r1 - eye) < np.linalg.norm(r0 - eye)
    np.testing.assert_allclose(r1.T @ r1, eye, atol=1e-9)


# ---------------------------------------------------------------------------
# Transforms: batch / jit / vmap / grad / dtype
# ---------------------------------------------------------------------------


def test_batched_equals_loop() -> None:
    rng = np.random.default_rng(7)
    a = rng.standard_normal((5, 20, 4))
    b = rng.standard_normal((5, 20, 4))
    r_batch = np.asarray(orthogonal_procrustes(jnp.asarray(a), jnp.asarray(b)))
    for i in range(5):
        r_i = np.asarray(
            orthogonal_procrustes(jnp.asarray(a[i]), jnp.asarray(b[i]))
        )
        np.testing.assert_allclose(r_batch[i], r_i, atol=1e-10)


def test_jit_and_vmap() -> None:
    rng = np.random.default_rng(8)
    a = rng.standard_normal((5, 20, 4))
    b = rng.standard_normal((5, 20, 4))
    eager = np.asarray(orthogonal_procrustes(jnp.asarray(a), jnp.asarray(b)))
    jitted = np.asarray(
        jax.jit(jax.vmap(orthogonal_procrustes))(
            jnp.asarray(a), jnp.asarray(b)
        )
    )
    np.testing.assert_allclose(jitted, eager, atol=1e-10)


def test_grad_finite_at_degenerate_spectrum() -> None:
    # Orthonormal columns => A^T A = I => C = I, fully degenerate singular
    # spectrum: the reverse-mode eigh-VJP is undefined without reconditioning.
    a = jnp.asarray(_random_orthogonal(5, seed=11)[:, :3])  # orthonormal cols
    b = a

    def loss(x: jax.Array) -> jax.Array:
        r = orthogonal_procrustes(x, b, psi=1e-3, key=jax.random.PRNGKey(0))
        return (r**2).sum()

    g = jax.grad(loss)(a)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_float32_orthogonal() -> None:
    rng = np.random.default_rng(12)
    a = jnp.asarray(rng.standard_normal((40, 6)), dtype=jnp.float32)
    b = jnp.asarray(rng.standard_normal((40, 6)), dtype=jnp.float32)
    r = orthogonal_procrustes(a, b)
    assert r.dtype == jnp.float32
    r_np = np.asarray(r)
    np.testing.assert_allclose(r_np.T @ r_np, np.eye(6), atol=1e-4)
