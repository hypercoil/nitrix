# -*- coding: utf-8 -*-
"""Tests for ``nitrix.linalg.spd`` (SPD manifold operations).

The SPEC §4.5 stability rewrite: legacy ``hypercoil.functional.symmap``
underflowed at small eigenvalues (silent ``NaN`` -> ``0`` via
``fill_nans=True``).  The green-field implementation clips
eigenvalues at the rank-truncation threshold and surfaces ``NaN``
when the input is genuinely indefinite.

Coverage:

- Roundtrip identities: ``symexp(symlog(X)) == X``,
  ``symsqrt(X) @ symsqrt(X) == X``, ``X @ sympower(X, -1) == I``.
- Ill-conditioned input: ``symlog`` produces finite output via
  the auto-clipping path.
- ``tangent_project_spd`` / ``cone_project_spd`` round-trip.
- ``mean_log_euclidean`` invariants: identity at a single
  input; commutes with scalar multiplication.
- Differentiability through ``symlog`` with and without
  ``psi`` reconditioning.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.linalg._solver import safe_eigh
from nitrix.linalg.spd import (
    cone_project_spd,
    mean_euclidean,
    mean_log_euclidean,
    symexp,
    symlog,
    symmap,
    sympower,
    symsqrt,
    tangent_project_spd,
)


def _spd_matrix(n=5, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return jnp.asarray(A @ A.T + 0.5 * np.eye(n))


def _orthogonal_via_safe_eigh(n=5, seed=0):
    """Build an orthogonal n x n matrix via safe_eigh of a random symm.

    Avoids ``jax.random.orthogonal`` which uses GPU QR (broken on
    the test runner's cuSolver).
    """
    rng = np.random.default_rng(seed)
    S = jnp.asarray(
        (rng.standard_normal((n, n)) + rng.standard_normal((n, n)).T) / 2
    )
    _, U = safe_eigh(S)
    return U


# ---------------------------------------------------------------------------
# Roundtrip identities
# ---------------------------------------------------------------------------


def test_symlog_symexp_roundtrip():
    X = _spd_matrix()
    np.testing.assert_allclose(symexp(symlog(X)), X, atol=1e-12)


def test_symsqrt_squared_recovers_input():
    X = _spd_matrix()
    sq = symsqrt(X)
    np.testing.assert_allclose(sq @ sq, X, atol=1e-12)


def test_sympower_neg_one_is_inverse():
    X = _spd_matrix()
    Xi = sympower(X, power=-1)
    # Verify via X @ Xi == I (avoids jnp.linalg.inv which is cuSolver-broken).
    np.testing.assert_allclose(X @ Xi, jnp.eye(5), atol=1e-12)


def test_sympower_half_equals_symsqrt():
    X = _spd_matrix()
    np.testing.assert_allclose(
        sympower(X, power=0.5),
        symsqrt(X),
        atol=1e-13,
    )


def test_sympower_identity_at_power_one():
    X = _spd_matrix()
    np.testing.assert_allclose(sympower(X, power=1.0), X, atol=1e-13)


# ---------------------------------------------------------------------------
# Ill-conditioned input (the SPEC §4.5 stability concern)
# ---------------------------------------------------------------------------


def test_symlog_ill_conditioned_is_finite():
    """symlog on a 20-order-of-magnitude eigenvalue spread should
    produce finite output via the auto-clipping path.  The legacy
    code's silent NaN -> 0 substitution is gone; if eigvalues drop
    below the clip threshold, log produces the (finite) clipped
    value's log.

    Note: 20 orders of magnitude is at the limit of fp64
    reconstruction; spreads beyond that exceed precision.
    """
    U = _orthogonal_via_safe_eigh()
    ev = jnp.array([1e-12, 1e-8, 1e-4, 1.0, 1e4])
    X_ill = U @ jnp.diag(ev) @ U.T
    out = symlog(X_ill)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_symlog_clip_floor_uses_eps_threshold():
    """With ``eigvalue_clip='auto'``, the floor is
    ``max(|L|) * d * eps``.  Eigenvalues at or below this threshold
    are clamped, so ``symlog`` returns ``log(threshold)`` for those
    -- much larger than the unclipped ``log(1e-15) ~ -34.5``.
    """
    U = _orthogonal_via_safe_eigh()
    ev = jnp.array([1e-15, 1e-10, 1e-5, 1.0, 1e5])
    X_ill = U @ jnp.diag(ev) @ U.T
    # max|L| = 1e5; d = 5; eps_f64 ~ 2.22e-16; threshold ~ 1.1e-10
    out_clip = symlog(X_ill, eigvalue_clip='auto')
    out_eigvals, _ = safe_eigh(out_clip)
    threshold = float(jnp.max(jnp.abs(ev)) * 5 * jnp.finfo(ev.dtype).eps)
    expected_min = float(jnp.log(threshold))
    # The smallest eigenvalues of the original spectrum (1e-15, 1e-10)
    # both clip to threshold; log(threshold) is the smallest eigenvalue
    # of symlog(X_ill).
    np.testing.assert_allclose(
        float(out_eigvals.min()),
        expected_min,
        atol=1e-8,
    )


def test_symlog_no_clip_produces_very_negative():
    """With ``eigvalue_clip='none'``, no protection.  Tiny eigvals
    produce very-negative log entries.

    Note: we use ``1e-12`` as the smallest eigenvalue (not ``1e-15``)
    because reconstructing ``U diag(L) U.T`` with eigenvalue spread
    > 20 orders of magnitude exceeds fp64 precision -- the tiny
    eigenvalues are dominated by reconstruction noise.
    """
    U = _orthogonal_via_safe_eigh()
    ev = jnp.array([1e-12, 1e-8, 1e-4, 1.0, 1e4])
    X_ill = U @ jnp.diag(ev) @ U.T
    out = symlog(X_ill, eigvalue_clip='none')
    out_eigvals, _ = safe_eigh(out)
    # Smallest log eigenvalue should be ~ log(1e-12) = -27.6.
    # Allow slop for reconstruction noise.
    assert float(out_eigvals.min()) < -20


def test_symlog_explicit_float_clip():
    """Explicit float threshold overrides the auto-clip heuristic."""
    U = _orthogonal_via_safe_eigh()
    ev = jnp.array([1e-15, 1.0, 2.0, 3.0, 4.0])
    X_ill = U @ jnp.diag(ev) @ U.T
    out = symlog(X_ill, eigvalue_clip=1e-3)
    out_eigvals, _ = safe_eigh(out)
    # Smallest log eigenvalue should be log(1e-3) ~ -6.9
    np.testing.assert_allclose(
        float(out_eigvals.min()),
        float(jnp.log(1e-3)),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Tangent / cone projections
# ---------------------------------------------------------------------------


def test_tangent_cone_roundtrip():
    X = _spd_matrix(seed=0)
    R = _spd_matrix(seed=1)
    T = tangent_project_spd(X, R)
    back = cone_project_spd(T, R)
    np.testing.assert_allclose(back, X, atol=1e-10)


def test_tangent_project_at_self_is_log_zero_at_identity():
    """``tangent_project_spd(X, X)`` should be the zero matrix (X is
    the tangent-space "origin" at itself).
    """
    X = _spd_matrix()
    T = tangent_project_spd(X, X)
    assert float(jnp.abs(T).max()) < 1e-10


# ---------------------------------------------------------------------------
# Means
# ---------------------------------------------------------------------------


def test_mean_log_euclidean_single_input_is_identity():
    """The mean of a singleton batch is the batch element itself."""
    X = _spd_matrix()
    batch = X[None, ...]  # (1, 5, 5)
    mu = mean_log_euclidean(batch, axis=0)
    np.testing.assert_allclose(mu, X, atol=1e-12)


def test_mean_log_euclidean_is_spd():
    """The log-Euclidean mean of SPD matrices is itself SPD."""
    batch = jnp.stack([_spd_matrix(seed=i) for i in range(3)])
    mu = mean_log_euclidean(batch, axis=0)
    eigvals, _ = safe_eigh(mu)
    assert bool(jnp.all(eigvals > 0))


def test_mean_log_euclidean_scaling():
    """``mean(log(c * X_i)) = mean(log(X_i)) + log(c) I``, so
    ``exp(...)`` scales by ``c``.
    """
    batch = jnp.stack([_spd_matrix(seed=i) for i in range(3)])
    c = 3.0
    mu = mean_log_euclidean(batch, axis=0)
    mu_scaled = mean_log_euclidean(c * batch, axis=0)
    np.testing.assert_allclose(mu_scaled, c * mu, atol=1e-10)


def test_mean_euclidean_is_plain_mean():
    batch = jnp.stack([_spd_matrix(seed=i) for i in range(4)])
    mu = mean_euclidean(batch, axis=0)
    np.testing.assert_allclose(mu, batch.mean(0), atol=1e-13)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


def test_symlog_differentiable():
    X = _spd_matrix()

    def loss(X):
        return jnp.trace(symlog(X) ** 2)

    g = jax.grad(loss)(X)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_symlog_with_psi_reconditioning_handles_degeneracy():
    """Near-degenerate spectrum: without psi, eigh-VJP can blow up;
    with psi > 0, the gradient is stable.
    """
    U = _orthogonal_via_safe_eigh()
    ev = jnp.array([1.0 + 1e-12, 1.0, 2.0, 3.0, 4.0])  # near-degenerate
    X_deg = U @ jnp.diag(ev) @ U.T

    def loss_psi(X):
        return jnp.trace(symlog(X, psi=1e-4, key=jax.random.key(0)) ** 2)

    g_psi = jax.grad(loss_psi)(X_deg)
    assert bool(jnp.all(jnp.isfinite(g_psi)))


def test_symmap_generic_with_custom_fn():
    """``symmap`` with a user-supplied function (e.g. exp(- x))
    matches the manual eigh-recompose path.
    """
    X = _spd_matrix()
    out = symmap(X, fn=lambda x: jnp.exp(-x))
    # Manual reference
    L, Q = safe_eigh(X)
    ref = Q @ jnp.diag(jnp.exp(-L)) @ Q.T
    ref = 0.5 * (ref + ref.T)
    np.testing.assert_allclose(out, ref, atol=1e-12)
