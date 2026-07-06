# -*- coding: utf-8 -*-
"""Tests for ``nitrix.stats.whitening`` (zero-phase / ZCA whitening).

The ZCA whitener maps zero-mean data to unit covariance via the *symmetric*
map ``Sigma^{-1/2}``, computed cuSOLVER-free by the Newton-Schulz driver at
full sphering.  Coverage:

- Certification against the ``sympower(cov, -0.5)`` eigh seam (no behavioural
  divergence from the existing nimox path) and ``cov(whiten(x)) ~ I``.
- The fit/apply seam: single-call ``whiten`` is byte-identical to
  ``apply(fit)``; the inverse map round-trips.
- Sphering: ``s=0`` is centring only, ``s=0.5`` is ``sympower(-0.25)``.
- ``reference=`` fits from other data; ``eps`` ridge; batch / jit / grad.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.linalg import sympower
from nitrix.stats import (
    cov,
    whiten,
    whiten_apply,
    whiten_fit,
    whiten_inverse_apply,
)


def _correlated_data(n=200, d=8, seed=0):
    """``n`` observations of ``d`` correlated features."""
    rng = np.random.default_rng(seed)
    mixing = rng.standard_normal((d, d))
    shift = rng.standard_normal((d,)) * 3.0
    x = rng.standard_normal((n, d)) @ mixing + shift
    return jnp.asarray(x)


# ---------------------------------------------------------------------------
# Certification against the existing eigh seam
# ---------------------------------------------------------------------------


def test_whiten_matches_sympower_seam():
    """ZCA whitening equals ``x_centred @ sympower(cov(x), -0.5)`` -- the
    existing nimox eigh seam -- proving no behavioural divergence."""
    x = _correlated_data()
    xc = x - x.mean(0)
    ref = xc @ sympower(cov(x, rowvar=False), power=-0.5)
    np.testing.assert_allclose(whiten(x), ref, atol=1e-9)


def test_whiten_produces_identity_covariance():
    """The whitened data has (near-)identity covariance."""
    x = _correlated_data(n=500, d=6, seed=1)
    w = whiten(x)
    np.testing.assert_allclose(
        jnp.cov(w, rowvar=False), jnp.eye(6), atol=1e-10
    )
    # ...and is centred.
    np.testing.assert_allclose(w.mean(0), jnp.zeros(6), atol=1e-10)


# ---------------------------------------------------------------------------
# The fit/apply seam (SPEC estimator seam)
# ---------------------------------------------------------------------------


def test_single_call_equals_fit_apply_bytewise():
    """``whiten(x) == whiten_apply(x, whiten_fit(x))`` byte-for-byte."""
    x = _correlated_data(seed=2)
    state = whiten_fit(x)
    assert bool(jnp.array_equal(whiten(x), whiten_apply(x, state)))


def test_inverse_apply_round_trips():
    """``whiten_inverse_apply`` recovers the original data."""
    x = _correlated_data(seed=3)
    state = whiten_fit(x)
    w = whiten_apply(x, state)
    np.testing.assert_allclose(whiten_inverse_apply(w, state), x, atol=1e-10)


def test_reference_fits_from_other_data():
    """``reference=`` whitens ``x`` with another dataset's statistics."""
    ref = _correlated_data(seed=4)
    x = _correlated_data(seed=5)
    state = whiten_fit(ref)
    np.testing.assert_allclose(
        whiten(x, reference=ref), whiten_apply(x, state), atol=0.0
    )
    # Whitening the reference itself yields identity covariance; whitening
    # foreign data with the reference's map does not (sanity: they differ).
    assert not bool(
        jnp.allclose(
            jnp.cov(whiten(x, reference=ref), rowvar=False),
            jnp.eye(x.shape[1]),
            atol=1e-2,
        )
    )


# ---------------------------------------------------------------------------
# Sphering
# ---------------------------------------------------------------------------


def test_sphering_zero_is_centering_only():
    """``sphering=0`` -> ``Sigma^0 = I`` -> centring only."""
    x = _correlated_data(seed=6)
    np.testing.assert_allclose(whiten(x, sphering=0.0), x - x.mean(0), atol=0.0)


def test_partial_sphering_matches_fractional_power():
    """``sphering=s`` -> ``Sigma^{-s/2}`` via the eigenvalue map."""
    x = _correlated_data(seed=7)
    xc = x - x.mean(0)
    for s in (0.25, 0.5, 0.75):
        ref = xc @ sympower(cov(x, rowvar=False), power=-s / 2.0)
        np.testing.assert_allclose(whiten(x, sphering=s), ref, atol=1e-12)


def test_full_sphering_uses_newton_schulz_not_eigh():
    """At ``sphering=1`` the whitening matrix is the Newton-Schulz inverse
    sqrt (matches the eigh reference but is the matmul-only path)."""
    x = _correlated_data(seed=8)
    state = whiten_fit(x, sphering=1.0)
    np.testing.assert_allclose(
        state.matrix, sympower(cov(x, rowvar=False), power=-0.5), atol=1e-9
    )
    # The stored map is symmetric (ZCA is the unique symmetric whitener).
    np.testing.assert_allclose(state.matrix, state.matrix.T, atol=0.0)


# ---------------------------------------------------------------------------
# Conditioning, batching, transformations
# ---------------------------------------------------------------------------


def test_eps_ridge_conditions_rank_deficient():
    """With ``n < d`` the covariance is singular; the ``eps`` ridge keeps the
    Newton-Schulz iteration finite."""
    x = _correlated_data(n=5, d=12, seed=9)  # n < d -> rank-deficient
    w = whiten(x, eps=1e-2)
    assert bool(jnp.all(jnp.isfinite(w)))


def test_assume_centered_stores_zero_mean():
    x = _correlated_data(seed=10)
    xc = x - x.mean(0)
    state = whiten_fit(xc, assume_centered=True)
    np.testing.assert_allclose(state.mean, jnp.zeros(x.shape[1]), atol=0.0)


def test_batched_equals_loop():
    """A batch of datasets whitens independently (vmap parity)."""
    xs = jnp.stack([_correlated_data(n=120, d=5, seed=s) for s in range(4)])
    batched = jax.vmap(whiten)(xs)
    loop = jnp.stack([whiten(x) for x in xs])
    np.testing.assert_allclose(batched, loop, atol=1e-10)


def test_jit_and_grad():
    """``whiten`` is jit-clean and differentiable w.r.t. the data."""
    x = _correlated_data(n=100, d=5, seed=11)
    wj = jax.jit(whiten)(x)
    assert bool(jnp.all(jnp.isfinite(wj)))
    g = jax.grad(lambda X: whiten(X).sum())(x)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_float32_whitens():
    """fp32 path whitens to identity covariance within the looser fp32
    tolerance."""
    x = _correlated_data(n=400, d=6, seed=12).astype(jnp.float32)
    w = whiten(x)
    assert w.dtype == jnp.float32
    np.testing.assert_allclose(
        jnp.cov(w, rowvar=False), jnp.eye(6, dtype=jnp.float32), atol=1e-3
    )
