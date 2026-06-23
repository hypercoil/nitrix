# -*- coding: utf-8 -*-
"""Tests for ``nitrix.nn.ssm.selective_scan`` -- the P1a reference + dispatch.

P1a ships the JAX reference (sequential ``lax.scan`` oracle + parallel
``associative_scan``, auto-selected by platform) and the dispatcher; the fused
chunked-scan kernel lands in P1b.  These tests pin the golden corpus, the scan
math against an independent naive oracle and its invariants (scan == associative,
D-skip linearity, ``A -> 0`` cumulative map), the autodiff gradient, and the
dispatch / loud-fallback contract.
"""

from __future__ import annotations

import warnings

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from _golden import load_golden, tol  # noqa: E402

from nitrix._internal.backend import (  # noqa: E402
    _HAS_AMPERE_NVIDIA,
    NitrixBackendError,
    NitrixBackendFallback,
    reset_fallback_state,
)
from nitrix.nn import selective_scan  # noqa: E402
from nitrix.nn.ssm import (  # noqa: E402
    reference_selective_scan as ref_scan,
)

pallas_only = pytest.mark.skipif(
    not _HAS_AMPERE_NVIDIA,
    reason='requires NVIDIA Ampere+ for the pallas-cuda backend',
)


def _inputs(b, length, d, n, seed, with_d=True, dtype=np.float64):
    rng = np.random.RandomState(seed)
    shp = (length, d) if b is None else (b, length, d)
    bshp = (length, n) if b is None else (b, length, n)
    x = jnp.asarray(rng.standard_normal(shp).astype(dtype))
    delta = jnp.asarray(
        np.log1p(np.exp(rng.standard_normal(shp))).astype(dtype)
    )
    A = jnp.asarray(-np.exp(rng.standard_normal((d, n))).astype(dtype))
    B = jnp.asarray(rng.standard_normal(bshp).astype(dtype))
    C = jnp.asarray(rng.standard_normal(bshp).astype(dtype))
    D = (
        jnp.asarray(rng.standard_normal((d,)).astype(dtype))
        if with_d
        else None
    )
    return x, delta, A, B, C, D


def _relerr(a, b):
    return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-30))


# --- golden corpus -------------------------------------------------------


def test_golden_reference_reproducible():
    data = load_golden('selective_scan_float32')
    out = ref_scan(
        jnp.asarray(data['x']),
        jnp.asarray(data['delta']),
        jnp.asarray(data['A']),
        jnp.asarray(data['B']),
        jnp.asarray(data['C']),
        jnp.asarray(data['D']),
        method='sequential',
    )
    atol, rtol = tol('selective_scan', np.float32)
    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float32), data['out'], atol=atol, rtol=rtol
    )


# --- correctness vs an independent naive oracle --------------------------


def _naive(x, delta, A, B, C, D=None):
    x, delta, A, B, C = map(np.asarray, (x, delta, A, B, C))
    length, d = x.shape
    n = A.shape[1]
    h = np.zeros((d, n))
    ys = []
    for t in range(length):
        dA = np.exp(delta[t][:, None] * A)
        dBx = (delta[t][:, None] * B[t][None, :]) * x[t][:, None]
        h = dA * h + dBx
        ys.append((h * C[t][None, :]).sum(-1))
    y = np.stack(ys)
    if D is not None:
        y = y + np.asarray(D) * x
    return y


def test_reference_matches_naive_oracle():
    x, delta, A, B, C, D = _inputs(None, 7, 4, 3, seed=1)
    got = ref_scan(x, delta, A, B, C, D, method='sequential')
    want = _naive(x, delta, A, B, C, D)
    assert np.allclose(np.asarray(got), want, atol=1e-10)


# --- scan invariants -----------------------------------------------------


@pytest.mark.parametrize('seed', [2, 3])
def test_sequential_equals_associative(seed):
    x, delta, A, B, C, D = _inputs(2, 8, 4, 3, seed=seed)
    seq = ref_scan(x, delta, A, B, C, D, method='sequential')
    aso = ref_scan(x, delta, A, B, C, D, method='associative')
    assert np.allclose(np.asarray(seq), np.asarray(aso), atol=1e-10)


def test_d_skip_is_linear():
    x, delta, A, B, C, D = _inputs(2, 6, 4, 3, seed=4)
    with_d = ref_scan(x, delta, A, B, C, D, method='sequential')
    without = ref_scan(x, delta, A, B, C, None, method='sequential')
    assert np.allclose(
        np.asarray(with_d), np.asarray(without) + np.asarray(D * x), atol=1e-10
    )


def test_zero_A_reduces_to_cumulative_map():
    # A -> 0 => dA = 1 => h_t is the cumulative sum of dBx over the sequence.
    x, delta, _, B, C, _ = _inputs(2, 6, 4, 3, seed=5)
    A0 = jnp.zeros((4, 3))
    got = ref_scan(x, delta, A0, B, C, None, method='sequential')
    dBx = delta[..., None] * B[..., None, :] * x[..., None]  # (b, l, d, n)
    H = jnp.cumsum(dBx, axis=-3)
    expected = (H * C[..., None, :]).sum(axis=-1)
    assert np.allclose(np.asarray(got), np.asarray(expected), atol=1e-10)


# --- autodiff ------------------------------------------------------------


def test_autodiff_gradient_matches_finite_difference():
    x, delta, A, B, C, D = _inputs(None, 5, 3, 2, seed=6)

    def loss(x):
        return jnp.sum(
            ref_scan(x, delta, A, B, C, D, method='sequential') ** 2
        )

    g = np.asarray(jax.grad(loss)(x))
    assert np.all(np.isfinite(g))
    eps = 1e-6
    xn = np.asarray(x)
    fd = np.zeros_like(xn)
    for idx in np.ndindex(xn.shape):
        xp = xn.copy()
        xp[idx] += eps
        xm = xn.copy()
        xm[idx] -= eps
        fd[idx] = (
            float(loss(jnp.asarray(xp))) - float(loss(jnp.asarray(xm)))
        ) / (2 * eps)
    assert np.allclose(g, fd, atol=1e-5, rtol=1e-5)


# --- dispatch & loud fallback -------------------------------------------


def test_backend_jax_byte_identical_to_reference():
    x, delta, A, B, C, D = _inputs(2, 8, 4, 3, seed=7)
    a = selective_scan(x, delta, A, B, C, D, backend='jax')
    b = ref_scan(x, delta, A, B, C, D)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_jit_compiles_and_matches():
    x, delta, A, B, C, D = _inputs(2, 8, 4, 3, seed=8)
    ref = ref_scan(x, delta, A, B, C, D)
    out = jax.jit(
        lambda x, delta, A, B, C, D: selective_scan(
            x, delta, A, B, C, D, backend='jax'
        )
    )(x, delta, A, B, C, D)
    assert np.allclose(np.asarray(out), np.asarray(ref), atol=1e-12)


def test_explicit_pallas_on_cpu_raises():
    if _HAS_AMPERE_NVIDIA:
        pytest.skip('CPU-only: explicit pallas-cuda errors without Ampere')
    x, delta, A, B, C, D = _inputs(1, 4, 3, 2, seed=9)
    with pytest.raises(NitrixBackendError):
        selective_scan(x, delta, A, B, C, D, backend='pallas-cuda')


@pallas_only
def test_pallas_falls_back_loudly_until_p1b():
    # P1a: no fused kernel yet -> pallas-cuda falls back loudly to the
    # reference (which still uses the parallel associative_scan on GPU).
    x, delta, A, B, C, D = _inputs(2, 8, 4, 3, seed=10)
    reset_fallback_state()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = selective_scan(x, delta, A, B, C, D, backend='pallas-cuda')
        fired = [
            x_ for x_ in rec if issubclass(x_.category, NitrixBackendFallback)
        ]
    assert len(fired) == 1
    assert np.allclose(
        np.asarray(out), np.asarray(ref_scan(x, delta, A, B, C, D)), atol=1e-12
    )


def test_validation_errors():
    x, delta, A, B, C, D = _inputs(2, 8, 4, 3, seed=11)
    with pytest.raises(ValueError):  # B last dim != n
        selective_scan(x, delta, A, B[..., :-1], C, D)
    with pytest.raises(ValueError):  # D wrong shape
        selective_scan(x, delta, A, B, C, D[:-1])
