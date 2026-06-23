# -*- coding: utf-8 -*-
"""Tests for the fused Pallas selective-scan kernel (suite P1b).

GPU-only (NVIDIA Ampere+).  P1b ships the fully-fused forward + backward
(chunked cumsum closed-form forward; reverse chunked-cumsum recompute-adjoint
backward) -- the (l, d, n) state trajectory is materialised in HBM in neither
pass.  These tests pin the forward parity (pallas ~= jax within the
``tolerance.toml`` pallas-cuda row on realistic-Mamba inputs), the forward and
backward gross-memory contracts, the fused-backward gradient parity, and the
dispatch / loud-fallback behaviour.

Note: the fused forward uses a cumsum closed-form with an fp32 within-chunk
dynamic-range limit (|A * cumsum(delta)| < ~80), satisfied by the realistic
Mamba regime (small delta).  Extreme ranges should use backend='jax'.
"""

from __future__ import annotations

import warnings

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from _golden import tol  # noqa: E402

from nitrix._internal.backend import (  # noqa: E402
    _HAS_AMPERE_NVIDIA,
    NitrixBackendFallback,
    reset_fallback_state,
)
from nitrix._kernels.cuda.selective_scan import (  # noqa: E402
    selective_scan_pallas as kern,
)
from nitrix.nn import selective_scan  # noqa: E402
from nitrix.nn.ssm import reference_selective_scan as ref_scan  # noqa: E402

pallas_only = pytest.mark.skipif(
    not _HAS_AMPERE_NVIDIA,
    reason='requires NVIDIA Ampere+ for the pallas-cuda backend',
)

_ATOL, _RTOL = tol('selective_scan', np.float32, 'pallas-cuda')
_GRAD_TOL = 3.0e-3


def _inputs(b, length, d, n, seed, with_d=True):
    # Realistic Mamba regime: small delta (dt ~ [0.01, 0.1]); modest A = -(1..n).
    rng = np.random.RandomState(seed)
    f32 = np.float32
    x = jnp.asarray(rng.standard_normal((b, length, d)).astype(f32))
    sig = 1.0 / (1.0 + np.exp(-rng.standard_normal((b, length, d))))
    delta = jnp.asarray((0.01 + 0.09 * sig).astype(f32))
    A = jnp.asarray((-(np.arange(n) + 1.0))[None, :].repeat(d, 0).astype(f32))
    B = jnp.asarray(rng.standard_normal((b, length, n)).astype(f32))
    C = jnp.asarray(rng.standard_normal((b, length, n)).astype(f32))
    D = jnp.asarray(rng.standard_normal((d,)).astype(f32)) if with_d else None
    return x, delta, A, B, C, D


def _relerr(a, b):
    return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-30))


# --- forward parity ------------------------------------------------------


@pallas_only
@pytest.mark.parametrize('with_d', [False, True])
def test_forward_parity_with_reference(with_d):
    x, delta, A, B, C, D = _inputs(2, 64, 8, 4, seed=0, with_d=with_d)
    got = kern(x, delta, A, B, C, D)
    want = ref_scan(x, delta, A, B, C, D, method='sequential')
    assert got.shape == want.shape
    np.testing.assert_allclose(
        np.asarray(got, np.float32),
        np.asarray(want, np.float32),
        atol=_ATOL,
        rtol=_RTOL,
    )


@pallas_only
def test_forward_parity_no_batch_and_longer_seq():
    # no batch dim + a longer (chunked) sequence.
    x, delta, A, B, C, D = _inputs(1, 128, 4, 4, seed=1)
    x = x[0]
    delta = delta[0]
    B = B[0]
    C = C[0]
    got = kern(x, delta, A, B, C, D)
    want = ref_scan(x, delta, A, B, C, D, method='sequential')
    assert _relerr(got, want) < _RTOL


# --- gross-memory contract ----------------------------------------------


@pallas_only
def test_forward_does_not_materialise_state_trajectory():
    # The fused path keeps the (d, n) state in SRAM; the reference's
    # associative_scan materialises the (l, d, n) trajectory in HBM.
    b, length, d, n = 2, 128, 8, 4
    x, delta, A, B, C, D = _inputs(b, length, d, n, seed=2)
    comp_k = (
        jax.jit(lambda *a: selective_scan(*a, backend='pallas-cuda'))
        .lower(x, delta, A, B, C, D)
        .compile()
    )
    comp_r = (
        jax.jit(lambda *a: selective_scan(*a, backend='jax'))
        .lower(x, delta, A, B, C, D)
        .compile()
    )
    temp_k = comp_k.memory_analysis().temp_size_in_bytes
    temp_r = comp_r.memory_analysis().temp_size_in_bytes
    assert temp_k < b * length * d * n * 4  # no full (l, d, n) trajectory
    assert temp_k < temp_r


@pallas_only
def test_backward_does_not_materialise_state_trajectory():
    # The fused recompute-adjoint backward keeps state in SRAM + only the
    # (l/chunk)-reduced chunk-start residual; the reference backward stores the
    # full (l, d, n) trajectory.
    b, length, d, n = 2, 128, 8, 4
    x, delta, A, B, C, D = _inputs(b, length, d, n, seed=7)

    def grad_of(backend):
        return jax.grad(
            lambda *a: jnp.sum(selective_scan(*a, backend=backend) ** 2),
            argnums=(0, 1, 2, 3, 4, 5),
        )

    comp_k = (
        jax.jit(grad_of('pallas-cuda')).lower(x, delta, A, B, C, D).compile()
    )
    comp_r = jax.jit(grad_of('jax')).lower(x, delta, A, B, C, D).compile()
    temp_k = comp_k.memory_analysis().temp_size_in_bytes
    temp_r = comp_r.memory_analysis().temp_size_in_bytes
    assert temp_k < b * length * d * n * 4  # no full (l, d, n) trajectory
    assert temp_k < temp_r


# --- gradient correctness (fused recompute-adjoint backward) ------------


@pallas_only
@pytest.mark.parametrize('with_d', [False, True])
def test_gradient_matches_reference(with_d):
    x, delta, A, B, C, D = _inputs(2, 64, 8, 4, seed=3, with_d=with_d)
    args = (x, delta, A, B, C) if D is None else (x, delta, A, B, C, D)
    nargs = len(args)

    def lk(*a):
        return jnp.sum(kern(*a) ** 2)

    def lr(*a):
        return jnp.sum(ref_scan(*a, method='sequential') ** 2)

    gk = jax.grad(lk, argnums=tuple(range(nargs)))(*args)
    gr = jax.grad(lr, argnums=tuple(range(nargs)))(*args)
    for a, b in zip(gk, gr):
        assert np.all(np.isfinite(np.asarray(a)))
        assert _relerr(a, b) < _GRAD_TOL


# --- dispatch & loud fallback -------------------------------------------


@pallas_only
def test_auto_dispatch_uses_kernel_without_warning():
    x, delta, A, B, C, D = _inputs(2, 64, 8, 4, seed=4)
    reset_fallback_state()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = selective_scan(x, delta, A, B, C, D, backend='auto')
        fired = [
            w for w in rec if issubclass(w.category, NitrixBackendFallback)
        ]
    assert len(fired) == 0
    ref = ref_scan(x, delta, A, B, C, D, method='sequential')
    assert _relerr(out, ref) < _RTOL


@pallas_only
def test_jit_through_kernel():
    x, delta, A, B, C, D = _inputs(2, 64, 8, 4, seed=5)
    out = jax.jit(lambda *a: selective_scan(*a, backend='pallas-cuda'))(
        x, delta, A, B, C, D
    )
    ref = ref_scan(x, delta, A, B, C, D, method='sequential')
    assert _relerr(out, ref) < _RTOL


@pallas_only
@pytest.mark.parametrize('reason', ['non_divisible_len', 'float64'])
def test_unsupported_shape_falls_back_loudly(reason):
    if reason == 'non_divisible_len':
        x, delta, A, B, C, D = _inputs(2, 40, 8, 4, seed=6)  # 40 % 16 != 0
    else:
        x, delta, A, B, C, D = _inputs(2, 64, 8, 4, seed=6)
        x = x.astype(jnp.float64)  # fused path is float32-only
    reset_fallback_state()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = selective_scan(x, delta, A, B, C, D, backend='pallas-cuda')
        fired = [
            w for w in rec if issubclass(w.category, NitrixBackendFallback)
        ]
    assert len(fired) == 1
    assert _relerr(out, ref_scan(x, delta, A, B, C, D)) < 1e-5
