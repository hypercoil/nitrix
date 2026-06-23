# -*- coding: utf-8 -*-
"""Tests for ``nitrix.nn.norm`` -- the reference + dispatch (suite §7.3).

P3 ships the pure-JAX references (LayerNorm / GroupNorm / InstanceNorm) with the
curse-of-depth ``out_scale`` hook and the backend dispatch; the fused single-pass
kernel is perf-gated (it falls back loudly until a profiler justifies it).  These
tests pin the golden corpus, the math against independent numpy oracles and its
invariants (out_scale linearity, instance == group(C), normalised statistics),
the autodiff gradient, and the dispatch / loud-fallback contract.
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
from nitrix.nn import group_norm, layer_norm  # noqa: E402
from nitrix.nn.norm import (  # noqa: E402
    reference_group_norm,
    reference_instance_norm,
    reference_layer_norm,
)

pallas_only = pytest.mark.skipif(
    not _HAS_AMPERE_NVIDIA,
    reason='requires NVIDIA Ampere+ for the pallas-cuda backend',
)


def _relerr(a, b):
    return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-30))


# --- golden corpus -------------------------------------------------------


def test_golden_layer_norm():
    data = load_golden('layer_norm_float32')
    out = reference_layer_norm(
        jnp.asarray(data['x']),
        jnp.asarray(data['weight']),
        jnp.asarray(data['bias']),
        out_scale=0.5,
    )
    atol, rtol = tol('layer_norm', np.float32)
    np.testing.assert_allclose(
        np.asarray(out, np.float32), data['out'], atol=atol, rtol=rtol
    )


@pytest.mark.parametrize(
    'name,fn',
    [
        (
            'group_norm_float32',
            lambda d: reference_group_norm(
                jnp.asarray(d['x']),
                4,
                jnp.asarray(d['weight']),
                jnp.asarray(d['bias']),
            ),
        ),
        (
            'instance_norm_float32',
            lambda d: reference_instance_norm(
                jnp.asarray(d['x']),
                jnp.asarray(d['weight']),
                jnp.asarray(d['bias']),
            ),
        ),
    ],
)
def test_golden_group_instance(name, fn):
    data = load_golden(name)
    op = name.split('_float32')[0]
    atol, rtol = tol(op, np.float32)
    np.testing.assert_allclose(
        np.asarray(fn(data), np.float32), data['out'], atol=atol, rtol=rtol
    )


# --- correctness vs independent numpy oracles ---------------------------


def _ln_np(x, w, b, eps=1e-5, out_scale=1.0):
    x = np.asarray(x)
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    xh = (x - mu) / np.sqrt(var + eps)
    if w is not None:
        xh = xh * np.asarray(w)
    if b is not None:
        xh = xh + np.asarray(b)
    return out_scale * xh


def _gn_np(x, g, w, b, eps=1e-5, out_scale=1.0):
    x = np.asarray(x)
    n, c = x.shape[:2]
    spatial = x.shape[2:]
    xg = x.reshape(n, g, c // g, *spatial)
    axes = tuple(range(2, xg.ndim))
    mu = xg.mean(axes, keepdims=True)
    var = xg.var(axes, keepdims=True)
    xh = ((xg - mu) / np.sqrt(var + eps)).reshape(x.shape)
    sh = (1, c) + (1,) * len(spatial)
    if w is not None:
        xh = xh * np.asarray(w).reshape(sh)
    if b is not None:
        xh = xh + np.asarray(b).reshape(sh)
    return out_scale * xh


def test_layer_norm_matches_numpy():
    rng = np.random.RandomState(0)
    x = jnp.asarray(rng.standard_normal((3, 4, 16)))
    w = jnp.asarray(rng.standard_normal(16))
    b = jnp.asarray(rng.standard_normal(16))
    got = reference_layer_norm(x, w, b, out_scale=0.7)
    assert np.allclose(
        np.asarray(got), _ln_np(x, w, b, out_scale=0.7), atol=1e-10
    )


def test_group_norm_matches_numpy():
    rng = np.random.RandomState(1)
    x = jnp.asarray(rng.standard_normal((2, 8, 5, 5)))
    w = jnp.asarray(rng.standard_normal(8))
    b = jnp.asarray(rng.standard_normal(8))
    got = reference_group_norm(x, 4, w, b)
    assert np.allclose(np.asarray(got), _gn_np(x, 4, w, b), atol=1e-10)


# --- invariants ----------------------------------------------------------


def test_out_scale_is_linear():
    rng = np.random.RandomState(2)
    x = jnp.asarray(rng.standard_normal((2, 3, 16)))
    w = jnp.asarray(rng.standard_normal(16))
    b = jnp.asarray(rng.standard_normal(16))
    base = reference_layer_norm(x, w, b, out_scale=1.0)
    scaled = reference_layer_norm(x, w, b, out_scale=2.5)
    assert np.allclose(np.asarray(scaled), 2.5 * np.asarray(base), atol=1e-10)


def test_instance_norm_equals_group_norm_per_channel():
    rng = np.random.RandomState(3)
    x = jnp.asarray(rng.standard_normal((2, 6, 4, 4)))
    a = reference_instance_norm(x)
    b = reference_group_norm(x, 6)  # one group per channel
    assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-12)


def test_normalised_statistics():
    # Without affine, each normalised group has ~zero mean and ~unit variance.
    rng = np.random.RandomState(4)
    x = jnp.asarray(rng.standard_normal((2, 8, 4, 4)))
    xh = np.asarray(reference_group_norm(x, 4, eps=1e-6))
    g = xh.reshape(2, 4, 2, 4, 4)
    axes = (2, 3, 4)
    assert np.allclose(g.mean(axes), 0.0, atol=1e-4)
    assert np.allclose(g.var(axes), 1.0, atol=1e-3)


# --- autodiff ------------------------------------------------------------


def test_autodiff_gradient_matches_finite_difference():
    rng = np.random.RandomState(5)
    x = jnp.asarray(rng.standard_normal((2, 8)))
    w = jnp.asarray(rng.standard_normal(8))
    b = jnp.asarray(rng.standard_normal(8))

    def loss(x, w, b):
        return jnp.sum(reference_layer_norm(x, w, b, out_scale=0.5) ** 2)

    g = jax.grad(loss, argnums=(0, 1, 2))(x, w, b)
    assert all(np.all(np.isfinite(np.asarray(gi))) for gi in g)
    eps = 1e-6
    gx = np.asarray(g[0])
    xn = np.asarray(x)
    fd = np.zeros_like(xn)
    for idx in np.ndindex(xn.shape):
        xp = xn.copy()
        xp[idx] += eps
        xm = xn.copy()
        xm[idx] -= eps
        fd[idx] = (
            float(loss(jnp.asarray(xp), w, b))
            - float(loss(jnp.asarray(xm), w, b))
        ) / (2 * eps)
    assert np.allclose(gx, fd, atol=1e-5, rtol=1e-5)


# --- dispatch & loud fallback -------------------------------------------


def test_backend_jax_byte_identical_to_reference():
    rng = np.random.RandomState(6)
    x = jnp.asarray(rng.standard_normal((2, 4, 16)))
    w = jnp.asarray(rng.standard_normal(16))
    b = jnp.asarray(rng.standard_normal(16))
    a = layer_norm(x, w, b, out_scale=0.3, backend='jax')
    r = reference_layer_norm(x, w, b, out_scale=0.3)
    assert np.array_equal(np.asarray(a), np.asarray(r))


def test_jit_compiles_and_matches():
    rng = np.random.RandomState(7)
    x = jnp.asarray(rng.standard_normal((2, 8, 4, 4)))
    ref = reference_group_norm(x, 4)
    out = jax.jit(lambda x: group_norm(x, 4, backend='jax'))(x)
    assert np.allclose(np.asarray(out), np.asarray(ref), atol=1e-12)


def test_explicit_pallas_on_cpu_raises():
    if _HAS_AMPERE_NVIDIA:
        pytest.skip('CPU-only: explicit pallas-cuda errors without Ampere')
    x = jnp.zeros((2, 4, 8))
    with pytest.raises(NitrixBackendError):
        layer_norm(x, backend='pallas-cuda')


@pallas_only
def test_pallas_falls_back_loudly_perf_gated():
    # The fused norm kernel is perf-gated -> pallas-cuda falls back loudly.
    rng = np.random.RandomState(8)
    x = jnp.asarray(rng.standard_normal((2, 4, 16)))
    w = jnp.asarray(rng.standard_normal(16))
    b = jnp.asarray(rng.standard_normal(16))
    reset_fallback_state()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = layer_norm(x, w, b, backend='pallas-cuda')
        fired = [
            r for r in rec if issubclass(r.category, NitrixBackendFallback)
        ]
    assert len(fired) == 1
    assert _relerr(out, reference_layer_norm(x, w, b)) < 1e-12


def test_validation_errors():
    x = jnp.zeros((2, 4, 16))
    with pytest.raises(ValueError):  # weight wrong shape
        layer_norm(x, jnp.zeros((8,)))
    with pytest.raises(ValueError):  # C not divisible by num_groups
        group_norm(jnp.zeros((2, 6, 4)), 4)
