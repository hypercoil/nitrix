# -*- coding: utf-8 -*-
"""Tests for the fused Pallas flash-attention kernel (suite Phase 2 / P0b).

GPU-only (NVIDIA Ampere+).  Phase 2 ships the fused *forward* kernel with a
``custom_vjp`` whose backward recomputes through the reference; these tests pin
the forward parity (pallas ~= jax within the ``tolerance.toml`` pallas-cuda
row), the gradient parity (incl. the learnable-bias gradient ``d_bias``), the
gross-memory contract (the ``(s, t)`` score matrix is never materialised), and
the dispatch / loud-fallback behaviour for shapes the kernel rejects.

The fully-fused Triton backward is a later increment; the forward already
delivers the inference / forward activation-memory win certified here.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _golden import tol

from nitrix._internal.backend import (
    _HAS_AMPERE_NVIDIA,
    NitrixBackendFallback,
    reset_fallback_state,
)
from nitrix._kernels.cuda.attention import (
    scaled_dot_product_attention_pallas as kern,
)
from nitrix.nn import scaled_dot_product_attention
from nitrix.nn.attention import (
    reference_scaled_dot_product_attention as ref_sdpa,
)

pallas_only = pytest.mark.skipif(
    not _HAS_AMPERE_NVIDIA,
    reason='requires NVIDIA Ampere+ for the pallas-cuda backend',
)

_ATOL, _RTOL = tol('attention', np.float32, 'pallas-cuda')


def _mk(shape, seed):
    rng = np.random.RandomState(seed)
    return jnp.asarray(rng.standard_normal(shape).astype(np.float32))


def _relerr(a, b):
    return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-30))


# Gradients span a wide dynamic range, so the fused-backward parity is checked
# with a max-relative tolerance rather than elementwise atol/rtol.
_GRAD_TOL = 3.0e-3


def _case(name):
    """Return ``(q, k, v, kwargs, scale)`` for a named attention path."""
    h, d = 2, 64
    b = 2
    if name == 'cross':
        s, t = 128, 256
    else:
        s = t = 128
    q = _mk((b, h, s, d), 0)
    k = _mk((b, h, t, d), 1)
    v = _mk((b, h, t, d), 2)
    scale = 1.0 / np.sqrt(d)
    kwargs = {}
    if name == 'bias':
        kwargs['bias'] = _mk((h, s, t), 3)  # Swin-style, shared over batch
    elif name == 'causal':
        kwargs['causal'] = True
    elif name == 'mask':
        m = np.random.RandomState(4).rand(b, h, s, t) > 0.3
        m[..., 0] = True  # keep >= 1 per row
        kwargs['mask'] = jnp.asarray(m)
    return q, k, v, kwargs, scale


# --- forward parity ------------------------------------------------------


@pallas_only
@pytest.mark.parametrize('name', ['dense', 'bias', 'causal', 'mask', 'cross'])
def test_forward_parity_with_reference(name):
    q, k, v, kwargs, scale = _case(name)
    got = kern(q, k, v, scale=scale, **kwargs)
    want = ref_sdpa(q, k, v, scale=scale, **kwargs)
    assert got.shape == want.shape
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(want), atol=_ATOL, rtol=_RTOL
    )


# --- gradient parity (fused Triton backward, incl. in-kernel d_bias) -----


@pallas_only
@pytest.mark.parametrize('name', ['dense', 'causal', 'mask', 'cross'])
def test_gradient_parity_dqkv(name):
    # Exercises the fused recompute-in-tile backward vs the autodiff
    # reference (dq / dk / dv).
    q, k, v, kwargs, scale = _case(name)

    def lk(q, k, v):
        return jnp.sum(kern(q, k, v, scale=scale, **kwargs) ** 2)

    def lr(q, k, v):
        return jnp.sum(ref_sdpa(q, k, v, scale=scale, **kwargs) ** 2)

    gk = jax.grad(lk, argnums=(0, 1, 2))(q, k, v)
    gr = jax.grad(lr, argnums=(0, 1, 2))(q, k, v)
    for a, b in zip(gk, gr):
        assert np.all(np.isfinite(np.asarray(a)))
        assert _relerr(a, b) < _GRAD_TOL


@pallas_only
def test_gradient_parity_with_bias_dbias():
    # The in-kernel d_bias path: the learnable-bias gradient, reduced to the
    # (shared-over-batch) bias shape.
    q, k, v, kwargs, scale = _case('bias')
    bias = kwargs['bias']

    def lk(q, k, v, bias):
        return jnp.sum(kern(q, k, v, scale=scale, bias=bias) ** 2)

    def lr(q, k, v, bias):
        return jnp.sum(ref_sdpa(q, k, v, scale=scale, bias=bias) ** 2)

    gk = jax.grad(lk, argnums=(0, 1, 2, 3))(q, k, v, bias)
    gr = jax.grad(lr, argnums=(0, 1, 2, 3))(q, k, v, bias)
    assert gk[3].shape == bias.shape  # d_bias reduced to bias shape
    for a, b in zip(gk, gr):
        assert np.all(np.isfinite(np.asarray(a)))
        assert _relerr(a, b) < _GRAD_TOL


# --- qk-norm (RMS pre-op outside the fused core) ------------------------


@pallas_only
@pytest.mark.parametrize('name', ['dense', 'bias', 'causal'])
def test_qk_norm_parity(name):
    q, k, v, kwargs, scale = _case(name)
    got = scaled_dot_product_attention(
        q, k, v, scale=scale, qk_norm=True, backend='pallas-cuda', **kwargs
    )
    want = ref_sdpa(q, k, v, scale=scale, qk_norm=True, **kwargs)
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(want), atol=_ATOL, rtol=_RTOL
    )


@pallas_only
def test_qk_norm_false_is_byte_identical_to_no_norm():
    # Regression guard: qk_norm=False does not perturb the fused path.
    q, k, v, _, scale = _case('dense')
    a = scaled_dot_product_attention(
        q, k, v, scale=scale, qk_norm=False, backend='pallas-cuda'
    )
    b = scaled_dot_product_attention(
        q, k, v, scale=scale, backend='pallas-cuda'
    )
    assert np.array_equal(np.asarray(a), np.asarray(b))


@pallas_only
def test_qk_norm_gradient_matches_reference():
    q, k, v, _, scale = _case('dense')

    def lk(q, k, v):
        return jnp.sum(
            scaled_dot_product_attention(
                q, k, v, scale=scale, qk_norm=True, backend='pallas-cuda'
            )
            ** 2
        )

    def lr(q, k, v):
        return jnp.sum(ref_sdpa(q, k, v, scale=scale, qk_norm=True) ** 2)

    gk = jax.grad(lk, argnums=(0, 1, 2))(q, k, v)
    gr = jax.grad(lr, argnums=(0, 1, 2))(q, k, v)
    for a, b in zip(gk, gr):
        assert np.all(np.isfinite(np.asarray(a)))
        assert _relerr(a, b) < _GRAD_TOL


# --- gross-memory contract ----------------------------------------------


@pallas_only
def test_forward_does_not_materialise_score_matrix():
    # The fused path streams the (s, t) scores; the reference materialises
    # them.  memory_analysis().temp_size_in_bytes is a deterministic probe.
    b, h, s, d = 1, 1, 512, 64
    q = _mk((b, h, s, d), 1)
    k = _mk((b, h, s, d), 2)
    v = _mk((b, h, s, d), 3)
    comp_k = (
        jax.jit(
            lambda q, k, v: scaled_dot_product_attention(
                q, k, v, backend='pallas-cuda'
            )
        )
        .lower(q, k, v)
        .compile()
    )
    comp_r = (
        jax.jit(
            lambda q, k, v: scaled_dot_product_attention(
                q, k, v, backend='jax'
            )
        )
        .lower(q, k, v)
        .compile()
    )
    temp_k = comp_k.memory_analysis().temp_size_in_bytes
    temp_r = comp_r.memory_analysis().temp_size_in_bytes
    score_bytes = b * h * s * s * 4
    assert temp_k < score_bytes  # never holds a full (s, t) score tile
    assert temp_k < temp_r


@pallas_only
def test_backward_does_not_materialise_score_matrix():
    # The fused backward recomputes score tiles in-SRAM; the reference
    # backward materialises the (s, t) scores.  (No bias -> no (s, t) output.)
    b, h, s, d = 1, 1, 512, 64
    q = _mk((b, h, s, d), 1)
    k = _mk((b, h, s, d), 2)
    v = _mk((b, h, s, d), 3)

    def grad_of(backend):
        def loss(q, k, v):
            return jnp.sum(
                scaled_dot_product_attention(q, k, v, backend=backend) ** 2
            )

        return jax.grad(loss, argnums=(0, 1, 2))

    comp_k = jax.jit(grad_of('pallas-cuda')).lower(q, k, v).compile()
    comp_r = jax.jit(grad_of('jax')).lower(q, k, v).compile()
    temp_k = comp_k.memory_analysis().temp_size_in_bytes
    temp_r = comp_r.memory_analysis().temp_size_in_bytes
    assert temp_k < b * h * s * s * 4  # dq/dk/dv recompute, no stored (s, t)
    assert temp_k < temp_r


# --- dispatch & loud fallback -------------------------------------------


@pallas_only
def test_auto_dispatch_uses_kernel_without_warning():
    q, k, v, _, scale = _case('dense')
    reset_fallback_state()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = scaled_dot_product_attention(q, k, v, backend='auto')
        fired = [
            x for x in rec if issubclass(x.category, NitrixBackendFallback)
        ]
    assert len(fired) == 0  # the kernel handles this shape
    assert _relerr(out, ref_sdpa(q, k, v, scale=scale)) < _ATOL


@pallas_only
def test_jit_through_kernel():
    q, k, v, _, scale = _case('causal')
    out = jax.jit(
        lambda q, k, v: scaled_dot_product_attention(
            q, k, v, causal=True, backend='pallas-cuda'
        )
    )(q, k, v)
    assert _relerr(out, ref_sdpa(q, k, v, scale=scale, causal=True)) < _ATOL


@pallas_only
@pytest.mark.parametrize('reason', ['dv_ne_d', 'non_pow2_d', 'odd_seq'])
def test_unsupported_shape_falls_back_loudly(reason):
    h, b = 2, 2
    if reason == 'dv_ne_d':
        q = _mk((b, h, 128, 64), 0)
        k = _mk((b, h, 128, 64), 1)
        v = _mk((b, h, 128, 48), 2)  # d_v != d
    elif reason == 'non_pow2_d':
        q = _mk((b, h, 128, 48), 0)  # head dim not a power of two
        k = _mk((b, h, 128, 48), 1)
        v = _mk((b, h, 128, 48), 2)
    else:  # odd_seq: not divisible by the block size
        q = _mk((b, h, 100, 64), 0)
        k = _mk((b, h, 100, 64), 1)
        v = _mk((b, h, 100, 64), 2)
    reset_fallback_state()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = scaled_dot_product_attention(q, k, v, backend='pallas-cuda')
        fired = [
            x for x in rec if issubclass(x.category, NitrixBackendFallback)
        ]
    assert len(fired) == 1
    # Default scale on both sides (a python float); the fallback must be the
    # byte-identical reference output.
    assert np.array_equal(np.asarray(out), np.asarray(ref_sdpa(q, k, v)))
