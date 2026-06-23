# -*- coding: utf-8 -*-
"""Tests for ``nitrix.nn.attention`` -- the Phase-1 reference + dispatch.

Phase 1 ships the JAX reference only (the fused Pallas path lands in
Phase 2).  These tests pin the golden corpus, the attention math against
an independent naive oracle and its invariants, the autodiff gradient,
and the dispatch / loud-fallback contract.
"""

from __future__ import annotations

import warnings

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from _golden import load_golden, tol  # noqa: E402
from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from nitrix._internal.backend import (  # noqa: E402
    _HAS_AMPERE_NVIDIA,
    NitrixBackendError,
    NitrixBackendFallback,
    reset_fallback_state,
)
from nitrix.nn import scaled_dot_product_attention  # noqa: E402
from nitrix.nn.attention import (  # noqa: E402
    reference_scaled_dot_product_attention as ref_sdpa,
)

pallas_only = pytest.mark.skipif(
    not _HAS_AMPERE_NVIDIA,
    reason='requires NVIDIA Ampere+ for the pallas-cuda backend',
)

ATTENTION_CASES = [
    'attention_dense_float32',
    'attention_windowed_bias_float32',
    'attention_causal_float32',
    'attention_cross_float32',
]


def _call_from_golden(data: dict[str, np.ndarray]) -> jnp.ndarray:
    return ref_sdpa(
        jnp.asarray(data['q']),
        jnp.asarray(data['k']),
        jnp.asarray(data['v']),
        bias=jnp.asarray(data['bias']) if 'bias' in data else None,
        mask=jnp.asarray(data['mask']) if 'mask' in data else None,
        causal=bool(data['causal']),
    )


# --- golden corpus -------------------------------------------------------


@pytest.mark.parametrize('name', ATTENTION_CASES)
def test_golden_reference_reproducible(name):
    data = load_golden(name)
    out = np.asarray(_call_from_golden(data), dtype=np.float32)
    atol, rtol = tol('attention', np.float32)
    np.testing.assert_allclose(out, data['out'], atol=atol, rtol=rtol)


# --- correctness vs an independent naive oracle --------------------------


def _naive(m_q, m_k, m_v, bias=None, scale=None, causal=False):
    # single (s, d) / (t, d) / (t, d_v); explicit per-row softmax.
    m_q, m_k, m_v = np.asarray(m_q), np.asarray(m_k), np.asarray(m_v)
    s, d = m_q.shape
    t = m_k.shape[0]
    if scale is None:
        scale = 1.0 / np.sqrt(d)
    out = np.zeros((s, m_v.shape[1]))
    for i in range(s):
        logit = scale * (m_k @ m_q[i])
        if bias is not None:
            logit = logit + np.asarray(bias)[i]
        if causal:
            logit = logit + np.where(np.arange(t) <= i, 0.0, -np.inf)
        logit = logit - logit.max()
        w = np.exp(logit)
        w = w / w.sum()
        out[i] = w @ m_v
    return out


def test_reference_matches_naive_oracle():
    rng = np.random.RandomState(7)
    s, t, d, dv = 4, 5, 3, 2
    q = rng.standard_normal((s, d))
    k = rng.standard_normal((t, d))
    v = rng.standard_normal((t, dv))
    bias = rng.standard_normal((s, t))
    got = ref_sdpa(
        jnp.asarray(q)[None, None],
        jnp.asarray(k)[None, None],
        jnp.asarray(v)[None, None],
        bias=jnp.asarray(bias)[None, None],
    )[0, 0]
    want = _naive(q, k, v, bias=bias)
    assert np.allclose(np.asarray(got), want, atol=1e-10)


# --- math properties -----------------------------------------------------


def test_output_is_convex_combination_of_values():
    rng = np.random.RandomState(11)
    q = jnp.asarray(rng.standard_normal((2, 3, 5, 4)))
    k = jnp.asarray(rng.standard_normal((2, 3, 6, 4)))
    v = jnp.asarray(rng.standard_normal((2, 3, 6, 4)))
    out = np.asarray(ref_sdpa(q, k, v))
    vmin = np.asarray(v.min(axis=2))[:, :, None, :]
    vmax = np.asarray(v.max(axis=2))[:, :, None, :]
    assert (out >= vmin - 1e-6).all()
    assert (out <= vmax + 1e-6).all()


def test_zero_query_gives_mean_value():
    rng = np.random.RandomState(12)
    v = jnp.asarray(rng.standard_normal((2, 2, 7, 3)))
    q = jnp.zeros((2, 2, 4, 5))
    k = jnp.asarray(rng.standard_normal((2, 2, 7, 5)))
    out = ref_sdpa(q, k, v)  # logits all 0 -> uniform -> mean over t
    mean = jnp.broadcast_to(v.mean(axis=2, keepdims=True), out.shape)
    assert np.allclose(np.asarray(out), np.asarray(mean), atol=1e-6)


def test_causal_equals_tril_mask():
    rng = np.random.RandomState(13)
    q = jnp.asarray(rng.standard_normal((2, 2, 6, 4)))
    k = jnp.asarray(rng.standard_normal((2, 2, 6, 4)))
    v = jnp.asarray(rng.standard_normal((2, 2, 6, 3)))
    via_causal = ref_sdpa(q, k, v, causal=True)
    via_mask = ref_sdpa(q, k, v, mask=jnp.tril(jnp.ones((6, 6), bool)))
    assert np.allclose(
        np.asarray(via_causal), np.asarray(via_mask), atol=1e-12
    )


# --- hypothesis properties ----------------------------------------------


@given(n=st.integers(2, 7), seed=st.integers(0, 2**16))
def test_causal_equiv_tril_property(n, seed):
    rng = np.random.RandomState(seed)
    d = 4
    q = jnp.asarray(rng.standard_normal((1, 2, n, d)))
    k = jnp.asarray(rng.standard_normal((1, 2, n, d)))
    v = jnp.asarray(rng.standard_normal((1, 2, n, 3)))
    a = ref_sdpa(q, k, v, causal=True)
    b = ref_sdpa(q, k, v, mask=jnp.tril(jnp.ones((n, n), bool)))
    assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-10)


@given(seed=st.integers(0, 2**16))
def test_perm_equivariance_over_keys_property(seed):
    rng = np.random.RandomState(seed)
    t, d = 6, 4
    q = jnp.asarray(rng.standard_normal((1, 1, 4, d)))
    k = jnp.asarray(rng.standard_normal((1, 1, t, d)))
    v = jnp.asarray(rng.standard_normal((1, 1, t, 3)))
    perm = rng.permutation(t)
    base = ref_sdpa(q, k, v)
    permuted = ref_sdpa(q, k[:, :, perm], v[:, :, perm])
    assert np.allclose(np.asarray(base), np.asarray(permuted), atol=1e-10)


@given(seed=st.integers(0, 2**16))
def test_masking_a_key_equals_deleting_it_property(seed):
    # Masking a key to False must give the same output as removing it.
    rng = np.random.RandomState(seed)
    t, d = 6, 4
    q = jnp.asarray(rng.standard_normal((1, 1, 3, d)))
    k = jnp.asarray(rng.standard_normal((1, 1, t, d)))
    v = jnp.asarray(rng.standard_normal((1, 1, t, 2)))
    keep = np.ones((1, 1, 3, t), bool)
    keep[..., t - 1] = False  # drop the last key for every query
    masked = ref_sdpa(q, k, v, mask=jnp.asarray(keep))
    deleted = ref_sdpa(q, k[:, :, : t - 1], v[:, :, : t - 1])
    assert np.allclose(np.asarray(masked), np.asarray(deleted), atol=1e-10)


# --- autodiff ------------------------------------------------------------


def test_autodiff_gradient_matches_finite_difference():
    rng = np.random.RandomState(21)
    q = jnp.asarray(rng.standard_normal((1, 1, 3, 3)))
    k = jnp.asarray(rng.standard_normal((1, 1, 4, 3)))
    v = jnp.asarray(rng.standard_normal((1, 1, 4, 2)))
    bias = jnp.asarray(rng.standard_normal((1, 1, 3, 4)))

    def loss(q, k, v, bias):
        return jnp.sum(ref_sdpa(q, k, v, bias=bias) ** 2)

    grads = jax.grad(loss, argnums=(0, 1, 2, 3))(q, k, v, bias)
    assert all(np.all(np.isfinite(np.asarray(g))) for g in grads)

    # central finite differences on q (x64 active -> accurate).
    eps = 1e-6
    gq = np.asarray(grads[0])
    qn = np.asarray(q)
    fd = np.zeros_like(qn)
    for idx in np.ndindex(qn.shape):
        qp = qn.copy()
        qp[idx] += eps
        qm = qn.copy()
        qm[idx] -= eps
        fp = float(loss(jnp.asarray(qp), k, v, bias))
        fm = float(loss(jnp.asarray(qm), k, v, bias))
        fd[idx] = (fp - fm) / (2 * eps)
    assert np.allclose(gq, fd, atol=1e-4, rtol=1e-4)


# --- dispatch & loud fallback -------------------------------------------


def test_backend_jax_byte_identical_to_reference():
    rng = np.random.RandomState(31)
    q = jnp.asarray(rng.standard_normal((2, 2, 5, 4)))
    k = jnp.asarray(rng.standard_normal((2, 2, 5, 4)))
    v = jnp.asarray(rng.standard_normal((2, 2, 5, 3)))
    a = scaled_dot_product_attention(q, k, v, backend='jax')
    b = ref_sdpa(q, k, v)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_auto_on_cpu_uses_reference_without_warning():
    if _HAS_AMPERE_NVIDIA:
        pytest.skip('CPU-path test; GPU host resolves auto to pallas-cuda')
    rng = np.random.RandomState(32)
    q = jnp.asarray(rng.standard_normal((1, 2, 4, 4)))
    k = jnp.asarray(rng.standard_normal((1, 2, 4, 4)))
    v = jnp.asarray(rng.standard_normal((1, 2, 4, 3)))
    reset_fallback_state()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = scaled_dot_product_attention(q, k, v, backend='auto')
        fired = [
            x for x in rec if issubclass(x.category, NitrixBackendFallback)
        ]
    assert np.array_equal(np.asarray(out), np.asarray(ref_sdpa(q, k, v)))
    assert len(fired) == 0


def test_explicit_pallas_on_cpu_raises():
    if _HAS_AMPERE_NVIDIA:
        pytest.skip('CPU-only: explicit pallas-cuda errors without Ampere')
    q = jnp.zeros((1, 1, 4, 4))
    with pytest.raises(NitrixBackendError):
        scaled_dot_product_attention(q, q, q, backend='pallas-cuda')


@pallas_only
def test_pallas_falls_back_loudly_until_phase2():
    # Phase 1: no fused kernel yet -> pallas-cuda falls back loudly to jax.
    rng = np.random.RandomState(33)
    q = jnp.asarray(rng.standard_normal((1, 2, 8, 8)))
    k = jnp.asarray(rng.standard_normal((1, 2, 8, 8)))
    v = jnp.asarray(rng.standard_normal((1, 2, 8, 4)))
    reset_fallback_state()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = scaled_dot_product_attention(q, k, v, backend='pallas-cuda')
        fired = [
            x for x in rec if issubclass(x.category, NitrixBackendFallback)
        ]
    assert len(fired) == 1
    assert np.allclose(
        np.asarray(out), np.asarray(ref_sdpa(q, k, v)), atol=1e-12
    )


def test_jit_compiles_and_matches():
    rng = np.random.RandomState(34)
    q = jnp.asarray(rng.standard_normal((2, 2, 6, 4)))
    k = jnp.asarray(rng.standard_normal((2, 2, 6, 4)))
    v = jnp.asarray(rng.standard_normal((2, 2, 6, 3)))
    ref = ref_sdpa(q, k, v, causal=True)
    out = jax.jit(
        lambda q, k, v: scaled_dot_product_attention(
            q, k, v, causal=True, backend='jax'
        )
    )(q, k, v)
    assert np.allclose(np.asarray(out), np.asarray(ref), atol=1e-12)


def test_validation_errors():
    q = jnp.zeros((1, 1, 4, 8))
    with pytest.raises(ValueError):
        scaled_dot_product_attention(
            q, jnp.zeros((1, 1, 4, 7)), jnp.zeros((1, 1, 4, 8))
        )
    with pytest.raises(ValueError):
        scaled_dot_product_attention(
            q, jnp.zeros((1, 1, 5, 8)), jnp.zeros((1, 1, 4, 8))
        )
