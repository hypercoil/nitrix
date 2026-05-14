# -*- coding: utf-8 -*-
"""Tests for ``nitrix.semiring.semiring_conv``.

Coverage:

- REAL conv matches ``lax.conv_general_dilated`` for 1D / 2D / 3D
  with VALID and SAME padding, strides, and dilation.
- Tropical, log, euclidean conv match naive sliding-window
  reductions on small inputs.
- Identity propagation: ``-inf`` in TROPICAL_MAX_PLUS / LOG
  propagates through the conv without producing NaN.
- Backward composes correctly through ``semiring_matmul``'s VJP
  (finite-difference check at fp64).
- ``backend="pallas-cuda"`` falls back to JAX with one warning per
  shape signature (current GA state: no native Pallas conv kernel).
"""
from __future__ import annotations

import warnings

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from nitrix._internal.backend import (
    NitrixBackendFallback,
    _HAS_AMPERE_NVIDIA,
    reset_fallback_state,
)
from nitrix.semiring import (
    EUCLIDEAN,
    LOG,
    REAL,
    TROPICAL_MAX_PLUS,
    TROPICAL_MIN_PLUS,
    semiring_conv,
)


jax.config.update('jax_enable_x64', True)


pallas_only = pytest.mark.skipif(
    not _HAS_AMPERE_NVIDIA,
    reason='requires NVIDIA Ampere+ for Pallas Triton backend',
)


# ---------------------------------------------------------------------------
# REAL conv vs lax
# ---------------------------------------------------------------------------


def _lax_conv_real(x, k, *, stride, padding, dilation, spatial_rank):
    '''Reference REAL convolution via ``lax.conv_general_dilated``.

    Channel-last in / out; internal NCHW transpose to match lax.
    '''
    x_nchw = jnp.moveaxis(x, -1, 1)
    # k: (*kspatial, c_in, c_out) -> (c_out, c_in, *kspatial)
    k_lax = jnp.moveaxis(jnp.moveaxis(k, -1, 0), -1, 1)
    strides = (
        (stride,) * spatial_rank if isinstance(stride, int)
        else tuple(stride)
    )
    dilations = (
        (dilation,) * spatial_rank if isinstance(dilation, int)
        else tuple(dilation)
    )
    out_nchw = lax.conv_general_dilated(
        x_nchw, k_lax,
        window_strides=strides,
        padding=padding if isinstance(padding, str) else tuple(padding),
        rhs_dilation=dilations,
    )
    return jnp.moveaxis(out_nchw, 1, -1)


def test_real_conv1d_matches_lax():
    x = jax.random.normal(jax.random.key(0), (2, 16, 3))
    k = jax.random.normal(jax.random.key(1), (3, 3, 4))
    for pad in ('SAME', 'VALID'):
        got = semiring_conv(x, k, semiring=REAL, padding=pad, backend='jax')
        ref = _lax_conv_real(
            x, k, stride=1, padding=pad, dilation=1, spatial_rank=1,
        )
        np.testing.assert_allclose(got, ref, atol=1e-10, rtol=1e-10)


def test_real_conv2d_matches_lax():
    x = jax.random.normal(jax.random.key(2), (1, 8, 8, 3))
    k = jax.random.normal(jax.random.key(3), (3, 3, 3, 4))
    for pad in ('SAME', 'VALID'):
        got = semiring_conv(x, k, semiring=REAL, padding=pad, backend='jax')
        ref = _lax_conv_real(
            x, k, stride=1, padding=pad, dilation=1, spatial_rank=2,
        )
        np.testing.assert_allclose(got, ref, atol=1e-10, rtol=1e-10)


def test_real_conv3d_matches_lax():
    x = jax.random.normal(jax.random.key(4), (1, 6, 6, 6, 2))
    k = jax.random.normal(jax.random.key(5), (3, 3, 3, 2, 3))
    got = semiring_conv(x, k, semiring=REAL, padding='VALID', backend='jax')
    ref = _lax_conv_real(
        x, k, stride=1, padding='VALID', dilation=1, spatial_rank=3,
    )
    np.testing.assert_allclose(got, ref, atol=1e-10, rtol=1e-10)


def test_real_conv2d_strided():
    x = jax.random.normal(jax.random.key(6), (1, 10, 10, 2))
    k = jax.random.normal(jax.random.key(7), (3, 3, 2, 4))
    got = semiring_conv(
        x, k, semiring=REAL, stride=2, padding='SAME', backend='jax',
    )
    ref = _lax_conv_real(
        x, k, stride=2, padding='SAME', dilation=1, spatial_rank=2,
    )
    np.testing.assert_allclose(got, ref, atol=1e-10, rtol=1e-10)


def test_real_conv2d_dilated():
    x = jax.random.normal(jax.random.key(8), (1, 16, 16, 2))
    k = jax.random.normal(jax.random.key(9), (3, 3, 2, 4))
    got = semiring_conv(
        x, k, semiring=REAL, dilation=2, padding='VALID', backend='jax',
    )
    ref = _lax_conv_real(
        x, k, stride=1, padding='VALID', dilation=2, spatial_rank=2,
    )
    np.testing.assert_allclose(got, ref, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# Tropical / log / euclidean conv vs naive
# ---------------------------------------------------------------------------


def test_tropical_max_plus_conv1d_zero_kernel_is_max_window():
    # Sliding max with kernel = 0 across width: window-max of x.
    x = jnp.array([[[1.], [2.], [3.], [2.], [1.], [0.], [4.], [3.]]])
    k = jnp.zeros((3, 1, 1))
    got = semiring_conv(
        x, k, semiring=TROPICAL_MAX_PLUS, padding='VALID', backend='jax',
    )
    xnp = np.asarray(x[0, :, 0])
    expected = np.array([float(max(xnp[i:i + 3])) for i in range(6)])
    np.testing.assert_allclose(got[0, :, 0], expected)


def test_tropical_min_plus_conv1d_zero_kernel_is_min_window():
    x = jnp.array([[[1.], [2.], [3.], [2.], [1.], [0.], [4.], [3.]]])
    k = jnp.zeros((3, 1, 1))
    got = semiring_conv(
        x, k, semiring=TROPICAL_MIN_PLUS, padding='VALID', backend='jax',
    )
    xnp = np.asarray(x[0, :, 0])
    expected = np.array([float(min(xnp[i:i + 3])) for i in range(6)])
    np.testing.assert_allclose(got[0, :, 0], expected)


def test_log_conv1d_matches_logsumexp_window():
    x = jax.random.normal(jax.random.key(0), (1, 8, 1))
    k = jnp.zeros((3, 1, 1))
    got = semiring_conv(x, k, semiring=LOG, padding='VALID', backend='jax')
    xnp = np.asarray(x[0, :, 0])
    expected = np.array([
        float(jax.scipy.special.logsumexp(xnp[i:i + 3])) for i in range(6)
    ])
    np.testing.assert_allclose(got[0, :, 0], expected, atol=1e-10, rtol=1e-10)


def test_euclidean_conv1d_matches_naive():
    x = jax.random.normal(jax.random.key(11), (1, 6, 1))
    k = jax.random.normal(jax.random.key(12), (3, 1, 1))
    got = semiring_conv(
        x, k, semiring=EUCLIDEAN, padding='VALID', backend='jax',
    )
    xnp = np.asarray(x[0, :, 0])
    knp = np.asarray(k[:, 0, 0])
    expected = np.array(
        [np.sqrt(np.sum((xnp[i:i + 3] - knp) ** 2)) for i in range(4)]
    )
    np.testing.assert_allclose(got[0, :, 0], expected, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# Identity propagation
# ---------------------------------------------------------------------------


def test_neg_inf_in_tropical_max_plus_propagates():
    # If x[0, 3, 0] = -inf and the kernel sees position 3 in every output
    # output position that overlaps it, those positions must come out
    # as -inf (since max + -inf = -inf for k_p == 0).
    x = jnp.ones((1, 8, 1))
    x = x.at[0, 3, 0].set(-jnp.inf)
    k = jnp.zeros((3, 1, 1))
    got = semiring_conv(
        x, k, semiring=TROPICAL_MAX_PLUS, padding='VALID', backend='jax',
    )
    # With a 3-window VALID conv, position 3 of x is reached by output
    # positions {1, 2, 3}; all other positions should match max(x[i:i+3]).
    # But TROPICAL_MAX_PLUS with k_p=0: max(x[i:i+3] + 0) = max(x[i:i+3]).
    # max of -inf and finite is finite -- so the -inf does NOT actually
    # contaminate the max.  The right invariant is "no NaN".
    assert bool(~jnp.any(jnp.isnan(got)))


def test_neg_inf_in_log_does_not_nan():
    x = jnp.zeros((1, 8, 1))
    x = x.at[0, 3, 0].set(-jnp.inf)
    k = jnp.zeros((3, 1, 1))
    got = semiring_conv(
        x, k, semiring=LOG, padding='VALID', backend='jax',
    )
    assert bool(~jnp.any(jnp.isnan(got)))


# ---------------------------------------------------------------------------
# Backward via finite-difference (composition through semiring_matmul VJP)
# ---------------------------------------------------------------------------


def _finite_diff(fn, x, eps=1e-5):
    '''JIT the loss once so per-element calls hit the trace cache.'''
    jit_fn = jax.jit(fn)
    x_flat = x.reshape(-1)
    out = np.zeros_like(np.asarray(x_flat, dtype=np.float64))
    for i in range(x_flat.size):
        e = jnp.zeros_like(x_flat).at[i].set(eps)
        e = e.reshape(x.shape)
        f_plus = float(jit_fn(x + e))
        f_minus = float(jit_fn(x - e))
        out[i] = (f_plus - f_minus) / (2 * eps)
    return out.reshape(x.shape)


def _conv_loss(x, k, semiring, pad='SAME'):
    return jnp.sum(
        semiring_conv(x, k, semiring=semiring, padding=pad, backend='jax')
        ** 2
    )


def test_real_conv_grad_matches_fd():
    x = jax.random.normal(jax.random.key(20), (1, 5, 2))
    k = jax.random.normal(jax.random.key(21), (3, 2, 2))
    gx = jax.grad(_conv_loss, argnums=0)(x, k, REAL)
    gx_fd = _finite_diff(lambda x: _conv_loss(x, k, REAL), x)
    np.testing.assert_allclose(gx, gx_fd, atol=1e-6, rtol=1e-6)
    gk = jax.grad(_conv_loss, argnums=1)(x, k, REAL)
    gk_fd = _finite_diff(lambda kk: _conv_loss(x, kk, REAL), k)
    np.testing.assert_allclose(gk, gk_fd, atol=1e-6, rtol=1e-6)


def test_log_conv_grad_matches_fd():
    x = jax.random.normal(jax.random.key(22), (1, 5, 2))
    k = jax.random.normal(jax.random.key(23), (3, 2, 2))
    gx = jax.grad(_conv_loss, argnums=0)(x, k, LOG)
    gx_fd = _finite_diff(lambda x: _conv_loss(x, k, LOG), x)
    np.testing.assert_allclose(gx, gx_fd, atol=1e-6, rtol=1e-6)


def test_tropical_max_plus_conv_grad_matches_fd():
    x = jax.random.normal(jax.random.key(24), (1, 5, 2))
    k = jax.random.normal(jax.random.key(25), (3, 2, 2))
    gx = jax.grad(_conv_loss, argnums=0)(x, k, TROPICAL_MAX_PLUS)
    gx_fd = _finite_diff(
        lambda x: _conv_loss(x, k, TROPICAL_MAX_PLUS), x, eps=1e-4,
    )
    np.testing.assert_allclose(gx, gx_fd, atol=1e-6, rtol=1e-6)


def test_euclidean_conv_grad_matches_fd():
    x = jax.random.normal(jax.random.key(26), (1, 5, 2))
    k = jax.random.normal(jax.random.key(27), (3, 2, 2))
    gx = jax.grad(_conv_loss, argnums=0)(x, k, EUCLIDEAN)
    gx_fd = _finite_diff(lambda x: _conv_loss(x, k, EUCLIDEAN), x)
    np.testing.assert_allclose(gx, gx_fd, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Backend fallback observability
# ---------------------------------------------------------------------------


@pallas_only
def test_pallas_conv_falls_back_with_warning():
    reset_fallback_state()
    # Shape chosen so the inner matmul *can* tile in Pallas (so we get
    # exactly one warning, from the conv layer).  Patches will be
    # (1*64*64, 4*8) = (4096, 32) and kernel reshape (32, 16) -- all
    # divisible by the Pallas block-size candidates {128, 64, 32, 16}.
    x = jax.random.normal(jax.random.key(30), (1, 64, 64, 8))
    k = jax.random.normal(jax.random.key(31), (2, 2, 8, 16))
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        out = semiring_conv(
            x, k, semiring=REAL, padding='SAME', backend='pallas-cuda',
        )
        n = sum(1 for w in ws if w.category is NitrixBackendFallback)
    # Conv layer warns once; inner matmul tiles cleanly so no second
    # warning.
    assert n == 1, f'expected 1 fallback warning, got {n}'
    # Output still matches the all-JAX reference.
    ref = semiring_conv(
        x, k, semiring=REAL, padding='SAME', backend='jax',
    )
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)


@pallas_only
def test_pallas_conv_emits_inner_warning_when_matmul_cant_tile():
    reset_fallback_state()
    # Shape where the inner matmul *cannot* tile: K = c_in * prod(kspatial)
    # not divisible by any of {32, 16, 8}.  Expect TWO warnings: one
    # from the conv layer (no native conv kernel), one from the matmul
    # layer (no viable tile).  Both legitimate observability events.
    x = jax.random.normal(jax.random.key(40), (1, 8, 8, 2))
    k = jax.random.normal(jax.random.key(41), (3, 3, 2, 4))  # K = 18
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        out = semiring_conv(
            x, k, semiring=REAL, padding='SAME', backend='pallas-cuda',
        )
        n = sum(1 for w in ws if w.category is NitrixBackendFallback)
    assert n == 2, f'expected 2 fallback warnings (conv + matmul), got {n}'


# ---------------------------------------------------------------------------
# Batched inputs
# ---------------------------------------------------------------------------


def test_batched_conv2d():
    x = jax.random.normal(jax.random.key(40), (3, 6, 6, 2))
    k = jax.random.normal(jax.random.key(41), (3, 3, 2, 4))
    got = semiring_conv(x, k, semiring=REAL, padding='SAME', backend='jax')
    assert got.shape == (3, 6, 6, 4)
    # Reference: process each batch element independently.
    refs = []
    for i in range(3):
        refs.append(_lax_conv_real(
            x[i:i + 1], k, stride=1, padding='SAME',
            dilation=1, spatial_rank=2,
        )[0])
    ref = jnp.stack(refs, axis=0)
    np.testing.assert_allclose(got, ref, atol=1e-10, rtol=1e-10)
