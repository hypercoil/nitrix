# -*- coding: utf-8 -*-
"""The fp32-accumulation invariant for the Bucket-A NN forward kernels.

P1 of the mixed-precision strategy
(``docs/feature-requests/mixed-precision-strategy.md``), enforcing SPEC §2
tenet 11 on ``nn.attention`` / ``nn.ssm`` / ``nn.norm``: a float16/bfloat16 op
is computed by upcasting to float32, running the whole op in float32, and
casting back at the I/O boundary -- so the reduced-precision output is the
float32 oracle quantised to the I/O dtype (platform-stable; no bf16 tensor-core
multiply variance).

Pinned here:

* **the invariant** -- ``ref(reduced)`` is bit-for-bit ``ref(reduced→f32)``
  cast back to the reduced dtype, across every family and SSM driver;
* **dtype** -- output dtype == input dtype, and float64 is preserved (not
  floored to float32);
* **no regression** -- the float32 path is byte-identical (the existing fp32
  golden suite still passes untouched);
* **load-bearing** -- a naive reduced-precision reduction drifts from the floor;
* **golden corpus** -- reduced precision stays within the published per-dtype
  tolerance of the checked-in float32 golden (reusing it as the oracle; the
  invariant makes the reduced output a deterministic quantisation of it);
* **jit / autodiff** behave in reduced precision.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from _golden import load_golden, tol  # noqa: E402

from nitrix.nn.attention import (  # noqa: E402
    reference_scaled_dot_product_attention as _attn,
)
from nitrix.nn.norm import (  # noqa: E402
    reference_group_norm as _gn,
)
from nitrix.nn.norm import (
    reference_instance_norm as _in,
)
from nitrix.nn.norm import (
    reference_layer_norm as _ln,
)
from nitrix.nn.ssm import reference_selective_scan as _ssm  # noqa: E402

REDUCED = [jnp.bfloat16, jnp.float16]
SSM_DRIVERS = ['sequential', 'associative', 'chunked']


def _f32(seed, *shape):
    return jnp.asarray(
        np.random.RandomState(seed).standard_normal(shape), jnp.float32
    )


def _delta(seed, *shape):
    # post-softplus (positive) Δ, as the model feeds it.
    z = np.random.RandomState(seed).standard_normal(shape)
    return jnp.asarray(np.log1p(np.exp(z)), jnp.float32)


def _cast(tree, dt):
    return {k: v.astype(dt) for k, v in tree.items()}


def _to_f32(tree):
    return {k: v.astype(jnp.float32) for k, v in tree.items()}


# --- builders: fp32 input dict + a runner that consumes that dict -----------


def _attn_in():
    return dict(
        q=_f32(0, 2, 2, 6, 8), k=_f32(1, 2, 2, 6, 8), v=_f32(2, 2, 2, 6, 8)
    )


def _attn_run(d, **kw):
    return _attn(d['q'], d['k'], d['v'], **kw)


def _ssm_in():
    return dict(
        x=_f32(3, 2, 16, 4),
        delta=_delta(4, 2, 16, 4),
        A=-jnp.exp(_f32(5, 4, 3)),
        B=_f32(6, 2, 16, 3),
        C=_f32(7, 2, 16, 3),
        D=_f32(8, 4),
    )


def _ssm_run(d, **kw):
    return _ssm(d['x'], d['delta'], d['A'], d['B'], d['C'], d['D'], **kw)


def _ln_in():
    return dict(x=_f32(9, 2, 6, 64), weight=_f32(10, 64), bias=_f32(11, 64))


def _ln_run(d, **kw):
    return _ln(d['x'], d['weight'], d['bias'], **kw)


def _gn_in():
    return dict(x=_f32(12, 2, 8, 4, 4), weight=_f32(13, 8), bias=_f32(14, 8))


def _gn_run(d, **kw):
    return _gn(d['x'], 4, d['weight'], d['bias'], **kw)


def _in_run(d, **kw):
    return _in(d['x'], d['weight'], d['bias'], **kw)


# --- the invariant: reduced == float32-path cast down (bit-for-bit) ---------


def _check_invariant(inputs, run, dt, **kw):
    red_in = _cast(inputs, dt)
    out = run(red_in, **kw)
    # The float32 oracle on the *same* reduced-quantised inputs, cast back.
    oracle = run(_to_f32(red_in), **kw).astype(dt)
    assert out.dtype == dt
    assert np.array_equal(
        np.asarray(out, np.float32), np.asarray(oracle, np.float32)
    ), f'{run.__name__} not fp32-accumulated for {dt.__name__}'


@pytest.mark.parametrize('dt', REDUCED)
def test_invariant_attention(dt):
    _check_invariant(_attn_in(), _attn_run, dt)


@pytest.mark.parametrize('dt', REDUCED)
def test_invariant_attention_causal_qk_norm(dt):
    _check_invariant(_attn_in(), _attn_run, dt, causal=True, qk_norm=True)


@pytest.mark.parametrize('dt', REDUCED)
@pytest.mark.parametrize('driver', SSM_DRIVERS)
def test_invariant_ssm(dt, driver):
    _check_invariant(_ssm_in(), _ssm_run, dt, driver=driver, chunk_size=8)


@pytest.mark.parametrize('dt', REDUCED)
def test_invariant_layer_norm(dt):
    _check_invariant(_ln_in(), _ln_run, dt, out_scale=0.5)


@pytest.mark.parametrize('dt', REDUCED)
def test_invariant_group_norm(dt):
    _check_invariant(_gn_in(), _gn_run, dt)


@pytest.mark.parametrize('dt', REDUCED)
def test_invariant_instance_norm(dt):
    _check_invariant(_gn_in(), _in_run, dt)


# --- dtype preservation: float64 is NOT floored to float32 ------------------


def test_float64_preserved():
    d = {k: v.astype(jnp.float64) for k, v in _attn_in().items()}
    assert _attn(d['q'], d['k'], d['v']).dtype == jnp.float64
    s = {k: v.astype(jnp.float64) for k, v in _ssm_in().items()}
    out = _ssm(s['x'], s['delta'], s['A'], s['B'], s['C'], s['D'])
    assert out.dtype == jnp.float64
    n = {k: v.astype(jnp.float64) for k, v in _ln_in().items()}
    assert _ln(n['x'], n['weight'], n['bias']).dtype == jnp.float64


def test_float32_unchanged_dtype():
    # The floor is a no-op for float32 (acc_dtype == io_dtype).
    d = _attn_in()
    assert _attn(d['q'], d['k'], d['v']).dtype == jnp.float32
    assert _ln_run(_ln_in()).dtype == jnp.float32


# --- the floor is load-bearing ---------------------------------------------


def test_floor_is_non_vacuous_layer_norm():
    # Long feature axis + large magnitude: a bf16 reduction drifts materially
    # from the fp32-floored reference, so the floor changes the answer.
    x = _f32(20, 1, 1, 4096) * 30.0
    floored = np.asarray(_ln(x.astype(jnp.bfloat16)), np.float32)
    xb = x.astype(jnp.bfloat16)
    mean_b = xb.mean(-1, keepdims=True)
    var_b = xb.var(-1, keepdims=True)
    naive = np.asarray(
        ((xb - mean_b) * jax.lax.rsqrt(var_b + 1e-5)).astype(jnp.bfloat16),
        np.float32,
    )
    assert np.max(np.abs(floored - naive)) > 1e-3


# --- golden corpus: reduced precision within the published tolerance --------


@pytest.mark.parametrize('dt', REDUCED)
def test_golden_attention_reduced(dt):
    d = load_golden('attention_dense_float32')
    out = _attn(
        jnp.asarray(d['q']).astype(dt),
        jnp.asarray(d['k']).astype(dt),
        jnp.asarray(d['v']).astype(dt),
    )
    atol, rtol = tol('attention', dt)
    np.testing.assert_allclose(
        np.asarray(out, np.float32), d['out'], atol=atol, rtol=rtol
    )


@pytest.mark.parametrize('dt', REDUCED)
def test_golden_selective_scan_reduced(dt):
    d = load_golden('selective_scan_float32')
    out = _ssm(
        jnp.asarray(d['x']).astype(dt),
        jnp.asarray(d['delta']).astype(dt),
        jnp.asarray(d['A']).astype(dt),
        jnp.asarray(d['B']).astype(dt),
        jnp.asarray(d['C']).astype(dt),
        jnp.asarray(d['D']).astype(dt),
        driver='sequential',
    )
    atol, rtol = tol('selective_scan', dt)
    np.testing.assert_allclose(
        np.asarray(out, np.float32), d['out'], atol=atol, rtol=rtol
    )


@pytest.mark.parametrize('dt', REDUCED)
@pytest.mark.parametrize(
    'name,op,run',
    [
        (
            'layer_norm_float32',
            'layer_norm',
            lambda d, dt: _ln(
                jnp.asarray(d['x']).astype(dt),
                jnp.asarray(d['weight']).astype(dt),
                jnp.asarray(d['bias']).astype(dt),
                out_scale=0.5,
            ),
        ),
        (
            'group_norm_float32',
            'group_norm',
            lambda d, dt: _gn(
                jnp.asarray(d['x']).astype(dt),
                4,
                jnp.asarray(d['weight']).astype(dt),
                jnp.asarray(d['bias']).astype(dt),
            ),
        ),
        (
            'instance_norm_float32',
            'instance_norm',
            lambda d, dt: _in(
                jnp.asarray(d['x']).astype(dt),
                jnp.asarray(d['weight']).astype(dt),
                jnp.asarray(d['bias']).astype(dt),
            ),
        ),
    ],
)
def test_golden_norm_reduced(name, op, run, dt):
    d = load_golden(name)
    out = run(d, dt)
    atol, rtol = tol(op, dt)
    np.testing.assert_allclose(
        np.asarray(out, np.float32), d['out'], atol=atol, rtol=rtol
    )


# --- jit + autodiff in reduced precision ------------------------------------


@pytest.mark.parametrize('dt', REDUCED)
def test_jit_matches_eager_reduced(dt):
    d = _cast(_ln_in(), dt)
    eager = _ln(d['x'], d['weight'], d['bias'], out_scale=0.5)
    jitted = jax.jit(lambda x, w, b: _ln(x, w, b, out_scale=0.5))(
        d['x'], d['weight'], d['bias']
    )
    assert jitted.dtype == dt
    assert np.array_equal(
        np.asarray(jitted, np.float32), np.asarray(eager, np.float32)
    )


@pytest.mark.parametrize('dt', REDUCED)
def test_autodiff_finite_reduced(dt):
    d = _cast(_ln_in(), dt)

    def loss(x):
        return jnp.sum(
            _ln(x, d['weight'], d['bias'], out_scale=0.5).astype(jnp.float32)
            ** 2
        )

    g = jax.grad(loss)(d['x'])
    assert g.dtype == dt
    assert np.all(np.isfinite(np.asarray(g, np.float32)))
