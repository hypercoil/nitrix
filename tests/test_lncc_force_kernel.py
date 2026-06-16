# -*- coding: utf-8 -*-
"""v4 Phase 5b / L2b: the fused sliding-window centre-only LNCC-force kernel.

The kernel (ITK scanning-window: scan x carrying the five window sums, drop the
trailing plane / add the leading one) is **pure perf** -- tolerance-equal to the
JAX ``LNCCForce(derivative='center').update`` (boundary included, via the one
symmetric pad), and it beats the JAX integral image above ~128^3 (the win grows
to ~4x at 256^3).  These tests pin the parity, the dispatch (default ``'jax'``
byte-identical; ``'pallas-cuda'`` opts in; ``'exact'`` never uses the kernel),
the **size gate** (small volumes defer to JAX so the coarse pyramid levels are
not slowed), and the loud fallbacks.  GPU-only.
"""

from __future__ import annotations

import warnings

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nitrix._internal.backend import (  # noqa: E402
    _HAS_AMPERE_NVIDIA,
    NitrixBackendFallback,
    reset_fallback_state,
)
from nitrix._kernels.cuda.lncc_force import (  # noqa: E402
    lncc_center_force_pallas,
)
from nitrix.register._force import LNCCForce  # noqa: E402

pallas_only = pytest.mark.skipif(
    not _HAS_AMPERE_NVIDIA,
    reason='requires NVIDIA Ampere+ for the Pallas Triton backend',
)


def _pair(shape, seed=0, dtype=np.float64):
    rng = np.random.RandomState(seed)
    f = jnp.asarray(rng.standard_normal(shape).astype(dtype))
    w = jnp.asarray(rng.standard_normal(shape).astype(dtype))
    return w, f


def _jax_center(w, f, r=2):
    return LNCCForce(radius=r, derivative='center').bind(f, ndim=w.ndim).update(w)


@pallas_only
@pytest.mark.parametrize('shape', [(32, 32, 32), (40, 48, 56)])
def test_kernel_parity_with_jax_center(shape):
    # fp64: the kernel reproduces the JAX centre force to machine precision,
    # boundary included (the single symmetric pad doubles as the edge pad).
    w, f = _pair(shape, seed=1)
    ref = _jax_center(w, f)
    got = lncc_center_force_pallas(w, f, radius=2, eps=1e-5)
    scale = float(jnp.max(jnp.abs(ref))) + 1e-30
    assert float(jnp.max(jnp.abs(got - ref))) / scale < 1e-12


@pallas_only
def test_kernel_parity_fp32():
    w, f = _pair((48, 48, 48), seed=2, dtype=np.float32)
    ref = _jax_center(w, f)
    got = lncc_center_force_pallas(w, f, radius=2, eps=1e-5)
    scale = float(jnp.max(jnp.abs(ref))) + 1e-30
    assert float(jnp.max(jnp.abs(got - ref))) / scale < 1e-5


@pallas_only
def test_dispatch_uses_kernel_above_size_gate():
    # A large enough volume (>= ~128^3) goes through the kernel and is
    # tolerance-equal to the JAX centre force.
    w, f = _pair((128, 128, 128), seed=3, dtype=np.float32)
    u_jax = LNCCForce(
        radius=2, derivative='center', backend='jax'
    ).bind(f, ndim=3).update(w)
    u_pal = LNCCForce(
        radius=2, derivative='center', backend='pallas-cuda'
    ).bind(f, ndim=3).update(w)
    scale = float(jnp.max(jnp.abs(u_jax))) + 1e-30
    assert float(jnp.max(jnp.abs(u_pal - u_jax))) / scale < 1e-5


@pallas_only
def test_size_gate_defers_small_volume_to_jax():
    # Below the gate the kernel must NOT fire (it would be slower) -- the
    # pallas-cuda path returns exactly the JAX result, silently.
    w, f = _pair((32, 32, 32), seed=4, dtype=np.float32)
    u_jax = LNCCForce(
        radius=2, derivative='center', backend='jax'
    ).bind(f, ndim=3).update(w)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        u_pal = LNCCForce(
            radius=2, derivative='center', backend='pallas-cuda'
        ).bind(f, ndim=3).update(w)
        fired = [x for x in rec if issubclass(x.category, NitrixBackendFallback)]
    assert np.array_equal(np.asarray(u_pal), np.asarray(u_jax))  # JAX path
    assert len(fired) == 0  # the size gate is a silent optimisation, not a warn


@pallas_only
def test_exact_derivative_never_uses_kernel():
    # backend='pallas-cuda' with derivative='exact' is a no-op (no kernel).
    w, f = _pair((128, 128, 128), seed=5, dtype=np.float32)
    u_ex = LNCCForce(
        radius=2, derivative='exact', backend='pallas-cuda'
    ).bind(f, ndim=3).update(w)
    u_exj = LNCCForce(
        radius=2, derivative='exact', backend='jax'
    ).bind(f, ndim=3).update(w)
    assert np.array_equal(np.asarray(u_ex), np.asarray(u_exj))


@pallas_only
@pytest.mark.parametrize('shape', [(50, 50, 50), (44, 52, 60), (55, 47, 63)])
def test_non_tileable_size_pads_and_matches(shape):
    # An untileable (y, z) extent is padded up to a boundary-safe tileable
    # size, run, and cropped -- so the kernel handles arbitrary volumes and is
    # still tolerance-equal to JAX (the win holds at scale; here just parity).
    w, f = _pair(shape, seed=6)
    ref = _jax_center(w, f)
    got = lncc_center_force_pallas(w, f, radius=2, eps=1e-5)
    scale = float(jnp.max(jnp.abs(ref))) + 1e-30
    assert float(jnp.max(jnp.abs(got - ref))) / scale < 1e-12


@pallas_only
def test_large_radius_falls_back_loudly():
    # A radius past the kernel range -> PallasNotTileable -> loud fallback.
    w, f = _pair((128, 128, 128), seed=9, dtype=np.float32)
    reset_fallback_state()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        u_pal = LNCCForce(
            radius=5, derivative='center', backend='pallas-cuda'
        ).bind(f, ndim=3).update(w)
        fired = [x for x in rec if issubclass(x.category, NitrixBackendFallback)]
    u_jax = LNCCForce(
        radius=5, derivative='center', backend='jax'
    ).bind(f, ndim=3).update(w)
    assert len(fired) == 1
    assert np.array_equal(np.asarray(u_pal), np.asarray(u_jax))


@pallas_only
def test_jit_under_kernel_path():
    w, f = _pair((128, 128, 128), seed=7, dtype=np.float32)
    bound = LNCCForce(radius=2, derivative='center', backend='pallas-cuda').bind(
        f, ndim=3
    )
    ref = bound.update(w)
    out = jax.jit(bound.update)(w)
    assert float(jnp.max(jnp.abs(out - ref))) < 1e-10


def test_default_backend_is_jax_byte_identical():
    # Default LNCCForce (no backend) == explicit backend='jax' -- CPU-safe, runs
    # everywhere: the kernel never perturbs default recipe output.
    w, f = _pair((24, 24, 24), seed=8, dtype=np.float32)
    u_default = LNCCForce(radius=2, derivative='center').bind(f, ndim=3).update(w)
    u_jax = LNCCForce(
        radius=2, derivative='center', backend='jax'
    ).bind(f, ndim=3).update(w)
    assert np.array_equal(np.asarray(u_default), np.asarray(u_jax))
    assert LNCCForce().backend == 'jax'
