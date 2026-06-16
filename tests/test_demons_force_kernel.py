# -*- coding: utf-8 -*-
"""v4 Phase 5a: the fused ESM Demons-force Pallas kernel.

The kernel (``∇warped`` stencil + the ESM force in one tiled pass) is **pure
perf**: it must be ULP/tolerance-equal to the pure-JAX ``_BoundDemons.update``
oracle (the 0a denom guard included).  These tests pin that parity (random,
3-D, and the matched-uniform guard path), the dispatch (default ``'jax'`` is
byte-identical; ``'auto'`` resolves to the kernel on a supported GPU), and the
loud fallbacks (anisotropic spacing, untileable shape).  GPU-only -- an explicit
``backend='pallas-cuda'`` request errors without an Ampere+ NVIDIA card.
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
from nitrix.register._force import DemonsForce  # noqa: E402

pallas_only = pytest.mark.skipif(
    not _HAS_AMPERE_NVIDIA,
    reason='requires NVIDIA Ampere+ for the Pallas Triton backend',
)


def _pair(shape, seed=0, dtype=np.float64, *, matched_corner=True):
    rng = np.random.RandomState(seed)
    fixed = jnp.asarray(rng.standard_normal(shape).astype(dtype))
    warped = jnp.asarray(rng.standard_normal(shape).astype(dtype))
    if matched_corner:
        # A matched-uniform corner exercises the 0a denom guard (0/0 -> 0).
        sl = tuple(slice(0, 4) for _ in shape)
        warped = warped.at[sl].set(fixed[sl])
    return warped, fixed


def _update(warped, fixed, *, backend, alpha=0.4, rel_spacing=None):
    return (
        DemonsForce(alpha=alpha, backend=backend)
        .bind(fixed, ndim=warped.ndim, rel_spacing=rel_spacing)
        .update(warped)
    )


@pallas_only
@pytest.mark.parametrize('shape', [(64, 64), (32, 32, 32), (48, 48, 48)])
def test_kernel_parity_with_jax_oracle(shape):
    # fp64: the kernel must reproduce the JAX force to machine precision.
    warped, fixed = _pair(shape, seed=1)
    u_jax = _update(warped, fixed, backend='jax')
    u_pal = _update(warped, fixed, backend='pallas-cuda')
    scale = float(jnp.max(jnp.abs(u_jax))) + 1e-30
    assert float(jnp.max(jnp.abs(u_pal - u_jax))) / scale < 1e-12


@pallas_only
def test_kernel_parity_fp32():
    # fp32: ULP-level agreement (op-ordering differs in the stencil).
    warped, fixed = _pair((48, 48, 48), seed=2, dtype=np.float32)
    u_jax = _update(warped, fixed, backend='jax')
    u_pal = _update(warped, fixed, backend='pallas-cuda')
    scale = float(jnp.max(jnp.abs(u_jax))) + 1e-30
    assert float(jnp.max(jnp.abs(u_pal - u_jax))) / scale < 1e-5


@pallas_only
def test_auto_resolves_to_kernel_byte_identical():
    # 'auto' on a supported GPU picks the kernel -> identical to explicit.
    warped, fixed = _pair((32, 32, 32), seed=3)
    u_auto = _update(warped, fixed, backend='auto')
    u_pal = _update(warped, fixed, backend='pallas-cuda')
    assert np.array_equal(np.asarray(u_auto), np.asarray(u_pal))


@pallas_only
def test_jit_under_kernel_path():
    warped, fixed = _pair((32, 32, 32), seed=4)
    ref = _update(warped, fixed, backend='pallas-cuda')
    fn = jax.jit(
        lambda w: DemonsForce(backend='pallas-cuda')
        .bind(fixed, ndim=3)
        .update(w)
    )
    assert float(jnp.max(jnp.abs(fn(warped) - ref))) < 1e-12


@pallas_only
def test_anisotropic_falls_back_loudly():
    # The fused kernel is isotropic-only; an anisotropic spacing must fall
    # back to the JAX voxel-native path (byte-identical) with one warning.
    warped, fixed = _pair((32, 32, 32), seed=5)
    rel = (1.0, 1.0, 2.0)
    reset_fallback_state()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        u_pal = _update(warped, fixed, backend='pallas-cuda', rel_spacing=rel)
        fired = [w for w in rec if issubclass(w.category, NitrixBackendFallback)]
    u_jax = _update(warped, fixed, backend='jax', rel_spacing=rel)
    assert len(fired) == 1
    assert np.array_equal(np.asarray(u_pal), np.asarray(u_jax))


@pallas_only
def test_untileable_shape_falls_back_loudly():
    # A prime extent admits no friendly tile -> PallasNotTileable -> JAX.
    warped, fixed = _pair((47, 48, 48), seed=6)
    reset_fallback_state()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        u_pal = _update(warped, fixed, backend='pallas-cuda')
        fired = [w for w in rec if issubclass(w.category, NitrixBackendFallback)]
    u_jax = _update(warped, fixed, backend='jax')
    assert len(fired) == 1
    assert np.array_equal(np.asarray(u_pal), np.asarray(u_jax))


def test_default_backend_is_jax_byte_identical():
    # The default force (no backend kwarg) and an explicit backend='jax' are
    # the same path -- CPU-safe, so this runs everywhere (the discipline gate:
    # the kernel never perturbs default recipe output).
    warped, fixed = _pair((24, 24, 24), seed=7)
    u_default = DemonsForce().bind(fixed, ndim=3).update(warped)
    u_jax = _update(warped, fixed, backend='jax')
    assert np.array_equal(np.asarray(u_default), np.asarray(u_jax))
    from nitrix.register._force import DemonsForce as DF

    assert DF().backend == 'jax'
