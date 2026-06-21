# -*- coding: utf-8 -*-
"""MIForce auto-derives its histogram range under jit (bins suffice).

The Mattes MI force needs a histogram *range* (bins alone do not define edges),
but it need not be user-supplied: ``_svf.pin_force_ranges`` pins it once from the
full-resolution inputs.  It does so with a ``stop_gradient``-ed reduction (not an
eager ``float()``), so the pin is jit-safe -- ``MIForce(bins=...)`` traces with no
explicit range -- while staying a stationary, constant-edge (piecewise-constant
Mattes) range.  Gate: jit + bins-only succeeds and is numerically ~the explicit
range (the only difference is the float() round-trip the eager path used to add).
"""

from __future__ import annotations

import functools

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import nitrix.register._svf as svf  # noqa: E402
from nitrix.register import MIForce, SyNSpec, greedy_syn_register  # noqa: E402
from nitrix.register._force import DemonsForce  # noqa: E402


def _blobs(n=64):
    yy, xx = np.mgrid[0:n, 0:n].astype('float64')

    def blob(cy, cx, s, a):
        return a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))

    return jnp.asarray(
        blob(0.31 * n, 0.38 * n, 0.13 * n, 1.0)
        + blob(0.62 * n, 0.69 * n, 0.16 * n, 0.7)
    )


def test_pin_range_is_jit_safe_and_stationary():
    # _pin_range traces (no float(tracer)) and is detached (constant edges).
    x = _blobs(48)
    lo, hi = jax.jit(svf._pin_range)(x)
    assert float(lo) == float(x.min())
    assert float(hi) == float(x.max())
    # detached: gradient of the bound w.r.t. the input is zero.
    g = jax.grad(lambda z: svf._pin_range(z)[1])(x)
    assert float(jnp.abs(g).max()) == 0.0


def test_mi_force_jit_needs_no_explicit_range():
    fixed = _blobs(64)
    moving = jnp.asarray(np.roll(np.asarray(fixed), 3, axis=0))
    spec = SyNSpec(levels=2, iterations=(15, 8))
    # MIForce with bins only (range_moving / range_fixed left None): traces.
    fn = jax.jit(
        functools.partial(
            greedy_syn_register, spec=spec, force=MIForce(bins=32)
        )
    )
    res = fn(moving, fixed)
    res.warped.block_until_ready()
    assert bool(jnp.all(jnp.isfinite(res.warped)))
    # an explicit pinned range produces ~the same result (ULP-level float()
    # round-trip aside).
    rm = (float(moving.min()), float(moving.max()))
    rf = (float(fixed.min()), float(fixed.max()))
    fn2 = jax.jit(
        functools.partial(
            greedy_syn_register,
            spec=spec,
            force=MIForce(bins=32, range_moving=rm, range_fixed=rf),
        )
    )
    res2 = fn2(moving, fixed)
    assert np.allclose(
        np.asarray(res.warped), np.asarray(res2.warped), atol=1e-3
    )


def test_pin_force_ranges_noop_for_non_mi():
    # a non-histogram force is untouched.
    df = DemonsForce()
    assert svf.pin_force_ranges(df, _blobs(32), _blobs(32)) is df
