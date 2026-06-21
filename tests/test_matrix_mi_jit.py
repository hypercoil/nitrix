# -*- coding: utf-8 -*-
"""Matrix-recipe MI/CR auto-derive their histogram range under jit (bins suffice).

The matrix driver pins a histogram metric's range once from the full-res inputs
(``_metric.pin_metric_ranges``).  It does so with a ``stop_gradient``-ed reduction
(not eager ``float()``), so ``MI(bins=...)`` / ``CorrelationRatio(bins=...)`` trace
under ``jax.jit`` with no explicit range -- the matrix-side analogue of the
``_svf.pin_force_ranges`` fix.  Gate: jit rigid/affine MI succeeds bins-only; the
pin is detached; a least-squares metric is untouched.
"""

from __future__ import annotations

import functools

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import nitrix.register._metric as metmod  # noqa: E402
from nitrix.metrics import ncc  # noqa: E402
from nitrix.register import (  # noqa: E402
    MI,
    SSD,
    RegistrationSpec,
    affine_register,
    rigid_register,
)


def _img(n=40, seed=0):
    # Non-periodic blob field (clear features -> a well-posed MI signal, unlike
    # a periodic sinusoid where MI/ncc barely move under a small shift).
    rng = np.random.RandomState(seed)
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype('float64')

    def blob(cz, cy, cx, s, a):
        return a * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) / (2 * s * s)
        )

    base = (
        blob(0.4 * n, 0.42 * n, 0.5 * n, 0.16 * n, 1.0)
        + blob(0.6 * n, 0.62 * n, 0.4 * n, 0.2 * n, 0.7)
        + blob(0.5 * n, 0.3 * n, 0.62 * n, 0.12 * n, 0.6)
    )
    return jnp.asarray(
        (base + 0.02 * rng.standard_normal((n, n, n))).astype('float32')
    )


def test_pin_range_jit_safe_and_detached():
    x = _img()
    lo, hi = jax.jit(metmod._pin_range)(x)
    assert float(lo) == float(x.min())
    assert float(hi) == float(x.max())
    g = jax.grad(lambda z: metmod._pin_range(z)[1])(x)
    assert float(jnp.abs(g).max()) == 0.0


def test_pin_metric_ranges_noop_for_ssd():
    assert metmod.pin_metric_ranges(SSD(), _img(), _img()) == SSD()


@functools.lru_cache(maxsize=None)
def _shift_pair():
    fixed = _img(40, 0)
    moving = jnp.asarray(np.roll(np.asarray(fixed), 4, axis=0))
    return moving, fixed


def test_rigid_mi_jit_needs_no_explicit_range():
    moving, fixed = _shift_pair()
    spec = RegistrationSpec(levels=2, iterations=(40, 15), metric=MI(bins=32))
    fn = jax.jit(functools.partial(rigid_register, spec=spec))
    res = fn(moving, fixed)
    res.warped.block_until_ready()
    # the point of the test is that MI *traces* under jit (the pin_metric_ranges
    # fix); recovery is a sanity check (a finite, non-diverged, improved warp).
    assert bool(jnp.all(jnp.isfinite(res.warped)))
    assert float(ncc(res.warped, fixed)) > float(ncc(moving, fixed))


def test_affine_mi_jit_needs_no_explicit_range():
    moving, fixed = _shift_pair()
    spec = RegistrationSpec(levels=2, iterations=(40, 15), metric=MI(bins=32))
    fn = jax.jit(functools.partial(affine_register, spec=spec))
    res = fn(moving, fixed)
    res.warped.block_until_ready()
    assert bool(jnp.all(jnp.isfinite(res.warped)))
