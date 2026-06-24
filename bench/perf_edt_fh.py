# -*- coding: utf-8 -*-
"""EDT engine decision benchmark: brute-force min-plus matmul vs the
Felzenszwalb-Huttenlocher (FH) O(n) lower-envelope scan.

``morphology.distance_transform(metric='euclidean')`` computes each 1D axis
pass as a tropical min-plus matmul against the (n, n) squared-distance matrix --
O(n^2) work but a dense, control-flow-free kernel.  FH computes the same exact
1D EDT in O(n) via a per-line parabola-envelope stack scan.  The question
(raised for the surface-metrics tier): does an FH path, dispatched on axis size,
beat the matmul at large n?

Run once per platform::

    JAX_PLATFORMS=cpu python bench/perf_edt_fh.py
    python bench/perf_edt_fh.py            # gpu, if visible

Verifies FH == brute-force == scipy (exact), then times the full 3D EDT across
cube sizes.  See bench/PERF_EDT_FH.md for the decision.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from nitrix.morphology import distance_transform_edt as nx_edt

_EDT_BIG = 1e10
_EDT_BIG_THRESHOLD = 1e9


# --- FH 1D squared EDT: D[p] = min_q (f[q] + a*(p-q)^2), a = spacing^2 ------
def _fh_edt_1d(f, a):
    n = f.shape[0]
    a = jnp.asarray(a, f.dtype)
    big = jnp.asarray(_EDT_BIG * 100.0, f.dtype)
    v0 = jnp.zeros(n, jnp.int32)
    z0 = jnp.full(n + 1, big, f.dtype).at[0].set(-big)

    def outer(q, carry):
        v, z, k = carry
        qf = q.astype(f.dtype)

        def s_at(kk):
            r = v[kk]
            rf = r.astype(f.dtype)
            return ((f[q] - f[r]) + a * (qf * qf - rf * rf)) / (
                2.0 * a * (qf - rf)
            )

        k = lax.while_loop(
            lambda kk: jnp.logical_and(kk >= 0, s_at(kk) <= z[kk]),
            lambda kk: kk - 1,
            k,
        )
        s = s_at(k)
        k = k + 1
        v = v.at[k].set(q)
        z = z.at[k].set(s)
        z = z.at[k + 1].set(big)
        return (v, z, k)

    v, z, _ = lax.fori_loop(1, n, outer, (v0, z0, jnp.int32(0)))

    def fill(p, carry):
        D, k = carry
        pf = p.astype(f.dtype)
        k = lax.while_loop(lambda kk: z[kk + 1] < pf, lambda kk: kk + 1, k)
        r = v[k]
        rf = r.astype(f.dtype)
        D = D.at[p].set(a * (pf - rf) ** 2 + f[r])
        return (D, k)

    D, _ = lax.fori_loop(0, n, fill, (jnp.zeros(n, f.dtype), jnp.int32(0)))
    return D


def _edt_axis_fh(g, axis, spacing):
    g = jnp.moveaxis(g, axis, -1)
    shape = g.shape
    n = shape[-1]
    a = jnp.asarray(spacing, g.dtype) ** 2
    out = jax.vmap(lambda f: _fh_edt_1d(f, a))(g.reshape(-1, n))
    return jnp.moveaxis(out.reshape(shape), -1, axis)


def edt_fh(mask):
    arr = jnp.asarray(mask)
    dtype = jnp.promote_types(arr.dtype, jnp.float32)
    g = jnp.where(arr == 0, jnp.zeros((), dtype), jnp.asarray(_EDT_BIG, dtype))
    for ax in range(arr.ndim):
        g = _edt_axis_fh(g, ax, 1.0)
    return jnp.where(g >= _EDT_BIG_THRESHOLD, jnp.inf, jnp.sqrt(g))


def main():
    from scipy.ndimage import distance_transform_edt as sp_edt

    plat = jax.devices()[0].platform
    print(f'=== EDT brute-force vs FH | platform: {plat} | fp32 ===')

    rng = np.random.RandomState(0)
    m = (rng.rand(16, 18, 14) > 0.2).astype(np.float32)
    fh = np.asarray(edt_fh(jnp.asarray(m)), np.float64)
    bf = np.asarray(nx_edt(jnp.asarray(m)), np.float64)
    sp = sp_edt(m > 0.5)
    print(
        f'correctness: FH-vs-scipy={np.max(np.abs(fh - sp)):.2e} '
        f'FH-vs-brute={np.max(np.abs(fh - bf)):.2e}'
    )

    print(f'{"D":>5} {"brute_ms":>10} {"fh_ms":>10} {"FH_speedup":>11}')
    for d in (32, 64, 96, 128, 160, 192, 256):
        vol = jnp.asarray((rng.rand(d, d, d) > 0.5).astype(np.float32))
        bfn = jax.jit(nx_edt)
        fhn = jax.jit(edt_fh)
        try:
            bfn(vol).block_until_ready()
            fhn(vol).block_until_ready()
        except Exception as e:  # noqa: BLE001
            print(f'{d:>5}  ERROR {type(e).__name__}: {str(e)[:50]}')
            continue

        def timed(fn, reps=5):
            s = time.perf_counter()
            for _ in range(reps):
                fn(vol).block_until_ready()
            return (time.perf_counter() - s) / reps * 1e3

        tb, tf = timed(bfn), timed(fhn)
        print(f'{d:>5} {tb:>10.2f} {tf:>10.2f} {tb / tf:>10.2f}x')


if __name__ == '__main__':
    main()
