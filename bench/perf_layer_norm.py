# -*- coding: utf-8 -*-
"""Decision benchmark: does a *fused* Pallas LayerNorm beat XLA on this host?

P3 gates the fused norm kernel on an empirical perf signal (suite §7.3): unlike
attention / selective-scan there is no activation cliff to remove -- the only
win is memory bandwidth, which XLA's own elementwise+reduction fusion already
largely captures.  Rather than build our own kernel to find out, we benchmark
the **stock** ``jax.experimental.pallas.ops.gpu.layer_norm`` (the exact kernel
the plan says to fork) against the nitrix XLA reference, forward and
forward+backward, across realistic transformer shapes.  If the off-the-shelf
fused kernel does not beat XLA here, our fork won't either.

Run: ``./.venv/bin/python bench/perf_layer_norm.py`` (needs an Ampere+ GPU).
"""

from __future__ import annotations

from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.gpu import layer_norm as stock

from nitrix.nn.norm import reference_layer_norm

_SHAPES = [
    (8, 1024, 1024),
    (8, 1024, 2048),
    (8, 1024, 4096),
    (8, 1024, 8192),
    (32, 512, 1024),
    (4, 4096, 2048),
]


def _bench(
    fn, args, iters: int = 100, warmup: int = 15, reps: int = 3
) -> float:
    f = jax.jit(fn)
    for _ in range(warmup):
        jax.block_until_ready(f(*args))
    best = float('inf')
    for _ in range(reps):
        t0 = perf_counter()
        out = None
        for _ in range(iters):
            out = f(*args)
        jax.block_until_ready(out)
        best = min(best, (perf_counter() - t0) / iters)
    return best * 1e3  # ms


def _grad(ln):
    def loss(x, w, b):
        return jnp.sum(ln(x, w, b) ** 2)

    return lambda x, w, b: jax.grad(loss, argnums=(0, 1, 2))(x, w, b)


def _peak_mb(fn, args) -> float:
    m = jax.jit(fn).lower(*args).compile().memory_analysis()
    total = (
        m.temp_size_in_bytes
        + m.output_size_in_bytes
        + m.argument_size_in_bytes
    )
    return total / 1e6


def main() -> None:
    print(f'device: {jax.devices()[0]}  dtype: float32\n')
    hdr = (
        f'{"shape (B,S,H)":>18} | {"fwd xla":>8} {"fwd pal":>8} {"x":>5} | '
        f'{"f+b xla":>8} {"f+b pal":>8} {"x":>5} | {"mem x/p":>10}'
    )
    print(hdr)
    print('-' * len(hdr))
    for b, s, h in _SHAPES:
        rng = np.random.RandomState(0)
        x = jnp.asarray(rng.standard_normal((b, s, h)).astype(np.float32))
        w = jnp.asarray(rng.standard_normal(h).astype(np.float32))
        bi = jnp.asarray(rng.standard_normal(h).astype(np.float32))
        args = (x, w, bi)

        fx = _bench(lambda x, w, b: reference_layer_norm(x, w, b), args)
        fp = _bench(lambda x, w, b: stock.layer_norm(x, w, b), args)
        gx = _bench(_grad(reference_layer_norm), args)
        gp = _bench(_grad(stock.layer_norm), args)
        mx = _peak_mb(lambda x, w, b: reference_layer_norm(x, w, b), args)
        mp = _peak_mb(lambda x, w, b: stock.layer_norm(x, w, b), args)
        print(
            f'{f"({b},{s},{h})":>18} | {fx:8.3f} {fp:8.3f} {fx / fp:5.2f} | '
            f'{gx:8.3f} {gp:8.3f} {gx / gp:5.2f} | {mx:5.0f}/{mp:<4.0f}'
        )


if __name__ == '__main__':
    main()
