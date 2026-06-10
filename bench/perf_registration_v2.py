# -*- coding: utf-8 -*-
"""Brain-scale certification benchmark for the registration-suite-v2 recipes.

The v1 ``perf_registration.py`` table covers the matrix recipes (rigid /
affine / demons / implicit-LNCC) at small sizes.  This v2 harness certifies
the **new** recipes added on the physical-space (R4) foundation, at
brain-realistic scale and on the paths that actually dominate their cost:

- ``greedy_syn_register`` -- single-pair symmetric diffeomorphic SyN
  (the LNCC-force SVF kernel), swept across 3-D volume size.  This is the
  SVF family the Pallas-ESM-force proposal targets, so its scaling is the
  evidence that trips or rules out that gate.
- ``volreg`` -- batched rigid motion realignment of a ``(T, *spatial)``
  series, the inverse-compositional path (constant-template Hessian built
  once, shared across frames *and* iterations).  Swept across frame count
  and volume size; we report both the whole-series time and the per-frame
  amortised cost (the number that matters for a long fMRI run).

Everything runs in **f32** (x64 left off): the deployment dtype on a
healthy GPU, and the realistic per-voxel bandwidth number -- not the
x64-doubled cost the correctness tests pay.

Run::

    ./.venv/bin/python bench/perf_registration_v2.py

Writes ``PERF_REGISTRATION_V2.md`` alongside this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from _util import (  # type: ignore[import-not-found]
    bench_call,
    host_summary,
    timed_jit,
)

from nitrix.geometry import (
    affine_grid,
    identity_grid,
    integrate_velocity_field,
    rigid_exp,
    spatial_transform,
)
from nitrix.register import (
    RegistrationSpec,
    SyNSpec,
    greedy_syn_register,
    volreg,
)
from nitrix.smoothing import gaussian


def _blobs(shape: tuple[int, ...], seed: int = 0) -> jax.Array:
    """A smooth multi-blob test image of the given spatial shape (f32)."""
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
    rng = np.random.RandomState(seed)
    img = np.zeros(shape, dtype='float32')
    for _ in range(6):
        center = [rng.uniform(0.25, 0.75) * s for s in shape]
        sigma = rng.uniform(0.1, 0.18) * min(shape)
        amp = rng.uniform(0.4, 1.0)
        r2 = sum((g - c) ** 2 for g, c in zip(grids, center))
        img += amp * np.exp(-r2 / (2 * sigma * sigma))
    return jnp.asarray(img)


def _smooth_velocity(
    shape: tuple[int, ...], sigma: float, scale: float, seed: int
) -> jax.Array:
    """A smooth random velocity field ``(*shape, ndim)`` (f32)."""
    ndim = len(shape)
    rng = np.random.RandomState(seed)
    v = rng.standard_normal(shape + (ndim,)).astype('float32')
    v = np.moveaxis(v, -1, 0)
    v = np.asarray(gaussian(jnp.asarray(v), sigma=sigma, spatial_rank=ndim))
    v = np.moveaxis(v, 0, -1)
    return jnp.asarray(scale * v)


def _deformable_pair(shape: tuple[int, ...]) -> tuple[jax.Array, jax.Array]:
    """Fixed multi-blob + moving warped by a known smooth diffeomorphism."""
    fixed = _blobs(shape)
    v = _smooth_velocity(
        shape, sigma=0.12 * min(shape), scale=0.06 * min(shape), seed=1
    )
    grid = identity_grid(shape, dtype=fixed.dtype)
    s = integrate_velocity_field(v)
    moving = spatial_transform(fixed[..., None], grid + s, mode='nearest')[
        ..., 0
    ]
    return moving, fixed


def _motion_series(shape: tuple[int, ...], n_frames: int) -> jax.Array:
    """A ``(T, *spatial)`` series: one base volume under T small rigid motions."""
    base = _blobs(shape)
    ndim = len(shape)
    center = (jnp.asarray(shape, dtype=base.dtype) - 1.0) / 2.0
    rng = np.random.RandomState(7)
    frames = []
    for t in range(n_frames):
        if ndim == 3:
            p = (
                rng.uniform(-0.04, 0.04, 3).tolist()
                + rng.uniform(-1.5, 1.5, 3).tolist()
            )
        else:
            p = [rng.uniform(-0.05, 0.05)] + rng.uniform(-2.0, 2.0, 2).tolist()
        params = jnp.asarray(p, dtype=base.dtype)
        grid = affine_grid(rigid_exp(params, ndim=ndim), shape, center=center)
        frames.append(
            spatial_transform(base[..., None], grid, mode='nearest')[..., 0]
        )
    return jnp.stack(frames, axis=0)


def _bench_syn(sizes: list[int], spec: SyNSpec) -> list[tuple]:
    rows = []
    for n in sizes:
        shape = (n, n, n)
        moving, fixed = _deformable_pair(shape)
        fwd = timed_jit(
            lambda m, f: greedy_syn_register(m, f, spec=spec).warped
        )
        s = bench_call(fwd, moving, fixed, warmup=2, repeats=5)
        mvox = (n**3) / 1e6
        rows.append(
            (f'{n}^3', f'{mvox:.2f}', s.compile_s, s.warm_s, s.warm_s / mvox)
        )
        print(
            f'  SyN {n}^3: cold {s.compile_s:.2f}s warm {s.warm_s * 1e3:.1f}ms'
        )
    return rows


def _bench_volreg(
    configs: list[tuple[int, int]], spec: RegistrationSpec
) -> list[tuple]:
    rows = []
    for n_frames, n in configs:
        shape = (n, n, n)
        series = _motion_series(shape, n_frames)
        fwd = timed_jit(lambda s: volreg(s, spec=spec).realigned)
        s = bench_call(fwd, series, warmup=2, repeats=5)
        rows.append(
            (
                f'T={n_frames}, {n}^3',
                n_frames,
                s.compile_s,
                s.warm_s,
                s.warm_s / n_frames * 1e3,
            )
        )
        print(
            f'  volreg T={n_frames} {n}^3: cold {s.compile_s:.2f}s '
            f'warm {s.warm_s * 1e3:.1f}ms ({s.warm_s / n_frames * 1e3:.2f}ms/frame)'
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--syn-sizes', type=int, nargs='+', default=[64, 96, 128]
    )
    parser.add_argument('--syn-levels', type=int, default=3)
    parser.add_argument('--syn-iters', type=int, default=20)
    parser.add_argument('--syn-radius', type=int, default=2)
    parser.add_argument('--volreg-levels', type=int, default=3)
    parser.add_argument('--volreg-iters', type=int, default=20)
    args = parser.parse_args()

    syn_spec = SyNSpec(
        levels=args.syn_levels,
        iterations=args.syn_iters,
        radius=args.syn_radius,
        step=0.5,
    )
    volreg_spec = RegistrationSpec(
        levels=args.volreg_levels, iterations=args.volreg_iters
    )
    volreg_configs = [(16, 64), (32, 64), (16, 96)]

    print('greedy SyN (single pair):')
    syn_rows = _bench_syn(args.syn_sizes, syn_spec)
    print('volreg (cohort, inverse-compositional):')
    volreg_rows = _bench_volreg(volreg_configs, volreg_spec)

    host = host_summary()
    lines = [
        '# Registration-suite-v2 scaling benchmark',
        '',
        f'- Device: `{host["device"]}`, jax `{host["jax_version"]}`',
        '- dtype: **f32** (deployment path; x64 left off).',
        f'- SyN spec: levels={args.syn_levels}, iterations={args.syn_iters}, '
        f'radius={args.syn_radius}, step=0.5.',
        f'- volreg spec: levels={args.volreg_levels}, '
        f'iterations={args.volreg_iters}, inverse-compositional, SSD.',
        '- Warm = median post-warmup wall-time; cold = first (trace+compile) '
        'call.',
        '',
        '## greedy SyN -- single-pair scaling',
        '',
        '| size | Mvox | cold (s) | warm (s) | warm/Mvox (ms) |',
        '|---|---|---|---|---|',
    ]
    for name, mvox, cold, warm, per in syn_rows:
        lines.append(
            f'| {name} | {mvox} | {cold:.2f} | {warm:.4f} | {per * 1e3:.3f} |'
        )
    lines += [
        '',
        '## volreg -- cohort (inverse-compositional)',
        '',
        '| config | T | cold (s) | warm (s) | warm/frame (ms) |',
        '|---|---|---|---|---|',
    ]
    for name, t, cold, warm, per in volreg_rows:
        lines.append(f'| {name} | {t} | {cold:.2f} | {warm:.4f} | {per:.3f} |')
    report = '\n'.join(lines) + '\n'
    out = Path(__file__).with_name('PERF_REGISTRATION_V2.md')
    out.write_text(report)
    print('\n' + report)


if __name__ == '__main__':
    main()
