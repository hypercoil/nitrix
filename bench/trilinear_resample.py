# -*- coding: utf-8 -*-
"""Baseline for 3D trilinear resampling (``geometry.spatial_transform``).

A downstream consumer asked for a Pallas kernel for 3D trilinear
resampling -- the workhorse behind ``spatial_transform`` /
``integrate_velocity_field`` (the SVF loop composes it ``n_steps``
times) and any voxelmorph-style warp.  Before committing kernel time we
benchmark the current path, because trilinear resampling is
*structurally* a gather (8 corner voxels at data-dependent positions),
and that is exactly the primitive the G0 gate found Pallas Triton cannot
lower on the pinned JAX -- the reason ``semiring_ell_matmul`` is
JAX-default today (see ``bench/G0_ELL_REPORT.md``).

So this script establishes (a) the steady-state wall time and effective
voxel throughput of the current ``jax.scipy.ndimage.map_coordinates``
path on the reference GPU, forward and forward+backward, and (b) a point
of comparison: an explicit pure-JAX 8-corner trilinear gather, which is
the closest pure-XLA analogue of what a Pallas kernel would fuse.  If
the explicit gather is not materially faster than ``map_coordinates``,
a Pallas kernel -- which faces the gather-lowering blocker on top -- is
unlikely to pay off until upstream Pallas grows gather.

Reproduce::

    python bench/trilinear_resample.py
    python bench/trilinear_resample.py --quick
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from nitrix.geometry import identity_grid, spatial_transform

from _util import BenchSample, bench_call, format_us, host_summary, timed_jit


HERE = Path(__file__).parent
REPORT_PATH = HERE / 'PERF_TRILINEAR.md'


# (D, H, W) single-channel volumes spanning the registration regime.
SHAPES = [(64, 64, 64), (128, 128, 128), (192, 192, 192), (256, 256, 256)]
QUICK_SHAPES = [(64, 64, 64), (128, 128, 128)]


def make_warp(shape, seed=0):
    '''A smooth-ish absolute deformation: identity grid + small noise.'''
    key = jax.random.key(seed)
    ki, kd = jax.random.split(key)
    image = jax.random.normal(ki, (*shape, 1), dtype=jnp.float32)
    disp = 2.0 * jax.random.normal(kd, (*shape, 3), dtype=jnp.float32)
    deform = identity_grid(shape, dtype=jnp.float32) + disp
    return (
        jax.block_until_ready(image),
        jax.block_until_ready(deform),
    )


def _explicit_trilinear(image, deform):
    '''Pure-JAX 8-corner trilinear gather (channel-last, single volume).

    The closest pure-XLA analogue of a fused Pallas trilinear kernel:
    compute the 8 corner indices, gather, and blend.  Boundary by edge
    clamp ("nearest").  Used only as a throughput reference here.
    '''
    D, H, W, _ = image.shape
    x = deform[..., 0]
    y = deform[..., 1]
    z = deform[..., 2]
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    z0 = jnp.floor(z).astype(jnp.int32)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    wx, wy, wz = x - x0, y - y0, z - z0

    def clamp(a, n):
        return jnp.clip(a, 0, n - 1)

    x0c, x1c = clamp(x0, D), clamp(x1, D)
    y0c, y1c = clamp(y0, H), clamp(y1, H)
    z0c, z1c = clamp(z0, W), clamp(z1, W)
    img = image[..., 0]

    def g(ix, iy, iz):
        return img[ix, iy, iz]

    c000 = g(x0c, y0c, z0c)
    c001 = g(x0c, y0c, z1c)
    c010 = g(x0c, y1c, z0c)
    c011 = g(x0c, y1c, z1c)
    c100 = g(x1c, y0c, z0c)
    c101 = g(x1c, y0c, z1c)
    c110 = g(x1c, y1c, z0c)
    c111 = g(x1c, y1c, z1c)
    c00 = c000 * (1 - wx) + c100 * wx
    c01 = c001 * (1 - wx) + c101 * wx
    c10 = c010 * (1 - wx) + c110 * wx
    c11 = c011 * (1 - wx) + c111 * wx
    c0 = c00 * (1 - wy) + c10 * wy
    c1 = c01 * (1 - wy) + c11 * wy
    return (c0 * (1 - wz) + c1 * wz)[..., None]


def run(shapes, warmup, repeats):
    rows = []
    for shape in shapes:
        image, deform = make_warp(shape, seed=hash(shape) & 0xFFFF)
        n_vox = int(shape[0] * shape[1] * shape[2])
        row = {'shape': 'x'.join(str(s) for s in shape), 'n_vox': n_vox}

        # 1. Forward: map_coordinates path (the shipped op).
        fwd = timed_jit(lambda im, df: spatial_transform(im, df, mode='nearest'))
        s_fwd: BenchSample = bench_call(fwd, image, deform, warmup=warmup, repeats=repeats)
        row['fwd_warm_s'] = s_fwd.warm_s
        row['fwd_compile_s'] = s_fwd.compile_s
        row['fwd_mvox_s'] = n_vox / s_fwd.warm_s / 1e6

        # 2. Forward+backward (registration-like: grad wrt the field).
        # NB: value_and_grad returns a (value, grad) tuple, which has no
        # ``.block_until_ready``; collapse to a single scalar that depends on
        # both so ``bench_call`` actually blocks on the gradient (otherwise it
        # would time async dispatch only).
        def loss(df, im=image):
            return jnp.sum(spatial_transform(im, df, mode='nearest') ** 2)
        _vg = jax.value_and_grad(loss)
        def vg_scalar(df):
            v, g = _vg(df)
            return v + jnp.sum(g)
        vg = timed_jit(vg_scalar)
        s_vg: BenchSample = bench_call(vg, deform, warmup=warmup, repeats=repeats)
        row['vg_warm_s'] = s_vg.warm_s
        row['vg_mvox_s'] = n_vox / s_vg.warm_s / 1e6

        # 3. Reference: explicit pure-JAX 8-corner gather (forward).
        ref = timed_jit(_explicit_trilinear)
        s_ref: BenchSample = bench_call(ref, image, deform, warmup=warmup, repeats=repeats)
        row['explicit_warm_s'] = s_ref.warm_s
        row['explicit_mvox_s'] = n_vox / s_ref.warm_s / 1e6

        rows.append(row)
    return rows


def render_report(rows, host) -> str:
    lines = [
        '# Trilinear resampling — baseline + Pallas decision',
        '',
        '> Consumer ask: a Pallas kernel for 3D trilinear resampling',
        '> (``geometry.spatial_transform`` / ``integrate_velocity_field``).',
        '> This report is the "benchmark first" step before any kernel work.',
        '',
        '## Host',
        '',
        f'- Device: {host["device"]}',
        f'- Platform: {host["platform"]}',
        f'- JAX: {host["jax_version"]}',
        '',
        '## Steady-state (post-warm-up median)',
        '',
        '`fwd` = shipped ``spatial_transform`` (``map_coordinates`` order=1).',
        '`fwd+bwd` = ``value_and_grad`` of a scalar loss wrt the field',
        '(the registration-training cost). `explicit` = pure-JAX 8-corner',
        'gather (forward), the closest pure-XLA analogue of a fused Pallas',
        'kernel.',
        '',
        '| shape | voxels | fwd | fwd Mvox/s | fwd+bwd | fwd+bwd Mvox/s | explicit fwd | explicit Mvox/s |',
        '|---|------:|----:|----:|----:|----:|----:|----:|',
    ]
    for r in rows:
        lines.append(
            '| {shape} | {n} | {fwd} | {fm:.0f} | {vg} | {vm:.0f} | {ex} | {em:.0f} |'.format(
                shape=r['shape'], n=r['n_vox'],
                fwd=format_us(r['fwd_warm_s']), fm=r['fwd_mvox_s'],
                vg=format_us(r['vg_warm_s']), vm=r['vg_mvox_s'],
                ex=format_us(r['explicit_warm_s']), em=r['explicit_mvox_s'],
            )
        )
    lines += [
        '',
        '## Reading',
        '',
        '- Trilinear resampling is gather-bound: the shipped',
        '  ``map_coordinates`` path and the explicit 8-corner gather both',
        '  lower to ``lax.gather`` over the volume.  Their throughput is',
        '  the practical ceiling for an XLA implementation on this device.',
        '- A Pallas Triton kernel would need to express the 8 corner loads',
        '  as data-dependent ``pl.load`` / pointer arithmetic.  The G0 gate',
        '  (``bench/G0_ELL_REPORT.md``) found Triton on the pinned JAX does',
        '  not lower the ``gather`` HLO primitive; a pointer-load',
        '  formulation *may* sidestep that, but it is unproven and is the',
        '  same upstream risk surface that kept ELL on the JAX path.',
        '',
        '## Decision',
        '',
        'See the inline summary printed by this script and the writeup in',
        'the feedback / backlog.  The kernel is gated on (a) these numbers',
        'showing the path is a real bottleneck in a consumer training loop,',
        'and (b) a pointer-load Pallas prototype clearing the gather-',
        'lowering risk -- otherwise ship JAX-default (the current state),',
        'with the kernel revisited when upstream Pallas grows gather.',
    ]
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    shapes = QUICK_SHAPES if args.quick else SHAPES
    host = host_summary()
    rows = run(shapes, warmup=args.warmup, repeats=args.repeats)
    report = render_report(rows, host)
    REPORT_PATH.write_text(report)
    print(report)
    print(f'Wrote {REPORT_PATH}.')


if __name__ == '__main__':
    main()
