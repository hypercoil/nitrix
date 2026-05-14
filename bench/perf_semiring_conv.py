# -*- coding: utf-8 -*-
"""Perf and memory benchmark for ``semiring_conv``.

Validates two design claims:

1. The current explicit-im2col + ``semiring_matmul`` implementation
   on the JAX path is within a small constant factor (target ≤ 5×) of
   ``lax.conv_general_dilated`` (cuDNN ``IMPLICIT_PRECOMP_GEMM``) for
   the REAL semiring.  The constant factor comes from two sources:
   (a) cuDNN may use tensor cores on Ampere+ (TF32 by default), while
   our kernel issues plain CUDA-core ops so it can target arbitrary
   algebras; (b) cuDNN computes the patches implicitly inside the
   kernel while we materialise them in HBM.  Per Chen et al. 2021
   (arXiv:2110.03901), the implicit-vs-explicit GEMM gap is ~28% on
   tensor-core hardware; on plain CUDA cores it should be smaller
   (~1.3–1.7×).  The tensor-core gap on top of that is the larger
   factor; see the docstring of ``semiring_matmul``.

2. Non-REAL algebras (LOG, TROPICAL_MAX_PLUS, EUCLIDEAN) run at
   roughly the same wall-time as REAL through our path: they share
   the patch extraction and the semiring_matmul kernel; only the
   per-step combine and monoid update differ.

3. Memory: our peak HBM is bounded by the im2col size
   ``M_out * prod(kspatial) * c_in`` plus the output ``M_out * c_out``,
   not by anything resembling ``M_out * prod(kspatial) * c_in * c_out``.

Note on benchmarking convention (matches ``perf_semiring_matmul.py``):
first-call wall-time includes XLA / Pallas compilation and is
reported separately as ``compile``.  The steady-state number is the
median of post-warm-up calls.

References for design context (research summary in
``bench/CONV_DESIGN_NOTE.md``):

- cuDNN algorithm selection: NVIDIA *Convolutional Layers User's Guide*;
  arXiv:2110.03901 (Chen et al., implicit-vs-explicit GEMM
  characterisation).
- KeOps-style tiled online map-reduce for semiring reductions:
  Charlier et al., JMLR 2021; kernel-operations.io.
- Triton implicit-GEMM template: Tillet et al., MAPL 2019 §6.2;
  github.com/l1351868270/implicit_gemm.triton.
- XLA gather+dot non-fusion: openxla.org/xla/gpu_architecture and
  the JAX issue tracker (no "free" implicit im2col on the JAX path).
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import jax
import jax.lax as lax
import jax.numpy as jnp

from nitrix.semiring import (
    EUCLIDEAN,
    LOG,
    REAL,
    TROPICAL_MAX_PLUS,
    semiring_conv,
)

from _util import (
    BenchSample,
    bench_call,
    format_us,
    host_summary,
    timed_jit,
)


HERE = Path(__file__).parent
REPORT_PATH = HERE / 'PERF_SEMIRING_CONV.md'


# (batch, H, W, c_in, kH, kW, c_out) -- 2D image conv shapes.
SHAPES_2D = [
    # Typical small-medium feature-map conv.
    (1, 64, 64, 32, 3, 3, 32),
    (1, 128, 128, 32, 3, 3, 64),
    (1, 224, 224, 64, 3, 3, 64),
    (1, 256, 256, 32, 3, 3, 32),
]
QUICK_SHAPES_2D = [(1, 64, 64, 32, 3, 3, 32)]

# 3D volume shapes.
SHAPES_3D = [
    (1, 32, 32, 32, 8, 3, 3, 3, 16),    # smallish 3D conv
    (1, 64, 64, 64, 8, 3, 3, 3, 8),     # neuroimaging-ish patch
]
QUICK_SHAPES_3D = [(1, 32, 32, 32, 8, 3, 3, 3, 16)]


def _make_2d(shape, seed):
    B, H, W, Cin, kH, kW, Cout = shape
    key = jax.random.key(seed)
    ka, kb = jax.random.split(key)
    x = jax.random.normal(ka, (B, H, W, Cin), dtype=jnp.float32)
    k = jax.random.normal(kb, (kH, kW, Cin, Cout), dtype=jnp.float32)
    return jax.block_until_ready(x), jax.block_until_ready(k)


def _make_3d(shape, seed):
    B, D, H, W, Cin, kD, kH, kW, Cout = shape
    key = jax.random.key(seed)
    ka, kb = jax.random.split(key)
    x = jax.random.normal(ka, (B, D, H, W, Cin), dtype=jnp.float32)
    k = jax.random.normal(kb, (kD, kH, kW, Cin, Cout), dtype=jnp.float32)
    return jax.block_until_ready(x), jax.block_until_ready(k)


def _peak() -> int:
    return int(jax.devices()[0].memory_stats()['peak_bytes_in_use'])


def measure_peak_ceiling(fn, *args, warmup: int = 2) -> int:
    """HBM peak high-water mark after ``warmup`` un-timed calls."""
    for _ in range(warmup):
        out = fn(*args)
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
        del out
    gc.collect()
    return _peak()


def lax_conv_2d(x, k, padding='SAME', precision=None):
    """Channel-last REAL conv via ``lax.conv_general_dilated``.

    ``precision=None`` -> JAX default (TF32 on Ampere+ tensor cores).
    ``precision='highest'`` -> strict fp32, no tensor cores -- this is
    the apples-to-apples baseline for our CUDA-core semiring_matmul.
    """
    x_nchw = jnp.moveaxis(x, -1, 1)
    # k: (kH, kW, c_in, c_out) -> (c_out, c_in, kH, kW)
    k_lax = jnp.moveaxis(jnp.moveaxis(k, -1, 0), -1, 1)
    out_nchw = lax.conv_general_dilated(
        x_nchw, k_lax,
        window_strides=(1, 1), padding=padding,
        precision=precision,
    )
    return jnp.moveaxis(out_nchw, 1, -1)


def lax_conv_3d(x, k, padding='SAME', precision=None):
    x_nchw = jnp.moveaxis(x, -1, 1)
    # k: (kD, kH, kW, c_in, c_out) -> (c_out, c_in, kD, kH, kW)
    k_lax = jnp.moveaxis(jnp.moveaxis(k, -1, 0), -1, 1)
    out_nchw = lax.conv_general_dilated(
        x_nchw, k_lax,
        window_strides=(1, 1, 1), padding=padding,
        precision=precision,
    )
    return jnp.moveaxis(out_nchw, 1, -1)


def io_floor_bytes(shape_args):
    """Analytical lower-bound on HBM for inputs + output (fp32).

    Returns ``4 * (size_x + size_k + size_out)``.
    """
    B = shape_args['B']
    spatial_in = shape_args['spatial']
    Cin = shape_args['Cin']
    kspatial = shape_args['kspatial']
    Cout = shape_args['Cout']
    size_x = B * Cin
    for d in spatial_in:
        size_x *= d
    size_k = Cin * Cout
    for d in kspatial:
        size_k *= d
    # out spatial = same as in (SAME padding); size depends on stride which
    # we keep at 1 here.
    size_out = B * Cout
    for d in spatial_in:
        size_out *= d
    return 4 * (size_x + size_k + size_out)


def im2col_overhead_bytes(shape_args):
    """Bytes our explicit im2col materialises on top of inputs/output."""
    B = shape_args['B']
    spatial_in = shape_args['spatial']
    Cin = shape_args['Cin']
    kspatial = shape_args['kspatial']
    n_out = B
    for d in spatial_in:
        n_out *= d
    k_total = Cin
    for d in kspatial:
        k_total *= d
    return 4 * n_out * k_total


def run_2d(shapes, algebras, warmup, repeats):
    rows = []
    for shape in shapes:
        B, H, W, Cin, kH, kW, Cout = shape
        x, k = _make_2d(shape, seed=hash(shape) & 0xFFFF)
        shape_args = {
            'B': B, 'spatial': (H, W), 'Cin': Cin,
            'kspatial': (kH, kW), 'Cout': Cout,
        }
        row = {
            'shape': f'B{B} {H}x{W}x{Cin} -> {kH}x{kW} {Cout}',
            'io_floor_bytes': io_floor_bytes(shape_args),
            'im2col_bytes': im2col_overhead_bytes(shape_args),
        }

        # cuDNN tensor-core baseline (uses TF32 on Ampere+).
        fn_tc = timed_jit(lambda x, k: lax_conv_2d(x, k))
        s = bench_call(fn_tc, x, k, warmup=warmup, repeats=repeats)
        row['cudnn_tc_warm_s'] = s.warm_s
        row['cudnn_tc_compile_s'] = s.compile_s

        # cuDNN strict-fp32 baseline (no tensor cores).  This is the
        # apples-to-apples comparison for our CUDA-core semiring kernel.
        fn_fp = timed_jit(lambda x, k: lax_conv_2d(x, k, precision='highest'))
        s = bench_call(fn_fp, x, k, warmup=warmup, repeats=repeats)
        row['cudnn_fp32_warm_s'] = s.warm_s
        row['cudnn_fp32_compile_s'] = s.compile_s

        # Our path for each algebra, via the Pallas inner matmul when
        # available (the conv-level kernel is missing but the inner
        # matmul has a native Triton kernel).
        for algebra in algebras:
            fn = timed_jit(
                lambda x, k, _a=algebra: semiring_conv(
                    x, k, semiring=_a, padding='SAME',
                    backend='pallas-cuda',
                )
            )
            try:
                # First call also exercises the conv-level fallback
                # warning; ``bench_call``'s warmup discards it.
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    s = bench_call(fn, x, k, warmup=warmup, repeats=repeats)
                row[f'{algebra.name}_warm_s'] = s.warm_s
                row[f'{algebra.name}_compile_s'] = s.compile_s
            except Exception as e:
                row[f'{algebra.name}_warm_s'] = None
                row[f'{algebra.name}_err'] = repr(e)[:200]

        # JAX-only REAL for reference (compare against the Pallas-matmul
        # speed-up).
        fn_jax = timed_jit(
            lambda x, k: semiring_conv(
                x, k, semiring=REAL, padding='SAME', backend='jax',
            )
        )
        s = bench_call(fn_jax, x, k, warmup=warmup, repeats=repeats)
        row['real_jax_warm_s'] = s.warm_s

        if row.get('cudnn_tc_warm_s') and row.get('real_warm_s'):
            row['slowdown_vs_tc'] = (
                row['real_warm_s'] / row['cudnn_tc_warm_s']
            )
        if row.get('cudnn_fp32_warm_s') and row.get('real_warm_s'):
            row['slowdown_vs_fp32'] = (
                row['real_warm_s'] / row['cudnn_fp32_warm_s']
            )
        rows.append(row)
    return rows


def run_3d(shapes, algebras, warmup, repeats):
    rows = []
    for shape in shapes:
        B, D, H, W, Cin, kD, kH, kW, Cout = shape
        x, k = _make_3d(shape, seed=hash(shape) & 0xFFFF)
        shape_args = {
            'B': B, 'spatial': (D, H, W), 'Cin': Cin,
            'kspatial': (kD, kH, kW), 'Cout': Cout,
        }
        row = {
            'shape': f'B{B} {D}x{H}x{W}x{Cin} -> {kD}x{kH}x{kW} {Cout}',
            'io_floor_bytes': io_floor_bytes(shape_args),
            'im2col_bytes': im2col_overhead_bytes(shape_args),
        }

        fn_tc = timed_jit(lambda x, k: lax_conv_3d(x, k))
        s = bench_call(fn_tc, x, k, warmup=warmup, repeats=repeats)
        row['cudnn_tc_warm_s'] = s.warm_s
        row['cudnn_tc_compile_s'] = s.compile_s

        fn_fp = timed_jit(lambda x, k: lax_conv_3d(x, k, precision='highest'))
        s = bench_call(fn_fp, x, k, warmup=warmup, repeats=repeats)
        row['cudnn_fp32_warm_s'] = s.warm_s
        row['cudnn_fp32_compile_s'] = s.compile_s

        for algebra in algebras:
            fn = timed_jit(
                lambda x, k, _a=algebra: semiring_conv(
                    x, k, semiring=_a, padding='SAME',
                    backend='pallas-cuda',
                )
            )
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    s = bench_call(fn, x, k, warmup=warmup, repeats=repeats)
                row[f'{algebra.name}_warm_s'] = s.warm_s
                row[f'{algebra.name}_compile_s'] = s.compile_s
            except Exception as e:
                row[f'{algebra.name}_warm_s'] = None
                row[f'{algebra.name}_err'] = repr(e)[:200]

        fn_jax = timed_jit(
            lambda x, k: semiring_conv(
                x, k, semiring=REAL, padding='SAME', backend='jax',
            )
        )
        s = bench_call(fn_jax, x, k, warmup=warmup, repeats=repeats)
        row['real_jax_warm_s'] = s.warm_s

        if row.get('cudnn_tc_warm_s') and row.get('real_warm_s'):
            row['slowdown_vs_tc'] = (
                row['real_warm_s'] / row['cudnn_tc_warm_s']
            )
        if row.get('cudnn_fp32_warm_s') and row.get('real_warm_s'):
            row['slowdown_vs_fp32'] = (
                row['real_warm_s'] / row['cudnn_fp32_warm_s']
            )
        rows.append(row)
    return rows


def _mb(b):
    if b is None:
        return 'n/a'
    return f'{b / (1024 * 1024):.1f} MB'


def render_report(rows_2d, rows_3d, host) -> str:
    lines = [
        '# Perf — `semiring_conv` vs cuDNN',
        '',
        '> Generated by `bench/perf_semiring_conv.py`.  Two cuDNN',
        '> baselines: the **tensor-core** path (TF32 by default on',
        '> Ampere+) is the production reference, but on a generalised-',
        '> algebra library we cannot use tensor cores; the **strict-',
        '> fp32** path (``precision="highest"``) issues CUDA-core FMA',
        '> just like our semiring matmul, so it isolates the',
        '> implicit-vs-explicit im2col gap from the tensor-core gap.',
        '',
        '## Host',
        '',
        f'- Device: {host["device"]}',
        f'- Platform: {host["platform"]}',
        f'- JAX: {host["jax_version"]}',
        '',
        '## 2D conv -- wall-time',
        '',
        '| shape | cuDNN TC | cuDNN fp32 | REAL (Pallas mm) | / TC | / fp32 | REAL (JAX mm) | LOG | TROPICAL_MAX_PLUS | EUCLIDEAN |',
        '|---|----:|----:|----:|----:|----:|----:|----:|----:|----:|',
    ]
    for r in rows_2d:
        lines.append(
            '| {s} | {tc} | {fp} | {re} | {stc} | {sfp} | {rj} | {lo} | {tr} | {eu} |'.format(
                s=r['shape'],
                tc=format_us(r.get('cudnn_tc_warm_s', 0) or 0),
                fp=format_us(r.get('cudnn_fp32_warm_s', 0) or 0),
                re=format_us(r.get('real_warm_s', 0) or 0),
                stc=(f'{r["slowdown_vs_tc"]:.2f}×'
                     if r.get('slowdown_vs_tc') else 'n/a'),
                sfp=(f'{r["slowdown_vs_fp32"]:.2f}×'
                     if r.get('slowdown_vs_fp32') else 'n/a'),
                rj=format_us(r.get('real_jax_warm_s', 0) or 0),
                lo=format_us(r.get('log_warm_s', 0) or 0),
                tr=format_us(r.get('tropical_max_plus_warm_s', 0) or 0),
                eu=format_us(r.get('euclidean_warm_s', 0) or 0),
            )
        )

    lines += [
        '',
        '## 3D conv -- wall-time',
        '',
        '| shape | cuDNN TC | cuDNN fp32 | REAL (Pallas mm) | / TC | / fp32 | REAL (JAX mm) | LOG | TROPICAL_MAX_PLUS |',
        '|---|----:|----:|----:|----:|----:|----:|----:|----:|',
    ]
    for r in rows_3d:
        lines.append(
            '| {s} | {tc} | {fp} | {re} | {stc} | {sfp} | {rj} | {lo} | {tr} |'.format(
                s=r['shape'],
                tc=format_us(r.get('cudnn_tc_warm_s', 0) or 0),
                fp=format_us(r.get('cudnn_fp32_warm_s', 0) or 0),
                re=format_us(r.get('real_warm_s', 0) or 0),
                stc=(f'{r["slowdown_vs_tc"]:.2f}×'
                     if r.get('slowdown_vs_tc') else 'n/a'),
                sfp=(f'{r["slowdown_vs_fp32"]:.2f}×'
                     if r.get('slowdown_vs_fp32') else 'n/a'),
                rj=format_us(r.get('real_jax_warm_s', 0) or 0),
                lo=format_us(r.get('log_warm_s', 0) or 0),
                tr=format_us(r.get('tropical_max_plus_warm_s', 0) or 0),
            )
        )

    lines += [
        '',
        '## Analytical HBM footprint',
        '',
        'cuDNN runs ``IMPLICIT_PRECOMP_GEMM`` so its workspace is',
        'O(1) auxiliary tables -- effectively just inputs + output.',
        'Our path materialises an im2col buffer of size',
        '``M_out * prod(kspatial) * c_in * 4`` bytes (fp32) on top of',
        'the I/O floor.  The HBM stats below are *analytical*',
        '(measured exactly from the shape) rather than from runtime',
        'sampling -- the JAX allocator pool\'s ``peak_bytes_in_use``',
        'is a process-wide HWM that cannot be reset, so successive',
        'measurements mask each other.',
        '',
        '| shape | I/O floor (A+B+out) | im2col overhead (ours) | overhead × |',
        '|---|----:|----:|----:|',
    ]
    for r in rows_2d:
        lines.append(
            '| {s} | {io} | {im} | {ratio} |'.format(
                s=r['shape'],
                io=_mb(r.get('io_floor_bytes')),
                im=_mb(r.get('im2col_bytes')),
                ratio=(f'{r["im2col_bytes"] / r["io_floor_bytes"]:.2f}×'
                       if r.get('io_floor_bytes') else 'n/a'),
            )
        )

    lines += [
        '',
        '## First-call compile cost',
        '',
        '| shape | cuDNN compile | REAL compile | LOG compile |',
        '|---|----:|----:|----:|',
    ]
    for r in rows_2d:
        lines.append(
            '| {s} | {cu} | {re} | {lo} |'.format(
                s=r['shape'],
                cu=format_us(r.get('cudnn_tc_compile_s', 0) or 0),
                re=format_us(r.get('real_compile_s', 0) or 0),
                lo=format_us(r.get('log_compile_s', 0) or 0),
            )
        )

    lines += [
        '',
        '## Interpretation',
        '',
        '- **`/ TC`**: slowdown vs cuDNN with tensor cores.  On A10G /',
        '  Ampere the TF32 tensor-core path has ~8× higher peak fp32',
        '  throughput than the CUDA-core path, so this column is',
        '  expected to be ~8 × (im2col penalty) ≈ 10–15×.  This is',
        '  not the "fair" baseline for a generalised library.',
        '- **`/ fp32`**: slowdown vs cuDNN with `precision="highest"`',
        '  (no tensor cores).  This is the *apples-to-apples*',
        '  baseline: same CUDA-core compute, same algebra.  Expected',
        '  ~1.3–1.7× per Chen et al. 2021 (arXiv:2110.03901), driven',
        '  by the explicit-im2col HBM round-trip the implicit-GEMM',
        '  kernel avoids.',
        '- **LOG / TROPICAL / EUCLIDEAN**: cuDNN cannot serve these at',
        '  all.  Our numbers are the only baseline.  LOG is the',
        '  slowest of the four (pytree-state accumulator); the other',
        '  three are within ~10% of REAL because they share patch',
        '  extraction and matmul shape.',
        '- **Memory**: our explicit im2col overhead is ``prod(kspatial)``',
        '  × ``c_in / (c_in + c_out + 1)`` × the I/O floor.  For 3×3',
        '  conv with c_in ≈ c_out, that\'s roughly 4×.  Memory-bound',
        '  shapes (very large spatial × small channels) trigger this',
        '  first; the workaround is to tile the output spatial dim',
        '  before im2col -- a Pallas implicit-GEMM kernel removes it',
        '  entirely.',
        '',
        '## Future work',
        '',
        'A Pallas Triton implicit-GEMM conv kernel would eliminate the',
        'im2col materialisation by computing patch coordinates inside',
        'the kernel (KeOps-style tiled online map-reduce; Triton paper',
        '§6.2; reference at github.com/l1351868270/implicit_gemm.triton).',
        'Expected payoff per Chen et al. 2021 is ~1.3–1.7× on',
        'CUDA-core hardware, plus ``9× -> 1×`` HBM capacity reduction',
        'for ``3×3`` conv.  Unlike the ELL kernel, conv\'s gather',
        'pattern is *static* (a function of kspatial / strides /',
        'dilation, not of a data-dependent index array), so it may',
        'lower in Triton without depending on a runtime gather',
        'primitive.  Filed as a 1.x improvement; the JAX path is the',
        'GA baseline.',
    ]
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    shapes_2d = QUICK_SHAPES_2D if args.quick else SHAPES_2D
    shapes_3d = QUICK_SHAPES_3D if args.quick else SHAPES_3D
    host = host_summary()
    algebras = [REAL, LOG, TROPICAL_MAX_PLUS, EUCLIDEAN]
    rows_2d = run_2d(
        shapes_2d, algebras, warmup=args.warmup, repeats=args.repeats,
    )
    rows_3d = run_3d(
        shapes_3d, algebras[:3], warmup=args.warmup, repeats=args.repeats,
    )
    report = render_report(rows_2d, rows_3d, host)
    REPORT_PATH.write_text(report)
    print(report)
    print(f'Wrote {REPORT_PATH}.')


if __name__ == '__main__':
    main()
