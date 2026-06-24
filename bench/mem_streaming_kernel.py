# -*- coding: utf-8 -*-
"""Peak-HBM measurement: does the streaming kernel materialise (M, K, N)?

Per SPEC §4.1 the KeOps-style streaming kernel claims peak on-chip
memory of ``O(BM*BN + BM*BK + BK*BN)`` — never the full
``(BM, BK, BN)`` value tensor.  This script verifies the *device*-side
shape of that claim by reading ``jax.devices()[0].memory_stats()
['peak_bytes_in_use']`` before and after each call.

What this measures and what it doesn't:

- It **does** measure HBM (global-memory) allocations made by JAX /
  XLA / Pallas while executing the call.  This catches the bad case:
  XLA failing to fuse the K loop body and emitting an explicit
  ``(M, K, N)`` intermediate buffer.
- It **does not** measure on-chip SMEM / register use.  Pallas
  kernels do their work in registers; the tile resident state never
  hits HBM, so it's invisible here by design.  The reported Pallas
  peak is effectively just ``M*N`` output + input read-back.

The naive ``(A[:, :, None] OP B[None, :, :]).reduce(axis=1)``
formulation is the failure mode the streaming design is meant to
avoid.  We *don't* run it here -- XLA's compile time grows with the
size of the materialised ``(M, K, N)`` intermediate (60+ seconds for
``(512, 256, 512)`` on Ampere even with ``jit``), and the answer is
analytical anyway: it allocates ``4 * M * K * N`` bytes at fp32.  The
table below reports the analytical naive floor next to the measured
streaming-path peaks so the comparison is explicit.

The measurement protocol drops first-call compile-induced
allocations via a warm-up run, then records the *delta* in peak
HBM around the timed run.  Because ``peak_bytes_in_use`` is a
high-water mark, we read it on the live process and reason in
deltas to avoid getting confused by cached compilations.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import jax
import jax.numpy as jnp
from _util import host_summary, timed_jit

from nitrix.semiring import (
    LOG,
    REAL,
    TROPICAL_MAX_PLUS,
    reference_semiring_matmul,
    semiring_matmul,
)

HERE = Path(__file__).parent
REPORT_PATH = HERE / 'MEM_STREAMING_KERNEL.md'


SHAPES = [
    # (M, K, N).  Both shapes have M*K*N >> M*N so streaming wins are visible.
    (256, 256, 256),  # naive-analytical: 64 MB intermediate vs 0.25 MB output
    (512, 256, 512),  # naive-analytical: 128 MB intermediate vs 1 MB output
    (1024, 256, 1024),  # naive-analytical: 1024 MB intermediate vs 4 MB output
    (2048, 512, 2048),  # naive-analytical: 8 GB intermediate vs 16 MB output
]
QUICK_SHAPES = [
    (256, 256, 256),
    (512, 256, 512),
]

ALGEBRAS = [REAL, LOG, TROPICAL_MAX_PLUS]


def make_pair(m: int, k: int, n: int, seed: int):
    key = jax.random.key(seed)
    ka, kb = jax.random.split(key)
    A = jax.random.normal(ka, (m, k), dtype=jnp.float32)
    B = jax.random.normal(kb, (k, n), dtype=jnp.float32)
    A.block_until_ready()
    B.block_until_ready()
    return A, B


def _peak() -> int:
    return int(jax.devices()[0].memory_stats()['peak_bytes_in_use'])


def measure_peak_delta(fn, *args, warmup: int = 2) -> int:
    """HBM peak-bytes growth attributable to one call to ``fn(*args)``.

    Strategy:

    1. Run ``warmup`` calls and discard them; this seeds the JIT
       cache and pushes the HWM to wherever a steady-state call
       leaves it.
    2. ``gc.collect()`` and snapshot the post-warm-up HWM.
    3. Run *one more* call and snapshot the HWM again.
    4. The delta is non-zero only if step 3 allocated something
       larger than anything seen during warm-up -- i.e., something
       unique to this measured call.

    For a steady-state kernel this delta will be exactly 0: warm-up
    has already exercised the same allocation pattern.  We use the
    delta as a regression detector rather than as the headline
    number.  The *peak-after-warmup* is what we report -- that's the
    headline ``HBM ceiling`` for the kernel.
    """
    for _ in range(warmup):
        out = fn(*args)
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
        del out
    gc.collect()
    before = _peak()
    out = fn(*args)
    if hasattr(out, 'block_until_ready'):
        out.block_until_ready()
    after = _peak()
    del out
    gc.collect()
    return after - before


def measure_peak_ceiling(fn, *args, warmup: int = 2) -> int:
    """Steady-state peak HBM HWM after ``warmup`` calls.

    This is the headline number: "running this kernel at this shape
    drives the JAX HBM pool to at least this size."  Subtract input
    + output bytes to back out the per-call intermediate footprint.
    """
    for _ in range(warmup):
        out = fn(*args)
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
        del out
    gc.collect()
    return _peak()


def streaming_jax(algebra):
    def f(A, B):
        return reference_semiring_matmul(A, B, semiring=algebra)

    return f


def pallas(algebra):
    def f(A, B):
        return semiring_matmul(A, B, semiring=algebra, backend='pallas-cuda')

    return f


def run(shapes, algebras, warmup):
    rows = []
    for shape in shapes:
        m, k, n = shape
        A, B = make_pair(m, k, n, seed=hash(shape) & 0xFFFF)
        # Sanity-check baseline so the ceiling we read is meaningful.
        io_floor = 4 * (m * k + k * n + m * n)
        for algebra in algebras:
            row = {
                'm': m,
                'k': k,
                'n': n,
                'algebra': algebra.name,
                'mkn_bytes_fp32': 4 * m * k * n,
                'mn_bytes_fp32': 4 * m * n,
                'io_floor_bytes_fp32': io_floor,
            }
            for label, fn_factory in [
                ('stream_jax', streaming_jax),
                ('pallas', pallas),
            ]:
                fn = timed_jit(fn_factory(algebra))
                try:
                    row[f'{label}_ceiling'] = measure_peak_ceiling(
                        fn,
                        A,
                        B,
                        warmup=warmup,
                    )
                    row[f'{label}_delta'] = measure_peak_delta(
                        fn,
                        A,
                        B,
                        warmup=0,  # already warm
                    )
                except Exception as e:
                    row[f'{label}_ceiling'] = None
                    row[f'{label}_err'] = repr(e)[:200]
            rows.append(row)
    return rows


def _mb(b: int | None) -> str:
    if b is None:
        return 'n/a'
    return f'{b / (1024 * 1024):.2f} MB'


def render_report(rows, host) -> str:
    lines = [
        '# Memory streaming check — `semiring_matmul`',
        '',
        '> Generated by `bench/mem_streaming_kernel.py`.  Reads',
        '> `jax.devices()[0].memory_stats()` to verify our streaming',
        '> kernels do not materialise an `(M, K, N)` intermediate in',
        '> HBM.',
        '',
        '## What the columns mean',
        '',
        '- **naive (M·K·N, fp32)**: *analytical* HBM cost of the',
        '  reference formulation `(A[:, :, None] OP B[None, :, :]).reduce(axis=1)`.',
        "  We do not run this -- XLA's compile time on the materialised",
        '  fusion blows past 60 s for shapes ≥ `(512, 256, 512)` on',
        '  Ampere.  Listed as the failure-mode the streaming kernels',
        '  are supposed to avoid.',
        '- **output (M·N, fp32)**: lower-bound on per-call HBM growth',
        '  (the output buffer itself).',
        '- **streaming JAX ceiling**: peak HBM `bytes_in_use` after',
        '  warm-up using `reference_semiring_matmul` (`lax.fori_loop`',
        '  over K, body folds rank-1 outer combines into a (M, N)',
        '  accumulator).  Includes A, B, output, and any per-step',
        '  intermediates XLA chose not to fuse.',
        '- **Pallas ceiling**: same metric for the Pallas kernel.',
        '  Per-tile intermediates live in registers / SMEM, so they',
        '  are *invisible* to HBM accounting; the ceiling here is the',
        '  irreducible A + B + output footprint plus pool slack.',
        '- **delta**: HBM growth on one additional steady-state call',
        '  after warm-up.  Expected 0 for a stable kernel; non-zero',
        '  indicates a leak or a per-call allocation that grew past',
        '  the warm-up HWM.',
        '',
        '## Host',
        '',
        f'- Device: {host["device"]}',
        f'- Platform: {host["platform"]}',
        f'- JAX: {host["jax_version"]}',
        '',
        '## Steady-state HBM footprint',
        '',
        '| m | k | n | algebra | naive (analytical) | output | streaming JAX ceiling | Δ | Pallas ceiling | Δ |',
        '|---|---|---|---------|-------------------:|-------:|----------------------:|--:|---------------:|--:|',
    ]
    for r in rows:
        lines.append(
            '| {m} | {k} | {n} | {a} | {mkn} | {mn} | {sj} | {sjd} | {p} | {pd} |'.format(
                m=r['m'],
                k=r['k'],
                n=r['n'],
                a=r['algebra'],
                mkn=_mb(r['mkn_bytes_fp32']),
                mn=_mb(r['mn_bytes_fp32']),
                sj=_mb(r.get('stream_jax_ceiling')),
                sjd=_mb(r.get('stream_jax_delta', 0)),
                p=_mb(r.get('pallas_ceiling')),
                pd=_mb(r.get('pallas_delta', 0)),
            )
        )
    lines += [
        '',
        '## Interpretation',
        '',
        '- A streaming-kernel ceiling close to `A + B + output ≈ ',
        '  4·(M·K + K·N + M·N) bytes` (the I/O floor) means the kernel',
        '  is doing what the spec promises: no full `(M, K, N)`',
        '  intermediate hits HBM.',
        '- A streaming-kernel ceiling that scales with `M·K·N` means',
        '  XLA failed to fuse the K loop body and is keeping a',
        '  per-step `(M, N)` intermediate live across steps, or worse,',
        '  the entire `(M, K, N)` tensor.  That is a regression to be',
        '  filed against the kernel.',
        '- The Pallas ceiling can be **lower** than the streaming JAX',
        '  ceiling for the same shape, because XLA may keep a couple',
        '  of (M, N) intermediates live for the JAX path while the',
        '  Pallas path resides entirely in registers (input buffers',
        '  read once + output written once).',
    ]
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    shapes = QUICK_SHAPES if args.quick else SHAPES
    host = host_summary()
    rows = run(shapes, ALGEBRAS, warmup=args.warmup)
    report = render_report(rows, host)
    REPORT_PATH.write_text(report)
    print(report)
    print(f'Wrote {REPORT_PATH}.')


if __name__ == '__main__':
    main()
