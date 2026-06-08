# -*- coding: utf-8 -*-
"""G0 — Ampere ELL policy + JAX-path baseline.

The G0 gate in IMPLEMENTATION_PLAN §3.1 asks whether Pallas Triton
is viable as the default backend for ``semiring_ell_matmul`` on
Ampere.  At first GA the answer is **no**, and the reason is
structural rather than perf-marginal: the Pallas Triton backend in
the pinned JAX version does not lower the ``gather`` primitive
(nor axis-0 ``concatenate`` of arbitrary fan-out), both of which the
ELL streaming kernel needs to do per-row neighbour lookups into the
dense operand.  See ``src/nitrix/_kernels/cuda/semiring_ell_matmul.py``
for the exact constraint and the upstream tracking surface.

Consequently the policy for first GA is:

  ``semiring_ell_matmul`` runs on the JAX backend unconditionally
  on Ampere+; passing ``backend="pallas-cuda"`` resolves to JAX and
  emits a ``NitrixBackendFallback`` warning on first use.

This script's job is therefore to (a) re-verify the structural
fallback is still in effect on the current JAX pin -- a future JAX
release adding gather lowering would change the picture -- and (b)
publish a JAX-path *baseline* so downstream callers know what
sustained throughput to expect on representative mesh-adjacency
shapes.

When the picture changes, switch this script back into a Pallas-vs-
JAX comparison and re-evaluate per the IMPLEMENTATION_PLAN §3.1
decision rule.

Reproduce::

    python bench/g0_ampere_ell.py
    python bench/g0_ampere_ell.py --quick
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
from _util import (
    BenchSample,
    bench_call,
    format_us,
    host_summary,
    timed_jit,
)

from nitrix._internal.backend import (
    NitrixBackendFallback,
    reset_fallback_state,
)
from nitrix.semiring import (
    LOG,
    REAL,
    TROPICAL_MAX_PLUS,
    semiring_ell_matmul,
)

HERE = Path(__file__).parent
REPORT_PATH = HERE / 'G0_ELL_REPORT.md'


SHAPES = [
    # (m, k_max, ncol) -- mesh-adjacency-like workload.
    (1024, 16, 32),
    (4096, 16, 64),
    (16384, 32, 32),
    (65536, 32, 32),
]
QUICK_SHAPES = [(1024, 16, 32), (4096, 32, 32)]
ALGEBRAS = [REAL, LOG, TROPICAL_MAX_PLUS]


def make_mesh_ell(m: int, k_max: int, ncol: int, seed: int = 0):
    key = jax.random.key(seed)
    kv, ki, kb = jax.random.split(key, 3)
    values = jax.random.normal(kv, (m, k_max), dtype=jnp.float32)
    indices = jax.random.randint(ki, (m, k_max), 0, m).astype(jnp.int32)
    B = jax.random.normal(kb, (m, ncol), dtype=jnp.float32)
    return (
        jax.block_until_ready(values),
        jax.block_until_ready(indices),
        jax.block_until_ready(B),
    )


def confirm_pallas_falls_back() -> bool:
    """Run a tiny Pallas-requested call and check the warning fires.

    Returns True iff the Pallas path resolved to a JAX fallback.
    A False here means the Pallas ELL kernel is now usable and the
    policy in this report needs revisiting.
    """
    m, k_max, ncol = 64, 8, 32
    v, idx, B = make_mesh_ell(m, k_max, ncol, seed=99)
    reset_fallback_state()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        _ = semiring_ell_matmul(
            v,
            idx,
            B,
            semiring=REAL,
            n_cols=m,
            backend='pallas-cuda',
        )
    return any(w.category is NitrixBackendFallback for w in ws)


def run(shapes, algebras, warmup, repeats):
    rows = []
    for shape in shapes:
        m, k_max, ncol = shape
        v, idx, B = make_mesh_ell(m, k_max, ncol, seed=hash(shape) & 0xFFFF)
        for algebra in algebras:
            row = {
                'm': m,
                'k_max': k_max,
                'ncol': ncol,
                'algebra': algebra.name,
            }
            fn = timed_jit(
                lambda v_, i_, B_, _a=algebra, _m=m: semiring_ell_matmul(
                    v_,
                    i_,
                    B_,
                    semiring=_a,
                    n_cols=_m,
                    backend='jax',
                )
            )
            try:
                sample: BenchSample = bench_call(
                    fn,
                    v,
                    idx,
                    B,
                    warmup=warmup,
                    repeats=repeats,
                )
                row['jax_compile_s'] = sample.compile_s
                row['jax_warm_s'] = sample.warm_s
                # Approximate effective throughput in M_entries/s.
                nnz = m * k_max
                row['effective_meps'] = (nnz * ncol) / sample.warm_s / 1e6
            except Exception as e:  # pragma: no cover
                row['jax_warm_s'] = None
                row['jax_err'] = repr(e)[:200]
            rows.append(row)
    return rows


def render_report(rows, host, pallas_falls_back: bool) -> str:
    lines = [
        '# G0 — Ampere ELL policy + JAX-path baseline',
        '',
        '> SPEC reference: IMPLEMENTATION_PLAN §3.1; SPEC_UPDATE_v0.2 §4.',
        '> This is a *policy* document plus the JAX-path baseline; the',
        '> Pallas-vs-JAX wall-time comparison is deferred until Pallas',
        '> Triton lowers the `gather` primitive on the pinned JAX.',
        '',
        '## Policy',
        '',
        (
            '`semiring_ell_matmul` runs on the **JAX backend unconditionally** '
            'on Ampere+; `backend="pallas-cuda"` resolves to JAX and emits '
            'one `NitrixBackendFallback` warning per `(shape, dtype, '
            'algebra)` signature.'
        ),
        '',
        f'Pallas-falls-back probe (current JAX pin): {"yes" if pallas_falls_back else "**no -- revisit policy**"}.',
        '',
        '## Host',
        '',
        f'- Device: {host["device"]}',
        f'- Platform: {host["platform"]}',
        f'- JAX: {host["jax_version"]}',
        '',
        '## JAX-path baseline (post-warm-up steady state)',
        '',
        '| m | k_max | ncol | algebra | JAX wall-time | JAX compile | effective Mevents/s |',
        '|---|------:|-----:|---------|--------------:|------------:|--------------------:|',
    ]
    for r in rows:
        j = r.get('jax_warm_s')
        jc = r.get('jax_compile_s')
        eps = r.get('effective_meps')
        lines.append(
            '| {m} | {k_max} | {ncol} | {a} | {j} | {jc} | {eps} |'.format(
                m=r['m'],
                k_max=r['k_max'],
                ncol=r['ncol'],
                a=r['algebra'],
                j=format_us(j) if j else 'n/a',
                jc=format_us(jc) if jc else 'n/a',
                eps=f'{eps:.1f}' if eps else 'n/a',
            )
        )

    lines += [
        '',
        '## Decision (per IMPLEMENTATION_PLAN §3.1)',
        '',
        '**JAX-default on Ampere+, unconditional for first GA.**  The',
        'reason is structural (Pallas Triton lacks gather lowering),',
        'not a performance-margin call.  Revisit when either: (a) JAX',
        'lands gather in the Triton lowering, or (b) we accept a',
        'shape-specialised Pallas variant that avoids gather (e.g.,',
        'small fixed `k_max` with fully Python-unrolled per-row',
        'loads, gated by tile size).',
        '',
        '## Why ELL is JAX-only at GA',
        '',
        'The ELL streaming kernel reads, for each output row, the',
        'rows of `B` at the column indices `indices[i, :]`.  In',
        'JAX this lowers to `jnp.take` / `lax.gather` over `B`, which',
        'Triton-on-Pallas does not currently lower.  Two workarounds',
        'were tried and rejected:',
        '',
        '- **Per-row `pl.ds` ref loads inside a Python-unrolled loop',
        '  over BM rows.**  The loaded rows must be stacked back into',
        '  the `(BM, BN)` tile shape via concatenation along axis 0;',
        '  Triton supports only axis-(-1) concatenate.',
        '- **`jnp.take` inside the kernel body.**  Lowers to `gather`',
        '  in HLO; same lowering gap.',
        '',
        'The right fix is upstream (Pallas Triton landing gather) or',
        'a separate hand-rolled kernel layout (Hopper Mosaic GPU,',
        'which we explicitly exclude per §1.1).  Both are 1.x',
        'conversations.',
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
    pallas_falls_back = confirm_pallas_falls_back()
    rows = run(shapes, ALGEBRAS, warmup=args.warmup, repeats=args.repeats)
    report = render_report(rows, host, pallas_falls_back)
    REPORT_PATH.write_text(report)
    print(report)
    print(f'Wrote {REPORT_PATH}.')


if __name__ == '__main__':
    main()
