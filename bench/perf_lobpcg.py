# -*- coding: utf-8 -*-
"""Performance benchmark for the differentiable LOBPCG paths.

Measures forward and forward-plus-backward wall-time for:

- ``lobpcg_top_k_dense`` (implicit-VJP) against ``jnp.linalg.eigh``
  (the dense-spectrum reference).
- ``lobpcg_top_k_ell`` (sparsity-projected backward) against the
  same dense ``lobpcg_top_k_dense`` at matched ``n``.

The ELL-vs-dense comparison is the load-bearing claim of the
sparsity-projected backward: the backward cost should scale with
``nnz * k + n * k^2``, **not** ``n^2``.  XLA can defeat this if it
fuses the gather into a dense materialisation of ``U K U^T``; we
guard against that with a separate HLO-shape audit (see
``_assert_no_n_squared_in_hlo``).

The HLO audit also reports the largest single dimension appearing
in the compiled program; for the ELL path this should be roughly
``5 n`` (the LOBPCG internal search-subspace footprint), not
``n^2``.

Run::

    python bench/perf_lobpcg.py

Writes ``PERF_LOBPCG.md`` alongside this script.  No environment
overrides are needed; the script picks the default JAX device.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from _util import BenchSample, bench_call, format_us, host_summary, timed_jit
from nitrix.graph._lobpcg_diff import lobpcg_top_k_dense, lobpcg_top_k_ell

from nitrix.semiring import REAL, semiring_ell_matmul

# ---------------------------------------------------------------------------
# Operand construction
# ---------------------------------------------------------------------------


def _ring_with_chords(
    n: int, n_chords: int = 0, *, seed: int = 0
) -> np.ndarray:
    """Build a dense ``(n, n)`` ring-with-chords adjacency.

    Used for paths that need the dense operand (the ``dense``
    LOBPCG row and the ``eigh`` reference).  ``n`` must fit in
    host memory at ``n^2 * 4`` bytes.  For ELL-only benches use
    ``_ring_with_chords_ell`` which builds the ELL representation
    directly without the dense intermediate.
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = 1.0
        A[(i + 1) % n, i] = 1.0
    for _ in range(n_chords):
        i, j = rng.integers(0, n, 2)
        if i != j:
            A[i, j] = A[j, i] = 1.0
    A += 0.5 * np.eye(n)
    return A.astype(np.float32)


def _ring_with_chords_ell(
    n: int,
    n_chords: int = 0,
    *,
    seed: int = 0,
) -> Tuple[jax.Array, jax.Array, int, int]:
    """Build a ring-with-chords adjacency directly in ELL form.

    Avoids the ``O(n^2)`` dense intermediate; scales to ``n = 1M+``
    if needed.  Returns ``(values, indices, n_cols, nnz)``.
    """
    rng = np.random.default_rng(seed)
    # Adjacency list per row.  Start with self-loop, ring neighbours.
    adj: list[list[int]] = [[i, (i - 1) % n, (i + 1) % n] for i in range(n)]
    for _ in range(n_chords):
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i == j:
            continue
        if j not in adj[i]:
            adj[i].append(j)
        if i not in adj[j]:
            adj[j].append(i)
    k_max = max(len(row) for row in adj)
    values = np.zeros((n, k_max), dtype=np.float32)
    indices = np.zeros((n, k_max), dtype=np.int32)
    nnz = 0
    for i, row in enumerate(adj):
        for p, j in enumerate(row):
            values[i, p] = 0.5 if i == j else 1.0  # self / off-diag weights
        indices[i, : len(row)] = row
        if len(row) < k_max:
            indices[i, len(row) :] = row[0]
        nnz += len(row)
    return jnp.asarray(values), jnp.asarray(indices), int(n), nnz


def _to_ell(A_dense: np.ndarray) -> Tuple[jax.Array, jax.Array, int]:
    n = A_dense.shape[0]
    k_max = int(np.max((A_dense != 0).sum(axis=1)))
    values = np.zeros((n, k_max), dtype=A_dense.dtype)
    indices = np.zeros((n, k_max), dtype=np.int32)
    for i in range(n):
        nz = np.nonzero(A_dense[i])[0]
        values[i, : len(nz)] = A_dense[i, nz]
        indices[i, : len(nz)] = nz
        if len(nz) < k_max:
            indices[i, len(nz) :] = nz[0]
    return jnp.asarray(values), jnp.asarray(indices), k_max


# ---------------------------------------------------------------------------
# Loss / gradient closures (jit-able)
# ---------------------------------------------------------------------------


def _make_dense_loss(
    X0: jax.Array,
    target: jax.Array,
    n_iters: int,
    k: int,
):
    def loss(M):
        _, U = lobpcg_top_k_dense(M, X0, n_iters, None, 1e-8)
        return jnp.trace(U.T @ M @ U @ target)

    return loss


def _make_ell_loss(
    indices: jax.Array,
    X0: jax.Array,
    target: jax.Array,
    n_cols: int,
    n_iters: int,
):
    def loss(values):
        _, U = lobpcg_top_k_ell(
            values,
            indices,
            X0,
            n_cols,
            n_iters,
            None,
            1e-8,
        )
        AU = semiring_ell_matmul(
            values,
            indices,
            U,
            semiring=REAL,
            n_cols=n_cols,
            backend='jax',
        )
        return jnp.trace(U.T @ AU @ target)

    return loss


# ---------------------------------------------------------------------------
# HLO inspection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HLOReport:
    """Summary of the compiled HLO's shape footprint.

    ``max_single_dim`` is the largest single axis appearing on any
    tensor in the compiled program; ``has_n_squared`` is True iff a
    tensor with two axes equal to ``n`` is materialised.
    """

    max_single_dim: int
    has_n_squared: bool
    distinct_shapes_top10: tuple[tuple[tuple[int, ...], int], ...]


def _audit_hlo(grad_fn, *args, n: int) -> HLOReport:
    """Compile ``grad_fn`` and audit its HLO shape footprint.

    The audit looks for ``f32[...]`` and ``f64[...]`` shape tokens
    in the textual HLO; XLA fold / fusion passes shouldn't introduce
    higher-rank intermediates than what we'd predict from the
    backward formula.
    """
    hlo = grad_fn.lower(*args).compile().as_text()
    shapes = re.findall(r'f(?:32|64)\[([0-9,]+)\]', hlo)
    counts: dict[tuple[int, ...], int] = {}
    max_dim = 0
    has_n2 = False
    for s in shapes:
        dims = tuple(int(x) for x in s.split(',') if x)
        if not dims:
            continue
        max_dim = max(max_dim, max(dims))
        counts[dims] = counts.get(dims, 0) + 1
        n_axes = [d for d in dims if d == n]
        if len(n_axes) >= 2:
            has_n2 = True
    top10 = tuple(
        sorted(
            counts.items(),
            key=lambda kv: -np.prod(kv[0]) * kv[1],
        )[:10]
    )
    return HLOReport(
        max_single_dim=max_dim,
        has_n_squared=has_n2,
        distinct_shapes_top10=top10,
    )


# ---------------------------------------------------------------------------
# Bench drivers
# ---------------------------------------------------------------------------


@dataclass
class LobpcgBenchRow:
    n: int
    k: int
    nnz: Optional[int]
    path: str  # 'dense' / 'ell' / 'eigh'
    forward: BenchSample
    grad: BenchSample
    hlo: Optional[HLOReport]

    @property
    def bwd_only(self) -> float:
        """Bwd wall-time = grad time minus forward time.

        Approximate (some kernel-launch overhead double-counts) but
        accurate to first order at fixed n.
        """
        return max(self.grad.warm_s - self.forward.warm_s, 0.0)

    def report_line(self) -> str:
        nnz_str = '-' if self.nnz is None else str(self.nnz)
        hlo_str = (
            '-'
            if self.hlo is None
            else (
                f'max={self.hlo.max_single_dim} '
                f'n2={"y" if self.hlo.has_n_squared else "n"}'
            )
        )
        return (
            f'| {self.n:>5d} | {self.k:>3d} | {nnz_str:>6s} | '
            f'{self.path:>6s} | {format_us(self.forward.warm_s):>10s} | '
            f'{format_us(self.grad.warm_s):>10s} | '
            f'{format_us(self.bwd_only):>10s} | {hlo_str} |'
        )


@dataclass
class PureVjpBenchRow:
    """Pure bwd-formula timing, no LOBPCG iteration in the path.

    Inputs to the bwd are the already-computed eigenpairs and the
    cotangents; the kernel only does the F-matrix construction plus
    the (gather + einsum / matmul) projection.  This is the cleanest
    measure of "did XLA preserve our O(nnz*k + n*k^2) ELL backward".
    """

    n: int
    k: int
    nnz: Optional[int]
    path: str  # 'dense' / 'ell'
    bench: BenchSample
    hlo: HLOReport

    def report_line(self) -> str:
        nnz_str = '-' if self.nnz is None else str(self.nnz)
        return (
            f'| {self.n:>5d} | {self.k:>3d} | {nnz_str:>6s} | '
            f'{self.path:>6s} | {format_us(self.bench.warm_s):>10s} | '
            f'max={self.hlo.max_single_dim} '
            f'n2={"y" if self.hlo.has_n_squared else "n"} |'
        )


def bench_dense_lobpcg(
    n: int,
    k: int,
    *,
    n_iters: int = 200,
    n_chords: int = 0,
    seed: int = 0,
    repeats: int = 8,
) -> LobpcgBenchRow:
    A = _ring_with_chords(n, n_chords=n_chords, seed=seed)
    M = jnp.asarray(A)
    X0 = jax.random.normal(jax.random.key(seed), (n, k), dtype=jnp.float32)
    target = jax.random.normal(jax.random.key(seed + 1), (k, k))
    target = (target + target.T) / 2

    loss = _make_dense_loss(X0, target, n_iters, k)
    loss_jit = timed_jit(loss)
    grad_jit = timed_jit(jax.grad(loss))

    fwd_sample = bench_call(loss_jit, M, warmup=2, repeats=repeats)
    grad_sample = bench_call(grad_jit, M, warmup=2, repeats=repeats)
    hlo = _audit_hlo(grad_jit, M, n=n)
    return LobpcgBenchRow(
        n=n,
        k=k,
        nnz=None,
        path='dense',
        forward=fwd_sample,
        grad=grad_sample,
        hlo=hlo,
    )


def bench_ell_lobpcg(
    n: int,
    k: int,
    *,
    n_iters: int = 200,
    n_chords: int = 0,
    seed: int = 0,
    repeats: int = 8,
) -> LobpcgBenchRow:
    values, indices, _, nnz = _ring_with_chords_ell(
        n,
        n_chords=n_chords,
        seed=seed,
    )
    X0 = jax.random.normal(jax.random.key(seed), (n, k), dtype=jnp.float32)
    target = jax.random.normal(jax.random.key(seed + 1), (k, k))
    target = (target + target.T) / 2

    loss = _make_ell_loss(indices, X0, target, n, n_iters)
    loss_jit = timed_jit(loss)
    grad_jit = timed_jit(jax.grad(loss))

    fwd_sample = bench_call(loss_jit, values, warmup=2, repeats=repeats)
    grad_sample = bench_call(grad_jit, values, warmup=2, repeats=repeats)
    hlo = _audit_hlo(grad_jit, values, n=n)
    return LobpcgBenchRow(
        n=n,
        k=k,
        nnz=nnz,
        path='ell',
        forward=fwd_sample,
        grad=grad_sample,
        hlo=hlo,
    )


def bench_pure_vjp_dense(
    n: int,
    k: int,
    *,
    repeats: int = 12,
    seed: int = 0,
) -> PureVjpBenchRow:
    """Time the dense bwd formula alone (no LOBPCG iteration).

    We synthesize concrete ``(eigvals, eigvecs)`` from a random
    symmetric ``M`` via ``jnp.linalg.eigh`` (or a synthetic
    orthogonal basis) and measure the time of the closed-form
    backward map ``(g_eigvals, g_eigvecs) -> dM``.  This isolates
    the per-call overhead of the implicit VJP kernel from the
    LOBPCG iteration body, exposing any XLA pessimisation cleanly.
    """
    from nitrix.graph._lobpcg_diff import _subspace_vjp_kernel

    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, k)).astype(np.float32))
    eigvals = jnp.asarray(
        np.sort(rng.uniform(0.5, 3.0, size=k)).astype(np.float32)[::-1]
    )
    eigvecs = jnp.asarray(Q)
    g_eigvals = jax.random.normal(
        jax.random.key(seed), (k,), dtype=jnp.float32
    )
    g_eigvecs = jax.random.normal(
        jax.random.key(seed + 1), (n, k), dtype=jnp.float32
    )

    def bwd_kernel(eigvals, eigvecs, g_eigvals, g_eigvecs):
        K = _subspace_vjp_kernel(eigvals, eigvecs, g_eigvals, g_eigvecs, 1e-8)
        return eigvecs @ K @ eigvecs.T

    bwd_jit = timed_jit(bwd_kernel)
    sample = bench_call(
        bwd_jit,
        eigvals,
        eigvecs,
        g_eigvals,
        g_eigvecs,
        warmup=3,
        repeats=repeats,
    )
    hlo = _audit_hlo(bwd_jit, eigvals, eigvecs, g_eigvals, g_eigvecs, n=n)
    return PureVjpBenchRow(
        n=n,
        k=k,
        nnz=None,
        path='dense',
        bench=sample,
        hlo=hlo,
    )


def bench_pure_vjp_ell(
    n: int,
    k: int,
    *,
    n_chords: int = 0,
    repeats: int = 12,
    seed: int = 0,
) -> PureVjpBenchRow:
    """Time the ELL bwd formula alone.

    Same construction as ``bench_pure_vjp_dense`` but the projection
    step gathers + einsums onto an ELL sparsity pattern.  The hot
    path: ``VK = U @ K`` ``(n,k) @ (k,k) -> (n,k)``;
    ``V_at_idx = U[indices]`` ``(n, k_max, k)`` gather;
    ``einsum('ij,ipj->ip', VK, V_at_idx)`` ``-> (n, k_max)``.
    Total ``O(nnz * k + n * k^2)``; nothing should materialise an
    ``(n, n)`` intermediate.
    """
    from nitrix.graph._lobpcg_diff import _subspace_vjp_kernel

    _, indices, _, nnz = _ring_with_chords_ell(
        n,
        n_chords=n_chords,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, k)).astype(np.float32))
    eigvals = jnp.asarray(
        np.sort(rng.uniform(0.5, 3.0, size=k)).astype(np.float32)[::-1]
    )
    eigvecs = jnp.asarray(Q)
    g_eigvals = jax.random.normal(
        jax.random.key(seed), (k,), dtype=jnp.float32
    )
    g_eigvecs = jax.random.normal(
        jax.random.key(seed + 1), (n, k), dtype=jnp.float32
    )

    def bwd_kernel(eigvals, eigvecs, g_eigvals, g_eigvecs, indices):
        K = _subspace_vjp_kernel(eigvals, eigvecs, g_eigvals, g_eigvecs, 1e-8)
        VK = eigvecs @ K  # (n, k)
        V_at_idx = eigvecs[indices]  # (n, k_max, k)
        return jnp.einsum('ij,ipj->ip', VK, V_at_idx)

    bwd_jit = timed_jit(bwd_kernel)
    sample = bench_call(
        bwd_jit,
        eigvals,
        eigvecs,
        g_eigvals,
        g_eigvecs,
        indices,
        warmup=3,
        repeats=repeats,
    )
    hlo = _audit_hlo(
        bwd_jit,
        eigvals,
        eigvecs,
        g_eigvals,
        g_eigvecs,
        indices,
        n=n,
    )
    return PureVjpBenchRow(
        n=n,
        k=k,
        nnz=nnz,
        path='ell',
        bench=sample,
        hlo=hlo,
    )


def bench_eigh(
    n: int,
    k: int,
    *,
    repeats: int = 8,
    seed: int = 0,
) -> LobpcgBenchRow:
    """Dense ``jnp.linalg.eigh`` reference: full spectrum then slice top-k.

    Backward gradient via JAX's built-in eigh VJP.  This is the
    upper bound on what dense LOBPCG should hope to beat (for n
    where eigh fits in memory) and the only reference for ELL at
    moderate n.
    """
    A = _ring_with_chords(n, seed=seed)
    M = jnp.asarray(A)
    target = jax.random.normal(jax.random.key(seed + 1), (k, k))
    target = (target + target.T) / 2

    def loss(M):
        ev_all, V_all = jnp.linalg.eigh(M)
        U = V_all[:, -k:]
        return jnp.trace(U.T @ M @ U @ target)

    loss_jit = timed_jit(loss)
    grad_jit = timed_jit(jax.grad(loss))
    fwd_sample = bench_call(loss_jit, M, warmup=2, repeats=repeats)
    grad_sample = bench_call(grad_jit, M, warmup=2, repeats=repeats)
    return LobpcgBenchRow(
        n=n,
        k=k,
        nnz=None,
        path='eigh',
        forward=fwd_sample,
        grad=grad_sample,
        hlo=None,
    )


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


def _write_report(
    rows: list[LobpcgBenchRow],
    pure_rows: list[PureVjpBenchRow],
    output: Path,
) -> None:
    host = host_summary()
    lines = []
    lines.append('# LOBPCG implicit-VJP performance')
    lines.append('')
    lines.append('## Host')
    for kk, vv in host.items():
        lines.append(f'- {kk}: {vv}')
    lines.append('')
    lines.append('## End-to-end: forward + backward through LOBPCG')
    lines.append('')
    lines.append(
        'Times include the LOBPCG iteration (typically dominant) '
        'plus the implicit-VJP backward.  ``bwd_only`` is the '
        'subtraction of ``forward`` from ``fwd+bwd``; for the ELL '
        'path it captures the backward formula only, since LOBPCG '
        'is run the same way in both calls.'
    )
    lines.append('')
    lines.append(
        '| n     | k   | nnz    |   path | forward    | fwd+bwd    | bwd_only   | HLO audit |'
    )
    lines.append(
        '|------:|----:|-------:|-------:|-----------:|-----------:|-----------:|:----------|'
    )
    for row in rows:
        lines.append(row.report_line())
    lines.append('')
    lines.append('## Pure backward kernel (no LOBPCG iteration)')
    lines.append('')
    lines.append(
        'This isolates the implicit-VJP formula itself.  For '
        'a synthetic orthogonal ``U`` and random ``Λ``, time the '
        'closed-form backward map ``(g_λ, g_U) -> dM`` (dense) or '
        '``(g_λ, g_U) -> d_values`` (ELL).  The ELL row tests the '
        'sparsity-projected ``O(nnz * k + n * k^2)`` claim '
        'directly, without LOBPCG iteration noise.'
    )
    lines.append('')
    lines.append('| n     | k   | nnz    |   path | time       | HLO audit |')
    lines.append('|------:|----:|-------:|-------:|-----------:|:----------|')
    for row in pure_rows:
        lines.append(row.report_line())
    lines.append('')
    lines.append('### Reading the table')
    lines.append('')
    lines.append(
        '- ``max`` is the largest single tensor axis in the '
        'compiled HLO.  For the ELL backward we expect '
        '``max ~ k_max`` or ``max ~ n * k`` at most -- definitely '
        'not ``n^2``.  For the dense backward, ``max == n`` is '
        'fine (the gradient itself is ``(n, n)``).'
    )
    lines.append(
        '- ``n2`` is "y" iff a tensor with two axes equal to ``n`` '
        'appears.  For the ELL backward this MUST be "n": "y" '
        'means XLA materialised an ``O(n^2)`` intermediate '
        'somewhere in the projection.'
    )
    lines.append(
        '- For ELL pure-bwd, doubling ``n`` at fixed per-row '
        'degree should roughly double the wall-time.  Quadratic '
        'scaling would mean XLA found a sneaky way around the '
        'sparsity projection.'
    )
    lines.append('')
    output.write_text('\n'.join(lines))
    print(f'wrote {output}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).parent / 'PERF_LOBPCG.md',
    )
    parser.add_argument(
        '--skip-eigh-large',
        action='store_true',
        help='Skip eigh reference at n >= 4000 (OOM on small GPUs).',
    )
    args = parser.parse_args()

    rows: list[LobpcgBenchRow] = []
    pure_rows: list[PureVjpBenchRow] = []

    # Dense LOBPCG sweep
    for n in (256, 1024, 4096):
        rows.append(bench_dense_lobpcg(n, k=4, n_iters=200))

    # ELL LOBPCG sweep -- ring-with-chords, ~constant per-row degree
    for n in (256, 1024, 4096, 16384):
        rows.append(bench_ell_lobpcg(n, k=4, n_iters=200, n_chords=n))

    # eigh reference at sizes where it fits
    for n in (256, 1024, 4096):
        if args.skip_eigh_large and n >= 4096:
            continue
        rows.append(bench_eigh(n, k=4))

    # Pure-VJP scaling: isolates the backward kernel from LOBPCG.
    # Dense bwd is dominated by the (n,n) gradient materialisation;
    # ELL bwd should grow linearly with n at fixed per-row degree.
    for n in (256, 1024, 4096, 16384):
        pure_rows.append(bench_pure_vjp_dense(n, k=4))
    for n in (256, 1024, 4096, 16384, 65536):
        pure_rows.append(bench_pure_vjp_ell(n, k=4, n_chords=n))

    _write_report(rows, pure_rows, args.output)


if __name__ == '__main__':
    main()
