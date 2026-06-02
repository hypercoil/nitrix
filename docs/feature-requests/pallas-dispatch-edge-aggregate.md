# B3. Pallas dispatch for `semiring_ell_edge_aggregate`

> **Status (2026-06-02): parked (engineering backlog) — measured signal
> says a Pallas dispatch is NOT motivated by GPU perf today.** Not a
> commitment — gated on the **Trigger** below. Provenance: migrated from the
> retired top-level `BACKLOG.md` (B-numbering preserved); ledger context in
> [`internal-backlog.md`](internal-backlog.md).

The implementation is JAX-only (gather + nested vmap + semiring
axis-reduce). At Topofit scale (`ico_7`, `d_in=64-128`) the inner edge-MLP
is a tiled matmul that *should* map to the existing `semiring_matmul` Pallas
backend.

**Trigger.** [B2](perf-bench-sprint-surfaces.md) baseline shows the JAX path
is the bottleneck in a real training loop, or a consumer explicitly asks.

**Notes.** Hard part: `edge_fn` is a user Python callable, so no precompiled
kernel — needs either (a) a Pallas template parameterised by a fixed-shape
MLP descriptor (works for DGCNN-shaped `edge_fn`s, not generic) or (b) the
KeOps-style symbolic-formula approach ([B5](keops-genred-research.md)).

**Measured signal (2026-05-23, nitrix-perf-bench).** Partially met for a
GCN-style *linear* `edge_fn` (`w·(W@h_j)`): on **CPU**, torch_geometric
`MessagePassing` is ~2–5× faster than the nitrix-jax reference (0.21× at
n=4096 → 0.53× at n=16384, degree 16, d=64), both REAL and
TROPICAL_MAX_PLUS — the cost of the gather + nested-vmap + axis-reduce
generality vs PyG's native `scatter_reduce`; narrows with graph size. **But
on the A10G (deployment target) it is a wash** — nitrix-jax within ~5% of
PyG for REAL and ~15% *faster* for TROPICAL_MAX_PLUS; XLA fuses the
gather+vmap+reduce well on Ampere. Net: a Pallas dispatch is **not**
motivated by GPU perf vs PyG; the only standing case is CPU (likely not the
target) or an `edge_fn` shape XLA fuses poorly. Keep low-priority until a
GPU shortfall or concrete consumer appears. See
`nitrix-perf-bench/reports/PERF_ELL_EDGE_AGGREGATE.md`.

## Cross-references

- [`internal-backlog.md`](internal-backlog.md) — the engineering-backlog
  ledger.
- [`perf-bench-sprint-surfaces.md`](perf-bench-sprint-surfaces.md) (B2),
  [`keops-genred-research.md`](keops-genred-research.md) (B5).
- `docs/design/streaming-kernel.md`, `docs/design/ell-on-triton.md`.
