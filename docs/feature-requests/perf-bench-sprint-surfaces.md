# B2. Perf-bench the Sprint A / Sprint B surfaces

> **Status (2026-06-02): parked (engineering backlog).** Not a commitment —
> gated on the **Trigger** below. Provenance: migrated from the retired
> top-level `BACKLOG.md` (B-numbering preserved); ledger context in
> [`internal-backlog.md`](internal-backlog.md).

Add bench scripts (now under `nitrix-perf-bench`) for
`semiring_ell_edge_aggregate` (GCN fwd+bwd, DGCNN fwd+bwd),
`max_pool_with_indices_nd` / `max_unpool_nd`, and the mesh wrappers at
icosphere-realistic shapes (`ico_6 = 40962`, `ico_7 = 163842` verts). Wire
results into the op-matrix `perf_ratio` column.

**Trigger.** First concrete consumer running these in a training loop where
wall-time matters (Topofit cascade at `ico_7`, any SphereMorph descendant,
any PGlandsSeg-lineage encoder-decoder UNet). Or: a Pallas-dispatch
experiment ([B3](pallas-dispatch-edge-aggregate.md)) needing a baseline.

**Notes.** The op matrix shows all four transforms passing on these
surfaces; what's missing is wall-time vs a natural reference —
`semiring_ell_edge_aggregate` vs PyTorch-Geometric `MessagePassing` (host
parity, JAX GPU timing); `max_pool_with_indices_nd` vs
`torch.nn.MaxPool3d(return_indices=True)`.

## Cross-references

- [`internal-backlog.md`](internal-backlog.md) — the engineering-backlog
  ledger (framing, closed-by-design, resolved).
- [`pallas-dispatch-edge-aggregate.md`](pallas-dispatch-edge-aggregate.md)
  (B3) — the dispatch experiment this baselines.
