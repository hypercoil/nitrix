# B5. KeOps `Genred`-primitive research

> **Status (2026-06-02): parked (research note).** Not a commitment — gated
> on the **Trigger** below; the deliverable is a design doc, not a
> primitive. Provenance: migrated from the retired top-level `BACKLOG.md`
> (B-numbering preserved); ledger context in
> [`internal-backlog.md`](internal-backlog.md).

KeOps's `Genred` takes a user formula in a string DSL and JIT-compiles a
CUDA kernel that streams the reduction without materialising the per-edge
tensor. The SPEC §4.1 "KeOps-style streaming kernel" claim invokes this
lineage; what's missing is a concrete research note on whether a
Pallas-backed analogue could host arbitrary user formulas (the
`edge_aggregate` generic case).

**Trigger.** [B3](pallas-dispatch-edge-aggregate.md) forces a decision
between "Pallas template per `edge_fn` shape" and "compile a formula on the
fly". Or a follow-up SPEC update revisits formula compilation independently.

**Notes.** Research deliverable — a design doc covering: (1) KeOps's formula
DSL (LazyTensor + reduction primitives); (2) the JAX/Pallas analogue
(jaxpr-level analysis of a callable? Pallas templating via concrete-shape
inputs?); (3) the boundary between compilable (closed polynomial /
elementwise) and non-compilable (data-dependent control flow, dynamic
shapes) formulas; (4) whether this belongs in nitrix or a separate package.

**Cross-refs.** `docs/design/streaming-kernel.md`,
`docs/design/ell-on-triton.md`; KeOps paper Charlier et al. 2021 (JMLR).

## Cross-references

- [`internal-backlog.md`](internal-backlog.md) — the engineering-backlog
  ledger.
- [`pallas-dispatch-edge-aggregate.md`](pallas-dispatch-edge-aggregate.md)
  (B3) — the experiment that would force the formula-compilation decision.
