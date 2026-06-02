# B11. Finish migrating perf to nitrix-perf-bench (op_matrix → capability-only)

> **Status (2026-06-02): in progress (transitional, non-regressive) —
> bench-report migration COMPLETE; the final op_matrix column-strip
> remains.** Effort **L**, cleanly sliceable, no nitrix API change.
> Provenance: migrated from the retired top-level `BACKLOG.md` (B-numbering
> preserved); ledger context in [`internal-backlog.md`](internal-backlog.md).

Performance is moving out of this repo into sibling **nitrix-perf-bench**
(cross-framework refs, multi-platform fan-out, durable result store,
fp64-oracle fidelity gating, regression gate, hosted dashboard). End state:
the op_matrix is **capability-only** (jit / grad / vmap / jit-of-grad +
invariants), perf lives entirely in nitrix-perf-bench.

**Status (transitional, non-regressive).** Migrated: `semiring_matmul`,
`semiring_ell_edge_aggregate` (cells render `↗ perf-bench`; see
`MIGRATED_TO_PERFBENCH` in `tools/op_matrix.py`). The other ~30 ops still
carry in-tree `bench/` numbers — nothing lost in the interim.

**Bench-report migration COMPLETE.** Every `bench/PERF_*` / `G0_*` / `MEM_*`
report is either ported to a nitrix-perf-bench case or intentionally kept
nitrix-side: `PERF_AUDIT` (all 13 ops ported), `PERF_SEMIRING_CONV`
(ported), `G0_ELL_REPORT` (ported), `PERF_TRILINEAR` (folded into the
`spatial_transform` case). **Kept nitrix-side:** `PERF_LOBPCG` (private
kernel; implicit-VJP perf + HLO no-n² scaling audit, no
public-op/forward-ratio fit) and `MEM_STREAMING_KERNEL` (memory-scaling /
leak study; core finding already captured by the `semiring_matmul` case's
`peak_hbm`) — both move to nitrix-side tests.

**Final step (this repo).** Drop the op_matrix `perf_*` columns +
`load_perf_data` + the `bench/PERF_*` scrapers, and retire `bench/` (perf
now lives in nitrix-perf-bench; the LOBPCG / MEM_STREAMING studies move to
tests).

**Trigger.** Incremental/opportunistic: port an op when touched or when its
perf becomes decision-relevant; do the final column-strip once
`MIGRATED_TO_PERFBENCH` covers the catalogue. Or a dedicated migration
sprint.

**Effort.** L overall, cleanly sliceable (one op = one small case, each S
and independently shippable). No nitrix API change — docs/tooling + the
eventual `bench/` retirement.

**Cross-refs.** `../nitrix-perf-bench/DESIGN.md` §6 + `src/nperf/cases/`;
`tools/op_matrix.py` `MIGRATED_TO_PERFBENCH`; `bench/PERF_AUDIT.md`.

**Inventory gap (tracked).** The public-surface inventory is not complete:
three shipping public ops have **no `op_matrix.json` entry** —
`nitrix.linalg.tangent_project_spd`, `nitrix.bias.histogram_match`,
`nitrix.bias.n4_bias_field_correction` — so perf-bench benchmarks for them
cannot be credited by `coverage_report.py` (the join is keyed on the
inventory). Mechanical add of one entry each (capabilities verified) — see
[`doc-op-matrix-inventory-gaps`](doc-op-matrix-inventory-gaps.md). Re-running
the public-surface inventory would catch any other omissions.

## Cross-references

- [`internal-backlog.md`](internal-backlog.md) — the engineering-backlog
  ledger.
- [`perf-bench-feedback.md`](perf-bench-feedback.md) — doc-drift findings
  surfaced during the same perf-bench build-out.
