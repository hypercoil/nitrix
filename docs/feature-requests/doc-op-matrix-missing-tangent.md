# Doc-fix: `tangent_project_spd` is missing from the op_matrix inventory

> **Status (2026-06-02): open — inventory-correctness fix (not a primitive
> proposal; the op exists and ships).** Provenance: surfaced shipping a
> `nitrix-perf-bench` case for it; ledger context in
> [`perf-bench-feedback.md`](perf-bench-feedback.md), and it directly touches
> the op_matrix migration tracked in [`perfbench-migration.md`](perfbench-migration.md)
> (B11).

`nitrix.linalg.tangent_project_spd` is a **public** op — exported in
`src/nitrix/linalg/__init__.py` (`__all__`, lines 59 + 93) — but it has **no
entry** in `docs/op_matrix.json` (`ops` list = 59 entries; `grep -c
tangent_project_spd docs/op_matrix.json` → 0). Its own building blocks
(`symlog`, `sympower`, `symsqrt`) are all cataloged; it was simply missed by
the public-surface inventory (`7ba3776 "matrix inventorying public surface
ops"`).

**Impact (perf-bench coverage cannot credit a shipped benchmark).**
nitrix-perf-bench shipped `tangent_project_spd` (commit `cfac230`: the
affine-invariant tangent embedding `log(R^-1/2 X R^-1/2)` vs nilearn's
`ConnectivityMeasure(kind='tangent')` kernel — the suite's *first nilearn
reference*). But `tools/coverage_report.py` joins `op_matrix.json` × the
perf-bench store, so an op absent from the inventory is invisible to coverage:
the benchmark exists (nitrix GPU 1.76/4.68/9.0 ms; 15–32× over the nilearn CPU
floor) yet coverage stays **27/52** instead of crediting it.

**Fix (mechanical: one `ops` entry, mirroring `symlog`).** Capability fields
verified on this checkout (CPU, `d=8` batch): `jit` / `vmap` (over the SPD
batch, reference fixed) / `grad` (finite) / `jit_of_grad` all **pass** — as
expected, since it composes `sympower` + `symlog` (both cataloged `pass`).
Suggested entry:

```json
{
  "qualname": "nitrix.linalg.tangent_project_spd",
  "jit": "pass", "grad": "pass", "vmap": "pass", "jit_of_grad": "pass",
  "invariants": ["affine-invariant log map at reference (not log-Euclidean)"],
  "notes": "log(R^-1/2 X R^-1/2); composes sympower(-1/2)+symlog (consumed eigh, GPU-robust through the cuSolver wedge)",
  "perf_cpu_baseline": "nilearn.connectome ConnectivityMeasure(kind='tangent')",
  "perf_cpu_ratio": null, "perf_gpu_baseline": null, "perf_gpu_ratio": null
}
```

(Perf is migrating to perf-bench, so the `perf_*_ratio` cells stay `null` /
carry `perf_source` per `MIGRATED_TO_PERFBENCH` — see B11.)

## Cross-references

- [`perf-bench-feedback.md`](perf-bench-feedback.md) — the perf-bench-surfaced
  correctness ledger.
- [`perfbench-migration.md`](perfbench-migration.md) — B11 op_matrix migration
  (this is an inventory-completeness item under that umbrella).
- `src/nitrix/linalg/__init__.py:59,93`; `docs/op_matrix.json` (`ops`).
- nitrix-perf-bench `cfac230` (case) + `src/nperf/cases/tangent_project_spd.py`.
