# Doc-fix: public ops missing from the op_matrix inventory

> **Status (2026-06-02): RESOLVED — and the full inventory re-run the request
> recommended is done.** The 3 named ops plus **75 further uncataloged public
> functions** (a full `__all__`-vs-catalogue diff found the matrix covered only
> 59 of ~163 ops) were added to `tools/op_matrix.py`; the matrix regenerated to
> **137 ops** (jit 122/137, grad 119/121 — the two N4 cells fail `grad` as
> flagged below). A **completeness guard** (`tests/test_op_matrix_completeness.py`)
> now fails CI if any public op is neither cataloged nor on an explicit EXCLUDE
> allowlist (aliases / reference impls / matvec closures / shape helpers / metric
> constructors), so the inventory cannot silently rot again. Also fixed a
> pre-existing stale `bilateral_gaussian` fixture (`sigma_features` → v0.4
> `metric`) and a missing `signal.tsconv` export from `signal.__all__`. See
> `IMPLEMENTATION_PLAN.md §10.3` (2026-06-02 op-matrix entry). Provenance:
> surfaced shipping `nitrix-perf-bench` cases; ledger context in
> [`perf-bench-feedback.md`](../perf-bench-feedback.md); part of the op_matrix
> migration tracked in [`perfbench-migration.md`](../perfbench-migration.md) (B11).

The public-surface inventory in `docs/op_matrix.json` (`ops` list, 59 entries)
is **incomplete**: three shipping public ops have no entry, so perf-bench
benchmarks built for them cannot be credited by `tools/coverage_report.py`
(the coverage join is keyed on the inventory). All three were verified
`grep -c <qualname> docs/op_matrix.json` → 0. **Recommend re-running the
public-surface inventory** (`7ba3776 "matrix inventorying public surface ops"`)
to catch any further omissions; the three confirmed below are the ones
perf-bench has hit.

Capability fields below are **verified on this checkout** (small inputs, CPU).
Mechanical fix: add one `ops` entry each. Perf is migrating to perf-bench, so
`perf_*_ratio` stay `null` / carry `perf_source` per `MIGRATED_TO_PERFBENCH`.

### 1. `nitrix.linalg.tangent_project_spd`

Public (`src/nitrix/linalg/__init__.py:59,93`); its building blocks
`symlog`/`sympower`/`symsqrt` are all cataloged. perf-bench case `cfac230`
(affine-invariant tangent embedding vs nilearn `ConnectivityMeasure(kind=
'tangent')`). Verified jit / vmap / grad (finite) / jit_of_grad all **pass**.

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

### 2. `nitrix.bias.histogram_match`

Public (`nitrix.bias.__all__`). perf-bench case `2bfe3c3` (Nyul-Udupa vs ITK
`HistogramMatchingImageFilter`; live-SimpleITK parity asserted). Verified jit /
vmap / grad / jit_of_grad all **pass**.

```json
{
  "qualname": "nitrix.bias.histogram_match",
  "jit": "pass", "grad": "pass", "vmap": "pass", "jit_of_grad": "pass",
  "invariants": ["Nyul-Udupa landmark map; ITK HistogramMatching parity"],
  "notes": "n_match_points=7, n_histogram_levels=1024, threshold-at-mean; matches sitk to <1e-3 of ref range",
  "perf_cpu_baseline": "SimpleITK HistogramMatchingImageFilter",
  "perf_cpu_ratio": null, "perf_gpu_baseline": null, "perf_gpu_ratio": null
}
```

### 3. `nitrix.bias.n4_bias_field_correction`

Public (`nitrix.bias.__all__`). perf-bench case `2bfe3c3` (vs ITK
`N4BiasFieldCorrectionImageFilter`; live-SimpleITK parity asserted, corr>0.999).
Verified jit / vmap **pass**; **reverse-mode `grad` FAILS** — `jax.grad` raises
*"Reverse-mode differentiation does not work for lax.while_loop"* (the iterative
N4 fit uses a `while_loop`). So `grad`/`jit_of_grad` should be recorded `fail`
(or `n/a`) unless a `custom_vjp` / forward-mode path is intended — flagging for
nitrix to set the intended status.

```json
{
  "qualname": "nitrix.bias.n4_bias_field_correction",
  "jit": "pass", "grad": "fail", "vmap": "pass", "jit_of_grad": "fail",
  "invariants": ["ITK/ANTs N4 algorithm + defaults; field defined up to global scale"],
  "notes": "iterative B-spline fit via lax.while_loop (reverse-grad unsupported); GPU-robust (lowers off cuSolver); N4BiasFieldCorrectionImageFilter parity",
  "perf_cpu_baseline": "SimpleITK N4BiasFieldCorrectionImageFilter",
  "perf_cpu_ratio": null, "perf_gpu_baseline": null, "perf_gpu_ratio": null
}
```

## Cross-references

- [`perf-bench-feedback.md`](../perf-bench-feedback.md) — the perf-bench-surfaced
  correctness ledger.
- [`perfbench-migration.md`](../perfbench-migration.md) — B11 op_matrix migration
  (this is an inventory-completeness item under that umbrella).
- `src/nitrix/linalg/__init__.py`; `src/nitrix/bias/__init__.py`;
  `docs/op_matrix.json` (`ops`).
- nitrix-perf-bench `cfac230` (tangent), `2bfe3c3` (N4 + histogram_match).
