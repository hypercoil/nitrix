# Doc-fix: lomb-scargle module docstring says "Cholesky"; code uses `eigh` + pseudo-inverse

> **Status (2026-06-02): open — documentation-correctness fix (not a
> primitive proposal).** Mechanical (file:line). Provenance: surfaced
> building a `nitrix-perf-bench` case; ledger context in
> [`perf-bench-feedback.md`](perf-bench-feedback.md).

The module "Memory regime" docstring (`src/nitrix/signal/lomb_scargle.py:43–49`)
describes the shared-mask interpolation path as *"compute the Gram matrix
`G = Bᵀ diag(mask) B` and its **Cholesky L** once, then solve … as a single
batched **triangular solve**"* and budgets *"Shared Gram / Cholesky"*. But
the implementation `_lomb_scargle_solve_shared_mask` (lines 246–261, and its
own docstring at 229–233) factors `G` via **`safe_eigh`** (symmetric
eigendecomposition) and applies a **threshold-truncated pseudo-inverse**
(`rcond·max(eigval)`) — no Cholesky, no triangular solve. The module
docstring is stale relative to the rank-deficiency-robust eigh path the code
actually ships.

**Fix.** Update the module "Memory regime" prose to describe the eigh /
pseudo-inverse factorisation.

## Cross-references

- [`perf-bench-feedback.md`](perf-bench-feedback.md) — the doc-drift ledger.
- [`doc-lomb-scargle-cpu-eigh-caveat.md`](doc-lomb-scargle-cpu-eigh-caveat.md)
  — the related `safe_eigh` device-placement caveat on the same path.
