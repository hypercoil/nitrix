# Doc-fix: `gaussian_kernel` sigma->gamma relation is wrong (½ factor)

> **Status (2026-06-03): open — documentation-correctness fix (one line, no
> behaviour change).** Provenance: surfaced matching `gaussian_kernel` to
> `sklearn.metrics.pairwise.rbf_kernel` for a `nitrix-perf-bench` case; ledger
> context in [`perf-bench-feedback.md`](perf-bench-feedback.md).

`src/nitrix/linalg/kernel.py:37` (module docstring) states that
`gaussian_kernel` and `rbf_kernel` *"are aliases up to the ``gamma = sigma^-2``
substitution."* The actual relation carries the standard **½** factor:
`gaussian_kernel(sigma=σ) == rbf_kernel(gamma = 1/(2·σ²))`, i.e. the kernel is
`exp(-‖x-y‖² / (2σ²))`, not `exp(-‖x-y‖²·σ⁻²)`.

**Verified (fp64, this checkout):** `gaussian_kernel(X, sigma=3)` matches
`rbf_kernel(X, gamma=1/18)` to **0.0**, and `rbf_kernel(X, gamma=1/9 = σ⁻²)` to
**0.25** (clearly off). Same at σ=2 (matches gamma=1/8, not 1/4).

**Fix (one line).** `gamma = sigma^-2` → `gamma = 1/(2 * sigma**2)` (or state
the kernel as `exp(-dist² / (2σ²))`). The code is correct; only the docstring's
substitution formula is wrong. (The benchmark uses the correct
`gamma = 1/(2σ²)` mapping; this is purely a docstring fix.)

## Cross-references

- [`perf-bench-feedback.md`](perf-bench-feedback.md) — the perf-bench-surfaced
  doc-drift ledger.
- `src/nitrix/linalg/kernel.py:37`; nitrix-perf-bench `gaussian_kernel` case.
