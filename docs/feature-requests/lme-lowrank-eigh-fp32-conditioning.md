# FR: low-rank REML — the `ZZ^T` eigendecomposition is fp32-conditioning-limited at large N

**Status:** open · **Severity:** medium (accuracy at scale; not a correctness
bug — exact in fp64) · **Source:** nitrix-perf-bench `reml_fit_lowrank` case.

## Summary

`reml_fit(low_rank=True)` (the FaST-LMM q-rank AI-REML in
`stats/lme/_lowrank.py`) is **numerically exact in float64** but
**fp32-precision-limited at large N**: the error vs the exact closed-form
balanced-one-way REML grows with the observation count `N` and, by `N ≈ 3·10⁴`,
exceeds even a loose `rtol=atol=1e-2` gate. The full-rank `reml_fit` path is far
less sensitive. The culprit is the one `ZZ^T` eigendecomposition
(`safe_eigh` → CPU): as `N` grows the nonzero eigenvalues `s²` span a wider
range, and the AI-REML terms evaluated in that diagonal basis lose fp32 digits.

## Evidence (L4, balanced one-way `y ~ 1 + (1|g)`, q = rank(Z) = 8)

`rel_to_tol = max|est − closed_form| / (1e-2 + 1e-2·|closed_form|)`, max over
the (β, σ_b², σ_e²) components and the voxel batch:

| N      | full-rank (fp32) | low-rank (fp32) | low-rank (fp64) |
|--------|------------------|-----------------|-----------------|
| 192    | 0.023            | 0.074           | 0.000           |
| 1600   | 0.032            | 0.307           | 0.000           |
| 3200   | —                | 0.337           | —               |
| 8192   | —                | 0.372           | —               |
| 16384  | —                | 0.627           | —               |
| 32768  | —                | **1.936** (FAIL)| 0.000           |

Two controls pin the cause:

- **Not convergence.** The low-rank `rel_to_tol` is **flat across `n_iter` =
  20…200** (e.g. 0.331 at N=1600 for every `n_iter`) — AI-REML has converged;
  it converges to a slightly fp32-biased fixed point.
- **Not the algorithm.** In **float64** the low-rank path matches the closed
  form (and the full-rank path) to `rel_to_tol = 0.000` at N=1600 *and* N=4096.
  So the q-rank formulation is exact; the loss is purely fp32 precision in the
  eigendecomposition / diagonal-basis arithmetic.

The low-rank path is **markedly more fp32-sensitive than full-rank**
(0.33 vs 0.03 at N=1600) — i.e. forming and reducing in the `ZZ^T` eigenbasis
amplifies the conditioning relative to the full-rank solve.

## Suggested fixes (any one)

1. **Promote the conditioning-critical step to float64 internally** even when
   the inputs are fp32: compute the `ZZ^T` eigh (and the diagonal-basis score /
   average-information aggregates) in float64, then cast back. This is the
   targeted fix — the rest of the per-voxel arithmetic can stay fp32.
2. **Reduce the eigenvalue spread** before the diagonal-basis arithmetic (e.g.
   work with `s² / mean(s²)` and rescale θ accordingly), so the fp32 terms span
   a narrower dynamic range.
3. At minimum, **document the fp32 large-N limit** (as the Lomb-Scargle eigh
   caveats already do — see `doc-lomb-scargle-cpu-eigh-caveat.md`) and steer
   large-N callers to float64.

## Impact

The perf-bench `reml_fit_lowrank` case caps its fp32 large tier at **N=16384**
(rel_to_tol 0.627); beyond that the fp32 result is unreliable against the exact
oracle, so the N-scaling benchmark stops there. With fix (1) the cap could be
removed and the low-rank path benched to arbitrary N at full accuracy. Related:
`lme-family-tiny-linalg-gpu-block-and-perf.md` (the broader LME-linalg item).
