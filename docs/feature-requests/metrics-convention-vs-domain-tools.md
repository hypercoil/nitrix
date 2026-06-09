# Convention check: `nitrix.metrics` similarity metrics vs domain-standard ITK/ANTs

> **Status (2026-06-09): OPEN — needs a nitrix decision before perf-bench
> benchmarks the metric family or any registration recipe that selects a
> metric.** Surfaced building `nitrix-perf-bench` cases for the registration
> surface (R0 metrics). Provenance + index in
> [`perf-bench-feedback.md`](perf-bench-feedback.md). This is a
> *convention/definition* finding, not a bug: the metrics compute well-defined
> quantities, but several differ from the domain-standard ITK/ANTs forms a
> practitioner would compare against, which determines what a *warranted*
> benchmark reference is.

## What was measured

Building a fp64 reference for each metric (an exact-convention numpy
reimplementation) and a SimpleITK `ImageRegistrationMethod.MetricEvaluate` at
identity, on the same image pair (axis order handled: numpy `(z,y,x)` ↔
`sitk.GetImageFromArray`). Two verified so far:

| nitrix metric | nitrix computes | numpy reimpl | SimpleITK metric | match? |
|---|---|---|---|---|
| `metrics.ssd` (`intensity.py`) | **mean** squared difference | `mean((m-f)**2)` ✓ (1e-6, fp32) | `MeanSquares` = **0.0914894** | **matches nitrix 0.0914893** ✓ |
| `metrics.ncc` (`intensity.py`) | **global Pearson** (mean-subtracted) `r` | `corrcoef` ✓ (4e-8) | `Correlation` = **−0.9152** | **diverges** (nitrix `r` = 0.9567) |

`ssd` ↔ ITK `MeanSquares` is a clean co-oracle. `ncc` is **not**: ITK's
`Correlation` is the normalized cross-correlation *without* mean subtraction
(and sign-flipped for minimization), a different definition from Pearson — the
same setup that made `MeanSquares` match exactly, so this is definitional, not
a harness artifact.

## Expected (not yet verified — deferred pending this decision)

- `metrics.mutual_information` (`information.py`) is a **soft (Parzen) joint
  histogram** → differentiable. ITK `MattesMutualInformation` (B-spline Parzen)
  and `sklearn.metrics.mutual_info_score` (hard binning) compute different
  numbers, and the sklearn/ITK forms are **not** differentiable. nitrix's
  differentiability here is a genuine capability worth surfacing — but it means
  the domain tools are divergent references, not oracles.
- `metrics.correlation_ratio` (Roche η²) and `metrics.lncc` (docstring claims
  the **ANTs** local-CC form) likewise need checking against ITK
  `ANTSNeighborhoodCorrelation` / the ANTs definition before they can be called
  matching references. The `lncc` ↔ ANTs claim in particular should be pinned.

## The ask (why perf-bench paused here)

1. **Are the divergences intentional?** Pearson `ncc` and soft-histogram,
   *differentiable* `mutual_information` look like deliberate choices (gradient
   flow for registration-as-a-layer). Confirm, so the benchmark frames them as
   "nitrix's convention" rather than "wrong vs ITK".
2. **Document the exact convention per metric** (mean-subtracted? sign? binning
   kernel? window normalization?), so a cross-tool comparison is warranted and
   the perf-bench reference choice is principled.
3. **Convention-matching variants?** Does nitrix want optional ITK-compatible
   forms (e.g. a no-mean-subtraction NCC, a hard-binned MI) for interop /
   cross-validation against the domain tools — or is divergence-by-design the
   position (and the domain tools are simply labeled divergent)?
4. **Confirm `lncc` == the ANTs neighborhood-CC form** the docstring claims.

## perf-bench plan (pending the above)

Each metric benchmarked vs a **numpy exact-convention reimplementation** (the
fp64 oracle + CPU floor) + a cupy GPU bar, and the domain tool (SimpleITK/ANTs)
included **only where its definition matches** (clean co-oracle: `ssd`) or, per
the perf-bench owner's call (2026-06-09), as a **clearly-labeled *divergent*
reference** elsewhere (fidelity reported, not gated — it computes a different
quantity; the speed + the convention gap are both the signal). The
soft-histogram MI/CR differentiability is surfaced as a nitrix capability the
hard-binned domain tools lack.

## Cross-references

- [`perf-bench-feedback.md`](perf-bench-feedback.md) — the doc-drift / convention ledger.
- `src/nitrix/metrics/intensity.py` (`ssd`, `ncc`, `lncc`), `src/nitrix/metrics/information.py` (`mutual_information`, `correlation_ratio`, `joint_histogram`).
- nitrix-perf-bench `DOMAIN_TOOL_BASELINES.md` §7 (the "match the right target" verification discipline; the SimpleITK `(x,y,z)`↔`(z,y,x)` axis trap).
