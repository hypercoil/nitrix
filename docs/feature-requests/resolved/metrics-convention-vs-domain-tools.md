# Convention check: `nitrix.metrics` similarity metrics vs domain-standard ITK/ANTs

> **Status (2026-06-09): RESOLVED — divergence is intentional convention, no
> nitrix lapse.** Investigated end-to-end (all five metrics, fp64 reference
> reimplementations of ITK's *published* v4 formulas; SimpleITK is not a nitrix
> dependency so the formulas were reimplemented, not invoked). Outcome:
> document-only — the divergences are convention/packaging differences,
> caller-recoverable; the exact conventions + ITK/ANTs parity are now pinned in
> each metric's docstring. Provenance + index in
> [`perf-bench-feedback.md`](../perf-bench-feedback.md).
>
> **One correction to the original finding:** the first pass characterised ITK
> `Correlation` as "the normalized cross-correlation *without* mean
> subtraction". That is wrong — it *is* mean-subtracted Pearson, returned as
> `-r**2` (see below). The numbers prove it: nitrix `r = 0.9567`, ITK `=
> -0.9152`, and `-0.9567**2 = -0.91527`.

## What was measured / verified

A fp64 reference for each metric (an exact-convention numpy reimplementation,
including ITK's published v4 formulas) on a controlled image pair. All five
relationships are now pinned (machine-precision agreement quoted):

| nitrix metric | nitrix computes | domain-standard form | exact relationship | verdict |
|---|---|---|---|---|
| `ssd` (`intensity.py`) | **mean** squared difference | ITK `MeanSquares` | **identical** (3e-15, fp64) | clean co-oracle |
| `ncc` (`intensity.py`) | **signed Pearson** `r` ∈ [−1,1] | ITK `Correlation` (`CorrelationImageToImageMetricv4`) | **ITK = −r²** (8e-12) — mean-subtracted, squared, negated | convention (by design) |
| `lncc` (`intensity.py`) | **squared local CC**, window-local means, similarity | ANTs `ANTSNeighborhoodCorrelation` | **interior-identical** (1e-10); nitrix = +mean, ANTs = −mean | convention (claim holds) |
| `mutual_information` (`information.py`) | **soft (order-1 B-spline Parzen)** MI, nats, differentiable | ITK `MattesMutualInformation` (order-3); sklearn (order-0 hard) | same Parzen family, different kernel order → different number at fixed `bins`; all → true MI as `bins → ∞` | divergent reference (by design) |
| `correlation_ratio` (`information.py`) | Roche η² (FSL `mcflirt` lineage) | — (SimpleITK ships no CR metric) | no domain oracle | nitrix-own |

ITK's published `CorrelationImageToImageMetricv4` definition is
`C = -<f-f̄, m-m̄>² / (<f-f̄,f-f̄> <m-m̄,m-m̄>) = -r²`. So nitrix `ncc` and ITK
`Correlation` compute the **same** mean-subtracted Pearson correlation; the gap
is packaging only: nitrix returns the *signed* similarity `r` (gradient-friendly,
sign-preserving), ITK returns the *negated-squared* minimisation cost (which is
sign-insensitive — an anti-correlated pair scores like a correlated one).

ANTs neighbourhood-CC is `-(1/N) Σ <f_i-f̄_i, m_i-m̄_i>² / (...)` over local
windows; nitrix `lncc` is the per-voxel `<...>²/(var_m var_f)` reduced by a
(domain-mask) mean — the identical squared local CC, returned as the
similarity. The only difference is boundary handling (nitrix uses `mode`,
default `reflect`, keeping the window count uniform; ANTs trims to valid
neighbourhoods).

## Soft- vs hard-binning convergence (the standing question)

Does the soft (relaxed) histogram converge to the hard one as some parameter
tends to an extremum? **No.** `_common._soft_bin` is a **linear order-1
B-spline / triangular Parzen** kernel with support fixed at exactly two bins —
there is no temperature / bandwidth parameter to take to a limit. Verified on a
bivariate Gaussian (ρ = 0.8, true MI = 0.51083 nats):

- At a **fixed** `bins`, soft ≠ hard (e.g. 8 bins: soft 0.220 < hard 0.347 <
  true 0.511 — the Parzen smoothing biases MI *downward* at coarse bins). No
  parameter closes this gap.
- Soft and hard coincide **exactly** only when every sample lands on a bin
  centre (`frac = 0`; verified to 2.6e-8) — a measure-zero, data-dependent
  case, not a parameter limit.
- As `bins → ∞` **both** converge to the true continuous MI (128 bins: soft
  0.509, hard 0.511) — the *continuum* limit, not "soft becoming hard".

A genuine soft→hard knob would need a different kernel family (Gaussian /
softmax-Parzen with bandwidth σ → 0). nitrix deliberately uses the linear
B-spline for cheap differentiability with fixed support; ITK Mattes (order-3)
is the same family, sklearn (order-0 hard) is the box-kernel endpoint.

## Resolution (the asks, answered)

1. **Are the divergences intentional?** Yes — confirmed for all five. `ncc`
   signed-`r`, `lncc` similarity-sign, and soft-Parzen `mutual_information` /
   `correlation_ratio` are deliberate (sign-preservation + gradient flow for
   registration-as-a-layer). No nitrix math lapse was found.
2. **Document the exact convention per metric.** Done — sign, mean-subtraction,
   squaring, reduction, and the ITK/ANTs relationship are now in each
   docstring (`ssd` = `MeanSquares`; `ncc` → ITK `-r²`; `lncc` =
   `ANTSNeighborhoodCorrelation` interior-exact; MI = order-1 Parzen, same
   family as Mattes; CR = FSL/Roche, no ITK oracle).
3. **Convention-matching variants?** Not added (document-only decision,
   2026-06-09). The `ncc`/`lncc` divergences are pure packaging, recoverable by
   the caller (`-ncc**2`, `1 - lncc`); MI/CR are differentiable-by-design and
   the hard-binned forms are non-differentiable. A hard-binned MI variant for
   ITK cross-validation remains a possible future opt-in if interop demand
   appears, but is a non-goal now.
4. **`lncc` == ANTs neighbourhood-CC?** Confirmed (interior-identical, fp64).

## perf-bench plan

Each metric is benchmarked vs a **numpy exact-convention reimplementation** (the
fp64 oracle + CPU floor) + a cupy GPU bar, and the domain tool included **only
where its definition matches** (clean co-oracle: `ssd`) or, per the perf-bench
owner's call (2026-06-09), as a **clearly-labeled *divergent* reference**
elsewhere (fidelity reported via the exact relationship — `ncc` ↔ `-r²`, `lncc`
↔ `-mean(localCC)` — not gated; the speed + the convention gap are both the
signal). The soft-Parzen MI/CR differentiability is surfaced as a nitrix
capability the hard-binned domain tools lack.

## Cross-references

- [`perf-bench-feedback.md`](../perf-bench-feedback.md) — the doc-drift / convention ledger.
- `src/nitrix/metrics/intensity.py` (`ssd`, `ncc`, `lncc`), `src/nitrix/metrics/information.py` (`mutual_information`, `correlation_ratio`, `joint_histogram`), `src/nitrix/metrics/_common.py` (`_soft_bin`).
- nitrix-perf-bench `DOMAIN_TOOL_BASELINES.md` §7 (the "match the right target" verification discipline; the SimpleITK `(x,y,z)`↔`(z,y,x)` axis trap).
