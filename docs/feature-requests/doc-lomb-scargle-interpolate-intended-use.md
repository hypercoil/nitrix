# Doc-fix: `lomb_scargle_interpolate` — document the *intended use* (spectral bridge, not durable imputation)

> **Status (2026-06-02): RESOLVED.** Added an "Intended use" Notes paragraph
> to `lomb_scargle_interpolate` (spectral bridge for AR/IIR filtering, not
> durable per-frame imputation; observed-frame splice-through and
> band-limited content are the only well-defined outputs; censored-frame
> values not bit-reproducible across precisions). See `IMPLEMENTATION_PLAN.md
> §10.3` (2026-06-02 entry). Provenance: surfaced building the perf-bench
> `lomb_scargle_interpolate` case; ledger context in
> [`perf-bench-feedback.md`](perf-bench-feedback.md). Measured from-scratch
> fp64 joint-GLM vs nitrix on this checkout.

The masked Gram `Bᵀ diag(mask) B` is **hugely ill-conditioned (cond ~1e32)**.
nitrix correctly regularises it with the rcond-truncated pseudo-inverse, but
**which** near-zero eigenvalues fall below the `rcond·max` threshold differs
between fp32 and fp64 (and between fp32/fp64 trial-freq grids), so the
**reconstruction at *censored* frames has no well-defined bit-level value** —
two valid regularised solutions disagree by **up to ~1.4 on O(1) signals**
(worst frame, `obs=256`, ~15% censoring). The **observed-frame splice-through
is exact** (`recon[obs] == data[obs]`, 0.0).

**This is not a bug — but it has a use-case implication worth documenting.**
The Power-2014 procedure's purpose is *not* to "repair" high-motion frames
for durable inclusion in analysis; it is to produce a **spectrally-consistent
bridge** so downstream **autoregressive / IIR temporal filters** (band-pass,
high-pass) can run across the gaps without ringing or spectral leakage. The
filled values are a transient means to that end; the censored frames are
typically dropped again after filtering. The fp32/regularisation sensitivity
lives in the **near-null-space** (high-frequency basis directions the
irregular observed grid cannot pin down — exactly what the rcond truncation
zeros), which a downstream low/band-pass filter attenuates anyway. So the
sensitivity is **benign for the intended use** but a real correctness trap if
a user treats the interpolated values as **durable per-frame imputations**.

**Fix (docstring, `lomb_scargle_interpolate` ~264–359).** State the intended
use explicitly: *"The filled values at censored frames are a
spectrally-consistent bridge for downstream temporal (AR/IIR) filtering — not
durable imputations for direct per-frame analysis. The reconstruction at
censored frames is regularised (rcond-truncated pseudo-inverse of an
ill-conditioned masked Gram) and is therefore not bit-reproducible across
precisions; only the observed-sample splice-through and the band-limited
spectral content are well-defined. If you need the filled values durably,
compute in fp64 and treat the high-frequency content as undetermined."*

**Perf-bench handling (FYI).** The case scores fidelity on the well-defined
property — the splice-through (asserted in tests) — and sets
`fp64_reference=None` (no cross-impl oracle) for the censored-frame values
rather than gating on an ill-posed target.

## Cross-references

- [`perf-bench-feedback.md`](perf-bench-feedback.md) — the doc-drift ledger.
- [`doc-lomb-scargle-normalisation.md`](doc-lomb-scargle-normalisation.md) —
  the sibling high-value normalisation docstring fix.
