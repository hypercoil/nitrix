# Doc-fix: `lomb_scargle_periodogram` normalisation docstring is wrong

> **Status (2026-06-02): RESOLVED.** Docstring rewritten to describe the
> math (`P_raw/var` ≡ `scipy.signal.lombscargle(normalize=False)/var`) with
> an explicit note on the `N/2` offset from scipy `normalize=True`; a
> scipy-parity regression test (`test_signal_interpolate.py`) pins both
> relations. See `IMPLEMENTATION_PLAN.md §10.3` (2026-06-02 entry).
> Provenance: surfaced building a `nitrix-perf-bench` case; ledger context in
> [`perf-bench-feedback.md`](perf-bench-feedback.md). Verified against
> `scipy 1.17.1` + a from-scratch fp64 Scargle-1982 reference.

`src/nitrix/signal/lomb_scargle.py:154` claims *"Normalisation matches
`scipy.signal.lombscargle(normalize=True)`."* Measured, it does **not**:

- nitrix returns the **classic Scargle-normalised** periodogram `P_raw / var`
  (`var` = masked-sample variance) — matches my from-scratch Scargle-1982
  `raw/var` and `scipy.signal.lombscargle(..., normalize=False)/var` to
  **~1e-9 in fp64 at every length 256–8192** (algorithmically exact).
- `scipy.signal.lombscargle(..., normalize=True)` on 1.17.1 returns
  `2·P_raw / (N·var)` — i.e. it differs from nitrix's output by a constant
  factor of **N/2** (measured exactly: 95.0 for N=190).

So anyone trusting the docstring and comparing nitrix to scipy
`normalize=True` sees a flat N/2 discrepancy and concludes nitrix is "wrong"
when it is correct.

**Fix.** Change the docstring to *"matches
`scipy.signal.lombscargle(normalize=False)` divided by the observed-sample
variance (the classic Scargle 1982 normalisation)."* (scipy's `normalize`
convention has drifted across versions, so naming a specific scipy flag is
fragile — describe the math.)

## Cross-references

- [`perf-bench-feedback.md`](perf-bench-feedback.md) — the doc-drift ledger.
- Sibling lomb-scargle doc fixes:
  [`doc-lomb-scargle-eigh-factorisation.md`](doc-lomb-scargle-eigh-factorisation.md),
  [`doc-lomb-scargle-cpu-eigh-caveat.md`](doc-lomb-scargle-cpu-eigh-caveat.md),
  [`doc-lomb-scargle-interpolate-intended-use.md`](doc-lomb-scargle-interpolate-intended-use.md).
