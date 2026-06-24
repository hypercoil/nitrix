# Compensated summation / mixed-precision — `nitrix.numerics.precision`

> **Status (2026-06-02): not started.** Brainstorm candidate; promotion
> gated by the §13 acceptance protocol. Provenance:
> `docs/feature-requests catalogue §12.10`.

**What.** Pure-numerics reduction utilities any substrate reduction can drop
in for accuracy / reproducibility / low-precision accumulation.

**Proposed surface.**

```python
def kahan_sum(x, axis): ...                       # compensated summation
def neumaier_sum(x, axis): ...                     # Kahan's improved variant
def stochastic_round(x, dtype): ...                # for FP8 / FP16 accumulation
def pairwise_sum(x, axis, blocksize): ...          # log-depth tree reduction
```

**Composition.** Standalone numerics utilities — no substrate dependency;
they are *dropped into* existing reductions (covariance accumulation,
semiring reductions) rather than composing them.

**Likely consumer.** Long-time-series fMRI covariance accumulation at FP32
(cross-volume drift), Blackwell FP8 paths when they arrive, reproducible
reductions for golden-corpus tests.

**Effort.** S.

**Live-code status.** No `kahan_sum` / `neumaier_sum` / `stochastic_round` /
`pairwise_sum`. `numerics/__init__` currently exposes normalisation helpers
(`demean`, `zscore_normalize`, …) and re-exported axis utilities only.

## Cross-references

- `docs/feature-requests catalogue §12.10` — origin entry; `§13` — acceptance protocol.
- `src/nitrix/stats/covariance.py` — a prime drop-in site (FP32 covariance
  drift over long series).
- [`docs/design/testing-strategy.md`](../design/testing-strategy.md) —
  golden-corpus reproducibility motivation.
