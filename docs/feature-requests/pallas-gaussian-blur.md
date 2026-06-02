# B6. Pallas kernel for the Gaussian blur primitive

> **Status (2026-06-02): parked (engineering backlog) — do not build
> speculatively.** Not a commitment — gated on the **Trigger** below.
> Provenance: migrated from the retired top-level `BACKLOG.md` (B-numbering
> preserved); ledger context in [`internal-backlog.md`](internal-backlog.md).

`smoothing.gaussian` is a separable n-D conv via `lax.conv_general_dilated`
(one 1-D pass/axis), lowering to cuDNN on Ampere+ — a strong baseline.

**Trigger.** A consumer with a wall-time wall on large-3-D Gaussian blur
(e.g. repeated 256³ smoothing in a training loop), *and* a benchmark showing
the separable-conv path is the bottleneck.

**Notes.** Gaussian is a stencil (not a gather), so Pallas-friendly. But
cuDNN is hard to beat per-pass; the only real win is **fusing the 3
separable axis passes** (+ boundary pad) into one kernel to save inter-pass
HBM round-trips on large volumes — marginal and bandwidth-bound; do not
build speculatively. Any kernel ships behind `backend=` with the
`conv_general_dilated` JAX floor (non-negotiable §2.2.3) and a golden-corpus
parity test; bench against `conv_general_dilated`, not a naive loop.

## Cross-references

- [`internal-backlog.md`](internal-backlog.md) — the engineering-backlog
  ledger; the house "benchmark-first, don't build Pallas speculatively"
  policy (cited from `src/nitrix/bias/n4.py`).
- `src/nitrix/smoothing/gaussian.py`.
