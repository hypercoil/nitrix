# Graph wavelet transforms (SGWT) — `nitrix.graph.wavelet`

> **Status (2026-06-02): not started — blocked on `matrix_polynomial`
> (§12.2).** Brainstorm candidate; promotion gated by the §13 acceptance
> protocol. Provenance: `SPEC_UPDATE_v0.3.md §12.13`.

**What.** Hammond et al.'s Spectral Graph Wavelet Transform — Chebyshev-
polynomial approximations of band-pass filters in the Laplacian
eigenspectrum.

**Composition.** Direct composition of `graph.laplacian` +
`matrix_polynomial` ([`matrix-functions.md`](matrix-functions.md), §12.2).
The Chebyshev recurrence is matvec-only against the Laplacian, so it reuses
the matrix-free operator interface — no eigendecomposition required.

**Likely consumer.** Multiscale features on cortical-surface graphs,
surface-domain wavelet shrinkage for denoising, SGWT-based feature
engineering for connectome analyses.

**Effort.** S — depends on §12.2.

**Live-code status.** No `graph.wavelet`. `graph.laplacian` /
`laplacian_matvec` are shipped; the missing piece is `matrix_polynomial`
(the Chebyshev band-pass filter), tracked in
[`matrix-functions.md`](matrix-functions.md).

## Cross-references

- `SPEC_UPDATE_v0.3.md §12.13` — origin entry; `§13` — acceptance protocol.
- [`matrix-functions.md`](matrix-functions.md) — `matrix_polynomial`
  dependency.
- `src/nitrix/graph/laplacian.py`.
