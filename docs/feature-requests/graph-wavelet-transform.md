# Graph wavelet transforms (SGWT) — `nitrix.graph.wavelet`

> **Status (2026-07-07): SHIPPED (`nitrix.graph.graph_wavelet_transform`).**
> Hammond et al.'s spectral graph wavelet transform: band-pass filtering in the
> Laplacian eigenspectrum via a Chebyshev approximation applied by **matvec only**
> (`laplacian_matvec` + the shipped `linalg.chebyshev_apply`/`chebyshev_coefficients`
> — the §12.2 dependency, now shipped), with power-iteration λmax estimation. No
> eigendecomposition → GPU-native and jit-clean on dense cortical graphs; the
> Chebyshev basis is built once (`O(order)` matvecs) and recombined per scale.
> Default band-pass `mexican_hat_kernel` (`g(x)=x e^{-x}`); verified against a
> dense eigendecomposition to machine precision. Provenance:
> `docs/feature-requests catalogue §12.13`.

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

- `docs/feature-requests catalogue §12.13` — origin entry; `§13` — acceptance protocol.
- [`matrix-functions.md`](matrix-functions.md) — `matrix_polynomial`
  dependency.
- `src/nitrix/graph/laplacian.py`.
