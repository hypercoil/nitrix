# Continuous wavelet transform — `nitrix.signal.cwt`

> **Status (2026-06-02): not started.** Brainstorm candidate; promotion
> gated by the §13 acceptance protocol. Provenance:
> `docs/feature-requests catalogue §12.12`.

**What.** Continuous-wavelet analysis at user-specified mother wavelets
(Ricker, Morlet, Paul).

**Proposed surface.**

```python
def cwt(x, scales, *, wavelet='morlet'): ...   # -> scalogram (scales × time)
```

**Composition.** Extends the existing `nitrix.signal` family
(`lomb_scargle_*`, `analytic_signal`, `tsconv`) with continuous-wavelet
analyses. The transform is a bank of scaled-wavelet convolutions, composing
the shipped `signal.tsconv` / FFT machinery.

**Likely consumer.** fMRI / EEG time-frequency analysis, non-stationary
signal characterisation, scalogram features for downstream classifiers.

**Effort.** S.

**Live-code status.** No `cwt`. `signal/__init__` ships
`lomb_scargle_periodogram` / `lomb_scargle_interpolate`, the IIR/SOS filter
family, windows, and detrending; `stats.analytic_signal` / `hilbert_transform`
cover the analytic-signal side. The convolution substrate (`signal.tsconv`)
is the natural host.

## Cross-references

- `docs/feature-requests catalogue §12.12` — origin entry; `§13` — acceptance protocol.
- `src/nitrix/signal/tsconv.py` — the convolution substrate.
- [`docs/design/signal-and-numerics.md`](../design/signal-and-numerics.md).
