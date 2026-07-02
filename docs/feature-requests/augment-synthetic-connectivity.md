# Synthetic connectivity & time-series generators in `nitrix.augment`

> **Status (2026-06-30): PROPOSED — weakest of the migration batch; admit a clean
> parameterised core only.** From the
> [`hypercoil-examples` migration](hypercoil-examples-migration.md)
> (`synthetic/scripts/{mix,sylo,corr,filter,denoise}.py`). Keyed forward models
> that synthesise fMRI-like signals / connectomes with **known ground-truth
> structure** — the substrate for testing/benchmarking the whole ecosystem
> (and nitrix's own suites + `nitrix-perf-bench`) against a known answer.
> Proposed home: `nitrix.augment`, beside the existing keyed synthesis atoms
> (`gmm_label_to_image`, `simulate_bias_field`).
>
> **Correctness mandate — theory over legacy.** The migrated functions are
> experiment scripts; the legacy is a starting point, not a spec. The defining
> obligation here is that **the generated data provably has the structure it
> claims** (a recoverable planted covariance / connectome / state sequence) — that
> is the test, not parity with the legacy output.

## 1. The clean core (a coherent keyed-generator family)

Each a pure function of a `jax.random` key (tenet 1 — RNG policy stays with the
caller), statically shaped, jit/vmap-clean:

- **Band-limited latent sources** (`synth_slow_signals`) — coloured/band-limited
  noise via spectral shaping.
- **Sparse mixing matrix** (`create_mixture_matrix`) — Poisson-cardinality rows,
  L1-normalised: a sparse linear map from latent to observed signals with a known
  mixing structure.
- **Mixture forward model** (`mix_data` / `synthesise_mixture`) — `mixture @
  sources` (+ optional local component): observed BOLD-like signals with a known
  latent covariance / connectome.
- **Low-rank-block connectome** (`synthesise_lowrank_block`) — `tanh(L Lᵀ) + N
  Nᵀ` with planted modular block structure + noise outer product: a symmetric
  low-rank-plus-noise connectivity matrix with known communities.
- **Markov state sequences** (`simulate_markov_transitions`) — keyed
  discrete-state trajectories from a (log-space) transition matrix, for dynamic-FC
  / state-switching synthesis.

These **compose** (latent sources → mixing → observed; state-switching for
dynamic FC), which is the family's coherence and its §9 admission story (named
synthesis vocabulary, like the existing augment atoms). The experiment-specific
variants (e.g. `denoise.py`'s 3-jitter QC-confound model) are **not** admitted —
they stay downstream; only the reusable core moves.

## 2. Correctness / quality points (theory over legacy)

1. **Generated structure is provable.** Each generator ships with the closed-form
   ground truth it plants (the mixing matrix / population covariance / community
   labels / stationary distribution), and the test recovers it (e.g. empirical
   `cov` of the mixture ≈ `mixture @ cov(sources) @ mixtureᵀ`).
2. **Spectral shaping ≠ FFT bin-zeroing.** The legacy band-limits by zeroing rfft
   bins (a brick-wall filter → spectral leakage / ringing). Use a documented
   smooth spectral window, or reuse `nitrix.signal` filtering — do not ship a
   leaky brick-wall as if it were a clean band-limit.
3. **jit-safety: fix the data-dependent loops.** `create_mixture_matrix`'s
   per-row `permutation(...)[:n]` (data-dependent slice) and
   `simulate_markov_transitions`' Python time-loop must become fixed-shape
   `vmap` + boolean mask and `lax.scan` respectively (the legacy is eager-only).
4. **Parameterise, don't hard-code.** `synthesise_lowrank_block` hard-codes a
   4-column block layout ("one-off" per its own comment); the migrated kernel
   takes the block structure / rank as arguments.

## 3. Surface & boundaries

- `augment` keyed generators returning plain arrays (signals, and optionally the
  ground-truth mixing/labels for tests). Names TBD but vocabulary-coherent with
  the existing atoms.
- Pure `jax`/`jaxtyping`/`numpy`; **drop** `hypercoil.engine.Tensor`; **replace**
  `scipy.ndimage.gaussian_filter1d` with `nitrix.smoothing`; remove all
  plotting / `print` / kmeans / autoencoder code (downstream).
- Augmentation *policy* (compose, registries, multi-crop) stays in
  `ilex`/`bitsjax` (SPEC §4.14); these are leaf generators only.

## 4. Acceptance

- Each generator's planted structure is recovered from its output within CI
  (covariance / community / transition-matrix recovery tests).
- jit + vmap clean (no data-dependent shapes); deterministic given `key`.
- Band-limited sources have the specified spectral support without brick-wall
  ringing artefacts.

## 5. Cross-references

- Ledger: [`hypercoil-examples-migration`](hypercoil-examples-migration.md).
- Sibling atoms: `augment.gmm_label_to_image`, `augment.simulate_bias_field`;
  reuse `signal` (filtering) + `smoothing` (Gaussian).
- Consumers: nitrix's own test/benchmark suites + `nitrix-perf-bench`
  (known-ground-truth synthetic data).
- Provenance: `hypercoil-examples/synthetic/scripts/{mix,sylo,corr,filter,denoise}.py`.
