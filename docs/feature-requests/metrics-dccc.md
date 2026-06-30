# Distance-controlled boundary / coaffiliation coefficient (DCBC / DCCC) in `nitrix.metrics`

> **Status (2026-06-30): PROPOSED.** From the
> [`hypercoil-examples` migration](hypercoil-examples-migration.md)
> (`atlas/dccc.py`). A **parcellation-quality score kernel**: how well a
> parcellation's boundaries separate functionally distinct cortex, *controlling
> for spatial distance* (the confound that nearby vertices are both more
> correlated and more likely co-parcellated). No DCBC/DCCC anywhere in nitrix
> (grep). Proposed home: `nitrix.metrics` (a comparison/validity score kernel,
> §5-clean: unreduced by default, no scalarisation).
>
> **Correctness mandate — theory over legacy.** Clean-room from Zhi et al. (2022,
> *Cerebral Cortex*, "Evaluating brain parcellations using the distance-controlled
> boundary coefficient"); the legacy `dccc` is the recovery oracle. The key
> obligation: **the continuous relaxation must reduce to the published DCBC** in
> the hard-assignment + indicator-bin limit (the headline property test).

## 1. Theory

**DCBC (hard, published).** Over all vertex pairs, bin by **geodesic distance**;
within each distance bin `b` compute `corr_within(b) − corr_between(b)` (mean
functional-profile correlation of same-parcel vs different-parcel pairs at that
distance); the DCBC is the pair-count-weighted average over bins. Distance
binning is what makes it *controlled* — it removes the spatial-autocorrelation
confound that inflates naive boundary measures.

**DCCC (continuous, differentiable — the migrated variant).** Replaces the hard
parcel indicator with a **soft co-assignment** `Σ_p Q_a,p Q_b,p` (probabilistic
coaffiliation) and the hard distance bin with a **distance kernel**, so the
coefficient is differentiable w.r.t. a soft parcellation `Q` — usable as a
quality *signal*, not just an evaluation. The legacy computes, per distance
sample, a co-assignment- and kernel-weighted within/between correlation contrast
(`einsum` over vertex pairs) and integrates over the distance axis.

## 2. Correctness points (theory over legacy)

1. **Reduce-to-DCBC limit.** With hard one-hot `Q` and an indicator (top-hat)
   distance kernel, `dccc` **must** equal the published DCBC numerically. This is
   the defining test; if the legacy `einsum`/normalisation does not satisfy it,
   the legacy is wrong and we follow the paper.
2. **The distance kernel is not part of canonical DCBC.** The legacy's default
   `gauss_kernel(2.)` weighting and the arbitrary `integral_samples =
   arange(40)*0.9` are hypercoil choices, not DCBC. Expose binning/kernel
   explicitly (an ADT or array args, not a captured `callable`), default to the
   indicator bins that reproduce DCBC, and document the Gaussian-kernel variant as
   a *soft* generalisation.
3. **Reuse the geodesic distance.** `dccc.spherical_distance` duplicates
   `geometry.spherical_geodesic_distance` (which already handles the `arccos`
   clipping / numerical edge); reuse it — do not re-port (and do not silently add
   the legacy's self-`eye` + radius factor without justification).
4. **Edge cases:** empty distance bins (no pairs → defined as 0 contribution, not
   NaN); single-parcel / degenerate `Q`; the soft↔hard correspondence at the
   boundaries; weighting by pair count vs by bin (match the paper).

## 3. Performance (the legacy is "extremely slow")

The legacy's own docstring flags it. The all-pairs `O(V²)` distance + correlation
contrast must be expressed as a vectorised / `vmap`-ed reduction (no Python bin
loop with `print`), and ideally as an **ELL/semiring reduction over a bounded
geodesic neighbourhood** (DCBC only uses pairs out to a max distance, so the full
`V²` is wasteful) — the same bounded-neighbourhood pattern as
`smoothing.bilateral_gaussian`. Wall-clock at brain scale → `nitrix-perf-bench`.

## 4. Surface & boundaries

- `metrics.dccc(coords, features, assignment, *, distance=None, bins=..., kernel=...,
  reduction='none')` — returns the **unreduced per-bin** within−between contrast
  by default (the score-kernel canonical form, §5), with the flat
  `reduction ∈ {'none','sum','mean'}` leaf and the pair-count-weighted mean (the
  §5.2 domain-mask mean analogue). Differentiable in `features` and the soft
  `assignment`.
- Pure `jax`/`jaxtyping`/`numpy`; `nibabel`/`pandas`/`seaborn`/plotting → dropped
  (legacy `main`/`plot_dccc` are downstream). `gauss_kernel`/`bin_kernel` do not
  migrate as named symbols (trivial closures → an explicit kernel arg/ADT).

## 5. Acceptance

- **Reduces to DCBC**: hard `Q` + indicator bins reproduce a published-DCBC
  reference (the defining oracle).
- Monotone sanity: a ground-truth parcellation scores higher than a random one;
  scrambling boundaries lowers the coefficient.
- `jax.grad` w.r.t. a soft `assignment` is finite and points toward
  boundary-improving moves.
- Empty bins / single parcel handled (no NaN); reuses
  `geometry.spherical_geodesic_distance`.

## 6. Cross-references

- Ledger: [`hypercoil-examples-migration`](hypercoil-examples-migration.md).
- Reuse: `geometry.spherical_geodesic_distance`; bounded-neighbourhood reduction
  pattern `smoothing.bilateral_gaussian` / `semiring.semiring_ell_*`.
- Sibling validity kernels: `graph` parcellation (`surface_boundary_map`,
  `eta_squared`), `metrics` overlap (`dice`/`jaccard`).
- Provenance: `hypercoil-examples/atlas/dccc.py`. Reference: Zhi, King,
  Hernandez-Castillo & Diedrichsen (2022), *Cerebral Cortex*.
