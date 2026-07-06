# Distance-controlled boundary / coaffiliation coefficient (DCBC / DCCC) in `nitrix.metrics`

> **Status (2026-07-06): DEFERRED to a dedicated parcellation sprint — the soft
> (DCCC) extension is mathematically flawed as formulated.** The hard DCBC is
> fine; the *migrated probabilistic relaxation* (marginal coaffiliation `c_ij =
> Σ_p Q_i,p Q_j,p`) does **not** yield a parcellation-quality score comparable
> across deterministic and probabilistic parcellations, and systematically
> **favours determinism** — verified empirically (§0). Relaxing DCBC to
> continuous/probabilistic parcellations is genuinely nontrivial (possibly
> ill-posed as a difference-of-conditional-means) and is the sprint's problem.
> This is **separate** from the distance-binning kernel (also added to the legacy
> DCCC), which is its own (independent) design axis. Do not ship the soft
> coaffiliation contrast as-is.
>
> **Correctness mandate — theory over legacy.** Clean-room from Zhi et al. (2022,
> *Cerebral Cortex*, "Evaluating brain parcellations using the distance-controlled
> boundary coefficient"); the legacy `dccc` is the recovery oracle. The key
> obligation: **the continuous relaxation must reduce to the published DCBC** in
> the hard-assignment + indicator-bin limit (the headline property test).

## 0. Why the soft coaffiliation extension is flawed (empirical, 2026-07-06)

The migrated DCCC generalises the hard within/between indicator to the **marginal
coaffiliation** `c_ij = <Q_i, Q_j>` (probability loci `i, j` share a parcel,
marginalising each locus's affiliation), then forms the coaffiliation-weighted
contrast `within − between` with `within = Σ c_ij r_ij / Σ c_ij`, `between = Σ
(1−c_ij) r_ij / Σ (1−c_ij)` (`r_ij` = functional correlation). For hard one-hot
`Q` this is exactly single-bin DCBC. **But the score conflates representation
*sharpness* with parcellation *quality*.** A simulation (block-structured signals,
genuine boundary ambiguity; `tools`-side, not shipped) shows:

- **Sharpness confound / non-comparability.** Take the *same* partition (argmax =
  ground truth at every temperature) and only vary how sharply `Q` represents it:
  DCCC collapses **monotonically** from ~0.50 (one-hot) to ~0.01 (near-uniform),
  in **8/8 seeds**. The identical partition decision scores 50× differently by
  representation entropy alone — so a soft parcellation's DCCC is *not* comparable
  to a hard one's. Analytic cause: as `Q_i → uniform`, `c_ij → 1/K` constant, so
  the within- and between-weightings converge and both correlations tend to the
  global mean ⇒ `DCCC → 0`, regardless of boundary quality.
- **Favours determinism.** The hard argmax parcellation beats the *honest* soft
  posterior calibrated to the true boundary mixing (`α` own / `1−α` neighbour) in
  **8/8 seeds** — the metric penalises correctly representing genuine uncertainty.
- **Refinement of the recollection (not an absolute ordering).** It is *not* that
  hard always wins: a sufficiently corrupted hard parcellation (10 % labels wrong)
  still scored below the honest soft one. The defect is a **systematic
  sharpness-dependent bias/offset**, so scores are only comparable *within* a
  fixed sharpness — invalid for the intended cross-parcellation comparison.

A valid probabilistic DCBC must **decouple sharpness from boundary quality**
(e.g. normalise by the sharpness-achievable contrast; or an information-theoretic
/ proper-scoring-rule reformulation) — deferred, non-trivial, sprint-scoped.

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
- **NEW gate for any soft formulation (§0):** *sharpness-invariance* — a fixed
  partition represented at different `Q`-entropies (same argmax) must score
  (near-)equally, and an honest calibrated soft posterior must not lose to its
  hard argmax purely for being soft. The marginal-coaffiliation contrast **fails**
  this; it is the acceptance bar the sprint's reformulation must clear.

## 6. Cross-references

- Ledger: [`hypercoil-examples-migration`](hypercoil-examples-migration.md).
- Reuse: `geometry.spherical_geodesic_distance`; bounded-neighbourhood reduction
  pattern `smoothing.bilateral_gaussian` / `semiring.semiring_ell_*`.
- Sibling validity kernels: `graph` parcellation (`surface_boundary_map`,
  `eta_squared`), `metrics` overlap (`dice`/`jaccard`).
- Provenance: `hypercoil-examples/atlas/dccc.py`. Reference: Zhi, King,
  Hernandez-Castillo & Diedrichsen (2022), *Cerebral Cortex*.
