# Non-parametric combination (NPC) in `stats.inference`

**Status:** OPEN · **Effort:** S · **Risk:** low · **Subpackage:** `stats.inference`

## Summary

Add **non-parametric combination** — joint inference across several partial tests (contrasts,
modalities, hemispheres, timepoints) by combining them **through the joint permutation
distribution**, rather than by combining marginal p-values under an independence assumption they do
not satisfy.

This is the one gap found by a coverage triage of mass-univariate modelling
(`nitrix-moonshot-round-2-candidates.md`, triage round 3). Everything else in that family is already
shipped: `sandwich_cov`, `fdr_bh`/`fdr_by`/`fdr_storey`, `gpd_pvalue`, `permutations(blocks=…)`,
`permutation_test` with Freedman–Lane, TFCE, cluster mass/size, `conjunction`, `sign_flips`.

## Why it is not `conjunction`, and not "just combine the p-values"

`conjunction` answers *"is the effect present in **all** partial tests?"* (an intersection–union
test). NPC answers the **union–intersection** question — *"is there evidence **somewhere** in this
set, accounting for the fact that the partial tests are dependent?"* — and it is the dependence that
makes it non-trivial.

Combining marginal p-values with Fisher / Tippett / Stouffer under an **independence** assumption is
anti-conservative when the partial tests are correlated, which they always are here (the same
subjects, overlapping data, spatially smooth statistics). The whole point of NPC is that **the
permutation distribution already carries the dependence**: because every partial test is re-computed
under the *same* permutation of the *same* subjects, the joint null is obtained by applying the
combining function **within each permutation** and taking the distribution of the combined statistic
across permutations. No independence assumption is made, and none is needed.

That is also why this is cheap for us: the `(n_perm, …)` axis is **already materialised** by
`permutation_test`. NPC is a reduction *along* it — not new machinery.

## Design

```
combine(p_partial, method, ...) -> combined statistic       # the combining function
npc(statistics_per_permutation, *, method, ...) -> NPCResult
```

- **Combining functions** as an enum (`Fisher`, `Tippett`, `Stouffer`, `Pearson`), not bare strings.
- The input is the **per-permutation partial statistics** — shape `(n_perm, n_partial, *spatial)` —
  i.e. the object `permutation_test` already computes. NPC converts each partial statistic to a
  *permutation* p-value **within** the permutation set, applies the combining function across the
  partial axis, then reads the combined statistic's own permutation distribution.
- Composes with the shipped FWE machinery: the max-statistic over space, of the *combined* statistic,
  gives FWE-corrected NPC — no new correction path.
- Fully vectorised over the spatial axis; no new solver, no new dependency.

## Validation

- **Exactness under exchangeability:** with a synthetic exchangeable design and a true null, the NPC
  p-value is uniform. Assert the *empirical* false-positive rate at α ∈ {0.05, 0.01}.
- **The dependence is what matters — test it:** construct partial tests with a *known, strong*
  correlation, and assert that (a) NPC controls the false-positive rate, and (b) **marginal Fisher
  combination under an independence assumption does not** — i.e. exhibit the failure the method
  exists to prevent, rather than merely asserting the method works. A corpus that the naïve
  combination also passes certifies nothing.
- **Power:** on a planted effect present in a *subset* of partial tests, NPC is more powerful than
  the conjunction (which requires the effect everywhere) and than a Bonferroni over partials.
- Oracle: a direct `numpy` reference over a small design.

## Non-goals

- No scalarisation and no objective composition — this returns statistic/p-value maps.
- No new permutation machinery: NPC **consumes** `permutations()` and its exchangeability blocks.

## Reference

Winkler et al., *Non-parametric combination and related permutation tests for neuroimaging*,
Human Brain Mapping 37:1486–1511 (2016). https://doi.org/10.1002/hbm.23115
