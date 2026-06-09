# nitrix — Specification update (v0.4 → v0.5)

> **Status.** Boundary-definition + ratification addendum. Apply on top of
> v0.4. All prior changes stand; this patch draws the **score-kernel ↔
> scalarisation boundary** between nitrix and nimox, blesses **keyed pure
> generators** as an in-scope symbol category, and ratifies the
> **`nitrix.augment`** subpackage — the three boundary questions surfaced
> (not created) by the ilex-backlog implementation sprint (Phases 1–5,
> 2026-06-08/09). It also records that sprint's shipped scope additions in
> the deviation log (§10.A).
>
> **Summary of changes.**
> - §1 — formal boundary: nitrix owns the *score kernel*; nimox owns the
>   *scalarisation*. Includes the one weighted reduction nitrix keeps (the
>   domain-mask weighted mean) and the rule that there is **no `loss`
>   namespace in nitrix**.
> - §2 — tenet-1 clarification: **keyed pure generators**
>   `(shape/params, key) -> Array` are in scope (a `key` is itself a typed
>   `Array`; `identity_grid` / `signal.window` are precedent).
> - §3 — `nitrix.augment` ratified, with its substrate-composition story and
>   the **primitive-admission rule** (resolving the uneven "primitive bar").
> - §4 — resolutions applied to the sprint's surface.
> - §10.A — deviation log for the sprint.

---

## §0. Rationale: the sprint surfaced an undrawn boundary

The ilex-backlog sprint shipped ~15 primitives and one new subpackage across
`metrics`, `stats`, `numerics`, `geometry`, `morphology`, `register`,
`sparse`, and `augment`. None of it violated the charter. But it revealed
that two lines were **never specified precisely enough** and now must be:

1. **Losses vs metrics vs substrate.** Earlier specs put losses in nimox
   (`hypercoil/loss/`). The registration suite (R0) already had to land
   `metrics.{ssd,ncc,lncc,…}` in nitrix because the *recipes* live here and
   need a differentiable similarity to optimise. This sprint widened that
   with overlap / classification / contrastive objectives. Where is the line?

2. **Reductions.** The sprint explicitly declined to implement
   scalarisation (nimox's `scalarise` + `scheme`), yet every new
   metric/loss/stat grew a `reduction=` kwarg, and the `metrics` variant a
   `mask`. The masked-mean branch is numerically `Σ(w·x)/Σw` — i.e.
   `scalarise.wmean`. nitrix appeared to ship a copy of a system it had
   declared out of scope.

§1 resolves both. The key realisation: nimox's `scalarise` is a
**higher-order combinator** system (`mean_scalarise(inner=…)`,
function→function); nitrix's `reduction=` is a **flat leaf** (value→value).
They are different *kinds* of abstraction, in a subset relationship, not
rivals. Drawing the boundary at that seam keeps one vocabulary.

This is consistent with §0 of v0.3: consumer-driven drift is expected; each
addition must respect the substrate's organising principles. The cure for
drift is to *write the principle down*, which is what §1–§3 do.

---

## §1. The score-kernel ↔ scalarisation boundary  [normative]

### §1.1 Definitions

- **Score kernel** (nitrix). A pure function that compares or transforms
  arrays and returns a numeric tensor carrying **irreducible numerical
  content** — a numerically-stable-from-logits rewrite, a soft-overlap
  coefficient, a distance / similarity, a distributional closed form, a
  field regulariser. Its canonical output is the **unreduced** tensor
  (per-element / per-region / per-sample). It is value → value:
  `(Array, …) -> Array`.

- **Scalarisation** (nimox). A higher-order map that **wraps** a score
  (function or tensor) into the single training scalar — composing
  reductions (`mean_scalarise(inner=…)`), weighting whole terms, and
  combining terms (`scheme`). It is function → function (a combinator).

### §1.2 The invariant

1. **nitrix owns the score kernel; nimox owns the scalarisation.** A nitrix
   score kernel returns the unreduced tensor by default
   (`reduction='none'`). It MAY expose a **flat, non-compositional**
   reduction — `reduction ∈ {'none','sum','mean'}` — as a leaf convenience
   for standalone and test use. That leaf is precisely the innermost element
   nimox's `inner=` composition can call or replace; the two read as
   "leaf ⊂ combinator system," not as two systems. nitrix does **not** own
   compositional, norm, max, softmax-self-weighted, or `inner=`-composed
   scalarisation.

2. **The one weighted reduction nitrix owns is the domain-mask weighted
   mean** `Σ(w·x) / Σw` (with `w` a per-element non-negative weight, the
   `mean` branch when a `weight`/`mask` is supplied). *Rationale — this is a
   property of the measurement, not the objective.* In neuroimaging a
   foreground / brain / validity mask is intrinsic to the *numeric*: an
   LNCC, Dice, or masked-token cross-entropy computed over background is
   numerically meaningless, so excluding it is part of computing the score,
   not part of weighting the objective. This is categorically distinct from
   **objective weighting** (class weights, per-term weights, hard-example
   weights), which is determined by the training scheme and is nimox's.
   nitrix `weight` ≡ a per-element **domain mask**; nimox scalarise weighting
   ≡ per-term / per-sample **objective weight** + composition.

3. **Objective structure is a recipe → nimox.** View-pair layout,
   masked-token *selection*, EMA / centre bookkeeping, multi-term weighting
   are recipe conventions. A nitrix score kernel either takes such structure
   as an explicit argument or exposes a structure-free core; it does not bake
   a recipe in.

4. **There is no `loss` namespace in nitrix.** "Loss" — a signed, scalarised,
   weighted objective — is a nimox concept. nitrix hosts *score kernels*
   (`metrics`, `stats`, `register.regulariser`); nimox forms a loss as
   `scalarise(scheme(± kernel(...)))`. `metrics` is therefore the namespace
   of *differentiable comparison kernels* (similarity / overlap /
   classification / contrastive), not "losses."

### §1.3 Consequences (where each existing symbol lands)

| Symbol(s) | Classification | Home |
|---|---|---|
| `ssd`, `ncc`, `lncc`, `joint_histogram`, `mutual_information`, `correlation_ratio` | similarity kernel | `metrics` |
| `dice`, `jaccard` | overlap kernel | `metrics` |
| `bce_with_logits`, `cross_entropy_with_logits`, `focal_loss` | classification kernel (stable-from-logits) | `metrics` |
| `nt_xent`/`info_nce`, `dino_cross_entropy`, `ibot_cross_entropy`, `koleo` | contrastive kernel | `metrics` |
| `gaussian_nll`, `kl_diagonal_gaussian` | distributional closed form | `stats` |
| `gradient_smoothness`, `bending_energy`, `jacobian_folding_penalty` | field regulariser kernel | `register.regulariser` |
| EMA / centre maintenance, view-pair / masked-token *layout*, term weighting, `scheme` | scalarisation / recipe | **nimox** |

### §1.4 The single reduction surface

nitrix's three duplicated `_reduce` helpers (`metrics/_common`,
`stats/gaussian`, `register/regulariser`) collapse to one
`nitrix._internal.reductions`:

```python
Reduction = Literal['none', 'sum', 'mean']

def reduce(values, *, axis=None, weight=None, reduction='mean') -> Array:
    # 'mean' with weight  -> Σ(w·x)/Σw   (the domain-mask weighted mean, §1.2.2)
    # 'mean' without       -> values.mean(axis)
    # 'sum'/'none'         -> the obvious
```

This is the leaf of §1.2.1, named and documented to read as the minimal
subset of nimox `scalarise` (one vocabulary, not two).

---

## §2. Tenet-1 clarification: keyed pure generators are in scope

SPEC §2 tenet 1 reads "every public symbol is a function
`(Array, …) -> Array`." This is **clarified** (not expanded) to admit:

> **Keyed pure generators.** A symbol may take a `jax.random` **key** and /
> or a static shape / parameter spec and return an `Array`:
> `(shape | params, key) -> Array`. A PRNG key is itself a typed `Array`, and
> shape-driven generators already exist (`geometry.identity_grid`,
> `signal.window`), so this is within the spirit of tenet 1. Such a generator
> is "pure" in the JAX sense: deterministic given `(inputs, key)`, no hidden
> state. RNG *policy* (which key, how it is split, sampling schedules) remains
> the caller's (bitsjax / nimox / thrux).

This blesses `augment.{gaussian_noise, rician_noise, random_*, gmm_label_to_image,
simulate_bias_field}` and `stats.pca(solver='randomized')`.

---

## §3. `nitrix.augment` — ratified, with its substrate-composition story

SPEC_UPDATE_v0.3 §14 forbids "new top-level subpackages without a clear
substrate-composition story," naming `nitrix.image.classifier` /
`nitrix.model.X` as the target — i.e. **model / application** packages.
`augment` is a **numeric category** (pure deterministic augmentation
kernels), not a model package, and it carries the required story:

- **Composes existing substrate.** `augment.geometric.random_resized_crop`
  calls `geometry.spatial_transform`; `random_affine_matrix` calls
  `geometry.params_to_affine_matrix`; `random_svf_displacement` calls
  `geometry.integrate_velocity_field`. These add **no** new numeric.
- **Irreducible new leaf atoms** (each a §13-qualifying primitive: concrete
  consumer = ilex FM-pretraining / synth\* augmentation; effort XS–S;
  separation-of-concerns held): `gamma_contrast`, `random_histogram_shift`,
  `gibbs_ringing`, `gaussian_noise`, `rician_noise`, `gmm_label_to_image`,
  `simulate_bias_field`. These are keyed generators / pointwise transforms,
  blessed by §2.
- **Policy stays out.** The `AugmentationSpec` / registry / compose /
  multi-crop view-fan-out (the role/shape vocabulary) stays in ilex /
  bitsjax; `augment` ships only the kernels they call.

### §3.1 Primitive-admission rule (resolving the uneven bar)

The sprint excluded `mse`/`l1` as "one-liners" yet admitted `gaussian_noise`
/ `random_flip`. The rule that makes this consistent:

> A symbol is admitted iff it has **irreducible numerical / structural
> content**, **or** it is a named member of a **coherent vocabulary family**
> whose value is discoverability + convention-consistency. A symbol is
> **excluded** when it is *(trivial elementwise op) ∘ (reduction)* with no
> content of its own — because the elementwise part is trivial and the
> reduction is nimox's (§1).

Therefore: `mse`/`l1` (= `square`/`abs(a-b)` ∘ reduction) stay **out** — they
are `scalarise(square(a - b))` in nimox. `gamma_contrast` (bracket-normalise
+ power), the noise generators (keyed field generation), `gibbs_ringing`
(k-space truncation), `gmm_label_to_image`, and the `augment.geometric`
vocabulary (`random_flip` / `random_crop` / `random_resized_crop`) stay
**in** — content and/or the `augment` vocabulary family.

---

## §4. Resolutions applied to the sprint surface

Ratified 2026-06-09 (the three boundary decisions), implemented on
`feat/ilex-backlog-impl`:

1. **`metrics` charter reframed** from "image-similarity metrics" to
   "differentiable comparison kernels (similarity / overlap / classification
   / contrastive)." No `loss` namespace (§1.2.4).
2. **`nt_xent` → layout-agnostic `info_nce(za, zb, *, temperature)`** (the
   positive of `za[i]` is `zb[i]`). This removes the baked adjacent-pair
   convention (§1.2.3) **and** the `-2/τ` self-mask and its
   partition-function bias (a cross-view similarity has no self-pairs). The
   stacked single-tensor layout becomes a nimox-side adapter.
3. **`ibot_cross_entropy` → `dino_cross_entropy(reduction='none')` per token
   + the domain-mask weighted reduction** over the token axis (mask = the
   masked-token indicator). The "baked masked averaging" becomes the one
   sanctioned domain-mask reduction (§1.2.2).
4. **One `nitrix._internal.reductions`** (§1.4) replaces the three `_reduce`
   copies.

---

## §10.A Deviation log — ilex-backlog sprint (Phases 1–5)

Extends `IMPLEMENTATION_PLAN.md §10` (see the dated pointer there). Each
addition has a concrete consumer (the 2026-06-02 / 06-08 ilex audits,
tracked in `docs/feature-requests/`), composes or extends the substrate, and
holds the separation-of-concerns invariant.

| Symbol(s) | Module | Consumer | Composition / note | Shape |
|---|---|---|---|---|
| `l2_normalize`, `lp_normalize`, `instance_norm` | `numerics.normalize` | SSL embeds, decoders | elementwise; clamp-denominator | JAX |
| `dice`, `jaccard` | `metrics.overlap` | seg ports | soft set-overlap; shared `_overlap_sums` | JAX |
| `bce_with_logits`, `cross_entropy_with_logits`, `focal_loss` | `metrics.classification` | seg/cls ports | stable-from-logits; dedup ×3 | JAX |
| `nt_xent`/`info_nce`, `dino_cross_entropy`, `ibot_cross_entropy`, `koleo` | `metrics.contrastive` | FM pretraining | log-softmax + stop-grad; §4.2/4.3 | JAX |
| `kl_diagonal_gaussian`, `gaussian_nll` | `stats.gaussian` | VAE / LDM | closed forms; log-var param | JAX |
| `PCAResult`, `pca_fit/transform/inverse` (full / gram / randomized / auto) | `stats.pca` | krakencoder, CompCor (thrux) | eigh-only (cuSolver-safe); §2 keyed | JAX |
| `params_to_affine_matrix`, `affine_matrix_to_params`, `angles_to_rotation_matrix`, `rotation_matrix_to_angles`, `fit_affine`, `make_square_affine`, `invert_affine`, `compose_affine` | `geometry.affine` | lab2im, synthmorph/voxelmorph | Euler/scale/shear chart; safe_* solvers | JAX |
| `sample_at_points` | `geometry` | cortex_ode / surfnet (grid_sample) | wraps `_sample_at_coords` | JAX |
| `connected_components`, `largest_connected_component` | `morphology` | strip/seg cleanup | label-propagation; pointer-jumping | JAX |
| `pad_to_multiple`, `crop_to_multiple`, `nonzero_bounding_box`, `gaussian_window`, `overlap_add` | `numerics.spatial` | every CNN/UNet v0 | shape/window math | JAX |
| `percentile_rescale(mask=)` | `numerics.normalize` | skull-strip norm | masked nanpercentile | JAX |
| `gradient_smoothness`, `bending_energy`, `jacobian_folding_penalty` | `register.regulariser` | voxelmorph/synthmorph | on `spatial_gradient` / `jacobian_det` | JAX |
| `compute_vertex_normals`, `mesh_laplacian_smooth` | `sparse.mesh` | topofit/cortex_ode/surfnet | per-vertex scatter | JAX |
| `euler`, `rk4`, `odeint` | `numerics.ode` | cortex_ode/surfnet NODE | `lax.scan`; diffrax-free | JAX |
| `safe_cholesky`, `_make_safe_op` | `linalg._solver` | affine/PCA on dead cuSolver | probe+latch→CPU | JAX |
| `nitrix.augment` (subpackage) | `augment.{intensity,geometric,synthesis}` | ilex FM / synth\* | §3 | JAX |

**Sealed-spec status.** SPEC §4 gains `nitrix.augment` (per §3); SPEC §2
tenet 1 gains the keyed-generator clarification (per §2); the boundary in §1
is added as a new SPEC §15 ("Score-kernel ↔ scalarisation boundary") at the
next consolidation.

---

## §11. Out-of-scope reminders (unchanged, reaffirmed)

v0.3 §14 stands. `augment` is **not** a counterexample: it is a numeric
category with a substrate-composition story (§3), not a model / application
package. The no-PyTree-modules, no-message-passing-base-class, and
prefer-a-kwarg-over-a-fork rules all stand.
