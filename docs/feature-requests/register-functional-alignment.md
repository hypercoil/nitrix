# Functional alignment (Procrustes / ProMises) in `nitrix.register`

> **Status (2026-06-30): PROPOSED.** Migration target from the deprecated
> `hypercoil-examples` staging repo (`atlas/promises.py`, the "empty ProMises"
> step). Adds **alignment in representation space** to `nitrix.register`
> alongside the existing **spatial** registration recipes: a closed-form
> orthogonal-Procrustes aligner with an optional **matrix von Mises–Fisher**
> (matrix-Langevin) spatial prior on the rotation (the ProMises model), exposed
> on the SPEC §6.5 `fit`/`apply` seam. Numerics-only; the Equinox alignment
> module + the parcellation objective that wrap it stay downstream (`nimox`).
>
> **Home decision (user, 2026-06-30): `nitrix.register`.** Functional alignment
> is pairwise alignment — the same concern as spatial registration, in feature
> space rather than voxel/world space — so it joins the registration surface as
> a sibling recipe family, not a new top-level subpackage (SPEC §6.4).
>
> **Naming (user, 2026-06-30): functional alignment is a *family*, ProMises is
> one *method*.** Other hyperalignment algorithms (ridge/regression
> hyperalignment, optimal-transport, shared-response) will follow, selected by a
> method ADT — ProMises must not claim the generic `functional_align` name as a
> synonym.
>
> **Scale (user, 2026-06-30): the dense path is searchlight/parcel scale; the
> whole-brain regime needs *efficient ProMises*.** See §6.

## 0. Correctness mandate — theory over the legacy code

**This is a clean-room reimplementation from the ProMises literature, not a port.**
The migrated `empty_promises` is a *reference point and recovery oracle*, not a
spec; where it disagrees with the theory, the theory wins and the deviation is
documented. The legacy code is **known-suspect** at several points that a faithful
kernel must resolve from first principles, not copy:

1. **The "empty" double-whitening is a self-flagged hack.** The author's own
   docstring — *"Our modification can't really make any promises, so we call it
   the empty ProMises algorithm"* — admits it deviates from ProMises. Rotating
   `X` into *its own* right-singular basis `Q` and `M` into *its own* `P`
   discards the **relative** orientation that Procrustes exists to recover, and
   injects the PCA gauge ambiguity (per-component sign/order indeterminacy, and
   arbitrary rotations within repeated-singular-value subspaces). The nitrix
   default implements **actual ProMises** (Andreella & Finos 2022) — Procrustes
   on the data against a shared/reference frame with the matrix-vMF prior — and
   admits any reduced-basis variant only if separately derived and separately
   recovery-tested, never as the silent default. (The *efficient* ProMises §6 is
   the principled subspace reduction — distinct from this gauge-breaking hack.)
2. **Cross-product orientation is unverified in the source.** The commented-out
   `#N.T @ Z +` beside the live `Z.T @ N` shows the original author was himself
   unsure which orientation is correct (`XᵀM` vs `MᵀX` differ by `R ↔ Rᵀ`, i.e.
   align-source-to-reference vs the inverse). Derive it from the objective
   `argmin ‖XR − M‖` and pin it with the planted-rotation recovery oracle.
   *(Resolved in `linalg.orthogonal_procrustes`: `C = aᵀb`, recovery-tested.)*
3. **Reflection handling is absent.** `svd → UVᵀ` yields an *improper* rotation
   (`det = −1`) whenever the data demand a reflection; ProMises alignment of
   neural representations should decide rotation-only (`SO`) vs full-orthogonal
   (`O`) **from the theory** and expose it (`allow_reflection=`), not inherit
   whatever the raw SVD returns. *(Resolved: `allow_reflection`, default `O`.)*
4. **The co-transport Jacobian is a flagged TODO.** The source comments that
   co-transporting *probabilities* through the map needs a Jacobian correction it
   never applies — a latent correctness bug. Resolve from the change-of-variables
   theory or document the restriction (orthogonal maps preserve the measure, so
   the correction may be identity — *prove it*, don't assume it). *(Resolved:
   `|det R| = 1` ⇒ identity correction, documented in `functional_align_apply`.)*
5. **The prior is the matrix-vMF natural parameter, derived — not the ELL
   symmetrisation copied.** Verify `F`'s construction (concentration `κ` × a
   reference orientation / spatial-location prior) against the paper rather than
   reproducing the legacy reduced-frame symmetrisation `½(QSPᵀ + (PSQᵀ)ᵀ)`,
   which is downstream of (1). Following the theory, `F` is simply **added** to
   the cross-product (`C = XᵀM + F`) — the legacy triple-matvec (and its BCOO)
   were artefacts of the double-whitening and dissolve here.

Every claim above becomes a test (recovery oracle + a property the theory
guarantees), so "faithful to the theory" is enforced, not asserted.

## 1. What

Two subjects' data matrices `X` (source, `n × p`) and `M` (reference) describe
the *same* underlying signal in two **arbitrarily rotated feature bases** (two
subjects' functional connectomes; two encoders' latent spaces over a shared
sample set). Functional alignment recovers the orthogonal map `R` that best
rotates `X` onto `M`, so the aligned `X R` is directly comparable across subjects
(the hyperalignment task).

- **Base case — orthogonal Procrustes.** `R = argmin_{RᵀR=I} ‖XR − M‖_F = U Vᵀ`,
  `U Σ Vᵀ = svd(Xᵀ M)`. The primitive: `linalg.orthogonal_procrustes`.
- **Regularised case — ProMises.** A **matrix von Mises–Fisher** prior on `R`,
  MAP `R = polar(Xᵀ M + F)` — the prior's natural-parameter matrix `F` added to
  the cross-product. The matrix-vMF normaliser cancels in the MAP and is never
  formed, so this depends on **no** vMF directional-statistics machinery.

## 2. Surface — a method family on the §6.5 seam

- `FunctionalAlignment` — the fitted state (the orthogonal map `R`, plain
  arrays; a `NamedTuple`, never a module).
- `functional_align_fit(source, reference, *, method=ProMises(), ...) -> FunctionalAlignment`
- `functional_align_apply(data, alignment) -> data @ R` (the original source —
  reproducing the alignment — or co-registered auxiliary data: co-transport).
- `functional_align(source, reference, ...) ≡ apply(source, fit(...))` (the
  single-call convenience; byte-faithful, §6.5 invariant 2).
- **Method ADT** (the `Metric`/`TransformModel`/`Interpolator` precedent):
  `ProMises(prior=None, prior_weight=1.0, allow_reflection=True)` is the first
  implementer (`prior=None` ⇒ plain Procrustes). Future methods
  (ridge/regression hyperalignment, optimal-transport, shared-response) are new
  implementers; `functional_align(method=...)` stays the stable entry point.

**Register integration.** A distinct recipe family — *not* a `TransformModel`
chart (the solve is closed-form, not an `exp`-map optimised by GN/LM) and *not* a
`CoordinateSpace` (feature space, not voxel/world). Reframe
`register/__init__.py` from "pairwise registration recipes" to "pairwise
**alignment** recipes (spatial registration + functional alignment)".

## 3. Why (and why here)

- **Genuine gap.** `nitrix.register` aligns *images*; it cannot align
  *representations*. Grep confirms no Procrustes/alignment-in-feature-space.
- **Substrate composition (SPEC §9 gate).** Composes the linalg primitive
  (`orthogonal_procrustes`) + the §6.5 seam + a method ADT — no parallel API, no
  new subpackage.
- **Separation of concerns holds.** nitrix owns the closed-form estimator + the
  conventions that cross the fit/apply boundary; the Equinox alignment layer, the
  running-template GPA *policy* (iterative reference building), and the
  parcellation *objective* that consume it stay in `nimox`.

## 4. Acceptance

- `functional_align` with `ProMises(prior=None)` matches
  `scipy.linalg.orthogonal_procrustes` on a planted random-rotation recovery.
- ProMises (`prior` set) improves recovery over plain Procrustes when the
  cross-product is rank-deficient / noisy, on a planted spatially-structured
  rotation.
- `functional_align(src, ref) == functional_align_apply(src,
  functional_align_fit(src, ref))` byte-for-byte (§6.5 invariant 2).
- jit/vmap-clean and differentiable w.r.t. `X`/`M` (subgradient at repeated
  singular values documented; `psi` reconditions).

## 5. Status (2026-06-30)

- ✅ **`linalg.orthogonal_procrustes`** shipped (the solver; cuSOLVER-free polar
  factor; matrix-vMF `prior=` additive term; `allow_reflection`).
- ✅ **`register.functional_align{,_fit,_apply}` + `ProMises` method ADT**
  (dense path) — the §6.5 seam + method family.
- ⏳ **Efficient ProMises** (§6) — the whole-brain-tractable method.

## 6. Efficient ProMises (the whole-brain regime) — REQUIRED follow-up

The dense path forms `Xᵀ M` and `R` explicitly: **`O(p²)` memory / `O(p³)` for
the polar**, where `p` = features (voxels). That is fine at **searchlight /
parcel** scale (small `p`) but **intractable for whole-brain** hyperalignment
(`p ~ 10⁴–10⁵`). The headline ProMises use case therefore needs the **efficient
ProMises** algorithm (Andreella et al.), which exploits `n ≪ p` (timepoints ≪
voxels): the data — and the relevant part of `R` — live in an `≤ n`-dimensional
subspace, so the alignment is computed in that small space and the map
represented implicitly (a low-rank factor + a chosen orthogonal completion),
never forming the `p × p` objects.

**This must be derived clean-room from the paper, not assumed** (the correctness
mandate): the subspace reduction, the orthogonal-completion convention on the
`(p − n)`-dim null space, and especially **how the matrix-vMF prior `F` enters
the reduced problem** are the parts to get right. Validation: equivalence to the
dense path on tractable `p` (where both run), on the row-space applications where
the completion is irrelevant; characterise any completion-dependent difference
for general co-transport. Likely shape: an `EfficientProMises` method (or a
`subspace=`/`efficient=` field on `ProMises`) — a method-ADT sibling, so the
public `functional_align` surface is unchanged.

## 7. Cross-references

- **Dependency (shipped):** [`linalg-orthogonal-procrustes`](linalg-orthogonal-procrustes.md).
- **Independent (explicitly *not* a dependency):** the vector-vMF directional
  family [`stats-directional-vmf`](stats-directional-vmf.md) — ProMises uses the
  *matrix* vMF, which never materialises its normaliser.
- **Seam precedent:** [`nimox-differentiable-registration-layer`](nimox-differentiable-registration-layer.md),
  [`nimox-histogram-match-fit-apply`](nimox-histogram-match-fit-apply.md).
- **ADT precedent:** `register._metric.Metric`, `register._model.TransformModel`.
- **Ledger:** [`registration-suite`](registration-suite.md) §9,
  [`hypercoil-examples-migration`](hypercoil-examples-migration.md).
- **Provenance:** `hypercoil-examples/atlas/promises.py` (`empty_promises`);
  legacy module wrapper `atlas/model.py::EmptyPromises` → `nimox`.
