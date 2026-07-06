# Functional alignment (Procrustes / ProMises) in `nitrix.register`

> **Status (2026-07-02): SHIPPED (dense + efficient).** Migration target from the
> deprecated `hypercoil-examples` staging repo (`atlas/promises.py`, the "empty
> ProMises" step). Adds **alignment in representation space** to `nitrix.register`
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

## 5. Status (2026-07-02)

- ✅ **`linalg.orthogonal_procrustes`** shipped (the solver; cuSOLVER-free polar
  factor; matrix-vMF `prior=` additive term; `allow_reflection`).
- ✅ **`register.functional_align{,_fit,_apply}` + `ProMises` method ADT**
  (dense path) — the §6.5 seam + method family.
- ✅ **`EfficientProMises` + the fitted-map ADT** (`FunctionalAlignment`
  Protocol; `DenseAlignment` / `SubspaceAlignment`) — the whole-brain subspace
  method (§6).

## 6. Efficient ProMises (the whole-brain regime) — ✅ SHIPPED (2026-07-02)

The dense path forms `Xᵀ M` and `R` explicitly: **`O(p²)` memory / `O(p³)` for
the polar**, where `p` = features (voxels) — fine at **searchlight / parcel**
scale but **intractable for whole-brain** hyperalignment (`p ~ 10⁴–10⁵`).
Because `Xᵀ M` has rank `≤ n`, the alignment lives in an `≤ n`-dimensional
subspace; `EfficientProMises` computes it there and represents the map
implicitly, never forming a `p × p` object on the plain path (`O(p n²)` time /
`O(p n)` memory).

**Derived clean-room from the paper** (Andreella & Finos 2022, Theorem 3 /
Lemma 5), as the mandate requires — and the legacy code was *not* followed where
it disagrees with the theory:

- **Subspace reduction (Theorem 3).** Per-matrix thin SVDs `X = Lₓ Sₓ Qₓᵀ`,
  `M = L_m S_m Q_mᵀ` give semi-orthogonal `Qₓ, Q_m` (`(p, l)`, `l ≤ n`). The
  reduced solve is `R* = polar((X Qₓ)ᵀ(M Q_m) + k F*)`, an honest `(l, l)`
  orthogonal rotation. Bases are computed **cuSOLVER-free** from the small
  `(n, n)` Gram via `safe_eigh` (as in `orthogonal_procrustes`), never a `(p, p)`
  eigendecomposition.
- **Prior projection (Lemma 5).** The matrix-vMF prior enters *only* as
  `F* = Qₓᵀ F Q_m`. `prior=None` stays fully `O(p n²)`; a dense `(p, p)` prior
  costs `O(p² l)` to project.
- **Coordinate-kernel spatial prior (Lemma 5, whole-brain) — ✅ SHIPPED
  (2026-07-06).** `CoordinateKernelPrior(coords, lengthscale, key, n_features)` +
  `EfficientProMises(spatial_prior=…)`. The anatomical location prior — an RBF
  kernel `K` over locus coordinates biasing nearby loci to align — is built at
  whole-brain scale by **random Fourier features**: `K ≈ Φ Φᵀ`, so `F* ≈ (Qₓᵀ Φ)
  (Φᵀ Q_m)` in `O(p·r·l)`, **never forming the `(p, p)` kernel**. Applied *inside*
  `fit`, where the row-space bases `Qₓ / Q_m` live (resolving the basis
  circularity — `F*` can't be built before the fit computes the bases). The kernel
  *choice* (RBF, lengthscale) is the modelling knob; the irreducible content is
  the never-materialise-`(p,p)` subspace projection. Validated: `F*_RFF → Qₓᵀ K
  Q_m` as `O(r^{-1/2})`; the whole-brain path matches the dense-`(p,p)`-prior
  projection to `< 5e-3` at high `n_features`.
- **Reconstruction convention — the theory-vs-legacy fork.** The map is
  `R = Qₓ R* Q_mᵀ` (source basis in, **reference** basis out), applied as
  `data @ Qₓ @ R* @ Q_mᵀ`. The legacy `EfficientProMises` lifts back with the
  *source* basis on both sides (`Qₓ R* Qₓᵀ`), which keeps the result in the
  source's row space and **does not reproduce the dense ProMises MAP**. nitrix
  follows the theory (the MAP); the deviation is documented in-code.
- **Map ADT.** The fitted map is now itself an ADT — a `FunctionalAlignment`
  Protocol with `DenseAlignment` (explicit `(p, p)`) and `SubspaceAlignment`
  (implicit `(Qₓ, Q_m, R*)`) implementers — so `apply` dispatches without the
  subspace method ever materialising `R`. `functional_align` surface unchanged.

**Validation.** (i) lossless reduction (`n_components` ≥ row rank, `prior=None`)
⇒ `EfficientProMises ≡ ProMises` on the aligned data; (ii) the subspace map
reproduces the dense MAP `X · scipy.orthogonal_procrustes(X, M)` to `1e-7`;
(iii) planted in-subspace rotation recovered to machine precision;
(iv) semi-orthogonal bases, orthogonal `R*`, `n_components` truncation, jit /
vmap / grad. **External oracle:** the authors' actual `alignProMises` R routines
(base-R LAPACK `svd`) — nitrix dense matches `GPASub` to `1e-12` (incl. the
matrix-vMF prior), nitrix efficient matches the dense MAP to `9e-15`, and the
legacy source-basis reconstruction is confirmed to deviate (‖Δ‖ ≈ 4.9).

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
