# Subspace geometry & orthogonal alignment in `nitrix.linalg`

> **Status (2026-06-30): `orthogonal_procrustes` SHIPPED; subspace angles
> PROPOSED.** A small coherent linalg vocabulary family of SVD-of-cross-product
> primitives, migrated from the deprecated `hypercoil-examples` repo. **Lead item
> `orthogonal_procrustes` is the dependency of
> [`register-functional-alignment`](register-functional-alignment.md)** (the
> matrix-vMF MAP reduces to a Procrustes solve with an additive prior term) and
> is now shipped; the subspace-angle siblings come from `atlas/totalangle.py`.
> Pure differentiable JAX; the only real in-tree dependency
> (`recondition_eigenspaces`) already ships in `nitrix.linalg`.
>
> **Correctness mandate — theory over the legacy code.** Clean-room from the
> linear-algebra literature (Schönemann; Björck–Golub canonical angles; scipy's
> `orth`/`subspace_angles` as test-only oracles), **not** a port. The legacy
> `atlas/totalangle.py` is a reference point, not a spec. Points the kernels must
> get right from first principles rather than copy: (i) the `arcsin`(small)/
> `arccos`(large) **split point and the ±1 boundary sub-derivatives** — the
> source's `σ² ≥ ½` switch must match the Björck–Golub stable formulation and
> stay finite-gradient at exactly 0 and π/2, the regimes that trivially break a
> naive `arccos`; (ii) **reflection handling** in Procrustes (`det < 0` →
> proper-rotation flip vs the full-orthogonal solution) decided by the theory and
> exposed, not inherited from the raw SVD; (iii) the **`rcond` default** for
> `image_basis` matched to a justified rule (`max(m,n)·eps`, scipy's `orth`), not
> an arbitrary constant; (iv) behaviour at **repeated singular values** (the
> SVD-gradient degeneracy) handled via `recondition_eigenspaces` and documented.
> Each becomes an oracle / property test.

## 1. Items

### P0 — `orthogonal_procrustes(a, b, *, prior=None, allow_reflection=True, psi=0., key=None)` ✅ SHIPPED

The orthogonal map best mapping `a` onto `b`: `R = U Vᵀ`, `U Σ Vᵀ = svd(aᵀb +
prior)`. Computed as the **orthogonal polar factor** `R = C (CᵀC)^{-1/2}` of the
cross-product via a single `safe_eigh` of the small `(p,p)` Gram — **cuSOLVER-
free** (never `jnp.linalg.svd/qr`, broken on the affected GPU stacks like
`decompose`/`pca`). `R` is invariant to the eigenvector gauge, so repeated
singular values leave the forward result well-defined; `psi` reconditions the
eigh-VJP for the gradient at degeneracy.

- **`prior`** — the **additive natural-parameter term** that turns the plain
  least-squares Procrustes into a MAP under a **matrix von Mises–Fisher** prior
  on `r` (the algebraic crux of ProMises). nitrix never materialises the
  matrix-vMF normaliser.
- **`allow_reflection=False`** — the proper-rotation (`SO`) Kabsch solution,
  flipping the least-significant singular direction when `det C < 0`. The sign is
  computed by a **pure-XLA partial-pivot determinant** (`_det_sign`;
  `jnp.linalg.det` also lowers to the dead cuSOLVER pool).
- Validated vs `scipy.linalg.orthogonal_procrustes` (1e-14), planted-rotation
  recovery, orthogonality/optimality, the prior MAP pull, batch/jit/vmap/grad,
  float32. 16 tests; op-matrix catalogued.

### P1 — `image_basis(x, *, rcond=None, rank=None) -> q` (PROPOSED)

A numerically-ranked **column-space (range) basis** via SVD with an `rcond`
singular-value tolerance — a differentiable analogue of `scipy.linalg.orth`
(scipy is the test-only oracle, not a runtime dep). Provenance:
`atlas/totalangle.py::image_basis`.

### P1 — `subspace_angles(x, y=None, *, rcond=None, rank=None, recondition=0.0) -> angles` (PROPOSED)

**Principal / canonical (Grassmann) angles** between two subspaces, computed
differentiably with the numerically-stable split — `arcsin` of the
residual-block singular values for small angles, `arccos` of the cross-Gram
singular values for large angles (the `σ² ≥ ½` switch) — with explicit ±1
boundary handling so the `arcsin'`/`arccos'` gradients stay finite, and optional
eigenspace reconditioning (`recondition`) to keep the SVD gradient defined under
degeneracy. Provenance: `atlas/totalangle.py::inner_angles`. **Its only real
dependency, `recondition_eigenspaces`, already lives in `nitrix.linalg`.**

## 2. Why (and why a family)

- **Unblocks functional alignment.** P0 is the closed-form solver under
  [`register-functional-alignment`](register-functional-alignment.md); landing it
  first (tiny, oracle-testable, boundary-question-free) de-risked that recipe.
- **Coherent vocabulary (SPEC §9 admission).** All three are SVD-of-a-cross-
  product subspace-geometry tools (Procrustes alignment, range basis, principal
  angles) — a named family, discoverable together, not piecemeal one-liners.
- **Differentiable Grassmann geometry is genuinely irreducible.** The
  `arcsin`/`arccos` stable split + boundary-derivative handling in
  `subspace_angles` is exactly the numerical content nitrix exists to own (and
  not have every consumer re-derive).

## 3. Separation of concerns / boundaries

- Pure `jax`/`jaxtyping`/`numpy`; **drop** the `hypercoil.engine.Tensor` alias on
  migration. `scipy.linalg.{orth,subspace_angles,orthogonal_procrustes}` are
  **test-only** oracles.
- No module state, no scalarisation — pure array→array kernels for
  `nitrix.linalg`.

## 4. Acceptance

- ✅ `orthogonal_procrustes` matches `scipy.linalg.orthogonal_procrustes`;
  `allow_reflection=False` is the Kabsch proper rotation; `prior` recovers the
  ProMises regularised solve.
- `image_basis` matches `scipy.linalg.orth` (rank + span) within tolerance.
- `subspace_angles` matches `scipy.linalg.subspace_angles` across the small-/
  large-angle regimes incl. the 0 and π/2 boundaries; gradients finite.
- All jit/vmap-clean and golden-tested per `(op, dtype)`.

## 5. Cross-references

- **Consumer:** [`register-functional-alignment`](register-functional-alignment.md)
  (P0 is its solver).
- **Existing in-tree dep:** `linalg.recondition_eigenspaces`
  (`linalg/matrix.py`) — `subspace_angles`' degeneracy guard.
- **Catalogue sibling:** [`matrix-functions`](matrix-functions.md) (§12.2).
- **Provenance:** `hypercoil-examples/atlas/totalangle.py` (`image_basis`,
  `inner_angles`) + `atlas/promises.py` (the Procrustes solve inside
  `empty_promises`).
