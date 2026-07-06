# Subspace geometry & orthogonal alignment in `nitrix.linalg`

> **Status (2026-07-02): ALL SHIPPED (`orthogonal_procrustes`, `image_basis`,
> `subspace_angles`).** A small coherent linalg vocabulary family of
> SVD-of-cross-product primitives, migrated from the deprecated
> `hypercoil-examples` repo. **Lead item `orthogonal_procrustes` is the
> dependency of
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

### P1 — `image_basis(x, *, rcond=None, rank=None) -> q` ✅ SHIPPED (2026-07-02)

A numerically-ranked **column-space (range) basis** — the differentiable,
cuSOLVER-free analogue of `scipy.linalg.orth` (test-only oracle). The leading
left singular vectors via the smaller-Gram `safe_eigh` (never `svd`/`qr`).
`rank` (static) is the jit-clean path; `rank=None` infers the numerical rank
from `rcond` **eagerly** (data-dependent shape). **Correctness note the theory
forced:** the rank threshold is applied to the *eigenvalues* `s²` (accurate to
~`eps`), not the singular values `s` — the Gram's null floor sits at ~`eps` on
`s²` but only ~`√eps` on `s = √(s²)`, so an `s`-scale cutoff over-counts the
rank. Validated vs `scipy.linalg.orth` (rank + projector) across tall/wide/
square and a planted rank-deficient case.

### P1 — `subspace_angles(x, y) -> angles` ✅ SHIPPED (2026-07-02)

**Principal / canonical (Grassmann) angles** between two subspaces (descending,
matching `scipy.linalg.subspace_angles`), via the Knyazev-Argentati (2002)
stable split — `arccos(σ)` for `σ² < ½` (large angle), `arcsin` of the residual
singular values for `σ² ≥ ½` (small angle) — with a **double-`where`** feeding
each branch boundary-safe inputs so the `arcsin'`/`arccos'` gradients stay
finite at exactly `0` and `π/2`. Validated vs scipy to machine precision across
the small/large regimes and both boundaries (incl. ~`8e-17` on small angles,
where a naive `arccos` loses half its digits).

**Stability decision (user, 2026-07-02): stable even for orthonormal inputs.**
An `eigh`-based orthonormalisation (`image_basis`) NaNs the gradient when the
input Gram is degenerate (`XᵀX = I` for orthonormal columns — a repeated
spectrum whose eigenvector VJP blows up). `subspace_angles` therefore forms its
bases by a matmul-only **Löwdin orthonormalisation** `Q = X (XᵀX)^{-1/2}`
(Newton-Schulz inverse-sqrt): deterministic, gauge-fixed, and gradient-finite
even at `XᵀX = I`, accurate for condition numbers up to ~`1e5` (pre-truncate a
near-rank-deficient basis with `image_basis`). Combined with the
*eigenvalues-only* angle computation (finite VJP at repeated/degenerate
principal angles), the op is differentiably stable end-to-end. `nimox`'s
`sympower` was ruled out here: its `eigh`-reconstruction VJP is *not* stable at a
repeated spectrum (needs random reconditioning + a key). `rcond`/`rank`/
`recondition` params were dropped as unnecessary under this design (full-column-
rank assumption; compose with `image_basis` for rank control).

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
- **Catalogue sibling:** [`matrix-functions`](../matrix-functions.md) (§12.2).
- **Provenance:** `hypercoil-examples/atlas/totalangle.py` (`image_basis`,
  `inner_angles`) + `atlas/promises.py` (the Procrustes solve inside
  `empty_promises`).
