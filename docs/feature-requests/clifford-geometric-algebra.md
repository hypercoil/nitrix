# Clifford (geometric) algebra primitives — PROPOSAL for review

> **Status (2026-07-02): PROPOSED — SPEC-review-gated, no code.** A proposal for
> a genuine **geometric-algebra** vocabulary in nitrix (the multivector
> geometric product, rotor exp/log, the sandwich transform, grade projection /
> involutions), distinct from the already-shipped `geometry.algebra` (which is
> *transform* algebra — Lie-group means/geodesics of homogeneous matrices).
> Raised because "Clifford algebra" was named as a candidate; this document
> establishes whether it belongs in nitrix, and if so, its surface and
> boundaries. **It commits nothing — it needs an explicit go-ahead** (the
> `geometry`-suite GS-2 precedent for SPEC-gated new capability).

## 0. Why this doc exists (disambiguation)

Two unrelated things share the word "algebra":

- **`nitrix.geometry.algebra` (SHIPPED).** *Transform* algebra: the Fréchet
  (Karcher) mean, geodesic interpolation, and fusion of homogeneous transforms
  (`SE(n) ⊂` affine) via `matrix_log` / `matrix_exp`. It represents rigid /
  affine motions as **matrices**.
- **Geometric (Clifford) algebra (THIS PROPOSAL).** Represents the same motions
  (and much else) as **multivectors** in a Clifford algebra `Cl(p, q, r)`, with
  the *geometric product* as the fundamental operation and **rotors / motors**
  as the group elements. This is a different representation with a different
  primitive vocabulary, and it is the substrate for **equivariant** geometric
  deep learning (the GATr lineage).

The only geometric-algebra artefact in the deprecated `hypercoil-examples`
(`atlas/gatrefpyg.py`, a GATr reference) was scoped **out** in the migration
ledger — it is torch/PyG and module-shaped. This proposal is *not* that: it is
the pure-function **numerical substrate** under such a layer, in the same
relation as `nitrix.nn` (forward kernels) to downstream NN modules.

## 1. What could belong in nitrix (the substrate cut)

Geometric algebra factors cleanly into (a) irreducible numerical primitives and
(b) downstream network/model machinery. nitrix would own only (a):

| Primitive | What it is | Why nitrix |
|---|---|---|
| `geometric_product(a, b)` | the bilinear multivector product (contraction of the two coefficient vectors with the algebra's Cayley tensor) | the irreducible core kernel; a fixed structure-constant einsum whose efficient, differentiable evaluation every consumer would otherwise re-derive |
| `outer_product` / `inner_product` | wedge (grade-raising) / contraction (grade-lowering) parts | same Cayley tensor, grade-masked |
| `reverse` / `grade_involution` / `conjugate` | the three canonical involutions | sign-flip-by-grade; needed by every downstream norm / inverse |
| `grade_project(x, grade)` | project onto a fixed grade | the graded structure |
| `rotor_exp(bivector)` / `rotor_log(rotor)` | the bivector exponential → rotor (and inverse) — the GA analogue of `matrix_exp`/`log` | **irreducible closed-form numerics** (per signature; e.g. the `Cl(3,0,0)` and PGA `Cl(3,0,1)` rotor exps have stable closed forms that beat a generic matrix exp) |
| `apply_versor(v, x)` | the sandwich `v x v⁻¹` (rotor/motor action) — the equivariant transform | the operation equivariant layers are built on |
| `multivector_norm` | the algebra norm `√⟨x x̃⟩₀` | composition of the above |

The **signature** `Cl(p, q, r)` is static config (the multivector dimension is
`2^(p+q+r)`, static). Proposed initial support: **PGA `Cl(3,0,1)`** (16-dim, 3-D
projective geometry — GATr's algebra, covers points/lines/planes and rigid
motors) and **`Cl(3,0,0)`** (8-dim, 3-D rotations/rotors); generalise later.

## 2. Admission story (SPEC §9)

SPEC §9 admits a new primitive with *irreducible numerical content* **or** a
*named vocabulary family*. Geometric algebra clears both:

1. **Named vocabulary family.** Geometric product, wedge, involutions, grade
   projection, rotor exp/log, and the sandwich are a coherent, textbook-named
   family — discoverable together, exactly the shape (`semiring`, the transform
   `geometry.algebra`, the `stats.directional` family just shipped) nitrix
   already hosts.
2. **Irreducible content.** The Cayley-tensor contraction (built once per
   signature from the metric) and the closed-form rotor `exp`/`log` are genuine
   numerical kernels — not one-liners a consumer should re-derive, and
   fp32/fp64-first (the equivariance guarantees are precision-sensitive).

**Substrate-composition test (the bar for a new subpackage).** The geometric
product is a fixed bilinear einsum against a static Cayley tensor; the rotor
`exp`/`log` compose with the involutions; nothing forks an existing surface. It
either lands as a new subpackage (`nitrix.algebra`, GA-only) or — if kept
tight — as `geometry.clifford` beside the transform algebra. **Open for the
review to decide** (§5).

## 3. Boundaries (what stays out)

- **Equivariant layers stay downstream** (`ilex` / `nimox`): the GATr-style
  equivariant linear map, attention, gated nonlinearities, and the network
  modules are module-shaped (Equinox) and objective-adjacent — the same cut as
  `nitrix.nn` kernels vs downstream blocks. nitrix ships the algebra ops they
  call, not the layers.
- **No I/O, no containers, no training** — as everywhere.
- **Runtime deps stay `jax`/`jaxtyping`/`numpy`.** The Cayley tensor is a static
  array built at import from the signature (numpy), never a runtime `scipy`/
  `clifford` import. The `clifford` (BSD) / `kingdon` / `galgebra` packages and a
  reference GATr implementation are **test-only** oracles.
- **Relationship to `geometry.algebra`.** Complementary, not redundant: rotors /
  motors are the GA representation of the rigid motions that `geometry.algebra`
  handles as matrices. If admitted, nitrix should expose **conversions**
  (rotor ↔ homogeneous matrix) and document when each representation is
  preferred (matrices for classical registration; multivectors for equivariant
  learning and for line/plane geometry the matrix form handles awkwardly).

## 4. Correctness mandate & validation plan

Clean-room from the geometric-algebra literature (Dorst, Fontijne & Mann,
*Geometric Algebra for Computer Science*; Doran & Lasenby), **not** a port of
GATr or `clifford`:

- `geometric_product` associative, distributive; reproduces the Cayley table of
  `clifford`/`kingdon` for the supported signatures (test-only oracle);
  recovers the quaternion product as the even subalgebra of `Cl(3,0,0)`.
- `rotor_exp`/`rotor_log` round-trip (`log(exp(B)) = B`); `apply_versor` with a
  rotor is a rotation (preserves the norm; `det = 1` on the vector grade);
  matches the corresponding `matrix_exp` rotation after the rotor→matrix
  conversion.
- `apply_versor` **equivariance**: `gp(v, gp(a, b)) sandwich = ...` — the
  defining identity `v (a b) ṽ = (v a ṽ)(v b ṽ)` to machine precision.
- Grade projection / involutions match their sign-table definitions; `norm`
  non-negative and zero only at 0.
- `jax.grad` finite through `geometric_product`, `rotor_exp`, `apply_versor`
  (the equivariant-learning use needs differentiability); jit/vmap-clean;
  fp32 and fp64.
- Golden entry per `(op, dtype)` (SPEC §8).

## 5. Open questions for the review

1. **Admit at all?** Is in-tree GA a capability nitrix wants, or does it stay
   downstream indefinitely (GATr fully in `ilex`)? The equivariant-learning
   demand is real but currently hypothetical for this ecosystem — is there a
   consumer lined up?
2. **Home & scope:** new `nitrix.algebra` subpackage vs `geometry.clifford`;
   which signatures at v1 (PGA `Cl(3,0,1)` only, or `+ Cl(3,0,0)`)?
3. **Representation:** dense `2^n` coefficient vectors (simplest, natural for the
   Cayley einsum) vs a graded/blade-sparse layout (leaner for high-`n`). Dense
   is proposed for v1.
4. **Overlap policy** with `geometry.algebra`: ship rotor↔matrix conversions, or
   keep the two representations independent?

## 6. Cross-references

- Disambiguated from: `nitrix.geometry.algebra` (transform algebra, SHIPPED).
- Substrate-vs-layer precedent: `nitrix.nn` (forward kernels) ↔ downstream
  modules; the `stats.directional` family (just shipped) as a "named vocabulary"
  admission precedent; the geometry-suite **GS-2** SPEC-gated-capability
  precedent.
- Provenance / out-of-scope: `hypercoil-examples/atlas/gatrefpyg.py` (GATr, torch/
  PyG) — the *downstream* consumer, explicitly not migrated
  ([`hypercoil-examples-migration`](hypercoil-examples-migration.md) §3.3).
- Admission gate: SPEC §9; concern boundaries SPEC §5 / §6.
