# genred × affinity kernels — coverage & wire-in tracker

*A standing map of every affinity / similarity / kernel-weight function in nitrix, its
streaming decomposition, and where the genred streaming-reduction engine can be wired in —
as a **replacement**, as an **auxiliary**, or **not at all**. Living document; update the
per-site rows as sites are wired.*

Related: the genred filing and its spec; the moonshot ledger's **X-9** (the genred /
sparse-product boundary) and **X-6** (measure before replacing a seam); the k-store spec
that closes the one reduction-side gap (`nitrix-moonshot/00.../docs/genred-kstore.md`).

---

## 1. The framing that makes this tractable

An affinity kernel is a **per-pair formula** `φ(xᵢ, yⱼ)`. genred streams
`reduce_j φ(xᵢ, yⱼ, …)` **without ever forming the `(m, n)` pairwise object**. So two axes
are orthogonal:

- **φ** — the kernel's closed form (a Gaussian, a cosine, an η²). This is the *affinity*.
- **the reduction** — the semiring/store folded over `j` (`Sum`=REAL, `LogSumExp`=LOG,
  `Max`/`Min`=TROPICAL, `Or`=BOOLEAN, `SoftmaxWeightedSum`, `EUCLIDEAN`). This is *what you
  do with* the affinity.

**Every kernel below composes with every reduction.** A Gaussian affinity under REAL is a
diffusion matvec; under LOG it is a log-domain softmax; under TROPICAL min-plus with
`φ = distance` it is nearest-neighbour; under BOOLEAN with a threshold it is adjacency. The
coverage question is therefore only: *does the IR's φ-vocabulary span every kernel's closed
form?* It does — see §2.

### The φ-vocabulary (verified against the IR, not assumed)

- **Unary:** `NEG EXP LOG SQRT SQUARE ABS RECIP SIN COS`
- **Binary:** `ADD SUB MUL DIV MIN MAX SQDIFF ATAN2` + relational (`EQ…GE`, for masks)
- **Structural:** `Contract` (inner product over a FEATURE axis), `Gather` (neighbour-list
  lookup into a tile-resident table)

The two load-bearing ops are `SQDIFF` (fused `(a−b)²` — the squared-distance atom) and
`RECIP`/`DIV` (every normalised or rational kernel). Two forms are *derived, not primitive*:
arbitrary real powers via `xᵅ = exp(α·log x)` (bases here are ≥ 1, so safe), and `tanh` via
`exp`/`recip`. One caveat, not a gap: kernels needing `d = √(d²)` (exponential/Laplacian,
Matérn) hit the `0×∞` reverse-mode NaN at coincident points — the floor must go **inside**
the sqrt (the filing-07 `_clamp`/`_distance` trap); it must live in the reference φ-lowering
so it is not re-discovered per kernel.

---

## 2. Coverage table

Wire-in mode legend:
**R** = *replace* (affinity never materialised; only ever reduced against — genred subsumes it, `O(N)` memory) ·
**A** = *auxiliary* (the affinity **object** is itself desired; genred streams the fold that *fills* or *applies* it, object still stored) ·
**B** = *blocked* (does not fit the pairwise-fold shape, or waits on the k-store).

| Kernel (nitrix site) | φ closed form | Streaming decomposition | Mode |
|---|---|---|---|
| **Linear** `linalg/kernel.py:113` | `⟨x,y⟩` (opt. Mahalanobis) | `Contract` over FEATURE | **R** |
| **Cosine** `linalg/kernel.py:454` | `⟨x,y⟩/(‖x‖‖y‖)` | `Contract` × per-i, per-j `RECIP(√)` payloads | **R** |
| **RBF / Gaussian** `linalg/kernel.py:298` | `exp(−γ‖x−y‖²)` | `EXP∘(SQDIFF-sum)`; the `‖·‖²` identity | **R** |
| **Polynomial** `linalg/kernel.py:369` | `(γ⟨x,y⟩+r)^d` | `Contract` then int power via `SQUARE`/`MUL` | **R** |
| **Sigmoid** `linalg/kernel.py:410` | `tanh(γ⟨x,y⟩+r)` | `tanh` derived via `EXP`/`RECIP` | **R** |
| **SE / Matérn spectral density** `linalg/kernel.py:505` | `(λ²+‖ω‖²)^{−(ν+D/2)}` | arbitrary power via `exp∘log` | **A** (HSGP design) |
| **Exponential / Laplacian** (implied) | `exp(−‖x−y‖/σ)` | `EXP∘NEG∘√(SQDIFF-sum)` — sqrt-at-0 caveat | **R** |
| **Student-t / Cauchy** (t-SNE form) | `1/(1+‖x−y‖²)` | `RECIP(1+SQDIFF-sum)` | **R** |
| **η² (Cohen 2008)** `graph/parcellation.py:73` | `1 − SS_within/SS_total` | **collapses**: `SS_within=½‖a−b‖²`, `SS_total=‖a‖²+‖b‖²−2d·M²`, `M=(μₐ+μ_b)/2` — all per-axis payloads + `DIV` | **A** (boundary map is the object) |
| **Pearson** `graph/parcellation.py:112` | `corr(a,b)` | pre-centre per-i → cosine | **A** |
| **Surface boundary dissimilarity** `graph/parcellation.py:175` | `1 − sim(hᵢ,hⱼ)` over mesh adj. | η²/Pearson φ, reduced by REAL(mean)/TROPICAL_MAX over ELL | **A** ✓ *already a genred-shaped fold* |
| **Symmetric-normalised affinity** `connectopy.py:287` | `D^{−½} A D^{−½}` | 2-pass: degrees, then normalised apply (per-i, per-j degree payloads) | **R** (as matvec) |
| **Diffusion operator (Coifman–Lafon)** `connectopy.py:287` | `D₂^{−½} Kₐ D₂^{−½}`, `Kₐ=K/(dᵅ dᵀᵅ)` | 3-pass composition; each pass streams | **R** (as matvec) |
| **Heat kernel** `connectopy.py:710` | `exp(−tL)` | Chebyshev/scaling matvecs — apply, never form | **R** (as matvec) |
| **Cotangent weights** `sparse/mesh.py:976` | `½(cot α + cot β)` | `cot θ = ⟨e₁,e₂⟩/‖e₁×e₂‖` (no trig); ELL edge-fold | **A** (Gather → reference tier) |
| **Bilateral** `smoothing/bilateral.py:130` | `exp(−½‖M(fᵢ−fⱼ)‖²)` over k-NN | `EXP∘SQUARE` over ELL neighbourhood | **A** |
| **Spherical-geodesic Gaussian** `geometry/sphere.py:241` | `exp(−½(geo/σ)²)` on k-NN | geodesic precomputed as ELL payload; `EXP∘SQUARE` | **A** |
| **BrainSMASH kernels** `stats/inference/brainsmash.py:96` | `exp(−d/b)`, `exp(−½(d/b)²)`, `𝟙[d≤b]` | `EXP`/`Where`; dense `O(n²)` or k-NN | **R/A** |
| **Soft Dice** `metrics/overlap.py:75` | `2Σ(p·t)/(Σp+Σt)` | `Contract` numerator; per-axis sum payloads; `DIV` | **A** |
| **Soft Jaccard** `metrics/overlap.py:130` | `Σ(p·t)/(Σp+Σt−Σ(p·t))` | as Dice | **A** |
| **vMF energy** `stats/directional.py:265` | `κ⟨μ,x⟩` | `Contract`; normaliser is a per-dist. `WithFinalize` scalar | **R** |
| **Watson** `stats/directional.py:640` | `κ⟨μ,x⟩²` | `SQUARE∘Contract` | **R** |
| **Kent / Fisher–Bingham** `stats/directional.py:825,933` | `κ⟨γ₁,x⟩ + Σβⱼ⟨γⱼ,x⟩²` | `Contract`s + `SQUARE` + weighted sum | **R** |
| **Angular affinity** (implied, S²) | `exp(−θ²/σ²)`, `θ=∠(x,y)` | `θ = ATAN2(‖x×y‖, ⟨x,y⟩)` — arccos-free | **R** |
| **InfoNCE / NT-Xent** `metrics/contrastive.py:41` | `⟨ẑₐ,ẑ_b⟩/τ` then softmax | `Contract`/τ under `SoftmaxWeightedSum` | **R** |
| **Kozachenko–Leonenko** `metrics/contrastive.py:236` | 1-NN by max cosine | `Max` with arg (k=1 store — *already present*) | **R** |
| **Coordinate RBF prior (RFF)** `register/_functional.py:381` | `exp(−‖s−s'‖²/2ℓ²)` via `ΦΦᵀ` | already never-dense (random features) | — (already streaming-shaped) |
| **Soft MI (Parzen / Mattes)** `metrics/information.py:227,305` | joint-histogram MI | **not a pairwise fold** — scatter into bins, then reduce over bins | **B** |
| **Correlation ratio** `metrics/information.py:518` | `Σ nₖ(μₖ−μ)²/Σ(m−μ)²` | soft-bin scatter, then bin reduction | **B** |
| **k-NN graph construction** (`ell_from_dense`, kNN sites) | select k nearest per point | needs `ArgKMin` — **specified, not yet built** | **B → A** once k-store lands |
| **Self-tuning / adaptive bandwidth** (implied) | `σᵢ = dist to k-th NN` | `best[…,−1]` of a `KMin` pass — **k-store** | **B → R** once k-store lands |

---

## 3. The three wire-in modes, and why *auxiliary* is the common case

**Replace (R).** The consumer only ever *reduces against* the affinity — it never inspects
or returns the matrix. Spectral embedding, diffusion maps, Laplacian eigenmaps and
connectopy all reach the affinity **only** through matvecs inside an iterative eigensolver
(Lanczos / LOBPCG). There, genred streams `A·v = Σⱼ k(xᵢ,yⱼ) vⱼ` at `O(N)` memory and the
dense `(n,n)` affinity — currently materialised at `connectopy.py:710` (heat kernel) and the
`linalg/kernel.py` builders — is **deleted outright**. This is the pure win, and it is
gated by **X-6**: claim it only with a head-to-head against the incumbent dense build at
target scale.

**Auxiliary (A) — the common case, and the one you flagged.** Often the affinity **object
itself is the deliverable**: the η² **boundary map** is returned and thresholded
(`parcellation.py`); a kernel matrix is handed to a downstream estimator; a stored ELL is
reused across many applies. Here genred does **not** replace the object — it *co-processes*
it:

- *construct*: stream the fold that **fills** each stored entry on a **given** neighbourhood
  (the boundary map is exactly η² over mesh adjacency, reduced by mean/max — already a
  genred-shaped `ell_edge` fold). Note: when *both* operands are sparse, filling the object
  is a sparse×sparse product and belongs to the masked-sparse-product filing, **not** genred
  (X-9).
- *apply*: stream `A·v` while the object is built and kept elsewhere.

The distinguishing test: **is the affinity matrix ever read as a value, or only summed
against?** Read-as-value ⇒ auxiliary (object stays); summed-against-only ⇒ replace.

**Blocked (B).** Two kinds. (i) **Wrong shape** — soft MI and the correlation ratio are
*image-to-image* similarities via a joint histogram (scatter over voxels into a bins×bins
grid, then a reduction over *bins*); this is not `reduce_j φ(xᵢ,yⱼ)` and genred should not be
forced onto it — it is `metrics.joint_histogram` + a bin reduction (also the live **X-4**
site). (ii) **Waiting on the k-store** — k-NN graph construction and k-th-NN bandwidths need
`ArgKMin`, now **specified** (`genred-kstore.md`) but not built; once it lands, both move to
A/R respectively, because `ArgKMin`'s `arg` output *is* the fixed-width ELL neighbour list.

---

## 4. Gaps & their status

| Gap | Impact | Status |
|---|---|---|
| **k-store (`ArgKMin`/`KMin`)** | k-NN graph construction; self-tuning/adaptive bandwidth; any k-selection | **Specified** in `nitrix-moonshot/00.../docs/genred-kstore.md`; not yet built. The one substantive gap. 1-NN already works (`Max` with arg). |
| **Joint-histogram MI** | soft MI, correlation ratio | **Out of scope by shape** — not a pairwise fold. Keep on `metrics.joint_histogram`. |
| **Diffusion normalisation coupling** | `D^{−1}K`, symmetric, α-normalised | **Not a gap** — a 2–3 pass *composition* (degrees, then normalised apply with per-i/per-j degree payloads). State it so callers don't expect one atom. |
| **sqrt-at-0 reverse-mode NaN** | exponential/Laplacian, Matérn | **Caveat** — floor inside the sqrt, in the reference φ-lowering. |

---

## 5. Governance

- **X-9 (the boundary).** genred streams a fold over a **given** domain; it does not
  construct a *data-dependent* one. The k-store is the single exception the static-shape
  model admits — because `k` is a compile-time constant, `ArgKMin` manufactures a **bounded,
  statically-shaped** neighbour list, not a data-dependent one. Constructing the k-NN
  structure and folding over it are two composed atoms.
- **X-6 (the gate).** No wire-in is claimed as a *replacement* without a measured
  head-to-head against the incumbent at target scale. A row moving from A to R in §2 must
  cite the measurement.

---

## 6. Priority wire-in candidates (highest leverage first)

1. **Connectopy / diffusion-embedding matvec** (`connectopy.py:287,710`) — **R**. The heat
   kernel and normalised affinity are built dense today and only ever applied; streaming the
   matvec deletes the `(n,n)` object. Highest fan-out.
2. **η² boundary map** (`parcellation.py:73,175`) — **A**. Already a mean/max fold over mesh
   adjacency; the η² collapse (§2) means it streams with no dense score. The object is kept.
3. **Bilateral / spherical-geodesic smoothing** (`smoothing/bilateral.py:130`,
   `geometry/sphere.py:241`) — **A**. Bounded-neighbourhood `EXP∘SQUARE` folds; natural
   `ell_edge` shape.
4. **BrainSMASH variogram** (`brainsmash.py:96`) — **R/A**. The dense `O(n²)` path is the
   documented bottleneck; a streamed distance-kernel fold removes the `n×n` temporary.
5. **k-NN construction everywhere** — **blocked on the k-store**; build that first (§4), then
   this unblocks the sparsification step that every A-row above depends on.
