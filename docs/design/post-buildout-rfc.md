# RFC — post-buildout: validation, rebuild, textbook; and the seam inventory that gates them

> **TL;DR.** After the moonshot filings land there are three tasks: **validate** against SOTA on
> real scientific objectives, **rebuild** nitrix from foundations by coalescing the shipped suites
> with the new kernels, and write a **textbook** that motivates every kernel for an audience with
> linear algebra but not numerics. They are not three sequential projects — each is the strongest
> available audit of the other two, and the correct order is **book spine → rebuild → book body**,
> with validation running *continuously* rather than as a phase. The rebuild's real content is not
> "add the new kernels": it is to **hoist the abstractions that thirty filings independently
> rediscovered**, and the prize is a **composable certificate algebra** that no library in this
> field has. This RFC records the argument and specifies the **seam inventory** — the concrete,
> evidence-based deliverable that must exist *before* any rebuild begins.

**Status:** RFC. **No code.** Parked deliberately: the gating risk is that several filings still
carry open blockers, and foundations must not be extracted from kernels that have not survived
review (§5). Return here when the gate in §6.2 is met.

---

## 1. The strategic frame: three views of one object

The instinct is to sequence: validate → rebuild → write. That is wrong in one important place.

**The textbook is the best architecture review available.** If a kernel cannot be motivated in a
chapter without hand-waving, it is in the wrong place or should not exist. A table of contents *is*
a proposed subpackage layout, and when the two disagree it is usually not the book that is wrong.
Code review asks *"is this correct?"*; pedagogy asks *"why does this belong here?"* — and only the
second question catches organisational defects. Writing the **spine** of the book costs weeks and
is the cheapest high-leverage audit we can buy.

**Validation is the only thing that says which kernels are load-bearing.** Rebuild first and the
foundations get organised around the kernels we *believe* matter rather than the ones that do.

Hence the order: **book spine → rebuild → book body**, validation continuous alongside.

### 1.1 One corpus, not three

The validation corpus, the golden/regression corpus, and the textbook's figures should be **the
same artefact**: one curated set of instances, each carrying either a known answer or a known
pathology, which simultaneously (a) gates the library in CI, (b) demonstrates where the incumbent
tools are silently wrong, and (c) illustrates the text.

Building it three times — once for CI, once for the paper, once for the book — is the default
failure mode and is pure waste. It is also the mechanism that makes the rebuild survivable (§3.3).
**Design it once, as a first-class deliverable with its own provenance and versioning.**

---

## 2. Validation — compete on certificates, not on metrics

The obvious framing is a benchmark table against the incumbent toolboxes. That framing loses, or
wins so narrowly that nobody switches. Three reasons:

- For most real objectives **there is no ground truth**.
- **Wall-clock at unmatched accuracy is meaningless.** We already learned this in the registration
  campaign: the fair comparison is *iso-accuracy*, not raw time.
- A 0.02 improvement in an overlap metric persuades no one.

**The moat is that no baseline can certify anything.** Every filing was built around a fail-closed
flag: a duality gap, strict complementarity, a genus certificate, a positive-Jacobian guarantee, a
spectral bulk-fit statistic, a structural/practical identifiability separation, a global-optimality
certificate. This is a category of evidence the incumbents **structurally cannot produce**.

So the campaign is not *"our number is better"*. It is:

> **Build an adversarial corpus on which the standard tools are silently wrong, and show that
> nitrix is either right or refuses to answer.**

That is a stronger scientific claim, it is publishable on its own terms, and — decisively — **the
adversarial reviews of the filings have already been finding these instances.** The corpus is
half-built and we have not been calling it that. Already on the books: correlated noise breaking
the Marchenko–Pastur bulk (filing 14 G1); cycle-skipping in narrowband lag estimation (30);
`det ∇f > 0` not implying global bijectivity on a closed surface; the non-physical residues of
unconstrained spectral-analysis perfusion deconvolution; geodesic tractography's intrinsic shortcut
pathology; the moment-map normaliser that is 30–38 % off.

### 2.1 Evidence hierarchy

"Validate on real objectives" hides a lot. Be explicit about what each kind of evidence can and
cannot support:

| evidence | supports | limitation |
|---|---|---|
| synthetic with known truth | correctness | necessary, **not** sufficient — simulation gap |
| **planted truth on real data** | correctness *under real statistics* | **the strongest bridge, and badly underused in this field** |
| test–retest reliability | stability | **necessary but not sufficient** — a method returning a constant is perfectly reliable |
| predictive validity | usefulness | circularity risk |
| **certificates** | *what the answer is worth* | **the incumbents cannot produce it at all** |

Lead with the last two rows.

### 2.2 Existence proofs

Include at least one objective that is **impossible** without end-to-end differentiability —
jointly inverting a generative model across stages that are currently run sequentially, for
instance. That is not a benchmark; it is a capability the field does not have, and it cannot be
argued with.

### 2.3 Placement

Real-data validation lives **outside** `nitrix` (correctness is nitrix's concern; wall-clock and
real-data parity are the sibling perf-bench suite's, per `CLAUDE.md`). Do not pollute the library.

---

## 3. The rebuild — organise by seam, not by subpackage

### 3.1 Why it is a rebuild and not an extension

The honest argument is already written down across the filing addenda, and **X-1 is the template**
(see the cross-filing table in `nitrix-moonshot-ledger.md`). The filings **independently
rediscovered the same abstractions**, and the current layout — fifteen subpackages organised by
*numerical kind* — has nowhere to put them:

- **The solution map with an implicit adjoint** is re-derived in at least six places: `linalg`
  Krylov solvers, `numerics.fixed_point_solve` / `implicit_minimize` / `implicit_least_squares`,
  `register`'s level solver, filing 15's nonsmooth solver, 19's Fredholm inversion, 30's lifted
  problem. **This is one concept.**
- **The spectral function** `X ↦ X·g(XᴴX)` is one object appearing in 02, 14, and
  `matrix_function` — which is precisely what X-1 says.
- **Streaming reduction over an implicit `n×m` object** underpins 25, 07, 29, 30. `genred` is the
  substrate, but each filing rolls its own.
- **Certificates** are invented ad hoc by *every single filing*.

That last one is the prize.

### 3.2 The organising idea: certificates must compose

> **A pipeline's guarantee is the *meet* of its stages' guarantees — and today nobody, us
> included, has a type that expresses this.**

A real workflow chains five to ten primitives: denoise (bulk-fit flag) → register (Jacobian
certificate) → estimate a field (duality gap) → fit a model (identifiability flag). Right now
there is **no principled statement of what the end-to-end claim is**, and an uncertified stage
degrades the chain *silently*.

The natural structure is a lattice: a certificate ADT with fail-closed semantics, a composition
rule (meet), and the property that an **uncertified stage poisons the chain visibly**. Build that
and nitrix becomes the only library in the field that can tell a scientist *what its output is
worth*. Every adversarial review across thirty filings has been circling this without naming it.

### 3.3 Two constraints that make a foundations rebuild survivable

1. **The golden corpus is the invariant.** Internals may be rewritten freely; a golden that *moves*
   is a contract change requiring justification. This is the safety net — and it is why §1.1
   matters so much.
2. **Strangle, do not bulldoze.** Build the new seams alongside the old, migrate a subpackage at a
   time, delete the old. A rebuild that breaks everything at once is where projects die.

---

## 4. The textbook — teach seven ideas, not thirty topics

Audience: neuroimagers proficient in linear algebra, not in numerics. The failure mode is a
kernel-by-kernel cookbook. The spine should be the ideas that **recur**, because recurrence is what
builds intuition:

1. **Everything is a solution map, and you can differentiate it without differentiating the
   solver.** The implicit function theorem is the highest-leverage idea in the library and exactly
   what this audience lacks.
2. **Ill-posedness and regularisation** — why the naïve answer is garbage; what a prior buys and
   what it costs.
3. **Conditioning and certificates** — the difference between *wrong* and *uncertain*, and why a
   method that refuses beats one that guesses.
4. **Gradients do not flow through `argmax`** — and what to do about it (relaxation, lifting,
   implicit rules).
5. **Spectra** — eigenvalues as the universal solvent, degeneracy as the universal hazard.
6. **Structure buys scale** — low rank, sparsity, band-limit, separability. The time–bandwidth
   argument that says "the answer is rank seven" (filing 30) is the perfect worked example.
7. **The data is not in a vector space** — SPD matrices, spheres, diffeomorphisms, simplices.

Each kernel is then a **case study of one or two ideas**, not a topic in its own right.

**On rigour:** be *rigorous about the statement and honest about the proof*. State theorems
precisely; prove what is provable in two lines; otherwise give the **reason** it is true (a picture,
the 1-D case, a numerical demonstration) and cite where the proof lives. **Never fake rigour.** This
audience are scientists: they will forgive "here is the precise claim and here is where it is
proved"; they will not forgive a hand-wave dressed as an argument.

**Two things make it uniquely good, and cheap for us specifically:**

- Every chapter carries a **"how this fails"** section — and **the adversarial reviews already wrote
  them.** Thirty filings' worth of catalogued, verified failure modes is content nobody else has.
- The book is **executable**: every figure regenerated from nitrix. The book then tests the API (if
  the example does not write cleanly, the API is wrong) and the library keeps the book from rotting.

---

## 5. The gating risk

**The filings are not done.** Several carry open blockers — 14 has five, 01 and 07 have scaling
blockers, 17's certificate is still unreachable, 05 is mid-reconciliation. **Foundations must not be
extracted from kernels that have not survived review.** X-1 is the warning shot: two filings landing
independently, each with a defensible local answer, would have put the *weaker* adjoint into the
*lower-level, more widely-consumed* seam, where every future spectral consumer would inherit it. The
damage there is done by the **order of folding**, which no single-filing review can see.

This is the whole reason the seam inventory exists, and the reason it is gated.

---

## 6. The deliverable: the seam inventory

### 6.1 What it is

An **evidence-based enumeration of every abstraction independently rediscovered** across the shipped
surface and the reviewed filings — with, for each: the sites that rediscovered it, how their answers
*diverge*, which divergences are **defects** and which are **principled**, the proposed single form,
and the migration cost.

It is simultaneously:
- the **rebuild's design input** (the seams *are* the new foundations),
- the **textbook's chapter spine** (a recurring abstraction is exactly a recurring *idea*),
- and the **evidence** that this is a rebuild and not an extension.

### 6.2 Gate — do not start before this holds

- Every filing intended as a foundation has **cleared adversarial review with no open blocker**.
  (Filings still blocked may be *catalogued* as evidence, but must not *define* a seam.)
- The cross-filing constraints table has been swept for further X-1-class collisions.

### 6.3 Evidence standard (the thing that keeps this honest)

> **A seam is admitted only on ≥ 2 independent rediscovery sites, each cited by file + symbol.**

No speculative seams. No abstraction admitted because it is elegant. This standard is what
distinguishes a seam inventory from architecture astronautics, and it is non-negotiable: the
rediscoveries are *empirical facts about what the problem domain forced*, and that is precisely why
they are trustworthy foundations.

### 6.4 Method

- **S0 — freeze the input set.** Enumerate the shipped surface (15 subpackages) and the filings that
  passed §6.2. Record what is excluded and why.
- **S1 — evidence sweep, one candidate seam at a time.** For each candidate, find every site that
  implements it, cited by file + symbol. **Serial, not parallel** — this is a low-memory box and a
  fan-out of agents can take the machine down. **No code execution**; static reading only.
- **S2 — classify.** Each candidate is a **true seam** (same concept, N implementations), a
  **coincidence** (superficial similarity, different mathematics), or **vocabulary** (a naming
  family, not an abstraction). Only the first survives.
- **S3 — per-seam dossier.** The N current implementations; the divergences between them; for each
  divergence, *defect or principled?*; the proposed single form; the consumers; the migration cost.
- **S4 — the collision matrix.** Which *pairs* of filings would land conflicting answers to the same
  question if folded independently. **X-1 is instance #1**; the matrix asks what else is out there.
- **S5 — output.** A **seam register** plus a **rebuild dependency order** (which seams must land
  before which, since a seam consumed by another cannot follow it).

### 6.5 Candidate seams to seed the sweep

Not the answer — the *starting hypotheses*, each already carrying ≥ 1 site from this discussion and
requiring a second to be admitted:

| candidate | seed evidence |
|---|---|
| **Solution map + implicit adjoint** | `linalg` Krylov; `numerics.fixed_point_solve` / `implicit_minimize` / `implicit_least_squares`; `register`'s level solver; filings 15, 19, 30 |
| **Certificate ADT + composition (meet)** | invented ad hoc by *every* filing; §3.2 |
| **Streaming reduction over an implicit `n×m` object** | `genred` / `semiring`; filings 25, 07, 29, 30 |
| **Spectral function `X·g(XᴴX)` + its adjoint** | filings 02, 14; `matrix_function` — **X-1** |
| **fit/apply estimator seam** | SPEC §6.5 — *already exists*; audit for drift |
| **Divergent-recipe registry** | `_divergent_ops`; filing 00's AST anti-bypass guard; filing 02's **per-site** canonicals |
| **Fail-closed flag discipline** | 15's `trustworthy`; 14's absent bulk-fit; 30's `certified` |
| **Memory-gate discipline (two-sided)** | filing 00's `grad_max > operand`; filing 14's missing `grad` witness |
| **Iteration-cap / non-convergence surfacing** | filing 14 G4; filing 02 G4 — *the same defect, twice, independently* |

The last row is worth noting: **two filings independently failed to surface non-convergence.** That
is not a coincidence — it is a seam announcing itself.

### 6.6 Non-goals

- **Not the rebuild.** The inventory is the *design input*; it writes no code and moves no file.
- **Not a SPEC rewrite.** The SPEC follows the seams, not the other way round.
- **Not a kernel triage.** Whether a kernel is *good* is the filings' reviews; whether it is
  *duplicated* is this.

---

## 7. Provenance

Recorded 2026-07-11/12 from the post-buildout strategy discussion, while the moonshot filings are
still in implementation. Parked deliberately (§5). Related: `nitrix-moonshot-ledger.md`
(cross-filing reconciliation constraints — **X-1**), `nitrix-moonshot-round-2-candidates.md`
(problem/kernel ledger, P1–P10 / KA–K15), and the per-filing `ADDENDUM.md` files in the
`nitrix-moonshot` workspace, which are the raw evidence for §6.
