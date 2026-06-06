# eigsolve: a dedicated extremal-eigendecomposition dispatcher

> **Status (2026-06-06): implemented.**  Both goals shipped: (1) the
> solver dispatch is lifted out of ``graph.connectopy`` into
> ``nitrix.linalg._eigsolve`` (``eigsolve_top_k`` + the frozen
> ``SolverSpec`` + three per-format ``custom_vjp`` entries), and (2)
> ELL / SectionedELL coverage now extends to ``shift_invert`` and
> ``poly``.  As predicted, the two were the **same** refactor: factoring
> the solver *forward* (method) apart from the gradient *backward*
> (operand format) made the sparse cells fall out for free.  The
> ``_lobpcg_diff`` compatibility shim has been removed; ``connectopy``
> and the direct-kernel tests import from ``nitrix.linalg`` directly.
> The sections below are the as-built record (the original plan, which
> this followed closely, is retained for its rationale).

## The problem

The extremal-eigendecomposition solver suite grew organically inside
``graph.connectopy``.  Today:

- **Dispatch is tangled into two domain functions.**  ``laplacian_eigenmap``
  and ``diffusion_embedding`` each carry a near-identical
  ``if solver == 'eigh' / 'lobpcg' / 'shift_invert'`` ladder
  (``connectopy.py:411`` and ``:542``), a ``preconditioner → lobpcg``
  reroute, ``_auto_solver``, and three wrappers (``_lobpcg_top_k``,
  ``_shift_invert_top_k``, ``_poly_top_k``) that each re-implement the
  **same** device routing (``_source_device`` / ``_solver_device`` /
  ``_device_put_graph`` / move-back), X0 initialisation, skip-trivial,
  transform, and sort.  Roughly six copies of the same plumbing.
- **Two solvers are stuck at dense.**  ``shift_invert`` and ``poly`` are
  asserted dense-only (``connectopy.py:780``, ``:823``, with the runtime
  guards at ``:785`` and ``:829``), even though nothing in their math
  needs a dense operand.  ``_poly_top_k``'s own docstring names the gap:
  *"Dense only for now (the differentiable sparse-pattern backward is the
  ELL follow-up)."*

This belongs in ``nitrix.linalg``, not ``graph``.  Extremal
eigendecomposition is a pure-numeric primitive with no connectome
semantics (SPEC §1: nitrix is the substrate; domain logic lives in
consumers).  ``linalg._solver.safe_eigh`` already lives here; the LOBPCG
kernels sit in ``graph/`` only by historical accident.  The salient
consumer today is connectopy, but the same facility serves community
detection (``graph.community``'s modularity operator, see
``community.py:142``) and future PCA-style consumers (aCompCor over a
nuisance Gram).

## The core observation

The **backward is solver-independent**.  Every solver's gradient is
``_subspace_vjp_kernel`` (Hellmann–Feynman eigenvalue term + F-matrix
subspace correction) followed by a projection that depends only on the
*operand format* — the dense ``V K Vᵀ``, the ELL gather-einsum onto the
sparsity pattern, or the SectionedELL per-section gather.  The shipped
kernels say so directly: shift-invert and poly both note *"the backward
is the same implicit VJP … solver-independent, depending only on the
converged pair"* (``_lobpcg_diff.py:267``, ``:366``).  See
[`lobpcg-implicit-vjp.md`](lobpcg-implicit-vjp.md) for the derivation.

The **forward is format-independent**.  Shift-invert's
``shifted(y) = (1−σ)y − M@y`` and poly's ``(M+σI)^d`` filter are written
purely against a matvec ``y ↦ M@y``.  The only thing that varies by
format is *how the matvec is built* (``M@X`` dense;
``semiring_ell_matmul`` for ELL; per-bucket for SectionedELL) — and those
builders already exist (``laplacian._ell_matvec``, ``_sectioned_matvec``).

So the two axes are orthogonal:

```
forward  =  method_transform( apply )                      # ⟂ format
backward =  format_projection( _subspace_vjp_kernel(...) )  # ⟂ method
```

The current code is factored the wrong way — each ``(method × format)``
pair re-derives both halves, so only the hand-written cells exist and the
dense column is the only full one:

| method          | dense | ELL | SectionedELL | differentiable |
|-----------------|:-----:|:---:|:------------:|:--------------:|
| ``eigh``        |  ✓    |  —  |      —       | native eigh VJP |
| ``lobpcg``      |  ✓    |  ✓  |      ✓       | implicit VJP   |
| ``shift_invert``|  ✓    |  ✓  |      ✓       | implicit VJP   |
| ``poly``        |  ✓    |  ✓  |      ✓       | implicit VJP   |

``—`` = unsupported by construction (``eigh`` materialises the full
spectrum).  The ``shift_invert`` / ``poly`` sparse cells (``✓`` above,
shipped in phase 4) came nearly free: the forwards are written once
against an abstract matvec, and the backwards already existed
(``_project_ell`` / ``_project_sectioned``, shared across methods).
``auto`` still routes sparse to ``lobpcg``; shift-invert / poly on sparse
are served on explicit request only (the spectrum-tuning caveat below).

## Scope: extremal only; ``eigh`` is folded in *in that role*

The dispatcher is the **top-k extremal** facility — its contract is "the
``k`` largest eigenpairs of this operator."  ``eigh`` becomes a *method*
of it only in that role: dense full ``safe_eigh`` **then slice top-k**,
returned in the same largest-first convention as the iterative methods.
That uniform return is what lets connectopy apply one post-processing path
(skip-trivial / ``1−μ`` / ``λ^t`` / right-eigenvector recovery) regardless
of solver, instead of the divergent slicing the eigh and lobpcg paths use
today.

It does **not** subsume the *full-spectrum* uses of ``safe_eigh``:
``linalg.spd`` (matrix log/sqrt/power over the whole spectrum),
``stats.lme.reml`` (the ZZᵀ rotation), ``signal.lomb_scargle`` (the Gram
pseudo-inverse).  Those need every eigenpair, not the extremal few; they
keep calling ``safe_eigh`` directly.  ``safe_eigh`` stays the shared
substrate primitive — including its cuSolver-broken-stack device latching
(see [`linalg.md`](linalg.md) and the B14 note in
[`../feature-requests/spectral-embedding-gpu-solver.md`](../feature-requests/spectral-embedding-gpu-solver.md))
— and the dispatcher's ``eigh`` method is *built on* it.

## Shape

### Types — rigorous, immutable, protocols where they earn it

```python
Method = Literal['eigh', 'lobpcg', 'shift_invert', 'poly']
EigOperand = Union[Float[Array, '... n n'], ELL, SectionedELL]

class EigPair(NamedTuple):                 # uniform, largest-first
    values: Float[Array, 'k']
    vectors: Float[Array, 'n k']

@dataclass(frozen=True)                    # hashable → rides custom_vjp nondiff args
class SolverSpec:
    method: Method
    n_iters: int = 200
    tol: Optional[float] = 1e-7
    sigma: float = -0.5
    outer_iters: int = 15
    cg_iters: int = 12
    degree: int = 4
    shift: float = 1.0
    eps_clamp: float = 1e-8
    # classmethod builders: SolverSpec.eigh(), .lobpcg(...), .shift_invert(...), .poly(...)
```

A **Protocol** is warranted for the one genuinely open set — the forward
kernel, the extension point for a future Lanczos / Chebyshev method:

```python
class ForwardKernel(Protocol):
    def __call__(
        self, apply: Callable[[Array], Array], X0: Array, spec: SolverSpec,
    ) -> EigPair: ...

_FORWARD: Mapping[Method, ForwardKernel] = {   # frozen registry of pure functions
    'lobpcg': _lobpcg_forward,
    'shift_invert': _shift_invert_forward,
    'poly': _poly_forward,
}
```

We deliberately do **not** introduce a Protocol over the three operand
formats: they are a closed, already-well-typed union, and the spec prefers
``isinstance`` / ``Literal`` dispatch (the codebase already narrows with
``TypeIs`` in ``laplacian._is_sparse``).  Protocol for the open set,
``isinstance`` for the closed one.

### The custom_vjp boundary — consistent with the prior decision

[`lobpcg-implicit-vjp.md`](lobpcg-implicit-vjp.md) §"What we considered and
didn't pick" rejected a unified ``lobpcg_top_k(A, …)`` because
``jax.custom_vjp`` has a *fixed* diff-arg structure — dense differentiates
w.r.t. ``M``, ELL w.r.t. ``values``, SectionedELL w.r.t. the per-section
``values`` tuple.  That decision stands.  We unify the **orthogonal** axes
instead.  Three per-format ``custom_vjp`` entry points survive, each
parameterised by the static ``SolverSpec``:

```python
_eig_top_k_dense(M, X0, spec)                                  # diff wrt M
_eig_top_k_ell(values, indices, X0, n_cols, spec)             # diff wrt values
_eig_top_k_sectioned(values_t, indices_t, row_groups_t, X0, n_cols, spec)
```

Within each, the **forward** branches on ``spec.method`` (static → resolved
at trace time, JIT-safe) and runs ``_FORWARD[spec.method](apply, X0, spec)``
on the format's matvec; the **backward** is the single shared per-format
projection.  ``spec`` packs all static config into one hashable nondiff arg
— the generalisation of the ``tol_eps_clamp`` tuple trick the sectioned
path already uses (``_lobpcg_diff.py:543``).  Net effect: 5 kernels → 3,
four duplicated backwards removed, and the ``(method × format)`` matrix
becomes dense automatically.

(Dense plain-``lobpcg`` keeps passing the concrete ``M`` to
``lobpcg_standard`` rather than a matvec closure, to preserve its
blocked-matmul performance; the ``apply`` abstraction is what
shift-invert / poly already build their ``shifted`` / ``filt`` closures
from, so no perf is lost there.)

### The dispatcher surface

```python
def eigsolve_top_k(
    operand: EigOperand, k: int, *, spec: SolverSpec, seed: int = 0,
) -> EigPair: ...
```

One pure function owning what is scattered today:

- **Validity table** — a frozen ``Mapping[Method, frozenset[format]]`` plus
  a ``differentiable(method, format)`` predicate — replacing the ad-hoc
  ``raise ValueError``s with one clear error path.  The only hard
  constraints: ``eigh`` requires dense (full spectrum), and lobpcg needs
  ``5·(k+1) < n``.
- **An ``auto`` policy** as a pure, testable function: format +
  "gradients needed?" + size + preconditioner-request → concrete
  ``Method``.  Replaces ``_auto_solver`` and the ``preconditioner != 'none'
  and solver == 'eigh'`` reroute.  Sparse default stays ``lobpcg`` (see
  below).
- **Device routing** once, not six times (``_device_put_graph`` + source /
  target + move-back).
- **``eigh`` folded in** as the dense method: ``safe_eigh`` then slice
  top-k, same largest-first return.

connectopy then becomes: ``_build_affinity_operator`` → build a
``SolverSpec`` → ``eigsolve_top_k(M, k_total, spec)`` → uniform
post-processing.  The two ~110-line ladders collapse to a few lines each.

## Sparse extension: ``shift_invert`` and ``poly`` over ELL / SectionedELL

Feasibility is high and low-novelty — no new math, no new kernels, no new
backward:

- ``shift_invert`` forward is matvec + inner ``jax.scipy.sparse.linalg.cg``
  (a callable-operator solver); both accept a matvec closure.  The backward
  is the existing ``_ell_bwd`` / ``_sectioned_bwd``.
- ``poly`` forward is ``degree`` repeated matvecs plus a Rayleigh-quotient
  recovery (matvec); same backward reuse.
- ``eigh`` stays dense — it genuinely needs a materialised matrix and the
  full spectrum.

Two caveats are *tuning*, not correctness:

- Shift-invert's ``σ`` / CG-iteration defaults (``_SI_*`` in
  ``connectopy.py``) were tuned on a dense ``n=1024`` clustered spectrum;
  inner-CG counts on large sparse operators may want retuning.  Ship the
  *capability*; keep ``auto`` routing sparse → ``lobpcg`` until a sparse
  bench earns the switch.  The extension honours an *explicit* request.
- Poly's ``shift=1`` assumes eigenvalues in ``[−1, 1]`` (the normalised
  affinity) — already true for the connectopy operator; documented as a
  precondition.

## Home and surface

- **Home: ``nitrix.linalg``** (new ``linalg/eigsolve.py`` +
  ``linalg/_eigsolve_kernels.py``).  ``linalg → semiring / sparse`` is
  acyclic (nothing under ``semiring/`` or ``sparse/`` imports ``linalg``),
  so the matvec-dependent kernels relocate cleanly; ``graph/_lobpcg_diff.py``
  is retired into the new kernels module (a thin re-export shim may bridge
  the transition).
- **Internal first, public later.**  Build it internal
  (``linalg/_eigsolve``), exercised by connectopy and community; promote
  ``eigsolve_top_k`` / ``EigPair`` / ``SolverSpec`` to public
  ``nitrix.linalg.eigsolve`` when a second consumer (aCompCor) lands, to
  avoid premature API lock-in (SPEC §2.6 golden-output stability).

## Plan of work

1. **Characterise.**  Lock current behaviour before touching anything:
   eigh-vs-lobpcg agreement (``test_graph.py:368``, ``:575``),
   sectioned-matches-dense (``:471``, ``:520``), source-device
   preservation, skip-trivial conventions.  Guard rail for the extraction.
2. **Refactor forwards to matvec-generic.**  Rewrite the three method
   forwards against an ``apply: X ↦ MX`` closure; behaviour-preserving.
3. **Land the dispatcher** (``linalg/_eigsolve``): ``SolverSpec``, validity
   table, ``auto`` policy, device routing, the three format entry points
   sharing per-format backwards.  Re-point connectopy; delete the
   duplicated ladders.  Behaviour-preserving — the characterisation suite
   is the proof.
4. **Extend sparse coverage** — the now-cheap part: register ELL +
   SectionedELL forwards for ``shift_invert`` and ``poly``.
5. **Promote to public** (gated on aCompCor): export the surface + a
   reference-doc update.

Phases 1–3 are the "dedicated dispatcher"; phase 4 is the "sparse
extension"; they share the phase-2 prerequisite.

**As built (2026-06-06):** phases 1–4 are complete.  Phase 3 landed in
three stages — A (consolidate the kernels + ``SolverSpec`` into
``linalg/_eigsolve`` behind a ``_lobpcg_diff`` compatibility shim), B
(rewrite connectopy onto ``eigsolve_top_k`` with the unified
post-processing), C (remove the shim, migrate the direct-kernel tests to
``nitrix.linalg``).  The two ``rejects_sparse`` tests were flipped to
sparse-agreement + grad-finiteness as phase 4 widened ``_SUPPORTED``.
Phase 5 (public promotion of ``eigsolve_top_k`` / ``SolverSpec`` /
``EigPair``) remains gated on a second consumer (e.g. aCompCor).

## Testing implications

- **Two tests flip from reject to accept**:
  ``test_shift_invert_rejects_sparse_input`` (``test_graph.py:416``) and
  ``test_polynomial_preconditioner_rejects_sparse_input`` (``:439``).
  Extending coverage deliberately removes the restriction they pin; they
  are rewritten to assert sparse ``shift_invert`` / ``poly`` agree with
  dense ``eigh`` and that their ELL / SectionedELL gradients match the
  dense-projected gradient (generalising
  ``test_lobpcg_ell_grad_matches_dense_projected``, ``:741``).  This is the
  one intentional behaviour change.
- New: dispatcher validity / ``auto``-policy unit tests; ``SolverSpec``
  hashability under ``custom_vjp``.
- The no-``(n, n)``-intermediate HLO audit (``bench/PERF_LOBPCG.md``)
  extends to the new sparse ``shift_invert`` / ``poly`` backwards — same
  projection, so it should pass; the bench is the proof.

## Future capability: a ``want='smallest'`` mode

Once dispatch is unified, finding the *smallest* eigenpairs of an operator
becomes a dispatcher **prefix + suffix**, not a new solver.  ``B = σI − A``
shares ``A``'s eigenvectors and has eigenvalues ``σ − λ_i`` in reversed
order, so a top-k *largest* solve on ``B`` returns the *smallest* pairs of
``A``.  The prefix wraps the matvec, ``apply_B(X) = σ·X − apply_A(X)``; the
suffix un-shifts the values, ``λ_i = σ − μ_i``; eigenvectors pass through
unchanged.  Because the wrap is on the *matvec* while the concrete operand
is still carried for the backward, this reuses ``_ell_bwd`` /
``_sectioned_bwd`` verbatim — the format backward projects onto ``A``'s own
pattern with un-shifted eigenvalues, so **no new backward is needed**.  It
is the same forward(``apply``)-⟂-backward(operand) split the main refactor
introduces; smallest-mode is the cleanest demonstration of why that split
is worth having.

**Choosing σ.**  Which subspace is found is shift-*invariant*
(``argmax_i (σ − λ_i) = argmin_i λ_i`` for any σ); σ is a conditioning
knob, with one hard constraint: ``poly`` needs ``σ ≥ λ_max`` so ``B`` is
PSD and the power filter stays monotone, and plain ``lobpcg`` wants
``σ ≈ λ_max`` for the best relative gap.  Three ways to get σ, increasing
in cost: (1) a **known a-priori bound** for bounded operators — the
normalised affinity is in ``[−1, 1]``, so ``σ = 1`` with no estimate (this
*is* the ``L = I − M`` shift connectopy hard-codes; the general mode just
lifts that special case out); (2) **Gershgorin / max-row-abs-sum**, a
matvec-free guaranteed upper bound (looser → slower); (3) a **prefix top-1
solve** for ``λ_max`` — literally a recursive ``eigsolve_top_k(A, 1)`` of a
few matvecs (tightest, needs a small safety margin so it stays an *upper*
bound for poly's PSD requirement).

**Policy and caveats.**  ``want='smallest'`` is a *mode*; the auto-policy
still picks the *method*.  ``eigh`` needs no shift (slice the other end of
the full spectrum).  For **clustered near-zero** spectra (Laplacians),
flip+lobpcg is the bad case — the tiny absolute gap near 0 becomes a tiny
*relative* gap at the top of a wide spectrum (the B14 measurement of plain
lobpcg ~12× off cupy) — so the policy should prefer **``shift_invert``**,
which amplifies the near-zero region directly.  The flip is most valuable
when the smallest pairs are well-separated and a matvec-only method is
wanted.

**Differentiability.**  Treat σ as a ``stop_gradient`` convergence device
(it is not part of the definition of the smallest pair): ``dB/dA = −I``
gives the matching sign on the operand cotangent, the value un-shift gives
the sign on the eigenvalue cotangent, and the recovered ``(λ_i, u_i)`` are
exactly ``A``'s — so the shared implicit VJP applies unchanged.

## What we considered and didn't pick

- **A single ``custom_vjp`` over the operand union.**  Rejected for the
  same reason as the prior ``lobpcg_top_k(A, …)`` proposal — ``custom_vjp``
  fixes the diff-arg structure, which differs per format.  We unify the
  method axis (static ``SolverSpec``) and share the backward across
  methods, keeping three concrete per-format entries.
- **A Protocol over the operand formats.**  Closed, well-typed set;
  ``isinstance`` / ``TypeIs`` dispatch is the established pattern and adds
  no indirection.  Protocol is reserved for the open forward-kernel set.
- **Subsuming ``safe_eigh``.**  The dispatcher is extremal-only; full-
  spectrum consumers (``spd`` / ``reml`` / ``lomb_scargle``) keep calling
  ``safe_eigh`` directly.  The dispatcher's ``eigh`` method is built on it.
- **Changing the sparse ``auto`` default to shift-invert.**  Deferred to a
  sparse benchmark; the extension adds the *capability* under an explicit
  request, not a new default.

## Cross-references

- [`graph.md`](graph.md) — the connectopy consumer; current dispatch site.
- [`lobpcg-implicit-vjp.md`](lobpcg-implicit-vjp.md) — the backward math
  and the fixed-diff-arg-structure rationale this design preserves.
- [`linalg.md`](linalg.md) — ``_solver.safe_eigh`` and the cuSolver-robust
  device pick.
- [`backend-selection.md`](backend-selection.md) — the three-level
  resolution precedent the ``auto`` policy mirrors in spirit.
- [`../feature-requests/spectral-embedding-gpu-solver.md`](../feature-requests/spectral-embedding-gpu-solver.md)
  — B14: the shift-invert / poly solvers and the eigh-wedge latch.
- ``src/nitrix/graph/connectopy.py``, ``src/nitrix/graph/_lobpcg_diff.py``
  — the code to be lifted into ``nitrix.linalg``.
- SPEC §1 (substrate role / separation of concerns), §2.1 (pure functional),
  §2.4 (typed boundaries), §2.6 (golden-output stability).
