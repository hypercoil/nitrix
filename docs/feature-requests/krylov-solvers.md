# Iterative Krylov solvers — `nitrix.linalg.krylov`

> **Status (2026-06-08): partial — `cg` SHIPPED** (`linalg.cg` /
> `linalg.krylov.cg`, matrix-free conjugate gradients for SPD systems, dense
> or matvec, differentiable), graduated by the registration suite as the
> wedge-resilient on-device solver for the GN/LM normal equations (see
> `docs/design/registration.md` and `IMPLEMENTATION_PLAN.md §10.3`). Still
> open: `minres` / `lsqr` / `bicgstab` (same template), gated by §13.
> Provenance: `docs/feature-requests catalogue §12.1`.

**What.** Matrix-free iterative linear solvers, generalising the
implicit-operator pattern already proven by the `laplacian_eigenmap`
LOBPCG path (operator is matvec-only — dense, `ELL`, or `SectionedELL` —
and implicit-VJP closes the backward).

**Proposed surface.**

```python
def cg(A, b, x0, *, tol, max_iter): ...        # symmetric positive-definite
def minres(A, b, x0, *, tol, max_iter): ...    # symmetric indefinite
def lsqr(A, b, x0, *, tol, max_iter): ...       # rectangular least-squares
def bicgstab(A, b, x0, *, tol, max_iter): ...   # nonsymmetric
```

`A` is a matvec callable (or a `nitrix.sparse` operator exposing one), so
the solver never materialises the matrix.

**Composition.** The matvec-only operator interface and the implicit-VJP
backward are already validated in `graph/_lobpcg_diff.py` /
`linalg/_solver.py` (`safe_eigh`/`safe_inv` device plumbing). CG is ~30
lines on top of a matvec; the other solvers follow the same template.

**Likely consumer.** Implicit smoothing `(I − αL)x = b`, gradient-flow
integration, surface harmonic-coordinate solves, neural-ODE adjoint passes.
Directly unblocks [`fixed-point-combinators.md`](fixed-point-combinators.md)
(§12.8) whose implicit-VJP needs an inner linear solve.

**Effort.** M. CG is XS; the implicit-VJP pattern is already proven. Other
solvers reuse the template.

**Live-code status.** No `cg`/`minres`/`lsqr`/`bicgstab` in
`linalg/__init__`. `linalg/_solver.py` exists but exposes only
device-placement helpers (`safe_eigh`, `safe_inv`) plus the LOBPCG plumbing
in `graph/_lobpcg_diff.py` — the matvec+implicit-VJP machinery this would
generalise, but no public Krylov surface.

## Cross-references

- `docs/feature-requests catalogue §12.1` — origin entry; `§13` — acceptance protocol.
- [`fixed-point-combinators.md`](fixed-point-combinators.md) — downstream
  consumer (inner solve).
- [`docs/design/lobpcg-implicit-vjp.md`](../design/lobpcg-implicit-vjp.md) —
  the implicit-VJP pattern to reuse.
