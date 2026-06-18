# Rolled Jacobi eigensolver — cuSOLVER-free, jittable small symmetric eig (`nitrix.linalg`)

> **Status (2026-06-17): parked — gated on a batched / vmapped small-eig
> consumer.** Internal-backlog item **B24**. Surfaced by the `randomized_svd`
> verification this session (commit `38d2f27`): on the cuSOLVER-affected dev L4
> every `safe_eigh` falls back to CPU, which is correct and negligible for a
> one-off decomposition but **cannot be `vmap`/`jit`-ed**, so a patch-wise
> consumer that batches thousands of small eigs would serialise the CPU bounces.

**What.** A `jacobi_eigh(A, *, n_sweeps, ...) -> (eigvals, eigvecs)` for a small
dense **symmetric** matrix, implemented as a *rolled* cyclic Jacobi rotation
sweep — pure matmuls / Givens rotations, **no cuSOLVER custom-call** and **no CPU
latch**. Unlike `_solver.safe_eigh` (which routes a dense `eigh` to a working
device, latching to CPU on the broken stacks and therefore staying eager), this
is a fully on-device, `jit`-able, `vmap`-able, differentiable kernel — at the
cost of being `O(n^3)` per sweep and accurate only for *small* `n` (the regime
where Jacobi is competitive and where the existing dense `eigh` is overkill).

**Why / Trigger.** The promotion trigger is a consumer that needs **many small
symmetric eigendecompositions at once**, i.e. one that would `vmap` the eig:

- **Patch-wise NORDIC / Marchenko-Pastur denoising** — thousands of
  `(patch_voxels, time)` blocks, each an SVD/eig of a small Gram. `vmap`-ing
  `randomized_svd` (or a direct small eig) over patches is the natural form, but
  `safe_eigh` is eager → the per-patch CPU round-trip (~3 ms latency measured
  this session) serialises and dominates.
- **Per-element / per-edge eigendecompositions** in a mass-univariate or
  per-region pipeline (e.g. local covariance structure, tensor-fit eigenvalues)
  where the eig sits *inside* a `vmap` over `~10^5-10^6` elements.

Until such a consumer exists, the one-off `safe_eigh` path is the right tool
(bounded, self-correcting across stacks); this stays parked.

**Proposed surface.**

```python
def jacobi_eigh(
    A: Float[Array, '... n n'],
    *,
    n_sweeps: int = 12,
    sort: bool = True,           # descending eigenvalues + matching vectors
) -> tuple[Float[Array, '... n'], Float[Array, '... n n']]:
    """Symmetric eigendecomposition via rolled cyclic Jacobi rotations.

    cuSOLVER-free and fully jit/vmap/grad-able; for small ``n`` (the sketch /
    per-element regime).  ``n_sweeps`` is a fixed iteration budget (vmap-clean);
    classical Jacobi converges quadratically, so ~8-12 sweeps reach fp tolerance
    for ``n`` up to a few hundred.
    """
```

**Algorithm sketch (rolled, vmap-clean).**

- Cyclic Jacobi: sweep over the `n(n-1)/2` off-diagonal `(p, q)` pairs; each
  applies a Givens rotation zeroing `A[p, q]` (`tan(2θ) = 2A_pq/(A_pp - A_qq)`),
  accumulating the rotations into the eigenvector matrix `V`.
- Keep the **pair loop and the sweep loop rolled** (`lax.fori_loop` /
  `lax.scan`, never Python-unrolled) — the rolled-Cholesky lesson: an unrolled
  `O(n^3)` sweep makes compile cubic in `n`. A fixed `n_sweeps` keeps it
  `vmap`-clean (no data-dependent convergence branch).
- Optionally the **parallel (block) Jacobi ordering** (round-robin pairing) so a
  whole sweep is a few batched rotations rather than `n^2/2` sequential ones —
  better GPU utilisation, same result.
- Differentiable: rotations are `sin`/`cos`/division, all with VJPs; the fixed
  budget unrolls cleanly for reverse-mode (the v1 LME-Newton pattern). An
  implicit-function VJP at the converged eigenbasis is the careful follow-up.

**Composition / where it plugs in.**

- A `randomized_svd(..., eig='jacobi')` (or an internal switch) that swaps the
  two `safe_eigh` calls on the `(l, l)` Gram / projected matrix for
  `jacobi_eigh`, making the *whole* sketch SVD jittable + vmappable + on-device
  (the NORDIC-patch path).
- A natural new **method in the `_eigsolve` dispatcher** (`SolverSpec.method =
  'jacobi'`) for the dense small-matrix extremal case, slotting beside `eigh` /
  `lobpcg` / `shift_invert` / `poly` — the dispatcher already separates
  `forward(method)` from `backward(format)`, so the jittable forward composes
  with the shared subspace VJP. Sibling of **B14**
  [`spectral-embedding-gpu-solver`](spectral-embedding-gpu-solver.md) (lobpcg /
  eigh on the broken GPU).
- Could back a future `stats.pca` `solver='jacobi'` and an aCompCor path that
  need a GPU-resident, batchable decomposition.

**Effort: M.** The rolled cyclic Jacobi + accumulation is compact; the care is
(a) numerical: rotation stability near degenerate eigenvalues, the
sweep-count/accuracy trade for the target `n` range, and (b) the parallel
ordering for GPU throughput. **Oracle:** `numpy.linalg.eigh` (CPU) on random
symmetric matrices — eigenvalues to ~1e-6, `V Λ Vᵀ` reconstruction, orthonormal
`V`; and a `vmap` over a batch reproducing the per-matrix references.

**Live-code status.** No `jacobi_eigh`. `nitrix.linalg.randomized_svd` and
`stats.pca` (randomized solver) use the **eigh-based** range finder routed
through `_solver.safe_eigh` (cuSOLVER-robust, CPU-latch on the broken stacks,
eager). `_eigsolve` ships `eigh` (extremal-sliced `safe_eigh`), `lobpcg`,
`shift_invert`, `poly` — all built on `safe_eigh` for the dense factorisation;
none is a cuSOLVER-free dense symmetric eig.

## Cross-references

- `src/nitrix/linalg/decompose.py` — `randomized_svd`; the `safe_eigh` calls
  this would replace for a vmappable variant (see the module docstring's
  "cuSOLVER-free" note).
- `src/nitrix/linalg/_solver.py` — `safe_eigh` and the probe/latch/CPU-fallback
  machinery this is the jittable, on-device alternative to.
- `src/nitrix/linalg/_eigsolve.py` — the extremal-eigensolver dispatcher a
  `'jacobi'` method would join (`docs/design/eigsolve-dispatcher.md`).
- [`gpu-cusolver-first-call-handle-failure.md`](gpu-cusolver-first-call-handle-failure.md)
  — the broken-cuSOLVER reality that motivates a solver-free eig.
- [`spectral-embedding-gpu-solver.md`](spectral-embedding-gpu-solver.md) (B14) —
  sibling GPU-eig robustness item.
- `docs/feature-requests/stats-modelling-suite-v2.md` — the v2 suite whose
  `randomized_svd` decomposition primitive raised this.
