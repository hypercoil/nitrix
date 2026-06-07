# B22. Register `ELL` / `SectionedELL` (and kin) as JAX pytrees

> **Status (2026-06-07): open — API/ergonomics + capability gap from
> `nitrix-perf-bench`.** Surfaced building the B18 Win 4 eigensolver cases:
> `eigsolve_top_k` / `laplacian_eigenmap` / `diffusion_embedding` advertise
> `ELL` / `SectionedELL` inputs *and* differentiability, but those dataclasses
> are not registered JAX pytrees, so they cannot cross a `jit` / `grad` / `vmap`
> boundary as an operand without the caller manually unpacking and repacking the
> array fields. Authored perf-bench-side (COVERAGE_MANDATE §5); nitrix disposes.

## TL;DR

`nitrix.sparse.ELL` (and `SectionedELL`, and the other array-bearing frozen
dataclasses) are **plain frozen dataclasses, not registered pytrees**:

```python
jax.tree_util.tree_leaves(ell)          # -> [ell]  (1 opaque leaf)
jax.jit(lambda e: laplacian_eigenmap(e, n_components=8))(ell)   # TypeError:
#   Argument 'ELL(...)' of type <class 'nitrix.sparse.ell.ELL'> is not a
#   valid JAX type
jax.vmap(lambda e: e.values.sum())(ell)                          # same TypeError
```

So any user (or downstream harness) that wants to `jit` / `grad` / `vmap` a
function taking an `ELL` operand must unpack it to `(values, indices)`, pass
those as the traced args, and rebuild `ELL(values, indices, n_cols, identity)`
inside the traced function — boilerplate at every boundary. The perf-bench Win 4
cases now carry exactly this workaround.

Worse than boilerplate: because `tree_leaves(ell)` returns the **whole `ELL` as
a single leaf**, a `jax.tree_util.tree_map(f, params)` over a parameter pytree
that *contains* an `ELL` silently applies `f` to the `ELL` *object*, not its
arrays — a quiet correctness trap, not a loud error.

## Why there's a case now

1. **The capability is advertised but incomplete at the boundary.**
   `eigsolve_top_k`, `laplacian_eigenmap`, `diffusion_embedding` all document
   `ELL` / `SectionedELL` inputs and document being *differentiable w.r.t. the
   operand*. But `jax.grad(f)(ell)` / `jax.jit(f)(ell)` fail unless the caller
   hand-unpacks — so the advertised "differentiable sparse eigensolver" can't be
   driven the natural JAX way.

2. **The pytree decomposition already exists internally.** `_eig_top_k_ell`'s
   `custom_vjp` differentiates w.r.t. `values` (returning a zero cotangent for
   `indices`); `_eig_top_k_sectioned` differentiates each section's `values`.
   That *is* the children-vs-aux split a pytree registration needs — the
   gradient code has already decided it. Registration just exposes it at the
   call boundary instead of unpacking inside `eigsolve_top_k`.

3. **Precedent is already in the tree.** nitrix already registers frozen
   dataclasses as pytrees with `@jax.tree_util.register_pytree_node_class`:
   `smoothing/metric.py` (`DiagonalMetric`, `FactorMetric`),
   `stats/lme/reml.py`, `stats/lme/flame.py`. This is applying an established
   in-house pattern to `nitrix.sparse`, not introducing new machinery.

4. **It is a recurring boundary, not a one-off.** The same unpack/repack is
   needed for *any* jitted/vmapped/autodiffed consumer of a sparse operator —
   graph learning loops (grad of an embedding loss w.r.t. edge weights), batched
   solves (`vmap` over a stack of ELL operators), `lax.scan` over iterations.

## Suggested direction (nitrix disposes)

Register the array-bearing `nitrix.sparse` dataclasses as pytrees, mirroring
`smoothing/metric.py`. The children/aux split is the design call; the existing
`custom_vjp` suggests:

- **`ELL`**: children `(values, indices)`; aux `(n_cols, identity)`. (Matching
  `_eig_top_k_ell`, which carries `values` + `indices` as traced args and
  `n_cols` as `nondiff`. Alternative: make `indices` static aux if the sparsity
  pattern should bake into the jaxpr — a deliberate choice, not obvious; nitrix
  decides. `values`-only-as-child also matches the gradient, which zeros
  `indices`.)
- **`SectionedELL`**: children = the per-section `ELL` children + `row_groups`;
  aux `(n_rows, n_cols, identity)` and the section structure. Nested but
  mechanical once `ELL` is registered.
- **Audit the family.** `Mesh` and `IcosphereHierarchy` (`sparse/mesh.py`) are
  the same shape — frozen dataclasses holding arrays, not registered. They
  likely have the same gap; worth registering (or consciously declining) in the
  same pass so `nitrix.sparse`'s pytree story is uniform.

Add a small test that each registered type round-trips through
`tree_flatten`/`tree_unflatten`, survives `jit` / `vmap` as an argument, and
that `jax.grad` w.r.t. a registered `ELL` returns an `ELL`-structured cotangent
consistent with the current `eigsolve` `custom_vjp` (values populated, indices
zero) — so the registration and the hand-written VJP stay in agreement.

## Risk / things to check

- Once `ELL` is a pytree, `eigsolve_top_k(operand=ELL)` should still unpack
  correctly (it already does internally) — confirm no double-handling and that
  the public `jax.grad(...)(ell)` path and the internal `custom_vjp` compose to
  the same cotangent.
- Any existing code that (intentionally) treated an `ELL` as an opaque leaf in a
  `tree_map` would change behaviour — a quick sweep for `tree_map` over
  structures holding sparse operators is worth it (expected: none rely on the
  leaf behaviour, since it is the trap described above).

## Cross-references

- [`morphology-explicit-se-im2col-cost.md`](morphology-explicit-se-im2col-cost.md)
  (B21) and the Win 4 eigensolver work that surfaced this.
- Precedent: `src/nitrix/smoothing/metric.py`
  (`@jax.tree_util.register_pytree_node_class`), `stats/lme/{reml,flame}.py`.
- Affected API: `linalg/_eigsolve.py` (`eigsolve_top_k`),
  `graph/connectopy.py` (`laplacian_eigenmap`, `diffusion_embedding`).
- perf-bench: `src/nperf/cases/_eigenmap.py` carries the unpack/repack
  workaround (ELL operand passed as `(values, indices)`, rebuilt inside the
  jitted baseline).
- [`internal-backlog.md`](internal-backlog.md) — ledger index (B22 pointer).
