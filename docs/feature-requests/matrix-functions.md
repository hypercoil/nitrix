# Matrix functions — `nitrix.linalg.matrix_function`

> **Status (2026-06-08): partial — `matrix_exp` SHIPPED** (`linalg.matrix_exp`,
> general non-symmetric matrix exponential via pure-matmul Taylor
> scaling-and-squaring — GPU-native, no cuSolver), graduated by the
> registration suite (the affine generator; see `docs/design/registration.md`
> and `IMPLEMENTATION_PLAN.md §10.3`). The `sym*` family was already shipped.
> Still open: a general `matrix_function(A, fn)` entry point, `matrix_log`,
> `matrix_polynomial`, `frechet_derivative`. Remaining items gated by the §13
> acceptance protocol. Provenance: `docs/feature-requests catalogue §12.2`.

**What.** A general `matrix_function(A, fn)` — apply `fn` to the
eigenvalues of a symmetric `A`, reassemble — plus three named
specialisations.

**Proposed surface.**

```python
def matrix_function(A, fn): ...              # apply fn to eigenvalues
def matrix_exp(A): ...                        # heat-kernel diffusion
def matrix_polynomial(A, coeffs): ...         # Chebyshev poly of the Laplacian
def frechet_derivative(A, fn, E): ...         # directional derivative
```

**Composition.** `linalg.symlog` / `symsqrt` / `sympower` / `symexp` /
`symmap` already implement matrix log / sqrt / power / exp via `eigh` —
i.e. `matrix_function` already exists in spirit, specialised per-`fn`. This
item is the explicit generalisation plus `matrix_polynomial` (needed for
ChebNet / SGWT band-pass filters) and `frechet_derivative` (matrix-exp
gradient).

**Likely consumer.** Heat-kernel signatures, ChebNet,
[`graph-wavelet-transform.md`](graph-wavelet-transform.md) (§12.13),
[`heat-kernel-diffusion.md`](resolved/heat-kernel-diffusion.md) (§12.3), group-level
fMRI dynamic-connectivity factorisations.

**Effort.** S.

**Live-code status.** `linalg/__init__` ships `symexp`, `symlog`, `symmap`,
`sympower`, `symsqrt` (the per-`fn` reassembly path, with `eigh` VJP). No
general `matrix_function`, `matrix_exp`, `matrix_polynomial`, or
`frechet_derivative` symbol.

## Cross-references

- `docs/feature-requests catalogue §12.2` — origin entry; `§13` — acceptance protocol.
- [`heat-kernel-diffusion.md`](resolved/heat-kernel-diffusion.md) and
  [`graph-wavelet-transform.md`](graph-wavelet-transform.md) — depend on
  `matrix_exp` / `matrix_polynomial`.
- `src/nitrix/linalg/spd.py` — the `sym*` family this generalises.
