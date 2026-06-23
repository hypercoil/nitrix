# selective-scan — state-space-model fused scan (Mamba/S6)

> **Status (2026-06-23), against `nitrix main@6449cfa`.** Not present. Part of
> the [`nn-forward-block-kernels.md`](nn-forward-block-kernels.md) bundle —
> **P1, ENABLING**. Read that ledger first for the shared framing.

**What.** A selective state-space scan primitive `nitrix.ssm.selective_scan`,
with a `jax` reference (today's `lax.scan` recurrence) and a `pallas-cuda`
fused kernel, lifting the inner contraction out of
`ilex/nimox/modules/_mamba.py`.

The current reference (verbatim shape) is the discretised selective recurrence

```
h_t = exp(Δ_t · A) ⊙ h_{t-1} + (Δ_t · B_t) ⊙ x_t          # state update
y_t = Σ_d  C_t · h_t  (+ D ⊙ x_t)                          # readout
```

implemented as a `jax.lax.scan` over the sequence with `softplus` /`silu`
gating in the surrounding module. Consumers: `neurostorm`, `swift` (the 4-D
Swin + SSM hybrids); the depthwise conv and gating stay in the nimox module.

**Why.** The sequential `lax.scan` is the throughput floor for the SSM models:
it is O(L) sequential steps with no work-parallelism across the sequence, and
materialises the per-step state. Mamba's published speedups come from a fused
**parallel-associative-scan** kernel that (a) keeps the state in SRAM across
the sequence tile and (b) recomputes the state in the backward pass instead of
storing all `(L, d_state)` intermediates. This is the SSM analogue of flash
attention — a memory-bandwidth + recompute win that only a fused kernel
delivers, and it belongs in nitrix for the same reason attention does.

**When it bites.** Every Neurostorm / Swift forward, training and inference.
The sequential floor is most acute on long 4-D token sequences, exactly the
neuroimaging-volume regime these models target.

**Proposed API.**

```python
def selective_scan(
    x: Float[Array, '... l d'],          # input sequence
    delta: Float[Array, '... l d'],      # per-step, per-channel Δ (post-softplus)
    A: Float[Array, 'd n'],              # state matrix (diagonal-plus form)
    B: Float[Array, '... l n'],          # input projection (selective)
    C: Float[Array, '... l n'],          # output projection (selective)
    D: Float[Array, 'd'] | None = None,  # skip / residual
    *,
    backend: Backend = 'auto',
) -> Float[Array, '... l d']:
    ...
```

- Selective (input-dependent `B`, `C`, `Δ`) — the S6 / Mamba form, not the
  fixed-kernel S4 convolution. The associative-scan combinator is
  `(a₁,b₁)∘(a₂,b₂) = (a₁a₂, a₂b₁+b₂)`; the reference uses `lax.associative_scan`
  (or the existing `lax.scan`, kept as the slow-but-obvious oracle).
- Discretisation (`Δ·A → exp(Δ·A)`) is inside the op; the surrounding
  `softplus(Δ)` / `silu` gating stays in the nimox module.

**Implementation shape (house pattern).**

- `nitrix/ssm/__init__.py` — public `selective_scan` + `resolve_backend`.
- `nitrix/ssm/_reference.py` — the current `lax.scan` recurrence, kept as the
  bit-exact oracle (ilex Tier-1 parity swaps onto this with no drift).
- `nitrix/_kernels/cuda/selective_scan.py` — fused block-parallel associative
  scan (state in SRAM per tile), `None` on tiling failure → reference + warn.
- Custom VJP: the Mamba backward (recompute states forward, accumulate grads
  in reverse) with a finite-difference check. The reference can lean on
  `lax.associative_scan`'s autodiff as a second oracle for the VJP.

**Tests.**

- Golden corpus: `tests/golden/selective_scan_float32.npy` from the reference.
- Backend parity: `pallas-cuda ≈ jax` within `tests/tolerance.toml`
  (`selective_scan` row).
- Hypothesis: equivalence of `associative_scan` and sequential `scan` forms;
  `D`-skip linearity; degenerate `A→0` reduces to a cumulative input map.
- Gradient: finite-difference VJP vs the `associative_scan` autodiff oracle.

**Acceptance.** nimox's `_mamba.py` inner recurrence becomes a call to this
op; `neurostorm` / `swift` parity stays green on `backend='jax'`; nitrix's
golden corpus certifies the fused path. The relationship to the shipped
`numerics.ode` adjoint pattern (recompute-forward backward) should be noted in
the impl plan — same idea, different combinator.
