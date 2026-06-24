# Backward kernels and the G1 gate

> **TL;DR.**  Each built-in algebra ships with a hand-derived backward
> rule attached to the ``Semiring`` dataclass as ``matmul_vjp`` /
> ``ell_matmul_vjp``.  The public ``semiring_matmul`` /
> ``semiring_ell_matmul`` wrap the 2-D core in ``jax.custom_vjp`` and
> dispatch the backward to the per-algebra rule.  All five
> differentiable algebras (REAL, LOG, TROPICAL_MAX/MIN_PLUS,
> EUCLIDEAN) pass the G1 finite-difference gate at the pinned
> ``float64`` tolerance.  ``BOOLEAN`` raises with a clear diagnostic.

## The differentiability vocabulary

From SPEC §4.1, the per-algebra story:

| Algebra | Forward | Backward strategy |
|---|---|---|
| REAL | inner product | transpose-matmul (``g @ B.T``, ``A.T @ g``) |
| LOG | log-sum-exp | softmax-weighted, K-loop recompute |
| TROPICAL_MAX_PLUS | max over sum | argmax-gather; one-hot subgradient |
| TROPICAL_MIN_PLUS | min over sum | argmin-gather |
| EUCLIDEAN | √∑(a-b)² | normalised-difference, ``√`` singularity guard |
| BOOLEAN | OR over AND | not differentiable -- raises |

The rules live in
[`semiring/_backward.py`](../../src/nitrix/semiring/_backward.py).  Each
function takes ``(residuals, g_out)`` where ``residuals = (A, B, C)``
(the forward stashes all three; backwards pick what they need) and
returns ``(grad_A, grad_B)``.

## Streaming in the backward, too

For ``LOG``, ``TROPICAL_*``, and the ELL variants of all algebras,
the backward is implemented as a ``lax.fori_loop`` over the
contraction axis so per-step intermediates are ``(M, N)`` rather
than ``(M, K, N)``.  This mirrors the forward streaming property
(see [`streaming-kernel.md`](streaming-kernel.md)).

The ``LOG`` backward is the most illustrative:

```python
def log_matmul_vjp(residuals, g_out):
    A, B, C = residuals
    def body(kk, carry):
        gA, gB = carry
        a_col = A[:, kk:kk+1]           # (M, 1)
        b_row = B[kk:kk+1, :]           # (1, N)
        log_w = a_col + b_row - C       # (M, N)
        w = safe_exp(log_w)             # (M, N), bounded [0, 1]
        contrib = g_out * w             # (M, N)
        gA = gA.at[:, kk].set(contrib.sum(axis=1))
        gB = gB.at[kk, :].set(contrib.sum(axis=0))
        return gA, gB
    return lax.fori_loop(0, K, body, (jnp.zeros_like(A), jnp.zeros_like(B)))
```

The softmax weight ``w[i, k, j] = exp(A[i, k] + B[k, j] - C[i, j])``
is bounded in ``[0, 1]`` because ``C`` is the logsumexp.  We never
materialise the full ``(M, K, N)`` softmax tensor; per-step we hold
``(M, N)``.

``TROPICAL_*`` backwards run two K-loops in sequence: one to find
``argmax_k`` (or ``argmin_k``) per ``(i, j)`` cell, and one to route
``g_out`` through the one-hot mask.  The K-loop computation of the
argmax is itself a streaming reduction (running max + running argmax)
so memory is again ``O(M·N)``.

``EUCLIDEAN`` has a closed-form gradient with a ``1 / C[i, j]``
factor.  We guard at ``|C| ≤ eps`` and zero out the contribution
there, using the "double-where with sentinel" trick to keep AD
NaN-free:

```python
h = safe_div(g_out, C)          # zero where C ≈ 0
grad_A = A * h.sum(1, keepdims=True) - h @ B.T
grad_B = B * h.sum(0, keepdims=True) - A.T @ h
```

No K loop needed -- the gradient factors into matmul-shaped pieces.

## How ``jax.custom_vjp`` is wired

The 2-D core function is decorated::

```python
@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _diff_matmul_2d(A, B, semiring, backend):
    return _forward_only_2d(A, B, semiring=semiring, backend=backend)
```

with the ``Semiring`` and ``Backend`` as non-diff (hashable, carried
identity).  Two subtleties caught us during implementation:

1. **The fwd's signature mirrors the primal's**, not the bwd's.
   Current JAX (>= 0.10) passes nondiff args at their *original*
   positions in the fwd, but prepends them in the bwd.  Earlier JAX
   prepended in both.  We learned this the hard way; the comment in
   ``_diff_matmul_2d_fwd`` is load-bearing.
2. **Integer-valued args (ELL ``indices``) cannot live in
   ``nondiff_argnums``.**  JAX (>= 0.10) refuses to allow Tracers
   through ``nondiff_argnums``, even when the tracer is integer-
   typed.  ELL ``indices`` is therefore a regular array argument;
   the bwd returns ``jnp.zeros_like(indices)`` for it.  This is the
   convention for integer arrays (no meaningful gradient).

Batching composes via ``jax.vmap`` over the 2-D core.  ``custom_vjp``
is vmap-compatible; the bwd dispatch sees the un-batched
``Semiring`` (it's not mapped, by construction) and delegates to the
per-algebra rule, which runs against the un-batched 2-D residuals
per vmap slice.

## The G1 gate

Per IMPLEMENTATION_PLAN §1.3 G1, each algebra must pass a
finite-difference check against ``jax.grad`` at the pinned
per-dtype tolerance, or else ship forward-only with a documented
raise.  The gate is encoded in
[`tests/test_backward.py`](../../tests/test_backward.py).

Headline numbers (all at ``float64`` with ``eps=1e-5`` and ``rtol=
atol=1e-6``, except ``TROPICAL_*`` where ``eps=1e-4`` to avoid
crossing the argmax tie boundary):

| algebra | matmul backward | ELL matmul backward |
|---|:---:|:---:|
| REAL | ✓ (matches analytical) | ✓ |
| LOG | ✓ | ✓ |
| TROPICAL_MAX_PLUS | ✓ (subgradient) | ✓ |
| TROPICAL_MIN_PLUS | ✓ (subgradient) | ✓ |
| EUCLIDEAN | ✓ (off-singularity) | ✓ |
| BOOLEAN | raises (asserted) | raises (asserted) |

The EUCLIDEAN singularity guard is also tested (``A_row == B_col``
configuration): gradient stays finite, no NaN.

## Pallas backwards (deferred)

Per IMPLEMENTATION_PLAN §5.2 (Phase 2.A.7), Pallas backward kernels
follow forward Pallas kernels.  At first GA all backwards run on
JAX -- including for ``backend="pallas-cuda"`` users, who get the
forward on Pallas and the backward on JAX (no fallback warning for
the backward; that's the documented behaviour, not a regression).

Steady-state cost of the JAX-side backward at 256×256 fp32 (from a
quick check in the perf bench):

| algebra | fwd | fwd + bwd | bwd overhead |
|---|---:|---:|---:|
| REAL | 1.65 ms | 71 µs | 0.04× (XLA dead-codes most fwd) |
| LOG | 1.64 ms | 5.74 ms | 3.50× (K-loop recompute) |
| TROPICAL_MAX_PLUS | 1.65 ms | 5.60 ms | 3.39× (argmax K-loop + routing) |
| EUCLIDEAN | 1.63 ms | 1.65 ms | 1.01× (two matmuls, no K loop) |

LOG and TROPICAL_MAX_PLUS pay ~3.5× for the K-loop recompute; this
is the price for not materialising the ``(M, K, N)`` weight tensor.
A future Pallas backward kernel for LOG specifically would fuse the
forward and backward into a single pass with the softmax weight
computed in registers (similar to flash attention's backward); est.
3-5× speedup over the current JAX bwd.

## What we considered and didn't pick

- **Materialising the softmax / argmax across K.**  Faster forward,
  but ``(M, K, N)`` memory cost for ``LOG`` on 1024² with K=1024 is
  4 GB.  We chose to pay the recompute cost; this matches FlashAttention
  v2's design choice and the SPEC §4.1 language explicitly
  ("recompute the softmax in the backward K loop, not materialised").
- **Symbolic autograd over the algebra's formula.**  KeOps does this
  via templated formulae; for us it would require expressing every
  ``Monoid.update`` as a closed-form symbolic expression, which is
  intractable for the pytree-state algebras.
- **Auto-derive subgradients via numerical perturbation at trace
  time.**  Considered for ``TROPICAL_*`` to handle ties gracefully;
  rejected as too magical.  Current behaviour: ties resolve to the
  *first* maximiser (consistent with ``jnp.argmax``), and the user
  is expected to construct test cases away from ties.

## Cross-references

- SPEC §4.1 "Differentiability vocabulary".
- IMPLEMENTATION_PLAN §5.2 (G1 gate at 2.A.5).
- ``src/nitrix/semiring/_backward.py`` -- the rules.
- ``tests/test_backward.py`` -- 15 finite-difference tests.
- [`semiring-protocols.md`](semiring-protocols.md) -- where the
  ``matmul_vjp`` / ``ell_matmul_vjp`` fields live.
