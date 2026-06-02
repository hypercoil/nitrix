# Semiring algebra Protocols

> **TL;DR.**  We decompose a "semiring" into two Python ``Protocol``s:
> a ``Semigroup`` for the per-element ``(*)`` combine and a ``Monoid``
> for the running ``(+)`` reduction.  The monoid carries a pytree
> accumulator so numerically-stable online reductions (log-sum-exp's
> ``(max, sum_exp)`` carry, Welford-style variance, etc.) can thread
> state through the K loop without materialising intermediates.  We
> further split the relaxed ``Semiring`` from a strict subtype
> ``StrictSemiring`` so functions that need free reassociation can
> declare it at the type level.

## Why two Protocols and not one

The first instinct is to model a semiring as a single dataclass with
``binary_op: Callable[[a, b], c]`` and ``reduce_op: Callable[[a, b], c]``.
That works for the simple algebras (``REAL``, ``TROPICAL_MAX_PLUS``)
but breaks the moment you need a stable LSE: the running
``logsumexp`` cannot be expressed as a single ``(acc, value) -> acc``
without losing the running max.  The "softmax / flash attention"
solution is to keep the accumulator as a pair ``(m, s)`` where ``m``
is the running max and ``s`` is the running sum of ``exp(value - m)``.
That state is *internal* to the reduction; it isn't part of the
user-facing output.

So the model is: a ``Monoid[S]`` carries a *state pytree* ``S`` that
need not match the output shape, with four methods --- ``init``,
``update`` (consume one value), ``merge`` (combine two states; needed
for tree reductions and cross-device reduce-scatter), and ``finalize``
(project state to user-facing output).  Strict semirings are then
those whose ``binary_op`` is associative and where the streaming K
loop's order doesn't matter; relaxed semirings (currently just
``EUCLIDEAN``, whose ``finalize`` is a non-monoidal ``sqrt``
projection) don't allow free reassociation.

This is the brainstorm shape from
[`_refstubs/semiring_gemm.py`](../../_refstubs/semiring_gemm.py)
preserved in
[`semiring/_types.py`](../../src/nitrix/semiring/_types.py).

## Why pytree state buys real things

The pytree-state pattern shows up in three different places:

- **``LogSumExp`` ``(m, s)`` carry** keeps the running normalised sum
  bounded in ``[0, K]`` even when the inputs have ±50× the dynamic
  range of fp32 (verified by
  ``tests/test_semiring.py::test_log_large_magnitudes_finite_jax``;
  inputs scaled by 50, output stays finite).
- **Euclidean's ``finalize = sqrt``** projects the running
  sum-of-squares to a distance once per output cell, with a clamp
  at the origin so ``sqrt`` does not see negative rounding noise.
- **Backward kernels (Phase 2.A.5)** can reuse the same pytree-state
  pattern in reverse: ``LOG``'s backward recomputes the softmax weight
  per-K-step from the residual ``(A, B, C)`` triple inside a
  ``lax.fori_loop`` body, so no ``(M, K, N)`` weight tensor hits HBM
  (see [`backward-kernels.md`](backward-kernels.md)).

For the kernel substrate, ``Monoid[S]`` and ``Semigroup`` are the
*only* algebra-specific knobs the streaming kernel needs.  The forward
Pallas kernel ([`_kernels/cuda/semiring_matmul.py`](../../src/nitrix/_kernels/cuda/semiring_matmul.py))
threads ``S`` through ``lax.fori_loop`` and lowers identically for
every algebra; we get one C++/Triton compile per ``(BM, BK, BN,
algebra)`` instead of one per algebra.

## Why relaxed vs strict

Two cases motivate the split:

- **Functions that re-associate the K loop.**  A tree reduction or a
  cross-device ``reduce_scatter`` needs ``(a (+) b) (+) c == a (+) (b
  (+) c)``.  These should annotate ``semiring: StrictSemiring`` so
  callers passing the relaxed ``EUCLIDEAN`` (whose ``binary_op`` is
  ``(a - b)**2``, non-associative in the algebra sense once a
  ``finalize=sqrt`` is in the picture) are rejected at the type-check
  site rather than producing wrong answers at scale.
- **Streaming kernels** (matmul, conv, ell-matmul) fix the reduction
  order; they accept the relaxed ``Semiring`` because their
  correctness does not depend on associativity.  ``EUCLIDEAN`` works
  here.

The strict-vs-relaxed boundary is *advisory*: we don't programmatically
verify associativity, we just type-tag the algebra and require the
caller to opt in via ``strict=True`` or the ``StrictSemiring(...)``
constructor.  Users supplying a custom algebra sign the contract by
choosing the constructor.

## User-defined algebras

The public surface lets a user wire up a custom algebra by
constructing ``Semiring(monoid=..., binary_op=..., identity=...,
name=..., matmul_vjp=..., ell_matmul_vjp=...)``.  The kernel substrate
treats it identically to a built-in.  Two things to know:

- **Forward-only by default.**  ``matmul_vjp=None`` and
  ``ell_matmul_vjp=None`` mean ``jax.grad`` over a call with this
  algebra will raise a clean ``TypeError`` at backward time.  To
  ship a differentiable custom algebra the user supplies a backward
  rule (signature ``(residuals, g_out) -> (grad_A, grad_B)``); the
  forward returns ``(out, residuals)`` where ``residuals`` is
  ``(A, B, out)`` by current convention.
- **Hashability.**  ``Semiring`` is a frozen dataclass, so it hashes
  by structure.  This is load-bearing because ``jax.custom_vjp``'s
  ``nondiff_argnums`` requires the carried algebra to be hashable.
  Users defining their own ``Monoid`` / ``Semigroup`` should use
  ``@dataclass(frozen=True)``.

## What we considered and didn't pick

- **A single ``reduce(...)`` callable.**  Loses the pytree-state slot
  needed for stable LSE / online variance.  Considered briefly,
  rejected on the first numerical-stability test draft.
- **KeOps-style symbolic autodiff.**  KeOps generates backwards by
  differentiating the user's formula DAG.  Out of scope for first
  GA; per SPEC_UPDATE §3.1, custom-algebra users wrap their own
  ``jax.custom_vjp`` if they want gradients.  Saves a lot of
  implementation cost; the marginal user-base benefiting is small.
- **``Algebra`` instead of ``Semiring``.**  The name "semiring" is
  arguably slightly wrong for ``EUCLIDEAN`` (since its ``(*)`` is
  non-associative), but the term "semiring-analogous algebra"
  per SPEC §3.1 is what the literature uses and the codebase is
  consistent with.

## Learning: `identity` is the monoid identity, not the annihilator

Surfaced while building `sparse.ell_mask` for masking incomplete
geometries (cortical medial wall, grey-matter volume masks).  A
masked / padded edge must contribute a no-op to
``(+)_p values[i, p] (*) B[indices[i, p]]``.  The value that achieves
this is the **`(*)`-annihilator** ``z`` (with ``z (*) b ==
monoid_identity`` for all ``b``), *not* the monoid identity.

``Semiring.identity`` holds the **monoid identity** (the
``monoid.init`` neutral element).  For every built-in *except*
``EUCLIDEAN`` the annihilator happens to equal it -- REAL ``0``,
LOG / TROPICAL_MAX_PLUS ``-inf``, TROPICAL_MIN_PLUS ``+inf``,
BOOLEAN ``False`` -- so ``identity`` doubles as the masking value.
``EUCLIDEAN``'s ``(a - b)**2`` has **no** annihilator, but its
``identity`` is ``0.0`` (the sum-monoid neutral); masking with that
value injects ``B[idx]**2`` rather than vanishing.  EUCLIDEAN
neighbourhoods must therefore be masked by dropping columns
structurally, not via a value.

Because overloading one field for two distinct algebraic roles is a
confusion risk as the algebra set grows, ``Semiring`` now carries an
explicit ``annihilator`` field (B8, shipped): ``0`` for REAL, ``-inf``
for LOG / TROPICAL_MAX_PLUS, ``+inf`` for TROPICAL_MIN_PLUS, ``False``
for BOOLEAN, and ``None`` for ``EUCLIDEAN`` (which has no annihilator).
``ell_mask`` accepts ``semiring=`` and reads ``semiring.annihilator``,
raising a clear error when it is ``None`` (EUCLIDEAN) instead of
silently injecting ``B[idx]**2``.  The legacy ``ell_mask(identity=...)``
form still works but emits a ``DeprecationWarning``.  See
``IMPLEMENTATION_PLAN.md §10.3`` (2026-06-02 entry).

## Cross-references

- SPEC §3.1 "Algebra representation"; SPEC_UPDATE §3.1 (strict/relaxed
  split).
- ``src/nitrix/semiring/_types.py`` -- the Protocol definitions.
- ``src/nitrix/semiring/algebras.py`` -- the six built-ins.
- ``tests/test_semiring.py`` -- algebra-level correctness; identity
  propagation; numerical stability.
