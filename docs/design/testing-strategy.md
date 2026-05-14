# Testing strategy

> **TL;DR.**  Five categories: backend resolution + fallback
> observability; per-algebra forward correctness vs naive
> broadcast; identity propagation and numerical stability; backend
> parity (Pallas vs JAX); finite-difference gradient checks.  All
> 90 tests run in ~45 s on a single Ampere GPU; the finite-diff
> tests are JIT-cached to keep the per-test cost bounded.

## What we check, and where

| Check | File | Skips on CPU? |
|---|---|---|
| Backend resolution + env vars | ``test_backend.py`` | no |
| Fallback warning deduplication | ``test_backend.py``, ``test_ell.py``, ``test_conv.py`` | partial |
| Per-algebra forward vs naive broadcast | ``test_semiring.py`` | no |
| Identity propagation (-inf, +inf, False) | ``test_semiring.py``, ``test_conv.py`` | no |
| Numerical stability (LSE with large magnitudes) | ``test_semiring.py`` | no |
| Backend parity (Pallas vs JAX, bitwise) | ``test_semiring.py`` | **yes** (Ampere+ only) |
| ELL format primitives | ``test_ell.py`` | no |
| Pallas ELL falls back with one warning | ``test_ell.py`` | **yes** (Ampere+ only) |
| Per-algebra backward, finite-diff | ``test_backward.py`` | no |
| EUCLIDEAN sqrt singularity guard | ``test_backward.py`` | no |
| BOOLEAN backward raises | ``test_backward.py`` | no |
| Custom-vjp wrapping doesn't change forward | ``test_backward.py`` | no |
| Batched gradient composes correctly | ``test_backward.py`` | no |
| REAL conv matches ``lax.conv_general_dilated`` | ``test_conv.py`` | no |
| Tropical / log / euclidean conv vs naive sliding-window | ``test_conv.py`` | no |
| NaN-safe patch extraction | ``test_conv.py`` (``test_neg_inf_in_tropical_max_plus_propagates``) | no |
| Conv backward via finite-diff | ``test_conv.py`` | no |
| Pallas conv falls back -- single warning | ``test_conv.py`` | **yes** (Ampere+ only) |
| Pallas conv inner-matmul second warning | ``test_conv.py`` | **yes** (Ampere+ only) |

The ``pallas_only`` marker on the parity / fallback tests checks
``_HAS_AMPERE_NVIDIA`` at collection time, so the full suite runs
clean on a CPU host (with the Pallas-specific cases skipped).

## How we make finite-diff tests fast

Naive finite-difference looks like::

```python
for i in range(x.size):
    out[i] = (fn(x + eps_i) - fn(x - eps_i)) / (2 * eps)
```

Without JIT, each ``fn(x + eps_i)`` retraces through the Python
``semiring_matmul`` dispatcher, the ``_diff_matmul_2d`` ``custom_vjp``
wrapper, the per-algebra reference implementation, etc.  That's
~5-10 ms of trace overhead per call.  For an 8×8 = 64-element array
that's 128 calls × ~7 ms = ~900 ms per finite-diff *call*, and each
test does several -- the original suite ran the backward tests at
5-7 s each.

We JIT the loss once inside ``_finite_diff``::

```python
def _finite_diff(fn, x, eps=1e-5):
    jit_fn = jax.jit(fn)
    # ... loop calling jit_fn(x + eps_i) ...
```

After warm-up the per-element call is microseconds.  The same JIT
cache is reused across the 2*N calls because the shape and dtype
don't change.  Backward tests drop from 5-7 s to under 1 s each;
total suite goes from 98 s to 44 s.

## Why we don't check Pallas / JAX parity to ``allclose``

Bit-exact equality.  Our Pallas kernel and our JAX reference are
*the same algorithm* in two languages, threading the same Monoid
state through the same K loop in the same order, issuing the same
floating-point ops without resorting to tensor cores or
reassociation.  At fp32, the only way they'd differ is via XLA
reordering (which Pallas bypasses) or non-determinism in the
tensor-core path (which we explicitly avoid).  The fact that
``np.testing.assert_array_equal`` passes is a regression detector
for any unintended reassociation; if it ever starts failing, we
know to investigate.

This is stricter than the SPEC §10 contract ("agree to pinned
tolerance"), but it's free at fp32 and gives us a sharper failure
mode.

## The golden-corpus scaffolding (sketch, not yet populated)

Per SPEC_UPDATE §2.8, we owe a ``tests/golden/`` checked-in array
per ``(kernel, dtype, algebra, backend)`` cell.  The directory
exists (``tests/golden/``) but is empty.  The current per-algebra
naive-broadcast tests serve as a quasi-golden corpus, but they
don't lock numerics across release boundaries.  This is a
post-Phase-2 task; see IMPLEMENTATION_PLAN §3.1 Phase 0.2 task
("Golden corpus scaffolding").

## What we considered and didn't pick

- **``jax.test_util.check_grads``.**  It's the canonical tool but
  has an opinionated default tolerance and limited control over
  the FD step size.  TROPICAL_* subgradients need a step size
  smaller than the gap to the second-best ``k``; tunable per-test.
  Our hand-rolled ``_finite_diff`` is ~15 lines and gives us that
  control.
- **Hypothesis property tests for identity propagation.**  Tempting
  for the algebraic invariants, but our concrete identity tests
  (``-inf`` row stays ``-inf``, no NaN under ``-inf`` mixed with
  finite, etc.) catch the regressions we actually care about.
  Hypothesis would mainly add randomness without coverage.  Worth
  revisiting if the algebra surface grows.
- **CI on multiple GPU generations.**  Currently A10G (Ampere)
  only.  Lovelace / Hopper / Blackwell support is on the roadmap
  per SPEC_UPDATE_v0.2 §1.1 but blocked on runner availability.

## Cross-references

- ``tests/`` -- all test files.
- ``tests/golden/`` -- empty; future golden corpus.
- IMPLEMENTATION_PLAN §3.1 Phase 0.2 -- the test scaffolding
  requirements.
- SPEC §8, SPEC_UPDATE §8, SPEC_UPDATE_v0.2 §8 -- the
  authoritative testing contract.
