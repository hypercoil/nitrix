# Backend selection and fallback observability

> **TL;DR.**  Every public kernel resolves a backend via *explicit
> kwarg → env var → autodetect*; mismatches emit a deduplicated
> ``NitrixBackendFallback`` warning that can be silenced or escalated
> to an error.  Per SPEC_UPDATE §2.7, silent perf regressions are a
> bug.  The library is restricted to NVIDIA Ampere+ for Pallas Triton
> and pure JAX everywhere else (SPEC_UPDATE_v0.2 §1.1).

## Why this exists

We commit to two contradictory-sounding things at once:

1. *Correctness floor:* every kernel has a working pure-JAX fallback
   exercised in CI (SPEC tenet 3).  A Pallas-only kernel is not
   shippable.
2. *Stable kernels:* once a shape × algebra × dtype combination has a
   golden output, it has that output across releases (SPEC §2.6).

These commitments together mean that when Pallas Triton breaks
(churn upstream, a kernel the Triton lowering cannot handle, an
unsupported dtype combination), the *user-visible behaviour* must
not change.  Falling back to JAX silently would honour both
commitments *and* hide perf regressions that downstream consumers
depend on us catching.  The fallback warning is what makes the
fallback non-silent.

## How resolution works

The rule is in [`_internal/backend.py`](../../src/nitrix/_internal/backend.py).
Three levels, in order:

1. **Explicit ``backend=`` kwarg** on the call site.  This wins
   unconditionally; if it asks for an unavailable backend the call
   raises (e.g. ``backend="pallas-cuda"`` on a CPU-only host raises
   ``NitrixBackendError`` rather than silently downgrading -- an
   explicit request is a deployment claim and a silent downgrade
   would mask a misconfigured environment).
2. **``NITRIX_BACKEND`` env var.**  Applies when ``backend="auto"``
   was passed.  Useful for CI matrix runs (``NITRIX_BACKEND=jax``
   forces the fallback path everywhere) and for ad-hoc debugging.
3. **Autodetect.**  ``pallas-cuda`` if ``jax.devices('gpu')`` lists
   at least one device of compute capability ``sm_80`` (Ampere) or
   newer; ``jax`` otherwise.  The compute-capability check happens
   once at module import; subsequent calls reuse the cached result.

The ``Backend`` literal type is exactly
``Literal["auto", "pallas-cuda", "jax"]`` per SPEC_UPDATE_v0.2 §3.1
-- no ``pallas-tpu``, no AMD, no Mosaic.

## How fallback observability works

When a resolved backend cannot actually serve a call (Pallas Triton
cannot tile the shape, a kernel module fails to import, etc.) the
public dispatcher catches the rejection, calls ``fallback(...)``,
and proceeds on the next-best backend.  ``fallback`` is the single
choke-point that:

- Tracks ``(function, shape-signature, dtype, requested_backend)`` in
  a process-wide ``set`` and emits the ``NitrixBackendFallback``
  warning **at most once per tuple**.  A 1000-iteration training loop
  that hits the same Pallas-untileable shape on every step gets one
  warning, not 1000.
- Honours ``NITRIX_SILENCE_FALLBACK=1`` (no warnings) and
  ``NITRIX_STRICT_BACKEND=1`` (warnings become ``NitrixBackendError``).
  Strict mode is the CI safety net: if a kernel that previously ran
  on Pallas regresses to JAX, the build fails.

The warning category is a real ``UserWarning`` subclass so it
participates in the standard ``warnings.filterwarnings`` machinery.

## What we considered and didn't pick

- **Silent fallback.**  Removed by SPEC_UPDATE §2.7.  The cost of
  silent fallbacks during a Pallas regression cycle is exactly the
  perf-cliff problem JAX users hit with TF32-vs-fp32: indistinguishable
  outputs at the dtype level, very different perf, no log trail.
- **Logging instead of warnings.**  ``warnings`` integrates with
  Jupyter and IDEs as a banner, with pytest as a failable category,
  and with the standard filterwarnings API.  Logging would require a
  separate channel and would not deduplicate by default.
- **Per-call fallback opt-in.**  A user wanting to ship over Pallas
  unconditionally is welcome to set ``NITRIX_STRICT_BACKEND=1``;
  there is no need for a per-call ``allow_fallback=False`` argument
  on every kernel.
- **Resetting the dedupe state automatically.**  Considered for tests;
  rejected because cross-test interference is more confusing than
  one explicit ``reset_fallback_state()`` call in fixtures.  The
  function is exported.

## What it looks like in practice

Forward call on a happy-path Pallas matmul -- no warning:

```
semiring_matmul(A, B, semiring=LOG)   # silent
```

Forward call on an ELL contraction (Pallas ELL kernel raises
``PallasELLNotTileable`` unconditionally at first GA -- see
[`ell-on-triton.md`](ell-on-triton.md)):

```
semiring_ell_matmul(v, idx, B, semiring=REAL)
# /nitrix/.../ell.py:124: NitrixBackendFallback:
#   semiring_ell_matmul: falling back to backend='jax' from
#   'pallas-cuda': algebra='real': Pallas Triton kernel unavailable
#   or cannot tile the requested shape. Set NITRIX_SILENCE_FALLBACK=1
#   to suppress this warning, or NITRIX_STRICT_BACKEND=1 to convert
#   it to an error.
```

The same call inside a training loop emits the warning once and then
goes silent, so logs stay clean.

## Cross-references

- SPEC §7.2, SPEC_UPDATE §7.2, SPEC_UPDATE_v0.2 §7.2 -- the
  authoritative backend-selection rules.
- ``tests/test_backend.py`` -- end-to-end coverage of resolution,
  deduplication, env-var knobs.
- The fallback machinery is exercised in real flow by
  ``tests/test_ell.py::test_ell_pallas_falls_back_with_warning`` and
  ``tests/test_conv.py::test_pallas_conv_*``.
