# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`nitrix` is the **lowest-level numerical substrate** of the diffprog neuroimaging
ecosystem: a pure-JAX/XLA library of differentiable neuroimaging primitives. Every
public symbol is a function over JAX arrays (or more generally PyTrees, optionally
with a `jax.random` key or static shape/param spec or static config args)
returning JAX arrays. It underpins `hypercoil`, `entense`, and other downstream
libraries.

**`SPEC.md` is the architectural source of record.** Read it before non-trivial
work — it defines what belongs here, the firm concern boundaries, and the dispatch
contracts. In-code comments cite it as `SPEC §x` (and `SPEC_UPDATE §x`, which now
resolves into `SPEC.md` via its §11 provenance map). Aspirationally, not currently: the SPEC should be written to invariants rather than drifting.

## Code style

Use rigorous typing, using protocols where appropriate, favour immutable objects and containers with pure functions, and adhere to the style prescribed by ruff. Enums are preferred over bare strings, match/case statements over repeated conditionals, explicit over implicit, hackability not magic. Comments should be used sparingly, for example to provide an explanation when a counterintuitive or suboptimal-looking routine was implemented for a principled reason. Code should be self-documenting. Docstrings should not contain opaque references to enumerated spec sections, FR headings, implementation plan codes, etc. (i.e., `SPEC §x` does not belong in any public-facing documentation surface unless specifically targeting developers). British English in comments and docstrings and internal variables, USAmerican English for public API function names and arguments.

### Docstring conventions

Docstrings are **NumPy-style reStructuredText** and are transcluded verbatim into the Quarto API reference (`hypercoil/doc/build/gen_reference_stubs.py`; audited by `inventory_docstrings.py`). Write them accordingly, not as ad-hoc plaintext:

- **Maths uses roles, never code spans.** Inline maths is `` :math:`…` ``; multi-line/display maths is a `.. math::` block. Use real LaTeX (`\top`, `\mu`, `\sum`, `\lambda`, `\operatorname{tr}`) — *not* informal ASCII/Unicode inside `` `` `` literals (`` ``X^T W X`` `` renders as monospace code, not maths). Reserve double-backtick literals for actual code: identifiers, arguments, dtypes.
- **A docstring containing any LaTeX backslash MUST be a raw string** (`r"""…"""`). Otherwise `\t`, `\r`, `\b`, `\v`, `\f` are *valid* escapes (tab, CR, …) that silently corrupt the string with **no** `SyntaxWarning` — e.g. `\top` becomes a literal tab. (Doubling the backslashes, `\\top`, also works but raw is preferred.)
- **Reference other API with cross-reference roles**, not bare code spans, so the generator links them: `` :func:`name` ``, `` :class:`Name` ``, `` :meth:`Class.method` ``, `` :attr:`field` ``, `` :mod:`pkg.mod` ``, `` :data:`CONST` ``.
- **Document every parameter and return** in `Parameters` / `Returns` sections, giving array shapes in the tensor-dimension vocabulary (`(V, p)`, `Float[Array, '*spatial ndim']`); document tuple returns element by element.
- **Public constants get a docstring too** — a PEP 224 attribute docstring (a string literal on the line after the assignment: `GAUSSIAN = Family(...)` then `"""The Gaussian family: identity link, unit variance."""`).
- **Cite sources with a DOI or stable URL**, not just author-year.
- First line is a one-sentence summary, then a blank line, then any extended description; prefer a clear 2–3 sentence summary over a cryptic one-liner.

## Status and imperatives

This is a prerelease, prealpha library. Backward compatibility is not a concern; do not bloat the code base with compat shims. Your imperatives are:
- mathematical correctness
- engineering rigour
- community needs and value
- consumer and user ergonomics
- suite performance
- code organisation
- abstraction and design
- hardware-awareness / GPU acceleration
Note 1: blast radius and implementation effort are only a concern vis-a-vis the above imperatives; they are not in and of themselves decision heuristics. If it will degrade performance or bloat the code base with redundancy, then reconsider how to do it.

## Development environment

`nitrix` is developed and deployed across several different environments (local laptop, HPC, cloud; CPU, GPU, GPU Ampere+, GPU Hopper+). Some numerical kernels are designed to be hardware-aware. When creating a venv, ensure it is suitable for the current compute environment: CUDA capability is required if a GPU is present, Pallas/Triton capability on Ampere+, Pallas/(Triton+Mosaic) on Hopper+. When working on a Code Ocean cloud instance / capsule, you must never set up the venv or any disk-heavy dependencies/installs on the space-limited root partition --- always use /scratch. Design custom performant numerical pathways with the capability to avoid cuSOLVER, as it has proven fragile and flaky in numerous dev and deployment environments.

## Commands

The environment is managed with `uv`. Tests currently run under `nox` across Python
3.11/3.12/3.13, but should be kept up to date across the latest 3-4 versions of Python. (If ruff/mypy version drift causes irreconcilable inconsistency, those should be tested against the latest build specifically.)

```bash
# Full CI gate (tests + ruff + typecheck on all Python versions) — what CI runs:
nox

# Individual sessions:
nox -s tests          # pytest --cov + ruff check + ruff format --check
nox -s typecheck      # mypy src/nitrix (strict: every def fully annotated)
nox -s report         # coverage report --fail-under=99 + html + xml

# Faster local iteration (current venv, no matrix):
uv sync --extra=dev                       # set up the dev environment
uv run pytest tests/test_affine.py        # one test file
uv run pytest tests/test_affine.py::test_name   # one test
uv run ruff check src/nitrix
uv run ruff format src/nitrix             # autofix formatting (single quotes, 79 cols)
uv run mypy src/nitrix

# Regenerate the golden reference corpus (after an intentional kernel change):
uv run python tools/regen_golden.py
```

Coverage must stay ≥ 99%. `mypy` is a hard gate, not aspirational: every `def`
carries a complete signature (`disallow_untyped_defs`).

### Test conventions

- Tests live in `tests/` (flat, ~145 files, one per module/topic).
- **x64 is enabled at the top of nearly every test file** —
  `jax.config.update('jax_enable_x64', True)` must run *before* importing `nitrix`
  (x64 is a trace-time setting). This is why test imports sit below the config call
  (ruff `E402` is ignored for `tests/*`).
- Numerical references (scipy / statsmodels / sklearn / pingouin / numpyro /
  nibabel / nilearn) are **test-only deps** — never runtime imports. Real-mesh
  loaders (`tests/_real_meshes.py`) are guarded by `importorskip`.
- Tests should not be limited to trivial toy cases. A thorough suite of property-based tests that probe numerical invariants is preferred in combination with oracle parity tests.
- Full-scale tests and benchmarks on real data, however, generally go in the sibling nitrix-perf-bench. File a benchmark request there rather than using the legacy bench/ namespace.
- When a test fails just outside of tolerance, provide a thorough justification and rule out other causes before relaxing the tolerance.

## Dispatch model — the four orthogonal axes

Keep these distinct (SPEC §3):

- **`backend=`** — execution engine: `'pallas-cuda'` vs `'jax'`. Resolves via
  `nitrix._internal.backend.resolve_backend`. Auto → `pallas-cuda` only on
  Ampere+ NVIDIA GPUs (probed once at import), else `jax`.
- **`driver=`** — which *numerically-divergent recipe* of the same math (e.g. FIR
  vs recursive Gaussian, sequential vs associative scan). Resolves via
  `nitrix._internal.config.resolve_driver`.
- **`method=`** — algorithm *family* (interpolator kind, eigensolver, …).
- **`dtype`** — precision of the data. Governed by input dtype + explicit knobs,
  **never** by auto-dispatch.

**Fallbacks are loud** (a silent perf regression is a bug): when a resolved backend
can't run, nitrix falls back to JAX and emits a deduplicated `NitrixBackendFallback`
warning. Env knobs: `NITRIX_BACKEND`, `NITRIX_SILENCE_FALLBACK=1`,
`NITRIX_STRICT_BACKEND=1` (escalate fallback to error), `NITRIX_REPRODUCIBLE=1`.

**Divergent ops** (operations that auto-select among numerically-different
implementations) are a registered, golden-tested contract. The 5 registered sites
live in `nitrix._internal._divergent_ops` (eager-imported). `nitrix.reproducible()`
/ `NITRIX_REPRODUCIBLE=1` forces every such site to its canonical variant for
cross-platform stability. Adding a new ungoverned platform-flip **fails CI** via
`tests/test_reproducible_dispatch_guard.py`. Enumerate sites with
`nitrix.divergent_ops()`.

## Architecture

### Layering
- `_internal/` — shared private machinery: `backend.py` (engine resolution),
  `config.py` + `_divergent_ops.py` (driver/reproducibility registry),
  `reductions.py` (the single `reduce()` backing all score kernels), `gaussian.py`,
  `separable.py`, `testutil.py`, `docutil.py`, `util.py`.
- `_kernels/cuda/` — private Pallas-Triton kernels (attention, selective_scan,
  norm, semiring_matmul, semiring_ell_matmul, demons_force, lncc_force). Each
  registers a `custom_vjp` with a paired backward kernel or JAX fallback. These are
  an implementation detail behind the public API.
- Public subpackages (each `__init__.py` is the surface-of-record):
  `semiring`, `sparse`, `morphology`, `smoothing`, `linalg`, `stats`, `signal`,
  `geometry`, `graph`, `metrics`, `register`, `numerics`, `nn`, `augment`, `bias`.
  See SPEC §4 for each subsystem's intent and key surface.

### Core substrate
`semiring` (streaming-kernel reductions over arbitrary algebras — REAL / LOG /
TROPICAL_* / EUCLIDEAN / BOOLEAN) is the foundation. `sparse` (ELL + SectionedELL,
**no BCOO**) is the storage layer, and `morphology` + `smoothing` are built atop
`semiring`. Mesh/graph ops are reductions over ELL — there is no message-passing
base class.

### Two-tier parity (SPEC §2 tenet 10)
A fused Pallas op ships alongside a bit-faithful JAX **reference** (`reference_*`)
that consumers may pin as an oracle. The fused path is certified `pallas ≈ jax`
within the tolerance in `tests/tolerance.toml`. The golden corpus
(`tests/golden/*.npz`, loader `tests/_golden.py`) is reference-vs-golden per
`(op, dtype)`, loosened per `(op, dtype, backend)` for fused paths. **A tolerance
change is a public-API change.**

## Boundaries (firm — these take precedence over convenience; SPEC §1, §5, §6)

- **Runtime imports are restricted to `jax`, `jaxtyping`, `numpy`.** `scipy`,
  `sklearn`, `nibabel`, `numpyro`, `equinox`, etc. are forbidden at runtime
  (test-only). Verify before adding a dependency.
- **No module classes in the public API** — return `NamedTuple`s, custom-registered PyTrees, or frozen
  dataclasses of arrays, never module objects. (Equinox modules are `nimox`.)
- **No `loss` namespace, no scalarisation.** nitrix hosts *score kernels* (return
  unreduced tensors by default; may expose only the flat leaf
  `reduction ∈ {'none','sum','mean'}` + the domain-mask weighted mean). Term
  weighting / objective composition is `nimox`. See SPEC §5.
- **No I/O, no containers, no pipelines, no training loops** — those are `thrux` /
  `bitsjax` / `entense` / `nimox`.
- **No new top-level subpackage without a substrate-composition story.** Prefer a
  keyword over forking a function. If you must fork, hoist out common subroutines as helpers. New primitives are gated by the admission rule
  (irreducible numerical content *or* a named vocabulary family).
- **Precision is fp32/fp64-first.** The scientific core is fp32/fp64-only; reduced
  precision there is a correctness bug. fp16/bf16 is admitted only at the NN-forward
  seam (`nn.attention`, `nn.ssm`, `nn.norm`) under a hard ≥float32-accumulation
  invariant (SPEC §2 tenet 11).
- **fit/apply seam (SPEC §6.5):** expose stateful estimators as a pure
  `fit(reference) -> state` / `apply(input, state) -> output` pair (state = plain
  arrays). The single-call convenience is *defined as* `apply(src, fit(ref))` so the
  two paths can't drift. nitrix owns any convention that crosses the boundary.

## Hardware scope

In scope: Ampere+ NVIDIA GPUs (via Pallas-Triton) and a pure-JAX CPU fallback
(the correctness floor, exercised in CI; CPU perf is not a goal). Out of scope:
TPU, ROCm, Apple Metal, Intel GPUs — all fall back to JAX-CPU. Hopper+ (via Pallas-Mosaic) is currently aspirational. `jax-metal` is
unsupported (missing `linalg.eigh` etc.). Wall-clock perf vs external references is
delegated to the sibling `nitrix-perf-bench` suite, not nitrix's correctness concern.
