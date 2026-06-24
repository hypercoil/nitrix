# Reproducible dispatch — a central design principle (P0: design, review-first)

> **Status (2026-06-24): proposed (design doc for review).** No code yet. This
> codifies how nitrix governs auto-selection among *numerically divergent*
> implementations of one operation, so that hardware-aware performance and
> scientific reproducibility are both first-class. Destined for `SPEC.md §2`
> (a new design tenet) once the shape is approved; this doc is the staging
> ground and the per-site rollout ledger.
>
> Driver: the ilex consumer request
> [`cubic-bspline-prefilter-backend-parity`](cubic-bspline-prefilter-backend-parity.md)
> (a `CubicBSpline` GPU-vs-CPU parity break), recognised as one instance of a
> codebase-wide pattern.

## 1. The problem

nitrix has **two orthogonal dispatch axes**, and only one is governed:

1. **Execution backend** — *where / which kernel* runs: `pallas-cuda` vs `jax`.
   Governed by `backend=` (keyword) → `NITRIX_BACKEND` (env) → Ampere autodetect,
   with loud fallback (`NitrixBackendFallback`), strict mode, and a certified
   `pallas ≈ jax` tolerance. **This axis is well-governed.**

2. **Numerical variant** — *which numerically-distinct algorithm* computes the
   same math: e.g. a sequential `lax.scan` vs a parallel `lax.associative_scan`
   (floating-point **reassociation**), or a deterministic one-hot histogram vs a
   non-associative atomic scatter-add (**run-to-run nondeterminism**). Today this
   axis is selected silently by `default_backend_is_gpu()` and is **ungoverned**.

Axis 2 matters because reproducibility is a scientific requirement: the same
input must give the same output across deployment platforms (CPU fixture vs GPU
bundle) and across runs, or processing heterogeneity manufactures artefactual
batch effects. A complete audit (2026-06-24) found **five** numerically-divergent
sites on axis 2, with three distinct failures of governance:

| # | Site | Divergence | Override today | Visibility |
|---|------|-----------|----------------|------------|
| 1 | `nn.ssm.selective_scan` | assoc vs sequential scan (reassociation) | `method=` | documented |
| 2 | `signal._iir` (`sosfilt`/`sosfiltfilt`/`iir_filter`) | fft vs scan vs assoc | `backend=` | documented + loud FFT-fallback |
| 3 | `register._svf` gaussian smoothing | FIR vs Young–van Vliet recursion (~1–2% at edges) | none | documented, silent |
| 4 | `metrics.information` joint histogram | one-hot matmul vs atomic scatter — **run-to-run nondeterminism** on GPU | none | documented, silent (has a determinism test) |
| 5 | `geometry.CubicBSpline` prefilter | assoc vs sequential scan (reassociation) | none | docstring **falsely** claims "agree to a ULP" |

Three problems:

- **Invisible.** 3 of 5 have no override and are silent; #5 actively *misstates*
  the contract (a long IIR pole filter breaks ULP agreement — the ilex break).
- **Incoherent vocabulary.** `backend=` means *kernel engine* in `nn` but
  *algorithm variant* in `signal`; `method=` means *variant* in `ssm`/`gaussian`
  but *kernel/algorithm family* in `linalg`/interpolation. Same word, two axes.
- **Two concerns tangled.** Cross-*platform* divergence (1, 2, 3, 5) and
  run-to-run *nondeterminism* (4). A complete principle covers both.

## 2. The principle

> **Reproducible dispatch.** Where nitrix auto-selects among *numerically
> divergent* implementations of one operation:
>
> 1. **the default is hardware-aware** — the fastest correct variant for the
>    detected platform;
> 2. **selection is overridable per-call** by a consistently-named keyword
>    `driver=`, distinct from `backend=` (which remains the execution-engine
>    axis);
> 3. **a single library-level reproducibility mode** (`NITRIX_REPRODUCIBLE=1`
>    env + `with nitrix.reproducible():` context) forces the **canonical**
>    variant at *every* such site — and a deterministic reduction wherever an op
>    is otherwise run-to-run nondeterministic — trading peak performance for
>    cross-platform / cross-run stability **up to a documented tolerance**;
> 4. **every divergent site is a registered contract** — `{op, canonical, fast,
>    driver, tolerance}` — and the golden corpus tests `variant ≈ canonical`
>    within that tolerance;
> 5. **the divergence is documented** on the public API (honest tolerances; no
>    false ULP claims).
>
> Dispatch among *mechanically-equivalent* implementations (bit-identical, or
> ULP-identical and tested as such — e.g. the order-0/1 `map_coordinates`
> gather, or a certified `pallas ≈ jax` kernel) is **exempt** and may dispatch
> freely.

This extends existing tenet **#3** ("Backend selection deterministic and
user-overridable") from the execution axis to the numerical-variant axis, and
tenet **#6** (cross-*release* reproducibility) to cross-*platform / -run*
reproducibility.

### What "reproducible" does and does not promise

- **Does:** select the *same algorithm and reduction order* on every platform
  (eliminating the reassociation divergence — the large, unbounded one), and a
  *deterministic reduction* where the default is nondeterministic (the atomic
  scatter). This removes the *algorithm-choice* source of divergence entirely.
- **Does not:** promise bit-identical results across different hardware.
  Residual differences from fused-multiply-add ordering, transcendental-function
  implementations, and XLA fusion remain even for one algorithm; they are small
  and **bounded by the site's documented tolerance**. Promising bit-equality
  across hardware would itself be a false guarantee.
- **Never reduces accuracy.** The canonical variant is the reference oracle (or
  the more faithful path, e.g. FIR over the ~1–2% Young–van Vliet edge
  approximation); reproducible mode trades *speed* for *consistency*, not
  *correctness*.

## 3. The resolution model

Precedence mirrors `resolve_backend` (keyword > env/context > auto):

```
driver= (per-call, not 'auto')   >   reproducible mode   >   hardware-auto (default)
```

- `driver='auto'` (the default everywhere) defers to mode, then to hardware.
- An explicit `driver=` overrides even reproducible mode (explicit is explicit;
  a power user who names a variant gets it). Reproducible mode sets the *default*
  to canonical, not a hard lock.

**Vocabulary (locked):**

| Keyword | Axis | Values | Status |
|---|---|---|---|
| `backend=` | execution engine | `auto` / `pallas-cuda` / `jax` | unchanged; **reserved for this axis only** |
| `driver=` | numerical variant | `auto` / per-op (e.g. `sequential`/`associative`/`fft`/`fir`/`onehot`) | **new, unified** |
| `method=` | algorithm *family* (different valid answers) | per-op (eigensolver, interpolator) | unchanged |
| `representation=` | group vs algebra (registration) | `group` / `algebra` | unchanged |

`driver` is the SciPy/Torch convention for "which routine computes this"
(`scipy.linalg.eigh(driver=)`, `torch.linalg.lstsq(driver=)`), unused as a
nitrix keyword, and orthogonal to all of the above. (`signal._iir`'s current
`backend=` — which is really this axis — is renamed to `driver=` with a
deprecation alias; `ssm`/`gaussian`'s `method=` for this axis likewise migrate
to `driver=` with aliases.)

### "Canonical" is per-site, not "the CPU path"

The canonical variant is the **reference the golden corpus is pinned to** —
chosen for faithfulness, which is not always the CPU default:

| Site | canonical (reproducible) | fast default (hardware-auto) |
|---|---|---|
| selective_scan | `sequential` | `associative` (GPU) / `sequential` (CPU) |
| sosfilt / iir | `scan` | `fft` (GPU) / `scan` (CPU) |
| gaussian | `fir` (more faithful) | `fir` (GPU) / `recursive` (CPU, σ≥0.5) |
| joint histogram | `onehot` (deterministic) | `onehot` (GPU, n≤200k) / `scatter` else |
| CubicBSpline prefilter | `sequential` | `associative` (GPU) / `sequential` (CPU) |

So for `gaussian`, reproducible mode makes the **CPU** adopt the GPU's FIR
(removing the YvV edge approximation) — i.e. the principle is "one designated
variant everywhere," not "force the CPU variant."

## 4. Public API shape

```python
import nitrix

# 1. Deployment-wide (the ilex bundle case — set in the environment, no code change)
#    NITRIX_REPRODUCIBLE=1

# 2. Scoped (preferred in code)
with nitrix.reproducible():
    y = nitrix.nn.selective_scan(x)      # -> 'sequential' on any platform
    z = nitrix.signal.sosfilt(sos, sig)  # -> 'scan'

# 3. Per-call escape hatch (power user; overrides even (1)/(2))
y = nitrix.nn.selective_scan(x, driver='associative')

# 4. Introspection (discoverability, in lieu of runtime warnings)
nitrix.divergent_ops()
# -> [{'op': 'nn.ssm.selective_scan', 'canonical': 'sequential',
#      'fast': {'gpu': 'associative', 'cpu': 'sequential'},
#      'driver_values': ('auto','sequential','associative','chunked'),
#      'tolerance': {'float32': 2e-3}}, ...]
```

Internals (P1):

- `nitrix._internal.config` — a trace-time-readable reproducibility flag held in
  a `contextvars.ContextVar` (initialised from `NITRIX_REPRODUCIBLE`), a
  `reproducible()` context manager, and `reproducible_enabled()`.
- `resolve_driver(local, *, op, canonical, fast)` — the single resolver every
  divergent site calls: returns `local` if not `'auto'`, else `canonical` if
  reproducible mode, else `fast()` (the hardware pick). Read at trace time, so
  `jit`-safe (the chosen variant is baked into the traced graph — see the
  caveat below).
- A registry: each site registers its `DivergentOp(op, canonical, fast,
  driver_values, tolerance)` at import; `nitrix.divergent_ops()` lists it, and a
  completeness guard (P4) fails CI if a `default_backend_is_gpu()` /
  `resolve_driver` site is unregistered.

**`jit` caveat (documented, not silently wrong):** like `jax.config` flags and
the existing `default_backend_is_gpu()`, the reproducibility flag is read at
*trace* time. A function traced outside `reproducible()` and called inside (from
the `jit` cache) keeps its traced variant. Set the mode before first trace
(deployment env var is the robust path). A future refinement can fold the flag
into the `jit` cache key; out of scope for P0.

## 5. The contract (what makes this real, not aspirational)

1. **Tested tolerances.** `tests/tolerance.toml` already pins
   `[op.dtype.backend]` for `pallas ≈ jax`; extend the same mechanism to
   `[op.dtype.driver]` cross-variant budgets, and add golden tests asserting
   `variant ≈ canonical` within them. A divergence that is *bounded and tested*
   is a contract, not a surprise.
2. **Honest docs.** Every divergent public function documents its cross-platform
   tolerance and its `driver=`. **Fix #5's docstring** (delete the false "agree
   to a ULP" — replace with the measured cross-variant tolerance).
3. **Enforcement guard (P4).** A completeness test (sibling to
   `test_op_matrix_completeness`) asserts every `resolve_driver`/
   `default_backend_is_gpu()` site is in the registry with a tolerance. This is
   what keeps the principle alive past this change.

## 6. Per-site rollout ledger

| # | Site | Change |
|---|------|--------|
| 1 | `nn.ssm.selective_scan` | `method=` → `driver=` (alias); route through `resolve_driver`; register; already has cross-variant test — pin its tolerance row |
| 2 | `signal._iir` ×3 | `backend=` → `driver=` (deprecation alias); route; register; keep the loud FFT-fallback (orthogonal) |
| 3 | `register._svf` gaussian | expose `driver=` on the public smoother; route the internal `_smooth_method`; canonical `fir`; register + tolerance row for the ~1–2% edge band |
| 4 | `metrics.information` histogram | expose `driver=`; reproducible → force `onehot` regardless of size; register; fold the existing determinism test into the contract |
| 5 | `geometry.CubicBSpline` | add a `driver` field (frozen record); route the prefilter; **fix docstring**; register + cross-variant tolerance row — *resolves the ilex FR* |

## 7. Rollout plan

- **P0 — this doc. ✅ SHIPPED** (commit a7fcbc4). Principle + per-site ledger.
- **P1 — substrate. ✅ SHIPPED** (commit b531955). `_internal/config.py`
  (reproducibility contextvar + env seed + `reproducible()` / `set_reproducible`
  / `reproducible_enabled`), `resolve_driver`, the `DivergentOp` registry +
  `nitrix.divergent_ops()`, public surface at the package root. 16 tests; no
  site behaviour changed yet (registry empty until P2 wires the real ops).
- **P2 — retrofit the 5 sites. ✅ SHIPPED** (commits 5b8f084, e5e3a95, ee8e84f,
  6f25518). All five route through `resolve_driver` and are registered in the
  central manifest (cold `divergent_ops()` lists all five). **Clean break, no
  deprecation shim** (all callers in-ecosystem, per the go-ahead): `signal`
  `backend=`→`driver=`, `ssm` `method=`→`driver=`, `smoothing.gaussian`
  `method=`→`driver=`; `CubicBSpline` gained a `driver` field; the gaussian-
  regulariser and histogram auto-picks now route through the axis. Added the
  backend-axis half: `auto_backend()`→`jax` under reproducibility mode (explicit
  `backend=`/env overrides). Verified on L4 (each variant diverges within its
  registered budget; `reproducible()` forces the canonical / deterministic path);
  demons-recipe + metrics regression byte-identical in normal mode (49 passed).
  *Per-recipe `driver=` threading for the gaussian/histogram sites deferred —
  the reproducibility mode is the lever; the underlying `_smooth_method` /
  `_joint_hist_from_softbins` already accept `driver`.*
- **P3 — contract. ✅ SHIPPED** (commit 84689f3). `tests/test_reproducible_
  dispatch_contract.py`: every variant of every registered op asserted
  ``variant ≈ canonical`` within the **registry** tolerance (read from
  `divergent_ops()` directly -- the registry is the single source of truth, no
  `tolerance.toml` mirror), plus a sync guard that fails if a new op lacks a
  runner. Tolerances measured + pinned honestly (iir/ssm/cubic ~ULP; histogram
  f32 budget 2e-4; `register.field_smooth` 3e-2, the looser FIR-vs-YvV
  approximation contract on a boundary-decaying field). CubicBSpline docstring
  fixed in P2. 7 tests pass.
- **P4 — enforcement guard.** Registry-completeness CI test.

The ilex request (`cubic-bspline-prefilter-backend-parity`) is satisfied by
P2–P3 for free: honest docstring + tested tolerance + global switch + a `driver`
field — all three options it offered, at once.

## 8. SPEC integration (on approval)

- Add SPEC §2 tenet (the boxed principle in §2 above).
- Amend tenet #3 to name both axes (`backend=` execution, `driver=` variant).
- Amend tenet #6 to include cross-platform / cross-run, scoped honestly (§2
  "does not promise bit-equality across hardware").

## Cross-references

- [`cubic-bspline-prefilter-backend-parity`](cubic-bspline-prefilter-backend-parity.md)
  — the ilex driver (site #5).
- [`interpolation-backend-cpu-gpu-gap`](interpolation-backend-cpu-gpu-gap.md)
  — the order-1 *perf* sibling (mechanically-equivalent; **exempt**).
- `src/nitrix/_internal/backend.py` (`resolve_backend`, `default_backend_is_gpu`,
  the `fallback` machinery this principle's resolver mirrors).
- Sites: `src/nitrix/nn/ssm/_reference.py:202`, `src/nitrix/signal/_iir.py:71`,
  `src/nitrix/register/_svf.py:245`, `src/nitrix/metrics/information.py:104`,
  `src/nitrix/geometry/_interpolate.py:611`.
