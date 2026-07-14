# Per-device autotuning and a dispatch perf floor

*Feature request / design. Concerns the fused (`pallas-cuda`) tier and the
backend-resolution seam (`_internal/backend.py`). Motivated by a measured regression:
a fused kernel hand-tuned on one device (L4, Ada / sm_89) running **slower than the
JAX reference** on another in-scope device (A100, Ampere / sm_80).*

---

## 1. The defect, stated as an invariant violation

SPEC's dispatch tenet is explicit: **"fallbacks are loud — a silent performance
regression is a bug."** The current dispatch violates it structurally.

- `auto_backend()` resolves `pallas-cuda` on **any** `sm ≥ 8.0` device
  (`backend.py:196`), a pure hardware-*family* gate computed once at import
  (`_HAS_AMPERE_NVIDIA`, `backend.py:133`). The embedded assumption is
  **"Ampere+ ⟹ the fused kernel beats the reference."**
- The loud fallback fires **only** when `requested != resolved` (`backend.py:345`),
  i.e. when the fused path *cannot run* (untileable shape, non-fp32, non-power-of-two).
  A fused path that **runs but loses** to the reference never trips it.

So on A100 the kernel — whose launch parameters were fixed on L4, or for SRAM-fit, or
for sm_80 depending on the kernel — runs, loses to XLA's own lowering (the "reference"),
and emits nothing. That is the "silent performance regression" the tenet names as a bug.
It is the assumption failing for the first time someone measured the untested device.

## 2. Why it recurs — it is structural, not incidental

The launch parameters encode one machine and are frozen, one config for all silicon
from Ampere through Blackwell:

| kernel | frozen tuning | provenance |
|---|---|---|
| attention (`_kernels/cuda/attention.py:68,249,257`) | `_BLOCK=32`, `num_stages=2`, `num_warps = 4 if d≤64 else 8` | `_BLOCK` chosen to **fit SRAM** ("~99 KB … Ampere/Lovelace SMs"); the file says *"perf-tuning the tile ladder is the sibling perf suite's job"* |
| semiring_matmul (`_kernels/cuda/semiring_matmul.py:65–71`) | block ladder `(128,64,32,16)×(32,16,8)` | *"an experimental sweet spot on **Ampere (sm_80)** cards"* — tuned for A100 |
| selective_scan (`_kernels/cuda/selective_scan.py:192,371`) | `num_warps=4, num_stages=1` | hardcoded |

A `grep` for any device branch (`sm_80`, `device_kind`, `A100`, `L4`, …) **inside** the
kernels returns zero. There is no per-device tuning, no perf measurement, no cache.

The reason one config cannot serve is physical. "Ampere+" spans A100 (sm_80, HBM2e
~2 TB/s, 108 SMs), L4 (sm_89, GDDR6 ~300 GB/s, 58 SMs), and aspirationally Hopper. A
tile that fits L4's SRAM and hides its (scarce) bandwidth is the wrong tile for A100,
whose tensor cores want bigger blocks and deeper pipelining (`num_stages` 3–4, not 2) to
amortise its far higher bandwidth. The optimum diverges by 2–4× in tile dimension. So:

**No kernel is tuned for the device it happens to run on, in general** — and the count of
`(kernel × device)` cells grows multiplicatively with every part admitted. Hand-tuning is
`O(kernels × devices)` of human effort; the problem is systemic and wants a systemic fix.

There is also a self-selection at work, the same one the post-buildout RFC §6.5.1 names:
the dev box **is** the L4 (the CUDA suites are run there), so the A100 regime is the one
that cannot be observed from the bench. The guarantee "pallas is faster" was certified
only on the device its author could test — the recurring **X-6** pattern (a claim
certified only on the path its author took), now in silicon.

## 3. Design — one substrate, two tiers

The perf **floor** and the **autotuner** are the same machine at two candidate-set sizes.
Build one substrate: **measure candidates ahead-of-time, cache the winner persistently,
keyed on the device.**

### 3.1 The cache key

```
(function, dtype, shape_bucket, device_key)
```

- `function` — the op id (`'scaled_dot_product_attention'`, `'semiring_matmul'`, …), the
  same string the dispatchers already pass to `fallback()`.
- `dtype` — precision is a first-class axis (SPEC): tensor-core throughput per dtype is
  exactly what differs across parts.
- `shape_bucket` — shapes bucketed (e.g. per-dim round-up to the next power of two, or the
  op's own tileability classes). You cannot tune every shape; the bucket granularity is a
  knob, coarse by default.
- `device_key` — the **specific** device, not "cuda": `device_kind`
  (`"NVIDIA A100-SXM4-40GB"` vs `"NVIDIA L4"`) plus `compute_capability`. This is the axis
  the current gate collapses.

### 3.2 Measurement — ahead-of-time, never at trace time

Measurement is **AOT and eager**: `jax.jit(fn).lower(*args).compile()` once, then time the
compiled callable warm (median of N, isolate compile from run-time). This is the discipline
the GPU-verify practice already prescribes, and the reason it is mandatory here is a scar:
**the genred work found the XLA fusion autotuner was itself the compile pathology** and
pinned `--xla_gpu_autotune_level=0`. An in-house tuner that measured inside a trace would
reintroduce exactly that cost. Measurement therefore runs from an explicit calibration
entry point (and/or an eager first-touch out of band), and the **result is persisted to
disk** so it is paid once per device, ever — not once per process.

### 3.3 Tier A — the perf floor (candidates = {default-pallas, jax})

For each fused op at a few representative buckets, measure the *existing* pallas config
against the reference, once per device; cache the winner. At dispatch, a dict lookup: if
the table says the reference wins for this `(function, bucket, device)`, resolve to `jax`
and emit the **existing** `NitrixBackendFallback` — same channel, same dedup, same
`STRICT`/`SILENCE` knobs — with a new reason: *"measured slower than the reference on this
device."* Small, and it restores the tenet: the A100 regression becomes a correct choice
plus a loud warning.

Uncalibrated `(function, bucket, device)` defaults to **pallas** — i.e. exactly today's
behaviour — so shipping Tier A changes nothing until a device is calibrated. Zero
regression risk from adoption.

### 3.4 Tier B — the autotuner (candidates = {pallas-config-0…N, jax})

The same cache, a richer candidate set: enumerate a **curated** list of
`(block, num_warps, num_stages)` per kernel — the author knows the plausible handful; this
is not a search over a large space — measure, keep the winner per bucket/device. This is
already specified in the genred backlog **M3.7** with precisely the right key: *"measure
tile/warp/stage candidates per (spec fingerprint, shapes, device) with a persistent cache;
measured per-class heuristics stand in meanwhile."* The work is to **lift that one cache
out of genred** and apply it to the standalone `_kernels/cuda/` kernels, so there is one
autotuner, not per-kernel ad-hoc tuning.

### 3.5 They compose

Tier A is Tier B with `|candidates| = 1`. Build the measure-cache-persist substrate once;
the floor is the degenerate candidate set, autotune is the full one — and the floor
**guarantees you are never below the reference even when autotune has no entry for a
shape.** Autotune optimises; the floor is the safety net.

## 4. Interactions

- **Reproducibility — clean, and this is the pleasant surprise.** `auto_backend()` already
  returns `'jax'` under `NITRIX_REPRODUCIBLE` (`backend.py:194`). Reproducible mode never
  touches the fused tier, so the tuner is *orthogonal* to it: no new governed divergent
  site, no interaction to reason about. Non-reproducible = pallas + tuning; reproducible =
  jax, deterministic.
- **fp accumulation order — the one real subtlety.** Different tiles change reduction order
  → different round-off. The golden corpus is per `(op, dtype, backend)`, "loosened for
  fused paths"; tile-induced round-off must stay inside that band, and if autotune picks
  different tiles per device, different devices land at different points *within* the band.
  Acceptable for tolerance, but **golden references must be generated against a pinned
  canonical config**, not the autotuned one, so the oracle does not chase the tuner. A
  tolerance change remains a public-API change.
- **Governance — the dispatch becomes empirical, not asserted.** This is the conceptual
  shift: the backend decision moves from "assume pallas wins on the family" to "know, per
  device, whether it does." That is X-6 discharged in the dispatcher — measure before you
  claim.

## 5. Non-goals

- **Not a general autotuning framework.** Curated per-kernel candidate lists, a coarse
  bucket, a dict cache. No search infrastructure.
- **No trace-time / per-call measurement.** AOT and cached only (§3.2).
- **No new dtype behaviour.** Precision stays governed by input dtype and explicit knobs;
  the tuner never changes dtype.
- **Does not replace the sibling perf-bench suite.** That suite certifies wall-clock vs
  external tools; this subsystem makes an *in-library dispatch decision*. They inform each
  other but are distinct.

## 6. Acceptance

1. On a device where the fused op loses, dispatch resolves to `jax` and a
   `NitrixBackendFallback` fires with the "measured slower" reason — asserted in a test that
   **injects** a perf table (does not depend on real hardware timing).
2. Uncalibrated behaviour is byte-identical to today (default pallas).
3. The cache round-trips to disk and is keyed on the specific device; two device kinds get
   two entries.
4. Reproducible mode is unaffected (still `jax`, no cache consulted).
5. Goldens unchanged (generated against the pinned canonical config).

## 7. Rollout

Ship **Tier A** first — small, restores a stated invariant, converts the A100 regression
into a correct choice + a loud warning. Then **Tier B**, generalising the genred M3.7
cache across all fused kernels. Lives in nitrix core (`_internal/autotune.py`) — it is
hardware-awareness infrastructure, not a numerical primitive, so not a moonshot filing.

---

## 8. Tier-A sketch (illustrative — not committed to `src/`)

Shows the blast radius: one new `_internal/` module, one helper in `backend.py`, and a
2–3 line insertion at each dispatcher. The existing `fallback()` is reused verbatim.

### 8.1 New module `_internal/autotune.py`

```python
"""Per-device perf floor (Tier A) and, later, launch-param autotuning (Tier B).

Measures fused-vs-reference AHEAD OF TIME, caches the winner per
(function, dtype, shape-bucket, device), and persists it. Never measures inside a
trace (that reintroduces the XLA-autotuner compile pathology). Uncalibrated keys
default to the fused path, so importing this changes nothing until calibrate() runs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

import jax

_Winner = str  # 'pallas-cuda' | 'jax'  (Tier B: 'pallas-cuda#<config-id>')
_Key = tuple[str, str, str, str]  # (function, dtype, shape_bucket, device_key)

_table: dict[_Key, _Winner] = {}
_loaded = False


def device_key() -> str:
    """Identity of the current GPU — the axis the sm>=8 gate collapses."""
    try:
        d = jax.devices('gpu')[0]
    except (RuntimeError, IndexError):
        return 'cpu'
    kind = (getattr(d, 'device_kind', '') or 'gpu').replace(' ', '_')
    cc = getattr(d, 'compute_capability', None)
    return f'{kind}@{cc}' if cc is not None else kind


def _pow2_ceil(n: int) -> int:
    return 1 << (max(1, n) - 1).bit_length()


def shape_bucket(shapes: tuple[tuple[int, ...], ...]) -> str:
    """Coarse bucket: per-dim round-up to a power of two. Granularity is a knob."""
    return ';'.join('x'.join(str(_pow2_ceil(d)) for d in s) for s in shapes)


def _cache_path() -> Path:
    root = os.environ.get('NITRIX_CACHE_DIR') or (Path.home() / '.cache' / 'nitrix')
    return Path(root) / 'perf_table.json'


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    p = _cache_path()
    if p.exists():
        for k, v in json.loads(p.read_text()).items():
            _table[tuple(k.split('|'))] = v  # type: ignore[assignment]
    _loaded = True


def prefers_reference(
    *, function: str, shapes: tuple[tuple[int, ...], ...], dtype: Any
) -> bool:
    """True iff a measurement says the reference wins here. Uncalibrated -> False."""
    _ensure_loaded()
    key = (function, str(dtype), shape_bucket(shapes), device_key())
    return _table.get(key) == 'jax'


def record(
    *, function: str, shapes: tuple[tuple[int, ...], ...], dtype: Any, winner: _Winner
) -> None:
    _ensure_loaded()
    _table[(function, str(dtype), shape_bucket(shapes), device_key())] = winner
    p = _cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({'|'.join(k): v for k, v in _table.items()}, indent=2))


def measure_winner(
    pallas_fn: Callable[..., Any], reference_fn: Callable[..., Any], *args: Any,
    reps: int = 30,
) -> _Winner:
    """AOT-compile both, time warm, return the faster. Eager only — never in a trace."""
    # timeit_compiled() := lower(*args).compile() then median of `reps` warm runs,
    # each block_until_ready()'d; isolate compile from runtime (the CUDA-verify rule).
    ...  # returns 'pallas-cuda' or 'jax'
```

### 8.2 Helper in `backend.py`

```python
def apply_perf_floor(
    resolved: ResolvedBackend, *, function: str,
    shapes: tuple[Any, ...], dtype: Any,
) -> ResolvedBackend:
    """Downgrade a fused resolution to 'jax' when it is measured slower here.

    Reuses fallback() verbatim, so the warning, dedup key, and STRICT/SILENCE
    knobs are exactly those of the can't-run path. No-op unless resolved is
    'pallas-cuda' and the perf table has an entry saying the reference wins.
    """
    if resolved != 'pallas-cuda':
        return resolved
    from .autotune import prefers_reference
    if not prefers_reference(function=function, shapes=shapes, dtype=dtype):
        return resolved
    return fallback(
        function=function, requested='pallas-cuda', resolved='jax',
        reason='the fused kernel measured slower than the reference on this device',
        shapes=shapes, dtype=dtype,
    )
```

### 8.3 Dispatcher integration — the whole change at each call site

```python
# nn/attention/__init__.py, replacing line 192:
resolved = apply_perf_floor(
    resolve_backend(backend),
    function='scaled_dot_product_attention',
    shapes=(tuple(q.shape), tuple(k.shape), tuple(v.shape)),
    dtype=q.dtype,
)
# ...the existing `if resolved == 'pallas-cuda': ...` block is unchanged.
```

```python
# semiring/matmul.py, at the resolve site (~line 220):
resolved = apply_perf_floor(
    resolve_backend(backend),
    function='semiring_matmul',
    shapes=(tuple(A.shape), tuple(B.shape)),
    dtype=A.dtype,
)
```

### 8.4 Blast radius

- **+1 file**: `_internal/autotune.py` (~90 lines, Tier A).
- **+1 function**: `apply_perf_floor` in `backend.py` (~15 lines); nothing existing changes.
- **~2 lines each** at the fused dispatchers (attention, semiring_matmul; norm and
  semiring_ell already always fall back, so they need nothing).
- **0 changes** to the kernels, to `fallback()`, to the reproducibility path, or to goldens.
- **1 new public entry**, `nitrix.calibrate()`, to populate the table on a box (thin wrapper
  over `measure_winner` + `record` for each fused op × representative bucket).

Uncalibrated, every path is byte-identical to today. The floor engages only where a
measurement exists and says the reference wins.
