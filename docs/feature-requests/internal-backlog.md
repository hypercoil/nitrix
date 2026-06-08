# nitrix internal backlog — ledger & index

> **This doc is now the engineering-backlog *ledger + index*.** Each open
> item has been atomised into its own tracking doc (one doc per item, to
> reduce duplicate-issue risk); this file keeps the shared framing, the
> **Closed by design** decision record, and the **Resolved** history, plus
> the index of the atomised open items. See [`README.md`](README.md) for the
> directory-wide index.

Parked engineering items — perf characterisation, Pallas kernels, kernel tuning,
research notes, and small API refinements — revisited when a concrete consumer
ask, perf regression, or phase boundary surfaces them. **Not commitments**:
parked because no consumer is currently blocked, or the cost/payoff favours
waiting for more signal. Each atomised item names the **Trigger** that would
promote it to a sprint.

Migrated 2026-06-02 from the retired top-level `BACKLOG.md` (B-numbering kept so
existing references stay valid) plus the one genuinely-open leftover from the
sealed `NITRIX_FEEDBACK_JOSA.md`. When an item graduates, move it to
`IMPLEMENTATION_PLAN.md` (or fold into an open phase) and leave a one-line pointer
in *Resolved* below.

## Open items (atomised)

| Item | Doc | Kind | Effort |
|---|---|---|---|
| B2 | [perf-bench-sprint-surfaces](perf-bench-sprint-surfaces.md) | perf characterisation | — |
| B3 | [pallas-dispatch-edge-aggregate](pallas-dispatch-edge-aggregate.md) | Pallas dispatch (measured: not GPU-motivated) | — |
| B4 | [edge-aggregate-log-euclidean](edge-aggregate-log-euclidean.md) | semiring coverage (canonical) | — |
| B5 | [keops-genred-research](keops-genred-research.md) | research note | — |
| B6 | [pallas-gaussian-blur](pallas-gaussian-blur.md) | Pallas kernel | — |
| B7 | [pallas-trilinear-resample](pallas-trilinear-resample.md) | Pallas kernel (interim JAX gather **shipped** 2026-06-07; kernel parked) | — |
| B10 | [retune-pallas-log-matmul](retune-pallas-log-matmul.md) | kernel tuning | M |
| B11 | [perfbench-migration](perfbench-migration.md) | tooling migration (in progress) | L (sliceable) |
| B12 | [iir-filter-gpu-backend](iir-filter-gpu-backend.md) | perf / API-default (IIR GPU backend) | S + M |
| B13 | [boundary-mode-parity](boundary-mode-parity.md) | API refinement (scipy/ITK boundary parity) | M |
| B14 | [spectral-embedding-gpu-solver](spectral-embedding-gpu-solver.md) | perf + robustness (lobpcg / eigh on GPU) | M |
| B15 | [interpolation-backend-cpu-gpu-gap](interpolation-backend-cpu-gpu-gap.md) | perf characterisation (map_coordinates CPU/GPU) | S char / M-L fix |
| B16 | [alternative-interp-backends-xla](alternative-interp-backends-xla.md) | research note (scipy/cupy backends in XLA) | M-L |
| B17 | [median-percentile-cpu-sort-cliff](median-percentile-cpu-sort-cliff.md) | perf characterisation (jnp.median/percentile CPU sort) | M |
| B18 | [perf-bench-case-hardening](perf-bench-case-hardening.md) | benchmark-integrity report (gameable hard-path branches) | S (report) |
| B20 | [distance-transform-anisotropic-sampling](distance-transform-anisotropic-sampling.md) | feature gap (euclidean EDT has no sampling=) | S |
| B21 | [morphology-explicit-se-im2col-cost](morphology-explicit-se-im2col-cost.md) | perf characterisation (explicit-SE/disk-footprint im2col cost, measured) | M |
| B22 | [register-sparse-dataclasses-as-pytrees](register-sparse-dataclasses-as-pytrees.md) | API/ergonomics (ELL/SectionedELL/Mesh not registered pytrees) | S-M |
| B23 | [perf-wins-must-certify-at-scale](perf-wins-must-certify-at-scale.md) | benchmark-integrity principle (a win must certify at brain scale, not the benched size) | S (principle) |
| G1 | [spatial-transform-linear-extrap](spatial-transform-linear-extrap.md) | boundary-mode extension | S |

(B1, B8, B9, and B19 are resolved — see below. `spatial_transform_batched`, JOSA §3,
shipped 2026-06-02 — see `IMPLEMENTATION_PLAN.md §10.3`;
[`spatial-transform-batched.md`](spatial-transform-batched.md) is its home doc.)

## Closed by design (recorded so the decision isn't lost on file deletion)

Items requested in the retired ledgers that were **deliberately not built** — kept
here as the rationale record:

- **`dilate` / `erode` with `padding='periodic'`** (JOSA §6). Rejected: the
  `dilate`/`erode` docstrings (`src/nitrix/morphology/_mm.py`) document the
  composition recipe instead — `jnp.pad(mask, mode='wrap')` → `dilate(padding='VALID')`
  → unpad — so spherical/toroidal boundaries work without threading a new mode
  through every morphology op.
- **`gaussian_dense_2d` (non-separable 2-D kernel)** (JOSA §5 option 2). Superseded
  by the shipped `gaussian(kernel_size=...)` override, which reaches the small-window
  case while preserving separability.
- **Out of nitrix scope** (correctly): trained-atlas-as-tensor (→ nimox `eqx.Module`
  field), FreeSurfer `.sphere`/`.mgz` I/O and surface↔sphere parameterisation
  (`surfa`) (→ consumer / thrux; SPEC §5.2), TF-checkpoint key renaming (→
  `ilex.core.adapters`).

## Resolved items

Full resolution history for the consumer-driven gaps (gaussian docstrings, the
edge-aggregate / icosphere-hierarchy / mesh-pool stack, `max_pool_with_indices_nd` /
`max_unpool_nd`, SUGAR deltas, GATv2 self-loops, Nyul–Udupa `histogram_match`) lives
in **`IMPLEMENTATION_PLAN.md §10.3`** (shipped-deviation log) and
**`SPEC_UPDATE_v0.3 §10.A`** — these supersede the retired `NITRIX_FEEDBACK_ILEX.md`
/ `NITRIX_FEEDBACK_JOSA.md` ledgers. Backlog items closed here:

- **B1. Move resolved findings in `NITRIX_FEEDBACK_ILEX.md`** — done 2026-05-20.
- **B8. Explicit `(*)`-annihilator on `Semiring`** — resolved 2026-06-02. Added an
  `annihilator` field to `Semiring` (`None` for EUCLIDEAN; equal to `identity` for the
  other built-ins) and an `ell_mask(semiring=)` path that reads it and raises clearly when
  `None`. The legacy `ell_mask(identity=...)` form is retained but now emits a
  `DeprecationWarning`. The masking-vs-identity footgun (`docs/design/semiring-protocols.md`)
  is now machine-checked. See `IMPLEMENTATION_PLAN.md §10.3` (2026-06-02 entry).
- **B19. `erode`/`dilate` flat-SE fast path breaks `jit(grad(...))`** — resolved
  2026-06-07. Root cause: the flat-box `lax.reduce_window` fast path passed a
  *traced* window init (`jnp.asarray`), which fails JAX's monoid-detection
  concreteness test and falls back to the generic `reduce_window_p` (no transpose
  rule) — so `jit(grad)` raised "Linearization failed …" while eager `grad`
  stayed green. Fix: pass a **concrete** init sourced from the algebra identity
  via `np.asarray`, routing to JAX's own differentiable
  `reduce_window_max_p`/`min_p` (no `custom_vjp` needed); folded in a uniform
  int/bool → `float32` promotion (`_to_float`) for a `float-in → float-out`
  contract across both paths. Gated by three new `test_flat_path_*` tests; the
  four `jit_of_grad` op-matrix cells flip `ValueError → pass`. See
  [`morphology-reduce-window-jitgrad.md`](morphology-reduce-window-jitgrad.md)
  (Resolution) and `docs/design/morphology.md` (flat-path section).
- **B7 (interim win) + `upsample-nearest-nd` — partially resolved 2026-06-07.**
  The interpolation-method dispatcher (`geometry/_interpolate.py`) shipped the
  B7 "explicit 8-corner gather" as `_separable_gather` — now the GPU engine
  for `Linear` / `NearestNeighbour` and the always-engine for `Lanczos`,
  **platform-branched** (`map_coordinates` on CPU, where the gather regresses;
  the `signal._iir` precedent). This also delivers `upsample-nearest-nd`'s
  capability (`resample(method=NearestNeighbour())`) and `point-sample`'s
  zero-fill sampling (`spatial_transform(mode='constant', cval=0)`). B7's
  **Pallas pointer-load kernel stays parked** (the gather-lowering risk is
  unchanged); B15's CPU throughput gap stands (the branch keeps CPU on
  `map_coordinates`). See `IMPLEMENTATION_PLAN.md §10.3` (2026-06-07) and
  `docs/design/geometry.md`.
- **B9. `residualise` robustness on ill-conditioned designs** — resolved 2026-05-22.
  Root cause: default `method='cholesky'` returns NaN for rank-deficient `X` (the
  singular Gram has no factor); the shipped `method='svd'` path (min-norm LSQ via
  `jnp.linalg.lstsq`) is exact there. Resolution was verification + docs:
  `tests/test_resid.py` exercises the SVD path across `p→obs` / `p>obs` and pins the
  cholesky-NaN / svd-finite contract; `method` docstring documents the min-norm
  semantics + failure mode. Default kept `cholesky` (~2× faster on the
  well-conditioned case); `svd` is the documented robust opt-in. See
  `IMPLEMENTATION_PLAN.md §10.3`.
