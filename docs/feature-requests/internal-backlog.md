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
| B7 | [pallas-trilinear-resample](pallas-trilinear-resample.md) | Pallas kernel (cheap JAX interim win) | — |
| B10 | [retune-pallas-log-matmul](retune-pallas-log-matmul.md) | kernel tuning | M |
| B11 | [perfbench-migration](perfbench-migration.md) | tooling migration (in progress) | L (sliceable) |
| B12 | [iir-filter-gpu-backend](iir-filter-gpu-backend.md) | perf / API-default (IIR GPU backend) | S + M |
| G1 | [spatial-transform-linear-extrap](spatial-transform-linear-extrap.md) | boundary-mode extension | S |

(B1, B8, and B9 are resolved — see below. `spatial_transform_batched`, JOSA §3,
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
- **B9. `residualise` robustness on ill-conditioned designs** — resolved 2026-05-22.
  Root cause: default `method='cholesky'` returns NaN for rank-deficient `X` (the
  singular Gram has no factor); the shipped `method='svd'` path (min-norm LSQ via
  `jnp.linalg.lstsq`) is exact there. Resolution was verification + docs:
  `tests/test_resid.py` exercises the SVD path across `p→obs` / `p>obs` and pins the
  cholesky-NaN / svd-finite contract; `method` docstring documents the min-norm
  semantics + failure mode. Default kept `cholesky` (~2× faster on the
  well-conditioned case); `svd` is the documented robust opt-in. See
  `IMPLEMENTATION_PLAN.md §10.3`.
