# nitrix internal backlog

Parked engineering items — perf characterisation, Pallas kernels, kernel tuning,
research notes, and small API refinements — revisited when a concrete consumer
ask, perf regression, or phase boundary surfaces them. **Not commitments**:
parked because no consumer is currently blocked, or the cost/payoff favours
waiting for more signal. Each item names the **Trigger** that would promote it to
a sprint.

Migrated 2026-06-02 from the retired top-level `BACKLOG.md` (B-numbering kept so
existing references stay valid) plus the one genuinely-open leftover from the
sealed `NITRIX_FEEDBACK_JOSA.md`. When an item graduates, move it to
`IMPLEMENTATION_PLAN.md` (or fold into an open phase) and leave a one-line pointer
in *Resolved* below.

## Open items

### B2. Perf-bench the Sprint A / Sprint B surfaces

Add bench scripts (now under `nitrix-perf-bench`) for `semiring_ell_edge_aggregate`
(GCN fwd+bwd, DGCNN fwd+bwd), `max_pool_with_indices_nd` / `max_unpool_nd`, and the
mesh wrappers at icosphere-realistic shapes (`ico_6 = 40962`, `ico_7 = 163842`
verts). Wire results into the op-matrix `perf_ratio` column.

**Trigger.** First concrete consumer running these in a training loop where
wall-time matters (Topofit cascade at `ico_7`, any SphereMorph descendant, any
PGlandsSeg-lineage encoder-decoder UNet). Or: a Pallas-dispatch experiment (B3)
needing a baseline.

**Notes.** The op matrix shows all four transforms passing on these surfaces;
what's missing is wall-time vs a natural reference — `semiring_ell_edge_aggregate`
vs PyTorch-Geometric `MessagePassing` (host parity, JAX GPU timing);
`max_pool_with_indices_nd` vs `torch.nn.MaxPool3d(return_indices=True)`.

### B3. Pallas dispatch for `semiring_ell_edge_aggregate`

The implementation is JAX-only (gather + nested vmap + semiring axis-reduce). At
Topofit scale (`ico_7`, `d_in=64-128`) the inner edge-MLP is a tiled matmul that
*should* map to the existing `semiring_matmul` Pallas backend.

**Trigger.** B2 baseline shows the JAX path is the bottleneck in a real training
loop, or a consumer explicitly asks.

**Notes.** Hard part: `edge_fn` is a user Python callable, so no precompiled
kernel — needs either (a) a Pallas template parameterised by a fixed-shape MLP
descriptor (works for DGCNN-shaped `edge_fn`s, not generic) or (b) the KeOps-style
symbolic-formula approach (B5).

**Measured signal (2026-05-23, nitrix-perf-bench).** Partially met for a GCN-style
*linear* `edge_fn` (`w·(W@h_j)`): on **CPU**, torch_geometric `MessagePassing` is
~2–5× faster than the nitrix-jax reference (0.21× at n=4096 → 0.53× at n=16384,
degree 16, d=64), both REAL and TROPICAL_MAX_PLUS — the cost of the gather +
nested-vmap + axis-reduce generality vs PyG's native `scatter_reduce`; narrows with
graph size. **But on the A10G (deployment target) it is a wash** — nitrix-jax within
~5% of PyG for REAL and ~15% *faster* for TROPICAL_MAX_PLUS; XLA fuses the
gather+vmap+reduce well on Ampere. Net: a Pallas dispatch is **not** motivated by
GPU perf vs PyG; the only standing case is CPU (likely not the target) or an
`edge_fn` shape XLA fuses poorly. Keep low-priority until a GPU shortfall or
concrete consumer appears. See `nitrix-perf-bench/reports/PERF_ELL_EDGE_AGGREGATE.md`.

### B4. LOG / EUCLIDEAN / BOOLEAN `edge_aggregate` semirings

`semiring_ell_edge_aggregate` supports REAL, TROPICAL_MAX_PLUS, TROPICAL_MIN_PLUS;
LOG / EUCLIDEAN raise `NotImplementedError` with a specific message
(`src/nitrix/semiring/ell_edge.py:143`); BOOLEAN is excluded (most `edge_fn`s
produce floats). *(Also surfaced consumer-side as the residual item E in
`ilex-pipeline-substrate.md`; this is the canonical entry.)*

**Trigger.** A consumer with a concrete use case where one composes naturally —
most likely LOG for attention-style softmax aggregation, though the
`ell_row_softmax`-then-REAL composition is usually cleaner than a fused logsumexp
at the edge.

**Notes.** The per-edge message shape is user-controlled, so the algebra's
identity-action on the message must be user-respected (same contract as the
existing semirings). LOG: `logsumexp` over the neighbour axis on
`log(edge_fn_output)`-type messages. EUCLIDEAN: `sqrt(sum_p edge_fn**2)`-shaped.

### B5. KeOps `Genred`-primitive research

KeOps's `Genred` takes a user formula in a string DSL and JIT-compiles a CUDA
kernel that streams the reduction without materialising the per-edge tensor. The
SPEC §3.1 "KeOps-style streaming kernel" claim invokes this lineage; what's missing
is a concrete research note on whether a Pallas-backed analogue could host
arbitrary user formulas (the `edge_aggregate` generic case).

**Trigger.** B3 forces a decision between "Pallas template per `edge_fn` shape" and
"compile a formula on the fly". Or a follow-up SPEC update revisits formula
compilation independently.

**Notes.** Research deliverable — a design doc covering: (1) KeOps's formula DSL
(LazyTensor + reduction primitives); (2) the JAX/Pallas analogue (jaxpr-level
analysis of a callable? Pallas templating via concrete-shape inputs?); (3) the
boundary between compilable (closed polynomial / elementwise) and non-compilable
(data-dependent control flow, dynamic shapes) formulas; (4) whether this belongs in
nitrix or a separate package.
**Cross-refs.** `docs/design/streaming-kernel.md`, `docs/design/ell-on-triton.md`;
KeOps paper Charlier et al. 2021 (JMLR).

### B6. Pallas kernel for the Gaussian blur primitive

`smoothing.gaussian` is a separable n-D conv via `lax.conv_general_dilated` (one
1-D pass/axis), lowering to cuDNN on Ampere+ — a strong baseline.

**Trigger.** A consumer with a wall-time wall on large-3-D Gaussian blur (e.g.
repeated 256³ smoothing in a training loop), *and* a benchmark showing the
separable-conv path is the bottleneck.

**Notes.** Gaussian is a stencil (not a gather), so Pallas-friendly. But cuDNN is
hard to beat per-pass; the only real win is **fusing the 3 separable axis passes**
(+ boundary pad) into one kernel to save inter-pass HBM round-trips on large
volumes — marginal and bandwidth-bound; do not build speculatively. Any kernel
ships behind `backend=` with the `conv_general_dilated` JAX floor (non-negotiable
§2.2.3) and a golden-corpus parity test; bench against `conv_general_dilated`, not a
naive loop.

### B7. Pallas kernel for 3-D trilinear resampling

`geometry.spatial_transform` / `integrate_velocity_field` resample via
`map_coordinates(order=1)`. (See also `ilex-pipeline-substrate.md` item D — cubic
order-3 resampling is a separate, parity-driven gap.)

**Trigger.** Both of: (a) the baseline shows resampling is a real bottleneck in a
consumer training loop, and (b) a pointer-load (`pl.load`/`pl.ds`) Pallas prototype
clears the gather-lowering risk — i.e. avoids the HLO `gather` primitive that Triton
on the pinned JAX cannot lower (the ELL blocker; `bench/G0_ELL_REPORT.md`).

**Notes.** Trilinear resampling is structurally a gather (8 corner voxels at
data-dependent positions), so it inherits the ELL gather blocker. Realistic
near-term outcome: JAX-default until upstream Pallas grows gather, unless the
pointer-load formulation works. A kernel must register a `custom_vjp` (backward is
scatter-add) and keep `map_coordinates` as the floor.
**Baseline (A10G, jax 0.10.0; `bench/PERF_TRILINEAR.md`).** Forward
`spatial_transform`: 256³ ~3.1 ms, 192³ ~1.4 ms; fwd+bwd 256³ ~13 ms. **Cheap interim
win, no Pallas:** an explicit pure-JAX 8-corner gather is ~1.5–1.7× faster than
`map_coordinates` (256³: 1.80 ms vs 3.14 ms) — `map_coordinates` carries dispatch
overhead this op doesn't need. If a consumer hits a resampling wall before the Pallas
gate clears, swap `_gather_coords_linear` to the explicit 8-corner form first.

### B8. Store the `(*)`-annihilator explicitly on `Semiring`

`Semiring.identity` is the **monoid (additive) identity** (`monoid.init`). ELL
padding and `sparse.ell_mask` (medial-wall / grey-matter masking) actually need the
**`(*)`-annihilator** — the `z` with `z (*) b = monoid_identity` for all `b` — so a
masked edge is a no-op. For every built-in *except* EUCLIDEAN the two coincide (REAL
`0`, LOG/TROPICAL_MAX `-inf`, TROPICAL_MIN `+inf`, BOOLEAN `False`), which is why
`identity` currently doubles as the masking value. EUCLIDEAN's `(a-b)**2` has **no**
annihilator yet `identity == 0.0`, so masking by that value silently injects
`B[idx]**2` instead of vanishing. (Verified: `Semiring` in
`src/nitrix/semiring/_types.py` carries `identity`, no `annihilator` field.)

**Consideration.** Add `annihilator: Optional[...]` to `Semiring` (`None` for
EUCLIDEAN). Then `ell_mask` takes a `semiring=` and pulls `semiring.annihilator`,
raising a clear error when `None`, instead of overloading `identity` and relying on
the caller to know the distinction.

**Trigger.** A second masking consumer, a user-defined semiring whose monoid
identity and annihilator differ, or a confusion report. Until then
`ell_mask(ell, valid, *, identity=...)` takes an explicit value and the docstring +
`tests/test_ell_masking_semirings.py` document the distinction (incl. the EUCLIDEAN
exception).

**Effort.** S — one field + a guarded `ell_mask(semiring=...)` overload;
backward-compatible with the explicit-`identity` form.

### B10. Re-tune the Pallas LOG (logsumexp) `semiring_matmul` kernel

The Pallas/Triton LOG-semiring kernel is markedly under-tuned vs both the pure-JAX
reference and a naive materialise-then-reduce — uniquely among the built-in
algebras.

**Trigger.** A consumer running the LOG matmul (logsumexp contraction —
softmax/attention aggregation) at scale where wall-time matters; or the next
`nitrix-perf-bench` sweep promoting this to a kernel-tuning sprint.

**Notes (evidence; A10G, jax 0.10.0,
`../nitrix-perf-bench/reports/PERF_SEMIRING_MATMUL.md`).** LOG Pallas vs JAX
reference: **1.22× at 256³ (slower)**, 0.59× at 512³. REAL / TROPICAL_MAX /
EUCLIDEAN Pallas hit 0.07–0.20× on the same shapes — tiling is fine in general; LOG
is the outlier. A naive materialise-then-`logsumexp` reaches 0.11× at 512³, so the
throughput is achievable. Suspect: the online-logsumexp `(m, s)` recurrence in
`_LogSumExpMonoid` (flash-attention style) serialises the K-loop / carries register
pressure the `sum`/`max` monoids don't. Direction: tile/unroll/occupancy pass on the
LOG kernel, following online-softmax literature; gate it benchmark-first (ship only
if it beats JAX). **Don't over-read the naive win** — its speed is steady-state
only; it pays a pathological cold compile (~45–585 s/point, ~580 s for 512³) and
elevated peak-HBM (68–85 MB vs streaming 2.6–23 MB). Pallas keeps O(M·N) streaming
memory + ~2 s compile, so it stays default; the task is narrow — close the LOG-kernel
throughput gap, not adopt materialisation. **Arch caveat:** numbers are A10G;
re-bench on target before acting.

**Effort.** M — single-kernel tuning + re-bench; no API change.
**Cross-refs.** `../nitrix-perf-bench/reports/PERF_SEMIRING_MATMUL.md`;
`docs/design/streaming-kernel.md`; `bench/G0_ELL_REPORT.md`;
`src/nitrix/semiring/algebras.py` `_LogSumExpMonoid`.

### B11. Finish migrating perf to nitrix-perf-bench (op_matrix → capability-only)

Performance is moving out of this repo into sibling **nitrix-perf-bench**
(cross-framework refs, multi-platform fan-out, durable result store, fp64-oracle
fidelity gating, regression gate, hosted dashboard). End state: the op_matrix is
**capability-only** (jit / grad / vmap / jit-of-grad + invariants), perf lives
entirely in nitrix-perf-bench.

**Status (transitional, non-regressive).** Migrated: `semiring_matmul`,
`semiring_ell_edge_aggregate` (cells render `↗ perf-bench`; see
`MIGRATED_TO_PERFBENCH` in `tools/op_matrix.py`). The other ~30 ops still carry
in-tree `bench/` numbers — nothing lost in the interim.

**Bench-report migration COMPLETE.** Every `bench/PERF_*` / `G0_*` / `MEM_*` report
is either ported to a nitrix-perf-bench case or intentionally kept nitrix-side:
`PERF_AUDIT` (all 13 ops ported), `PERF_SEMIRING_CONV` (ported), `G0_ELL_REPORT`
(ported), `PERF_TRILINEAR` (folded into the `spatial_transform` case). **Kept
nitrix-side:** `PERF_LOBPCG` (private kernel; implicit-VJP perf + HLO no-n² scaling
audit, no public-op/forward-ratio fit) and `MEM_STREAMING_KERNEL` (memory-scaling /
leak study; core finding already captured by the `semiring_matmul` case's `peak_hbm`)
— both move to nitrix-side tests.

**Final step (this repo).** Drop the op_matrix `perf_*` columns + `load_perf_data` +
the `bench/PERF_*` scrapers, and retire `bench/` (perf now lives in
nitrix-perf-bench; the LOBPCG / MEM_STREAMING studies move to tests).

**Trigger.** Incremental/opportunistic: port an op when touched or when its perf
becomes decision-relevant; do the final column-strip once `MIGRATED_TO_PERFBENCH`
covers the catalogue. Or a dedicated migration sprint.

**Effort.** L overall, cleanly sliceable (one op = one small case, each S and
independently shippable). No nitrix API change — docs/tooling + the eventual `bench/`
retirement.
**Cross-refs.** `../nitrix-perf-bench/DESIGN.md` §6 + `src/nperf/cases/`;
`tools/op_matrix.py` `MIGRATED_TO_PERFBENCH`; `bench/PERF_AUDIT.md`.

### G1. `spatial_transform(mode='linear_extrap')` — genuine linear extrapolator

`geometry.spatial_transform`'s `BoundaryMode` is
`{'constant','nearest','wrap','mirror','reflect'}` — edge-replicate (`'nearest'`,
the voxelmorph `fill_value=None` convention) is covered, but there is no true
*linear extrapolation* at the boundary. Distinct from `'nearest'`: pad-by-1 +
edge-replicate the image + adjust coords by 1 to continue the end-segment slope.

**Trigger.** A consumer where gradient continuity at the sampling boundary matters
(a sampler inside a learned generative model). **Not** required for voxelmorph /
JOSA parity (those use edge-replicate), so explicitly low priority. Originally a
JOSA-port request (low priority).

**Effort.** S — one `BoundaryMode` value + the pad/coord-shift inside
`_gather_coords_linear`.

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

`spatial_transform_batched` (JOSA §3) is genuinely open but lives in
`ilex-pipeline-substrate.md` (residual item D, consumer-convenience home), not here.

## Resolved items

Full resolution history for the consumer-driven gaps (gaussian docstrings, the
edge-aggregate / icosphere-hierarchy / mesh-pool stack, `max_pool_with_indices_nd` /
`max_unpool_nd`, SUGAR deltas, GATv2 self-loops, Nyul–Udupa `histogram_match`) lives
in **`IMPLEMENTATION_PLAN.md §10.3`** (shipped-deviation log) and
**`SPEC_UPDATE_v0.3 §10.A`** — these supersede the retired `NITRIX_FEEDBACK_ILEX.md`
/ `NITRIX_FEEDBACK_JOSA.md` ledgers. Backlog items closed here:

- **B1. Move resolved findings in `NITRIX_FEEDBACK_ILEX.md`** — done 2026-05-20.
- **B9. `residualise` robustness on ill-conditioned designs** — resolved 2026-05-22.
  Root cause: default `method='cholesky'` returns NaN for rank-deficient `X` (the
  singular Gram has no factor); the shipped `method='svd'` path (min-norm LSQ via
  `jnp.linalg.lstsq`) is exact there. Resolution was verification + docs:
  `tests/test_resid.py` exercises the SVD path across `p→obs` / `p>obs` and pins the
  cholesky-NaN / svd-finite contract; `method` docstring documents the min-norm
  semantics + failure mode. Default kept `cholesky` (~2× faster on the
  well-conditioned case); `svd` is the documented robust opt-in. See
  `IMPLEMENTATION_PLAN.md §10.3`.
