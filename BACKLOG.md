# nitrix backlog

Ideas to revisit when a concrete consumer ask, perf regression, or
phase boundary surfaces them.  Items here are **not commitments** —
they're parked because (a) no consumer is currently blocked, or
(b) the cost / payoff ratio favours waiting for more signal.
Each entry should name the trigger that would promote it to a sprint.

## Conventions

- One section per item.  Lead with a one-sentence summary, then a
  **Trigger** line stating the condition under which we'd revisit,
  then **Notes** with the design context that would otherwise
  evaporate.
- When an item graduates to a sprint, move it to ``IMPLEMENTATION_PLAN.md``
  (or fold into an open phase) and replace the body here with a
  one-line pointer at the merging commit.

## Open items

### B2. Perf-bench the Sprint A / Sprint B surfaces

Add bench scripts under ``bench/`` for
``semiring_ell_edge_aggregate`` (GCN forward + bwd, DGCNN forward
+ bwd), ``max_pool_with_indices_nd`` /  ``max_unpool_nd``, and
the mesh wrappers at icosphere-realistic shapes
(``ico_6 = 40962`` verts, ``ico_7 = 163842`` verts).  Wire
results into the op-matrix ``perf_ratio`` column.

**Trigger.** First concrete consumer running these in a training
loop where wall-time matters (Topofit cascade at ``ico_7``, any
SphereMorph descendant, any encoder-decoder UNet variant from
the PGlandsSeg lineage).  Or: a Pallas-dispatch experiment
(B3) needing a baseline.

**Notes.**  The op matrix shows all four transformations passing
green on these surfaces; what's missing is the wall-time
characterisation against a natural reference.  Natural references:
- ``semiring_ell_edge_aggregate`` vs PyTorch Geometric ``MessagePassing``
  on the same graph (host-side parity check, JAX timing on GPU)
- ``max_pool_with_indices_nd`` vs ``torch.nn.MaxPool3d(return_indices=True)``
  (the PGlandsSeg reference)

### B3. Pallas dispatch for ``semiring_ell_edge_aggregate``

The current implementation is JAX-only: gather + nested vmap +
semiring axis-reduction.  At Topofit-scale (``ico_7``, ``d_in =
64-128``) the inner edge-MLP is a tiled matmul that **should**
map to the existing ``semiring_matmul`` Pallas backend.  The
feedback flagged Pallas-fast as a stretch goal.

**Trigger.** B2 baseline shows the JAX path is the bottleneck
in a real training loop, or a consumer explicitly asks.

**Notes.** The hard part is that ``edge_fn`` is a user-supplied
Python callable, so we can't precompile a kernel — we'd need
either (a) a Pallas template parameterised by a fixed-shape MLP
descriptor (works for DGCNN-shaped edge_fns; not generic), or
(b) the KeOps-style symbolic formula approach in B5.  Decide
between these only after B2 shows the JAX path matters.

**Measured signal (2026-05-23, nitrix-perf-bench).** The trigger is
now partially met: the `ell_edge_aggregate` case (a GCN-style *linear*
`edge_fn`, ``w·(W@h_j)``, so JAX / torch / the fp64 oracle compute
identical math) shows **torch_geometric's `MessagePassing` is ~2–5×
faster than the `nitrix-jax` reference on CPU** — ratio 0.21× at
``n=4096`` rising to 0.53× at ``n=16384`` (degree 16, ``d_in=d_out=64``),
for both REAL (sum / `aggr='add'`) and TROPICAL_MAX_PLUS (max /
`aggr='max'`).  The gap is the cost of the gather + nested-vmap +
axis-reduce generality vs PyG's specialised native `scatter_reduce`; it
*narrows* with graph size as the per-edge vmap overhead amortises.  This
is the JAX-path-is-the-bottleneck evidence the trigger asks for (against
an external reference, not just B2's internal baseline).  See
`nitrix-perf-bench/reports/PERF_ELL_EDGE_AGGREGATE.md`.

**But the GPU result tempers it — measure before you kernel.**  On the
A10G (the actual deployment target) the same comparison is a *wash*:
nitrix-jax is within ~5% of PyG for REAL (pyg 0.94–0.95×) and ~15%
**faster** for TROPICAL_MAX_PLUS (pyg 1.14–1.15×).  XLA fuses the
gather + vmap + axis-reduce well on Ampere, so the large CPU gap does
**not** carry to the GPU.  Net: a Pallas dispatch for `edge_aggregate`
is **not** motivated by *GPU* performance against PyG — the only standing
perf case is CPU (likely not the deployment target) or an `edge_fn`
shape XLA fuses poorly.  Recommend keeping B3 low-priority on perf
grounds until a GPU shortfall or a concrete consumer appears; the
benchmark (`ell_edge_aggregate`, both platforms) is now the standing
watch for that.

### B4. LOG / EUCLIDEAN / BOOLEAN ``edge_aggregate`` semirings

The current implementation supports REAL, TROPICAL_MAX_PLUS,
TROPICAL_MIN_PLUS.  LOG / EUCLIDEAN raise
``NotImplementedError`` with a specific message; BOOLEAN is
excluded for footgun reasons (most edge_fns produce floats).

**Trigger.** A consumer presents a concrete use case where one
of these would compose naturally.  The most likely is LOG for
attention-style softmax aggregation, but the
``ell_row_softmax``-then-REAL composition is usually cleaner
than a fused logsumexp at the edge.

**Notes.** The shape of the per-edge message is user-controlled,
so the algebra's identity-action on the message must be
user-respected (the same contract as the existing semirings).
LOG: ``logsumexp`` over the neighbour axis on
``log(edge_fn_output)``-type messages.  EUCLIDEAN:
``sqrt(sum_p edge_fn ** 2)``-shaped reductions.

### B5. KeOps ``Genred``-primitive research

KeOps's ``Genred`` (Generic Reductions) takes a user-supplied
mathematical formula expressed in a string DSL and JIT-compiles
a CUDA kernel that streams the reduction without materialising
the per-edge tensor.  The SPEC's §3.1 "KeOps-style streaming
kernel" claim already invokes this lineage; what's missing is a
concrete research note on whether a Pallas-backed analogue
could host arbitrary user formulas (the edge_aggregate path's
generic case).

**Trigger.** B3 forces a decision between "Pallas template per
edge_fn shape" and "compile a formula on the fly".  Or: a
follow-up SPEC update revisits the formula-compilation idea
independently.

**Notes.**  The research deliverable is a design doc covering:
1. KeOps's formula DSL (LazyTensor + reduction primitives).
2. What the JAX / Pallas analogue would look like
   (jaxpr-level analysis of a user-supplied callable?
   Pallas templating via concrete-shape inputs?).
3. The boundary between formulas that can be compiled (closed
   polynomial / elementwise) and those that can't (data-
   dependent control flow, dynamic shapes).
4. Whether this belongs in nitrix or as a separate package.

**Cross-references.**
- ``docs/design/streaming-kernel.md`` already documents the
  current ``O(M·N)`` peak-HBM claim and the deliberate
  avoidance of tensor cores.
- ``docs/design/ell-on-triton.md`` for the current Triton /
  Pallas gap.
- KeOps paper: Charlier et al. 2021, "Kernel Operations on the
  GPU, with Autodiff, without Memory Overflows", JMLR.

### B6. Pallas kernel for the Gaussian blur primitive

``smoothing.gaussian`` is a separable n-D convolution implemented
via ``lax.conv_general_dilated`` (one 1-D pass per axis), which
lowers to **cuDNN** on Ampere+ -- a strong baseline.  A consumer
flagged a Pallas kernel as a (low-priority) eventual want.

**Trigger.** A consumer with a wall-time wall on large-3-D Gaussian
blur (e.g. repeated 256³ smoothing inside a training loop), *and* a
benchmark showing the separable-conv path is the bottleneck.

**Notes.** Unlike the ELL / trilinear gather case, the Gaussian is a
**stencil**, not a gather, so it is Pallas-friendly (the dense
``semiring_matmul`` Pallas kernel already works).  But cuDNN is hard
to beat per-pass; the only real Pallas win is **fusing the 3
separable axis passes** (and the boundary pad) into one kernel to
save the inter-pass HBM round-trips on large volumes.  That is
marginal and bandwidth-bound; do not build speculatively.  Any kernel
ships behind the existing ``backend=`` dispatch with the
``conv_general_dilated`` path as the JAX floor (non-negotiable
§2.2.3) and a golden-corpus parity test.  Bench against
``lax.conv_general_dilated`` (not against a naive loop) so the
comparison is honest.

### B7. Pallas kernel for 3-D trilinear resampling

``geometry.spatial_transform`` / ``integrate_velocity_field`` resample
via ``jax.scipy.ndimage.map_coordinates(order=1)``.  A consumer asked
for a Pallas kernel; the "benchmark first" baseline lives in
``bench/trilinear_resample.py`` / ``bench/PERF_TRILINEAR.md``.

**Trigger.** Both of: (a) the baseline shows the resampling path is a
real bottleneck in a consumer training loop, and (b) a pointer-load
(``pl.load`` / ``pl.ds``) Pallas prototype clears the gather-lowering
risk -- i.e. it avoids the HLO ``gather`` primitive that Triton on the
pinned JAX cannot lower (the same blocker as ELL; see
``bench/G0_ELL_REPORT.md``).

**Notes.** Trilinear resampling is structurally a gather (8 corner
voxels at data-dependent positions), so it inherits the ELL gather
blocker.  The realistic near-term outcome is JAX-default (current
state) until upstream Pallas grows gather, *unless* the pointer-load
formulation works.  A kernel must register a ``custom_vjp`` (the
backward is a scatter-add) and keep the ``map_coordinates`` path as
the JAX floor.  Cross-reference: ``bench/G0_ELL_REPORT.md`` for the
gather-lowering precedent.

**Baseline (A10G, jax 0.10.0; ``bench/PERF_TRILINEAR.md``).**  Forward
``spatial_transform`` is fast in absolute terms: 256³ in ~3.1 ms,
192³ in ~1.4 ms; fwd+bwd 256³ in ~13 ms.  **Cheap interim win, no
Pallas:** an explicit pure-JAX 8-corner gather is ~1.5–1.7× faster
than ``map_coordinates`` (256³: 1.80 ms vs 3.14 ms), because
``map_coordinates`` carries dispatch / generality overhead this op
doesn't need.  If a consumer hits a resampling wall before the Pallas
gate clears, swap ``_gather_coords_linear`` to the explicit 8-corner
form first (it's the same gather XLA already lowers, just leaner).

### B8. Store the `(*)`-annihilator explicitly on `Semiring`

`Semiring.identity` is the **monoid (additive) identity** (the
`monoid.init` value).  ELL padding and `sparse.ell_mask`
(medial-wall / grey-matter masking) actually need the
**`(*)`-annihilator** -- the `z` with `z (*) b = monoid_identity`
for all `b` -- so a masked edge is a no-op in
`(+)_p values[i,p] (*) B[...]`.  For every built-in *except*
`EUCLIDEAN` the two coincide (REAL `0`, LOG/TROPICAL_MAX `-inf`,
TROPICAL_MIN `+inf`, BOOLEAN `False`), which is why `identity`
currently doubles as the masking value.  `EUCLIDEAN`'s `(a-b)**2`
has **no** annihilator yet `identity == 0.0`, so masking by that
value silently injects `B[idx]**2` instead of vanishing.

**Consideration.** Add an explicit `annihilator: Optional[...]`
field to `Semiring` (`None` for `EUCLIDEAN`).  Then `ell_mask`
takes a `semiring=` and pulls `semiring.annihilator`, raising a
clear error when it is `None`, instead of overloading `identity`
and relying on the caller to know the distinction.  Overloading
`identity` for two concepts (monoid neutral vs multiplicative
absorber) is a confusion risk as the algebra set grows.

**Trigger.** A second masking consumer, a user-defined semiring
whose monoid identity and annihilator differ, or any confusion
report.  Until then `ell_mask(ell, valid, *, identity=...)` takes
an explicit value and the docstring + `tests/test_ell_masking_semirings.py`
document the distinction (incl. the EUCLIDEAN exception).

**Effort.** S.  One field + a guarded `ell_mask(semiring=...)`
overload; backward-compatible with the explicit-`identity` form.

### B10. Re-tune the Pallas LOG (logsumexp) `semiring_matmul` kernel

The Pallas / Triton kernel for the **LOG** semiring is markedly
under-tuned versus both the pure-JAX reference and a naive
materialise-then-reduce alternative -- uniquely among the built-in
algebras.

**Trigger.** A consumer running the LOG semiring matmul (the
logsumexp contraction -- softmax / attention-style aggregation) at
scale where wall-time matters; or the next `nitrix-perf-bench`
sweep promoting this to a kernel-tuning sprint.

**Notes (evidence).** `nitrix-perf-bench` `semiring_matmul` case,
A10G, jax 0.10.0 (`../nitrix-perf-bench/reports/PERF_SEMIRING_MATMUL.md`):
- LOG Pallas vs the JAX reference: **1.22× at 256³ (slower than the
  reference)**, **0.59× at 512³ (1.7×)**.
- REAL / TROPICAL_MAX_PLUS / EUCLIDEAN Pallas hit 0.07–0.20×
  (5–14×) on the same shapes -- so the tiling is fine in general;
  LOG is the outlier.
- A naive materialise-then-`logsumexp` baseline reaches **0.11× at
  512³ (≈9×)**, so the throughput is achievable -- the kernel is
  leaving it on the table.

The suspect is the online-logsumexp recurrence (the running
`(m, s)` max / sum-exp state in `_LogSumExpMonoid`, flash-attention
style): it likely serialises the K-loop and/or carries register
pressure the additive (`sum`) and `max` monoids don't, since those
share the same tiling and are fast.  Direction: a tile / unroll /
occupancy pass on the LOG kernel specifically, following the
online-softmax / flash-attention literature for the streaming
logsumexp recurrence.  Gate it the G0 way (benchmark-first; ship
only if it beats the JAX path).  Not new functionality, and the
JAX path stays the floor.

**Don't over-read the naive win.** `naive-dense`'s speed is
steady-state *only*: it pays a pathological **cold** compile (XLA
`input_reduce_fusion` over the materialised operand -- ~45–585 s
per reduction point, ~580 s for 512³ logsumexp; this hits every
algebra including `max`, once each attempt runs in its own process
-- the earlier ~70 ms figures were an in-process XLA cross-attempt
cache artifact) and elevated peak-HBM (~68–85 MB vs the streaming
kernels' 2.6–23 MB).  Pallas keeps the O(M·N) streaming memory and
a ~2 s compile, so it stays the default -- the task is narrow:
close the LOG-kernel throughput gap, not adopt materialisation.

**Arch caveat.** Numbers are A10G; the naive↔Pallas crossover and
the compile magnitudes may shift on other GPUs (e.g. Lovelace), so
re-bench on the target before acting.

**Effort.** M.  Single-kernel tuning + re-bench; no API change.

**Cross-references.**
- `../nitrix-perf-bench/reports/PERF_SEMIRING_MATMUL.md` -- the
  evidence; supersedes the hand-built `bench/PERF_SEMIRING_MATMUL.md`.
- `docs/design/streaming-kernel.md` -- the O(M·N) streaming claim
  and the online logsumexp `(m, s)` state.
- `bench/G0_ELL_REPORT.md` -- the benchmark-gated kernel-decision
  precedent.
- `src/nitrix/semiring/algebras.py` `_LogSumExpMonoid` -- the
  online recurrence under suspicion.

### B11. Finish migrating perf to nitrix-perf-bench (op_matrix → capability-only)

Performance is moving out of this repo into the sibling
**nitrix-perf-bench** suite, which is strictly richer than the
in-tree `bench/` + the op_matrix's two perf columns:
cross-framework refs (torch / PyG), multi-platform fan-out
(CPU + A10G), a durable result store with history, fidelity
gating against an fp64 oracle, a regression gate, decision-input
bundles, and a hosted HTML dashboard.  The end state: the
op_matrix is **capability-only** (jit / grad / vmap / jit-of-grad
+ invariants -- intrinsic to the op, regenerates standalone), and
perf lives entirely in nitrix-perf-bench.

**Status (transitional, non-regressive).** Two ops are migrated --
`semiring_matmul` and `semiring_ell_edge_aggregate`; their
op_matrix perf cells now render `↗ perf-bench` (see
`MIGRATED_TO_PERFBENCH` in `tools/op_matrix.py`).  The other ~30
ops still carry their in-tree `bench/` numbers, so nothing is lost
in the interim.

**Plan.** Port each remaining benchmarked op to a nitrix-perf-bench
*case* (its competing baselines + the external reference it is
rated against + an fp64 oracle), add it to `MIGRATED_TO_PERFBENCH`,
and let the op's cell point out.  When the set covers the
catalogue, delete the op_matrix perf columns, `load_perf_data`, the
`bench/PERF_*` scrapers, and retire `bench/` itself.  The bulk is
the `PERF_AUDIT` external-library comparisons (numpy / scipy /
sklearn), which are mechanical now that the perf harness exists;
`PERF_SEMIRING_CONV`, `PERF_LOBPCG`, `PERF_TRILINEAR`,
`MEM_STREAMING_KERNEL`, `G0_ELL_REPORT` are the rest.

**Migration status (perf-bench).**  PERF_AUDIT: all 13 ops ported.
`PERF_SEMIRING_CONV`: ported (`semiring_conv` case).
**`PERF_LOBPCG`: intentionally NOT migrated** -- `lobpcg_top_k_*`
is a *private* kernel (`nitrix.graph._lobpcg_diff`, no public op /
no op_matrix cell), and the report is an implicit-VJP perf + HLO
*scaling-correctness* study (the "no O(n^2) materialisation"
audit), which does not fit perf-bench's public-op / forward-ratio
model and has no HLO-shape metric there.  Keep PERF_LOBPCG (esp.
the no-n^2 HLO audit) as a nitrix-side study/test; it is out of the
op-matrix migration's scope.  `PERF_TRILINEAR`: folded into the
existing `spatial_transform` case (it benchmarks the same op) -- 3-D
points + an explicit 2^ndim-corner gather baseline (the
shipped-vs-explicit kernel-decision comparison); its fwd+bwd
timing is a gradient cost outside the forward-case model.
`G0_ELL_REPORT`: ported (`semiring_ell_matmul` case -- nitrix-jax
vs scipy.sparse @ dense for REAL + the non-real algebras; the
"Pallas falls back to JAX on Ampere" policy is a nitrix-side
fact, not a perf-bench ratio).
**`MEM_STREAMING_KERNEL`: intentionally NOT migrated.**  It is a
*memory-scaling / leak* study for `semiring_matmul` (already a
case), not a forward ratio; its core finding (streaming peak HBM
<< the naive O(M*K*N) materialisation) is **already captured** by
that case's `peak_hbm` metric -- measured streaming 5.8/23 MB
(nitrix-jax/pallas) vs naive-dense 85/73 MB at 256/512 log on the
A10G.  Its unique parts (the per-call HBM *delta* leak-check and
the analytical M*K*N floor) are memory-correctness checks that
don't fit perf-bench's per-attempt-HWM model; keep them as a
nitrix study/test.

**Bench-report migration COMPLETE.**  Every `bench/PERF_*` /
`bench/G0_*` / `bench/MEM_*` report is either ported to a
nitrix-perf-bench case or intentionally kept nitrix-side
(`PERF_LOBPCG`, `MEM_STREAMING_KERNEL`).  `MIGRATED_TO_PERFBENCH`
covers every public op that had an in-tree perf number.
**Final step (this repo):** drop the op_matrix `perf_*` columns +
`load_perf_data` + the `bench/PERF_*` scrapers, and retire
`bench/` (its perf now lives in nitrix-perf-bench; the LOBPCG /
MEM_STREAMING studies move to tests).

**Trigger.** Incremental + opportunistic: port an op when it is
touched or when its perf becomes decision-relevant; do the final
column-strip once `MIGRATED_TO_PERFBENCH` covers the catalogue.  Or
a dedicated migration sprint if a stakeholder wants the whole
matrix's perf on the hosted dashboard at once.

**Effort.** L overall, but cleanly sliceable (one op = one small
case); each slice is S and independently shippable.  No nitrix API
change -- only docs/tooling + the eventual `bench/` retirement.

**Cross-references.**
- `../nitrix-perf-bench/DESIGN.md` §6 (phase plan; the perf suite
  executes this) and its `src/nperf/cases/` (the case pattern to
  copy per op).
- `tools/op_matrix.py` `MIGRATED_TO_PERFBENCH` -- the per-op switch;
  `load_perf_data` + `_parse_perf_*` + `bench/` are what retire at
  the end.
- `bench/PERF_AUDIT.md` -- the ~30-op external-library reference
  numbers to reproduce as cases.

## Resolved items

- **B9. `residualise` robustness on ill-conditioned designs** — resolved
  2026-05-22.  Root cause: the default ``method='cholesky'`` (Cholesky of the
  Gram ``X Xᵀ``) returns NaN for rank-deficient ``X`` (``p > obs`` / collinear
  columns) because the singular Gram has no factor; the already-shipped
  ``method='svd'`` path (min-norm least-squares via ``jnp.linalg.lstsq``) is
  exact there.  Resolution is verification + documentation, not new numerics:
  ``tests/test_resid.py`` now exercises the SVD path across the full
  ``p -> obs`` and ``p > obs`` regime (``test_residual_decomposition_svd_robust``)
  and pins the cholesky-NaN / svd-finite contract
  (``test_svd_robust_where_cholesky_degenerates``); the ``method`` docstring
  documents the min-norm semantics, the unique-projection guarantee, the
  cholesky NaN failure mode, and the ``lstsq`` ``rcond``-cutoff pitfall.
  Default kept as ``cholesky`` (≈2× faster on the common well-conditioned
  case); ``svd`` is the documented robust opt-in.  See
  ``IMPLEMENTATION_PLAN.md §10.3``.

- **B1. Move resolved findings in NITRIX_FEEDBACK_ILEX.md** — done
  2026-05-20. The gaussian-docstring, edge-aggregate/mesh-conv-stack,
  max_pool_with_indices, and SUGAR-delta findings are now in that
  doc's "Resolved findings" section; the open entries carry RESOLVED
  markers. (Commit refs land on merge.)
