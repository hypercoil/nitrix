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

### B9. `residualise` robustness on ill-conditioned design matrices

``linalg.residualise`` loses the exact ``residual + projection == Y``
decomposition (at ``atol=1e-5``, float32) when the design matrix ``X``
is ill-conditioned -- i.e. as the number of regressors ``p`` approaches
the observation count ``obs`` and the Gram ``X Xᵀ`` becomes near-
singular.  This is a **long-standing, documented limitation** (it
predates the ``functional`` -> ``linalg`` migration; both
implementations are numerically identical and both fail it), recorded
in the original ``tests/test_resid.py`` generator comment ("too good at
coming up with adversarial examples ... will likely need a strong
background in error analysis and numerical linear algebra").

It was previously *masked* because the hypothesis property tests rarely
reached the ill-conditioned regime under the default deadline / health
checks; the 2026-05-21 conftest deflake (``deadline=None``) made
hypothesis explore fully and surface it deterministically.  The tests
now constrain ``generate_valid_arrays(well_conditioned=True)`` to
``p <= obs/2`` (the well-conditioned regime where the property holds),
so the suite is honest and green, and this entry tracks the real fix.

**Trigger.** A consumer needs residualisation of near-rank-deficient
designs (``p`` close to ``obs``), or we invest in the numerics.

**Notes / candidate fix.** Replace the Cholesky-of-Gram solve with an
SVD- / QR-based projector (``Q Qᵀ`` from a thin QR of ``Xᵀ``, or an
SVD with a singular-value floor), which is stable for rank-deficient /
ill-conditioned ``X``.  ``residualise`` already exposes ``method=``;
add ``method='svd'`` (or ``'qr'``) as the robust path and consider
making it the default.  Pair the fix with re-widening the
``well_conditioned`` cap (or removing it) in ``test_resid`` so the
exact-decomposition properties once again exercise ``p -> obs``.

## Resolved items

- **B1. Move resolved findings in NITRIX_FEEDBACK_ILEX.md** — done
  2026-05-20. The gaussian-docstring, edge-aggregate/mesh-conv-stack,
  max_pool_with_indices, and SUGAR-delta findings are now in that
  doc's "Resolved findings" section; the open entries carry RESOLVED
  markers. (Commit refs land on merge.)
