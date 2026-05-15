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

### B1. Move resolved findings in NITRIX_FEEDBACK_ILEX.md

One-line bookkeeping: the three addressed findings (gaussian
docstrings, edge-aggregate + mesh-conv stack, max_pool_with_indices)
should be relocated from "Open findings" to "Resolved findings"
with commit / PR references each.

**Trigger.** Next time we touch the feedback file (likely when a
new consumer files something fresh), or before any external
publication of the feedback document.

**Notes.** The "Resolved" section is currently empty.  The
existing scaffolding in the doc's "How to use" instructions
specifies the format: one line per resolved item pointing at the
resolving commit / PR.

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

## Resolved items

*(empty — first entry will be transferred here when an item
above lands as a merged sprint or is explicitly rejected.)*
