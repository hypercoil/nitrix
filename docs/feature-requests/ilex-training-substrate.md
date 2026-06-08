# ilex training-substrate — survey context & index

> **This doc is the survey *context + index* for the renewed
> 2026-06-08 audit.** It is the sibling of
> [`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md) (which covered
> the *vendored-model inference* pipelines). This one covers the surfaces
> ilex grew **after** the 2026-06-02 survey: the **augmentation pipelines**
> (`ilex/train/augment/`, `ilex/train/pipelines/`), the **loss functions**
> (`ilex/nimox/loss/functional/`), and the **numerical substrate of the
> nimox modules + the newest models** (`ilex/nimox/modules/`; `krakencoder`,
> `cortex_ode`, `surfnet`). One doc per primitive; this file keeps the shared
> framing, the boundary, the negative record, and the index.

Driven by a 2026-06-08 audit of:

- `ilex/train/augment/{intensity,lab2im,geometric,compose,spec}.py` and
  `ilex/train/pipelines/{dino3d,simclr3d,shared_private}.py` — the FM
  pretraining (3DINO / SimCLR / BrainIAC) augmentation stack and the
  SynthSeg-family `lab2im` generative augmentation.
- `ilex/nimox/loss/functional/{segmentation,classification,contrastive,dino,
  registration,regression,vae,swap}.py` — the loss library spun out with
  nimox.
- `ilex/nimox/modules/{affine,ode,instance_norm,batchnorm,ema,hyperconv,
  prelu,unetr_blocks}.py` and the post-survey models `krakencoder`
  (`_pca.py`, `_krakencoder.py`), `cortex_ode`, `surfnet` (per-vertex
  neural-ODE / diffeomorphic surface models).

## The boundary (post bitsjax–thrux–nitrix realignment)

The ilex `augment/__init__.py` docstring still names `bitsjax.augmentation.*`
as the migration target — **that note is stale**, written before the
realignment. The current split, which this audit applies:

- **Pure numerics → `nitrix`.** The deterministic numerical kernel — gamma
  transform, GMM render, bias-field generation, histogram remap, crop-index
  math, Dice, focal CE, NT-Xent, field regularisers, KL/NLL, PCA, mesh
  Laplacian. RNG-keyed pure functions (a `jax.random.key` argument) are
  in-scope: keyed sampling is still a pure `(…, key) -> Array`.
- **Image-op wrappers → `thrux`.** The lift from `(Array,…)->Array` to a
  container-/affine-aware operator on images (world-space, channels
  convention, sidecar metadata).
- **Dataset-coupled chains → `bitsjax`.** Once a BITS / tensorbids dataset
  is in play — the `AugmentationSpec`, the op registry, `compose`, multi-crop
  view fan-out, the role/shape vocabulary (source / spatial-paired /
  image-only / view-multiplier).
- **Trainable modules + training-step builders → `nimox` / ilex.** Every
  `eqx.Module`; the DINO/SimCLR pipelines and step builders; EMA-teacher
  bookkeeping; `loss/scalarise.py` + `loss/scheme.py` (reduction + weighting).

A primitive belongs in nitrix iff it is a pure `(Array, …) -> Array`
expressible in `jax` + `jaxtyping` + `numpy` only, **and** is reusable
substrate rather than model glue, container/IO orchestration, or a trainable
module. The recurring pattern in `augment/`: the *deterministic apply* is the
nitrix atom; the `U(range)` parameter draw is a thin keyed wrapper that can
live in nitrix too (it is pure) or be re-bound upstream by thrux/bitsjax.

## Scope boundary (what is NOT nitrix)

- **`labels_to_image`** (`lab2im.py:254`) — chain orchestration. Its
  *constituent atoms* (GMM render, bias, gamma, Rician, deform) are the
  nitrix targets; the chain is glue (→ bitsjax operator chain).
- **`compose.py` / `spec.py`** — `compose_pipeline`, `multi_crop`,
  `AugmentationSpec`, the `_OP_REGISTRY`, `Lab2imSource`, `_with_prob`. The
  declarative augmentation vocabulary (Tier 1/2) → bitsjax.
- **`pipelines/{dino3d,simclr3d,shared_private}.py`** — `eqx.Module` heads,
  `DINOState`/`SimCLRState`, `build_*_step`, EMA-teacher, optax glue → nimox.
- **`loss/scalarise.py`, `loss/scheme.py`** — reduction strategies + the
  `LossScheme` weighting container → nimox.
- **`mse`, `l1`, `mse_recon`** — one-liners (`jnp.mean((a-b)**2)`); below the
  primitive bar. **`cross_reconstruction`, `bidirectional_consistency`** —
  take a model `decode_fn` / `similarity_fn`; model-coupled glue.
- **`integrate_vertex_flow`** (`nimox/modules/ode.py`) — imports `diffrax`,
  off the nitrix allowlist; the *general pure-jax `odeint`* it wants is
  tracked in [`ode-integrators.md`](ode-integrators.md), not a lift of this.
- **All `eqx.Module`s** — `unetr_blocks`, `PReLU`, `BatchNorm2d`,
  `HyperConv3DFromDense`, `Krakencoder`, the CNN/MLP velocity nets — stay in
  nimox; only their *extractable pure kernels* (below) move.

## Already in nitrix (delegate, do not re-request)

- **Windowed NCC** — `local_ncc_3d/2d` (`loss/registration.py`) duplicate
  `nitrix.metrics.lncc` (ANTs squared form, separable box filter, n-D). ilex
  should call `lncc(..., reduction='none')`.
- **Jacobian determinant** of a displacement — `nitrix.geometry.
  jacobian_det_displacement` (the folding-penalty in
  [`field-regularisers.md`](field-regularisers.md) builds on it).
- **SVF integration** — `nitrix.geometry.integrate_velocity_field`
  (scaling-and-squaring); `lab2im.sample_svf_displacement` already calls it.
- **Average-pool pyramid** — `cortex_ode.build_volume_pyramid` duplicates
  `nitrix.geometry.pyramid` (`downsample`/`gaussian_pyramid`).
- **Lie-algebra affine** — `nitrix.geometry.transform` `rigid_exp`/`affine_exp`
  (axis-angle/twist); **complementary** to the Euler/scale/shear convention
  in [`affine-matrix-algebra.md`](affine-matrix-algebra.md), not a duplicate.
- **Histogram matching** — `nitrix.bias.histogram_match` (Nyúl–Udupa);
  `lifespan_strip`/`bme_x` already migrated to it (2026-05-29).

## Atomised items

Severity: **ENABLING** = a downstream surface is blocked / hand-rolling it;
**CONVENIENCE** = works inline today, a shared primitive removes duplication;
**MISMATCH** = a documented parity deviation.

### Augmentation substrate (FM pretraining + lab2im synth)

| Item | Doc | Severity | Home |
|---|---|---|---|
| GMM label→image render | [lab2im-gmm-synthesis](lab2im-gmm-synthesis.md) | ENABLING | `augment` |
| Generative bias field (simulated INU) | [generative-bias-field](generative-bias-field.md) | ENABLING | `augment` / `bias` |
| Gamma, histogram-shift, gaussian/rician noise | [intensity-augmentation-ops](intensity-augmentation-ops.md) | CONVENIENCE | `augment.intensity` |
| Random flip / crop / resized-crop / affine / SVF gen | [geometric-augmentation-ops](geometric-augmentation-ops.md) | CONVENIENCE | `augment.geometric` |

### Loss / metric numerics (nimox loss library)

| Item | Doc | Severity | Home |
|---|---|---|---|
| Soft / binary Dice | [dice-loss](dice-loss.md) | ENABLING | `metrics.dice` |
| Stable BCE / categorical CE / focal | [cross-entropy-focal](cross-entropy-focal.md) | CONVENIENCE (dedup ×3) | `metrics` |
| NT-Xent / DINO / iBOT / KoLeo | [contrastive-ssl-losses](contrastive-ssl-losses.md) | CONVENIENCE | `metrics` / `stats` |
| Field gradient/bending + Jacobian-folding penalty | [field-regularisers](field-regularisers.md) | ENABLING | `register.regulariser` |
| Diagonal-Gaussian KL / NLL | [gaussian-kl-nll](gaussian-kl-nll.md) | CONVENIENCE | `stats` |

### Model numerics (nimox modules + new models)

| Item | Doc | Severity | Home |
|---|---|---|---|
| Affine param↔matrix / euler↔R / fit / homog. | [affine-matrix-algebra](affine-matrix-algebra.md) | ENABLING | `geometry.transform` |
| PCA fit / transform / inverse (SVD) | [pca-svd](pca-svd.md) | CONVENIENCE | `stats.pca` |
| Lp / unit normalize + instance-norm stats | [lp-normalize](lp-normalize.md) | CONVENIENCE | `numerics.normalize` |
| Uniform 1-ring mesh Laplacian smoothing | [mesh-laplacian-smoothing](mesh-laplacian-smoothing.md) | CONVENIENCE | `geometry.mesh` |

### Expansions to existing docs (new drivers / scope)

- [`compute-vertex-normals`](compute-vertex-normals.md) — add `cortex_ode` /
  `surfnet` as drivers (was `topofit`-only).
- [`point-sample`](point-sample.md) — the arbitrary-point trilinear
  `grid_sample` (align_corners + border-clamp + multichannel) has **3 live
  duplicates** across `cortex_ode`/`surfnet`; planned as nitrix task #138.
- [`ode-integrators`](ode-integrators.md) — `cortex_ode`/`surfnet` per-vertex
  neural-ODE is a concrete consumer for a general pure-jax `odeint`
  (the diffrax dependency cannot follow into nitrix).

## Namespace — resolved (2026-06-08)

**Decision: a thin `nitrix.augment` that *re-exports*.** The canonical home of
each atom is its existing module — `numerics` (crop/flip/normalize), `signal`
(noise), `geometry`/`geometry.transform` (deformation/affine), `metrics` /
`stats` / `register.regulariser` (losses), `bias` (corrective). `nitrix.augment`
adds **only** the genuinely-new generative primitives that have no other home
(GMM label→image render, generative/INU bias field) and re-exports the rest as
a cohesive augmentation surface. This keeps one discoverable namespace for the
augmentation story without duplicating primitives or fragmenting the taxonomy.
The per-atom "Home" lines below name the canonical module; `nitrix.augment` is
the re-export facade over them.
