# v4 Phase 1a — closed-form Mattes MI force (implementation design)

> **Status (2026-06-12): implementation-ready design, no code yet.** The headline
> fMRIPrep-parity deliverable of [`registration-suite-v4.md`](registration-suite-v4.md)
> §3.1 / Phase 1a: a closed-form **Mattes mutual-information** gradient and a
> Tier-1 `MIForce`, replacing the autodiff escape hatch (`MetricForce(MI())`).
> Also specs the **matrix-side A6 range pinning** (followup 4d), since it is the
> same range-stationarity concern. Unnormalised Mattes MI only; **NMI deferred**
> to the escape hatch on use-case grounds (§6).

## 1. Scope & deliverables

| # | Deliverable | Home |
|---|---|---|
| 1a-i | `mi_grad(moving, fixed, *, bins, range_moving, range_fixed)` — closed-form `∂MI/∂moving` | `metrics/information.py` |
| 1a-ii | `MIForce` (+ `_BoundMI`) Tier-1 implementer of the `Force` protocol | `register/_force.py` |
| 1a-iii | `pin_force_ranges(force, moving, fixed)` — eager range resolution in the SVF recipes | `register/_svf.py`, called by `diffeomorphic.py` / `_syn.py` |
| 4d | `range_moving` / `range_fixed` fields on the `MI` / `CorrelationRatio` records + eager pinning in the matrix driver | `register/_metric.py`, `register/_core.py` |
| (opt) | `cr_grad` + `CRForce` (followup 1b) — same machinery, deferred unless a consumer asks | `metrics/information.py`, `_force.py` |

Public surface delta: one new metric function (`mi_grad`), one new force (`MIForce`),
two new optional fields on two metric records. `MetricForce(MI())` stays as the
parity oracle and the NMI path.

## 2. The closed form (derivation, verified)

The joint histogram (`joint_histogram`) is the normalised linear-Parzen table

```
P[a,b] = (1/N) Σ_x β_m(x,a) · β_f(x,b),   Σ_a β_m(x,a) = Σ_b β_f(x,b) = 1  ⇒  Z = N
```

with `β_m(x,·)` the moving soft weights (nonzero on the two bins `k = lower_m(x)`,
`k+1`, weights `1−t_m`, `t_m` where `t_m = frac_m`) and `β_f` likewise. From
`MI = Σ_{a,b} P[a,b] log(P[a,b] / (P_m[a] P_f[b]))`, differentiating **only**
through the moving weights (the fixed weights are constant in `moving`):

```
∂MI/∂m(x) = (1/N) Σ_{a,b} (∂β_m(x,a)/∂m) · β_f(x,b) · log( P[a,b] / P_m[a] )
```

The `+1` and `log P_f[b]` terms vanish because the soft weights sum to one, so
`Σ_a ∂β_m(x,a)/∂m = 0` (any term independent of `a` drops). This is the Mattes
2003 form; it was verified against finite differences to relative L2 ≈ 2e-6 and the
reduced table `W[a,b] = log(P[a,b]/P_m[a])` against the full table `log(P/(P_m P_f))−1`
to machine precision.

**Per-voxel evaluation.** With `s_m = (bins−1)/span_m`, `∂β_m(x,k)/∂m = −s_m`,
`∂β_m(x,k+1)/∂m = +s_m`, and `β_f(x,·)` nonzero on `l, l+1`:

```
∂MI/∂m(x) = (s_m / N) · [ (1−t_f)·D[k,l] + t_f·D[k,l+1] ],
            D[a,b] = W[a+1,b] − W[a,b]   (forward difference along the moving axis)
```

— a gather of two `D` values per voxel (`D` is a tiny `(bins−1, bins)` table) plus a
2-term blend. `s_m` is **0** for any moving voxel outside `[lo_m, hi_m]` (its `frac`
is clipped, so `∂frac/∂m = 0`).

### 2.1 Reference algorithm

```python
def mi_grad(warped, fixed, *, bins, range_moving, range_fixed, eps=1e-10):
    m, f = warped.reshape(-1), fixed.reshape(-1)
    lo_m, hi_m = range_moving; lo_f, hi_f = range_fixed
    span_m = jnp.maximum(hi_m - lo_m, SPAN_FLOOR)          # intensity-scaled floor, not 1e-12
    s_m = (bins - 1) / span_m
    k, t_m = _soft_bin(m, bins, lo_m, hi_m)                # lower idx (∈[0,bins-2]), frac
    l, t_f = _soft_bin(f, bins, lo_f, hi_f)
    P  = _joint_hist_from_softbins(k, t_m, l, t_f, bins)   # shared with the cost (scatter)
    Pm = P.sum(axis=1)
    # Empty-bin convention MUST match `mutual_information` (where(hist>0,...)):
    logP  = jnp.where(P  > 0, jnp.log(P),  0.0)
    logPm = jnp.where(Pm > 0, jnp.log(Pm), 0.0)
    W = jnp.where(P > 0, logP - logPm[:, None], 0.0)       # (bins, bins)
    D = W[1:, :] - W[:-1, :]                               # (bins-1, bins)
    g = (s_m / m.size) * ((1 - t_f) * D[k, l] + t_f * D[k, l + 1])
    g = jnp.where((m >= lo_m) & (m <= hi_m), g, 0.0)       # clip → zero derivative
    return g.reshape(warped.shape)
```

`_joint_hist_from_softbins` is `joint_histogram`'s scatter, factored so the cost and
the gradient share it. The empty-bin masking is deliberately identical to
`mutual_information`'s so `mi_grad == jax.grad(mutual_information)` holds to tolerance
on populated bins (the parity oracle); the gradient at exactly-empty bins is a
documented divergence (the `where(hist>0)` mask is non-smooth there — the analogue of
`lncc_grad`'s documented boundary divergence).

## 3. `MIForce` — the Tier-1 implementer

The joint histogram depends on **both** images at the current warp, so — unlike
`DemonsForce`'s `∇fixed` — there is **nothing image-dependent to hoist in `bind`**;
the histogram is recomputed every iteration (correctly). `bind` carries only `fixed`
+ the (static, pinned) ranges.

```python
@dataclass(frozen=True)
class MIForce:
    bins: int = 32
    range_moving: Optional[tuple[float, float]] = None   # pinned by the recipe if None (§4)
    range_fixed:  Optional[tuple[float, float]] = None
    # `normalized` intentionally absent: NMI is the deferred quotient-rule form (§6).

    def bind(self, fixed, *, ndim, rel_spacing=None) -> _BoundMI:
        return _BoundMI(self.bins, self.range_moving, self.range_fixed,
                        fixed, ndim, rel_spacing)

@dataclass(frozen=True)
class _BoundMI:
    bins; range_moving; range_fixed; fixed; ndim; rel_spacing
    def update(self, warped):
        g = mi_grad(warped, self.fixed, bins=self.bins,
                    range_moving=self.range_moving, range_fixed=self.range_fixed)
        grad = spatial_gradient(warped, spacing=_grad_spacing(self.rel_spacing))
        return _to_voxel(g[..., None] * grad, self.rel_spacing)        # force convention
    def cost(self, warped):
        return -mutual_information(warped, self.fixed, bins=self.bins,
                                   range_moving=self.range_moving,
                                   range_fixed=self.range_fixed)
```

- **Direction/sign.** The Force convention is `u = −∂cost/∂warped · ∇warped` with
  `cost = −MI`, so `u = +∂MI/∂warped · ∇warped = mi_grad · ∇warped` — identical in
  shape to `LNCCForce.update`, voxel-converted by `_to_voxel`. Validated against
  `MetricForce(MI())` (direction) and `jax.grad` (magnitude) — the parity oracle.
- **Magnitude / driver interaction.** `mi_grad` carries the true `(1/N)` MI-gradient
  magnitude. The **SyN** driver clamps the force (`step` set), so the magnitude is
  controlled; the **Demons** driver runs `step=None`, so `MIForce`-driven Demons must
  use the 0c RMS/max normalisation (this is exactly followup B2). Recommend `MIForce`
  with the clamped SyN/greedy driver; document the Demons caveat.

## 4. Range pinning (the stationarity requirement, both halves of A6)

A data-`min/max` range makes the bin grid drift as the moving image deforms → a
**non-stationary objective** (the target moves between iterations), and the extreme
voxels sit at the clip boundary where the force truncates to 0. So the range is
pinned **once** from the full-resolution images and reused at every level.

- **The range is static config (Python floats), not traced.** It is resolved
  **eagerly** in the recipe (`float(img.min())` / `float(img.max())`, or the Phase-4a
  winsorised percentiles), before the pyramid, and baked into the frozen force /
  metric record (so it rides jit static args). Pinning to a constant is also the
  *correct* gradient (a data-derived range is piecewise-constant — zero derivative —
  per `_resolve_range`). **jit caveat:** under `jax.jit` with *traced* images,
  `float(tracer)` cannot run, so the caller must pass explicit ranges (the documented
  contract, matching `_resolve_range`'s existing note).
- **Two ranges, not one `value_range`.** `range_fixed` pinned to the fixed support;
  `range_moving` pinned to the moving support. (`joint_histogram` already takes
  `range_moving`/`range_fixed` — do **not** introduce a single `value_range`.)
  Pinning `range_moving` *below* the data extent silently zeroes the force on
  over-range voxels (the clip), so pin to cover the full support (the winsorised
  extent if 4a is enabled).

**Force side (`pin_force_ranges`).** Before the pyramid, each SVF recipe resolves any
`None` ranges on an `MIForce`/`CRForce`:

```python
def pin_force_ranges(force, moving, fixed):   # in _svf; called per force in the schedule
    if isinstance(force, (MIForce, CRForce)):
        rm = force.range_moving or (float(moving.min()), float(moving.max()))
        rf = force.range_fixed  or (float(fixed.min()),  float(fixed.max()))
        return dataclasses.replace(force, range_moving=rm, range_fixed=rf)
    return force
```

**Matrix side (4d, followup A6).** Add `range_moving` / `range_fixed` (Optional float
tuples) to the `MI` and `CorrelationRatio` records; have `MetricObjective` /
`register_core` resolve them once from the full-res images and thread them into
`metric.cost` per level. This fixes `affine_register(metric=MI())` /
`rigid_register(...)`, whose MI/CR cost otherwise drifts on the per-batch data
min/max — the bug A6 was originally raised against.

## 5. Test plan & parity oracles

1. **Closed-form parity.** `mi_grad(w, f) == jax.grad(λw: mutual_information(w, f, range…))`
   to tolerance on populated bins (pinned range); documented empty-bin divergence.
2. **Force-direction parity.** `MIForce.update` cosine-aligns with `MetricForce(MI())`
   (the existing escape hatch) to machine precision; magnitude matches `mi_grad·∇w`.
3. **Cross-modal recovery.** `MIForce(MI)` drives a synthetic T1↔T2-style deformable
   recovery (group SyN) to NCC tolerance — the use-case end-to-end test.
4. **Stationarity / pinning.** With a pinned range the per-iteration cost is monotone
   under a known warp sequence; with `range=None` + drifting moving it is not (the
   regression the pin fixes). The matrix path (`affine_register(metric=MI())`)
   recovers a known affine that the unpinned path fails on.
5. **Flat / empty-bin robustness.** A uniform-background image: `span` floor prevents
   `s_m` blow-up; finite force everywhere (cross-ref 0a / D1).
6. **Scaling case.** SyN-MI at 256³ single + cohort: wall-clock vs `MetricForce(MI())`
   (the tape removal) and the **forced histogram-scatter throughput** at 256³ (the
   §3.2-plan brain-scale risk — measured here, before claiming brain-scale parity;
   the 5c kernel closes it if needed).

## 6. The NMI / CR decision (use-case-grounded)

- **Ship Mattes MI** (unnormalised): the ANTs/ITK and fMRIPrep cross-modal default
  (`antsRegistration --metric MI`/`Mattes`).
- **Defer NMI** (Studholme `(H_m+H_f)/H_mf`): its gradient is a different quotient-rule
  derivation (`∂NMI ∝ [H_mf·(∂H_m+∂H_f) − (H_m+H_f)·∂H_mf]/H_mf²` — the `+1`/`P_f`
  terms do **not** cancel) with a near-convergence `1/H_mf` blow-up. Its distinctive
  value (overlap invariance) is a **global/affine, large-FOV/partial-overlap** concern,
  ~moot for the dense deformable force (per-level overlap ≈ constant). NMI routes
  through the correct-but-slower `MetricForce(MI(normalized=True))` meanwhile (with 0c
  normalisation). The quotient-rule closed form is scoped (shared contraction
  machinery, only the `W` table changes) for when a deformable-NMI consumer appears.
- **CR** (`cr_grad` + `CRForce`): same machinery on `correlation_ratio`'s `fixed`-binned
  group means; oracle = `jax.grad(correlation_ratio)`; built only if a consumer asks.

## 7. Dependencies, risks

- **Depends on** the `Force` protocol (unchanged) and `joint_histogram` (factor out
  `_joint_hist_from_softbins`). Composes with 0c (Demons magnitude), 0d (RMS clamp),
  Phase 2 (the group driver consumes `MIForce` unchanged), Phase 4a (winsorised range).
- **Risk — empty-bin parity gap:** mitigated by matching the cost's `where(hist>0)`
  masking exactly; validate to tolerance, document the divergence.
- **Risk — Demons + `MIForce` magnitude untuned:** mitigated by 0c; recommend the
  clamped SyN/greedy driver; add the recovery test (#3) on the Demons path too.
- **Risk — jit-with-traced-images can't auto-pin:** documented contract (pass explicit
  ranges under trace), matching `_resolve_range`.

## 8. Cross-references

- [`registration-suite-v4.md`](registration-suite-v4.md) §3.1 (the MI/NMI decision),
  Phase 1a / 4d; `registration-suite-v3-followups.md` B1 (MI force), B2 (MetricForce
  magnitude), A6 (range pinning).
- `src/nitrix/metrics/information.py` (`joint_histogram`, `mutual_information`,
  `correlation_ratio`), `metrics/_common.py` (`_soft_bin`, `_resolve_range`),
  `register/_force.py` (`LNCCForce` pattern, `MetricForce` oracle), `register/_metric.py`
  (`MI`, `CorrelationRatio`), `register/_core.py` (`MetricObjective`, `register_core`).
- [`registration-suite-v4-force-kernels.md`](registration-suite-v4-force-kernels.md)
  §5c (the fused MI histogram+gradient kernel that closes the forward-scatter risk).
