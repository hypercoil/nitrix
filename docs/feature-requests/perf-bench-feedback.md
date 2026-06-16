# nitrix-perf-bench feedback — ledger & index

> **This doc is now the doc-drift *ledger + index*.** Each finding has been
> atomised into its own tracking doc (one doc per fix, to reduce
> duplicate-issue risk); this file keeps the framing and the index. See
> [`README.md`](README.md) for the directory-wide index.

Documentation / definition drift and consumer-facing gaps surfaced while
building benchmark cases in `nitrix-perf-bench` (the perf migration of the op
matrix; see DESIGN there). Each atomised entry cites file:line and the
measurement that surfaced it so the fix is mechanical. Perf *numbers* live in
the perf-bench `COVERAGE_DEFICIT` report; these entries are for
**correctness-of-documentation** findings only — they are doc fixes, not
primitive proposals.

## Open (atomised)

Surfaced 2026-06-02 while building perf-bench cases; verified against
`scipy 1.17.1` + from-scratch fp64 references on this checkout.

| Finding | Doc | Site | Priority |
|---|---|---|---|
| `lomb_scargle_periodogram` normalisation claim is wrong | [doc-lomb-scargle-normalisation](doc-lomb-scargle-normalisation.md) | `signal/lomb_scargle.py:154` | high-value |
| module docstring says "Cholesky"; code uses `eigh` + pseudo-inverse | [doc-lomb-scargle-eigh-factorisation](doc-lomb-scargle-eigh-factorisation.md) | `signal/lomb_scargle.py:43–49` | normal |
| `lomb_scargle_interpolate` silently runs eigh on CPU (cuSolver-broken stacks) | [doc-lomb-scargle-cpu-eigh-caveat](doc-lomb-scargle-cpu-eigh-caveat.md) | `linalg/_solver.py:147` | normal |
| `tsconv` documented as "convolution" but is cross-correlation | [doc-tsconv-cross-correlation](doc-tsconv-cross-correlation.md) | `signal/tsconv.py:45` | low (clarity) |
| `lomb_scargle_interpolate` intended-use (spectral bridge, not durable imputation) | [doc-lomb-scargle-interpolate-intended-use](doc-lomb-scargle-interpolate-intended-use.md) | `signal/lomb_scargle.py:~264–359` | normal |
| `gaussian_kernel` sigma->gamma relation wrong (missing ½ factor) | [doc-gaussian-kernel-gamma](doc-gaussian-kernel-gamma.md) | `linalg/kernel.py:37` | low (clarity) |
| `relaxed_modularity` doesn't reduce to Newman modularity — it's `Q_N / 2` (double-corrected undirected count); default `exclude_diag=True` also drops the diagonal | [doc-relaxed-modularity-newman-factor](doc-relaxed-modularity-newman-factor.md) | `graph/community.py:245` | low (clarity) |
| `_iir.py` module docstring says `backend='scan' (default)`; real default is `'auto'` (→ fft on GPU / scan on CPU). Function docstrings are correct; only the module header is stale | [doc-iir-backend-default](doc-iir-backend-default.md) | `signal/_iir.py:~22` | normal |
| `pairedcorr` forms the full `cov(X)`/`cov(Y)` to read their diagonals — ~3× matmul (the direct-variance ref is ~2× faster from `c≳512` on CPU+GPU; `pairedcov` control is at parity) | [pairedcorr-redundant-full-cov](pairedcorr-redundant-full-cov.md) | `stats/covariance.py:313–328` | perf (micro-opt) |
| **perf:** LME family calls `jnp.linalg.cholesky` on the tiny `(p,p)` (p=1 → 1×1) fixed-effect system inside the per-voxel `vmap`, and takes its score/Hessian by 2nd-order autodiff *through* it → CPU compile scales linearly in `V` + OOM at brain-volume `V`. Prototype (Cholesky→scalar, autodiff kept): **3–6× CPU steady, 3–5× + flatter compile**, identical accuracy. `reml_fit` adds `jax.hessian`/`(2,2) solve`/full-`N×N` eigh to clean up. Dispatch on `p` (closed-form 1×1/2×2 hot path, capable fallback for p>2) | [lme-family-tiny-linalg-gpu-block-and-perf](lme-family-tiny-linalg-gpu-block-and-perf.md) | `stats/lme/flame.py:113`, `stats/lme/reml.py:188,231,239,448` | perf |
| **GPU-availability (observational / cause unknown):** `flame_two_level` **skips** on GPU — `gpusolverDnCreate` fails when `potrf` (or `syevd`) is the *first* cuSOLVER routine a process touches; a prior `getrf`/matmul clears it. `reml_fit` runs **OK** (its `2×2` Newton `getrf` inits the handle first — so reml's `ok` GPU rows are correct, NOT stale). Cause NOT established; provisional cuBLAS-warmup workaround needs robust repeated-trial verification. Closed-form perf fix sidesteps it (no cuSOLVER) | [gpu-cusolver-first-call-handle-failure](gpu-cusolver-first-call-handle-failure.md) | `stats/lme/flame.py:113` (jaxlib `solver_handle_pool.cc`) | GPU-availability (env) |

_(The five lomb/tsconv findings above resolved 2026-06-02 — see below;
`doc-gaussian-kernel-gamma` and `doc-relaxed-modularity-newman-factor` are
newly open 2026-06-03; `doc-iir-backend-default` newly open 2026-06-06;
`pairedcorr-redundant-full-cov` newly open 2026-06-10;
`lme-family-tiny-linalg-gpu-block-and-perf` +
`gpu-cusolver-first-call-handle-failure` newly open 2026-06-12.
`register-affine-small-grid-divergence` + `register-demons-force-divide-by-zero`
were open 2026-06-11 and **resolved 2026-06-12 in v4 `63d69e7`** — see Resolved.)_

## Resolved

The five lomb-scargle / tsconv doc-drift findings were fixed on 2026-06-02
(docstring-only, no behaviour change; the normalisation fix additionally
carries a scipy-parity regression test). See `IMPLEMENTATION_PLAN.md §10.3`
(2026-06-02 entry) and each item's own doc for the per-fix record.

| Finding | Doc | Resolution |
|---|---|---|
| `lomb_scargle_periodogram` normalisation | [doc-lomb-scargle-normalisation](doc-lomb-scargle-normalisation.md) | docstring rewritten to the math + `N/2` note; regression test added |
| eigh-vs-Cholesky module docstring | [doc-lomb-scargle-eigh-factorisation](doc-lomb-scargle-eigh-factorisation.md) | prose rewritten to the eigh / pseudo-inverse path |
| `safe_eigh` CPU-routing caveat | [doc-lomb-scargle-cpu-eigh-caveat](doc-lomb-scargle-cpu-eigh-caveat.md) | "Device placement" Notes added |
| `lomb_scargle_interpolate` intended use | [doc-lomb-scargle-interpolate-intended-use](doc-lomb-scargle-interpolate-intended-use.md) | "Intended use" Notes added |
| `tsconv` cross-correlation | [doc-tsconv-cross-correlation](doc-tsconv-cross-correlation.md) | Notes clarification added |
| op_matrix inventory gaps | [doc-op-matrix-inventory-gaps](doc-op-matrix-inventory-gaps.md) | full inventory re-run: catalogue 59 → 137 ops; completeness-guard test added; `signal.tsconv` export + stale `bilateral_gaussian` fixture fixed |
| `metrics` vs ITK/ANTs convention divergence | [metrics-convention-vs-domain-tools](metrics-convention-vs-domain-tools.md) | resolved 2026-06-09: divergence is intentional convention, no nitrix lapse; verified all five (`ssd`=`MeanSquares`; ITK `Correlation`=`−ncc²`, mean-subtracted — original "no mean subtraction" note was wrong; `lncc`=`ANTSNeighborhoodCorrelation` interior-exact; soft-Parzen MI/CR differentiable-by-design, no soft→hard parameter limit). Exact conventions + parity pinned in each docstring; document-only (no variants) |
| **correctness:** demons ESM force `0/0 → NaN` on uniform regions | [register-demons-force-divide-by-zero](register-demons-force-divide-by-zero.md) | resolved 2026-06-12 (v4 `63d69e7`): gradient-safe double-`where` zeroes the force where `|j|²+α²·diff²≈0`; regression test `test_demons_finite_on_uniform_background` (2026-06-16) |
| **regression:** `affine_register` diverges at small coarse grids | [register-affine-small-grid-divergence](register-affine-small-grid-divergence.md) | resolved 2026-06-12 (v4 `63d69e7`): IC geometric trust region + affine pyramid-depth cap (`AffinePyramidDepthWarning`, ≥16 vox/axis); regression test `test_affine_small_grid_stays_bounded` (2026-06-16). perf-bench can un-`xfail` its 28³ case |
