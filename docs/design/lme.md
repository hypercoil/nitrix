# lme: voxelwise linear mixed-effects models

> **TL;DR.** ``nitrix.stats.lme`` ships two layered primitives
> for voxelwise LME fitting on fMRI / dMRI data: ``reml_fit``
> (general variance-components REML via the FaST-LMM spectral
> trick) and ``flame_two_level`` (the FSL FLAME equivalent, a
> single-parameter REML with known per-subject within-variance).
> Both are vmap-batched over voxels with shared design and use
> Newton scoring on log-parameterised variance components in a
> fixed-iteration scan; the design ensures no ``V × N × N``
> intermediate at any point in the compiled HLO, so memory
> scales as ``V·N`` not ``V·N²``.  Matches ``statsmodels.MixedLM``
> reference to ~5e-3 at fp64 (the convergence floor of both
> solvers).

## Why a separate LME namespace

Voxelwise LME is the standard tool for fMRI group analysis -- one
LME fit per voxel, often 100k-1M voxels per brain.  The naive
implementation (loop over voxels, call R / statsmodels per voxel)
takes hours to days.  A vmap-batched GPU implementation should
finish in seconds-to-minutes.

The two models we ship cover the bulk of practical fMRI use cases:

- **General REML** (``reml_fit``): for cross-sectional designs
  with subject random effects -- e.g., voxelwise random
  intercepts to account for subject-level variation in a
  task-fMRI analysis.
- **FLAME two-level** (``flame_two_level``): the FSL FLAME model
  for fMRI group analysis given per-subject level-1 GLM output.
  The standard tool for "compute group statistics from
  subject-level betas" workflows.

## `reml_fit`: general variance-components REML

### The model

For each voxel ``v``, the linear mixed-effects model is::

    y_v = X beta_v + Z b_v + eps_v
    b_v ~ N(0, sigma_b^2 I_q)
    eps_v ~ N(0, sigma_e^2 I_N)
    Cov(y_v) = V = sigma_b^2 ZZ^T + sigma_e^2 I_N

Two variance components: between-group (``sigma_b^2``) and
within-group / residual (``sigma_e^2``).  Fixed effects ``beta``
are profiled out analytically.

The fixed-effect design ``X`` and random-effect design ``Z`` are
**shared across voxels** -- the typical fMRI case: one
group-level design, applied to every voxel's response.  Only
``y_v`` varies per voxel.

### The FaST-LMM spectral trick

Lippert et al. 2011's FaST-LMM idea: eigendecompose
``ZZ^T = U Λ U^T`` once at the outer call.  In the rotated basis::

    y_rot = U^T y,  X_rot = U^T X
    V_rot = sigma_b^2 Λ + sigma_e^2 I = diag(d)
    d_i = sigma_b^2 lambda_i + sigma_e^2

Every operation in the Newton iteration becomes elementwise
on ``d``:

- ``log|V| = sum_i log(d_i)`` -- ``O(N)``.
- ``V^{-1} = diag(1 / d)`` -- ``O(N)``.
- ``X^T V^{-1} X = sum_i x_i x_i^T / d_i`` -- ``O(N p^2)``.
- ``X^T V^{-1} y = sum_i x_i y_i / d_i`` -- ``O(N p)``.

Per-Newton-step cost drops from ``O(N^3)`` (naive Cholesky on V)
to ``O(N p^2)``.  Since typical fMRI has ``p ≪ N`` (small
fixed-effect dimensionality), this is a substantial speedup.

The eigendecomposition is computed **once** at the outer call
via ``nitrix.linalg.safe_eigh`` (cuSolver-CPU fallback for the
broken-driver case) and shared across all voxels via vmap closure.

### Newton scoring with damping + backtracking

The inner loop is Newton on the negative profile REML log-
likelihood with the parameter ``theta = (log(sigma_b^2),
log(sigma_e^2))``.  Log-space parameterisation enforces
positivity automatically; the unconstrained optimisation is
cleaner.

Two robustness mechanisms:

1. **Levenberg damping** (``damping=1e-6``) on the Hessian
   diagonal -- handles ill-conditioned steps near boundaries
   (where one variance component collapses).
2. **Step clipping + backtracking** -- caps per-axis log-
   variance update at ``max_step=1.0``; if the full step
   doesn't decrease nll, try halving (4 backtrack steps via
   ``lax.scan``).  Both run as fixed-iteration scans so they
   vmap cleanly.

Default ``n_iter=20`` Newton iterations.  For well-conditioned
data, 5-10 are enough; the extra iterations are cheap safety.

### Memory regime

At ``V`` voxels, ``N`` subjects, ``p`` fixed-effect coefficients,
``q`` random-effect columns:

- Shared (computed once, ``O(N²)`` storage):
  - ``ZZ^T``: ``(N, N)`` ~``N²·4`` bytes.
  - ``U``, ``Λ`` from ``safe_eigh``: ``(N, N)`` + ``(N,)``.
  - ``X_rot = U^T X``: ``(N, p)``.

- Per-voxel (vmapped, ``O(V·N)`` storage):
  - ``y_rot = Y @ U``: ``(V, N)``.
  - ``inv_d``, intermediate ``Xw``: ``(V, N)`` and ``(V, N, p)``.

- Per-voxel parameters: ``(V, 2)`` + ``(V, p)`` + ``(V,)`` ~ tiny.

Total HBM at ``V = 100k``, ``N = 30``, ``p = 5``:
- ``Y``: 12 MB.
- ``X_rot``, ``U``, ``Lambda``: < 1 MB shared.
- Per-Newton step intermediates: ~``V * N * p * 4 = 60 MB`` peak.
- Final params: 3 MB.

Total ~80 MB.  Fits trivially on any modern GPU.  At ``V = 1M``,
``N = 100``, ``p = 10``: ~5 GB intermediates -- still comfortable
on a 24 GB A10G.

**HLO audit** (in ``test_reml_max_tensor_size_within_budget``):
the compiled HLO is grepped for the largest tensor; we assert
it's well below ``V·N²/2``.  This is the regression-safety
against a future refactor that might inadvertently materialise
a per-voxel covariance.

## `flame_two_level`: FSL FLAME for fMRI group analysis

### Why a dedicated solver

FLAME differs from general REML in one key way: the per-subject
**within-variance is known** (from a level-1 GLM output), not
estimated.  The model is::

    beta_i = X_group gamma + b_i + eps_i
    b_i ~ N(0, sigma_b^2)         (between-subject, unknown)
    eps_i ~ N(0, s_i^2)           (within-subject, known)
    V = sigma_b^2 I + diag(s_i^2)

This is naturally a **single-parameter** REML: only
``sigma_b^2`` is estimated; ``s_i^2`` is fixed.

Initial implementation attempted to reuse ``reml_fit`` with
``V_basis_diag = [ones, var_within]`` and two free parameters.
This was **non-identifiable**: a free scaling on the
``var_within`` component traded off with ``sigma_b^2``,
producing biased estimates (``sigma_b_sq`` ~ 0.13 instead of
0.25 on synthetic FLAME data).  The fix: a dedicated
single-parameter solver in ``lme/flame.py``.

### Single-parameter Newton

For 1-D Newton on ``log_sigma_b_sq``:

- Gradient: scalar.
- Hessian: scalar (second derivative).
- Newton step: ``new = old - grad / hess`` (with Hessian
  positivity guard).

Identical robustness machinery as ``reml_fit`` (damping, step
clipping, backtracking).  Fixed-iteration scan; vmap-batches.

### Validation

- ``test_flame_recovers_true_between_variance``: synthetic
  FLAME data at ``V = 200``, ``N = 60``, ``sigma_b = 0.5``;
  recovers ``sigma_b_sq`` to within 5% relative error.
- ``test_flame_voxelwise_per_voxel_match_unbatched``: batched
  vs per-voxel match to ``1e-7`` (numerical noise from
  iterative solver).

## Differentiability

Both solvers use ``lax.scan`` over a fixed number of Newton
steps; each step is fully differentiable, so backward-mode AD
through the fit works (unrolled gradient through the scan).
``test_reml_differentiable`` verifies finite gradients via
``jax.grad`` over a small ``n_iter=10`` fit.

Caveat: unrolled-Newton AD has a ``O(n_iter)`` memory cost
during backward (the scan stacks per-step intermediates).  For
applications that differentiate through the fit (e.g.,
differentiable model selection), pass a smaller ``n_iter`` or
wait for the implicit-function-theorem VJP follow-up that would
collapse the ``n_iter`` memory to a single inverse-Hessian
solve at the fixed point.

## Validation against `statsmodels.MixedLM`

The gold-standard reference for LME is ``statsmodels.MixedLM``
(R's ``lme4`` equivalent in Python).
``test_reml_matches_statsmodels_reference`` checks that
``reml_fit`` produces:

- ``beta_hat`` within ``5e-3`` of statsmodels.
- ``sigma_b_sq`` within ``5e-3``.
- ``sigma_e_sq`` within ``5e-3``.

These are different solvers (Newton-scoring vs L-BFGS) hitting
the same optimum; ``5e-3`` is the convergence floor of both,
not a slack on accuracy.  At higher ``n_iter`` and tighter
``damping`` we can drive this lower; but the practical fMRI
use case doesn't need more precision than the level-1 inputs'
sampling noise.

## What we considered and didn't pick

- **Multiple random-effect variance components.** The current
  implementation assumes a single between-variance ``sigma_b^2``
  (``Z`` columns share a common variance).  For models with
  multiple independent random effects (e.g., random intercept +
  random slope), we'd need ``K > 2`` variance components.  The
  spectral trick still applies (per-component basis matrices),
  but the API surface and the ``V_basis_diag`` extension is
  defered until a consumer asks.
- **Maximum likelihood (ML)** instead of REML.  REML is the
  default for LME variance-component estimation; ML estimates
  are biased downward for variance components.  Adding an
  ``estimator='ml' | 'reml'`` knob is a one-line change but
  defered.
- **L-BFGS via ``jax.scipy.optimize.minimize``.** Has line
  search built in; but its ``lax.while_loop`` is not
  cleanly vmap-compatible (variable iteration counts across
  voxels).  Newton scoring with fixed iterations is more
  vmap-friendly.
- **Implicit-function-theorem VJP.** Would collapse the
  ``n_iter``-deep unrolled gradient to a single inverse-Hessian
  solve at the fixed point.  Math is clear (same pattern as
  LOBPCG-IVJP) but the gain is marginal until a consumer wants
  gradients through a converged LME fit.
- **EM algorithm** for variance components.  Slower than
  Newton (linear vs quadratic convergence) but boundary-stable.
  Newton + backtracking handles the boundary cases well enough.
- **Kullback-Leibler / robust LME.** Out of scope at first GA.

## Cross-references

- ``src/nitrix/stats/lme/{reml, flame}.py``.
- ``tests/test_lme.py`` -- 10 tests including balanced one-way
  closed form, statsmodels parity, FLAME bias check, HLO memory
  audit, differentiability.
- ``src/nitrix/linalg/_solver.safe_eigh`` -- the cuSolver-robust
  eigendecomposition used for the FaST-LMM rotation.
- Lippert, C., Listgarten, J., et al. (2011).  FaST linear mixed
  models for genome-wide association studies. *Nat. Methods* 8,
  833-835.
- Beckmann, C. F., Jenkinson, M., & Smith, S. M. (2003).  General
  multilevel linear modeling for group analysis in fMRI.
  *NeuroImage* 20, 1052-1063.
- Lindstrom, M. J., & Bates, D. M. (1990). Newton-Raphson and
  EM algorithms for linear mixed-effects models. *JASA* 83,
  1014-1022.
