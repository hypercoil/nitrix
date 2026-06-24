# linalg: matrix utilities, kernels, residualisation, SPD ops

> **TL;DR.**  ``nitrix.linalg`` ships four submodules consolidated
> from the legacy ``hypercoil.functional.{matrix, resid, kernel,
> symmap, semidefinite}`` and the partial ``nitrix.functional.{matrix,
> residual}``: ``matrix`` (sym↔vec bijection, Toeplitz, diagonal
> helpers, eigenspace reconditioning), ``residual``
> (OLS / WLS / ridge regression via Cholesky-default with SVD
> fallback), ``kernel`` (linear / RBF / polynomial / sigmoid /
> cosine; squared-L2 via the identity-formula to avoid the legacy
> ``O(n m d)`` materialisation), ``spd`` (matrix log / exp / sqrt /
> power / tangent-space projection / log-Euclidean mean with the
> SPEC §4.5 stability rewrite), and a shared ``_solver`` helper
> for cuSolver-robust eigh.

## Why a green-field rewrite

The user invoked the "code salvage and retrofit" mode with explicit
permission to drop backward-compatible naming and signatures.  The
legacy code is treated as a template for *what* to compute, not
*how*.  Three patterns drove material changes:

1. **The ``form_docstring`` decoration machinery.**  Adds ~50 lines
   of dim-spec / param-spec template per module.  Dropped in favour
   of plain docstrings -- jaxtyping handles the type signature; one-
   line summaries are clearer than templated tables for the
   call-site reader.
2. **The legacy "JIT trap" pattern.**  Several legacy functions did
   runtime checks on tracer values (``_is_diagonal`` in covariance;
   per-shape branching in window-sampling).  Replaced with shape-
   based dispatch only (JIT-stable).
3. **The legacy ``fill_nans`` / ``truncate_eigenvalues`` flags.**
   These masked numerical issues rather than fixing them.  The new
   code surfaces ``NaN`` when the input is genuinely ill-conditioned
   and uses *principled* clipping (rank-truncation threshold from
   numerical linear algebra) when the input is well-formed but
   ill-conditioned.

## `matrix`: the sym↔vec bijection plus utilities

Direct port of the existing ``nitrix.functional.matrix`` symbols
with documentation cleanup.  The custom-VJP rules for ``sym2vec``
and ``vec2sym`` are load-bearing:

- ``sym2vec`` strictly **drops** the lower triangle, so the backward
  must place zeros there -- not mirror.  Without the custom VJP,
  JAX would symmetrise the gradient and double-count the
  off-diagonals.
- ``vec2sym`` implicitly **mirrors** the upper triangle, so the
  backward must scale off-diagonal cotangents by 2 to account for
  the duplication.

These are tested explicitly:

- ``test_sym2vec_grad_zeros_lower_triangle`` -- forward drops it;
  backward zeros it.
- ``test_vec2sym_grad_doubles_off_diagonal`` -- ``dL/dv_k = 4 * v_k``
  for an ``||M||^2`` loss (factor of 4 from the mirror plus the
  quadratic).

Dropped from the legacy hypercoil version:

- ``cholesky_invert`` -- slower than ``jnp.linalg.inv`` in nearly
  all cases.
- ``spd`` -- the legacy crude PSD-projection.  Use
  ``recondition_eigenspaces`` for the differentiability-stability
  case; use ``linalg.spd`` for SPD-manifold ops proper.
- ``expand_outer`` -- just write the ``jnp.einsum`` you want.

## `residual`: Cholesky-default OLS / WLS / ridge

Two key changes from ``hypercoil.functional.resid`` and the
prior ``nitrix.functional.residual``:

1. **Default solver is Cholesky on the normal equations**, not SVD
   via ``lstsq``.  For tall-and-skinny ``X`` (typical fMRI:
   400 TRs × 24 confounds), Cholesky is ~9× faster than the SVD
   path; the heavy step ``X^T X`` is bandwidth-limited at
   ``O(obs * k^2)`` and the ``k × k`` Cholesky is trivial
   (``O(k^3)``).  We verified the Cholesky / SVD agreement to
   ``2e-15`` at fp64 (machine eps) on well-conditioned random data
   -- the ``test_residualise_cholesky_vs_svd_parity_at_fp64`` is
   the load-bearing test of that claim.
2. **Per-observation weights (WLS / GLS-style)** via the
   ``weights`` argument.  Internally pre-scales rows by
   ``sqrt(weights)``; algebraically identical to forming
   ``X^T W X`` directly.  Tested against the
   "replicated-observations" equivalence: integer weights
   ``[1, 2]`` repeated should match the unweighted fit on
   `[1, 2, 1, 2, ...]`-replicated data.

A QR path was considered (and is asymptotically the fastest stable
option for over-determined systems) but **QR doesn't work on the
test runner's GPU**: ``jnp.linalg.qr`` hits the same cuSolver-broken
handle that's documented in
[`graph.md`](graph.md) and [`lobpcg-implicit-vjp.md`](lobpcg-implicit-vjp.md).
Cholesky uses different routines and works fine; SVD works fine
via a different code path.  If a future stack fixes the cuSolver
issue, adding a ``method='qr'`` is a one-function addition; until
then, exposing it would just confuse users.

## `kernel`: kernels + the squared-L2-via-identity trick

The single material improvement over the legacy
``hypercoil.functional.kernel`` is in ``linear_distance``:

- **Legacy**: ``D = X0[:, None, :] - X1[None, :, :]`` then squared.
  Materialises ``(n, m, d)`` -- ``O(n m d)`` memory.  At
  ``n = m = 1000, d = 1000`` (e.g., voxel-by-voxel similarity over
  ROI signatures), that's 4 GB at fp32 just for the difference
  tensor.
- **New**: ``|x - y|^2 = |x|^2 + |y|^2 - 2 x . y``.  Memory is
  ``(n, m)`` -- ``O(n m)``.  At the same shape, 4 MB.  **1000×
  memory reduction.**  The numerical pitfall (catastrophic
  cancellation when ``x ≈ y`` produces small negative values from
  roundoff) is handled by ``jnp.maximum(dist_sq, 0)``.

Tested against brute-force reference for the L2 and the
diagonal- and full-Mahalanobis cases; all agree to ``5e-14`` at
fp64.

Dropped from the legacy:

- ``cov_kernel`` / ``corr_kernel`` -- thin wrappers around
  ``stats.pairedcov`` / ``stats.pairedcorr``.  Call those directly.
- The ``singledispatch`` sparse overloads for ``BCOO`` / ``TopK``.
  ELL and SectionedELL are the project's sparse formats; users
  wanting a sparse kernel matrix build it via ``semiring_ell_matmul``
  with the appropriate algebra.

## `spd`: the SPEC §4.5 stability rewrite

The legacy ``symmap`` had a documented stability gap (SPEC §4.5):
applying ``log`` / ``sqrt`` to eigenvalues underflowed for small
eigenvalues, and the ``fill_nans=True`` flag silently substituted
``0`` for ``NaN`` -- mathematically wrong (``log(0) = -inf``,
not ``0``), but masked the problem.

The new ``symmap`` separates two concerns:

1. **Forward stability**: ``eigvalue_clip='auto'`` (default) clips
   eigenvalues at the rank-truncation threshold ``max(|L|) * d *
   eps(dtype)`` -- matches ``numpy.linalg.matrix_rank``'s default.
   At fp64 with ``max = 1e5, d = 5``, threshold is ``~1e-10``;
   eigenvalues below that are floored to the threshold, so
   ``log`` produces a (finite, very negative) value rather than
   ``-inf``.
2. **Backward stability**: ``psi > 0`` reconditions the input
   before ``eigh`` (perturbing degenerate eigenvalues).  This is
   the differentiability hook: ``eigh``'s registered VJP F-matrix
   formula divides by ``(λ_j - λ_i)`` which blows up at near-
   degenerate spectra.  The legacy code had this; we keep it but
   make it default ``psi = 0`` so the *forward* is exact unless
   the user explicitly asks for gradient stability.

Tested explicitly:

- ``test_symlog_ill_conditioned_is_finite`` -- 20 orders-of-
  magnitude eigenvalue spread; output is finite.
- ``test_symlog_clip_floor_uses_eps_threshold`` -- the clipped
  output's smallest log-eigval matches ``log(threshold)`` to
  ``1e-8``.
- ``test_symlog_with_psi_reconditioning_handles_degeneracy`` --
  near-degenerate spectrum produces finite gradients with
  ``psi > 0``.

Dropped from the legacy ``mean_*`` family:

- ``mean_euc_spd`` -- now just ``mean_euclidean``.
- ``mean_harm_spd``, ``mean_kullback_spd``, ``mean_geom_spd`` --
  the iterative means.  Defer to follow-up; the closed-form
  ``mean_log_euclidean`` covers the common fMRI / dMRI use case
  (Pennec et al. 2006 show log-Euclidean is a near-perfect proxy
  for the affine-invariant Fréchet mean at typical batch spreads).

## `_solver`: the cuSolver-robust eigh helper

Extracted from ``graph.connectopy._safe_eigh`` (where it was
originally inlined).  Now shared between ``graph.connectopy``,
``graph._lobpcg_diff``, ``linalg.spd``, and
``signal.lomb_scargle``.  Centralising the device-probe + cache
means a single source of truth for the cuSolver-broken stack
detection; one bug to fix when a future driver / cuSolver build
fixes the underlying issue.

The key correctness fix made during the migration: **always
``device_put`` to the safe device, even under trace**.  Under
``jax.grad`` the input is a tracer with no concrete device, so
the original "if source is concrete, move; else trust dispatch"
logic fell through to the broken GPU path.  The new
unconditional ``device_put`` is a no-op at JIT time and a real
move at concrete-eval time -- correct in both regimes.

## Cross-references

- ``src/nitrix/linalg/{matrix,residual,kernel,spd,_solver}.py``.
- ``tests/test_linalg.py``, ``tests/test_linalg_spd.py`` -- 29 + 18 tests.
- [`graph.md`](graph.md) -- the ``connectopy`` consumer of ``_solver``.
- [`lobpcg-implicit-vjp.md`](lobpcg-implicit-vjp.md) -- the LOBPCG
  consumer of ``_solver``.
- SPEC §4.5 (SPD stability), §4.2 (kernel surface).
