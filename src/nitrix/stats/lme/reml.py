# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Variance-components REML for voxelwise LMEs.

The model
---------

For each voxel ``v``, the LME is::

    y_v = X beta_v + Z b_v + eps_v
    b_v ~ N(0, sigma_b^2 I_q)
    eps_v ~ N(0, sigma_e^2 I_N)
    Cov(y_v) = V = sigma_b^2 ZZ^T + sigma_e^2 I_N

The fixed-effect design ``X`` and random-effect design ``Z`` are
**shared** across voxels (the typical fMRI / dMRI case: one
group-level design, applied to every voxel's response).  Only
``y_v`` varies per voxel.

The profile REML (Restricted Maximum Likelihood) negative log-
likelihood, after profiling out ``beta``, is::

    nll(theta) = 0.5 * [log|V| + log|X^T V^{-1} X| + r^T V^{-1} r]

where ``r = y - X beta_hat`` and ``beta_hat = (X^T V^{-1} X)^{-1}
X^T V^{-1} y``.  We parameterise ``theta = log(sigma^2)`` (log-
space) so optimisation is unconstrained.

The FaST-LMM spectral trick
---------------------------

Lippert et al. 2011 (FaST-LMM): eigendecompose ``ZZ^T = U Lambda
U^T``.  In the rotated basis ``y_rot = U^T y``, ``X_rot = U^T X``,
the total covariance is diagonal::

    V_rot = sigma_b^2 Lambda + sigma_e^2 I = diag(d)
    d_i = sigma_b^2 lambda_i + sigma_e^2

Every operation in the Newton iteration becomes elementwise on
``d``: ``log|V| = sum_i log d_i``; ``V^{-1}`` is ``diag(1/d)``;
``X^T V^{-1} X = sum_i x_i x_i^T / d_i``.  Per-iteration cost
drops from ``O(N^3)`` (naive) to ``O(N p^2 + N)`` (rotated).

The eigendecomposition of ``ZZ^T`` is computed **once** at the
outer call -- shared across all voxels via vmap closure.

Memory regime
-------------

For ``V`` voxels, ``N`` subjects, ``p`` fixed-effect coefficients:

- Shared (computed once):
  - ``U``, ``Lambda``: ``(N, N)`` + ``(N,)``  --  ``~N^2 * 4`` bytes.
  - ``X_rot``: ``(N, p)``.
- Per-voxel (vmap):
  - ``y_rot``: ``(N,)``.
  - ``XtVinvX``: ``(p, p)``.  Tiny.
  - ``beta``: ``(p,)``.
- Per-Newton-step intermediates: ``(N,)`` arrays.

Total HBM at ``V = 100k``, ``N = 30``, ``p = 5``:

- ``Y``: ``100k * 30 * 4 = 12 MB``.
- Per-voxel results: ``100k * (2 + 5 + 1) * 4 = ~3 MB``.
- Newton-step intermediates (vmapped): ``100k * 30 * 4 = 12 MB`` peak.

Total ~30 MB.  Fits trivially.  At ``V = 1M``, ``N = 100``,
``p = 10``: ``~500 MB`` -- still comfortable on a 24 GB GPU.

Solver
------

The per-voxel inner loop is the shared ``_varcomp`` AI-REML engine:
size-dispatched fixed-effect algebra (closed-form for ``p <= 2``,
``LU`` for ``p > 2`` -- no per-voxel cuSOLVER ``potrf`` / ``syevd``)
and *analytic* REML score + average-information curvature (no
second-order autodiff through a Cholesky).  See ``_varcomp.py`` for
the derivation.  The ``ZZ^T`` eigendecomposition that supplies the
diagonalising basis is the one shared, one-off ``safe_eigh`` call.

Differentiability
-----------------

Implementation uses ``jax.lax.scan`` over a fixed number of
Newton steps; each step is fully differentiable, so backward-
mode AD through the fit works (unrolled gradient through the
scan).  At ``n_iter = 20`` the unrolled grad has ~20 stacked
sub-graphs -- a real memory cost.  For applications that
differentiate through the fit (e.g., differentiable model
selection), pass a smaller ``n_iter`` or wait for the implicit-
function-theorem VJP follow-up.

References
----------
- Lindstrom, M. J., & Bates, D. M. (1990). Newton-Raphson and
  EM algorithms for linear mixed-effects models.  JASA 83.
- Lippert, C., Listgarten, J., et al. (2011). FaST linear mixed
  models for genome-wide association studies. Nat. Methods 8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._varcomp import VarCompSpec, fit_varcomp_diagonal

__all__ = ['REMLResult', 'reml_fit']


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class REMLResult:
    """Per-voxel REML fit output.

    Attributes
    ----------
    theta_hat
        ``(V, 2)`` -- ``[log(sigma_b^2), log(sigma_e^2)]`` per voxel.
        Take ``jnp.exp`` for the natural variance scale.
    beta_hat
        ``(V, p)`` -- fixed-effect estimates per voxel.
    log_lik
        ``(V,)`` -- profile REML log-likelihood at the fit.
    """

    theta_hat: Float[Array, 'V 2']
    beta_hat: Float[Array, 'V p']
    log_lik: Float[Array, 'V']

    @property
    def sigma_b_sq(self) -> Float[Array, 'V']:
        return jnp.exp(self.theta_hat[..., 0])

    @property
    def sigma_e_sq(self) -> Float[Array, 'V']:
        return jnp.exp(self.theta_hat[..., 1])

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[Float[Array, 'V 2'], Float[Array, 'V p'], Float[Array, 'V']],
        None,
    ]:
        return (self.theta_hat, self.beta_hat, self.log_lik), None

    @classmethod
    def tree_unflatten(
        cls, _aux: None, children: Tuple[Any, ...]
    ) -> REMLResult:
        return cls(*children)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _default_theta_init(
    y_rot: Float[Array, '... N'],
    V_basis_diag: Float[Array, 'K N'],
) -> Float[Array, '... K']:
    """Initialise log-variance components from the empirical data.

    Heuristic: start with a small random-effect variance and most
    of the variance assigned to the residual component.  This
    starting point is closer to the "no random effects" boundary
    than to the "all random" boundary, which makes Newton more
    likely to converge to the true optimum without overshooting
    when the true ``sigma_b^2`` is small.

    Concretely: ``sigma_e^2_init = 0.5 * var(y)``,
    ``sigma_b^2_init = 0.1 * var(y) / mean(lambda)`` (so the
    contribution to the diagonal is comparable in magnitude to
    ``sigma_e^2_init``).  For the FLAME case where the first basis
    is ``ones`` and the second is ``var_within``, this gives
    sensible scaling automatically.
    """
    y_var = jnp.var(y_rot, axis=-1, keepdims=True)  # (..., 1)
    K = V_basis_diag.shape[0]
    # Average diagonal contribution of each basis matrix to V.
    basis_scale = jnp.mean(V_basis_diag, axis=-1)  # (K,)
    basis_scale = jnp.where(basis_scale > 1e-12, basis_scale, 1.0)
    # Allocate var(y) primarily to the LAST component (typically the
    # residual identity); 10% to the others.
    weights = jnp.full((K,), 0.1, dtype=y_var.dtype).at[-1].set(0.5)
    weights = weights / jnp.sum(weights)
    # Per-component initial variance: var(y) * weight / basis_scale.
    init = jnp.log(
        jnp.maximum(y_var * weights / basis_scale, 1e-6),
    )
    return init


def reml_fit(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    Z: Float[Array, 'N q'],
    *,
    theta_init: Optional[Float[Array, 'V 2']] = None,
    n_iter: int = 20,
    damping: float = 1e-6,
    block: Optional[int] = None,
) -> REMLResult:
    """Voxelwise variance-components REML fit.

    Fits the LME

    .. code::

        y_v = X beta_v + Z b_v + eps_v
        b_v ~ N(0, sigma_b^2 I_q)
        eps_v ~ N(0, sigma_e^2 I_N)

    independently for each voxel, sharing the fixed-effect design
    ``X`` and the random-effect design ``Z`` across voxels.  Uses
    the FaST-LMM spectral trick: eigendecompose ``ZZ^T`` once,
    rotate ``y`` and ``X`` into the diagonalising basis, then
    Newton-score on the variance components in log-space.

    Parameters
    ----------
    Y
        Response tensor, ``(V, N)`` -- ``V`` voxels, ``N`` subjects
        (or observations).
    X
        Fixed-effect design, ``(N, p)``.  Shared across voxels.
    Z
        Random-effect design, ``(N, q)``.  Shared across voxels.
        The random-effect covariance is ``sigma_b^2 I_q`` (single
        variance component); pass each component's design matrix
        column to ``Z`` to model multiple random effects (currently
        all share ``sigma_b^2``; for separate variance components
        per random effect, the lower-level ``_varcomp`` core accepts a
        general ``(K, N)`` basis).
    theta_init
        Per-voxel initial log-variances, ``(V, 2)``.  Defaults to a
        heuristic split of empirical variance.
    n_iter
        Fixed Newton-scoring iteration count.  Default ``20`` --
        typically converges in 5-10 for well-conditioned data.
    damping
        Levenberg-Marquardt-style damping on the average-information
        matrix.  Default ``1e-6``; raise if Newton steps are unstable
        near boundaries.
    block
        Optional voxel-block size: cap the number of voxels whose
        per-Newton intermediates are live at once, bounding peak memory
        on brain-scale ``V``.  ``None`` (default) is a single ``vmap``
        over all voxels (identical numerics and HLO to before).

    Returns
    -------
    ``REMLResult`` with per-voxel ``theta_hat`` (log-variances),
    ``beta_hat`` (fixed effects), and ``log_lik``.

    Notes
    -----
    Eigendecomposition of ``ZZ^T`` uses ``safe_eigh`` (cuSolver-
    robust fallback) -- the one shared, one-off solver call.  The
    per-voxel inner loop (``_varcomp``) makes no cuSOLVER call: the
    fixed-effect algebra is closed-form for ``p <= 2`` and ``LU``-based
    for ``p > 2``, and the REML derivatives are analytic.
    """
    from ...linalg._solver import safe_eigh

    n = X.shape[0]
    if Z.shape[0] != n:
        raise ValueError(
            f'reml_fit: Z.shape[0]={Z.shape[0]} must match X.shape[0]={n}.'
        )
    if Y.shape[-1] != n:
        raise ValueError(
            f'reml_fit: Y.shape[-1]={Y.shape[-1]} must match N={n}.'
        )

    # Eigendecompose ZZ^T (shared across voxels).
    ZZt = Z @ Z.T
    ZZt = 0.5 * (ZZt + ZZt.T)  # symmetrise against drift
    eigvals, U = safe_eigh(ZZt)
    # Clamp negative eigenvalues from roundoff.
    lambdas = jnp.clip(eigvals, 0.0, None)

    # Rotate Y (per-voxel) and X (shared).
    Y_rot = Y @ U  # (V, N)
    X_rot = U.T @ X  # (N, p)

    # V_basis_diag = (lambdas, ones) -- shared across voxels.
    V_basis_diag = jnp.stack(
        [lambdas, jnp.ones_like(lambdas)],
        axis=0,
    )

    if theta_init is None:
        theta_init = _default_theta_init(Y_rot, V_basis_diag)

    spec = VarCompSpec.reml(n_iter=n_iter, damping=damping)
    theta_hat, beta_hat, log_lik = fit_varcomp_diagonal(
        Y_rot,
        X_rot,
        V_basis_diag,
        theta_init,
        spec=spec,
        block=block,
    )
    return REMLResult(
        theta_hat=theta_hat,
        beta_hat=beta_hat,
        log_lik=log_lik,
    )
