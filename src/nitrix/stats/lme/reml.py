# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Variance-components REML for voxelwise linear mixed-effects models.

The model
---------

For each voxel :math:`v`, the linear mixed-effects model is

.. code::

    y_v = X beta_v + Z b_v + eps_v

with :math:`b_v \\sim \\mathcal{N}(0, \\sigma_b^2 I_q)`,
:math:`\\epsilon_v \\sim \\mathcal{N}(0, \\sigma_e^2 I_N)`, so the marginal
covariance of the response is
:math:`V = \\sigma_b^2 Z Z^{\\top} + \\sigma_e^2 I_N`.

The fixed-effect design ``X`` and random-effect design ``Z`` are
**shared** across voxels (the typical fMRI / dMRI case: one
group-level design, applied to every voxel's response).  Only the
per-voxel response :math:`y_v` varies.

The profile REML (Restricted Maximum Likelihood) negative log-
likelihood, after profiling out :math:`\\beta`, is

.. math::

    \\mathrm{nll}(\\theta) = \\tfrac{1}{2}
        \\bigl[\\log|V| + \\log|X^{\\top} V^{-1} X|
        + r^{\\top} V^{-1} r\\bigr]

where :math:`r = y - X \\hat{\\beta}` and
:math:`\\hat{\\beta} = (X^{\\top} V^{-1} X)^{-1} X^{\\top} V^{-1} y`.  The
variance components are parameterised as :math:`\\theta = \\log(\\sigma^2)`
(log-space) so optimisation is unconstrained.

The FaST-LMM spectral trick
---------------------------

Following Lippert et al. (2011), eigendecompose
:math:`Z Z^{\\top} = U \\Lambda U^{\\top}`.  In the rotated basis
:math:`y_{\\mathrm{rot}} = U^{\\top} y`,
:math:`X_{\\mathrm{rot}} = U^{\\top} X`, the total covariance is diagonal:

.. math::

    V_{\\mathrm{rot}} = \\sigma_b^2 \\Lambda + \\sigma_e^2 I
        = \\operatorname{diag}(d), \\qquad
    d_i = \\sigma_b^2 \\lambda_i + \\sigma_e^2 .

Every operation in the Newton iteration becomes elementwise on
:math:`d`: :math:`\\log|V| = \\sum_i \\log d_i`; :math:`V^{-1}` is
:math:`\\operatorname{diag}(1/d)`;
:math:`X^{\\top} V^{-1} X = \\sum_i x_i x_i^{\\top} / d_i`.  Per-iteration
cost drops from :math:`O(N^3)` (naive) to :math:`O(N p^2 + N)` (rotated).

The eigendecomposition of :math:`Z Z^{\\top}` is computed **once** at the
outer call -- shared across all voxels via a vmap closure.

Memory regime
-------------

For ``V`` voxels, ``N`` subjects, ``p`` fixed-effect coefficients:

- Shared (computed once):
  - ``U``, :math:`\\Lambda`: ``(N, N)`` + ``(N,)``  --  :math:`\\sim 4 N^2`
    bytes.
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
size-dispatched fixed-effect algebra (closed-form for :math:`p \\le 2`,
``LU`` for :math:`p > 2` -- no per-voxel cuSOLVER ``potrf`` / ``syevd``)
and *analytic* REML score + average-information curvature (no
second-order autodiff through a Cholesky).  See ``_varcomp`` for
the derivation.  The :math:`Z Z^{\\top}` eigendecomposition that supplies
the diagonalising basis is the one shared, one-off ``safe_eigh`` call.

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
- Lindstrom, M. J., & Bates, D. M. (1988). Newton-Raphson and
  EM algorithms for linear mixed-effects models for repeated-measures
  data.  Journal of the American Statistical Association, 83(404),
  1014-1022.  https://doi.org/10.1080/01621459.1988.10478693
- Lippert, C., Listgarten, J., Liu, Y., Kadie, C. M., Davidson, R. I.,
  & Heckerman, D. (2011). FaST linear mixed models for genome-wide
  association studies.  Nature Methods, 8(10), 833-835.
  https://doi.org/10.1038/nmeth.1681
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal, NamedTuple, Optional, Tuple, Union, cast

import jax
import jax.numpy as jnp
from jax.scipy.special import betainc
from jaxtyping import Array, Float, Int

from ...linalg._smalllinalg import small_inv_logdet, sym_eig_jacobi
from .._batching import blocked_vmap
from .._result import register_result
from ._blup import _blups_standard
from ._corr import CorrSpec, resolve_corr
from ._corrfit import CorrLMEResult
from ._kr import kr_cov_and_scaled_f
from ._varcomp import VarCompSpec, fit_varcomp_diagonal, varcomp_inference

# MC7: the profile-REML cores (_varcomp / _blockwoodbury / _crossed / _nested)
# return ``log_lik = -0.5 [log|V| + log|X^T V^{-1} X| + r^T V^{-1} r]`` -- the
# *non-normalised* restricted log-likelihood, omitting the ``(N - p) log(2 pi)``
# Gaussian constant that the GLS path (and statsmodels) include.  Subtracting
# ``(N - p) * _HALF_LOG_2PI`` makes every tier report the **full** restricted
# log-likelihood, so log_lik is comparable across tiers (and AIC/BIC valid).
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def _kr_rotated_basis(
    X: Float[Array, 'N p'], Z: Float[Array, 'N q']
) -> Tuple[Float[Array, 'N p'], Float[Array, 'N']]:
    """Rebuild the shared FaST-LMM rotation from the random-effect design.

    Eigendecomposes :math:`Z Z^{\\top}` (the one-off ``safe_eigh`` call the
    Kenward-Roger correction needs to rebuild the rotated basis) and returns the
    fixed-effect design rotated into the diagonalising eigenbasis together with
    the clamped eigenvalues.

    Parameters
    ----------
    X
        Fixed-effect design, ``(N, p)``.
    Z
        Random-effect design, ``(N, q)``.

    Returns
    -------
    X_rot : Float[Array, 'N p']
        The fixed-effect design rotated into the eigenbasis of
        :math:`Z Z^{\\top}`, :math:`U^{\\top} X`.
    lambdas : Float[Array, 'N']
        The eigenvalues of :math:`Z Z^{\\top}`, clamped to be non-negative.
    """
    from ...linalg._solver import safe_eigh

    Xj, Zj = jnp.asarray(X), jnp.asarray(Z)
    ZZt = Zj @ Zj.T
    eigvals, U = safe_eigh(0.5 * (ZZt + ZZt.T))
    return U.T @ Xj, jnp.clip(eigvals, 0.0, None)


__all__ = [
    'REMLResult',
    'LMEResult',
    'NestedLMEResult',
    'CrossedLMEResult',
    'LMEContrast',
    'LMEFContrast',
    'reml_fit',
    'lme_fit',
    'lme_t_contrast',
    'lme_f_contrast',
]

Structure = Literal['unstructured', 'diagonal']
ContrastDof = Literal['satterthwaite', 'kr']


@register_result(
    children=(
        'theta_hat',
        'beta_hat',
        'log_lik',
        'fixed_cov',
        'theta_cov',
        'grad_m',
        'blups',
    ),
)
@dataclass(frozen=True)
class REMLResult:
    """Per-voxel REML fit output.

    Attributes
    ----------
    theta_hat
        ``(V, 2)`` -- :math:`[\\log(\\sigma_b^2), \\log(\\sigma_e^2)]` per voxel.
        Take ``jnp.exp`` for the natural variance scale.
    beta_hat
        ``(V, p)`` -- fixed-effect estimates per voxel.
    log_lik
        ``(V,)`` -- profile REML log-likelihood at the fit.
    fixed_cov
        ``(V, p, p)`` -- :math:`\\operatorname{Cov}(\\hat{\\beta})
        = (X^{\\top} V^{-1} X)^{-1}` at the fit.  The fixed-effect covariance for
        a contrast test (:func:`lme_t_contrast`).
    theta_cov
        ``(V, 2, 2)`` -- :math:`\\operatorname{Cov}(\\hat{\\theta})` (inverse
        average-information), for the Satterthwaite denominator degrees of
        freedom.
    grad_m
        ``(V, 2, p, p)`` -- the contrast-independent Satterthwaite gradient
        tensors :math:`M_k` (see ``_varcomp.varcomp_inference``).
    """

    theta_hat: Float[Array, 'V 2']
    beta_hat: Float[Array, 'V p']
    log_lik: Float[Array, 'V']
    fixed_cov: Float[Array, 'V p p']
    theta_cov: Float[Array, 'V 2 2']
    grad_m: Float[Array, 'V 2 p p']
    blups: Optional[Float[Array, 'V q']] = None
    """``(V, q)`` per-group random-intercept BLUPs, or ``None`` when the fit did
    not retain them (``lme_fit(..., retain_blups=False)``, the default).  Read
    via :func:`~nitrix.stats.ranef`; consumed by
    :func:`~nitrix.stats.lme_predict` at ``level='conditional'``."""

    @property
    def sigma_b_sq(self) -> Float[Array, 'V']:
        return jnp.exp(self.theta_hat[..., 0])

    @property
    def sigma_e_sq(self) -> Float[Array, 'V']:
        return jnp.exp(self.theta_hat[..., 1])

    @property
    def cov_re(self) -> Float[Array, 'V 1 1']:
        """Random-effect covariance as the uniform ``(V, k, k)`` block (D2): the
        single scalar variance as a ``(V, 1, 1)``, matching
        :attr:`LMEResult.cov_re`."""
        return self.sigma_b_sq[:, None, None]

    @property
    def re_labels(self) -> Tuple[str, ...]:
        """Names of the ``k`` random-effect dimensions of :attr:`cov_re`."""
        return ('group',)

    @property
    def coef(self) -> Float[Array, 'V p']:
        """Alias for :attr:`beta_hat` -- the fixed-effect coefficients, named
        ``coef`` for cross-suite parity with GLM / GAM / GP / HGP results (UX1)."""
        return self.beta_hat


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
    when the true :math:`\\sigma_b^2` is small.

    Concretely, the residual component is initialised at
    :math:`0.5 \\cdot \\operatorname{var}(y)` and each random component at
    :math:`0.1 \\cdot \\operatorname{var}(y) / \\operatorname{mean}(\\mathrm{basis})`
    (so its contribution to the diagonal is comparable in magnitude to the
    residual initialiser), floored at ``1e-6`` so the log is finite.

    This is the default for :func:`reml_fit` only;
    :func:`~nitrix.stats.flame_two_level` supplies its own initialiser.

    Parameters
    ----------
    y_rot
        ``(..., N)`` per-voxel response rotated into the diagonalising basis
        (leading dimensions broadcast over voxels).
    V_basis_diag
        ``(K, N)`` diagonal contribution of each of the ``K`` variance
        components to the marginal covariance in the rotated basis (rows are the
        clamped eigenvalues and the residual ones-row).

    Returns
    -------
    Float[Array, '... K']
        The per-component initial log-variances, one entry per variance
        component.

    Notes
    -----
    For a random-effect design whose basis (:math:`Z Z^{\\top}` spectrum) has a
    large mean, the random-component initialiser can hit the ``1e-6`` floor and
    start Newton near the :math:`\\sigma_b^2 \\to 0` boundary; the damped and
    backtracked iteration still climbs off it given enough iterations, but raise
    ``n_iter`` for such designs.
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


def _lowrank_theta_init(
    Y: Float[Array, 'V N'],
    s2: Float[Array, 'q'],
    n: int,
) -> Float[Array, 'V 2']:
    """Low-rank counterpart of :func:`_default_theta_init`.

    Reproduces the dense heuristic's split (10% of :math:`\\operatorname{var}(y)`
    to the random component, 50% to the residual, normalised) using the *full*
    basis scale :math:`\\operatorname{mean}_N(\\lambda) = \\sum s^2 / N` -- so the
    low-rank fit starts from the same :math:`\\theta` as the dense path and
    converges to the same optimum.

    Parameters
    ----------
    Y
        ``(V, N)`` per-voxel responses.
    s2
        ``(q,)`` range spectrum -- the non-negative eigenvalues of
        :math:`Z^{\\top} Z` (equivalently the non-zero eigenvalues of
        :math:`Z Z^{\\top}`).
    n
        The number of observations ``N`` (used to form the full basis scale).

    Returns
    -------
    Float[Array, 'V 2']
        The per-voxel initial log-variances
        :math:`[\\log \\sigma_b^2, \\log \\sigma_e^2]`.
    """
    y_var = jnp.var(Y, axis=-1, keepdims=True)  # (V, 1)
    basis_scale_b = jnp.maximum(jnp.sum(s2) / n, 1e-12)  # mean_N(lambda)
    w = jnp.asarray([0.1, 0.5], dtype=Y.dtype)
    w = w / jnp.sum(w)
    sb2 = jnp.maximum(y_var * w[0] / basis_scale_b, 1e-6)
    se2 = jnp.maximum(y_var * w[1], 1e-6)
    return jnp.concatenate([jnp.log(sb2), jnp.log(se2)], axis=-1)  # (V, 2)


def _reml_fit_lowrank(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    Z: Float[Array, 'N q'],
    theta_init: Optional[Float[Array, 'V 2']],
    n_iter: int,
    damping: float,
    block: Optional[int],
) -> REMLResult:
    """FaST-LMM low-rank REML fit.

    Runs the FaST-LMM spectral fit through the low-rank decomposition: a
    :math:`q \\times q` eigendecomposition of :math:`Z^{\\top} Z` in place of the
    dense :math:`N \\times N` eigendecomposition of :math:`Z Z^{\\top}`.  Requires
    :math:`q \\le N`.  The range factor is
    :math:`Z Z^{\\top} = U_r \\operatorname{diag}(s^2) U_r^{\\top}` with
    :math:`U_r = Z W / \\sqrt{s^2}` (shape ``(N, q)``), and the :math:`N - q` null
    directions enter only through per-voxel Gram aggregates -- never an
    :math:`N \\times N` factor (see ``_lowrank``).

    Parameters
    ----------
    Y
        ``(V, N)`` per-voxel responses.
    X
        Fixed-effect design, ``(N, p)``.  Shared across voxels.
    Z
        Random-effect design, ``(N, q)``.  Shared across voxels.
    theta_init
        ``(V, 2)`` per-voxel initial log-variances, or ``None`` to use the
        low-rank heuristic (:func:`_lowrank_theta_init`).
    n_iter
        Fixed Newton-scoring iteration count.
    damping
        Levenberg-Marquardt-style damping on the average-information matrix.
    block
        Optional voxel-block size bounding the number of voxels live at once, or
        ``None`` for a single ``vmap`` over all voxels.

    Returns
    -------
    REMLResult
        The per-voxel fit, with the fixed-effect / variance-component inference
        fields populated.
    """
    from ...linalg._solver import safe_eigh
    from ._lowrank import fit_lowrank_reml, lowrank_inference

    n, q = Z.shape
    if q > n:
        raise ValueError(
            f'reml_fit(low_rank=True): needs q <= N, got q={q}, N={n}; the '
            'low-rank form assumes a tall random-effect design.'
        )

    ZtZ = Z.T @ Z  # (q, q) -- the cheap factor
    ZtZ = 0.5 * (ZtZ + ZtZ.T)
    evals, W = safe_eigh(ZtZ)
    s2 = jnp.clip(evals, 0.0, None)  # (q,) range spectrum (ZZ^T eigenvalues)
    s = jnp.sqrt(jnp.clip(s2, jnp.finfo(Y.dtype).tiny, None))
    u_r = (Z @ W) / s[None, :]  # (N, q) range left-singular vectors

    y_range = Y @ u_r  # (V, q)
    x_range = u_r.T @ X  # (q, p)
    # Null-space Gram aggregates: X^T X - X_r^T X_r etc. (the orthogonal
    # complement of the range, summarised -- no per-null-coordinate work).
    gxx = X.T @ X - x_range.T @ x_range  # (p, p) shared
    gxy = Y @ X - y_range @ x_range  # (V, p): X^T y_v - X_r^T y_r,v
    gyy = jnp.sum(Y * Y, axis=-1) - jnp.sum(y_range * y_range, axis=-1)  # (V,)
    n0 = n - q

    if theta_init is None:
        theta_init = _lowrank_theta_init(Y, s2, n)

    spec = VarCompSpec(n_iter=n_iter, damping=damping)
    theta_hat, beta_hat, log_lik = fit_lowrank_reml(
        y_range,
        x_range,
        s2,
        gxx,
        gxy,
        gyy,
        theta_init,
        n0,
        spec=spec,
        block=block,
    )
    p = X.shape[-1]

    def _inf(
        y: Float[Array, 'q'],
        gx: Float[Array, 'p'],
        gy: Float[Array, ''],
        th: Float[Array, '2'],
    ) -> Tuple[
        Float[Array, 'p p'], Float[Array, '2 2'], Float[Array, '2 p p']
    ]:
        return lowrank_inference(th, y, x_range, s2, gxx, gx, gy, n0, p, spec)

    fixed_cov, theta_cov, grad_m = cast(
        Tuple[Array, Array, Array],
        blocked_vmap(_inf, (y_range, gxy, gyy, theta_hat), block=block),
    )
    return REMLResult(
        theta_hat=theta_hat,
        beta_hat=beta_hat,
        log_lik=log_lik - (X.shape[0] - X.shape[1]) * _HALF_LOG_2PI,
        fixed_cov=fixed_cov,
        theta_cov=theta_cov,
        grad_m=grad_m,
    )


# ---------------------------------------------------------------------------
# Fixed-effect contrast inference (Satterthwaite)
# ---------------------------------------------------------------------------


class LMEContrast(NamedTuple):
    """Per-voxel mixed-model t-contrast result.

    Attributes
    ----------
    effect
        ``(V,)`` -- the contrast estimate :math:`c^{\\top} \\hat{\\beta}`.
    se
        ``(V,)`` -- its standard error
        :math:`\\sqrt{c^{\\top} \\operatorname{Cov}(\\hat{\\beta})\\, c}`.
    t
        ``(V,)`` -- the t-statistic ``effect / se``.
    df
        ``(V,)`` -- Satterthwaite denominator degrees of freedom.
    p_value
        ``(V,)`` -- two-sided p-value from ``t`` on ``df`` degrees of freedom.
    """

    effect: Float[Array, 'V']
    se: Float[Array, 'V']
    t: Float[Array, 'V']
    df: Float[Array, 'V']
    p_value: Float[Array, 'V']


def lme_t_contrast(
    result: 'Union[REMLResult, LMEResult]',
    contrast: Float[Array, 'p'],
    *,
    dof: ContrastDof = 'satterthwaite',
    X: Optional[Float[Array, 'N p']] = None,
    Z: Optional[Float[Array, 'N q']] = None,
) -> LMEContrast:
    """Per-voxel t-test of a fixed-effect contrast :math:`c^{\\top} \\beta = 0`.

    The standard error is
    :math:`\\sqrt{c^{\\top} (X^{\\top} V^{-1} X)^{-1} c}` from the fitted
    ``result.fixed_cov``; the denominator degrees of freedom are
    **Satterthwaite** (the default, ``dof='satterthwaite'``):

    .. math::

        \\mathrm{df} = \\frac{2 (c^{\\top} C c)^2}
            {g^{\\top} \\operatorname{Cov}(\\hat{\\theta})\\, g}, \\qquad
        g_k = \\frac{\\partial (c^{\\top} C c)}{\\partial \\theta_k}
            = w^{\\top} M_k w, \\quad w = C c

    using the per-voxel ``theta_cov`` / ``grad_m`` the fit already formed.
    cuSOLVER-free and differentiable.

    ``dof='kr'`` instead applies the **Kenward-Roger** small-sample correction:
    the adjusted covariance :math:`\\Phi_A` (inflated for variance-component
    uncertainty) and the scaled-:math:`F` denominator df, with
    :math:`t = \\operatorname{sign}(\\mathrm{effect}) \\sqrt{F_{\\mathrm{KR}}}`
    on :math:`\\mathrm{df} = m`.  The Kenward-Roger path rebuilds the rotated
    basis, so the original design must be re-passed as ``X=`` / ``Z=`` (the
    compressed :class:`REMLResult` does not retain it).

    Parameters
    ----------
    result
        A :class:`REMLResult` (:func:`reml_fit`, the scalar-intercept tier)
        **or** an :class:`LMEResult` (:func:`lme_fit` with a random slope) --
        both carry the ``fixed_cov`` / ``theta_cov`` / ``grad_m`` inference
        fields the Satterthwaite df consumes.  ``dof='kr'`` accepts only a
        :class:`REMLResult`.
    contrast
        ``(p,)`` contrast vector :math:`c` over the fixed effects.
    dof
        ``'satterthwaite'`` (default) or ``'kr'`` (Kenward-Roger, accepts only a
        :class:`REMLResult`; pass ``X`` / ``Z``).
    X, Z
        The original fixed ``(N, p)`` / random ``(N, q)`` design -- required for
        ``dof='kr'``.

    Returns
    -------
    LMEContrast
        The per-voxel contrast record :class:`LMEContrast`
        ``(effect, se, t, df, p_value)``.
    """
    if dof not in ('satterthwaite', 'kr'):
        raise ValueError(
            f'lme_t_contrast: dof={dof!r}; expected "satterthwaite" or "kr".'
        )
    if not isinstance(result, (REMLResult, LMEResult)):
        raise TypeError(
            'lme_t_contrast: needs a REMLResult (reml_fit / R1) or the R2 '
            f'LMEResult (lme_fit with a random slope); got '
            f'{type(result).__name__}, which does not surface the fixed-effect '
            'inference fields.  Nested (R3) / crossed (R4) / structured-residual '
            '(+corr) contrasts are not yet supported.'
        )
    c = jnp.asarray(contrast, dtype=result.beta_hat.dtype)
    effect = result.beta_hat @ c  # (V,)

    if dof == 'kr':
        if not isinstance(result, REMLResult):
            raise NotImplementedError(
                "lme_t_contrast: dof='kr' (Kenward-Roger) is R1-only (a "
                'REMLResult from reml_fit); use the default '
                "dof='satterthwaite' for the R2 LMEResult."
            )
        if X is None or Z is None:
            raise ValueError(
                "lme_t_contrast: dof='kr' needs the original design -- pass "
                'X= and Z= (the Kenward-Roger correction rebuilds the rotated '
                'basis the compressed REMLResult does not retain).'
            )
        p = c.shape[0]
        c_row = c[None, :]  # (1, p)
        x_rot, lambdas = _kr_rotated_basis(X, Z)

        def kr_t(
            th: Float[Array, '2'], beta: Float[Array, 'p']
        ) -> Tuple[Float[Array, ''], Float[Array, '']]:
            f_kr, df2_kr, _ = kr_cov_and_scaled_f(
                th, beta, x_rot, lambdas, c_row, p, 1
            )
            return f_kr, df2_kr

        f_kr, df = jax.vmap(kr_t)(result.theta_hat, result.beta_hat)
        t = jnp.sign(effect) * jnp.sqrt(jnp.clip(f_kr, 0.0, None))
        se = jnp.abs(effect) / jnp.sqrt(jnp.clip(f_kr, 1e-30, None))
        x = df / (df + t * t)
        p_value = betainc(0.5 * df, 0.5, x)  # two-sided
        return LMEContrast(effect=effect, se=se, t=t, df=df, p_value=p_value)

    w = jnp.einsum('vij,j->vi', result.fixed_cov, c)  # C c, (V, p)
    var = jnp.clip(jnp.einsum('vi,i->v', w, c), 1e-30, None)  # c^T C c
    se = jnp.sqrt(var)
    t = effect / se

    # Satterthwaite: g_k = w^T M_k w; df = 2 var^2 / (g^T theta_cov g).
    g = jnp.einsum('vi,vkij,vj->vk', w, result.grad_m, w)  # (V, K)
    denom = jnp.clip(
        jnp.einsum('vk,vkl,vl->v', g, result.theta_cov, g), 1e-30, None
    )
    df = 2.0 * var * var / denom
    x = df / (df + t * t)
    p_value = betainc(0.5 * df, 0.5, x)  # two-sided
    return LMEContrast(effect=effect, se=se, t=t, df=df, p_value=p_value)


class LMEFContrast(NamedTuple):
    """Per-voxel mixed-model F-contrast result.

    Attributes
    ----------
    f
        ``(V,)`` -- the Wald F-statistic
        :math:`(C \\beta)^{\\top} (C \\operatorname{Cov}(\\beta) C^{\\top})^{-1}
        (C \\beta) / L` for the :math:`L`-row contrast matrix :math:`C`.
    df1
        ``(V,)`` -- numerator degrees of freedom ``= L`` (the contrast rank;
        constant across voxels, returned per-voxel for a uniform record).
    df2
        ``(V,)`` -- Satterthwaite (Fai-Cornelius) denominator degrees of freedom.
    p_value
        ``(V,)`` -- upper-tail p-value of ``f`` on ``(df1, df2)``.
    """

    f: Float[Array, 'V']
    df1: Float[Array, 'V']
    df2: Float[Array, 'V']
    p_value: Float[Array, 'V']


def lme_f_contrast(
    result: 'Union[REMLResult, LMEResult]',
    contrast: Float[Array, 'L p'],
    *,
    dof: ContrastDof = 'satterthwaite',
    X: Optional[Float[Array, 'N p']] = None,
    Z: Optional[Float[Array, 'N q']] = None,
) -> LMEFContrast:
    """Per-voxel F-test of a multi-row mixed-model contrast :math:`C \\beta = 0`.

    For an :math:`L`-row contrast matrix :math:`C` (a single ``(p,)`` row is
    promoted to :math:`L = 1`), the Wald F-statistic is

    .. math::

        F = (C \\beta)^{\\top} (C \\operatorname{Cov}(\\beta) C^{\\top})^{-1}
            (C \\beta) / L

    with :math:`\\operatorname{Cov}(\\beta) = (X^{\\top} V^{-1} X)^{-1}` from
    ``result.fixed_cov``.  The denominator degrees of freedom use the
    **Fai-Cornelius / lmerTest** multivariate-Satterthwaite construction
    (``dof='satterthwaite'``, the default):

    1. eigendecompose
       :math:`M = C \\operatorname{Cov}(\\beta) C^{\\top}
       = P \\operatorname{diag}(d) P^{\\top}` (a small :math:`L \\times L`
       symmetric solve, cuSOLVER-free ``sym_eig_jacobi``);
    2. each eigendirection :math:`l_m = (P^{\\top} C)_m` is an independent 1-df
       contrast with variance :math:`d_m` and a per-direction Satterthwaite
       :math:`\\nu_m = 2 d_m^2 / (g_m^{\\top} \\operatorname{Cov}(\\theta)\\, g_m)`,
       :math:`g_{m,k} = w_m^{\\top} M_k w_m`,
       :math:`w_m = \\operatorname{Cov}(\\beta)\\, l_m` (the same :math:`M_k` /
       :math:`\\operatorname{Cov}(\\theta)` the fit formed, ``result.grad_m`` /
       ``result.theta_cov``);
    3. combine by the E-method:
       :math:`E = \\sum_m \\nu_m / (\\nu_m - 2)` over :math:`\\nu_m > 2`, and
       :math:`\\mathrm{df}_2 = 2 E / (E - L)`.

    For :math:`L = 1` this collapses **exactly** to :func:`lme_t_contrast`
    (:math:`F = t^2`, ``df2`` the t-Satterthwaite df, same p-value).  The
    F-statistic is differentiable; the denominator df is not (the iterative
    eigensolver feeds it under ``stop_gradient`` -- a model-selection-style
    scalar).

    ``dof='kr'`` instead applies the **Kenward-Roger** small-sample correction:
    the adjusted covariance :math:`\\Phi_A` and the moment-matched
    scaled-:math:`F`, referred to :math:`F(L, m)`.  It rebuilds the rotated
    basis, so re-pass the original design as ``X=`` / ``Z=`` (the compressed
    :class:`REMLResult` does not retain it); :math:`F = t^2` consistency with
    ``lme_t_contrast(dof='kr')`` holds for :math:`L = 1`.

    Parameters
    ----------
    result
        A :class:`REMLResult` (:func:`reml_fit`, the scalar-intercept tier)
        **or** an :class:`LMEResult` (:func:`lme_fit` with a random slope) --
        both carry the ``fixed_cov`` / ``theta_cov`` / ``grad_m`` inference
        fields.  ``dof='kr'`` accepts only a :class:`REMLResult`.
    contrast
        ``(L, p)`` contrast matrix (or a ``(p,)`` single-row contrast).
    dof
        ``'satterthwaite'`` (default) or ``'kr'`` (Kenward-Roger; pass ``X`` /
        ``Z``).
    X, Z
        The original fixed ``(N, p)`` / random ``(N, q)`` design -- required for
        ``dof='kr'``.

    Returns
    -------
    LMEFContrast
        The per-voxel F-contrast record :class:`LMEFContrast`
        ``(f, df1, df2, p_value)``.
    """
    if dof not in ('satterthwaite', 'kr'):
        raise ValueError(
            f'lme_f_contrast: dof={dof!r}; expected "satterthwaite" or "kr".'
        )
    if not isinstance(result, (REMLResult, LMEResult)):
        raise TypeError(
            'lme_f_contrast: needs a REMLResult (reml_fit / R1) or the R2 '
            f'LMEResult (lme_fit with a random slope); got '
            f'{type(result).__name__}, which does not surface the fixed-effect '
            'inference fields.  Nested (R3) / crossed (R4) / structured-residual '
            '(+corr) contrasts are not yet supported.'
        )
    dtype = result.beta_hat.dtype
    c_mat = jnp.atleast_2d(jnp.asarray(contrast, dtype=dtype))  # (L, p)
    n_rows = c_mat.shape[0]
    tiny = jnp.finfo(dtype).tiny

    if dof == 'kr':
        if not isinstance(result, REMLResult):
            raise NotImplementedError(
                "lme_f_contrast: dof='kr' (Kenward-Roger) is R1-only (a "
                'REMLResult from reml_fit); use the default '
                "dof='satterthwaite' for the R2 LMEResult."
            )
        if X is None or Z is None:
            raise ValueError(
                "lme_f_contrast: dof='kr' needs the original design -- pass "
                'X= and Z= (the Kenward-Roger correction rebuilds the rotated '
                'basis the compressed REMLResult does not retain).'
            )
        p = c_mat.shape[1]
        x_rot, lambdas = _kr_rotated_basis(X, Z)

        def kr_voxel(
            th: Float[Array, '2'], beta: Float[Array, 'p']
        ) -> Tuple[Float[Array, ''], Float[Array, '']]:
            f_kr, df2_kr, _ = kr_cov_and_scaled_f(
                th, beta, x_rot, lambdas, c_mat, p, n_rows
            )
            return f_kr, df2_kr

        f, df2 = jax.vmap(kr_voxel)(result.theta_hat, result.beta_hat)
        df1 = jnp.full_like(f, float(n_rows))
        x = df2 / (df2 + df1 * f)
        p_value = betainc(0.5 * df2, 0.5 * df1, x)
        return LMEFContrast(f=f, df1=df1, df2=df2, p_value=p_value)

    def per_voxel(
        beta: Float[Array, 'p'],
        sigma: Float[Array, 'p p'],
        theta_cov: Float[Array, 'K K'],
        grad_m: Float[Array, 'K p p'],
    ) -> Tuple[Float[Array, ''], Float[Array, '']]:
        cb = c_mat @ beta  # (L,)
        m_cov = c_mat @ sigma @ c_mat.T  # (L, L) = C Cov(beta) C^T
        m_inv, _ = small_inv_logdet(m_cov, n_rows)
        f = (cb @ m_inv @ cb) / n_rows  # the Wald F (differentiable)

        # Denominator df (Fai-Cornelius E-method) -- not differentiated through.
        d, eigvecs = sym_eig_jacobi(jax.lax.stop_gradient(m_cov), n_rows)
        l_proj = eigvecs.T @ c_mat  # (L, p): per-eigendirection contrasts
        w = l_proj @ sigma  # (L, p): w_m = Cov(beta) l_m
        g = jnp.einsum('mp,kpq,mq->mk', w, grad_m, w)  # (L, K)
        denom = jnp.einsum('mk,kl,ml->m', g, theta_cov, g)  # (L,)
        d_safe = jnp.clip(d, tiny, None)
        nu = 2.0 * d_safe * d_safe / jnp.clip(denom, tiny, None)  # (L,)
        ratio = jnp.where(nu > 2.0, nu / (nu - 2.0), 0.0)
        e_sum = jnp.sum(ratio)
        # E-method df2 = 2E/(E-L) is only valid when E > L; in the degenerate
        # small-sample / near-boundary case (no eigendirection has nu > 2, so
        # E <= L) it would give df2 <= 0 and a NaN F p-value.  Fall back to a
        # conservative df2 = 1 there (wide tails -> larger p), keeping betainc
        # well-defined.
        df2 = jnp.where(
            e_sum > n_rows,
            2.0 * e_sum / jnp.maximum(e_sum - n_rows, tiny),
            1.0,
        )
        return f, df2

    f, df2 = jax.vmap(per_voxel)(
        result.beta_hat,
        result.fixed_cov,
        result.theta_cov,
        result.grad_m,
    )
    df1 = jnp.full_like(f, float(n_rows))
    x = df2 / (df2 + df1 * f)
    p_value = betainc(0.5 * df2, 0.5 * df1, x)  # upper-tail F survival
    return LMEFContrast(f=f, df1=df1, df2=df2, p_value=p_value)


def reml_fit(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    Z: Float[Array, 'N q'],
    *,
    theta_init: Optional[Float[Array, 'V 2']] = None,
    n_iter: int = 20,
    damping: float = 1e-6,
    block: Optional[int] = None,
    low_rank: Optional[bool] = None,
) -> REMLResult:
    """Voxelwise variance-components REML fit.

    Fits the linear mixed-effects model

    .. math::

        y_v = X \\beta_v + Z b_v + \\epsilon_v, \\qquad
        b_v \\sim \\mathcal{N}(0, \\sigma_b^2 I_q), \\quad
        \\epsilon_v \\sim \\mathcal{N}(0, \\sigma_e^2 I_N)

    independently for each voxel, sharing the fixed-effect design
    ``X`` and the random-effect design ``Z`` across voxels.  Uses
    the FaST-LMM spectral trick: eigendecompose :math:`Z Z^{\\top}` once,
    rotate ``y`` and ``X`` into the diagonalising basis, then
    Newton-score on the variance components in log-space.

    Parameters
    ----------
    Y
        Response tensor, ``(V, N)`` -- ``V`` voxels, ``N`` subjects
        (or observations).
    X
        Fixed-effect design, ``(N, p)``.  Shared across voxels; carries its own
        intercept column (no intercept is added).
    Z
        Random-effect design, ``(N, q)``.  Shared across voxels.
        The random-effect covariance is :math:`\\sigma_b^2 I_q` (single
        variance component); pass each component's design matrix
        column to ``Z`` to model multiple random effects (currently
        all share :math:`\\sigma_b^2`; for separate variance components
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
    low_rank
        Use the FaST-LMM **low-rank** decomposition: a :math:`q \\times q`
        eigendecomposition of :math:`Z^{\\top} Z` (:math:`O(N q^2 + q^3)`)
        instead of the dense :math:`N \\times N` eigendecomposition of
        :math:`Z Z^{\\top}` (:math:`O(N^3)`).  Requires :math:`q \\le N` (the
        usual tall random-effect design); the :math:`N - q` null directions
        enter only through per-voxel Gram aggregates, and a rank-deficient ``Z``
        (e.g. phantom one-hot columns from gappy labels) is absorbed by that null
        space.  **Default ``None``** auto-selects: low-rank when :math:`q < N`
        (the brain-scale case), dense otherwise.  Pass ``True`` / ``False`` to
        force; the two match to the iterative tolerance.

    Returns
    -------
    REMLResult
        The per-voxel fit, with ``theta_hat`` (log-variances), ``beta_hat``
        (fixed effects), ``log_lik`` and the fixed-effect / variance-component
        inference fields.

    Notes
    -----
    Eigendecomposition of :math:`Z Z^{\\top}` uses ``safe_eigh`` (a
    cuSOLVER-robust fallback) -- the one shared, one-off solver call.  The
    per-voxel inner loop (``_varcomp``) makes no cuSOLVER call: the fixed-effect
    algebra is closed-form for :math:`p \\le 2` and ``LU``-based for
    :math:`p > 2`, and the REML derivatives are analytic.
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

    # Auto-select the low-rank decomposition when the random-effect design is
    # genuinely tall (q < N) -- the common brain-scale case, where the q x q eig
    # of Z^T Z is 1-2 orders cheaper than the dense N x N eig of ZZ^T and matches
    # it to the iterative tolerance.  Explicit low_rank=True/False overrides.
    use_low_rank = (Z.shape[-1] < n) if low_rank is None else low_rank
    if use_low_rank:
        return _reml_fit_lowrank(Y, X, Z, theta_init, n_iter, damping, block)

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

    spec = VarCompSpec(n_iter=n_iter, damping=damping)
    theta_hat, beta_hat, log_lik = fit_varcomp_diagonal(
        Y_rot,
        X_rot,
        V_basis_diag,
        theta_init,
        spec=spec,
        block=block,
    )
    p = X.shape[-1]
    zero_off = jnp.zeros(n, dtype=Y.dtype)

    def _inf(
        y: Float[Array, 'N'], th: Float[Array, '2']
    ) -> Tuple[
        Float[Array, 'p p'], Float[Array, '2 2'], Float[Array, '2 p p']
    ]:
        return varcomp_inference(th, y, X_rot, V_basis_diag, zero_off, p, spec)

    fixed_cov, theta_cov, grad_m = cast(
        Tuple[Array, Array, Array],
        blocked_vmap(_inf, (Y_rot, theta_hat), block=block),
    )
    return REMLResult(
        theta_hat=theta_hat,
        beta_hat=beta_hat,
        log_lik=log_lik - (X.shape[0] - X.shape[1]) * _HALF_LOG_2PI,
        fixed_cov=fixed_cov,
        theta_cov=theta_cov,
        grad_m=grad_m,
    )


# ---------------------------------------------------------------------------
# The structure-dispatched LME (v3 §1.1): cheapest exact solver per random spec
# ---------------------------------------------------------------------------


@register_result(
    children=('beta_hat', 'var_outer', 'var_inner', 'sigma_e_sq', 'log_lik'),
    aux=('tier',),
)
@dataclass(frozen=True)
class NestedLMEResult:
    """Nested two-level mixed-model fit output (the :func:`lme_fit` R3 path).

    Attributes
    ----------
    beta_hat
        ``(V, p)`` fixed-effect estimates.
    var_outer
        ``(V,)`` outer random-intercept variance :math:`\\sigma_1^2`
        (``(1 | g1)``).
    var_inner
        ``(V,)`` inner (nested) random-intercept variance :math:`\\sigma_2^2`
        (``(1 | g1:g2)``).
    sigma_e_sq
        ``(V,)`` residual variance.
    log_lik
        ``(V,)`` profile REML log-likelihood.
    tier
        ``'R3'`` (the telescoping-Woodbury nested solver).
    """

    beta_hat: Float[Array, 'V p']
    var_outer: Float[Array, 'V']
    var_inner: Float[Array, 'V']
    sigma_e_sq: Float[Array, 'V']
    log_lik: Float[Array, 'V']
    tier: str

    @property
    def cov_re(self) -> Float[Array, 'V 2 2']:
        """Random-effect covariance as the uniform ``(V, k, k)`` block (D2): the
        outer / inner factor variances on the diagonal of a ``(V, 2, 2)`` -- the
        two nested factors are independent, so the off-diagonals are zero."""
        v = jnp.stack([self.var_outer, self.var_inner], axis=-1)
        return v[..., None] * jnp.eye(2, dtype=v.dtype)

    @property
    def re_labels(self) -> Tuple[str, ...]:
        """Names of the ``k`` random-effect dimensions of :attr:`cov_re`."""
        return ('outer', 'inner')

    @property
    def coef(self) -> Float[Array, 'V p']:
        """Alias for :attr:`beta_hat` -- the fixed-effect coefficients, named
        ``coef`` for cross-suite parity with GLM / GAM / GP / HGP results (UX1)."""
        return self.beta_hat


@register_result(
    children=('beta_hat', 'var_group', 'var_cross', 'sigma_e_sq', 'log_lik'),
    aux=('tier',),
)
@dataclass(frozen=True)
class CrossedLMEResult:
    """Crossed two-factor mixed-model fit output (the :func:`lme_fit` R4 path).

    Attributes
    ----------
    beta_hat
        ``(V, p)`` fixed-effect estimates.
    var_group
        ``(V,)`` random-intercept variance of the ``group`` factor.
    var_cross
        ``(V,)`` random-intercept variance of the crossed ``cross`` factor.
    sigma_e_sq
        ``(V,)`` residual variance.
    log_lik
        ``(V,)`` profile REML log-likelihood.
    tier
        ``'R4'`` (the crossed Woodbury + diagonal-Schur solver).
    """

    beta_hat: Float[Array, 'V p']
    var_group: Float[Array, 'V']
    var_cross: Float[Array, 'V']
    sigma_e_sq: Float[Array, 'V']
    log_lik: Float[Array, 'V']
    tier: str

    @property
    def cov_re(self) -> Float[Array, 'V 2 2']:
        """Random-effect covariance as the uniform ``(V, k, k)`` block (D2): the
        group / cross factor variances on the diagonal of a ``(V, 2, 2)`` -- the
        two crossed factors are independent, so the off-diagonals are zero."""
        v = jnp.stack([self.var_group, self.var_cross], axis=-1)
        return v[..., None] * jnp.eye(2, dtype=v.dtype)

    @property
    def re_labels(self) -> Tuple[str, ...]:
        """Names of the ``k`` random-effect dimensions of :attr:`cov_re`."""
        return ('group', 'cross')

    @property
    def coef(self) -> Float[Array, 'V p']:
        """Alias for :attr:`beta_hat` -- the fixed-effect coefficients, named
        ``coef`` for cross-suite parity with GLM / GAM / GP / HGP results (UX1)."""
        return self.beta_hat


@register_result(
    children=(
        'beta_hat',
        'cov_re',
        'sigma_e_sq',
        'log_lik',
        'fixed_cov',
        'theta_cov',
        'grad_m',
        'blups',
    ),
    aux=('tier',),
)
@dataclass(frozen=True)
class LMEResult:
    """Block-Woodbury (R2) mixed-model fit output for one random slope.

    The output of the R2 tier -- one correlated or diagonal random slope
    (``(1 + x | g)`` or ``(x || g)``).  The :func:`lme_fit` dispatcher returns
    this for the R2 tier; the scalar-intercept R1 tier returns a
    :class:`REMLResult` (the FaST-LMM path), and the nested / crossed /
    structured-residual tiers their own result types.

    Carries the fixed-effect contrast-inference tensors (``fixed_cov`` /
    ``theta_cov`` / ``grad_m``), so :func:`lme_t_contrast` /
    :func:`lme_f_contrast` accept it with the **same** Satterthwaite /
    Kenward-Roger df machinery as R1.

    Attributes
    ----------
    beta_hat
        ``(V, p)`` fixed-effect estimates.
    cov_re
        ``(V, r, r)`` random-effect covariance ``G`` (the within-group
        covariance of the random effects).
    sigma_e_sq
        ``(V,)`` residual variance.
    log_lik
        ``(V,)`` profile REML log-likelihood.
    fixed_cov
        ``(V, p, p)`` :math:`\\operatorname{Cov}(\\hat{\\beta})
        = (X^{\\top} V^{-1} X)^{-1}` (the contrast numerator covariance).
    theta_cov
        ``(V, nt, nt)`` :math:`\\operatorname{Cov}(\\hat{\\theta})` (inverse
        average-information; for the Satterthwaite denominator df).
        ``nt = r(r+1)/2 + 1`` unstructured, or ``r + 1`` diagonal.
    grad_m
        ``(V, nt, p, p)`` the Satterthwaite gradient tensors
        :math:`M_k = X^{\\top} V^{-1} (\\partial V / \\partial \\theta_k)
        V^{-1} X`.
    tier
        ``'R2'`` (block-Woodbury, one correlated / diagonal factor).
    """

    beta_hat: Float[Array, 'V p']
    cov_re: Float[Array, 'V r r']
    sigma_e_sq: Float[Array, 'V']
    log_lik: Float[Array, 'V']
    fixed_cov: Float[Array, 'V p p']
    theta_cov: Float[Array, 'V nt nt']
    grad_m: Float[Array, 'V nt p p']
    tier: str
    blups: Optional[Float[Array, 'V q r']] = None
    """``(V, q, r)`` per-group random-effect modes, or ``None`` when the fit did
    not retain them (the default ``retain_blups=False``).  Read via
    :func:`~nitrix.stats.ranef`; used by :func:`~nitrix.stats.lme_predict` at
    ``level='conditional'``."""

    @property
    def re_labels(self) -> Tuple[str, ...]:
        """Names of the ``r`` within-factor random-effect dimensions of
        :attr:`cov_re` (D2); the off-diagonals are genuine intercept/slope
        covariances (unlike the block-diagonal multi-factor tiers)."""
        return tuple(f're{j}' for j in range(self.cov_re.shape[-1]))

    @property
    def coef(self) -> Float[Array, 'V p']:
        """Alias for :attr:`beta_hat` -- the fixed-effect coefficients, named
        ``coef`` for cross-suite parity with GLM / GAM / GP / HGP results (UX1)."""
        return self.beta_hat


def lme_fit(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    *,
    group: Int[Array, 'N'],
    z: Optional[Float[Array, 'N r']] = None,
    inner: Optional[Int[Array, 'N']] = None,
    cross: Optional[Int[Array, 'N']] = None,
    corr: Optional[Union[str, CorrSpec]] = None,
    time: Optional[Float[Array, 'N']] = None,
    structure: Structure = 'unstructured',
    n_iter: int = 20,
    damping: float = 1e-6,
    block: Optional[int] = None,
    low_rank: Optional[bool] = None,
    retain_blups: bool = False,
) -> Union[
    REMLResult, LMEResult, NestedLMEResult, CorrLMEResult, CrossedLMEResult
]:
    """Voxelwise mixed model, dispatched to the cheapest exact solver.

    A single grouping factor ``group`` with per-observation random covariates
    ``z`` (``None`` -> a scalar random intercept).  The dispatcher honours a
    **performance-preserving** invariant: it routes to the cheapest solver that
    is *exact* for the given random structure, so the shipped fast paths stay the
    realised path for the cases they already serve.

    Dispatch ladder
    ---------------

    - **R1** -- one **scalar** random effect (``z`` is ``None`` or a single
      column): the FaST-LMM spectral :func:`reml_fit` (dense, or q-rank via
      ``low_rank=``).  Returns a :class:`REMLResult` (with the fixed-effect
      inference fields).  *Bit-for-bit the shipped path.*
    - **R2** -- one **correlated / diagonal** random effect (``z`` has ``r >= 2``
      columns, e.g. ``[1, x]`` for ``(1 + x | g)``): the block-Woodbury REML
      (``_blockwoodbury``) -- per-group ``r x r`` Woodbury, no ``N x N``
      intermediate, cuSOLVER-free.  Returns an :class:`LMEResult`.
      ``structure='unstructured'`` fits a full ``r x r`` ``G``;
      ``structure='diagonal'`` constrains ``G`` to be diagonal -- the
      uncorrelated ``(x || g)`` random effect (independent intercept and slope),
      fitting only the ``r`` variances (``nt = r + 1`` parameters).

    - **R3** -- a **nested** two-level hierarchy ``(1 | g1/g2)`` (pass the inner
      factor as ``inner``; ``z`` must be ``None``): the telescoping-Woodbury
      nested solver (``_nested``) over the three variance components
      :math:`[\\sigma_1^2, \\sigma_2^2, \\sigma_e^2]` -- block-diagonal across
      ``g1``, Sherman-Morrison at each nested level, no :math:`N \\times N`
      intermediate, cuSOLVER-free.  Returns a :class:`NestedLMEResult`.
    - **R2 + corr** -- a random effect **and** a structured within-group residual
      :math:`\\operatorname{Cov}(\\epsilon) = \\sigma_e^2 R(\\rho)` (pass
      ``corr=`` and, for :func:`~nitrix.stats.car1`, ``time=``).  Whitening by
      :math:`R` reduces each group to a standard block-Woodbury, so the per-group
      Woodbury algebra is reused with :math:`\\rho` joining the REML
      :math:`\\theta`; cuSOLVER-free.  Returns a :class:`CorrLMEResult`.
      ``z=None`` is the random intercept; ``z`` a random slope
      (``structure='diagonal'`` for ``(x || g)``).
    - **R4** -- two **crossed** scalar random intercepts ``(1 | group) +
      (1 | cross)`` (pass the second factor as ``cross``; ``z`` / ``inner`` /
      ``corr`` must be unset).  :math:`V` is not block-diagonal across either
      factor, so this is the sparse mixed-model-equations regime: a Woodbury with
      the combined design, whose inner matrix has diagonal blocks (the level
      counts) and an off-diagonal incidence, so a diagonal Schur eliminates the
      larger factor and leaves a single dense
      :math:`\\min(q_{\\mathrm{group}}, q_{\\mathrm{cross}})^3` solve per Newton
      step (``_crossed``).  cuSOLVER-free; **cheap when one factor is small,
      expensive when both are large.**  Returns a :class:`CrossedLMEResult`.

    Parameters
    ----------
    Y, X
        ``(V, N)`` responses and ``(N, p)`` shared fixed-effect design.  ``X``
        carries its own intercept column (no intercept is added).
    group
        ``(N,)`` integer group labels (``0 .. M-1``).  The **outer** factor
        ``g1`` when ``inner`` is given (R3).
    z
        ``(N, r)`` per-observation random covariates; ``None`` is the scalar
        random intercept (``r = 1``, ``z = 1``).
    inner
        ``(N,)`` **nested** inner factor ``g2`` (each ``(group, inner)`` pair is
        a sublevel ``g1:g2``).  When given, routes to the R3 nested solver
        (``z`` must be ``None``); ``None`` (default) keeps the single-factor
        R1/R2 dispatch.
    cross
        ``(N,)`` second **crossed** factor.  When given, routes to the R4 crossed
        solver -- ``(1 | group) + (1 | cross)``, the two factors crossed (not
        nested).  ``z`` / ``inner`` / ``corr`` must be unset.  Tier-2; cost scales
        with the *smaller* factor's level count cubed.
    corr
        Within-group residual correlation structure (``'ar1'`` / ``'car1'`` /
        ``'cs'`` or a :class:`CorrSpec`).  When given, routes to the R2 + corr
        whitened block-Woodbury (a random effect plus a structured residual);
        ``None`` (default) is the :math:`\\sigma_e^2 I` residual.
    time
        ``(N,)`` observation times for a ``car1`` residual (see ``corr``).
    structure
        ``'unstructured'`` (full ``r x r`` ``G``, ``(1 + x | g)``) or
        ``'diagonal'`` (independent variance components, ``(x || g)`` -- the
        off-diagonal of ``G`` is held at zero).  Both run the R2 block-Woodbury.
    n_iter, damping, block
        Newton iterations, AI damping, and voxel-block chunking (all tiers).
    low_rank
        **R1-only.** Toggles the FaST-LMM q-rank decomposition on the
        single-scalar-factor (R1) path; ``None`` (default) auto-selects low-rank
        when ``q < N`` (the brain-scale case; matches the dense fit to tolerance).
        **Ignored on the R2 / R3 / R4 / +corr tiers**, which have no
        dense-vs-low-rank choice -- passing it there is a silent no-op.
    retain_blups
        Retain the per-group random-effect modes (BLUPs) on the returned result
        for later conditional prediction.  ``False`` (default) discards them.
        Currently supported only on the single-factor R1 (scalar intercept), R2
        (random slope) and R2 + corr (structured residual) tiers; requesting it
        alongside ``cross=`` or ``inner=`` raises.  Read the retained modes via
        :func:`~nitrix.stats.ranef`.

    Returns
    -------
    REMLResult or LMEResult or NestedLMEResult or CorrLMEResult or CrossedLMEResult
        The fit for the selected tier: a :class:`REMLResult` (R1), an
        :class:`LMEResult` (R2), a :class:`NestedLMEResult` (R3), a
        :class:`CorrLMEResult` (R2 + corr), or a :class:`CrossedLMEResult`
        (R4 crossed).
    """
    group = jnp.asarray(group)
    n_groups = int(jnp.max(group)) + 1
    if retain_blups and (cross is not None or inner is not None):
        raise NotImplementedError(
            'lme_fit: retain_blups is currently supported for the single-factor '
            'R1 (scalar intercept), R2 (random slope) and R2 + corr (structured '
            'residual) tiers.  Conditional BLUPs for crossed (cross=) and nested '
            '(inner=) fits are staged; population-level lme_predict works for '
            'every tier without retaining modes.'
        )
    if z is not None:
        # ER4: a 1-D random covariate (N,) is a single random slope; coerce it to
        # the (N, r) contract so r = z.shape[-1] = 1 (a bare (N,) would otherwise
        # set r = N and misroute the dispatch into a deep IndexError).
        z = jnp.asarray(z, dtype=Y.dtype)
        if z.ndim == 1:
            z = z[:, None]
        elif z.ndim != 2 or z.shape[0] != group.shape[0]:
            raise ValueError(
                f'lme_fit: z must be (N, r) with N={group.shape[0]}; got shape '
                f'{tuple(z.shape)}.'
            )
    r = 1 if z is None else z.shape[-1]

    if cross is not None:
        # R4: two crossed scalar random intercepts (1|group) + (1|cross).
        if z is not None or inner is not None or corr is not None:
            raise NotImplementedError(
                'lme_fit: crossed random effects (cross=) compose only with two '
                'scalar intercepts; z / inner / corr are not yet supported '
                'alongside it.'
            )
        from ._crossed import fit_crossed_reml

        cross = jnp.asarray(cross)
        q_g = n_groups
        q_c = int(jnp.max(cross)) + 1
        oh_g = jax.nn.one_hot(group, q_g, dtype=Y.dtype)
        oh_c = jax.nn.one_hot(cross, q_c, dtype=Y.dtype)
        # Order so the dense Schur solve is on the smaller factor (oh1 larger).
        swap = q_c > q_g
        oh1, oh2 = (oh_c, oh_g) if swap else (oh_g, oh_c)
        var_y = jnp.var(Y, axis=-1, keepdims=True)  # (V, 1)
        w = jnp.asarray([0.25, 0.25, 0.5], dtype=Y.dtype)
        theta_init = jnp.log(jnp.maximum(var_y * w, 1e-6))  # (V, 3)
        spec = VarCompSpec(n_iter=n_iter, damping=damping)
        theta_hat, beta_hat, log_lik = fit_crossed_reml(
            Y, X, oh1, oh2, theta_init, spec=spec, block=block
        )
        var1 = jnp.exp(theta_hat[:, 0])  # variance of oh1
        var2 = jnp.exp(theta_hat[:, 1])  # variance of oh2
        var_group = var2 if swap else var1
        var_cross = var1 if swap else var2
        return CrossedLMEResult(
            beta_hat=beta_hat,
            var_group=var_group,
            var_cross=var_cross,
            sigma_e_sq=jnp.exp(theta_hat[:, 2]),
            log_lik=log_lik - (X.shape[0] - X.shape[1]) * _HALF_LOG_2PI,
            tier='R4',
        )

    if corr is not None:
        # R2 + corr: a random effect *and* a structured within-group residual.
        if inner is not None:
            raise NotImplementedError(
                'lme_fit: a structured residual (corr=) on a nested design '
                '(inner=) is a further composition; not yet supported.'
            )
        from ._corrfit import fit_corr_lme

        z_design = (
            jnp.ones((group.shape[0], 1), dtype=Y.dtype)
            if z is None
            else jnp.asarray(z, dtype=Y.dtype)
        )
        result = fit_corr_lme(
            Y,
            X,
            z_design,
            group,
            resolve_corr(corr),
            time=None if time is None else jnp.asarray(time),
            diagonal=structure == 'diagonal',
            n_iter=n_iter,
            damping=damping,
            block=block,
            retain_blups=retain_blups,
        )
        if retain_blups and z is None and result.blups is not None:
            # Scalar intercept -> (V, q) (matches R1); an explicit slope keeps
            # the (V, q, r) modes for the z-weighted conditional predictor.
            result = replace(result, blups=result.blups[..., 0])
        return result

    if inner is not None:
        # R3: nested two-level hierarchy (1 | g1/g2).
        if z is not None:
            raise ValueError(
                'lme_fit: nested fit (inner=...) takes scalar intercepts at '
                'both levels; pass z=None (random slopes nested under a factor '
                'are a Tier-2 follow-up).'
            )
        from ._nested import fit_nested_reml

        inner_arr = jnp.asarray(inner)
        var_y = jnp.var(Y, axis=-1, keepdims=True)  # (V, 1)
        w = jnp.asarray([0.25, 0.25, 0.5], dtype=Y.dtype)
        theta_init = jnp.log(jnp.maximum(var_y * w, 1e-6))  # (V, 3)
        spec = VarCompSpec(n_iter=n_iter, damping=damping)
        theta_hat, beta_hat, log_lik = fit_nested_reml(
            Y, X, group, inner_arr, theta_init, spec=spec, block=block
        )
        return NestedLMEResult(
            beta_hat=beta_hat,
            var_outer=jnp.exp(theta_hat[:, 0]),
            var_inner=jnp.exp(theta_hat[:, 1]),
            sigma_e_sq=jnp.exp(theta_hat[:, 2]),
            log_lik=log_lik - (X.shape[0] - X.shape[1]) * _HALF_LOG_2PI,
            tier='R3',
        )

    if r == 1:
        # R1: one scalar random effect -> the FaST-LMM spectral fast path.
        onehot = jax.nn.one_hot(group, n_groups, dtype=Y.dtype)  # (N, M)
        z_design = onehot if z is None else onehot * jnp.asarray(z)
        result = reml_fit(
            Y,
            X,
            z_design,
            n_iter=n_iter,
            damping=damping,
            block=block,
            low_rank=low_rank,
        )
        if retain_blups:
            z_eff = (
                jnp.ones((group.shape[0], 1), dtype=Y.dtype)
                if z is None
                else jnp.asarray(z, dtype=Y.dtype)
            )
            b = _blups_standard(
                Y,
                X,
                result.beta_hat,
                z_eff,
                group,
                result.cov_re,
                result.sigma_e_sq,
                n_groups,
            )  # (V, q, 1)
            # Scalar intercept -> (V, q); an explicit 1-col slope keeps (V, q, 1).
            result = replace(result, blups=b[..., 0] if z is None else b)
        return result

    # R2: one correlated / diagonal random effect -> block-Woodbury REML.
    from ._blockwoodbury import fit_blockwoodbury_reml
    from ._recov import _param_layout, cov_re_from_chol

    diagonal = structure == 'diagonal'
    z_arr = jnp.asarray(z, dtype=Y.dtype)
    layout = _param_layout(r, diagonal)
    diag_mask = jnp.asarray([i == j for (i, j) in layout], dtype=bool)  # (m,)
    var_y = jnp.var(Y, axis=-1)  # (V,)
    chol_diag = 0.5 * jnp.log(jnp.maximum(0.1 * var_y, 1e-6))  # (V,)
    chol = jnp.where(diag_mask[None, :], chol_diag[:, None], 0.0)  # (V, m)
    log_se2 = jnp.log(jnp.maximum(0.5 * var_y, 1e-6))[:, None]  # (V, 1)
    theta_init = jnp.concatenate([chol, log_se2], axis=1)  # (V, nt)

    spec = VarCompSpec(n_iter=n_iter, damping=damping)
    theta_hat, beta_hat, log_lik, fixed_cov, theta_cov, grad_m = (
        fit_blockwoodbury_reml(
            Y,
            X,
            z_arr,
            group,
            n_groups,
            theta_init,
            spec=spec,
            block=block,
            diagonal=diagonal,
            inference=True,
        )
    )
    cov_re = jax.vmap(lambda th: cov_re_from_chol(th[:-1], r, diagonal))(
        theta_hat
    )  # (V, r, r)
    sigma_e_sq = jnp.exp(theta_hat[:, -1])
    blups = (
        _blups_standard(
            Y, X, beta_hat, z_arr, group, cov_re, sigma_e_sq, n_groups
        )
        if retain_blups
        else None
    )
    return LMEResult(
        beta_hat=beta_hat,
        cov_re=cov_re,
        sigma_e_sq=sigma_e_sq,
        log_lik=log_lik - (X.shape[0] - X.shape[1]) * _HALF_LOG_2PI,
        fixed_cov=fixed_cov,
        theta_cov=theta_cov,
        grad_m=grad_m,
        tier='R2',
        blups=blups,
    )
