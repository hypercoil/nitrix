# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Laplace-approximate marginal GLMM solvers -- the ``method='laplace'`` family.

The scalar random-intercept Laplace fit and its ``r``-dimensional random-slope
lift (Fisher-scoring mode plus determinant correction).
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float, Int

from ...linalg._smalllinalg import small_inv_logdet
from .._batching import blocked_vmap
from .._family import Family
from .._optimise import damped_newton
from ..lme._recov import _param_layout, cov_re_from_chol
from ..lme._varcomp import VarCompSpec
from ._base import _EPS, GLMMResult

# ---------------------------------------------------------------------------
# Laplace-approximate GLMM (scalar random intercept) -- the §11 follow-up to PQL
# ---------------------------------------------------------------------------


def _laplace_conditional_modes(
    theta: Float[Array, 'p1'],
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    group: Int[Array, 'N'],
    n_groups: int,
    family: Family,
    p: int,
    n_mode: int,
) -> Tuple[Float[Array, 'q'], Float[Array, 'q'], Float[Array, 'N']]:
    """Per-group conditional modes :math:`\\hat b` with curvature and predictor.

    Given the fixed effects and the random-intercept variance
    :math:`\\sigma_b^2` (packed as ``theta = [beta, log_sigma_b_squared]``),
    this locates the per-group mode of
    :math:`\\sum_i \\log p(y_i \\mid \\beta, b_g) - b_g^2 / (2\\sigma_b^2)`
    by per-group Newton iteration (Fisher scoring), vectorised over groups via
    ``segment_sum``.  This costs :math:`O(N)` and never forms a random-effect-wide
    linear system.

    Parameters
    ----------
    theta
        ``(p + 1,)`` parameter vector ``[beta, log_sigma_b_squared]``: the first
        ``p`` entries are the fixed-effect coefficients :math:`\\beta`, the last is
        :math:`\\log \\sigma_b^2`.
    y
        ``(N,)`` response vector.
    X
        ``(N, p)`` fixed-effect design matrix.
    group
        ``(N,)`` integer group (grouping-factor level) index for each observation,
        with values in ``[0, n_groups)``.
    n_groups
        Number of grouping-factor levels ``q``.
    family
        Exponential-family specification providing the link, inverse link,
        variance function and derivative used in Fisher scoring.
    p
        Number of fixed-effect columns (length of :math:`\\beta`).
    n_mode
        Number of inner Newton (Fisher-scoring) steps used to locate the modes.

    Returns
    -------
    b : Float[Array, 'q']
        ``(q,)`` per-group conditional modes :math:`\\hat b`.
    sw : Float[Array, 'q']
        ``(q,)`` per-group Fisher curvature :math:`\\sum_i w_i` at the mode
        (the observation-weight sum :math:`-\\ell''` per group).
    eta : Float[Array, 'N']
        ``(N,)`` linear predictor :math:`\\eta` evaluated at the modes.
    """
    beta = theta[:p]
    sb2 = jnp.exp(theta[p])
    eta_fix = X @ beta

    def mode_step(
        b: Float[Array, 'q'], _: Array
    ) -> Tuple[Float[Array, 'q'], None]:
        eta = eta_fix + b[group]
        mu = family.linkinv(eta)
        var = jnp.clip(family.variance(mu), _EPS, None)
        dmu = family.mu_eta(eta)
        score = (y - mu) * dmu / var  # d log p / d eta
        w = dmu * dmu / var  # Fisher weight
        sg = jax.ops.segment_sum(score, group, num_segments=n_groups) - b / sb2
        sh = jax.ops.segment_sum(w, group, num_segments=n_groups) + 1.0 / sb2
        return b + sg / sh, None

    b, _ = lax.scan(
        mode_step,
        jnp.zeros((n_groups,), dtype=X.dtype),
        xs=None,
        length=n_mode,
    )
    eta = eta_fix + b[group]
    mu = family.linkinv(eta)
    var = jnp.clip(family.variance(mu), _EPS, None)
    sw = jax.ops.segment_sum(
        family.mu_eta(eta) ** 2 / var, group, num_segments=n_groups
    )  # curvature -ell'' at the mode (Fisher)
    return b, sw, eta


def _laplace_nll(
    theta: Float[Array, 'p1'],
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    group: Int[Array, 'N'],
    n_groups: int,
    family: Family,
    p: int,
    n_mode: int,
) -> Float[Array, '']:
    """Laplace-approximate marginal negative log-likelihood at ``theta``.

    Evaluates
    :math:`-\\sum_g [\\, \\ell_g(\\hat b) - \\hat b^2 / (2\\sigma_b^2)
    - 0.5\\log\\sigma_b^2 - 0.5\\log(\\sum_i w_i + 1/\\sigma_b^2) \\,]`,
    the Laplace approximation to the random-effect integral.  It matches
    Gauss-Hermite quadrature to the order of the approximation and corrects the
    penalised quasi-likelihood attenuation for binary and low-count responses.

    Parameters
    ----------
    theta
        ``(p + 1,)`` parameter vector ``[beta, log_sigma_b_squared]`` (see
        :func:`_laplace_conditional_modes`).
    y
        ``(N,)`` response vector.
    X
        ``(N, p)`` fixed-effect design matrix.
    group
        ``(N,)`` integer group index for each observation, in ``[0, n_groups)``.
    n_groups
        Number of grouping-factor levels ``q``.
    family
        Exponential-family specification.
    p
        Number of fixed-effect columns.
    n_mode
        Number of inner Newton steps used to locate the conditional modes.

    Returns
    -------
    Float[Array, '']
        Scalar Laplace-approximate marginal negative log-likelihood.
    """
    sb2 = jnp.exp(theta[p])
    b, sw, eta = _laplace_conditional_modes(
        theta, y, X, group, n_groups, family, p, n_mode
    )
    mu = family.linkinv(eta)
    ll = jax.ops.segment_sum(
        family.loglik(y, mu, jnp.asarray(1.0, dtype=X.dtype)),
        group,
        num_segments=n_groups,
    )
    log_lg = (
        ll
        - b * b / (2.0 * sb2)
        - 0.5 * jnp.log(sb2)
        - 0.5 * jnp.log(sw + 1.0 / sb2)
    )
    return -jnp.sum(log_lg)


def _glmm_laplace_one(
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    group: Int[Array, 'N'],
    n_groups: int,
    family: Family,
    p: int,
    n_mode: int,
    spec: VarCompSpec,
) -> Tuple[
    Float[Array, 'p'],
    Float[Array, 'q'],
    Float[Array, ''],
    Float[Array, ''],
]:
    """Single-element Laplace GLMM fit with a scalar random intercept.

    Optimises the Laplace-approximate marginal negative log-likelihood
    (:func:`_laplace_nll`) over ``theta = [beta, log_sigma_b_squared]`` by damped
    Newton iteration, then recovers the fixed effects, the conditional-mode
    random intercepts and the deviance for a single response vector.

    Parameters
    ----------
    y
        ``(N,)`` response vector for one element.
    X
        ``(N, p)`` fixed-effect design matrix.
    group
        ``(N,)`` integer group index for each observation, in ``[0, n_groups)``.
    n_groups
        Number of grouping-factor levels ``q``.
    family
        Exponential-family specification.
    p
        Number of fixed-effect columns.
    n_mode
        Number of inner Newton steps used to locate the conditional modes.
    spec
        Variance-component specification supplying the outer damped-Newton
        keyword arguments.

    Returns
    -------
    beta : Float[Array, 'p']
        ``(p,)`` fixed-effect estimates :math:`\\hat\\beta`.
    blups : Float[Array, 'q']
        ``(q,)`` per-group random-intercept modes (the BLUPs).
    re_var : Float[Array, '']
        Scalar random-intercept variance :math:`\\sigma_b^2`.
    deviance : Float[Array, '']
        Scalar deviance :math:`-2\\log L_{\\mathrm{Laplace}}` (the ``glmer``
        deviance) at the optimum.
    """

    def nll(theta: Float[Array, 'p1']) -> Float[Array, '']:
        return _laplace_nll(theta, y, X, group, n_groups, family, p, n_mode)

    theta0 = jnp.concatenate(
        [jnp.zeros((p,), dtype=X.dtype), jnp.asarray([0.0], dtype=X.dtype)]
    )  # beta = 0, log sigma_b^2 = 0
    theta = damped_newton(nll, theta0, **spec.newton_kwargs)
    beta = theta[:p]
    sb2 = jnp.exp(theta[p])
    b, _, _ = _laplace_conditional_modes(
        theta, y, X, group, n_groups, family, p, n_mode
    )
    deviance = 2.0 * nll(theta)  # -2 log L_Laplace (the glmer deviance)
    return beta, b, sb2, deviance


def _glmm_laplace(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    group: Int[Array, 'N'],
    n_groups: int,
    family: Family,
    n_outer: int,
    n_mode: int,
    damping: float,
    block: Optional[int],
) -> GLMMResult:
    """Laplace-approximate GLMM over ``V`` elements with a scalar random intercept.

    Fits an independent Laplace-approximate random-intercept GLMM to each of the
    ``V`` response rows in ``Y`` by mapping :func:`_glmm_laplace_one` over the
    elements (optionally in blocks of ``block`` to bound peak memory) and packs
    the per-element estimates into a single :class:`GLMMResult`.

    Parameters
    ----------
    Y
        ``(V, N)`` responses -- one length-``N`` response vector per element.
    X
        ``(N, p)`` fixed-effect design matrix, shared across elements.
    group
        ``(N,)`` integer group index for each observation, in ``[0, n_groups)``.
    n_groups
        Number of grouping-factor levels ``q``.
    family
        Exponential-family specification.
    n_outer
        Number of outer damped-Newton iterations over ``theta``.
    n_mode
        Number of inner Newton steps used to locate the conditional modes.
    damping
        Levenberg-style damping factor for the outer Newton optimiser.
    block
        Optional element-block size for :func:`blocked_vmap`; ``None`` maps over
        all ``V`` elements at once.

    Returns
    -------
    GLMMResult
        Per-element fit with ``tier='laplace'``.  ``re_var`` is reshaped to
        ``(V, 1, 1)`` for the uniform random-effect covariance layout,
        ``dispersion`` and ``edf_total`` are placeholders (ones, and the
        fixed-effect count ``p``, respectively).
    """
    p = X.shape[-1]
    spec = VarCompSpec(n_iter=n_outer, damping=damping)

    def per_voxel(
        y: Float[Array, 'N'],
    ) -> Tuple[Array, Array, Array, Array]:
        return _glmm_laplace_one(
            y, X, group, n_groups, family, p, n_mode, spec
        )

    beta, blups, re_var, deviance = blocked_vmap(per_voxel, (Y,), block=block)
    return GLMMResult(
        beta_hat=beta,
        blups=blups,
        re_var=re_var[:, None, None],  # (V, 1, 1) -- uniform G shape (D4)
        dispersion=jnp.ones_like(re_var),
        deviance=deviance,
        edf_total=jnp.full_like(re_var, float(p)),
        family=family,
        n_obs=int(X.shape[0]),
        n_groups=n_groups,
        tier='laplace',
    )


# ---------------------------------------------------------------------------
# Laplace-approximate random *slope* -- the r-dimensional lift of the scalar fit
# ---------------------------------------------------------------------------


def _laplace_slope_modes(
    theta: Float[Array, 'pm'],
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    z: Float[Array, 'N r'],
    group: Int[Array, 'N'],
    n_groups: int,
    family: Family,
    p: int,
    r: int,
    n_mode: int,
    diagonal: bool,
) -> Tuple[Float[Array, 'q r'], Float[Array, 'q r r'], Float[Array, 'N']]:
    """Per-group conditional modes (``r``-vectors) with curvature and predictor.

    The ``r``-dimensional lift of :func:`_laplace_conditional_modes`.  Given the
    fixed effects and the random-effect covariance :math:`G` (packed as
    ``theta = [beta, chol_G]`` with the Cholesky factor of :math:`G`), it locates
    the per-group mode of
    :math:`\\sum_i \\log p(y_i \\mid \\beta, b_g) - 0.5\\, b_g^{\\top} G^{-1} b_g`
    by per-group :math:`r \\times r` Newton iteration (Fisher scoring) --
    gradient :math:`\\sum_i s_i z_i - G^{-1} b_g` and curvature
    :math:`H_g = \\sum_i w_i z_i z_i^{\\top} + G^{-1}` -- vectorised over groups
    via ``segment_sum`` of the per-observation outer products.

    The curvature :math:`H_g` uses the **expected** (Fisher) information
    :math:`w_i = (d\\mu/d\\eta)^2 / V`, not the observed Hessian
    :math:`-d^2\\ell/db^2` -- a deliberate, ``glmer``-consistent
    "Fisher-scoring Laplace".  For canonical links (logit / log) the two
    coincide; for a non-canonical link (probit / cloglog slope) they differ at
    higher order, so this is an approximation choice, not the exact
    observed-information Laplace.

    Parameters
    ----------
    theta
        ``(p + m,)`` parameter vector ``[beta, chol_G]``: the first ``p`` entries
        are the fixed-effect coefficients, the remaining ``m`` entries
        parametrise the Cholesky factor of :math:`G` (``m = r`` for a diagonal
        structure, ``r(r+1)/2`` for an unstructured one).
    y
        ``(N,)`` response vector.
    X
        ``(N, p)`` fixed-effect design matrix.
    z
        ``(N, r)`` random-effect design matrix (the slope covariates).
    group
        ``(N,)`` integer group index for each observation, in ``[0, n_groups)``.
    n_groups
        Number of grouping-factor levels ``q``.
    family
        Exponential-family specification.
    p
        Number of fixed-effect columns.
    r
        Number of random-effect columns (random-slope dimension).
    n_mode
        Number of inner Newton (Fisher-scoring) steps used to locate the modes.
    diagonal
        Whether :math:`G` is parametrised as diagonal (``True``) or unstructured
        (``False``), controlling how ``theta[p:]`` maps to the covariance.

    Returns
    -------
    b : Float[Array, 'q r']
        ``(q, r)`` per-group conditional-mode random-effect vectors.
    h_mat : Float[Array, 'q r r']
        ``(q, r, r)`` per-group Fisher curvature :math:`H_g` at the mode, used
        for the Laplace determinant correction.
    eta : Float[Array, 'N']
        ``(N,)`` linear predictor :math:`\\eta` evaluated at the modes.
    """

    beta = theta[:p]
    g_inv, _ = small_inv_logdet(cov_re_from_chol(theta[p:], r, diagonal), r)
    eta_fix = X @ beta

    def _curvature(eta: Array) -> Tuple[Array, Array, Array]:
        mu = family.linkinv(eta)
        var = jnp.clip(family.variance(mu), _EPS, None)
        dmu = family.mu_eta(eta)
        w = dmu * dmu / var  # (N,) Fisher weight
        wzz = jax.ops.segment_sum(
            w[:, None, None] * (z[:, :, None] * z[:, None, :]),
            group,
            num_segments=n_groups,
        )  # (q, r, r) = sum_i w_i z_i z_i^T
        return mu, dmu / var, wzz + g_inv[None]

    def mode_step(
        b: Float[Array, 'q r'], _: Array
    ) -> Tuple[Float[Array, 'q r'], None]:
        eta = family.clip_eta(eta_fix + jnp.einsum('nr,nr->n', z, b[group]))
        mu, dlog, h_mat = _curvature(eta)
        score = (y - mu) * dlog  # (N,) d log p / d eta
        grad = jax.ops.segment_sum(
            score[:, None] * z, group, num_segments=n_groups
        ) - jnp.einsum('rs,qs->qr', g_inv, b)  # (q, r)
        h_inv = jax.vmap(lambda a: small_inv_logdet(a, r)[0])(h_mat)
        return b + jnp.einsum('qrs,qs->qr', h_inv, grad), None

    b, _ = lax.scan(
        mode_step,
        jnp.zeros((n_groups, r), dtype=X.dtype),
        xs=None,
        length=n_mode,
    )
    eta = family.clip_eta(eta_fix + jnp.einsum('nr,nr->n', z, b[group]))
    _, _, h_mat = _curvature(eta)
    return b, h_mat, eta


def _laplace_slope_nll(
    theta: Float[Array, 'pm'],
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    z: Float[Array, 'N r'],
    group: Int[Array, 'N'],
    n_groups: int,
    family: Family,
    p: int,
    r: int,
    n_mode: int,
    diagonal: bool,
) -> Float[Array, '']:
    """Laplace-approximate marginal negative log-likelihood for a random slope.

    Evaluates
    :math:`-\\sum_g [\\, \\ell_g(\\hat b) - 0.5\\, \\hat b^{\\top} G^{-1} \\hat b
    - 0.5\\log\\det G - 0.5\\log\\det H_g \\,]`, the ``r``-dimensional
    generalisation of the scalar determinant correction.  It reduces to the
    scalar case at :math:`r = 1`, where :math:`\\log\\det G = \\log\\sigma_b^2`
    and :math:`\\log\\det H_g = \\log(\\sum_i w_i + 1/\\sigma_b^2)`.

    Parameters
    ----------
    theta
        ``(p + m,)`` parameter vector ``[beta, chol_G]`` (see
        :func:`_laplace_slope_modes`).
    y
        ``(N,)`` response vector.
    X
        ``(N, p)`` fixed-effect design matrix.
    z
        ``(N, r)`` random-effect design matrix (the slope covariates).
    group
        ``(N,)`` integer group index for each observation, in ``[0, n_groups)``.
    n_groups
        Number of grouping-factor levels ``q``.
    family
        Exponential-family specification.
    p
        Number of fixed-effect columns.
    r
        Number of random-effect columns (random-slope dimension).
    n_mode
        Number of inner Newton steps used to locate the conditional modes.
    diagonal
        Whether :math:`G` is parametrised as diagonal (``True``) or unstructured
        (``False``).

    Returns
    -------
    Float[Array, '']
        Scalar Laplace-approximate marginal negative log-likelihood.
    """

    g_inv, logdet_g = small_inv_logdet(
        cov_re_from_chol(theta[p:], r, diagonal), r
    )
    b, h_mat, eta = _laplace_slope_modes(
        theta, y, X, z, group, n_groups, family, p, r, n_mode, diagonal
    )
    mu = family.linkinv(eta)
    ll = jax.ops.segment_sum(
        family.loglik(y, mu, jnp.asarray(1.0, dtype=X.dtype)),
        group,
        num_segments=n_groups,
    )  # (q,)
    logdet_h = jax.vmap(lambda a: small_inv_logdet(a, r)[1])(h_mat)  # (q,)
    quad = 0.5 * jnp.einsum('qr,rs,qs->q', b, g_inv, b)  # 0.5 b^T G^{-1} b
    log_lg = ll - quad - 0.5 * logdet_g - 0.5 * logdet_h
    return -jnp.sum(log_lg)


def _glmm_laplace_slope_one(
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    z: Float[Array, 'N r'],
    group: Int[Array, 'N'],
    n_groups: int,
    family: Family,
    p: int,
    r: int,
    n_mode: int,
    spec: VarCompSpec,
    diagonal: bool,
) -> Tuple[
    Float[Array, 'p'],
    Float[Array, 'q r'],
    Float[Array, 'r r'],
    Float[Array, ''],
]:
    """Single-element Laplace random-slope GLMM fit.

    Optimises the Laplace-approximate marginal negative log-likelihood
    (:func:`_laplace_slope_nll`) over ``theta = [beta, chol_G]`` by damped Newton
    iteration from an IRLS warm start for :math:`\\beta` and a small initial
    :math:`G`, then recovers the fixed effects, the conditional-mode random
    slopes, the fitted covariance and the deviance for a single response vector.

    Parameters
    ----------
    y
        ``(N,)`` response vector for one element.
    X
        ``(N, p)`` fixed-effect design matrix.
    z
        ``(N, r)`` random-effect design matrix (the slope covariates).
    group
        ``(N,)`` integer group index for each observation, in ``[0, n_groups)``.
    n_groups
        Number of grouping-factor levels ``q``.
    family
        Exponential-family specification.
    p
        Number of fixed-effect columns.
    r
        Number of random-effect columns (random-slope dimension).
    n_mode
        Number of inner Newton steps used to locate the conditional modes.
    spec
        Variance-component specification supplying the outer damped-Newton
        keyword arguments and the IRLS warm-start ridge.
    diagonal
        Whether :math:`G` is parametrised as diagonal (``True``) or unstructured
        (``False``).

    Returns
    -------
    beta : Float[Array, 'p']
        ``(p,)`` fixed-effect estimates :math:`\\hat\\beta`.
    blups : Float[Array, 'q r']
        ``(q, r)`` per-group random-slope modes (the BLUPs).
    g_cov : Float[Array, 'r r']
        ``(r, r)`` fitted random-effect covariance :math:`G`.
    deviance : Float[Array, '']
        Scalar deviance :math:`-2\\log L_{\\mathrm{Laplace}}` at the optimum.
    """
    from .._irls import irls_warm_start

    def nll(theta: Float[Array, 'pm']) -> Float[Array, '']:
        return _laplace_slope_nll(
            theta, y, X, z, group, n_groups, family, p, r, n_mode, diagonal
        )

    # Link-scale init: warm-start beta, a small G (cf. the structured PQL slope).
    var_e = jnp.var(family.link(family.init_mu(y)))
    layout = _param_layout(r, diagonal)
    diag = jnp.asarray([i == j for (i, j) in layout])
    chol0 = jnp.where(diag, 0.5 * jnp.log(jnp.maximum(0.1 * var_e, 1e-6)), 0.0)
    beta0 = irls_warm_start(
        y,
        X,
        family,
        penalty=jnp.zeros((p, p), dtype=y.dtype),
        ridge=spec.ridge,
    )
    theta0 = jnp.concatenate([beta0, chol0])
    theta = damped_newton(nll, theta0, **spec.newton_kwargs)
    beta = theta[:p]
    g_cov = cov_re_from_chol(theta[p:], r, diagonal)
    b, _, _ = _laplace_slope_modes(
        theta, y, X, z, group, n_groups, family, p, r, n_mode, diagonal
    )
    deviance = 2.0 * nll(theta)
    return beta, b, g_cov, deviance


def _glmm_laplace_slope(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    group: Int[Array, 'N'],
    n_groups: int,
    z: Float[Array, 'N r'],
    family: Family,
    n_outer: int,
    n_mode: int,
    damping: float,
    diagonal: bool,
    block: Optional[int],
) -> GLMMResult:
    """Laplace-approximate random-slope GLMM over ``V`` elements.

    Fits an independent Laplace-approximate random-slope GLMM to each of the
    ``V`` response rows in ``Y`` by mapping :func:`_glmm_laplace_slope_one` over
    the elements (optionally in blocks of ``block``) and packs the per-element
    estimates into a single :class:`GLMMResult`.

    Parameters
    ----------
    Y
        ``(V, N)`` responses -- one length-``N`` response vector per element.
    X
        ``(N, p)`` fixed-effect design matrix, shared across elements.
    group
        ``(N,)`` integer group index for each observation, in ``[0, n_groups)``.
    n_groups
        Number of grouping-factor levels ``q``.
    z
        ``(N, r)`` random-effect design matrix (the slope covariates).
    family
        Exponential-family specification.
    n_outer
        Number of outer damped-Newton iterations over ``theta``.
    n_mode
        Number of inner Newton steps used to locate the conditional modes.
    damping
        Levenberg-style damping factor for the outer Newton optimiser.
    diagonal
        Whether the random-effect covariance :math:`G` is parametrised as
        diagonal (``True``) or unstructured (``False``).
    block
        Optional element-block size for :func:`blocked_vmap`; ``None`` maps over
        all ``V`` elements at once.

    Returns
    -------
    GLMMResult
        Per-element fit with ``tier='laplace'``.  ``re_var`` carries the
        ``(V, r, r)`` random-effect covariances (diagonal-valued when
        ``diagonal`` is ``True``); ``dispersion`` and ``edf_total`` are
        placeholders (ones, and the fixed-effect count ``p``, respectively).
    """
    p = X.shape[-1]
    r = z.shape[-1]
    spec = VarCompSpec(n_iter=n_outer, damping=damping)

    def per_voxel(
        y: Float[Array, 'N'],
    ) -> Tuple[Array, Array, Array, Array]:
        return _glmm_laplace_slope_one(
            y, X, z, group, n_groups, family, p, r, n_mode, spec, diagonal
        )

    beta, blups, g_cov, deviance = blocked_vmap(per_voxel, (Y,), block=block)
    # D4: uniform (V, r, r) G -- diagonal-valued in the diagonal case.
    re_var = g_cov
    nv = beta.shape[0]
    return GLMMResult(
        beta_hat=beta,
        blups=blups,
        re_var=re_var,
        dispersion=jnp.ones((nv,), dtype=beta.dtype),
        deviance=deviance,
        edf_total=jnp.full((nv,), float(p), dtype=beta.dtype),
        family=family,
        n_obs=int(X.shape[0]),
        n_groups=n_groups,
        tier='laplace',
    )
