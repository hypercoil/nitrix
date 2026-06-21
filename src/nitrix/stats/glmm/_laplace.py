# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Laplace-approximate marginal GLMM solvers -- the ``method='laplace'`` family.

The scalar random-intercept Laplace fit and its r-dimensional random-slope lift
(Fisher-scoring mode + determinant correction).  Split from the ``glmm`` monolith
(audit O1).
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
    """Per-group conditional modes ``b_hat`` (+ the curvature ``sw`` and ``eta``).

    Given the fixed effects and ``sigma_b^2`` (in ``theta = [beta, log
    sigma_b^2]``), the mode of ``sum_i log p(y_i | beta, b_g) - b_g^2/(2
    sigma_b^2)`` is found by per-group Newton (Fisher scoring), vectorised over
    groups via ``segment_sum`` -- ``O(N)``, no random-effect-wide system.
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

    ``-sum_g [ ell_g(b_hat) - b_hat^2/(2 sigma_b^2) - 0.5 log sigma_b^2
    - 0.5 log(sum_i w_i + 1/sigma_b^2) ]`` -- the Laplace approximation to the
    random-effect integral (matches Gauss-Hermite to the approximation order, and
    corrects the PQL attenuation for binary / low-count responses).
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
    """Single-element Laplace GLMM fit.  Returns ``(beta, blups, re_var,
    deviance)``."""

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
    """Laplace GLMM over ``V`` elements (scalar random intercept)."""
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
    """Per-group conditional modes ``b_hat`` (r-vector) + curvature ``H`` + eta.

    The ``r``-dimensional lift of :func:`_laplace_conditional_modes`: given the
    fixed effects and ``G`` (in ``theta = [beta, chol(G)]``), the mode of
    ``sum_i log p(y_i | beta, b_g) - 0.5 b_g^T G^{-1} b_g`` is found by per-group
    ``r x r`` Newton (Fisher scoring) -- gradient ``sum_i s_i z_i - G^{-1} b_g``,
    curvature ``H_g = sum_i w_i z_i z_i^T + G^{-1}`` -- vectorised over groups via
    ``segment_sum`` of the per-observation outer products.  Returns the modes, the
    per-group curvature ``H_g`` at the mode (for the determinant correction), and
    the linear predictor.

    ``H_g`` uses the **expected** (Fisher) information ``w_i = (dmu/deta)^2 / V``,
    not the observed Hessian ``-d^2 ell/db^2`` -- the deliberate ``glmer``-
    consistent "Fisher-scoring Laplace".  For the canonical links (logit / log)
    the two coincide; for a non-canonical link (probit / cloglog slope) they
    differ at higher order, so this is an approximation choice, not the exact
    observed-information Laplace.
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
    """Laplace-approximate marginal NLL for a random slope.

    ``-sum_g [ ell_g(b_hat) - 0.5 b_hat^T G^{-1} b_hat - 0.5 logdet G
    - 0.5 logdet H_g ]`` -- the ``r``-dimensional generalisation of the scalar
    determinant correction (reduces to it at ``r = 1``: ``logdet G = log
    sigma_b^2``, ``logdet H_g = log(sum_i w_i + 1/sigma_b^2)``).
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
    """Single-element Laplace random-slope GLMM fit.  Returns ``(beta, blups,
    G, deviance)``."""
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
    """Laplace random-slope GLMM over ``V`` elements."""
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
