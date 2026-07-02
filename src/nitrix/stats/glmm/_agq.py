# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Adaptive Gauss-Hermite quadrature GLMM solver -- the ``method='agq'`` family.

The accuracy tier above Laplace: it evaluates the random-slope marginal
integral on an ``n_quad ** r`` tensor grid of Gauss-Hermite nodes centred and
scaled at each group's conditional mode and curvature.  The mode and curvature
themselves are supplied by the sibling Laplace solver.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float, Int

from ...linalg._smalllinalg import small_inv_logdet, spd_chol
from .._batching import blocked_vmap
from .._family import Family
from .._optimise import damped_newton
from ..lme._recov import _param_layout, cov_re_from_chol
from ..lme._varcomp import VarCompSpec
from ._base import GLMMResult
from ._laplace import _laplace_slope_modes

# ---------------------------------------------------------------------------
# Adaptive Gauss-Hermite quadrature -- the accuracy tier above Laplace
# ---------------------------------------------------------------------------


def _gh_tensor_nodes(
    n_quad: int, r: int
) -> Tuple[Float[Array, 'K r'], Float[Array, 'K'], Float[Array, 'K']]:
    """Tensor-product Gauss-Hermite nodes for an ``r``-dimensional integral.

    Builds the ``K = n_quad ** r`` tensor-product nodes of the physicists'
    Gauss-Hermite rule :math:`\\int \\exp(-t^2)\\, g(t)\\, \\mathrm{d}t \\approx
    \\sum_k w_k\\, g(t_k)` in :math:`r` dimensions.  The one-dimensional nodes
    and weights come from :func:`numpy.polynomial.hermite.hermgauss`; the
    multi-dimensional grid takes their Cartesian product and the log weight of
    each tensor node is the sum of the per-axis log weights.  This is a static
    computation, evaluated once per fit in NumPy.

    Parameters
    ----------
    n_quad : int
        Number of quadrature nodes per axis (``nAGQ``).  ``n_quad = 1`` recovers
        the single-node Laplace approximation.
    r : int
        Dimension of the random-effect vector, i.e. the number of integration
        axes.

    Returns
    -------
    nodes : Float[Array, 'K r']
        The node coordinates :math:`t_k`, one row per tensor node.
    logw : Float[Array, 'K']
        The log product weight of each tensor node,
        :math:`\\sum_j \\log w_{k_j}`.
    sumsq : Float[Array, 'K']
        The squared norm :math:`\\lVert t_k \\rVert^2` of each node.
    """
    x, w = np.polynomial.hermite.hermgauss(n_quad)
    if r == 1:
        nodes = x[:, None]
        logw = np.log(w)
    else:
        mesh = np.meshgrid(*([x] * r), indexing='ij')
        nodes = np.stack([m.ravel() for m in mesh], axis=-1)  # (K, r)
        wmesh = np.meshgrid(*([w] * r), indexing='ij')
        logw = sum(np.log(wm.ravel()) for wm in wmesh)  # (K,)
    sumsq = np.sum(nodes**2, axis=-1)
    return jnp.asarray(nodes), jnp.asarray(logw), jnp.asarray(sumsq)


def _agq_slope_nll(
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
    nodes: Float[Array, 'K r'],
    logw: Float[Array, 'K'],
    sumsq: Float[Array, 'K'],
) -> Float[Array, '']:
    """Adaptive Gauss-Hermite marginal negative log-likelihood for a random slope.

    Evaluates the marginal negative log-likelihood (summed over groups) by
    adaptive Gauss-Hermite quadrature.  The tensor Gauss-Hermite nodes are
    centred and scaled at each group's conditional mode and curvature,
    :math:`b = \\hat{b}_g + \\sqrt{2}\\, L_g t_k` with
    :math:`L_g L_g^{\\top} = H_g^{-1}` (the principal-axis adaptive curvature
    factor), so that the per-group log marginal likelihood is

    .. math::

        \\log L_g = \\tfrac{r}{2}\\log 2 - \\tfrac{1}{2}\\log\\det H_g
        + \\operatorname{logsumexp}_k\\!\\bigl[\\log w_k + \\lVert t_k\\rVert^2
        + \\log g(\\hat{b}_g + \\mathrm{offset}_k)\\bigr],

    where :math:`g(b) = \\exp(\\ell_g(b))\\, \\mathcal{N}(b; 0, G)` is the
    integrand.  With ``n_quad = 1`` (:math:`t = 0`, :math:`w = \\sqrt{\\pi}`)
    the single term collapses exactly to the Laplace determinant correction;
    more nodes integrate the random-effect density directly, correcting the
    Laplace bias for small or low-count clusters (the ``nAGQ`` control in lme4).

    Parameters
    ----------
    theta : Float[Array, 'pm']
        Packed parameter vector ``[beta, chol(G)]``: the ``p`` fixed-effect
        coefficients followed by the Cholesky parameters of the random-effect
        covariance ``G``.
    y : Float[Array, 'N']
        Response vector over ``N`` observations.
    X : Float[Array, 'N p']
        Fixed-effect design matrix, ``(N, p)``.
    z : Float[Array, 'N r']
        Random-effect (slope) design matrix, ``(N, r)``.
    group : Int[Array, 'N']
        Per-observation group index in ``[0, n_groups)``.
    n_groups : int
        Number of groups.
    family : Family
        Exponential-family specification providing the link, inverse link,
        log-likelihood and ``eta`` clipping.
    p : int
        Number of fixed-effect coefficients.
    r : int
        Dimension of the random-effect vector.
    n_mode : int
        Number of inner Newton (Fisher-scoring) iterations used to locate the
        per-group conditional mode and curvature.
    diagonal : bool
        Whether the random-effect covariance ``G`` is constrained to be
        diagonal.
    nodes : Float[Array, 'K r']
        Tensor Gauss-Hermite node coordinates, ``(K, r)``.
    logw : Float[Array, 'K']
        Log product weight of each tensor node, ``(K,)``.
    sumsq : Float[Array, 'K']
        Squared norm of each node, ``(K,)``.

    Returns
    -------
    Float[Array, '']
        The scalar marginal negative log-likelihood, summed over groups.
    """

    beta = theta[:p]
    g_inv, logdet_g = small_inv_logdet(
        cov_re_from_chol(theta[p:], r, diagonal), r
    )
    b_hat, h_mat, _ = _laplace_slope_modes(
        theta, y, X, z, group, n_groups, family, p, r, n_mode, diagonal
    )
    # Per-group curvature factor L_g (L_g L_g^T = H_g^{-1}) and logdet H_g.
    h_inv, logdet_h = jax.vmap(lambda h: small_inv_logdet(h, r))(h_mat)
    l_scale = jax.vmap(lambda hi: spd_chol(hi, r))(h_inv)  # (q, r, r)
    sqrt2 = jnp.sqrt(jnp.asarray(2.0, dtype=X.dtype))
    offset = sqrt2 * jnp.einsum('grs,ks->gkr', l_scale, nodes)  # (q, K, r)
    b_node = b_hat[:, None, :] + offset  # (q, K, r)

    eta_fix = X @ beta
    eta = family.clip_eta(
        eta_fix[:, None] + jnp.einsum('nr,nkr->nk', z, b_node[group])
    )  # (N, K)
    ll = jax.ops.segment_sum(
        family.loglik(
            y[:, None], family.linkinv(eta), jnp.asarray(1.0, dtype=X.dtype)
        ),
        group,
        num_segments=n_groups,
    )  # (q, K)
    quad = 0.5 * jnp.einsum('gkr,rs,gks->gk', b_node, g_inv, b_node)
    log_g = ll - quad - 0.5 * logdet_g - 0.5 * r * jnp.log(2.0 * jnp.pi)
    log_marg = (
        0.5 * r * jnp.log(2.0)
        - 0.5 * logdet_h
        + logsumexp(logw[None, :] + sumsq[None, :] + log_g, axis=-1)
    )
    return -jnp.sum(log_marg)


def _glmm_agq_slope_one(
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
    nodes: Float[Array, 'K r'],
    logw: Float[Array, 'K'],
    sumsq: Float[Array, 'K'],
) -> Tuple[
    Float[Array, 'p'],
    Float[Array, 'q r'],
    Float[Array, 'r r'],
    Float[Array, ''],
]:
    """Fit a single-element adaptive Gauss-Hermite random-slope GLMM.

    Warm-starts the fixed effects with IRLS, initialises the covariance
    Cholesky parameters, then minimises the adaptive Gauss-Hermite marginal
    negative log-likelihood by damped Newton.  This mirrors the Laplace fit,
    except that the marginal likelihood is the ``n_quad``-point adaptive
    Gauss-Hermite integral rather than the single-node Laplace approximation.
    The random-effect predictions are the per-group conditional modes at the
    fitted parameters.

    Parameters
    ----------
    y : Float[Array, 'N']
        Response vector over ``N`` observations.
    X : Float[Array, 'N p']
        Fixed-effect design matrix, ``(N, p)``.
    z : Float[Array, 'N r']
        Random-effect (slope) design matrix, ``(N, r)``.
    group : Int[Array, 'N']
        Per-observation group index in ``[0, n_groups)``.
    n_groups : int
        Number of groups.
    family : Family
        Exponential-family specification.
    p : int
        Number of fixed-effect coefficients.
    r : int
        Dimension of the random-effect vector.
    n_mode : int
        Number of inner Newton iterations for the per-group conditional mode.
    spec : VarCompSpec
        Variance-component solver settings supplying the ridge penalty for the
        IRLS warm start and the keyword arguments for the outer damped-Newton
        optimiser.
    diagonal : bool
        Whether the random-effect covariance ``G`` is constrained to be
        diagonal.
    nodes : Float[Array, 'K r']
        Tensor Gauss-Hermite node coordinates, ``(K, r)``.
    logw : Float[Array, 'K']
        Log product weight of each tensor node, ``(K,)``.
    sumsq : Float[Array, 'K']
        Squared norm of each node, ``(K,)``.

    Returns
    -------
    beta : Float[Array, 'p']
        Fitted fixed-effect coefficients.
    blups : Float[Array, 'q r']
        Per-group conditional modes of the random slopes (the ``q`` groups by
        ``r`` random-effect dimensions).
    g_cov : Float[Array, 'r r']
        Fitted random-effect covariance ``G``.
    deviance : Float[Array, '']
        Twice the marginal negative log-likelihood at the fitted parameters.
    """
    from .._irls import irls_warm_start

    def nll(theta: Float[Array, 'pm']) -> Float[Array, '']:
        return _agq_slope_nll(
            theta,
            y,
            X,
            z,
            group,
            n_groups,
            family,
            p,
            r,
            n_mode,
            diagonal,
            nodes,
            logw,
            sumsq,
        )

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
    theta = damped_newton(
        nll, jnp.concatenate([beta0, chol0]), **spec.newton_kwargs
    )
    beta = theta[:p]
    g_cov = cov_re_from_chol(theta[p:], r, diagonal)
    b, _, _ = _laplace_slope_modes(
        theta, y, X, z, group, n_groups, family, p, r, n_mode, diagonal
    )
    return beta, b, g_cov, 2.0 * nll(theta)


def _glmm_agq_slope(
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
    n_quad: int,
    block: Optional[int],
) -> GLMMResult:
    """Adaptive Gauss-Hermite random-slope GLMM over ``V`` elements.

    Builds the static tensor Gauss-Hermite grid once, then fits the adaptive
    Gauss-Hermite random-slope model independently for each of the ``V`` rows
    (elements) of ``Y`` via a blocked ``vmap`` over :func:`_glmm_agq_slope_one`,
    assembling the per-element results into a single :class:`GLMMResult`.

    Parameters
    ----------
    Y : Float[Array, 'V N']
        Responses for ``V`` elements, each over the same ``N`` observations.
    X : Float[Array, 'N p']
        Shared fixed-effect design matrix, ``(N, p)``.
    group : Int[Array, 'N']
        Per-observation group index in ``[0, n_groups)``.
    n_groups : int
        Number of groups.
    z : Float[Array, 'N r']
        Shared random-effect (slope) design matrix, ``(N, r)``.
    family : Family
        Exponential-family specification.
    n_outer : int
        Number of outer damped-Newton iterations for the variance-component
        optimisation.
    n_mode : int
        Number of inner Newton iterations for the per-group conditional mode.
    damping : float
        Damping factor for the outer damped-Newton optimiser.
    diagonal : bool
        Whether the random-effect covariance ``G`` is constrained to be
        diagonal.
    n_quad : int
        Number of Gauss-Hermite quadrature nodes per axis (``nAGQ``).
    block : int, optional
        Block size for the blocked ``vmap`` over elements; ``None`` maps every
        element in one batch.

    Returns
    -------
    GLMMResult
        The assembled fit with ``tier='agq'``.  ``beta_hat`` is ``(V, p)``,
        ``blups`` is ``(V, q, r)``, ``re_var`` is the uniform ``(V, r, r)``
        random-effect covariance, ``deviance`` is ``(V,)`` twice the marginal
        negative log-likelihood, ``dispersion`` is a placeholder ``1`` and
        ``edf_total`` a placeholder fixed-effect count ``p`` per element.
    """
    p = X.shape[-1]
    r = z.shape[-1]
    spec = VarCompSpec(n_iter=n_outer, damping=damping)
    nodes, logw, sumsq = _gh_tensor_nodes(n_quad, r)

    def per_voxel(
        y: Float[Array, 'N'],
    ) -> Tuple[Array, Array, Array, Array]:
        return _glmm_agq_slope_one(
            y,
            X,
            z,
            group,
            n_groups,
            family,
            p,
            r,
            n_mode,
            spec,
            diagonal,
            nodes,
            logw,
            sumsq,
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
        tier='agq',
    )
