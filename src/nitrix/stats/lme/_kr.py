# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Kenward-Roger small-sample correction for fixed-effect inference.

The naive mixed-model Wald covariance :math:`\Phi = (X^{\top} V^{-1} X)^{-1}`
understates the true variance of :math:`\hat\beta` for two reasons that matter
when the number of groups is small: it ignores the downward bias from plugging
in the estimated variance components :math:`\hat\theta`, and it uses an
asymptotic (chi-squared) reference. Kenward & Roger (1997) correct both -- an
inflated covariance :math:`\Phi_A` and a scaled-:math:`F` reference with
adjusted denominator degrees of freedom -- giving close-to-nominal coverage
where the Satterthwaite approximation alone is anticonservative.

For the single-random-intercept model :math:`V = \sigma_b^2 Z Z^{\top} +
\sigma_e^2 I` the variance :math:`V` is linear in the natural components
:math:`(\sigma_b^2, \sigma_e^2)` (so the second-derivative term
:math:`R_{ij}` vanishes), and every quantity reduces to a diagonal weighting in
the FaST-LMM rotated basis :math:`V = \operatorname{diag}(d)`,
:math:`d_k = \sigma_b^2 \lambda_k + \sigma_e^2`:

.. math::

    \Phi   &= \left(\sum_k x_k x_k^{\top} / d_k\right)^{-1} \\
    P_i    &= \sum_k v_{i,k}\, x_k x_k^{\top} / d_k^2
              \quad (v_1 = \lambda,\ v_2 = 1) \\
    Q_{ij} &= \sum_k v_{i,k} v_{j,k}\, x_k x_k^{\top} / d_k^3 \\
    \Phi_A &= \Phi + 2 \Phi
              \left[\sum_{ij} W_{ij} (Q_{ij} - P_i \Phi P_j)\right] \Phi

with :math:`W` the inverse REML expected information of :math:`\theta` (also a
set of diagonal sums in the rotated basis). The scaled-:math:`F` maps the Wald
:math:`F` to :math:`F_{q, m}` via a moment-matched scale and denominator degrees
of freedom :math:`m`, both built from :math:`W` and
:math:`\partial\Phi/\partial\theta_i = \Phi P_i \Phi` contracted with the
contrast. Everything is closed-form and free of cuSOLVER (small
:math:`p \times p`, :math:`2 \times 2` and :math:`q \times q` solves via
:func:`small_inv_logdet`); the correction is computed under ``stop_gradient`` on
:math:`\hat\theta` (an inference read-out, like the Satterthwaite degrees of
freedom).

References
----------
- Kenward, M. G. & Roger, J. H. (1997). Small sample inference for fixed effects
  from restricted maximum likelihood. Biometrics 53, 983-997.
  https://doi.org/10.2307/2533558
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ...linalg._smalllinalg import small_inv_logdet

__all__ = ['kr_cov_and_scaled_f']

_TINY = 1e-30


def _rotated_reductions(
    theta: Float[Array, '2'],
    x_rot: Float[Array, 'N p'],
    lambdas: Float[Array, 'N'],
    p: int,
) -> Tuple[
    Float[Array, 'p p'],  # Phi
    Float[Array, '2 p p'],  # P_i
    Float[Array, '2 2 p p'],  # Q_ij
    Float[Array, '2 2'],  # W (inverse REML expected info)
]:
    r"""
    Kenward-Roger rotated-basis reductions at a variance-component estimate.

    Evaluates the four building blocks of the small-sample correction in the
    FaST-LMM rotated basis, where the marginal variance is diagonal,
    :math:`d_k = \sigma_b^2 \lambda_k + \sigma_e^2` with
    :math:`(\sigma_b^2, \sigma_e^2) = \exp(\theta)`. The blocks are the Wald
    covariance :math:`\Phi = (\sum_k x_k x_k^{\top} / d_k)^{-1}`, its first- and
    second-order weighted Gram sums :math:`P_i` and :math:`Q_{ij}`, and the
    inverse REML expected information :math:`W` of :math:`\theta`.

    Parameters
    ----------
    theta : Float[Array, '2']
        Log variance components :math:`(\log \sigma_b^2, \log \sigma_e^2)`.
    x_rot : Float[Array, 'N p']
        Design matrix rotated into the eigenbasis of the random-effect
        structure, one row per observation.
    lambdas : Float[Array, 'N']
        Eigenvalues of the random-effect Gram matrix in the rotated basis,
        i.e. the coefficients :math:`\lambda_k` multiplying
        :math:`\sigma_b^2` in :math:`d_k`.
    p : int
        Number of fixed-effect coefficients (the size of the
        :math:`p \times p` solves).

    Returns
    -------
    phi : Float[Array, 'p p']
        Naive Wald covariance :math:`\Phi = (X^{\top} V^{-1} X)^{-1}`.
    pmat : Float[Array, '2 p p']
        First-order weighted Gram sums :math:`P_i` for
        :math:`i \in \{b, e\}`.
    qmat : Float[Array, '2 2 p p']
        Second-order weighted Gram sums :math:`Q_{ij}`.
    w : Float[Array, '2 2']
        Inverse REML expected information :math:`W` of :math:`\theta`, i.e.
        the asymptotic covariance of the variance-component estimates.
    """
    sb2 = jnp.exp(theta[0])
    se2 = jnp.exp(theta[1])
    d = sb2 * lambdas + se2  # (N,)
    v = jnp.stack([lambdas, jnp.ones_like(lambdas)], axis=0)  # (2, N)

    def gram(weight: Float[Array, 'N']) -> Float[Array, 'p p']:
        return jnp.einsum('np,nq,n->pq', x_rot, x_rot, weight)

    xtvix = gram(1.0 / d)
    phi, _ = small_inv_logdet(xtvix, p)
    pmat = jnp.stack([gram(v[i] / d**2) for i in range(2)], axis=0)  # (2,p,p)
    qmat = jnp.stack(
        [
            jnp.stack([gram(v[i] * v[j] / d**3) for j in range(2)], axis=0)
            for i in range(2)
        ],
        axis=0,
    )  # (2, 2, p, p)

    # REML expected information of theta: I_ij = 0.5 tr(P V_i P V_j), with
    # P = V^{-1} - V^{-1} X Phi X^T V^{-1}.  In the diagonal basis this is
    # 0.5[ sum v_i v_j / d^2  - 2 sum v_i v_j (x^T Phi x) / d^3  + tr(Phi P_i Phi P_j) ].
    xpx = jnp.einsum('np,pq,nq->n', x_rot, phi, x_rot)  # (N,) x_k^T Phi x_k
    info = jnp.zeros((2, 2), dtype=x_rot.dtype)
    for i in range(2):
        for j in range(2):
            t1 = jnp.sum(v[i] * v[j] / d**2)
            t2 = jnp.sum(v[i] * v[j] * xpx / d**3)
            t3 = jnp.trace(phi @ pmat[i] @ phi @ pmat[j])
            info = info.at[i, j].set(0.5 * (t1 - 2.0 * t2 + t3))
    w, _ = small_inv_logdet(info, 2)
    return phi, pmat, qmat, w


def kr_cov_and_scaled_f(
    theta: Float[Array, '2'],
    beta: Float[Array, 'p'],
    x_rot: Float[Array, 'N p'],
    lambdas: Float[Array, 'N'],
    contrast: Float[Array, 'q p'],
    p: int,
    q: int,
) -> Tuple[Float[Array, ''], Float[Array, ''], Float[Array, 'p p']]:
    r"""
    Kenward-Roger scaled-:math:`F` statistic and adjusted covariance for one
    voxel.

    Builds the small-sample correction for a single fixed-effect contrast: the
    inflated covariance :math:`\Phi_A`, a Wald :math:`F` evaluated on it, and a
    moment-matched scale that maps that statistic onto an :math:`F(q, df_2)`
    reference. The variance-component dependence is taken at the converged
    :math:`\theta` under ``stop_gradient`` (an inference read-out); the returned
    statistic stays differentiable through ``beta``.

    Parameters
    ----------
    theta : Float[Array, '2']
        Converged log variance components
        :math:`(\log \sigma_b^2, \log \sigma_e^2)`. Detached from the gradient.
    beta : Float[Array, 'p']
        Fixed-effect coefficient estimates :math:`\hat\beta`.
    x_rot : Float[Array, 'N p']
        Design matrix rotated into the random-effect eigenbasis, one row per
        observation.
    lambdas : Float[Array, 'N']
        Eigenvalues of the random-effect Gram matrix in the rotated basis.
    contrast : Float[Array, 'q p']
        Contrast matrix :math:`C` selecting the :math:`q` linear combinations
        of ``beta`` under test.
    p : int
        Number of fixed-effect coefficients.
    q : int
        Number of contrast rows (numerator degrees of freedom).

    Returns
    -------
    f_kr : Float[Array, '']
        Kenward-Roger scaled Wald statistic
        :math:`F_{KR} = \mathrm{scale} \cdot F_{\mathrm{Wald}}`, to be referred
        to an :math:`F(q, df_2)` distribution.
    df2 : Float[Array, '']
        Adjusted denominator degrees of freedom :math:`df_2`, floored just
        above ``2`` to guard the degenerate near-boundary case.
    phi_a : Float[Array, 'p p']
        Kenward-Roger adjusted covariance :math:`\Phi_A` of :math:`\hat\beta`.
    """
    theta = jax.lax.stop_gradient(theta)
    phi, pmat, qmat, w = _rotated_reductions(theta, x_rot, lambdas, p)

    # Adjusted covariance Phi_A = Phi + 2 Phi [ sum_ij W_ij (Q_ij - P_i Phi P_j) ] Phi.
    middle = jnp.zeros((p, p), dtype=x_rot.dtype)
    for i in range(2):
        for j in range(2):
            middle = middle + w[i, j] * (qmat[i, j] - pmat[i] @ phi @ pmat[j])
    phi_a = phi + 2.0 * phi @ middle @ phi

    # Wald F on the adjusted covariance (differentiable through beta).
    cb = contrast @ beta  # (q,)
    m_cov = contrast @ phi_a @ contrast.T  # (q, q)
    m_inv, _ = small_inv_logdet(m_cov, q)
    f_wald = (cb @ m_inv @ cb) / q

    # KR2 scale + denominator df, from W and dPhi/dtheta_i = Phi P_i Phi.
    ptil = jnp.stack([phi @ pmat[i] @ phi for i in range(2)], axis=0)
    a1 = jnp.asarray(0.0, dtype=x_rot.dtype)
    a2 = jnp.asarray(0.0, dtype=x_rot.dtype)
    for i in range(2):
        ti = m_inv @ (contrast @ ptil[i] @ contrast.T)  # (q, q)
        for j in range(2):
            tj = m_inv @ (contrast @ ptil[j] @ contrast.T)
            a1 = a1 + w[i, j] * jnp.trace(ti) * jnp.trace(tj)
            a2 = a2 + w[i, j] * jnp.trace(ti @ tj)

    qf = float(q)
    b = (a1 + 6.0 * a2) / (2.0 * qf)
    g = ((qf + 1.0) * a1 - (qf + 4.0) * a2) / jnp.clip(
        (qf + 2.0) * a2, _TINY, None
    )
    den = 3.0 * qf + 2.0 * (1.0 - g)
    c1 = g / den
    c2 = (qf - g) / den
    c3 = (qf + 2.0 - g) / den
    e_star = 1.0 / jnp.clip(1.0 - a2 / qf, _TINY, None)
    v_star = (
        (2.0 / qf)
        * (1.0 + c1 * b)
        / (jnp.clip((1.0 - c2 * b) ** 2 * (1.0 - c3 * b), _TINY, None))
    )
    rho = v_star / (2.0 * e_star**2)
    df2 = 4.0 + (qf + 2.0) / jnp.clip(qf * rho - 1.0, _TINY, None)
    # The moment-matched scale; F_kr = scale * F_wald ~ F(q, df2).  Guard the
    # degenerate near-boundary case (df2 <= 2) with a conservative fallback.
    df2 = jnp.where(df2 > 2.0, df2, 2.0 + _TINY)
    scale = df2 / (e_star * (df2 - 2.0))
    f_kr = jnp.clip(scale, _TINY, None) * f_wald
    return f_kr, df2, phi_a
