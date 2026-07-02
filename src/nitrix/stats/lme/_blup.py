# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Random-effect modes (BLUPs) and conditional prediction for the LME fitters.

The *apply* half of the mixed-model estimator: :func:`lme_fit` estimates the
variance components and fixed effects, and this module reads off the random-
effect modes :math:`b_g` (the BLUPs) and predicts on a new design at either the
**population** (marginal, :math:`X\\beta`) or **conditional** (subject-specific,
:math:`X\\beta + Z b_g`) level.

The BLUP for a single block-diagonal-by-group factor (the scalar-intercept and
random-slope tiers) is the mixed-model-equation form

.. math::

    b_g = \\left(Z_g^{\\top} Z_g / \\sigma_e^2 + G^{-1}\\right)^{-1}
          \\left(Z_g^{\\top} r_g / \\sigma_e^2\\right),
    \\qquad r = y - X\\beta,

computed once at fit (opt-in ``retain_blups=True``) from the converged
:math:`(\\beta, G, \\sigma_e^2)` and the training ``(y, X, Z, group)`` -- a
post-fit pass that never touches the inner REML solver, so the default fit path
is byte-identical.

The structured-residual (random-slope + correlated-residual) tier reuses the
**same** mixed-model equation on **whitened** data: with
:math:`\\operatorname{Cov}(\\varepsilon_g) = \\sigma_e^2 R_g(\\rho)` and the
per-group whitener :math:`W_g` (:math:`W_g R_g W_g^{\\top} = I`, so
:math:`W_g^{\\top} W_g = R_g^{-1}`),
:math:`Z_g^{\\top} R_g^{-1} Z_g = (W_g Z_g)^{\\top} (W_g Z_g)` and
:math:`Z_g^{\\top} R_g^{-1} r_g = (W_g Z_g)^{\\top} (W_g r_g)` -- the standard
BLUP on whitened :math:`(Z_g, r_g)`.  That pass lives in the correlated-residual
fitting module (where the group/whitening layout is), sharing the cuSOLVER-free
per-group solve (:func:`_solve_blup_system`) here.  The crossed and nested tiers
do **not** have this single block-diagonal-by-group structure and are not yet
covered (``retain_blups`` raises there); the population level works for every
tier.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ...linalg._smalllinalg import small_inv_logdet

__all__ = ['ranef', 'lme_predict']

PredictLevel = Literal['population', 'conditional']


def _solve_blup_system(
    a: Float[Array, 'q r r'],
    rhs: Float[Array, 'q r'],
    r_dim: int,
) -> Float[Array, 'q r']:
    """Per-group mixed-model-equation solve :math:`b_g = a_g^{-1} \\mathrm{rhs}_g`.

    The shared cuSOLVER-free atom of every BLUP: with
    :math:`a_g = Z_g^{\\top} \\Sigma_g^{-1} Z_g + G^{-1}` and
    :math:`\\mathrm{rhs}_g = Z_g^{\\top} \\Sigma_g^{-1} r_g` per group, the mode
    :math:`b_g` is recovered by inverting each :math:`r \\times r` system, vmapped
    over the :math:`q` groups via :func:`~nitrix.linalg._smalllinalg.small_inv_logdet`.
    :func:`_blups_standard` (segment-sum Grams, :math:`\\Sigma = \\sigma_e^2 I`)
    and the correlated-residual path (whitened Grams,
    :math:`\\Sigma = \\sigma_e^2 R(\\rho)`) assemble ``a`` / ``rhs`` in their own
    layout and call this for the solve.

    Parameters
    ----------
    a
        ``(q, r, r)`` per-group left-hand-side matrices
        :math:`Z_g^{\\top} \\Sigma_g^{-1} Z_g + G^{-1}`, one for each of the
        ``q`` groups.
    rhs
        ``(q, r)`` per-group right-hand sides
        :math:`Z_g^{\\top} \\Sigma_g^{-1} r_g`.
    r_dim
        Random-effect dimension :math:`r` (the block size passed to the small
        inverse solve).

    Returns
    -------
    Float[Array, 'q r']
        The ``(q, r)`` per-group random-effect modes :math:`b_g`.
    """
    solved = jax.vmap(lambda av, bv: small_inv_logdet(av, r_dim)[0] @ bv)(
        a, rhs
    )
    return cast(Float[Array, 'q r'], solved)


def _blups_standard(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    beta: Float[Array, 'V p'],
    z: Float[Array, 'N r'],
    group: Int[Array, ' N'],
    cov_re: Float[Array, 'V r r'],
    sigma_e_sq: Float[Array, ' V'],
    n_groups: int,
) -> Float[Array, 'V q r']:
    """Per-group random-effect modes for a single block-diagonal factor.

    The mixed-model-equation BLUP
    :math:`b_g = (Z_g^{\\top} Z_g / \\sigma_e^2 + G^{-1})^{-1}
    Z_g^{\\top} r_g / \\sigma_e^2` (with residual :math:`r = y - X\\beta`),
    vectorised over elements and groups via segment sums and the cuSOLVER-free
    :math:`r \\times r` solve (:func:`_solve_blup_system`).  For a scalar
    intercept (:math:`r = 1`) this reduces to the shrinkage estimator
    :math:`\\sigma_b^2 \\sum_g r / (\\sigma_e^2 + n_g \\sigma_b^2)`.

    Parameters
    ----------
    Y
        ``(V, N)`` responses: one length-``N`` observation vector per element.
    X
        ``(N, p)`` fixed-effect design (shared across elements).
    beta
        ``(V, p)`` fitted fixed-effect coefficients, one row per element.
    z
        ``(N, r)`` random-effect design for the block-diagonal factor.
    group
        ``(N,)`` integer group labels in ``0..n_groups-1`` assigning each row to
        one of the random-effect blocks.
    cov_re
        ``(V, r, r)`` per-element random-effect covariance :math:`G`.
    sigma_e_sq
        ``(V,)`` per-element residual variance :math:`\\sigma_e^2`.
    n_groups
        Number of groups :math:`q` (segment count for the group reductions).

    Returns
    -------
    Float[Array, 'V q r']
        The per-element, per-group random-effect modes :math:`b_g`: one
        length-``r`` vector for each (element, group) pair.
    """
    r_dim = z.shape[-1]
    resid = Y - beta @ X.T  # (V, N)
    # Z_g^T Z_g per group (element-independent).
    ztz = jax.ops.segment_sum(
        z[:, :, None] * z[:, None, :], group, num_segments=n_groups
    )  # (q, r, r)

    def _ztr(resid_v: Float[Array, ' N']) -> Float[Array, 'q r']:
        return jax.ops.segment_sum(
            z * resid_v[:, None], group, num_segments=n_groups
        )

    ztr = jax.vmap(_ztr)(resid)  # (V, q, r)
    g_inv = jax.vmap(lambda g: small_inv_logdet(g, r_dim)[0])(
        cov_re
    )  # (V,r,r)
    inv_s = 1.0 / sigma_e_sq  # (V,)
    a = ztz[None] * inv_s[:, None, None, None] + g_inv[:, None, :, :]
    rhs = ztr * inv_s[:, None, None]  # (V, q, r)
    return jax.vmap(lambda a_v, rhs_v: _solve_blup_system(a_v, rhs_v, r_dim))(
        a, rhs
    )  # (V, q, r)


def ranef(result: Any) -> Float[Array, 'V q r']:
    """Random-effect modes (BLUPs) of a fitted mixed model.

    Returns the per-level modes :math:`b_g` stored on the result -- ``(V, q)``
    for a scalar random intercept, ``(V, q, r)`` for a random slope.  Available
    on a :class:`~nitrix.stats.GLMMResult` (always) and on an LME result fitted
    with :func:`lme_fit` under ``retain_blups=True``.

    Parameters
    ----------
    result
        A fitted mixed-model result carrying a ``blups`` attribute -- a
        :class:`~nitrix.stats.GLMMResult`, or an LME result fitted with
        ``retain_blups=True``.

    Returns
    -------
    Float[Array, 'V q r']
        The per-element, per-group random-effect modes :math:`b_g` -- ``(V, q)``
        for a scalar random intercept, ``(V, q, r)`` for a random slope.

    Raises
    ------
    ValueError
        If the result carries no modes (an LME fit with the default
        ``retain_blups=False``, or a tier whose conditional BLUPs are not yet
        supported).
    """
    blups = getattr(result, 'blups', None)
    if blups is None:
        raise ValueError(
            'ranef: this result carries no random-effect modes.  Re-fit with '
            'lme_fit(..., retain_blups=True) (the modes are opt-in so the '
            'default fit path stays free), or use a GLMM result (which always '
            'retains its BLUPs).'
        )
    return cast(Float[Array, 'V q r'], blups)


def _conditional_eta(
    beta: Float[Array, 'V p'],
    X: Float[Array, 'N p'],
    *,
    level: PredictLevel,
    blups: Optional[Float[Array, 'V q r']],
    z: Optional[Float[Array, 'N r']],
    group: Optional[Int[Array, ' N']],
) -> Float[Array, 'V N']:
    """Linear predictor at a design ``X`` at the population or conditional level.

    Returns :math:`X\\beta` at the ``'population'`` level, and adds the per-row
    random-effect contribution :math:`Z b_{\\mathrm{group}}` at the
    ``'conditional'`` level.  Unseen or unspecified groups fall back to the
    marginal mean (random mode 0): ``group=None`` yields the population level for
    every row.  Shared by :func:`lme_predict` and
    :func:`~nitrix.stats.glmm_predict`.

    Parameters
    ----------
    beta
        ``(V, p)`` fitted fixed-effect coefficients, one row per element.
    X
        ``(N, p)`` fixed-effect design for the rows to predict.
    level
        ``'population'`` for the marginal mean :math:`X\\beta`, or
        ``'conditional'`` to add the subject-specific random-effect term.
    blups
        ``(V, q)`` (scalar intercept) or ``(V, q, r)`` (random slope)
        random-effect modes; required for the conditional level, ``None``
        otherwise.
    z
        ``(N, r)`` random-effect design for the new rows; required for the
        conditional level of a random-slope fit, ``None`` for a scalar intercept.
    group
        ``(N,)`` integer group labels indexing the fitted levels ``0..q-1``;
        labels outside that range, or ``group=None``, fall back to the marginal
        mean for those rows.

    Returns
    -------
    Float[Array, 'V N']
        The ``(V, N)`` linear predictor, one length-``N`` vector per element.
    """
    eta = beta @ X.T  # (V, N)
    if level == 'population':
        return eta
    if level != 'conditional':
        raise ValueError(
            f"level={level!r}; expected 'population' or 'conditional'."
        )
    if blups is None:
        raise ValueError(
            "level='conditional' needs the random-effect modes; see ranef()."
        )
    if group is None:
        return eta  # no group assignment -> marginal fallback for all rows
    group = jnp.asarray(group)
    n_groups = blups.shape[1]
    seen = group < n_groups  # unseen levels (>= q) fall back to the margin
    safe_group = jnp.where(seen, group, 0)
    if blups.ndim == 2:  # scalar intercept: (V, q)
        contrib = blups[:, safe_group]  # (V, N)
    else:  # random slope: (V, q, r) with the per-row design z (N, r)
        if z is None:
            raise ValueError(
                "level='conditional' on a random-slope fit needs z (the new "
                'rows random-effect design).'
            )
        b_rows = blups[:, safe_group, :]  # (V, N, r)
        contrib = jnp.einsum('vnr,nr->vn', b_rows, jnp.asarray(z, eta.dtype))
    return eta + jnp.where(seen[None, :], contrib, 0.0)


def lme_predict(
    result: Any,
    X: Float[Array, 'N p'],
    *,
    z: Optional[Float[Array, 'N r']] = None,
    group: Optional[Int[Array, ' N']] = None,
    level: PredictLevel = 'population',
) -> Float[Array, 'V N']:
    """Per-element LME prediction on a (new) design ``X``.

    ``level='population'`` (default) returns the marginal mean
    :math:`X\\hat{\\beta}` (defined for **every** tier).  ``level='conditional'``
    adds the subject-specific random-effect contribution
    :math:`Z b_{\\mathrm{group}}` using the fitted modes (:func:`ranef`), so the
    result must come from :func:`lme_fit` under ``retain_blups=True``.  A
    ``group`` label outside the fitted range (an unseen subject) -- or
    ``group=None`` -- falls back to the marginal mean.

    Parameters
    ----------
    result
        An LME result (:class:`~nitrix.stats.REMLResult`,
        :class:`~nitrix.stats.LMEResult`, or another fitted-LME result carrying
        ``beta_hat`` and, for the conditional level, ``blups``).
    X
        ``(N, p)`` fixed-effect design (same columns as the fit).
    z
        ``(N, r)`` random-effect design for the new rows (a random-slope fit);
        ``None`` for a scalar intercept.
    group
        ``(N,)`` integer group labels indexing the fitted levels ``0..q-1``;
        out-of-range / ``None`` -> the marginal mean for those rows.
    level
        ``'population'`` (marginal) or ``'conditional'`` (subject-specific BLUP).

    Returns
    -------
    Float[Array, 'V N']
        The ``(V, N)`` predicted means, one length-``N`` vector per element.
        Differentiable with respect to ``X`` / ``z`` (and the fitted
        coefficients / modes).
    """
    blups = getattr(result, 'blups', None)
    return _conditional_eta(
        result.beta_hat,
        jnp.asarray(X, result.beta_hat.dtype),
        level=level,
        blups=blups,
        z=z,
        group=group,
    )
