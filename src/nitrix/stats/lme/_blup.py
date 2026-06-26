# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Random-effect modes (BLUPs) and conditional prediction for the LME fitters.

The *apply* half of the mixed-model estimator: `lme_fit` estimates the
variance components and fixed effects, and this module reads off the random-
effect modes ``b_g`` (the BLUPs) and predicts on a new design at either the
**population** (marginal, ``X beta``) or **conditional** (subject-specific,
``X beta + Z b_g``) level.

The BLUP for a single block-diagonal-by-group factor (the R1 scalar-intercept
and R2 random-slope tiers) is the mixed-model-equation form

    b_g = (Z_g^T Z_g / sigma_e^2 + G^{-1})^{-1} (Z_g^T r_g / sigma_e^2),
    r = y - X beta,

computed once at fit (opt-in ``retain_blups=True``) from the converged
``(beta, G, sigma_e^2)`` and the training ``(y, X, Z, group)`` -- a post-fit
pass that never touches the inner REML solver, so the default fit path is
byte-identical.

The structured-residual (R2 + corr) tier reuses the **same** mixed-model
equation on **whitened** data: with ``Cov(eps_g) = sigma_e^2 R_g(rho)`` and the
per-group whitener ``W_g`` (``W_g R_g W_g^T = I``, so ``W_g^T W_g = R_g^{-1}``),
``Z_g^T R_g^{-1} Z_g = (W_g Z_g)^T (W_g Z_g)`` and ``Z_g^T R_g^{-1} r_g =
(W_g Z_g)^T (W_g r_g)`` -- the standard BLUP on whitened ``(Z_g, r_g)``.  That
pass lives in :mod:`._corrfit` (where the group/whitening layout is), sharing
the cuSOLVER-free per-group solve (:func:`_solve_blup_system`) here.  The
crossed (R4) and nested (R3) tiers do **not** have this single block-diagonal-
by-group structure and are not yet covered (``retain_blups`` raises there); the
population level works for every tier.
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
    """Per-group mixed-model-equation solve ``b_g = a_g^{-1} rhs_g``.

    The shared cuSOLVER-free atom of every BLUP: ``a_g = Z_g^T Sigma_g^{-1} Z_g +
    G^{-1}`` and ``rhs_g = Z_g^T Sigma_g^{-1} r_g`` per group, vmapped over the
    ``q`` groups via the ``r x r`` ``small_inv_logdet`` solve.  ``_blups_standard``
    (segment-sum Grams, ``Sigma = sigma_e^2 I``) and ``_corrfit._blups_corr``
    (whitened Grams, ``Sigma = sigma_e^2 R(rho)``) assemble ``a`` / ``rhs`` in
    their own layout and call this for the solve.
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

    The mixed-model-equation BLUP ``b_g = (Z_g^T Z_g / sigma_e^2 + G^{-1})^{-1}
    Z_g^T r_g / sigma_e^2`` (``r = y - X beta``), vectorised over elements and
    groups via ``segment_sum`` + the cuSOLVER-free ``r x r`` solve.  Returns
    ``(V, q, r)`` -- one ``r``-vector per (element, group).  For a scalar
    intercept (``r = 1``) this is the shrinkage estimator ``sigma_b^2
    sum_g(r) / (sigma_e^2 + n_g sigma_b^2)``.
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

    Returns the per-level modes ``b_g`` stored on the result -- ``(V, q)`` for a
    scalar random intercept, ``(V, q, r)`` for a random slope.  Available on a
    :class:`~nitrix.stats.GLMMResult` (always) and on an LME result fitted with
    ``lme_fit(..., retain_blups=True)``.

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
    """Linear predictor at ``X`` -- ``X beta`` (population) plus the per-row
    random-effect contribution ``Z b_group`` (conditional).

    Unseen / unspecified groups fall back to the marginal mean (random mode 0):
    ``group=None`` is the population level for every row.  Shared by
    :func:`lme_predict` and ``glmm_predict``.
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

    ``level='population'`` (default) returns the marginal mean ``X beta_hat``
    (defined for **every** tier).  ``level='conditional'`` adds the
    subject-specific random-effect contribution ``Z b_group`` using the fitted
    modes (:func:`ranef`), so the result must come from ``lme_fit(...,
    retain_blups=True)``.  A ``group`` label outside the fitted range (an unseen
    subject) -- or ``group=None`` -- falls back to the marginal mean.

    Parameters
    ----------
    result
        An LME result (``REMLResult`` / ``LMEResult`` / ...).
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
    ``(V, N)`` predicted means.  Differentiable w.r.t. ``X`` / ``z`` (and the
    fitted coefficients / modes).
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
