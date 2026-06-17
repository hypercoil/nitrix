# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Permutation inference for the GLM -- the on-device ``randomise`` engine.

``permutation_test`` is the kernel ``niffi`` needs in place of FSL
``randomise``: a non-parametric, family-wise-error-controlling test of a GLM
contrast over a statistic image, built from the parts in this subpackage.

The procedure
-------------

For data ``Y`` (one response per observation per voxel), design ``X``, and a
t-contrast ``c``:

1. Fit OLS and form the observed statistic image (t, or a variance-smoothed
   pseudo-t, FSL ``-v``), then **enhance** it (TFCE / cluster / raw voxel).
2. For each of ``n_perm`` relabellings (sign flips or permutations, honouring
   exchangeability blocks), refit and re-enhance.  Nuisance regressors are
   handled by **Freedman-Lane**: permute the residuals of the reduced
   (nuisance-only) model and add the nuisance fit back, so only the tested
   effect is exchanged.
3. Accumulate, without storing every permutation: the **uncorrected** p-map
   (fraction of permutations whose voxel statistic >= the observed) and the
   **FWE** p-map (fraction whose *spatial maximum* enhanced statistic >= the
   observed enhanced value).  The first permutation is the identity, so the
   observed is included and ``p >= 1 / n_perm``.

Everything is cuSOLVER-free (the only solve is a shared ``(p, p)`` inverse via
``stats._smalllinalg``) and runs on the broken-cuSOLVER GPU.  The enhancement
is a non-differentiable inference kernel (it forms discrete clusters); the
returned maps are arrays -- the CLI / containers / design parsing stay in the
consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Bool, Float, Int

from .._smalllinalg import small_inv_logdet
from .permutation import permutations, sign_flips
from .tfce import tfce

__all__ = ['PermResult', 'permutation_test']

Enhancement = Literal['tfce', 'voxel']
Exchange = Literal['sign', 'perm']

# A voxel whose observed residual variance is below this fraction of the
# typical in-mask variance is treated as degenerate (constant / no information)
# and excluded -- its SE-floored statistic would otherwise be a spurious +inf.
_VAR_REL_FLOOR = 1e-10


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PermResult:
    """Permutation-test output (all maps in the input spatial shape).

    Attributes
    ----------
    stat
        The observed statistic image (t / pseudo-t).
    enhanced
        The observed enhanced image (TFCE / cluster / |t|).
    p_fwe
        Family-wise-error-corrected p-map (max-statistic null).
    p_uncorrected
        Uncorrected voxelwise permutation p-map.
    null_max
        ``(n_perm,)`` null distribution of the spatial-maximum enhanced
        statistic (the FWE reference distribution).
    """

    stat: Float[Array, '*spatial']
    enhanced: Float[Array, '*spatial']
    p_fwe: Float[Array, '*spatial']
    p_uncorrected: Float[Array, '*spatial']
    null_max: Float[Array, 'n_perm']

    def tree_flatten(self) -> Tuple[Tuple[Array, ...], None]:
        return (
            self.stat,
            self.enhanced,
            self.p_fwe,
            self.p_uncorrected,
            self.null_max,
        ), None

    @classmethod
    def tree_unflatten(
        cls, _aux: None, children: Tuple[Any, ...]
    ) -> 'PermResult':
        return cls(*children)


def _ols_pre(
    X: Float[Array, 'N p'],
) -> Tuple[Float[Array, 'p N'], Float[Array, '']]:
    """Shared OLS pieces: hat pre-multiplier ``M = (X^T X)^{-1} X^T`` and the
    contrast-independent variance factor is computed by the caller."""
    p = X.shape[-1]
    xtx_inv, _ = small_inv_logdet(X.T @ X, p)
    return xtx_inv @ X.T, xtx_inv


def _residual_former(
    Z: Optional[Float[Array, 'N q']], n: int, dtype: Any
) -> Tuple[Float[Array, 'N N'], Float[Array, 'N N']]:
    """Reduced-model fit and residual-forming matrices for Freedman-Lane.

    Returns ``(fit_z, res_z)`` with ``fit_z = Z (Z^T Z)^{-1} Z^T`` and
    ``res_z = I - fit_z``.  ``Z = None`` -> ``fit_z = 0``, ``res_z = I`` (the
    no-nuisance case: permute the data directly)."""
    eye = jnp.eye(n, dtype=dtype)
    if Z is None:
        return jnp.zeros((n, n), dtype=dtype), eye
    q = Z.shape[-1]
    ztz_inv, _ = small_inv_logdet(Z.T @ Z, q)
    fit_z = Z @ (ztz_inv @ Z.T)
    return fit_z, eye - fit_z


def permutation_test(
    data: Float[Array, '*spatial N'],
    design: Float[Array, 'N p'],
    contrast: Float[Array, 'p'],
    *,
    key: Array,
    n_perm: int = 500,
    enhancement: Enhancement = 'tfce',
    exchange: Exchange = 'sign',
    nuisance: Optional[Float[Array, 'N q']] = None,
    blocks: Optional[Int[Array, 'N']] = None,
    mask: Optional[Bool[Array, '*spatial']] = None,
    two_sided: bool = True,
    var_smooth: Optional[float] = None,
    connectivity: int = 1,
    tfce_E: float = 0.5,
    tfce_H: float = 2.0,
    tfce_steps: int = 100,
) -> PermResult:
    """Permutation test of a GLM t-contrast with FWE control.

    Parameters
    ----------
    data
        ``(*spatial, N)`` responses (the observation axis is last).
    design
        ``(N, p)`` GLM design (includes the effect and any nuisance columns).
    contrast
        ``(p,)`` t-contrast vector ``c`` (tests ``c^T beta = 0``).
    key
        ``jax.random`` key (RNG policy is the caller's).
    n_perm
        Number of permutations incl. the identity (default ``500``).
    enhancement
        ``'tfce'`` (default) or ``'voxel'`` (raw statistic, voxelwise FWE).
    exchange
        ``'sign'`` (sign-flipping; symmetric / one-sample) or ``'perm'``
        (row permutation; two-sample / regression).
    nuisance
        Optional ``(N, q)`` nuisance design for Freedman-Lane.  ``None`` permutes
        the data directly (exact for designs with no nuisance).
    blocks
        Optional per-observation exchangeability-block labels.
    mask
        Optional spatial mask; out-of-mask voxels do not contribute to clusters
        or the maximum.
    two_sided
        Two-sided test (default ``True``).
    var_smooth
        Optional Gaussian sigma for variance smoothing (pseudo-t, FSL ``-v``).
    connectivity, tfce_E, tfce_H, tfce_steps
        Cluster / TFCE parameters.

    Returns
    -------
    ``PermResult`` with the observed statistic and enhanced maps, the FWE and
    uncorrected p-maps, and the null max distribution.
    """
    spatial = data.shape[:-1]
    n = data.shape[-1]
    p = design.shape[-1]
    v = prod(spatial)
    if design.shape[0] != n:
        raise ValueError(
            f'permutation_test: design has {design.shape[0]} rows; N={n}.'
        )
    Y = data.reshape(v, n)
    c = jnp.asarray(contrast)
    mask_flat = (
        jnp.ones((v,), dtype=bool) if mask is None else mask.reshape(v)
    )

    M, xtx_inv = _ols_pre(design)
    c_var = c @ (xtx_inv @ c)  # scalar
    dof = float(n - p)
    fit_z, res_z = _residual_former(nuisance, n, Y.dtype)
    e0 = Y @ res_z.T  # reduced-model residuals (V, N)
    base = Y @ fit_z.T  # nuisance fit to add back (V, N); 0 if no nuisance

    # Fold zero-variance (constant / degenerate) voxels into the mask: their
    # statistic is undefined and the SE floor would otherwise inflate it into a
    # spurious maximum that corrupts the max-statistic null.  Identified once
    # from the observed data (relative to the typical in-mask variance) and
    # excluded everywhere -- clustering, the max statistic, and the p-maps.
    beta_obs = Y @ M.T
    resid_obs = Y - beta_obs @ design.T
    sigma2_obs = jnp.sum(resid_obs * resid_obs, axis=-1) / dof
    var_scale = jnp.max(jnp.where(mask_flat, sigma2_obs, 0.0))
    mask_flat = mask_flat & (sigma2_obs > _VAR_REL_FLOOR * var_scale)

    def statistic(Yp: Float[Array, 'V N']) -> Float[Array, 'V']:
        beta = Yp @ M.T  # (V, p)
        resid = Yp - beta @ design.T
        sigma2 = jnp.sum(resid * resid, axis=-1) / dof
        if var_smooth is not None:
            sigma2 = jnp.reshape(
                _smooth(jnp.reshape(sigma2, spatial), var_smooth), (v,)
            )
        effect = beta @ c
        se = jnp.sqrt(jnp.clip(sigma2 * c_var, 1e-30, None))
        return effect / se

    def enhance(stat_v: Float[Array, 'V']) -> Float[Array, 'V']:
        stat_v = jnp.where(mask_flat, stat_v, 0.0)
        if enhancement == 'voxel':
            out = jnp.abs(stat_v) if two_sided else jnp.clip(stat_v, 0.0, None)
            return out
        enhanced = tfce(
            jnp.reshape(stat_v, spatial),
            E=tfce_E,
            H=tfce_H,
            n_steps=tfce_steps,
            connectivity=connectivity,
            two_sided=two_sided,
            mask=None if mask is None else mask,
        )
        return jnp.reshape(enhanced, (v,))

    # Relabellings (row 0 = identity).
    if exchange == 'sign':
        ops: Array = sign_flips(n, n_perm, key, blocks=blocks)
    else:
        ops = permutations(n, n_perm, key, blocks=blocks)

    def per_perm(
        op: Array,
    ) -> Tuple[Float[Array, 'V'], Float[Array, 'V'], Float[Array, '']]:
        if exchange == 'sign':
            Yp = base + e0 * op[None, :]
        else:
            Yp = base + e0[:, op]
        stat_p = statistic(Yp)
        enhanced_p = enhance(stat_p)
        return (
            stat_p,
            enhanced_p,
            jnp.max(jnp.where(mask_flat, enhanced_p, -jnp.inf)),
        )

    # The observed result IS permutation 0 (the identity).  We capture its
    # statistic / enhanced map from *inside* the scan (iteration 0), so the
    # reference used for the comparisons is bit-identical to the identity
    # permutation's own enhanced map -- guaranteeing the identity contributes a
    # ``+1`` everywhere and ``p_fwe >= 1 / n_perm``.  (TFCE is discretely
    # sensitive to float reassociation, so a separately-compiled observed path
    # would not match the scan body.)
    Carry = Tuple[
        Float[Array, 'V'],
        Float[Array, 'V'],
        Float[Array, 'V'],
        Float[Array, 'V'],
    ]

    def scan_body(
        carry: Carry, step: Tuple[Array, Array]
    ) -> Tuple[Carry, Float[Array, '']]:
        i, op = step
        fwe_count, unc_count, stat_obs, enhanced_obs = carry
        stat_p, enhanced_p, m = per_perm(op)
        is_first = i == 0
        stat_obs = jnp.where(is_first, stat_p, stat_obs)
        enhanced_obs = jnp.where(is_first, enhanced_p, enhanced_obs)
        # FWE: the spatial-max enhanced statistic vs the observed enhanced map.
        fwe_count = fwe_count + (m >= enhanced_obs)
        # Uncorrected: the *raw* per-voxel statistic vs the observed (FSL
        # convention) -- two-sided compares magnitudes.  Not the enhanced value.
        cstat_p = jnp.abs(stat_p) if two_sided else stat_p
        cstat_obs = jnp.abs(stat_obs) if two_sided else stat_obs
        unc_count = unc_count + (cstat_p >= cstat_obs)
        return (fwe_count, unc_count, stat_obs, enhanced_obs), m

    zeros = jnp.zeros((v,), Y.dtype)
    init: Carry = (zeros, zeros, zeros, zeros)
    (fwe_count, unc_count, stat_obs, enhanced_obs), null_max = lax.scan(
        scan_body, init, (jnp.arange(n_perm), ops)
    )

    p_fwe = jnp.where(mask_flat, fwe_count / n_perm, 1.0)
    p_unc = jnp.where(mask_flat, unc_count / n_perm, 1.0)
    # Zero the observed statistic at excluded voxels (out-of-mask or degenerate)
    # so the returned map carries no SE-floor artifact.
    stat_obs = jnp.where(mask_flat, stat_obs, 0.0)

    return PermResult(
        stat=jnp.reshape(stat_obs, spatial),
        enhanced=jnp.reshape(enhanced_obs, spatial),
        p_fwe=jnp.reshape(p_fwe, spatial),
        p_uncorrected=jnp.reshape(p_unc, spatial),
        null_max=null_max,
    )


def _smooth(x: Float[Array, '*spatial'], sigma: float) -> Float[Array, '*spatial']:
    from ...smoothing import gaussian

    return gaussian(x, sigma=sigma)
