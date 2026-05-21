# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Empirical covariance / correlation / partial / conditional measures.

Differentiable, batched (n-D leading batch), JIT-safe.  Supports:

- Unweighted, scalar bias / unbias.
- **Vector weights** (per-observation; the diagonal of W).
- **Matrix weights** (full ``(obs, obs)`` coupling; useful for
  inter-temporal weighting, lagged covariance, etc.).
- Ridge regularisation via the ``l2`` knob (adds ``l2 I`` to the
  estimated covariance).

The green-field rewrite drops three warts from
``nitrix.functional.covariance``:

1. **The JIT trap.**  Legacy ``_is_diagonal(W)`` inspected the
   weight matrix's contents at trace time -- under JIT, this
   silently bypassed the nondiagonal-weight path and produced
   wrong results when the function was first traced with a
   diagonal weight then re-invoked with a non-diagonal one.
   The new version dispatches *only* on ``W.ndim`` (a shape
   property, JIT-stable) and treats the user-supplied matrix as
   given.
2. **The implicit ``_conform_bform_weight`` reshape.**  Removed.
   ``weights`` is required to broadcast against ``X`` in the
   standard way.
3. **The ``avg`` precomputation for matrix weights.**  Legacy
   computed ``avg = X @ (W / sum(W))`` which is the "right
   marginal of W" weighting -- but with a per-row sum, not a
   matrix product.  The new version computes the correct
   weighted mean ``mu[i] = sum_t marginal_weight_t * X[i, t] /
   sum(marginal_weight)`` explicitly.

Functions exposed:

- ``cov`` / ``corr`` -- unary covariance / correlation.
- ``pairedcov`` / ``pairedcorr`` -- bivariate (X, Y).
- ``partialcov`` / ``partialcorr`` -- conditioned on all other
  variables (inverse-covariance-derived).
- ``conditionalcov`` / ``conditionalcorr`` -- conditioned on
  an external set of variables (residualise + cov).
- ``precision`` -- inverse covariance (matrix inverse or
  Moore-Penrose pseudoinverse).
- Aliases: ``ccov``, ``ccorr``, ``pcorr``, ``corrcoef``.
"""
from __future__ import annotations

from typing import Any, Optional, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Num

from ..linalg.residual import residualise


__all__ = [
    'cov',
    'corr',
    'pairedcov',
    'pairedcorr',
    'partialcov',
    'partialcorr',
    'conditionalcov',
    'conditionalcorr',
    'precision',
    # aliases
    'ccov',
    'ccorr',
    'pcorr',
    'corrcoef',
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _orient_obs_last(X: Array, rowvar: bool) -> Array:
    '''Move the observation axis to the trailing position.'''
    X = jnp.atleast_2d(X)
    if not rowvar and X.shape[-2] != 1:
        X = X.swapaxes(-1, -2)
    return X


def _weighted_mean(
    X: Num[Array, '... c obs'],
    weights: Optional[Num[Array, '... obs']] = None,
    W: Optional[Num[Array, '... obs obs']] = None,
) -> Num[Array, '... c 1']:
    '''Per-channel weighted mean along the observation axis.

    At most one of ``weights`` / ``W`` is non-``None``.  ``W`` is
    a full matrix weight; the marginal observation weight is
    ``W.sum(-1)`` (right marginal; equals ``W.sum(-2)`` when ``W``
    is symmetric, which it should be for a valid covariance
    weight).
    '''
    if W is not None:
        marg = W.sum(axis=-1)
        return (X * marg[..., None, :]).sum(-1, keepdims=True) / marg.sum(
            axis=-1, keepdims=True
        )[..., None, :]
    if weights is not None:
        w_sum = weights.sum(-1, keepdims=True)
        return (X * weights[..., None, :]).sum(-1, keepdims=True) / w_sum[..., None, :]
    return X.mean(-1, keepdims=True)


def _denom_factor(
    n_obs: int,
    *,
    weights: Optional[Array] = None,
    W: Optional[Array] = None,
    ddof: int,
) -> Array:
    '''Bessel-style denominator for (un)biased covariance.

    For unweighted: ``n_obs - ddof``.
    For vector weights ``w``: ``sum(w) - ddof * sum(w**2) / sum(w)``
    (the "effective sample size" reduction for finite-sample
    weighting).
    For matrix weights: ``sum(W) - ddof * sum(W @ W.T) / sum(W)``
    (the natural generalisation; reduces to the vector case for
    diagonal W).
    '''
    if weights is None and W is None:
        return jnp.asarray(n_obs - ddof, dtype=jnp.float32)
    if weights is not None:
        w_sum = weights.sum(-1, keepdims=True)
        if ddof == 0:
            return w_sum
        return w_sum - ddof * (weights ** 2).sum(-1, keepdims=True) / w_sum
    # Matrix weight: by elimination ``W`` is the non-None argument here.
    assert W is not None
    w_sum = W.sum(axis=(-1, -2), keepdims=True)
    if ddof == 0:
        return w_sum[..., 0]
    wwT = (W @ W.swapaxes(-1, -2)).sum(axis=(-1, -2), keepdims=True)
    return (w_sum - ddof * wwT / w_sum)[..., 0]


def _cov_core(
    X: Num[Array, '... c obs'],
    Y: Num[Array, '... d obs'],
    *,
    weights: Optional[Num[Array, '... obs']],
    W: Optional[Num[Array, '... obs obs']],
    ddof: int,
    l2: float,
) -> Num[Array, '... c d']:
    '''Unbatched-shape covariance between channel sets X and Y.

    Caller is responsible for centring against the appropriate mean
    (we compute it here).  Handles unweighted, vector-weighted, and
    matrix-weighted cases via three explicit branches.
    '''
    mu_X = _weighted_mean(X, weights=weights, W=W)
    if Y is X:
        mu_Y = mu_X
    else:
        mu_Y = _weighted_mean(Y, weights=weights, W=W)

    Xc = X - mu_X
    Yc = (Y - mu_Y).conj()

    n_obs = X.shape[-1]
    fact = _denom_factor(n_obs, weights=weights, W=W, ddof=ddof)

    if W is not None:
        sigma = Xc @ W @ Yc.swapaxes(-1, -2) / fact[..., None, :]
    elif weights is not None:
        sigma = (Xc * weights[..., None, :]) @ Yc.swapaxes(-1, -2) / fact[..., None, :]
    else:
        sigma = Xc @ Yc.swapaxes(-1, -2) / fact

    if l2 > 0.0 and sigma.shape[-1] == sigma.shape[-2]:
        sigma = sigma + l2 * jnp.eye(sigma.shape[-1], dtype=sigma.dtype)
    return sigma


# ---------------------------------------------------------------------------
# Public API: cov / corr
# ---------------------------------------------------------------------------


def cov(
    X: Num[Array, '... c obs'],
    *,
    rowvar: bool = True,
    bias: bool = False,
    ddof: Optional[int] = None,
    weights: Optional[Num[Array, '... obs']] = None,
    weight_matrix: Optional[Num[Array, '... obs obs']] = None,
    l2: float = 0.0,
) -> Num[Array, '... c c']:
    '''Empirical covariance of variables in a tensor batch.

    Parameters
    ----------
    X
        Sample tensor.  Default layout ``(..., c, obs)`` with the
        last axis as observations and the penultimate axis as
        variables; pass ``rowvar=False`` to flip.
    rowvar
        ``True`` (default): observation axis is the *last* axis.
    bias
        If ``True``, divide by ``n``; otherwise (default) by
        ``n - 1`` (Bessel's correction).  Ignored if ``ddof`` is
        given.
    ddof
        Explicit degrees-of-freedom override.  ``None`` means
        ``int(not bias)``.
    weights
        Per-observation weights, ``(..., obs)``.  Mutually
        exclusive with ``weight_matrix``.
    weight_matrix
        Full observation coupling matrix, ``(..., obs, obs)``.
        Used for inter-temporal weighting (e.g. lagged or
        smoothed covariance).  Must be symmetric for the result
        to be a valid covariance.  Mutually exclusive with
        ``weights``.
    l2
        Ridge regularisation strength.  ``l2 > 0`` adds
        ``l2 I`` to the result.

    Returns
    -------
    Covariance matrix, ``(..., c, c)``.
    '''
    if weights is not None and weight_matrix is not None:
        raise ValueError(
            'cov: pass at most one of `weights` (per-observation) '
            'and `weight_matrix` (full).'
        )
    if ddof is None:
        ddof = int(not bias)
    X = _orient_obs_last(X, rowvar)
    return _cov_core(
        X, X, weights=weights, W=weight_matrix, ddof=ddof, l2=l2,
    )


def _corrnorm(A: Num[Array, '... c c']) -> Num[Array, '... c c']:
    '''Normalisation matrix for ``corr``: ``sqrt(diag) outer sqrt(diag)``.'''
    d = jnp.diagonal(A, axis1=-2, axis2=-1)
    fact = jnp.sqrt(d)[..., None]
    # ``jnp.diagonal`` resolves to Any; restore the array type.
    return cast(
        Num[Array, '... c c'],
        fact @ fact.swapaxes(-1, -2) + jnp.finfo(fact.dtype).eps,
    )


def corr(
    X: Num[Array, '... c obs'],
    **kwargs: Any,
) -> Num[Array, '... c c']:
    '''Pearson correlation of variables in a tensor batch.

    All arguments forwarded to ``cov``; the result is divided by
    the geometric mean of the diagonal entries.
    '''
    sigma = cov(X, **kwargs)
    return sigma / _corrnorm(sigma)


def pairedcov(
    X: Num[Array, '... c obs'],
    Y: Num[Array, '... d obs'],
    *,
    rowvar: bool = True,
    bias: bool = False,
    ddof: Optional[int] = None,
    weights: Optional[Num[Array, '... obs']] = None,
    weight_matrix: Optional[Num[Array, '... obs obs']] = None,
    l2: float = 0.0,
) -> Num[Array, '... c d']:
    '''Cross-covariance between two sets of variables.

    Same semantics as ``cov`` but with two separate variable
    blocks.  Returns ``(..., c, d)``.
    '''
    if weights is not None and weight_matrix is not None:
        raise ValueError(
            'pairedcov: pass at most one of `weights` and `weight_matrix`.'
        )
    if ddof is None:
        ddof = int(not bias)
    X = _orient_obs_last(X, rowvar)
    Y = _orient_obs_last(Y, rowvar)
    return _cov_core(
        X, Y, weights=weights, W=weight_matrix, ddof=ddof, l2=l2,
    )


def pairedcorr(
    X: Num[Array, '... c obs'],
    Y: Num[Array, '... d obs'],
    **kwargs: Any,
) -> Num[Array, '... c d']:
    '''Cross-correlation between two sets of variables.

    Normalised version of ``pairedcov``: each entry divided by
    the geometric mean of the corresponding diagonal entries of
    ``cov(X)`` and ``cov(Y)``.
    '''
    sigma_xy = pairedcov(X, Y, **kwargs)
    sigma_xx_diag = jnp.diagonal(cov(X, **kwargs), axis1=-2, axis2=-1)
    sigma_yy_diag = jnp.diagonal(cov(Y, **kwargs), axis1=-2, axis2=-1)
    norm = jnp.sqrt(sigma_xx_diag[..., :, None] * sigma_yy_diag[..., None, :])
    # ``jnp.diagonal`` feeds Any into ``norm``; restore the array type.
    return cast(
        Num[Array, '... c d'],
        sigma_xy / (norm + jnp.finfo(norm.dtype).eps),
    )


# ---------------------------------------------------------------------------
# precision / partial
# ---------------------------------------------------------------------------


def precision(
    X: Num[Array, '... c obs'],
    *,
    require_nonsingular: bool = True,
    **kwargs: Any,
) -> Num[Array, '... c c']:
    '''Inverse covariance (precision) matrix.

    Parameters
    ----------
    X
        See ``cov``.
    require_nonsingular
        ``True`` (default) -- use ``jnp.linalg.inv`` (fails on
        singular sigma).  ``False`` -- use the Moore-Penrose
        pseudoinverse.  Pass ``l2 > 0`` to ``kwargs`` to
        regularise rather than rely on the pseudoinverse for
        rank-deficient cases.
    '''
    sigma = cov(X, **kwargs)
    if require_nonsingular:
        # ``jnp.linalg.inv`` is typed as returning Any; restore.
        return cast(Num[Array, '... c c'], jnp.linalg.inv(sigma))
    return jnp.linalg.pinv(sigma)


def partialcov(
    X: Num[Array, '... c obs'],
    *,
    require_nonsingular: bool = True,
    **kwargs: Any,
) -> Num[Array, '... c c']:
    '''Partial covariance: conditioning each pair on all others.

    Computed from the precision matrix by negating off-diagonal
    entries.  Interpretation: direct (conditional on the rest)
    relationships between variable pairs.
    '''
    omega = precision(X, require_nonsingular=require_nonsingular, **kwargs)
    n = omega.shape[-1]
    sign = 2 * jnp.eye(n, dtype=omega.dtype) - 1
    return omega * sign


def partialcorr(
    X: Num[Array, '... c obs'],
    *,
    require_nonsingular: bool = True,
    **kwargs: Any,
) -> Num[Array, '... c c']:
    '''Partial correlation: normalised partial covariance.'''
    omega = partialcov(X, require_nonsingular=require_nonsingular, **kwargs)
    # For partial *correlation*, sign(diag) is sign(precision diag) = positive.
    # Normalise by geometric mean of |diag|.
    diag = jnp.abs(jnp.diagonal(omega, axis1=-2, axis2=-1))
    fact = jnp.sqrt(diag)[..., None]
    norm = fact @ fact.swapaxes(-1, -2) + jnp.finfo(fact.dtype).eps
    # ``jnp.diagonal`` feeds Any into ``omega``/``norm``; restore.
    return cast(Num[Array, '... c c'], omega / norm)


# ---------------------------------------------------------------------------
# Conditional cov / corr
# ---------------------------------------------------------------------------


def conditionalcov(
    X: Num[Array, '... c obs'],
    Y: Num[Array, '... d obs'],
    *,
    rowvar: bool = True,
    bias: bool = False,
    ddof: Optional[int] = None,
    weights: Optional[Num[Array, '... obs']] = None,
    weight_matrix: Optional[Num[Array, '... obs obs']] = None,
    l2: float = 0.0,
    residualise_l2: float = 0.0,
) -> Num[Array, '... c c']:
    '''Covariance of ``X`` conditioned on the variables in ``Y``.

    Operationally: residualise ``X`` against ``Y``, then take the
    covariance of the residuals.  This is the "explained out the
    Y subspace, look at what's left" view.

    Parameters
    ----------
    X, Y
        Variable / conditioning tensors.
    rowvar, bias, ddof, weights, weight_matrix, l2
        See ``cov``.
    residualise_l2
        Ridge for the residualisation step (passed to
        ``residualise``).  Typically left at ``0``; raise if
        ``Y`` is near-collinear.
    '''
    X = _orient_obs_last(X, rowvar)
    Y = _orient_obs_last(Y, rowvar)
    Xr = residualise(X, Y, l2=residualise_l2, rowvar=True)
    return cov(
        Xr, rowvar=True, bias=bias, ddof=ddof,
        weights=weights, weight_matrix=weight_matrix, l2=l2,
    )


def conditionalcorr(
    X: Num[Array, '... c obs'],
    Y: Num[Array, '... d obs'],
    **kwargs: Any,
) -> Num[Array, '... c c']:
    '''Correlation of ``X`` conditioned on the variables in ``Y``.

    Normalised version of ``conditionalcov``.
    '''
    sigma = conditionalcov(X, Y, **kwargs)
    return sigma / _corrnorm(sigma)


# ---------------------------------------------------------------------------
# Aliases (stats conventions)
# ---------------------------------------------------------------------------


def ccov(*args: Any, **kwargs: Any) -> Num[Array, '... c c']:
    '''Alias for ``conditionalcov``.'''
    return conditionalcov(*args, **kwargs)


def ccorr(*args: Any, **kwargs: Any) -> Num[Array, '... c c']:
    '''Alias for ``conditionalcorr``.'''
    return conditionalcorr(*args, **kwargs)


def pcorr(*args: Any, **kwargs: Any) -> Num[Array, '... c c']:
    '''Alias for ``partialcorr``.'''
    return partialcorr(*args, **kwargs)


def corrcoef(*args: Any, **kwargs: Any) -> Num[Array, '... c c']:
    '''Alias for ``corr`` (numpy convention).'''
    return corr(*args, **kwargs)
