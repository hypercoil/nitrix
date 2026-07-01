# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Empirical covariance / correlation / partial / conditional measures.

Differentiable, batched (n-D leading batch), JIT-safe.  Supports:

- Unweighted, scalar bias / unbias.
- **Vector weights** (per-observation; the diagonal of :math:`W`).
- **Matrix weights** (full ``(obs, obs)`` coupling; useful for
  inter-temporal weighting, lagged covariance, etc.).
- Ridge regularisation via the ``l2`` knob (adds :math:`\\lambda I`
  to the estimated covariance).

The weighted mean underlying the matrix-weight path computes
:math:`\\mu_i = \\sum_t m_t X_{it} / \\sum_t m_t`, where the
per-observation marginal weight :math:`m_t` is the row sum of the
supplied weight matrix.

Functions exposed:

- :func:`cov` / :func:`corr` -- unary covariance / correlation.
- :func:`pairedcov` / :func:`pairedcorr` -- bivariate (X, Y).
- :func:`partialcov` / :func:`partialcorr` -- conditioned on all
  other variables (inverse-covariance-derived).
- :func:`conditionalcov` / :func:`conditionalcorr` -- conditioned on
  an external set of variables (residualise + cov).
- :func:`precision` -- inverse covariance (matrix inverse or
  Moore-Penrose pseudoinverse).
- Aliases: :func:`ccov`, :func:`ccorr`, :func:`pcorr`, :func:`corrcoef`.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Num

from ..linalg._solver import safe_eigh, safe_inv
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
    """Move the observation axis to the trailing position.

    Promotes ``X`` to at least two dimensions, then transposes the
    final two axes when the observations lie along the penultimate
    axis, so that downstream reductions can assume observations are
    last.

    Parameters
    ----------
    X : Array
        Sample tensor.  Arrays of fewer than two dimensions are
        promoted via :func:`jax.numpy.atleast_2d`.
    rowvar : bool
        If ``True``, observations are already along the last axis and
        the array is returned unchanged.  If ``False`` (and the array
        is not a single row), the last two axes are swapped so that
        observations become the trailing axis.

    Returns
    -------
    Array
        The input with the observation axis in the trailing position.
    """
    X = jnp.atleast_2d(X)
    if not rowvar and X.shape[-2] != 1:
        X = X.swapaxes(-1, -2)
    return X


def _weighted_mean(
    X: Num[Array, '... c obs'],
    weights: Optional[Num[Array, '... obs']] = None,
    W: Optional[Num[Array, '... obs obs']] = None,
) -> Num[Array, '... c 1']:
    """Per-channel weighted mean along the observation axis.

    At most one of ``weights`` / ``W`` is non-``None``.  With no
    weighting, this is the plain mean.  With a matrix weight ``W`` the
    per-observation weight is the right marginal :math:`W_t = \\sum_j
    W_{tj}` (equal to the left marginal when ``W`` is symmetric, as it
    should be for a valid covariance weight).

    Parameters
    ----------
    X : Num[Array, '... c obs']
        Sample tensor with variables/channels on the penultimate axis
        and observations on the trailing axis.
    weights : Num[Array, '... obs'], optional
        Per-observation vector weights.  Mutually exclusive with
        ``W``.
    W : Num[Array, '... obs obs'], optional
        Full observation-coupling weight matrix; the marginal
        observation weight is its row sum.  Mutually exclusive with
        ``weights``.

    Returns
    -------
    Num[Array, '... c 1']
        The per-channel weighted mean, retaining the observation axis
        as a singleton so it broadcasts back against ``X``.
    """
    if W is not None:
        marg = W.sum(axis=-1)
        return (X * marg[..., None, :]).sum(-1, keepdims=True) / marg.sum(
            axis=-1, keepdims=True
        )[..., None, :]
    if weights is not None:
        w_sum = weights.sum(-1, keepdims=True)
        return (X * weights[..., None, :]).sum(-1, keepdims=True) / w_sum[
            ..., None, :
        ]
    return X.mean(-1, keepdims=True)


def _denom_factor(
    n_obs: int,
    *,
    weights: Optional[Array] = None,
    W: Optional[Array] = None,
    ddof: int,
    dtype: Any,
) -> Array:
    """Bessel-style denominator for (un)biased covariance.

    The three weighting regimes each use the natural generalisation of
    the degrees-of-freedom correction:

    - Unweighted: :math:`n_{obs} - \\mathrm{ddof}`.
    - Vector weights :math:`w`: :math:`\\sum_t w_t - \\mathrm{ddof}
      \\, \\sum_t w_t^2 / \\sum_t w_t` (the effective-sample-size
      reduction for finite-sample weighting).
    - Matrix weights :math:`W`: :math:`\\sum W - \\mathrm{ddof} \\,
      \\sum (W W^{\\top}) / \\sum W`, which reduces to the vector case
      for diagonal :math:`W`.

    Parameters
    ----------
    n_obs : int
        Number of observations along the observation axis.
    weights : Array, optional
        Per-observation vector weights.  Mutually exclusive with
        ``W``.
    W : Array, optional
        Full observation-coupling weight matrix.  Mutually exclusive
        with ``weights``.
    ddof : int
        Degrees-of-freedom correction; ``0`` gives the biased
        denominator, ``1`` Bessel's correction.
    dtype : Any
        Data dtype adopted by the unweighted count so that an x64
        covariance denominator is not silently downcast; the weighted
        branches already inherit the weights' dtype.

    Returns
    -------
    Array
        The scalar (or batched) denominator by which the
        (cross-)products are divided.
    """
    if weights is None and W is None:
        return jnp.asarray(n_obs - ddof, dtype=dtype)
    if weights is not None:
        w_sum = weights.sum(-1, keepdims=True)
        if ddof == 0:
            return w_sum
        return w_sum - ddof * (weights**2).sum(-1, keepdims=True) / w_sum
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
    """Covariance between two channel sets after mean-centring.

    Centres both inputs against their (optionally weighted) mean, then
    forms the (cross-)product :math:`X_c W Y_c^{\\top}` divided by the
    appropriate denominator.  Handles unweighted, vector-weighted, and
    matrix-weighted cases via three explicit branches, and optionally
    adds a ridge term when the result is square.

    A host-side check warns when a supplied matrix weight is not
    symmetric, since the weighted mean uses the right marginal and an
    asymmetric weight then yields an invalid covariance weight; this
    check is skipped under tracing.

    Parameters
    ----------
    X : Num[Array, '... c obs']
        First channel set, observations on the trailing axis.
    Y : Num[Array, '... d obs']
        Second channel set, observations on the trailing axis.  May be
        the same array object as ``X``, in which case its mean is
        reused.
    weights : Num[Array, '... obs'], optional
        Per-observation vector weights.  Mutually exclusive with
        ``W``.
    W : Num[Array, '... obs obs'], optional
        Full observation-coupling weight matrix.  Mutually exclusive
        with ``weights``.
    ddof : int
        Degrees-of-freedom correction applied in the denominator.
    l2 : float
        Ridge regularisation strength; when positive and the result is
        square, adds :math:`\\lambda I`.

    Returns
    -------
    Num[Array, '... c d']
        The (cross-)covariance matrix between the channels of ``X``
        and ``Y``.
    """
    # Round 4: the matrix-weight path uses the right marginal W.sum(-1) as the
    # observation weight, which is only a valid covariance weight when W is
    # symmetric. Flag an asymmetric W host-side (skip under jit).
    if W is not None and not isinstance(W, jax.core.Tracer):
        w_host = np.asarray(W)
        if not np.allclose(w_host, np.swapaxes(w_host, -1, -2)):
            warnings.warn(
                'cov/corr: the matrix weight W is not symmetric; the weighted '
                'mean uses the right marginal W.sum(-1), so an asymmetric W '
                'yields an asymmetric, generally invalid covariance weight. '
                'Pass a symmetric (ideally PSD) W.',
                stacklevel=3,
            )
    mu_X = _weighted_mean(X, weights=weights, W=W)
    if Y is X:
        mu_Y = mu_X
    else:
        mu_Y = _weighted_mean(Y, weights=weights, W=W)

    Xc = X - mu_X
    Yc = (Y - mu_Y).conj()

    n_obs = X.shape[-1]
    fact = _denom_factor(
        n_obs, weights=weights, W=W, ddof=ddof, dtype=Xc.dtype
    )

    if W is not None:
        sigma = Xc @ W @ Yc.swapaxes(-1, -2) / fact[..., None, :]
    elif weights is not None:
        sigma = (
            (Xc * weights[..., None, :])
            @ Yc.swapaxes(-1, -2)
            / fact[..., None, :]
        )
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
    """Empirical covariance of variables in a tensor batch.

    Parameters
    ----------
    X
        Sample tensor.  Default layout ``(..., c, obs)`` with the
        last axis as observations and the penultimate axis as
        variables; pass ``rowvar=False`` to flip.
    rowvar
        ``True`` (default): observation axis is the *last* axis.
    bias
        If ``True``, divide by :math:`n`; otherwise (default) by
        :math:`n - 1` (Bessel's correction).  Ignored if ``ddof`` is
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
        :math:`\\lambda I` to the result.

    Returns
    -------
    Num[Array, '... c c']
        Covariance matrix over the ``c`` variables, ``(..., c, c)``.
    """
    if weights is not None and weight_matrix is not None:
        raise ValueError(
            'cov: pass at most one of `weights` (per-observation) '
            'and `weight_matrix` (full).'
        )
    if ddof is None:
        ddof = int(not bias)
    X = _orient_obs_last(X, rowvar)
    return _cov_core(
        X,
        X,
        weights=weights,
        W=weight_matrix,
        ddof=ddof,
        l2=l2,
    )


def _corrnorm(A: Num[Array, '... c c']) -> Num[Array, '... c c']:
    """Normalisation matrix for :func:`corr`.

    Forms the outer product of the elementwise square roots of the
    diagonal, :math:`\\sqrt{\\operatorname{diag} A} \\,
    \\sqrt{\\operatorname{diag} A}^{\\top}`, so that dividing a
    covariance by this matrix yields correlations.  A machine-epsilon
    term is added to guard against division by zero.

    Parameters
    ----------
    A : Num[Array, '... c c']
        A (covariance) matrix whose diagonal supplies the per-variable
        scales.

    Returns
    -------
    Num[Array, '... c c']
        The normalisation matrix, ``(..., c, c)``.
    """
    d = jnp.diagonal(A, axis1=-2, axis2=-1)
    fact = jnp.sqrt(d)[..., None]
    return fact @ fact.swapaxes(-1, -2) + jnp.finfo(fact.dtype).eps


def corr(
    X: Num[Array, '... c obs'],
    **kwargs: Any,
) -> Num[Array, '... c c']:
    """Pearson correlation of variables in a tensor batch.

    Computes the covariance via :func:`cov` and normalises it by the
    outer product of the per-variable standard deviations, yielding
    correlations with unit diagonal.

    Parameters
    ----------
    X : Num[Array, '... c obs']
        Sample tensor; see :func:`cov`.
    **kwargs
        Additional keyword arguments forwarded to :func:`cov`
        (``rowvar``, ``bias``, ``ddof``, ``weights``,
        ``weight_matrix``, ``l2``).

    Returns
    -------
    Num[Array, '... c c']
        Correlation matrix over the ``c`` variables, ``(..., c, c)``.
    """
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
    """Cross-covariance between two sets of variables.

    Same semantics as :func:`cov` but with two separate variable
    blocks ``X`` and ``Y`` sharing the same observation axis.  The
    result is the rectangular block of covariances between every
    variable of ``X`` and every variable of ``Y``.

    Parameters
    ----------
    X : Num[Array, '... c obs']
        First variable block; observations along the observation axis.
    Y : Num[Array, '... d obs']
        Second variable block, sharing the observation axis with
        ``X``.
    rowvar : bool
        If ``True`` (default), the observation axis is the last axis;
        pass ``False`` to flip.
    bias : bool
        If ``True``, divide by :math:`n`; otherwise by :math:`n - 1`.
        Ignored if ``ddof`` is given.
    ddof : int, optional
        Explicit degrees-of-freedom override.  ``None`` means
        ``int(not bias)``.
    weights : Num[Array, '... obs'], optional
        Per-observation vector weights.  Mutually exclusive with
        ``weight_matrix``.
    weight_matrix : Num[Array, '... obs obs'], optional
        Full observation-coupling matrix.  Must be symmetric for the
        result to be a valid covariance.  Mutually exclusive with
        ``weights``.
    l2 : float
        Ridge regularisation strength; ``l2 > 0`` adds
        :math:`\\lambda I` when the result is square.

    Returns
    -------
    Num[Array, '... c d']
        Cross-covariance block, ``(..., c, d)``.
    """
    if weights is not None and weight_matrix is not None:
        raise ValueError(
            'pairedcov: pass at most one of `weights` and `weight_matrix`.'
        )
    if ddof is None:
        ddof = int(not bias)
    X = _orient_obs_last(X, rowvar)
    Y = _orient_obs_last(Y, rowvar)
    return _cov_core(
        X,
        Y,
        weights=weights,
        W=weight_matrix,
        ddof=ddof,
        l2=l2,
    )


def pairedcorr(
    X: Num[Array, '... c obs'],
    Y: Num[Array, '... d obs'],
    **kwargs: Any,
) -> Num[Array, '... c d']:
    """Cross-correlation between two sets of variables.

    Normalised version of :func:`pairedcov`: each cross-covariance
    entry is divided by the square root of the product of the
    corresponding variances, i.e. the diagonal entries of
    :func:`cov` applied to ``X`` and to ``Y`` respectively.

    Parameters
    ----------
    X : Num[Array, '... c obs']
        First variable block; see :func:`pairedcov`.
    Y : Num[Array, '... d obs']
        Second variable block; see :func:`pairedcov`.
    **kwargs
        Additional keyword arguments forwarded to :func:`pairedcov`
        and :func:`cov` (``rowvar``, ``bias``, ``ddof``, ``weights``,
        ``weight_matrix``, ``l2``).

    Returns
    -------
    Num[Array, '... c d']
        Cross-correlation block, ``(..., c, d)``.
    """
    sigma_xy = pairedcov(X, Y, **kwargs)
    sigma_xx_diag = jnp.diagonal(cov(X, **kwargs), axis1=-2, axis2=-1)
    sigma_yy_diag = jnp.diagonal(cov(Y, **kwargs), axis1=-2, axis2=-1)
    norm = jnp.sqrt(sigma_xx_diag[..., :, None] * sigma_yy_diag[..., None, :])
    return sigma_xy / (norm + jnp.finfo(norm.dtype).eps)


# ---------------------------------------------------------------------------
# precision / partial
# ---------------------------------------------------------------------------


def precision(
    X: Num[Array, '... c obs'],
    *,
    require_nonsingular: bool = True,
    **kwargs: Any,
) -> Num[Array, '... c c']:
    """Inverse covariance (precision) matrix.

    Estimates the covariance via :func:`cov`, then inverts it.  Both
    inversion routes go through the robust ``safe_*`` linear-algebra
    wrappers (with an adaptive CPU fallback on GPU stacks where the
    vendor solver is broken) rather than a raw ``jnp.linalg.inv`` /
    ``jnp.linalg.pinv``.  For the per-element / sparse-precision route
    see :func:`~nitrix.stats.connectivity.glasso`; this function is
    the dense, once-per-matrix estimator.

    Parameters
    ----------
    X : Num[Array, '... c obs']
        Sample tensor; see :func:`cov`.
    require_nonsingular : bool
        If ``True`` (default), use a dense matrix inverse, which fails
        on a singular covariance.  If ``False``, use the symmetric
        eigen-truncated Moore-Penrose pseudoinverse.  Pass ``l2 > 0``
        via ``kwargs`` to regularise rather than rely on the
        pseudoinverse for rank-deficient cases.
    **kwargs
        Additional keyword arguments forwarded to :func:`cov`
        (``rowvar``, ``bias``, ``ddof``, ``weights``,
        ``weight_matrix``, ``l2``).

    Returns
    -------
    Num[Array, '... c c']
        Precision (inverse covariance) matrix, ``(..., c, c)``.
    """
    sigma = cov(X, **kwargs)
    if require_nonsingular:
        return safe_inv(sigma)
    return _sym_pinv(sigma)


def _sym_pinv(sigma: Num[Array, '... c c']) -> Num[Array, '... c c']:
    """Symmetric Moore-Penrose pseudoinverse via a robust ``safe_eigh``.

    Because ``sigma`` is a symmetric covariance, the SVD-based
    ``jnp.linalg.pinv`` is replaced by an eigendecomposition: tiny
    eigenvalues below the standard relative cutoff
    :math:`c \\cdot \\varepsilon \\cdot \\max_i |w_i|` are treated as
    zero and their reciprocals set to zero before reconstruction.

    Parameters
    ----------
    sigma : Num[Array, '... c c']
        A symmetric (covariance) matrix to pseudo-invert.

    Returns
    -------
    Num[Array, '... c c']
        The symmetric Moore-Penrose pseudoinverse, ``(..., c, c)``.
    """
    w, v = safe_eigh(sigma)
    cutoff = (
        sigma.shape[-1]
        * jnp.finfo(sigma.dtype).eps
        * jnp.max(jnp.abs(w), axis=-1, keepdims=True)
    )
    w_inv = jnp.where(jnp.abs(w) > cutoff, 1.0 / w, 0.0)
    return cast(
        Num[Array, '... c c'],
        (v * w_inv[..., None, :]) @ jnp.swapaxes(v, -1, -2),
    )


def partialcov(
    X: Num[Array, '... c obs'],
    *,
    require_nonsingular: bool = True,
    **kwargs: Any,
) -> Num[Array, '... c c']:
    """Partial covariance: conditioning each pair on all others.

    Computed from the precision matrix (see :func:`precision`) by
    negating its off-diagonal entries while leaving the diagonal
    unchanged.  Each off-diagonal entry then expresses the direct
    relationship between a pair of variables after conditioning on all
    the remaining variables.

    Parameters
    ----------
    X : Num[Array, '... c obs']
        Sample tensor; see :func:`cov`.
    require_nonsingular : bool
        Forwarded to :func:`precision`: use a dense inverse
        (``True``, default) or the eigen-truncated pseudoinverse
        (``False``).
    **kwargs
        Additional keyword arguments forwarded to :func:`precision`
        and :func:`cov`.

    Returns
    -------
    Num[Array, '... c c']
        Partial covariance matrix, ``(..., c, c)``.
    """
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
    """Partial correlation: normalised partial covariance.

    Normalises the partial covariance (see :func:`partialcov`) by the
    outer product of the square roots of the absolute diagonal
    entries, giving the correlation between each pair of variables
    after conditioning on all the others.

    Parameters
    ----------
    X : Num[Array, '... c obs']
        Sample tensor; see :func:`cov`.
    require_nonsingular : bool
        Forwarded to :func:`precision` via :func:`partialcov`.
    **kwargs
        Additional keyword arguments forwarded to :func:`partialcov`.

    Returns
    -------
    Num[Array, '... c c']
        Partial correlation matrix, ``(..., c, c)``.
    """
    omega = partialcov(X, require_nonsingular=require_nonsingular, **kwargs)
    # For partial *correlation*, sign(diag) is sign(precision diag) = positive.
    # Normalise by geometric mean of |diag|.
    diag = jnp.abs(jnp.diagonal(omega, axis1=-2, axis2=-1))
    fact = jnp.sqrt(diag)[..., None]
    norm = fact @ fact.swapaxes(-1, -2) + jnp.finfo(fact.dtype).eps
    return omega / norm


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
    """Covariance of ``X`` conditioned on the variables in ``Y``.

    Operationally, ``X`` is residualised against ``Y`` (see
    :func:`~nitrix.linalg.residual.residualise`) and the covariance
    of the residuals is returned.  This is the "explain away the
    ``Y`` subspace, then look at what remains" view of conditioning.

    Parameters
    ----------
    X : Num[Array, '... c obs']
        Variables whose conditional covariance is sought.
    Y : Num[Array, '... d obs']
        Conditioning variables to residualise out of ``X``.
    rowvar : bool
        If ``True`` (default), the observation axis is the last axis;
        see :func:`cov`.
    bias : bool
        Bias flag; see :func:`cov`.
    ddof : int, optional
        Degrees-of-freedom override; see :func:`cov`.
    weights : Num[Array, '... obs'], optional
        Per-observation vector weights; see :func:`cov`.
    weight_matrix : Num[Array, '... obs obs'], optional
        Full observation-coupling matrix; see :func:`cov`.
    l2 : float
        Ridge regularisation strength for the covariance step; see
        :func:`cov`.
    residualise_l2 : float
        Ridge for the residualisation step, passed to
        :func:`~nitrix.linalg.residual.residualise`.  Typically left
        at ``0``; raise it if ``Y`` is near-collinear.

    Returns
    -------
    Num[Array, '... c c']
        Conditional covariance matrix over the variables of ``X``,
        ``(..., c, c)``.
    """
    X = _orient_obs_last(X, rowvar)
    Y = _orient_obs_last(Y, rowvar)
    Xr = residualise(X, Y, l2=residualise_l2, rowvar=True)
    return cov(
        Xr,
        rowvar=True,
        bias=bias,
        ddof=ddof,
        weights=weights,
        weight_matrix=weight_matrix,
        l2=l2,
    )


def conditionalcorr(
    X: Num[Array, '... c obs'],
    Y: Num[Array, '... d obs'],
    **kwargs: Any,
) -> Num[Array, '... c c']:
    """Correlation of ``X`` conditioned on the variables in ``Y``.

    Normalised version of :func:`conditionalcov`: the conditional
    covariance is divided by the outer product of the residual
    standard deviations, yielding conditional correlations with unit
    diagonal.

    Parameters
    ----------
    X : Num[Array, '... c obs']
        Variables whose conditional correlation is sought.
    Y : Num[Array, '... d obs']
        Conditioning variables to residualise out of ``X``.
    **kwargs
        Additional keyword arguments forwarded to
        :func:`conditionalcov` (``rowvar``, ``bias``, ``ddof``,
        ``weights``, ``weight_matrix``, ``l2``, ``residualise_l2``).

    Returns
    -------
    Num[Array, '... c c']
        Conditional correlation matrix, ``(..., c, c)``.
    """
    sigma = conditionalcov(X, Y, **kwargs)
    return sigma / _corrnorm(sigma)


# ---------------------------------------------------------------------------
# Aliases (stats conventions)
# ---------------------------------------------------------------------------


def ccov(*args: Any, **kwargs: Any) -> Num[Array, '... c c']:
    """Alias for :func:`conditionalcov`.

    Parameters
    ----------
    *args, **kwargs
        Forwarded verbatim to :func:`conditionalcov`.

    Returns
    -------
    Num[Array, '... c c']
        Conditional covariance matrix, ``(..., c, c)``.
    """
    return conditionalcov(*args, **kwargs)


def ccorr(*args: Any, **kwargs: Any) -> Num[Array, '... c c']:
    """Alias for :func:`conditionalcorr`.

    Parameters
    ----------
    *args, **kwargs
        Forwarded verbatim to :func:`conditionalcorr`.

    Returns
    -------
    Num[Array, '... c c']
        Conditional correlation matrix, ``(..., c, c)``.
    """
    return conditionalcorr(*args, **kwargs)


def pcorr(*args: Any, **kwargs: Any) -> Num[Array, '... c c']:
    """Alias for :func:`partialcorr`.

    Parameters
    ----------
    *args, **kwargs
        Forwarded verbatim to :func:`partialcorr`.

    Returns
    -------
    Num[Array, '... c c']
        Partial correlation matrix, ``(..., c, c)``.
    """
    return partialcorr(*args, **kwargs)


def corrcoef(*args: Any, **kwargs: Any) -> Num[Array, '... c c']:
    """Alias for :func:`corr` (numpy convention).

    Parameters
    ----------
    *args, **kwargs
        Forwarded verbatim to :func:`corr`.

    Returns
    -------
    Num[Array, '... c c']
        Correlation matrix, ``(..., c, c)``.
    """
    return corr(*args, **kwargs)
