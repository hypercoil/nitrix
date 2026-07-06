# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Zero-phase (ZCA) whitening as a findable fit/apply primitive.

Whitening maps zero-mean data to unit covariance.  The **symmetric** (ZCA,
zero-phase) whitening uses the *unique* symmetric map
:math:`W = \Sigma^{-1/2}`, so it rotates the data as little as possible --
unlike PCA whitening, which additionally rotates onto the eigenbasis (that
variant lives with its estimator, ``nimox.estimators.pca.PCA(whiten=True)``,
and is not duplicated here).

The inverse square root is computed **cuSOLVER-free** by the matmul-only
Newton-Schulz coupled iteration (:func:`nitrix.linalg.symsqrt` with
``driver='newton_schulz'``) for the full-sphering default, rather than the
eigendecomposition of the covariance: forming the covariance already squares
the condition number, and the eigen-path is the fragile solver route nitrix is
chartered to avoid.  Partial sphering (:math:`\Sigma^{-s/2}`, ``0 < s < 1``) is
a genuine fractional power and falls back to the eigenvalue map
(:func:`nitrix.linalg.sympower`).

The single-call :func:`whiten` is *defined as* ``whiten_apply(x,
whiten_fit(reference))`` so the fitted and one-shot paths cannot drift.  The
state is plain arrays (a :class:`WhiteningState`), so a stateful estimator (the
``nimox`` ``Whitening`` façade) re-uses this rather than re-deriving the
convention.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..linalg import sympower, symsqrt
from .covariance import cov

__all__ = [
    'WhiteningState',
    'whiten',
    'whiten_fit',
    'whiten_apply',
    'whiten_inverse_apply',
]


class WhiteningState(NamedTuple):
    """A fitted ZCA whitener: the learned mean and symmetric maps.

    Attributes
    ----------
    mean : Float[Array, '... d']
        The per-feature mean subtracted before whitening (zeros when the fit
        assumed centred data).
    matrix : Float[Array, '... d d']
        The symmetric whitening map :math:`\\Sigma^{-s/2}` applied on the right:
        ``whitened = (x - mean) @ matrix``.
    inverse_matrix : Float[Array, '... d d']
        The un-whitening map :math:`\\Sigma^{+s/2}`:
        ``x = whitened @ inverse_matrix + mean``.
    """

    mean: Float[Array, '... d']
    matrix: Float[Array, '... d d']
    inverse_matrix: Float[Array, '... d d']


def whiten_fit(
    x: Float[Array, '... n d'],
    *,
    sphering: float = 1.0,
    eps: float = 0.0,
    assume_centered: bool = False,
) -> WhiteningState:
    r"""Fit a ZCA whitener from reference data.

    Estimates the mean and covariance of ``x`` (observations along the
    second-to-last axis) and forms the symmetric whitening map
    :math:`\Sigma^{-s/2}` and its inverse.

    Parameters
    ----------
    x : Float[Array, '... n d']
        Reference data: ``n`` observations of ``d`` features (batched over any
        leading axes).
    sphering : float, optional
        Sphering exponent :math:`s`.  ``1`` (default) is full whitening
        (:math:`\Sigma^{-1/2}`, via the cuSOLVER-free Newton-Schulz driver);
        ``0`` leaves the data unscaled (identity map, centring only); ``0 < s <
        1`` is partial sphering :math:`\Sigma^{-s/2}` via the eigenvalue map.
        A static scalar (it selects the code path).
    eps : float, optional
        Ridge added to the covariance diagonal (Tikhonov / SPD floor) before
        the root.  The conditioner for the ``n < d`` / near-singular regime on
        the matmul-only path (which cannot truncate small eigenvalues); loud in
        that it shifts the spectrum uniformly.  Default ``0``.
    assume_centered : bool, optional
        If ``True``, treat ``x`` as already mean-zero: the stored ``mean`` is
        zero (the covariance estimate still centres internally).  Default
        ``False``.

    Returns
    -------
    WhiteningState
        The fitted mean and symmetric maps (plain arrays).
    """
    mean: Float[Array, '... d'] = jnp.mean(x, axis=-2)
    if assume_centered:
        mean = jnp.zeros_like(mean)
    sigma: Float[Array, '... d d'] = cov(x, rowvar=False, l2=eps)
    d = sigma.shape[-1]
    if sphering == 0.0:
        eye = jnp.broadcast_to(jnp.eye(d, dtype=sigma.dtype), sigma.shape)
        return WhiteningState(mean=mean, matrix=eye, inverse_matrix=eye)
    if sphering == 1.0:
        matrix = symsqrt(sigma, inverse=True, driver='newton_schulz')
        inverse_matrix = symsqrt(sigma, inverse=False, driver='newton_schulz')
    else:
        matrix = sympower(sigma, power=-sphering / 2.0)
        inverse_matrix = sympower(sigma, power=sphering / 2.0)
    return WhiteningState(
        mean=mean,
        matrix=matrix,
        inverse_matrix=inverse_matrix,
    )


def whiten_apply(
    x: Float[Array, '... n d'],
    state: WhiteningState,
) -> Float[Array, '... n d']:
    """Whiten ``x`` with a fitted state: ``(x - mean) @ matrix``.

    Parameters
    ----------
    x : Float[Array, '... n d']
        Data to whiten (the reference itself, or auxiliary data sharing the
        reference's feature basis).
    state : WhiteningState
        A state from :func:`whiten_fit`.

    Returns
    -------
    Float[Array, '... n d']
        The whitened data (unit covariance when ``x`` is the reference and
        ``sphering=1``).
    """
    out: Float[Array, '... n d'] = (
        x - state.mean[..., None, :]
    ) @ state.matrix
    return out


def whiten_inverse_apply(
    z: Float[Array, '... n d'],
    state: WhiteningState,
) -> Float[Array, '... n d']:
    """Un-whiten ``z`` with a fitted state: ``z @ inverse_matrix + mean``.

    The inverse of :func:`whiten_apply` (exact for ``sphering > 0``; the
    ``sphering=0`` identity map is trivially its own inverse).

    Parameters
    ----------
    z : Float[Array, '... n d']
        Whitened data.
    state : WhiteningState
        A state from :func:`whiten_fit`.

    Returns
    -------
    Float[Array, '... n d']
        The recovered data in the original coordinates.
    """
    out: Float[Array, '... n d'] = (
        z @ state.inverse_matrix + state.mean[..., None, :]
    )
    return out


def whiten(
    x: Float[Array, '... n d'],
    *,
    reference: Optional[Float[Array, '... n d']] = None,
    sphering: float = 1.0,
    eps: float = 0.0,
    assume_centered: bool = False,
) -> Float[Array, '... n d']:
    r"""Zero-phase (ZCA) whiten ``x``.

    The single-call convenience, defined as ``whiten_apply(x,
    whiten_fit(reference))`` so it cannot drift from the fitted path.

    Parameters
    ----------
    x : Float[Array, '... n d']
        Data to whiten.
    reference : Float[Array, '... n d'] or None, optional
        Data the whitener is fitted from.  ``None`` (default) self-whitens
        (``reference = x``).
    sphering : float, optional
        Sphering exponent (see :func:`whiten_fit`).  Default ``1`` (full).
    eps : float, optional
        Covariance ridge (see :func:`whiten_fit`).  Default ``0``.
    assume_centered : bool, optional
        Treat the reference as already mean-zero (see :func:`whiten_fit`).

    Returns
    -------
    Float[Array, '... n d']
        The whitened data.
    """
    ref = x if reference is None else reference
    state = whiten_fit(
        ref,
        sphering=sphering,
        eps=eps,
        assume_centered=assume_centered,
    )
    return whiten_apply(x, state)
