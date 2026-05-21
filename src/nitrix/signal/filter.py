# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Linear filtering: polynomial detrend.

Polynomial detrending: fit a polynomial of degree ``d`` to a time
series and subtract the fit.  Used for low-frequency drift removal
in fMRI BOLD signals (the classic linear, quadratic, or cubic
detrend depending on scan duration).

Implementation: build a polynomial regressor basis and call
``nitrix.linalg.residualise`` with it.  The basis is
``[1, t, t^2, ..., t^d]`` for time indices ``t in [0, n_obs)``
rescaled to ``[-1, 1]`` for numerical stability of the
Vandermonde-like matrix (avoid powers of large integers).

What changed from the legacy ``entense.confound_regression_p``:

- ``residualise`` is now the workhorse; the polynomial part is
  just a basis-construction step.
- Stability: the time index is rescaled to ``[-1, 1]`` before
  taking powers.  For ``n_obs = 500`` and ``degree = 3`` this
  changes ``500^3 ~ 1.25e8`` into ``1`` -- huge stability win
  for the QR / SVD fit, although in practice it only matters
  for ``degree > 5`` or so.

Frequency-domain filtering (bandpass / lowpass / highpass) lives
in ``nitrix.stats.fourier`` (``product_filter``); IIR filters
are not currently shipped.
"""
from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Num

from ..linalg.residual import residualise


__all__ = ['polynomial_detrend']


def _polynomial_basis(
    n_obs: int,
    degree: int,
    dtype: DTypeLike,
) -> Float[Array, 'degree+1 obs']:
    '''Build a polynomial basis ``[1, t, t^2, ..., t^d]`` along
    rescaled time ``t in [-1, 1]``.

    Rescaling is essential for numerical stability when ``degree``
    is moderately large; integer powers of ``[0, n_obs)`` blow up
    fast and ill-condition the regression matrix.
    '''
    if n_obs < 2:
        # Edge case: too few samples for any polynomial.
        return jnp.ones((degree + 1, n_obs), dtype=dtype)
    t = jnp.linspace(-1.0, 1.0, n_obs, dtype=dtype)
    return jnp.stack([t ** k for k in range(degree + 1)], axis=0)


def polynomial_detrend(
    X: Num[Array, '... obs'],
    *,
    degree: int = 1,
    rowvar: bool = True,
) -> Num[Array, '... obs']:
    '''Subtract a polynomial fit of the named ``degree`` from each
    observation channel.

    Equivalent to ``residualise(X, polynomial_basis)`` where the
    basis is the rescaled Vandermonde matrix
    ``[1, t, t^2, ..., t^degree]``.

    Parameters
    ----------
    X
        Time series, observation axis is last (``rowvar=True``,
        default).
    degree
        Polynomial degree.  ``0`` -- demean only.  ``1`` --
        linear detrend.  ``2`` -- quadratic.  ``3`` -- cubic.
    rowvar
        ``True`` (default): observation axis is the *last* axis.

    Returns
    -------
    Detrended time series, same shape as ``X``.

    Notes
    -----
    Differentiable through ``X``; ``degree`` is static.
    '''
    if degree < 0:
        raise ValueError(f'degree must be >= 0; got {degree}.')
    # Determine n_obs from X (last axis if rowvar, else penultimate).
    n_obs = X.shape[-1] if rowvar else X.shape[-2]
    basis = _polynomial_basis(n_obs, degree, X.dtype)
    # Broadcast basis to share leading dims with X.
    while basis.ndim < X.ndim:
        basis = basis[None, ...]
    if not rowvar:
        basis = basis.swapaxes(-1, -2)
    return residualise(X, basis, rowvar=rowvar, method='cholesky')
