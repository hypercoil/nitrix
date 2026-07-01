# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Random-effect covariance in log-Cholesky coordinates.

This is the shared parameterisation behind every block-Woodbury and Laplace
random-effect fit. The covariance is written as :math:`G = L L^{\\top}` with
:math:`L` lower-triangular, its diagonal entries exponentiated and its
off-diagonals free, so that :math:`G` stays positive-definite under an
unconstrained Newton step on the free parameters ``chol_params``. Setting
``diagonal=True`` keeps only the diagonal of :math:`L` (an uncorrelated
``(x || g)`` random effect); the full lower triangle gives the unstructured
``(1 + x | g)`` random effect.

This module is the single source of truth for the log-Cholesky mapping, shared by
the REML, correlated-effect, and Laplace / GLMM fitting routines.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = ['cov_re_from_chol']


def _tril_layout(r: int) -> Tuple[Tuple[int, int], ...]:
    """Row-major lower-triangular ``(i, j)`` index pairs of an ``r x r`` factor.

    Parameters
    ----------
    r : int
        Dimension of the square factor.

    Returns
    -------
    tuple of (int, int)
        The lower-triangular positions ``(i, j)`` with ``j <= i``, enumerated
        row by row. There are :math:`r(r + 1)/2` of them.
    """
    return tuple((i, j) for i in range(r) for j in range(i + 1))


def _param_layout(
    r: int, diagonal: bool = False
) -> Tuple[Tuple[int, int], ...]:
    """Free ``(i, j)`` positions of the Cholesky factor for :math:`G`.

    With ``diagonal=False`` the free positions are the full lower triangle
    (:math:`r(r + 1)/2` parameters), giving an unstructured :math:`r \\times r`
    within-group covariance (the ``(1 + x | g)`` random effect). With
    ``diagonal=True`` only the diagonal is free (:math:`r` parameters), giving an
    independent (diagonal-:math:`G`) random effect ``(x || g)`` in which
    intercept and slope share no covariance.

    Parameters
    ----------
    r : int
        Dimension of the square factor.
    diagonal : bool, optional
        If ``True``, return only the diagonal positions; otherwise return the
        full lower triangle. Default is ``False``.

    Returns
    -------
    tuple of (int, int)
        The free positions ``(i, j)`` of the Cholesky factor, in the order the
        free parameters are consumed.
    """
    if diagonal:
        return tuple((i, i) for i in range(r))
    return _tril_layout(r)


def _build_chol(
    chol_params: Float[Array, 'm'], r: int, diagonal: bool = False
) -> Float[Array, 'r r']:
    """Build the lower-triangular Cholesky factor :math:`L` from free parameters.

    Diagonal entries are exponentiated (hence positive) and off-diagonal entries
    are left free, so that :math:`G = L L^{\\top}` is positive-definite for any
    real ``chol_params``. The factor is assembled by scattering each free
    parameter into the position dictated by the static layout returned by
    :func:`_param_layout`: the full lower triangle when ``diagonal=False``, or
    just the diagonal when ``diagonal=True`` (yielding a diagonal :math:`G`, i.e.
    an uncorrelated ``(x || g)`` random effect).

    Parameters
    ----------
    chol_params : Float[Array, 'm']
        The ``m`` free Cholesky parameters, ordered to match
        :func:`_param_layout`. Here ``m`` is :math:`r(r + 1)/2` when
        ``diagonal=False`` and ``r`` when ``diagonal=True``.
    r : int
        Dimension of the square factor.
    diagonal : bool, optional
        If ``True``, populate only the diagonal of :math:`L`; otherwise populate
        the full lower triangle. Default is ``False``.

    Returns
    -------
    Float[Array, 'r r']
        The lower-triangular Cholesky factor :math:`L`.
    """
    L = jnp.zeros((r, r), dtype=chol_params.dtype)
    for k, (i, j) in enumerate(_param_layout(r, diagonal)):
        val = jnp.exp(chol_params[k]) if i == j else chol_params[k]
        L = L.at[i, j].set(val)
    return L


def cov_re_from_chol(
    chol_params: Float[Array, 'm'], r: int, diagonal: bool = False
) -> Float[Array, 'r r']:
    """Random-effect covariance :math:`G = L L^{\\top}` from Cholesky parameters.

    Reconstructs the lower-triangular factor :math:`L` from its free parameters
    (via :func:`_build_chol`) and forms the positive-definite random-effect
    covariance :math:`G = L L^{\\top}`. This is the single source of truth for the
    log-Cholesky mapping shared by the REML, correlated-effect, and Laplace / AGQ
    random-slope GLMM fitting routines.

    Parameters
    ----------
    chol_params : Float[Array, 'm']
        The ``m`` free Cholesky parameters. Here ``m`` is :math:`r(r + 1)/2` when
        ``diagonal=False`` and ``r`` when ``diagonal=True``.
    r : int
        Dimension of the random-effect covariance.
    diagonal : bool, optional
        If ``True``, use a diagonal Cholesky factor, yielding a diagonal
        (uncorrelated) covariance; otherwise use the full lower triangle,
        yielding an unstructured covariance. Default is ``False``.

    Returns
    -------
    Float[Array, 'r r']
        The symmetric positive-definite random-effect covariance :math:`G`.
    """
    L = _build_chol(chol_params, r, diagonal)
    return L @ L.T
