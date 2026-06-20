# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Random-effect covariance in **log-Cholesky** coordinates -- the shared
parameterisation behind every block-Woodbury / Laplace random-effect fit.

``G = L L^T`` with ``L`` lower-triangular, its diagonal entries exponentiated and
its off-diagonals free, so ``G`` stays positive-definite under an *unconstrained*
Newton step on the free parameters ``chol_params``.  ``diagonal=True`` keeps only
the diagonal of ``L`` (an uncorrelated ``(x || g)`` random effect); the full lower
triangle is the unstructured ``(1 + x | g)``.

Lifted out of ``_blockwoodbury.py`` (audit D7) so ``reml.py`` / ``_corrfit.py`` /
``glmm.py`` share **one** source of truth instead of reaching across module
boundaries for these private names.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = ['cov_re_from_chol']


def _tril_layout(r: int) -> Tuple[Tuple[int, int], ...]:
    """Row-major lower-triangular ``(i, j)`` index pairs of an ``r x r`` factor."""
    return tuple((i, j) for i in range(r) for j in range(i + 1))


def _param_layout(
    r: int, diagonal: bool = False
) -> Tuple[Tuple[int, int], ...]:
    """Free ``(i, j)`` positions of the Cholesky factor for ``G``.

    ``diagonal=False`` -- the full lower triangle (``r(r+1)/2`` params): an
    **unstructured** ``r x r`` within-group covariance ``(1 + x | g)``.
    ``diagonal=True`` -- only the diagonal (``r`` params): an **independent**
    (diagonal-``G``) random effect ``(x || g)``, where intercept and slope share
    no covariance.  Both are tier-R2 (one grouping factor, block-diagonal ``V``).
    """
    if diagonal:
        return tuple((i, i) for i in range(r))
    return _tril_layout(r)


def _build_chol(
    chol_params: Float[Array, 'm'], r: int, diagonal: bool = False
) -> Float[Array, 'r r']:
    """Lower-triangular Cholesky factor ``L`` from its free parameters.

    Diagonal entries are exponentiated (positive), off-diagonal entries are
    free -- so ``G = L L^T`` is positive-definite for any real ``chol_params``.
    The loop is over the static layout (unrolled; ``r`` is tiny): the full lower
    triangle (``diagonal=False``) or just the diagonal (``diagonal=True`` -> a
    diagonal ``G``, i.e. an uncorrelated ``(x || g)`` random effect).
    """
    L = jnp.zeros((r, r), dtype=chol_params.dtype)
    for k, (i, j) in enumerate(_param_layout(r, diagonal)):
        val = jnp.exp(chol_params[k]) if i == j else chol_params[k]
        L = L.at[i, j].set(val)
    return L


def cov_re_from_chol(
    chol_params: Float[Array, 'm'], r: int, diagonal: bool = False
) -> Float[Array, 'r r']:
    """Random-effect covariance ``G = L L^T`` from the free Cholesky parameters.

    The single source of truth shared by the R2 (``reml.lme_fit``), R2+corr
    (``_corrfit.fit_corr_lme``), and the Laplace / AGQ random-slope GLMM fits --
    each previously reached into ``_blockwoodbury`` for it.
    """
    L = _build_chol(chol_params, r, diagonal)
    return L @ L.T
