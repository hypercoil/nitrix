# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Randomized (sketch) low-rank SVD.

:func:`randomized_svd` recovers the top-``k`` singular triplets of a (possibly
huge) :math:`(m, n)` matrix in :math:`O(m n (k + p))` matmuls -- the Halko,
Martinsson & Tropp range finder with subspace (power) iteration -- instead of
the :math:`O(m n \min(m, n))` dense SVD.  It is the decomposition primitive the
small-sample regime wants: :math:`m` (voxels) is 100k--500k but only the leading
:math:`k` modes are needed (principal-component / aCompCor confound bases,
NORDIC-style singular value denoising, spatial eigenimages, the top-``k``
eigensolve consumer).

cuSOLVER-free
-------------

The construction is entirely matmuls plus solver calls on tiny
:math:`(l, l)` matrices (:math:`l = k + p` for oversampling ``p``).
Orthonormalisation is done through the eigendecomposition of the small Gram
matrix :math:`g^{\top} g` (giving :math:`Q = g V \Lambda^{-1/2}`), never ``qr``
or ``svd``: on the cuSOLVER-affected GPU stacks ``qr`` / ``svd`` are broken while
the routed symmetric eigendecomposition falls back to a device where it works.
So the large data matrix never enters a solver; only the :math:`(l, l)` Gram
does.  This mirrors the randomised principal-component solver in
:mod:`nitrix.stats` (the same eigh-based range finder), promoted here to a
shared, response-agnostic SVD.

It is *not* a mass-univariate fit accelerator: projecting the response onto a
low-rank basis and recombining is exact only for estimators *linear* in the
data, where a single matmul already beats forming the sketch; the iterative
nonlinear fits (IRLS / GAM / LME) do not commute with the projection.  This
primitive is for decomposition / denoising, where the low-rank factor *is* the
object of interest.

Notes
-----
Call this eagerly, like the one-off principal-component and REML
decompositions.  The routed symmetric-eigendecomposition CPU fallback is an
*eager* ``try``/``except`` (it needs concrete arrays to catch the cuSOLVER
failure), so on a broken-cuSOLVER stack the eigendecomposition lowers to the
dead GPU solver under ``jit`` and crashes at runtime -- wrapping this (or a
pipeline containing it) in a single ``jit`` / ``vmap`` / ``grad`` fails there.
Decompose eagerly, then ``jit`` the heavy downstream separately.  The large
matmuls still run on-device eagerly; only the tiny :math:`(l, l)` Gram
eigendecomposition bounces to CPU.  On a backend with a healthy symmetric
eigendecomposition (a working-cuSOLVER GPU, or CPU) the whole pipeline *is*
``jit``-able with ``k`` passed statically.

Singular values match a dense SVD's leading ``k`` once the spectrum is resolved
(raise ``n_power`` for slowly decaying spectra); the reconstruction
:math:`U \operatorname{diag}(s) V^{\top}` is the rank-``k`` approximation
regardless of sign.

References
----------
Halko, N., Martinsson, P.-G., & Tropp, J. A. (2011).  Finding structure with
randomness: probabilistic algorithms for constructing approximate matrix
decompositions.  *SIAM Review*, 53(2), 217--288.
https://doi.org/10.1137/090771806
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._solver import safe_eigh

__all__ = ['randomized_svd']


def _orthonormalize(g: Float[Array, 'm l']) -> Float[Array, 'm l']:
    r"""Orthonormalise the columns of ``g`` via the Gram eigendecomposition.

    Forms :math:`Q = g V \Lambda^{-1/2}` from the symmetric eigendecomposition
    :math:`g^{\top} g = V \Lambda V^{\top}` -- an eigendecomposition-based
    substitute for QR (which is broken on the cuSOLVER-affected GPU stacks).  The
    eigendecomposition is on the small :math:`(l, l)` Gram matrix only; the large
    :math:`(m, l)` factor never enters a solver.

    Parameters
    ----------
    g
        ``(m, l)`` factor whose ``l`` columns are to be orthonormalised.

    Returns
    -------
    Float[Array, 'm l']
        ``(m, l)`` factor with orthonormal columns spanning the same column
        space as ``g``.
    """
    w, v = safe_eigh(g.T @ g)
    w = jnp.maximum(w, 1e-12)
    return (g @ v) * (1.0 / jnp.sqrt(w))[None, :]


def randomized_svd(
    A: Float[Array, 'm n'],
    k: int,
    *,
    key: Array,
    n_oversample: int = 10,
    n_power: int = 2,
) -> Tuple[Float[Array, 'm k'], Float[Array, 'k'], Float[Array, 'k n']]:
    r"""Top-``k`` randomised SVD :math:`A \approx U \operatorname{diag}(s) V^{\top}`.

    Parameters
    ----------
    A
        ``(m, n)`` matrix to decompose.
    k
        Number of singular triplets to return (``k <= min(m, n)``).
    key
        ``jax.random`` key for the Gaussian sketch (required -- the algorithm is
        randomized; the same key reproduces the result).
    n_oversample
        Oversampling :math:`p`; the sketch width is :math:`l = k + p` (clamped to
        :math:`\min(m, n)`).  Larger :math:`p` improves accuracy at more cost.
    n_power
        Subspace (power) iterations :math:`q`.  Each reapplies :math:`A A^{\top}`
        (with re-orthonormalisation) to sharpen the spectral gap -- raise it for
        slowly decaying spectra; ``0`` is the plain one-pass sketch.

    Returns
    -------
    U : Float[Array, 'm k']
        ``(m, k)`` orthonormal left singular vectors.
    s : Float[Array, 'k']
        ``(k,)`` singular values, descending and non-negative.
    Vt : Float[Array, 'k n']
        ``(k, n)`` right singular vectors.  ``U @ jnp.diag(s) @ Vt`` is the
        rank-``k`` approximation of ``A``.
    """
    m, n = A.shape
    if not 1 <= k <= min(m, n):
        raise ValueError(
            f'randomized_svd: k={k} must satisfy 1 <= k <= min(m, n)='
            f'{min(m, n)}.'
        )
    ell = min(k + n_oversample, min(m, n))

    omega = jax.random.normal(key, (n, ell), dtype=A.dtype)
    g = A @ omega  # (m, l) -- a random slice of the range of A
    for _ in range(n_power):
        g = _orthonormalize(A @ (A.T @ g))  # subspace iteration, re-orthonorm
    q = _orthonormalize(g)  # (m, l) orthonormal basis of the captured range

    b = q.T @ A  # (l, n) -- the small projected matrix, A ~= q b
    # SVD of b through the eigh of b b^T (l, l): b = Ub Sigma Vb^T, so
    # b b^T = Ub Sigma^2 Ub^T; then U = q Ub, V^T = Sigma^{-1} Ub^T b.
    s2, ub = safe_eigh(b @ b.T)
    order = jnp.argsort(s2)[::-1]
    s2 = s2[order]
    ub = ub[:, order]
    s = jnp.sqrt(jnp.maximum(s2, 0.0))
    s_inv = 1.0 / jnp.sqrt(jnp.maximum(s2, 1e-12))

    u = (q @ ub)[:, :k]  # (m, k)
    vt = ((ub.T @ b) * s_inv[:, None])[:k]  # (k, n)
    return u, s[:k], vt
