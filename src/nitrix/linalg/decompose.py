# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Randomized (sketch) low-rank SVD.

``randomized_svd`` recovers the top-``k`` singular triplets of a (possibly
huge) ``(m, n)`` matrix in ``O(m n (k + p))`` matmuls -- the Halko, Martinsson
& Tropp (2011) range finder with subspace (power) iteration -- instead of the
``O(m n min(m, n))`` dense SVD.  It is the decomposition primitive the
small-sample regime wants: ``m`` (voxels) is ``100k-500k`` but only the leading
``k`` modes are needed (PCA / aCompCor confound bases, NORDIC-style singular
value denoising, the spatial eigenimages, the eigsolve top-``k`` consumer).

cuSOLVER-free
-------------

The construction is **all matmuls plus solver calls on tiny ``(l, l)``
matrices** (``l = k + n_oversample``).  Orthonormalisation is done through the
**eigendecomposition of the small Gram** ``g^T g`` (``Q = g V Lambda^{-1/2}``),
*never* ``qr`` / ``svd`` -- on the cuSOLVER-affected GPU stacks ``qr`` / ``svd``
are broken while the routed ``eigh`` (``_solver.safe_eigh``) falls back to a
device where it works.  So the large data matrix never enters a solver; only the
``(l, l)`` Gram does.  This mirrors the ``stats.pca`` randomized solver (same
eigh-based range finder), promoted here to a shared, response-agnostic SVD.

It is **not** a mass-univariate fit accelerator: projecting the response onto a
low-rank basis and recombining is exact only for estimators *linear* in the
data, where a single matmul already beats forming the sketch; the iterative
nonlinear fits (IRLS / GAM / LME) do not commute with the projection.  This
primitive is for decomposition / denoising, where the low-rank factor *is* the
object of interest.

Notes
-----

- **Call it eagerly, like the ``pca`` / ``reml`` one-off decompositions.**
  ``safe_eigh``'s CPU fallback is an *eager* ``try/except`` (it needs concrete
  arrays to catch the cuSOLVER failure), so on a **broken-cuSOLVER** stack the
  ``eigh`` lowers to the dead GPU solver *under ``jit``* and crashes at runtime
  -- wrapping this (or a pipeline containing it) in a single ``jit`` / ``vmap``
  / ``grad`` fails there. Decompose eagerly, then ``jit`` the heavy downstream
  separately. The large matmuls still run on-device eagerly; only the tiny
  ``(l, l)`` Gram eigh bounces to CPU. **On a healthy-``eigh`` backend** (a
  working-cuSOLVER GPU, or CPU) the whole pipeline *is* ``jit``-able with ``k``
  passed statically. A fully jittable / vmappable variant (the patch-wise
  NORDIC case) wants a cuSOLVER-free small symmetric eig -- see the backlog
  ``jacobi-eigensolver-cusolver-free`` (B24).
- Singular values match a dense SVD's leading ``k`` once the spectrum is
  resolved (raise ``n_power`` for slowly decaying spectra); the reconstruction
  ``U diag(s) V^T`` is the rank-``k`` approximation regardless of sign.

References
----------
- Halko, N., Martinsson, P.-G., & Tropp, J. A. (2011).  Finding structure with
  randomness: probabilistic algorithms for constructing approximate matrix
  decompositions.  SIAM Review 53(2), 217-288.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._solver import safe_eigh

__all__ = ['randomized_svd']


def _orthonormalize(g: Float[Array, 'm l']) -> Float[Array, 'm l']:
    """Orthonormalise the columns of ``g`` via the Gram eigendecomposition.

    ``Q = g V Lambda^{-1/2}`` from ``g^T g = V Lambda V^T`` -- an ``eigh``-based
    substitute for QR (broken on the cuSOLVER-affected GPU stacks).  The eigh is
    on the small ``(l, l)`` Gram only; the large ``(m, l)`` factor never enters
    a solver.
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
    """Top-``k`` randomized SVD ``A ~= U diag(s) V^T``.

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
        Oversampling ``p``; the sketch width is ``l = k + p`` (clamped to
        ``min(m, n)``).  Larger ``p`` improves accuracy at more cost.
    n_power
        Subspace (power) iterations ``q``.  Each reapplies ``A A^T`` (with
        re-orthonormalisation) to sharpen the spectral gap -- raise it for
        slowly decaying spectra; ``0`` is the plain one-pass sketch.

    Returns
    -------
    ``(U, s, Vt)``
        ``U`` ``(m, k)`` orthonormal left singular vectors, ``s`` ``(k,)``
        singular values (descending, non-negative), ``Vt`` ``(k, n)`` right
        singular vectors.  ``U @ jnp.diag(s) @ Vt`` is the rank-``k`` approximation.
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
