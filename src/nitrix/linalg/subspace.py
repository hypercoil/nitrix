# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Subspace geometry and orthogonal alignment.

``orthogonal_procrustes`` -- the orthogonal matrix best mapping one
configuration onto another (Schoenemann 1966), optionally regularised by an
additive natural-parameter term.  The latter is the algebraic crux of the
**ProMises** functional-alignment model (Andreella & Finos 2022): a *matrix*
von Mises-Fisher (matrix-Langevin) prior on the rotation contributes its
natural-parameter matrix ``F`` additively to the cross-product, and the MAP
estimate is the Procrustes solution of ``A^T B + F`` -- so the prior never
materialises its (matrix-argument hypergeometric) normaliser.  The
``nitrix.register`` functional-alignment recipe composes this.

cuSOLVER-free
-------------

The Procrustes solution is the orthogonal **polar factor** of the cross-product
``C = A^T B (+ prior)``::

    C = U Sigma V^T   =>   R = U V^T = C (C^T C)^{-1/2}

computed through the eigendecomposition of the small ``(p, p)`` Gram ``C^T C``
(``safe_eigh`` -- the Jacobi path that survives the cuSOLVER-affected GPU
stacks), *never* ``jnp.linalg.svd`` / ``qr`` (broken there, like the
``decompose`` / ``pca`` randomized solvers).  The polar factor is invariant to
the eigenvector sign / in-eigenspace-rotation gauge of the ``eigh``, so ``R`` is
well-defined even at repeated singular values (the standard Procrustes
degeneracy only bites the reverse-mode gradient, stabilised by ``psi``).

References
----------
- Schoenemann, P. H. (1966).  A generalized solution of the orthogonal
  Procrustes problem.  Psychometrika 31(1), 1-10.
- Andreella, A. & Finos, L. (2022).  Procrustes analysis for high-dimensional
  data.  Human Brain Mapping (the ProMises model; the ``prior`` term).
"""

from __future__ import annotations

from typing import Optional, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._solver import safe_eigh
from .matrix import recondition_eigenspaces, symmetric

__all__ = ['orthogonal_procrustes']


def _det_sign(m: Float[Array, '... p p']) -> Float[Array, '...']:
    """Sign of ``det(m)`` in ``{-1, 0, +1}`` via pure-XLA pivoted elimination.

    ``jnp.linalg.det`` lowers to an LU (``getrf``) on the cuSOLVER pool, which is
    broken on the affected GPU stacks (like ``svd`` / ``inv``).  Only the *sign*
    is needed (the reflection test), so this runs Gaussian elimination with
    partial pivoting in matmul/select primitives: the sign is the swap parity
    times the product of the pivot signs.  cuSOLVER-free and ``jit``-safe.
    """
    p = m.shape[-1]

    def single(mat: Float[Array, 'p p']) -> Float[Array, '']:
        def body(k: int, carry: tuple) -> tuple:  # type: ignore[type-arg]
            a, sign = carry
            col = jnp.where(jnp.arange(p) >= k, jnp.abs(a[:, k]), -1.0)
            piv = jnp.argmax(col)
            row_k, row_p = a[k], a[piv]
            a = a.at[k].set(row_p).at[piv].set(row_k)
            sign = sign * jnp.where(piv != k, -1.0, 1.0)
            pivot = a[k, k]
            sign = sign * jnp.sign(pivot)
            denom = jnp.where(pivot == 0.0, 1.0, pivot)
            factor = jnp.where(jnp.arange(p) > k, a[:, k] / denom, 0.0)
            a = a - factor[:, None] * a[k][None, :]
            return a, sign

        _, sign = jax.lax.fori_loop(0, p, body, (mat, jnp.ones((), mat.dtype)))
        return cast(Float[Array, ''], sign)

    return cast(
        Float[Array, '...'],
        jnp.vectorize(single, signature='(p,p)->()')(m),
    )


def orthogonal_procrustes(
    a: Float[Array, '... n p'],
    b: Float[Array, '... n p'],
    *,
    prior: Optional[Float[Array, '... p p']] = None,
    allow_reflection: bool = True,
    psi: float = 0.0,
    key: Optional[jax.Array] = None,
) -> Float[Array, '... p p']:
    r"""Orthogonal matrix ``R`` best mapping ``a`` onto ``b``.

    Solves ``argmin_{R^T R = I} || a R - b ||_F`` -- the orthogonal Procrustes
    problem.  With ``C = a^T b`` (plus the optional ``prior``) and the SVD
    ``C = U Sigma V^T``, the minimiser is ``R = U V^T``, computed here as the
    orthogonal polar factor ``C (C^T C)^{-1/2}`` via a single symmetric
    eigendecomposition of the ``(p, p)`` Gram (cuSOLVER-free; see the module
    docstring).

    Parameters
    ----------
    a, b
        ``(..., n, p)`` configurations of ``n`` observations in a shared
        ``p``-dimensional feature space (the natural convention: ``a R``
        rotates ``a``'s feature axes onto ``b``'s).  Leading dims broadcast/batch.
    prior
        Optional ``(..., p, p)`` additive term on the cross-product ``a^T b``.
        This is the **matrix von Mises-Fisher natural parameter** of the
        ProMises model -- it regularises the rotation toward a prior orientation
        (the MAP estimate is the Procrustes solution of ``a^T b + prior``).
        ``None`` recovers the plain (maximum-likelihood) Procrustes solution.
    allow_reflection
        ``True`` (default) returns the closest matrix in the full orthogonal
        group ``O(p)`` -- a reflection if the data demand one (this matches
        ``scipy.linalg.orthogonal_procrustes`` and is the usual choice for
        representational alignment).  ``False`` constrains to the proper-rotation
        group ``SO(p)`` (``det R = +1``; the Kabsch variant) by flipping the
        least-significant singular direction when ``det C < 0``.
    psi
        Reverse-mode reconditioning strength.  ``0`` (default) leaves the Gram
        untouched -- correct for the forward solve and for gradients away from
        degeneracy.  ``psi > 0`` perturbs the Gram via
        ``recondition_eigenspaces`` before ``eigh`` so the eigh-VJP stays finite
        at repeated singular values (the Procrustes gradient degeneracy).
    key
        PRNG key, required when ``psi > 0``.

    Returns
    -------
    ``R``
        ``(..., p, p)`` orthogonal matrix.  ``a @ R`` is the aligned ``a``.

    Notes
    -----
    ``R`` is invariant to the eigenvector gauge of the internal ``eigh`` (a sign
    flip of column ``i`` flips the matching column of ``U`` too, leaving
    ``U V^T`` unchanged), so repeated singular values do not make the *forward*
    result ambiguous; only the reverse-mode gradient needs ``psi``.  For a
    rank-deficient ``C`` the orthogonal completion is genuinely non-unique
    (documented; the floored Gram keeps the result finite).
    """
    c = jnp.swapaxes(a, -1, -2) @ b  # (..., p, p) -- the cross-product A^T B
    if prior is not None:
        c = c + prior
    gram = symmetric(jnp.swapaxes(c, -1, -2) @ c)  # C^T C = V diag(s2) V^T
    if psi > 0:
        gram = recondition_eigenspaces(gram, psi=psi, xi=psi, key=key)
    # C^T C = V diag(s2) V^T: s2 ascending = sigma^2, v the eigenvectors.
    s2, v = safe_eigh(gram)
    p = c.shape[-1]
    eps = jnp.finfo(c.dtype).eps
    floor = jnp.maximum(s2[..., -1:], 0.0) * p * eps  # rank-trunc threshold
    sigma = jnp.sqrt(jnp.where(s2 < floor, floor, s2))
    # U = C V Sigma^{-1} has orthonormal columns; R = U V^T is the polar factor.
    u = (c @ v) / sigma[..., None, :]
    vt = jnp.swapaxes(v, -1, -2)
    if allow_reflection:
        return u @ vt
    # SO(p): det(U V^T) = sign(det C) since Sigma >= 0; flip the smallest
    # singular direction (column 0 in ascending order) when it is a reflection.
    sign = _det_sign(c)
    sign = jnp.where(sign == 0, 1.0, sign)
    flip = jnp.where(jnp.arange(p) == 0, sign[..., None], 1.0)
    return (u * flip[..., None, :]) @ vt
