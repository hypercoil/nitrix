# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Subspace geometry and orthogonal alignment.

A small vocabulary of SVD-of-a-cross-product subspace-geometry primitives, all
computed cuSOLVER-free (small-Gram ``safe_eigh``, never ``jnp.linalg.svd`` /
``qr``):

- ``orthogonal_procrustes`` -- the orthogonal matrix best mapping one
  configuration onto another (Schoenemann 1966), optionally regularised by an
  additive natural-parameter term.
- ``image_basis`` -- an orthonormal basis for the range (column space) of a
  matrix, the differentiable analogue of ``scipy.linalg.orth``.
- ``subspace_angles`` -- the principal / canonical (Grassmann) angles between
  two subspaces, with the numerically-stable ``arcsin`` / ``arccos`` split
  (Knyazev-Argentati 2002) that keeps the gradient finite at ``0`` and
  ``pi / 2``.

``orthogonal_procrustes`` is the algebraic crux of the **ProMises**
functional-alignment model (Andreella & Finos 2022): a *matrix* von
Mises-Fisher (matrix-Langevin) prior on the rotation contributes its
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

__all__ = ['orthogonal_procrustes', 'image_basis', 'subspace_angles']


def _mT(x: Float[Array, '... a b']) -> Float[Array, '... b a']:
    """Batched transpose of the trailing two axes."""
    return jnp.swapaxes(x, -1, -2)


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


def image_basis(
    x: Float[Array, '... m n'],
    *,
    rcond: Optional[float] = None,
    rank: Optional[int] = None,
) -> Float[Array, '... m k']:
    r"""Orthonormal basis for the range (column space) of ``x``.

    The differentiable, cuSOLVER-free analogue of ``scipy.linalg.orth``: the
    leading left singular vectors of ``x`` (spanning its column space),
    computed through the eigendecomposition of the smaller of the two Gram
    matrices ``x x^T`` / ``x^T x`` (``safe_eigh``), never ``jnp.linalg.svd``.

    Parameters
    ----------
    x
        ``(..., m, n)`` matrix whose range (column space, a subspace of
        :math:`\mathbb{R}^m`) is sought.
    rcond
        Relative singular-value cutoff for the numerical rank when ``rank`` is
        ``None``: singular directions with :math:`\sigma_i > \text{rcond} \cdot
        \sigma_{\max}` are kept.  Default ``max(m, n) * eps`` (the
        ``scipy.linalg.orth`` rule).  **Determining the rank from ``rcond`` is
        data-dependent, hence eager-only** (the returned column count is a
        concrete integer); pass ``rank`` explicitly under ``jit``.
    rank
        Static number of leading columns to return.  When given (the
        ``jit``-clean path) the numerical-rank estimation is skipped and exactly
        ``rank`` orthonormal columns are returned.

    Returns
    -------
    ``q``
        ``(..., m, k)`` matrix with orthonormal columns spanning the range of
        ``x`` (``k`` the numerical rank, or ``rank`` if given).

    Notes
    -----
    Any orthonormal basis of the range is a valid answer; the particular
    columns carry the ``eigh`` sign / in-eigenspace-rotation gauge, so the
    *basis* is gauge-dependent (only the projector ``q q^T`` and gauge-invariant
    functionals such as :func:`subspace_angles` are canonical).
    """
    m, n = x.shape[-2], x.shape[-1]
    eps = jnp.finfo(x.dtype).eps
    if m <= n:
        gram = symmetric(x @ _mT(x))  # (..., m, m)
        s2_asc, u_asc = safe_eigh(gram)
        s2 = s2_asc[..., ::-1]  # descending
        u = u_asc[..., ::-1]  # (..., m, m) left singular vectors
    else:
        gram = symmetric(_mT(x) @ x)  # (..., n, n)
        s2_asc, v_asc = safe_eigh(gram)
        s2 = s2_asc[..., ::-1]
        v = v_asc[..., ::-1]  # (..., n, n) right singular vectors
        floor = jnp.maximum(s2[..., :1], 0.0) * max(m, n) * eps
        s_safe = jnp.sqrt(jnp.where(s2 < floor, floor, s2))
        u = (x @ v) / s_safe[..., None, :]  # (..., m, n)
    if rank is None:
        # Threshold on the *eigenvalues* s^2 (accurate to ~eps relative), not
        # the singular values s: the Gram's null floor sits at ~eps on s^2 but
        # only ~sqrt(eps) on s = sqrt(s^2), so an s-scale cutoff over-counts the
        # numerical rank.  A user-supplied rcond is a singular-value cutoff
        # (scipy.linalg.orth semantics), applied here as rcond^2 on s^2.
        s2_max = s2[..., :1]
        thr = (
            float(rcond) ** 2 * s2_max
            if rcond is not None
            else max(m, n) * float(eps) * s2_max
        )
        k = int(jnp.max(jnp.sum(s2 > thr, axis=-1)))  # eager (data-dependent)
    else:
        k = rank
    return u[..., :k]


def _svdvals_desc(mat: Float[Array, '... a b']) -> Float[Array, '... s']:
    """Singular values of ``mat`` in descending order, cuSOLVER-free.

    The square roots of the eigenvalues of the smaller Gram (``safe_eigh``);
    returns ``min(a, b)`` values.  Only the *eigenvalues* are used downstream,
    so the reverse-mode VJP stays finite even at repeated singular values (the
    ``1/(lambda_i - lambda_j)`` eigenvector-sensitivity terms never arise).
    """
    a, b = mat.shape[-2], mat.shape[-1]
    gram = symmetric(mat @ _mT(mat)) if a <= b else symmetric(_mT(mat) @ mat)
    s2_asc, _ = safe_eigh(gram)
    return jnp.sqrt(jnp.maximum(s2_asc[..., ::-1], 0.0))


# Newton-Schulz inverse-sqrt iterations for the Loewdin orthonormalisation
# below; 30 covers condition numbers up to ~1e5 (well past where a subspace is
# numerically well-defined) at double precision.
_LOEWDIN_ITERS = 30


def _loewdin_orthonormalize(
    x: Float[Array, '... m p'],
) -> Float[Array, '... m p']:
    r"""Symmetric (Loewdin) orthonormalisation ``Q = x (x^T x)^{-1/2}``.

    The inverse square root is computed by a Newton-Schulz iteration (matmuls
    only), so -- unlike an ``eigh``-based orthonormalisation -- the reverse-mode
    VJP is finite even when ``x`` already has orthonormal columns (``x^T x = I``,
    a repeated spectrum that makes the eigenvector VJP blow up).  The Loewdin
    factor is the *unique* symmetric orthonormaliser (no eigenvector gauge), so
    ``Q`` is a smooth function of ``x`` wherever ``x`` has full column rank.
    Assumes ``x`` is reasonably well-conditioned (the iteration is accurate for
    condition numbers up to ~1e5); pre-truncate a near-rank-deficient basis with
    :func:`image_basis`.
    """
    gram = symmetric(_mT(x) @ x)  # (..., p, p) SPD
    p = gram.shape[-1]
    eye = jnp.eye(p, dtype=x.dtype)
    # Upper bound on the largest eigenvalue (a few power iterations) to scale the
    # spectrum into the Newton-Schulz convergence region.
    v = jnp.ones((*gram.shape[:-1], 1), x.dtype)
    for _ in range(5):
        v = gram @ v
        v = v / jnp.linalg.norm(v, axis=-2, keepdims=True)
    lam = jnp.sum(v * (gram @ v), axis=-2, keepdims=True)  # (..., 1, 1)
    scaled = gram / lam
    y = scaled
    z = jnp.broadcast_to(eye, scaled.shape)
    for _ in range(_LOEWDIN_ITERS):
        t = 1.5 * eye - 0.5 * (z @ y)
        y = y @ t
        z = t @ z
    inv_sqrt = z / jnp.sqrt(lam)  # (x^T x)^{-1/2}
    return x @ inv_sqrt


def subspace_angles(
    x: Float[Array, '... m p'],
    y: Float[Array, '... m q'],
) -> Float[Array, '... k']:
    r"""Principal (canonical / Grassmann) angles between two subspaces.

    The angles :math:`0 \le \theta_1 \le \dots \le \theta_k \le \pi/2` between
    the ranges of ``x`` and ``y`` (returned in **descending** order,
    :math:`\theta_k` first, matching ``scipy.linalg.subspace_angles``).
    Computed by the numerically-stable Knyazev-Argentati (2002) scheme: the
    cosines are the singular values of :math:`Q_x^\top Q_y` (orthonormal bases),
    and each angle is taken as :math:`\arccos(\sigma)` when :math:`\sigma^2 <
    \tfrac12` (large angle) but as :math:`\arcsin` of the corresponding *sine*
    (the singular values of the residual :math:`Q_y - Q_x Q_x^\top Q_y`) when
    :math:`\sigma^2 \ge \tfrac12` (small angle).

    The split keeps each inverse trig function away from its
    infinite-derivative endpoint, so -- unlike a naive ``arccos`` -- the
    **gradient stays finite at both boundaries** (:math:`\theta = 0`, where
    :math:`\sigma \to 1`, and :math:`\theta = \pi/2`, where :math:`\sigma \to
    0`); the branches are fed boundary-safe inputs so the unused branch cannot
    inject a NaN.  Fully **cuSOLVER-free** and differentiably stable: the bases
    are formed by a matmul-only Loewdin orthonormalisation (finite VJP even for
    already-orthonormal inputs) and only eigen*values* feed the angles (finite
    VJP at repeated / degenerate principal angles).

    Parameters
    ----------
    x, y
        ``(..., m, p)`` / ``(..., m, q)`` matrices whose column spaces
        (subspaces of :math:`\mathbb{R}^m`, assumed full column rank and
        reasonably conditioned) are compared.  ``k = min(p, q)`` angles are
        returned.  Pre-truncate a near-rank-deficient basis with
        :func:`image_basis`.

    Returns
    -------
    ``angles``
        ``(..., min(p, q))`` principal angles in radians, descending.
    """
    p, q = x.shape[-1], y.shape[-1]
    qa = _loewdin_orthonormalize(x)  # (..., m, p)
    qb = _loewdin_orthonormalize(y)  # (..., m, q)
    cross = _mT(qa) @ qb  # (..., p, q)
    # Cosines (ascending -> descending angle) and sines (descending sine ->
    # descending angle) are co-indexed: position i is the i-th largest angle.
    sigma_asc = _svdvals_desc(cross)[..., ::-1]
    if p >= q:
        resid = qb - qa @ cross  # (..., m, q)
    else:
        resid = qa - qb @ _mT(cross)  # (..., m, p)
    mu_desc = _svdvals_desc(resid)
    # sigma^2 >= 1/2 marks a small angle: take arcsin(sine), else arccos(cosine).
    # Feed each branch a boundary-safe input so the unused branch's gradient
    # (arccos' near sigma=1, arcsin' near mu=1) cannot become a NaN.
    small = sigma_asc**2 >= 0.5
    cos_safe = jnp.where(small, 0.0, sigma_asc)
    sin_safe = jnp.where(small, mu_desc, 0.0)
    return jnp.where(
        small,
        jnp.arcsin(jnp.clip(sin_safe, -1.0, 1.0)),
        jnp.arccos(jnp.clip(cos_safe, -1.0, 1.0)),
    )
