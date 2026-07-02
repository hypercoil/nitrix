# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Matrix utilities for symmetric matrices, diagonals, and Toeplitz structure.

A collection of the bread-and-butter primitives used by downstream
consumers that work with covariance and connectivity matrices: symmetric
and antisymmetric views, diagonal masking, Toeplitz construction, and
eigenspace reconditioning for stable differentiation.

The main surfaces are:

- :func:`symmetric` -- symmetrise (or antisymmetrise) any square block.
- :func:`sym2vec` / :func:`vec2sym` / :func:`squareform` -- bijection
  between symmetric matrices and the vec of their strict upper triangle.
  Both carry hand-rolled ``custom_vjp`` rules: the backward of
  :func:`sym2vec` sums diagonal entries' gradients only once, and the
  backward of :func:`vec2sym` accounts for the symmetrising reflection.
- :func:`delete_diagonal` / :func:`fill_diagonal` -- masked diagonal
  operations.
- :func:`recondition_eigenspaces` -- stabilise differentiation through SVD
  and symmetric eigendecomposition by perturbing zero or near-degenerate
  eigenvalues.
- :func:`toeplitz` / :func:`toeplitz_2d` -- Toeplitz construction using a
  vmap-over-roll recipe that is faster than scipy on GPU but degrades on
  TPU.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Num

__all__ = [
    'delete_diagonal',
    'fill_diagonal',
    'recondition_eigenspaces',
    'squareform',
    'sym2vec',
    'symmetric',
    'toeplitz',
    'toeplitz_2d',
    'vec2sym',
]


# ---------------------------------------------------------------------------
# Symmetrisation
# ---------------------------------------------------------------------------


def symmetric(
    X: Num[Array, '... i i'],
    *,
    skew: bool = False,
    axes: Tuple[int, int] = (-2, -1),
) -> Num[Array, '... i i']:
    """Average ``X`` with its transpose across the named axes.

    Returns the symmetric part :math:`(X + X^{\\top}) / 2` of the square
    block, or its antisymmetric part :math:`(X - X^{\\top}) / 2` when
    ``skew`` is set.

    Parameters
    ----------
    X : Num[Array, '... i i']
        Tensor with a square block at the named axes.
    skew : bool
        If ``True``, antisymmetrise (:math:`(X - X^{\\top}) / 2`) instead
        of symmetrising.
    axes : tuple of int
        The two axes defining the square block.  Defaults to the trailing
        two axes.

    Returns
    -------
    Num[Array, '... i i']
        The symmetrised (or antisymmetrised) tensor, of the same shape as
        ``X``.
    """
    if skew:
        return (X - X.swapaxes(*axes)) / 2
    return (X + X.swapaxes(*axes)) / 2


# ---------------------------------------------------------------------------
# Eigenspace reconditioning
# ---------------------------------------------------------------------------


def recondition_eigenspaces(
    A: Num[Array, '... i i'],
    *,
    psi: float,
    xi: float = 0.0,
    key: Optional['jax.Array'] = None,
) -> Num[Array, '... i i']:
    r"""Perturb ``A`` so it has no zero or degenerate eigenvalues.

    Stabilises reverse-mode differentiation through ``jnp.linalg.eigh``
    and SVD on near-singular inputs.  The transformation is
    :math:`A := A + (\psi - \xi) I + \operatorname{diag}(x)`, where
    :math:`x \sim \operatorname{Uniform}(0, \xi)` is drawn fresh per
    call.  The additive shift lifts zero eigenvalues away from the origin,
    while the random diagonal splits any exactly-degenerate eigenvalues so
    that the eigenvectors are well defined.

    Parameters
    ----------
    A : Num[Array, '... i i']
        Symmetric matrix block.
    psi : float
        Reconditioning constant for ensuring nonzero eigenvalues.  Must
        satisfy ``psi >= xi``.
    xi : float
        Reconditioning constant for ensuring nondegenerate eigenvalues.
        Default ``0`` (no random perturbation).
    key : jax.Array, optional
        PRNG key.  Required when ``xi > 0``.

    Returns
    -------
    Num[Array, '... i i']
        The reconditioned matrix, of the same shape as ``A``.
    """
    if psi < xi:
        raise ValueError(
            f'psi={psi} must satisfy psi >= xi={xi}: the nondegenerate-'
            'eigenvalue perturbation cannot exceed the nonzero-eigenvalue '
            'perturbation, or A may become indefinite.'
        )
    n = A.shape[-1]
    eye = jnp.eye(n, dtype=A.dtype)
    if xi > 0:
        if key is None:
            raise ValueError(
                'recondition_eigenspaces requires key when xi > 0.'
            )
        x = jax.random.uniform(
            key=key,
            shape=A.shape[-1:],
            maxval=xi,
            dtype=A.dtype,
        )
        return A + (psi - xi) * eye + jnp.diag(x)
    return A + psi * eye


# ---------------------------------------------------------------------------
# Diagonal helpers
# ---------------------------------------------------------------------------


def delete_diagonal(A: Num[Array, '... i i']) -> Num[Array, '... i i']:
    """Zero out the main diagonal of a square matrix block.

    Parameters
    ----------
    A : Num[Array, '... i i']
        Square matrix block.

    Returns
    -------
    Num[Array, '... i i']
        A copy of ``A`` with its main diagonal set to zero and all
        off-diagonal entries unchanged.
    """
    mask = ~jnp.eye(A.shape[-1], dtype=bool)
    return A * mask


def fill_diagonal(
    A: Num[Array, '... i j'],
    *,
    fill: float = 0.0,
    offset: int = 0,
) -> Num[Array, '... i j']:
    """Set the named diagonal of ``A`` to ``fill``.

    Parameters
    ----------
    A : Num[Array, '... i j']
        Matrix block (need not be square).
    fill : float
        Value to write into the diagonal cells.
    offset : int
        Diagonal offset.  ``0`` is the main diagonal, positive is above,
        negative is below.

    Returns
    -------
    Num[Array, '... i j']
        A copy of ``A`` with the cells on the named diagonal replaced by
        ``fill`` and all other entries unchanged.  Diagonal cells that
        fall outside the matrix are ignored.
    """
    m, n = A.shape[-2], A.shape[-1]
    rows = jnp.arange(m)
    cols = rows + offset
    valid = (cols >= 0) & (cols < n)
    rows = jnp.where(valid, rows, 0)
    cols = jnp.where(valid, cols, 0)
    # Build the indexing mask: True at (i, i + offset) for valid (i).
    indicator = jnp.zeros((m, n), dtype=bool).at[rows, cols].set(valid)
    return jnp.where(indicator, fill, A)


# ---------------------------------------------------------------------------
# Toeplitz construction
# ---------------------------------------------------------------------------


def toeplitz_2d(
    c: Float[Array, 'mc'],
    r: Optional[Float[Array, 'nr']] = None,
    *,
    shape: Optional[Tuple[int, int]] = None,
    fill_value: float = 0.0,
) -> Float[Array, 'm n']:
    """Construct a 2-D Toeplitz matrix from a first column and first row.

    Based on the recipe in
    https://github.com/google/jax/issues/1646#issuecomment-1139044324
    (the @mattjj / @blakehechtman "vmap-over-roll" pattern).  This
    underperforms on TPU but is the fastest approach on GPU.

    Unlike ``scipy.linalg.toeplitz``, when ``c[0] != r[0]`` the
    *column*-leading element wins (``r[0]`` is ignored).  The
    output supports rectangular shapes and a fill value for cells
    outside the originally-specified support.

    Parameters
    ----------
    c
        First column.  Length ``mc``; extended to ``m`` with
        ``fill_value`` if needed.
    r
        First row.  Length ``nr``; extended to ``n`` with
        ``fill_value`` if needed.  If ``None``, defaults to ``c``
        (symmetric Toeplitz).
    shape
        Output ``(m, n)``.  Defaults to ``(len(c), len(r))``.
    fill_value
        Fill for cells outside the support of ``c`` / ``r``.

    Returns
    -------
    Float[Array, 'm n']
        The ``(m, n)`` Toeplitz matrix whose first column is ``c`` and
        whose first row is ``r``, with cells beyond the support of ``c``
        and ``r`` set to ``fill_value``.
    """
    if r is None:
        r = c
    m_in, n_in = c.shape[-1], r.shape[-1]
    m, n = shape if shape is not None else (m_in, n_in)
    d = max(m, n)
    if (m != n) or (m_in != n_in) or (m_in != d):
        r_arg = jnp.full(d, fill_value, dtype=r.dtype).at[:n_in].set(r)
        c_arg = jnp.full(d, fill_value, dtype=c.dtype).at[:m_in].set(c)
    else:
        r_arg, c_arg = r, c
    # Index-based reverse rather than ``jnp.flip``: the ``reverse`` HLO that
    # ``flip`` emits trips an XLA CPU ``AlgebraicSimplifier::HandleReverse``
    # CHECK-failure ("Invalid binary instruction opcode map") on this graph
    # under jax >= 0.10 (a hard compiler abort, not a Python error).  A gather
    # produces identical output at negligible cost and avoids the bug.
    c_arg = c_arg[..., jnp.arange(d - 1, -1, -1)]

    mask = jnp.zeros(2 * d - 1, dtype=bool).at[: (d - 1)].set(True)
    iota = jnp.arange(d)

    def roll_one(
        c: Num[Array, '...'],
        r: Num[Array, '...'],
        i: Int[Array, '...'],
        mask: Bool[Array, '...'],
    ) -> Num[Array, '...']:
        rs = jnp.roll(r, i, axis=-1)
        cs = jnp.roll(c, i + 1, axis=-1)
        ms = jnp.roll(mask, i, axis=-1)[-d:]
        return jnp.where(ms, cs, rs)

    f = jax.vmap(roll_one, in_axes=(None, None, 0, None))
    return f(c_arg, r_arg, iota[..., None], mask)[..., :m, :n]


def toeplitz(
    c: Float[Array, '... mc'],
    r: Optional[Float[Array, '... nr']] = None,
    *,
    shape: Optional[Tuple[int, int]] = None,
    fill_value: float = 0.0,
) -> Float[Array, '... m n']:
    """Toeplitz matrix construction with leading batch support.

    Vectorises :func:`toeplitz_2d` over any leading axes; both ``c`` and
    ``r`` must share the same leading batch shape.  See
    :func:`toeplitz_2d` for the per-sample semantics, including the
    handling of ``shape`` and ``fill_value``.

    Parameters
    ----------
    c : Float[Array, '... mc']
        First column of each Toeplitz matrix, with arbitrary leading
        batch axes and a trailing length ``mc``.
    r : Float[Array, '... nr'], optional
        First row of each Toeplitz matrix, sharing the leading batch axes
        of ``c`` and with trailing length ``nr``.  If ``None``, defaults
        to ``c`` (symmetric Toeplitz).
    shape : tuple of int, optional
        Output ``(m, n)`` of each Toeplitz block.  Defaults to
        ``(mc, nr)``.
    fill_value : float
        Fill for cells outside the support of ``c`` / ``r``.

    Returns
    -------
    Float[Array, '... m n']
        The batch of ``(m, n)`` Toeplitz matrices, carrying the shared
        leading batch axes of ``c`` and ``r``.
    """
    if r is None:
        r = c
    if c.shape[:-1] != r.shape[:-1]:
        raise ValueError(
            f'toeplitz: leading batch dims must match; got '
            f'c.shape={c.shape}, r.shape={r.shape}.'
        )

    core = partial(toeplitz_2d, shape=shape, fill_value=fill_value)
    fn = core
    for _ in range(c.ndim - 1):
        fn = jax.vmap(fn, in_axes=(0, 0))
    return fn(c, r)


# ---------------------------------------------------------------------------
# Symmetric <-> vec bijection
# ---------------------------------------------------------------------------


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def sym2vec(
    sym: Num[Array, '... i i'],
    offset: int = 1,
) -> Num[Array, '... n']:
    r"""Vec of the upper triangle of a symmetric matrix block.

    Output entries follow the row-major order of the upper triangle
    starting at the diagonal offset by ``offset``.  This is the inverse
    of :func:`vec2sym`.

    Parameters
    ----------
    sym : Num[Array, '... i i']
        Symmetric matrix block.  (Symmetry is assumed, not checked.)
    offset : int
        Diagonal offset.  Default ``1`` (skip the main diagonal,
        producing :math:`\binom{i}{2} = i (i - 1) / 2` entries).  Use
        ``offset=0`` to include the main diagonal.

    Returns
    -------
    Num[Array, '... n']
        The vectorised upper triangle, carrying the leading batch axes of
        ``sym`` and a trailing axis of length ``n`` equal to the number
        of selected entries.

    Notes
    -----
    The hand-rolled ``custom_vjp`` reflects that this operation strictly
    drops half the matrix, so its backward is masked to the upper
    triangle: the gradient through the dropped lower triangle is zero
    rather than mirrored.
    """
    idx = jnp.triu_indices(m=sym.shape[-2], n=sym.shape[-1], k=offset)
    vec = sym[..., idx[0], idx[1]]
    return vec.reshape(*sym.shape[:-2], -1)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def vec2sym(
    vec: Num[Array, '... n'],
    offset: int = 1,
) -> Num[Array, '... i i']:
    r"""Symmetric matrix from the vec of its upper triangle.

    The inverse of :func:`sym2vec`.  The matrix side length is recovered
    from ``n`` in closed form as
    :math:`\lfloor (\sqrt{8 n + 1} + 1) / 2 \rfloor + (\text{offset} - 1)`,
    then the vector is written into the upper triangle at the named
    offset and reflected across it: entries off the named diagonal are
    duplicated into the lower triangle, while entries on it are left
    unduplicated.

    Parameters
    ----------
    vec : Num[Array, '... n']
        Vectorised upper triangle, with arbitrary leading batch axes and
        a trailing axis of length ``n``.
    offset : int
        Diagonal offset at which ``vec`` was extracted.  Default ``1``
        (the strict upper triangle, excluding the main diagonal).  Use
        ``offset=0`` if ``vec`` includes the main diagonal.

    Returns
    -------
    Num[Array, '... i i']
        The reconstructed symmetric matrix block, carrying the leading
        batch axes of ``vec``.
    """
    n = vec.shape[-1]
    side = int(0.5 * (math.sqrt(8 * n + 1) + 1)) + (offset - 1)
    idx = jnp.triu_indices(m=side, n=side, k=offset)
    sym = jnp.zeros((*vec.shape[:-1], side, side), dtype=vec.dtype)
    sym = sym.at[..., idx[0], idx[1]].set(vec)
    # Mirror the off-(named-)diagonal entries; the named diagonal
    # is duplicated by the addition so we exclude it via the mask.
    return jnp.where(
        jnp.eye(side, dtype=bool), sym, sym + sym.swapaxes(-1, -2)
    )


def squareform(X: Num[Array, '...']) -> Num[Array, '...']:
    """Toggle between vec and square forms of a symmetric matrix.

    A convenience wrapper: if ``X`` looks square and symmetric (up to a
    rough tolerance), return ``sym2vec(X, offset=1)``; otherwise treat it
    as a vector and return ``vec2sym(X, offset=1)``.

    Parameters
    ----------
    X : Num[Array, '...']
        Either a square symmetric matrix block to vectorise, or the
        vectorised strict upper triangle to expand into a matrix.

    Returns
    -------
    Num[Array, '...']
        The complementary form: the vectorised strict upper triangle if
        ``X`` was a symmetric matrix, otherwise the reconstructed
        symmetric matrix.

    Notes
    -----
    Unlike ``scipy.spatial.distance.squareform``, this does not verify
    that the input is a strictly conforming form.  When in doubt, call
    :func:`sym2vec` / :func:`vec2sym` directly with the explicit
    ``offset``.
    """
    if (
        X.ndim >= 2
        and X.shape[-2] == X.shape[-1]
        and jnp.allclose(
            X,
            X.swapaxes(-1, -2),
        )
    ):
        return sym2vec(X, offset=1)
    return vec2sym(X, offset=1)


# ---------------------------------------------------------------------------
# custom_vjp rules for sym2vec / vec2sym
# ---------------------------------------------------------------------------
#
# The handle-rolled VJPs exist because the default would either:
# (a) propagate the gradient through the implicit symmetrisation in
#     ``vec2sym``, double-counting the off-diagonal entries; or
# (b) propagate gradients into the lower triangle of ``sym2vec``,
#     which has no information in a truly symmetric input.
# The rules below match the natural ``jax.grad`` semantics for the
# *upper-triangle-only* parameterisation that downstream consumers
# typically intend.


def _sym2vec_fwd(
    sym: Num[Array, '...'], offset: int
) -> Tuple[Num[Array, '...'], None]:
    return sym2vec(sym, offset=offset), None


def _sym2vec_bwd(
    offset: int, _res: None, g: Num[Array, '...']
) -> Tuple[Num[Array, '...']]:
    # Backprop: place the cotangents back into the upper triangle.
    # Do *not* mirror; the lower triangle of the input had no
    # information.
    unmirrored = vec2sym(g, offset=offset)
    mask = jnp.triu(
        jnp.ones(unmirrored.shape[-2:], dtype=unmirrored.dtype), offset
    )
    return (unmirrored * mask,)


def _vec2sym_fwd(
    vec: Num[Array, '...'], offset: int
) -> Tuple[Num[Array, '...'], None]:
    return vec2sym(vec, offset=offset), None


def _vec2sym_bwd(
    offset: int, _res: None, g: Num[Array, '...']
) -> Tuple[Num[Array, '...']]:
    # Backprop: scale by 2 for off-(named-)diagonal entries to account
    # for the implicit mirroring in the forward; 1 for on-diagonal.
    scale = jnp.where(jnp.eye(g.shape[-1], dtype=bool), 1.0, 2.0)
    return (sym2vec(scale * symmetric(g), offset=offset),)


sym2vec.defvjp(_sym2vec_fwd, _sym2vec_bwd)
vec2sym.defvjp(_vec2sym_fwd, _vec2sym_bwd)
