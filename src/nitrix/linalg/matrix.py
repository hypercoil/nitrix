# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Matrix utilities: symmetric / triangular views, diagonal helpers,
Toeplitz construction, and eigenspace reconditioning.

The bread-and-butter primitives used by downstream consumers that
work with covariance / connectivity matrices:

- ``symmetric`` -- symmetrise (or antisymmetrise) any square block.
- ``sym2vec`` / ``vec2sym`` / ``squareform`` -- bijection between
  symmetric matrices and the vec of their strict upper triangle.
  Both carry hand-rolled ``custom_vjp`` rules: ``sym2vec``'s
  backward sums diagonal entries' gradients only once;
  ``vec2sym``'s backward accounts for the symmetrising
  reflection.
- ``delete_diagonal`` / ``fill_diagonal`` -- standard masked
  operations.
- ``recondition_eigenspaces`` -- stabilise differentiation
  through SVD / eigh by perturbing zero / near-degenerate
  eigenvalues.
- ``toeplitz`` / ``toeplitz_2d`` -- vmap-over-batch Toeplitz
  construction following the @mattjj / @blakehechtman 2022
  recipe (see hypercoil legacy comment).  Faster than scipy on
  GPU; degrades on TPU per the same reference.

What the legacy ``hypercoil.functional.matrix`` had that we drop:

- ``cholesky_invert`` -- in nearly all cases slower than
  ``jnp.linalg.inv``.
- ``spd`` -- the legacy crude PSD enforcement.  Use
  ``recondition_eigenspaces`` for the differentiable-stability
  use case; use ``nitrix.linalg.spd`` for SPD-manifold ops
  proper (matrix log / exp / sqrt).
- ``expand_outer`` -- just write the ``jnp.einsum`` you want.
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

    Parameters
    ----------
    X
        Tensor with a square block at the named axes.
    skew
        If ``True``, antisymmetrise (``(X - X.T) / 2``) instead.
    axes
        Two axes defining the square block.  Defaults to the
        trailing two axes.
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

    Stabilises reverse-mode differentiation through
    ``jnp.linalg.eigh`` / SVD on near-singular inputs.  The
    transformation is

    ``A := A + (psi - xi / 2) I + diag(x)``

    where ``x ~ Uniform(0, xi)`` is fresh per call.

    Parameters
    ----------
    A
        Symmetric matrix block.
    psi
        Reconditioning constant for ensuring nonzero eigenvalues.
        Must satisfy ``psi >= xi``.
    xi
        Reconditioning constant for ensuring nondegenerate
        eigenvalues.  Default ``0`` (no random perturbation).
    key
        PRNG key.  Required when ``xi > 0``.

    Returns
    -------
    Reconditioned matrix.
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
    """Zero out the diagonal of a square matrix block."""
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
    A
        Matrix block (need not be square).
    fill
        Value to write into the diagonal cells.
    offset
        Diagonal offset.  ``0`` is the main diagonal, positive is
        above, negative is below.
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

    Vectorises ``toeplitz_2d`` over any leading axes; both ``c``
    and ``r`` must share the same leading shape.

    See ``toeplitz_2d`` for the per-sample semantics.
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
    """Vec of the upper triangle of a symmetric matrix block.

    Output entries follow the row-major order of the upper triangle
    starting at the diagonal offset by ``offset``.

    Parameters
    ----------
    sym
        Symmetric matrix block.  (Not checked.)
    offset
        Diagonal offset.  Default ``1`` (skip main diagonal,
        producing ``binom(i, 2) = i (i - 1) / 2`` entries).  Use
        ``offset=0`` to include the main diagonal.

    Returns
    -------
    The vec, shape ``(..., n)``.

    Notes
    -----
    The hand-rolled ``custom_vjp``: ``sym2vec`` strictly drops half
    the matrix, so the backward (``vec2sym``-like) is masked to the
    upper triangle so the gradient through the dropped half is zero
    (rather than mirrored).
    """
    idx = jnp.triu_indices(m=sym.shape[-2], n=sym.shape[-1], k=offset)
    vec = sym[..., idx[0], idx[1]]
    return vec.reshape(*sym.shape[:-2], -1)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def vec2sym(
    vec: Num[Array, '... n'],
    offset: int = 1,
) -> Num[Array, '... i i']:
    """Symmetric matrix from the vec of its upper triangle.

    The matrix size ``i`` is recovered from ``n`` by solving
    ``n = binom(i - offset + 1, 2)`` (the closed-form
    ``i = ceil((sqrt(8n + 1) + 1) / 2) + (offset - 1)``).

    The forward symmetrises by mirroring (entries off the named
    diagonal are duplicated; entries on the diagonal are not).

    Parameters / Returns
    --------------------
    See ``sym2vec``.
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

    A convenience: if ``X`` looks square and symmetric (with rough
    tolerance), return ``sym2vec(X, offset=1)``; otherwise return
    ``vec2sym(X, offset=1)``.

    Notes
    -----
    Unlike ``scipy.spatial.distance.squareform``, we do not verify
    that the input is a strictly conforming form.  When in doubt,
    call ``sym2vec`` / ``vec2sym`` directly with the explicit
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
