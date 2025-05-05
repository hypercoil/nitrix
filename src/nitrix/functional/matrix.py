# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Special matrix functions.
"""

import math
from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from nitrix._internal import Tensor
from nitrix._internal.docutil import (
    DocTemplateFormat,
    form_docstring,
    tensor_dimensions,
)
from nitrix._internal.util import conform_mask, vmap_over_outer

# Functions *not* upstreamed from `hypercoil`:
# - `cholesky_invert`: In nearly all cases slower than a simple
#   `jnp.linalg.inv`.
# - `spd`: There are a number of different ways to force a matrix to be
#   positive semidefinite or to find the closest positive semidefinite matrix.
#   The `spd` function used a particularly crude approach, and so we leave the
#   implementation of this family of transformations to the user.
# - `expand_outer`: Just use `jnp.einsum` for this. It's nealry always more
#   readable than the implementation downstream.


@form_docstring
def document_symmetric() -> DocTemplateFormat:
    tensor_dim_spec = """

    | Dim | Description | Notes |
    |-----|-------------|-------|
    | $*$ | Any number of prefix, intervening, and suffix dimensions ||
    | $i$ | {desc_i} ||
    """.format(
        desc_i=(
            'Dimensions of the input tensor corresponding to the axes '
            'over which symmetry is imposed.'
        ),
    )
    tensor_dim_spec = tensor_dimensions(tensor_dim_spec)
    fmt = DocTemplateFormat(
        tensor_dim_spec=tensor_dim_spec,
    )
    return fmt


@form_docstring
def document_spd() -> DocTemplateFormat:
    tensor_dim_spec = """

    | Dim | Description | Notes |
    |-----|-------------|-------|
    | $*$ | Any number of prefix dimensions ||
    | $i$ | {desc_i} ||
    """.format(
        desc_i=(
            'Dimensions of the input tensor corresponding to the axes '
            'that together form the square positive semidefinite matrix.'
        ),
    )
    reconditioning_expr = r"""
    $A := A + \left(\psi - \frac{\xi}{2}\right) I + I\mathbf{x}$

    $x_i \sim \mathrm{Uniform}(0, \xi) \forall x_i$

    $\psi > \xi$"""
    tensor_dim_spec = tensor_dimensions(tensor_dim_spec)
    fmt = DocTemplateFormat(
        tensor_dim_spec=tensor_dim_spec,
        reconditioning_expr=reconditioning_expr,
    )
    return fmt


@form_docstring
def document_diagonal() -> DocTemplateFormat:
    tensor_dim_spec = """

    | Dim | Description | Notes |
    |-----|-------------|-------|
    | $*$ | Any number of prefix dimensions ||
    | $i$ | {desc_i} ||
    """.format(
        desc_i=('Size of the diagonal.'),
    )
    tensor_dim_spec = tensor_dimensions(tensor_dim_spec)
    size_plus_offset = r'i + |\mathrm{offset}|'
    fmt = DocTemplateFormat(
        tensor_dim_spec=tensor_dim_spec,
        size_plus_offset=size_plus_offset,
    )
    return fmt


@form_docstring
def document_toeplitz() -> DocTemplateFormat:
    tensor_dim_spec = """

    | Dim | Description | Notes |
    |-----|-------------|-------|
    | $*$ | Any number of prefix dimensions ||
    | $r$ | {desc_r} ||
    | $c$ | {desc_c} ||
    | $m$ | {desc_m} ||
    | $n$ | {desc_n} ||
    """.format(
        desc_r=('Size of the generating row vector.'),
        desc_c=('Size of the generating column vector.'),
        desc_m=('Size of the rows of the returned Toeplitz matrix.'),
        desc_n=('Size of the columns of the returned Toeplitz matrix.'),
    )
    tensor_dim_spec = tensor_dimensions(tensor_dim_spec)
    fmt = DocTemplateFormat(
        tensor_dim_spec=tensor_dim_spec,
    )
    return fmt


@form_docstring
def document_symform() -> DocTemplateFormat:
    tensor_dim_spec = """

    | Dim | Description | Notes |
    |-----|-------------|-------|
    | $*$ | Any number of prefix dimensions ||
    | $i$ | {desc_i} ||
    | $n$ | {desc_n} | {desc_n_note} |
    """.format(
        desc_i=('Size of the square matrix.'),
        desc_n=('Size of the vectorised form of the matrix.'),
        desc_n_note=(
            'This is equal to the number of elements in the specified '
            'upper triangle of the matrix, plus one. For the default '
            'offset of 1, this is equal to $\binom{i}{2}$ '
            'or equivalently $\frac{i(i - 1)}{2}$.'
        ),
    )
    tensor_dim_spec = tensor_dimensions(tensor_dim_spec)
    fmt = DocTemplateFormat(
        tensor_dim_spec=tensor_dim_spec,
    )
    return fmt


@document_symmetric
def symmetric(
    X: Tensor,
    skew: bool = False,
    axes: Tuple[int, int] = (-2, -1),
) -> Tensor:
    """
    Impose symmetry on a tensor block.

    The input tensor block is averaged with its transpose across the slice
    delineated by the specified axes.
    \
    {tensor_dim_spec}

    Parameters
    ----------
    X : ($*$, $i$, $*$, $i$, $*$) tensor
        Input to be symmetrised.
    skew : bool (default False)
        Indicates whether skew-symmetry (antisymmetry) should be imposed on
        the input.
    axes : tuple(int, int) (default (-2, -1))
        Axes that delineate the square slices of the input on which symmetry
        is imposed. By default, symmetry is imposed on the last 2 slices.

    Returns
    -------
    output : ($*$, $i$, $*$, $i$, $*$) tensor
        Input with symmetry imposed across specified slices.
    """
    if not skew:
        return (X + X.swapaxes(*axes)) / 2
    else:
        return (X - X.swapaxes(*axes)) / 2


@document_spd
def recondition_eigenspaces(
    A: Tensor,
    psi: float,
    xi: float = 0,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    r"""
    Recondition a positive semidefinite matrix such that it has no zero
    eigenvalues, and all of its eigenspaces have dimension one.

    This reconditioning operation should help stabilise differentiation
    through singular value decomposition at the cost of technical correctness.

    This operation modifies the input matrix A following

    {reconditioning_expr}
    \
    {tensor_dim_spec}

    Parameters
    ----------
    A : ($*$, $i$, $i$) tensor
        Matrix or matrix block to be reconditioned.
    psi : float
        Reconditioning parameter for ensuring nonzero eigenvalues.
    xi : float
        Reconditioning parameter for ensuring nondegenerate eigenvalues.
    key : jax.random.PRNGKey
        Random number generator key. Required if ``xi > 0``.

    Returns
    -------
    ($*$, $i$, $i$) tensor
        Reconditioned matrix or matrix block.
    """
    assert psi >= xi, (
        'Reconditioning parameter for ensuring nonzero eigenvalues must be '
        'greater than that for ensuring nondegenerate eigenvalues.'
    )
    if xi > 0:
        x = jax.random.uniform(key=key, shape=A.shape, maxval=xi)
    else:
        x = 0
    mask = jnp.eye(A.shape[-1])
    return A + (psi - xi + x) * mask


@document_diagonal
def delete_diagonal(A: Tensor) -> Tensor:
    """
    Delete the diagonal from a block of square matrices. Dimension is inferred
    from the final axis.
    \
    {tensor_dim_spec}

    Parameters
    ----------
    A : ($*$, $i$, $i$) tensor
        Block of square matrices.

    Returns
    -------
    ($*$, $i$, $i$) tensor
        Block of square matrices with the diagonal deleted.
    """
    mask = ~jnp.eye(A.shape[-1], dtype=bool)
    return A * mask


@document_diagonal
def diag_embed(v: Tensor, offset: int = 0) -> Tensor:
    """
    Embed a vector into a diagonal matrix.
    \
    {tensor_dim_spec}

    Parameters
    ----------
    v : ($*$, $i$) tensor
        Vector to embed.
    offset : int (default 0)
        Offset from the main diagonal where the vector should be placed.

    Returns
    -------
    ($*$, {size_plus_offset}, {size_plus_offset}) tensor
        Diagonal matrix with the vector embedded.
    """
    return vmap_over_outer(partial(jnp.diagflat, k=offset), 1)((v,))


@document_diagonal
def fill_diagonal(A: Tensor, fill: float = 0, offset: int = 0) -> Tensor:
    """
    Fill a selected diagonal in a block of square matrices. Dimension is
    inferred from the final axes.
    \
    {tensor_dim_spec}

    Parameters
    ----------
    A : ($*$, $i$, $i$) tensor
        Block of square matrices.
    fill : float (default 0)
        Value to fill the diagonal with.
    offset : int (default 0)
        Offset from the main diagonal where the diagonal should be filled.

    Returns
    -------
    ($*$, $i$, $i$) tensor
        Block of square matrices with the diagonal filled.
    """
    dim = A.shape[-2:]
    mask = jnp.ones(max(dim) - abs(offset), dtype=bool)
    mask = diag_embed(mask, offset=offset)
    mask = mask[: dim[0], : dim[1]]
    mask = conform_mask(A, mask, axis=(-2, -1))
    return jnp.where(mask, fill, A)


@document_toeplitz
def toeplitz_2d(
    c: Tensor,
    r: Optional[Tensor] = None,
    shape: Optional[Tuple[int, int]] = None,
    fill_value: float = 0,
) -> Tensor:
    """
    Construct a 2D Toeplitz matrix from a column and row vector.

    Based on the second method posted by @mattjj and evaluated by
    @blakehechtman here:
    https://github.com/google/jax/issues/1646#issuecomment-1139044324
    Apparently this method underperforms on TPU. But given that TPUs are
    only available for Google, this is probably not a big deal.

    Our method is more flexible in that it admits Toeplitz matrices without
    circulant structure. Our API is also closer to that of the scipy toeplitz
    function. We also support a fill value for the matrix. See
    :func:`toeplitz` for complete API documentation.

    .. note::

        Use the :func:`toeplitz` function for an API that supports any number
        of leading dimensions.

    .. note::

        The input vectors ``c`` and ``r`` must contain the same first element
        for functionality to match ``scipy.toeplitz``. This is not checked.
        In the event that this is not the case, ``c[0]`` is ignored.
        Note that this is the opposite of ``scipy.toeplitz``.
    \
    {tensor_dim_spec}

    Parameters
    ----------
    c : ($c$,) tensor
        Column vector of the Toeplitz matrix.
    r : ($r$,) tensor
        Row vector of the Toeplitz matrix.
    shape : tuple(int, int) or None (default)
        Shape of the Toeplitz matrix. Sets dimensions of the output tensor.
    fill_value : float (default 0)
        Value to fill the Toeplitz matrix with.

    Returns
    -------
    ($m$, $n$) tensor
        Toeplitz matrix of shape specified by `shape`.
    """
    if r is None:
        r = c
    m_in, n_in = c.shape[-1], r.shape[-1]
    m, n = shape if shape is not None else (m_in, n_in)
    d = max(m, n)
    if (m != n) or (m_in != n_in) or (m_in != d):
        r_arg = jnp.full(d, fill_value, dtype=r.dtype).at[:n_in].set(r)
        c_arg = jnp.full(d, fill_value, dtype=c.dtype).at[:m_in].set(c)
        # if fill_value!= 0 : breakpoint()
    else:
        r_arg, c_arg = r, c
    c_arg = jnp.flip(c_arg, -1)

    mask = jnp.zeros(2 * d - 1, dtype=bool)
    mask = mask.at[: (d - 1)].set(True)
    iota = jnp.arange(d)

    def roll(c, r, i, mask):
        rs = jnp.roll(r, i, axis=-1)
        cs = jnp.roll(c, i + 1, axis=-1)
        ms = jnp.roll(mask, i, axis=-1)[-d:]
        return jnp.where(ms, cs, rs)

    f = jax.vmap(roll, in_axes=(None, None, 0, None))
    return f(c_arg, r_arg, iota[..., None], mask)[..., :m, :n]


@document_toeplitz
def toeplitz(
    c: Tensor,
    r: Optional[Tensor] = None,
    shape: Optional[Tuple[int, int]] = None,
    fill_value: float = 0.0,
) -> Tensor:
    r"""
    Populate a block of tensors with Toeplitz banded structure.

    .. warning::

        Inputs ``c`` and ``r`` must contain the same first element for
        functionality to match ``scipy.toeplitz``. This is not checked.
        In the event that this is not the case, ``c[0]`` is ignored.
        Note that this is the opposite of ``scipy.toeplitz``.
    \
    {tensor_dim_spec}

    Parameters
    ----------
    c : ($*$, $c$) tensor
        Tensor of entries in the first column of each Toeplitz matrix. The
        first axis corresponds to a single matrix column; additional
        dimensions correspond to concatenation of Toeplitz matrices into a
        stack or block tensor.
    r : ($*$, $r$) tensor
        Tensor of entries in the first row of each Toeplitz matrix. The first
        axis corresponds to a single matrix row; additional dimensions
        correspond to concatenation of Toeplitz matrices into a stack or block
        tensor. The first entry in each column should be the same as the first
        entry in the corresponding column of `c`; otherwise, it will be
        ignored.
    shape : tuple(int, int) or None (default)
        Dimension of each Toeplitz banded matrix in the output block. If this
        is None or unspecified, it defaults to the sizes of the first axes of
        inputs `c` and `r`. Otherwise, the row and column inputs are extended
        until their dimensions equal those specified here. This can be useful,
        for instance, to create a large banded matrix with mostly zero
        off-diagonals.
    fill_value: Tensor or float (default 0)
        Specifies the value that should be used to populate the off-diagonals
        of each Toeplitz matrix if the specified row and column elements are
        extended to conform with the specified `dim`. If this is a tensor,
        then each entry corresponds to the fill value in a different data
        channel. Has no effect if ``dim`` is None.

    Returns
    -------
    out : ($*$, $m$, $n$) tensor
        Block of Toeplitz matrices populated from the specified row and column
        elements.
    """
    return vmap_over_outer(
        partial(toeplitz_2d, shape=shape, fill_value=fill_value), 1
    )((c, r))


@document_symform
@partial(jax.custom_vjp, nondiff_argnums=(1,))
def sym2vec(sym: Tensor, offset: int = 1) -> Tensor:
    """
    Convert a block of symmetric matrices into ravelled vector form.

    Ordering in the ravelled form follows row-major order of the upper
    triangle of the matrix block.
    \
    {tensor_dim_spec}

    Parameters
    ----------
    sym : ($*$, $i$, $i$) tensor
        Block of tensors to convert. The last two dimensions should be equal,
        with each slice along the final 2 axes being a square, symmetric
        matrix.
    offset : int (default 1)
        Offset from the main diagonal where the upper triangle begins. By
        default, the main diagonal is not included. Set this to 0 to include
        the main diagonal.

    Returns
    -------
    vec : ($*$, $n$) tensor
        Block of ravelled vectors formed from the upper triangles of the input
        `sym`, beginning with the diagonal offset from the main by the input
        `offset`.
    """
    idx = jnp.triu_indices(m=sym.shape[-2], n=sym.shape[-1], k=offset)
    shape = sym.shape[:-2]
    # print(idx, shape)
    vec = sym[..., idx[0], idx[1]]
    return vec.reshape(*shape, -1)


@document_symform
@partial(jax.custom_vjp, nondiff_argnums=(1,))
def vec2sym(vec: Tensor, offset: int = 1) -> Tensor:
    """
    Convert a block of ravelled vectors into symmetric matrices.

    The ordering of the input vectors should follow the upper triangle of the
    matrices to be formed.
    \
    {tensor_dim_spec}

    Parameters
    ----------
    vec : ($*$, $n$) tensor
        Block of vectors to convert. Input vectors should be of length
        (n choose 2), where n is the number of elements on the offset
        diagonal, plus 1.
    offset : int (default 1)
        Offset from the main diagonal where the upper triangle begins. By
        default, the main diagonal is not included. Set this to 0 to place
        elements along the main diagonal.

    Returns
    -------
    sym : ($*$, $i$, $i$) tensor
        Block of symmetric matrices formed by first populating the offset
        upper triangle with the elements from the input `vec`, then
        symmetrising.
    """
    shape = vec.shape[:-1]
    vec = vec.reshape(*shape, -1)
    cn2 = vec.shape[-1]
    side = int(0.5 * (math.sqrt(8 * cn2 + 1) + 1)) + (offset - 1)
    idx = jnp.triu_indices(m=side, n=side, k=offset)
    sym = jnp.zeros((*shape, side, side))
    sym = sym.at[..., idx[0], idx[1]].set(vec)
    sym = jnp.where(jnp.eye(side, dtype=bool), sym, sym + sym.swapaxes(-1, -2))
    return sym


@document_symform
def squareform(X: Tensor) -> Tensor:
    """
    Convert between symmetric matrix and vector forms.

    .. warning::
        Unlike numpy or matlab implementations, this does not verify a
        conformant input.
    \
    {tensor_dim_spec}

    Parameters
    ----------
    X : ($*$, $i$, $i$) tensor or ($*$, $n$) tensor
        Block of symmetric matrices, in either square matrix or vectorised
        form.

    Returns
    -------
    ($*$, $i$, $i$) tensor or ($*$, $n$) tensor
        If the input block is in square matrix form, returns it
        :doc:`in vector form <nitrix.functional.matrix.sym2vec>`.
        If the input block is in vector form, returns it
        :doc:`in square matrix form <nitrix.functional.matrix.vec2sym>`.
    """
    if X.shape[-2] == X.shape[-1] and jnp.allclose(X, X.swapaxes(-1, -2)):
        return sym2vec(X, offset=1)
    else:
        return vec2sym(X, offset=1)


def sym2vec_fwd(sym, offset):
    return sym2vec(sym, offset=offset), None


def sym2vec_bwd(offset, res, g):
    unscaled = vec2sym(g, offset=offset)
    scalar = jnp.triu(jnp.ones(unscaled.shape[-2:]), offset)
    return (unscaled * scalar,)


def vec2sym_fwd(vec, offset):
    return vec2sym(vec, offset=offset), None


def vec2sym_bwd(offset, res, g):
    scalar = jnp.where(jnp.eye(g.shape[-1]), 1, 2)
    scalar = sym2vec(scalar, offset=offset)
    return (scalar * sym2vec(symmetric(g), offset=offset),)


sym2vec.defvjp(sym2vec_fwd, sym2vec_bwd)
vec2sym.defvjp(vec2sym_fwd, vec2sym_bwd)
