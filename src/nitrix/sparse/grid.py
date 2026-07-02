# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Regular-grid stencil specialisation of the ELL sparse format.

When every voxel of a regular n-D grid uses the same neighbour offset pattern
(the "stencil"), the resulting linear operator has a special structure: row
:math:`i` indexes neighbours at fixed offsets from row :math:`i`'s spatial
coordinate. This is stored as a plain :class:`~nitrix.sparse.ELL` with:

- ``n_rows = prod(grid_shape)`` (one row per voxel).
- ``k_max = len(offsets)`` (one column per stencil tap).
- ``indices[v, k]`` is the linearised index of the ``k``-th neighbour of
  voxel ``v``.
- ``values[v, k]`` is the stencil weight at offset ``k``.

This is the substrate equivalent of the regular finite-difference operators
(the 5-point Laplacian, Sobel filters, and so on). Once constructed, the
machinery of :func:`~nitrix.semiring.semiring_ell_matmul`,
:func:`~nitrix.graph.laplacian_eigenmap`, and related routines flows naturally.

The public surface consists of :func:`regular_grid_stencil` (the general n-D
stencil constructor with replicate / periodic / reflect boundary modes),
:func:`grid_laplacian` (a convenience for the (2n+1)-point n-D Laplacian on a
regular grid), and :func:`grid_identity` (the identity operator, trivial but
composable with :func:`grid_laplacian` to form :math:`I - L` smoothers).

Boundary handling
-----------------
- ``'replicate'``: at the grid edge, the out-of-bounds neighbour index is
  clamped to the edge cell. This effectively duplicates the boundary value
  when applied via matmul, matching scipy ``mode='nearest'``.
- ``'periodic'``: wrap around (toroidal topology), matching scipy
  ``mode='wrap'``.
- ``'reflect'``: mirror at the edge with the edge cell repeated, matching
  scipy ``mode='reflect'`` (numpy ``mode='symmetric'``).

For more exotic topologies (e.g., a parameterised sphere with pole flip),
pre-pad the spatial axes via the routines in :mod:`nitrix.geometry.sphere_grid`
and construct the stencil on the padded grid.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Sequence, Tuple, Union, cast

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Num
from numpy.typing import NDArray

from .ell import ELL

__all__ = [
    'regular_grid_stencil',
    'grid_laplacian',
    'grid_identity',
]


BoundaryMode = Literal['replicate', 'periodic', 'reflect']


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _linearise_offsets(
    grid_shape: Tuple[int, ...],
    offsets: NDArray[Any],
    boundary: BoundaryMode,
) -> NDArray[Any]:
    """Map per-voxel and per-offset neighbour coordinates to linear indices.

    For each voxel of the grid, the stencil offsets are added to the voxel's
    spatial coordinate, the named boundary handling is applied at the grid
    edges, and the resulting neighbour coordinates are flattened to linear
    (C-order) indices.

    This is done in NumPy at construction time, off any JIT path. The result
    is a fixed integer array passed to :class:`~nitrix.sparse.ELL`; the actual
    matmul (via :func:`~nitrix.semiring.semiring_ell_matmul`) is pure JAX.

    Parameters
    ----------
    grid_shape
        Spatial grid dimensions, with voxels linearised in C order. Its length
        is the grid dimensionality ``ndim``.
    offsets
        Integer offsets of shape ``(n_offsets, ndim)`` defining the stencil
        relative to each voxel. The second axis must match ``len(grid_shape)``.
    boundary
        Boundary handling at the grid edges, one of ``'replicate'``,
        ``'periodic'`` or ``'reflect'``. See the module docstring.

    Returns
    -------
    NDArray
        Integer (``int32``) array of shape ``(n_voxels, n_offsets)`` where
        ``n_voxels = prod(grid_shape)`` and entry ``[v, k]`` is the linearised
        C-order index of the ``k``-th neighbour of voxel ``v`` after boundary
        handling.
    """
    ndim = len(grid_shape)
    if offsets.shape[1] != ndim:
        raise ValueError(
            f'offsets.shape[1]={offsets.shape[1]} must match '
            f'len(grid_shape)={ndim}.'
        )
    n_offsets = offsets.shape[0]
    n_voxels = int(np.prod(grid_shape))
    grid_shape_arr = np.asarray(grid_shape, dtype=np.int64)

    # Build the (n_voxels, ndim) coordinate grid via NumPy meshgrid.
    coord_axes = [np.arange(s, dtype=np.int64) for s in grid_shape]
    coords = np.stack(np.meshgrid(*coord_axes, indexing='ij'), axis=-1)
    coords = coords.reshape(n_voxels, ndim)  # (n_voxels, ndim)

    # Neighbour coordinates: (n_voxels, n_offsets, ndim)
    neigh = coords[:, None, :] + offsets[None, :, :]
    if boundary == 'replicate':
        neigh = np.clip(neigh, 0, grid_shape_arr[None, None, :] - 1)
    elif boundary == 'periodic':
        neigh = neigh % grid_shape_arr[None, None, :]
    elif boundary == 'reflect':
        # scipy.ndimage convention: 'reflect' is "with edge repetition"
        # (half-sample symmetric): (d c b a | a b c d | d c b a).
        # Formula: period = 2 * n; x % period; if result >= n,
        # mirror to (period - 1 - result).
        period = 2 * grid_shape_arr
        period = np.maximum(period, 1)
        neigh = neigh % period[None, None, :]
        neigh = np.where(
            neigh >= grid_shape_arr[None, None, :],
            period[None, None, :] - 1 - neigh,
            neigh,
        )
    else:
        raise ValueError(
            f"boundary={boundary!r}; expected 'replicate', 'periodic', "
            "or 'reflect'."
        )

    # Linearise via NumPy ravel_multi_index for the C-order convention.
    neigh_flat = neigh.reshape(-1, ndim)
    linear = np.ravel_multi_index(
        [neigh_flat[:, d] for d in range(ndim)],
        dims=grid_shape,
        order='C',
    )
    # ``np.ravel_multi_index`` is typed as returning Any here; restore.
    return cast(
        NDArray[Any], linear.reshape(n_voxels, n_offsets).astype(np.int32)
    )


# ---------------------------------------------------------------------------
# Public constructors
# ---------------------------------------------------------------------------


def regular_grid_stencil(
    grid_shape: Sequence[int],
    offsets: Sequence[Sequence[int]],
    weights: Num[Array, 'n_offsets'],
    *,
    boundary: BoundaryMode = 'replicate',
    n_cols: Optional[int] = None,
    identity: Any = 0.0,
) -> ELL:
    """Build an ELL for an n-D regular-grid stencil.

    The result represents a linear operator ``Op`` such that
    ``(Op @ x)[v] = sum_k weights[k] * x[neighbour(v, offsets[k])]``,
    where ``neighbour`` applies the named boundary handling at the
    grid edges.

    Parameters
    ----------
    grid_shape
        Spatial grid dimensions, e.g. ``(D, H, W)`` for a 3-D
        volume.  Voxels are linearised in C order.
    offsets
        ``(n_offsets, ndim)`` integer offsets defining the stencil
        relative to each voxel.  E.g. for a 2-D 5-point Laplacian
        stencil:
        ``[[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]``.
    weights
        Per-offset stencil weights, length ``n_offsets``.  May be a
        jax or numpy array; promoted to the ELL ``values`` dtype.
    boundary
        Boundary handling at the grid edges.  See module docstring.
        Default ``'replicate'``.
    n_cols
        Optional override for the ELL ``n_cols``.  Defaults to
        ``prod(grid_shape)`` -- the natural self-coupling case.
        Set explicitly when the operator maps the grid to a
        different-sized domain.
    identity
        Semiring identity for the ELL.  Default ``0.0`` (REAL
        identity).

    Returns
    -------
    ELL with shape ``(prod(grid_shape), n_cols)`` and ``k_max =
    n_offsets``.
    """
    grid_shape_t = tuple(int(s) for s in grid_shape)
    offsets_arr = np.asarray(offsets, dtype=np.int64)
    if offsets_arr.ndim != 2:
        raise ValueError(
            f'offsets must be 2-D (n_offsets, ndim); got shape '
            f'{offsets_arr.shape}.'
        )
    n_voxels = int(np.prod(grid_shape_t))
    n_offsets = offsets_arr.shape[0]
    if n_cols is None:
        n_cols = n_voxels
    indices_np = _linearise_offsets(grid_shape_t, offsets_arr, boundary)

    # Tile per-offset weights across all voxels.
    weights_arr = jnp.asarray(weights)
    if weights_arr.shape != (n_offsets,):
        raise ValueError(
            f'weights must have shape ({n_offsets},); got {weights_arr.shape}.'
        )
    values = jnp.broadcast_to(weights_arr[None, :], (n_voxels, n_offsets))

    return ELL(
        values=values,
        indices=jnp.asarray(indices_np),
        n_cols=int(n_cols),
        identity=identity,
    )


def _axis_unit_offsets(ndim: int) -> NDArray[Any]:
    """Return the :math:`2\\,\\mathrm{ndim} + 1` axis-aligned offsets.

    The offsets are the origin together with the positive and negative unit
    steps along each axis, i.e. :math:`\\{0,\\ \\pm e_d\\}` for every axis
    :math:`d`. They are ordered as centre first, then
    :math:`(+e_0, -e_0, +e_1, -e_1, \\dots)`.

    Parameters
    ----------
    ndim
        Number of spatial dimensions of the grid.

    Returns
    -------
    NDArray
        Integer array of shape ``(2 * ndim + 1, ndim)`` holding the stencil
        offsets in the order described above.
    """
    rows = [np.zeros(ndim, dtype=np.int64)]
    for d in range(ndim):
        for sign in (1, -1):
            v = np.zeros(ndim, dtype=np.int64)
            v[d] = sign
            rows.append(v)
    return np.stack(rows, axis=0)


def grid_laplacian(
    grid_shape: Sequence[int],
    *,
    boundary: BoundaryMode = 'replicate',
    spacing: Union[float, Sequence[float]] = 1.0,
    sign: Literal['positive', 'negative'] = 'negative',
) -> ELL:
    """The standard (2n+1)-point n-D Laplacian stencil on a regular grid.

    This builds the classic second-order finite-difference Laplacian: the
    5-point stencil in 2-D, the 7-point stencil in 3-D, and so on. In 1-D the
    stencil is :math:`\\Delta u(i) = u(i-1) - 2\\,u(i) + u(i+1)`; in 2-D each
    voxel is coupled to its four axis-aligned neighbours with a centre weight
    of :math:`-4` (for unit spacing).

    Parameters
    ----------
    grid_shape
        Spatial grid dimensions, with voxels linearised in C order.
    boundary
        Boundary handling. ``'replicate'`` is the standard free-boundary
        convention (zero flux); ``'periodic'`` gives a toroidal Laplacian;
        ``'reflect'`` gives a Neumann-like condition with the boundary
        mirrored. Default ``'replicate'``.
    spacing
        Voxel spacing. A ``float`` is isotropic; a sequence gives per-axis
        spacing for anisotropic grids (e.g., fMRI ``(3, 3, 4)``). The
        Laplacian weight along each axis is scaled by :math:`1/h^2`, where
        :math:`h` is that axis's spacing. Default ``1.0``.
    sign
        ``'negative'`` (default) gives the standard discrete Laplacian
        :math:`\\Delta u`, whose eigenvalues are :math:`\\le 0` on a finite
        grid (matching the continuous Laplacian sign). ``'positive'`` gives
        :math:`-\\Delta u`, the positive-definite form used in spectral graph
        theory.

    Returns
    -------
    ELL
        An :class:`~nitrix.sparse.ELL` operator of shape
        ``(prod(grid_shape), prod(grid_shape))`` with :math:`2\\,n + 1` taps
        per row, representing the Laplacian.
    """
    grid_shape_t = tuple(int(s) for s in grid_shape)
    ndim = len(grid_shape_t)

    if isinstance(spacing, (int, float)):
        spacing_per_axis = np.full(ndim, float(spacing))
    else:
        spacing_per_axis = np.asarray(spacing, dtype=np.float64)
        if spacing_per_axis.shape != (ndim,):
            raise ValueError(
                f'spacing must be scalar or length-{ndim}; got '
                f'{spacing_per_axis.shape}.'
            )

    inv_h2 = 1.0 / (spacing_per_axis**2)
    offsets = _axis_unit_offsets(ndim)
    # Weights: centre = -sum(2 * 1/h^2); +e_d and -e_d each get 1/h^2.
    weights_np = np.empty(offsets.shape[0], dtype=np.float64)
    weights_np[0] = -2.0 * inv_h2.sum()
    for d in range(ndim):
        weights_np[1 + 2 * d] = inv_h2[d]
        weights_np[2 + 2 * d] = inv_h2[d]
    if sign == 'positive':
        weights_np = -weights_np
    elif sign != 'negative':
        raise ValueError(f"sign={sign!r}; expected 'negative' or 'positive'.")

    return regular_grid_stencil(
        grid_shape_t,
        offsets.tolist(),
        jnp.asarray(weights_np),
        boundary=boundary,
    )


def grid_identity(
    grid_shape: Sequence[int],
    *,
    n_cols: Optional[int] = None,
) -> ELL:
    """The identity operator on a regular grid.

    Returned as an :class:`~nitrix.sparse.ELL` with a single tap per row (the
    centre offset, with unit weight). This is composable with
    :func:`grid_laplacian` and other stencil operators to form "shifted
    Laplacian" preconditioners and "implicit smoothing" operators of the form
    :math:`I - \\alpha L`.

    Parameters
    ----------
    grid_shape
        Spatial grid dimensions, with voxels linearised in C order.
    n_cols
        Optional override for the number of columns of the returned operator.
        Defaults to ``prod(grid_shape)`` (the square self-coupling case). Set
        explicitly when the operator maps the grid into a different-sized
        domain.

    Returns
    -------
    ELL
        An :class:`~nitrix.sparse.ELL` operator of shape
        ``(prod(grid_shape), n_cols)`` with one tap per row (weight ``1.0`` on
        the centre voxel).
    """
    grid_shape_t = tuple(int(s) for s in grid_shape)
    ndim = len(grid_shape_t)
    offsets = [[0] * ndim]
    weights = jnp.ones(1, dtype=jnp.float32)
    return regular_grid_stencil(
        grid_shape_t,
        offsets,
        weights,
        boundary='replicate',
        n_cols=n_cols,
    )
