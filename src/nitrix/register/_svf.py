# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Shared substrate for the stationary-velocity-field (SVF) recipes.

The log-Demons (``diffeomorphic``) and greedy-SyN (``_syn``) recipes are
the same family: a coarse-to-fine flow of one or two voxel-unit velocity
fields, regularised by separable Gaussian smoothing, with the only
difference being the per-level update.  This module holds what they share
-- the anisotropy-aware smoothing helpers and the coarse-to-fine scaffold
(``svf_coarse_to_fine``) -- so each recipe is just its per-level update
plus its finalisation, not a re-derived multiresolution loop.

(The matrix-transform driver, by contrast, carries a small parameter
*vector* and a coordinate-space sampler, not a velocity *field*, so it is
a genuinely different state machine and stays in ``_core`` -- the SVF
unification is the warranted one, not a forced matrix+SVF merge.)
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Union

import jax.numpy as jnp
from jaxtyping import Array

from ..geometry import upsample
from ..smoothing import gaussian

__all__ = [
    'svf_coarse_to_fine',
]


def _smooth_vector(
    field: Array, sigma: Union[float, Sequence[float]], ndim: int
) -> Array:
    """Separable Gaussian over the spatial axes of a channel-last field.

    ``sigma`` is a scalar (isotropic) or a length-``ndim`` per-axis
    sequence (anisotropic regularisation).
    """
    moved = jnp.moveaxis(field, -1, 0)
    smoothed = gaussian(moved, sigma=sigma, spatial_rank=ndim)
    return jnp.moveaxis(smoothed, 0, -1)


def _spacing_tuple(
    spacing: Union[float, Sequence[float]], ndim: int
) -> tuple[float, ...]:
    if isinstance(spacing, (int, float)):
        return (float(spacing),) * ndim
    out = tuple(float(s) for s in spacing)
    if len(out) != ndim:
        raise ValueError(
            f'spacing must be a scalar or a length-{ndim} sequence; '
            f'got {spacing!r}.'
        )
    return out


def _relative_spacing(
    spacing: Optional[Union[float, Sequence[float]]], ndim: int
) -> Optional[tuple[float, ...]]:
    """Anisotropy-only spacing ``spacing / geomean(spacing)``.

    Level-independent (the coarse-to-fine align-corners scale cancels in
    the ratio) and ``1`` for isotropic spacing, so the regularisation /
    force only see the per-axis *anisotropy*, not an absolute scale --
    isotropic data is unchanged.
    """
    if spacing is None:
        return None
    sp = _spacing_tuple(spacing, ndim)
    geomean = math.prod(sp) ** (1.0 / ndim)
    rel = tuple(s / geomean for s in sp)
    if all(r == 1.0 for r in rel):
        return None
    return rel


# Per-level update: ``(moving_level, fixed_level, state) -> (state, cost)``
# where ``state`` is a tuple of ``n_fields`` velocity fields.
LevelSolve = Callable[
    [Array, Array, tuple[Array, ...]], tuple[tuple[Array, ...], Array]
]


def svf_coarse_to_fine(
    pyr_m: tuple[Array, ...],
    pyr_f: tuple[Array, ...],
    *,
    ndim: int,
    dtype: jnp.dtype,
    n_fields: int,
    level_solve: LevelSolve,
) -> tuple[tuple[Array, ...], Array]:
    """Coarse-to-fine driver for the SVF recipes.

    Carries a tuple of ``n_fields`` voxel-unit velocity fields from the
    coarsest pyramid level to the finest, prolonging between levels
    (interpolating upsample + the per-axis voxel-scale that keeps the field
    in voxel units), and delegating the per-level update to ``level_solve``.

    Parameters
    ----------
    pyr_m, pyr_f
        Channel-last Gaussian pyramids (finest first) of the moving / fixed
        images.
    ndim, dtype
        Spatial rank and dtype of the velocity fields.
    n_fields
        Number of velocity fields carried (1 for Demons, 2 for symmetric
        SyN).
    level_solve
        ``(moving_level, fixed_level, state) -> (state, cost_trace)`` for
        one resolution; ``state`` is the ``n_fields``-tuple of velocities.

    Returns
    -------
    ``(state, cost_history)`` -- the finest-level velocity tuple and the
    concatenated per-level cost traces.
    """
    state: Optional[tuple[Array, ...]] = None
    prev_shape: Optional[tuple[int, ...]] = None
    histories = []
    for level in range(len(pyr_m) - 1, -1, -1):
        m_l = pyr_m[level][..., 0]
        f_l = pyr_f[level][..., 0]
        shape_l = f_l.shape
        if state is None:
            state = tuple(
                jnp.zeros(shape_l + (ndim,), dtype=dtype)
                for _ in range(n_fields)
            )
        else:
            ratio = jnp.asarray(shape_l, dtype=dtype) / jnp.asarray(
                prev_shape, dtype=dtype
            )
            state = tuple(upsample(s, shape_l) * ratio for s in state)
        state, hist = level_solve(m_l, f_l, state)
        histories.append(hist)
        prev_shape = shape_l

    assert state is not None
    return state, jnp.concatenate(histories)
