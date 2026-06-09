# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Explicit penalties on a displacement / velocity field.

The diffeomorphic recipe in ``diffeomorphic`` regularises *implicitly*,
by Gaussian-smoothing the update (the Green's function of the
fluid/diffusion operator).  These are the *explicit* penalties a
learned or directly-optimised warp adds to its loss:

- ``gradient_smoothness`` -- the diffusion (first-order) penalty
  ``‖∇u‖²`` per voxel, summed over components: the squared Frobenius norm
  of the displacement Jacobian.  Penalises sharp / incoherent flow.
- ``bending_energy`` -- the thin-plate (second-order) penalty
  ``‖∇²u‖²``: the squared Frobenius norm of the per-voxel Hessian.
  Penalises curvature rather than slope (an affine flow is free).
- ``jacobian_folding_penalty`` -- penalises folding (loss of
  invertibility), ``relu(-det J)`` of the deformation Jacobian
  ``J = I + ∇u``: zero where the map is locally orientation-preserving
  (``det J > 0``), growing with the degree of fold (``det J <= 0``).

All take a channels-last field ``(*spatial, ndim)``, build on the shipped
``geometry`` differential operators, and are differentiable.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from .._internal.reductions import Reduction, reduce
from ..geometry import (
    jacobian_det_displacement,
    jacobian_displacement,
    spatial_gradient,
)

__all__ = [
    'gradient_smoothness',
    'bending_energy',
    'jacobian_folding_penalty',
]


def _grad_components(
    field: Float[Array, '*spatial k'], spatial_rank: int
) -> Float[Array, '*spatial k ndim']:
    """Spatial gradient of each trailing component: ``(..., k) -> (..., k, ndim)``."""
    grads = [
        spatial_gradient(field[..., c], spatial_rank=spatial_rank)
        for c in range(field.shape[-1])
    ]
    return jnp.stack(grads, axis=-2)


def gradient_smoothness(
    field: Float[Array, '*spatial ndim'],
    *,
    reduction: Reduction = 'mean',
) -> Float[Array, '...']:
    """Diffusion (first-order) smoothness penalty ``‖∇u‖²``.

    The squared Frobenius norm of the displacement Jacobian ``∇u``
    (recovered as ``J - I`` from :func:`geometry.jacobian_displacement`),
    summed over the ``ndim x ndim`` derivative entries at each voxel.

    Parameters
    ----------
    field
        Displacement field ``(*spatial, ndim)``.
    reduction
        ``"mean"`` (default), ``"sum"``, or ``"none"`` (the per-voxel
        energy map).
    """
    ndim = field.shape[-1]
    grad_u = jacobian_displacement(field) - jnp.eye(ndim, dtype=field.dtype)
    energy = jnp.sum(grad_u**2, axis=(-2, -1))
    return reduce(energy, reduction=reduction)


def bending_energy(
    field: Float[Array, '*spatial ndim'],
    *,
    reduction: Reduction = 'mean',
) -> Float[Array, '...']:
    """Thin-plate (second-order) bending penalty ``‖∇²u‖²``.

    The squared Frobenius norm of the per-voxel Hessian (all second
    partials ``∂²u_c / ∂x_a ∂x_b``), computed by differentiating the
    displacement Jacobian a second time.  Unlike
    :func:`gradient_smoothness` it does not penalise a uniform (affine)
    flow -- only curvature.

    Parameters
    ----------
    field
        Displacement field ``(*spatial, ndim)``.
    reduction
        ``"mean"`` (default), ``"sum"``, or ``"none"`` (the per-voxel
        energy map).
    """
    ndim = field.shape[-1]
    grad_u = jacobian_displacement(field) - jnp.eye(ndim, dtype=field.dtype)
    flat = grad_u.reshape(field.shape[:-1] + (ndim * ndim,))
    hessian = _grad_components(flat, ndim)  # (*spatial, ndim*ndim, ndim)
    energy = jnp.sum(hessian**2, axis=(-2, -1))
    return reduce(energy, reduction=reduction)


def jacobian_folding_penalty(
    field: Float[Array, '*spatial ndim'],
    *,
    reduction: Reduction = 'mean',
) -> Float[Array, '...']:
    """Folding penalty ``relu(-det J)`` of the deformation Jacobian.

    ``J = I + ∇u`` (via :func:`geometry.jacobian_det_displacement`); the
    per-voxel penalty is ``max(-det J, 0)`` -- zero where the map is
    locally orientation-preserving (``det J > 0``) and growing with the
    magnitude of any fold (``det J <= 0``).

    Parameters
    ----------
    field
        Displacement field ``(*spatial, ndim)``.
    reduction
        ``"mean"`` (default), ``"sum"``, or ``"none"`` (the per-voxel
        penalty map).
    """
    det = jacobian_det_displacement(field)
    return reduce(jnp.maximum(-det, 0.0), reduction=reduction)
