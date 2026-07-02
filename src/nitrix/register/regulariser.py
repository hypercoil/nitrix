# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Explicit penalties on a displacement / velocity field.

The diffeomorphic Demons recipe regularises *implicitly*, by
Gaussian-smoothing the update (the Green's function of the
fluid/diffusion operator). These are the *explicit* penalties a learned
or directly-optimised warp adds to its objective:

- :func:`gradient_smoothness` -- the diffusion (first-order) penalty
  :math:`\\|\\nabla u\\|^2` per voxel, summed over components: the
  squared Frobenius norm of the displacement Jacobian. Penalises sharp or
  incoherent flow.
- :func:`bending_energy` -- the thin-plate (second-order) penalty
  :math:`\\|\\nabla^2 u\\|^2`: the squared Frobenius norm of the
  per-voxel Hessian. Penalises curvature rather than slope (an affine
  flow is free).
- :func:`jacobian_folding_penalty` -- penalises folding (loss of
  invertibility) via :math:`\\operatorname{relu}(-\\det J)` of the
  deformation Jacobian :math:`J = I + \\nabla u`: zero where the map is
  locally orientation-preserving (:math:`\\det J > 0`), growing with the
  degree of fold (:math:`\\det J \\le 0`).

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
    """Spatial gradient of each trailing component of a field.

    Stacks the per-axis spatial gradient of every trailing component,
    mapping ``(*spatial, k) -> (*spatial, k, ndim)``.

    Parameters
    ----------
    field
        Channels-last field ``(*spatial, k)`` whose ``k`` trailing
        components are each differentiated over the ``spatial_rank``
        leading spatial axes.
    spatial_rank
        Number of leading spatial axes (``ndim``) over which the spatial
        gradient is taken.

    Returns
    -------
    Float[Array, '*spatial k ndim']
        The stacked spatial gradients, with a new trailing axis of length
        ``ndim`` holding the derivative of each component along each
        spatial axis.
    """
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
    """Diffusion (first-order) smoothness penalty :math:`\\|\\nabla u\\|^2`.

    The squared Frobenius norm of the displacement Jacobian
    :math:`\\nabla u` (recovered as :math:`J - I` from
    :func:`geometry.jacobian_displacement`), summed over the
    :math:`\\mathrm{ndim} \\times \\mathrm{ndim}` derivative entries at
    each voxel. Penalises sharp or incoherent flow.

    Parameters
    ----------
    field : Float[Array, '*spatial ndim']
        Channels-last displacement field ``(*spatial, ndim)``, with the
        trailing axis holding the ``ndim`` displacement components.
    reduction : {'mean', 'sum', 'none'}, optional
        How the per-voxel energy is reduced: ``"mean"`` (default) or
        ``"sum"`` collapse the spatial axes to a scalar, while ``"none"``
        returns the unreduced per-voxel energy map.

    Returns
    -------
    Float[Array, '...']
        The scalar total (for ``"mean"`` / ``"sum"``) or the per-voxel
        energy map of shape ``(*spatial,)`` (for ``"none"``).
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
    """Thin-plate (second-order) bending penalty :math:`\\|\\nabla^2 u\\|^2`.

    The squared Frobenius norm of the per-voxel Hessian (all second
    partials :math:`\\partial^2 u_c / \\partial x_a \\partial x_b`),
    computed by differentiating the displacement Jacobian a second time.
    Unlike :func:`gradient_smoothness` it does not penalise a uniform
    (affine) flow -- only curvature.

    Parameters
    ----------
    field : Float[Array, '*spatial ndim']
        Channels-last displacement field ``(*spatial, ndim)``, with the
        trailing axis holding the ``ndim`` displacement components.
    reduction : {'mean', 'sum', 'none'}, optional
        How the per-voxel energy is reduced: ``"mean"`` (default) or
        ``"sum"`` collapse the spatial axes to a scalar, while ``"none"``
        returns the unreduced per-voxel energy map.

    Returns
    -------
    Float[Array, '...']
        The scalar total (for ``"mean"`` / ``"sum"``) or the per-voxel
        energy map of shape ``(*spatial,)`` (for ``"none"``).
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
    """Folding penalty :math:`\\operatorname{relu}(-\\det J)` of the deformation Jacobian.

    The deformation Jacobian is :math:`J = I + \\nabla u`, whose
    determinant is obtained via :func:`geometry.jacobian_det_displacement`.
    The per-voxel penalty is :math:`\\max(-\\det J, 0)` -- zero where the
    map is locally orientation-preserving (:math:`\\det J > 0`) and
    growing with the magnitude of any fold (:math:`\\det J \\le 0`).

    Parameters
    ----------
    field : Float[Array, '*spatial ndim']
        Channels-last displacement field ``(*spatial, ndim)``, with the
        trailing axis holding the ``ndim`` displacement components.
    reduction : {'mean', 'sum', 'none'}, optional
        How the per-voxel penalty is reduced: ``"mean"`` (default) or
        ``"sum"`` collapse the spatial axes to a scalar, while ``"none"``
        returns the unreduced per-voxel penalty map.

    Returns
    -------
    Float[Array, '...']
        The scalar total (for ``"mean"`` / ``"sum"``) or the per-voxel
        penalty map of shape ``(*spatial,)`` (for ``"none"``).
    """
    det = jacobian_det_displacement(field)
    return reduce(jnp.maximum(-det, 0.0), reduction=reduction)
