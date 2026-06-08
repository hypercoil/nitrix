# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Deformation- and velocity-field algebra.

The operations the diffeomorphic (log-Demons) recipe needs on top of the
SVF stack (``integrate_velocity_field`` is the exponential ``exp(v)``):

- ``compose_displacement`` -- compose two displacement fields,
  ``(id + u) ∘ (id + v)``, the warp-by-then-warp operation.
- ``compose_velocity`` -- the BCH approximation of the velocity whose
  exponential is ``exp(v) ∘ exp(u)`` (the log-domain update); first
  order is plain addition, second order adds ½ the Lie bracket.
- ``invert_displacement`` -- the inverse displacement, as the fixed
  point ``s_inv = -s ∘ (id + s_inv)`` (``numerics.fixed_point_solve``,
  so it is differentiable).

Channel-last fields ``(*spatial, ndim)``; coordinates index-space
(``identity_grid`` convention).
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..numerics.fixed_point import fixed_point_solve
from ._interpolate import BoundaryMode
from .grid import identity_grid, jacobian_displacement, spatial_transform

__all__ = [
    'compose_displacement',
    'compose_velocity',
    'invert_displacement',
]


def compose_displacement(
    outer: Float[Array, '*spatial ndim'],
    inner: Float[Array, '*spatial ndim'],
    *,
    mode: BoundaryMode = 'nearest',
) -> Float[Array, '*spatial ndim']:
    """Displacement of ``(id + outer) ∘ (id + inner)``.

    The composed deformation maps ``x -> x + inner(x) + outer(x +
    inner(x))``, so the displacement is ``inner + outer∘(id + inner)``.
    Warping an image by the result equals warping by ``inner`` then by
    ``outer``.

    Parameters
    ----------
    outer, inner
        Displacement fields, ``(*spatial, ndim)`` (the trailing axis is
        the displacement vector).
    mode
        Boundary mode for sampling ``outer`` at the deformed positions
        (default ``"nearest"`` -- edge-replicate, the flow-field
        convention).
    """
    spatial_shape = inner.shape[:-1]
    grid = identity_grid(spatial_shape, dtype=inner.dtype) + inner
    warped_outer = spatial_transform(outer, grid, mode=mode)
    return inner + warped_outer


def _grad_field(
    field: Float[Array, '*spatial ndim'],
) -> Float[Array, '*spatial ndim ndim']:
    """Spatial Jacobian ``∂ field_i / ∂ x_j`` (``[..., i, j]``)."""
    ndim = field.shape[-1]
    eye = jnp.eye(ndim, dtype=field.dtype)
    return jacobian_displacement(field) - eye


def _lie_bracket(
    v: Float[Array, '*spatial ndim'],
    u: Float[Array, '*spatial ndim'],
) -> Float[Array, '*spatial ndim']:
    """Lie bracket ``[v, u] = (v·∇)u - (u·∇)v`` of two velocity fields."""
    du = _grad_field(u)
    dv = _grad_field(v)
    du_v = jnp.einsum('...ij,...j->...i', du, v)
    dv_u = jnp.einsum('...ij,...j->...i', dv, u)
    return du_v - dv_u


def compose_velocity(
    v: Float[Array, '*spatial ndim'],
    u: Float[Array, '*spatial ndim'],
    *,
    order: int = 1,
) -> Float[Array, '*spatial ndim']:
    """BCH composition of stationary velocity fields.

    Approximates the velocity ``z`` with ``exp(z) ≈ exp(v) ∘ exp(u)``:

    - ``order == 1`` -- ``z = v + u`` (the standard additive log-domain
      update; exact when the fields commute).
    - ``order == 2`` -- ``z = v + u + ½ [v, u]`` (the first
      Baker-Campbell-Hausdorff correction).

    Default ``order == 1``: the additive update most diffeomorphic-demons
    implementations use.
    """
    if order == 1:
        return v + u
    if order == 2:
        return v + u + 0.5 * _lie_bracket(v, u)
    raise ValueError(f'order must be 1 or 2; got {order}.')


def invert_displacement(
    s: Float[Array, '*spatial ndim'],
    *,
    tol: float = 1e-5,
    max_iter: int = 50,
    mode: BoundaryMode = 'nearest',
) -> Float[Array, '*spatial ndim']:
    """Inverse displacement field of ``φ = id + s``.

    Returns ``s_inv`` with ``(id + s) ∘ (id + s_inv) ≈ id``, found as the
    fixed point ``s_inv = -s ∘ (id + s_inv)`` via
    ``numerics.fixed_point_solve`` (so it is differentiable w.r.t. ``s``
    by the implicit-function theorem).  Converges for displacements with
    ``‖∇s‖ < 1`` (the diffeomorphic regime).

    Parameters
    ----------
    s
        Displacement field, ``(*spatial, ndim)``.
    tol, max_iter
        Fixed-point convergence controls.
    mode
        Boundary mode for the inner sampling.
    """
    spatial_shape = s.shape[:-1]

    def update(field: Array, s_inv: Array) -> Array:
        grid = identity_grid(spatial_shape, dtype=field.dtype) + s_inv
        return -spatial_transform(field, grid, mode=mode)

    return fixed_point_solve(
        update, s, jnp.zeros_like(s), tol=tol, max_iter=max_iter
    )
