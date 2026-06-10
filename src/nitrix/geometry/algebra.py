# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Transform algebra: Lie-group barycentre, geodesic interpolation, fusion.

Beyond the exp/log chart (``transform``) and the pairwise compose / invert
(``affine``, ``deformation``), registration needs to *average* and *interpolate*
transforms on their manifold and to *fuse* a multi-stage chain into a single
resampling.  The averaging is the substrate for groupwise / template
construction (a template is the Fréchet mean of the subject transforms), motion
summary, and temporal regularisation.

- ``transform_mean`` -- the Fréchet (Karcher) mean of homogeneous transforms
  (rigid ``SE(n)`` ⊂ affine), the Riemannian barycentre via the log/exp fixed
  point in the true matrix chart (``linalg.matrix_log`` / ``matrix_exp``).
- ``transform_geodesic`` -- the point at fraction ``t`` along the geodesic from
  identity to a transform (``exp(t · log T)``; ``½``-point squares to ``T``).
- ``velocity_mean`` -- the Fréchet mean of stationary velocity fields, which is
  the (weighted) arithmetic mean (an SVF *is* its own Lie-algebra element).

The matrix chart (not the closed-form ``rigid_exp`` / ``rigid_log``) is used so
the mean / geodesic are the *true* group operations -- no ``so(n)`` log
singularity at identity (which the Karcher init hits), and a genuine geodesic
(``transform_geodesic(T, ½) @ transform_geodesic(T, ½) == T``).  ``matrix_log``
routes through ``safe_inv``, so these are **offline** barycentre / interpolation
ops (a healthy GPU runs them jit- and grad-clean; the wedged-cuSolver dev box
runs the forward pass eagerly via the CPU fallback -- hence the Python loop, not
``lax.scan``).  Transform *fusion* lands alongside these as ``fuse_transforms``
in V3b.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..linalg._solver import safe_inv
from ..linalg.matrix_function import matrix_exp, matrix_log

__all__ = [
    'transform_mean',
    'transform_geodesic',
    'velocity_mean',
]


def transform_mean(
    transforms: Float[Array, 'k d1 d1'],
    *,
    weights: Optional[Float[Array, ' k']] = None,
    iters: int = 10,
) -> Float[Array, 'd1 d1']:
    """Fréchet (Karcher) mean of homogeneous transforms (rigid or affine).

    The Riemannian barycentre, found by the standard log/exp fixed point
    ``μ ← μ · exp(Σ_k w_k · log(μ⁻¹ T_k))`` (``iters`` steps from
    ``transforms[0]``) in the true matrix chart.  The mean of rigid transforms
    is rigid; of affines, affine.  The Lie-algebra mean is a plain weighted sum,
    and the matrix log is smooth at identity (unlike the ``so(n)`` rotation-
    vector log), so the iteration converges from the first-element init.

    Parameters
    ----------
    transforms
        ``(K, ndim+1, ndim+1)`` stack of homogeneous transforms (rigid or
        affine; the linear blocks need positive determinant -- the
        ``rigid_exp`` / ``affine_exp`` regime).
    weights
        Optional ``(K,)`` non-negative weights (normalised internally); ``None``
        -> uniform.
    iters
        Karcher fixed-point iterations (converges fast for clustered inputs).

    Returns
    -------
    The mean transform, ``(ndim+1, ndim+1)``.

    Notes
    -----
    Uses ``matrix_log`` (hence ``safe_inv``): jit- and grad-clean on a healthy
    GPU; a forward / eager op on the wedged-cuSolver dev box.
    """
    k = transforms.shape[0]
    if weights is None:
        w = jnp.full((k,), 1.0 / k, dtype=transforms.dtype)
    else:
        w = weights / jnp.sum(weights)

    mu = transforms[0]
    for _ in range(iters):
        tangents = matrix_log(safe_inv(mu) @ transforms)
        mean_tangent = jnp.tensordot(w, tangents, axes=(0, 0))
        mu = mu @ matrix_exp(mean_tangent)
    return mu


def transform_geodesic(
    transform: Float[Array, 'd1 d1'],
    t: float,
) -> Float[Array, 'd1 d1']:
    """Point at fraction ``t`` along the geodesic identity->``transform``.

    ``exp(t · log transform)`` in the true matrix chart: ``t = 0`` is identity,
    ``t = 1`` is ``transform``, and ``transform_geodesic(T, ½) @
    transform_geodesic(T, ½) == T``.  Interpolate between two transforms ``A``,
    ``B`` as ``A @ transform_geodesic(invert(A) @ B, t)``.  Offline (uses
    ``matrix_log``); see :func:`transform_mean`.
    """
    return matrix_exp(t * matrix_log(transform))


def velocity_mean(
    velocities: Float[Array, 'k *spatial ndim'],
    *,
    weights: Optional[Float[Array, ' k']] = None,
) -> Float[Array, '*spatial ndim']:
    """Fréchet mean of stationary velocity fields (the SVF barycentre).

    A stationary velocity field *is* its own Lie-algebra element, so the
    barycentre is just the (weighted) arithmetic mean -- the deformable template
    centre.  ``velocities`` is ``(K, *spatial, ndim)``.
    """
    if weights is None:
        return velocities.mean(axis=0)
    w = weights / jnp.sum(weights)
    return jnp.tensordot(w, velocities, axes=(0, 0))
