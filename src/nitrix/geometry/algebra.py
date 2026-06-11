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
- ``fuse_transforms`` -- collapse a multi-stage chain (matrices + displacement
  fields) into one displacement, so ``moving`` is resampled **once** (no
  compounded interpolation blur; one gather).

The matrix chart (not the closed-form ``rigid_exp`` / ``rigid_log``) is used so
the mean / geodesic are the *true* group operations -- no ``so(n)`` log
singularity at identity (which the Karcher init hits), and a genuine geodesic
(``transform_geodesic(T, ½) @ transform_geodesic(T, ½) == T``).  ``matrix_log``
routes through ``safe_inv``, so these are **offline** barycentre / interpolation
ops (a healthy GPU runs them jit- and grad-clean; the wedged-cuSolver dev box
runs the forward pass eagerly via the CPU fallback -- hence the Python loop, not
``lax.scan``).  ``fuse_transforms`` is pure resampling (no ``safe_inv``), so it
is GPU-native.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..linalg._solver import safe_inv
from ..linalg.matrix_function import matrix_exp, matrix_log
from ._interpolate import BoundaryMode
from .grid import identity_grid, spatial_transform
from .transform import apply_affine

__all__ = [
    'transform_mean',
    'transform_geodesic',
    'velocity_mean',
    'fuse_transforms',
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


def fuse_transforms(
    transforms: Sequence[Union[Float[Array, 'd1 d1'], Float[Array, '*spatial ndim']]],
    shape: tuple[int, ...],
    *,
    mode: BoundaryMode = 'nearest',
) -> Float[Array, '*spatial ndim']:
    """Fuse a chain of transforms into ONE displacement field on ``shape``.

    ``transforms`` are the stages applied to ``moving`` **in order**
    (``transforms[0]`` first, as a ``rigid -> affine -> deformable`` pipeline
    does); each is a homogeneous matrix ``(ndim+1, ndim+1)`` (applied about the
    grid centre, the registration convention) or a displacement field on
    ``shape`` (``(*shape, ndim)``).  Returns the single displacement ``d`` for
    which ``spatial_transform(moving, identity_grid + d)`` reproduces warping
    ``moving`` through the **whole** chain -- **one** interpolation instead of
    ``N`` (a quality win: no compounded resampling blur; and a throughput win:
    one gather).

    The stages compose on the sampling grid in reverse (the last stage maps the
    output grid first, the first stage maps into ``moving`` last):
    ``g = T₁(T₂(… T_N(x)))``.
    """
    if not transforms:
        raise ValueError('fuse_transforms: empty chain')
    dtype = transforms[-1].dtype
    id_grid = identity_grid(shape, dtype=dtype)
    centre = (jnp.asarray(shape, dtype=dtype) - 1.0) / 2.0
    g = id_grid
    for t in reversed(transforms):
        if t.ndim == 2:
            g = apply_affine(g, t, center=centre)
        else:
            g = g + spatial_transform(t, g, mode=mode)
    return g - id_grid
