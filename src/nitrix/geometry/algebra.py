# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Transform algebra: Lie-group barycentre, geodesic interpolation, fusion.

Beyond the exp/log chart and the pairwise compose / invert operations,
registration needs to *average* and *interpolate* transforms on their manifold
and to *fuse* a multi-stage chain into a single resampling.  The averaging is
the substrate for groupwise / template construction (a template is the Fréchet
mean of the subject transforms), motion summary, and temporal regularisation.

- :func:`transform_mean` -- the Fréchet (Karcher) mean of homogeneous
  transforms (rigid :math:`SE(n) \\subset` affine), the Riemannian barycentre
  via the log/exp fixed point in the true matrix chart (:func:`matrix_log` /
  :func:`matrix_exp`).
- :func:`transform_geodesic` -- the point at fraction :math:`t` along the
  geodesic from identity to a transform (:math:`\\exp(t \\cdot \\log T)`; the
  :math:`\\tfrac{1}{2}`-point squares to :math:`T`).
- :func:`velocity_mean` -- the Fréchet mean of stationary velocity fields, which
  is the (weighted) arithmetic mean (a stationary velocity field *is* its own
  Lie-algebra element).
- :func:`fuse_transforms` -- collapse a multi-stage chain (matrices +
  displacement fields) into one displacement, so the moving image is resampled
  **once** (no compounded interpolation blur; one gather).

The matrix chart (not the closed-form :func:`rigid_exp` / :func:`rigid_log`) is
used so the mean / geodesic are the *true* group operations: no
:math:`\\mathfrak{so}(n)` log singularity at identity (which the Karcher init
hits), and a genuine geodesic (so that composing the half-way transform with
itself recovers the whole, :math:`T_{1/2} \\, T_{1/2} = T`).  Because
:func:`matrix_log` routes through :func:`safe_inv`, the mean and geodesic are
**offline** barycentre / interpolation ops: a healthy GPU runs them jit- and
grad-clean, whereas a machine with a wedged cuSOLVER runs the forward pass
eagerly via the CPU fallback (hence the Python loop, not a scan).
:func:`fuse_transforms` is pure resampling (no :func:`safe_inv`), so it is
GPU-native.
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
    return_residual: bool = False,
) -> Float[Array, 'd1 d1'] | tuple[Float[Array, 'd1 d1'], Float[Array, '']]:
    """Fréchet (Karcher) mean of homogeneous transforms (rigid or affine).

    The Riemannian barycentre, found by the standard log/exp fixed point
    :math:`\\mu \\leftarrow \\mu \\cdot \\exp(\\sum_k w_k \\cdot \\log(\\mu^{-1} T_k))`
    (``iters`` steps started from ``transforms[0]``) in the true matrix chart.
    The mean of rigid transforms is rigid; of affines, affine.  The Lie-algebra
    mean is a plain weighted sum, and the matrix log is smooth at identity
    (unlike the :math:`\\mathfrak{so}(n)` rotation-vector log), so the iteration
    converges from the first-element initialisation.

    ``iters`` is a *hard cap*, not a tolerance loop: a data-dependent
    ``float(residual) < tol`` break would forfeit the jit- and grad-cleanliness
    this op advertises (the Python loop unrolls under trace; a host-synced break
    does not).  It converges in a few steps for a *clustered* cohort, but a
    **widely dispersed** one (rotation spread :math:`\\gtrsim \\pi/2`) may not
    converge within ``iters`` -- raise ``iters`` and use ``return_residual`` to
    *check* rather than trust convergence.  A near-**antipodal** cohort (a pair
    :math:`\\approx \\pi` apart) lies on the negative-real-axis boundary of
    :func:`matrix_log`: the tangent is NaN there, which propagates to a NaN mean
    (a loud "the mean is ill-defined", not silent garbage).

    Parameters
    ----------
    transforms
        ``(K, ndim+1, ndim+1)`` stack of homogeneous transforms (rigid or
        affine; the linear blocks need positive determinant -- the
        :func:`rigid_exp` / :func:`affine_exp` regime).
    weights
        Optional ``(K,)`` array of non-negative weights (normalised internally
        to sum to one); ``None`` gives uniform weights.
    iters
        Karcher fixed-point iteration **cap** (converges fast for clustered
        inputs; a dispersed cohort may need more -- check ``return_residual``).
    return_residual
        If ``True``, also return the scalar Karcher residual (see Returns).

    Returns
    -------
    Float[Array, 'd1 d1'] or tuple
        The mean transform, of shape ``(ndim+1, ndim+1)``, when
        ``return_residual=False``.  When ``return_residual=True``, the pair
        ``(mean, residual)``, where ``residual`` is the scalar norm of the final
        update tangent :math:`\\|\\sum_k w_k \\log(\\mu^{-1} T_k)\\|` -- which tends
        to zero at a converged barycentre.  A non-small (or NaN) residual flags
        a non-converged or ill-defined mean, so the caller can assert
        convergence rather than trust it.

    Notes
    -----
    Uses :func:`matrix_log` (hence :func:`safe_inv`): jit- and grad-clean on a
    healthy GPU; a forward / eager op on a machine with a wedged cuSOLVER.
    """
    k = transforms.shape[0]
    if weights is None:
        w = jnp.full((k,), 1.0 / k, dtype=transforms.dtype)
    else:
        w = weights / jnp.sum(weights)

    mu = transforms[0]
    mean_tangent = jnp.zeros_like(mu)
    for _ in range(iters):
        tangents = matrix_log(safe_inv(mu) @ transforms)
        mean_tangent = jnp.tensordot(w, tangents, axes=(0, 0))
        mu = mu @ matrix_exp(mean_tangent)
    if return_residual:
        return mu, jnp.linalg.norm(mean_tangent)
    return mu


def transform_geodesic(
    transform: Float[Array, 'd1 d1'],
    t: float,
) -> Float[Array, 'd1 d1']:
    """Point at fraction ``t`` along the geodesic from identity to ``transform``.

    Evaluates :math:`\\exp(t \\cdot \\log \\mathrm{transform})` in the true matrix
    chart: ``t = 0`` gives the identity, ``t = 1`` gives ``transform``, and the
    half-way point squares back to the whole, :math:`T_{1/2} \\, T_{1/2} = T`.
    Interpolate between two transforms ``A`` and ``B`` as
    ``A @ transform_geodesic(invert(A) @ B, t)``.  This is an offline op (it
    uses :func:`matrix_log`); see :func:`transform_mean`.

    Parameters
    ----------
    transform
        ``(ndim+1, ndim+1)`` homogeneous transform whose geodesic from the
        identity is traced (its linear block needs positive determinant).
    t
        Scalar fraction along the geodesic; ``0`` is the identity and ``1`` is
        ``transform``.  Values outside ``[0, 1]`` extrapolate along the same
        geodesic.

    Returns
    -------
    Float[Array, 'd1 d1']
        The interpolated homogeneous transform, of shape ``(ndim+1, ndim+1)``.
    """
    return matrix_exp(t * matrix_log(transform))


def velocity_mean(
    velocities: Float[Array, 'k *spatial ndim'],
    *,
    weights: Optional[Float[Array, ' k']] = None,
) -> Float[Array, '*spatial ndim']:
    """Fréchet mean of stationary velocity fields (the SVF barycentre).

    A stationary velocity field *is* its own Lie-algebra element, so its
    Fréchet barycentre is just the (weighted) arithmetic mean -- the deformable
    template centre.

    Parameters
    ----------
    velocities
        ``(K, *spatial, ndim)`` stack of ``K`` stationary velocity fields over a
        shared spatial grid, each with ``ndim`` vector components per voxel.
    weights
        Optional ``(K,)`` array of non-negative weights (normalised internally
        to sum to one); ``None`` gives uniform weights.

    Returns
    -------
    Float[Array, '*spatial ndim']
        The mean velocity field, of shape ``(*spatial, ndim)``.
    """
    if weights is None:
        return velocities.mean(axis=0)
    w = weights / jnp.sum(weights)
    return jnp.tensordot(w, velocities, axes=(0, 0))


def fuse_transforms(
    transforms: Sequence[
        Union[Float[Array, 'd1 d1'], Float[Array, '*spatial ndim']]
    ],
    shape: tuple[int, ...],
    *,
    mode: BoundaryMode = 'nearest',
) -> Float[Array, '*spatial ndim']:
    """Fuse a chain of transforms into ONE displacement field on ``shape``.

    The stages are applied to the moving image **in order** (``transforms[0]``
    first, as a rigid then affine then deformable pipeline does); each is a
    homogeneous matrix ``(ndim+1, ndim+1)`` (applied about the grid centre, the
    registration convention) or a displacement field on ``shape``, of shape
    ``(*shape, ndim)``.  The single returned displacement ``d`` is the one for
    which ``spatial_transform(moving, identity_grid + d)`` reproduces warping the
    moving image through the **whole** chain -- **one** interpolation instead of
    ``N`` (a quality win, since there is no compounded resampling blur, and a
    throughput win, since it is a single gather).

    The stages compose on the sampling grid in reverse (the last stage maps the
    output grid first, the first stage maps into the moving image last):
    :math:`g = T_1(T_2(\\dots T_N(x)))`.

    Parameters
    ----------
    transforms
        Ordered sequence of transform stages, applied to the moving image with
        ``transforms[0]`` first.  Each element is either a homogeneous matrix of
        shape ``(ndim+1, ndim+1)`` (applied about the grid centre) or a
        displacement field of shape ``(*shape, ndim)``.  Must be non-empty.
    shape
        Spatial shape of the output grid on which the fused displacement is
        defined; its length is the spatial rank ``ndim``.
    mode
        Boundary handling used when a displacement-field stage is itself sampled
        onto the running grid, one of ``'constant'``, ``'nearest'``, ``'wrap'``,
        ``'mirror'`` or ``'reflect'``.  Defaults to ``'nearest'``.

    Returns
    -------
    Float[Array, '*spatial ndim']
        The fused displacement field, of shape ``(*shape, ndim)``: adding it to
        :func:`identity_grid` yields the absolute sample coordinates that warp
        the moving image through the whole chain in a single resampling.

    Raises
    ------
    ValueError
        If ``transforms`` is empty.
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
