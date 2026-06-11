# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Inverse-compositional rigid registration (constant-template Hessian).

The Baker-Matthews (2004) inverse-compositional Lucas-Kanade scheme,
specialised to the centred rigid warp in **index space** -- the lean,
on-device frame ``IndexSpace`` was set up to host.  Where the forward
Gauss-Newton path re-linearises the warped *moving* image every iteration,
the inverse-compositional scheme linearises the **reference** (template)
instead, so the steepest-descent images ``∂F·∂W/∂θ|₀`` and the Gauss-Newton
Hessian ``H = SDᵀSD`` are evaluated **once** on the reference and reused
across every iteration -- the 3dvolreg lineage's speed.

For ``volreg`` the reference is shared across the whole series, so the
per-level ``SD`` / ``H⁻¹`` are computed a single time for *all* frames and
*all* iterations (``_ic_reference``), and only the per-frame update
(warp -> error -> ``SDᵀe`` -> ``H⁻¹`` matvec -> compositional matrix
update) is ``vmap``-ed.  Memory is ``O(M·P)`` for the shared steepest-
descent plus ``O(M)`` per frame -- not the forward path's ``O(T·M·P)``.

The recovered warp is carried as the homogeneous matrix ``T``
(fixed-voxel -> moving-voxel), updated by the inverse-compositional rule
``T ← T · W(Δθ)⁻¹`` with the closed-form rigid inverse (no general solve).
SSD only (it is a least-squares scheme); ``IndexSpace`` only (the template
linearisation is in voxel coordinates).
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..geometry import (
    affine_grid,
    gaussian_pyramid,
    identity_grid,
    rigid_exp,
    rigid_log,
    spatial_gradient,
    spatial_transform,
)
from ..linalg._solver import safe_inv
from ._core import RegistrationResult, RegistrationSpec

# Per-level precomputed reference data: (fixed, steepest-descent (M, P),
# inverse Hessian (P, P), grid centre (ndim,)).
ReferenceLevel = tuple[Array, Array, Array, Array]

_RIDGE = 1e-4  # relative Levenberg ridge on the (constant) Hessian.


def _rotation_generators(ndim: int, dtype: Any) -> Array:
    """``so(n)`` generators: ``(n_rot, ndim, ndim)`` skew bases.

    ``∂R(ω)/∂ω_k|₀`` for the ``rigid_exp`` rotation -- one matrix in 2-D,
    the three skew bases in 3-D (matching ``geometry.transform._skew3``).
    """
    if ndim == 2:
        return jnp.asarray([[[0.0, -1.0], [1.0, 0.0]]], dtype=dtype)
    gx = [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    gy = [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    gz = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    return jnp.asarray([gx, gy, gz], dtype=dtype)


def _steepest_descent(
    fixed: Float[Array, '*spatial'],
    center: Float[Array, ' d'],
    ndim: int,
) -> Float[Array, ' m p']:
    """Steepest-descent images of the centred rigid warp at identity.

    ``SD[x, j] = ∇F(x) · ∂W(x;θ)/∂θ_j|₀``.  Rotation columns use the
    generator fields ``G_k·(x − c)``; translation columns are the gradient
    components.  Order matches ``rigid_exp`` (rotation params then
    translation).  Returns ``(M, P)`` (M voxels, P parameters).
    """
    grad = spatial_gradient(fixed)
    grid_c = identity_grid(fixed.shape, dtype=fixed.dtype) - center
    gens = _rotation_generators(ndim, fixed.dtype)
    # sd_rot[..., k] = Σ_ij G[k,i,j] (x−c)[..., j] ∇F[..., i]
    sd_rot = jnp.einsum('kij,...j,...i->...k', gens, grid_c, grad)
    sd = jnp.concatenate([sd_rot, grad], axis=-1)  # translation SD = ∇F
    return sd.reshape(-1, sd.shape[-1])


def _hessian_inv(sd: Float[Array, ' m p']) -> Float[Array, ' p p']:
    """Inverse Gauss-Newton Hessian ``(SDᵀSD + λI)⁻¹`` (computed once)."""
    h = sd.T @ sd
    p = h.shape[0]
    damp = _RIDGE * jnp.trace(h) / p
    return safe_inv(h + damp * jnp.eye(p, dtype=sd.dtype))


def _rigid_inverse(matrix: Array, ndim: int) -> Array:
    """Closed-form inverse of a rigid homogeneous matrix (no solve)."""
    r = matrix[:ndim, :ndim]
    t = matrix[:ndim, ndim]
    rt = r.T
    inv = jnp.eye(ndim + 1, dtype=matrix.dtype)
    inv = inv.at[:ndim, :ndim].set(rt)
    return inv.at[:ndim, ndim].set(-rt @ t)


def _ic_level(
    moving: Array,
    ref: ReferenceLevel,
    matrix: Array,
    *,
    spec: RegistrationSpec,
    ndim: int,
) -> tuple[Array, Array]:
    """Run the inverse-compositional iterations on one resolution."""
    fixed, sd, h_inv, center = ref

    def step(matrix: Array, _: Any) -> tuple[Array, Array]:
        grid = affine_grid(matrix, fixed.shape, center=center)
        warped = spatial_transform(
            moving[..., None],
            grid,
            mode=spec.boundary_mode,
            cval=spec.cval,
            method=spec.interpolation,
        )[..., 0]
        err = (warped - fixed).reshape(-1)
        delta = h_inv @ (sd.T @ err)
        update = _rigid_inverse(rigid_exp(delta, ndim=ndim), ndim)
        return matrix @ update, 0.5 * jnp.sum(err * err)

    return jax.lax.scan(step, matrix, xs=None, length=spec.iterations)


def ic_reference(
    pyr_f: tuple[Float[Array, '*spatial 1'], ...],
    ndim: int,
) -> list[ReferenceLevel]:
    """Precompute the per-level reference data (finest first).

    The shared, frame-independent work: the steepest-descent images and
    the inverse Hessian on every reference pyramid level.  Computed once;
    reused across all frames and all iterations.
    """
    levels = []
    for level in pyr_f:
        fixed = level[..., 0]
        center = (jnp.asarray(fixed.shape, dtype=fixed.dtype) - 1.0) / 2.0
        sd = _steepest_descent(fixed, center, ndim)
        levels.append((fixed, sd, _hessian_inv(sd), center))
    return levels


def ic_register_core(
    moving: Float[Array, '*spatial'],
    ref_levels: list[ReferenceLevel],
    *,
    ndim: int,
    spec: RegistrationSpec,
    init_matrix: Array,
) -> RegistrationResult:
    """Inverse-compositional register ``moving`` against a precomputed ref.

    ``ref_levels`` (from :func:`ic_reference`) is built once by the caller
    and ``vmap``-ed over a series, so the steepest-descent / Hessian work
    is shared.  Builds the moving pyramid, runs the coarse-to-fine IC
    iterations carrying the warp matrix, and finalises.
    """
    dtype = moving.dtype
    pyr_m = gaussian_pyramid(
        moving[..., None],
        levels=spec.levels,
        factor=spec.pyramid_factor,
        sigma=spec.pyramid_sigma,
    )
    matrix = init_matrix
    histories = []
    prev_shape = None
    for level in range(spec.levels - 1, -1, -1):
        ref = ref_levels[level]
        m_l = pyr_m[level][..., 0]
        f_shape = ref[0].shape
        if prev_shape is not None:
            # Voxel-unit translation column rescales to the finer grid.
            ratio = jnp.asarray(f_shape, dtype=dtype) / jnp.asarray(
                prev_shape, dtype=dtype
            )
            matrix = matrix.at[:ndim, ndim].set(matrix[:ndim, ndim] * ratio)
        matrix, costs = _ic_level(m_l, ref, matrix, spec=spec, ndim=ndim)
        histories.append(costs)
        prev_shape = f_shape

    full_shape = ref_levels[0][0].shape
    center = (jnp.asarray(full_shape, dtype=dtype) - 1.0) / 2.0
    grid = affine_grid(matrix, full_shape, center=center)
    warped = spatial_transform(
        moving[..., None],
        grid,
        mode=spec.boundary_mode,
        cval=spec.cval,
        method=spec.interpolation,
    )[..., 0]
    return RegistrationResult(
        matrix=matrix,
        params=rigid_log(matrix, ndim=ndim),
        warped=warped,
        cost_history=jnp.concatenate(histories),
    )


def ic_rigid_register(
    moving: Float[Array, '*mspatial'],
    fixed: Float[Array, '*fspatial'],
    *,
    ndim: int,
    spec: RegistrationSpec,
    init_matrix: Optional[Array] = None,
) -> RegistrationResult:
    """Single-pair inverse-compositional rigid registration.

    The fast path ``rigid_register`` dispatches to when its preconditions hold
    (``IndexSpace`` + a least-squares / SSD metric + a Rigid model): builds the
    reference steepest-descent / Hessian **once** (:func:`ic_reference`) and
    runs the constant-template iterations (:func:`ic_register_core`) -- the
    3dvolreg lineage's per-iteration speed (one warp + one projection per step
    vs the forward path's ``jacfwd`` over the warp).  Returns the same
    ``RegistrationResult`` as the forward driver, so the two are interchangeable
    (the forward path is the parity oracle).
    """
    pyr_f = gaussian_pyramid(
        fixed[..., None],
        levels=spec.levels,
        factor=spec.pyramid_factor,
        sigma=spec.pyramid_sigma,
    )
    ref_levels = ic_reference(pyr_f, ndim)
    if init_matrix is None:
        init_matrix = jnp.eye(ndim + 1, dtype=moving.dtype)
    return ic_register_core(
        moving, ref_levels, ndim=ndim, spec=spec, init_matrix=init_matrix
    )
