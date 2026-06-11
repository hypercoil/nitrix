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

from typing import Any, Callable, Optional

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
from ..linalg.matrix_function import matrix_exp, matrix_log
from ._core import (
    Convergence,
    RegistrationResult,
    RegistrationSpec,
    resolve_iterations,
)

# A model's steepest-descent images and its compositional update / params
# recovery -- the only pieces of the IC kernel that differ by transform group.
SteepestDescent = Callable[[Array, Array, int], Array]
CompositionalUpdate = Callable[[Array, int], Array]
ParamsFromMatrix = Callable[[Array, int], Array]

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


def _affine_steepest_descent(
    fixed: Float[Array, '*spatial'],
    center: Float[Array, ' d'],
    ndim: int,
) -> Float[Array, ' m p']:
    """Steepest-descent images of the centred affine warp at identity.

    The 12-DOF (3-D) / 6-DOF (2-D) affine: the linear-block generators are the
    full ``gl(n)`` basis ``E_ij`` (``∂ matrix_exp(A)/∂A_ij|₀ = E_ij``), so
    ``SD[x, (i,j)] = ∇F(x)_i · (x − c)_j`` (``n²`` columns, row-major to match
    ``affine_exp``); the translation columns are ``∇F`` (``n`` columns).
    Returns ``(M, P)`` with ``P = n² + n``.
    """
    grad = spatial_gradient(fixed)
    grid_c = identity_grid(fixed.shape, dtype=fixed.dtype) - center
    # sd_lin[..., i, j] = ∇F[..., i] · (x − c)[..., j], row-major (i, j).
    sd_lin = (grad[..., :, None] * grid_c[..., None, :]).reshape(
        fixed.shape + (ndim * ndim,)
    )
    sd = jnp.concatenate([sd_lin, grad], axis=-1)
    return sd.reshape(-1, sd.shape[-1])


def _rigid_compositional_update(delta: Array, ndim: int) -> Array:
    """Inverse-compositional warp update ``W(Δθ)⁻¹`` for a rigid step.

    Closed-form: ``W(Δθ) = rigid_exp(Δθ)``, inverted by the rigid transpose
    rule (no solve).
    """
    return _rigid_inverse(rigid_exp(delta, ndim=ndim), ndim)


def _affine_compositional_update(delta: Array, ndim: int) -> Array:
    """Inverse-compositional warp update ``W(Δθ)⁻¹`` for an affine step.

    ``W(Δθ) = [[matrix_exp(A), t], [0, 1]]`` (``affine_exp``), whose inverse is
    ``[[matrix_exp(−A), −matrix_exp(−A)·t], [0, 1]]`` -- the linear-block
    inverse is ``matrix_exp(A)⁻¹ = matrix_exp(−A)``, so it stays **GPU-native**
    (pure matmul, no solve).
    """
    a = delta[: ndim * ndim].reshape(ndim, ndim)
    t = delta[ndim * ndim :]
    l_inv = matrix_exp(-a)
    inv = jnp.eye(ndim + 1, dtype=delta.dtype)
    inv = inv.at[:ndim, :ndim].set(l_inv)
    return inv.at[:ndim, ndim].set(-l_inv @ t)


def _rigid_params_from_matrix(matrix: Array, ndim: int) -> Array:
    """Rigid Lie parameters of a recovered rigid homogeneous matrix."""
    return rigid_log(matrix, ndim=ndim)


def _affine_params_from_matrix(matrix: Array, ndim: int) -> Array:
    """Affine Lie parameters of a recovered affine homogeneous matrix.

    The linear params are ``matrix_log`` of the linear block (the ``gl(n)``
    generator, row-major); the translation params are the literal translation
    (``affine_exp`` uses a direct translation) -- matching ``affine_exp``'s
    layout.  Uses ``matrix_log`` (offline, not in the iteration loop).
    """
    linear = matrix_log(matrix[:ndim, :ndim]).reshape(-1)
    return jnp.concatenate([linear, matrix[:ndim, ndim]])


def _hessian_inv(sd: Float[Array, ' m p']) -> Float[Array, ' p p']:
    """Inverse Gauss-Newton Hessian (Jacobi-preconditioned, computed once).

    The affine steepest-descent columns span orders of magnitude -- the
    linear-block columns scale with the voxel coordinate ``x − c`` (``O(n)`` for
    an ``n``-voxel axis), the translation columns are ``O(1)`` -- so ``H = SDᵀSD``
    is badly conditioned and a single scalar Levenberg ridge mis-damps the
    small-diagonal (translation) directions.  Precondition by the diagonal
    (Jacobi / Marquardt's relative ridge): scale ``H`` to a unit diagonal
    ``Ĥ = D⁻¹ H D⁻¹`` with ``D = diag(√Hᵢᵢ)``, ridge the *conditioned* matrix,
    and unscale the inverse, i.e. ``H⁻¹ = D⁻¹ (Ĥ + λI)⁻¹ D⁻¹``.  This is exactly
    ``(H + λ·diag(Hᵢᵢ))⁻¹`` -- a scale-invariant per-direction ridge -- and the
    solved system is well-conditioned regardless of the column scaling.  The
    rigid path (balanced columns) is essentially unchanged; the affine path
    becomes robust.
    """
    h = sd.T @ sd
    p = h.shape[0]
    d = jnp.sqrt(jnp.diagonal(h)) + jnp.finfo(sd.dtype).eps
    scale = d[:, None] * d[None, :]
    h_hat = h / scale
    h_hat_inv = safe_inv(h_hat + _RIDGE * jnp.eye(p, dtype=sd.dtype))
    return h_hat_inv / scale


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
    iterations: int,
    compositional_update: CompositionalUpdate,
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
        update = compositional_update(delta, ndim)
        return matrix @ update, 0.5 * jnp.sum(err * err)

    return jax.lax.scan(step, matrix, xs=None, length=iterations)


def _ic_level_converge(
    moving: Array,
    ref: ReferenceLevel,
    matrix0: Array,
    *,
    spec: RegistrationSpec,
    ndim: int,
    iterations: int,
    compositional_update: CompositionalUpdate,
    convergence: Convergence,
) -> tuple[Array, Array]:
    """IC iterations on one resolution with ANTs-style early-exit.

    A ``lax.while_loop`` that stops when the windowed normalised cost slope (a
    least-squares-line fit over the last ``window`` per-iteration costs) drops
    below ``threshold``, or ``iterations`` (the hard cap) is reached.  Returns
    ``(matrix, costs)`` with ``costs`` a fixed ``(iterations,)`` trace (the
    unrun tail padded with the final cost), so the result's ``cost_history``
    keeps its shape.
    """
    fixed, sd, h_inv, center = ref
    dtype = moving.dtype
    window = convergence.window
    threshold = convergence.threshold
    t = jnp.arange(window, dtype=dtype)
    t_centred = t - jnp.mean(t)
    t_var = jnp.sum(t_centred * t_centred)

    def cost_and_delta(matrix: Array) -> tuple[Array, Array]:
        grid = affine_grid(matrix, fixed.shape, center=center)
        warped = spatial_transform(
            moving[..., None],
            grid,
            mode=spec.boundary_mode,
            cval=spec.cval,
            method=spec.interpolation,
        )[..., 0]
        err = (warped - fixed).reshape(-1)
        return 0.5 * jnp.sum(err * err), h_inv @ (sd.T @ err)

    def converged(buf: Array) -> Array:
        slope = jnp.sum(t_centred * (buf - jnp.mean(buf))) / t_var
        return jnp.abs(slope) / (jnp.abs(jnp.mean(buf)) + 1e-12) < threshold

    def cond(carry: tuple[Array, Array, Array, Array]) -> Array:
        _, i, buf, _ = carry
        return (i < iterations) & ((i < window) | ~converged(buf))

    def body(
        carry: tuple[Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array]:
        matrix, i, buf, hist = carry
        cost, delta = cost_and_delta(matrix)
        update = compositional_update(delta, ndim)
        buf = jnp.concatenate([buf[1:], cost[None]])
        return matrix @ update, i + 1, buf, hist.at[i].set(cost)

    cost0, _ = cost_and_delta(matrix0)
    init = (
        matrix0,
        jnp.asarray(0),
        jnp.full((window,), cost0, dtype=dtype),
        jnp.full((iterations,), cost0, dtype=dtype),
    )
    matrix, last_i, buf, hist = jax.lax.while_loop(cond, body, init)
    # Pad the unrun tail of the trace with the final cost.
    hist = jnp.where(jnp.arange(iterations) < last_i, hist, buf[-1])
    return matrix, hist


def ic_reference(
    pyr_f: tuple[Float[Array, '*spatial 1'], ...],
    ndim: int,
    *,
    steepest_descent: SteepestDescent = _steepest_descent,
) -> list[ReferenceLevel]:
    """Precompute the per-level reference data (finest first).

    The shared, frame-independent work: the steepest-descent images
    (``steepest_descent``; rigid by default, affine for the affine recipe) and
    the inverse Hessian on every reference pyramid level.  Computed once; reused
    across all frames and all iterations.
    """
    levels = []
    for level in pyr_f:
        fixed = level[..., 0]
        center = (jnp.asarray(fixed.shape, dtype=fixed.dtype) - 1.0) / 2.0
        sd = steepest_descent(fixed, center, ndim)
        levels.append((fixed, sd, _hessian_inv(sd), center))
    return levels


def ic_register_core(
    moving: Float[Array, '*spatial'],
    ref_levels: list[ReferenceLevel],
    *,
    ndim: int,
    spec: RegistrationSpec,
    init_matrix: Array,
    compositional_update: CompositionalUpdate = _rigid_compositional_update,
    params_from_matrix: ParamsFromMatrix = _rigid_params_from_matrix,
) -> RegistrationResult:
    """Inverse-compositional register ``moving`` against a precomputed ref.

    ``ref_levels`` (from :func:`ic_reference`) is built once by the caller
    and ``vmap``-ed over a series, so the steepest-descent / Hessian work
    is shared.  Builds the moving pyramid, runs the coarse-to-fine IC
    iterations carrying the warp matrix, and finalises.  The transform group is
    the ``compositional_update`` (``W(Δθ)⁻¹``) + ``params_from_matrix`` pair
    (rigid by default; affine for ``affine_register``).
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
    iters_per_level = resolve_iterations(spec.iterations, spec.levels)
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
        if spec.convergence is None:
            matrix, costs = _ic_level(
                m_l,
                ref,
                matrix,
                spec=spec,
                ndim=ndim,
                iterations=iters_per_level[level],
                compositional_update=compositional_update,
            )
        else:
            matrix, costs = _ic_level_converge(
                m_l,
                ref,
                matrix,
                spec=spec,
                ndim=ndim,
                iterations=iters_per_level[level],
                compositional_update=compositional_update,
                convergence=spec.convergence,
            )
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
        params=params_from_matrix(matrix, ndim),
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


def ic_affine_register(
    moving: Float[Array, '*mspatial'],
    fixed: Float[Array, '*fspatial'],
    *,
    ndim: int,
    spec: RegistrationSpec,
    init_matrix: Optional[Array] = None,
) -> RegistrationResult:
    """Single-pair inverse-compositional **affine** registration.

    The affine analogue of :func:`ic_rigid_register` (lever A′): the reference
    steepest-descent uses the full ``gl(n)`` generators (12-DOF 3-D / 6-DOF
    2-D), the compositional update inverts ``affine_exp(Δθ)`` GPU-natively via
    ``matrix_exp(−A)``, and the params are recovered with ``matrix_log`` of the
    linear block.  Affine has the largest parameter count, so the forward
    path's per-iteration ``jacfwd`` (≈ 14 tangent warps) is the costliest --
    the constant-template Hessian saves the most here.
    """
    pyr_f = gaussian_pyramid(
        fixed[..., None],
        levels=spec.levels,
        factor=spec.pyramid_factor,
        sigma=spec.pyramid_sigma,
    )
    ref_levels = ic_reference(
        pyr_f, ndim, steepest_descent=_affine_steepest_descent
    )
    if init_matrix is None:
        init_matrix = jnp.eye(ndim + 1, dtype=moving.dtype)
    return ic_register_core(
        moving,
        ref_levels,
        ndim=ndim,
        spec=spec,
        init_matrix=init_matrix,
        compositional_update=_affine_compositional_update,
        params_from_matrix=_affine_params_from_matrix,
    )
