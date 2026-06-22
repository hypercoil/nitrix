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
from ._converge import run_iterations
from ._core import (
    Convergence,
    RegistrationResult,
    RegistrationSpec,
    resolve_convergence_mode,
    resolve_iterations,
)
from ._space import _conjugate_about

# A model's steepest-descent images and its compositional update / params
# recovery -- the only pieces of the IC kernel that differ by transform group.
SteepestDescent = Callable[[Array, Array, int], Array]
CompositionalUpdate = Callable[[Array, int], Array]
ParamsFromMatrix = Callable[[Array, int], Array]
# Project the per-iteration moments to the steepest-descent projection
# ``SDᵀe (P,)``: ``(m (ndim, ndim), g (ndim,), ndim) -> (P,)`` (3a).  Rigid
# contracts ``m`` against the ``so(n)`` generators; affine flattens it.
ProjectMoments = Callable[[Array, Array, int], Array]

# Per-level precomputed reference data: (fixed, **gradient** ``∇F (M, ndim)``,
# inverse Hessian (P, P), grid centre (ndim,)).  The IC projection ``SDᵀe`` is
# the moment tensor of ``∇F`` against the centred grid and the error (3a), so
# only ``∇F`` is stored -- not the ``(M, P)`` steepest-descent buffer (``P``-fold
# resident memory; ``P = ndim² + ndim`` for affine, so ``∇F`` is a ``P/ndim``
# = 4× saving at 3-D affine, ~600 MB at 256³).
ReferenceLevel = tuple[Array, Array, Array, Array]

_RIDGE = 1e-4  # relative Levenberg ridge on the (constant) Hessian.

# Geometric trust region (`register-affine-small-grid-divergence`).  The IC step
# uses the constant *template* Hessian, which is unreliable on a few-voxel coarse
# grid or a low-gradient image; the raw Gauss-Newton step can then overshoot and
# ``matrix_exp`` explode (params -> thousands, the warp anti-correlates).  Rather
# than damp *every* step (a larger ridge slows the precise, well-conditioned path
# and throttles early-exit), clamp only the step's **induced grid displacement**:
# a step that would move the sampling grid by more than the grid's own extent is
# physically meaningless (you cannot align by moving farther than the image), so
# it is shortened to that bound.  A normal step (displacement <= extent) is left
# byte-unchanged -- so the well-conditioned path keeps its single-step Gauss-Newton
# convergence (and the ridge stays tiny), while an explosive step is bounded.
_TRUST_EXTENT_FACTOR = (
    1.0  # cap the per-step grid motion at this * the grid extent
)


def _grid_corners(shape: tuple[int, ...], center: Array, dtype: Any) -> Array:
    """Homogeneous, centred grid corners ``(2**ndim, ndim+1)`` -- where an affine
    step's induced displacement is extremal."""
    axes = [
        jnp.asarray([0.0, s - 1.0], dtype=dtype) - center[i]
        for i, s in enumerate(shape)
    ]
    mesh = jnp.meshgrid(*axes, indexing='ij')
    pts = jnp.stack([m.reshape(-1) for m in mesh], axis=-1)
    ones = jnp.ones((pts.shape[0], 1), dtype=dtype)
    return jnp.concatenate([pts, ones], axis=-1)


def _trust_scale(
    matrix: Array,
    update: Array,
    corners: Array,
    ndim: int,
    step_max: float,
) -> Array:
    """Scale factor in ``(0, 1]`` capping the step's induced grid displacement.

    The step ``matrix -> matrix @ update`` moves grid corner ``x`` by
    ``matrix @ (update − I) @ x``; the largest such corner motion is clamped to
    ``step_max``.  Model-generic (reads the homogeneous ``update`` matrix, so it
    serves rigid and affine alike).
    """
    m_du = matrix @ (update - jnp.eye(ndim + 1, dtype=matrix.dtype))
    disp = (corners @ m_du.T)[:, :ndim]
    max_disp = jnp.max(jnp.sqrt(jnp.sum(disp * disp, axis=-1)))
    return jnp.minimum(1.0, step_max / (max_disp + 1e-12))


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


def _rigid_project_moments(m: Array, g: Array, ndim: int) -> Array:
    """Rigid ``SDᵀe`` from the shared moments (3a).

    The rotation columns of ``SDᵀe`` are a fixed contraction of the moment
    tensor ``m_ij = Σ_x ∇F_i·(x−c)_j·e`` against the ``so(n)`` generators
    (``(SDᵀe)_k = Σ_ij G[k,i,j]·m_ij``); the translation columns are ``g_i =
    Σ_x ∇F_i·e``.  Order (rotation then translation) matches ``_steepest_descent``.
    """
    gens = _rotation_generators(ndim, m.dtype)
    return jnp.concatenate([jnp.einsum('kij,ij->k', gens, m), g])


def _affine_project_moments(m: Array, g: Array, ndim: int) -> Array:
    """Affine ``SDᵀe`` from the shared moments (3a).

    The linear-block columns are the moment tensor itself, row-major
    (``(SDᵀe)_{(i,j)} = m_ij = Σ_x ∇F_i·(x−c)_j·e``); the translation columns are
    ``g``.  Order matches ``_affine_steepest_descent``.
    """
    return jnp.concatenate([m.reshape(-1), g])


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
    project_moments: ProjectMoments,
    convergence: Optional[Convergence] = None,
) -> tuple[Array, Array]:
    """Run the inverse-compositional iterations on one resolution.

    The per-iteration projection ``SDᵀe`` is reconstructed from the moments of
    ``∇F`` (3a) -- ``m_ij = Σ_x ∇F_i·(x−c)_j·e`` and ``g_i = Σ_x ∇F_i·e``, a
    fused reduction over only ``∇F`` rather than a re-read of the ``(M, P)``
    steepest-descent buffer -- then ``project_moments`` assembles the model's
    ``SDᵀe`` (rigid contracts ``m``, affine flattens it) and ``H⁻¹`` solves.

    The Gauss-Newton step is **geometric-trust-region-clamped** (``_trust_scale``):
    a step whose induced grid motion would exceed the grid extent is shortened, so
    the unreliable few-voxel / low-gradient Hessian cannot drive ``matrix_exp`` to
    explode; a normal step is left unchanged.  ``convergence`` (default ``None``)
    selects the fixed ``scan`` or the windowed-slope early-exit (``run_iterations``).
    """
    fixed, grad, h_inv, center = ref
    grid_c = (identity_grid(fixed.shape, dtype=moving.dtype) - center).reshape(
        -1, ndim
    )
    corners = _grid_corners(fixed.shape, center, moving.dtype)
    step_max = _TRUST_EXTENT_FACTOR * float(max(fixed.shape))

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
        m = grad.T @ (grid_c * err[:, None])  # (ndim, ndim) moment tensor
        g = grad.T @ err  # (ndim,) translation moments
        delta = h_inv @ project_moments(m, g, ndim)
        update = compositional_update(delta, ndim)
        scale = _trust_scale(matrix, update, corners, ndim, step_max)
        update = compositional_update(delta * scale, ndim)
        return matrix @ update, 0.5 * jnp.sum(err * err)

    return run_iterations(
        step,
        matrix,
        iterations=iterations,
        convergence=convergence,
        dtype=moving.dtype,
    )


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
        # The (M, P) steepest-descent images build the Hessian **once** here, but
        # are not stored: the per-iteration projection is reconstructed from the
        # moments of ``∇F`` (3a), so only the gradient is kept resident.
        sd = steepest_descent(fixed, center, ndim)
        h_inv = _hessian_inv(sd)
        grad = spatial_gradient(fixed).reshape(-1, ndim)
        levels.append((fixed, grad, h_inv, center))
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
    project_moments: ProjectMoments = _rigid_project_moments,
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
    # Single-pair IC path: always early-exit-capable (the constant-template
    # while_loop).  mode='early_exit' (the rigid/affine recipe default) -> the
    # windowed loop; mode='fixed' -> the scan.
    convergence = resolve_convergence_mode(
        spec.mode,
        spec.convergence,
        supports_early_exit=True,
        path='the inverse-compositional path',
    )
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
        matrix, costs = _ic_level(
            m_l,
            ref,
            matrix,
            spec=spec,
            ndim=ndim,
            iterations=iters_per_level[level],
            compositional_update=compositional_update,
            project_moments=project_moments,
            convergence=convergence,
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
    # Return the self-contained centred matrix (the warp applies ``matrix``
    # about ``center``), matching the IndexSpace forward path; ``params`` keeps
    # the raw about-origin Lie coordinates of the un-centred matrix.
    return RegistrationResult(
        matrix=_conjugate_about(matrix, center, ndim, dtype),
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
        project_moments=_affine_project_moments,
    )
