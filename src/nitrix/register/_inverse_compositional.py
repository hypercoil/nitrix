# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Inverse-compositional rigid registration (constant-template Hessian).

The Baker-Matthews inverse-compositional Lucas-Kanade scheme, specialised to
the centred rigid warp in **index space** -- the lean, on-device frame that
:class:`IndexSpace` was set up to host.  Where the forward Gauss-Newton path
re-linearises the warped *moving* image every iteration, the
inverse-compositional scheme linearises the **reference** (template) instead,
so the steepest-descent images
:math:`\\partial F \\cdot \\partial W / \\partial\\theta|_0` and the
Gauss-Newton Hessian :math:`H = \\mathrm{SD}^{\\top}\\mathrm{SD}` are evaluated
**once** on the reference and reused across every iteration -- the same
constant-template speed as the classic ``3dvolreg`` lineage.

For a motion-correction series the reference is shared across the whole
series, so the per-level :math:`\\mathrm{SD}` / :math:`H^{-1}` are computed a
single time for *all* frames and *all* iterations (see :func:`ic_reference`),
and only the per-frame update (warp, then error, then
:math:`\\mathrm{SD}^{\\top}e`, then the :math:`H^{-1}` mat-vec, then the
compositional matrix update) is vmapped.  Memory is :math:`O(M \\cdot P)` for
the shared steepest-descent plus :math:`O(M)` per frame -- not the forward
path's :math:`O(T \\cdot M \\cdot P)`.

The recovered warp is carried as the homogeneous matrix :math:`T`
(fixed-voxel to moving-voxel), updated by the inverse-compositional rule
:math:`T \\leftarrow T \\cdot W(\\Delta\\theta)^{-1}` with the closed-form
rigid inverse (no general solve).  SSD only (it is a least-squares scheme);
index-space only (the template linearisation is in voxel coordinates).

References
----------
Baker, S. and Matthews, I. (2004). Lucas-Kanade 20 years on: a unifying
framework. *International Journal of Computer Vision*, 56, 221-255.
doi:10.1023/B:VISI.0000011205.11775.fd
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
from ._converge import (
    Convergence,
    resolve_convergence_mode,
    run_iterations,
)
from ._core import (
    RegistrationResult,
    RegistrationSpec,
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

# F1 backtracking ladder: step fractions along the trust-clamped Gauss-Newton
# direction (in the Lie algebra) the cost-decrease guard tries, largest first.
# The largest fraction that decreases the cost is accepted (so a step that
# already decreases at full length -- the well-conditioned case -- is taken
# byte-unchanged at 1.0); if none decreases (a bad-direction step from the
# constant-template Hessian on a hard case), the iterate is left unmoved, which
# makes the per-level cost monotone non-increasing.  Three rungs keep the extra
# warps bounded; the no-move fallback is the implicit fourth.
_BACKTRACK_ALPHAS = (1.0, 0.5, 0.25)


def _grid_corners(shape: tuple[int, ...], center: Array, dtype: Any) -> Array:
    """Homogeneous, centred coordinates of the sampling grid's corners.

    The corners are where an affine step's induced grid displacement is
    extremal, so they suffice to bound the step's maximum motion.

    Parameters
    ----------
    shape : tuple of int
        Spatial shape of the sampling grid; one axis per dimension.
    center : Array
        Grid centre, shape ``(ndim,)``, subtracted from every corner so the
        corners are expressed relative to the rotation/scaling centre.
    dtype
        Floating dtype of the returned corner coordinates.

    Returns
    -------
    Array
        Corner coordinates in homogeneous form, shape ``(2 ** ndim, ndim + 1)``
        (a trailing column of ones).
    """
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

    The step ``matrix -> matrix @ update`` moves grid corner :math:`x` by
    :math:`\\mathrm{matrix} \\cdot (\\mathrm{update} - I) \\cdot x`; the largest
    such corner motion is clamped to ``step_max``.  Model-generic (reads the
    homogeneous ``update`` matrix, so it serves rigid and affine alike).

    Parameters
    ----------
    matrix : Array
        Current warp as a homogeneous matrix, shape ``(ndim + 1, ndim + 1)``.
    update : Array
        Candidate compositional update as a homogeneous matrix, shape
        ``(ndim + 1, ndim + 1)``; the proposed step is ``matrix @ update``.
    corners : Array
        Homogeneous, centred grid corners, shape ``(2 ** ndim, ndim + 1)``,
        as returned by :func:`_grid_corners`.
    ndim : int
        Number of spatial dimensions.
    step_max : float
        Maximum permitted corner displacement (in voxels).

    Returns
    -------
    Array
        Scalar scale factor in ``(0, 1]`` by which to shorten the step so the
        largest corner motion does not exceed ``step_max`` (``1`` when the raw
        step is already within the bound).
    """
    m_du = matrix @ (update - jnp.eye(ndim + 1, dtype=matrix.dtype))
    disp = (corners @ m_du.T)[:, :ndim]
    max_disp = jnp.max(jnp.sqrt(jnp.sum(disp * disp, axis=-1)))
    # Dtype-derived 0/0 guard (E1): a fixed 1e-12 sits below float32's ~1.2e-7
    # precision floor, so it is meaningless in the production float32 path; the
    # finfo epsilon scales the guard to the working dtype.  Only active when the
    # step induces ~zero corner motion (a converged step), so its exact value
    # never affects a real update.
    return jnp.minimum(
        1.0, step_max / (max_disp + jnp.finfo(max_disp.dtype).eps)
    )


def _rotation_generators(ndim: int, dtype: Any) -> Array:
    """Skew-symmetric generators of the rotation group :math:`so(n)`.

    These are the derivatives :math:`\\partial R(\\omega)/\\partial\\omega_k|_0`
    of the :func:`rigid_exp` rotation at identity -- a single matrix in 2-D and
    the three skew bases in 3-D (matching ``geometry.transform._skew3``).

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions (2 or 3).
    dtype
        Floating dtype of the returned generators.

    Returns
    -------
    Array
        Stacked skew bases, shape ``(n_rot, ndim, ndim)`` where ``n_rot`` is 1
        for ``ndim == 2`` and 3 for ``ndim == 3``.
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

    Each column is
    :math:`\\mathrm{SD}[x, j] = \\nabla F(x) \\cdot \\partial W(x;\\theta)/\\partial\\theta_j|_0`.
    Rotation columns use the generator fields :math:`G_k \\cdot (x - c)`;
    translation columns are the gradient components.  Column order matches
    :func:`rigid_exp` (rotation parameters then translation).

    Parameters
    ----------
    fixed : Float[Array, '*spatial']
        Reference (template) image over the spatial grid.
    center : Float[Array, ' d']
        Grid centre, shape ``(ndim,)``, about which the rigid warp rotates.
    ndim : int
        Number of spatial dimensions.

    Returns
    -------
    Float[Array, ' m p']
        Steepest-descent images, shape ``(M, P)`` for ``M`` voxels and ``P``
        rigid parameters (rotation columns then translation columns).
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
    full :math:`gl(n)` basis :math:`E_{ij}`
    (:math:`\\partial\\,\\mathrm{matrix\\_exp}(A)/\\partial A_{ij}|_0 = E_{ij}`),
    so
    :math:`\\mathrm{SD}[x, (i,j)] = \\nabla F(x)_i \\cdot (x - c)_j`
    (:math:`n^2` columns, row-major to match :func:`affine_exp`); the
    translation columns are :math:`\\nabla F` (:math:`n` columns).

    Parameters
    ----------
    fixed : Float[Array, '*spatial']
        Reference (template) image over the spatial grid.
    center : Float[Array, ' d']
        Grid centre, shape ``(ndim,)``, relative to which the linear block acts.
    ndim : int
        Number of spatial dimensions.

    Returns
    -------
    Float[Array, ' m p']
        Steepest-descent images, shape ``(M, P)`` with ``P = ndim ** 2 + ndim``
        (linear-block columns row-major, then translation columns).
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
    """Rigid steepest-descent projection from the shared moments.

    The rotation entries of the projection :math:`\\mathrm{SD}^{\\top}e` are a
    fixed contraction of the moment tensor
    :math:`m_{ij} = \\sum_x \\nabla F_i \\cdot (x - c)_j \\cdot e` against the
    :math:`so(n)` generators
    (:math:`(\\mathrm{SD}^{\\top}e)_k = \\sum_{ij} G[k,i,j] \\cdot m_{ij}`); the
    translation entries are :math:`g_i = \\sum_x \\nabla F_i \\cdot e`.  Order
    (rotation then translation) matches :func:`_steepest_descent`.

    Parameters
    ----------
    m : Array
        Moment tensor of the gradient against the centred grid and error, shape
        ``(ndim, ndim)``.
    g : Array
        Translation moments (gradient against the error), shape ``(ndim,)``.
    ndim : int
        Number of spatial dimensions.

    Returns
    -------
    Array
        The rigid steepest-descent projection, shape ``(P,)`` with
        ``P = n_rot + ndim`` (rotation entries then translation entries).
    """
    gens = _rotation_generators(ndim, m.dtype)
    return jnp.concatenate([jnp.einsum('kij,ij->k', gens, m), g])


def _affine_project_moments(m: Array, g: Array, ndim: int) -> Array:
    """Affine steepest-descent projection from the shared moments.

    The linear-block entries of :math:`\\mathrm{SD}^{\\top}e` are the moment
    tensor itself, flattened row-major
    (:math:`(\\mathrm{SD}^{\\top}e)_{(i,j)} = m_{ij} = \\sum_x \\nabla F_i \\cdot (x - c)_j \\cdot e`);
    the translation entries are :math:`g`.  Order matches
    :func:`_affine_steepest_descent`.

    Parameters
    ----------
    m : Array
        Moment tensor of the gradient against the centred grid and error, shape
        ``(ndim, ndim)``.
    g : Array
        Translation moments (gradient against the error), shape ``(ndim,)``.
    ndim : int
        Number of spatial dimensions.

    Returns
    -------
    Array
        The affine steepest-descent projection, shape ``(P,)`` with
        ``P = ndim ** 2 + ndim`` (flattened linear block then translation).
    """
    return jnp.concatenate([m.reshape(-1), g])


def _rigid_compositional_update(delta: Array, ndim: int) -> Array:
    """Inverse-compositional warp update :math:`W(\\Delta\\theta)^{-1}` for a rigid step.

    Closed-form: :math:`W(\\Delta\\theta) = \\mathrm{rigid\\_exp}(\\Delta\\theta)`,
    inverted by the rigid transpose rule (no solve).

    Parameters
    ----------
    delta : Array
        Rigid Lie-algebra increment, shape ``(P,)`` (rotation parameters then
        translation), as returned by the projection/Hessian solve.
    ndim : int
        Number of spatial dimensions.

    Returns
    -------
    Array
        The inverse warp update as a homogeneous matrix, shape
        ``(ndim + 1, ndim + 1)``.
    """
    return _rigid_inverse(rigid_exp(delta, ndim=ndim), ndim)


def _affine_compositional_update(delta: Array, ndim: int) -> Array:
    """Inverse-compositional warp update :math:`W(\\Delta\\theta)^{-1}` for an affine step.

    The forward warp is
    :math:`W(\\Delta\\theta) = \\begin{bmatrix} \\mathrm{matrix\\_exp}(A) & t \\\\ 0 & 1 \\end{bmatrix}`
    (as built by :func:`affine_exp`), whose inverse is
    :math:`\\begin{bmatrix} \\mathrm{matrix\\_exp}(-A) & -\\mathrm{matrix\\_exp}(-A) \\cdot t \\\\ 0 & 1 \\end{bmatrix}`
    -- the linear-block inverse is
    :math:`\\mathrm{matrix\\_exp}(A)^{-1} = \\mathrm{matrix\\_exp}(-A)`, so it
    stays **GPU-native** (pure matmul, no solve).

    Parameters
    ----------
    delta : Array
        Affine Lie-algebra increment, shape ``(P,)`` with ``P = ndim ** 2 + ndim``
        (flattened linear block :math:`A` then translation :math:`t`).
    ndim : int
        Number of spatial dimensions.

    Returns
    -------
    Array
        The inverse warp update as a homogeneous matrix, shape
        ``(ndim + 1, ndim + 1)``.
    """
    a = delta[: ndim * ndim].reshape(ndim, ndim)
    t = delta[ndim * ndim :]
    l_inv = matrix_exp(-a)
    inv = jnp.eye(ndim + 1, dtype=delta.dtype)
    inv = inv.at[:ndim, :ndim].set(l_inv)
    return inv.at[:ndim, ndim].set(-l_inv @ t)


def _rigid_params_from_matrix(matrix: Array, ndim: int) -> Array:
    """Rigid Lie parameters of a recovered rigid homogeneous matrix.

    Parameters
    ----------
    matrix : Array
        Recovered rigid warp as a homogeneous matrix, shape
        ``(ndim + 1, ndim + 1)``.
    ndim : int
        Number of spatial dimensions.

    Returns
    -------
    Array
        Rigid Lie coordinates, shape ``(P,)`` (rotation parameters then
        translation), matching :func:`rigid_exp`'s layout.
    """
    return rigid_log(matrix, ndim=ndim)


def _affine_params_from_matrix(matrix: Array, ndim: int) -> Array:
    """Affine Lie parameters of a recovered affine homogeneous matrix.

    The linear parameters are :func:`matrix_log` of the linear block (the
    :math:`gl(n)` generator, flattened row-major); the translation parameters
    are the literal translation (:func:`affine_exp` uses a direct translation)
    -- matching :func:`affine_exp`'s layout.  This uses :func:`matrix_log` and
    runs offline (not inside the iteration loop).

    Parameters
    ----------
    matrix : Array
        Recovered affine warp as a homogeneous matrix, shape
        ``(ndim + 1, ndim + 1)``.
    ndim : int
        Number of spatial dimensions.

    Returns
    -------
    Array
        Affine Lie coordinates, shape ``(P,)`` with ``P = ndim ** 2 + ndim``
        (flattened linear-block generator then translation).
    """
    linear = matrix_log(matrix[:ndim, :ndim]).reshape(-1)
    return jnp.concatenate([linear, matrix[:ndim, ndim]])


def _hessian_inv(sd: Float[Array, ' m p']) -> Float[Array, ' p p']:
    """Inverse Gauss-Newton Hessian (Jacobi-preconditioned, computed once).

    The affine steepest-descent columns span orders of magnitude -- the
    linear-block columns scale with the voxel coordinate :math:`x - c`
    (:math:`O(n)` for an :math:`n`-voxel axis), the translation columns are
    :math:`O(1)` -- so :math:`H = \\mathrm{SD}^{\\top}\\mathrm{SD}` is badly
    conditioned and a single scalar Levenberg ridge mis-damps the small-diagonal
    (translation) directions.  Precondition by the diagonal (Jacobi /
    Marquardt's relative ridge): scale :math:`H` to a unit diagonal
    :math:`\\hat{H} = D^{-1} H D^{-1}` with
    :math:`D = \\operatorname{diag}(\\sqrt{H_{ii}})`, ridge the *conditioned*
    matrix, and unscale the inverse, i.e.
    :math:`H^{-1} = D^{-1} (\\hat{H} + \\lambda I)^{-1} D^{-1}`.  This is exactly
    :math:`(H + \\lambda \\cdot \\operatorname{diag}(H_{ii}))^{-1}` -- a
    scale-invariant per-direction ridge -- and the solved system is
    well-conditioned regardless of the column scaling.  The rigid path (balanced
    columns) is essentially unchanged; the affine path becomes robust.

    Parameters
    ----------
    sd : Float[Array, ' m p']
        Steepest-descent images, shape ``(M, P)`` for ``M`` voxels and ``P``
        parameters.

    Returns
    -------
    Float[Array, ' p p']
        The preconditioned inverse Gauss-Newton Hessian, shape ``(P, P)``.
    """
    h = sd.T @ sd
    p = h.shape[0]
    d = jnp.sqrt(jnp.diagonal(h)) + jnp.finfo(sd.dtype).eps
    scale = d[:, None] * d[None, :]
    h_hat = h / scale
    h_hat_inv = safe_inv(h_hat + _RIDGE * jnp.eye(p, dtype=sd.dtype))
    return h_hat_inv / scale


def _rigid_inverse(matrix: Array, ndim: int) -> Array:
    """Closed-form inverse of a rigid homogeneous matrix (no solve).

    Exploits orthogonality of the rotation block (its inverse is its transpose),
    so no linear solve is required.

    Parameters
    ----------
    matrix : Array
        Rigid warp as a homogeneous matrix, shape ``(ndim + 1, ndim + 1)``.
    ndim : int
        Number of spatial dimensions.

    Returns
    -------
    Array
        The inverse rigid warp as a homogeneous matrix, shape
        ``(ndim + 1, ndim + 1)``.
    """
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

    The per-iteration projection :math:`\\mathrm{SD}^{\\top}e` is reconstructed
    from the moments of :math:`\\nabla F` --
    :math:`m_{ij} = \\sum_x \\nabla F_i \\cdot (x - c)_j \\cdot e` and
    :math:`g_i = \\sum_x \\nabla F_i \\cdot e`, a fused reduction over only
    :math:`\\nabla F` rather than a re-read of the ``(M, P)`` steepest-descent
    buffer -- then ``project_moments`` assembles the model's
    :math:`\\mathrm{SD}^{\\top}e` (rigid contracts :math:`m`, affine flattens
    it) and the inverse Hessian solves.

    The Gauss-Newton step is **geometric-trust-region-clamped**
    (:func:`_trust_scale`): a step whose induced grid motion would exceed the
    grid extent is shortened, so the unreliable few-voxel / low-gradient Hessian
    cannot drive :func:`matrix_exp` to explode; a normal step is left unchanged.
    ``convergence`` (default ``None``) selects the fixed scan or the
    windowed-slope early-exit (see :func:`run_iterations`).

    Parameters
    ----------
    moving : Array
        Moving image at this pyramid resolution.
    ref : ReferenceLevel
        Precomputed reference data for this level: the tuple
        ``(fixed, grad, h_inv, center)`` from :func:`ic_reference`.
    matrix : Array
        Current warp as a homogeneous matrix, shape ``(ndim + 1, ndim + 1)``,
        carried into this level.
    spec : RegistrationSpec
        Registration configuration (boundary mode, fill value, interpolation,
        and the optional inverse-compositional line search).
    ndim : int
        Number of spatial dimensions.
    iterations : int
        Number of iterations to run at this level.
    compositional_update : CompositionalUpdate
        Maps a Lie-algebra increment to the inverse warp update
        :math:`W(\\Delta\\theta)^{-1}` for the transform group.
    project_moments : ProjectMoments
        Assembles the model's steepest-descent projection from the shared
        moments.
    convergence : Convergence, optional
        Early-exit policy; ``None`` (default) runs a fixed-length scan.

    Returns
    -------
    tuple of Array
        The updated warp matrix, shape ``(ndim + 1, ndim + 1)``, and the
        per-iteration cost history for this level.
    """
    fixed, grad, h_inv, center = ref
    grid_c = (identity_grid(fixed.shape, dtype=moving.dtype) - center).reshape(
        -1, ndim
    )
    corners = _grid_corners(fixed.shape, center, moving.dtype)
    step_max = _TRUST_EXTENT_FACTOR * float(max(fixed.shape))

    def warp_at(mat: Array) -> Array:
        return spatial_transform(
            moving[..., None],
            affine_grid(mat, fixed.shape, center=center),
            mode=spec.boundary_mode,
            cval=spec.cval,
            method=spec.interpolation,
        )[..., 0]

    def step(matrix: Array, _: Any) -> tuple[Array, Array]:
        warped = warp_at(matrix)
        err = (warped - fixed).reshape(-1)
        c0 = 0.5 * jnp.sum(err * err)
        m = grad.T @ (grid_c * err[:, None])  # (ndim, ndim) moment tensor
        g = grad.T @ err  # (ndim,) translation moments
        delta = h_inv @ project_moments(m, g, ndim)
        update = compositional_update(delta, ndim)
        scale = _trust_scale(matrix, update, corners, ndim, step_max)
        clamped = delta * scale  # trust-clamped GN increment (Lie algebra)

        if spec.ic_line_search:
            # F1 cost-decrease guard (opt-in): a clamped Gauss-Newton step from
            # the *constant* template Hessian can still increase the SSD on a
            # hard case (the direction, not just the magnitude, can be wrong).
            # Backtrack along the clamped direction and accept the largest
            # fraction that decreases the cost; if none does, leave the iterate
            # unmoved -- so the per-level cost is monotone non-increasing.  A
            # full-length decrease (the well-conditioned case) is taken
            # byte-unchanged at alpha=1.  ``spec.ic_line_search`` is static, so
            # the candidate warps are traced only when the guard is on.
            alphas = jnp.asarray(_BACKTRACK_ALPHAS, dtype=moving.dtype)

            def cost_at(alpha: float) -> Array:
                cand = warp_at(
                    matrix @ compositional_update(clamped * alpha, ndim)
                )
                return 0.5 * jnp.sum((cand - fixed) ** 2)

            costs = jnp.stack([cost_at(a) for a in _BACKTRACK_ALPHAS])
            decreased = costs < c0
            clamped = clamped * jnp.where(
                jnp.any(decreased), alphas[jnp.argmax(decreased)], 0.0
            )

        return matrix @ compositional_update(clamped, ndim), c0

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

    Parameters
    ----------
    pyr_f : tuple of Float[Array, '*spatial 1']
        Reference (fixed) Gaussian pyramid, finest level first, each with a
        trailing singleton channel axis.
    ndim : int
        Number of spatial dimensions.
    steepest_descent : SteepestDescent, optional
        Function computing the steepest-descent images for the transform group;
        :func:`_steepest_descent` (rigid) by default.

    Returns
    -------
    list of ReferenceLevel
        One tuple per pyramid level (finest first), each holding the fixed
        image, the flattened gradient ``(M, ndim)``, the inverse Hessian
        ``(P, P)``, and the grid centre ``(ndim,)``.
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
    """Inverse-compositional register ``moving`` against a precomputed reference.

    ``ref_levels`` (from :func:`ic_reference`) is built once by the caller and
    vmapped over a series, so the steepest-descent / Hessian work is shared.
    Builds the moving pyramid, runs the coarse-to-fine inverse-compositional
    iterations carrying the warp matrix, and finalises.  The transform group is
    determined by the ``compositional_update``
    (:math:`W(\\Delta\\theta)^{-1}`) and ``params_from_matrix`` pair (rigid by
    default; affine for :func:`affine_register`).

    Parameters
    ----------
    moving : Float[Array, '*spatial']
        Moving image to align to the reference.
    ref_levels : list of ReferenceLevel
        Precomputed per-level reference data from :func:`ic_reference`
        (finest first).
    ndim : int
        Number of spatial dimensions.
    spec : RegistrationSpec
        Registration configuration (pyramid, iteration, convergence, and
        sampling settings).
    init_matrix : Array
        Initial warp as a homogeneous matrix, shape ``(ndim + 1, ndim + 1)``.
    compositional_update : CompositionalUpdate, optional
        Inverse warp update for the transform group;
        :func:`_rigid_compositional_update` by default.
    params_from_matrix : ParamsFromMatrix, optional
        Recovers the Lie parameters from the final warp matrix;
        :func:`_rigid_params_from_matrix` by default.
    project_moments : ProjectMoments, optional
        Assembles the steepest-descent projection from the shared moments;
        :func:`_rigid_project_moments` by default.

    Returns
    -------
    RegistrationResult
        The recovered warp matrix (centred about the grid centre), the Lie
        parameters of the un-centred matrix, the warped moving image, and the
        concatenated per-level cost history.
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

    The fast path that :func:`rigid_register` dispatches to when its
    preconditions hold (an :class:`IndexSpace` frame, a least-squares / SSD
    metric, and a rigid model): builds the reference steepest-descent / Hessian
    **once** (:func:`ic_reference`) and runs the constant-template iterations
    (:func:`ic_register_core`) -- the classic ``3dvolreg`` per-iteration speed
    (one warp plus one projection per step versus the forward path's ``jacfwd``
    over the warp).  Returns the same :class:`RegistrationResult` as the forward
    driver, so the two are interchangeable (the forward path is the parity
    oracle).

    Parameters
    ----------
    moving : Float[Array, '*mspatial']
        Moving image to align to ``fixed``.
    fixed : Float[Array, '*fspatial']
        Reference (template) image.
    ndim : int
        Number of spatial dimensions.
    spec : RegistrationSpec
        Registration configuration.
    init_matrix : Array, optional
        Initial warp as a homogeneous matrix, shape ``(ndim + 1, ndim + 1)``;
        the identity when omitted.

    Returns
    -------
    RegistrationResult
        The recovered rigid warp, its parameters, the warped moving image, and
        the cost history.
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

    The affine analogue of :func:`ic_rigid_register`: the reference
    steepest-descent uses the full :math:`gl(n)` generators (12-DOF in 3-D,
    6-DOF in 2-D), the compositional update inverts
    :math:`\\mathrm{affine\\_exp}(\\Delta\\theta)` GPU-natively via
    :math:`\\mathrm{matrix\\_exp}(-A)`, and the parameters are recovered with
    :func:`matrix_log` of the linear block.  Affine has the largest parameter
    count, so the forward path's per-iteration ``jacfwd`` (roughly 14 tangent
    warps) is the costliest -- the constant-template Hessian saves the most
    here.

    Parameters
    ----------
    moving : Float[Array, '*mspatial']
        Moving image to align to ``fixed``.
    fixed : Float[Array, '*fspatial']
        Reference (template) image.
    ndim : int
        Number of spatial dimensions.
    spec : RegistrationSpec
        Registration configuration.
    init_matrix : Array, optional
        Initial warp as a homogeneous matrix, shape ``(ndim + 1, ndim + 1)``;
        the identity when omitted.

    Returns
    -------
    RegistrationResult
        The recovered affine warp, its parameters, the warped moving image, and
        the cost history.
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
