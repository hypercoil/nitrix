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
- ``field_log`` -- the stationary-velocity *logarithm* of a deformation
  (the inverse of ``integrate_velocity_field``), by inverse scaling-and-
  squaring; the dense analogue of ``linalg.matrix_log``.  Recovers an SVF
  parameterisation of a deformation solved directly in the group (the
  greedy registration path).  **``exp`` is not surjective** -- a general
  diffeomorphism (e.g. a greedy composition of many non-commuting warps)
  is *not* ``exp(v)`` for any single stationary ``v`` -- so ``field_log``
  returns the **best SVF fit**: the round-trip ``exp(field_log(φ)) == φ``
  is exact by construction, but ``field_log`` is a true inverse of ``exp``
  only on the SVF submanifold (a nonzero fit residual off it).

Channel-last fields ``(*spatial, ndim)``; coordinates index-space
(``identity_grid`` convention).
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..numerics.fixed_point import fixed_point_solve
from ._interpolate import BoundaryMode
from .grid import identity_grid, jacobian_displacement, spatial_transform

__all__ = [
    'compose_displacement',
    'compose_velocity',
    'invert_displacement',
    'field_log',
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
    acceleration: Literal['picard', 'anderson'] = 'picard',
    return_residual: bool = False,
) -> (
    Float[Array, '*spatial ndim']
    | tuple[Float[Array, '*spatial ndim'], Float[Array, '']]
):
    """Inverse displacement field of ``φ = id + s``.

    Returns ``s_inv`` with ``(id + s) ∘ (id + s_inv) ≈ id``, found as the
    fixed point ``s_inv = -s ∘ (id + s_inv)`` via
    ``numerics.fixed_point_solve`` (so it is differentiable w.r.t. ``s``
    by the implicit-function theorem).  Converges geometrically with
    factor ``‖∇s‖`` (the diffeomorphic regime ``‖∇s‖ < 1``) -- fast for the
    smoothed deformations registration produces.

    **Robustness (B5): ``return_residual``, not a silent ``max_iter``.**
    The failure mode is a *silent* non-converged inverse returned at the
    iteration cap (a stiff ``‖∇s‖ → 1`` deformation).  ``return_residual``
    returns the realised inversion error so the caller can assert
    convergence rather than trust it; for a genuinely stiff fixed point
    (where plain Picard plateaus) pass ``acceleration='anderson'`` (a
    quasi-Newton mixing that converges where Picard stalls).  Plain Picard
    is the default because it is faster *and* more accurate on the
    smoothed (non-stiff) regime registration actually hits -- Anderson
    helps only once Picard genuinely stalls.

    Parameters
    ----------
    s
        Displacement field, ``(*spatial, ndim)``.
    tol, max_iter
        Fixed-point convergence controls.
    mode
        Boundary mode for the inner sampling.
    acceleration
        ``'picard'`` (default) -- the plain iteration, best on the
        smoothed regime; ``'anderson'`` -- Anderson-accelerated, for a
        stiff ``‖∇s‖ → 1`` deformation where Picard plateaus.
    return_residual
        If ``True``, also return the scalar inversion residual
        ``rms((id + s) ∘ (id + s_inv) − id)`` (zero at a perfect inverse),
        so the caller can assert convergence rather than trust it.

    Returns
    -------
    ``s_inv`` (``return_residual=False``) or ``(s_inv, residual)``.
    """
    spatial_shape = s.shape[:-1]

    def update(field: Array, s_inv: Array) -> Array:
        grid = identity_grid(spatial_shape, dtype=field.dtype) + s_inv
        return -spatial_transform(field, grid, mode=mode)

    s_inv = fixed_point_solve(
        update,
        s,
        jnp.zeros_like(s),
        tol=tol,
        max_iter=max_iter,
        acceleration=acceleration,
    )
    if return_residual:
        composed = compose_displacement(s, s_inv, mode=mode)
        residual = jnp.sqrt(jnp.mean(composed * composed))
        return s_inv, residual
    return s_inv


def _diffeo_sqrt(
    s: Float[Array, '*spatial ndim'],
    *,
    tol: float,
    max_iter: int,
    mode: BoundaryMode,
) -> Float[Array, '*spatial ndim']:
    """Displacement of the diffeomorphism square root ``(id + s)^{1/2}``.

    Returns ``w`` with ``(id + w) ∘ (id + w) = id + s`` (i.e.
    ``compose_displacement(w, w) = s``), as the **½-damped** fixed point
    ``w ← w + ½·(s − compose_displacement(w, w))`` from ``w₀ = s/2``.

    The damping is essential *and* sufficient: the un-damped
    ``w ← s − w∘(id+w)`` has a near-identity Jacobian of ``−I`` and
    *oscillates*; the ½-damping makes the near-identity Jacobian ``0``
    (``compose(w, w) ≈ 2w`` there), so the iteration is **super-linear**
    (a residual ``~1e-10`` in well under ten steps even for a large
    deformation) -- plain Picard, no acceleration needed.
    """

    def half_step(s_param: Array, w: Array) -> Array:
        return w + 0.5 * (s_param - compose_displacement(w, w, mode=mode))

    return fixed_point_solve(half_step, s, 0.5 * s, tol=tol, max_iter=max_iter)


def _directional_grad(
    w: Float[Array, '*spatial ndim'],
) -> Float[Array, '*spatial ndim']:
    """``(w·∇)w`` -- the field with component ``i`` equal to
    ``Σ_j (∂w_i/∂x_j)·w_j`` (the first BCH log correction term)."""
    ndim = w.shape[-1]
    grad_w = jacobian_displacement(w) - jnp.eye(ndim, dtype=w.dtype)
    return jnp.einsum('...ij,...j->...i', grad_w, w)


def field_log(
    displacement: Float[Array, '*spatial ndim'],
    *,
    n_sqrt: int = 6,
    sqrt_tol: float = 1e-6,
    sqrt_max_iter: int = 50,
    correction: Literal['first_order', 'bch'] = 'first_order',
    mode: BoundaryMode = 'nearest',
) -> Float[Array, '*spatial ndim']:
    """Stationary-velocity logarithm of ``φ = id + displacement``.

    The inverse of ``integrate_velocity_field`` by **inverse scaling-and-
    squaring** (the dense analogue of ``linalg.matrix_log``): take
    ``n_sqrt`` diffeomorphism square roots until near identity, then
    ``v ≈ 2**n_sqrt · w`` where ``w`` is the ``n_sqrt``-times square root.

    **``exp`` is not surjective -- the recovered ``v`` is a best fit.** A
    general diffeomorphism (a greedy composition of non-commuting warps)
    is *not* ``exp(v)`` for any single stationary ``v``, so ``field_log``
    returns the **best SVF parameterisation**, not a true inverse of an
    ``exp`` that does not exist there.  Concretely, two distinct accuracy
    notions:

    - **Round-trip fidelity** -- ``integrate_velocity_field(field_log(s),
      n_steps=n_sqrt) == id + s`` is **exact** (to ``sqrt_tol``),
      *unconditionally* (``v/2**n_sqrt`` *is* the square-root displacement
      by construction).  This is what reproduces the given warp -- what
      most consumers need.
    - **Generating-velocity fidelity** -- if ``φ = exp(v_true)`` then the
      returned ``v = v_true + O(‖v‖² / 2**n_sqrt)`` (the log approximation,
      §below); on a *non-SVF* ``φ`` there is no ``v_true`` and the SVF fit
      carries a nonzero residual ``‖exp(v) − φ‖`` (measure it via the
      round-trip if needed).

    The recovered ``v`` feeds the velocity barycentre / template machinery
    (``geometry.velocity_mean`` / ``transform_mean``).  Differentiable
    (fixed-point IFT) and GPU-native (no ``safe_inv``, unlike
    ``matrix_log``).

    Parameters
    ----------
    displacement
        The deformation as a displacement field ``s`` (``φ = id + s``),
        ``(*spatial, ndim)``.
    n_sqrt
        Number of inverse-squaring (diffeomorphism square-root) steps.
        Matches ``integrate_velocity_field``'s default ``n_steps`` so the
        ``exp``/``log`` charts are consistent; the generating-velocity
        error halves with each extra root.
    sqrt_tol, sqrt_max_iter
        Convergence controls for each square-root fixed point.
    correction
        ``'first_order'`` (default) -- ``v = 2**n_sqrt · w`` (``log(id+w)
        ≈ w``); **preserves the exact round-trip**.  ``'bch'`` -- the
        one-term correction ``v = 2**n_sqrt · (w − ½(w·∇)w)``, more
        accurate as the *generating* velocity but it **breaks the exact
        round-trip** (use only when the velocity is interpreted as the
        true generator, e.g. geodesic / momentum analysis).
    mode
        Boundary mode for the inner compositions (edge-replicate by
        default, the flow-field convention).

    Returns
    -------
    The stationary velocity field ``v``, ``(*spatial, ndim)``.
    """
    w = displacement
    for _ in range(n_sqrt):
        w = _diffeo_sqrt(w, tol=sqrt_tol, max_iter=sqrt_max_iter, mode=mode)
    scale = float(2**n_sqrt)
    if correction == 'bch':
        return (w - 0.5 * _directional_grad(w)) * scale
    if correction != 'first_order':
        raise ValueError(
            f"correction must be 'first_order' or 'bch'; got {correction!r}."
        )
    return w * scale
