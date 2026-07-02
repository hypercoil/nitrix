# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Deformation- and velocity-field algebra.

The operations the diffeomorphic (log-Demons) recipe needs on top of the
stationary-velocity-field stack (:func:`~nitrix.geometry.integrate_velocity_field`
is the exponential :math:`\\exp(v)`):

- :func:`compose_displacement` -- compose two displacement fields,
  :math:`(\\mathrm{id} + u) \\circ (\\mathrm{id} + v)`, the warp-by-then-warp
  operation.
- :func:`compose_velocity` -- the Baker-Campbell-Hausdorff approximation of
  the velocity whose exponential is :math:`\\exp(v) \\circ \\exp(u)` (the
  log-domain update); first order is plain addition, second order adds half
  the Lie bracket.
- :func:`invert_displacement` -- the inverse displacement, as the fixed
  point :math:`s_{\\mathrm{inv}} = -s \\circ (\\mathrm{id} + s_{\\mathrm{inv}})`
  (via :func:`~nitrix.numerics.fixed_point_solve`, so it is differentiable).
- :func:`field_log` -- the stationary-velocity *logarithm* of a deformation
  (the inverse of :func:`~nitrix.geometry.integrate_velocity_field`), by
  inverse scaling-and-squaring; the dense analogue of
  :func:`~nitrix.linalg.matrix_log`.  Recovers a stationary-velocity-field
  parameterisation of a deformation solved directly in the group (the greedy
  registration path).  The exponential is not surjective -- a general
  diffeomorphism (e.g. a greedy composition of many non-commuting warps) is
  *not* :math:`\\exp(v)` for any single stationary :math:`v` -- so
  :func:`field_log` returns the *best stationary-velocity-field fit*: the
  round-trip :math:`\\exp(\\mathrm{field\\_log}(\\varphi)) = \\varphi` is exact
  by construction, but :func:`field_log` is a true inverse of the exponential
  only on the stationary-velocity-field submanifold (a nonzero fit residual
  off it).

Channel-last fields ``(*spatial, ndim)``; coordinates are in index space
(the :func:`~nitrix.geometry.identity_grid` convention).
"""

from __future__ import annotations

from typing import Literal, overload

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
    """Displacement of the composed deformation
    :math:`(\\mathrm{id} + \\mathrm{outer}) \\circ (\\mathrm{id} + \\mathrm{inner})`.

    The composed deformation maps
    :math:`x \\mapsto x + \\mathrm{inner}(x) + \\mathrm{outer}(x + \\mathrm{inner}(x))`,
    so the displacement is
    :math:`\\mathrm{inner} + \\mathrm{outer} \\circ (\\mathrm{id} + \\mathrm{inner})`.
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

    Returns
    -------
    Float[Array, '*spatial ndim']
        The displacement field of the composed deformation,
        ``(*spatial, ndim)``.
    """
    spatial_shape = inner.shape[:-1]
    grid = identity_grid(spatial_shape, dtype=inner.dtype) + inner
    warped_outer = spatial_transform(outer, grid, mode=mode)
    return inner + warped_outer


def _grad_field(
    field: Float[Array, '*spatial ndim'],
) -> Float[Array, '*spatial ndim ndim']:
    """Spatial Jacobian :math:`\\partial\\, \\mathrm{field}_i / \\partial x_j`.

    Differentiates the displacement (not the full deformation): the
    identity is subtracted from the deformation Jacobian so that a zero
    field yields a zero matrix.

    Parameters
    ----------
    field
        Displacement field, ``(*spatial, ndim)``.

    Returns
    -------
    Float[Array, '*spatial ndim ndim']
        The Jacobian of the displacement, ``(*spatial, ndim, ndim)``,
        with entry ``[..., i, j]`` equal to
        :math:`\\partial\\, \\mathrm{field}_i / \\partial x_j`.
    """
    ndim = field.shape[-1]
    eye = jnp.eye(ndim, dtype=field.dtype)
    return jacobian_displacement(field) - eye


def _lie_bracket(
    v: Float[Array, '*spatial ndim'],
    u: Float[Array, '*spatial ndim'],
) -> Float[Array, '*spatial ndim']:
    """Lie bracket
    :math:`[v, u] = (v \\cdot \\nabla) u - (u \\cdot \\nabla) v` of two velocity
    fields.

    Parameters
    ----------
    v, u
        Stationary velocity fields, ``(*spatial, ndim)``.

    Returns
    -------
    Float[Array, '*spatial ndim']
        The Lie bracket :math:`[v, u]`, ``(*spatial, ndim)``.
    """
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
    """Baker-Campbell-Hausdorff composition of stationary velocity fields.

    Approximates the velocity :math:`z` for which
    :math:`\\exp(z) \\approx \\exp(v) \\circ \\exp(u)`:

    - ``order == 1`` -- :math:`z = v + u` (the standard additive log-domain
      update; exact when the fields commute).
    - ``order == 2`` -- :math:`z = v + u + \\tfrac{1}{2} [v, u]` (the first
      Baker-Campbell-Hausdorff correction).

    Parameters
    ----------
    v, u
        Stationary velocity fields to compose, ``(*spatial, ndim)``.  The
        result approximates the velocity of the exponential
        :math:`\\exp(v) \\circ \\exp(u)`.
    order
        Order of the Baker-Campbell-Hausdorff expansion: ``1`` (default)
        for the plain additive update most diffeomorphic-demons
        implementations use, or ``2`` to add half the Lie bracket.

    Returns
    -------
    Float[Array, '*spatial ndim']
        The composed stationary velocity field :math:`z`,
        ``(*spatial, ndim)``.

    Raises
    ------
    ValueError
        If ``order`` is neither ``1`` nor ``2``.
    """
    if order == 1:
        return v + u
    if order == 2:
        return v + u + 0.5 * _lie_bracket(v, u)
    raise ValueError(f'order must be 1 or 2; got {order}.')


@overload
def invert_displacement(
    s: Float[Array, '*spatial ndim'],
    *,
    tol: float = ...,
    max_iter: int = ...,
    mode: BoundaryMode = ...,
    acceleration: Literal['picard', 'anderson'] = ...,
    return_residual: Literal[False] = ...,
) -> Float[Array, '*spatial ndim']: ...


@overload
def invert_displacement(
    s: Float[Array, '*spatial ndim'],
    *,
    tol: float = ...,
    max_iter: int = ...,
    mode: BoundaryMode = ...,
    acceleration: Literal['picard', 'anderson'] = ...,
    return_residual: Literal[True],
) -> tuple[Float[Array, '*spatial ndim'], Float[Array, '']]: ...


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
    """Inverse displacement field of the deformation
    :math:`\\varphi = \\mathrm{id} + s`.

    Returns ``s_inv`` with
    :math:`(\\mathrm{id} + s) \\circ (\\mathrm{id} + s_{\\mathrm{inv}}) \\approx \\mathrm{id}`,
    found as the fixed point
    :math:`s_{\\mathrm{inv}} = -s \\circ (\\mathrm{id} + s_{\\mathrm{inv}})` via
    :func:`~nitrix.numerics.fixed_point_solve` (so it is differentiable with
    respect to ``s`` by the implicit-function theorem).  The iteration
    converges geometrically with factor :math:`\\lVert \\nabla s \\rVert` (the
    diffeomorphic regime :math:`\\lVert \\nabla s \\rVert < 1`) -- fast for the
    smoothed deformations registration produces.

    The failure mode of a bare iteration cap is a *silent* non-converged
    inverse returned at ``max_iter`` (a stiff
    :math:`\\lVert \\nabla s \\rVert \\to 1` deformation).  Setting
    ``return_residual`` returns the realised inversion error so the caller
    can assert convergence rather than trust it; for a genuinely stiff fixed
    point (where plain Picard plateaus) pass ``acceleration='anderson'`` (a
    quasi-Newton mixing that converges where Picard stalls).  Plain Picard is
    the default because it is faster *and* more accurate on the smoothed
    (non-stiff) regime registration actually hits -- Anderson helps only once
    Picard genuinely stalls.

    Parameters
    ----------
    s
        Displacement field, ``(*spatial, ndim)``.
    tol, max_iter
        Fixed-point convergence controls: the residual tolerance and the
        iteration cap.
    mode
        Boundary mode for the inner sampling.
    acceleration
        ``'picard'`` (default) -- the plain iteration, best on the
        smoothed regime; ``'anderson'`` -- Anderson-accelerated, for a
        stiff :math:`\\lVert \\nabla s \\rVert \\to 1` deformation where Picard
        plateaus.
    return_residual
        If ``True``, also return the scalar inversion residual
        :math:`\\mathrm{rms}\\big((\\mathrm{id} + s) \\circ (\\mathrm{id} + s_{\\mathrm{inv}}) - \\mathrm{id}\\big)`
        (zero at a perfect inverse), so the caller can assert convergence
        rather than trust it.

    Returns
    -------
    Float[Array, '*spatial ndim'] or tuple of (Float[Array, '*spatial ndim'], Float[Array, ''])
        The inverse displacement field ``s_inv``, ``(*spatial, ndim)``, if
        ``return_residual`` is ``False``; otherwise the pair
        ``(s_inv, residual)`` where ``residual`` is the scalar
        root-mean-square inversion error.
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
    """Displacement of the diffeomorphism square root
    :math:`(\\mathrm{id} + s)^{1/2}`.

    Returns ``w`` with
    :math:`(\\mathrm{id} + w) \\circ (\\mathrm{id} + w) = \\mathrm{id} + s` (i.e.
    ``compose_displacement(w, w) = s``), found as the half-damped fixed point
    :math:`w \\leftarrow w + \\tfrac{1}{2}\\,(s - \\mathrm{compose\\_displacement}(w, w))`
    from :math:`w_0 = s/2`.

    The damping is essential *and* sufficient: the un-damped iteration
    :math:`w \\leftarrow s - w \\circ (\\mathrm{id} + w)` has a near-identity
    Jacobian of :math:`-I` and *oscillates*; the half-damping makes the
    near-identity Jacobian :math:`0` (:math:`\\mathrm{compose}(w, w) \\approx 2w`
    there), so the iteration is super-linear (a residual around ``1e-10`` in
    well under ten steps even for a large deformation) -- plain Picard, no
    acceleration needed.

    Parameters
    ----------
    s
        Displacement field of the deformation to take the square root of,
        ``(*spatial, ndim)``.
    tol, max_iter
        Convergence controls for the fixed-point iteration: the residual
        tolerance and the iteration cap.
    mode
        Boundary mode for the inner compositions.

    Returns
    -------
    Float[Array, '*spatial ndim']
        The displacement field ``w`` of the square-root deformation,
        ``(*spatial, ndim)``.
    """

    def half_step(s_param: Array, w: Array) -> Array:
        return w + 0.5 * (s_param - compose_displacement(w, w, mode=mode))

    return fixed_point_solve(half_step, s, 0.5 * s, tol=tol, max_iter=max_iter)


def _directional_grad(
    w: Float[Array, '*spatial ndim'],
) -> Float[Array, '*spatial ndim']:
    """Directional gradient :math:`(w \\cdot \\nabla) w` of a field along
    itself.

    The first Baker-Campbell-Hausdorff correction term for the logarithm:
    the field whose component :math:`i` equals
    :math:`\\sum_j (\\partial w_i / \\partial x_j)\\, w_j`.

    Parameters
    ----------
    w
        Displacement (or velocity) field, ``(*spatial, ndim)``.

    Returns
    -------
    Float[Array, '*spatial ndim']
        The directional gradient :math:`(w \\cdot \\nabla) w`,
        ``(*spatial, ndim)``.
    """
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
    """Stationary-velocity logarithm of the deformation
    :math:`\\varphi = \\mathrm{id} + \\mathrm{displacement}`.

    The inverse of :func:`~nitrix.geometry.integrate_velocity_field` by
    *inverse scaling-and-squaring* (the dense analogue of
    :func:`~nitrix.linalg.matrix_log`): take ``n_sqrt`` diffeomorphism square
    roots until near identity, then :math:`v \\approx 2^{n_{\\mathrm{sqrt}}} \\cdot w`
    where :math:`w` is the ``n_sqrt``-times square root.

    The exponential is not surjective, so the recovered :math:`v` is a best
    fit.  A general diffeomorphism (a greedy composition of non-commuting
    warps) is *not* :math:`\\exp(v)` for any single stationary :math:`v`, so
    this function returns the *best stationary-velocity-field
    parameterisation*, not a true inverse of an exponential that does not
    exist there.  Concretely, two distinct accuracy notions:

    - **Round-trip fidelity** -- reintegrating the result reproduces the
      input, ``integrate_velocity_field(field_log(s), n_steps=n_sqrt)`` equals
      :math:`\\mathrm{id} + s` *exactly* (to ``sqrt_tol``) and
      unconditionally, because :math:`v / 2^{n_{\\mathrm{sqrt}}}` *is* the
      square-root displacement by construction.  This is what reproduces the
      given warp -- what most consumers need.
    - **Generating-velocity fidelity** -- if :math:`\\varphi = \\exp(v_{\\mathrm{true}})`
      then the returned velocity is
      :math:`v = v_{\\mathrm{true}} + O(\\lVert v \\rVert^2 / 2^{n_{\\mathrm{sqrt}}})`
      (the logarithm approximation); on a deformation that is not a
      stationary velocity field there is no :math:`v_{\\mathrm{true}}` and the
      fit carries a nonzero residual :math:`\\lVert \\exp(v) - \\varphi \\rVert`
      (measure it via the round-trip if needed).

    The recovered velocity feeds the velocity-barycentre / template machinery
    (:func:`~nitrix.geometry.velocity_mean` /
    :func:`~nitrix.geometry.transform_mean`).  It is differentiable (via the
    implicit-function theorem through the fixed points) and GPU-native (no
    dense inverse, unlike :func:`~nitrix.linalg.matrix_log`).

    Parameters
    ----------
    displacement
        The deformation as a displacement field :math:`s`
        (:math:`\\varphi = \\mathrm{id} + s`), ``(*spatial, ndim)``.
    n_sqrt
        Number of inverse-squaring (diffeomorphism square-root) steps.
        Matches the default ``n_steps`` of
        :func:`~nitrix.geometry.integrate_velocity_field` so the exponential
        and logarithm charts are consistent; the generating-velocity error
        halves with each extra root.
    sqrt_tol, sqrt_max_iter
        Convergence controls for each square-root fixed point: the residual
        tolerance and the iteration cap.
    correction
        ``'first_order'`` (default) -- :math:`v = 2^{n_{\\mathrm{sqrt}}} \\cdot w`
        (using :math:`\\log(\\mathrm{id} + w) \\approx w`); this *preserves the
        exact round-trip*.  ``'bch'`` -- the one-term correction
        :math:`v = 2^{n_{\\mathrm{sqrt}}} \\cdot (w - \\tfrac{1}{2}(w \\cdot \\nabla) w)`,
        more accurate as the *generating* velocity but it *breaks the exact
        round-trip* (use only when the velocity is interpreted as the true
        generator, e.g. geodesic / momentum analysis).
    mode
        Boundary mode for the inner compositions (edge-replicate by
        default, the flow-field convention).

    Returns
    -------
    Float[Array, '*spatial ndim']
        The stationary velocity field :math:`v`, ``(*spatial, ndim)``.

    Raises
    ------
    ValueError
        If ``correction`` is neither ``'first_order'`` nor ``'bch'``.
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
