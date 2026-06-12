# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Fixed-point solver with implicit differentiation.

``fixed_point_solve(f, params, x0)`` returns ``x*`` with ``x* = f(params,
x*)`` and -- via ``jax.custom_vjp`` -- differentiates through the
*solution* by the implicit-function theorem rather than by unrolling the
iteration.  At the fixed point, ``dx*/dparams`` satisfies
``(I - ∂_x f) dx* = ∂_params f``, so the backward solves the adjoint
system ``w = ḡ + (∂_x f)ᵀ w`` (itself a fixed point -- a contraction at a
stable solution) and pushes ``w`` back through ``∂_params f``.  Memory is
O(1) in the iteration count: the backward needs only ``x*``.

**Two forward engines** (the backward is the same IFT adjoint either way):

- ``acceleration='picard'`` (default) -- plain Picard iteration
  ``x ← f(params, x)``.  Converges geometrically with factor ``‖∂_x f‖``;
  fine when that is comfortably below 1.
- ``acceleration='anderson'`` -- windowed Anderson acceleration (Walker-
  Ni), depth ``depth``.  A quasi-Newton secant mixing of the last
  ``depth`` residuals; converges where Picard *genuinely* stalls
  (``‖∂_x f‖ → 1``).  An opt-in escape hatch -- Picard is the better
  choice whenever it converges comfortably (Anderson's least-squares over
  a near-collinear residual history is marginally noisier there), which
  includes the smoothed ``invert_displacement`` / ``field_log`` square
  root (the ½-damping already makes the latter super-linear).  The tiny
  ``depth×depth`` least-squares is solved by an inlined conjugate gradient
  -- **no factorisation, no cuSolver** -- so it runs inside the traced
  ``while_loop`` on every backend (incl. the wedged-cuSolver dev box).

This is the §12.8 graduation (``numerics.fixed_point``); registration is
the named consumer -- ``geometry.invert_displacement`` and
``geometry.field_log`` are fixed points, and the same machinery underpins
the differentiable-registration story.  Generalises the scaling-and-
squaring special case in ``geometry.integrate_velocity_field``.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Literal, Tuple, cast

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

__all__ = ['fixed_point_solve']

FixedPointFn = Callable[[Array, Array], Array]
Acceleration = Literal['picard', 'anderson']


def fixed_point_solve(
    f: FixedPointFn,
    params: Float[Array, '...'],
    x0: Float[Array, '...'],
    *,
    tol: float = 1e-6,
    max_iter: int = 100,
    acceleration: Acceleration = 'picard',
    depth: int = 4,
    beta: float = 1.0,
    reg: float = 1e-8,
) -> Float[Array, '...']:
    """Solve ``x = f(params, x)`` for ``x``, differentiably.

    Parameters
    ----------
    f
        The update map ``(params, x) -> x``.  Must be a contraction near
        the solution for both the forward iteration and the implicit
        backward to converge.
    params
        The differentiable parameters ``f`` depends on (gradients flow
        here via the implicit-function theorem).
    x0
        Initial guess (same structure as the solution; its value does
        not affect the gradient -- ``dx*/dx0 = 0``).
    tol
        Relative convergence tolerance on the residual ``‖f(x) - x‖``.
    max_iter
        Iteration cap (forward and adjoint).
    acceleration
        ``'picard'`` (default) -- plain ``x ← f(x)``.  ``'anderson'`` --
        windowed Anderson acceleration of depth ``depth``, for the stiff
        ``‖∂_x f‖ → 1`` regime (a solve-free quasi-Newton mixing; runs in
        the traced loop on every backend).
    depth
        Anderson window (number of past residuals mixed).  Larger
        converges in fewer steps but holds ``~2·depth`` extra
        iterate-shaped buffers -- lower it for a brain-scale field where
        memory is tight.  Ignored for ``'picard'``.
    beta
        Anderson mixing (relaxation) factor; ``1.0`` is the standard
        choice.  Ignored for ``'picard'``.
    reg
        Relative Tikhonov ridge on the Anderson least-squares (keeps it
        well-posed when the residual history is near-collinear).  Ignored
        for ``'picard'``.

    Returns
    -------
    The fixed point ``x*``.
    """
    d = 0 if acceleration == 'picard' else int(depth)
    return _fixed_point(
        f, tol, max_iter, d, float(beta), float(reg), params, x0
    )


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4, 5))
def _fixed_point(
    f: FixedPointFn,
    tol: float,
    max_iter: int,
    depth: int,
    beta: float,
    reg: float,
    params: Array,
    x0: Array,
) -> Array:
    if depth <= 0:
        return _picard(f, tol, max_iter, params, x0)
    return _anderson(f, tol, max_iter, depth, beta, reg, params, x0)


def _residual_converged(
    r: Array, x: Array, i: Array, tol: float, max_iter: int
) -> Array:
    """Shared stopping rule: relative residual ``‖r‖ ≤ tol·‖x‖`` or cap."""
    moved = jnp.sqrt(jnp.vdot(r, r).real)
    scale = jnp.sqrt(jnp.vdot(x, x).real)
    return (i < max_iter) & (moved > tol * (scale + 1e-12))


def _picard(
    f: FixedPointFn,
    tol: float,
    max_iter: int,
    params: Array,
    x0: Array,
) -> Array:
    def cond(carry: Tuple[Array, Array, Array]) -> Array:
        x_prev, x, i = carry
        return _residual_converged(x - x_prev, x, i, tol, max_iter)

    def body(
        carry: Tuple[Array, Array, Array],
    ) -> Tuple[Array, Array, Array]:
        _, x, i = carry
        return (x, f(params, x), i + 1)

    x1 = f(params, x0)
    _, x_star, _ = lax.while_loop(cond, body, (x0, x1, jnp.asarray(1)))
    return x_star


def _small_spd_solve(a: Array, b: Array, iters: int) -> Array:
    """Solve the tiny SPD system ``a x = b`` by conjugate gradient.

    ``a`` is the ``(depth, depth)`` Anderson normal matrix (SPD after the
    ridge); CG is exact in ``≤ depth`` steps and uses only matvecs, so it
    avoids cuSolver and runs inside the traced ``while_loop``.
    """
    x = jnp.zeros_like(b)
    r = b - a @ x
    p = r
    rs = jnp.vdot(r, r).real

    def step(
        carry: Tuple[Array, Array, Array, Array], _: Any
    ) -> Tuple[Tuple[Array, Array, Array, Array], None]:
        x, r, p, rs = carry
        ap = a @ p
        alpha = rs / (jnp.vdot(p, ap).real + 1e-30)
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = jnp.vdot(r, r).real
        p = r + (rs_new / (rs + 1e-30)) * p
        return (x, r, p, rs_new), None

    (x, _, _, _), _ = lax.scan(step, (x, r, p, rs), xs=None, length=iters)
    return x


def _anderson(
    f: FixedPointFn,
    tol: float,
    max_iter: int,
    m: int,
    beta: float,
    reg: float,
    params: Array,
    x0: Array,
) -> Array:
    shape = x0.shape
    n = x0.size
    eye_m = jnp.eye(m, dtype=x0.dtype)

    def residual(xf: Array) -> Array:
        x = xf.reshape(shape)
        return (f(params, x) - x).reshape(-1)

    x0f = x0.reshape(-1)
    r0 = residual(x0f)
    # ``m + 1`` recent iterates / residuals -> ``m`` consecutive differences.
    x_buf = jnp.broadcast_to(x0f, (m + 1, n))
    r_buf = jnp.broadcast_to(r0, (m + 1, n))

    def cond(carry: Tuple[Array, Array, Array, Array, Array]) -> Array:
        _, _, x, r, i = carry
        return _residual_converged(r, x, i, tol, max_iter)

    def body(
        carry: Tuple[Array, Array, Array, Array, Array],
    ) -> Tuple[Array, Array, Array, Array, Array]:
        x_buf, r_buf, x, r, i = carry
        x_buf = jnp.concatenate([x_buf[1:], x[None]], axis=0)
        r_buf = jnp.concatenate([r_buf[1:], r[None]], axis=0)
        dx = x_buf[1:] - x_buf[:-1]  # (m, n)
        dr = r_buf[1:] - r_buf[:-1]  # (m, n)
        gram = dr @ dr.T  # (m, m)
        lam = reg * (jnp.trace(gram) / m + 1e-30)
        gamma = _small_spd_solve(gram + lam * eye_m, dr @ r, m)  # (m,)
        x_new = x + beta * r - (dx + beta * dr).T @ gamma
        return (x_buf, r_buf, x_new, residual(x_new), i + 1)

    init = (x_buf, r_buf, x0f, r0, jnp.asarray(1))
    _, _, x_star, _, _ = lax.while_loop(cond, body, init)
    return x_star.reshape(shape)


def _fixed_point_fwd(
    f: FixedPointFn,
    tol: float,
    max_iter: int,
    depth: int,
    beta: float,
    reg: float,
    params: Array,
    x0: Array,
) -> Tuple[Array, Tuple[Array, Array]]:
    x_star = _fixed_point(f, tol, max_iter, depth, beta, reg, params, x0)
    return x_star, (params, x_star)


def _fixed_point_bwd(
    f: FixedPointFn,
    tol: float,
    max_iter: int,
    depth: int,
    beta: float,
    reg: float,
    res: Tuple[Array, Array],
    g: Array,
) -> Tuple[Array, Array]:
    params, x_star = res
    # Adjoint fixed point: w = g + (∂_x f)ᵀ w (same engine / depth, so the
    # stiff regime is accelerated on the backward too).
    _, vjp_x = jax.vjp(lambda x: f(params, x), x_star)

    def adjoint(_: Array, w: Array) -> Array:
        return g + cast(Array, vjp_x(w)[0])

    w = _fixed_point(adjoint, tol, max_iter, depth, beta, reg, g, g)
    # Push the adjoint through the parameter dependence.
    _, vjp_params = jax.vjp(lambda p: f(p, x_star), params)
    (params_bar,) = vjp_params(w)
    return cast(Array, params_bar), jnp.zeros_like(x_star)


_fixed_point.defvjp(_fixed_point_fwd, _fixed_point_bwd)
