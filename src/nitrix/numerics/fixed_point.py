# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Fixed-point solver with implicit differentiation.

:func:`fixed_point_solve` returns a solution :math:`x^\\ast` satisfying
:math:`x^\\ast = f(\\mathrm{params}, x^\\ast)` and -- via
``jax.custom_vjp`` -- differentiates through the *solution* by the
implicit-function theorem rather than by unrolling the iteration.  At the
fixed point, :math:`\\mathrm{d}x^\\ast / \\mathrm{d}\\,\\mathrm{params}`
satisfies :math:`(I - \\partial_x f)\\,\\mathrm{d}x^\\ast =
\\partial_{\\mathrm{params}} f`, so the backward pass solves the adjoint
system :math:`w = \\bar{g} + (\\partial_x f)^{\\top} w` (itself a fixed
point -- a contraction at a stable solution) and pushes :math:`w` back
through :math:`\\partial_{\\mathrm{params}} f`.  Memory is :math:`O(1)` in
the iteration count: the backward needs only :math:`x^\\ast`.

The two forward engines share the same implicit-function-theorem adjoint
on the backward pass:

- ``acceleration='picard'`` (default) -- plain Picard iteration
  :math:`x \\leftarrow f(\\mathrm{params}, x)`.  Converges geometrically
  with factor :math:`\\|\\partial_x f\\|`; fine when that is comfortably
  below one.
- ``acceleration='anderson'`` -- windowed Anderson acceleration (Walker
  and Ni), of the given depth.  A quasi-Newton secant mixing of the last
  ``depth`` residuals; converges where Picard *genuinely* stalls
  (:math:`\\|\\partial_x f\\| \\to 1`).  An opt-in escape hatch -- Picard
  is the better choice whenever it converges comfortably (Anderson's
  least-squares over a near-collinear residual history is marginally
  noisier there), which includes the smoothed
  :func:`~nitrix.geometry.invert_displacement` /
  :func:`~nitrix.geometry.field_log` square root (the half-damping
  already makes the latter super-linear).  The tiny
  :math:`\\mathrm{depth} \\times \\mathrm{depth}` least-squares is solved
  by an inlined conjugate gradient -- no factorisation, no cuSolver -- so
  it runs inside the traced ``while_loop`` on every backend.

The named consumers are :func:`~nitrix.geometry.invert_displacement` and
:func:`~nitrix.geometry.field_log`, which are fixed points, and the same
machinery underpins the differentiable-registration story.  It
generalises the scaling-and-squaring special case in
:func:`~nitrix.geometry.integrate_velocity_field`.
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
    """Solve :math:`x = f(\\mathrm{params}, x)` for :math:`x`, differentiably.

    Iterates the update map ``f`` to a fixed point and differentiates
    through the solution by the implicit-function theorem, so gradients
    with respect to ``params`` flow without unrolling the iteration.

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
        Relative convergence tolerance on the residual
        :math:`\\|f(x) - x\\|`.
    max_iter
        Iteration cap (forward and adjoint).
    acceleration
        ``'picard'`` (default) -- plain :math:`x \\leftarrow f(x)`.
        ``'anderson'`` -- windowed Anderson acceleration of the given
        depth, for the stiff :math:`\\|\\partial_x f\\| \\to 1` regime (a
        solve-free quasi-Newton mixing; runs in the traced loop on every
        backend).
    depth
        Anderson window (number of past residuals mixed).  Larger
        converges in fewer steps but holds roughly :math:`2 \\cdot
        \\mathrm{depth}` extra iterate-shaped buffers -- lower it for a
        brain-scale field where memory is tight.  Ignored for
        ``'picard'``.
    beta
        Anderson mixing (relaxation) factor; ``1.0`` is the standard
        choice.  Ignored for ``'picard'``.
    reg
        Relative Tikhonov ridge on the Anderson least-squares (keeps it
        well-posed when the residual history is near-collinear).  Ignored
        for ``'picard'``.

    Returns
    -------
    Float[Array, '...']
        The fixed point :math:`x^\\ast`, with the same shape as ``x0``.
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
    """Shared stopping rule for the fixed-point iterations.

    Reports whether the iteration should continue: it stops once the
    relative residual has converged (:math:`\\|r\\| \\le \\mathrm{tol}
    \\cdot \\|x\\|`) or the iteration cap has been reached.

    Parameters
    ----------
    r
        Current residual (the difference between successive iterates for
        Picard, or the fixed-point residual for Anderson).
    x
        Current iterate, used to scale the tolerance relatively.
    i
        Current iteration count.
    tol
        Relative convergence tolerance on the residual norm.
    max_iter
        Iteration cap.

    Returns
    -------
    Array
        A scalar boolean array that is ``True`` while the iteration
        should continue and ``False`` once it has converged or reached
        ``max_iter``.
    """
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

    Here ``a`` is the ``(depth, depth)`` Anderson normal matrix
    (symmetric positive definite after the ridge); conjugate gradient is
    exact in at most ``depth`` steps and uses only matrix-vector
    products, so it avoids cuSolver and runs inside the traced
    ``while_loop``.

    Parameters
    ----------
    a
        The ``(depth, depth)`` symmetric positive-definite normal matrix.
    b
        The ``(depth,)`` right-hand side.
    iters
        Number of conjugate-gradient steps to run (fixed for the traced
        scan; ``depth`` steps suffice for an exact solve).

    Returns
    -------
    Array
        The ``(depth,)`` solution ``x`` of ``a x = b``.
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
