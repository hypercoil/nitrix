# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Small-residual nonlinear least squares: Gauss-Newton & Levenberg-Marquardt.

Minimise ``½ ‖r(x)‖²`` for a residual map ``r: ℝᴾ -> ℝᴹ``.  The
registration recipes are the driving consumer (``r`` = warped-moving
minus fixed, ``x`` = the transform's Lie parameters), but the optimiser
is generic.

**Matrix-free and GPU-native.**  Each step solves the Gauss-Newton
normal equations ``(JᵀJ + λI) δ = -Jᵀr``.  The operator ``v ↦ Jᵀ(Jv)`` is
assembled from a cached linearisation (``jax.linearize`` for ``Jv``, its
``jax.linear_transpose`` for ``Jᵀu``) -- so the ``M×P`` Jacobian is
**never materialised** -- and the SPD system is solved with ``cg``
(matvecs only).  Both facts keep the hot loop on the GPU through the
cuSolver wedge: no factorisation, no host round-trip.

The unrolled fixed-iteration loop is differentiable, so ``jax.grad``
through a registration works out of the box.  ``implicit_least_squares``
adds the implicit-function-theorem path: exact gradients at the optimum
without unrolling (O(1) memory in the iteration count), the
differentiable-layer entry point for consumers like ``entense``.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, NamedTuple, Optional, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .krylov import cg

__all__ = [
    'OptimizeResult',
    'gauss_newton',
    'levenberg_marquardt',
    'implicit_least_squares',
]

ResidualFn = Callable[[Float[Array, ' p']], Float[Array, ' m']]
DataResidualFn = Callable[[Any, Float[Array, ' p']], Float[Array, ' m']]


class OptimizeResult(NamedTuple):
    """Result of a nonlinear-least-squares solve.

    Attributes
    ----------
    params
        The optimised parameter vector, ``(P,)``.
    cost
        Final objective ``½ ‖r(params)‖²`` (scalar).
    cost_history
        Objective before each step and after the last, ``(n_iters + 1,)``
        -- a monotone-decreasing trace for LM (accept/reject guarantees
        it).
    """

    params: Float[Array, ' p']
    cost: Float[Array, '']
    cost_history: Float[Array, ' h']


def _half_sq(r: Array) -> Array:
    return 0.5 * jnp.vdot(r, r).real


def _normal_operators(
    residual_fn: ResidualFn,
    x: Array,
) -> tuple[Array, Callable[[Array], Array], Array]:
    """Linearise ``residual_fn`` at ``x`` once and return ``(r, JᵀJ-op,
    Jᵀr)`` -- the cached pieces a Gauss-Newton step needs.

    ``jax.linearize`` caches the forward linearisation (``jv = J v``);
    ``jax.linear_transpose`` gives the adjoint (``Jᵀ u``).  The returned
    operator ``v -> Jᵀ(Jv)`` is the SPD normal-equation matvec.
    """
    r, jvp_fn = jax.linearize(residual_fn, x)
    vjp_fn = jax.linear_transpose(jvp_fn, x)

    def jtj(v: Array) -> Array:
        (out,) = vjp_fn(jvp_fn(v))
        return cast(Array, out)

    (jt_r,) = vjp_fn(r)
    return r, jtj, cast(Array, jt_r)


def gauss_newton(
    residual_fn: ResidualFn,
    x0: Float[Array, ' p'],
    *,
    n_iters: int = 10,
    damping: float = 0.0,
    cg_tol: float = 1e-6,
    cg_maxiter: Optional[int] = None,
) -> OptimizeResult:
    """Gauss-Newton minimisation of ``½ ‖r(x)‖²``.

    Full undamped steps ``δ = -(JᵀJ + damping·I)⁻¹ Jᵀr`` (set
    ``damping > 0`` for a fixed Tikhonov ridge if ``JᵀJ`` is
    near-singular).  Use ``levenberg_marquardt`` when robustness to a
    poor starting point matters.

    Parameters
    ----------
    residual_fn
        Map ``x -> r(x)`` (``(P,) -> (M,)``).
    x0
        Initial parameters, ``(P,)``.
    n_iters
        Number of Gauss-Newton steps.
    damping
        Fixed diagonal ridge added to ``JᵀJ`` (default ``0``).
    cg_tol, cg_maxiter
        Tolerance / iteration cap for the inner ``cg`` solve.

    Returns
    -------
    ``OptimizeResult``.
    """
    x = x0
    costs = []
    for _ in range(n_iters):
        r, jtj, jt_r = _normal_operators(residual_fn, x)
        costs.append(_half_sq(r))
        delta = cg(jtj, -jt_r, l2=damping, tol=cg_tol, maxiter=cg_maxiter)
        x = x + delta
    r_final = residual_fn(x)
    costs.append(_half_sq(r_final))
    return OptimizeResult(
        params=x, cost=costs[-1], cost_history=jnp.stack(costs)
    )


def levenberg_marquardt(
    residual_fn: ResidualFn,
    x0: Float[Array, ' p'],
    *,
    n_iters: int = 20,
    init_lambda: float = 1e-3,
    lambda_up: float = 10.0,
    lambda_down: float = 0.1,
    cg_tol: float = 1e-6,
    cg_maxiter: Optional[int] = None,
) -> OptimizeResult:
    """Levenberg-Marquardt minimisation of ``½ ‖r(x)‖²``.

    Each step solves the damped normal equations ``(JᵀJ + λI) δ =
    -Jᵀr`` and accepts ``x + δ`` only if the objective decreases;
    ``λ`` shrinks on accept (toward Gauss-Newton) and grows on reject
    (toward gradient descent).  The accept/reject branch is a
    ``jnp.where`` so the whole solve stays ``jit`` / ``grad``-friendly.

    Parameters
    ----------
    residual_fn, x0, cg_tol, cg_maxiter
        As ``gauss_newton``.
    n_iters
        Number of LM steps (each is one trial step + accept/reject).
    init_lambda
        Initial damping ``λ``.
    lambda_up, lambda_down
        Multiplicative ``λ`` adjustment on reject / accept.

    Returns
    -------
    ``OptimizeResult`` with a monotone-decreasing ``cost_history``.
    """
    x = x0
    lam = jnp.asarray(init_lambda, dtype=x0.dtype)
    cost = _half_sq(residual_fn(x))
    costs = [cost]
    for _ in range(n_iters):
        r, jtj, jt_r = _normal_operators(residual_fn, x)
        delta = cg(jtj, -jt_r, l2=lam, tol=cg_tol, maxiter=cg_maxiter)
        x_new = x + delta
        cost_new = _half_sq(residual_fn(x_new))
        accept = cost_new < cost
        x = jnp.where(accept, x_new, x)
        cost = jnp.where(accept, cost_new, cost)
        lam = jnp.where(accept, lam * lambda_down, lam * lambda_up)
        costs.append(cost)
    return OptimizeResult(params=x, cost=cost, cost_history=jnp.stack(costs))


def implicit_least_squares(
    residual_fn: DataResidualFn,
    data: Any,
    x0: Float[Array, ' p'],
    *,
    n_iters: int = 20,
    init_lambda: float = 1e-3,
    cg_tol: float = 1e-6,
    cg_maxiter: Optional[int] = None,
) -> Float[Array, ' p']:
    """Argmin of ``½‖r(data, x)‖²`` that is differentiable w.r.t. ``data``
    by the **implicit-function theorem** (not by unrolling the solver).

    The forward solve is Levenberg-Marquardt; the backward differentiates
    through the *converged* ``x*`` directly.  At the optimum the
    stationarity ``g(data, x*) = Jᵀr = 0`` gives, by the IFT,
    ``∂x*/∂data = -(JᵀJ)⁻¹ ∂_data g`` (Gauss-Newton Hessian -- exact when
    the residual is small or the model is linear, the standard
    least-squares implicit-diff approximation).  The backward solves
    ``(JᵀJ) w = x̄`` with ``cg`` (SPD, matrix-free -- stays on-device) and
    pushes ``-w`` back through ``∂_data g``.  Memory is O(1) in the
    iteration count, vs unrolling's O(n_iters).

    This is the differentiable-layer entry point: a registration whose
    residual is ``r(images, θ) = warp(moving, θ) - fixed`` becomes
    differentiable w.r.t. the images by calling this with ``data =
    (moving, fixed)`` (the unrolled recipe is the always-available
    alternative).

    Parameters
    ----------
    residual_fn
        Map ``(data, x) -> r`` (``data`` is the differentiable argument,
        an array or pytree; ``x`` is the optimisation variable).
    data
        The differentiable parameters / inputs.
    x0
        Initial guess for ``x``.
    n_iters, init_lambda
        Forward Levenberg-Marquardt controls.
    cg_tol, cg_maxiter
        Backward adjoint-solve (``cg``) controls.

    Returns
    -------
    The minimiser ``x*``.  ``dx*/dx0 = 0`` (the initial guess does not
    affect the gradient).
    """
    return _implicit_ls(
        residual_fn, n_iters, init_lambda, cg_tol, cg_maxiter, data, x0
    )


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def _implicit_ls(
    residual_fn: DataResidualFn,
    n_iters: int,
    init_lambda: float,
    cg_tol: float,
    cg_maxiter: Optional[int],
    data: Any,
    x0: Array,
) -> Array:
    res = levenberg_marquardt(
        lambda x: residual_fn(data, x),
        x0,
        n_iters=n_iters,
        init_lambda=init_lambda,
        cg_tol=cg_tol,
    )
    return res.params


def _implicit_ls_fwd(
    residual_fn: DataResidualFn,
    n_iters: int,
    init_lambda: float,
    cg_tol: float,
    cg_maxiter: Optional[int],
    data: Any,
    x0: Array,
) -> tuple[Array, tuple[Any, Array]]:
    x_star = _implicit_ls(
        residual_fn, n_iters, init_lambda, cg_tol, cg_maxiter, data, x0
    )
    return x_star, (data, x_star)


def _implicit_ls_bwd(
    residual_fn: DataResidualFn,
    n_iters: int,
    init_lambda: float,
    cg_tol: float,
    cg_maxiter: Optional[int],
    res: tuple[Any, Array],
    x_bar: Array,
) -> tuple[Any, Array]:
    data, x_star = res
    # Gauss-Newton Hessian operator JᵀJ at x*, then solve (JᵀJ) w = x̄.
    _, jvp_fn = jax.linearize(lambda x: residual_fn(data, x), x_star)
    vjp_fn = jax.linear_transpose(jvp_fn, x_star)

    def jtj(v: Array) -> Array:
        (out,) = vjp_fn(jvp_fn(v))
        return cast(Array, out)

    w = cg(jtj, x_bar, tol=cg_tol, maxiter=cg_maxiter)

    # Stationarity g(data) = Jᵀr at fixed x*; data_bar = -(∂_data g)ᵀ w.
    def stationarity(d: Any) -> Array:
        r, vjp_x = jax.vjp(lambda x: residual_fn(d, x), x_star)
        return cast(Array, vjp_x(r)[0])

    _, vjp_data = jax.vjp(stationarity, data)
    (data_bar,) = vjp_data(w)
    data_bar = jax.tree_util.tree_map(lambda z: -z, data_bar)
    return data_bar, jnp.zeros_like(x_star)


_implicit_ls.defvjp(_implicit_ls_fwd, _implicit_ls_bwd)
