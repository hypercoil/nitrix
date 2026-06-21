# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Small-residual nonlinear least squares: Gauss-Newton & Levenberg-Marquardt.

Minimise ``½ ‖r(x)‖²`` for a residual map ``r: ℝᴾ -> ℝᴹ``.  The
registration recipes are the driving consumer (``r`` = warped-moving
minus fixed, ``x`` = the transform's Lie parameters), but the optimiser
is generic.

**Two Jacobian strategies, both GPU-native.**  Each step solves the
Gauss-Newton normal equations ``(JᵀJ + λI) δ = -Jᵀr``.

- ``jacobian='assembled'`` -- the ``auto`` default when the parameter
  count ``P`` is small (the rigid/affine regime).  The ``M×P`` Jacobian
  is materialised in ``P`` forward-mode passes (``jax.jacfwd``), the
  explicit ``P×P`` Gram matrix ``JᵀJ`` is formed, and the tiny SPD
  system is solved by ``cg`` on that explicit matrix.  Cheaper than the
  matrix-free matvec for small ``P``: ``cg`` then costs ``P×P`` matmuls,
  not a full warp tangent + transpose per inner iteration.
- ``jacobian='matrix_free'`` -- for a large ``P``.  The operator
  ``v ↦ Jᵀ(Jv)`` is assembled from a cached linearisation
  (``jax.linearize`` for ``Jv``, its ``jax.linear_transpose`` for
  ``Jᵀu``) -- so the ``M×P`` Jacobian is **never materialised** -- and
  the SPD system is solved with ``cg`` (matvecs only).

Both keep the hot loop on the GPU through the cuSolver wedge: no
factorisation, no host round-trip.

**The fixed-iteration loop is rolled with ``lax.scan``** (not
Python-unrolled), so cold-compile time is ~constant in ``n_iters``
rather than linear -- one iteration is compiled, not ``n_iters`` copies.
It stays differentiable (``scan`` is), so ``jax.grad`` through a
registration works out of the box.  ``implicit_least_squares`` adds the
implicit-function-theorem path: exact gradients at the optimum without
unrolling (O(1) memory in the iteration count), the differentiable-layer
entry point for consumers like ``entense``.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Literal, NamedTuple, Optional, Union, cast

import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from jaxtyping import Array, Float

from .krylov import cg

__all__ = [
    'OptimizeResult',
    'gauss_newton',
    'levenberg_marquardt',
    'implicit_least_squares',
    'implicit_minimize',
]

ResidualFn = Callable[[Float[Array, ' p']], Float[Array, ' m']]
DataResidualFn = Callable[[Any, Float[Array, ' p']], Float[Array, ' m']]
ScalarObjectiveFn = Callable[[Any, Float[Array, ' p']], Float[Array, '']]

# A Gauss-Newton step needs ``(JᵀJ + λI) δ = -Jᵀr`` solved.  The normal
# operator ``JᵀJ`` is supplied either as an explicit ``P×P`` matrix
# (``'assembled'``) or as a matvec callable (``'matrix_free'``); ``cg``
# accepts both.
NormalOperator = Union[Float[Array, ' p p'], Callable[[Array], Array]]
JacobianStrategy = Literal['auto', 'matrix_free', 'assembled']

# ``'auto'`` materialises the Jacobian at or below this parameter count
# (the rigid/affine regime is ``P ≤ 12``); above it the matrix-free
# matvec wins.
_ASSEMBLE_MAX_P = 64


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


def _assembled_operators(
    residual_fn: ResidualFn,
    x: Array,
    jacobian_fn: Optional[ResidualFn] = None,
) -> tuple[Array, Array, Array]:
    """Materialise the Jacobian at ``x`` and return ``(r, JᵀJ, Jᵀr)``.

    For a small parameter count ``P`` the dense ``M×P`` Jacobian is cheap
    to form (``P`` forward-mode passes), and the explicit ``P×P`` Gram
    matrix turns the normal-equation solve into a tiny on-device ``cg``
    -- far less work than the matrix-free ``Jᵀ(Jv)`` matvec, which runs a
    full warp tangent + transpose per inner ``cg`` iteration.

    ``jacobian_fn`` (optional) supplies the ``M×P`` Jacobian in **closed form**
    -- when the residual is a registration warp, the analytic
    ``∇M(grid) · ∂grid/∂θ`` does the expensive gather a handful of times rather
    than ``jax.jacfwd``'s ``P`` times (the warp-tangent gather per parameter).
    Its result must equal ``jax.jacfwd(residual_fn)(x)`` (the parity oracle).
    """
    r = residual_fn(x)
    jac = jax.jacfwd(residual_fn)(x) if jacobian_fn is None else jacobian_fn(x)
    jac_h = jac.conj().T
    return r, jac_h @ jac, jac_h @ r


def _resolve_jacobian(strategy: JacobianStrategy, p: int) -> str:
    """Resolve ``'auto'`` to ``'assembled'`` / ``'matrix_free'`` by ``P``."""
    if strategy == 'auto':
        return 'assembled' if p <= _ASSEMBLE_MAX_P else 'matrix_free'
    return strategy


def _operators(
    residual_fn: ResidualFn,
    x: Array,
    method: str,
    jacobian_fn: Optional[ResidualFn] = None,
) -> tuple[Array, NormalOperator, Array]:
    """``(r, JᵀJ, Jᵀr)`` with ``JᵀJ`` explicit (``'assembled'``) or a
    matvec callable (``'matrix_free'``) -- both accepted by ``cg``."""
    if method == 'assembled':
        return _assembled_operators(residual_fn, x, jacobian_fn)
    return _normal_operators(residual_fn, x)


def _iterate(
    step: Callable[[Any, Any], tuple[Any, Array]],
    init_state: Any,
    *,
    n_iters: int,
    early_stop: Optional[tuple[float, int]],
    dtype: Any,
) -> tuple[Any, Array]:
    """Run ``step`` for ``n_iters`` (fixed scan) or until the cost slope converges.

    ``early_stop=None`` (default) is exactly ``lax.scan`` over ``n_iters`` --
    reproducible, ``vmap``-batchable and reverse-differentiable (the unchanged
    optimiser behaviour).  An ``early_stop=(threshold, window)`` runs a
    ``lax.while_loop`` that stops once the least-squares-line slope over the last
    ``window`` per-iteration costs, normalised by the mean, drops below
    ``threshold`` (the ANTs criterion).  The returned ``costs`` is length
    ``n_iters`` with the unrun tail padded by the final cost, so the caller
    assembles ``cost_history`` identically either way.  **Not
    reverse-differentiable** (a ``while_loop`` has no reverse rule); use
    ``early_stop=None`` for gradients.  (Mirrors
    ``register._converge.run_iterations`` -- kept local to avoid a
    ``linalg``->``register`` import inversion.)
    """
    if early_stop is None:
        return jax.lax.scan(step, init_state, xs=None, length=n_iters)
    threshold, window = early_stop
    t = jnp.arange(window, dtype=dtype)
    t_centred = t - jnp.mean(t)
    t_var = jnp.sum(t_centred * t_centred)

    def converged(buf: Array) -> Array:
        slope = jnp.sum(t_centred * (buf - jnp.mean(buf))) / t_var
        return jnp.abs(slope) / (jnp.abs(jnp.mean(buf)) + 1e-12) < threshold

    def cond(carry: tuple[Any, Array, Array, Array]) -> Array:
        _, i, buf, _ = carry
        return (i < n_iters) & ((i < window) | ~converged(buf))

    def body(
        carry: tuple[Any, Array, Array, Array],
    ) -> tuple[Any, Array, Array, Array]:
        state, i, buf, hist = carry
        state, cost = step(state, None)
        buf = jnp.concatenate([buf[1:], cost[None]])
        return state, i + 1, buf, hist.at[i].set(cost)

    init = (
        init_state,
        jnp.asarray(0),
        jnp.zeros((window,), dtype=dtype),
        jnp.zeros((n_iters,), dtype=dtype),
    )
    state, last_i, buf, hist = jax.lax.while_loop(cond, body, init)
    hist = jnp.where(jnp.arange(n_iters) < last_i, hist, buf[-1])
    return state, hist


def gauss_newton(
    residual_fn: ResidualFn,
    x0: Float[Array, ' p'],
    *,
    n_iters: int = 10,
    damping: float = 0.0,
    cg_tol: float = 1e-6,
    cg_maxiter: Optional[int] = None,
    jacobian: JacobianStrategy = 'auto',
    jacobian_fn: Optional[ResidualFn] = None,
    early_stop: Optional[tuple[float, int]] = None,
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
        Number of Gauss-Newton steps.  The loop is rolled with
        ``lax.scan``, so compile time is ~constant in ``n_iters``.
    damping
        Fixed diagonal ridge added to ``JᵀJ`` (default ``0``).
    cg_tol, cg_maxiter
        Tolerance / iteration cap for the inner ``cg`` solve.
    jacobian
        Normal-operator strategy: ``'assembled'`` (materialise ``JᵀJ``,
        for a small ``P``), ``'matrix_free'`` (matvec only, for a large
        ``P``), or ``'auto'`` (default; assembled when ``P`` is small).
    early_stop
        Opt-in early-exit ``(threshold, window)``: stop once the windowed
        normalised cost slope drops below ``threshold`` (else run all
        ``n_iters``).  ``None`` (default) keeps the fixed ``lax.scan`` --
        reproducible, ``vmap``-batchable, reverse-differentiable.  An
        ``early_stop`` runs a ``lax.while_loop`` (single-instance only; **not**
        reverse-differentiable -- a ``while_loop`` has no reverse rule).

    Returns
    -------
    ``OptimizeResult``.
    """
    method = _resolve_jacobian(jacobian, x0.shape[-1])

    def step(x: Array, _: Any) -> tuple[Array, Array]:
        r, a, jt_r = _operators(residual_fn, x, method, jacobian_fn)
        delta = cg(a, -jt_r, l2=damping, tol=cg_tol, maxiter=cg_maxiter)
        return x + delta, _half_sq(r)

    x, costs = _iterate(
        step, x0, n_iters=n_iters, early_stop=early_stop, dtype=x0.dtype
    )
    cost_final = _half_sq(residual_fn(x))
    cost_history = jnp.concatenate([costs, cost_final[None]])
    return OptimizeResult(params=x, cost=cost_final, cost_history=cost_history)


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
    jacobian: JacobianStrategy = 'auto',
    jacobian_fn: Optional[ResidualFn] = None,
    early_stop: Optional[tuple[float, int]] = None,
) -> OptimizeResult:
    """Levenberg-Marquardt minimisation of ``½ ‖r(x)‖²``.

    Each step solves the damped normal equations ``(JᵀJ + λI) δ =
    -Jᵀr`` and accepts ``x + δ`` only if the objective decreases;
    ``λ`` shrinks on accept (toward Gauss-Newton) and grows on reject
    (toward gradient descent).  The accept/reject branch is a
    ``jnp.where`` and the iteration is a ``lax.scan``, so the whole solve
    stays ``jit`` / ``grad``-friendly and compiles in ~constant time in
    ``n_iters``.

    Parameters
    ----------
    residual_fn, x0, cg_tol, cg_maxiter, jacobian
        As ``gauss_newton``.
    n_iters
        Number of LM steps (each is one trial step + accept/reject).
    init_lambda
        Initial damping ``λ``.
    lambda_up, lambda_down
        Multiplicative ``λ`` adjustment on reject / accept.
    early_stop
        Opt-in early-exit ``(threshold, window)`` -- as ``gauss_newton``
        (default ``None`` keeps the fixed ``lax.scan``).

    Returns
    -------
    ``OptimizeResult`` with a monotone-decreasing ``cost_history``.
    """
    method = _resolve_jacobian(jacobian, x0.shape[-1])
    lam0 = jnp.asarray(init_lambda, dtype=x0.dtype)
    cost0 = _half_sq(residual_fn(x0))

    def step(
        carry: tuple[Array, Array, Array], _: Any
    ) -> tuple[tuple[Array, Array, Array], Array]:
        x, lam, cost = carry
        r, a, jt_r = _operators(residual_fn, x, method, jacobian_fn)
        delta = cg(a, -jt_r, l2=lam, tol=cg_tol, maxiter=cg_maxiter)
        x_new = x + delta
        cost_new = _half_sq(residual_fn(x_new))
        accept = cost_new < cost
        x = jnp.where(accept, x_new, x)
        cost = jnp.where(accept, cost_new, cost)
        lam = jnp.where(accept, lam * lambda_down, lam * lambda_up)
        return (x, lam, cost), cost

    (x, _, cost), costs = _iterate(
        step,
        (x0, lam0, cost0),
        n_iters=n_iters,
        early_stop=early_stop,
        dtype=x0.dtype,
    )
    cost_history = jnp.concatenate([cost0[None], costs])
    return OptimizeResult(params=x, cost=cost, cost_history=cost_history)


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
    iteration count (vs unrolling's O(n_iters)) and ``O(M + P)`` per
    call: neither the ``M×P`` Jacobian nor the ``P×P`` normal matrix is
    materialised -- ``cg`` holds O(P) iterates and the linearisation
    caches one residual evaluation (O(M) voxels), independent of ``P²``.

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


def implicit_minimize(
    objective_fn: ScalarObjectiveFn,
    data: Any,
    x0: Float[Array, ' p'],
    *,
    maxiter: int = 50,
    ridge: float = 0.0,
    cg_tol: float = 1e-6,
    cg_maxiter: Optional[int] = None,
) -> Float[Array, ' p']:
    """Argmin of a scalar objective ``f(data, x)`` that is differentiable
    w.r.t. ``data`` by the **implicit-function theorem**.

    The general-objective counterpart of ``implicit_least_squares``: where
    that assumes a least-squares residual and uses the Gauss-Newton
    Hessian ``JᵀJ``, this differentiates the argmin of an *arbitrary*
    scalar ``f`` through the **exact** Hessian ``∇²_x f`` -- so a
    registration whose cost is LNCC / MI / correlation-ratio (not a
    least-squares residual) becomes a differentiable layer too.

    The forward solve is BFGS (``jax.scipy.optimize.minimize``); the
    backward uses the stationarity ``g(data, x*) = ∇_x f = 0``, which by
    the IFT gives ``∂x*/∂data = -(∇²_x f)⁻¹ ∂_data g``.  The adjoint
    ``(∇²_x f) w = x̄`` is solved matrix-free with ``cg`` (Hessian-vector
    products via ``jax.linearize`` of the gradient -- stays on-device),
    then ``-w`` is pushed back through ``∂_data g``.

    **Memory footprint.**  O(1) in the BFGS iteration count -- nothing is
    unrolled.  The Hessian is never materialised: ``cg`` applies it as a
    matrix-free Hessian-vector product, so the backward holds only the
    ``cg`` iterates (a handful of ``x``-shaped vectors, **O(P)** in the
    parameter count) plus the ``jax.linearize`` cache of a *single*
    ``∇_x f`` evaluation.  For an image-similarity objective the latter
    dominates and scales as the cost of one gradient -- **O(M)** in the
    number of voxels -- **not O(P²)**: the dense ``P×P`` Hessian is never
    formed.  So peak backward memory is ``O(M + P)``, independent of both
    the iteration count and ``P²``.

    Parameters
    ----------
    objective_fn
        Map ``(data, x) -> scalar`` (``data`` the differentiable
        argument, an array or pytree; ``x`` the optimisation variable).
    data
        The differentiable parameters / inputs.
    x0
        Initial guess for ``x``.
    maxiter
        Forward BFGS iteration cap.
    ridge
        Non-negative Tikhonov ridge added to the Hessian in the backward
        adjoint solve (default ``0``); raise it if the Hessian at the
        optimum is near-singular (a min's Hessian is only PSD).
    cg_tol, cg_maxiter
        Backward adjoint-solve (``cg``) controls.

    Returns
    -------
    The minimiser ``x*``.  ``dx*/dx0 = 0`` (the initial guess does not
    affect the gradient).

    Notes
    -----
    The implicit gradient is exact only at a true stationary point (the
    backward assumes ``∇_x f(data, x*) = 0``), so run the forward solve to
    convergence.  At a minimum ``∇²_x f`` is positive semidefinite, so the
    SPD ``cg`` applies -- pass ``ridge > 0`` if it is only semidefinite.
    """
    return _implicit_min(
        objective_fn, maxiter, ridge, cg_tol, cg_maxiter, data, x0
    )


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def _implicit_min(
    objective_fn: ScalarObjectiveFn,
    maxiter: int,
    ridge: float,
    cg_tol: float,
    cg_maxiter: Optional[int],
    data: Any,
    x0: Array,
) -> Array:
    res = minimize(
        lambda x: objective_fn(data, x),
        x0,
        method='BFGS',
        options={'maxiter': maxiter},
    )
    return res.x


def _implicit_min_fwd(
    objective_fn: ScalarObjectiveFn,
    maxiter: int,
    ridge: float,
    cg_tol: float,
    cg_maxiter: Optional[int],
    data: Any,
    x0: Array,
) -> tuple[Array, tuple[Any, Array]]:
    x_star = _implicit_min(
        objective_fn, maxiter, ridge, cg_tol, cg_maxiter, data, x0
    )
    return x_star, (data, x_star)


def _grad_x(objective_fn: ScalarObjectiveFn, d: Any, x: Array) -> Array:
    """``∇_x f(d, x)`` -- the stationarity map the IFT backward uses."""
    return cast(Array, jax.grad(lambda y: objective_fn(d, y))(x))


def _implicit_min_bwd(
    objective_fn: ScalarObjectiveFn,
    maxiter: int,
    ridge: float,
    cg_tol: float,
    cg_maxiter: Optional[int],
    res: tuple[Any, Array],
    x_bar: Array,
) -> tuple[Any, Array]:
    data, x_star = res
    # Exact-Hessian operator ∇²_x f at x* (Hvp via linearise of the
    # gradient), then solve (∇²_x f + ridge·I) w = x̄.
    _, hvp = jax.linearize(lambda x: _grad_x(objective_fn, data, x), x_star)

    def hessian(v: Array) -> Array:
        return cast(Array, hvp(v)) + ridge * v

    w = cg(hessian, x_bar, tol=cg_tol, maxiter=cg_maxiter)

    # Stationarity g(data) = ∇_x f at fixed x*; data_bar = -(∂_data g)ᵀ w.
    _, vjp_data = jax.vjp(lambda d: _grad_x(objective_fn, d, x_star), data)
    (data_bar,) = vjp_data(w)
    data_bar = jax.tree_util.tree_map(lambda z: -z, data_bar)
    return data_bar, jnp.zeros_like(x_star)


_implicit_min.defvjp(_implicit_min_fwd, _implicit_min_bwd)
