# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Small-residual nonlinear least squares: Gauss-Newton & Levenberg-Marquardt.

Minimise :math:`\\tfrac{1}{2}\\lVert r(x)\\rVert^2` for a residual map
:math:`r: \\mathbb{R}^P \\to \\mathbb{R}^M`.  The registration recipes are
the driving consumer (:math:`r` = warped-moving minus fixed, :math:`x` =
the transform's Lie parameters), but the optimiser is generic.

**Two Jacobian strategies, both GPU-native.**  Each step solves the
Gauss-Newton normal equations
:math:`(J^{\\top}J + \\lambda I)\\,\\delta = -J^{\\top}r`.

- ``jacobian='assembled'`` -- the ``'auto'`` default when the parameter
  count :math:`P` is small (the rigid/affine regime).  The
  :math:`M \\times P` Jacobian is materialised in :math:`P` forward-mode
  passes (``jax.jacfwd``), the explicit :math:`P \\times P` Gram matrix
  :math:`J^{\\top}J` is formed, and the tiny SPD system is solved by
  :func:`cg` on that explicit matrix.  Cheaper than the matrix-free matvec
  for small :math:`P`: :func:`cg` then costs :math:`P \\times P` matmuls,
  not a full warp tangent + transpose per inner iteration.
- ``jacobian='matrix_free'`` -- for a large :math:`P`.  The operator
  :math:`v \\mapsto J^{\\top}(Jv)` is assembled from a cached linearisation
  (``jax.linearize`` for :math:`Jv`, its ``jax.linear_transpose`` for
  :math:`J^{\\top}u`) -- so the :math:`M \\times P` Jacobian is **never
  materialised** -- and the SPD system is solved with :func:`cg` (matvecs
  only).

Both keep the hot loop on the GPU through the cuSolver wedge: no
factorisation, no host round-trip.

**The fixed-iteration loop is rolled with ``lax.scan``** (not
Python-unrolled), so cold-compile time is roughly constant in ``n_iters``
rather than linear -- one iteration is compiled, not ``n_iters`` copies.
It stays differentiable (``scan`` is), so ``jax.grad`` through a
registration works out of the box.  :func:`implicit_least_squares` adds the
implicit-function-theorem path: exact gradients at the optimum without
unrolling (:math:`O(1)` memory in the iteration count), the
differentiable-layer entry point for consumers like ``entense``.
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
        Final objective :math:`\\tfrac{1}{2}\\lVert r(\\mathrm{params})\\rVert^2`
        (scalar).
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
    """Build the matrix-free Gauss-Newton normal operator at ``x``.

    Linearise ``residual_fn`` at ``x`` once and return the cached pieces a
    Gauss-Newton step needs: the residual, the normal-equation matvec
    :math:`v \\mapsto J^{\\top}(Jv)`, and the right-hand side
    :math:`J^{\\top}r`.

    ``jax.linearize`` caches the forward linearisation (:math:`Jv`);
    ``jax.linear_transpose`` gives the adjoint (:math:`J^{\\top}u`).  The
    returned operator :math:`v \\mapsto J^{\\top}(Jv)` is the SPD
    normal-equation matvec, so the :math:`M \\times P` Jacobian is never
    materialised.

    Parameters
    ----------
    residual_fn
        Map :math:`x \\mapsto r(x)` (``(P,) -> (M,)``).
    x
        Point at which to linearise, ``(P,)``.

    Returns
    -------
    r : Array
        The residual :math:`r(x)`, ``(M,)``.
    jtj : Callable[[Array], Array]
        The SPD normal-equation matvec :math:`v \\mapsto J^{\\top}(Jv)`,
        mapping ``(P,) -> (P,)``.
    jt_r : Array
        The right-hand side :math:`J^{\\top}r`, ``(P,)``.
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
    """Materialise the Jacobian at ``x`` and return the explicit normal system.

    Returns the residual, the explicit :math:`P \\times P` Gram matrix
    :math:`J^{\\top}J`, and the right-hand side :math:`J^{\\top}r`.

    For a small parameter count :math:`P` the dense :math:`M \\times P`
    Jacobian is cheap to form (:math:`P` forward-mode passes), and the
    explicit :math:`P \\times P` Gram matrix turns the normal-equation solve
    into a tiny on-device :func:`cg` -- far less work than the matrix-free
    :math:`J^{\\top}(Jv)` matvec, which runs a full warp tangent + transpose
    per inner :func:`cg` iteration.

    Parameters
    ----------
    residual_fn
        Map :math:`x \\mapsto r(x)` (``(P,) -> (M,)``).
    x
        Point at which to materialise the Jacobian, ``(P,)``.
    jacobian_fn
        Optional map supplying the :math:`M \\times P` Jacobian in **closed
        form**.  When the residual is a registration warp, the analytic
        :math:`\\nabla M(\\mathrm{grid}) \\cdot \\partial \\mathrm{grid} /
        \\partial \\theta` does the expensive gather a handful of times rather
        than ``jax.jacfwd``'s :math:`P` times (the warp-tangent gather per
        parameter).  Its result must equal ``jax.jacfwd(residual_fn)(x)`` (the
        parity oracle).  If ``None`` (default), the Jacobian is obtained by
        forward-mode differentiation.

    Returns
    -------
    r : Array
        The residual :math:`r(x)`, ``(M,)``.
    jtj : Array
        The explicit Gram matrix :math:`J^{\\top}J`, ``(P, P)``.
    jt_r : Array
        The right-hand side :math:`J^{\\top}r`, ``(P,)``.
    """
    r = residual_fn(x)
    jac = jax.jacfwd(residual_fn)(x) if jacobian_fn is None else jacobian_fn(x)
    jac_h = jac.conj().T
    return r, jac_h @ jac, jac_h @ r


def _resolve_jacobian(strategy: JacobianStrategy, p: int) -> str:
    """Resolve ``'auto'`` to a concrete Jacobian strategy by parameter count.

    Parameters
    ----------
    strategy
        The requested strategy: ``'auto'``, ``'assembled'`` or
        ``'matrix_free'``.
    p
        The parameter count :math:`P`.

    Returns
    -------
    str
        The resolved strategy: ``'assembled'`` when ``strategy`` is
        ``'auto'`` and :math:`P` does not exceed the assemble threshold,
        ``'matrix_free'`` when it does, otherwise ``strategy`` unchanged.
    """
    if strategy == 'auto':
        return 'assembled' if p <= _ASSEMBLE_MAX_P else 'matrix_free'
    return strategy


def _operators(
    residual_fn: ResidualFn,
    x: Array,
    method: str,
    jacobian_fn: Optional[ResidualFn] = None,
) -> tuple[Array, NormalOperator, Array]:
    """Build the Gauss-Newton normal system via the selected strategy.

    Dispatches on ``method`` to return the residual, the normal operator
    :math:`J^{\\top}J` -- explicit ``(P, P)`` matrix for ``'assembled'`` or a
    matvec callable for ``'matrix_free'``, both accepted by :func:`cg` -- and
    the right-hand side :math:`J^{\\top}r`.

    Parameters
    ----------
    residual_fn
        Map :math:`x \\mapsto r(x)` (``(P,) -> (M,)``).
    x
        Point at which to build the normal system, ``(P,)``.
    method
        The resolved Jacobian strategy: ``'assembled'`` or ``'matrix_free'``.
    jacobian_fn
        Optional closed-form Jacobian, used only by the ``'assembled'`` path;
        see :func:`_assembled_operators`.

    Returns
    -------
    r : Array
        The residual :math:`r(x)`, ``(M,)``.
    a : NormalOperator
        The normal operator :math:`J^{\\top}J`: an explicit ``(P, P)`` matrix
        (``'assembled'``) or a ``(P,) -> (P,)`` matvec callable
        (``'matrix_free'``).
    jt_r : Array
        The right-hand side :math:`J^{\\top}r`, ``(P,)``.
    """
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
    ``threshold``.  The returned costs trace is length ``n_iters`` with the
    unrun tail padded by the final cost, so the caller assembles
    ``cost_history`` identically either way.  **Not reverse-differentiable** (a
    ``while_loop`` has no reverse rule); use ``early_stop=None`` for gradients.

    Parameters
    ----------
    step
        The per-iteration update, ``(state, None) -> (state, cost)``, in the
        signature ``lax.scan`` expects; ``cost`` is a scalar.
    init_state
        Initial carry for the iteration (an array or pytree).
    n_iters
        Number of iterations (the scan length, or the while-loop cap).
    early_stop
        Optional ``(threshold, window)`` enabling the windowed slope
        early-exit; ``None`` runs the fixed ``lax.scan``.
    dtype
        Floating dtype for the cost trace and slope buffers.

    Returns
    -------
    state : Any
        The final carry after the iterations.
    costs : Array
        The per-iteration cost, ``(n_iters,)``; when ``early_stop`` triggers,
        the unrun tail is padded by the final cost.
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
    """Gauss-Newton minimisation of :math:`\\tfrac{1}{2}\\lVert r(x)\\rVert^2`.

    Full undamped steps
    :math:`\\delta = -(J^{\\top}J + \\mathrm{damping} \\cdot I)^{-1} J^{\\top}r`
    (set ``damping > 0`` for a fixed Tikhonov ridge if :math:`J^{\\top}J` is
    near-singular).  Use :func:`levenberg_marquardt` when robustness to a poor
    starting point matters.

    Parameters
    ----------
    residual_fn
        Map :math:`x \\mapsto r(x)` (``(P,) -> (M,)``).
    x0
        Initial parameters, ``(P,)``.
    n_iters
        Number of Gauss-Newton steps.  The loop is rolled with
        ``lax.scan``, so compile time is roughly constant in ``n_iters``.
    damping
        Fixed diagonal ridge added to :math:`J^{\\top}J` (default ``0``).
    cg_tol
        Convergence tolerance for the inner :func:`cg` solve.
    cg_maxiter
        Iteration cap for the inner :func:`cg` solve; ``None`` uses the
        solver default.
    jacobian
        Normal-operator strategy: ``'assembled'`` (materialise
        :math:`J^{\\top}J`, for a small :math:`P`), ``'matrix_free'`` (matvec
        only, for a large :math:`P`), or ``'auto'`` (default; assembled when
        :math:`P` is small).
    jacobian_fn
        Optional closed-form Jacobian used only by the ``'assembled'`` path;
        its result must equal ``jax.jacfwd(residual_fn)(x)``.  If ``None``
        (default), the Jacobian is obtained by forward-mode differentiation.
    early_stop
        Opt-in early-exit ``(threshold, window)``: stop once the windowed
        normalised cost slope drops below ``threshold`` (else run all
        ``n_iters``).  ``None`` (default) keeps the fixed ``lax.scan`` --
        reproducible, ``vmap``-batchable, reverse-differentiable.  An
        ``early_stop`` runs a ``lax.while_loop`` (single-instance only; **not**
        reverse-differentiable -- a ``while_loop`` has no reverse rule).

    Returns
    -------
    OptimizeResult
        The minimiser and its cost trace.
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
    """Levenberg-Marquardt minimisation of
    :math:`\\tfrac{1}{2}\\lVert r(x)\\rVert^2`.

    Each step solves the damped normal equations
    :math:`(J^{\\top}J + \\lambda I)\\,\\delta = -J^{\\top}r` and accepts
    :math:`x + \\delta` only if the objective decreases; :math:`\\lambda`
    shrinks on accept (toward Gauss-Newton) and grows on reject (toward
    gradient descent).  The accept/reject branch is a ``jnp.where`` and the
    iteration is a ``lax.scan``, so the whole solve stays ``jit`` /
    ``grad``-friendly and compiles in roughly constant time in ``n_iters``.

    Parameters
    ----------
    residual_fn, x0, cg_tol, cg_maxiter, jacobian, jacobian_fn
        As :func:`gauss_newton`.
    n_iters
        Number of LM steps (each is one trial step + accept/reject).
    init_lambda
        Initial damping :math:`\\lambda`.
    lambda_up, lambda_down
        Multiplicative :math:`\\lambda` adjustment on reject / accept.
    early_stop
        Opt-in early-exit ``(threshold, window)`` -- as :func:`gauss_newton`
        (default ``None`` keeps the fixed ``lax.scan``).

    Returns
    -------
    OptimizeResult
        The minimiser and its cost trace; the ``cost_history`` is
        monotone-decreasing (guaranteed by the accept/reject rule).
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
    """Differentiable argmin of a least-squares residual by the implicit
    theorem.

    Compute the argmin of :math:`\\tfrac{1}{2}\\lVert r(\\mathrm{data}, x)
    \\rVert^2`, differentiable with respect to ``data`` by the
    **implicit-function theorem** rather than by unrolling the solver.

    The forward solve is Levenberg-Marquardt; the backward differentiates
    through the *converged* :math:`x^*` directly.  At the optimum the
    stationarity :math:`g(\\mathrm{data}, x^*) = J^{\\top}r = 0` gives, by the
    implicit-function theorem,
    :math:`\\partial x^* / \\partial \\mathrm{data} = -(J^{\\top}J)^{-1}
    \\partial_{\\mathrm{data}} g` (Gauss-Newton Hessian -- exact when the
    residual is small or the model is linear, the standard least-squares
    implicit-diff approximation).  The backward solves
    :math:`(J^{\\top}J)\\, w = \\bar{x}` with :func:`cg` (SPD, matrix-free --
    stays on-device) and pushes :math:`-w` back through
    :math:`\\partial_{\\mathrm{data}} g`.  Memory is :math:`O(1)` in the
    iteration count (vs unrolling's :math:`O(n\\_iters)`) and
    :math:`O(M + P)` per call: neither the :math:`M \\times P` Jacobian nor the
    :math:`P \\times P` normal matrix is materialised -- :func:`cg` holds
    :math:`O(P)` iterates and the linearisation caches one residual evaluation
    (:math:`O(M)` voxels), independent of :math:`P^2`.

    This is the differentiable-layer entry point: a registration whose
    residual is
    :math:`r(\\mathrm{images}, \\theta) = \\mathrm{warp}(\\mathrm{moving},
    \\theta) - \\mathrm{fixed}` becomes differentiable with respect to the
    images by calling this with ``data = (moving, fixed)`` (the unrolled
    recipe is the always-available alternative).

    Parameters
    ----------
    residual_fn
        Map ``(data, x) -> r`` (``data`` is the differentiable argument,
        an array or pytree; ``x`` is the optimisation variable, ``(P,)``,
        mapping to a residual ``(M,)``).
    data
        The differentiable parameters / inputs (an array or pytree).
    x0
        Initial guess for ``x``, ``(P,)``.
    n_iters, init_lambda
        Forward Levenberg-Marquardt controls.
    cg_tol, cg_maxiter
        Backward adjoint-solve (:func:`cg`) controls.

    Returns
    -------
    Float[Array, ' p']
        The minimiser :math:`x^*`, ``(P,)``.  :math:`\\mathrm{d}x^* /
        \\mathrm{d}x_0 = 0` (the initial guess does not affect the gradient).
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
    """Differentiable argmin of a scalar objective by the implicit theorem.

    Compute the argmin of a scalar objective :math:`f(\\mathrm{data}, x)`,
    differentiable with respect to ``data`` by the **implicit-function
    theorem**.

    The general-objective counterpart of :func:`implicit_least_squares`: where
    that assumes a least-squares residual and uses the Gauss-Newton Hessian
    :math:`J^{\\top}J`, this differentiates the argmin of an *arbitrary* scalar
    :math:`f` through the **exact** Hessian :math:`\\nabla^2_x f` -- so a
    registration whose cost is LNCC / MI / correlation-ratio (not a
    least-squares residual) becomes a differentiable layer too.

    The forward solve is BFGS (``jax.scipy.optimize.minimize``); the
    backward uses the stationarity
    :math:`g(\\mathrm{data}, x^*) = \\nabla_x f = 0`, which by the
    implicit-function theorem gives
    :math:`\\partial x^* / \\partial \\mathrm{data} = -(\\nabla^2_x f)^{-1}
    \\partial_{\\mathrm{data}} g`.  The adjoint
    :math:`(\\nabla^2_x f)\\, w = \\bar{x}` is solved matrix-free with
    :func:`cg` (Hessian-vector products via ``jax.linearize`` of the gradient
    -- stays on-device), then :math:`-w` is pushed back through
    :math:`\\partial_{\\mathrm{data}} g`.

    **Memory footprint.**  :math:`O(1)` in the BFGS iteration count -- nothing
    is unrolled.  The Hessian is never materialised: :func:`cg` applies it as a
    matrix-free Hessian-vector product, so the backward holds only the
    :func:`cg` iterates (a handful of ``x``-shaped vectors, :math:`O(P)` in the
    parameter count) plus the ``jax.linearize`` cache of a *single*
    :math:`\\nabla_x f` evaluation.  For an image-similarity objective the
    latter dominates and scales as the cost of one gradient -- :math:`O(M)` in
    the number of voxels -- *not* :math:`O(P^2)`: the dense :math:`P \\times P`
    Hessian is never formed.  So peak backward memory is :math:`O(M + P)`,
    independent of both the iteration count and :math:`P^2`.

    Parameters
    ----------
    objective_fn
        Map ``(data, x) -> scalar`` (``data`` the differentiable
        argument, an array or pytree; ``x`` the optimisation variable,
        ``(P,)``, mapping to a scalar objective).
    data
        The differentiable parameters / inputs (an array or pytree).
    x0
        Initial guess for ``x``, ``(P,)``.
    maxiter
        Forward BFGS iteration cap.
    ridge
        Non-negative Tikhonov ridge added to the Hessian in the backward
        adjoint solve (default ``0``); raise it if the Hessian at the
        optimum is near-singular (a minimum's Hessian is only positive
        semidefinite).
    cg_tol, cg_maxiter
        Backward adjoint-solve (:func:`cg`) controls.

    Returns
    -------
    Float[Array, ' p']
        The minimiser :math:`x^*`, ``(P,)``.  :math:`\\mathrm{d}x^* /
        \\mathrm{d}x_0 = 0` (the initial guess does not affect the gradient).

    Notes
    -----
    The implicit gradient is exact only at a true stationary point (the
    backward assumes :math:`\\nabla_x f(\\mathrm{data}, x^*) = 0`), so run the
    forward solve to convergence.  At a minimum :math:`\\nabla^2_x f` is
    positive semidefinite, so the SPD :func:`cg` applies -- pass ``ridge > 0``
    if it is only semidefinite.
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
    """Gradient :math:`\\nabla_x f(d, x)` -- the stationarity map the
    implicit-function backward uses.

    Parameters
    ----------
    objective_fn
        Map ``(data, x) -> scalar``.
    d
        The differentiable data argument (an array or pytree).
    x
        The point at which to take the gradient, ``(P,)``.

    Returns
    -------
    Array
        The gradient :math:`\\nabla_x f(d, x)`, ``(P,)``.
    """
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
