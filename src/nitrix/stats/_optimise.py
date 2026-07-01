# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
The one shared damped/saddle-free Newton optimiser for the small-parameter fits.

Every restricted-maximum-likelihood (REML) variance-component fit -- and the
ordinal cumulative-link fit -- minimises a small, unconstrained
negative-log-likelihood by a fixed-iteration damped Newton with a backtracking
line search.  This module is the single source of truth for that loop, so the
step rule cannot drift between solvers.

It is a generic small-problem optimiser, so it takes its iteration budget as
primitive keyword arguments (``n_iter`` / ``damping`` / ``max_step`` /
``n_backtrack``) rather than a mixed-model variance-components specification --
the non-mixed-model callers (e.g. the ordinal fit) do not have to build a
specification they have no use for.  A :class:`VarCompSpec` unpacks itself via
``**spec.newton_kwargs`` at the mixed-model call sites.

Two curvature sources, one optimiser (the closed-form vs autodiff fork)
-----------------------------------------------------------------------

:func:`damped_newton` takes the objective and, optionally, an explicit
``curvature`` callback returning ``(gradient, curvature_matrix)``:

- ``curvature=None`` -- the **autodiff** path: ``jax.grad`` + ``jax.hessian`` of
  ``nll``.  Used by the block-Woodbury / nested / correlation solvers, whose
  closed-form objectives are differentiated rather than hand-derived.
- ``curvature=<fn>`` -- the **analytic** path: the solver supplies its own score
  and curvature (e.g. the average-information REML matrix), so no autodiff
  Hessian is formed.  This is the seam the analytic average-information REML
  engines adopt.

The step rule (``step=``)
-------------------------

- ``'saddle_free'`` (default) -- the profile-REML objective is **non-convex away
  from the optimum** (the log-Cholesky / log-variance / correlation
  parameterisation has saddles), so the raw Hessian is indefinite there and a
  plain damped Newton step :math:`(H + \\lambda I)^{-1} g` is *not* a descent
  direction -- the line search then stalls the iterate at its start and the fit
  silently returns the wrong variance components.  The saddle-free step
  eigendecomposes :math:`H` and replaces each eigenvalue by :math:`|\\lambda|`
  (floored), guaranteeing a descent direction (Dauphin et al. 2014); the
  backtracking line search then guarantees a monotone decrease.  The
  eigendecomposition is the cuSOLVER-free :func:`sym_eig_jacobi` and rides under
  ``stop_gradient`` (it shapes the step, not the value, and differentiating the
  iterative eigendecomposition would overflow).
- ``'damped'`` -- the classic Levenberg step :math:`(H + \\lambda I)^{-1} g` via
  :func:`small_inv_logdet`.  Correct when :math:`H` is positive semi-definite by
  construction -- e.g. the average-information REML matrix -- so it is the right
  choice for the analytic-curvature solvers.

References
----------
Dauphin, Y., Pascanu, R., Gulcehre, C., Cho, K., Ganguli, S., & Bengio, Y.
(2014). Identifying and attacking the saddle point problem in high-dimensional
non-convex optimization. *Advances in Neural Information Processing Systems 27*.
https://arxiv.org/abs/1406.2572
"""

from __future__ import annotations

from typing import Callable, Literal, Optional, Tuple, cast

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from ..linalg._smalllinalg import small_inv_logdet, sym_eig_jacobi

__all__ = ['damped_newton']

# theta -> (gradient, curvature_matrix)
Curvature = Callable[[Float[Array, 'nt']], Tuple[Array, Array]]
StepRule = Literal['saddle_free', 'damped']


def _newton_direction(
    g: Float[Array, 'nt'],
    h: Float[Array, 'nt nt'],
    nt: int,
    damping: float,
    max_step: float,
    step: StepRule,
) -> Float[Array, 'nt']:
    """Compute the clipped Newton step :math:`-H^{-1} g` under a step rule.

    For ``step='saddle_free'`` the curvature matrix is eigendecomposed and each
    eigenvalue is replaced by its absolute value (floored at ``damping``),
    yielding a guaranteed descent direction even at an indefinite (saddle)
    Hessian.  For ``step='damped'`` the classic Levenberg step
    :math:`(H + \\lambda I)^{-1} g` is formed instead.  The resulting update is
    clipped to ``[-max_step, max_step]`` per axis.

    Parameters
    ----------
    g
        The gradient (score) vector, shape ``(nt,)``.
    h
        The curvature matrix, shape ``(nt, nt)``.
    nt
        The number of parameters (the leading dimension of ``g`` and ``h``).
    damping
        Eigenvalue floor (saddle-free) or Levenberg ridge (damped) stabilising
        the step.
    max_step
        Per-axis magnitude at which the update is clipped.
    step
        Either ``'saddle_free'`` or ``'damped'``, selecting the step rule.

    Returns
    -------
    Float[Array, 'nt']
        The clipped Newton update :math:`\\delta`, shape ``(nt,)``.
    """
    if step == 'saddle_free':
        # Replace eigenvalues by |lambda| (floored) -> guaranteed descent
        # direction even at an indefinite (saddle) Hessian.  The iterative eig
        # shapes the step only, so it rides under stop_gradient.
        evals, evecs = sym_eig_jacobi(lax.stop_gradient(h), nt)
        evals_pd = jnp.maximum(jnp.abs(evals), damping)
        h_inv = (evecs / evals_pd) @ evecs.T
    else:
        h_damped = h + damping * jnp.eye(nt, dtype=h.dtype)
        h_inv, _ = small_inv_logdet(h_damped, nt)
    return jnp.clip(h_inv @ g, -max_step, max_step)


def _backtrack(
    nll: Callable[[Float[Array, 'nt']], Float[Array, '']],
    theta: Float[Array, 'nt'],
    delta: Float[Array, 'nt'],
    n_backtrack: int,
) -> Float[Array, 'nt']:
    """Backtracking halving line search along a proposed Newton direction.

    Try step scales :math:`1, \\tfrac{1}{2}, \\tfrac{1}{4}, \\dots` for
    ``n_backtrack`` halvings and keep the trial ``theta - scale * delta`` with
    the lowest objective seen; if no trial beats the starting objective, the
    starting ``theta`` is returned unchanged.

    Parameters
    ----------
    nll
        The scalar objective ``theta -> negative log-likelihood``.
    theta
        The current parameter vector, shape ``(nt,)``.
    delta
        The proposed update direction, shape ``(nt,)``; trials subtract scaled
        multiples of it from ``theta``.
    n_backtrack
        The number of halvings to try.

    Returns
    -------
    Float[Array, 'nt']
        The accepted parameter vector, shape ``(nt,)`` -- the best trial found,
        or ``theta`` if none decreased the objective.
    """
    nll_old = nll(theta)

    def bt(
        _: Array, carry: Tuple[Array, Array, Array]
    ) -> Tuple[Array, Array, Array]:
        scale, best, best_nll = carry
        trial = theta - scale * delta
        trial_nll = nll(trial)
        ok = trial_nll < best_nll
        return (
            scale * 0.5,
            jnp.where(ok, trial, best),
            jnp.where(ok, trial_nll, best_nll),
        )

    init = (jnp.asarray(1.0, theta.dtype), theta, nll_old)
    _, theta_new, _ = lax.fori_loop(0, n_backtrack, bt, init)
    return cast(Float[Array, 'nt'], theta_new)


def damped_newton(
    nll: Callable[[Float[Array, 'nt']], Float[Array, '']],
    theta0: Float[Array, 'nt'],
    *,
    n_iter: int,
    damping: float = 1e-6,
    max_step: float = 1.0,
    n_backtrack: int = 4,
    curvature: Optional[Curvature] = None,
    step: StepRule = 'saddle_free',
) -> Float[Array, 'nt']:
    """Minimise ``nll`` from ``theta0`` by ``n_iter`` damped Newton steps.

    Parameters
    ----------
    nll
        The scalar objective ``theta -> negative log-likelihood``.
    theta0
        Initial parameter vector.
    n_iter
        Fixed number of Newton-scoring iterations.
    damping
        Levenberg / eigenvalue floor stabilising the step near a boundary.
    max_step
        Per-axis clip on the update (Newton overshoot guard).
    n_backtrack
        Backtracking halvings tried when the full step does not decrease ``nll``.
    curvature
        Optional ``theta -> (grad, curvature_matrix)``.  ``None`` (default) uses
        ``jax.grad`` + ``jax.hessian`` of ``nll`` (the autodiff fork); supply a
        callback to use an analytic score + curvature (e.g. an AI matrix).
    step
        ``'saddle_free'`` (default) or ``'damped'`` -- see the module docstring.

    Returns
    -------
    The optimised ``theta`` (the caller re-evaluates the objective for the
    fitted statistics).  Mixed-model callers pass ``**spec.newton_kwargs`` to
    forward their :class:`VarCompSpec` budget.
    """
    nt = theta0.shape[0]
    if curvature is None:
        grad_fn = jax.grad(nll)
        hess_fn = jax.hessian(nll)

        def curv(th: Float[Array, 'nt']) -> Tuple[Array, Array]:
            return grad_fn(th), hess_fn(th)

        curvature = curv

    curv_fn: Curvature = curvature

    def newton(theta: Float[Array, 'nt'], _: Array) -> Tuple[Array, None]:
        g, h = curv_fn(theta)
        delta = _newton_direction(g, h, nt, damping, max_step, step)
        return _backtrack(nll, theta, delta, n_backtrack), None

    theta, _ = lax.scan(newton, theta0, xs=None, length=n_iter)
    return theta
