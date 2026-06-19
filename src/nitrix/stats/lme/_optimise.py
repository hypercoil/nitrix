# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
The one shared damped/saddle-free Newton optimiser for the variance-component
fits.

Every REML variance-component fit in this package minimises a (small,
unconstrained) profile negative-log-likelihood by a fixed-iteration damped
Newton with a backtracking line search.  Historically each solver
(``_blockwoodbury`` / ``_nested`` / ``_corrfit``) re-implemented that loop as its
own closure, which let the step rule drift -- the saddle-free fix below was at
one point present in only one copy, leaving the others exposed to a silent
non-convergence bug.  This module is the single source of truth.

Two curvature sources, one optimiser (the closed-form vs autodiff *fork*)
----------------------------------------------------------------------

``damped_newton`` takes the objective and, optionally, an explicit ``curvature``
callback returning ``(gradient, curvature_matrix)``:

- ``curvature=None`` -- the **autodiff** path: ``jax.grad`` + ``jax.hessian`` of
  ``nll``.  Used by the block-Woodbury / nested / correlation solvers, whose
  closed-form objectives are differentiated rather than hand-derived.
- ``curvature=<fn>`` -- the **analytic** path: the solver supplies its own score
  and curvature (e.g. the AI-REML average-information matrix), so no autodiff
  Hessian is formed.  This is the seam the analytic AI-REML engines
  (``_varcomp`` / ``_lowrank``) can adopt.

The step rule (``step=``)
------------------------

- ``'saddle_free'`` (default) -- the profile-REML objective is **non-convex away
  from the optimum** (the log-Cholesky / log-variance / correlation
  parameterisation has saddles), so the raw Hessian is indefinite there and a
  plain damped Newton step ``(H + lambda I)^{-1} g`` is *not* a descent
  direction -- the line search then stalls the iterate at its start and the fit
  silently returns the wrong variance components.  The saddle-free step
  eigendecomposes ``H`` and replaces each eigenvalue by ``|lambda|`` (floored),
  guaranteeing a descent direction (Dauphin et al. 2014); the backtracking line
  search then guarantees a monotone decrease.  The eigendecomposition is the
  cuSOLVER-free ``sym_eig_jacobi`` and rides under ``stop_gradient`` (it shapes
  the step, not the value, and differentiating the iterative eig would overflow).
- ``'damped'`` -- the classic Levenberg step ``(H + lambda I)^{-1} g`` via
  ``small_inv_logdet``.  Correct when ``H`` is PSD by construction -- e.g. the
  AI-REML average-information matrix -- so it is the right choice for the
  analytic-curvature solvers (and bit-for-bit with their current behaviour).

References
----------
- Dauphin, Y. et al. (2014). Identifying and attacking the saddle point problem
  in high-dimensional non-convex optimization. NeurIPS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, Optional, Tuple, cast

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from ...linalg._smalllinalg import small_inv_logdet, sym_eig_jacobi

if TYPE_CHECKING:
    # Annotation only -- importing at runtime would cycle (``_varcomp`` imports
    # ``damped_newton``).  Only ``spec``'s primitive fields are used here.
    from ._varcomp import VarCompSpec

__all__ = ['damped_newton']

# theta -> (gradient, curvature_matrix)
Curvature = Callable[[Float[Array, 'nt']], Tuple[Array, Array]]
StepRule = Literal['saddle_free', 'damped']


def _newton_direction(
    g: Float[Array, 'nt'],
    h: Float[Array, 'nt nt'],
    nt: int,
    spec: VarCompSpec,
    step: StepRule,
) -> Float[Array, 'nt']:
    """The (clipped) Newton step ``-H^{-1} g`` under the chosen step rule."""
    if step == 'saddle_free':
        # Replace eigenvalues by |lambda| (floored) -> guaranteed descent
        # direction even at an indefinite (saddle) Hessian.  The iterative eig
        # shapes the step only, so it rides under stop_gradient.
        evals, evecs = sym_eig_jacobi(lax.stop_gradient(h), nt)
        evals_pd = jnp.maximum(jnp.abs(evals), spec.damping)
        h_inv = (evecs / evals_pd) @ evecs.T
    else:
        h_damped = h + spec.damping * jnp.eye(nt, dtype=h.dtype)
        h_inv, _ = small_inv_logdet(h_damped, nt)
    return jnp.clip(h_inv @ g, -spec.max_step, spec.max_step)


def _backtrack(
    nll: Callable[[Float[Array, 'nt']], Float[Array, '']],
    theta: Float[Array, 'nt'],
    delta: Float[Array, 'nt'],
    spec: VarCompSpec,
) -> Float[Array, 'nt']:
    """Halving line search: take the largest ``scale`` in ``{1, 1/2, ...}`` that
    decreases ``nll`` (keep ``theta`` if none does)."""
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
    _, theta_new, _ = lax.fori_loop(0, spec.n_backtrack, bt, init)
    return cast(Float[Array, 'nt'], theta_new)


def damped_newton(
    nll: Callable[[Float[Array, 'nt']], Float[Array, '']],
    theta0: Float[Array, 'nt'],
    *,
    spec: VarCompSpec,
    curvature: Optional[Curvature] = None,
    step: StepRule = 'saddle_free',
) -> Float[Array, 'nt']:
    """Minimise ``nll`` from ``theta0`` by ``spec.n_iter`` damped Newton steps.

    Parameters
    ----------
    nll
        The scalar objective ``theta -> negative log-likelihood``.
    theta0
        Initial parameter vector.
    spec
        :class:`~nitrix.stats.lme._varcomp.VarCompSpec` supplying ``n_iter``,
        ``damping``, ``max_step``, ``n_backtrack``.
    curvature
        Optional ``theta -> (grad, curvature_matrix)``.  ``None`` (default) uses
        ``jax.grad`` + ``jax.hessian`` of ``nll`` (the autodiff fork); supply a
        callback to use an analytic score + curvature (e.g. an AI matrix).
    step
        ``'saddle_free'`` (default) or ``'damped'`` -- see the module docstring.

    Returns
    -------
    The optimised ``theta`` (the caller re-evaluates the objective for the
    fitted statistics).
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
        delta = _newton_direction(g, h, nt, spec, step)
        return _backtrack(nll, theta, delta, spec), None

    theta, _ = lax.scan(newton, theta0, xs=None, length=spec.n_iter)
    return theta
