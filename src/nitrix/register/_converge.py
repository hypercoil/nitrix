# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Shared iteration driver with optional windowed-slope early-exit (C7 / A3).

The ANTs-style convergence criterion -- stop when the least-squares-line slope
over the last ``window`` per-iteration costs, normalised by the mean cost, drops
below ``threshold`` -- factored out of the matrix inverse-compositional path
(``_inverse_compositional._ic_level_converge``) so the dense-field SVF drivers
(``_svf``) share one implementation (followup C7; A3 is its first SVF consumer).

``run_iterations`` runs a per-iteration ``step`` either as a fixed-length
``lax.scan`` (the default; reproducible and ``vmap``-batchable) or, when a
:class:`Convergence` is supplied, as a ``lax.while_loop`` that early-exits.  The
early-exit is offered **single-pair**: a ``vmap``-ed ``while_loop`` *does* batch
-- it runs to the **all-lanes** exit, the slowest lane setting the trip count
(``volreg`` uses exactly that batch-aggregate form) -- but the per-pair
early-exit advantage then degrades to the cohort's slowest pair.  It is
differentiable only through the implicit path (a ``while_loop`` has no reverse
rule), so use ``implicit_*`` for gradients, not the unrolled trajectory.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array

from ._core import Convergence

__all__ = ['Convergence', 'run_iterations']

_EARLY_EXIT_NO_REVERSE = (
    'reverse-mode differentiation through an early-exit (lax.while_loop) '
    'registration is not supported -- a while_loop has no reverse rule, so the '
    'data-dependent iteration count cannot be back-propagated.  For gradients, '
    'use the implicit-function entry points (linalg.implicit_least_squares / '
    'implicit_minimize), which differentiate at the optimum and are '
    "trajectory-independent; or set mode='fixed' to restore the "
    'reverse-differentiable fixed-iteration scan.'
)


@jax.custom_vjp
def _early_exit_barrier(state: Any) -> Any:
    """Identity on the forward pass; a loud, actionable raise on the reverse.

    Applied to the ``while_loop`` early-exit output so that a ``jax.grad``
    through the early-exit path raises the registration-specific contract error
    (above) -- the "loud fallbacks" tenet -- instead of JAX's generic
    while_loop-no-reverse error.  ``convergence=None`` (the ``lax.scan`` path)
    never goes through the barrier, so it stays reverse-differentiable.
    """
    return state


def _barrier_fwd(state: Any) -> tuple[Any, None]:
    return state, None


def _barrier_bwd(_: None, cotangent: Any) -> tuple[Any]:
    raise RuntimeError(_EARLY_EXIT_NO_REVERSE)


_early_exit_barrier.defvjp(_barrier_fwd, _barrier_bwd)

# Scan-style step: ``(state, xs) -> (state, cost)`` with ``cost`` a scalar; the
# same signature ``lax.scan`` consumes, so a level fn's body is reused verbatim.
StepFn = Callable[[Any, Any], Tuple[Any, Array]]


def run_iterations(
    step_fn: StepFn,
    init_state: Any,
    *,
    iterations: int,
    convergence: Optional[Convergence],
    dtype: Any,
) -> Tuple[Any, Array]:
    """Run ``step_fn`` for ``iterations``, or until the cost slope converges.

    ``step_fn(state, _) -> (state, cost)`` (a scalar per-iteration cost).  With
    ``convergence=None`` (default) this is exactly ``lax.scan`` over a fixed
    ``iterations``.  With a :class:`Convergence` it runs a ``lax.while_loop``
    that stops once the windowed normalised cost slope drops below
    ``threshold`` (or the ``iterations`` hard cap is hit), returning a
    ``(iterations,)`` ``costs`` trace whose unrun tail is padded with the final
    cost -- so ``cost_history`` keeps its shape either way.
    """
    if convergence is None:
        return lax.scan(step_fn, init_state, xs=None, length=iterations)

    window = convergence.window
    threshold = convergence.threshold
    t = jnp.arange(window, dtype=dtype)
    t_centred = t - jnp.mean(t)
    t_var = jnp.sum(t_centred * t_centred)

    def converged(buf: Array) -> Array:
        slope = jnp.sum(t_centred * (buf - jnp.mean(buf))) / t_var
        return jnp.abs(slope) / (jnp.abs(jnp.mean(buf)) + 1e-12) < threshold

    def cond(carry: Tuple[Any, Array, Array, Array]) -> Array:
        _, i, buf, _ = carry
        return (i < iterations) & ((i < window) | ~converged(buf))

    def body(
        carry: Tuple[Any, Array, Array, Array],
    ) -> Tuple[Any, Array, Array, Array]:
        state, i, buf, hist = carry
        state, cost = step_fn(state, None)
        buf = jnp.concatenate([buf[1:], cost[None]])
        return state, i + 1, buf, hist.at[i].set(cost)

    # The buffer / trace seeds are fully overwritten before they are read (the
    # ``i < window`` guard delays the slope test until the buffer is real; the
    # final ``where`` pads every unrun index), so any seed value works.
    init = (
        init_state,
        jnp.asarray(0),
        jnp.zeros((window,), dtype=dtype),
        jnp.zeros((iterations,), dtype=dtype),
    )
    state, last_i, buf, hist = lax.while_loop(cond, body, init)
    hist = jnp.where(jnp.arange(iterations) < last_i, hist, buf[-1])
    # Loud, actionable error on reverse-mode through the early-exit (the state
    # carries the differentiable result -- matrix / velocity field).
    return _early_exit_barrier(state), hist
