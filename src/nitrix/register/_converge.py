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
early-exit is **single-pair only** -- a data-dependent trip count is not
``vmap``-batchable -- exactly as on the matrix IC path; it is differentiable
only through the implicit path (a ``while_loop`` has no reverse rule), so use
``implicit_*`` for gradients, not the unrolled trajectory.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array

from ._core import Convergence

__all__ = ['Convergence', 'run_iterations']

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
        return lax.scan(
            step_fn, init_state, xs=None, length=iterations
        )

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
    return state, hist
