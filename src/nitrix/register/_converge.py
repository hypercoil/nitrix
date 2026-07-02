# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Shared iteration driver with optional windowed-slope early-exit.

The ANTs-style convergence criterion -- stop when the least-squares-line slope
over the last ``window`` per-iteration costs, normalised by the mean cost, drops
below ``threshold`` -- is factored out of the matrix inverse-compositional path
so that the dense-field stationary-velocity-field drivers share one
implementation.

:func:`run_iterations` runs a per-iteration ``step`` either as a fixed-length
``lax.scan`` (the default; reproducible and ``vmap``-batchable) or, when a
:class:`Convergence` is supplied, as a ``lax.while_loop`` that early-exits.  The
early-exit is offered **single-pair**: a ``vmap``-ed ``while_loop`` *does* batch
-- it runs to the **all-lanes** exit, the slowest lane setting the trip count
(volume registration uses exactly that batch-aggregate form) -- but the per-pair
early-exit advantage then degrades to the cohort's slowest pair.  It is
differentiable only through the implicit path (a ``while_loop`` has no reverse
rule), so use the implicit-function entry points for gradients, not the
unrolled trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array

__all__ = [
    'Convergence',
    'ConvergenceMode',
    'resolve_convergence_mode',
    'run_iterations',
]


@dataclass(frozen=True)
class Convergence:
    """Windowed cost-slope criterion for the ``mode='early_exit'`` iteration.

    The threshold/window pair that parameterises the early-exit ``lax.while_loop``
    (see :data:`ConvergenceMode`): per level, the loop runs until the cost has
    plateaued -- a least-squares-line fit over the last ``window`` per-iteration
    costs whose normalised slope falls below ``threshold`` -- or the level's
    iteration count (the hard cap) is reached.  It is **inert** under
    ``mode='fixed'`` (the fixed ``lax.scan``); the mode and this criterion are
    orthogonal, so a spec always carries a concrete :class:`Convergence`, used
    only when ``mode='early_exit'``.

    Attributes
    ----------
    threshold
        Stop when the windowed normalised cost slope drops below this.
    window
        Number of recent costs the slope is fit over (the convergence window).

    Notes
    -----
    An early-exit forward is **not** reverse-differentiable (``lax.while_loop``
    has no reverse rule); for a differentiable registration layer use
    ``mode='fixed'`` (the ``scan``) or the implicit-function path
    (``linalg.implicit_least_squares`` / ``implicit_minimize``), whose adjoint is
    solved at the optimum and is trajectory-independent.
    """

    threshold: float = 1e-6
    window: int = 10


# The iteration strategy, orthogonal to the :class:`Convergence`
# parameters: ``'fixed'`` is the reproducible, reverse-differentiable,
# ``vmap``-batchable ``lax.scan``; ``'early_exit'`` is the windowed-slope
# ``lax.while_loop`` (single-pair; not reverse-differentiable).  Eligibility is
# checked in the one ``resolve_convergence_mode`` gate below.
ConvergenceMode = Literal['fixed', 'early_exit']
"""The iteration strategy selecting how :func:`run_iterations` loops.

``'fixed'`` runs a fixed-length ``lax.scan``: reproducible,
reverse-differentiable, and ``vmap``-batchable.  ``'early_exit'`` runs a
windowed cost-slope ``lax.while_loop`` (see :class:`Convergence`) that stops
once the cost has plateaued; it is offered single-pair and is not
reverse-differentiable.  This mode is orthogonal to the :class:`Convergence`
threshold/window pair, which is consulted only when ``'early_exit'`` is chosen.
"""


def resolve_convergence_mode(
    mode: ConvergenceMode,
    convergence: Convergence,
    *,
    supports_early_exit: bool,
    path: str,
) -> Optional[Convergence]:
    """Resolve ``(mode, convergence)`` into the run-time iteration control.

    The single eligibility gate mapping a requested iteration ``mode`` onto the
    concrete criterion :func:`run_iterations` consumes.  ``mode='early_exit'``
    on a path that cannot honour a data-dependent trip count
    (``supports_early_exit=False`` -- for example a monolithic scalar/BFGS
    forward optimiser) raises a loud, actionable error naming ``path`` rather
    than failing obscurely downstream.  Every recipe routes its mode decision
    through here.

    Parameters
    ----------
    mode
        Requested iteration strategy: ``'fixed'`` for the reproducible,
        reverse-differentiable fixed-length ``lax.scan``, or ``'early_exit'``
        for the windowed cost-slope ``lax.while_loop``.
    convergence
        The threshold/window criterion to apply under ``'early_exit'``.  Always
        carried, but inert (returned only) when it is used.
    supports_early_exit
        Whether the calling path can honour a data-dependent trip count.  If
        ``False``, requesting ``'early_exit'`` raises rather than proceeding.
    path
        Human-readable name of the calling recipe, embedded in the error message
        when ``'early_exit'`` is unsupported.

    Returns
    -------
    Optional[Convergence]
        The concrete :class:`Convergence` to drive an early-exit
        ``lax.while_loop`` when ``mode='early_exit'``, or ``None`` for the fixed
        ``lax.scan`` when ``mode='fixed'``.

    Raises
    ------
    ValueError
        If ``mode`` is neither ``'fixed'`` nor ``'early_exit'``, or if
        ``mode='early_exit'`` is requested where ``supports_early_exit`` is
        ``False``.
    """
    if mode not in ('fixed', 'early_exit'):
        raise ValueError(
            f"mode must be 'fixed' or 'early_exit'; got {mode!r}."
        )
    if mode == 'fixed':
        return None
    if not supports_early_exit:
        raise ValueError(
            f"mode='early_exit' is not supported on {path} -- it cannot honour "
            "a data-dependent trip count.  Use mode='fixed' (the reproducible "
            'fixed scan), or the inverse-compositional recipes (IndexSpace + a '
            'least-squares metric such as SSD), which do.'
        )
    return convergence


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
    instead of JAX's generic while-loop-has-no-reverse-rule error.  The
    ``convergence=None`` (``lax.scan``) path never passes through the barrier,
    so it stays reverse-differentiable.

    Parameters
    ----------
    state
        The differentiable early-exit result (for example the estimated matrix
        or velocity field) carried out of the ``lax.while_loop``.

    Returns
    -------
    Any
        The same ``state``, unchanged, on the forward pass.
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
    """Run ``step_fn`` for a fixed count, or until the cost slope converges.

    With ``convergence=None`` (the default) this is exactly ``lax.scan`` over a
    fixed ``iterations``.  With a :class:`Convergence` it runs a
    ``lax.while_loop`` that stops once the windowed normalised cost slope drops
    below ``threshold`` (or the ``iterations`` hard cap is hit), returning a
    cost trace of length ``iterations`` whose unrun tail is padded with the
    final cost -- so the history keeps its shape either way.

    The fixed-``scan`` body is wrapped in ``jax.checkpoint``: reverse-mode
    through the unrolled trajectory then rematerialises each step in the backward
    pass instead of storing every iterate, so ``grad`` memory scales with the
    size of a single ``state`` plus recompute, rather than with
    ``iterations`` times the size of ``state`` (the dense-field
    stationary-velocity-field carry is large).  ``checkpoint`` is a no-op on the
    forward pass, so the (common, grad-free) forward result and its compile are
    byte-unchanged; only reverse mode trades the recompute for the memory.

    Parameters
    ----------
    step_fn
        Per-iteration body with the ``lax.scan`` signature
        ``step_fn(state, _) -> (state, cost)``, where ``cost`` is a scalar array
        giving the per-iteration cost.  The second argument is ignored (there
        are no scanned inputs).
    init_state
        Initial carry state (an arbitrary PyTree), threaded through every
        iteration.
    iterations
        Number of iterations for the fixed ``lax.scan``, or the hard cap on the
        early-exit ``lax.while_loop``.
    convergence
        If ``None``, use the fixed ``lax.scan``.  Otherwise, the threshold/window
        criterion driving the early-exit ``lax.while_loop``.
    dtype
        Floating-point dtype for the internal slope-fit time axis, cost buffer,
        and cost history.

    Returns
    -------
    state : Any
        The final carry state after the last iteration (same PyTree structure as
        ``init_state``).  On the early-exit path it is routed through an identity
        barrier that raises on reverse-mode differentiation.
    costs : Array
        The per-iteration cost trace, shape ``(iterations,)``.  On the early-exit
        path the entries after convergence are padded with the final cost.
    """
    if convergence is None:
        # remat the body so grad through the unrolled scan stays memory-bounded.
        return lax.scan(
            jax.checkpoint(step_fn), init_state, xs=None, length=iterations
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
    # Loud, actionable error on reverse-mode through the early-exit (the state
    # carries the differentiable result -- matrix / velocity field).
    return _early_exit_barrier(state), hist
