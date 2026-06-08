# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Fixed-point solver with implicit differentiation.

``fixed_point_solve(f, params, x0)`` returns ``x*`` with ``x* = f(params,
x*)`` by Picard iteration, and -- via ``jax.custom_vjp`` -- differentiates
through the *solution* by the implicit-function theorem rather than by
unrolling the iteration.  At the fixed point, ``dx*/dparams`` satisfies
``(I - ∂_x f) dx* = ∂_params f``, so the backward solves the adjoint
system ``w = ḡ + (∂_x f)ᵀ w`` (itself a fixed point -- a contraction at a
stable solution) and pushes ``w`` back through ``∂_params f``.  Memory is
O(1) in the iteration count: the backward needs only ``x*``.

This is the §12.8 graduation (``numerics.fixed_point``); registration is
the named consumer -- ``geometry.invert_displacement`` is a fixed point
(``s_inv = -s∘(id + s_inv)``), and the same machinery underpins the
differentiable-registration story.  Generalises the scaling-and-squaring
special case in ``geometry.integrate_velocity_field``.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Tuple, cast

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

__all__ = ['fixed_point_solve']

FixedPointFn = Callable[[Array, Array], Array]


def fixed_point_solve(
    f: FixedPointFn,
    params: Float[Array, '...'],
    x0: Float[Array, '...'],
    *,
    tol: float = 1e-6,
    max_iter: int = 100,
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
        Relative convergence tolerance on ``‖x_{k+1} - x_k‖``.
    max_iter
        Iteration cap (forward and adjoint).

    Returns
    -------
    The fixed point ``x*``.
    """
    return _fixed_point(f, tol, max_iter, params, x0)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _fixed_point(
    f: FixedPointFn,
    tol: float,
    max_iter: int,
    params: Array,
    x0: Array,
) -> Array:
    def cond(carry: Tuple[Array, Array, Array]) -> Array:
        x_prev, x, i = carry
        delta = x - x_prev
        moved = jnp.sqrt(jnp.vdot(delta, delta).real)
        scale = jnp.sqrt(jnp.vdot(x, x).real)
        return (i < max_iter) & (moved > tol * (scale + 1e-12))

    def body(
        carry: Tuple[Array, Array, Array],
    ) -> Tuple[Array, Array, Array]:
        _, x, i = carry
        return (x, f(params, x), i + 1)

    x1 = f(params, x0)
    _, x_star, _ = lax.while_loop(cond, body, (x0, x1, jnp.asarray(1)))
    return x_star


def _fixed_point_fwd(
    f: FixedPointFn,
    tol: float,
    max_iter: int,
    params: Array,
    x0: Array,
) -> Tuple[Array, Tuple[Array, Array]]:
    x_star = _fixed_point(f, tol, max_iter, params, x0)
    return x_star, (params, x_star)


def _fixed_point_bwd(
    f: FixedPointFn,
    tol: float,
    max_iter: int,
    res: Tuple[Array, Array],
    g: Array,
) -> Tuple[Array, Array]:
    params, x_star = res
    # Adjoint fixed point: w = g + (∂_x f)ᵀ w.
    _, vjp_x = jax.vjp(lambda x: f(params, x), x_star)

    def adjoint(_: Array, w: Array) -> Array:
        return g + cast(Array, vjp_x(w)[0])

    w = _fixed_point(adjoint, tol, max_iter, g, g)
    # Push the adjoint through the parameter dependence.
    _, vjp_params = jax.vjp(lambda p: f(p, x_star), params)
    (params_bar,) = vjp_params(w)
    return cast(Array, params_bar), jnp.zeros_like(x_star)


_fixed_point.defvjp(_fixed_point_fwd, _fixed_point_bwd)
