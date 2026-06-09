# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Fixed-step ODE integrators.

Pure-JAX explicit integrators for ``dy/dt = f(t, y)``, returning the state
at each requested time point.  These are the substrate for continuous-time
("neural-ODE") models -- a portable, differentiable alternative to a
third-party ODE library (the integration is a ``lax.scan``, so reverse-mode
``grad`` differentiates straight through the solver).

- ``euler`` -- the first-order explicit step (cheap, low accuracy).
- ``rk4`` -- the classic fourth-order Runge--Kutta step (the default).
- ``odeint`` -- dispatch over ``method``.

One step is taken per consecutive pair of time points, so granularity is
controlled by how densely ``t`` is sampled.  The vector field takes
``f(t, y)`` (close over any extra parameters in the caller).

Roadmap (per the §12.11 catalogue entry): adaptive steppers
(Dormand--Prince), symplectic steppers (leapfrog / implicit-midpoint), and
the memory-efficient adjoint backward (a fixed-point + Krylov solve) layer
on the same ``f(t, y)`` interface.
"""

from __future__ import annotations

from typing import Callable, Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = ['euler', 'rk4', 'odeint']

VectorField = Callable[[Array, Array], Array]
Method = Literal['euler', 'rk4']


def _euler_step(f: VectorField, t: Array, dt: Array, y: Array) -> Array:
    return y + dt * f(t, y)


def _rk4_step(f: VectorField, t: Array, dt: Array, y: Array) -> Array:
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _integrate(
    step: Callable[[VectorField, Array, Array, Array], Array],
    f: VectorField,
    y0: Float[Array, '...'],
    t: Float[Array, 't'],
) -> Float[Array, 't ...']:
    def scan_step(y: Array, pair: Array) -> tuple[Array, Array]:
        t0, t1 = pair[0], pair[1]
        y_next = step(f, t0, t1 - t0, y)
        return y_next, y_next

    pairs = jnp.stack([t[:-1], t[1:]], axis=-1)
    _, ys = jax.lax.scan(scan_step, y0, pairs)
    return jnp.concatenate([y0[None], ys], axis=0)


def euler(
    f: VectorField,
    y0: Float[Array, '...'],
    t: Float[Array, 't'],
) -> Float[Array, 't ...']:
    """Integrate ``dy/dt = f(t, y)`` with the explicit Euler method.

    Parameters
    ----------
    f
        Vector field ``f(t, y) -> dy/dt`` (``t`` scalar, ``y`` the state).
    y0
        Initial state at ``t[0]``.
    t
        Strictly increasing (or decreasing) time points; one Euler step is
        taken per interval.

    Returns
    -------
    States at each time point, shape ``(len(t), *y0.shape)`` (``ys[0]`` is
    ``y0``).
    """
    return _integrate(_euler_step, f, y0, t)


def rk4(
    f: VectorField,
    y0: Float[Array, '...'],
    t: Float[Array, 't'],
) -> Float[Array, 't ...']:
    """Integrate ``dy/dt = f(t, y)`` with classic 4th-order Runge--Kutta.

    Same interface as :func:`euler`; far more accurate per step (4th-order
    local error), the sensible default for smooth vector fields.
    """
    return _integrate(_rk4_step, f, y0, t)


def odeint(
    f: VectorField,
    y0: Float[Array, '...'],
    t: Float[Array, 't'],
    *,
    method: Method = 'rk4',
) -> Float[Array, 't ...']:
    """Integrate ``dy/dt = f(t, y)`` with the chosen fixed-step ``method``.

    ``method="rk4"`` (default) or ``"euler"``.
    """
    if method == 'rk4':
        return rk4(f, y0, t)
    if method == 'euler':
        return euler(f, y0, t)
    raise ValueError(f"method={method!r}; expected 'rk4' or 'euler'.")
