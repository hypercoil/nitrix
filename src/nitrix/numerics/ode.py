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
- ``midpoint`` -- the second-order explicit-midpoint step (RK2).
- ``rk4`` -- the classic fourth-order Runge--Kutta step (the default).
- ``local_linearization`` -- the exponential / local-linearization step
  (Ozaki; the scheme behind SPM's ``spm_int``): linearize ``f`` about the
  current state and integrate the linear part *exactly* via a matrix
  exponential.  A-stable -- the method for *stiff* systems (e.g. the
  Balloon--Windkessel haemodynamic model) where an explicit step needs a
  punishingly small ``dt``.  Exact (to round-off) for affine-autonomous
  ``f(y) = A y + b``.  Requires a flat 1-D state (it builds the Jacobian);
  ``vmap`` it for batched / multi-region integration.
- ``odeint`` -- dispatch over ``method``.

One step is taken per consecutive pair of time points, so granularity is
controlled by how densely ``t`` is sampled.  The vector field takes
``f(t, y)``; extra parameters are closed over in the caller -- e.g.
``odeint(lambda t, y: field(t, y, args), y0, t)``.  There is deliberately
*no* ``args=`` keyword: the integrator's contract is a bare ``f(t, y)``, and
closed-over arrays differentiate and ``jit`` exactly as a kwarg would.

The step arithmetic is the textbook explicit-Runge--Kutta tableau (the same
``Euler`` / explicit-``Midpoint`` / classic ``RK4`` steps a library such as
``diffrax`` takes for those fixed-step methods), so a model whose weights
were trained or evaluated against that arithmetic reproduces here -- the
bring-your-own-weights parity requirement of the surface neural-ODE backend
swap (see ``docs/feature-requests/ode-integrators.md``).

Roadmap (per the §12.11 catalogue entry and ``dynamics-suite.md``): *adaptive*
steppers (the embedded Dormand--Prince ``dopri5`` / ``dopri8`` and ``tsit5``
pairs -- a ``while_loop`` + step-size controller, a different control-flow
shape from this static ``scan``), *symplectic* steppers (leapfrog /
velocity-Verlet, for Hamiltonian flows and HMC), *stochastic* steppers (the
SDE family -- Euler--Maruyama / stochastic Heun), and the memory-efficient
continuous-adjoint backward (a fixed-point + Krylov solve) layering on the
same ``f(t, y)`` interface.
"""

from __future__ import annotations

from typing import Callable, Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..linalg import matrix_exp

__all__ = ['euler', 'midpoint', 'rk4', 'local_linearization', 'odeint']

VectorField = Callable[[Array, Array], Array]
Method = Literal['euler', 'midpoint', 'rk4', 'local_linearization']


def _euler_step(f: VectorField, t: Array, dt: Array, y: Array) -> Array:
    return y + dt * f(t, y)


def _midpoint_step(f: VectorField, t: Array, dt: Array, y: Array) -> Array:
    # Explicit midpoint (RK2): evaluate the field at the half-step.
    k1 = f(t, y)
    return y + dt * f(t + 0.5 * dt, y + 0.5 * dt * k1)


def _rk4_step(f: VectorField, t: Array, dt: Array, y: Array) -> Array:
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _local_linearization_step(
    f: VectorField, t: Array, dt: Array, y: Array
) -> Array:
    # Local-linearization (Ozaki; SPM ``spm_int``).  Freeze the input over the
    # step, linearize ``f`` about ``y`` -- ``f(y + d) ~ f_n + J d`` with
    # ``J = df/dy|_y`` -- and integrate that linear ODE *exactly*.  The exact
    # increment is ``(e^{dt J} - I) J^{-1} f_n``, which the augmented matrix
    # exponential delivers without inverting ``J`` (so it is singular-safe):
    #
    #     e^{dt [[J, f_n],[0, 0]]} = [[e^{dt J}, (e^{dt J} - I) J^{-1} f_n],
    #                                 [0,        1                        ]]
    #
    # -> the top-right column is the increment.  Exact (to round-off) for an
    # affine-autonomous ``f(y) = A y + b`` (the linearization is then exact),
    # and A-stable -- the reason it integrates stiff systems where an explicit
    # step would diverge.  ``y`` must be a flat 1-D state (the Jacobian is an
    # ``(n, n)`` matrix); ``vmap`` over a batch / region axis for many states.
    n = y.shape[-1]
    f_n = f(t, y)
    jac = jax.jacobian(lambda yy: f(t, yy))(y)
    top = jnp.concatenate([jac, f_n[:, None]], axis=-1)
    bottom = jnp.zeros((1, n + 1), dtype=y.dtype)
    aug = jnp.concatenate([top, bottom], axis=0)
    propagated = matrix_exp(dt * aug)
    return y + propagated[:n, n]


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


def midpoint(
    f: VectorField,
    y0: Float[Array, '...'],
    t: Float[Array, 't'],
) -> Float[Array, 't ...']:
    """Integrate ``dy/dt = f(t, y)`` with the explicit midpoint method (RK2).

    Same interface as :func:`euler`; second-order local error -- a step takes
    two field evaluations (``f`` at the start and at the half-step) for an
    extra order of accuracy.  This is the explicit RK2 stepper, *not* the
    symplectic implicit-midpoint method (which requires a per-step solve);
    it completes the ``[euler, midpoint, rk4]`` method set of the surface
    neural-ODE consumers.
    """
    return _integrate(_midpoint_step, f, y0, t)


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


def local_linearization(
    f: VectorField,
    y0: Float[Array, 'n'],
    t: Float[Array, 't'],
) -> Float[Array, 't n']:
    """Integrate ``dy/dt = f(t, y)`` by exponential / local linearization.

    The Ozaki local-linearization scheme (the integrator behind SPM's
    ``spm_int``): each step linearizes ``f`` about the current state and
    advances the linear part *exactly* with a matrix exponential.  Unlike the
    explicit ``euler`` / ``midpoint`` / ``rk4`` steppers it is **A-stable**, so
    it integrates **stiff** systems -- such as the Balloon--Windkessel
    haemodynamic model -- at a step size where an explicit method would need a
    far finer grid or diverge outright.  It is **exact (to round-off)** for an
    affine-autonomous field ``f(y) = A y + b``.

    Same ``f(t, y)`` / close-over-args contract as :func:`euler`, with two
    differences from the explicit steppers:

    - ``y0`` must be a **flat 1-D** state ``(n,)`` -- the step forms the
      ``(n, n)`` Jacobian ``df/dy``.  For a batched / multi-region system,
      ``jax.vmap`` this integrator over the batch axis.
    - the input is held constant across each step (piecewise-constant /
      frozen-input linearization), so accuracy on a fast-varying drive is
      controlled, as for the explicit methods, by how densely ``t`` is sampled.

    Differentiable straight through the ``scan`` (the matrix exponential is
    pure matmul); composes with the shipped :func:`nitrix.linalg.matrix_exp`.
    """
    return _integrate(_local_linearization_step, f, y0, t)


def odeint(
    f: VectorField,
    y0: Float[Array, '...'],
    t: Float[Array, 't'],
    *,
    method: Method = 'rk4',
) -> Float[Array, 't ...']:
    """Integrate ``dy/dt = f(t, y)`` with the chosen ``method``.

    ``method`` is one of ``"rk4"`` (default), ``"midpoint"``, ``"euler"``, or
    ``"local_linearization"`` (the A-stable exponential integrator for stiff
    systems; requires a flat 1-D state).  The adaptive methods (``dopri5`` /
    ``dopri8`` / ``tsit5``) are not part of this set; passing one raises with a
    pointer to that gap.
    """
    if method == 'rk4':
        return rk4(f, y0, t)
    if method == 'midpoint':
        return midpoint(f, y0, t)
    if method == 'euler':
        return euler(f, y0, t)
    if method == 'local_linearization':
        return local_linearization(f, y0, t)
    if method in ('dopri5', 'dopri8', 'tsit5'):
        raise ValueError(
            f'method={method!r} is an adaptive stepper, not part of the '
            "fixed-step set; use 'euler', 'midpoint', 'rk4', or "
            "'local_linearization'. Adaptive integration is roadmapped "
            '(see ode-integrators.md).'
        )
    raise ValueError(
        f"method={method!r}; expected 'euler', 'midpoint', 'rk4', or "
        "'local_linearization'."
    )
