# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Entropic optimal transport by the Sinkhorn algorithm.

Given a cost matrix :math:`C \in \mathbb{R}^{n \times m}` and two probability
vectors :math:`a, b`, entropic optimal transport solves

.. math::

    \min_{P \in U(a, b)} \langle P, C \rangle - \varepsilon\, H(P),
    \qquad U(a,b) = \{P \ge 0 : P\mathbf{1} = a,\; P^{\top}\mathbf{1} = b\},

for the coupling :math:`P` (the transport plan) with entropic regularisation
:math:`\varepsilon`. The optimum has the form
:math:`P_{ij} = \exp\!\big((f_i + g_j - C_{ij})/\varepsilon\big)` and is found by
Sinkhorn's alternating projection on the dual potentials :math:`f, g`.

**Substrate composition.** Each Sinkhorn half-step is a log-domain *softmin*
against the cost -- :math:`\operatorname{logsumexp}_j(g_j/\varepsilon -
C_{ij}/\varepsilon)` -- which is exactly a matrix product in the ``LOG`` semiring
(:math:`\oplus = \operatorname{logsumexp}`, :math:`\otimes = +`). The iteration
therefore *composes* :func:`nitrix.semiring.semiring_matmul` with the ``LOG``
algebra rather than introducing a new kernel: it inherits the semiring's
streaming, hardware-aware reduction, and the whole loop stays in the log domain
(numerically stable for small :math:`\varepsilon`).

References
----------
Cuturi M (2013). Sinkhorn distances: lightspeed computation of optimal transport.
*Advances in Neural Information Processing Systems*, 26, 2292-2300.

Peyré G, Cuturi M (2019). Computational optimal transport. *Foundations and
Trends in Machine Learning*, 11(5-6), 355-607.
https://doi.org/10.1561/2200000073
"""

from __future__ import annotations

from typing import NamedTuple

import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..semiring import LOG, semiring_matmul

__all__ = [
    'SinkhornResult',
    'barycentric_map',
    'sinkhorn',
    'wasserstein_distance',
]


class SinkhornResult(NamedTuple):
    """A solved entropic optimal transport problem.

    Attributes
    ----------
    plan : Float[Array, 'n m']
        The entropic transport plan (coupling) :math:`P`, with row sums
        :math:`\\approx a` and column sums :math:`\\approx b`.
    potential_f : Float[Array, 'n']
        The row dual potential :math:`f`.
    potential_g : Float[Array, 'm']
        The column dual potential :math:`g`.
    """

    plan: Float[Array, 'n m']
    potential_f: Float[Array, 'n']
    potential_g: Float[Array, 'm']


def _softmin(
    neg_cost_scaled: Float[Array, 'p q'],
    potential_scaled: Float[Array, 'q'],
) -> Float[Array, 'p']:
    r"""One log-domain softmin against the cost, as a ``LOG`` semiring matvec.

    Returns :math:`\operatorname{logsumexp}_j(-C_{ij}/\varepsilon +
    h_j/\varepsilon)` -- the entropic Sinkhorn half-step -- by composing
    :func:`~nitrix.semiring.semiring_matmul` with the ``LOG`` algebra (no new
    kernel).
    """
    out = semiring_matmul(
        neg_cost_scaled, potential_scaled[:, None], semiring=LOG
    )
    return out[..., 0]


def sinkhorn(
    cost: Float[Array, 'n m'],
    a: Float[Array, 'n'],
    b: Float[Array, 'm'],
    *,
    epsilon: float = 0.1,
    n_iter: int = 100,
) -> SinkhornResult:
    r"""Entropic optimal transport plan by log-domain Sinkhorn iteration.

    Alternately updates the dual potentials so the coupling
    :math:`P_{ij} = \exp((f_i + g_j - C_{ij})/\varepsilon)` matches the marginals
    :math:`a` and :math:`b`. Runs a fixed number of iterations in the log domain
    (stable for small :math:`\varepsilon`); the softmin against the cost is a
    ``LOG`` semiring matmul.

    Parameters
    ----------
    cost : Float[Array, 'n m']
        The transport cost matrix :math:`C_{ij}` between ``n`` source and ``m``
        target bins.
    a : Float[Array, 'n']
        Source marginal (a probability vector, ``sum a == 1``).
    b : Float[Array, 'm']
        Target marginal (a probability vector, ``sum b == 1``).
    epsilon : float, optional
        Entropic regularisation strength, in the units of ``cost``. Smaller is
        closer to the unregularised (sharp) optimal transport but converges more
        slowly. Default ``0.1``.
    n_iter : int, optional
        Number of Sinkhorn iterations (fixed, for differentiability). Default
        ``100``.

    Returns
    -------
    SinkhornResult
        The transport ``plan`` and the dual potentials.
    """
    n, m = cost.shape
    log_a = jnp.log(a)
    log_b = jnp.log(b)
    neg_cost = -cost / epsilon
    neg_cost_t = neg_cost.T

    def step(
        _: int, fg: tuple[Float[Array, 'n'], Float[Array, 'm']]
    ) -> tuple[Float[Array, 'n'], Float[Array, 'm']]:
        f, g = fg
        f = epsilon * (log_a - _softmin(neg_cost, g / epsilon))
        g = epsilon * (log_b - _softmin(neg_cost_t, f / epsilon))
        return f, g

    f0 = jnp.zeros((n,), cost.dtype)
    g0 = jnp.zeros((m,), cost.dtype)
    f, g = lax.fori_loop(0, n_iter, step, (f0, g0))
    plan = jnp.exp((f[:, None] + g[None, :] - cost) / epsilon)
    return SinkhornResult(plan=plan, potential_f=f, potential_g=g)


def wasserstein_distance(
    cost: Float[Array, 'n m'],
    a: Float[Array, 'n'],
    b: Float[Array, 'm'],
    *,
    epsilon: float = 0.1,
    n_iter: int = 100,
) -> Float[Array, '']:
    r"""Entropic optimal transport cost :math:`\langle P, C \rangle`.

    Solves the entropic OT problem (:func:`sinkhorn`) and returns the transport
    cost :math:`\sum_{ij} P_{ij} C_{ij}` of the resulting plan. As
    :math:`\varepsilon \to 0` this approaches the (unregularised) Wasserstein
    transport cost for the given ground cost.

    Parameters
    ----------
    cost, a, b, epsilon, n_iter
        As in :func:`sinkhorn`.

    Returns
    -------
    Float[Array, '']
        The scalar transport cost of the entropic plan.
    """
    result = sinkhorn(cost, a, b, epsilon=epsilon, n_iter=n_iter)
    return jnp.sum(result.plan * cost)


def barycentric_map(
    plan: Float[Array, 'n m'],
    a: Float[Array, 'n'],
    y: Float[Array, 'm d'],
) -> Float[Array, 'n d']:
    r"""Barycentric projection of a transport plan (the pushforward map).

    Maps each source point to the plan-weighted average of the target points it
    is coupled to, :math:`T(x_i) = \tfrac{1}{a_i} \sum_j P_{ij}\, y_j` -- the
    standard barycentric estimate of the optimal transport map (e.g. for mesh
    correspondence or distribution alignment).

    Parameters
    ----------
    plan : Float[Array, 'n m']
        A transport plan (e.g. :attr:`SinkhornResult.plan`).
    a : Float[Array, 'n']
        The source marginal (the plan's row sums; used to renormalise).
    y : Float[Array, 'm d']
        The target point coordinates.

    Returns
    -------
    Float[Array, 'n d']
        The image of each source point under the barycentric map.
    """
    return (plan @ y) / a[:, None]
