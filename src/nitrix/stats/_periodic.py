# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Reduced-rank periodic Gaussian-process basis (the periodic kernel).

The exact periodic (MacKay) covariance
:math:`k(\tau) = \sigma^2 \exp(-2\sin^2(\pi\tau/T)/\ell^2)` has a *discrete*
spectrum, so -- unlike the squared-exponential / Matern kernels -- it does not
fit the Hilbert-space (Laplace-eigenfunction) reduced-rank construction. It
instead admits an exact Fourier expansion in harmonics of the period: writing
:math:`z = 1/\ell^2` and using the Jacobi--Anger identity
:math:`\exp(z\cos\theta) = I_0(z) + 2\sum_{j\ge 1} I_j(z)\cos(j\theta)`,

.. math::

    k(\tau) = \sigma^2 e^{-z}\Big[ I_0(z) + 2\sum_{j\ge 1} I_j(z)
    \cos\!\big(2\pi j\tau/T\big) \Big],

so the reduced-rank basis is :math:`\{\cos(2\pi j x/T), \sin(2\pi j x/T)\}` with
spectral weights :math:`w_j = 2\sigma^2 I_j(z) e^{-z}` (the mean absorbs the DC
term :math:`I_0`, which stays in the unpenalised fixed block). Exactly as for the
Hilbert-space kernels, the length-scale :math:`\ell` enters *only* as a diagonal
reweighting of a fixed basis, so it slots into the same reduced-rank GP engine.

The weights use the exponentially-scaled modified Bessel functions
:math:`\tilde I_j(z) = I_j(z) e^{-z}`, which JAX does not provide for general
order. They are computed from the continued fraction of the ratios
:math:`r_j = I_j(z)/I_{j-1}(z) = 1/((2j/z) + r_{j+1})`, evaluated downward from a
high order (Miller's algorithm) -- each :math:`r_j \in (0, 1)` so the cumulative
products :math:`\tilde I_j = \tilde I_0 \prod_{k\le j} r_k` decrease monotonically
and never overflow (the naive downward recurrence on :math:`I_j` itself does).
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp
from jax.scipy.special import i0e
from jaxtyping import Array, Float


def _ive(
    z: Float[Array, ''], order: int, buffer: Optional[int] = None
) -> Float[Array, ' order_plus_1']:
    r"""Exponentially-scaled modified Bessel :math:`\tilde I_j(z)`, ``j = 0..order``.

    Overflow-free via the continued fraction of the ratios
    :math:`r_j = \tilde I_j / \tilde I_{j-1}`; the absolute scale is set by
    :math:`\tilde I_0 = ` :func:`~jax.scipy.special.i0e`.
    """
    buf = buffer if buffer is not None else max(25, 2 * order)
    upper = order + buf
    r = jnp.zeros_like(z)
    ratios: dict[int, Array] = {}
    for j in range(upper, 0, -1):
        r = 1.0 / ((2.0 * j / z) + r)
        if j <= order:
            ratios[j] = r
    ive = [i0e(z)]
    for j in range(1, order + 1):
        ive.append(ive[-1] * ratios[j])
    return jnp.stack(ive)


def periodic_features(
    x: Float[Array, ' n'],
    period: float,
    order: int,
) -> Float[Array, 'n two_order']:
    r"""Fourier design of the periodic basis (harmonics ``1..order``).

    Columns are interleaved ``[cos_1, sin_1, cos_2, sin_2, ...]`` with angular
    frequencies :math:`2\pi j / T`; the DC term is omitted (absorbed by the
    model intercept).
    """
    j = jnp.arange(1, order + 1, dtype=x.dtype)
    omega = 2.0 * jnp.pi * j / period
    arg = omega[None, :] * x[:, None]
    stacked = jnp.stack([jnp.cos(arg), jnp.sin(arg)], axis=-1)
    return stacked.reshape(x.shape[0], 2 * order)


def periodic_penalty_diag(
    order: int,
    rho: Float[Array, ''],
    n_fixed: int,
) -> Tuple[Float[Array, ' p'], Float[Array, '']]:
    r"""Diagonal penalty core and log-pseudo-determinant of the periodic basis.

    Mirrors :func:`~nitrix.stats._hsgp._penalty_diag`: returns the penalty
    diagonal :math:`d = [0, \ldots, 0,\, 1/s_1, \ldots]` over the
    :math:`n_\mathrm{fixed} + 2\,\mathrm{order}` columns and
    :math:`\sum_j \log(1/s_j)`, with :math:`s_j = \tilde I_j(1/\rho^2)` for each
    harmonic (shared by its cosine and sine column). The amplitude folds into the
    smoothing parameter, exactly as for the spectral-density kernels.
    """
    z = 1.0 / (rho * rho)
    ive = _ive(z, order)
    s = jnp.repeat(ive[1:], 2)
    inv_s = 1.0 / jnp.clip(s, 1e-30, None)
    d = jnp.concatenate([jnp.zeros((n_fixed,), dtype=inv_s.dtype), inv_s])
    return d, jnp.sum(jnp.log(inv_s))
