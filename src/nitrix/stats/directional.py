# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Directional statistics on the sphere -- the von Mises--Fisher family.

The spherical analogue of :mod:`nitrix.stats.gaussian`: the von Mises--Fisher
(vMF) distribution is the canonical law for unit vectors :math:`x \in S^{p-1}`,
and vMF mixtures are the emission model for surface parcellation.  The
substrate this subpackage was missing -- nitrix has a rich spherical *geometry*
stack but had **zero** spherical statistics.

For :math:`x \in S^{p-1}`, mean direction :math:`\mu \in S^{p-1}` and
concentration :math:`\kappa > 0`,

.. math::

    f(x; \mu, \kappa) = C_p(\kappa)\, e^{\kappa\, \mu^\top x},
    \qquad
    C_p(\kappa) = \frac{\kappa^{p/2 - 1}}{(2\pi)^{p/2}\, I_{p/2-1}(\kappa)},

so the log-normaliser is
:math:`\log C_p(\kappa) = (p/2-1)\log\kappa - (p/2)\log 2\pi - \log I_\nu(\kappa)`
with order :math:`\nu = p/2 - 1` and :math:`I_\nu` the modified Bessel function
of the first kind.

- :func:`log_iv` -- :math:`\log I_\nu(\kappa)`, the normaliser of everything
  else, accurate across the *full* :math:`\kappa` range (not a large-:math:`\kappa`
  asymptotic).
- :func:`vmf_log_prob` -- the vMF log-density score kernel (differentiable in
  :math:`\mu`, :math:`\kappa`), the apply half of the fit/apply seam.

The ``log_iv`` accuracy note (theory over the legacy code)
----------------------------------------------------------

JAX ships only :math:`I_0, I_1` (orders 0 and 1); the general-order
:math:`\log I_\nu` is built here.  The legacy implementation used a single
large-:math:`\kappa` asymptotic term, which is materially wrong at small and
moderate :math:`\kappa` and corrupts the normaliser -> the log-density -> the
MLE.  This implementation is regime-split and validated to :math:`< 4\times
10^{-9}` against a high-precision oracle over :math:`\nu \in [0, \infty)`,
:math:`\kappa \in [10^{-3}, 10^6]`.  Because the order :math:`\nu = p/2 - 1` is
fixed by the sphere dimension, it is a **static** argument, and the choice of
regime is a compile-time branch:

- **small / moderate order** (:math:`\nu < 15`): the ascending series
  :math:`I_\nu(\kappa) = \sum_m (\kappa/2)^{2m+\nu} / (m!\,\Gamma(m+\nu+1))`
  in log-space (log-sum-exp) for :math:`\kappa \le 120`, and the
  large-argument asymptotic series (DLMF 10.40.1) for :math:`\kappa > 120`;
- **large order** (:math:`\nu \ge 15`): the uniform (Debye) asymptotic
  expansion (DLMF 10.41.3), keeping the terms :math:`U_0, \dots, U_5`, which is
  accurate uniformly in :math:`\kappa` when :math:`\nu` is large.

References
----------
- Mardia, K. V. & Jupp, P. E. (2000).  *Directional Statistics.* Wiley.
- DLMF (NIST), chapter 10 (Bessel functions), 10.25.2 / 10.40.1 / 10.41.3.
  https://dlmf.nist.gov/10
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, Tuple, Union, cast

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp
from jaxtyping import Array, Bool, Float

from .._internal.reductions import Reduction, reduce

__all__ = ['log_iv', 'vmf_log_prob', 'vmf_fit', 'vmf_sample', 'VMFFit']

_AxisArg = Union[int, Tuple[int, ...]]

# Regime boundaries (see the module docstring; pinned against an mpmath oracle).
_NU_UNIFORM = 15.0  # order at / above which the uniform asymptotic is used
_KAPPA_SWITCH = 120.0  # argument at which series -> large-argument asymptotic
_N_SERIES = 256  # ascending-series terms (covers kappa <= _KAPPA_SWITCH)
_N_LARGE_ARG = 24  # large-argument asymptotic terms

_LOG_2PI = math.log(2.0 * math.pi)


def _log_iv_series(
    nu: float, kappa: Float[Array, '...']
) -> Float[Array, '...']:
    """Ascending power series (DLMF 10.25.2) in log-space; small/moderate kappa."""
    m = jnp.arange(_N_SERIES, dtype=kappa.dtype)  # (K,)
    log_half_kappa = jnp.log(kappa / 2.0)[..., None]  # (..., 1)
    log_terms = (
        (2.0 * m + nu) * log_half_kappa
        - gammaln(m + 1.0)
        - gammaln(m + nu + 1.0)
    )  # (..., K)
    return logsumexp(log_terms, axis=-1)


def _large_arg_coeffs(nu: float) -> Tuple[float, ...]:
    """Static asymptotic coefficients ``a_k(nu)`` for DLMF 10.40.1.

    ``a_k = prod_{j=1}^{k} (4 nu^2 - (2j-1)^2) / (k! 8^k)``, ``a_0 = 1``; the
    series is ``sum_k (-1)^k a_k / kappa^k``.  ``nu`` is static, so these are
    plain Python floats folded into the trace.
    """
    coeffs = [1.0]
    a = 1.0
    four_nu2 = 4.0 * nu * nu
    for k in range(1, _N_LARGE_ARG):
        a = a * (four_nu2 - (2 * k - 1) ** 2) / (k * 8.0)
        coeffs.append(a)
    return tuple(coeffs)


def _log_iv_large_arg(
    nu: float, kappa: Float[Array, '...']
) -> Float[Array, '...']:
    """Large-argument asymptotic (DLMF 10.40.1); large kappa, small/moderate nu."""
    inv = 1.0 / kappa
    coeffs = _large_arg_coeffs(nu)
    series = jnp.ones_like(kappa) * coeffs[0]
    inv_pow = jnp.ones_like(kappa)
    for k in range(1, _N_LARGE_ARG):
        inv_pow = inv_pow * inv
        series = series + ((-1.0) ** k) * coeffs[k] * inv_pow
    return kappa - 0.5 * jnp.log(2.0 * jnp.pi * kappa) + jnp.log(series)


# Debye polynomials U_0..U_5 (DLMF 10.41.10): U_k(t) = sum coeff * t^power.
_DEBYE_U: Tuple[Tuple[Tuple[float, int], ...], ...] = (
    ((1.0, 0),),
    ((3.0 / 24.0, 1), (-5.0 / 24.0, 3)),
    ((81.0 / 1152.0, 2), (-462.0 / 1152.0, 4), (385.0 / 1152.0, 6)),
    (
        (30375.0 / 414720.0, 3),
        (-369603.0 / 414720.0, 5),
        (765765.0 / 414720.0, 7),
        (-425425.0 / 414720.0, 9),
    ),
    (
        (4465125.0 / 39813120.0, 4),
        (-94121676.0 / 39813120.0, 6),
        (349922430.0 / 39813120.0, 8),
        (-446185740.0 / 39813120.0, 10),
        (185910725.0 / 39813120.0, 12),
    ),
    (
        (1519035525.0 / 6688604160.0, 5),
        (-49286948607.0 / 6688604160.0, 7),
        (284499769554.0 / 6688604160.0, 9),
        (-614135872350.0 / 6688604160.0, 11),
        (566098157625.0 / 6688604160.0, 13),
        (-188699385875.0 / 6688604160.0, 15),
    ),
)


def _log_iv_uniform(
    nu: float, kappa: Float[Array, '...']
) -> Float[Array, '...']:
    """Uniform (Debye) asymptotic expansion (DLMF 10.41.3); large order nu."""
    u = kappa / nu
    root = jnp.sqrt(1.0 + u * u)
    t = 1.0 / root
    eta = root + jnp.log(u / (1.0 + root))
    series = jnp.zeros_like(kappa)
    for k, poly in enumerate(_DEBYE_U):
        u_k = sum(coeff * t**power for coeff, power in poly)
        series = series + u_k / (nu**k)
    return (
        nu * eta
        - 0.5 * jnp.log(2.0 * jnp.pi * nu)
        - 0.25 * jnp.log(1.0 + u * u)
        + jnp.log(series)
    )


def log_iv(nu: float, kappa: Float[Array, '...']) -> Float[Array, '...']:
    r"""Logarithm of the modified Bessel function of the first kind.

    Computes :math:`\log I_\nu(\kappa)` -- the log-normaliser of the von
    Mises--Fisher (and related directional) densities -- accurately across the
    *full* range of :math:`\kappa`, not merely the large-:math:`\kappa`
    asymptotic regime.  Differentiable in :math:`\kappa`.

    The order :math:`\nu` is a **static** (non-traced) scalar -- it is fixed by
    the sphere dimension (:math:`\nu = p/2 - 1`) -- and selects the evaluation
    regime at trace time (see the module docstring): the ascending series /
    large-argument asymptotic for :math:`\nu < 15`, and the uniform (Debye)
    asymptotic for :math:`\nu \ge 15`.

    Parameters
    ----------
    nu : float
        The order :math:`\nu \ge 0` of the Bessel function; a static scalar.
    kappa : Float[Array, '...']
        The argument :math:`\kappa > 0`, of arbitrary shape.  For :math:`\nu >
        0`, :math:`\log I_\nu(\kappa) \to -\infty` as :math:`\kappa \to 0^+`.

    Returns
    -------
    Float[Array, '...']
        :math:`\log I_\nu(\kappa)`, the same shape as ``kappa``.
    """
    if nu >= _NU_UNIFORM:
        return _log_iv_uniform(nu, kappa)
    # Split on kappa; feed each branch inputs kept inside its valid region so
    # the unused branch cannot contaminate the gradient with a NaN (the
    # large-argument series diverges for small kappa).
    kappa_small = jnp.minimum(kappa, _KAPPA_SWITCH)
    kappa_large = jnp.maximum(kappa, _KAPPA_SWITCH)
    return jnp.where(
        kappa <= _KAPPA_SWITCH,
        _log_iv_series(nu, kappa_small),
        _log_iv_large_arg(nu, kappa_large),
    )


def vmf_log_prob(
    x: Float[Array, '... p'],
    mu: Float[Array, '... p'],
    kappa: Float[Array, '...'],
    *,
    axis: Optional[_AxisArg] = None,
    reduction: Reduction = 'none',
) -> Float[Array, '...']:
    r"""Von Mises--Fisher log-density of unit vectors.

    Evaluates :math:`\log f(x; \mu, \kappa) = \kappa\, \mu^\top x + \log
    C_p(\kappa)` for :math:`x \in S^{p-1}`, with :math:`p` taken from the
    trailing axis of ``x`` and order :math:`\nu = p/2 - 1`.  A distributional
    score kernel (the spherical sibling of :func:`~nitrix.stats.gaussian_nll`),
    differentiable in ``mu`` and ``kappa``, and the apply half of the vMF
    fit/apply seam.

    ``x`` and ``mu`` are assumed unit-norm along the trailing axis; this is not
    enforced (normalise beforehand if in doubt).

    Parameters
    ----------
    x : Float[Array, '... p']
        Observations on :math:`S^{p-1}` (unit vectors), of arbitrary batch
        shape.
    mu : Float[Array, '... p']
        Mean direction :math:`\mu \in S^{p-1}`, broadcastable to ``x``.
    kappa : Float[Array, '...']
        Concentration :math:`\kappa > 0`, broadcastable to ``x``'s batch shape
        (``x`` without its trailing axis).
    axis : int or tuple of int, optional
        Axes to reduce over. Default ``None`` reduces over all axes (only
        relevant when ``reduction != "none"``).
    reduction : {'none', 'sum', 'mean'}, optional
        How to reduce the per-observation log-density. ``"none"`` (the default,
        per the score-kernel convention) returns the unreduced tensor;
        ``"sum"`` gives the log-likelihood; ``"mean"`` its average.

    Returns
    -------
    Float[Array, '...']
        The per-observation vMF log-density (the broadcast batch shape of ``x``,
        ``mu`` and ``kappa``) when ``reduction="none"``, otherwise its reduction
        over ``axis``.
    """
    p = x.shape[-1]
    nu = p / 2.0 - 1.0
    dot = jnp.sum(mu * x, axis=-1)  # mu^T x, per observation
    log_norm = nu * jnp.log(kappa) - (p / 2.0) * _LOG_2PI - log_iv(nu, kappa)
    per_obs = kappa * dot + log_norm
    return reduce(per_obs, axis=axis, reduction=reduction)


class VMFFit(NamedTuple):
    """A fitted von Mises--Fisher distribution: the MLE state.

    State only (plain arrays; the fit/apply seam), consumed by
    :func:`vmf_log_prob`.

    Attributes
    ----------
    mu
        The maximum-likelihood mean direction :math:`\\hat\\mu \\in S^{p-1}`
        (the normalised resultant), of shape ``(..., p)``.
    kappa
        The maximum-likelihood concentration :math:`\\hat\\kappa`, of shape
        ``(...)``.
    """

    mu: Float[Array, '... p']
    kappa: Float[Array, '...']


def _a_ratio(p: int, kappa: Float[Array, '...']) -> Float[Array, '...']:
    """The mean-resultant function :math:`A_p(\\kappa) = I_{p/2}/I_{p/2-1}`."""
    return jnp.exp(log_iv(p / 2.0, kappa) - log_iv(p / 2.0 - 1.0, kappa))


def vmf_fit(
    x: Float[Array, '... n p'],
    *,
    weights: Optional[Float[Array, '... n']] = None,
    axis: int = -2,
    n_newton: int = 10,
) -> VMFFit:
    r"""Maximum-likelihood fit of a von Mises--Fisher distribution.

    The *fit* half of the fit/apply seam (:func:`vmf_log_prob` is the apply).
    The mean direction is the normalised (weighted) resultant
    :math:`\hat\mu = \sum_i w_i x_i / \lVert \sum_i w_i x_i \rVert`; with mean
    resultant length :math:`\bar R = \lVert \sum_i w_i x_i \rVert / \sum_i w_i`,
    the concentration :math:`\hat\kappa` solves :math:`A_p(\kappa) = \bar R`,
    where :math:`A_p(\kappa) = I_{p/2}(\kappa) / I_{p/2-1}(\kappa)`.

    The root is found by Newton's method on :math:`A_p(\kappa) - \bar R`, using
    the exact derivative :math:`A_p'(\kappa) = 1 - A_p(\kappa)^2 -
    \tfrac{p-1}{\kappa} A_p(\kappa)`, warm-started from the Banerjee et al.
    (2005) closed form :math:`\hat\kappa_0 = \bar R (p - \bar R^2) / (1 - \bar
    R^2)` -- so the returned :math:`\hat\kappa` matches the exact
    :math:`A_p(\kappa) = \bar R` root, not merely that approximation.

    Parameters
    ----------
    x : Float[Array, '... n p']
        Observations on :math:`S^{p-1}` (unit vectors); ``n`` along ``axis``.
    weights : Float[Array, '... n'], optional
        Per-observation non-negative weights (e.g. soft cluster
        responsibilities). ``None`` (default) weights all observations equally.
    axis : int, optional
        The observation axis to reduce over. Default ``-2`` (features last).
    n_newton : int, optional
        Number of Newton refinement steps on the exact :math:`A_p` root.
        Default 10 (converges to machine precision well within this).

    Returns
    -------
    VMFFit
        The fitted ``(mu, kappa)``.
    """
    p = x.shape[-1]
    if weights is None:
        resultant = jnp.sum(x, axis=axis)  # (..., p)
        n_eff: Float[Array, '...'] = jnp.asarray(
            float(x.shape[axis]), dtype=x.dtype
        )
    else:
        resultant = jnp.sum(weights[..., None] * x, axis=axis)
        # weights lack x's trailing feature axis, so their observation axis is
        # one position later than x's when counted from the end.
        w_axis = axis + 1 if axis < 0 else axis
        n_eff = jnp.sum(weights, axis=w_axis)
    r_norm = jnp.linalg.norm(resultant, axis=-1)  # (...,)
    mu = resultant / r_norm[..., None]
    r_bar = r_norm / n_eff  # mean resultant length in [0, 1)

    tiny = jnp.finfo(x.dtype).tiny
    # Banerjee warm start, then Newton on the exact A_p(kappa) = r_bar root.
    kappa = r_bar * (p - r_bar**2) / jnp.maximum(1.0 - r_bar**2, tiny)
    for _ in range(n_newton):
        a = _a_ratio(p, kappa)
        deriv = 1.0 - a * a - ((p - 1.0) / kappa) * a
        kappa = jnp.maximum(kappa - (a - r_bar) / deriv, tiny)
    return VMFFit(mu=mu, kappa=kappa)


def _sample_w(
    key: jax.Array, kappa: Float[Array, ''], p: int, n: int
) -> Float[Array, ' n']:
    """Sample the mu-component ``w`` (density :math:`\\propto (1-w^2)^{(p-3)/2}
    e^{\\kappa w}`) via Wood (1994) rejection, with guaranteed acceptance.

    A :func:`jax.lax.while_loop` resamples the not-yet-accepted draws until all
    are accepted -- the rejection envelope has bounded efficiency, so this
    terminates almost surely.  There is no fixed iteration cap and no
    possibility of returning an unaccepted (invalid) draw.
    """
    dim = p - 1.0
    b = (-2.0 * kappa + jnp.sqrt(4.0 * kappa**2 + dim**2)) / dim
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + dim * jnp.log(1.0 - x0**2)

    def cond(state: Tuple[jax.Array, Array, Array]) -> Array:
        _, _, accepted = state
        return jnp.logical_not(jnp.all(accepted))

    def body(
        state: Tuple[jax.Array, Array, Array],
    ) -> Tuple[jax.Array, Array, Array]:
        key, w, accepted = state
        key, key_beta, key_unif = jax.random.split(key, 3)
        z = jax.random.beta(key_beta, dim / 2.0, dim / 2.0, shape=(n,))
        w_cand = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        log_u = jnp.log(jax.random.uniform(key_unif, shape=(n,)))
        test = kappa * w_cand + dim * jnp.log(1.0 - x0 * w_cand) - c >= log_u
        newly = test & jnp.logical_not(accepted)
        w = jnp.where(newly, w_cand, w)
        return key, w, accepted | test

    w0 = jnp.zeros((n,), dtype=kappa.dtype)
    accepted0: Bool[Array, ' n'] = jnp.zeros((n,), dtype=bool)
    _, w, _ = jax.lax.while_loop(cond, body, (key, w0, accepted0))
    return w


def vmf_sample(
    key: jax.Array,
    mu: Float[Array, ' p'],
    kappa: Float[Array, ''],
    shape: Tuple[int, ...] = (),
) -> Float[Array, '... p']:
    r"""Draw samples from a von Mises--Fisher distribution (Wood 1994).

    Each draw decomposes as :math:`x = w\,\mu + \sqrt{1 - w^2}\, v`, where the
    :math:`\mu`-component :math:`w` has density :math:`\propto (1 -
    w^2)^{(p-3)/2} e^{\kappa w}` (sampled by Wood's rejection scheme with
    **guaranteed acceptance** -- a :func:`jax.lax.while_loop`, never a fixed cap
    that could return an invalid sample) and the tangent direction :math:`v` is
    uniform on the sub-sphere :math:`S^{p-2}` orthogonal to :math:`\mu`.

    Sampling is **not differentiable** (the rejection loop and the discrete
    accept step have no reparameterisation here); use it for simulation, not for
    pathwise gradients.

    Parameters
    ----------
    key : jax.Array
        A PRNG key.
    mu : Float[Array, 'p']
        The mean direction :math:`\mu \in S^{p-1}` (assumed unit-norm).
    kappa : Float[Array, '']
        The concentration :math:`\kappa > 0` (a scalar).
    shape : tuple of int, optional
        The batch shape of draws. Default ``()`` (a single sample).

    Returns
    -------
    Float[Array, '... p']
        Samples of shape ``(*shape, p)``, each a unit vector on
        :math:`S^{p-1}`.
    """
    p = mu.shape[-1]
    n = math.prod(shape) if shape else 1
    kappa = jnp.asarray(kappa, dtype=mu.dtype)
    key_w, key_v = jax.random.split(key)

    w = _sample_w(key_w, kappa, p, n)  # (n,)
    # Tangent component: uniform on S^{p-2} (project a Gaussian onto mu^perp).
    v = jax.random.normal(key_v, (n, p), dtype=mu.dtype)
    v = v - jnp.sum(v * mu, axis=-1, keepdims=True) * mu
    v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)

    x = w[:, None] * mu + jnp.sqrt(1.0 - w**2)[:, None] * v
    return cast(Float[Array, '... p'], x.reshape((*shape, p)))
