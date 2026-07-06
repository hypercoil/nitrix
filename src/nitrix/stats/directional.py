# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Directional statistics on the sphere -- the von Mises--Fisher and Watson families.

The spherical analogue of :mod:`nitrix.stats.gaussian`: the von Mises--Fisher
(vMF) distribution is the canonical law for unit vectors :math:`x \in S^{p-1}`,
and vMF mixtures are the emission model for surface parcellation.  The
substrate this subpackage was missing -- nitrix has a rich spherical *geometry*
stack but had **zero** spherical statistics.

The **Watson** distribution is the *axial* sibling (:math:`x` and :math:`-x`
identified): rotationally symmetric about an axis :math:`\mu` with concentration
:math:`\kappa` (bipolar for :math:`\kappa > 0`, girdle for :math:`\kappa < 0`),
density :math:`\propto \exp(\kappa (\mu^\top x)^2)`, normalised by Kummer's
confluent hypergeometric :math:`M(\tfrac12, p/2, \kappa)` -- the law for
undirected orientations / axial data (:func:`log_kummer_m`,
:func:`watson_log_prob`, :func:`watson_fit`).

The **Kent** (Fisher--Bingham FB5) distribution is the *elliptical* vMF on
:math:`S^2`: density :math:`\propto \exp(\kappa\, \gamma_1^\top x + \beta
[(\gamma_2^\top x)^2 - (\gamma_3^\top x)^2])` with an orthonormal frame and
ovalness :math:`\beta` (:math:`\beta = 0` recovers vMF); its normaliser is a
series in half-integer-order Bessel functions, reusing :func:`log_iv`
(:func:`log_kent_normaliser`, :func:`kent_log_prob`, :func:`kent_fit`).

Each ``*_log_prob`` takes ``normalize=False`` to return the **unnormalised**
log-density (energy) -- the normaliser is dropped *and never computed*, so it is
tractable in high dimensions and composes additively as a per-site potential in
a Gibbs / Markov-random-field energy (the parcellation setting).
:func:`fisher_bingham_energy` is the *general* such energy, :math:`\kappa
\gamma_1^\top x + \sum_j \beta_j (\gamma_j^\top x)^2` on any :math:`S^{p-1}`
(orthonormal frame + coefficient vector) -- the quadratic-exponential family that
**subsumes** vMF (:math:`\beta = 0`), Watson (rank-one quadratic), Bingham
(:math:`\kappa = 0`), and Kent (:math:`p = 3`); its normaliser is intractable in
general, so only the energy is provided.

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
from ..linalg._solver import safe_eigh

__all__ = [
    'log_iv',
    'vmf_log_prob',
    'vmf_fit',
    'vmf_sample',
    'VMFFit',
    'log_kummer_m',
    'watson_log_prob',
    'watson_fit',
    'WatsonFit',
    'log_kent_normaliser',
    'kent_log_prob',
    'kent_fit',
    'KentFit',
    'fisher_bingham_energy',
]

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
    normalize: bool = True,
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
    normalize : bool, optional
        If ``True`` (default), the full log-density.  If ``False``, the
        **unnormalised** log-density :math:`\kappa\, \mu^\top x` -- the
        normaliser term is dropped *and never computed* (no :func:`log_iv`).
        Its negative is the Gibbs energy (:math:`p \propto e^{-E}`): a
        per-site clique **potential** that composes additively into a
        Gibbs/Markov-random-field energy (the parcellation setting) and is
        tractable in high dimensions where the normaliser is expensive.

    Returns
    -------
    Float[Array, '...']
        The per-observation vMF log-density (the broadcast batch shape of ``x``,
        ``mu`` and ``kappa``) when ``reduction="none"``, otherwise its reduction
        over ``axis``.
    """
    p = x.shape[-1]
    dot = jnp.sum(mu * x, axis=-1)  # mu^T x, per observation
    per_obs = kappa * dot
    if normalize:
        nu = p / 2.0 - 1.0
        log_norm = (
            nu * jnp.log(kappa) - (p / 2.0) * _LOG_2PI - log_iv(nu, kappa)
        )
        per_obs = per_obs + log_norm
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


# ===========================================================================
# Watson distribution -- the axial (antipodally symmetric) sibling of vMF.
#
# Density on S^{p-1}: f(x; mu, kappa) = M(1/2, p/2, kappa)^{-1} exp(kappa
# (mu^T x)^2), with axis mu (mu == -mu) and concentration kappa (kappa > 0
# bipolar / clustered at +-mu; kappa < 0 girdle / equatorial).  The normaliser
# is the Kummer confluent hypergeometric M(1/2, p/2, kappa) = 1F1(1/2; p/2;
# kappa).  Regime-split + validated to < 3e-12 vs mpmath over p in [2, 50],
# |kappa| in [1e-3, 5e3] (both bipolar and girdle).
# ===========================================================================

_KUMMER_SWITCH = 100.0  # |z| at which series -> large-argument asymptotic
_N_KUMMER_SERIES = 300  # ascending-series terms (covers |z| <= switch)
_N_KUMMER_ASYM = 24  # large-argument asymptotic terms


def _kummer_series_logcoeffs(a: float, b: float, n: int) -> Tuple[float, ...]:
    """Static ``log[(a)_k / ((b)_k k!)]`` for the ascending series of M(a,b,z)."""
    return tuple(
        (math.lgamma(a + k) - math.lgamma(a))
        - (math.lgamma(b + k) - math.lgamma(b))
        - math.lgamma(k + 1)
        for k in range(n)
    )


def _kummer_asym_coeffs(a: float, b: float, n: int) -> Tuple[float, ...]:
    """Static (signed) ``(b-a)_k (1-a)_k / k!`` for the large-argument series."""
    out = []
    for k in range(n):
        factors = [b - a + i for i in range(k)] + [
            1.0 - a + i for i in range(k)
        ]
        if any(f == 0.0 for f in factors):
            out.append(0.0)
            continue
        log_mag = sum(math.log(abs(f)) for f in factors) - math.lgamma(k + 1)
        sign = 1.0
        for f in factors:
            sign *= 1.0 if f > 0.0 else -1.0
        out.append(sign * math.exp(log_mag))
    return tuple(out)


def _log_m_pos(
    a: float, b: float, y: Float[Array, '...']
) -> Float[Array, '...']:
    """``log M(a, b, y)`` for ``y >= 0`` (static ``a``, ``b``); regime-split.

    Ascending series (DLMF 13.2.2) for ``y <= _KUMMER_SWITCH``, else the
    large-argument asymptotic (DLMF 13.7.1).  Each branch is fed a clamped
    argument so the unused branch cannot inject a ``NaN`` (the ``log_iv``
    pattern).
    """
    dtype = y.dtype
    cn = jnp.asarray(
        _kummer_series_logcoeffs(a, b, _N_KUMMER_SERIES), dtype=dtype
    )
    dk = jnp.asarray(_kummer_asym_coeffs(a, b, _N_KUMMER_ASYM), dtype=dtype)
    n = jnp.arange(_N_KUMMER_SERIES, dtype=dtype)
    k = jnp.arange(_N_KUMMER_ASYM, dtype=dtype)
    eps = jnp.finfo(dtype).eps

    # Additive (not clamping) floor: keeps ``log`` finite at ``y = 0`` *and*
    # preserves the gradient there -- the ``1/(y+eps)`` blow-up is exactly
    # cancelled by the vanishing softmax weight of the ``n >= 1`` terms, so
    # ``d/dy log M|_0 = a/b`` (whereas ``maximum(y, tiny)`` would zero it, and a
    # denormal-edge ``tiny`` floor would underflow the cancellation).
    y_small = jnp.where(y <= _KUMMER_SWITCH, y + eps, 1.0)
    log_series = logsumexp(cn + n * jnp.log(y_small)[..., None], axis=-1)

    y_large = jnp.where(y > _KUMMER_SWITCH, y, _KUMMER_SWITCH + 1.0)
    asym_sum = jnp.sum(dk * (1.0 / y_large)[..., None] ** k, axis=-1)
    log_asym = (
        y_large
        + (a - b) * jnp.log(y_large)
        + (math.lgamma(b) - math.lgamma(a))
        + jnp.log(asym_sum)
    )
    return jnp.where(y <= _KUMMER_SWITCH, log_series, log_asym)


def log_kummer_m(
    a: float, b: float, z: Float[Array, '...']
) -> Float[Array, '...']:
    r"""Log of Kummer's confluent hypergeometric function :math:`\log M(a, b, z)`.

    :math:`M(a, b, z) = {}_1F_1(a; b; z) = \sum_{k \ge 0} \frac{(a)_k}{(b)_k}
    \frac{z^k}{k!}`, the normaliser of the Watson family (at :math:`a =
    \tfrac12`, :math:`b = p/2`).  Static ``a``, ``b`` select the regime at trace
    time; the argument ``z`` is an array of either sign.  For ``z < 0`` Kummer's
    transformation :math:`M(a, b, z) = e^{z} M(b - a, b, -z)` is used, so the
    all-positive-term series/asymptotic always run on a non-negative argument
    (the girdle branch).  Differentiable in ``z``.

    Parameters
    ----------
    a, b : float
        Static parameters (``b > 0``); ``a = 1/2``, ``b = p/2`` for Watson.
    z : Float[Array, '...']
        Argument (the Watson concentration :math:`\kappa`), any sign.

    Returns
    -------
    Float[Array, '...']
        :math:`\log M(a, b, z)`, elementwise.
    """
    # Branch-safe arguments (the double-``where`` trick): the bipolar branch
    # sees ``z >= 0`` and the girdle branch ``-z > 0``, with the *unused* branch
    # fed ``0`` -- so neither a ``NaN`` nor the ``|z|`` chain-rule (which would
    # zero the gradient at ``z = 0``) can leak through :func:`jax.grad`.
    z_pos = jnp.where(z >= 0.0, z, 0.0)
    z_neg = jnp.where(z < 0.0, -z, 0.0)
    bipolar = _log_m_pos(a, b, z_pos)
    girdle = z + _log_m_pos(b - a, b, z_neg)
    return jnp.where(z >= 0.0, bipolar, girdle)


def watson_log_prob(
    x: Float[Array, '... p'],
    mu: Float[Array, '... p'],
    kappa: Float[Array, '...'],
    *,
    axis: Optional[_AxisArg] = None,
    reduction: Reduction = 'none',
    normalize: bool = True,
) -> Float[Array, '...']:
    r"""Watson log-density of axial unit vectors.

    :math:`\log f(x; \mu, \kappa) = \kappa (\mu^\top x)^2 - \log M(\tfrac12,
    p/2, \kappa)` for :math:`x \in S^{p-1}`, with :math:`p` from the trailing
    axis.  Antipodally symmetric (:math:`x` and :math:`-x`, :math:`\mu` and
    :math:`-\mu`, give the same density): the axial score kernel.  ``kappa > 0``
    is bipolar (mass at :math:`\pm\mu`), ``kappa < 0`` girdle (mass on the
    equator :math:`\perp \mu`).  Differentiable in ``mu`` and ``kappa``; the
    apply half of the Watson fit/apply seam.

    ``x`` and ``mu`` are assumed unit-norm along the trailing axis.

    Parameters
    ----------
    x : Float[Array, '... p']
        Observations on :math:`S^{p-1}` (unit vectors, sign-agnostic).
    mu : Float[Array, '... p']
        Axis :math:`\mu \in S^{p-1}` (sign-agnostic), broadcastable to ``x``.
    kappa : Float[Array, '...']
        Concentration (real; sign selects bipolar/girdle), broadcastable to
        ``x``'s batch shape.
    axis : int or tuple of int, optional
        Axes to reduce over when ``reduction != "none"``.
    reduction : {'none', 'sum', 'mean'}, optional
        Reduction of the per-observation log-density (default ``"none"``).
    normalize : bool, optional
        If ``False``, the **unnormalised** log-density :math:`\kappa (\mu^\top
        x)^2` -- the Kummer normaliser is dropped *and never computed* (no
        :func:`log_kummer_m`).  The axial per-site potential for a Gibbs/Markov-
        random-field energy, tractable at any :math:`p`.  Default ``True``.

    Returns
    -------
    Float[Array, '...']
        The per-observation Watson log-density (``reduction="none"``) or its
        reduction.
    """
    p = x.shape[-1]
    dot = jnp.sum(mu * x, axis=-1)
    per_obs = kappa * dot**2
    if normalize:
        # Surface-measure normalisation (as for :func:`vmf_log_prob`): divide by
        # the area of S^{p-1}, omega = 2 pi^{p/2} / Gamma(p/2), so the density
        # integrates to one against the sphere's Lebesgue measure.
        log_area = (
            math.log(2.0)
            + (p / 2.0) * math.log(math.pi)
            - math.lgamma(p / 2.0)
        )
        per_obs = per_obs - log_kummer_m(0.5, p / 2.0, kappa) - log_area
    return reduce(per_obs, axis=axis, reduction=reduction)


class WatsonFit(NamedTuple):
    """A fitted Watson distribution: the MLE state.

    State only (plain arrays; the fit/apply seam), consumed by
    :func:`watson_log_prob`.

    Attributes
    ----------
    mu
        The maximum-likelihood axis :math:`\\hat\\mu \\in S^{p-1}` (an
        eigenvector of the scatter matrix; sign-agnostic), shape ``(..., p)``.
    kappa
        The maximum-likelihood concentration :math:`\\hat\\kappa` (positive
        bipolar / negative girdle), shape ``(...)``.
    """

    mu: Float[Array, '... p']
    kappa: Float[Array, '...']


def _watson_g(b: float, kappa: Float[Array, '...']) -> Float[Array, '...']:
    """:math:`g(\\kappa) = \\partial_\\kappa \\log M(\\tfrac12, b, \\kappa) =
    \\mathbb{E}[(\\mu^\\top x)^2]`, elementwise (via autodiff of the normaliser).
    """
    g = jax.grad(lambda k: log_kummer_m(0.5, b, k).sum())(kappa)
    return cast(Float[Array, '...'], g)


def _solve_watson_kappa(
    b: float,
    r: Float[Array, '...'],
    n_bisect: int,
    k_max: float = 1.0e4,
) -> Float[Array, '...']:
    """Solve :math:`g(\\kappa) = r` for :math:`\\kappa` by bisection.

    :math:`g` increases monotonically from ``0`` (:math:`\\kappa \\to -\\infty`)
    through ``1/p`` (:math:`\\kappa = 0`) to ``1`` (:math:`\\kappa \\to
    +\\infty`), so the bracket ``[-k_max, k_max]`` contains the unique root for
    any ``r`` in ``(0, 1)`` (bipolar ``r > 1/p`` gives ``kappa > 0``, girdle
    ``r < 1/p`` gives ``kappa < 0``).
    """
    lo = jnp.full_like(r, -k_max)
    hi = jnp.full_like(r, k_max)

    def body(_: Array, state: Tuple[Array, Array]) -> Tuple[Array, Array]:
        lo, hi = state
        mid = 0.5 * (lo + hi)
        below = _watson_g(b, mid) < r
        return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

    lo, hi = jax.lax.fori_loop(0, n_bisect, body, (lo, hi))
    return cast(Float[Array, '...'], 0.5 * (lo + hi))


def watson_fit(
    x: Float[Array, '... n p'],
    *,
    weights: Optional[Float[Array, '... n']] = None,
    axis: int = -2,
    n_bisect: int = 60,
) -> WatsonFit:
    r"""Maximum-likelihood fit of a Watson distribution.

    The *fit* half of the seam (:func:`watson_log_prob` is the apply).  Forms
    the (weighted) scatter matrix :math:`S = \sum_i w_i x_i x_i^\top / \sum_i
    w_i` and eigendecomposes it.  The MLE axis is an eigenvector of ``S`` and
    the concentration solves :math:`g(\hat\kappa) = \hat\mu^\top S \hat\mu`,
    where :math:`g(\kappa) = \partial_\kappa \log M(\tfrac12, p/2, \kappa)`.
    Both candidates are evaluated -- **bipolar** (leading eigenvector,
    :math:`\bar\lambda = \lambda_{\max} \ge 1/p`, :math:`\kappa \ge 0`) and
    **girdle** (trailing eigenvector, :math:`\lambda_{\min} \le 1/p`,
    :math:`\kappa \le 0`) -- and the higher-likelihood one is returned, so both
    axial regimes are recovered.  The root is found by bisection on the
    monotone :math:`g`.

    Because it eigendecomposes the scatter, the fit shares the ``safe_eigh``
    contract (jit-clean on healthy backends; eager CPU fallback on the
    cuSolver-affected GPU stacks, like :func:`nitrix.stats.pca_fit`).

    Parameters
    ----------
    x : Float[Array, '... n p']
        Axial observations on :math:`S^{p-1}` (unit vectors, sign-agnostic);
        ``n`` along ``axis``.
    weights : Float[Array, '... n'], optional
        Per-observation non-negative weights (e.g. soft responsibilities).
        ``None`` (default) weights observations equally.
    axis : int, optional
        The observation axis. Default ``-2`` (features last).
    n_bisect : int, optional
        Bisection steps for the concentration root (default 60 -> machine
        precision over the ``[-1e4, 1e4]`` bracket).

    Returns
    -------
    WatsonFit
        The fitted ``(mu, kappa)``.
    """
    p = x.shape[-1]
    b = p / 2.0
    xm = jnp.moveaxis(x, axis, -2)  # (..., n, p)
    if weights is None:
        scatter = jnp.einsum('...np,...nq->...pq', xm, xm) / xm.shape[-2]
    else:
        w_axis = axis if axis >= 0 else axis + 1
        wm = jnp.moveaxis(weights, w_axis, -1)  # (..., n)
        wx = wm[..., :, None] * xm
        scatter = (
            jnp.einsum('...np,...nq->...pq', wx, xm)
            / jnp.sum(wm, axis=-1)[..., None, None]
        )
    scatter = 0.5 * (scatter + jnp.swapaxes(scatter, -1, -2))

    evals, evecs = safe_eigh(scatter)  # ascending eigenvalues
    lam_max = evals[..., -1]
    lam_min = evals[..., 0]
    v_max = evecs[..., :, -1]
    v_min = evecs[..., :, 0]

    kappa_bip = _solve_watson_kappa(b, lam_max, n_bisect)
    kappa_gir = _solve_watson_kappa(b, lam_min, n_bisect)
    ll_bip = kappa_bip * lam_max - log_kummer_m(0.5, b, kappa_bip)
    ll_gir = kappa_gir * lam_min - log_kummer_m(0.5, b, kappa_gir)
    use_bip = ll_bip >= ll_gir

    mu = jnp.where(use_bip[..., None], v_max, v_min)
    kappa = jnp.where(use_bip, kappa_bip, kappa_gir)
    return WatsonFit(mu=mu, kappa=kappa)


# ===========================================================================
# Kent (Fisher--Bingham FB5) distribution -- the elliptical vMF on S^2.
#
# Density on S^2: f(x) = c(kappa, beta)^{-1} exp(kappa g1^T x + beta [(g2^T x)^2
# - (g3^T x)^2]), with an orthonormal frame (g1 mean, g2 major axis, g3 minor
# axis), concentration kappa >= 0 and ovalness 0 <= 2 beta < kappa (beta = 0
# recovers vMF).  The normaliser is a series in half-integer-order Bessel
# functions (reuses `log_iv`); validated < 6e-14 vs mpmath and reduces to the
# vMF C_3 normaliser at beta = 0.
# ===========================================================================

_KENT_TERMS = 64  # normaliser-series terms (machine precision to kappa ~ 1e3)


def log_kent_normaliser(
    kappa: Float[Array, '...'],
    beta: Float[Array, '...'],
    *,
    n_terms: int = _KENT_TERMS,
) -> Float[Array, '...']:
    r"""Log of the Kent (FB5) normalising constant :math:`\log c(\kappa, \beta)`.

    :math:`c(\kappa, \beta) = 2\pi \sum_{j \ge 0} \frac{\Gamma(j + \tfrac12)}{
    \Gamma(j + 1)} \beta^{2j} (2/\kappa)^{2j + 1/2} I_{2j + 1/2}(\kappa)`, the
    surface-measure normaliser (so :math:`c(\kappa, 0) = 1/C_3(\kappa)`, the vMF
    constant).  The half-integer Bessel terms reuse :func:`log_iv`; the sum is
    taken in log-space.  Differentiable in ``kappa`` and ``beta``; requires
    ``kappa > 0`` and ``0 <= 2 beta < kappa``.

    Parameters
    ----------
    kappa, beta : Float[Array, '...']
        Concentration and ovalness (broadcastable).
    n_terms : int, optional
        Series terms (default 64; machine precision to ``kappa`` ~ 1e3).

    Returns
    -------
    Float[Array, '...']
        :math:`\log c(\kappa, \beta)`.
    """
    kappa = jnp.asarray(kappa)
    beta = jnp.asarray(beta)
    dtype = jnp.result_type(kappa.dtype, beta.dtype, jnp.float32)
    kappa = kappa.astype(dtype)
    beta = beta.astype(dtype)
    eps = jnp.finfo(dtype).eps
    j = jnp.arange(n_terms, dtype=dtype)
    lg = jnp.asarray(
        [math.lgamma(k + 0.5) - math.lgamma(k + 1) for k in range(n_terms)],
        dtype=dtype,
    )
    # log I_{2j+1/2}(kappa) for each static order; stacked on a new trailing axis.
    log_bessel = jnp.stack(
        [log_iv(2.0 * k + 0.5, kappa) for k in range(n_terms)], axis=-1
    )
    log_beta = jnp.log(beta + eps)[..., None]  # additive floor: beta=0 safe
    log_2_over_kappa = jnp.log(2.0 / kappa)[..., None]
    terms = (
        lg
        + 2.0 * j * log_beta
        + (2.0 * j + 0.5) * log_2_over_kappa
        + log_bessel
    )
    return math.log(2.0 * math.pi) + logsumexp(terms, axis=-1)


def kent_log_prob(
    x: Float[Array, '... 3'],
    gamma1: Float[Array, '... 3'],
    gamma2: Float[Array, '... 3'],
    gamma3: Float[Array, '... 3'],
    kappa: Float[Array, '...'],
    beta: Float[Array, '...'],
    *,
    axis: Optional[_AxisArg] = None,
    reduction: Reduction = 'none',
    normalize: bool = True,
) -> Float[Array, '...']:
    r"""Kent (FB5) log-density of unit vectors on :math:`S^2`.

    :math:`\log f(x) = \kappa\, \gamma_1^\top x + \beta [(\gamma_2^\top x)^2 -
    (\gamma_3^\top x)^2] - \log c(\kappa, \beta)`.  The elliptical generalisation
    of vMF: :math:`\gamma_1` is the mean direction, :math:`\gamma_2` /
    :math:`\gamma_3` the major / minor axes of the elliptical contours, and
    :math:`\beta` the ovalness (:math:`\beta = 0` is vMF).  Differentiable in the
    frame, ``kappa`` and ``beta``.

    Parameters
    ----------
    x : Float[Array, '... 3']
        Observations on :math:`S^2`.
    gamma1, gamma2, gamma3 : Float[Array, '... 3']
        Orthonormal frame: mean direction, major axis, minor axis.
    kappa, beta : Float[Array, '...']
        Concentration and ovalness (``0 <= 2 beta < kappa``).
    axis : int or tuple of int, optional
        Axes to reduce over when ``reduction != "none"``.
    reduction : {'none', 'sum', 'mean'}, optional
        Reduction of the per-observation log-density (default ``"none"``).
    normalize : bool, optional
        If ``False``, the **unnormalised** log-density (energy) -- the series
        normaliser :func:`log_kent_normaliser` is dropped *and never computed*.
        The per-site potential for a Gibbs/Markov-random-field energy.  Default
        ``True``.

    Returns
    -------
    Float[Array, '...']
        The per-observation Kent log-density (``reduction="none"``) or its
        reduction.
    """
    d1 = jnp.sum(gamma1 * x, axis=-1)
    d2 = jnp.sum(gamma2 * x, axis=-1)
    d3 = jnp.sum(gamma3 * x, axis=-1)
    per_obs = kappa * d1 + beta * (d2**2 - d3**2)
    if normalize:
        per_obs = per_obs - log_kent_normaliser(kappa, beta)
    return reduce(per_obs, axis=axis, reduction=reduction)


class KentFit(NamedTuple):
    """A fitted Kent (FB5) distribution: the moment-estimator state.

    Attributes
    ----------
    gamma1, gamma2, gamma3
        The orthonormal frame: mean direction, major axis, minor axis, each
        ``(..., 3)``.
    kappa
        Concentration :math:`\\hat\\kappa`, ``(...)``.
    beta
        Ovalness :math:`\\hat\\beta \\ge 0`, ``(...)``.
    """

    gamma1: Float[Array, '... 3']
    gamma2: Float[Array, '... 3']
    gamma3: Float[Array, '... 3']
    kappa: Float[Array, '...']
    beta: Float[Array, '...']


def _tangent_frame(gamma1: Float[Array, '... 3']) -> Float[Array, '... 3']:
    """A unit vector orthogonal to ``gamma1`` (jit-safe reference choice)."""
    e_x = jnp.zeros_like(gamma1).at[..., 0].set(1.0)
    e_y = jnp.zeros_like(gamma1).at[..., 1].set(1.0)
    # Use e_x unless gamma1 is nearly parallel to it, then e_y.
    ref = jnp.where(jnp.abs(gamma1[..., :1]) < 0.9, e_x, e_y)
    h2 = ref - jnp.sum(ref * gamma1, axis=-1, keepdims=True) * gamma1
    unit = h2 / jnp.linalg.norm(h2, axis=-1, keepdims=True)
    return cast(Float[Array, '... 3'], unit)


def kent_fit(
    x: Float[Array, '... n 3'],
    *,
    weights: Optional[Float[Array, '... n']] = None,
    axis: int = -2,
) -> KentFit:
    r"""Moment-estimator fit of a Kent (FB5) distribution (Kent 1982).

    The mean direction is the normalised resultant :math:`\hat\gamma_1 =
    \bar{x}/\bar R`.  The major / minor axes diagonalise the sample scatter in
    the tangent plane :math:`\perp \hat\gamma_1` (rotation :math:`\psi = \tfrac12
    \operatorname{atan2}(2 B_{23}, B_{22} - B_{33})`), oriented so
    :math:`\hat\gamma_2` carries the larger tangent variance.  With :math:`\bar
    R` the mean resultant length and :math:`Q` the tangent eigenvalue gap, the
    moment estimators are :math:`\hat\kappa = (2 - 2\bar R - Q)^{-1} + (2 - 2\bar
    R + Q)^{-1}` and :math:`\hat\beta = \tfrac12[(2 - 2\bar R - Q)^{-1} - (2 -
    2\bar R + Q)^{-1}]` (at :math:`\beta = 0` this is the S^2 vMF
    :math:`\hat\kappa = 1/(1 - \bar R)`).

    Parameters
    ----------
    x : Float[Array, '... n 3']
        Observations on :math:`S^2`; ``n`` along ``axis``.
    weights : Float[Array, '... n'], optional
        Per-observation non-negative weights.  ``None`` weights equally.
    axis : int, optional
        The observation axis. Default ``-2``.

    Returns
    -------
    KentFit
        The fitted frame, ``kappa`` and ``beta``.

    Notes
    -----
    The moment estimator is derived from the near-pole tangent-Gaussian
    approximation, so :math:`\hat\gamma_1` and the axes are recovered accurately,
    and :math:`\hat\kappa` closely, but :math:`\hat\beta` carries the estimator's
    known **downward finite-**:math:`\kappa` **bias**, worsening as the ovalness
    :math:`2\beta/\kappa \to 1` (where the minor-axis spread is large).  For a
    high-eccentricity / low-:math:`\kappa` regime, refine with a numerical MLE
    seeded from this fit (maximising :func:`kent_log_prob`).
    """
    xm = jnp.moveaxis(x, axis, -2)  # (..., n, 3)
    if weights is None:
        w = jnp.ones(xm.shape[:-1], dtype=xm.dtype)  # (..., n)
    else:
        w_axis = axis if axis >= 0 else axis + 1
        w = jnp.moveaxis(weights, w_axis, -1)
    w_sum = jnp.sum(w, axis=-1, keepdims=True)  # (..., 1)

    def wmean(v: Float[Array, '... n']) -> Float[Array, '...']:
        return jnp.sum(w * v, axis=-1) / w_sum[..., 0]

    resultant = jnp.sum(w[..., None] * xm, axis=-2) / w_sum  # (..., 3)
    r_bar = jnp.linalg.norm(resultant, axis=-1)  # (...)
    gamma1 = resultant / r_bar[..., None]

    h2 = _tangent_frame(gamma1)
    h3 = jnp.cross(gamma1, h2)
    t2 = jnp.sum(h2[..., None, :] * xm, axis=-1)  # (..., n)
    t3 = jnp.sum(h3[..., None, :] * xm, axis=-1)
    b22 = wmean(t2**2)
    b33 = wmean(t3**2)
    b23 = wmean(t2 * t3)
    psi = 0.5 * jnp.arctan2(2.0 * b23, b22 - b33)
    cos_psi = jnp.cos(psi)[..., None]
    sin_psi = jnp.sin(psi)[..., None]
    a2 = cos_psi * h2 + sin_psi * h3
    a3 = -sin_psi * h2 + cos_psi * h3
    l2 = wmean(jnp.sum(a2[..., None, :] * xm, axis=-1) ** 2)
    l3 = wmean(jnp.sum(a3[..., None, :] * xm, axis=-1) ** 2)

    # Orient so gamma2 carries the larger tangent variance (beta >= 0).
    major = l2 >= l3
    gamma2 = jnp.where(major[..., None], a2, a3)
    gamma3 = jnp.where(major[..., None], a3, a2)
    q = jnp.abs(l2 - l3)

    tiny = jnp.finfo(xm.dtype).tiny
    denom_minus = jnp.maximum(2.0 - 2.0 * r_bar - q, tiny)
    denom_plus = jnp.maximum(2.0 - 2.0 * r_bar + q, tiny)
    kappa = 1.0 / denom_minus + 1.0 / denom_plus
    beta = 0.5 * (1.0 / denom_minus - 1.0 / denom_plus)
    return KentFit(
        gamma1=gamma1, gamma2=gamma2, gamma3=gamma3, kappa=kappa, beta=beta
    )


def fisher_bingham_energy(
    x: Float[Array, '... p'],
    gamma: Float[Array, '... p p'],
    kappa: Float[Array, '...'],
    beta: Float[Array, '... p'],
    *,
    axis: Optional[_AxisArg] = None,
    reduction: Reduction = 'none',
) -> Float[Array, '...']:
    r"""Unnormalised log-density (energy) of the Fisher--Bingham distribution.

    On :math:`S^{p-1}`,

    .. math::

        E(x) = \kappa\, \gamma_1^\top x + \sum_{j=1}^{p} \beta_j\,
        (\gamma_j^\top x)^2,

    the **general quadratic-exponential family on the sphere** (Fisher--Bingham,
    :math:`\exp(\kappa \mu^\top x + x^\top A x)` with :math:`A = \Gamma
    \operatorname{diag}(\beta) \Gamma^\top`) in the Kent (diagonalised)
    parametrisation: an orthonormal frame ``gamma`` (columns :math:`\gamma_1,
    \dots, \gamma_p`, with :math:`\gamma_1` the mean direction) and a per-axis
    coefficient vector ``beta``.  It **subsumes** the classical spherical laws:

    - **von Mises--Fisher** -- :math:`\beta = 0` (linear only);
    - **Bingham** -- :math:`\kappa = 0` (the pure quadratic :math:`x^\top A x`);
    - **Watson** -- :math:`\kappa = 0` with a single non-zero :math:`\beta_j`
      (a rank-one quadratic on one axis);
    - **Kent (FB5)** -- :math:`p = 3`, :math:`\beta = (0, \beta, -\beta)`.

    Only the **energy** is provided: the Fisher--Bingham normaliser is
    intractable in general (a hypergeometric function of matrix argument), so
    this is the tractable-at-any-:math:`p` quantity.  Its negative is the Gibbs
    energy (:math:`p \propto e^{-E}`): a per-site clique **potential** that
    composes additively into a Gibbs/Markov-random-field energy.  (The
    fully-normalised densities: :math:`S^2` Kent :func:`kent_log_prob`; vMF /
    Watson :func:`vmf_log_prob` / :func:`watson_log_prob`.)

    .. note::

        The subsumption above is **mathematical, not an implementation-sharing
        directive** -- do *not* collapse the specialised ``normalize=False``
        energies onto this general form.  This form projects onto the full
        frame (an :math:`O(p^2)` ``gamma^T x`` matvec) and always evaluates the
        quadratic term, whereas the vMF / Watson energies are :math:`O(p)`
        single-direction dot products (``mu`` is a vector, not a frame); since
        ``kappa`` / ``beta`` are runtime values, XLA cannot recover the
        specialised path (measured 4--19x slower at :math:`p = 64`--256).  The
        specialised energies are retained deliberately; use this general form
        only when the parameters genuinely are a full frame plus a coefficient
        vector.

    For a *proper* density the frame is orthonormal, :math:`\beta` is fixed only
    up to a common additive shift (the :math:`\beta_1 = 0` gauge is conventional
    -- no quadratic on the mean axis), and the concentration dominates the
    ovalness; none is enforced -- the energy is well defined for any parameters.

    Parameters
    ----------
    x : Float[Array, '... p']
        Observations on :math:`S^{p-1}`.
    gamma : Float[Array, '... p p']
        Orthonormal frame; column ``j`` is the axis :math:`\gamma_j` (column 0
        the mean direction).  Assumed orthonormal (not enforced).
    kappa : Float[Array, '...']
        Concentration of the linear (mean-direction) term.
    beta : Float[Array, '... p']
        Per-axis quadratic (ovalness) coefficients; ``beta[..., 0]`` is the
        mean-axis coefficient (conventionally ``0``).
    axis : int or tuple of int, optional
        Axes to reduce over when ``reduction != "none"``.
    reduction : {'none', 'sum', 'mean'}, optional
        Reduction of the per-observation energy (default ``"none"``).

    Returns
    -------
    Float[Array, '...']
        The per-observation energy (``reduction="none"``) or its reduction.
    """
    proj = jnp.einsum(
        '...jk,...j->...k', gamma, x
    )  # gamma_k^T x, shape (..., p)
    per_obs = kappa * proj[..., 0] + jnp.sum(beta * proj**2, axis=-1)
    return reduce(per_obs, axis=axis, reduction=reduction)
