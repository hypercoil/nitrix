# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Within-group residual-correlation structures (v3 §1.4).

A structured residual replaces the ``sigma_e^2 I`` error model with
``sigma_e^2 R(rho)``, where ``R`` is a within-group correlation matrix carrying
one (or few) correlation parameters: ``ar1`` (discrete AR(1)), ``car1``
(continuous-time AR(1), unequally-spaced times), and ``cs`` (compound symmetry).
This is the ``nlme`` ``correlation=corAR1 / corCAR1 / corCompSymm`` surface.

The unifying device is **whitening**.  Each structure supplies, per group, a
closed-form transform ``W_i`` with ``W_i R_i W_i^T = I`` -- so on whitened data
(``W_i y_i``, ``W_i X_i``, ``W_i Z_i``) the residual is i.i.d. and the rest of
the LME machinery (GLS / block-Woodbury) applies unchanged, with a single extra
``0.5 * log|R_i|`` per group added to the REML objective (the whitening
Jacobian).  Every transform is **closed-form and cuSOLVER-free**:

- ``ar1`` / ``car1`` -- the innovations form ``w_t = (z_t - phi_t z_{t-1}) /
  sqrt(1 - phi_t^2)`` (``w_0 = z_0``), a one-shift elementwise recurrence (no
  matrix inverse, no scan).  ``phi_t = rho`` (AR1) or ``rho^{Delta t_t}`` (CAR1).
  ``log|R_i| = sum_{t>=1} log(1 - phi_t^2)``.
- ``cs`` -- the rank-one whitener ``w = a z + (b - a) mean(z) 1`` with
  ``a = (1-rho)^{-1/2}``, ``b = (1 + (n_i-1) rho)^{-1/2}``; ``log|R_i| =
  (n_i-1) log(1-rho) + log(1 + (n_i-1) rho)``.

Groups are stored **left-packed and time-sorted** in a ``(G, T)`` padded layout
(``T`` = max group size) with a boolean ``mask``; padded positions are zeroed and
excluded from every reduction, so ragged group sizes are handled without dynamic
shapes.  The correlation parameter is carried **unconstrained** (the fit
optimises a real ``raw``; ``rho`` is recovered by a bounded transform) so Newton
stays unconstrained.

References
----------
- Pinheiro, J. C. & Bates, D. M. (2000). Mixed-Effects Models in S and S-PLUS.
  Springer.  (corAR1 / corCAR1 / corCompSymm; the innovations whitening.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

__all__ = [
    'CorrSpec',
    'ar1',
    'car1',
    'cs',
    'resolve_corr',
]

_EPS = 1e-6


@dataclass(frozen=True)
class CorrSpec:
    """A within-group correlation structure, as a record of pure functions.

    Frozen and hashable so it rides as a static config (a ``vmap`` /
    ``custom_vjp`` nondiff argument), like ``Family`` / ``VarCompSpec``.

    Fields
    ------
    name
        Structure name (``'ar1'`` / ``'car1'`` / ``'cs'``).
    n_params
        Number of correlation parameters (``1`` for all three here).
    whiten
        ``(z_pad, gaps, nsize, mask, raw) -> (w_pad, half_logdet)``: apply the
        per-group whitener to a ``(G, T, k)`` padded stack ``z_pad`` (zeroed pad),
        returning the whitened stack and ``0.5 * sum_i log|R_i|`` (summed over
        groups).  ``gaps`` is the ``(G, T)`` time gap to the previous in-group
        observation (``car1``); ``nsize`` the ``(G,)`` real group sizes; ``mask``
        the ``(G, T)`` validity mask; ``raw`` the ``(n_params,)`` unconstrained
        parameter.
    to_natural
        ``raw -> rho`` (the bounded natural correlation, for reporting).
    init_raw
        A reasonable unconstrained start.
    """

    name: str
    n_params: int
    whiten: Callable[
        [
            Float[Array, 'G T k'],
            Float[Array, 'G T'],
            Float[Array, 'G'],
            Bool[Array, 'G T'],
            Float[Array, 'n'],
        ],
        Tuple[Float[Array, 'G T k'], Float[Array, '']],
    ]
    to_natural: Callable[[Float[Array, 'n']], Float[Array, '']]
    init_raw: Callable[[Any], Float[Array, 'n']]  # dtype -> initial raw params


# ---------------------------------------------------------------------------
# Innovations whitening shared by ar1 / car1 (phi differs)
# ---------------------------------------------------------------------------


def _innovations_whiten(
    z_pad: Float[Array, 'G T k'],
    phi: Float[Array, 'G T'],
    mask: Bool[Array, 'G T'],
) -> Tuple[Float[Array, 'G T k'], Float[Array, '']]:
    """Innovations whitening ``w_t = (z_t - phi_t z_{t-1}) / sqrt(1 - phi_t^2)``.

    ``phi[:, 0]`` is taken as ``0`` (so ``w_0 = z_0``); ``phi`` for ``t >= 1`` is
    the lag-1 (AR1) or time-decayed (CAR1) coefficient.  A single shift along the
    time axis implements the recurrence -- it depends on the *original* lagged
    value, not the whitened one, so no scan is needed.  Padded positions are
    re-zeroed; ``half_logdet = 0.5 sum_{real, t>=1} log(1 - phi_t^2)``.
    """
    g, t = phi.shape
    phi = phi.at[:, 0].set(0.0)
    z_prev = jnp.concatenate(
        [jnp.zeros_like(z_pad[:, :1]), z_pad[:, :-1]], axis=1
    )  # (G, T, k)
    one_m_phi2 = jnp.clip(1.0 - phi * phi, _EPS, None)  # (G, T)
    denom = jnp.sqrt(one_m_phi2)
    w = (z_pad - phi[..., None] * z_prev) / denom[..., None]
    w = w * mask[..., None]
    # log|R| contribution: only real positions with a predecessor (t >= 1).
    col = jnp.arange(t)
    contrib = jnp.where(mask & (col[None, :] >= 1), jnp.log(one_m_phi2), 0.0)
    half_logdet = 0.5 * jnp.sum(contrib)
    return w, half_logdet


def ar1() -> CorrSpec:
    """Discrete AR(1) within-group correlation (``nlme`` ``corAR1``).

    ``R_{ij} = rho^{|i - j|}`` over the time-ordered observations; ``rho`` may be
    negative (oscillating), so ``rho = tanh(raw) in (-1, 1)``.
    """

    def whiten(
        z_pad: Float[Array, 'G T k'],
        gaps: Float[Array, 'G T'],
        nsize: Float[Array, 'G'],
        mask: Bool[Array, 'G T'],
        raw: Float[Array, 'n'],
    ) -> Tuple[Float[Array, 'G T k'], Float[Array, '']]:
        rho = jnp.tanh(raw[0])
        phi = jnp.broadcast_to(rho, mask.shape)
        return _innovations_whiten(z_pad, phi, mask)

    return CorrSpec(
        name='ar1',
        n_params=1,
        whiten=whiten,
        to_natural=lambda raw: jnp.tanh(raw[0]),
        init_raw=lambda dtype: jnp.zeros((1,), dtype=dtype),
    )


def car1() -> CorrSpec:
    """Continuous-time AR(1) for unequally-spaced times (``nlme`` ``corCAR1``).

    ``R_{ij} = rho^{|t_i - t_j|}`` with ``rho in (0, 1)`` (a positive decay), so
    ``rho = sigmoid(raw)``.  ``phi_t = rho^{Delta t_t}`` is the decay over the gap
    to the previous in-group observation; AR(1) is the unit-gap special case.
    """

    def whiten(
        z_pad: Float[Array, 'G T k'],
        gaps: Float[Array, 'G T'],
        nsize: Float[Array, 'G'],
        mask: Bool[Array, 'G T'],
        raw: Float[Array, 'n'],
    ) -> Tuple[Float[Array, 'G T k'], Float[Array, '']]:
        rho = jnp.clip(_sigmoid(raw[0]), _EPS, 1.0 - _EPS)
        # phi_t = rho ** gap_t = exp(gap_t * log rho); gaps >= 0 (sorted).
        phi = jnp.exp(jnp.clip(gaps, 0.0, None) * jnp.log(rho))
        return _innovations_whiten(z_pad, phi, mask)

    return CorrSpec(
        name='car1',
        n_params=1,
        whiten=whiten,
        to_natural=lambda raw: _sigmoid(raw[0]),
        init_raw=lambda dtype: jnp.zeros((1,), dtype=dtype),
    )


def cs() -> CorrSpec:
    """Compound symmetry (exchangeable) within-group correlation
    (``nlme`` ``corCompSymm``).

    ``R = (1 - rho) I + rho 11^T`` (constant off-diagonal ``rho``).  Whitened by
    the rank-one transform ``w = a z + (b - a) mean(z) 1``,
    ``a = (1-rho)^{-1/2}``, ``b = (1 + (n_i-1) rho)^{-1/2}``.  ``rho in (0, 1)``
    via ``sigmoid`` (the common positive-correlation regime; the small negative
    range admissible for finite ``n`` is not exposed).
    """

    def whiten(
        z_pad: Float[Array, 'G T k'],
        gaps: Float[Array, 'G T'],
        nsize: Float[Array, 'G'],
        mask: Bool[Array, 'G T'],
        raw: Float[Array, 'n'],
    ) -> Tuple[Float[Array, 'G T k'], Float[Array, '']]:
        rho = jnp.clip(_sigmoid(raw[0]), _EPS, 1.0 - _EPS)
        n_g = nsize  # (G,)
        a = (1.0 - rho) ** -0.5
        lam1 = 1.0 + (n_g - 1.0) * rho  # (G,) top eigenvalue
        b = lam1**-0.5  # (G,)
        # group mean over real observations: sum(z*mask)/n_g
        denom = jnp.clip(n_g, 1.0, None)[:, None, None]
        z_mean = jnp.sum(z_pad, axis=1, keepdims=True) / denom  # (G, 1, k)
        w = a * z_pad + (b - a)[:, None, None] * z_mean
        w = w * mask[..., None]
        # log|R_i| = (n_i - 1) log(1 - rho) + log(lam1_i), summed over groups.
        valid = nsize > 0
        half_logdet = 0.5 * jnp.sum(
            jnp.where(
                valid,
                (n_g - 1.0) * jnp.log(1.0 - rho) + jnp.log(lam1),
                0.0,
            )
        )
        return w, half_logdet

    return CorrSpec(
        name='cs',
        n_params=1,
        whiten=whiten,
        to_natural=lambda raw: _sigmoid(raw[0]),
        init_raw=lambda dtype: jnp.zeros((1,), dtype=dtype),
    )


def _sigmoid(x: Float[Array, '']) -> Float[Array, '']:
    return 0.5 * (1.0 + jnp.tanh(0.5 * x))


_CORRS: Mapping[str, Callable[[], CorrSpec]] = {
    'ar1': ar1,
    'car1': car1,
    'cs': cs,
}


def resolve_corr(corr: Union[str, CorrSpec]) -> CorrSpec:
    """Resolve a ``str`` name (a built-in) or a ``CorrSpec`` to a ``CorrSpec``.

    Built-ins: ``'ar1'`` / ``'car1'`` / ``'cs'``.
    """
    if isinstance(corr, CorrSpec):
        return corr
    try:
        return _CORRS[corr]()
    except KeyError:
        raise ValueError(
            f'unknown correlation {corr!r}; built-ins are {sorted(_CORRS)}, '
            f'or pass a CorrSpec.'
        ) from None
