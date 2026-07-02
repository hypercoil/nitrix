# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Within-group residual **variance functions** (heteroscedasticity).

A variance function relaxes the constant-variance assumption: the residual scale
varies across observations as :math:`\operatorname{Var}(\varepsilon_i) =
\sigma_e^2 g_i^2`, with :math:`g_i` a parametric function of a covariate or
stratum.  This is ``nlme``'s ``weights = varPower / varIdent`` surface, composing
with the :class:`~nitrix.stats.lme._corr.CorrSpec` correlation structures: the
full residual is :math:`\sigma_e^2 \operatorname{diag}(g) R(\rho)
\operatorname{diag}(g)`.

The unifying device is again **whitening**: :math:`g` enters as a diagonal
pre-scale.  On data divided by :math:`g` (:math:`y_i / g_i`, :math:`X_i / g_i`)
the residual is :math:`\sigma_e^2 R` -- the ordinary structured-residual problem
-- so the correlation whitener applies unchanged, with a single extra
:math:`\sum_i \log g_i` added to the REML objective (the diagonal whitening
Jacobian, since :math:`\log|\operatorname{diag}(g) R \operatorname{diag}(g)| =
2 \sum \log g + \log|R|`).  Two structures:

- ``varPower`` -- :math:`g_i = |v_i|^\delta` for a variance covariate :math:`v`
  (an external, non-circular covariate -- e.g. a baseline level), one parameter
  :math:`\delta` carried **unconstrained** (any real power; :math:`\delta = 0` is
  homoscedastic).  :math:`\log g_i = \delta \log|v_i|`.
- ``varIdent`` -- :math:`g_i = \exp(\tau_{s(i)})` for a stratum factor :math:`s`
  with :math:`S` levels, :math:`S - 1` parameters (the first stratum is the
  reference, :math:`\tau_0 = 0`, :math:`g = 1`).  This is a separate residual
  variance per stratum.

The parameters join the correlation ``raw`` in the GLS Newton vector; every
quantity is closed-form and cuSOLVER-free.

References
----------
- Pinheiro, J. C. & Bates, D. M. (2000). Mixed-Effects Models in S and S-PLUS.
  Springer.  (varPower / varIdent; the ``weights=`` variance-function surface.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

__all__ = [
    'VarFunc',
    'var_power',
    'var_ident',
]

_EPS = 1e-8


@dataclass(frozen=True)
class VarFunc:
    r"""A within-group residual variance function, as a record of pure functions.

    Frozen and hashable so it rides as a static config (a ``vmap`` nondiff
    argument), like :class:`~nitrix.stats.lme._corr.CorrSpec`.

    Fields
    ------
    name
        Structure name (``'varPower'`` / ``'varIdent'``).
    n_params
        Number of variance-function parameters (``1`` for ``varPower``;
        ``S - 1`` for ``varIdent`` over ``S`` strata).
    covariate
        The ``(N,)`` variance covariate: the continuous :math:`v` for
        ``varPower``, or the integer stratum index ``0 .. S-1`` for
        ``varIdent``.  It rides through the same ``(G, T)`` padded group layout
        as the design.
    log_g
        ``(cov_pad, mask, raw) -> log_g_pad``: the per-observation
        :math:`\log g_i` on the ``(G, T)`` padded grid given the padded
        covariate and the ``(n_params,)`` unconstrained parameter.
    init_raw
        A reasonable unconstrained start (homoscedastic, :math:`g = 1`).
    """

    name: str
    n_params: int
    covariate: Float[Array, 'N']
    log_g: Callable[
        [Float[Array, 'G T'], Bool[Array, 'G T'], Float[Array, 'n']],
        Float[Array, 'G T'],
    ]
    init_raw: Callable[[Any], Float[Array, 'n']]


def var_power(v: Float[Array, 'N']) -> VarFunc:
    r"""Power-of-covariate residual variance function (``nlme`` ``varPower``).

    Models the residual scale as a power of an external covariate,
    :math:`\operatorname{Var}(\varepsilon_i) = \sigma_e^2 |v_i|^{2 \delta}`.
    ``v`` is passed explicitly, so the fit stays non-circular -- it is *not* the
    fitted value.  There is one unconstrained parameter :math:`\delta` (any real;
    :math:`\delta = 0` recovers homoscedasticity), with :math:`\log g_i = \delta
    \log|v_i|`.

    Parameters
    ----------
    v : Float[Array, 'N']
        The ``(N,)`` variance covariate, one value per observation.  An external
        (non-circular) covariate such as a baseline level; its magnitude
        :math:`|v_i|` drives the residual scale.

    Returns
    -------
    VarFunc
        A variance-function record named ``'varPower'`` with a single parameter,
        carrying ``v`` as its covariate and the :math:`\log g_i = \delta
        \log|v_i|` log-scale map.
    """
    v_arr = jnp.asarray(v)

    def log_g(
        cov_pad: Float[Array, 'G T'],
        mask: Bool[Array, 'G T'],
        raw: Float[Array, 'n'],
    ) -> Float[Array, 'G T']:
        return raw[0] * jnp.log(jnp.clip(jnp.abs(cov_pad), _EPS, None))

    return VarFunc(
        name='varPower',
        n_params=1,
        covariate=v_arr,
        log_g=log_g,
        init_raw=lambda dtype: jnp.zeros((1,), dtype=dtype),
    )


def var_ident(strata: Float[Array, 'N']) -> VarFunc:
    r"""Per-stratum residual variance function (``nlme`` ``varIdent``).

    Fits a separate residual variance for each stratum,
    :math:`\operatorname{Var}(\varepsilon_i) = \sigma_e^2 \exp(2 \tau_{s(i)})`.
    ``strata`` is an integer factor with levels ``0 .. S-1``; the first level is
    the reference (:math:`\tau_0 = 0`, :math:`g = 1`), leaving :math:`S - 1` free
    parameters, with :math:`\log g_i = \tau_{s(i)}`.

    Parameters
    ----------
    strata : Float[Array, 'N']
        The ``(N,)`` stratum factor, one integer-valued level per observation in
        ``0 .. S-1``.  The number of strata :math:`S` is inferred as one plus the
        maximum level; level ``0`` is the reference.

    Returns
    -------
    VarFunc
        A variance-function record named ``'varIdent'`` with :math:`S - 1`
        parameters, carrying the stratum index as its covariate and the
        stratum-lookup log-scale map.
    """
    s_arr = jnp.asarray(strata)
    n_str = int(s_arr.max()) + 1 if s_arr.shape[0] else 1

    def log_g(
        cov_pad: Float[Array, 'G T'],
        mask: Bool[Array, 'G T'],
        raw: Float[Array, 'n'],
    ) -> Float[Array, 'G T']:
        tau_full = jnp.concatenate(
            [jnp.zeros((1,), dtype=raw.dtype), raw]
        )  # (S,), reference tau_0 = 0
        return tau_full[jnp.clip(cov_pad, 0.0, None).astype(jnp.int32)]

    return VarFunc(
        name='varIdent',
        n_params=max(n_str - 1, 0),
        covariate=s_arr.astype(jnp.float32),
        log_g=log_g,
        init_raw=lambda dtype: jnp.zeros((max(n_str - 1, 0),), dtype=dtype),
    )


def _apply_var_scale(
    stack: Float[Array, 'G T k'],
    cov_pad: Float[Array, 'G T'],
    mask: Bool[Array, 'G T'],
    varfunc: VarFunc,
    raw: Float[Array, 'n'],
) -> Tuple[Float[Array, 'G T k'], Float[Array, '']]:
    r"""Pre-scale a padded stack by :math:`1 / g` and accumulate the log-Jacobian.

    Divides the padded observation stack by the per-observation scale :math:`g =
    \exp(\log g_i)` and returns the whitening Jacobian term :math:`\sum_i \log
    g_i`.  On the pad :math:`g = 1` (so padded zeros stay zero); the Jacobian
    sums :math:`\log g_i` over real observations only.

    Parameters
    ----------
    stack : Float[Array, 'G T k']
        The padded stack to pre-scale, ``G`` groups by ``T`` in-group slots by
        ``k`` columns (e.g. the stacked response and design).
    cov_pad : Float[Array, 'G T']
        The variance covariate laid out on the same ``(G, T)`` padded grid.
    mask : Bool[Array, 'G T']
        Boolean mask marking real (``True``) versus padded (``False``)
        observations.
    varfunc : VarFunc
        The variance function supplying the ``log_g`` map.
    raw : Float[Array, 'n']
        The ``(n_params,)`` unconstrained variance-function parameter.

    Returns
    -------
    scaled : Float[Array, 'G T k']
        ``stack`` divided by :math:`g` per observation, with padded slots left
        unchanged.
    sum_log_g : Float[Array, '']
        The scalar Jacobian term :math:`\sum_i \log g_i`, summed over real
        observations only.
    """
    lg = varfunc.log_g(cov_pad, mask, raw)  # (G, T)
    g = jnp.where(mask, jnp.exp(lg), 1.0)
    scaled = stack / g[..., None]
    sum_log_g = jnp.sum(jnp.where(mask, lg, 0.0))
    return scaled, sum_log_g
