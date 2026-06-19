# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Within-group residual **variance functions** (heteroscedasticity, v3 §1.4).

A variance function relaxes the constant-variance assumption: the residual scale
varies across observations as ``Var(eps_i) = sigma_e^2 g_i^2``, with ``g_i`` a
parametric function of a covariate or stratum.  This is ``nlme``'s ``weights =
varPower / varIdent`` surface, composing with the :mod:`._corr` correlation
structures -- the full residual is ``sigma_e^2 diag(g) R(rho) diag(g)``.

The unifying device is again **whitening**: ``g`` enters as a diagonal pre-scale.
On data divided by ``g`` (``y_i / g_i``, ``X_i / g_i``) the residual is
``sigma_e^2 R`` -- the ordinary structured-residual problem -- so the
:mod:`._corr` whitener applies unchanged, with a single extra ``sum_i log g_i``
added to the REML objective (the diagonal whitening Jacobian, since
``log|diag(g) R diag(g)| = 2 sum log g + log|R|``).  Two structures:

- ``varPower`` -- ``g_i = |v_i|^delta`` for a variance covariate ``v`` (an
  external, non-circular covariate -- e.g. a baseline level), one parameter
  ``delta`` carried **unconstrained** (any real power; ``delta = 0`` is
  homoscedastic).  ``log g_i = delta log|v_i|``.
- ``varIdent`` -- ``g_i = exp(tau_{s(i)})`` for a stratum factor ``s`` with
  ``S`` levels, ``S - 1`` parameters (the first stratum is the reference,
  ``tau_0 = 0``, ``g = 1``).  This is a separate residual variance per stratum.

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
    """A within-group residual variance function, as a record of pure functions.

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
        The ``(N,)`` variance covariate: the continuous ``v`` for ``varPower``,
        or the integer stratum index ``0 .. S-1`` for ``varIdent``.  It rides
        through the same ``(G, T)`` padded group layout as the design.
    log_g
        ``(cov_pad, mask, raw) -> log_g_pad``: the per-observation ``log g_i``
        on the ``(G, T)`` padded grid given the padded covariate and the
        ``(n_params,)`` unconstrained parameter.
    init_raw
        A reasonable unconstrained start (homoscedastic, ``g = 1``).
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
    """Power-of-covariate variance ``Var(eps_i) = sigma_e^2 |v_i|^{2 delta}``
    (``nlme`` ``varPower``).

    ``v`` is an external variance covariate (passed explicitly, so the fit stays
    non-circular -- it is *not* the fitted value).  One unconstrained parameter
    ``delta`` (any real; ``delta = 0`` -> homoscedastic).  ``log g_i = delta
    log|v_i|``.
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
    """Per-stratum residual variance ``Var(eps_i) = sigma_e^2 exp(2 tau_{s(i)})``
    (``nlme`` ``varIdent``).

    ``strata`` is an integer factor with levels ``0 .. S-1``; the first level is
    the reference (``tau_0 = 0``, ``g = 1``), leaving ``S - 1`` free parameters
    -- a separate residual variance per stratum.  ``log g_i = tau_{s(i)}``.
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
    """Pre-scale a padded stack by ``1 / g`` and return ``(scaled, sum log g)``.

    ``g = 1`` on the pad (so padded zeros stay zero); the Jacobian sums
    ``log g_i`` over real observations only.
    """
    lg = varfunc.log_g(cov_pad, mask, raw)  # (G, T)
    g = jnp.where(mask, jnp.exp(lg), 1.0)
    scaled = stack / g[..., None]
    sum_log_g = jnp.sum(jnp.where(mask, lg, 0.0))
    return scaled, sum_log_g
