# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Shared Hilbert-space approximate-GP (HSGP) eigenstructure.

The fixed Laplace-Dirichlet eigenfunction basis on a box domain (1-D and the
multi-dimensional tensor product), the (rho-independent) eigenfunction design,
and the diagonal spectral-reweighting penalty -- the single source of truth used
by :mod:`nitrix.stats.gp` (``gp_fit``), :mod:`nitrix.stats.hgp` (``hgp_fit``), and
:mod:`nitrix.stats.basis` (``hsgp_basis``).  Neutral: depends only on the kernel
spectral densities, so it carries no import cycle between those modules.

``phi_j(x) = sqrt(1/L) sin(sqrt(lambda_j) (x - c + L))`` on ``[c - L, c + L]``; the
kernel enters only as the diagonal reweighting ``1 / S_rho(sqrt(lambda_j))``.
"""

from __future__ import annotations

from typing import Any, Tuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from ..linalg.kernel import spectral_density

__all__ = [
    '_hsgp_domain',
    '_hsgp_eigen',
    '_hsgp_features',
    '_penalty_diag',
    '_hsgp_eigen_nd',
    '_hsgp_features_nd',
    '_penalty_diag_nd_iso',
    '_penalty_diag_nd_ard',
]


def _hsgp_domain(
    lo: float, hi: float, boundary: float
) -> Tuple[float, float]:
    """``(c, L)``: the domain midrange and the half-width of ``[c - L, c + L]``."""
    c = 0.5 * (lo + hi)
    half = max(0.5 * (hi - lo), 1e-6)
    return c, float(boundary) * half


def _hsgp_eigen(
    rank: int, c: float, L: float, dtype: Any
) -> Tuple[Float[Array, ' m'], Float[Array, ' m'], float]:
    """The fixed Laplace-Dirichlet eigen-frequencies, per-mode phase, and the
    ``sqrt(1/L)`` amplitude -- everything the eigenfunctions need, independent of
    the kernel and ``rho``.

    ``phi_j(x) = sqrt(1/L) sin(sqrt(lambda_j) (x - c + L)) = sqrt(1/L)
    sin(sqrt(lambda_j) x + phase_j)`` with ``phase_j = sqrt(lambda_j) (L - c)``
    (folding the centring constant ``c`` into the stored phase).
    """
    j = np.arange(1, rank + 1, dtype=np.float64)
    sqrt_lambda = j * np.pi / (2.0 * L)
    phase = sqrt_lambda * (L - c)
    return (
        jnp.asarray(sqrt_lambda, dtype=dtype),
        jnp.asarray(phase, dtype=dtype),
        float(np.sqrt(1.0 / L)),
    )


def _hsgp_features(
    x: Float[Array, ' g'],
    sqrt_lambda: Float[Array, ' m'],
    phase: Float[Array, ' m'],
    inv_sqrt_L: float,
) -> Float[Array, 'g m']:
    """The (rho-independent) eigenfunction design ``Phi`` at covariate ``x``."""
    return inv_sqrt_L * jnp.sin(
        sqrt_lambda[None, :] * x[:, None] + phase[None, :]
    )


def _penalty_diag(
    sqrt_lambda: Float[Array, ' m'],
    kernel: str,
    rho: Float[Array, ''],
    n_fixed: int,
) -> Tuple[Float[Array, ' p'], Float[Array, '']]:
    """The diagonal penalty core ``d`` over the full ``p = n_fixed + m`` columns
    and the smooth-block log-pseudo-determinant contribution ``sum_j log d_j``.

    The penalty is ``S_lambda = lambda diag(d)`` with ``d = [0, ..., 0, 1/s_1,
    ..., 1/s_m]`` (zeros on the unpenalised fixed-effect columns); ``s_j =
    S_rho(sqrt(lambda_j))`` is the spectral density (amplitude folded into
    ``lambda``).  Returns ``(d, sum_j log(1/s_j))``.
    """
    s = spectral_density(sqrt_lambda, kernel=kernel, rho=rho, amplitude=1.0)
    inv_s = 1.0 / jnp.clip(s, 1e-30, None)
    d = jnp.concatenate(
        [jnp.zeros((n_fixed,), dtype=inv_s.dtype), inv_s]
    )
    return d, jnp.sum(jnp.log(inv_s))


def _hsgp_eigen_nd(
    x_np: np.ndarray, m_per: Tuple[int, ...], boundary: float, dtype: Any
) -> Tuple[Array, Array, Array, Array, Tuple[Tuple[float, float], ...]]:
    """Tensor-product Laplace eigenstructure on the box domain (host build).

    Returns ``(freqs, phase, inv_sqrt_L, omega_norm, bounds)``: per-mode per-axis
    eigen-frequencies ``(M, D)``, phases ``(M, D)``, the per-axis ``sqrt(1/L_d)``
    ``(D,)``, the mode-frequency magnitude ``||w|| = sqrt(sum_d lambda_{j_d})``
    ``(M,)``, and the per-axis ``(lo, hi)`` (recorded for re-evaluation).
    ``M = prod_d m_per[d]``."""
    lo = x_np.min(axis=0)
    hi = x_np.max(axis=0)
    c_mid = 0.5 * (lo + hi)
    big_l = float(boundary) * np.maximum(0.5 * (hi - lo), 1e-6)
    d_in = x_np.shape[1]
    sqrt_lams = [
        np.arange(1, m_per[d] + 1, dtype=np.float64) * np.pi / (2.0 * big_l[d])
        for d in range(d_in)
    ]
    grids = np.meshgrid(*sqrt_lams, indexing='ij')
    freqs = np.stack([g.ravel() for g in grids], axis=1)  # (M, D)
    phase = freqs * (big_l - c_mid)[None, :]
    inv_sqrt_L = np.sqrt(1.0 / big_l)
    omega_norm = np.sqrt((freqs**2).sum(axis=1))
    bounds = tuple((float(lo[d]), float(hi[d])) for d in range(d_in))
    return (
        jnp.asarray(freqs, dtype=dtype),
        jnp.asarray(phase, dtype=dtype),
        jnp.asarray(inv_sqrt_L, dtype=dtype),
        jnp.asarray(omega_norm, dtype=dtype),
        bounds,
    )


def _hsgp_features_nd(
    X: Float[Array, 'g D'],
    freqs: Float[Array, 'M D'],
    phase: Float[Array, 'M D'],
    inv_sqrt_L: Float[Array, ' D'],
) -> Float[Array, 'g M']:
    """The (rho-independent) tensor-product eigenfunction design ``Phi`` at ``X``:
    ``prod_d sqrt(1/L_d) sin(w_{m,d} x_d + phase_{m,d})``."""
    X = jnp.asarray(X)
    arg = freqs[None, :, :] * X[:, None, :] + phase[None, :, :]  # (g, M, D)
    terms = inv_sqrt_L[None, None, :] * jnp.sin(arg)
    return jnp.prod(terms, axis=2)  # (g, M)


def _penalty_diag_nd_iso(
    omega_norm: Float[Array, ' M'],
    kernel: str,
    rho: float,
    dim: int,
    n_fixed: int,
) -> Tuple[Float[Array, ' p'], Float[Array, '']]:
    """Isotropic tensor-HSGP diagonal penalty: ``1/S_D(||w||; rho)`` (the ``D``-dim
    radial spectral density), zeros on the fixed columns."""
    s = spectral_density(
        omega_norm, kernel=kernel, rho=jnp.asarray(rho, dtype=omega_norm.dtype),
        amplitude=1.0, dim=dim,
    )
    inv_s = 1.0 / jnp.clip(s, 1e-30, None)
    d = jnp.concatenate([jnp.zeros((n_fixed,), dtype=inv_s.dtype), inv_s])
    return d, jnp.sum(jnp.log(inv_s))


def _penalty_diag_nd_ard(
    freqs: Float[Array, 'M D'],
    kernel: str,
    rho_vec: Tuple[float, ...],
    n_fixed: int,
) -> Tuple[Float[Array, ' p'], Float[Array, '']]:
    """Separable / ARD tensor-HSGP penalty: ``1 / prod_d S_1(w_{m,d}; rho_d)`` (a
    per-axis 1-D density), zeros on the fixed columns."""
    dtype = freqs.dtype
    s = jnp.ones((freqs.shape[0],), dtype=dtype)
    for d in range(freqs.shape[1]):
        s = s * spectral_density(
            freqs[:, d], kernel=kernel,
            rho=jnp.asarray(rho_vec[d], dtype=dtype), amplitude=1.0, dim=1,
        )
    inv_s = 1.0 / jnp.clip(s, 1e-30, None)
    d_full = jnp.concatenate([jnp.zeros((n_fixed,), dtype=dtype), inv_s])
    return d_full, jnp.sum(jnp.log(inv_s))
