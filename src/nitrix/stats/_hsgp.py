# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Shared Hilbert-space approximate-GP (HSGP) eigenstructure.

The fixed Laplace--Dirichlet eigenfunction basis on a box domain (both the 1-D
case and the multi-dimensional tensor product), the eigenfunction design (which
is independent of the length-scale :math:`\\rho`), and the diagonal
spectral-reweighting penalty. This module is the single source of truth for that
eigenstructure, shared by :mod:`nitrix.stats.gp` (:func:`~nitrix.stats.gp.gp_fit`),
:mod:`nitrix.stats.hgp` (:func:`~nitrix.stats.hgp.hgp_fit`), and
:mod:`nitrix.stats.basis` (:func:`~nitrix.stats.basis.hsgp_basis`). It depends only
on the kernel spectral densities, so it introduces no import cycle between those
modules.

The eigenfunctions are
:math:`\\phi_j(x) = \\sqrt{1/L}\\,\\sin(\\sqrt{\\lambda_j}\\,(x - c + L))` on the
box :math:`[c - L,\\, c + L]`; the kernel enters only through the diagonal
reweighting :math:`1 / S_\\rho(\\sqrt{\\lambda_j})`.
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


def _hsgp_domain(lo: float, hi: float, boundary: float) -> Tuple[float, float]:
    """Centre and half-width of the (padded) box domain :math:`[c - L,\\, c + L]`.

    The centre :math:`c` is the midrange of the observed interval and the
    half-width :math:`L` is the observed half-range scaled by ``boundary`` (with a
    small floor to avoid a degenerate zero-width domain).

    Parameters
    ----------
    lo : float
        Lower extent of the observed covariate range.
    hi : float
        Upper extent of the observed covariate range.
    boundary : float
        Multiplicative padding factor applied to the observed half-range, so the
        eigenfunction support extends beyond the data.

    Returns
    -------
    c : float
        Domain centre :math:`c`, the midrange :math:`\\tfrac{1}{2}(lo + hi)`.
    L : float
        Domain half-width :math:`L`, the padded half-range.
    """
    c = 0.5 * (lo + hi)
    half = max(0.5 * (hi - lo), 1e-6)
    return c, float(boundary) * half


def _hsgp_eigen(
    rank: int, c: float, L: float, dtype: Any
) -> Tuple[Float[Array, ' m'], Float[Array, ' m'], float]:
    """Fixed Laplace--Dirichlet eigenstructure for the 1-D eigenfunctions.

    Returns the eigen-frequencies, the per-mode phase, and the common amplitude
    :math:`\\sqrt{1/L}` -- everything the eigenfunctions require, independent of
    the kernel and of the length-scale :math:`\\rho`. Rewriting the basis as
    :math:`\\phi_j(x) = \\sqrt{1/L}\\,\\sin(\\sqrt{\\lambda_j}\\,(x - c + L))
    = \\sqrt{1/L}\\,\\sin(\\sqrt{\\lambda_j}\\,x + \\mathrm{phase}_j)` folds the
    centring constant :math:`c` into the stored phase
    :math:`\\mathrm{phase}_j = \\sqrt{\\lambda_j}\\,(L - c)`.

    Parameters
    ----------
    rank : int
        Number of basis functions :math:`m`; modes are indexed
        :math:`j = 1, \\ldots, m`.
    c : float
        Domain centre :math:`c`.
    L : float
        Domain half-width :math:`L`.
    dtype : Any
        Floating dtype of the returned frequency and phase arrays.

    Returns
    -------
    sqrt_lambda : Float[Array, ' m']
        Square-root eigenvalues :math:`\\sqrt{\\lambda_j} = j\\pi / (2L)`.
    phase : Float[Array, ' m']
        Per-mode phase :math:`\\mathrm{phase}_j = \\sqrt{\\lambda_j}\\,(L - c)`.
    inv_sqrt_L : float
        Common eigenfunction amplitude :math:`\\sqrt{1/L}`.
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
    """Eigenfunction design matrix :math:`\\Phi` at covariate ``x``.

    Evaluates the fixed 1-D eigenfunctions, which are independent of the
    length-scale :math:`\\rho`, at every observation.

    Parameters
    ----------
    x : Float[Array, ' g']
        Covariate values at the ``g`` evaluation points.
    sqrt_lambda : Float[Array, ' m']
        Square-root eigenvalues :math:`\\sqrt{\\lambda_j}`, one per basis mode.
    phase : Float[Array, ' m']
        Per-mode phase :math:`\\mathrm{phase}_j`.
    inv_sqrt_L : float
        Common eigenfunction amplitude :math:`\\sqrt{1/L}`.

    Returns
    -------
    Float[Array, 'g m']
        Design matrix :math:`\\Phi` with entries
        :math:`\\Phi_{ij} = \\sqrt{1/L}\\,\\sin(\\sqrt{\\lambda_j}\\,x_i
        + \\mathrm{phase}_j)`.
    """
    return inv_sqrt_L * jnp.sin(
        sqrt_lambda[None, :] * x[:, None] + phase[None, :]
    )


def _penalty_diag(
    sqrt_lambda: Float[Array, ' m'],
    kernel: str,
    rho: Float[Array, ''],
    n_fixed: int,
) -> Tuple[Float[Array, ' p'], Float[Array, '']]:
    """Diagonal penalty core and its smooth-block log-pseudo-determinant.

    Builds the diagonal penalty core :math:`d` over the full
    :math:`p = \\mathtt{n\\_fixed} + m` columns and returns the smooth-block
    log-pseudo-determinant contribution :math:`\\sum_j \\log d_j`. The full penalty
    is :math:`S_\\lambda = \\lambda\\,\\operatorname{diag}(d)` with
    :math:`d = [0, \\ldots, 0,\\, 1/s_1, \\ldots, 1/s_m]` -- zeros on the
    unpenalised fixed-effect columns -- where
    :math:`s_j = S_\\rho(\\sqrt{\\lambda_j})` is the kernel spectral density (the
    amplitude is folded into the scale :math:`\\lambda`). A small floor guards the
    reciprocal against numerically vanishing densities.

    Parameters
    ----------
    sqrt_lambda : Float[Array, ' m']
        Square-root eigenvalues :math:`\\sqrt{\\lambda_j}` of the ``m`` basis modes.
    kernel : str
        Kernel name selecting the spectral density :math:`S_\\rho`.
    rho : Float[Array, '']
        Kernel length-scale :math:`\\rho`.
    n_fixed : int
        Number of leading unpenalised fixed-effect columns.

    Returns
    -------
    d : Float[Array, ' p']
        Diagonal penalty core over all :math:`p = \\mathtt{n\\_fixed} + m` columns,
        zero on the fixed block and :math:`1/s_j` on the smooth block.
    logdet : Float[Array, '']
        Smooth-block contribution :math:`\\sum_j \\log(1/s_j)` to the
        log-pseudo-determinant.
    """
    s = spectral_density(sqrt_lambda, kernel=kernel, rho=rho, amplitude=1.0)
    inv_s = 1.0 / jnp.clip(s, 1e-30, None)
    d = jnp.concatenate([jnp.zeros((n_fixed,), dtype=inv_s.dtype), inv_s])
    return d, jnp.sum(jnp.log(inv_s))


def _hsgp_eigen_nd(
    x_np: np.ndarray, m_per: Tuple[int, ...], boundary: float, dtype: Any
) -> Tuple[Array, Array, Array, Array, Tuple[Tuple[float, float], ...]]:
    """Tensor-product Laplace eigenstructure on the box domain (host build).

    Constructs, on the host in double precision, the multi-dimensional
    Laplace--Dirichlet eigenstructure as the tensor product of the per-axis 1-D
    bases. The per-axis box is centred and padded exactly as in the 1-D case, and
    the total number of modes is :math:`M = \\prod_d \\mathtt{m\\_per}[d]`.

    Parameters
    ----------
    x_np : numpy.ndarray
        Covariate array of shape ``(n, D)`` whose per-axis minima and maxima define
        the box domain.
    m_per : Tuple[int, ...]
        Number of basis functions per axis; its length is the input dimension
        :math:`D`.
    boundary : float
        Multiplicative padding factor applied to each axis half-range.
    dtype : Any
        Floating dtype of the returned arrays.

    Returns
    -------
    freqs : Array
        Per-mode, per-axis eigen-frequencies :math:`w_{m,d}`, shape ``(M, D)``.
    phase : Array
        Per-mode, per-axis phases, shape ``(M, D)``.
    inv_sqrt_L : Array
        Per-axis amplitude :math:`\\sqrt{1/L_d}`, shape ``(D,)``.
    omega_norm : Array
        Mode-frequency magnitude
        :math:`\\lVert w \\rVert = \\sqrt{\\sum_d \\lambda_{j_d}}`, shape ``(M,)``.
    bounds : Tuple[Tuple[float, float], ...]
        Per-axis observed ``(lo, hi)`` extents, recorded for re-evaluation.
    """
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
    """Tensor-product eigenfunction design matrix :math:`\\Phi` at ``X``.

    Evaluates the multi-dimensional eigenfunctions, which are independent of the
    length-scale, as the per-axis product
    :math:`\\prod_d \\sqrt{1/L_d}\\,\\sin(w_{m,d}\\,x_d + \\mathrm{phase}_{m,d})`.

    Parameters
    ----------
    X : Float[Array, 'g D']
        Covariate matrix of the ``g`` evaluation points across ``D`` axes.
    freqs : Float[Array, 'M D']
        Per-mode, per-axis eigen-frequencies :math:`w_{m,d}`.
    phase : Float[Array, 'M D']
        Per-mode, per-axis phases :math:`\\mathrm{phase}_{m,d}`.
    inv_sqrt_L : Float[Array, ' D']
        Per-axis amplitude :math:`\\sqrt{1/L_d}`.

    Returns
    -------
    Float[Array, 'g M']
        Design matrix :math:`\\Phi` whose ``(i, m)`` entry is the product over
        axes of the per-axis eigenfunctions evaluated at observation :math:`i`.
    """
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
    """Isotropic tensor-HSGP diagonal penalty.

    Reweights each mode by the reciprocal
    :math:`1 / S_D(\\lVert w \\rVert;\\, \\rho)` of the :math:`D`-dimensional radial
    spectral density evaluated at the mode-frequency magnitude, placing zeros on the
    leading fixed-effect columns. A small floor guards the reciprocal against
    numerically vanishing densities.

    Parameters
    ----------
    omega_norm : Float[Array, ' M']
        Mode-frequency magnitude :math:`\\lVert w \\rVert` for each of the ``M``
        modes.
    kernel : str
        Kernel name selecting the spectral density.
    rho : float
        Isotropic kernel length-scale :math:`\\rho`.
    dim : int
        Dimension :math:`D` of the radial spectral density.
    n_fixed : int
        Number of leading unpenalised fixed-effect columns.

    Returns
    -------
    d : Float[Array, ' p']
        Diagonal penalty core over all :math:`p = \\mathtt{n\\_fixed} + M` columns,
        zero on the fixed block and :math:`1/S_D(\\lVert w \\rVert;\\, \\rho)` on the
        smooth block.
    logdet : Float[Array, '']
        Smooth-block contribution to the log-pseudo-determinant,
        :math:`\\sum_m \\log(1/s_m)`.
    """
    s = spectral_density(
        omega_norm,
        kernel=kernel,
        rho=jnp.asarray(rho, dtype=omega_norm.dtype),
        amplitude=1.0,
        dim=dim,
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
    """Separable / ARD tensor-HSGP diagonal penalty.

    Reweights each mode by the reciprocal of the separable spectral density
    :math:`1 / \\prod_d S_1(w_{m,d};\\, \\rho_d)`, the product of per-axis 1-D
    densities evaluated at the per-axis frequency, with an independent length-scale
    :math:`\\rho_d` on each axis (automatic relevance determination). Zeros are
    placed on the leading fixed-effect columns, and a small floor guards the
    reciprocal against numerically vanishing densities.

    Parameters
    ----------
    freqs : Float[Array, 'M D']
        Per-mode, per-axis eigen-frequencies :math:`w_{m,d}`.
    kernel : str
        Kernel name selecting the per-axis 1-D spectral density.
    rho_vec : Tuple[float, ...]
        Per-axis length-scales :math:`\\rho_d`, one entry per axis.
    n_fixed : int
        Number of leading unpenalised fixed-effect columns.

    Returns
    -------
    d_full : Float[Array, ' p']
        Diagonal penalty core over all :math:`p = \\mathtt{n\\_fixed} + M` columns,
        zero on the fixed block and
        :math:`1 / \\prod_d S_1(w_{m,d};\\, \\rho_d)` on the smooth block.
    logdet : Float[Array, '']
        Smooth-block contribution to the log-pseudo-determinant,
        :math:`\\sum_m \\log(1/s_m)`.
    """
    dtype = freqs.dtype
    s = jnp.ones((freqs.shape[0],), dtype=dtype)
    for d in range(freqs.shape[1]):
        s = s * spectral_density(
            freqs[:, d],
            kernel=kernel,
            rho=jnp.asarray(rho_vec[d], dtype=dtype),
            amplitude=1.0,
            dim=1,
        )
    inv_s = 1.0 / jnp.clip(s, 1e-30, None)
    d_full = jnp.concatenate([jnp.zeros((n_fixed,), dtype=dtype), inv_s])
    return d_full, jnp.sum(jnp.log(inv_s))
