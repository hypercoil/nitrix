# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Spherical harmonic transform (analysis / synthesis).

The forward and inverse transforms between a scalar field on the 2-sphere and
its complex spherical-harmonic coefficients :math:`f_{\ell m}`,

.. math::

    f(\theta, \phi) = \sum_{\ell=0}^{L} \sum_{m=-\ell}^{\ell} f_{\ell m}\,
    Y_\ell^m(\theta, \phi), \qquad
    f_{\ell m} = \int_{S^2} f\, \overline{Y_\ell^m}\, d\Omega,

with the orthonormal, Condon--Shortley :math:`Y_\ell^m` (so
:math:`\int_{S^2} |Y_\ell^m|^2 d\Omega = 1`). The field is sampled on a
**Gauss--Legendre** grid: :math:`L+1` Gauss--Legendre colatitudes (exact for the
degree-:math:`2L` products the analysis integrand reaches) by :math:`2L+1`
equiangular longitudes. On that grid the transform is *exact* for any field
band-limited to degree :math:`L` -- ``sht_inverse(sht_forward(f)) == f`` to
rounding.

The longitude integral is a fast Fourier transform; the colatitude integral is a
matmul against the fully-normalised associated Legendre functions
:math:`\bar P_\ell^m` evaluated at the Gauss--Legendre nodes. Those nodes,
weights and the Legendre table depend only on the (static) band-limit, so they
are precomputed once (host-side) as a fixed plan and the data path is a pure
FFT + contraction.

Coefficient layout: coefficients are returned as ``(..., L+1, 2L+1)``, indexed
``[..., ell, m + L]`` for :math:`m \in [-\ell, \ell]`; entries with
:math:`|m| > \ell` are zero. :func:`sht_grid` returns the sampling
``(colatitude, longitude)``.

Notes
-----
The Wigner-D rotation of coefficients (``sht_rotation_matrix``) is a separate
follow-up; this module ships the analysis / synthesis pair and the grid.

References
----------
Driscoll JR, Healy DM (1994). Computing Fourier transforms and convolutions on
the 2-sphere. *Advances in Applied Mathematics*, 15(2), 202-250.
https://doi.org/10.1006/aama.1994.1008
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float

__all__ = ['SHTGrid', 'sht_forward', 'sht_grid', 'sht_inverse']


class SHTGrid(NamedTuple):
    """The Gauss--Legendre sampling grid a field must be evaluated on.

    Attributes
    ----------
    colatitude : Float[Array, 'n_lat']
        The ``L + 1`` Gauss--Legendre colatitudes :math:`\\theta \\in [0, \\pi]`
        (polar angle), ascending.
    longitude : Float[Array, 'n_lon']
        The ``2L + 1`` equiangular longitudes :math:`\\phi \\in [0, 2\\pi)`.
    """

    colatitude: Float[Array, 'n_lat']
    longitude: Float[Array, 'n_lon']


def _fnalp(x: np.ndarray, band_limit: int) -> np.ndarray:
    """Fully-normalised associated Legendre table (Condon--Shortley).

    Returns ``P[ell, m + L, j] = \\bar P_ell^m(x_j)`` for ``m in [-L, L]`` (zero
    where ``|m| > ell``), matching ``scipy.special.sph_harm_y(ell, m, ., 0)``.
    Host-side (numpy): the nodes ``x`` and the band-limit are static.
    """
    length = band_limit + 1
    s = np.sqrt(1.0 - x * x)
    pos = np.zeros((length, length, x.shape[0]))
    pos[0, 0] = 1.0 / np.sqrt(4.0 * np.pi)
    for m in range(1, length):
        pos[m, m] = -np.sqrt((2 * m + 1) / (2 * m)) * s * pos[m - 1, m - 1]
    for m in range(band_limit):
        pos[m + 1, m] = np.sqrt(2 * m + 3) * x * pos[m, m]
    for m in range(length):
        for ell in range(m + 2, length):
            a = np.sqrt(
                (2 * ell + 1) * (2 * ell - 1) / ((ell - m) * (ell + m))
            )
            b = np.sqrt(
                (2 * ell + 1)
                * (ell - 1 - m)
                * (ell - 1 + m)
                / ((2 * ell - 3) * (ell - m) * (ell + m))
            )
            pos[ell, m] = a * x * pos[ell - 1, m] - b * pos[ell - 2, m]

    table = np.zeros((length, 2 * band_limit + 1, x.shape[0]))
    for ell in range(length):
        for m in range(-ell, ell + 1):
            table[ell, m + band_limit] = (
                (-1) ** m * pos[ell, -m] if m < 0 else pos[ell, m]
            )
    return table


class _Plan(NamedTuple):
    weight: Array  # (n_lat,) Gauss-Legendre weights
    legendre: Array  # (L+1, 2L+1, n_lat) associated Legendre table
    m_to_fft: Array  # (2L+1,) index of order m in the FFT spectrum
    n_lon: int


def _plan(band_limit: int) -> _Plan:
    """Precompute the Gauss--Legendre nodes, weights and Legendre table."""
    x, w = np.polynomial.legendre.leggauss(band_limit + 1)
    n_lon = 2 * band_limit + 1
    table = _fnalp(x, band_limit)
    orders = np.arange(-band_limit, band_limit + 1)
    m_to_fft = np.where(orders >= 0, orders, orders + n_lon)
    return _Plan(
        weight=jnp.asarray(w),
        legendre=jnp.asarray(table),
        m_to_fft=jnp.asarray(m_to_fft),
        n_lon=n_lon,
    )


def sht_grid(band_limit: int) -> SHTGrid:
    r"""The Gauss--Legendre sampling grid for a given band-limit.

    A field must be sampled on this grid (colatitude :math:`\times` longitude,
    the outer product) before :func:`sht_forward`.

    Parameters
    ----------
    band_limit : int
        Maximum spherical-harmonic degree :math:`L`.

    Returns
    -------
    SHTGrid
        The ``L + 1`` colatitudes and ``2L + 1`` longitudes.
    """
    x, _ = np.polynomial.legendre.leggauss(band_limit + 1)
    colat = np.arccos(x)
    lon = 2.0 * np.pi * np.arange(2 * band_limit + 1) / (2 * band_limit + 1)
    return SHTGrid(colatitude=jnp.asarray(colat), longitude=jnp.asarray(lon))


def sht_forward(
    f: Float[Array, '... n_lat n_lon'],
    *,
    band_limit: int,
) -> Complex[Array, '... l_dim m_dim']:
    r"""Forward spherical harmonic transform (analysis).

    Projects a field sampled on the :func:`sht_grid` onto the orthonormal
    spherical harmonics up to degree ``band_limit``.

    Parameters
    ----------
    f : Float[Array, '... n_lat n_lon']
        The field on the Gauss--Legendre grid (``L + 1`` colatitudes by
        ``2L + 1`` longitudes), batching over leading dimensions. Real or
        complex.
    band_limit : int
        Maximum degree :math:`L` (a static argument -- it sets the grid).

    Returns
    -------
    Complex[Array, '... l_dim m_dim']
        Coefficients ``(..., L+1, 2L+1)``, indexed ``[..., ell, m + L]``; zero
        where :math:`|m| > \ell`.
    """
    plan = _plan(band_limit)
    f_hat = jnp.fft.fft(f, axis=-1)  # (..., n_lat, n_lon)
    f_hat_m = f_hat[..., plan.m_to_fft]  # reorder to m in [-L, L]
    dphi = 2.0 * np.pi / plan.n_lon
    coeffs = dphi * jnp.einsum(
        'j,lmj,...jm->...lm', plan.weight, plan.legendre, f_hat_m
    )
    return coeffs


def sht_inverse(
    coeffs: Complex[Array, '... l_dim m_dim'],
) -> Complex[Array, '... n_lat n_lon']:
    r"""Inverse spherical harmonic transform (synthesis).

    Reconstructs the field on the :func:`sht_grid` from its coefficients. The
    band-limit :math:`L` is inferred from the coefficient shape
    (``coeffs.shape[-2] == L + 1``).

    Parameters
    ----------
    coeffs : Complex[Array, '... l_dim m_dim']
        Coefficients ``(..., L+1, 2L+1)`` as returned by :func:`sht_forward`.

    Returns
    -------
    Complex[Array, '... n_lat n_lon']
        The field on the Gauss--Legendre grid. Real (up to rounding) when the
        coefficients carry the Hermitian symmetry
        :math:`f_{\ell,-m} = (-1)^m \overline{f_{\ell m}}` of a real field.
    """
    band_limit = coeffs.shape[-2] - 1
    plan = _plan(band_limit)
    g_m = jnp.einsum('...lm,lmj->...jm', coeffs, plan.legendre)
    spectrum = (
        jnp.zeros(g_m.shape[:-1] + (plan.n_lon,), dtype=g_m.dtype)
        .at[..., plan.m_to_fft]
        .set(g_m)
    )
    field = plan.n_lon * jnp.fft.ifft(spectrum, axis=-1)
    return field
