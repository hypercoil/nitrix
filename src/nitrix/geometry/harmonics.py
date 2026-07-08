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

Both the complex (:func:`sht_forward` / :func:`sht_inverse`) and the **real**
(:func:`real_sht_forward` / :func:`real_sht_inverse`, the redundancy-free real-SH
basis used for e.g. dMRI fibre orientation) transforms are provided, on either
the Gauss--Legendre grid (default) or the **Driscoll--Healy** equiangular grid
(``grid='driscoll_healy'``, the uniform sampling SH-equivariant spherical CNNs
use).

Notes
-----
The Wigner-D rotation of coefficients (``sht_rotation_matrix``) is a separate
follow-up; this module ships the complex + real analysis / synthesis pairs and
the two grids.

References
----------
Driscoll JR, Healy DM (1994). Computing Fourier transforms and convolutions on
the 2-sphere. *Advances in Applied Mathematics*, 15(2), 202-250.
https://doi.org/10.1006/aama.1994.1008
"""

from __future__ import annotations

from typing import Literal, NamedTuple, Tuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float

Grid = Literal['gauss', 'driscoll_healy']

__all__ = [
    'SHTGrid',
    'real_sht_forward',
    'real_sht_inverse',
    'sht_forward',
    'sht_grid',
    'sht_inverse',
]


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


def _dh_nodes(band_limit: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Driscoll--Healy equiangular colatitude nodes, weights and longitude count.

    The ``2(L+1)`` pole-avoiding equiangular colatitudes with the exact
    Driscoll--Healy latitude-quadrature weights (Driscoll & Healy 1994, eqn 6):
    :math:`w_j = \\tfrac{2}{B}\\sin\\theta_j \\sum_{k=0}^{B-1}
    \\frac{\\sin((2k+1)\\theta_j)}{2k+1}`, :math:`B = L + 1`.
    """
    b = band_limit + 1
    n_lat = 2 * b
    theta = np.pi * (2 * np.arange(n_lat) + 1) / (4 * b)
    k = 2 * np.arange(b) + 1
    weight = (
        (2.0 / b)
        * np.sin(theta)
        * (np.sin(theta[:, None] * k[None, :]) / k[None, :]).sum(axis=1)
    )
    return np.cos(theta), weight, 2 * b


def _plan(band_limit: int, grid: str) -> _Plan:
    """Precompute the sampling nodes, weights and Legendre table for a grid."""
    if grid == 'gauss':
        x, w = np.polynomial.legendre.leggauss(band_limit + 1)
        n_lon = 2 * band_limit + 1
    elif grid == 'driscoll_healy':
        x, w, n_lon = _dh_nodes(band_limit)
    else:
        raise ValueError(
            f"grid must be 'gauss'/'driscoll_healy'; got {grid!r}."
        )
    table = _fnalp(x, band_limit)
    orders = np.arange(-band_limit, band_limit + 1)
    m_to_fft = np.where(orders >= 0, orders, orders + n_lon)
    return _Plan(
        weight=jnp.asarray(w),
        legendre=jnp.asarray(table),
        m_to_fft=jnp.asarray(m_to_fft),
        n_lon=n_lon,
    )


def sht_grid(band_limit: int, *, grid: Grid = 'gauss') -> SHTGrid:
    r"""The sampling grid for a given band-limit.

    A field must be sampled on this grid (colatitude :math:`\times` longitude,
    the outer product) before :func:`sht_forward`.

    Parameters
    ----------
    band_limit : int
        Maximum spherical-harmonic degree :math:`L`.
    grid : {'gauss', 'driscoll_healy'}, optional
        ``'gauss'`` (default) uses ``L+1`` Gauss--Legendre colatitudes (fewest
        points, non-uniform); ``'driscoll_healy'`` uses ``2(L+1)`` equiangular
        (uniform) colatitudes -- the grid SH-equivariant spherical CNNs sample on.

    Returns
    -------
    SHTGrid
        The colatitudes and longitudes (``L+1`` by ``2L+1`` for ``'gauss'``,
        ``2(L+1)`` by ``2(L+1)`` for ``'driscoll_healy'``).
    """
    if grid == 'gauss':
        x, _ = np.polynomial.legendre.leggauss(band_limit + 1)
        n_lon = 2 * band_limit + 1
    elif grid == 'driscoll_healy':
        x, _, n_lon = _dh_nodes(band_limit)
    else:
        raise ValueError(
            f"grid must be 'gauss'/'driscoll_healy'; got {grid!r}."
        )
    lon = 2.0 * np.pi * np.arange(n_lon) / n_lon
    return SHTGrid(
        colatitude=jnp.asarray(np.arccos(x)), longitude=jnp.asarray(lon)
    )


def sht_forward(
    f: Float[Array, '... n_lat n_lon'],
    *,
    band_limit: int,
    grid: Grid = 'gauss',
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
    grid : {'gauss', 'driscoll_healy'}, optional
        The sampling grid ``f`` lives on (must match :func:`sht_grid`). Default
        ``'gauss'``.

    Returns
    -------
    Complex[Array, '... l_dim m_dim']
        Coefficients ``(..., L+1, 2L+1)``, indexed ``[..., ell, m + L]``; zero
        where :math:`|m| > \ell`.
    """
    plan = _plan(band_limit, grid)
    f_hat = jnp.fft.fft(f, axis=-1)  # (..., n_lat, n_lon)
    f_hat_m = f_hat[..., plan.m_to_fft]  # reorder to m in [-L, L]
    dphi = 2.0 * np.pi / plan.n_lon
    coeffs = dphi * jnp.einsum(
        'j,lmj,...jm->...lm', plan.weight, plan.legendre, f_hat_m
    )
    return coeffs


def sht_inverse(
    coeffs: Complex[Array, '... l_dim m_dim'],
    *,
    grid: Grid = 'gauss',
) -> Complex[Array, '... n_lat n_lon']:
    r"""Inverse spherical harmonic transform (synthesis).

    Reconstructs the field on the :func:`sht_grid` from its coefficients. The
    band-limit :math:`L` is inferred from the coefficient shape
    (``coeffs.shape[-2] == L + 1``).

    Parameters
    ----------
    coeffs : Complex[Array, '... l_dim m_dim']
        Coefficients ``(..., L+1, 2L+1)`` as returned by :func:`sht_forward`.
    grid : {'gauss', 'driscoll_healy'}, optional
        The sampling grid to synthesise on. Default ``'gauss'``.

    Returns
    -------
    Complex[Array, '... n_lat n_lon']
        The field on the sampling grid. Real (up to rounding) when the
        coefficients carry the Hermitian symmetry
        :math:`f_{\ell,-m} = (-1)^m \overline{f_{\ell m}}` of a real field.
    """
    band_limit = coeffs.shape[-2] - 1
    plan = _plan(band_limit, grid)
    g_m = jnp.einsum('...lm,lmj->...jm', coeffs, plan.legendre)
    spectrum = (
        jnp.zeros(g_m.shape[:-1] + (plan.n_lon,), dtype=g_m.dtype)
        .at[..., plan.m_to_fft]
        .set(g_m)
    )
    field = plan.n_lon * jnp.fft.ifft(spectrum, axis=-1)
    return field


def _complex_to_real(
    c: Complex[Array, '... l_dim m_dim'],
) -> Float[Array, '... l_dim m_dim']:
    """Recombine complex SH coefficients into the real-SH basis."""
    band_limit = c.shape[-2] - 1
    sqrt2 = jnp.sqrt(jnp.asarray(2.0, c.real.dtype))
    sign = (-1.0) ** jnp.arange(1, band_limit + 1)
    positive = c[..., band_limit + 1 :]  # m = 1..L
    cos_part = sqrt2 * sign * positive.real
    sin_part = (-sqrt2 * sign * positive.imag)[..., ::-1]  # m = -1..-L
    m0 = c[..., band_limit : band_limit + 1].real
    return jnp.concatenate([sin_part, m0, cos_part], axis=-1)


def _real_to_complex(
    r: Float[Array, '... l_dim m_dim'],
) -> Complex[Array, '... l_dim m_dim']:
    """Recombine real-SH coefficients back into the complex SH basis."""
    band_limit = r.shape[-2] - 1
    sqrt2 = jnp.sqrt(jnp.asarray(2.0, r.dtype))
    sign = (-1.0) ** jnp.arange(1, band_limit + 1)
    r_cos = r[..., band_limit + 1 :]  # cos, m = 1..L
    r_sin = r[..., :band_limit][..., ::-1]  # sin, aligned to m = 1..L
    c_pos = sign * (r_cos - 1j * r_sin) / sqrt2
    c_neg = ((r_cos + 1j * r_sin) / sqrt2)[..., ::-1]  # m = -L..-1
    c0 = r[..., band_limit : band_limit + 1].astype(c_pos.dtype)
    return jnp.concatenate([c_neg, c0, c_pos], axis=-1)


def real_sht_forward(
    f: Float[Array, '... n_lat n_lon'],
    *,
    band_limit: int,
    grid: Grid = 'gauss',
) -> Float[Array, '... l_dim m_dim']:
    r"""Forward transform onto the **real** spherical harmonics.

    The real-valued analogue of :func:`sht_forward` for a real field: projects
    onto the orthonormal real spherical harmonics (the cosine harmonics at
    :math:`m > 0`, the sine harmonics at :math:`m < 0`, indexed
    ``[..., ell, m + L]``). Computed by recombining the complex transform, so it
    inherits its exactness; the coefficients are real and carry no redundancy
    (a real field's complex coefficients are Hermitian-symmetric).

    Parameters
    ----------
    f : Float[Array, '... n_lat n_lon']
        The real field on the :func:`sht_grid`.
    band_limit : int
        Maximum degree :math:`L`.

    Returns
    -------
    Float[Array, '... l_dim m_dim']
        Real coefficients ``(..., L+1, 2L+1)``.
    """
    return _complex_to_real(sht_forward(f, band_limit=band_limit, grid=grid))


def real_sht_inverse(
    coeffs: Float[Array, '... l_dim m_dim'],
    *,
    grid: Grid = 'gauss',
) -> Float[Array, '... n_lat n_lon']:
    r"""Inverse transform from **real** spherical-harmonic coefficients.

    The real-valued analogue of :func:`sht_inverse`; reconstructs the real field
    on the :func:`sht_grid` from real coefficients as returned by
    :func:`real_sht_forward`.

    Parameters
    ----------
    coeffs : Float[Array, '... l_dim m_dim']
        Real coefficients ``(..., L+1, 2L+1)``.

    Returns
    -------
    Float[Array, '... n_lat n_lon']
        The real field on the sampling grid.
    """
    return jnp.real(sht_inverse(_real_to_complex(coeffs), grid=grid))
