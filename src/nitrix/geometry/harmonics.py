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
are assembled once (host-side) by :func:`sht_plan` and reused across transforms;
the data path -- :func:`sht_forward` / :func:`sht_inverse` given that plan -- is
a pure FFT + contraction that runs under :func:`jax.jit`.

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

Coefficients are rotated in place by a 3-D rotation via the Wigner-D matrices
(:func:`sht_rotation_matrix` / :func:`sht_rotate`), computed as a matrix
exponential of the angular-momentum generator.

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
from jaxtyping import Array, Complex, Float, Int

from ..linalg import matrix_exp

Grid = Literal['gauss', 'driscoll_healy']

__all__ = [
    'SHTGrid',
    'SHTPlan',
    'real_sht_forward',
    'real_sht_inverse',
    'sht_forward',
    'sht_grid',
    'sht_inverse',
    'sht_plan',
    'sht_rotate',
    'sht_rotation_matrix',
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


class SHTPlan(NamedTuple):
    r"""The fixed transform plan for a band-limit and grid (the fit state).

    Everything the forward / inverse transforms need that depends only on the
    (static) band-limit and grid choice -- the sampling nodes, the quadrature
    weights, the fully-normalised associated Legendre table and the FFT-order
    map -- assembled once by :func:`sht_plan`. Passing it to :func:`sht_forward`
    / :func:`sht_inverse` runs the transform under :func:`jax.jit` without
    rebuilding the :math:`O(L^2)` Legendre table. A superset of
    :class:`SHTGrid`: it also carries the sampling grid a field is evaluated on.

    Attributes
    ----------
    colatitude : Float[Array, 'n_lat']
        The colatitudes :math:`\theta \in [0, \pi]` (as :class:`SHTGrid`).
    longitude : Float[Array, 'n_lon']
        The equiangular longitudes :math:`\phi \in [0, 2\pi)`; its length is the
        FFT width ``n_lon``.
    weight : Float[Array, 'n_lat']
        The latitude-quadrature weights.
    legendre : Float[Array, 'l_dim m_dim n_lat']
        The fully-normalised associated Legendre table
        :math:`\bar P_\ell^m(\cos\theta_j)`, shape ``(L+1, 2L+1, n_lat)``.
    m_to_fft : Int[Array, 'm_dim']
        The index of order :math:`m` in the FFT spectrum, shape ``(2L+1,)``.
    """

    colatitude: Float[Array, 'n_lat']
    longitude: Float[Array, 'n_lon']
    weight: Float[Array, 'n_lat']
    legendre: Float[Array, 'l_dim m_dim n_lat']
    m_to_fft: Int[Array, 'm_dim']


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


def _nodes(band_limit: int, grid: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Colatitude cosine-nodes, quadrature weights and longitude count."""
    if grid == 'gauss':
        x, w = np.polynomial.legendre.leggauss(band_limit + 1)
        return x, w, 2 * band_limit + 1
    if grid == 'driscoll_healy':
        return _dh_nodes(band_limit)
    raise ValueError(f"grid must be 'gauss'/'driscoll_healy'; got {grid!r}.")


def sht_plan(band_limit: int, *, grid: Grid = 'gauss') -> SHTPlan:
    r"""Build the transform plan (the fit state) for a band-limit and grid.

    The construction half of the transform's fit/apply pair: precomputes the
    nodes, quadrature weights, Legendre table and FFT-order map once
    (host-side) so that :func:`sht_forward` / :func:`sht_inverse` -- which take
    this plan -- run under :func:`jax.jit` and reuse it across many fields.

    Parameters
    ----------
    band_limit : int
        Maximum spherical-harmonic degree :math:`L`.
    grid : {'gauss', 'driscoll_healy'}, optional
        The sampling grid (see :func:`sht_grid`). Default ``'gauss'``.

    Returns
    -------
    SHTPlan
        The reusable transform plan.
    """
    x, w, n_lon = _nodes(band_limit, grid)
    table = _fnalp(x, band_limit)
    orders = np.arange(-band_limit, band_limit + 1)
    m_to_fft = np.where(orders >= 0, orders, orders + n_lon)
    lon = 2.0 * np.pi * np.arange(n_lon) / n_lon
    return SHTPlan(
        colatitude=jnp.asarray(np.arccos(x)),
        longitude=jnp.asarray(lon),
        weight=jnp.asarray(w),
        legendre=jnp.asarray(table),
        m_to_fft=jnp.asarray(m_to_fft),
    )


def sht_grid(band_limit: int, *, grid: Grid = 'gauss') -> SHTGrid:
    r"""The sampling grid for a given band-limit.

    A field must be sampled on this grid (colatitude :math:`\times` longitude,
    the outer product) before :func:`sht_forward`.

    The nodes depend only on the static ``band_limit`` (and ``grid``), so this
    is a cheap host-side constructor; for the transforms themselves use
    :func:`sht_plan`, whose :class:`SHTPlan` carries this same grid plus the
    quadrature plan.

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
    x, _, n_lon = _nodes(band_limit, grid)
    lon = 2.0 * np.pi * np.arange(n_lon) / n_lon
    return SHTGrid(
        colatitude=jnp.asarray(np.arccos(x)), longitude=jnp.asarray(lon)
    )


def sht_forward(
    f: Float[Array, '... n_lat n_lon'],
    plan: SHTPlan,
) -> Complex[Array, '... l_dim m_dim']:
    r"""Forward spherical harmonic transform (analysis).

    Projects a field sampled on the plan's grid (:func:`sht_grid` /
    :func:`sht_plan`) onto the orthonormal spherical harmonics. The apply half
    of the transform's fit/apply pair: as ``plan`` is a plain-array pytree, this
    call is fully jittable and differentiable with respect to ``f``.

    Parameters
    ----------
    f : Float[Array, '... n_lat n_lon']
        The field on the plan's grid (``n_lat`` colatitudes by ``n_lon``
        longitudes), batching over leading dimensions. Real or complex.
    plan : SHTPlan
        The transform plan from :func:`sht_plan`; it sets the band-limit
        :math:`L` and the sampling grid.

    Returns
    -------
    Complex[Array, '... l_dim m_dim']
        Coefficients ``(..., L+1, 2L+1)``, indexed ``[..., ell, m + L]``; zero
        where :math:`|m| > \ell`.
    """
    n_lon = plan.longitude.shape[0]
    f_hat = jnp.fft.fft(f, axis=-1)  # (..., n_lat, n_lon)
    f_hat_m = f_hat[..., plan.m_to_fft]  # reorder to m in [-L, L]
    dphi = 2.0 * np.pi / n_lon
    coeffs = dphi * jnp.einsum(
        'j,lmj,...jm->...lm', plan.weight, plan.legendre, f_hat_m
    )
    return coeffs


def sht_inverse(
    coeffs: Complex[Array, '... l_dim m_dim'],
    plan: SHTPlan,
) -> Complex[Array, '... n_lat n_lon']:
    r"""Inverse spherical harmonic transform (synthesis).

    Reconstructs the field on the plan's grid from its coefficients. The apply
    half of the transform's fit/apply pair; jittable and differentiable with
    respect to ``coeffs``.

    Parameters
    ----------
    coeffs : Complex[Array, '... l_dim m_dim']
        Coefficients ``(..., L+1, 2L+1)`` as returned by :func:`sht_forward`.
    plan : SHTPlan
        The transform plan from :func:`sht_plan` (its band-limit must match
        ``coeffs``).

    Returns
    -------
    Complex[Array, '... n_lat n_lon']
        The field on the sampling grid. Real (up to rounding) when the
        coefficients carry the Hermitian symmetry
        :math:`f_{\ell,-m} = (-1)^m \overline{f_{\ell m}}` of a real field.
    """
    n_lon = plan.longitude.shape[0]
    g_m = jnp.einsum('...lm,lmj->...jm', coeffs, plan.legendre)
    spectrum = (
        jnp.zeros(g_m.shape[:-1] + (n_lon,), dtype=g_m.dtype)
        .at[..., plan.m_to_fft]
        .set(g_m)
    )
    field = n_lon * jnp.fft.ifft(spectrum, axis=-1)
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
    plan: SHTPlan,
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
        The real field on the plan's grid.
    plan : SHTPlan
        The transform plan from :func:`sht_plan`.

    Returns
    -------
    Float[Array, '... l_dim m_dim']
        Real coefficients ``(..., L+1, 2L+1)``.
    """
    return _complex_to_real(sht_forward(f, plan))


def real_sht_inverse(
    coeffs: Float[Array, '... l_dim m_dim'],
    plan: SHTPlan,
) -> Float[Array, '... n_lat n_lon']:
    r"""Inverse transform from **real** spherical-harmonic coefficients.

    The real-valued analogue of :func:`sht_inverse`; reconstructs the real field
    on the plan's grid from real coefficients as returned by
    :func:`real_sht_forward`.

    Parameters
    ----------
    coeffs : Float[Array, '... l_dim m_dim']
        Real coefficients ``(..., L+1, 2L+1)``.
    plan : SHTPlan
        The transform plan from :func:`sht_plan`.

    Returns
    -------
    Float[Array, '... n_lat n_lon']
        The real field on the sampling grid.
    """
    return jnp.real(sht_inverse(_real_to_complex(coeffs), plan))


def _wigner_generator(degree: int) -> np.ndarray:
    """The real antisymmetric angular-momentum-y generator :math:`-i J_y`.

    A ``(2l+1, 2l+1)`` tridiagonal matrix (rows/cols index :math:`m = -l..l`)
    whose exponential :math:`\\exp(\\beta\\,A)` is the Wigner (small-)d matrix
    :math:`d^l(\\beta)`. Host-side constant of the degree only.
    """
    dim = 2 * degree + 1
    a = np.zeros((dim, dim))
    for m in range(-degree, degree):
        c = np.sqrt(degree * (degree + 1) - m * (m + 1)) / 2.0
        a[(m + 1) + degree, m + degree] = -c
        a[m + degree, (m + 1) + degree] = c
    return a


def sht_rotation_matrix(
    rotation: Float[Array, '3 3'],
    *,
    band_limit: int,
) -> Complex[Array, 'l_dim m_dim m_dim']:
    r"""Wigner-D matrices rotating spherical-harmonic coefficients.

    Builds the block-diagonal Wigner-D representation of a 3-D rotation ``R`` in
    the spherical-harmonic basis, up to degree ``band_limit``. Rotating a field
    by ``R`` mixes only the coefficients within each degree :math:`\ell`:
    :math:`c'_{\ell m} = \sum_{m'} D^\ell_{m m'}(R)\, c_{\ell m'}`
    (:func:`sht_rotate`).

    The Wigner-d part :math:`d^\ell(\beta)` is computed as the matrix exponential
    :math:`\exp(\beta A^\ell)` of the (real, antisymmetric) angular-momentum-y
    generator -- pure matmul (:func:`~nitrix.linalg.matrix_exp`), no recurrence --
    with the ZYZ Euler angles read off ``R``; the azimuthal phases
    :math:`e^{-i m \alpha}`, :math:`e^{-i m' \gamma}` complete
    :math:`D^\ell_{m m'} = e^{-i m \alpha}\, d^\ell_{m m'}(\beta)\,
    e^{-i m' \gamma}`. Differentiable in ``R``.

    Parameters
    ----------
    rotation : Float[Array, '3 3']
        A rotation matrix :math:`R \in \mathrm{SO}(3)`.
    band_limit : int
        Maximum degree :math:`L`.

    Returns
    -------
    Complex[Array, 'l_dim m_dim m_dim']
        The stacked blocks ``(L+1, 2L+1, 2L+1)``, indexed
        ``[ell, m + L, m' + L]`` and zero-padded outside the degree-``ell`` block.
    """
    band = band_limit
    beta = jnp.arccos(jnp.clip(rotation[2, 2], -1.0, 1.0))
    # Gimbal lock at beta ~ 0 / pi: the ZYZ split of alpha and gamma is
    # degenerate (only their combination is defined and the z-column carries no
    # azimuth), so recover the combined z-rotation from the top-left block.
    sin_beta = jnp.sqrt(jnp.clip(1.0 - rotation[2, 2] ** 2, 0.0, 1.0))
    generic = sin_beta > 1e-6
    alpha = jnp.where(
        generic,
        jnp.arctan2(rotation[1, 2], rotation[0, 2]),
        jnp.arctan2(rotation[1, 0], rotation[0, 0]),
    )
    gamma = jnp.where(
        generic, jnp.arctan2(rotation[2, 1], -rotation[2, 0]), 0.0
    )

    blocks = []
    for ell in range(band + 1):
        gen = jnp.asarray(_wigner_generator(ell), dtype=beta.dtype)
        d = matrix_exp(beta * gen)  # (2l+1, 2l+1) real Wigner-d
        m = jnp.arange(-ell, ell + 1)
        phase = jnp.exp(-1j * m * alpha)[:, None] * jnp.exp(-1j * m * gamma)
        block = d * phase
        padded = jnp.zeros((2 * band + 1, 2 * band + 1), dtype=block.dtype)
        padded = padded.at[
            band - ell : band + ell + 1, band - ell : band + ell + 1
        ].set(block)
        blocks.append(padded)
    return jnp.stack(blocks)


def sht_rotate(
    coeffs: Complex[Array, '... l_dim m_dim'],
    rotation: Float[Array, '3 3'],
) -> Complex[Array, '... l_dim m_dim']:
    r"""Rotate spherical-harmonic coefficients by a 3-D rotation.

    Applies the Wigner-D representation (:func:`sht_rotation_matrix`) of
    ``rotation`` to ``coeffs``, degree by degree
    (:math:`c'_{\ell m} = \sum_{m'} D^\ell_{m m'}\, c_{\ell m'}`) -- the exact
    coefficient-space rotation, equivalent to resampling the synthesised field on
    the rotated sphere but without any interpolation loss.

    Parameters
    ----------
    coeffs : Complex[Array, '... l_dim m_dim']
        Coefficients ``(..., L+1, 2L+1)`` (as from :func:`sht_forward`).
    rotation : Float[Array, '3 3']
        A rotation matrix :math:`R \in \mathrm{SO}(3)`.

    Returns
    -------
    Complex[Array, '... l_dim m_dim']
        The rotated coefficients, same shape as ``coeffs``.
    """
    band_limit = coeffs.shape[-2] - 1
    wigner = sht_rotation_matrix(rotation, band_limit=band_limit)
    return jnp.einsum('lmk,...lk->...lm', wigner, coeffs)
