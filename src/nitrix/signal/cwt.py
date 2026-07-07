# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Continuous wavelet transform.

Time-frequency analysis by a bank of scaled, translated mother wavelets: the
continuous wavelet transform :math:`W(s, b) = \int x(t)\, \tfrac{1}{\sqrt s}\,
\psi^{*}\!\big(\tfrac{t-b}{s}\big)\, dt` evaluated at every time :math:`b` and a
set of scales :math:`s`. Computed in the Fourier domain -- for each scale,
multiply the signal spectrum by the (conjugated) transform of the scaled wavelet
and invert -- so the whole bank is a handful of FFTs (the standard
Torrence--Compo construction), reusing the module's FFT machinery rather than a
time-domain convolution per scale.

Three mother wavelets, in the Fourier-domain normalisation of Torrence & Compo
(1998): the complex analytic **Morlet** (a modulated Gaussian, the default;
sharp frequency localisation), the real **Ricker** / Mexican-hat (the second
derivative of a Gaussian, a DOG of order 2; sharp time localisation), and the
complex analytic **Paul** (asymmetric, good for oscillatory transients).

References
----------
Torrence C, Compo GP (1998). A practical guide to wavelet analysis. *Bulletin
of the American Meteorological Society*, 79(1), 61-78.
https://doi.org/10.1175/1520-0477(1998)079<0061:APGTWA>2.0.CO;2
"""

from __future__ import annotations

import math
from typing import Literal, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

__all__ = ['cwt']

Wavelet = Literal['morlet', 'ricker', 'paul']


def _wavelet_ft(
    wavelet: str, w: Float[Array, 'n'], omega0: float, order: int
) -> Float[Array, 'n']:
    r"""Fourier transform of the mother wavelet at scaled angular frequency ``w``.

    ``w`` is :math:`s\omega` (scale times angular frequency). Torrence--Compo
    normalisation of the (unit-energy) mother wavelet :math:`\hat\psi_0`.
    """
    match wavelet:
        case 'morlet':
            # analytic: the Heaviside gate zeroes the negative frequencies.
            gate = (w > 0.0).astype(w.dtype)
            psi = math.pi**-0.25 * gate * jnp.exp(-0.5 * (w - omega0) ** 2)
        case 'ricker':
            # DOG of order 2: real, symmetric in w (no Heaviside gate).
            norm = 1.0 / math.sqrt(math.gamma(2.5))
            psi = norm * w**2 * jnp.exp(-0.5 * w**2)
        case 'paul':
            m = order
            norm = 2.0**m / math.sqrt(m * math.factorial(2 * m - 1))
            safe = jnp.where(w > 0.0, w, 0.0)
            psi = jnp.where(w > 0.0, norm * safe**m * jnp.exp(-safe), 0.0)
        case _:
            raise ValueError(
                f"wavelet must be 'morlet'/'ricker'/'paul'; got {wavelet!r}."
            )
    return cast(Float[Array, 'n'], psi)


def cwt(
    x: Float[Array, '... n'],
    scales: Float[Array, 's'],
    *,
    wavelet: Wavelet = 'morlet',
    omega0: float = 6.0,
    order: int = 4,
    dt: float = 1.0,
) -> Complex[Array, 's ... n']:
    r"""Continuous wavelet transform (scalogram) of a signal.

    Convolves ``x`` with a bank of scaled mother wavelets in the Fourier domain,
    returning the complex wavelet coefficients at every scale and time. The
    coefficient magnitude :math:`|W(s, b)|` is the scalogram; for an analytic
    wavelet (Morlet, Paul) its argument is the local phase.

    Parameters
    ----------
    x : Float[Array, '... n']
        Real signal(s); the transform is over the trailing (time) axis, batching
        over leading dimensions.
    scales : Float[Array, 's']
        The wavelet scales (in the same time units as ``dt`` -- samples when
        ``dt = 1``). Larger scales resolve lower frequencies. For the Morlet
        wavelet the Fourier period is :math:`\approx 4\pi s /
        (\omega_0 + \sqrt{2 + \omega_0^2})` (:math:`\approx s` at
        :math:`\omega_0 = 6`).
    wavelet : {'morlet', 'ricker', 'paul'}, optional
        Mother wavelet. Default ``'morlet'``.
    omega0 : float, optional
        Morlet central angular frequency. Default ``6.0`` (the usual value, near
        the admissibility floor). Ignored by the other wavelets.
    order : int, optional
        Paul wavelet order :math:`m`. Default ``4``. Ignored by the others.
    dt : float, optional
        Sample spacing. Default ``1.0``.

    Returns
    -------
    Complex[Array, 's ... n']
        Wavelet coefficients, scales along the leading axis. Real-valued for the
        Ricker wavelet (zero imaginary part up to rounding), complex for the
        analytic Morlet / Paul wavelets.
    """
    n = x.shape[-1]
    omega = 2.0 * math.pi * jnp.fft.fftfreq(n, d=dt)
    x_hat = jnp.fft.fft(x, axis=-1)

    def one_scale(s: Float[Array, '']) -> Complex[Array, '... n']:
        psi = _wavelet_ft(wavelet, s * omega, omega0, order) * jnp.sqrt(
            2.0 * math.pi * s / dt
        )
        return jnp.fft.ifft(x_hat * jnp.conj(psi), axis=-1)

    return jax.vmap(one_scale)(jnp.asarray(scales))
