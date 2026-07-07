# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Spectral graph wavelet transform (SGWT).

Multiscale analysis of a signal on a graph by band-pass filtering in the
Laplacian eigenspectrum (Hammond, Vandergheynst & Gribonval 2011): the wavelet
coefficients at scale :math:`t` are :math:`W_t f = g(t\mathcal{L})\, f`, where
:math:`g` is a band-pass kernel of the eigenvalue and :math:`\mathcal{L}` the
graph Laplacian. Rather than eigendecompose :math:`\mathcal{L}` (infeasible on a
dense cortical-surface graph), :math:`g(t\mathcal{L})` is approximated by a
Chebyshev polynomial and applied by **matvec only** -- the transform never
materialises the Laplacian and needs no eigensolver, so it runs GPU-native and
``jit``-clean on graphs far too large for a dense spectrum.

The Chebyshev polynomials :math:`T_k(\tilde{\mathcal{L}})\, f` are the same
across every scale, so they are computed once (``order`` matvecs) and linearly
recombined per scale with the scale-specific Chebyshev coefficients of
:math:`g(t\,\cdot\,)` -- the wavelet bank costs :math:`O(\mathrm{order})`
matvecs in total, not :math:`O(\mathrm{order} \times \text{scales})`.

References
----------
Hammond DK, Vandergheynst P, Gribonval R (2011). Wavelets on graphs via spectral
graph theory. *Applied and Computational Harmonic Analysis*, 30(2), 129-150.
https://doi.org/10.1016/j.acha.2010.04.005
"""

from __future__ import annotations

from typing import Callable, Optional, Union, cast

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float, Num

from ..linalg.matrix_function import chebyshev_coefficients
from ..sparse import ELL, SectionedELL
from .laplacian import _Normalisation, laplacian_matvec

__all__ = ['graph_wavelet_transform', 'mexican_hat_kernel']

_GraphInput = Union[Num[Array, '... n n'], ELL, SectionedELL]


def mexican_hat_kernel(x: Float[Array, '...']) -> Float[Array, '...']:
    r"""The spectral Mexican-hat band-pass kernel :math:`g(x) = x\, e^{-x}`.

    A band-pass filter of the (scaled) eigenvalue: zero at :math:`x = 0`
    (annihilating the constant / DC eigenvector), peaked at :math:`x = 1`, and
    decaying for large :math:`x`. The default SGWT wavelet kernel.
    """
    return x * jnp.exp(-x)


def _estimate_lmax(
    matvec: Callable[[Array], Array],
    v0: Float[Array, 'n'],
    n_iter: int,
) -> Float[Array, '']:
    """Largest Laplacian eigenvalue by power iteration (Rayleigh quotient).

    Seeded from a fixed non-constant vector (the constant vector is the
    combinatorial Laplacian's null eigenvector, so it must be avoided). The
    result is inflated slightly to stay an upper bound of the spectrum, which the
    Chebyshev domain rescaling requires.
    """
    v = v0 / (jnp.linalg.norm(v0) + 1e-12)

    def body(_: int, v: Float[Array, 'n']) -> Float[Array, 'n']:
        w = matvec(v)
        return cast(Float[Array, 'n'], w / (jnp.linalg.norm(w) + 1e-12))

    v = lax.fori_loop(0, n_iter, body, v)
    lam = jnp.vdot(v, matvec(v)) / jnp.vdot(v, v)
    return jnp.real(lam) * 1.01


def graph_wavelet_transform(
    adjacency: _GraphInput,
    signal: Float[Array, '... n'],
    scales: Float[Array, 's'],
    *,
    kernel: Callable[[Array], Array] = mexican_hat_kernel,
    normalisation: _Normalisation = 'symmetric',
    order: int = 30,
    lmax: Optional[float] = None,
    lmax_iter: int = 30,
) -> Float[Array, 's ... n']:
    r"""Spectral graph wavelet transform of a signal (Chebyshev, matvec-only).

    Filters ``signal`` on the graph by the band-pass ``kernel`` at each scale,
    :math:`W_t f = g(t\mathcal{L}) f`, via a Chebyshev approximation applied with
    Laplacian matvecs -- no eigendecomposition. The Chebyshev basis
    :math:`T_k(\tilde{\mathcal{L}}) f` is built once and recombined per scale.

    Parameters
    ----------
    adjacency : Num[Array, '... n n'] or ELL or SectionedELL
        The graph adjacency (dense or sparse); the Laplacian is applied by
        :func:`~nitrix.graph.laplacian_matvec`, never materialised.
    signal : Float[Array, '... n']
        Graph signal(s) over the ``n`` vertices, batching over leading dims.
    scales : Float[Array, 's']
        Wavelet scales :math:`t`. Larger scales probe lower graph frequencies
        (coarser structure).
    kernel : Callable[[Array], Array], optional
        The spectral band-pass response :math:`g(\lambda)` (a function of the
        eigenvalue). Default :func:`mexican_hat_kernel` (:math:`x e^{-x}`).
    normalisation : {'symmetric', 'combinatorial', 'random_walk'}, optional
        Which graph Laplacian to use. Default ``'symmetric'`` (spectrum in
        :math:`[0, 2]`).
    order : int, optional
        Chebyshev polynomial degree. Default ``30``. Higher sharpens the filter
        approximation at the cost of more matvecs.
    lmax : float, optional
        Upper bound on the largest Laplacian eigenvalue (the Chebyshev domain).
        Default ``None`` estimates it by power iteration. For the symmetric /
        random-walk Laplacian ``2.0`` is always a valid bound.
    lmax_iter : int, optional
        Power-iteration steps when estimating ``lmax``. Default ``30``.

    Returns
    -------
    Float[Array, 's ... n']
        Wavelet coefficients, scales along the leading axis.
    """
    n = signal.shape[-1]

    def lmatvec(v: Float[Array, '... n']) -> Float[Array, '... n']:
        return laplacian_matvec(
            adjacency, v[..., None], normalisation=normalisation
        )[..., 0]

    if lmax is None:
        v0 = jnp.cos(jnp.arange(n, dtype=signal.dtype) + 1.0)
        lmax_val = _estimate_lmax(lmatvec, v0, lmax_iter)
    else:
        lmax_val = jnp.asarray(lmax, signal.dtype)

    # Scaled Laplacian: spectrum [0, lmax] -> [-1, 1], the Chebyshev domain.
    def op(v: Float[Array, '... n']) -> Float[Array, '... n']:
        return (2.0 / lmax_val) * lmatvec(v) - v

    # Chebyshev basis T_k(op) @ signal, built once (order matvecs).
    basis = [signal, op(signal)]
    for _ in range(2, order + 1):
        basis.append(2.0 * op(basis[-1]) - basis[-2])
    basis_stack = jnp.stack(basis)  # (order + 1, ..., n)

    def coeffs_for(t: Float[Array, '']) -> Float[Array, 'order_plus_1']:
        return chebyshev_coefficients(
            lambda lam: kernel(t * lam), order, domain=(0.0, lmax_val)
        )

    coeffs = jax.vmap(coeffs_for)(jnp.asarray(scales))  # (s, order + 1)
    return jnp.einsum('sk,k...->s...', coeffs, basis_stack)
