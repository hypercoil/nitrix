# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pairwise kernel and distance functions.

The bread-and-butter primitives for kernel methods: linear,
polynomial, sigmoid, RBF / Gaussian, cosine; plus the
underlying squared-L2 ``linear_distance``.

All accept a ``theta`` argument for parameterised (Mahalanobis-
style) inner products:

- ``theta is None`` -- the standard inner product / distance.
- ``theta`` is **vector** ``(..., d)`` -- diagonal Mahalanobis;
  the kernel acts as ``sum_d theta_d x_d y_d``.
- ``theta`` is **matrix** ``(..., d, d)`` -- full Mahalanobis;
  the kernel acts as ``x^T theta y``.

Performance: the legacy ``hypercoil.functional.kernel.linear_distance``
materialised an ``(..., n, m, d)`` intermediate to compute the
pairwise difference, costing ``O(n m d)`` memory.  The green-field
rewrite uses the identity ``|x - y|^2 = |x|^2 + |y|^2 - 2 x·y``,
which needs only ``(..., n, m)`` memory -- a ``d``-fold reduction.
For ``d`` in the thousands (covariance vectors, voxelwise features)
this is the difference between OOM and fitting in HBM.

What we ship:

- ``linear_kernel`` -- ``X0 @ X1.T``, optionally Mahalanobis-weighted.
- ``linear_distance`` -- squared L2 distance (or Mahalanobis-
  weighted squared distance), computed via the BLAS-friendly
  identity.
- ``parameterised_norm`` -- per-row norm, optionally Mahalanobis-
  weighted.
- ``rbf_kernel`` / ``gaussian_kernel`` -- ``exp(-gamma * dist^2)``;
  the two are aliases up to the ``gamma = sigma^-2`` substitution.
- ``polynomial_kernel`` -- ``(gamma * linear_kernel + r)^order``.
- ``sigmoid_kernel`` -- ``tanh(gamma * linear_kernel + r)``.
- ``cosine_kernel`` -- ``linear_kernel`` on row-normalised inputs.

What the legacy had that we drop:

- ``cov_kernel`` / ``corr_kernel`` -- thin wrappers around
  ``nitrix.stats.pairedcov`` / ``pairedcorr``.  Use those directly.
- The ``singledispatch`` sparse-tensor overloads (BCOO / TopK).
  ELL / SectionedELL are the project's sparse formats; if you
  need a sparse kernel matrix, build it via ``semiring_ell_matmul``
  with the ``EUCLIDEAN`` semiring on a pre-thresholded
  adjacency.
"""

from __future__ import annotations

from typing import Optional, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Num

__all__ = [
    'linear_kernel',
    'linear_distance',
    'parameterised_norm',
    'rbf_kernel',
    'gaussian_kernel',
    'polynomial_kernel',
    'sigmoid_kernel',
    'cosine_kernel',
]


# ---------------------------------------------------------------------------
# Theta dispatch helper
# ---------------------------------------------------------------------------


def _theta_kind(theta: Optional[Array]) -> str:
    """Classify ``theta`` as one of ``'none'``, ``'vector'``, ``'matrix'``."""
    if theta is None:
        return 'none'
    if theta.ndim == 1 or theta.shape[-1] != theta.shape[-2]:
        return 'vector'
    return 'matrix'


# ---------------------------------------------------------------------------
# Linear primitives
# ---------------------------------------------------------------------------


def linear_kernel(
    X0: Float[Array, '... n d'],
    X1: Optional[Float[Array, '... m d']] = None,
    *,
    theta: Optional[Num[Array, '... d']] = None,
) -> Float[Array, '... n m']:
    """Parameterised linear kernel: ``X0 @ theta @ X1.T``.

    Parameters
    ----------
    X0, X1
        Sample tensors ``(..., n, d)`` and ``(..., m, d)``.
        If ``X1 is None``, uses ``X0`` (self-kernel; result is
        ``(..., n, n)``).
    theta
        Optional inner-product weights.  ``None`` -> standard
        Euclidean; vector ``(..., d)`` -> diagonal Mahalanobis;
        matrix ``(..., d, d)`` -> full Mahalanobis.

    Returns
    -------
    Kernel matrix ``(..., n, m)``.
    """
    if X1 is None:
        X1 = X0
    kind = _theta_kind(theta)
    if kind == 'none':
        return X0 @ X1.swapaxes(-1, -2)
    # kind is 'vector' or 'matrix' here, so theta is set.
    assert theta is not None
    if kind == 'vector':
        # (X0 * theta) treats theta as the diagonal of a weight matrix.
        return (X0 * theta[..., None, :]) @ X1.swapaxes(-1, -2)
    # matrix
    return X0 @ theta @ X1.swapaxes(-1, -2)


def linear_distance(
    X0: Float[Array, '... n d'],
    X1: Optional[Float[Array, '... m d']] = None,
    *,
    theta: Optional[Num[Array, '... d']] = None,
) -> Float[Array, '... n m']:
    """Squared L2 (or Mahalanobis) distance ``|x - y|^2``.

    Computed via ``|x|^2 + |y|^2 - 2 x.y`` -- avoiding the
    ``(n, m, d)`` intermediate that a naive subtract-then-square
    would materialise.  At ``d = 1000`` this is a ``1000x``
    memory reduction.

    Parameters
    ----------
    X0, X1
        Sample tensors ``(..., n, d)`` and ``(..., m, d)``.
        Defaults ``X1 = X0`` (self-distance).
    theta
        Optional metric tensor.  ``None`` -> standard squared L2;
        vector -> diagonal Mahalanobis ``sum_d theta_d (x_d - y_d)^2``;
        matrix -> full Mahalanobis ``(x - y)^T theta (x - y)``.

    Returns
    -------
    Squared-distance matrix ``(..., n, m)``.

    Notes
    -----
    The result is clipped to ``>= 0`` to suppress catastrophic
    cancellation when ``X0`` ~ ``X1`` (the identity-form
    computation can produce small negative entries via roundoff;
    a true squared distance is always non-negative).
    """
    if X1 is None:
        X1 = X0
    kind = _theta_kind(theta)
    cross = linear_kernel(X0, X1, theta=theta)
    if kind == 'none':
        x0_sq = jnp.sum(X0 * X0, axis=-1)[..., :, None]
        x1_sq = jnp.sum(X1 * X1, axis=-1)[..., None, :]
    elif kind == 'vector':
        assert theta is not None
        x0_sq = jnp.sum(X0 * X0 * theta[..., None, :], axis=-1)[..., :, None]
        x1_sq = jnp.sum(X1 * X1 * theta[..., None, :], axis=-1)[..., None, :]
    else:
        # full matrix: (x^T theta x) = sum_ij theta_ij x_i x_j
        # Compute as einsum to keep memory ~(n, d) not (n, d, d).
        assert theta is not None
        x0_sq = jnp.einsum('...id,...de,...ie->...i', X0, theta, X0)[
            ..., :, None
        ]
        x1_sq = jnp.einsum('...id,...de,...ie->...i', X1, theta, X1)[
            ..., None, :
        ]
    dist_sq = x0_sq + x1_sq - 2.0 * cross
    return jnp.maximum(dist_sq, 0.0)


def parameterised_norm(
    X: Float[Array, '... n d'],
    *,
    theta: Optional[Num[Array, '... d']] = None,
    squared: bool = False,
) -> Float[Array, '... n d']:
    """Per-row Mahalanobis-style normalisation.

    Returns ``X / norm`` (or ``X / norm^2`` with ``squared=True``)
    where ``norm`` is the per-row Mahalanobis norm under ``theta``.
    For ``theta = None`` this is L2-normalisation along the last
    axis.

    Parameters
    ----------
    X
        Sample tensor ``(..., n, d)``.
    theta
        Optional metric.  Same shape conventions as
        ``linear_kernel``'s ``theta``.
    squared
        If ``True``, divide by ``norm^2`` instead of ``norm``.

    Returns
    -------
    Normalised tensor, same shape as ``X``.
    """
    kind = _theta_kind(theta)
    if kind == 'none':
        n_sq = jnp.sum(X * X, axis=-1, keepdims=True)
    elif kind == 'vector':
        assert theta is not None
        n_sq = jnp.sum(X * X * theta[..., None, :], axis=-1, keepdims=True)
    else:
        # full matrix Mahalanobis norm-sq
        assert theta is not None
        n_sq = jnp.einsum(
            '...id,...de,...ie->...i',
            X,
            theta,
            X,
        )[..., None]
    # ``n_sq`` may flow from ``jnp.einsum`` (typed Any); restore.
    if squared:
        return cast(Float[Array, '...'], X / (n_sq + jnp.finfo(X.dtype).eps))
    return cast(
        Float[Array, '...'], X / (jnp.sqrt(n_sq) + jnp.finfo(X.dtype).eps)
    )


# ---------------------------------------------------------------------------
# Common nonlinear kernels
# ---------------------------------------------------------------------------


def _default_gamma(X: Array, gamma: Optional[float]) -> float:
    """scikit-learn convention: default ``gamma = 1 / n_features``."""
    if gamma is not None:
        return gamma
    return 1.0 / X.shape[-1]


def rbf_kernel(
    X0: Float[Array, '... n d'],
    X1: Optional[Float[Array, '... m d']] = None,
    *,
    theta: Optional[Num[Array, '... d']] = None,
    gamma: Optional[float] = None,
) -> Float[Array, '... n m']:
    """RBF (Gaussian) kernel: ``exp(-gamma * |x - y|^2)``.

    Default ``gamma = 1 / d`` matches scikit-learn.
    """
    g = _default_gamma(X0, gamma)
    return jnp.exp(-g * linear_distance(X0, X1, theta=theta))


def gaussian_kernel(
    X0: Float[Array, '... n d'],
    X1: Optional[Float[Array, '... m d']] = None,
    *,
    theta: Optional[Num[Array, '... d']] = None,
    sigma: Optional[float] = None,
) -> Float[Array, '... n m']:
    """Gaussian kernel: ``exp(-|x - y|^2 / (2 sigma^2))``.

    Same as ``rbf_kernel`` with ``gamma = 1 / (2 sigma^2)``;
    provided for the ``sigma`` parameterisation common in
    classical Gaussian-process literature.
    """
    gamma = None if sigma is None else 1.0 / (2.0 * sigma * sigma)
    return rbf_kernel(X0, X1, theta=theta, gamma=gamma)


def polynomial_kernel(
    X0: Float[Array, '... n d'],
    X1: Optional[Float[Array, '... m d']] = None,
    *,
    theta: Optional[Num[Array, '... d']] = None,
    gamma: Optional[float] = None,
    order: int = 3,
    r: float = 0.0,
) -> Float[Array, '... n m']:
    """Polynomial kernel: ``(gamma * <x, y> + r)^order``."""
    g = _default_gamma(X0, gamma)
    K = linear_kernel(X0, X1, theta=theta)
    return (g * K + r) ** order


def sigmoid_kernel(
    X0: Float[Array, '... n d'],
    X1: Optional[Float[Array, '... m d']] = None,
    *,
    theta: Optional[Num[Array, '... d']] = None,
    gamma: Optional[float] = None,
    r: float = 0.0,
) -> Float[Array, '... n m']:
    """Sigmoid (hyperbolic tangent) kernel: ``tanh(gamma * <x, y> + r)``.

    Note: not positive semidefinite for arbitrary gamma / r --
    use cautiously in kernel-method pipelines.
    """
    g = _default_gamma(X0, gamma)
    K = linear_kernel(X0, X1, theta=theta)
    return jax.nn.tanh(g * K + r)


def cosine_kernel(
    X0: Float[Array, '... n d'],
    X1: Optional[Float[Array, '... m d']] = None,
    *,
    theta: Optional[Num[Array, '... d']] = None,
) -> Float[Array, '... n m']:
    """Cosine kernel: row-normalised linear kernel.

    Equivalent to ``linear_kernel(unit(X0), unit(X1), theta=theta)``
    where ``unit(X) = X / |X|``.
    """
    X0_n = parameterised_norm(X0, theta=theta)
    X1_n = X0_n if X1 is None else parameterised_norm(X1, theta=theta)
    return linear_kernel(X0_n, X1_n, theta=theta)
