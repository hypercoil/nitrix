# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Pairwise kernel and distance functions.

The bread-and-butter primitives for kernel methods: linear,
polynomial, sigmoid, RBF / Gaussian, cosine; plus the underlying
squared-L2 distance :func:`linear_distance`.

All accept a ``theta`` argument for parameterised (Mahalanobis-style)
inner products:

- ``theta is None`` -- the standard inner product / distance.
- ``theta`` a **vector** :math:`(\dots, d)` -- diagonal Mahalanobis;
  the kernel acts as :math:`\sum_d \theta_d x_d y_d`.
- ``theta`` a **matrix** :math:`(\dots, d, d)` -- full Mahalanobis;
  the kernel acts as :math:`x^{\top} \theta y`.

The squared-distance path uses the identity
:math:`\|x - y\|^2 = \|x\|^2 + \|y\|^2 - 2\, x \cdot y`, which needs
only :math:`(\dots, n, m)` memory rather than materialising an
:math:`(\dots, n, m, d)` pairwise-difference intermediate -- a
:math:`d`-fold reduction.  For :math:`d` in the thousands (covariance
vectors, voxelwise features) this is the difference between OOM and
fitting in device memory.

What we ship:

- :func:`linear_kernel` -- :math:`X_0 X_1^{\top}`, optionally
  Mahalanobis-weighted.
- :func:`linear_distance` -- squared L2 distance (or Mahalanobis-
  weighted squared distance), computed via the BLAS-friendly
  identity.
- :func:`parameterised_norm` -- per-row norm, optionally Mahalanobis-
  weighted.
- :func:`rbf_kernel` / :func:`gaussian_kernel` --
  :math:`\exp(-\gamma \|x - y\|^2)`; the two are aliases up to the
  :math:`\gamma = 1 / (2\sigma^2)` substitution.
- :func:`polynomial_kernel` -- :math:`(\gamma\, x^{\top} y + r)^{\mathrm{order}}`.
- :func:`sigmoid_kernel` -- :math:`\tanh(\gamma\, x^{\top} y + r)`.
- :func:`cosine_kernel` -- linear kernel on row-normalised inputs.

For covariance / correlation kernels, use
:func:`nitrix.stats.pairedcov` and :func:`nitrix.stats.pairedcorr`
directly.  There are no sparse-tensor overloads here; ELL and
SectionedELL are the project's sparse formats, so build a sparse
kernel matrix via :func:`nitrix.semiring.semiring_ell_matmul` with the
``EUCLIDEAN`` semiring on a pre-thresholded adjacency.
"""

from __future__ import annotations

from typing import Optional, Union, cast

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
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
    'se_spectral_density',
    'matern_spectral_density',
    'spectral_density',
]


# ---------------------------------------------------------------------------
# Theta dispatch helper
# ---------------------------------------------------------------------------


def _theta_kind(theta: Optional[Array]) -> str:
    r"""Classify a ``theta`` metric argument by its rank.

    Determines how a metric tensor should be interpreted: absent, a
    diagonal (vector) Mahalanobis weighting, or a full (matrix)
    Mahalanobis weighting.  A ``theta`` is treated as a vector when it
    is one-dimensional or its trailing two axes are non-square.

    Parameters
    ----------
    theta
        Optional metric tensor.  ``None``; a diagonal weight vector
        :math:`(\dots, d)`; or a full weight matrix :math:`(\dots, d, d)`.

    Returns
    -------
    str
        One of ``'none'`` (``theta is None``), ``'vector'`` (diagonal
        Mahalanobis), or ``'matrix'`` (full Mahalanobis).
    """
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
    r"""Pairwise squared L2 (or Mahalanobis) distance :math:`\|x - y\|^2`.

    Computed via the identity
    :math:`\|x\|^2 + \|y\|^2 - 2\, x \cdot y`, avoiding the
    :math:`(n, m, d)` intermediate that a naive subtract-then-square
    would materialise.  At :math:`d = 1000` this is a thousand-fold
    memory reduction.

    Parameters
    ----------
    X0, X1
        Sample tensors of shape :math:`(\dots, n, d)` and
        :math:`(\dots, m, d)`.  If ``X1 is None``, uses ``X0`` (self-
        distance; the result is :math:`(\dots, n, n)`).
    theta
        Optional metric tensor.  ``None`` gives the standard squared
        L2 distance; a vector :math:`(\dots, d)` gives the diagonal
        Mahalanobis distance :math:`\sum_d \theta_d (x_d - y_d)^2`; a
        matrix :math:`(\dots, d, d)` gives the full Mahalanobis
        distance :math:`(x - y)^{\top} \theta (x - y)`.  The matrix
        form is exact only for a **symmetric** ``theta`` (the
        identity-form expansion uses one cross term
        :math:`2\, x^{\top} \theta y`); pass
        :math:`\tfrac{1}{2}(\theta + \theta^{\top})` for an asymmetric
        input.

    Returns
    -------
    Float[Array, '... n m']
        Squared-distance matrix of shape :math:`(\dots, n, m)`.

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
    r"""Per-row Mahalanobis-style normalisation.

    Returns :math:`X / \mathrm{norm}` (or :math:`X / \mathrm{norm}^2`
    with ``squared=True``), where :math:`\mathrm{norm}` is the per-row
    Mahalanobis norm under ``theta``.  For ``theta = None`` this is L2
    normalisation along the last axis.

    Parameters
    ----------
    X
        Sample tensor of shape :math:`(\dots, n, d)`.
    theta
        Optional metric tensor.  Same shape conventions as the
        ``theta`` argument of :func:`linear_kernel`.
    squared
        If ``True``, divide by :math:`\mathrm{norm}^2` instead of
        :math:`\mathrm{norm}`.

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
    if squared:
        return X / (n_sq + jnp.finfo(X.dtype).eps)
    return X / (jnp.sqrt(n_sq) + jnp.finfo(X.dtype).eps)


# ---------------------------------------------------------------------------
# Common nonlinear kernels
# ---------------------------------------------------------------------------


def _default_gamma(X: Array, gamma: Optional[float]) -> float:
    r"""Resolve a kernel bandwidth, defaulting to the inverse feature count.

    If ``gamma`` is given it is returned unchanged; otherwise the
    scikit-learn convention :math:`\gamma = 1 / d` is used, where
    :math:`d` is the trailing (feature) dimension of ``X``.

    Parameters
    ----------
    X
        Sample tensor whose trailing axis length is the feature count
        :math:`d`.
    gamma
        Optional explicit bandwidth.  ``None`` requests the default.

    Returns
    -------
    float
        The resolved bandwidth: ``gamma`` if provided, else
        :math:`1 / d`.
    """
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
    r"""RBF (Gaussian) kernel :math:`\exp(-\gamma \|x - y\|^2)`.

    The squared distance is the (optionally Mahalanobis-weighted)
    squared L2 distance from :func:`linear_distance`.  The default
    bandwidth :math:`\gamma = 1 / d` matches scikit-learn.

    Parameters
    ----------
    X0, X1
        Sample tensors of shape :math:`(\dots, n, d)` and
        :math:`(\dots, m, d)`.  If ``X1 is None``, uses ``X0`` (self-
        kernel; the result is :math:`(\dots, n, n)`).
    theta
        Optional metric tensor for the distance.  Same shape
        conventions as the ``theta`` argument of :func:`linear_kernel`.
    gamma
        Optional bandwidth.  ``None`` uses :math:`\gamma = 1 / d`.

    Returns
    -------
    Float[Array, '... n m']
        Kernel matrix of shape :math:`(\dots, n, m)`.
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
    r"""Gaussian kernel :math:`\exp(-\|x - y\|^2 / (2 \sigma^2))`.

    Identical to :func:`rbf_kernel` with
    :math:`\gamma = 1 / (2 \sigma^2)`; provided for the :math:`\sigma`
    parameterisation common in the classical Gaussian-process
    literature.

    Parameters
    ----------
    X0, X1
        Sample tensors of shape :math:`(\dots, n, d)` and
        :math:`(\dots, m, d)`.  If ``X1 is None``, uses ``X0`` (self-
        kernel; the result is :math:`(\dots, n, n)`).
    theta
        Optional metric tensor for the distance.  Same shape
        conventions as the ``theta`` argument of :func:`linear_kernel`.
    sigma
        Optional length scale.  ``None`` defers to the default
        bandwidth of :func:`rbf_kernel` (:math:`\gamma = 1 / d`);
        otherwise :math:`\gamma = 1 / (2 \sigma^2)`.

    Returns
    -------
    Float[Array, '... n m']
        Kernel matrix of shape :math:`(\dots, n, m)`.
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
    r"""Polynomial kernel :math:`(\gamma \langle x, y \rangle + r)^{\mathrm{order}}`.

    The inner product :math:`\langle x, y \rangle` is the (optionally
    Mahalanobis-weighted) linear kernel from :func:`linear_kernel`.

    Parameters
    ----------
    X0, X1
        Sample tensors of shape :math:`(\dots, n, d)` and
        :math:`(\dots, m, d)`.  If ``X1 is None``, uses ``X0`` (self-
        kernel; the result is :math:`(\dots, n, n)`).
    theta
        Optional metric tensor for the inner product.  Same shape
        conventions as the ``theta`` argument of :func:`linear_kernel`.
    gamma
        Optional scaling of the inner product.  ``None`` uses
        :math:`\gamma = 1 / d`.
    order
        Polynomial degree (the exponent).
    r
        Additive offset (independent term) inside the power.

    Returns
    -------
    Float[Array, '... n m']
        Kernel matrix of shape :math:`(\dots, n, m)`.
    """
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
    r"""Sigmoid (hyperbolic tangent) kernel :math:`\tanh(\gamma \langle x, y \rangle + r)`.

    The inner product :math:`\langle x, y \rangle` is the (optionally
    Mahalanobis-weighted) linear kernel from :func:`linear_kernel`.

    Parameters
    ----------
    X0, X1
        Sample tensors of shape :math:`(\dots, n, d)` and
        :math:`(\dots, m, d)`.  If ``X1 is None``, uses ``X0`` (self-
        kernel; the result is :math:`(\dots, n, n)`).
    theta
        Optional metric tensor for the inner product.  Same shape
        conventions as the ``theta`` argument of :func:`linear_kernel`.
    gamma
        Optional scaling of the inner product.  ``None`` uses
        :math:`\gamma = 1 / d`.
    r
        Additive offset (independent term) inside the tanh.

    Returns
    -------
    Float[Array, '... n m']
        Kernel matrix of shape :math:`(\dots, n, m)`.

    Notes
    -----
    This kernel is not positive semidefinite for arbitrary
    :math:`\gamma` and :math:`r`; use it cautiously in kernel-method
    pipelines.
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
    r"""Cosine kernel: the linear kernel on row-normalised inputs.

    Equivalent to ``linear_kernel(unit(X0), unit(X1), theta=theta)``,
    where each row of ``X`` is normalised to unit (optionally
    Mahalanobis) norm via :func:`parameterised_norm` before the linear
    kernel of :func:`linear_kernel` is applied.

    Parameters
    ----------
    X0, X1
        Sample tensors of shape :math:`(\dots, n, d)` and
        :math:`(\dots, m, d)`.  If ``X1 is None``, uses the normalised
        ``X0`` (self-kernel; the result is :math:`(\dots, n, n)`).
    theta
        Optional metric tensor for both the normalisation and the
        linear kernel.  Same shape conventions as the ``theta``
        argument of :func:`linear_kernel`.

    Returns
    -------
    Float[Array, '... n m']
        Kernel matrix of shape :math:`(\dots, n, m)`.
    """
    X0_n = parameterised_norm(X0, theta=theta)
    X1_n = X0_n if X1 is None else parameterised_norm(X1, theta=theta)
    return linear_kernel(X0_n, X1_n, theta=theta)


# ---------------------------------------------------------------------------
# Stationary-kernel spectral densities (1-D)
# ---------------------------------------------------------------------------
#
# ``S_theta(omega)`` is the power spectral density of a stationary kernel
# ``k(r)`` (the Fourier transform: ``k(r) = (1/pi) int_0^inf S(omega) cos(omega
# r) domega``).  Evaluated at the Laplace eigen-frequencies it supplies the prior
# variances of a Hilbert-space approximate Gaussian process (HSGP); see
# :func:`nitrix.stats.basis.hsgp_basis`.  Parameterisation (lengthscale ``rho``,
# amplitude ``amplitude``) matches scikit-learn's ``RBF`` / ``Matern`` kernels
# scaled by ``amplitude**2``.

# Matern smoothness -> spectral closed form is given per-nu below (nu in
# {1/2, 3/2, 5/2}); a general-nu form via ``gammaln`` is a future add.
_MATERN_NU = {0.5, 1.5, 2.5}


def se_spectral_density(
    omega: Float[Array, '...'],
    *,
    rho: Union[float, Float[Array, '']],
    amplitude: float = 1.0,
    dim: int = 1,
) -> Float[Array, '...']:
    r"""Squared-exponential (RBF) spectral density in ``dim`` dimensions.

    Evaluates the power spectral density
    :math:`S(w) = a^2 (2\pi)^{D/2} \rho^{D} \exp(-\rho^2 \|w\|^2 / 2)`
    (with amplitude :math:`a` and input dimension :math:`D`) -- the
    :math:`D`-dimensional Fourier transform of
    :math:`k(r) = a^2 \exp(-r^2 / (2 \rho^2))`, i.e. scikit-learn's
    ``RBF(length_scale=rho)`` scaled by :math:`a^2`.  At ``dim=1`` this
    reduces to :math:`a^2 \sqrt{2\pi}\, \rho \exp(-\rho^2 w^2 / 2)`.

    Parameters
    ----------
    omega
        Frequency magnitude :math:`\|w\|`, of arbitrary shape.
        Integer inputs are promoted to the default float dtype before
        the density is formed (an integer ``omega`` would otherwise
        truncate ``rho`` to zero).
    rho
        Length scale :math:`\rho` (scalar or scalar array).
    amplitude
        Kernel amplitude :math:`a`; the density scales as :math:`a^2`.
    dim
        Input dimension :math:`D` (``1`` for the one-dimensional case).

    Returns
    -------
    Float[Array, '...']
        Spectral density evaluated elementwise at ``omega``, matching
        its (float-promoted) shape and dtype.
    """
    omega = jnp.asarray(omega)
    # Promote an integer omega to the default float (f64 under x64, else f32)
    # BEFORE deriving rho/amp -- else `dtype=omega.dtype` truncates rho/amplitude
    # to int (rho->0 zeros the density). A float omega is preserved as-is
    # (float32 stays float32 for the x32 path).
    if not jnp.issubdtype(omega.dtype, jnp.floating):
        omega = omega.astype(float)
    dt = omega.dtype
    rho = jnp.asarray(rho, dtype=dt)
    amp2 = jnp.asarray(amplitude, dtype=dt) ** 2
    const = (2.0 * jnp.pi) ** (dim / 2.0) * rho**dim
    return cast(Array, amp2 * const * jnp.exp(-0.5 * (rho * omega) ** 2))


def matern_spectral_density(
    omega: Float[Array, '...'],
    *,
    rho: Union[float, Float[Array, '']],
    nu: float,
    amplitude: float = 1.0,
    dim: int = 1,
) -> Float[Array, '...']:
    r"""Matern spectral density in ``dim`` dimensions for ``nu`` in ``{0.5, 1.5, 2.5}``.

    Evaluates the power spectral density
    :math:`S(w) = a^2 C (\lambda^2 + \|w\|^2)^{-(\nu + D/2)}` with the
    Matern rate :math:`\lambda = \sqrt{2\nu} / \rho`, normaliser
    :math:`C = 2^{D} \pi^{D/2} \Gamma(\nu + D/2) (2\nu)^{\nu} /
    (\Gamma(\nu) \rho^{2\nu})`, amplitude :math:`a`, and input
    dimension :math:`D` -- the :math:`D`-dimensional Fourier transform
    of scikit-learn's ``Matern(length_scale=rho, nu=nu)`` scaled by
    :math:`a^2`.  At ``dim=1`` this reduces to the closed forms
    :math:`2\lambda / (\lambda^2 + w^2)`,
    :math:`4\lambda^3 / (\lambda^2 + w^2)^2`, and
    :math:`\tfrac{16}{3}\lambda^5 / (\lambda^2 + w^2)^3` for
    :math:`\nu = 1/2, 3/2, 5/2` respectively, evaluated directly to
    keep the well-tested one-dimensional path byte-identical.

    Parameters
    ----------
    omega
        Frequency magnitude :math:`\|w\|`, of arbitrary shape.
        Integer inputs are promoted to the default float dtype before
        the density is formed (an integer ``omega`` would otherwise
        truncate ``rho`` and yield ``NaN``).
    rho
        Length scale :math:`\rho` (scalar or scalar array).
    nu
        Matern smoothness :math:`\nu`; must be ``0.5``, ``1.5``, or
        ``2.5`` (raises :class:`ValueError` otherwise).
    amplitude
        Kernel amplitude :math:`a`; the density scales as :math:`a^2`.
    dim
        Input dimension :math:`D` (``1`` for the one-dimensional case).

    Returns
    -------
    Float[Array, '...']
        Spectral density evaluated elementwise at ``omega``, matching
        its (float-promoted) shape and dtype.
    """
    if nu not in _MATERN_NU:
        raise ValueError(
            f'matern_spectral_density: nu={nu!r} unsupported; use 0.5, 1.5, or '
            '2.5.'
        )
    omega = jnp.asarray(omega)
    # See se_spectral_density: promote an integer omega to the default float
    # before casting rho/amplitude (otherwise an integer omega truncates rho ->
    # lam=1/0 -> NaN). A float omega is preserved, keeping the dim=1 closed forms
    # byte-identical.
    if not jnp.issubdtype(omega.dtype, jnp.floating):
        omega = omega.astype(float)
    dt = omega.dtype
    rho = jnp.asarray(rho, dtype=dt)
    amp2 = jnp.asarray(amplitude, dtype=dt) ** 2
    w2 = omega**2
    if dim == 1:
        if nu == 0.5:
            lam = 1.0 / rho
            return amp2 * 2.0 * lam / (lam**2 + w2)
        if nu == 1.5:
            lam = jnp.sqrt(3.0) / rho
            return amp2 * 4.0 * lam**3 / (lam**2 + w2) ** 2
        lam = jnp.sqrt(5.0) / rho  # nu == 2.5
        return amp2 * (16.0 / 3.0) * lam**5 / (lam**2 + w2) ** 3
    # General D-dimensional isotropic form (gammaln for the normaliser).
    half = nu + dim / 2.0
    lam2 = 2.0 * nu / rho**2
    log_c = (
        dim * jnp.log(2.0)
        + (dim / 2.0) * jnp.log(jnp.pi)
        + gammaln(half)
        + nu * jnp.log(2.0 * nu)
        - gammaln(nu)
        - 2.0 * nu * jnp.log(rho)
    )
    return amp2 * jnp.exp(log_c) * (lam2 + w2) ** (-half)


_KERNEL_NU = {
    'matern12': 0.5,
    'exp': 0.5,
    'exponential': 0.5,
    'matern32': 1.5,
    'matern52': 2.5,
}


def spectral_density(
    omega: Float[Array, '...'],
    *,
    kernel: str,
    rho: Union[float, Float[Array, '']],
    amplitude: float = 1.0,
    dim: int = 1,
) -> Float[Array, '...']:
    r"""Dispatch a stationary-kernel spectral density (in ``dim`` dimensions).

    Selects and evaluates a spectral density by name, routing to
    :func:`se_spectral_density` for the squared-exponential family and
    to :func:`matern_spectral_density` for the Matern families.  The
    ``kernel`` string is matched case-insensitively after stripping
    ``/``, ``-``, and ``_``.

    Parameters
    ----------
    omega
        Frequency magnitude :math:`\|w\|`, of arbitrary shape.
    kernel
        Kernel name.  Squared-exponential: ``'rbf'``, ``'se'``,
        ``'sqexp'``, ``'squaredexponential'``, ``'gaussian'``.  Matern:
        ``'matern12'`` / ``'exp'`` / ``'exponential'`` (:math:`\nu = 1/2`),
        ``'matern32'`` (:math:`\nu = 3/2`), ``'matern52'``
        (:math:`\nu = 5/2`).  An unknown name raises
        :class:`ValueError`.
    rho
        Length scale :math:`\rho` (scalar or scalar array).
    amplitude
        Kernel amplitude :math:`a`; the density scales as :math:`a^2`.
    dim
        Input dimension :math:`D` (``1`` for the one-dimensional case).

    Returns
    -------
    Float[Array, '...']
        Spectral density evaluated elementwise at ``omega``.
    """
    k = kernel.lower().replace('/', '').replace('-', '').replace('_', '')
    if k in ('rbf', 'se', 'sqexp', 'squaredexponential', 'gaussian'):
        return se_spectral_density(
            omega, rho=rho, amplitude=amplitude, dim=dim
        )
    if k in _KERNEL_NU:
        return matern_spectral_density(
            omega, rho=rho, nu=_KERNEL_NU[k], amplitude=amplitude, dim=dim
        )
    raise ValueError(
        f'spectral_density: unknown kernel {kernel!r}; expected one of '
        "'rbf'/'se', 'matern12'/'exp', 'matern32', 'matern52'."
    )
