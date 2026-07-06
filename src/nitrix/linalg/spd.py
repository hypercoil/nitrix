# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Symmetric positive-definite manifold operations.

Applying :math:`\\log` / :math:`\\sqrt{\\cdot}` / a fractional power to the
eigenvalues of a symmetric positive-definite (SPD) matrix is numerically
unstable when *any* eigenvalue is at or below the machine-precision floor of
the input's dtype.

This module handles that instability through three design choices:

1. **Eigenvalue clipping is on by default.**  ``eigvalue_clip='auto'``
   clips at :math:`\\max_i |\\lambda_i| \\cdot d \\cdot \\varepsilon`, the
   standard rank-truncation threshold (matching
   :func:`numpy.linalg.matrix_rank`), where :math:`d` is the matrix
   dimension and :math:`\\varepsilon` is the dtype's machine epsilon.
   Eigenvalues below the threshold are set to the threshold itself rather
   than to zero -- :math:`\\log(\\mathrm{threshold})` is finite, whereas
   :math:`\\log(0)` is :math:`-\\infty` and would contaminate everything
   downstream.
2. **No silent NaN filling.**  If the input is genuinely indefinite the
   operation is mathematically undefined, so the ``NaN`` is surfaced rather
   than silently replaced with ``0``.
3. **The differentiability hook is explicit.**  Pass ``psi > 0`` to
   recondition the input before the eigendecomposition (perturbing
   degenerate eigenvalues so the eigendecomposition VJP does not blow up).
   This is a backward-only concern; for well-conditioned input the forward
   result is the same either way.

Public surface:

- :func:`symmap` -- generic eigenvalue-function map.
- :func:`symlog`, :func:`symexp`, :func:`symsqrt` -- the common cases.
- :func:`sympower` -- arbitrary real power of an SPD matrix.
- :func:`tangent_project_spd` / :func:`cone_project_spd` -- log-Euclidean
  tangent-space projection used in the Riemannian mean / connectivity
  embedding literature.
- :func:`mean_log_euclidean` -- closed-form Fréchet mean under the
  log-Euclidean metric, :math:`\\exp(\\operatorname{mean}_i \\log X_i)`.
- :func:`mean_euclidean` -- the trivial Euclidean mean, provided for
  symmetry with the SPD mean family.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional, Sequence, Union, assert_never

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._solver import safe_eigh
from .matrix import recondition_eigenspaces, symmetric

__all__ = [
    'symmap',
    'symlog',
    'symexp',
    'symsqrt',
    'sympower',
    'tangent_project_spd',
    'cone_project_spd',
    'mean_log_euclidean',
    'mean_euclidean',
]


EigvalueClip = Union[Literal['auto', 'none'], float]

#: Which recipe computes the SPD square root / inverse square root.  ``'eigh'``
#: is the eigendecomposition map (the reference oracle; routes through
#: :func:`safe_eigh`).  ``'newton_schulz'`` is the matmul-only coupled iteration
#: -- cuSOLVER-free, GPU-friendly, and gradient-stable at a repeated spectrum
#: (no eigenvector VJP).  A ``driver`` axis in the sense of the dispatch model:
#: numerically-divergent recipes of the *same* map.
SqrtDriver = Literal['eigh', 'newton_schulz']


def _clip_eigvals(
    L: Float[Array, '... d'],
    clip: EigvalueClip,
) -> Float[Array, '... d']:
    """Clip eigenvalues to a floor defined by ``clip``.

    JIT-safe: never calls ``float()`` on a tracer; the threshold is a JAX
    scalar derived from ``L``.

    Parameters
    ----------
    L : Float[Array, '... d']
        Eigenvalues to clip, batched over any leading axes with the
        eigenvalue index on the trailing axis of length ``d``.
    clip : {'auto', 'none'} or float
        Clipping policy.  ``'auto'`` floors eigenvalues at the
        rank-truncation threshold
        :math:`\\max_i |\\lambda_i| \\cdot d \\cdot \\varepsilon` (matching
        :func:`numpy.linalg.matrix_rank`).  ``'none'`` passes eigenvalues
        through unchanged.  A ``float`` sets an explicit threshold.

    Returns
    -------
    Float[Array, '... d']
        Eigenvalues with every entry below the threshold replaced by the
        threshold; same shape and dtype as ``L``.
    """
    if clip == 'none':
        return L
    if clip == 'auto':
        d = L.shape[-1]
        eps = jnp.finfo(L.dtype).eps
        max_abs = jnp.abs(L).max(axis=-1, keepdims=True)
        threshold = max_abs * d * eps
    else:
        threshold = jnp.asarray(clip, dtype=L.dtype)
    return jnp.where(L < threshold, threshold, L)


def _eigh_with_clip(
    input: Float[Array, '... d d'],
    *,
    psi: float,
    key: Optional[jax.Array],
    eigvalue_clip: EigvalueClip,
) -> tuple[Float[Array, '... d'], Float[Array, '... d d']]:
    """Symmetrised eigendecomposition with optional eigenvalue clipping.

    The input is symmetrised and eigendecomposed, then the eigenvalues are
    clipped according to ``eigvalue_clip``.  When clipping is active the
    returned eigenvalues are guaranteed non-negative and nonzero, so they
    are safe for logarithm, square-root and fractional-power maps
    downstream.

    Parameters
    ----------
    input : Float[Array, '... d d']
        Symmetric (assumed SPD) matrix batch, with the two trailing axes
        forming each ``d`` by ``d`` matrix.
    psi : float
        Reconditioning strength.  When ``psi > 0`` the input is perturbed
        via :func:`recondition_eigenspaces` before the eigendecomposition so
        that its VJP does not blow up at degenerate eigenvalues; the price is
        a small forward perturbation (roughly ``psi`` in absolute eigenvalue
        shift).
    key : jax.Array or None
        PRNG key forwarded to :func:`recondition_eigenspaces`; required when
        ``psi > 0``.
    eigvalue_clip : {'auto', 'none'} or float
        Clipping policy passed through to :func:`_clip_eigvals`.

    Returns
    -------
    L : Float[Array, '... d']
        Clipped eigenvalues, ascending along the trailing axis.
    Q : Float[Array, '... d d']
        Corresponding orthonormal eigenvectors as matrix columns.

    Notes
    -----
    The eigendecomposition is routed through :func:`safe_eigh`, which falls
    back to CPU transparently and restores the source device to work around
    a flaky GPU eigensolver ABI mismatch.
    """
    if psi > 0:
        input = recondition_eigenspaces(input, psi=psi, xi=psi, key=key)
    L, Q = safe_eigh(symmetric(input))
    return _clip_eigvals(L, eigvalue_clip), Q


def _recompose(
    L: Float[Array, '... d'],
    Q: Float[Array, '... d d'],
) -> Float[Array, '... d d']:
    """Reassemble :math:`Q \\operatorname{diag}(L) Q^{\\top}` and symmetrise.

    A final symmetrisation guards against small numerical drift away from
    exact symmetry introduced by the matrix products.

    Parameters
    ----------
    L : Float[Array, '... d']
        Eigenvalues to place on the reconstructed diagonal.
    Q : Float[Array, '... d d']
        Orthonormal eigenvectors as matrix columns.

    Returns
    -------
    Float[Array, '... d d']
        The symmetric matrix batch reconstructed from ``L`` and ``Q``.
    """
    return symmetric(Q @ (L[..., None] * Q.swapaxes(-1, -2)))


# ---------------------------------------------------------------------------
# Generic eigenvalue map
# ---------------------------------------------------------------------------


def symmap(
    input: Float[Array, '... d d'],
    *,
    fn: Callable[[Array], Array],
    psi: float = 0.0,
    key: Optional[jax.Array] = None,
    eigvalue_clip: EigvalueClip = 'auto',
) -> Float[Array, '... d d']:
    r"""Apply an elementwise function to the eigenvalues of an SPD batch.

    Eigendecomposes each matrix and returns
    :math:`Q \operatorname{diag}(\mathrm{fn}(\lambda)) Q^{\top}`, where
    :math:`\lambda` and :math:`Q` are the eigenvalues and eigenvectors of the
    (symmetrised) input.  This is the workhorse behind :func:`symlog`,
    :func:`symexp`, :func:`symsqrt` and :func:`sympower`.

    Parameters
    ----------
    input : Float[Array, '... d d']
        Symmetric (assumed SPD) matrix batch, with the two trailing axes
        forming each ``d`` by ``d`` matrix.
    fn : Callable[[Array], Array]
        Elementwise function applied to the eigenvalues.  It must be
        well-defined on the (clipped) eigenvalues; if ``fn`` produces a
        ``NaN`` for any eigenvalue, that ``NaN`` propagates to the result
        rather than being silently replaced with ``0``.
    psi : float, optional
        Reconditioning strength.  ``0`` (default) leaves the input untouched.
        ``psi > 0`` perturbs the input via :func:`recondition_eigenspaces`
        before the eigendecomposition, stabilising the gradient under
        near-degenerate spectra.
    key : jax.Array or None, optional
        PRNG key, required when ``psi > 0``.
    eigvalue_clip : {'auto', 'none'} or float, optional
        Eigenvalue clipping policy.  ``'auto'`` (default) clips at the
        rank-truncation threshold
        :math:`\max_i |\lambda_i| \cdot d \cdot \varepsilon`.  ``'none'``
        disables clipping (use only when the input is verified strictly
        positive definite).  A ``float`` sets an explicit threshold.

    Returns
    -------
    Float[Array, '... d d']
        The symmetric matrix batch obtained by mapping ``fn`` over the
        eigenvalues; same shape and dtype as ``input``.

    Notes
    -----
    Reverse-mode differentiation through the eigendecomposition is supported
    by JAX but degrades to NaN gradients at degenerate eigenvalues.  Use
    ``psi > 0`` to stabilise the gradient at the cost of a small forward
    perturbation.
    """
    L, Q = _eigh_with_clip(
        input,
        psi=psi,
        key=key,
        eigvalue_clip=eigvalue_clip,
    )
    return _recompose(fn(L), Q)


# ---------------------------------------------------------------------------
# Common eigenvalue maps
# ---------------------------------------------------------------------------


def symlog(
    input: Float[Array, '... d d'],
    *,
    psi: float = 0.0,
    key: Optional[jax.Array] = None,
    eigvalue_clip: EigvalueClip = 'auto',
) -> Float[Array, '... d d']:
    r"""Matrix logarithm of an SPD batch.

    Applies :math:`\log` to the eigenvalues via :func:`symmap`.  With
    ``eigvalue_clip='auto'`` (default), eigenvalues below the rank-truncation
    threshold are floored to that threshold before the logarithm, producing a
    finite negative entry rather than :math:`-\infty`.

    Parameters
    ----------
    input : Float[Array, '... d d']
        Symmetric (assumed SPD) matrix batch.
    psi : float, optional
        Reconditioning strength forwarded to :func:`symmap`.
    key : jax.Array or None, optional
        PRNG key, required when ``psi > 0``.
    eigvalue_clip : {'auto', 'none'} or float, optional
        Eigenvalue clipping policy forwarded to :func:`symmap`.

    Returns
    -------
    Float[Array, '... d d']
        The matrix logarithm of each input matrix; same shape as ``input``.
    """
    return symmap(
        input,
        fn=jnp.log,
        psi=psi,
        key=key,
        eigvalue_clip=eigvalue_clip,
    )


def symexp(
    input: Float[Array, '... d d'],
    *,
    psi: float = 0.0,
    key: Optional[jax.Array] = None,
) -> Float[Array, '... d d']:
    r"""Matrix exponential of a symmetric batch.

    Applies :math:`\exp` to the eigenvalues via :func:`symmap`.  The scalar
    exponential is well-defined for any real eigenvalue (no underflow floor
    is needed), so the clipping step is skipped and there is no
    ``eigvalue_clip`` knob.

    Parameters
    ----------
    input : Float[Array, '... d d']
        Symmetric matrix batch.
    psi : float, optional
        Reconditioning strength forwarded to :func:`symmap`.
    key : jax.Array or None, optional
        PRNG key, required when ``psi > 0``.

    Returns
    -------
    Float[Array, '... d d']
        The matrix exponential of each input matrix; same shape as
        ``input``.
    """
    return symmap(
        input,
        fn=jnp.exp,
        psi=psi,
        key=key,
        eigvalue_clip='none',
    )


# ---------------------------------------------------------------------------
# Newton-Schulz (coupled Denman-Beavers) square root -- the cuSOLVER-free driver
# ---------------------------------------------------------------------------

#: Newton-Schulz iteration count.  30 covers condition numbers up to ~1e5 (well
#: past where an SPD square root is numerically meaningful) at double precision.
_NEWTON_SCHULZ_ITERS = 30


def _newton_schulz_isqrt(
    A: Float[Array, '... d d'],
    *,
    iters: int = _NEWTON_SCHULZ_ITERS,
) -> tuple[Float[Array, '... d d'], Float[Array, '... d d']]:
    r"""Coupled square root / inverse square root of an SPD batch (matmul-only).

    Returns ``(A^{1/2}, A^{-1/2})`` via the Denman-Beavers coupled Newton-Schulz
    iteration -- matrix multiplies only, no eigendecomposition.  This is
    **cuSOLVER-free** (never lowers to the fragile solver pool) and has a finite
    reverse-mode VJP even at a repeated spectrum (there is no eigenvector
    sensitivity term to blow up).  The input is first scaled by an upper bound
    on its largest eigenvalue -- a few power iterations -- into the iteration's
    convergence region :math:`(0, 1]`.  Assumes ``A`` is SPD and reasonably
    conditioned; ridge a near-singular covariance before calling.
    """
    A = symmetric(A)
    d = A.shape[-1]
    eye = jnp.eye(d, dtype=A.dtype)
    # Upper bound on lambda_max (a few power iterations) to scale the spectrum
    # into the Newton-Schulz convergence region.
    v = jnp.ones((*A.shape[:-1], 1), A.dtype)
    for _ in range(5):
        v = A @ v
        v = v / jnp.linalg.norm(v, axis=-2, keepdims=True)
    lam = jnp.sum(v * (A @ v), axis=-2, keepdims=True)  # (..., 1, 1)
    y = A / lam
    z = jnp.broadcast_to(eye, y.shape)
    for _ in range(iters):
        t = 1.5 * eye - 0.5 * (z @ y)
        y = y @ t
        z = t @ z
    sqrt_lam = jnp.sqrt(lam)
    return symmetric(y * sqrt_lam), symmetric(z / sqrt_lam)


def symsqrt(
    input: Float[Array, '... d d'],
    *,
    inverse: bool = False,
    driver: SqrtDriver = 'eigh',
    psi: float = 0.0,
    key: Optional[jax.Array] = None,
    eigvalue_clip: EigvalueClip = 'auto',
) -> Float[Array, '... d d']:
    r"""Matrix square root (or inverse square root) of an SPD batch.

    With ``inverse=False`` returns :math:`A^{1/2}`; with ``inverse=True`` the
    inverse square root :math:`A^{-1/2}` (equivalent to
    ``sympower(A, power=-0.5)`` on the ``'eigh'`` driver, and the zero-phase
    whitening matrix).

    Two ``driver`` recipes of the same map:

    - ``'eigh'`` (default) -- the eigenvalue map via :func:`symmap`
      (:math:`Q \operatorname{diag}(\lambda^{\pm 1/2}) Q^{\top}`).  The scalar
      square root is undefined for negative reals; the default clipping floors
      near-zero eigenvalues (genuinely indefinite input surfaces as ``NaN``).
      Honours ``psi`` / ``key`` / ``eigvalue_clip``.
    - ``'newton_schulz'`` -- the matmul-only coupled iteration
      (:func:`_newton_schulz_isqrt`): **cuSOLVER-free**, GPU-friendly, and
      differentiably stable at a repeated spectrum.  ``psi`` / ``key`` /
      ``eigvalue_clip`` do not apply (there is no eigendecomposition to
      recondition or clip; ridge a near-singular input instead).

    Parameters
    ----------
    input : Float[Array, '... d d']
        Symmetric (assumed SPD) matrix batch.
    inverse : bool, optional
        Return :math:`A^{-1/2}` instead of :math:`A^{1/2}`.  Default ``False``.
    driver : {'eigh', 'newton_schulz'}, optional
        Which recipe computes the root.  Default ``'eigh'``.
    psi : float, optional
        Reconditioning strength forwarded to :func:`symmap` (``'eigh'`` only).
    key : jax.Array or None, optional
        PRNG key, required when ``psi > 0`` (``'eigh'`` only).
    eigvalue_clip : {'auto', 'none'} or float, optional
        Eigenvalue clipping policy forwarded to :func:`symmap` (``'eigh'``
        only).

    Returns
    -------
    Float[Array, '... d d']
        The (inverse) matrix square root of each input matrix; same shape as
        ``input``.
    """
    match driver:
        case 'newton_schulz':
            a_sqrt, a_isqrt = _newton_schulz_isqrt(input)
            return a_isqrt if inverse else a_sqrt
        case 'eigh':
            fn: Callable[[Array], Array] = (
                (lambda x: 1.0 / jnp.sqrt(x))
                if inverse
                else (lambda x: jnp.sqrt(x))
            )
            return symmap(
                input,
                fn=fn,
                psi=psi,
                key=key,
                eigvalue_clip=eigvalue_clip,
            )
        case _:
            assert_never(driver)


def sympower(
    input: Float[Array, '... d d'],
    *,
    power: float,
    psi: float = 0.0,
    key: Optional[jax.Array] = None,
    eigvalue_clip: EigvalueClip = 'auto',
) -> Float[Array, '... d d']:
    r"""Arbitrary real power of an SPD batch.

    Raises the eigenvalues to ``power`` via :func:`symmap`, computing
    :math:`Q \operatorname{diag}(\lambda^{p}) Q^{\top}`.  For ``power > 0``
    and SPD input the result is well-defined.  For ``power < 0`` this returns
    the inverse (negative-power) variant, and eigenvalue clipping prevents
    division by zero.

    Parameters
    ----------
    input : Float[Array, '... d d']
        Symmetric (assumed SPD) matrix batch.
    power : float
        Real exponent applied to each eigenvalue.  Common cases:
        ``-0.5`` gives the inverse square root used in tangent-space
        projection, and ``2`` gives self-multiplication (equivalent to
        ``input @ input`` for SPD input; the eigendecomposition path is more
        stable for ill-conditioned inputs but slower).
    psi : float, optional
        Reconditioning strength forwarded to :func:`symmap`.
    key : jax.Array or None, optional
        PRNG key, required when ``psi > 0``.
    eigvalue_clip : {'auto', 'none'} or float, optional
        Eigenvalue clipping policy forwarded to :func:`symmap`.

    Returns
    -------
    Float[Array, '... d d']
        Each input matrix raised to the given real power; same shape as
        ``input``.
    """
    return symmap(
        input,
        fn=lambda x: x**power,
        psi=psi,
        key=key,
        eigvalue_clip=eigvalue_clip,
    )


# ---------------------------------------------------------------------------
# Tangent-space projection (the log-Euclidean geometry)
# ---------------------------------------------------------------------------


def tangent_project_spd(
    input: Float[Array, '... d d'],
    reference: Float[Array, '... d d'],
    *,
    psi: float = 0.0,
    key: Optional[jax.Array] = None,
    eigvalue_clip: EigvalueClip = 'auto',
) -> Float[Array, '... d d']:
    r"""Map SPD matrices into the tangent space at ``reference``.

    Computes :math:`\log(R^{-1/2} X R^{-1/2})`, where :math:`R` is the
    reference and :math:`X` the input.  This is the affine-invariant
    Riemannian log map at the reference point; the log-Euclidean
    approximation :math:`\log(X) - \log(R)` is *not* what this returns.

    Use this to convert a batch of SPD matrices (e.g. subject covariance
    matrices) into a flat tangent-space representation suitable for
    downstream linear-model fitting.  :func:`cone_project_spd` inverts the
    map.

    Parameters
    ----------
    input : Float[Array, '... d d']
        Symmetric (assumed SPD) matrix batch to project into the tangent
        space.
    reference : Float[Array, '... d d']
        Symmetric (assumed SPD) reference point at which the tangent space is
        taken, broadcastable against ``input``.
    psi : float, optional
        Reconditioning strength forwarded to :func:`sympower` and
        :func:`symlog`.
    key : jax.Array or None, optional
        PRNG key, required when ``psi > 0``.
    eigvalue_clip : {'auto', 'none'} or float, optional
        Eigenvalue clipping policy forwarded to :func:`sympower` and
        :func:`symlog`.

    Returns
    -------
    Float[Array, '... d d']
        The symmetric tangent-space representation of each input matrix at
        ``reference``; same shape as ``input``.
    """
    R_inv_sqrt = sympower(
        reference,
        power=-0.5,
        psi=psi,
        key=key,
        eigvalue_clip=eigvalue_clip,
    )
    return symlog(
        R_inv_sqrt @ input @ R_inv_sqrt,
        psi=psi,
        key=key,
        eigvalue_clip=eigvalue_clip,
    )


def cone_project_spd(
    input: Float[Array, '... d d'],
    reference: Float[Array, '... d d'],
    *,
    psi: float = 0.0,
    key: Optional[jax.Array] = None,
    eigvalue_clip: EigvalueClip = 'auto',
) -> Float[Array, '... d d']:
    r"""Map tangent-space matrices back into the SPD cone at ``reference``.

    Computes :math:`R^{1/2} \exp(X) R^{1/2}`, where :math:`R` is the reference
    and :math:`X` the tangent-space input.  This is the inverse of
    :func:`tangent_project_spd`.

    Parameters
    ----------
    input : Float[Array, '... d d']
        Symmetric tangent-space matrix batch to map back into the SPD cone.
    reference : Float[Array, '... d d']
        Symmetric (assumed SPD) reference point defining the cone, the same
        one used for the forward projection, broadcastable against ``input``.
    psi : float, optional
        Reconditioning strength forwarded to :func:`symsqrt` and
        :func:`symexp`.
    key : jax.Array or None, optional
        PRNG key, required when ``psi > 0``.
    eigvalue_clip : {'auto', 'none'} or float, optional
        Eigenvalue clipping policy forwarded to :func:`symsqrt`.

    Returns
    -------
    Float[Array, '... d d']
        The reconstructed SPD matrix batch at ``reference``; same shape as
        ``input``.
    """
    R_sqrt = symsqrt(
        reference,
        psi=psi,
        key=key,
        eigvalue_clip=eigvalue_clip,
    )
    return R_sqrt @ symexp(input, psi=psi, key=key) @ R_sqrt


# ---------------------------------------------------------------------------
# Means on the SPD manifold
# ---------------------------------------------------------------------------


def mean_euclidean(
    input: Float[Array, '... batch d d'],
    *,
    axis: Union[int, Sequence[int]] = 0,
) -> Float[Array, '... d d']:
    """Euclidean mean: the plain element-wise average.

    Provided for symmetry with the SPD-specific means.  The Euclidean mean of
    SPD matrices is itself SPD (the cone is convex), so this is geometrically
    valid; it is simply *not* the geodesic mean.

    Parameters
    ----------
    input : Float[Array, '... batch d d']
        Batch of symmetric matrices, with each ``d`` by ``d`` matrix on the
        two trailing axes and one or more batch axes to average over.
    axis : int or sequence of int, optional
        Axis or axes over which to average (default ``0``).

    Returns
    -------
    Float[Array, '... d d']
        The element-wise mean matrix, with the averaged axes removed.
    """
    return jnp.mean(input, axis=axis)


def mean_log_euclidean(
    input: Float[Array, '... batch d d'],
    *,
    axis: Union[int, Sequence[int]] = 0,
    psi: float = 0.0,
    key: Optional[jax.Array] = None,
    eigvalue_clip: EigvalueClip = 'auto',
) -> Float[Array, '... d d']:
    r"""Log-Euclidean Fréchet mean, :math:`\exp(\operatorname{mean}_i \log X_i)`.

    Closed-form geodesic mean on the SPD manifold under the log-Euclidean
    metric: each matrix is mapped through :func:`symlog`, the results are
    averaged, and the mean is mapped back with :func:`symexp`.  This is cheap
    (one logarithm per matrix in the batch plus one final exponential),
    unlike the affine-invariant Fréchet mean, which requires an iterative
    fixed-point computation.

    For practical fMRI / dMRI use the log-Euclidean mean is a near-perfect
    proxy for the affine-invariant mean when the spread of the input batch is
    moderate.

    Parameters
    ----------
    input : Float[Array, '... batch d d']
        Batch of symmetric (assumed SPD) matrices, with each ``d`` by ``d``
        matrix on the two trailing axes and one or more batch axes to average
        over.
    axis : int or sequence of int, optional
        Axis or axes over which to average the matrix logarithms
        (default ``0``).
    psi : float, optional
        Reconditioning strength forwarded to :func:`symlog` and
        :func:`symexp`.
    key : jax.Array or None, optional
        PRNG key, required when ``psi > 0``.
    eigvalue_clip : {'auto', 'none'} or float, optional
        Eigenvalue clipping policy forwarded to :func:`symlog`.

    Returns
    -------
    Float[Array, '... d d']
        The log-Euclidean mean matrix, with the averaged axes removed.
    """
    logs = symlog(input, psi=psi, key=key, eigvalue_clip=eigvalue_clip)
    mean_log = jnp.mean(logs, axis=axis)
    return symexp(mean_log, psi=psi, key=key)
