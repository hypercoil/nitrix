# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Symmetric positive-definite manifold operations.

The numerical-stability story: applying ``log`` / ``sqrt`` / fractional
power to the eigenvalues of an SPD matrix is unstable when *any*
eigenvalue is at or below the machine precision floor of the
input's dtype.  The legacy ``hypercoil.functional.symmap`` had two
loosely-coupled knobs (``fill_nans`` + ``truncate_eigenvalues``) that
masked the issue rather than fixing it.

The green-field rewrite:

1. **Eigenvalue clipping is on by default.**  ``eigvalue_clip='auto'``
   clips at ``max(L) * D * eps(L.dtype)``, the standard rank-truncation
   threshold (matches ``numpy.linalg.matrix_rank``).  Below the
   threshold, eigenvalues are set to the threshold itself (not to
   zero -- ``log(threshold)`` is finite; ``log(0)`` is ``-inf``,
   contaminating downstream).
2. **No ``fill_nans``.**  If the input is genuinely indefinite,
   the operation is mathematically undefined; we surface the
   ``NaN`` rather than silently replacing it with ``0``.
3. **Differentiability hook is explicit.**  Pass ``psi > 0`` to
   recondition the input before ``eigh`` (perturbing degenerate
   eigenvalues so the eigh-VJP F-matrix doesn't blow up).  This
   is a *backward-only* concern; the forward result is the same
   either way for well-conditioned input.

What we ship:

- ``symmap`` -- generic eigenvalue-function map.
- ``symlog``, ``symexp``, ``symsqrt`` -- the common cases.
- ``sympower`` -- arbitrary real power of an SPD matrix.
- ``tangent_project_spd`` / ``cone_project_spd`` -- log-Euclidean
  tangent-space projection used in the riemannian mean / connectivity
  embedding literature.
- ``mean_log_euclidean`` -- closed-form Fréchet mean on the
  log-Euclidean metric: ``exp(mean(log(X_i)))``.  Closed form;
  cheap.
- ``mean_euclidean`` -- the trivial Euclidean mean (provided for
  symmetry with the SPD mean family).

What we don't ship at first GA (carried over from the legacy):

- ``mean_kullback`` (Kullback-Leibler mean) -- iterative;
  defer until a consumer needs it.
- The affine-invariant mean (also iterative) -- same reason.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional, Sequence, Union

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


def _clip_eigvals(
    L: Float[Array, '... d'],
    clip: EigvalueClip,
) -> Float[Array, '... d']:
    """Clip eigenvalues to a floor defined by ``clip``.

    JIT-safe: never calls ``float()`` on a tracer; threshold is
    a JAX scalar derived from ``L``.

    ``'auto'`` -> ``max(|L|) * d * eps(L.dtype)`` (rank-truncation
    threshold matching ``numpy.linalg.matrix_rank``).
    ``'none'`` -> no clipping (eigvals passed through).
    ``float`` -> explicit Python-float threshold.
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
    """Symmetrised eigendecomposition with optional clipping.

    Returns ``(eigvals_clipped, eigvecs)``.  Clipped eigvals are
    guaranteed nonzero and non-negative -- safe for log / sqrt /
    fractional power downstream.

    ``psi > 0`` reconditions the input before ``eigh`` so the
    eigh-VJP F-matrix doesn't blow up at degenerate eigenvalues.
    A small forward perturbation is the price (typically ~``psi``
    in absolute eigenvalue shift).

    Routes ``eigh`` through ``_safe_eigh`` to handle the cuSolver
    ABI mismatch on the test runner (GPU eigh broken, CPU eigh
    fine; ``_safe_eigh`` falls back transparently and restores
    the source device).
    """
    if psi > 0:
        input = recondition_eigenspaces(input, psi=psi, xi=psi, key=key)
    L, Q = safe_eigh(symmetric(input))
    return _clip_eigvals(L, eigvalue_clip), Q


def _recompose(
    L: Float[Array, '... d'],
    Q: Float[Array, '... d d'],
) -> Float[Array, '... d d']:
    """Reassemble ``Q diag(L) Q.T`` then symmetrise against drift."""
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
    """Apply an elementwise function ``fn`` to the eigenvalues of an SPD batch.

    Computes ``Q diag(fn(L)) Q.T`` where ``(L, Q) = eigh(input)``.

    Parameters
    ----------
    input
        Symmetric (assumed SPD) matrix batch.
    fn
        Elementwise function applied to the eigenvalues.  Must be
        well-defined on positive reals; if ``fn`` produces ``NaN``
        for any eigenvalue, the result will contain ``NaN`` --
        we do NOT silently replace ``NaN`` with ``0`` (this was
        the legacy ``fill_nans`` footgun).
    psi
        Reconditioning strength.  ``0`` (default) leaves the
        input untouched.  ``psi > 0`` perturbs the input via
        ``recondition_eigenspaces`` before ``eigh`` -- useful
        when the eigh-VJP must be stable under near-degenerate
        spectra (training-style use).
    key
        PRNG key, required when ``psi > 0``.
    eigvalue_clip
        ``'auto'`` (default) -- clip at the rank-truncation
        threshold ``max(|L|) * d * eps``.  ``'none'`` -- no
        clipping (use only when you've verified the input is
        strictly positive definite).  ``float`` -- explicit
        threshold.

    Notes
    -----
    Differentiability: reverse-mode through ``eigh`` is supported
    by JAX but degrades to NaN-grads at degenerate eigenvalues.
    Use ``psi > 0`` to stabilise the gradient at the cost of a
    small forward perturbation.
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
    """Matrix logarithm of an SPD batch.

    With ``eigvalue_clip='auto'`` (default), eigenvalues below the
    rank-truncation threshold are floored to that threshold before
    ``log``, producing a finite negative entry rather than ``-inf``.
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
    """Matrix exponential of a symmetric batch.

    ``exp`` is well-defined for any real eigenvalue (no underflow
    floor needed), so we skip the clip step.  No
    ``eigvalue_clip`` knob.
    """
    return symmap(
        input,
        fn=jnp.exp,
        psi=psi,
        key=key,
        eigvalue_clip='none',
    )


def symsqrt(
    input: Float[Array, '... d d'],
    *,
    psi: float = 0.0,
    key: Optional[jax.Array] = None,
    eigvalue_clip: EigvalueClip = 'auto',
) -> Float[Array, '... d d']:
    """Matrix square root of an SPD batch.

    Sqrt is undefined for negative reals; clipping handles the
    boundary.  For genuinely indefinite input the result is
    undefined and we raise via the eigvalue check.
    """
    return symmap(
        input,
        fn=jnp.sqrt,
        psi=psi,
        key=key,
        eigvalue_clip=eigvalue_clip,
    )


def sympower(
    input: Float[Array, '... d d'],
    *,
    power: float,
    psi: float = 0.0,
    key: Optional[jax.Array] = None,
    eigvalue_clip: EigvalueClip = 'auto',
) -> Float[Array, '... d d']:
    """Arbitrary real power of an SPD batch.

    For ``power > 0`` and SPD input the result is well-defined.
    For ``power < 0`` you get the inverse (negative-power) variant;
    eigenvalue clipping prevents division by zero.

    Common use cases:
    - ``power = -0.5`` -- inverse square root, for tangent-space
      projection.
    - ``power = 2`` -- self-multiplication (equivalent to
      ``input @ input`` for SPD; the eigh path is more stable for
      ill-conditioned inputs but slower).
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

    Computes ``log(R^{-1/2} X R^{-1/2})`` where ``R = reference``.
    This is the affine-invariant Riemannian log map at the
    reference point; the legacy log-Euclidean approximation
    (``log(X) - log(R)``) is *not* what this returns.

    Use this to convert a batch of SPD matrices (e.g., subject
    covariance matrices) into a flat tangent-space representation
    suitable for downstream linear-model fitting.

    Parameters / Returns
    --------------------
    See ``symmap`` for the shared knobs.
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
    """Map tangent-space matrices back into the SPD cone at ``reference``.

    Computes ``R^{1/2} exp(X) R^{1/2}`` -- the inverse of
    ``tangent_project_spd``.
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
    """Euclidean mean: plain element-wise average.

    Provided for symmetry with the SPD-specific means.  Note: the
    Euclidean mean of SPD matrices is SPD (the cone is convex),
    so this is geometrically valid; it is just *not* the
    geodesic mean.
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
    r"""Log-Euclidean Fréchet mean: ``exp(mean(log(X_i)))``.

    Closed-form geodesic mean on the SPD manifold under the
    log-Euclidean metric.  Cheap (one ``symlog`` per matrix in
    the batch + one ``symexp`` at the end); cheaper than the
    affine-invariant Fréchet mean which requires an iterative
    fixed-point computation.

    For practical fMRI / dMRI use, the log-Euclidean mean is a
    near-perfect proxy for the affine-invariant mean when the
    spread of the input batch is moderate.
    """
    logs = symlog(input, psi=psi, key=key, eigvalue_clip=eigvalue_clip)
    mean_log = jnp.mean(logs, axis=axis)
    return symexp(mean_log, psi=psi, key=key)
