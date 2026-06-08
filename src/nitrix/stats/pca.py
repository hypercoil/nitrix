# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Principal-component analysis.

``pca_fit`` learns an orthonormal basis ordered by explained variance;
``pca_transform`` / ``pca_inverse_transform`` project to and from that
basis.  The basis is the eigendecomposition of the (sample) covariance:

    C = Xc.T Xc / (n - 1),   C v_i = lambda_i v_i

with ``Xc`` the centred data.  The ``i``-th component is the
eigenvector with the ``i``-th largest eigenvalue, and ``lambda_i`` is
the variance captured along it.

Solvers (``solver=``):

- ``"full"`` -- exact: ``eigh`` of the ``(d, d)`` covariance.
- ``"randomized"`` -- approximate: a randomised range finder that
  recovers the top ``k`` components in ``O(n d (k + p))`` matmuls, for
  ``k`` much smaller than ``d`` (the regime where forming / factoring
  the full covariance is wasteful).  Keyed (needs a PRNG key).

Implementation notes:

- Every factorisation goes through ``eigh`` (``nitrix.linalg._solver.
  safe_eigh``), *never* ``svd`` or ``qr``.  This is deliberate: on the
  cuSolver-affected GPU stacks ``svd`` / ``qr`` are broken while the
  routed ``eigh`` falls back to a device where it works.  The
  randomised solver in particular uses the eigh-based range finder
  (orthonormalise via the eigendecomposition of the small Gram matrix)
  rather than the textbook QR + small-SVD, so it too stays portable;
  its only solver calls are on tiny ``(k + p) x (k + p)`` matrices, the
  rest is matmuls.
- Eigenvector signs are arbitrary; we fix them deterministically
  (largest-magnitude entry of each component made positive, the
  ``svd_flip`` convention) so a fit is reproducible across runs and
  devices.

Roadmap (the family will grow along the ``solver=`` seam, mirroring the
extremal-eigensolver dispatcher): a ``"gram"`` solver (eigh of the
*smaller* of ``X Xᵀ`` / ``Xᵀ X`` -- the efficient path when
``n << d``, e.g. component-correlation / CompCor on voxel time series),
and a sparse-``X`` path routed to the sparse eig solvers.
"""

from __future__ import annotations

from typing import Literal, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..linalg._solver import safe_eigh

Solver = Literal['full', 'randomized']

__all__ = [
    'PCAResult',
    'pca_fit',
    'pca_transform',
    'pca_inverse_transform',
]


class PCAResult(NamedTuple):
    """Fitted PCA basis (a NamedTuple of arrays, not a module).

    Attributes
    ----------
    components
        ``(k, d)`` orthonormal basis, row ``i`` the ``i``-th principal
        axis (descending explained variance).
    mean
        ``(d,)`` feature mean subtracted before projection.
    explained_variance
        ``(k,)`` variance captured by each component (the eigenvalues
        of the covariance, descending).
    """

    components: Float[Array, 'k d']
    mean: Float[Array, 'd']
    explained_variance: Float[Array, 'k']


def _svd_flip_sign(components: Float[Array, 'k d']) -> Float[Array, 'k d']:
    """Deterministic sign: make each row's largest-|.| entry positive."""
    max_abs = jnp.argmax(jnp.abs(components), axis=1)
    picked = jnp.take_along_axis(components, max_abs[:, None], axis=1)[:, 0]
    signs = jnp.where(picked < 0, -1.0, 1.0)
    return components * signs[:, None]


def _orthonormalize(g: Float[Array, 'd l']) -> Float[Array, 'd l']:
    """Orthonormalise the columns of ``g`` via the Gram eigendecomposition.

    ``Q = g V Λ^{-1/2}`` from ``gᵀ g = V Λ Vᵀ`` -- an ``eigh``-based
    substitute for QR (unavailable on the cuSolver-affected GPU stacks).
    The eigh is on the small ``(l, l)`` Gram matrix.
    """
    w, v = safe_eigh(g.T @ g)
    w = jnp.maximum(w, 1e-12)
    return (g @ v) * (1.0 / jnp.sqrt(w))[None, :]


def _pca_full(
    xc: Float[Array, 'n d'], k: int, denom: int
) -> tuple[Float[Array, 'k d'], Float[Array, 'k']]:
    """Exact PCA: ``eigh`` of the ``(d, d)`` covariance."""
    cov = (xc.T @ xc) / denom
    eigvals, eigvecs = safe_eigh(cov)
    order = jnp.argsort(eigvals)[::-1]
    components = eigvecs[:, order][:, :k].T
    explained_variance = eigvals[order][:k]
    return components, explained_variance


def _pca_randomized(
    xc: Float[Array, 'n d'],
    k: int,
    denom: int,
    key: Array,
    n_oversamples: int,
    n_power_iterations: int,
) -> tuple[Float[Array, 'k d'], Float[Array, 'k']]:
    """Randomised PCA via the eigh-based range finder (no QR / SVD).

    Sketches the row space with a random ``(n, l)`` projection
    (``l = k + n_oversamples``), refines it with power iterations,
    orthonormalises through the small Gram eigh, then solves the
    projected ``(l, l)`` eigenproblem.  All large operations are
    matmuls; the only solver calls are on ``(l, l)`` matrices.
    """
    n, d = xc.shape
    ell = min(k + n_oversamples, min(n, d))
    omega = jax.random.normal(key, (n, ell), dtype=xc.dtype)
    g = xc.T @ omega  # (d, l) -- spans (approximately) the top row space
    for _ in range(n_power_iterations):
        g = _orthonormalize(xc.T @ (xc @ g))
    q = _orthonormalize(g)  # (d, l)
    b = xc @ q  # (n, l)
    s2, wb = safe_eigh(b.T @ b)  # (l,), (l, l)
    order = jnp.argsort(s2)[::-1]
    v = (q @ wb)[:, order]  # (d, l) approx right singular vectors of xc
    components = v[:, :k].T
    explained_variance = s2[order][:k] / denom
    return components, explained_variance


def pca_fit(
    X: Float[Array, 'n d'],
    *,
    n_components: Optional[int] = None,
    center: bool = True,
    solver: Solver = 'full',
    n_oversamples: int = 10,
    n_power_iterations: int = 2,
    key: Optional[Array] = None,
) -> PCAResult:
    """Fit a PCA basis to ``X`` (rows = samples, columns = features).

    Parameters
    ----------
    X
        Data matrix ``(n_samples, n_features)``.
    n_components
        Number of components to keep.  ``None`` (default) keeps
        ``min(n_samples, n_features)`` -- the components with
        potentially non-zero variance.  Must be a static (Python)
        ``int`` so the output shape is known at trace time.
    center
        Subtract the feature mean before fitting (default ``True``).
        When ``False`` the returned ``mean`` is zeros and the
        decomposition is of the uncentred second-moment matrix.
    solver
        ``"full"`` (default) for the exact covariance eigendecomposition,
        or ``"randomized"`` for the randomised range finder (top ``k``
        components only, far cheaper when ``k << d``).
    n_oversamples, n_power_iterations
        Randomised-solver controls: the oversampling ``p`` (sketch width
        ``k + p``) and the number of power iterations ``q`` (more ``q``
        sharpens the spectral separation, improving accuracy on slowly
        decaying spectra).  Ignored for ``solver="full"``.
    key
        PRNG key.  Required for ``solver="randomized"``; ignored
        otherwise.

    Returns
    -------
    PCAResult
        ``components`` ``(k, d)``, ``mean`` ``(d,)``,
        ``explained_variance`` ``(k,)``.
    """
    n, d = X.shape
    mean = jnp.mean(X, axis=0) if center else jnp.zeros(d, X.dtype)
    xc = X - mean
    denom = max(n - 1, 1)
    k = min(n, d) if n_components is None else n_components
    if solver == 'full':
        components, explained_variance = _pca_full(xc, k, denom)
    elif solver == 'randomized':
        if key is None:
            raise ValueError("solver='randomized' requires a PRNG key=.")
        components, explained_variance = _pca_randomized(
            xc, k, denom, key, n_oversamples, n_power_iterations
        )
    else:
        raise ValueError(
            f"solver={solver!r}; expected 'full' or 'randomized'."
        )
    return PCAResult(
        components=_svd_flip_sign(components),
        mean=mean,
        explained_variance=explained_variance,
    )


def pca_transform(
    X: Float[Array, 'n d'],
    components: Float[Array, 'k d'],
    mean: Float[Array, 'd'],
) -> Float[Array, 'n k']:
    """Project ``X`` onto a PCA basis: ``(X - mean) @ components.T``.

    Takes ``components`` / ``mean`` explicitly (rather than a
    :class:`PCAResult`) so a *pre-fitted* basis -- e.g. components
    loaded from elsewhere -- can be applied without a local
    :func:`pca_fit`.
    """
    return (X - mean) @ components.T


def pca_inverse_transform(
    Z: Float[Array, 'n k'],
    components: Float[Array, 'k d'],
    mean: Float[Array, 'd'],
) -> Float[Array, 'n d']:
    """Reconstruct from PCA coordinates: ``Z @ components + mean``.

    Exact (up to the discarded components) inverse of
    :func:`pca_transform`.
    """
    return Z @ components + mean
