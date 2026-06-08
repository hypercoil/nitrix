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

Implementation notes:

- The basis is recovered via ``eigh`` of the ``(d, d)`` covariance,
  *not* an SVD of the data matrix.  This is deliberate: on the
  cuSolver-affected GPU stacks ``svd`` is broken while the routed
  ``eigh`` (``nitrix.linalg._solver.safe_eigh``) falls back to a
  device where it works, so the eigen route is the portable one.  (For
  ``d >> n`` an SVD / Gram-matrix route would be cheaper; the
  covariance route is chosen for portability, and the fit is a
  one-off.)
- Eigenvector signs are arbitrary; we fix them deterministically
  (largest-magnitude entry of each component made positive, the
  ``svd_flip`` convention) so a fit is reproducible across runs and
  devices.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..linalg._solver import safe_eigh

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


def pca_fit(
    X: Float[Array, 'n d'],
    *,
    n_components: Optional[int] = None,
    center: bool = True,
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

    Returns
    -------
    PCAResult
        ``components`` ``(k, d)``, ``mean`` ``(d,)``,
        ``explained_variance`` ``(k,)``.
    """
    n, d = X.shape
    mean = jnp.mean(X, axis=0) if center else jnp.zeros(X.shape[1], X.dtype)
    xc = X - mean
    denom = max(n - 1, 1)
    cov = (xc.T @ xc) / denom
    # ``eigh`` returns ascending eigenvalues; reverse for descending.
    eigvals, eigvecs = safe_eigh(cov)
    order = jnp.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    k = min(n, d) if n_components is None else n_components
    components = _svd_flip_sign(eigvecs[:, :k].T)
    explained_variance = eigvals[:k]
    return PCAResult(
        components=components,
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
