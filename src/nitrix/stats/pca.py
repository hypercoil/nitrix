# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Principal-component analysis.

:func:`pca_fit` learns an orthonormal basis ordered by explained
variance; :func:`pca_transform` and :func:`pca_inverse_transform`
project to and from that basis.  The basis is the eigendecomposition of
the (sample) covariance

.. math::

    C = X_c^{\\top} X_c / (n - 1), \\qquad C v_i = \\lambda_i v_i

with ``xc`` the centred data.  The :math:`i`-th component is the
eigenvector with the :math:`i`-th largest eigenvalue, and
:math:`\\lambda_i` is the variance captured along it.

Solvers (``solver=``):

- ``"full"`` -- exact: eigendecomposition of the :math:`(d, d)`
  covariance.
- ``"gram"`` -- exact, but via the :math:`(n, n)` Gram matrix
  :math:`X_c X_c^{\\top}` instead of the :math:`(d, d)` covariance; far
  cheaper when :math:`n \\ll d` (many features, few samples -- e.g.
  component-correlation / CompCor on a voxel-by-time matrix).  The
  components are recovered from the Gram eigenvectors as
  :math:`V = X_c^{\\top} U / \\Sigma`.  Bit-identical to ``"full"``
  (same spectrum, same sign convention).
- ``"randomized"`` -- approximate: a randomised range finder that
  recovers the top :math:`k` components in :math:`O(n d (k + p))`
  matmuls, for :math:`k` much smaller than :math:`d` (the regime where
  forming / factoring the full covariance is wasteful).  Keyed (needs a
  PRNG key).
- ``"auto"`` -- pick the cheaper exact solver: ``"gram"`` when
  :math:`n < d`, else ``"full"``.

Implementation notes:

- Every factorisation goes through the symmetric eigensolver
  (``safe_eigh``), never SVD or QR.  This is deliberate: on the
  cuSolver-affected GPU stacks SVD and QR are broken while the routed
  eigendecomposition falls back to a device where it works.  The
  randomised solver in particular uses the eigendecomposition-based
  range finder (orthonormalise via the eigendecomposition of the small
  Gram matrix) rather than the textbook QR + small-SVD, so it too stays
  portable; its only solver calls are on tiny
  :math:`(k + p) \\times (k + p)` matrices, the rest is matmuls.
- Eigenvector signs are arbitrary; they are fixed deterministically
  (largest-magnitude entry of each component made positive, the
  ``svd_flip`` convention) so a fit is reproducible across runs and
  devices.
"""

from __future__ import annotations

from typing import Literal, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..linalg._solver import safe_eigh

Solver = Literal['full', 'gram', 'randomized', 'auto']

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
    """Fix each component's sign deterministically.

    Applies the ``svd_flip`` convention: the entry of largest absolute
    value in each row is made positive, resolving the arbitrary sign of
    each eigenvector so that a fit is reproducible across runs and
    devices.

    Parameters
    ----------
    components
        ``(k, d)`` array whose rows are the principal axes.

    Returns
    -------
    Float[Array, 'k d']
        The same ``(k, d)`` array with each row scaled by
        :math:`\\pm 1` so that its largest-magnitude entry is positive.
    """
    max_abs = jnp.argmax(jnp.abs(components), axis=1)
    picked = jnp.take_along_axis(components, max_abs[:, None], axis=1)[:, 0]
    signs = jnp.where(picked < 0, -1.0, 1.0)
    return components * signs[:, None]


def _orthonormalize(g: Float[Array, 'd l']) -> Float[Array, 'd l']:
    """Orthonormalise the columns of ``g`` via the Gram eigendecomposition.

    Forms :math:`Q = g V \\Lambda^{-1/2}` from the eigendecomposition
    :math:`g^{\\top} g = V \\Lambda V^{\\top}` -- an
    eigendecomposition-based substitute for QR (unavailable on the
    cuSolver-affected GPU stacks).  The factorisation acts on the small
    :math:`(l, l)` Gram matrix.

    Parameters
    ----------
    g
        ``(d, l)`` matrix whose columns are to be orthonormalised.

    Returns
    -------
    Float[Array, 'd l']
        ``(d, l)`` matrix with orthonormal columns spanning the same
        column space as ``g``.
    """
    w, v = safe_eigh(g.T @ g)
    w = jnp.maximum(w, 1e-12)
    return (g @ v) * (1.0 / jnp.sqrt(w))[None, :]


def _pca_full(
    xc: Float[Array, 'n d'], k: int, denom: int
) -> tuple[Float[Array, 'k d'], Float[Array, 'k']]:
    """Exact PCA via eigendecomposition of the covariance matrix.

    Forms the :math:`(d, d)` covariance ``xc.T @ xc / denom``,
    eigendecomposes it with ``safe_eigh``, and returns the top ``k``
    eigenvectors and eigenvalues in descending order of variance.

    Parameters
    ----------
    xc
        ``(n, d)`` centred data matrix.
    k
        Number of leading components to return.
    denom
        Divisor applied to the second-moment matrix (typically
        :math:`n - 1`) to yield the covariance.

    Returns
    -------
    components : Float[Array, 'k d']
        ``(k, d)`` orthonormal basis, row ``i`` the ``i``-th principal
        axis.
    explained_variance : Float[Array, 'k']
        ``(k,)`` covariance eigenvalues (variance captured per
        component), descending.
    """
    cov = (xc.T @ xc) / denom
    eigvals, eigvecs = safe_eigh(cov)
    order = jnp.argsort(eigvals)[::-1]
    components = eigvecs[:, order][:, :k].T
    explained_variance = eigvals[order][:k]
    return components, explained_variance


def _pca_gram(
    xc: Float[Array, 'n d'], k: int, denom: int
) -> tuple[Float[Array, 'k d'], Float[Array, 'k']]:
    """Exact PCA via the Gram matrix (cheap when :math:`n \\ll d`).

    Eigendecomposition of the :math:`(n, n)` Gram matrix
    :math:`X_c X_c^{\\top} / (n - 1)` gives the left singular structure
    :math:`(U, \\lambda)`, with :math:`\\lambda` the same variances as
    the covariance eigenvalues; the components (right singular vectors)
    are recovered as :math:`V = X_c^{\\top} U / \\Sigma` with
    :math:`\\Sigma = \\sqrt{\\lambda (n - 1)}`.  Yields results identical
    to :func:`_pca_full` while avoiding the :math:`(d, d)` covariance.

    Parameters
    ----------
    xc
        ``(n, d)`` centred data matrix.
    k
        Number of leading components to return.
    denom
        Divisor applied to the Gram matrix (typically :math:`n - 1`);
        also used to rescale the eigenvalues into singular values.

    Returns
    -------
    components : Float[Array, 'k d']
        ``(k, d)`` orthonormal basis, row ``i`` the ``i``-th principal
        axis.
    explained_variance : Float[Array, 'k']
        ``(k,)`` variance captured per component, descending.
    """
    gram = (xc @ xc.T) / denom
    eigvals, u = safe_eigh(gram)
    order = jnp.argsort(eigvals)[::-1]
    eigvals = eigvals[order][:k]
    u = u[:, order][:, :k]  # (n, k)
    sigma = jnp.sqrt(jnp.maximum(eigvals * denom, 1e-12))
    components = (xc.T @ u / sigma[None, :]).T  # (k, d)
    return components, eigvals


def _pca_randomized(
    xc: Float[Array, 'n d'],
    k: int,
    denom: int,
    key: Array,
    n_oversamples: int,
    n_power_iterations: int,
) -> tuple[Float[Array, 'k d'], Float[Array, 'k']]:
    """Randomised PCA via the eigendecomposition-based range finder.

    Uses neither QR nor SVD.  Sketches the row space with a random
    :math:`(n, l)` projection (:math:`l = k + p`, ``p = n_oversamples``),
    refines it with ``n_power_iterations`` power iterations,
    orthonormalises through the small Gram eigendecomposition
    (:func:`_orthonormalize`), then solves the projected :math:`(l, l)`
    eigenproblem.  All large operations are matmuls; the only solver
    calls are on :math:`(l, l)` matrices.

    Parameters
    ----------
    xc
        ``(n, d)`` centred data matrix.
    k
        Number of leading components to return.
    denom
        Divisor applied to the recovered eigenvalues (typically
        :math:`n - 1`) to yield explained variances.
    key
        PRNG key seeding the random :math:`(n, l)` sketch.
    n_oversamples
        Oversampling :math:`p`; the sketch width is :math:`k + p`
        (clamped to :math:`\\min(n, d)`).  Larger values improve
        accuracy at higher cost.
    n_power_iterations
        Number of power iterations :math:`q` used to sharpen the
        spectral separation before orthonormalisation.

    Returns
    -------
    components : Float[Array, 'k d']
        ``(k, d)`` approximate orthonormal basis, row ``i`` the ``i``-th
        principal axis.
    explained_variance : Float[Array, 'k']
        ``(k,)`` approximate variance captured per component, descending.
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
        ``"full"`` (default) -- exact covariance eigendecomposition;
        ``"gram"`` -- exact, via the ``(n, n)`` Gram (cheap when
        ``n < d``); ``"randomized"`` -- randomised range finder (top ``k``
        only, cheap when ``k << d``); ``"auto"`` -- ``"gram"`` if
        ``n < d`` else ``"full"``.
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
    if n_components is not None and not 1 <= n_components <= min(n, d):
        raise ValueError(
            f'pca_fit: n_components={n_components} must be in '
            f'[1, min(n, d)={min(n, d)}] (n={n} samples, d={d} features).'
        )
    mean = jnp.mean(X, axis=0) if center else jnp.zeros(d, X.dtype)
    xc = X - mean
    denom = max(n - 1, 1)
    k = min(n, d) if n_components is None else n_components
    resolved = ('gram' if n < d else 'full') if solver == 'auto' else solver
    if resolved == 'full':
        components, explained_variance = _pca_full(xc, k, denom)
    elif resolved == 'gram':
        components, explained_variance = _pca_gram(xc, k, denom)
    elif resolved == 'randomized':
        if key is None:
            raise ValueError("solver='randomized' requires a PRNG key=.")
        components, explained_variance = _pca_randomized(
            xc, k, denom, key, n_oversamples, n_power_iterations
        )
    else:
        raise ValueError(
            f"solver={solver!r}; expected 'full', 'gram', "
            "'randomized', or 'auto'."
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

    Centres ``X`` by ``mean`` and projects onto the rows of
    ``components`` to obtain the principal-component coordinates.  Takes
    ``components`` and ``mean`` explicitly (rather than a
    :class:`PCAResult`) so a pre-fitted basis -- e.g. components loaded
    from elsewhere -- can be applied without a local :func:`pca_fit`.

    Parameters
    ----------
    X
        ``(n, d)`` data matrix (rows = samples, columns = features).
    components
        ``(k, d)`` orthonormal basis, row ``i`` the ``i``-th principal
        axis.
    mean
        ``(d,)`` feature mean subtracted before projection.

    Returns
    -------
    Float[Array, 'n k']
        ``(n, k)`` coordinates of each sample in the principal-component
        basis.
    """
    return (X - mean) @ components.T


def pca_inverse_transform(
    Z: Float[Array, 'n k'],
    components: Float[Array, 'k d'],
    mean: Float[Array, 'd'],
) -> Float[Array, 'n d']:
    """Reconstruct from PCA coordinates: ``Z @ components + mean``.

    Maps principal-component coordinates back to feature space, the
    exact inverse of :func:`pca_transform` up to the components that were
    discarded when the basis was truncated.

    Parameters
    ----------
    Z
        ``(n, k)`` principal-component coordinates.
    components
        ``(k, d)`` orthonormal basis, row ``i`` the ``i``-th principal
        axis.
    mean
        ``(d,)`` feature mean added back after reconstruction.

    Returns
    -------
    Float[Array, 'n d']
        ``(n, d)`` reconstruction in the original feature space.
    """
    return Z @ components + mean
