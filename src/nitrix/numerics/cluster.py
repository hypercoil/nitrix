# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Geometric clustering primitives.

Domain-agnostic clustering of a point cloud ``X`` of shape ``(n, p)`` -- the
*geometric / algebraic* clustering kind (minimising within-cluster spread),
distinct from the *probabilistic* mixture models (GMM) that belong in
:mod:`nitrix.stats` and the *spectral* graph clustering
(:func:`nitrix.graph.normalized_cut`) that composes this module with a graph
eigensolver.

The estimator follows the fit/apply seam: :func:`kmeans_fit` returns array
state (:class:`KMeansState`), :func:`kmeans_predict` assigns points to that
state, and the single-call :func:`kmeans` is *defined as*
``kmeans_predict(X, kmeans_fit(X, ...))`` so the two paths cannot drift. The
``similarity`` convention (euclidean / cosine / correlation) crosses the
fit->predict boundary, so it is owned here (carried as static aux on the state,
not re-specified by the caller at predict time).

The similarity working space reuses the shipped
:func:`nitrix.numerics.l2_normalize` (cosine, correlation); the centroid update
is a scatter-mean; the Lloyd sweep is a rolled ``lax.while_loop`` (compile flat
in the iteration count).
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .normalize import l2_normalize

__all__ = [
    'KMeansState',
    'Similarity',
    'kmeans',
    'kmeans_fit',
    'kmeans_predict',
]

Similarity = Literal['euclidean', 'cosine', 'correlation']
"""Assignment metric: squared-``L2`` (``'euclidean'``), cosine similarity
(``'cosine'``), or Pearson correlation (``'correlation'``, cosine of the
feature-demeaned rows)."""


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class KMeansState:
    """Fitted k-means state (the fit/apply seam's array state).

    Attributes
    ----------
    centroids : Float[Array, 'k p']
        Cluster centroids in the ``similarity`` working space (the raw feature
        space for ``'euclidean'``; the row-normalised space for ``'cosine'`` /
        ``'correlation'``).
    inertia : Float[Array, '']
        Sum of squared distances of points to their assigned centroid (in the
        working space) at convergence -- the objective used to pick the best of
        ``n_init`` restarts. Lower is better.
    similarity : str
        The assignment metric (static aux); :func:`kmeans_predict` reuses it so
        it need not be re-specified, and it is kept out of the JAX trace.
    """

    centroids: Float[Array, 'k p']
    inertia: Float[Array, '']
    similarity: str

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Array, Array], Tuple[str]]:
        return (self.centroids, self.inertia), (self.similarity,)

    @classmethod
    def tree_unflatten(
        cls, aux: Tuple[str], children: Tuple[Array, Array]
    ) -> 'KMeansState':
        centroids, inertia = children
        (similarity,) = aux
        return cls(centroids=centroids, inertia=inertia, similarity=similarity)


def _prepare(X: Float[Array, 'n p'], similarity: str) -> Float[Array, 'n p']:
    """Map ``X`` into the working space where the metric is squared-``L2``."""
    if similarity == 'euclidean':
        return X
    if similarity == 'cosine':
        return l2_normalize(X, axis=-1)
    if similarity == 'correlation':
        return l2_normalize(X - X.mean(axis=-1, keepdims=True), axis=-1)
    raise ValueError(
        f'similarity must be euclidean/cosine/correlation; got {similarity!r}.'
    )


def _sq_dists(
    Xp: Float[Array, 'n p'], centroids: Float[Array, 'k p']
) -> Float[Array, 'n k']:
    """Pairwise squared distances via the BLAS-friendly expansion."""
    return (
        (Xp**2).sum(-1)[:, None]
        - 2.0 * Xp @ centroids.T
        + (centroids**2).sum(-1)[None, :]
    )


def _lloyd(
    Xp: Float[Array, 'n p'],
    centroids0: Float[Array, 'k p'],
    max_iter: int,
    tol: float,
) -> Tuple[Float[Array, 'k p'], Float[Array, '']]:
    """One Lloyd run to convergence (rolled ``while_loop``)."""
    k = centroids0.shape[0]

    def step(
        carry: Tuple[Array, Float[Array, 'k p'], Array],
    ) -> Tuple[Array, Float[Array, 'k p'], Array]:
        i, centroids, _ = carry
        labels = jnp.argmin(_sq_dists(Xp, centroids), axis=-1)
        onehot = jax.nn.one_hot(labels, k, dtype=Xp.dtype)  # (n, k)
        counts = onehot.sum(0)  # (k,)
        new = (onehot.T @ Xp) / jnp.maximum(counts[:, None], 1.0)
        # Empty clusters keep their previous centroid (no NaN).
        new = jnp.where(counts[:, None] > 0, new, centroids)
        moved = jnp.max(jnp.abs(new - centroids))
        return i + 1, new, moved

    def cond(carry: Tuple[Array, Array, Array]) -> Array:
        i, _, moved = carry
        return (i < max_iter) & (moved > tol)

    init = (jnp.asarray(0), centroids0, jnp.asarray(jnp.inf, Xp.dtype))
    _, centroids, _ = jax.lax.while_loop(cond, step, init)
    inertia = _sq_dists(Xp, centroids).min(axis=-1).sum()
    return centroids, inertia


def kmeans_fit(
    X: Float[Array, 'n p'],
    k: int,
    *,
    key: Array,
    similarity: Similarity = 'euclidean',
    n_init: int = 1,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> KMeansState:
    """Fit k-means by Lloyd's algorithm; return the array state.

    Runs ``n_init`` restarts from random point-seeded centroids and keeps the
    lowest-inertia one. The ``fit`` half of the estimator seam.

    Parameters
    ----------
    X : Float[Array, 'n p']
        ``n`` observations in ``p`` features.
    k : int
        Number of clusters (static; ``k <= n``).
    key : Array
        A :func:`jax.random.key` for the (random point-subset) initialisation.
    similarity : {'euclidean', 'cosine', 'correlation'}, optional
        Assignment metric (see :data:`Similarity`). Default ``'euclidean'``.
    n_init : int, optional
        Number of random restarts; the best (lowest-inertia) is returned.
        Default ``1``.
    max_iter : int, optional
        Maximum Lloyd sweeps per restart. Default ``100``.
    tol : float, optional
        Convergence tolerance on the maximum centroid movement. Default
        ``1e-4``.

    Returns
    -------
    KMeansState
        The fitted centroids (in the ``similarity`` working space), the
        inertia, and the ``similarity`` convention.
    """
    Xp = _prepare(X, similarity)
    keys = jax.random.split(key, n_init)

    def one(subkey: Array) -> Tuple[Float[Array, 'k p'], Float[Array, '']]:
        idx = jax.random.choice(subkey, Xp.shape[0], (k,), replace=False)
        return _lloyd(Xp, Xp[idx], max_iter, tol)

    centroids, inertia = jax.vmap(one)(keys)
    best = jnp.argmin(inertia)
    return KMeansState(
        centroids=centroids[best],
        inertia=inertia[best],
        similarity=similarity,
    )


def kmeans_predict(
    X: Float[Array, 'n p'], state: KMeansState
) -> Int[Array, 'n']:
    """Assign points to a fitted state's nearest centroid.

    The ``apply`` half of the seam: a pure function of ``(X, state)`` (the
    ``similarity`` convention is read from ``state``).

    Parameters
    ----------
    X : Float[Array, 'n p']
        Observations to label.
    state : KMeansState
        A state from :func:`kmeans_fit`.

    Returns
    -------
    Int[Array, 'n']
        The cluster index of each observation.
    """
    Xp = _prepare(X, state.similarity)
    return jnp.argmin(_sq_dists(Xp, state.centroids), axis=-1)


def kmeans(
    X: Float[Array, 'n p'],
    k: int,
    *,
    key: Array,
    similarity: Similarity = 'euclidean',
    n_init: int = 1,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> Int[Array, 'n']:
    """Cluster ``X`` into ``k`` groups; return the per-point labels.

    The single-call convenience, *defined as*
    ``kmeans_predict(X, kmeans_fit(X, ...))`` so it cannot drift from the
    fit/apply pair. For the centroids, call :func:`kmeans_fit` and read
    :attr:`KMeansState.centroids`. See :func:`kmeans_fit` for the arguments.
    """
    return kmeans_predict(
        X,
        kmeans_fit(
            X,
            k,
            key=key,
            similarity=similarity,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
        ),
    )
