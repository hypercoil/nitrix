# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Functional alignment -- pairwise alignment in representation space.

The feature-space sibling of the spatial registration recipes: given two data
matrices that describe the *same* signal in two arbitrarily-rotated feature
bases (two subjects' functional connectomes, two encoders' latent spaces over a
shared sample set), recover the linear map that takes one onto the other, so the
aligned representations are directly comparable (the hyperalignment task).

**Functional alignment is a family, not a single algorithm.** The method is an
ADT (the :class:`Metric` / :class:`TransformModel` / interpolator precedent):
``functional_align(..., method=ProMises())`` dispatches on it, and other
hyperalignment algorithms (ridge / regression hyperalignment, optimal-transport,
shared-response) become new :class:`AlignmentMethod` implementers without
touching the public surface.  :class:`ProMises` (dense) and
:class:`EfficientProMises` (subspace) are the first two; neither is to be
conflated with "functional alignment" itself.

- :func:`functional_align_fit` / :func:`functional_align_apply` -- the fit/apply
  seam: ``fit`` returns the fitted map (a :class:`FunctionalAlignment`, plain
  arrays); ``apply`` pushes any data in that feature space through it.
- :func:`functional_align` -- the single-call convenience, *defined as*
  ``apply(source, fit(source, reference))`` so the two paths cannot drift.

The fitted map is itself an ADT.  :class:`DenseAlignment` carries the
orthogonal map as an explicit :math:`(p, p)` matrix (searchlight / parcel
scale); :class:`SubspaceAlignment` carries it *implicitly* as a semi-orthogonal
basis and a small reduced rotation (whole-brain scale), so the :math:`(p, p)`
map is never materialised.

ProMises -- the model (theory, not the legacy "empty" port)
-----------------------------------------------------------

Let :math:`X` (source) and :math:`M` (reference) be :math:`(n, p)` matrices of
:math:`n` matched observations in a shared :math:`p`-dimensional feature space.
The base case is the orthogonal Procrustes solution

.. math::

    R = \operatorname*{arg\,min}_{R^\top R = I} \lVert X R - M \rVert_F
      = U V^\top,\qquad U \Sigma V^\top = \operatorname{svd}(X^\top M).

The regularised **ProMises** model places a *matrix* von Mises--Fisher
(matrix-Langevin) prior on :math:`R`, density
:math:`\propto \exp\{\operatorname{tr}(F^\top R)\}`, whose maximum-a-posteriori
estimate is

.. math::

    \hat R = \operatorname{polar}(X^\top M + k F),

i.e. the prior contributes its natural-parameter (location) matrix :math:`F`
**additively** to the cross-product, scaled by the concentration :math:`k`, and
:math:`\hat R` is the orthogonal polar factor of the sum
(:func:`~nitrix.linalg.orthogonal_procrustes` with ``prior``).  The matrix-vMF
normaliser (a hypergeometric of a matrix argument) cancels in the MAP and is
never formed, so this depends on *no* vMF directional-statistics machinery.

This is the theory-faithful ProMises, **not** the legacy ``empty_promises``
port: it does not rotate ``source`` and ``reference`` into their own (separate)
principal bases before aligning -- that "empty" double-whitening was self-flagged
in the source as making no promises, and it injects the PCA sign/permutation
gauge ambiguity the Procrustes solve exists to resolve.

Efficient ProMises -- the whole-brain (subspace) method
-------------------------------------------------------

Forming :math:`X^\top M` and the polar factor of a :math:`(p, p)` matrix is
:math:`O(p^2)` memory / :math:`O(p^3)` time -- fine at searchlight / parcel
scale, intractable for whole-brain hyperalignment (:math:`p \sim 10^4-10^5`).
But :math:`X^\top M` has rank at most :math:`n`, so the alignment lives in a
subspace of dimension :math:`\le n \ll p`.  Take the thin singular
decompositions :math:`X = L_X S_X Q_X^\top` and :math:`M = L_M S_M Q_M^\top`,
with the semi-orthogonal :math:`Q_X, Q_M` (each :math:`(p, l)`, :math:`l \le n`,
:math:`Q^\top Q = I_l`) spanning the source's and reference's row spaces.  Then
the full-space problem reduces without loss to an :math:`(l, l)` one (Theorem 3,
under isotropic column covariance),

.. math::

    \hat R^\star
      = \operatorname{polar}\bigl((X Q_X)^\top (M Q_M) + k F^\star\bigr),
    \qquad F^\star = Q_X^\top F Q_M,

and the full map is the partial isometry :math:`R = Q_X \hat R^\star Q_M^\top`,
represented implicitly by :math:`(Q_X, Q_M, \hat R^\star)` and applied as
``data @ Q_X @ R_star @ Q_M.T`` -- the :math:`(p, p)` object is never formed.
The plain (``prior=None``) path is :math:`O(p n^2)` time / :math:`O(p n)`
memory.

When the reduction is lossless -- ``n_components`` at least each matrix's row
rank, and ``prior=None`` -- :math:`Q_X \hat R^\star Q_M^\top` equals the dense
polar factor on the identified subspace, so ``EfficientProMises`` and
``ProMises`` agree exactly on the aligned data.  Fewer components trade that
equivalence for a smaller, cheaper reduced problem.  The prior enters only
through its projection :math:`Q_X^\top F Q_M` (Lemma 5): an *informative* prior
supplied as a dense :math:`(p, p)` matrix costs :math:`O(p^2 l)` to project (the
:math:`(p, p)` matrix must exist), whereas ``prior=None`` stays fully
:math:`O(p n^2)`.  A tractable informative-prior construction (projecting voxel
coordinates and rebuilding a spatial kernel in the reduced space, never forming
the :math:`(p, p)` prior) is a modelling policy that belongs downstream.

The lift back uses the *reference* basis :math:`Q_M`, not the source basis --
``data @ Q_X @ R_star @ Q_M.T``.  The legacy implementation reconstructs the
aligned data with the source basis on both sides (:math:`Q_X \hat R^\star
Q_X^\top`); that keeps the result in the source's row space and does **not**
reproduce the dense ProMises MAP, whereas the :math:`Q_M`-lift does (verified to
machine precision against the dense polar factor, and against the reference R
implementation).

References
----------
- Schoenemann, P. H. (1966).  A generalized solution of the orthogonal
  Procrustes problem.  Psychometrika 31(1), 1-10.
  https://doi.org/10.1007/BF02289451
- Andreella, A. & Finos, L. (2022).  Procrustes analysis for high-dimensional
  data.  Psychometrika 87, 1422-1438 (the ProMises model, Theorem 2, and the
  Efficient ProMises reduction, Theorem 3 / Lemma 5).
  https://arxiv.org/abs/2008.04631
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..linalg import orthogonal_procrustes, symmetric
from ..linalg._solver import safe_eigh

__all__ = [
    'AlignmentMethod',
    'FunctionalAlignment',
    'DenseAlignment',
    'SubspaceAlignment',
    'ProMises',
    'EfficientProMises',
    'functional_align',
    'functional_align_fit',
    'functional_align_apply',
]


def _mT(x: Float[Array, '... a b']) -> Float[Array, '... b a']:
    """Transpose the trailing two axes (batched matrix transpose)."""
    return jnp.swapaxes(x, -1, -2)


# --------------------------------------------------------------------------- #
# The fitted map (an ADT): dense or implicit-in-a-subspace.                    #
# --------------------------------------------------------------------------- #


@runtime_checkable
class FunctionalAlignment(Protocol):
    """A fitted functional alignment: the feature-space map, and how to apply it.

    State only (plain arrays; a pytree) -- no module, no reference data
    retained.  :meth:`transform` co-transports any array in the fitted feature
    space through the map (``data @ R``); :func:`functional_align_apply` is the
    seam-level spelling of the same operation.  :class:`DenseAlignment` and
    :class:`SubspaceAlignment` are the two implementers (explicit vs implicit
    map); structural conformance (not inheritance) keeps them plain
    ``NamedTuple`` states.
    """

    def transform(
        self, data: Float[Array, '... m p']
    ) -> Float[Array, '... m p']: ...


class DenseAlignment(NamedTuple):
    """A fitted map materialised as a dense orthogonal matrix.

    The searchlight / parcel-scale representation (:class:`ProMises`).

    Attributes
    ----------
    matrix
        The :math:`(..., p, p)` orthogonal map :math:`R`; ``data @ R`` rotates
        ``data`` (with :math:`p` features) into the reference frame.
    """

    matrix: Float[Array, '... p p']

    def transform(
        self, data: Float[Array, '... m p']
    ) -> Float[Array, '... m p']:
        """Push ``data`` through the map: ``data @ matrix``."""
        return data @ self.matrix


class SubspaceAlignment(NamedTuple):
    """A fitted map represented implicitly in a low-dimensional subspace.

    The whole-brain-scale representation (:class:`EfficientProMises`).  The full
    :math:`(p, p)` map :math:`R = Q_X R^\\star Q_M^\\top` (a rank-:math:`l`
    partial isometry from the source's to the reference's row space) is **never
    materialised** -- it is carried by the two semi-orthogonal bases and the
    small reduced rotation, and applied as
    ``((data @ source_basis) @ reduced) @ reference_basis.T``.

    Attributes
    ----------
    source_basis
        The :math:`(..., p, l)` semi-orthogonal basis :math:`Q_X`
        (:math:`Q_X^\\top Q_X = I_l`) for the source's row space -- the input is
        projected onto it.
    reference_basis
        The :math:`(..., p, l)` semi-orthogonal basis :math:`Q_M` for the
        reference's row space -- the rotated result is lifted back through it.
    reduced
        The :math:`(..., l, l)` orthogonal rotation :math:`R^\\star` solving the
        Procrustes problem in the reduced coordinates.
    """

    source_basis: Float[Array, '... p l']
    reference_basis: Float[Array, '... p l']
    reduced: Float[Array, '... l l']

    def transform(
        self, data: Float[Array, '... m p']
    ) -> Float[Array, '... m p']:
        """Push ``data`` through the implicit map without forming ``R``.

        Right-to-left: project onto the source subspace (``data @ Q_X``), rotate
        (``@ R_star``), lift into the reference subspace (``@ Q_M.T``).
        Components of ``data`` outside the source's fitted row space are
        annihilated -- the map acts only on the identified :math:`l`-dimensional
        signal subspace.
        """
        return ((data @ self.source_basis) @ self.reduced) @ _mT(
            self.reference_basis
        )


# --------------------------------------------------------------------------- #
# The alignment method (an ADT): the algorithm + its hyper-parameters.        #
# --------------------------------------------------------------------------- #


@runtime_checkable
class AlignmentMethod(Protocol):
    """A functional-alignment algorithm: fit a feature-space map.

    The single operation a method provides: given matched ``source`` /
    ``reference`` matrices, return the fitted :class:`FunctionalAlignment`.
    :class:`ProMises` / :class:`EfficientProMises` are the first implementers;
    structural conformance (not inheritance) keeps implementers plain frozen
    dataclasses (clean pytree registration, no ``Protocol`` MRO interaction).
    """

    def fit(
        self,
        source: Float[Array, '... n p'],
        reference: Float[Array, '... n p'],
        *,
        psi: float = 0.0,
        key: Optional[jax.Array] = None,
    ) -> FunctionalAlignment: ...


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ProMises:
    """Dense Procrustes / ProMises alignment (a :class:`AlignmentMethod`).

    Solves the orthogonal Procrustes problem, optionally regularised by a matrix
    von Mises--Fisher prior on the rotation (the ProMises MAP): the map is the
    orthogonal polar factor of ``source^T reference + prior_weight * prior``,
    materialised as a dense :math:`(p, p)` :class:`DenseAlignment`.  Ideal at
    searchlight / parcel scale; for whole-brain :math:`p` use
    :class:`EfficientProMises`.

    Attributes
    ----------
    prior
        Optional :math:`(..., p, p)` matrix-vMF natural-parameter (location)
        matrix :math:`F` -- the ProMises prior on :math:`R`.  ``None`` (default)
        recovers the plain (maximum-likelihood) orthogonal Procrustes alignment.
        A pytree child (differentiable).
    prior_weight
        Scalar concentration :math:`k` multiplying ``prior`` (the prior's
        strength); ignored when ``prior is None``.
    allow_reflection
        ``True`` (default) allows an improper rotation (a reflection) when the
        data demand one -- the usual choice for representational alignment, and
        the ``scipy.linalg.orthogonal_procrustes`` convention.  ``False``
        constrains to a proper rotation (``det R = +1``).
    """

    prior: Optional[Float[Array, '... p p']] = None
    prior_weight: float = 1.0
    allow_reflection: bool = True

    def fit(
        self,
        source: Float[Array, '... n p'],
        reference: Float[Array, '... n p'],
        *,
        psi: float = 0.0,
        key: Optional[jax.Array] = None,
    ) -> FunctionalAlignment:
        f = None if self.prior is None else self.prior_weight * self.prior
        matrix = orthogonal_procrustes(
            source,
            reference,
            prior=f,
            allow_reflection=self.allow_reflection,
            psi=psi,
            key=key,
        )
        return DenseAlignment(matrix=matrix)

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Optional[Array]], Tuple[float, bool]]:
        return (self.prior,), (self.prior_weight, self.allow_reflection)

    @classmethod
    def tree_unflatten(
        cls,
        aux: Tuple[float, bool],
        children: Tuple[Any, ...],
    ) -> 'ProMises':
        prior_weight, allow_reflection = aux
        return cls(
            prior=children[0],
            prior_weight=prior_weight,
            allow_reflection=allow_reflection,
        )


def _right_basis(
    x: Float[Array, '... n p'],
    n_components: Optional[int],
) -> Float[Array, '... p l']:
    """Semi-orthogonal basis for the row space of a single matrix.

    The top-``l`` right singular vectors :math:`Q` of :math:`X = L S Q^\\top`
    (:math:`Q^\\top Q = I_l`), computed cuSOLVER-free from the small
    :math:`(n, n)` Gram :math:`X X^\\top` (``safe_eigh``, as in
    :func:`~nitrix.linalg.orthogonal_procrustes`) -- never a :math:`(p, p)`
    object.  ``l = min(n_components, n, p)`` (or ``min(n, p)`` when
    ``n_components is None``, which spans the full row space and makes the
    reduction lossless).
    """
    n, p = x.shape[-2], x.shape[-1]
    if n_components is None:
        n_components = min(n, p)
    rank = min(n_components, n, p)
    # X = L S Q^T (thin); X X^T = L S^2 L^T, so eigh of the small Gram gives the
    # left factor L and the singular values; Q = X^T L S^{-1}.
    gram = symmetric(x @ _mT(x))  # (..., n, n)
    s2, u = safe_eigh(gram)  # eigenvalues ascending
    s2 = s2[..., ::-1][..., :rank]  # descending -> top-l
    u = u[..., ::-1][..., :rank]
    eps = jnp.finfo(x.dtype).eps
    floor = jnp.maximum(s2[..., :1], 0.0) * n * eps  # rank-trunc threshold
    s = jnp.sqrt(jnp.where(s2 < floor, floor, s2))
    q = (_mT(x) @ u) / s[..., None, :]  # (..., p, l), orthonormal columns
    return q


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EfficientProMises:
    """Subspace (whole-brain) ProMises alignment (a :class:`AlignmentMethod`).

    Reduces the :math:`(p, p)` Procrustes problem to an :math:`(l, l)` one
    (:math:`l \\le n`) in per-matrix semi-orthogonal bases for the row spaces of
    ``source`` / ``reference`` (Theorem 3), then returns the map implicitly as a
    :class:`SubspaceAlignment` -- the :math:`(p, p)` rotation is never formed.
    Equivalent to :class:`ProMises` when the reduction is lossless
    (``n_components`` at least each matrix's row rank, ``prior=None``); cheaper
    and rank-``l``-restricted otherwise.

    Attributes
    ----------
    prior
        Optional :math:`(..., p, p)` matrix-vMF location matrix :math:`F`,
        projected into the subspace as :math:`V^\\top F V` (Lemma 5) before the
        reduced solve.  ``None`` (default) is plain Procrustes and stays fully
        :math:`O(p n^2)`; a dense ``prior`` costs :math:`O(p^2 l)` to project
        (the :math:`(p, p)` matrix must exist).  A pytree child.
    prior_weight
        Scalar concentration :math:`k` multiplying ``prior``; ignored when
        ``prior is None``.
    allow_reflection
        Whether the reduced rotation may be improper (a reflection within the
        subspace).  See :class:`ProMises`.
    n_components
        The reduced dimension :math:`l` (number of leading singular directions
        of each matrix's row space to keep).  ``None`` (default) keeps
        :math:`\\min(n, p)` -- the full row space, making the reduction
        lossless.  A static (non-traced) integer.
    """

    prior: Optional[Float[Array, '... p p']] = None
    prior_weight: float = 1.0
    allow_reflection: bool = True
    n_components: Optional[int] = None

    def fit(
        self,
        source: Float[Array, '... n p'],
        reference: Float[Array, '... n p'],
        *,
        psi: float = 0.0,
        key: Optional[jax.Array] = None,
    ) -> FunctionalAlignment:
        qx = _right_basis(source, self.n_components)  # (..., p, l)  Q_X
        qm = _right_basis(reference, self.n_components)  # (..., p, l)  Q_M
        a = source @ qx  # (..., n, l) reduced source
        b = reference @ qm  # (..., n, l) reduced reference
        f = None
        if self.prior is not None:
            # Lemma 5: project the (p, p) prior into the subspace, Q_X^T F Q_M.
            f = _mT(qx) @ (self.prior_weight * self.prior) @ qm  # (..., l, l)
        reduced = orthogonal_procrustes(
            a,
            b,
            prior=f,
            allow_reflection=self.allow_reflection,
            psi=psi,
            key=key,
        )
        return SubspaceAlignment(
            source_basis=qx, reference_basis=qm, reduced=reduced
        )

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Optional[Array]], Tuple[float, bool, Optional[int]]]:
        return (
            (self.prior,),
            (self.prior_weight, self.allow_reflection, self.n_components),
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux: Tuple[float, bool, Optional[int]],
        children: Tuple[Any, ...],
    ) -> 'EfficientProMises':
        prior_weight, allow_reflection, n_components = aux
        return cls(
            prior=children[0],
            prior_weight=prior_weight,
            allow_reflection=allow_reflection,
            n_components=n_components,
        )


# --------------------------------------------------------------------------- #
# The fit / apply seam.                                                        #
# --------------------------------------------------------------------------- #


def functional_align_fit(
    source: Float[Array, '... n p'],
    reference: Float[Array, '... n p'],
    *,
    method: AlignmentMethod = ProMises(),
    psi: float = 0.0,
    key: Optional[jax.Array] = None,
) -> FunctionalAlignment:
    """Fit the map aligning ``source`` onto ``reference``.

    The *fit* half of the fit/apply seam: dispatch to ``method`` (default
    :class:`ProMises`) and return the fitted map as state, so it applies to
    ``source`` *and* to co-registered data (co-transport) via
    :func:`functional_align_apply` without re-deriving it.

    Parameters
    ----------
    source, reference
        ``(..., n, p)`` matrices of ``n`` matched observations in a shared
        ``p``-dimensional feature space.  The fitted map rotates ``source``'s
        feature axes onto ``reference``'s.
    method
        The :class:`AlignmentMethod` (the algorithm + its hyper-parameters).
        Defaults to plain ProMises (orthogonal Procrustes, no prior).
    psi, key
        Reverse-mode reconditioning forwarded to the method's solver (stabilises
        the gradient at repeated singular values; ``key`` required when
        ``psi > 0``).

    Returns
    -------
    FunctionalAlignment
        The fitted map -- a :class:`DenseAlignment` (from :class:`ProMises`) or
        a :class:`SubspaceAlignment` (from :class:`EfficientProMises`).
    """
    return method.fit(source, reference, psi=psi, key=key)


def functional_align_apply(
    data: Float[Array, '... m p'],
    alignment: FunctionalAlignment,
) -> Float[Array, '... m p']:
    """Push ``data`` through a fitted alignment.

    The *apply* half of the seam: ``alignment.transform(data)`` (``data @ R``
    for the dense map; the implicit product for the subspace map).  ``data`` is
    any array in the fitted feature space -- the original ``source`` (reproducing
    the alignment) or co-registered auxiliary data (co-transport).  For the
    orthogonal-map methods this is measure-preserving (``|det R| = 1`` on the
    map's support), so no Jacobian correction is needed when transporting
    densities -- resolving the open ``TODO`` the legacy port flagged.
    """
    return alignment.transform(data)


def functional_align(
    source: Float[Array, '... n p'],
    reference: Float[Array, '... n p'],
    *,
    method: AlignmentMethod = ProMises(),
    psi: float = 0.0,
    key: Optional[jax.Array] = None,
) -> Float[Array, '... n p']:
    """Align ``source`` onto ``reference`` and return the mapped ``source``.

    The single-call convenience, *defined as*
    ``functional_align_apply(source, functional_align_fit(source, reference,
    ...))`` (the fit/apply seam) so the split path cannot drift from the fused
    one.  Use the ``fit`` / ``apply`` pair directly to reuse one fitted map
    across many inputs (co-transport) or to serialise the map downstream.
    """
    alignment = functional_align_fit(
        source, reference, method=method, psi=psi, key=key
    )
    return functional_align_apply(source, alignment)
