# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Functional alignment -- pairwise alignment in representation space.

The feature-space sibling of the spatial registration recipes: given two data
matrices that describe the *same* signal in two arbitrarily-rotated feature
bases (two subjects' functional connectomes, two encoders' latent spaces over a
shared sample set), recover the linear map that takes one onto the other, so the
aligned representations are directly comparable (the hyperalignment task).

**Functional alignment is a family, not a single algorithm.** The method is an
ADT (the ``Metric`` / ``TransformModel`` / ``Interpolator`` precedent):
``functional_align(..., method=ProMises())`` dispatches on it, and other
hyperalignment algorithms (ridge / regression hyperalignment, optimal-transport,
shared-response) become new ``AlignmentMethod`` implementers without touching the
public surface.  ``ProMises`` is the first -- it must not be conflated with
"functional alignment" itself.

- ``functional_align_fit`` / ``functional_align_apply`` -- the SPEC 6.5 seam:
  ``fit`` returns the map (plain arrays, a ``FunctionalAlignment``); ``apply``
  pushes any data in that feature space through it.
- ``functional_align`` -- the single-call convenience, *defined as*
  ``apply(source, fit(source, reference))`` so the two paths cannot drift.

ProMises (the first method; theory, not the legacy "empty" port)
----------------------------------------------------------------

The base case is the orthogonal Procrustes solution
``R = argmin_{R^T R = I} || source R - reference ||_F``.  The regularised case
is the **ProMises** model (Andreella & Finos 2022): a *matrix* von Mises-Fisher
(matrix-Langevin) prior on ``R``, density ``∝ exp(tr(F^T R))``, whose MAP is

    R = polar(source^T reference + F)

i.e. the prior contributes its natural-parameter matrix ``F`` **additively** to
the cross-product, and ``R`` is the orthogonal polar factor of the sum
(``linalg.orthogonal_procrustes`` with ``prior=F``).  The matrix-vMF normaliser
(a hypergeometric of a matrix argument) cancels in the MAP and is never formed,
so this depends on *no* vMF directional-statistics machinery.

This is the theory-faithful ProMises, **not** the legacy ``empty_promises``
port: it does not rotate ``source`` and ``reference`` into their own (separate)
principal bases before aligning -- that "empty" double-whitening was self-flagged
in the source as making no promises, and it injects the PCA sign/permutation
gauge ambiguity the Procrustes solve exists to resolve.

Scale.  This is the **dense** path -- ``O(p^2)`` memory / ``O(p^3)`` polar, with
``p`` the feature (voxel) count.  Correct and ideal at searchlight / parcel
scale; the **whole-brain** regime (``p ~ 1e4-1e5``) needs the *efficient
ProMises* subspace method (exploiting ``n << p``), a forthcoming
``AlignmentMethod`` sibling -- see
``docs/feature-requests/register-functional-alignment.md`` section 6.

References
----------
- Schoenemann, P. H. (1966).  A generalized solution of the orthogonal
  Procrustes problem.  Psychometrika 31(1), 1-10.
- Andreella, A. & Finos, L. (2022).  Procrustes analysis for high-dimensional
  data.  Human Brain Mapping (the ProMises model).
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
from jaxtyping import Array, Float

from ..linalg import orthogonal_procrustes

__all__ = [
    'AlignmentMethod',
    'ProMises',
    'FunctionalAlignment',
    'functional_align',
    'functional_align_fit',
    'functional_align_apply',
]


@runtime_checkable
class AlignmentMethod(Protocol):
    """A functional-alignment algorithm: fit a feature-space map.

    The single operation a method provides: given matched ``source`` /
    ``reference`` matrices, return the linear map ``R`` (``data @ R`` aligns
    ``data`` into the reference frame).  ``ProMises`` is the first implementer;
    structural conformance (not inheritance) keeps implementers plain frozen
    dataclasses (clean pytree registration, no ``Protocol`` MRO interaction).
    """

    def solve(
        self,
        source: Float[Array, '... n p'],
        reference: Float[Array, '... n p'],
        *,
        psi: float = 0.0,
        key: Optional[jax.Array] = None,
    ) -> Float[Array, '... p p']: ...


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ProMises:
    """Procrustes / ProMises alignment (the first ``AlignmentMethod``).

    Solves the orthogonal Procrustes problem, optionally regularised by a matrix
    von Mises-Fisher prior on the rotation (the ProMises MAP): the map is the
    orthogonal polar factor of ``source^T reference + prior_weight * prior``.

    Attributes
    ----------
    prior
        Optional ``(..., p, p)`` matrix-vMF natural-parameter (location) matrix
        ``F`` -- the ProMises prior on ``R``.  ``None`` (default) recovers the
        plain (maximum-likelihood) orthogonal Procrustes alignment.  A pytree
        child (differentiable).
    prior_weight
        Scalar concentration multiplying ``prior`` (the prior's strength);
        ignored when ``prior is None``.
    allow_reflection
        ``True`` (default) allows an improper rotation (a reflection) when the
        data demand one -- the usual choice for representational alignment, and
        the ``scipy.linalg.orthogonal_procrustes`` convention.  ``False``
        constrains to a proper rotation (``det R = +1``).
    """

    prior: Optional[Float[Array, '... p p']] = None
    prior_weight: float = 1.0
    allow_reflection: bool = True

    def solve(
        self,
        source: Float[Array, '... n p'],
        reference: Float[Array, '... n p'],
        *,
        psi: float = 0.0,
        key: Optional[jax.Array] = None,
    ) -> Float[Array, '... p p']:
        f = None if self.prior is None else self.prior_weight * self.prior
        return orthogonal_procrustes(
            source,
            reference,
            prior=f,
            allow_reflection=self.allow_reflection,
            psi=psi,
            key=key,
        )

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


class FunctionalAlignment(NamedTuple):
    """A fitted functional alignment: the feature-space map.

    ``matrix`` is the ``(..., p, p)`` map ``R`` such that ``data @ R`` rotates
    ``data`` (with ``p`` features) into the reference frame.  State only (plain
    arrays, SPEC 6.5) -- no module, no reference data retained.
    """

    matrix: Float[Array, '... p p']


def functional_align_fit(
    source: Float[Array, '... n p'],
    reference: Float[Array, '... n p'],
    *,
    method: AlignmentMethod = ProMises(),
    psi: float = 0.0,
    key: Optional[jax.Array] = None,
) -> FunctionalAlignment:
    """Fit the map aligning ``source`` onto ``reference``.

    The *fit* half of the SPEC 6.5 seam: dispatch to ``method`` (default
    :class:`ProMises`) and return the map as state, so it applies to ``source``
    *and* to co-registered data (co-transport) via
    :func:`functional_align_apply` without re-deriving it.

    Parameters
    ----------
    source, reference
        ``(..., n, p)`` matrices of ``n`` matched observations in a shared
        ``p``-dimensional feature space.  ``R`` rotates ``source``'s feature
        axes onto ``reference``'s.
    method
        The :class:`AlignmentMethod` (the algorithm + its hyper-parameters).
        Defaults to plain ProMises (orthogonal Procrustes, no prior).
    psi, key
        Reverse-mode reconditioning forwarded to the method's solver (stabilises
        the gradient at repeated singular values; ``key`` required when
        ``psi > 0``).

    Returns
    -------
    :class:`FunctionalAlignment`
        The fitted map ``R``.
    """
    matrix = method.solve(source, reference, psi=psi, key=key)
    return FunctionalAlignment(matrix=matrix)


def functional_align_apply(
    data: Float[Array, '... m p'],
    alignment: FunctionalAlignment,
) -> Float[Array, '... m p']:
    """Push ``data`` through a fitted alignment.

    The *apply* half of the seam: ``data @ R``.  ``data`` is any array in the
    fitted feature space -- the original ``source`` (reproducing the alignment)
    or co-registered auxiliary data (co-transport).  For the orthogonal-map
    methods this is measure-preserving (``|det R| = 1``), so no Jacobian
    correction is needed when transporting densities -- resolving the open
    ``TODO`` the legacy port flagged.
    """
    return data @ alignment.matrix


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
    ...))`` (the SPEC 6.5 fit/apply seam) so the split path cannot drift from the
    fused one.  Use the ``fit`` / ``apply`` pair directly to reuse one fitted map
    across many inputs (co-transport) or to serialise the map downstream.
    """
    alignment = functional_align_fit(
        source, reference, method=method, psi=psi, key=key
    )
    return functional_align_apply(source, alignment)
