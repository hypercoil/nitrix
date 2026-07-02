# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Semiring algebra Protocols and value classes.

This module fixes the type shape that the rest of the substrate code
expresses against. It defines a relaxed-versus-strict split:

- :class:`Semiring` (relaxed) is the default; it makes no associativity
  promise on ``binary_op``. Kernels that fix a sequential reduction order
  (streaming :func:`semiring_matmul`, :func:`semiring_ell_matmul`, etc.)
  accept a :class:`Semiring`.
- :class:`StrictSemiring` is a structural subtype that additionally
  asserts ``binary_op`` is associative and (where required) distributes
  over the monoid combine. Functions whose correctness depends on free
  reassociation (tree reductions, multi-stage cross-device reductions)
  must annotate :class:`StrictSemiring`.

Users opt into the strict variant by constructing a
:class:`StrictSemiring` directly (or calling
:meth:`Semiring.as_strict`). No structural property is *checked* at
runtime -- the choice is the user's signed declaration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    runtime_checkable,
)

import jax.numpy as jnp
from jaxtyping import Array, Num

__all__ = [
    'Monoid',
    'Semigroup',
    'Semiring',
    'StrictSemiring',
]


# State pytree carried through the K loop.
S = TypeVar('S')


@runtime_checkable
class Monoid(Protocol[S]):
    """Associative reduction with identity, carrying optional pytree state.

    A monoid supplies the additive ``(+)`` structure of a semiring: an
    identity element and an associative combine over an accumulator that
    may itself be an arbitrary pytree of auxiliary state.

    Required (informal) properties:

    - :meth:`init` returns the identity element broadcast to ``shape``.
      The returned object is the same pytree shape used by :meth:`update`
      and :meth:`merge`.
    - :meth:`update` folds a single value into the accumulator. Used
      inside the K loop of the streaming kernels.
    - :meth:`merge` is associative and has :meth:`init` as identity. Used
      between fully-reduced sub-blocks (e.g. when joining results across
      Pallas blocks, or in cross-device tree reductions). :meth:`merge`
      and :meth:`update` must agree: ``update(acc, v)`` must equal
      ``merge(acc, update(init, v))``.
    - :meth:`finalize` projects the (possibly auxiliary) state to the
      user-facing array. Applied once at the end of the contraction.

    The state ``S`` may be any pytree; this is what enables online
    numerically-stable reductions such as the ``(running_max, sum_exp)``
    state used by a log-sum-exp monoid.
    """

    def init(self, shape: tuple[int, ...], dtype: jnp.dtype[Any]) -> S:
        """Return the identity element broadcast to ``shape``.

        Parameters
        ----------
        shape
            Target shape of the accumulator array (or of the array leaves
            of the state pytree).
        dtype
            Element dtype of the identity.

        Returns
        -------
        S
            The identity element of the monoid, as a pytree of the same
            shape carried by :meth:`update` and :meth:`merge`.
        """
        ...

    def update(self, acc: S, value: Num[Array, '*shape']) -> S:
        """Fold a single value into the accumulator.

        Parameters
        ----------
        acc
            Current accumulator state.
        value
            Value to fold in, broadcastable to the accumulator shape.

        Returns
        -------
        S
            The updated accumulator state.
        """
        ...

    def merge(self, a: S, b: S) -> S:
        """Combine two fully-reduced accumulator states.

        Associative, with :meth:`init` as identity, so that partially
        reduced sub-blocks may be recombined in any order.

        Parameters
        ----------
        a, b
            Accumulator states to combine.

        Returns
        -------
        S
            The combined accumulator state.
        """
        ...

    def finalize(self, acc: S) -> Num[Array, '*shape']:
        """Project the accumulator state to the user-facing array.

        Applied once at the end of the contraction to discard any
        auxiliary state (such as a running maximum) and return the
        reduced value.

        Parameters
        ----------
        acc
            Final accumulator state after all values have been folded in.

        Returns
        -------
        Num[Array, '*shape']
            The reduced result array.
        """
        ...


@runtime_checkable
class Semigroup(Protocol):
    """A binary operation supporting NumPy-style broadcasting.

    Used as the semiring's ``(*)``: the per-:math:`(i, k, j)` combine
    inside the contraction. Need not be commutative. Need not be
    associative (it is the *outer* :math:`+` reduction that imposes the
    algebraic structure; see :class:`Monoid`).

    The kernel only relies on the broadcast call
    ``combine(a_col, b_row) -> value``, so :meth:`combine` should be a
    pure JAX function with no captured state.
    """

    def combine(
        self, a: Num[Array, '*shape'], b: Num[Array, '*shape']
    ) -> Num[Array, '*shape']:
        """Combine two operands elementwise with broadcasting.

        Parameters
        ----------
        a, b
            Operands, broadcastable against one another under
            NumPy-style rules.

        Returns
        -------
        Num[Array, '*shape']
            The elementwise combination, of the broadcast shape.
        """
        ...


SemiringMatmulVJP = Callable[
    [Tuple[Any, ...], Any],  # residuals, g_out
    Tuple[Any, Any],  # grad_A, grad_B
]

SemiringELLMatmulVJP = Callable[
    [Tuple[Any, ...], Any],  # residuals, g_out
    Tuple[Any, Any],  # grad_values, grad_B  (indices is non-diff)
]


@dataclass(frozen=True)
class Semiring(Generic[S]):
    """The relaxed semiring: a frozen bundle of the algebraic operations.

    A semiring pairs a :class:`Monoid` (the additive reduction ``(+)``)
    with a :class:`Semigroup` (the multiplicative combine ``(*)``), plus
    the identity/annihilator scalars and the optional differentiation
    rules needed by the streaming kernels. The relaxed variant makes no
    associativity promise on ``binary_op``.

    Parameters
    ----------
    monoid
        :class:`Monoid` implementing the reduction ``(+)``.
    binary_op
        :class:`Semigroup` implementing the per-element combine ``(*)``.
    identity
        A scalar (or ``None``) representing the value with which the
        unused / padded entries of an ELL row are filled prior to the
        reduction. For :data:`REAL` this is ``0``; for
        :data:`TROPICAL_MAX_PLUS` it is :math:`-\\infty`; for
        :data:`BOOLEAN` it is ``False``. Distinct from :meth:`Monoid.init`:
        ``identity`` is the value that, when passed through ``binary_op``
        and reduced, leaves the accumulator unchanged. Setting this to
        ``None`` disables ELL pad-fill (the caller is responsible for
        sentinel handling).
    annihilator
        The ``(*)``-annihilator: the scalar :math:`z` with
        ``binary_op(z, b)`` equal to the monoid identity for every
        :math:`b`, i.e. the value an ELL edge must carry to be a true
        no-op under masking (:func:`nitrix.sparse.ell_mask`). This is
        **not** the same concept as ``identity`` (the additive / monoid
        identity), though for most algebras the two coincide:
        :data:`REAL` ``0``, :data:`LOG` / :data:`TROPICAL_MAX_PLUS`
        :math:`-\\infty`, :data:`TROPICAL_MIN_PLUS` :math:`+\\infty`,
        :data:`BOOLEAN` ``False``. :data:`EUCLIDEAN` is the exception:
        :math:`(a - b)^2` has **no** annihilator, so ``annihilator=None``
        -- and masking a :data:`EUCLIDEAN` neighbourhood by a value is
        impossible (drop the columns structurally instead). Defaults to
        ``None``; a user-defined algebra whose monoid identity and
        annihilator differ should set this explicitly so value-masking is
        correct.
    name
        Short string used in fallback warnings and debug output.
        Defaults to ``'semiring'``.
    matmul_vjp
        Optional callable implementing the backward rule for
        :func:`semiring_matmul` under this algebra. Receives a
        ``(residuals, g_out)`` pair and returns ``(grad_A, grad_B)``.
        Built-in algebras supply this; user-defined algebras default to
        ``None``, in which case ``jax.grad`` over :func:`semiring_matmul`
        raises with a clear message. Defaults to ``None``.
    ell_matmul_vjp
        Same shape, for :func:`semiring_ell_matmul`. ``indices`` is
        non-differentiable; the rule returns gradients for the ``values``
        and ``B`` arguments only. Defaults to ``None``.

    Notes
    -----
    Construct via ``Semiring(...)`` for the relaxed variant or
    ``StrictSemiring(...)`` directly for the strict variant.
    :meth:`Semiring.as_strict` and :meth:`StrictSemiring.as_relaxed`
    convert between them; the conversion is a re-tagging, not a check.

    User-defined algebras may subclass this dataclass only to attach
    extra metadata (e.g., a custom gradient hint); the kernel surface
    relies on the ``monoid`` / ``binary_op`` / ``identity`` / ``name``
    fields.
    """

    monoid: Monoid[S]
    binary_op: Semigroup
    identity: Any = None
    annihilator: Any = None
    name: str = 'semiring'
    matmul_vjp: Optional[SemiringMatmulVJP] = None
    ell_matmul_vjp: Optional[SemiringELLMatmulVJP] = None

    def as_strict(self) -> 'StrictSemiring[S]':
        """Re-tag this semiring as a :class:`StrictSemiring`.

        The caller is asserting that ``binary_op`` is associative and
        distributes over ``monoid.merge`` for the dtypes in use. No check
        is performed; the type system carries the obligation.

        Returns
        -------
        StrictSemiring[S]
            A :class:`StrictSemiring` carrying the same fields as this
            semiring.
        """
        return StrictSemiring(
            monoid=self.monoid,
            binary_op=self.binary_op,
            identity=self.identity,
            annihilator=self.annihilator,
            name=self.name,
            matmul_vjp=self.matmul_vjp,
            ell_matmul_vjp=self.ell_matmul_vjp,
        )

    def __post_init__(self) -> None:
        # Validate Protocol shape so a misshapen algebra fails at
        # construction rather than deep inside a Pallas trace.
        if not isinstance(self.monoid, Monoid):
            raise TypeError(
                f'Semiring {self.name!r}: monoid={self.monoid!r} does '
                'not satisfy the Monoid Protocol '
                '(init / update / merge / finalize).'
            )
        if not isinstance(self.binary_op, Semigroup):
            raise TypeError(
                f'Semiring {self.name!r}: binary_op={self.binary_op!r} '
                'does not satisfy the Semigroup Protocol (combine).'
            )


@dataclass(frozen=True)
class StrictSemiring(Semiring[S]):
    """A semiring whose ``binary_op`` is associative and distributive.

    This is the strict variant of :class:`Semiring`: it additionally
    asserts that ``binary_op`` is associative and distributes over
    ``monoid.merge``. The strict subtype unlocks reduction reorderings
    that are unsafe in general: tree reductions, sub-block recombination,
    cross-device reduce-scatter. Functions whose correctness depends on
    those transforms annotate their argument as :class:`StrictSemiring`
    so the type-checker rejects callers passing a relaxed
    :class:`Semiring`.

    The Pallas streaming kernels accept either type: they fix the
    reduction order, so they do not rely on associativity. Annotating
    them with :class:`Semiring` (rather than :class:`StrictSemiring`)
    reflects this.
    """

    def as_relaxed(self) -> 'Semiring[S]':
        """Re-tag this strict semiring as a relaxed :class:`Semiring`.

        Drops the associativity/distributivity obligation, yielding a
        plain :class:`Semiring` carrying the same fields.

        Returns
        -------
        Semiring[S]
            A relaxed :class:`Semiring` with identical fields.
        """
        return Semiring(
            monoid=self.monoid,
            binary_op=self.binary_op,
            identity=self.identity,
            annihilator=self.annihilator,
            name=self.name,
            matmul_vjp=self.matmul_vjp,
            ell_matmul_vjp=self.ell_matmul_vjp,
        )


# ---------------------------------------------------------------------------
# Sentinel for "this algebra has no gradient".
# ---------------------------------------------------------------------------


_NO_GRADIENT = object()
