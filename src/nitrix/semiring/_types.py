# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Semiring algebra Protocols and value classes.

This module fixes the type shape that the rest of the substrate code
expresses against. The relaxed-vs-strict split follows SPEC_UPDATE §3.1:

- ``Semiring`` (relaxed) is the default; makes no associativity promise on
  ``binary_op``.  Kernels that fix a sequential reduction order
  (streaming ``semiring_matmul``, ``semiring_ell_matmul``, etc.) accept
  ``Semiring``.
- ``StrictSemiring`` is a structural subtype that additionally asserts
  ``binary_op`` is associative and (where required) distributes over the
  monoid combine.  Functions whose correctness depends on free
  reassociation (tree reductions, multi-stage cross-device reductions)
  must annotate ``StrictSemiring``.

Users opt into strict via ``strict=True`` on the ``Semiring`` constructor;
this returns a ``StrictSemiring``.  No structural property is *checked*
at runtime -- the flag is the user's signed declaration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    '''Associative reduction with identity, carrying optional pytree state.

    Required (informal) properties:

    - ``init(shape, dtype)`` returns the identity element broadcast to
      ``shape``.  The returned object is the same pytree shape used by
      ``update`` and ``merge``.
    - ``update(acc, value)`` folds a value into the accumulator.  Used
      inside the K loop of the streaming kernels.
    - ``merge(a, b)`` is associative and has ``init`` as identity.  Used
      between fully-reduced sub-blocks (e.g. when joining results across
      Pallas blocks, or in cross-device tree reductions).  ``merge`` and
      ``update`` must agree: ``update(acc, v)`` must equal ``merge(acc,
      update(init, v))``.
    - ``finalize(acc)`` projects the (possibly auxiliary) state to the
      user-facing array.  Applied once at the end of the contraction.

    The state ``S`` may be any pytree; this is what enables online
    numerically-stable reductions such as the ``(running_max, sum_exp)``
    state used by ``LogSumExpMonoid``.
    '''

    def init(
        self, shape: tuple[int, ...], dtype: jnp.dtype
    ) -> S: ...

    def update(
        self, acc: S, value: Num[Array, '*shape']
    ) -> S: ...

    def merge(self, a: S, b: S) -> S: ...

    def finalize(self, acc: S) -> Num[Array, '*shape']: ...


@runtime_checkable
class Semigroup(Protocol):
    '''A binary operation supporting NumPy-style broadcasting.

    Used as the semiring's ``(*)``: the per-``(i, k, j)`` combine inside
    the contraction.  Need not be commutative.  Need not be associative
    (it is the *outer* `+` reduction that imposes the algebraic
    structure; see ``Monoid``).

    The kernel only relies on the broadcast call
    ``combine(a_col, b_row) -> value``, so ``combine`` should be a pure
    JAX function with no captured state.
    '''

    def combine(
        self, a: Num[Array, '*shape'], b: Num[Array, '*shape']
    ) -> Num[Array, '*shape']: ...


SemiringMatmulVJP = Callable[
    [Tuple[Any, ...], Any],  # residuals, g_out
    Tuple[Any, Any],         # grad_A, grad_B
]

SemiringELLMatmulVJP = Callable[
    [Tuple[Any, ...], Any],  # residuals, g_out
    Tuple[Any, Any],         # grad_values, grad_B  (indices is non-diff)
]


@dataclass(frozen=True)
class Semiring(Generic[S]):
    '''The relaxed semiring: a frozen ``(monoid, binary_op, identity, name)`` tuple.

    Parameters
    ----------
    monoid
        Monoid implementing the reduction ``(+)``.
    binary_op
        Semigroup implementing the per-element combine ``(*)``.
    identity
        A scalar (or ``None``) representing the value with which the
        unused / padded entries of an ELL row are filled prior to the
        reduction.  For ``REAL`` this is ``0``; for ``TROPICAL_MAX_PLUS``
        it is ``-inf``; for ``BOOLEAN`` it is ``False``.  Distinct from
        ``Monoid.init``: ``identity`` is the value that, when passed
        through ``binary_op`` and reduced, leaves the accumulator
        unchanged.  Setting this to ``None`` disables ELL pad-fill (the
        caller is responsible for sentinel handling).
    name
        Short string used in fallback warnings and debug output.
    matmul_vjp
        Optional callable implementing the backward rule for
        ``semiring_matmul`` under this algebra.  Receives a
        ``(residuals, g_out)`` pair and returns ``(grad_A, grad_B)``.
        Built-in algebras supply this; user-defined algebras default
        to ``None``, in which case ``jax.grad`` over
        ``semiring_matmul`` raises with a clear message.  Per
        SPEC_UPDATE §3.1 differentiability vocabulary, the per-algebra
        backwards live in ``nitrix.semiring._backward``.
    ell_matmul_vjp
        Same shape, for ``semiring_ell_matmul``.  ``indices`` is
        non-differentiable; the rule returns gradients for the
        ``values`` and ``B`` arguments only.

    Notes
    -----
    Construct via ``Semiring(...)`` for the relaxed variant or
    ``StrictSemiring(...)`` directly for the strict variant.
    ``Semiring(...).as_strict()`` and ``StrictSemiring(...).as_relaxed()``
    convert between them; the conversion is a re-tagging, not a check.

    User-defined algebras may subclass this dataclass only to attach
    extra metadata (e.g., a custom gradient hint); the kernel surface
    relies on the ``monoid`` / ``binary_op`` / ``identity`` / ``name``
    fields.
    '''

    monoid: Monoid[S]
    binary_op: Semigroup
    identity: Any = None
    name: str = 'semiring'
    matmul_vjp: Optional[SemiringMatmulVJP] = None
    ell_matmul_vjp: Optional[SemiringELLMatmulVJP] = None

    def as_strict(self) -> 'StrictSemiring[S]':
        '''Re-tag this semiring as a ``StrictSemiring``.

        The caller is asserting that ``binary_op`` is associative and
        distributes over ``monoid.merge`` for the dtypes in use.  No
        check is performed; the type system carries the obligation.
        '''
        return StrictSemiring(
            monoid=self.monoid,
            binary_op=self.binary_op,
            identity=self.identity,
            name=self.name,
            matmul_vjp=self.matmul_vjp,
            ell_matmul_vjp=self.ell_matmul_vjp,
        )

    def __post_init__(self):
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
    '''A ``Semiring`` whose ``binary_op`` is associative and distributes
    over ``monoid.merge``.

    The strict subtype unlocks reduction reorderings that are unsafe in
    general: tree reductions, sub-block recombination, cross-device
    reduce-scatter.  Functions whose correctness depends on those
    transforms annotate their argument as ``StrictSemiring`` so the
    type-checker rejects callers passing a relaxed ``Semiring``.

    The Pallas streaming kernels accept either type: they fix the
    reduction order, so they do not rely on associativity.  ``Semiring``
    in their signatures (rather than ``StrictSemiring``) reflects this.
    '''

    def as_relaxed(self) -> 'Semiring[S]':
        return Semiring(
            monoid=self.monoid,
            binary_op=self.binary_op,
            identity=self.identity,
            name=self.name,
            matmul_vjp=self.matmul_vjp,
            ell_matmul_vjp=self.ell_matmul_vjp,
        )


# ---------------------------------------------------------------------------
# Sentinel for "this algebra has no gradient".
# ---------------------------------------------------------------------------


_NO_GRADIENT = object()
