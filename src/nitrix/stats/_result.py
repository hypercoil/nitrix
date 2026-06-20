# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Shared JAX-pytree registration for the stats ``*Result`` types.

Every fitter in :mod:`nitrix.stats` returns a frozen-dataclass result that
doubles as a JAX pytree: the array fields are the **children** (traced, stacked
under ``vmap``, differentiated under ``grad``), while the non-array metadata --
``family`` / ``n_obs`` / ``rank`` / ``tier`` / ``corr`` -- are **static aux**
(kept out of the trace and compared *by value* for the ``jit`` cache key).

Hand-rolling ``tree_flatten`` / ``tree_unflatten`` on each result is the
add-a-field-touch-three-places footgun the audit (D1) flagged, and the
historical ``lme_fit`` ``NamedTuple`` results got it *wrong* in the opposite
direction -- a bare ``NamedTuple`` flattens **every** field as a child, so
``tier`` / ``corr`` / ``df_resid`` leaked into the trace as dynamic leaves.

:func:`register_result` is the single source of truth: name the children and
the aux, and the decorator synthesises both methods, registers the class, and
-- crucially -- **asserts the named fields exactly cover the dataclass**, so a
later field that is added but not registered fails loudly at import instead of
silently dropping out of (or into) the trace.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence, Tuple, Type, TypeVar

import jax

__all__ = ['register_result']

_T = TypeVar('_T')


def register_result(
    *,
    children: Sequence[str],
    aux: Sequence[str] = (),
) -> Callable[[Type[_T]], Type[_T]]:
    """Register a frozen-dataclass result as a JAX pytree.

    Stacks **above** ``@dataclass(frozen=True)`` (so the fields exist when the
    decorator runs)::

        @register_result(children=('coef', 'cov'), aux=('family', 'n_obs'))
        @dataclass(frozen=True)
        class GLMResult:
            coef: Float[Array, 'V p']
            cov: Float[Array, 'V p p']
            family: Family
            n_obs: int

    Parameters
    ----------
    children
        Field names that are array leaves -- traced / stacked / differentiated.
    aux
        Field names that are static metadata -- hashable, compared by value for
        the ``jit`` cache key (``family`` / ``n_obs`` / ``tier`` / ...).

    Raises
    ------
    TypeError
        If ``children`` and ``aux`` together do not *exactly* partition the
        dataclass fields (a field left out, or a name that is not a field).
    """
    child_names = tuple(children)
    aux_names = tuple(aux)

    def decorate(cls: Type[_T]) -> Type[_T]:
        declared = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
        registered = set(child_names) | set(aux_names)
        missing = declared - registered
        unknown = registered - declared
        if missing or unknown:
            raise TypeError(
                f'{cls.__name__}: register_result fields do not partition the '
                f'dataclass -- missing={sorted(missing)} '
                f'unknown={sorted(unknown)}'
            )
        if len(registered) != len(child_names) + len(aux_names):
            raise TypeError(
                f'{cls.__name__}: a field appears in both children and aux'
            )

        def tree_flatten(
            self: _T,
        ) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
            return (
                tuple(getattr(self, n) for n in child_names),
                tuple(getattr(self, n) for n in aux_names),
            )

        def tree_unflatten(
            c: Type[_T],
            aux_vals: Tuple[Any, ...],
            child_vals: Tuple[Any, ...],
        ) -> _T:
            kw = dict(zip(child_names, child_vals))
            kw.update(zip(aux_names, aux_vals))
            return c(**kw)  # type: ignore[call-arg]

        cls.tree_flatten = tree_flatten  # type: ignore[attr-defined]
        cls.tree_unflatten = classmethod(tree_unflatten)  # type: ignore[assignment]
        return jax.tree_util.register_pytree_node_class(cls)

    return decorate
