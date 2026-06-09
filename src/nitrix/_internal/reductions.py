# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
The one reduction surface for nitrix score kernels.

Per ``SPEC_UPDATE_v0.5 §1`` (the score-kernel ↔ scalarisation boundary):
a nitrix score kernel returns the **unreduced** tensor by default and may
expose a *flat, non-compositional* reduction as a leaf convenience. This is
that leaf -- the single ``Reduction`` literal and ``reduce`` helper shared
by ``metrics``, ``stats``, and ``register.regulariser`` (collapsing three
divergent private ``_reduce`` copies into one).

It is deliberately **not** the scalarisation system. nimox's ``scalarise``
is a higher-order combinator (``mean_scalarise(inner=…)``,
function → function) that composes, weights whole terms, and combines
objectives; ``reduce`` here is the innermost value → value leaf that such a
composition bottoms out at. Read this as "the minimal subset of
``scalarise``," not a rival.

The one weighted reduction nitrix owns is the **domain-mask weighted mean**
``Σ(w·x) / Σw`` (``reduction='mean'`` with a ``weight``): a per-element
foreground / validity mask is part of the *measurement* (an LNCC / Dice /
masked-token CE over background is numerically meaningless), distinct from
the per-term *objective* weighting that belongs to nimox. ``weight`` here is
a domain mask, never an objective weight.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = ['Reduction', 'reduce']

Reduction = Literal['none', 'sum', 'mean']
_AxisArg = Union[int, Tuple[int, ...]]


def reduce(
    values: Float[Array, '...'],
    *,
    axis: Optional[_AxisArg] = None,
    weight: Optional[Float[Array, '...']] = None,
    reduction: Reduction = 'mean',
) -> Float[Array, '...']:
    """Apply an optional domain-mask ``weight`` and reduce over ``axis``.

    Parameters
    ----------
    values
        The unreduced score tensor.
    axis
        Axis (or axes) to reduce over; ``None`` (default) reduces over all.
    weight
        Optional per-element non-negative **domain mask** (same shape as
        ``values`` or broadcastable). When given, it multiplies ``values``,
        and ``reduction='mean'`` becomes the weighted mean ``Σ(w·x)/Σw``.
    reduction
        ``'none'`` (return the -- optionally masked -- tensor), ``'sum'``,
        or ``'mean'``.

    Returns
    -------
    The reduced tensor (scalar for ``axis=None``), or the ``values`` tensor
    (masked, if ``weight`` was given) for ``reduction='none'``.
    """
    if weight is not None:
        values = values * weight
    if reduction == 'none':
        return values
    if reduction == 'sum':
        return values.sum(axis)
    if reduction == 'mean':
        if weight is not None:
            return values.sum(axis) / jnp.maximum(weight.sum(axis), 1e-12)
        return values.mean(axis)
    raise ValueError(
        f'reduction={reduction!r}; expected "none", "sum", or "mean".'
    )
