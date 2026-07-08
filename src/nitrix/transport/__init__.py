# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
nitrix.transport -- entropic optimal transport.

Optimal-transport primitives built by *composition* over the ``LOG`` semiring
rather than a new kernel: the Sinkhorn iteration's log-domain softmin against the
cost is a ``LOG`` matrix product (:func:`~nitrix.semiring.semiring_matmul`), so
the whole family inherits the semiring's streaming, hardware-aware reduction.

- :func:`sinkhorn` -- the entropic optimal transport plan (coupling) between two
  marginals under a cost matrix, with its dual potentials.
- :func:`wasserstein_distance` -- the transport cost of that plan.
- :func:`barycentric_map` -- the barycentric projection of a plan (the pushforward
  optimal-transport map, e.g. for mesh correspondence).

Transport *distances* are numerical primitives here; composing one into a
training objective (a transport loss) stays downstream.
"""

from .sinkhorn import (
    SinkhornResult,
    barycentric_map,
    sinkhorn,
    wasserstein_distance,
)

__all__ = [
    'SinkhornResult',
    'sinkhorn',
    'wasserstein_distance',
    'barycentric_map',
]
