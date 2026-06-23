# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pallas Triton fused selective state-space scan (Mamba / S6).

**P1a placeholder.** The fused chunked block-parallel scan (state in SRAM per
tile, recompute-forward adjoint backward -- the SSM analogue of the
``numerics.ode`` adjoint) lands in suite P1b.  Until then this module exists so
the dispatcher import succeeds and the loud-fallback contract is exercised:
``selective_scan_pallas`` always raises ``PallasNotTileable`` and the public op
runs the JAX reference (which already auto-selects the parallel
``associative_scan`` on GPU).

Implementation detail: never import from ``nitrix._kernels.cuda`` directly.  Use
``nitrix.nn.ssm.selective_scan`` which handles backend dispatch and fallback
observability.
"""

from __future__ import annotations

from typing import Optional

from jaxtyping import Array, Float

__all__ = [
    'selective_scan_pallas',
    'PallasNotTileable',
]


class PallasNotTileable(RuntimeError):
    """The Pallas kernel rejected the requested shape / host.

    Caught by the dispatcher in ``nitrix.nn.ssm`` and translated into a
    ``NitrixBackendFallback`` warning (the JAX reference runs instead).
    """


def selective_scan_pallas(
    x: Float[Array, '... l d'],
    delta: Float[Array, '... l d'],
    A: Float[Array, 'd n'],
    B: Float[Array, '... l n'],
    C: Float[Array, '... l n'],
    D: Optional[Float[Array, 'd']] = None,
) -> Float[Array, '... l d']:
    """Fused selective scan (suite P1b); currently always falls back."""
    raise PallasNotTileable(
        'fused selective-scan kernel not yet implemented (suite P1b); '
        'falling back to the JAX reference.'
    )
