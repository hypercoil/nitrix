# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pallas Triton fused scaled-dot-product (flash) attention.

**Phase-1 placeholder.** The fused kernel -- forked from
``jax.experimental.pallas.ops.gpu.attention`` and extended with
additive-bias / boolean-mask tiles threaded through the online softmax and
a ``d_bias`` term in the backward -- lands in suite Phase 2.  Until then
this module exists so the dispatcher import succeeds and the loud-fallback
contract is exercised: ``scaled_dot_product_attention_pallas`` always
raises ``PallasNotTileable`` and the public op runs the JAX reference.

Implementation detail: never import from ``nitrix._kernels.cuda`` directly.
Use ``nitrix.nn.attention.scaled_dot_product_attention`` which handles
backend dispatch and fallback observability.
"""

from __future__ import annotations

from typing import Optional

from jaxtyping import Array, Bool, Float

__all__ = [
    'scaled_dot_product_attention_pallas',
    'PallasNotTileable',
]


class PallasNotTileable(RuntimeError):
    """The Pallas kernel rejected the requested shape / host.

    Caught by the dispatcher in ``nitrix.nn.attention`` and translated into
    a ``NitrixBackendFallback`` warning (the JAX reference runs instead).
    """


def scaled_dot_product_attention_pallas(
    q: Float[Array, '... h s d'],
    k: Float[Array, '... h t d'],
    v: Float[Array, '... h t d_v'],
    *,
    scale: float,
    bias: Optional[Float[Array, '... h s t']] = None,
    mask: Optional[Bool[Array, '... h s t']] = None,
    causal: bool = False,
) -> Float[Array, '... h s d_v']:
    """Fused flash attention (suite Phase 2); currently always falls back."""
    raise PallasNotTileable(
        'fused attention kernel not yet implemented (suite Phase 2); '
        'falling back to the JAX reference.'
    )
