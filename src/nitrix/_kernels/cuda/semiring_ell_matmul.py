# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pallas Triton implementation of ``semiring_ell_matmul``.

Status (as of first GA effort): the ELL gather pattern lowers via the
``gather`` primitive in JAX, which the Pallas Triton backend in the
pinned JAX version cannot yet handle ("Unimplemented primitive in
Pallas GPU lowering: gather").  The kernel therefore raises
``PallasELLNotTileable`` unconditionally; the public dispatcher
catches this and routes to the JAX path, emitting a
``NitrixBackendFallback`` warning on first use.

This module exists so the import contract in
``nitrix.semiring.ell._semiring_ell_matmul_pallas`` is intact, and so
that when Pallas Triton grows a gather primitive (or we move to a
hand-rolled `pl.ds`-based pattern), the swap is local to this file.

The decision to ship JAX-default for ELL on Ampere is recorded in
``bench/G0_ELL_REPORT.md`` (per IMPLEMENTATION_PLAN §3.1, the G0 gate
branch where Triton is ``>= 5× slower or unstable``).
"""
from __future__ import annotations

from jaxtyping import Array, Int, Num

from ...semiring._types import Semiring


__all__ = [
    'semiring_ell_matmul_pallas',
    'PallasELLNotTileable',
]


class PallasELLNotTileable(RuntimeError):
    '''The Pallas Triton ELL kernel cannot tile the request.

    At present this is raised unconditionally because the underlying
    ``gather`` primitive (and the axis-0 concat we would need as a
    workaround) is not lowered by Pallas Triton in the pinned JAX
    version.  When Pallas grows a workable gather we switch this to a
    proper shape-tileability check.
    '''


def semiring_ell_matmul_pallas(
    values: Num[Array, 'm kmax'],
    indices: Int[Array, 'm kmax'],
    B: Num[Array, 'n_cols ncol'],
    *,
    semiring: Semiring,
) -> Num[Array, 'm ncol']:
    raise PallasELLNotTileable(
        'semiring_ell_matmul: Pallas Triton lacks gather / axis-0 '
        'concat lowering required for the ELL streaming kernel. '
        'Falling back to the JAX path. See bench/G0_ELL_REPORT.md.'
    )
