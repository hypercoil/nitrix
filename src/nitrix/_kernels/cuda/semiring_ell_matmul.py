# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pallas Triton implementation of ``semiring_ell_matmul``.

The ELL gather pattern lowers via the ``gather`` primitive in JAX,
which the Pallas Triton backend in the pinned JAX version cannot yet
handle ("Unimplemented primitive in Pallas GPU lowering: gather").
The kernel therefore raises :class:`PallasELLNotTileable`
unconditionally; the public dispatcher catches this and routes to the
JAX path, emitting a
:class:`~nitrix._internal.backend.NitrixBackendFallback` warning on
first use.

This module exists so the import contract in
``nitrix.semiring.ell._semiring_ell_matmul_pallas`` is intact, and so
that when Pallas Triton grows a gather primitive (or we move to a
hand-rolled ``pl.ds``-based pattern), the swap is local to this file.

The library ships the JAX path as the default for ELL on Ampere
because the Triton path was measured to be at least :math:`5\\times`
slower or unstable.
"""

from __future__ import annotations

from typing import Any

from jaxtyping import Array, Int, Num

from ...semiring._types import Semiring

__all__ = [
    'semiring_ell_matmul_pallas',
    'PallasELLNotTileable',
]


class PallasELLNotTileable(RuntimeError):
    """The Pallas Triton ELL kernel cannot tile the request.

    At present this is raised unconditionally because the underlying
    ``gather`` primitive (and the axis-0 concat we would need as a
    workaround) is not lowered by Pallas Triton in the pinned JAX
    version.  When Pallas grows a workable gather we switch this to a
    proper shape-tileability check.
    """


def semiring_ell_matmul_pallas(
    values: Num[Array, 'm kmax'],
    indices: Int[Array, 'm kmax'],
    B: Num[Array, 'n_cols ncol'],
    *,
    semiring: Semiring[Any],
) -> Num[Array, 'm ncol']:
    """Pallas Triton entry point for the ELL semiring matmul.

    This computes, in principle, the semiring product of an ELL-format
    sparse matrix (given as its packed ``values`` and column
    ``indices``) with a dense right operand ``B``. In the pinned JAX
    version the Pallas Triton backend cannot lower the ``gather``
    primitive the ELL streaming pattern requires, so the kernel is not
    yet realised: it always raises :class:`PallasELLNotTileable`, which
    the public dispatcher intercepts to fall back to the JAX path.

    Parameters
    ----------
    values : Num[Array, 'm kmax']
        Packed non-zero values of the ELL sparse operand, one row of up
        to ``kmax`` entries per output row.
    indices : Int[Array, 'm kmax']
        Column indices into ``B`` aligned with ``values``, sharing its
        ``(m, kmax)`` layout.
    B : Num[Array, 'n_cols ncol']
        Dense right operand of the product.
    semiring : Semiring
        The algebra supplying the "multiply" and "add" operations that
        replace ordinary multiplication and summation in the product.

    Returns
    -------
    Num[Array, 'm ncol']
        The dense result of the semiring product. Never returned in the
        current implementation, which always raises instead.

    Raises
    ------
    PallasELLNotTileable
        Always, because Pallas Triton lacks the gather lowering the
        kernel needs.
    """
    raise PallasELLNotTileable(
        'semiring_ell_matmul: Pallas Triton lacks gather / axis-0 '
        'concat lowering required for the ELL streaming kernel. '
        'Falling back to the JAX path. See bench/G0_ELL_REPORT.md.'
    )
