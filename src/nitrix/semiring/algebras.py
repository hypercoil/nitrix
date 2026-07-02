# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Built-in algebras shipped with ``nitrix.semiring``.

Each algebra is exported as a module-level :class:`Semiring` (relaxed) or
:class:`StrictSemiring` instance:

================== ===================== ============== ========================
Algebra            Type                  Identity       Backward
================== ===================== ============== ========================
REAL               StrictSemiring        ``0``          transpose-matmul
LOG                StrictSemiring        ``-inf``       softmax-weighted
TROPICAL_MAX_PLUS  StrictSemiring        ``-inf``       argmax-gather (subgrad)
TROPICAL_MIN_PLUS  StrictSemiring        ``+inf``       argmin-gather (subgrad)
BOOLEAN            StrictSemiring        ``False``      n/a (raises)
EUCLIDEAN          Semiring (relaxed)    ``0``          normalised-diff w/ guard
================== ===================== ============== ========================

The pre-built instances are intended for module-level use::

    from nitrix.semiring import REAL, LOG, TROPICAL_MAX_PLUS
    semiring_matmul(A, B, semiring=LOG)

A user can compose their own :class:`Semiring` by combining a
:class:`Monoid` with a :class:`Semigroup`; the kernel substrate does not
care whether the algebra is built-in or user-defined as long as the
protocol shape is honoured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float, Num

from ._backward import (
    boolean_ell_matmul_vjp,
    boolean_matmul_vjp,
    euclidean_ell_matmul_vjp,
    euclidean_matmul_vjp,
    log_ell_matmul_vjp,
    log_matmul_vjp,
    real_ell_matmul_vjp,
    real_matmul_vjp,
    tropical_max_plus_ell_matmul_vjp,
    tropical_max_plus_matmul_vjp,
    tropical_min_plus_ell_matmul_vjp,
    tropical_min_plus_matmul_vjp,
)
from ._types import Semiring, StrictSemiring

# ---------------------------------------------------------------------------
# Monoids
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SumMonoid:
    """The :math:`(\\mathbb{R}, +, 0)` monoid; :meth:`finalize` is identity."""

    def init(
        self, shape: tuple[int, ...], dtype: jnp.dtype[Any]
    ) -> Num[Array, '*shape']:
        return jnp.zeros(shape, dtype=dtype)

    def update(
        self, acc: Num[Array, '*shape'], value: Num[Array, '*shape']
    ) -> Num[Array, '*shape']:
        return acc + value

    def merge(
        self, a: Num[Array, '*shape'], b: Num[Array, '*shape']
    ) -> Num[Array, '*shape']:
        return a + b

    def finalize(self, acc: Num[Array, '*shape']) -> Num[Array, '*shape']:
        return acc


@dataclass(frozen=True)
class _MaxMonoid:
    """The :math:`(\\mathbb{R} \\cup \\{-\\infty\\}, \\max, -\\infty)` monoid."""

    def init(
        self, shape: tuple[int, ...], dtype: jnp.dtype[Any]
    ) -> Num[Array, '*shape']:
        return jnp.full(shape, -jnp.inf, dtype=dtype)

    def update(
        self, acc: Num[Array, '*shape'], value: Num[Array, '*shape']
    ) -> Num[Array, '*shape']:
        return jnp.maximum(acc, value)

    def merge(
        self, a: Num[Array, '*shape'], b: Num[Array, '*shape']
    ) -> Num[Array, '*shape']:
        return jnp.maximum(a, b)

    def finalize(self, acc: Num[Array, '*shape']) -> Num[Array, '*shape']:
        return acc


@dataclass(frozen=True)
class _MinMonoid:
    """The :math:`(\\mathbb{R} \\cup \\{+\\infty\\}, \\min, +\\infty)` monoid."""

    def init(
        self, shape: tuple[int, ...], dtype: jnp.dtype[Any]
    ) -> Num[Array, '*shape']:
        return jnp.full(shape, jnp.inf, dtype=dtype)

    def update(
        self, acc: Num[Array, '*shape'], value: Num[Array, '*shape']
    ) -> Num[Array, '*shape']:
        return jnp.minimum(acc, value)

    def merge(
        self, a: Num[Array, '*shape'], b: Num[Array, '*shape']
    ) -> Num[Array, '*shape']:
        return jnp.minimum(a, b)

    def finalize(self, acc: Num[Array, '*shape']) -> Num[Array, '*shape']:
        return acc


@dataclass(frozen=True)
class _OrMonoid:
    """Boolean OR monoid; identity is ``False``."""

    def init(
        self, shape: tuple[int, ...], dtype: jnp.dtype[Any]
    ) -> Num[Array, '*shape']:
        return jnp.zeros(shape, dtype=dtype)

    def update(
        self, acc: Num[Array, '*shape'], value: Num[Array, '*shape']
    ) -> Num[Array, '*shape']:
        return jnp.logical_or(acc, value)

    def merge(
        self, a: Num[Array, '*shape'], b: Num[Array, '*shape']
    ) -> Num[Array, '*shape']:
        return jnp.logical_or(a, b)

    def finalize(self, acc: Num[Array, '*shape']) -> Num[Array, '*shape']:
        return acc


@dataclass(frozen=True)
class _SumThenSqrtMonoid:
    """Sum monoid whose :meth:`finalize` step is a square root, for
    L2-style aggregations.

    The finalize step is a non-monoidal projection applied once at the
    end of the contraction; this is why :attr:`EUCLIDEAN` is a relaxed
    :class:`Semiring` rather than a :class:`StrictSemiring`.
    """

    def init(
        self, shape: tuple[int, ...], dtype: jnp.dtype[Any]
    ) -> Num[Array, '*shape']:
        return jnp.zeros(shape, dtype=dtype)

    def update(
        self, acc: Num[Array, '*shape'], value: Num[Array, '*shape']
    ) -> Num[Array, '*shape']:
        return acc + value

    def merge(
        self, a: Num[Array, '*shape'], b: Num[Array, '*shape']
    ) -> Num[Array, '*shape']:
        return a + b

    def finalize(self, acc: Num[Array, '*shape']) -> Num[Array, '*shape']:
        # Clamp tiny rounding-induced negative values to 0 before sqrt
        # so neither the value nor a downstream √' grad sees NaN.
        return jnp.sqrt(jnp.maximum(acc, jnp.zeros_like(acc)))


class LogSumExpAcc(NamedTuple):
    """Online state for logsumexp reductions.

    Invariant: the logsumexp of the values seen so far equals
    ``m + log(s)``, with the convention that ``s == 0`` corresponds to
    ``-inf`` (no values seen, or all ``-inf``).
    """

    m: Float[Array, '*shape']
    s: Float[Array, '*shape']


def _safe_exp_diff(
    x: Float[Array, '*shape'], m: Float[Array, '*shape']
) -> Float[Array, '*shape']:
    """Compute :math:`\\exp(x - m)`, defined to be 0 wherever ``x`` is
    :math:`-\\infty`.

    Uses the "double-where with sentinel" trick to keep both forward
    and reverse-mode AD NaN-free.  When ``x`` is :math:`-\\infty` (and
    possibly ``m`` is :math:`-\\infty` too, which would make the naive
    subtraction produce ``NaN``), the inner computation is
    short-circuited via a sentinel zero so no ``NaN`` enters either the
    forward value or the gradient.

    Parameters
    ----------
    x : Float[Array, '*shape']
        Values whose exponentiated difference is required. Entries equal
        to :math:`-\\infty` map to 0 in the output.
    m : Float[Array, '*shape']
        Reference to subtract from ``x`` before exponentiating (typically
        a running maximum). Broadcast against ``x``.

    Returns
    -------
    Float[Array, '*shape']
        The elementwise value :math:`\\exp(x - m)`, with every position
        where ``x`` is :math:`-\\infty` set to 0.
    """
    finite = jnp.isfinite(x)
    safe_diff = jnp.where(finite, x - m, jnp.zeros_like(x))
    return jnp.where(finite, jnp.exp(safe_diff), jnp.zeros_like(x))


@dataclass(frozen=True)
class _LogSumExpMonoid:
    """The logsumexp monoid :math:`(\\mathbb{R} \\cup \\{-\\infty\\},
    \\operatorname{logsumexp}, -\\infty)` with online :math:`(m, s)` state.

    The auxiliary :math:`(\\text{max}, \\text{sum\\_exp})` representation
    keeps the running normalised sum bounded in :math:`[0, K]` -- exactly
    the trick used in online softmax / flash attention. :meth:`finalize`
    re-materialises :math:`m + \\log(s)`.

    All operations are NaN-safe even when both operands are
    :math:`-\\infty`.
    """

    def init(
        self, shape: tuple[int, ...], dtype: jnp.dtype[Any]
    ) -> LogSumExpAcc:
        return LogSumExpAcc(
            m=jnp.full(shape, -jnp.inf, dtype=dtype),
            s=jnp.zeros(shape, dtype=dtype),
        )

    def update(
        self, acc: LogSumExpAcc, value: Float[Array, '*shape']
    ) -> LogSumExpAcc:
        new_m = jnp.maximum(acc.m, value)
        old_term = acc.s * _safe_exp_diff(acc.m, new_m)
        new_term = _safe_exp_diff(value, new_m)
        return LogSumExpAcc(m=new_m, s=old_term + new_term)

    def merge(self, a: LogSumExpAcc, b: LogSumExpAcc) -> LogSumExpAcc:
        new_m = jnp.maximum(a.m, b.m)
        sa = a.s * _safe_exp_diff(a.m, new_m)
        sb = b.s * _safe_exp_diff(b.m, new_m)
        return LogSumExpAcc(m=new_m, s=sa + sb)

    def finalize(self, acc: LogSumExpAcc) -> Float[Array, '*shape']:
        positive = acc.s > 0
        safe_s = jnp.where(positive, acc.s, jnp.ones_like(acc.s))
        return jnp.where(
            positive,
            acc.m + jnp.log(safe_s),
            jnp.full_like(acc.m, -jnp.inf),
        )


# ---------------------------------------------------------------------------
# Semigroups
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ProductSemigroup:
    def combine(
        self, a: Num[Array, '*shape'], b: Num[Array, '*shape']
    ) -> Num[Array, '*shape']:
        return a * b


@dataclass(frozen=True)
class _SumSemigroup:
    def combine(
        self, a: Num[Array, '*shape'], b: Num[Array, '*shape']
    ) -> Num[Array, '*shape']:
        return a + b


@dataclass(frozen=True)
class _AndSemigroup:
    def combine(
        self, a: Num[Array, '*shape'], b: Num[Array, '*shape']
    ) -> Num[Array, '*shape']:
        return jnp.logical_and(a, b)


@dataclass(frozen=True)
class _SquaredDiffSemigroup:
    def combine(
        self, a: Num[Array, '*shape'], b: Num[Array, '*shape']
    ) -> Num[Array, '*shape']:
        return (a - b) ** 2


# ---------------------------------------------------------------------------
# Pre-built algebras
# ---------------------------------------------------------------------------


REAL: StrictSemiring[Any] = StrictSemiring(
    monoid=_SumMonoid(),
    binary_op=_ProductSemigroup(),
    identity=0.0,
    annihilator=0.0,  # 0 * b == 0
    name='real',
    matmul_vjp=real_matmul_vjp,
    ell_matmul_vjp=real_ell_matmul_vjp,
)
"""The ordinary real algebra :math:`(\\mathbb{R}, +, \\times)`.

Sum-of-products contraction: the monoid is addition with identity 0 and
the binary operation is multiplication, with 0 as annihilator so a zero
entry masks its product. Recovers standard matrix multiplication.
"""

LOG: StrictSemiring[Any] = StrictSemiring(
    monoid=_LogSumExpMonoid(),
    binary_op=_SumSemigroup(),
    identity=-jnp.inf,
    annihilator=-jnp.inf,  # -inf + b == -inf
    name='log',
    matmul_vjp=log_matmul_vjp,
    ell_matmul_vjp=log_ell_matmul_vjp,
)
"""The log-space algebra: logsumexp combined with addition.

Products of ordinary reals become sums in the log domain, and sums
become numerically-stable logsumexp reductions. The identity and
annihilator are :math:`-\\infty` (the log of 0), so the contraction
computes probabilities and partition functions in log space without
underflow.
"""

TROPICAL_MAX_PLUS: StrictSemiring[Any] = StrictSemiring(
    monoid=_MaxMonoid(),
    binary_op=_SumSemigroup(),
    identity=-jnp.inf,
    annihilator=-jnp.inf,  # -inf + b == -inf, annihilates under max
    name='tropical_max_plus',
    matmul_vjp=tropical_max_plus_matmul_vjp,
    ell_matmul_vjp=tropical_max_plus_ell_matmul_vjp,
)
"""The max-plus tropical algebra :math:`(\\max, +)`.

The monoid is maximisation and the binary operation is addition, with
identity and annihilator :math:`-\\infty`. Contractions compute
longest-path / best-score quantities; the backward pass gathers the
subgradient at the maximising index.
"""

TROPICAL_MIN_PLUS: StrictSemiring[Any] = StrictSemiring(
    monoid=_MinMonoid(),
    binary_op=_SumSemigroup(),
    identity=jnp.inf,
    annihilator=jnp.inf,  # +inf + b == +inf, annihilates under min
    name='tropical_min_plus',
    matmul_vjp=tropical_min_plus_matmul_vjp,
    ell_matmul_vjp=tropical_min_plus_ell_matmul_vjp,
)
"""The min-plus tropical algebra :math:`(\\min, +)`.

The monoid is minimisation and the binary operation is addition, with
identity and annihilator :math:`+\\infty`. Contractions compute
shortest-path / least-cost quantities; the backward pass gathers the
subgradient at the minimising index.
"""

BOOLEAN: StrictSemiring[Any] = StrictSemiring(
    monoid=_OrMonoid(),
    binary_op=_AndSemigroup(),
    identity=False,
    annihilator=False,  # False and b == False
    name='boolean',
    matmul_vjp=boolean_matmul_vjp,
    ell_matmul_vjp=boolean_ell_matmul_vjp,
)
"""The Boolean algebra: logical OR combined with logical AND.

The monoid is OR (identity and annihilator ``False``) and the binary
operation is AND, giving reachability / transitive-closure contractions.
Being non-differentiable, this algebra raises rather than providing a
backward pass.
"""

# EUCLIDEAN is the relaxed-Semiring test case: its `binary_op` is non-
# associative (squared difference does not satisfy
# (a*b)*c == a*(b*c) in the algebra sense; the contraction shape is fixed
# by the kernel's K-loop order, not by associativity), and `finalize`
# applies a non-monoidal sqrt projection at the end.  Functions that
# require a `StrictSemiring` will reject this at the type-check site.
EUCLIDEAN: Semiring[Any] = Semiring(
    monoid=_SumThenSqrtMonoid(),
    binary_op=_SquaredDiffSemigroup(),
    identity=0.0,
    annihilator=None,  # (a - b)**2 has no annihilator: cannot value-mask
    name='euclidean',
    matmul_vjp=euclidean_matmul_vjp,
    ell_matmul_vjp=euclidean_ell_matmul_vjp,
)
"""The Euclidean-distance algebra: sum-of-squared-differences with a
final square root.

The binary operation is the squared difference :math:`(a - b)^2`, the
monoid sums those contributions (identity 0), and :meth:`finalize`
applies a guarded square root. The squared-difference operation has no
annihilator, so entries cannot be value-masked, and the trailing square
root is a non-monoidal projection; this makes ``EUCLIDEAN`` a relaxed
:class:`Semiring` rather than a :class:`StrictSemiring`.
"""


__all__ = [
    'REAL',
    'LOG',
    'TROPICAL_MAX_PLUS',
    'TROPICAL_MIN_PLUS',
    'BOOLEAN',
    'EUCLIDEAN',
    'LogSumExpAcc',
]
