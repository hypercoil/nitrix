# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Compensated summation and precision-aware reductions.

Pure-numerics reduction utilities that any substrate reduction can drop in
where floating-point accumulation error matters: accuracy (long fMRI
time-series covariance accumulation drifts at ``float32``), reproducibility
(a golden-corpus reduction that must not depend on the backend's reduction
order), and unbiased low-precision accumulation (``float16`` / ``bfloat16``
paths).

The surface:

- :func:`kahan_sum` -- Kahan (1965) compensated summation: carry the running
  round-off error and feed it back, recovering roughly one extra working
  precision.
- :func:`neumaier_sum` -- the Kahan--Babuška--Neumaier variant, correct also
  when a summand is larger in magnitude than the running total.
- :func:`pairwise_sum` -- a balanced log-depth tree reduction (the NumPy
  default), a fully parallel accuracy improvement over the naive left fold.
- :func:`compensated_dot` -- the Ogita--Rump--Oishi ``Dot2`` compensated inner
  product: as accurate as accumulating the products in twice the working
  precision, using an FMA-free Dekker two-product.
- :func:`stochastic_round` -- unbiased rounding to a lower-precision dtype, so
  that :math:`\mathbb{E}[\operatorname{round}(x)] = x` (the FP16 / FP8
  accumulation convention).

The compensated reductions (:func:`kahan_sum`, :func:`neumaier_sum`,
:func:`compensated_dot`) are implemented with :func:`jax.lax.scan` so the
error-feedback dependency is explicit and the compiler cannot reassociate it
away; :func:`pairwise_sum` unrolls a balanced add tree at trace time. All are
differentiable.

References
----------
Kahan W (1965). Further remarks on reducing truncation errors.
*Communications of the ACM*, 8(1), 40. https://doi.org/10.1145/363707.363723

Neumaier A (1974). Rundungsfehleranalyse einiger Verfahren zur Summation
endlicher Summen. *ZAMM*, 54(1), 39-51.
https://doi.org/10.1002/zamm.19740540106

Ogita T, Rump SM, Oishi S (2005). Accurate sum and dot product. *SIAM Journal
on Scientific Computing*, 26(6), 1955-1988.
https://doi.org/10.1137/030601818
"""

from __future__ import annotations

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float

__all__ = [
    'compensated_dot',
    'kahan_sum',
    'neumaier_sum',
    'pairwise_sum',
    'stochastic_round',
]


def _restore(
    reduced: Float[Array, '...'], ndim: int, axis: int, keepdims: bool
) -> Float[Array, '...']:
    """Re-insert the reduced axis as a singleton when ``keepdims``."""
    if keepdims:
        return jnp.expand_dims(reduced, axis % ndim)
    return reduced


def kahan_sum(
    x: Float[Array, '...'],
    *,
    axis: int = -1,
    keepdims: bool = False,
) -> Float[Array, '...']:
    r"""Kahan compensated sum of ``x`` along an axis.

    Carries the running round-off error :math:`c` and subtracts it from the
    next summand, so the accumulation behaves as if it had roughly one extra
    working precision. For a length-:math:`n` sum the error bound improves from
    the naive :math:`O(n\,\varepsilon)` to :math:`O(\varepsilon) +
    O(n\,\varepsilon^2)`.

    Parameters
    ----------
    x : Float[Array, '...']
        Values to sum.
    axis : int, optional
        Axis to reduce. Default ``-1`` (the trailing / observation axis).
    keepdims : bool, optional
        If ``True``, retain the reduced axis as a singleton. Default ``False``.

    Returns
    -------
    Float[Array, '...']
        The compensated sum, with ``axis`` removed (or kept as a singleton).
    """
    xm = jnp.moveaxis(x, axis, 0)
    zero = jnp.zeros(xm.shape[1:], xm.dtype)

    def body(
        carry: tuple[Float[Array, '...'], Float[Array, '...']],
        xi: Float[Array, '...'],
    ) -> tuple[tuple[Float[Array, '...'], Float[Array, '...']], None]:
        total, comp = carry
        y = xi - comp
        t = total + y
        comp = (t - total) - y
        return (t, comp), None

    (total, _), _ = lax.scan(body, (zero, zero), xm)
    return _restore(total, x.ndim, axis, keepdims)


def neumaier_sum(
    x: Float[Array, '...'],
    *,
    axis: int = -1,
    keepdims: bool = False,
) -> Float[Array, '...']:
    r"""Kahan--Babuška--Neumaier compensated sum of ``x`` along an axis.

    An improvement on :func:`kahan_sum` that stays accurate even when the next
    summand is larger in magnitude than the running total (the case Kahan's
    original loses): the low-order bits are taken from whichever of the two is
    smaller, and the accumulated compensation is applied once at the end.

    Parameters
    ----------
    x : Float[Array, '...']
        Values to sum.
    axis : int, optional
        Axis to reduce. Default ``-1``.
    keepdims : bool, optional
        If ``True``, retain the reduced axis as a singleton. Default ``False``.

    Returns
    -------
    Float[Array, '...']
        The compensated sum.
    """
    xm = jnp.moveaxis(x, axis, 0)
    zero = jnp.zeros(xm.shape[1:], xm.dtype)

    def body(
        carry: tuple[Float[Array, '...'], Float[Array, '...']],
        xi: Float[Array, '...'],
    ) -> tuple[tuple[Float[Array, '...'], Float[Array, '...']], None]:
        total, comp = carry
        t = total + xi
        larger_total = jnp.abs(total) >= jnp.abs(xi)
        comp = comp + jnp.where(
            larger_total, (total - t) + xi, (xi - t) + total
        )
        return (t, comp), None

    (total, comp), _ = lax.scan(body, (zero, zero), xm)
    return _restore(total + comp, x.ndim, axis, keepdims)


def _pairwise_tree(
    a: Float[Array, '...'], block_size: int
) -> Float[Array, '...']:
    """Balanced tree reduction over the leading axis of ``a``.

    Recurses at trace time (the split is on the static leading size), summing
    naively within blocks of at most ``block_size`` and pairing the partial
    sums, giving an add tree of depth :math:`O(\\log n)`.
    """
    n = a.shape[0]
    if n <= block_size:
        return a.sum(0)
    half = n // 2
    return _pairwise_tree(a[:half], block_size) + _pairwise_tree(
        a[half:], block_size
    )


def pairwise_sum(
    x: Float[Array, '...'],
    *,
    axis: int = -1,
    block_size: int = 128,
    keepdims: bool = False,
) -> Float[Array, '...']:
    r"""Pairwise (log-depth tree) sum of ``x`` along an axis.

    Splits the axis recursively and pairs partial sums, so the accumulation
    error grows as :math:`O(\varepsilon \log n)` rather than the naive left
    fold's :math:`O(\varepsilon n)`. Unlike :func:`kahan_sum` this adds no
    per-step overhead and stays fully parallel; it is the accuracy floor that
    NumPy's :func:`numpy.sum` provides and JAX's does not guarantee.

    Parameters
    ----------
    x : Float[Array, '...']
        Values to sum.
    axis : int, optional
        Axis to reduce. Default ``-1``.
    block_size : int, optional
        Leaves of at most this many elements are summed naively; above it the
        axis is halved recursively. Default ``128``.
    keepdims : bool, optional
        If ``True``, retain the reduced axis as a singleton. Default ``False``.

    Returns
    -------
    Float[Array, '...']
        The pairwise sum.
    """
    xm = jnp.moveaxis(x, axis, 0)
    total = _pairwise_tree(xm, block_size)
    return _restore(total, x.ndim, axis, keepdims)


def _split(a: Float[Array, '...']) -> tuple[Float[Array, '...'], ...]:
    """Veltkamp split ``a = hi + lo`` with non-overlapping significands."""
    p = jnp.finfo(a.dtype).nmant + 1
    s = (p + 1) // 2
    factor = jnp.asarray((1 << s) + 1, a.dtype)
    q = factor * a
    hi = q - (q - a)
    lo = a - hi
    return hi, lo


def _two_product(
    a: Float[Array, '...'], b: Float[Array, '...']
) -> tuple[Float[Array, '...'], Float[Array, '...']]:
    """Exact product: return ``(p, e)`` with ``a * b == p + e`` in exact
    arithmetic (Dekker's FMA-free two-product).

    An FMA-based two-product (``e = fma(a, b, -p)``) would be ~2 flops rather
    than Dekker's ~17, but is unreachable: JAX surfaces no ``fma`` primitive,
    and the ``a * b - p`` contraction idiom is inert under XLA (it rounds
    ``a * b`` to ``p`` twice and cancels to exactly zero, silently degrading the
    dot to uncompensated). Dekker is therefore the portable path -- and it is
    *safe*: the ``a_hi * b_hi - p`` error term would be corrupted if the backend
    contracted it into an FMA, but this was verified bit-exact on both the JAX
    CPU backend and an Ampere-class GPU (ptxas does not contract across it).
    """
    p = a * b
    a_hi, a_lo = _split(a)
    b_hi, b_lo = _split(b)
    e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    return p, e


def _two_sum(
    a: Float[Array, '...'], b: Float[Array, '...']
) -> tuple[Float[Array, '...'], Float[Array, '...']]:
    """Exact sum: return ``(s, e)`` with ``a + b == s + e`` (Knuth)."""
    s = a + b
    bv = s - a
    e = (a - (s - bv)) + (b - bv)
    return s, e


def compensated_dot(
    a: Float[Array, '...'],
    b: Float[Array, '...'],
    *,
    axis: int = -1,
    keepdims: bool = False,
) -> Float[Array, '...']:
    r"""Compensated inner product of ``a`` and ``b`` along an axis (``Dot2``).

    The Ogita--Rump--Oishi ``Dot2`` algorithm: each product is split into its
    rounded value plus an exact error (an FMA-free Dekker two-product), and the
    products *and* their errors are accumulated with a compensated sum. The
    result is as accurate as forming the dot product in twice the working
    precision and rounding once -- the summation error is removed, not merely
    bounded.

    Parameters
    ----------
    a, b : Float[Array, '...']
        Operands, broadcast against each other; reduced along ``axis``.
    axis : int, optional
        Axis to contract. Default ``-1``.
    keepdims : bool, optional
        If ``True``, retain the contracted axis as a singleton. Default
        ``False``.

    Returns
    -------
    Float[Array, '...']
        The compensated dot product.
    """
    a, b = jnp.broadcast_arrays(a, b)
    am = jnp.moveaxis(a, axis, 0)
    bm = jnp.moveaxis(b, axis, 0)
    zero = jnp.zeros(am.shape[1:], am.dtype)

    def body(
        carry: tuple[Float[Array, '...'], Float[Array, '...']],
        ab: tuple[Float[Array, '...'], Float[Array, '...']],
    ) -> tuple[tuple[Float[Array, '...'], Float[Array, '...']], None]:
        total, comp = carry
        ai, bi = ab
        p, e_p = _two_product(ai, bi)
        total, e_s = _two_sum(total, p)
        comp = comp + (e_p + e_s)
        return (total, comp), None

    (total, comp), _ = lax.scan(body, (zero, zero), (am, bm))
    return _restore(total + comp, a.ndim, axis, keepdims)


def stochastic_round(
    x: Float[Array, '...'],
    dtype: DTypeLike,
    *,
    key: Array,
) -> Float[Array, '...']:
    r"""Unbiased stochastic rounding of ``x`` to a lower-precision ``dtype``.

    Rounds each element to one of the two ``dtype`` grid points bracketing it,
    up with probability equal to the fractional position between them, so the
    rounding is unbiased: :math:`\mathbb{E}[\operatorname{round}(x)] = x`. This
    is the accumulation convention for reduced precision (``float16`` /
    ``bfloat16``): repeated stochastic rounding does not systematically drift
    the way round-to-nearest does, at the cost of per-element noise.

    Parameters
    ----------
    x : Float[Array, '...']
        Values to round, in a precision at least that of ``dtype``.
    dtype : DTypeLike
        Target (lower-precision) floating dtype, e.g. ``jnp.float16`` or
        ``jnp.bfloat16``.
    key : Array
        A :func:`jax.random.key`.

    Returns
    -------
    Float[Array, '...']
        ``x`` rounded to ``dtype``, unbiased.

    Notes
    -----
    Exactly representable values are returned unchanged. Implemented via
    :func:`jax.numpy.nextafter` in the target dtype; targets ``float16`` /
    ``bfloat16`` (and ``float32``), the precisions with reliable ``nextafter``
    support.
    """
    hi_dtype = x.dtype
    neg_inf = jnp.asarray(-jnp.inf, dtype)
    pos_inf = jnp.asarray(jnp.inf, dtype)

    down = x.astype(dtype)
    overshot = down.astype(hi_dtype) > x
    down = jnp.where(overshot, jnp.nextafter(down, neg_inf), down)
    down_hi = down.astype(hi_dtype)
    up = jnp.nextafter(down, pos_inf)
    up_hi = up.astype(hi_dtype)

    span = up_hi - down_hi
    prob = jnp.where(span > 0, (x - down_hi) / span, jnp.zeros_like(x))
    noise = jax.random.uniform(key, x.shape, hi_dtype)
    return jnp.where(noise < prob, up, down)
