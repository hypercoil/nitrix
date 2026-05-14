'''Semiring matrix multiplication via JAX Pallas.

Implements ``C[i, j] = (+)_k (A[i, k] (*) B[k, j])`` over user-specified
semirings, where ``(+)`` is a Monoid and ``(*)`` is a (broadcasting) Semigroup.

Key design choices (see module docstring of the constructor for details):

* The accumulator is a pytree, not necessarily a single array, so online /
  numerically-stable reductions like log-semiring's ``(running_max, sum_exp)``
  can be carried through the K loop.
* The K reduction is performed *inside* the kernel via ``jax.lax.fori_loop``
  with an inner Python-unrolled loop over each block's K dimension. This
  keeps peak on-chip memory at O(BM*BN + BM*BK + BK*BN) (KeOps-style
  streaming) rather than materializing the full (BM, BK, BN) value tensor.
* No tensor-core / ``dot`` primitives are used; we issue plain CUDA-core
  SIMD ops so the kernel works for arbitrary semirings.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Generic,
    NamedTuple,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jaxtyping import Array, Float, Num


# -----------------------------------------------------------------------------
# Protocols
# -----------------------------------------------------------------------------

S = TypeVar('S')  # accumulator-state pytree


@runtime_checkable
class Monoid(Protocol[S]):
    '''An associative reduction with identity, with optional auxiliary state.

    Required (informal) properties:

    * ``init(shape, dtype)`` returns the identity, broadcast to ``shape``.
    * ``update(acc, value)`` folds a value into the accumulator.
    * ``merge(a, b)`` is associative and has ``init`` as identity.
    * ``finalize(acc)`` projects the (possibly auxiliary) state to the
      user-facing value.

    The state ``S`` may be any pytree; this is what enables online algorithms
    such as the (max, sum_exp) state used by ``LogSumExpMonoid``.
    '''

    def init(self, shape: tuple[int, ...], dtype: jnp.dtype) -> S: ...

    def update(self, acc: S, value: Num[Array, '*shape']) -> S: ...

    def merge(self, a: S, b: S) -> S: ...

    def finalize(self, acc: S) -> Num[Array, '*shape']: ...


@runtime_checkable
class Semigroup(Protocol):
    '''A binary operation supporting NumPy-style broadcasting.

    Used as the semiring's ``(*)``: the per-(i, k, j) combine inside the
    contraction. Need not be commutative; the kernel only relies on the
    elementwise-broadcast call ``combine(a_col, b_row)``.
    '''

    def combine(
        self, a: Num[Array, '*shape'], b: Num[Array, '*shape']
    ) -> Num[Array, '*shape']: ...


@dataclass(frozen=True)
class Semiring(Generic[S]):
    '''A (numerical) semiring as a pair ``(monoid, semigroup)``.

    The kernel uses ``semigroup.combine`` for ``(*)`` and the four monoid
    methods for ``(+)`` and the auxiliary state lifecycle.
    '''

    monoid: Monoid[S]
    semigroup: Semigroup
    name: str = 'semiring'


# -----------------------------------------------------------------------------
# Concrete monoids
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SumMonoid:
    '''Standard ``(R, +, 0)`` monoid; ``finalize`` is identity.'''

    def init(self, shape, dtype):
        return jnp.zeros(shape, dtype=dtype)

    def update(self, acc, value):
        return acc + value

    def merge(self, a, b):
        return a + b

    def finalize(self, acc):
        return acc


@dataclass(frozen=True)
class MaxMonoid:
    '''``(R u {-inf}, max, -inf)`` monoid; ``finalize`` is identity.'''

    def init(self, shape, dtype):
        return jnp.full(shape, -jnp.inf, dtype=dtype)

    def update(self, acc, value):
        return jnp.maximum(acc, value)

    def merge(self, a, b):
        return jnp.maximum(a, b)

    def finalize(self, acc):
        return acc


@dataclass(frozen=True)
class MinMonoid:
    '''``(R u {+inf}, min, +inf)`` monoid; ``finalize`` is identity.'''

    def init(self, shape, dtype):
        return jnp.full(shape, jnp.inf, dtype=dtype)

    def update(self, acc, value):
        return jnp.minimum(acc, value)

    def merge(self, a, b):
        return jnp.minimum(a, b)

    def finalize(self, acc):
        return acc


@dataclass(frozen=True)
class SumThenSqrtMonoid:
    '''Sum monoid with ``finalize = sqrt``; for L2-style aggregations.

    Used by ``EuclideanSemiring`` together with ``SquaredDiffSemigroup``.
    The finalize-step is a *non-monoid* projection applied once at the end
    of the contraction; in particular it does not commute with ``merge``,
    so consumers must call ``finalize`` only on a fully-reduced state.
    '''

    def init(self, shape, dtype):
        return jnp.zeros(shape, dtype=dtype)

    def update(self, acc, value):
        return acc + value

    def merge(self, a, b):
        return a + b

    def finalize(self, acc):
        # clamp tiny negative values from rounding in fp32 to 0 before sqrt
        return jnp.sqrt(jnp.maximum(acc, jnp.zeros_like(acc)))


class LogSumExpAcc(NamedTuple):
    '''Online state for ``logsumexp`` reductions.

    Invariant: the logsumexp of the values seen so far equals
    ``m + log(s)``, with the convention that ``s == 0`` corresponds to
    ``-inf`` (no values seen, or all ``-inf``).
    '''

    m: Float[Array, '*shape']
    s: Float[Array, '*shape']


def _safe_exp_diff(
    x: Float[Array, '*shape'], m: Float[Array, '*shape']
) -> Float[Array, '*shape']:
    '''Compute ``exp(x - m)`` defined to be 0 whenever ``x = -inf``.

    Uses the "double-where with sentinel" trick to keep both forward and
    reverse-mode AD NaN-free. Specifically:

    * When ``x`` is ``-inf`` (and possibly ``m`` is ``-inf`` too, which
      would make the naive subtraction produce ``NaN``), the inner
      computation is short-circuited via a sentinel zero so no ``NaN``
      enters either the forward value or the gradient.
    '''
    finite = jnp.isfinite(x)
    safe_diff = jnp.where(finite, x - m, jnp.zeros_like(x))
    return jnp.where(finite, jnp.exp(safe_diff), jnp.zeros_like(x))


@dataclass(frozen=True)
class LogSumExpMonoid:
    '''``(R u {-inf}, logsumexp, -inf)`` with online (max, sum_exp) state.

    The auxiliary ``(m, s)`` representation keeps the running normalized
    sum bounded in ``[0, K]``, exactly the trick used in online softmax /
    flash attention. ``finalize`` re-materializes ``m + log(s)``.

    All operations are NaN-safe even when both operands are ``-inf``.
    '''

    def init(self, shape, dtype):
        return LogSumExpAcc(
            m=jnp.full(shape, -jnp.inf, dtype=dtype),
            s=jnp.zeros(shape, dtype=dtype),
        )

    def update(self, acc, value):
        new_m = jnp.maximum(acc.m, value)
        old_term = acc.s * _safe_exp_diff(acc.m, new_m)
        new_term = _safe_exp_diff(value, new_m)
        return LogSumExpAcc(m=new_m, s=old_term + new_term)

    def merge(self, a, b):
        new_m = jnp.maximum(a.m, b.m)
        sa = a.s * _safe_exp_diff(a.m, new_m)
        sb = b.s * _safe_exp_diff(b.m, new_m)
        return LogSumExpAcc(m=new_m, s=sa + sb)

    def finalize(self, acc):
        # s == 0 means no values were seen / all were -inf -> -inf result.
        # log(0) would be -inf already, but jnp.log(0) emits a divide-by-zero
        # warning in some configs, so we mask explicitly.
        positive = acc.s > 0
        safe_s = jnp.where(positive, acc.s, jnp.ones_like(acc.s))
        return jnp.where(
            positive, acc.m + jnp.log(safe_s), jnp.full_like(acc.m, -jnp.inf)
        )


# -----------------------------------------------------------------------------
# Concrete semigroups
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ProductSemigroup:
    def combine(self, a, b):
        return a * b


@dataclass(frozen=True)
class SumSemigroup:
    def combine(self, a, b):
        return a + b


@dataclass(frozen=True)
class SquaredDiffSemigroup:
    def combine(self, a, b):
        return (a - b) ** 2


# -----------------------------------------------------------------------------
# Pre-built semirings
# -----------------------------------------------------------------------------


def real_semiring() -> Semiring[Float[Array, '*shape']]:
    '''Standard real semiring: ``(R, +, *)``.'''
    return Semiring(SumMonoid(), ProductSemigroup(), name='real')


def tropical_max_plus_semiring() -> Semiring[Float[Array, '*shape']]:
    '''Tropical (max, +) semiring; useful for shortest-path / dilation.'''
    return Semiring(MaxMonoid(), SumSemigroup(), name='tropical_max_plus')


def tropical_min_plus_semiring() -> Semiring[Float[Array, '*shape']]:
    '''Tropical (min, +) semiring; useful for shortest-path / erosion.'''
    return Semiring(MinMonoid(), SumSemigroup(), name='tropical_min_plus')


def log_semiring() -> Semiring[LogSumExpAcc]:
    '''Log-semiring: ``logsumexp_k (a_k + b_k)`` with online stable state.'''
    return Semiring(LogSumExpMonoid(), SumSemigroup(), name='log')


def euclidean_semiring() -> Semiring[Float[Array, '*shape']]:
    '''Euclidean: ``C[i,j] = sqrt(sum_k (A[i,k] - B[k,j])**2)``.

    Strictly speaking this is *not* a semiring — ``(*)`` is non-associative
    in general — but it slots into the same kernel because we only require
    ``combine`` to be a broadcasting binary op and ``(+)`` to be a monoid.
    '''
    return Semiring(SumThenSqrtMonoid(), SquaredDiffSemigroup(), name='euclidean')


# -----------------------------------------------------------------------------
# Reference (pure-JAX, non-Pallas) implementation
# -----------------------------------------------------------------------------


def reference_semiring_gemm(
    semiring: Semiring[S],
    a: Float[Array, 'm p'],
    b: Float[Array, 'p n'],
) -> Float[Array, 'm n']:
    '''Pure-JAX reference using the same ``Semiring`` API as the kernel.

    Walks the contraction dim with ``lax.fori_loop`` and the supplied
    ``update``. Used as a ground-truth in tests; correctness here is just a
    function of the algebra, not of the kernel.
    '''
    p = a.shape[1]
    monoid = semiring.monoid
    semigroup = semiring.semigroup
    acc_init = monoid.init((a.shape[0], b.shape[1]), a.dtype)

    def body(k: int, acc: S) -> S:
        value = semigroup.combine(a[:, k : k + 1], b[k : k + 1, :])
        return monoid.update(acc, value)

    acc = lax.fori_loop(0, p, body, acc_init)
    return monoid.finalize(acc)


# -----------------------------------------------------------------------------
# Pallas kernel builder
# -----------------------------------------------------------------------------


GemmFn = Callable[
    [Float[Array, 'm p'], Float[Array, 'p n']], Float[Array, 'm n']
]


def make_gemm_kernel(
    semiring: Semiring[S],
    m: int,
    n: int,
    p: int,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    dtype: jnp.dtype = jnp.float32,
    out_dtype: jnp.dtype | None = None,
    interpret: bool = False,
) -> GemmFn:
    '''Build a Pallas kernel computing ``C = A (semiring-matmul) B``.

    The kernel evaluates::

        C[i, j] = (+)_k ( A[i, k] (*) B[k, j] )

    where ``(*)`` is ``semiring.semigroup.combine`` and ``(+)`` is the
    monoid encoded by ``semiring.monoid``. If the monoid carries auxiliary
    state (a pytree), it is threaded through ``lax.fori_loop`` and
    ``finalize``-d before storing the result block.

    Memory shape (per CTA):

    * ``acc``: ``(BM, BN)`` of monoid state (multiplied by leaf-count for
      pytree states, e.g. 2 leaves for ``LogSumExpAcc``).
    * Per K block: a ``(BM, BK)`` slice of ``A`` and a ``(BK, BN)`` slice
      of ``B`` are loaded; the ``(BM, BK, BN)`` value tensor is *never*
      materialized -- the inner Python loop folds rank-1 outer combinations
      directly into ``acc``.

    Args:
        semiring: defines ``(*)`` and ``(+)``.
        m, n, p: matrix dimensions. ``A: (m, p)``, ``B: (p, n)``,
          ``C: (m, n)``.
        block_m, block_n, block_k: tile sizes; must divide m, n, p
          respectively. Reasonable defaults are 64, 64, 32 -- tuned to fit
          comfortably in registers/SMEM on most NVIDIA SMs (Ampere+).
        dtype: dtype of inputs.
        out_dtype: dtype of output (defaults to ``dtype``).
        interpret: run the kernel in JAX-interpret mode (CPU; for testing).

    Returns:
        A jit-compiled function ``gemm(A, B) -> C``.
    '''
    if out_dtype is None:
        out_dtype = dtype

    if m % block_m:
        raise ValueError(f'block_m={block_m} must divide m={m}.')
    if n % block_n:
        raise ValueError(f'block_n={block_n} must divide n={n}.')
    if p % block_k:
        raise ValueError(f'block_k={block_k} must divide p={p}.')

    n_k_steps = p // block_k
    monoid = semiring.monoid
    semigroup = semiring.semigroup

    def kernel(a_ref, b_ref, o_ref):
        # a_ref: (block_m, p) logical block (whole row-stripe of A)
        # b_ref: (p, block_n) logical block (whole col-stripe of B)
        # o_ref: (block_m, block_n) output block

        acc_init = monoid.init((block_m, block_n), dtype)

        def body(k_step, acc):
            a_blk = pl.load(
                a_ref, (slice(None), pl.ds(k_step * block_k, block_k))
            )
            b_blk = pl.load(
                b_ref, (pl.ds(k_step * block_k, block_k), slice(None))
            )

            # Inner k-fold, Python-unrolled at trace time so each step is a
            # rank-1 outer combine + monoid update on (BM, BN). The compiler
            # keeps `acc` in registers across all `block_k` iterations.
            for kk in range(block_k):
                a_col = a_blk[:, kk : kk + 1]  # (BM, 1)
                b_row = b_blk[kk : kk + 1, :]  # (1, BN)
                value = semigroup.combine(a_col, b_row)  # (BM, BN)
                acc = monoid.update(acc, value)

            return acc

        acc = lax.fori_loop(0, n_k_steps, body, acc_init)
        out = monoid.finalize(acc).astype(out_dtype)
        pl.store(o_ref, (slice(None), slice(None)), out)

    @jax.jit
    def gemm(
        a: Float[Array, 'm p'], b: Float[Array, 'p n']
    ) -> Float[Array, 'm n']:
        return pl.pallas_call(
            kernel,
            grid=(m // block_m, n // block_n),
            in_specs=[
                pl.BlockSpec((block_m, p), lambda i, j: (i, 0)),
                pl.BlockSpec((p, block_n), lambda i, j: (0, j)),
            ],
            out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
            interpret=interpret,
        )(a, b)

    return gemm


__all__ = [
    'Monoid',
    'Semigroup',
    'Semiring',
    'SumMonoid',
    'MaxMonoid',
    'MinMonoid',
    'SumThenSqrtMonoid',
    'LogSumExpMonoid',
    'LogSumExpAcc',
    'ProductSemigroup',
    'SumSemigroup',
    'SquaredDiffSemigroup',
    'real_semiring',
    'tropical_max_plus_semiring',
    'tropical_min_plus_semiring',
    'log_semiring',
    'euclidean_semiring',
    'reference_semiring_gemm',
    'make_gemm_kernel',
]
