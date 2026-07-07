# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Matrix functions of a *general* (non-symmetric) matrix.

The symmetric family (:func:`~nitrix.linalg.spd.symexp` /
:func:`~nitrix.linalg.spd.symlog` / :func:`~nitrix.linalg.spd.symsqrt` /
:func:`~nitrix.linalg.spd.sympower`) applies a scalar function to the
eigenvalues of a symmetric matrix.  :func:`matrix_exp` is the general-matrix
counterpart: the true matrix exponential :math:`e^{A}` for an arbitrary square
:math:`A`, needed to exponentiate a non-symmetric Lie-algebra element --
specifically the affine generator :math:`[[A, b], [0, 0]]` whose exponential is
the affine transform :func:`~nitrix.geometry.transform.affine_exp` applies (a
rigid generator's rotation block is skew-symmetric, also non-symmetric).
:func:`matrix_log` is the inverse, warranted by the affine Fréchet/Karcher mean
:func:`~nitrix.geometry.algebra.transform_mean`.

:func:`matrix_exp` uses **scaling-and-squaring with a truncated Taylor
series**.  Unlike ``jax.scipy.linalg.expm`` (Padé, which needs a linear solve
and so wedges / falls back to CPU on the affected cuSolver stacks), this path is
**pure matmul** -- GPU-native, ``jit`` / ``grad``-clean, and never mixes
devices inside a traced optimiser.  :math:`A` is scaled by :math:`2^{-s}`, the
Taylor series of :math:`e^{A/2^{s}}` is summed, and the result is squared
:math:`s` times.  With the defaults (:math:`s = 8`, order 12) the scaled norm is
tiny for any :math:`\\lVert A \\rVert` that arises from a registration generator,
so the result matches a reference ``expm`` to machine precision; raise them for
very large :math:`\\lVert A \\rVert`.

:func:`matrix_log` is the inverse: **inverse scaling-and-squaring** -- take
:math:`s` matrix square roots (Denman--Beavers) to drive :math:`A` toward
:math:`I`, sum the Taylor series of :math:`\\log(I + X)` on the small remainder,
and multiply by :math:`2^{s}`.  Registration affines carry large translations
(:math:`\\lVert A - I \\rVert \\gg 1`), which rules out a pure-matmul (binomial)
square root, so the Denman--Beavers step routes its inverse through
:func:`~nitrix.linalg._solver.safe_inv` (GPU-native on a healthy stack;
CPU-fallback on the wedged-cuSolver dev box).  Unlike :func:`matrix_exp` this is
**not** for an optimiser hot loop -- it backs the offline Karcher mean -- so the
fallback is acceptable.  An out-of-domain input (an eigenvalue on the negative
real axis) returns **NaN**, not finite garbage, caught by a cuSolver-free
round-trip-through-:func:`matrix_exp` guard (see :func:`matrix_log`).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._solver import safe_eigh, safe_inv
from .matrix import symmetric
from .spd import symmap

__all__ = [
    'chebyshev_apply',
    'frechet_derivative',
    'matrix_exp',
    'matrix_function',
    'matrix_log',
    'matrix_polynomial',
]

# Relative round-trip residual ``‖exp(log A) − A‖ / ‖A‖`` above which
# ``matrix_log`` returns NaN rather than a silently-wrong finite log (B3): the
# input is then outside the convergence domain -- an eigenvalue on the closed
# negative real axis (a 180-deg rotation, a reflection), or ``‖A−I‖`` too large
# for ``n_sqrt``.  Generous: valid inputs (incl. large-translation / ill-scaled
# affines) round-trip at ``~1e-10``, and the transition to garbage is sharp (it
# tracks whether the spectrum reaches the Taylor radius, a discontinuity, not a
# gradual decay), so anything in the wide gap separates cleanly.  The check is a
# single ``matrix_exp`` (pure matmul) -- cuSolver-free and jit-clean, unlike an
# eigenvalue / determinant domain test (which would wedge on the dev box).
_LOG_RECON_RTOL = 1e-2


def matrix_exp(
    a: Float[Array, '... n n'],
    *,
    n_squarings: int = 8,
    taylor_order: int = 12,
) -> Float[Array, '... n n']:
    """Matrix exponential :math:`e^{A}` of a general square matrix.

    Uses scaling-and-squaring with a truncated Taylor series: :math:`A` is
    scaled by :math:`2^{-s}`, the Taylor series of :math:`e^{A/2^{s}}` is
    summed, and the result is squared :math:`s` times, so that
    :math:`e^{A} = (e^{A/2^{s}})^{2^{s}}`.

    Parameters
    ----------
    a : Float[Array, '... n n']
        Square matrix, ``(..., n, n)``.  Need not be symmetric.  Batches
        over leading dimensions.
    n_squarings : int, optional
        Scaling-and-squaring doublings :math:`s`: :math:`A` is exponentiated
        as :math:`(e^{A/2^{s}})^{2^{s}}`.  The default ``8`` keeps the scaled
        norm small for any registration generator.
    taylor_order : int, optional
        Number of Taylor terms for :math:`e^{A/2^{s}}`.  Default ``12``.

    Returns
    -------
    Float[Array, '... n n']
        The matrix exponential :math:`e^{A}`, same shape as ``a``.

    Notes
    -----
    Pure matmul (no factorisation), so it is GPU-native and differentiable
    by autodiff.  Identities used as correctness anchors in the tests:
    :math:`e^{0} = I`; :math:`\\det(e^{A}) = e^{\\operatorname{tr} A}`; for a
    skew-symmetric :math:`A`, :math:`e^{A}` is a rotation (orthogonal,
    :math:`\\det = +1`).
    """
    n = a.shape[-1]
    eye = jnp.broadcast_to(jnp.eye(n, dtype=a.dtype), a.shape)
    scaled = a / float(2**n_squarings)
    # Taylor series e^B = I + B + B^2/2! + ... + B^q/q!.
    term = eye
    result = eye
    for k in range(1, taylor_order + 1):
        term = (term @ scaled) / k
        result = result + term
    # Square s times to undo the scaling.
    for _ in range(n_squarings):
        result = result @ result
    return result


def _db_sqrtm(
    m: Float[Array, '... n n'], iters: int
) -> Float[Array, '... n n']:
    """Matrix square root by the Denman--Beavers iteration (batched).

    Iterates :math:`Y_{k+1} = \\tfrac{1}{2}(Y_k + Z_k^{-1})` and
    :math:`Z_{k+1} = \\tfrac{1}{2}(Z_k + Y_k^{-1})` from :math:`Y_0 = M`,
    :math:`Z_0 = I`, so that :math:`Y_k \\to M^{1/2}` quadratically.  The
    inverses route through :func:`~nitrix.linalg._solver.safe_inv`
    (CPU-fallback on the wedged cuSolver stack).

    Parameters
    ----------
    m : Float[Array, '... n n']
        Square matrix, ``(..., n, n)``, to take the square root of.  Batches
        over leading dimensions.
    iters : int
        Number of Denman--Beavers iterations to perform.

    Returns
    -------
    Float[Array, '... n n']
        The matrix square root :math:`M^{1/2}`, same shape as ``m``.
    """
    n = m.shape[-1]
    eye = jnp.broadcast_to(jnp.eye(n, dtype=m.dtype), m.shape)
    y, z = m, eye
    for _ in range(iters):
        y, z = 0.5 * (y + safe_inv(z)), 0.5 * (z + safe_inv(y))
    return y


def matrix_log(
    a: Float[Array, '... n n'],
    *,
    n_sqrt: int = 6,
    db_iters: int = 8,
    taylor_order: int = 12,
) -> Float[Array, '... n n']:
    """Matrix logarithm :math:`\\log A` of a general square matrix.

    This is the inverse of :func:`matrix_exp`, computed by inverse
    scaling-and-squaring: :math:`\\log A = 2^{s} \\cdot \\log(A^{1/2^{s}})`.
    Take :math:`s` Denman--Beavers square roots to drive :math:`A` toward
    :math:`I`, sum the Taylor series of :math:`\\log(I + X)` on the small
    remainder :math:`X = A^{1/2^{s}} - I`, and scale back by :math:`2^{s}`.

    :math:`A` must have no eigenvalue on the closed negative real axis (a
    positive-determinant registration affine / its relative transform
    qualifies).  Outside that domain -- a reflection (:math:`\\det < 0`) or a
    180-degree rotation (eigenvalue :math:`-1`) -- the principal log does not
    exist, and rather than the **finite garbage** the bare series would return
    (a reflection NaNs naturally in the square root; a 180-degree rotation
    returns a huge, non-reconstructing matrix), this returns **NaN** so the
    failure is loud.  The test is the round-trip residual itself
    (:math:`\\lVert e^{\\log A} - A \\rVert`, pure matmul -- no
    eigendecomposition, no :math:`\\det`, so it survives the dev box): a *valid*
    large-translation / ill-scaled affine whose Frobenius
    :math:`\\lVert A - I \\rVert` is large but whose **spectrum** the square
    roots still drive to :math:`I` round-trips at :math:`\\sim 10^{-10}` and is
    untouched; only genuine garbage trips the guard.  A NaN therefore means
    "raise ``n_sqrt`` first (the input may merely exceed the Taylor radius); if
    still NaN, the input is out of domain."  **No balancing step is applied** --
    the inverse scaling-and-squaring already equilibrates the spectrum for the
    transforms registration produces (verified to :math:`\\sim 10^{-12}` on a
    180-voxel-translation, 6x-anisotropic, sheared affine), so ``n_sqrt`` is the
    lever, not a diagonal balance.

    Parameters
    ----------
    a : Float[Array, '... n n']
        Square matrix, ``(..., n, n)`` with no eigenvalue on the closed
        negative real axis (above).  Batches over leading dimensions.
    n_sqrt : int, optional
        Number of square-root steps :math:`s`.  The default ``6`` brings a
        registration affine (rotation + :math:`\\lesssim 60` voxel translation)
        within the Taylor radius; raise it for larger transforms (the lever
        before concluding an input is out of domain).
    db_iters : int, optional
        Denman--Beavers iterations per square root (quadratic convergence).
        Default ``8``.
    taylor_order : int, optional
        Number of terms of the :math:`\\log(I + X)` series.  Default ``12``.

    Returns
    -------
    Float[Array, '... n n']
        The matrix logarithm :math:`\\log A`, same shape as ``a`` -- or **NaN**
        (same shape) when ``a`` is outside the convergence domain (above).

    Notes
    -----
    Routes its inverses through :func:`~nitrix.linalg._solver.safe_inv`
    (GPU-native on a healthy stack, CPU-fallback otherwise).  Correctness
    anchor: :func:`matrix_log` is the inverse of :func:`matrix_exp` (round-trips
    to machine precision in the registration regime); :math:`\\log(I) = 0`;
    :math:`\\operatorname{tr}(\\log A) = \\log(\\det A)`.
    """
    n = a.shape[-1]
    eye = jnp.broadcast_to(jnp.eye(n, dtype=a.dtype), a.shape)
    root = a
    for _ in range(n_sqrt):
        root = _db_sqrtm(root, db_iters)
    x = root - eye
    xk = x
    series = x
    for k in range(2, taylor_order + 1):
        xk = xk @ x
        series = series + ((-1.0) ** (k + 1)) * xk / k
    log_a = series * float(2**n_sqrt)
    # Loud domain / convergence guard: NaN a non-reconstructing log rather than
    # return finite garbage.  ``<=`` (not ``>``) so a non-finite residual -- a
    # reflection whose square root already NaN'd -- also fails the test.
    recon = matrix_exp(log_a)
    residual = jnp.linalg.norm(recon - a, axis=(-2, -1)) / (
        jnp.linalg.norm(a, axis=(-2, -1)) + 1e-30
    )
    ok = (residual <= _LOG_RECON_RTOL)[..., None, None]
    return jnp.where(ok, log_a, jnp.asarray(jnp.nan, dtype=a.dtype))


# ---------------------------------------------------------------------------
# Symmetric spectral functions: the general eigenvalue-map entry point and the
# Frechet derivative (eigh-based), plus the eigh-free Chebyshev operator family
# (pure matmul / matvec -- GPU-native, cuSOLVER-free) that the spectral-graph
# filters build on.
# ---------------------------------------------------------------------------


def matrix_function(
    a: Float[Array, '... n n'],
    fn: Callable[[Array], Array],
    *,
    psi: float = 0.0,
    key: Optional[Array] = None,
) -> Float[Array, '... n n']:
    r"""Apply a scalar function to the eigenvalues of a symmetric matrix.

    Eigendecomposes the (symmetrised) input :math:`A = Q \Lambda Q^{\top}` and
    reassembles :math:`f(A) = Q\, \operatorname{diag}(f(\lambda))\, Q^{\top}`.
    This is the **general** entry point of which
    :func:`~nitrix.linalg.spd.symexp` / :func:`~nitrix.linalg.spd.symlog` /
    :func:`~nitrix.linalg.spd.symsqrt` are named specialisations. It is
    :func:`~nitrix.linalg.spd.symmap` with the SPD eigenvalue clipping switched
    off (``eigvalue_clip='none'``) -- so ``fn`` sees the raw spectrum, negative
    eigenvalues included, and must itself be well-defined there -- delegating to
    that shipped workhorse rather than reimplementing the eigendecomposition and
    its VJP.

    Parameters
    ----------
    a : Float[Array, '... n n']
        Symmetric matrix batch (symmetrised internally).
    fn : Callable[[Array], Array]
        Elementwise scalar function applied to the eigenvalue array, e.g.
        ``jnp.exp`` or ``lambda x: jnp.exp(-t * x)`` for a heat kernel. A
        ``NaN`` from ``fn`` propagates rather than being silently zeroed.
    psi : float, optional
        Reconditioning strength for the eigendecomposition VJP; ``psi > 0``
        stabilises the gradient at near-degenerate spectra (see
        :func:`~nitrix.linalg.spd.symmap`). Default ``0.0``.
    key : Array, optional
        PRNG key, required when ``psi > 0``.

    Returns
    -------
    Float[Array, '... n n']
        The symmetric matrix :math:`f(A)`.

    Notes
    -----
    Routed through :func:`~nitrix.linalg._solver.safe_eigh` (GPU-native on a
    healthy stack, CPU-fallback on a wedged-cuSOLVER box; the fallback is
    eager-only). For a polynomial ``fn`` prefer the eigh-free
    :func:`matrix_polynomial`, which stays on the device under ``jit``.
    """
    return symmap(a, fn=fn, psi=psi, key=key, eigvalue_clip='none')


def chebyshev_apply(
    op: Callable[[Array], Array],
    x: Array,
    coeffs: Union[Sequence[float], Float[Array, 'k']],
) -> Array:
    r"""Apply a Chebyshev series of a linear operator to an operand.

    Evaluates :math:`\big(\sum_{k=0}^{K} c_k\, T_k(\mathcal{L})\big)\, x` by the
    three-term recurrence :math:`T_0 = I`, :math:`T_1 = \mathcal{L}`,
    :math:`T_{k+1} = 2\mathcal{L}\,T_k - T_{k-1}`, using only applications of
    ``op`` -- no eigendecomposition. This is the shared kernel behind
    :func:`matrix_polynomial` (``op`` a dense left-multiply, ``x`` the identity)
    and the spectral-graph filters (``op`` a sparse Laplacian matvec, ``x`` a
    signal), so a filter costs :math:`K` operator applications rather than an
    :math:`O(n^3)` eigensolve.

    Parameters
    ----------
    op : Callable[[Array], Array]
        The linear operator :math:`\mathcal{L}`, as a callable ``op(v)``. Its
        spectrum **must already lie in** :math:`[-1, 1]` (the domain of the
        Chebyshev polynomials); rescale a raw operator first (see
        :func:`matrix_polynomial` for the dense rescaling).
    x : Array
        The operand ``op`` is applied to -- an identity matrix (giving the dense
        polynomial), a vector, or a batch of signals.
    coeffs : sequence of float or Float[Array, 'k']
        Chebyshev-basis coefficients :math:`c_0, \dots, c_K` (the plain basis;
        if using a Chebyshev *series* with the :math:`c_0/2` convention, pass
        ``c_0/2`` as the first entry).

    Returns
    -------
    Array
        :math:`\sum_k c_k\, T_k(\mathcal{L})\, x`, the shape of ``x``.
    """
    c = jnp.asarray(coeffs)
    order = c.shape[0]
    t_prev = x
    acc = c[0] * t_prev
    if order == 1:
        return acc
    t_curr = op(x)
    acc = acc + c[1] * t_curr
    for k in range(2, order):
        t_next = 2.0 * op(t_curr) - t_prev
        acc = acc + c[k] * t_next
        t_prev, t_curr = t_curr, t_next
    return acc


def matrix_polynomial(
    a: Float[Array, '... n n'],
    coeffs: Union[Sequence[float], Float[Array, 'k']],
    *,
    domain: Optional[Tuple[float, float]] = None,
) -> Float[Array, '... n n']:
    r"""Chebyshev matrix polynomial :math:`\sum_k c_k\, T_k(A)` (eigh-free).

    Evaluates the Chebyshev-basis polynomial of ``a`` by the pure-matmul
    three-term recurrence -- GPU-native, ``jit`` / ``grad``-clean, and free of
    any eigendecomposition or solve. With ``domain=(lambda_min, lambda_max)`` the
    matrix is first affinely rescaled so that interval maps onto :math:`[-1, 1]`
    (:math:`\tilde A = (2A - (\lambda_{\min}+\lambda_{\max})I) /
    (\lambda_{\max}-\lambda_{\min})`), the standard conditioning for a spectral
    filter over a known spectral range (e.g. a graph Laplacian's
    :math:`[0, \lambda_{\max}]`).

    Parameters
    ----------
    a : Float[Array, '... n n']
        Square matrix batch (real spectrum assumed when ``domain`` is given).
    coeffs : sequence of float or Float[Array, 'k']
        Chebyshev-basis coefficients :math:`c_0, \dots, c_K`.
    domain : tuple of float, optional
        ``(lambda_min, lambda_max)`` bounding the spectrum; the matrix is
        rescaled onto :math:`[-1, 1]` before the recurrence. ``None`` (default)
        assumes the spectrum already lies in :math:`[-1, 1]`.

    Returns
    -------
    Float[Array, '... n n']
        The matrix polynomial :math:`\sum_k c_k\, T_k(\tilde A)`.
    """
    n = a.shape[-1]
    eye = jnp.broadcast_to(jnp.eye(n, dtype=a.dtype), a.shape)
    if domain is not None:
        lo, hi = domain
        a = (2.0 * a - (lo + hi) * eye) / (hi - lo)
    return chebyshev_apply(lambda m: a @ m, eye, coeffs)


def frechet_derivative(
    a: Float[Array, '... n n'],
    fn: Callable[[Array], Array],
    e: Float[Array, '... n n'],
    *,
    dfn: Optional[Callable[[Array], Array]] = None,
) -> Float[Array, '... n n']:
    r"""Frechet (directional) derivative of a symmetric matrix function.

    The derivative of :math:`f(A)` in the direction :math:`E`,
    :math:`L_f(A)[E] = \lim_{h\to 0} \tfrac{f(A + hE) - f(A)}{h}`, computed by
    the Daleckii--Krein formula: in the eigenbasis :math:`A = Q\Lambda
    Q^{\top}`, :math:`L_f(A)[E] = Q\,\big(\Psi \odot (Q^{\top} E Q)\big)\,
    Q^{\top}` where :math:`\Psi` is the Loewner matrix of divided differences
    :math:`\Psi_{ij} = \frac{f(\lambda_i) - f(\lambda_j)}{\lambda_i -
    \lambda_j}` (and :math:`\Psi_{ii} = f'(\lambda_i)` on the diagonal / for
    degenerate eigenvalues).

    Parameters
    ----------
    a : Float[Array, '... n n']
        Symmetric matrix batch (symmetrised internally).
    fn : Callable[[Array], Array]
        Elementwise scalar function (as in :func:`matrix_function`).
    e : Float[Array, '... n n']
        Direction matrix, same shape as ``a``.
    dfn : Callable[[Array], Array], optional
        The elementwise derivative :math:`f'`. Default ``None`` obtains it by
        :func:`jax.grad` of ``fn`` (requires ``fn`` to be a scalar-to-scalar
        differentiable function).

    Returns
    -------
    Float[Array, '... n n']
        The Frechet derivative :math:`L_f(A)[E]`, a symmetric matrix.
    """
    lam, q = safe_eigh(symmetric(a))
    f_lam = fn(lam)
    grad_fn = jax.grad(fn) if dfn is None else dfn
    df_lam = jax.vmap(grad_fn)(lam.reshape(-1)).reshape(lam.shape)

    diff = lam[..., :, None] - lam[..., None, :]
    num = f_lam[..., :, None] - f_lam[..., None, :]
    diag = 0.5 * (df_lam[..., :, None] + df_lam[..., None, :])
    tol = jnp.sqrt(jnp.finfo(a.dtype).eps) * (jnp.max(jnp.abs(lam)) + 1.0)
    degenerate = jnp.abs(diff) <= tol
    safe_diff = jnp.where(degenerate, 1.0, diff)
    loewner = jnp.where(degenerate, diag, num / safe_diff)

    qt = jnp.swapaxes(q, -1, -2)
    e_eig = qt @ symmetric(e) @ q
    return q @ (loewner * e_eig) @ qt
