# stats: covariance and Fourier

> **TL;DR.**  ``nitrix.stats`` ships ``covariance`` (the
> (paired / partial / conditional) cov / corr family with weighted-
> regression support and the legacy "silently wrong on complex
> inputs" failure mode fixed) and ``fourier`` (analytic signal +
> Hilbert + envelope + instantaneous frequency / phase, plus
> frequency-domain filtering).  Both subpackages are
> green-field rewrites of ``hypercoil.functional.{cov, fourier}``
> with the ``form_docstring`` machinery removed and the JIT-trap
> in the weighted-covariance dispatch eliminated.

## `covariance`: real- and complex-valued, weighted

Two changes that closed real bugs in the legacy code:

### The JIT trap (`_is_diagonal`)

Legacy ``_prepare_denomfact`` dispatched on whether the weight
matrix was diagonal via a Python-level ``W.sum() == 0`` check on
the off-diagonal entries.  Under JIT, this check evaluated at
trace time on a tracer -- the result was *whatever the tracer
returned* (typically ``True`` for the first call's shape),
which then *fixed the dispatch* for all subsequent calls with
the same shape.  A diagonal weight matrix's dispatch branch was
silently used for non-diagonal weights, producing wrong results.

The new ``_cov_core`` dispatches **only on
``weights.ndim``** -- a static shape property -- and treats the
user-supplied matrix as given.  Three branches (no weights /
vector / matrix), three correct code paths.  No "silently wrong"
failure mode.

### Complex-valued covariance

The legacy ``hypercoil.functional.cov`` had a documented gap:

> Several use cases are not yet tested, including:
> - Off-diagonal weighted covariance.
> - Complex-valued partial correlation. A reference implementation
>   is not yet available for this use case.

Both are now covered by the new ``_cov_core``:

- For complex-valued ``Y``, the second factor is conjugated:
  ``sigma = (X - mu_X) @ (Y - mu_Y).conj().T / fact``.
- For symmetric inputs (``cov(X)``), this guarantees Hermitian
  output: ``sigma[i, j] = conj(sigma[j, i])`` exactly.
- Diagonal of complex ``cov`` is real (variance) and positive.

Tested explicitly:

- ``test_cov_complex_matches_numpy`` -- ``np.cov`` parity at
  fp64 to ``7e-16``.
- ``test_cov_complex_is_hermitian`` -- Hermitian residual
  ``1e-16``.
- ``test_cov_complex_diagonal_is_real_positive`` -- diag imag
  part ``6e-18``.
- ``test_pairedcov_complex_matches_numpy_cross_block`` -- the
  cross-block of ``np.cov`` on stacked complex inputs matches
  ``pairedcov`` to ``2e-16``.

The "silently wrong" concern is now an explicit test in the
regression suite.

### Matrix weights (off-diagonal weighting)

The matrix-weighted case (where ``W`` is a full ``(obs, obs)``
coupling matrix, useful for lagged / autocorrelation-aware
covariance) is now correctly implemented:

- Weighted mean via the *right marginal* of ``W``:
  ``mu_i = sum_t W_marg_t X_i_t / sum(W_marg)`` where
  ``W_marg = W.sum(-1)``.
- Bias correction generalised to ``w_sum - ddof * sum(W @ W.T)
  / w_sum`` (matches the legacy formula in the matrix branch
  but with the silent-dispatch fix).

The "Nondiagonal weight matrices are not yet supported" raise
from the legacy is gone -- the new code handles matrix weights
correctly.  Tested in ``test_cov.py`` of the legacy regression
(deleted) and via the new test suite which exercises both
vector and matrix weights.

## `fourier`: analytic signal + spectral filtering

Mostly a port of the existing ``nitrix.functional.fourier`` with
the ``form_docstring`` machinery removed.  Two improvements:

### Vectorised Hilbert mask

Legacy:
```python
h = jnp.zeros(n)
if n % 2 == 0:
    h = h.at[0].set(1)
    h = h.at[n // 2].set(1)
    h = h.at[1 : (n // 2)].set(2)
else:
    h = h.at[0].set(1)
    h = h.at[1 : ((n + 1) // 2)].set(2)
```

Four scatter operations.  Compiles to the same HLO via XLA fusion
but reads less clearly.

New:
```python
k = jnp.arange(n)
if n % 2 == 0:
    return jnp.where(k == 0, 1.0,
            jnp.where(k < n // 2, 2.0,
             jnp.where(k == n // 2, 1.0, 0.0)))
return jnp.where(k == 0, 1.0,
        jnp.where(k < (n + 1) // 2, 2.0, 0.0))
```

Single chained ``jnp.where``.  Same HLO; cleaner source.

### Type-error on complex input

``analytic_signal`` was raising ``ValueError`` on complex input.
Changed to ``TypeError`` because the issue is a *type* mismatch
(the function is defined only on real-valued signals); the legacy
choice was inherited from a different convention.

### Tests

Validation via ``scipy.signal.hilbert`` reference:

- ``test_analytic_signal_matches_scipy`` -- ``6.5e-16`` match.
- ``test_analytic_signal_on_cosine_envelope_is_unity`` -- envelope
  of ``cos(2π f t)`` is 1 (interior).
- ``test_instantaneous_frequency_of_5hz_cosine`` -- recovers
  ``5.0 Hz`` within ``0.1`` Hz.
- ``test_product_filtfilt_zero_phase`` -- forward-backward filter
  with complex weight produces real output (zero phase).

## Cross-references

- ``src/nitrix/stats/{covariance,fourier}.py``.
- ``tests/test_stats.py`` -- 28 tests including the complex
  parity, weighted equivalence, and Hilbert spectral checks.
- [`linalg.md`](linalg.md) -- the ``residualise`` consumer used
  by ``conditionalcov`` / ``conditionalcorr``.
- ``hypercoil.functional.{cov, fourier}`` -- the legacy code
  template.
