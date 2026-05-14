# LOBPCG implicit-VJP performance

## Host
- device: NVIDIA A10G (gpu)
- platform: Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39
- jax_version: 0.10.0

## End-to-end: forward + backward through LOBPCG

Times include the LOBPCG iteration (typically dominant) plus the implicit-VJP backward.  ``bwd_only`` is the subtraction of ``forward`` from ``fwd+bwd``; for the ELL path it captures the backward formula only, since LOBPCG is run the same way in both calls.

| n     | k   | nnz    |   path | forward    | fwd+bwd    | bwd_only   | HLO audit |
|------:|----:|-------:|-------:|-----------:|-----------:|-----------:|:----------|
|   256 |   4 |      - |  dense |   80.83 ms |   81.02 ms |   197.5 µs | max=256 n2=y |
|  1024 |   4 |      - |  dense |   52.80 ms |   53.40 ms |   605.8 µs | max=1024 n2=y |
|  4096 |   4 |      - |  dense |   29.96 ms |   31.25 ms |    1.29 ms | max=4096 n2=y |
|   256 |   4 |   1274 |    ell |   35.60 ms |   35.97 ms |   370.2 µs | max=2304 n2=n |
|  1024 |   4 |   5112 |    ell |   35.43 ms |   36.02 ms |   592.0 µs | max=11264 n2=n |
|  4096 |   4 |  20478 |    ell |   40.88 ms |   41.08 ms |   201.5 µs | max=45056 n2=n |
| 16384 |   4 |  81916 |    ell |   19.48 ms |   19.68 ms |   206.9 µs | max=212992 n2=n |
|   256 |   4 |      - |   eigh |    2.15 ms |    2.19 ms |    35.9 µs | - |
|  1024 |   4 |      - |   eigh |   11.49 ms |   11.88 ms |   386.2 µs | - |

## Pure backward kernel (no LOBPCG iteration)

This isolates the implicit-VJP formula itself.  For a synthetic orthogonal ``U`` and random ``Λ``, time the closed-form backward map ``(g_λ, g_U) -> dM`` (dense) or ``(g_λ, g_U) -> d_values`` (ELL).  The ELL row tests the sparsity-projected ``O(nnz * k + n * k^2)`` claim directly, without LOBPCG iteration noise.

| n     | k   | nnz    |   path | time       | HLO audit |
|------:|----:|-------:|-------:|-----------:|:----------|
|   256 |   4 |      - |  dense |   147.9 µs | max=256 n2=y |
|  1024 |   4 |      - |  dense |   150.6 µs | max=1024 n2=y |
|  4096 |   4 |      - |  dense |   299.9 µs | max=4096 n2=y |
| 16384 |   4 |      - |  dense |    2.37 ms | max=16384 n2=y |
|   256 |   4 |   1274 |    ell |   165.8 µs | max=2304 n2=n |
|  1024 |   4 |   5112 |    ell |   171.9 µs | max=11264 n2=n |
|  4096 |   4 |  20478 |    ell |   144.1 µs | max=45056 n2=n |
| 16384 |   4 |  81916 |    ell |   168.2 µs | max=212992 n2=n |
| 65536 |   4 | 327670 |    ell |   163.1 µs | max=983040 n2=n |

### Reading the table

- ``max`` is the largest single tensor axis in the compiled HLO.  For the ELL backward we expect ``max ~ k_max`` or ``max ~ n * k`` at most -- definitely not ``n^2``.  For the dense backward, ``max == n`` is fine (the gradient itself is ``(n, n)``).
- ``n2`` is "y" iff a tensor with two axes equal to ``n`` appears.  For the ELL backward this MUST be "n": "y" means XLA materialised an ``O(n^2)`` intermediate somewhere in the projection.
- For ELL pure-bwd, doubling ``n`` at fixed per-row degree should roughly double the wall-time.  Quadratic scaling would mean XLA found a sneaky way around the sparsity projection.
