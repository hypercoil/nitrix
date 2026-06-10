# Registration-suite-v2 scaling benchmark

- Device: `NVIDIA L4 (gpu)`, jax `0.10.1`
- dtype: **f32** (deployment path; x64 left off).
- SyN spec: levels=3, iterations=20, radius=2, step=0.5.
- volreg spec: levels=3, iterations=20, inverse-compositional, SSD.
- Warm = median post-warmup wall-time; cold = first (trace+compile) call.

## greedy SyN -- single-pair scaling

| size | Mvox | cold (s) | warm (s) | warm/Mvox (ms) |
|---|---|---|---|---|
| 64^3 | 0.26 | 9.64 | 0.0534 | 203.776 |
| 96^3 | 0.88 | 10.15 | 0.1987 | 224.595 |
| 128^3 | 2.10 | 30.48 | 0.6502 | 310.028 |

## volreg -- cohort (inverse-compositional)

| config | T | cold (s) | warm (s) | warm/frame (ms) |
|---|---|---|---|---|
| T=16, 64^3 | 16 | 7.30 | 0.0655 | 4.096 |
| T=32, 64^3 | 32 | 7.48 | 0.1360 | 4.251 |
| T=16, 96^3 | 16 | 8.15 | 0.3382 | 21.139 |
