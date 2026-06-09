# Registration recipe benchmark

- Backend: `gpu` (NVIDIA L4)
- Spec: levels=3, rigid_iters=30, demons_iters=20, lncc_iters=30
- Warm time is the median post-warmup wall-time; compile is the first (trace+compile) call.
- Compile is **cold** (fresh XLA cache); JAX's persistent compilation cache amortises it across runs of the same shapes, so a deployment pays each shape once, not per process.

| case | recipe | fwd compile (s) | fwd warm (s) | grad compile (s) | grad warm (s) |
|---|---|---|---|---|---|
| 2-D 128x128 | rigid (SSD/LM) | 5.22 | 0.0086 | 11.36 | 0.0241 |
| 2-D 128x128 | affine (SSD/LM) | 5.77 | 0.0239 | 14.48 | 0.0762 |
| 2-D 128x128 | demons | 3.17 | 0.0073 | 7.72 | 0.0287 |
| 2-D 128x128 | rigid LNCC (implicit) | 2.73 | 0.0054 | 5.28 | 0.0064 |
| 3-D 48x48x48 | rigid (SSD/LM) | 9.33 | 0.0197 | 21.80 | 0.1018 |
| 3-D 48x48x48 | affine (SSD/LM) | 9.67 | 0.0300 | 25.95 | 0.2331 |
| 3-D 48x48x48 | demons | 4.82 | 0.0143 | 13.33 | 0.1866 |
| 3-D 48x48x48 | rigid LNCC (implicit) | 4.53 | 0.0012 | 8.91 | 0.0047 |
