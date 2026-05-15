# Perf audit -- nitrix vs natural references

Sorted by wall-time ratio (nitrix / reference) descending; the worst gap is at the top.  ``>1`` means nitrix is slower.

## Host
- device: NVIDIA A10G (gpu)
- platform: Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39
- jax_version: 0.10.0

| op | shape | nitrix | ref | ref time | ratio | agreement | notes |
|---|---|---:|---|---:|---:|---|---|
| morphology.distance_transform | (32, 32) | 2.28 ms | scipy.ndimage.dt_edt | 0.08 ms | 30.29x | algorithm-different | nitrix=iterative tropical, scipy=PMA |
| morphology.distance_transform | (128, 128) | 8.91 ms | scipy.ndimage.dt_edt | 1.09 ms | 8.18x | algorithm-different | nitrix=iterative tropical, scipy=PMA |
| morphology.distance_transform | (64, 64, 64) | 61.71 ms | scipy.ndimage.dt_edt | 45.44 ms | 1.36x | algorithm-different | nitrix=iterative tropical, scipy=PMA |
| morphology.distance_transform | (32, 32, 32) | 5.24 ms | scipy.ndimage.dt_edt | 4.05 ms | 1.29x | algorithm-different | nitrix=iterative tropical, scipy=PMA |
| stats.cov | (50, 500) | 0.15 ms | numpy.cov | 0.13 ms | 1.15x | 7.8e-05 |  |
| lme.flame_two_level (voxelwise) | V=10, N=60, p=2 | 15.17 ms | statsmodels.wls (per-voxel loop) | 49.63 ms | 0.31x | ~5e-3 | ref extrapolated from V=10 |
| linalg.rbf_kernel | (500, 32) | 0.14 ms | sklearn | 2.15 ms | 0.06x | 2.1e-03 |  |
| lme.flame_two_level (voxelwise) | V=100, N=60, p=2 | 15.19 ms | statsmodels.wls (per-voxel loop) | 451.61 ms | 0.03x | ~5e-3 | ref extrapolated from V=100 |
| stats.cov | (500, 2000) | 0.22 ms | numpy.cov | 9.04 ms | 0.02x | 4.8e-05 |  |
| linalg.residualise (Cholesky) | V=1000, N=400, K=24 | 0.20 ms | numpy.linalg.lstsq | 9.15 ms | 0.02x | 6.1e-04 |  |
| linalg.rbf_kernel | (1000, 128) | 0.15 ms | sklearn | 10.42 ms | 0.01x | 5.1e-03 |  |
| stats.cov | (2000, 1000) | 0.53 ms | numpy.cov | 68.96 ms | 0.01x | 8.2e-05 |  |
| linalg.rbf_kernel | (2000, 32) | 0.25 ms | sklearn | 42.01 ms | 0.01x | 4.0e-03 |  |
| lme.flame_two_level (voxelwise) | V=1000, N=60, p=2 | 15.34 ms | statsmodels.wls (per-voxel loop) | 4587.67 ms | 0.00x | skipped | ref extrapolated from V=100 |
| linalg.rbf_kernel | (5000, 32) | 0.80 ms | sklearn | 305.58 ms | 0.00x | 3.0e-06 |  |
| linalg.residualise (Cholesky) | V=10000, N=400, K=24 | 0.46 ms | numpy.linalg.lstsq | 185.45 ms | 0.00x | 7.0e-04 |  |
| linalg.residualise (Cholesky) | V=100000, N=400, K=24 | 3.04 ms | numpy.linalg.lstsq | 2425.95 ms | 0.00x | 8.2e-04 |  |

## Reading the table

- **ratio > 1**: nitrix slower than reference.  For GPU paths this often happens at small problem sizes where kernel-launch overhead dominates.  Pay attention to ratios > 5 at the LARGEST measured size for each op -- those are the real gaps.
- **ratio < 1**: nitrix faster.  Typical at large sizes where GPU bandwidth dominates.
- **agreement**: max absolute difference between nitrix and reference outputs.  ``algorithm-different`` means the two implementations don't compute identical results (e.g., different distance-transform algorithms); we report wall-time only.