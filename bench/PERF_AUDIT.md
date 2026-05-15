# Perf audit -- nitrix vs natural references

Sorted by wall-time ratio (nitrix / reference) descending; the worst gap is at the top.  ``>1`` means nitrix is slower.

## Host
- device: NVIDIA A10G (gpu)
- platform: Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39
- jax_version: 0.10.0

| op | shape | nitrix | ref | ref time | ratio | agreement | notes |
|---|---|---:|---|---:|---:|---|---|
| morphology.distance_transform | (32, 32) | 2.25 ms | scipy.ndimage.dt_edt | 0.07 ms | 30.14x | algorithm-different | nitrix=iterative tropical, scipy=PMA |
| graph.laplacian | n=64 sparse | 0.14 ms | scipy.sparse.csgraph.laplacian | 0.01 ms | 12.22x | 0.0e+00 |  |
| morphology.distance_transform | (128, 128) | 8.56 ms | scipy.ndimage.dt_edt | 1.09 ms | 7.83x | algorithm-different | nitrix=iterative tropical, scipy=PMA |
| stats.analytic_signal | n=256 | 0.15 ms | scipy.signal.hilbert | 0.04 ms | 3.80x | 7.5e-07 |  |
| graph.laplacian | n=256 sparse | 0.15 ms | scipy.sparse.csgraph.laplacian | 0.04 ms | 3.40x | 0.0e+00 |  |
| morphology.erode | 64x64 | 0.19 ms | scipy.ndimage.grey_erosion | 0.08 ms | 2.43x | 0.0e+00 |  |
| morphology.dilate | 64x64 | 0.19 ms | scipy.ndimage.grey_dilation | 0.08 ms | 2.26x | 0.0e+00 |  |
| stats.analytic_signal | n=2048 | 0.14 ms | scipy.signal.hilbert | 0.07 ms | 2.10x | 1.2e-06 |  |
| smoothing.gaussian | 64x64 | 0.16 ms | scipy.ndimage.gaussian_filter | 0.08 ms | 2.02x | 1.2e-07 |  |
| morphology.distance_transform | (64, 64, 64) | 61.68 ms | scipy.ndimage.dt_edt | 35.68 ms | 1.73x | algorithm-different | nitrix=iterative tropical, scipy=PMA |
| morphology.distance_transform | (32, 32, 32) | 5.09 ms | scipy.ndimage.dt_edt | 4.03 ms | 1.27x | algorithm-different | nitrix=iterative tropical, scipy=PMA |
| stats.cov | (50, 500) | 0.14 ms | numpy.cov | 0.13 ms | 1.10x | 7.8e-05 |  |
| stats.corr | (50, 500) | 0.14 ms | numpy.corrcoef | 0.15 ms | 0.91x | 4.4e-05 |  |
| geometry.spatial_transform | 64x64 c=1 | 0.14 ms | scipy.ndimage.map_coordinates | 0.25 ms | 0.57x | skipped | linear interpolation; modes match |
| graph.laplacian | n=1024 sparse | 0.17 ms | scipy.sparse.csgraph.laplacian | 0.42 ms | 0.41x | 0.0e+00 |  |
| stats.analytic_signal | n=16384 | 0.15 ms | scipy.signal.hilbert | 0.40 ms | 0.38x | 1.7e-06 |  |
| lme.flame_two_level (voxelwise) | V=10, N=60, p=2 | 15.26 ms | statsmodels.wls (per-voxel loop) | 50.88 ms | 0.30x | ~5e-3 | ref extrapolated from V=10 |
| morphology.median_filter | 64x64 | 0.17 ms | scipy.ndimage.median_filter | 0.67 ms | 0.25x | 9.9e-01 |  |
| smoothing.gaussian | 256x256 | 0.16 ms | scipy.ndimage.gaussian_filter | 0.89 ms | 0.18x | 1.8e-07 |  |
| morphology.dilate | 256x256 | 0.20 ms | scipy.ndimage.grey_dilation | 1.32 ms | 0.15x | 0.0e+00 |  |
| morphology.erode | 256x256 | 0.20 ms | scipy.ndimage.grey_erosion | 1.37 ms | 0.14x | 0.0e+00 |  |
| linalg.rbf_kernel | (500, 32) | 0.14 ms | sklearn | 2.05 ms | 0.07x | 2.1e-03 |  |
| geometry.spatial_transform | 256x256 c=1 | 0.14 ms | scipy.ndimage.map_coordinates | 2.54 ms | 0.06x | skipped | linear interpolation; modes match |
| lme.flame_two_level (voxelwise) | V=100, N=60, p=2 | 15.30 ms | statsmodels.wls (per-voxel loop) | 448.25 ms | 0.03x | ~5e-3 | ref extrapolated from V=100 |
| morphology.median_filter | 256x256 | 0.32 ms | scipy.ndimage.median_filter | 10.27 ms | 0.03x | 1.0e+00 |  |
| smoothing.gaussian | 64x64x64 | 0.19 ms | scipy.ndimage.gaussian_filter | 6.80 ms | 0.03x | 1.2e-07 |  |
| stats.cov | (500, 2000) | 0.22 ms | numpy.cov | 10.14 ms | 0.02x | 4.8e-05 |  |
| stats.corr | (500, 2000) | 0.21 ms | numpy.corrcoef | 9.97 ms | 0.02x | 3.3e-05 |  |
| linalg.residualise (Cholesky) | V=1000, N=400, K=24 | 0.19 ms | numpy.linalg.lstsq | 9.61 ms | 0.02x | 6.1e-04 |  |
| linalg.rbf_kernel | (1000, 128) | 0.15 ms | sklearn | 11.70 ms | 0.01x | 5.1e-03 |  |
| stats.cov | (2000, 1000) | 0.52 ms | numpy.cov | 72.28 ms | 0.01x | 8.2e-05 |  |
| stats.corr | (2000, 1000) | 0.57 ms | numpy.corrcoef | 82.80 ms | 0.01x | 4.6e-05 |  |
| linalg.rbf_kernel | (2000, 32) | 0.26 ms | sklearn | 39.97 ms | 0.01x | 4.0e-03 |  |
| lme.flame_two_level (voxelwise) | V=1000, N=60, p=2 | 15.38 ms | statsmodels.wls (per-voxel loop) | 4466.68 ms | 0.00x | skipped | ref extrapolated from V=100 |
| linalg.rbf_kernel | (5000, 32) | 0.78 ms | sklearn | 287.30 ms | 0.00x | 3.0e-06 |  |
| linalg.residualise (Cholesky) | V=10000, N=400, K=24 | 0.47 ms | numpy.linalg.lstsq | 189.60 ms | 0.00x | 7.0e-04 |  |
| linalg.residualise (Cholesky) | V=100000, N=400, K=24 | 3.05 ms | numpy.linalg.lstsq | 2426.51 ms | 0.00x | 8.2e-04 |  |

## Reading the table

- **ratio > 1**: nitrix slower than reference.  For GPU paths this often happens at small problem sizes where kernel-launch overhead dominates.  Pay attention to ratios > 5 at the LARGEST measured size for each op -- those are the real gaps.
- **ratio < 1**: nitrix faster.  Typical at large sizes where GPU bandwidth dominates.
- **agreement**: max absolute difference between nitrix and reference outputs.  ``algorithm-different`` means the two implementations don't compute identical results (e.g., different distance-transform algorithms); we report wall-time only.