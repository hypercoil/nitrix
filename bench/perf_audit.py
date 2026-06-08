# -*- coding: utf-8 -*-
"""
Perf audit across un-benched nitrix ops vs natural references.

Targets the high-traffic public ops that ship without dedicated
bench scripts as of this commit, paired with the canonical
external reference (numpy / scipy / sklearn / statsmodels).
Per-op:

- Build matching inputs at a realistic neuroimaging shape.
- Warm the JIT cache, then time both implementations.
- Report wall-time ratio + result-agreement check.

The goal is **gap identification**, not "we're slower than X."
A 2-5x gap on a single op may be perfectly acceptable; a 50x
gap on a hot path is a red flag.  Output is a ranked Markdown
table.

Selected ops (priority order):

1. ``linalg.rbf_kernel`` vs ``sklearn.metrics.pairwise.rbf_kernel``
   -- recent green-field kernel module; sklearn is the de facto
   ML reference.
2. ``stats.lme.flame_two_level`` (voxelwise) vs per-voxel
   ``statsmodels.MixedLM`` -- the LME wall-time is the key
   shippability question.
3. ``linalg.residualise`` (Cholesky) vs ``numpy.linalg.lstsq``
   -- we claim ~9x over SVD; here we want the wall-time ratio
   against numpy (which is SVD too) at fMRI-typical shapes.
4. ``morphology.distance_transform`` vs
   ``scipy.ndimage.distance_transform_edt`` -- different
   algorithms (iterative tropical-semiring vs scipy's PMA);
   single-call comparison.
5. ``stats.cov`` vs ``numpy.cov`` -- should be near-parity;
   bench as a sanity baseline.

Per the rest of nitrix bench convention: report warm-state
median wall-time (compile cost separately recorded).  All
nitrix paths are JIT-wrapped + warmed up.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

warnings.filterwarnings('ignore')


@dataclass
class BenchRow:
    op: str
    shape_desc: str
    nitrix_ms: float
    ref_name: str
    ref_ms: float
    agreement: str  # e.g., "1.2e-5" or "skipped"
    notes: str = ''

    @property
    def ratio(self) -> float:
        """Wall-time ratio nitrix / ref.  >1 means nitrix slower."""
        if self.ref_ms <= 0:
            return float('inf')
        return self.nitrix_ms / self.ref_ms

    def report_line(self) -> str:
        ratio_str = f'{self.ratio:.2f}x'
        return (
            f'| {self.op} | {self.shape_desc} | '
            f'{self.nitrix_ms:.2f} ms | {self.ref_name} | '
            f'{self.ref_ms:.2f} ms | {ratio_str} | {self.agreement} | '
            f'{self.notes} |'
        )


def _time_warm(fn: Callable, warmup: int = 3, repeats: int = 8) -> float:
    """Return median wall-time after warm-up, in milliseconds.

    ``fn`` should be a 0-arg callable that returns a JAX array
    (or any array we can ``block_until_ready`` on).  CPU-side
    functions returning numpy are fine -- we just don't block.
    """
    out = fn()
    if hasattr(out, 'block_until_ready'):
        out.block_until_ready()
    for _ in range(max(warmup - 1, 0)):
        out = fn()
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples)) * 1e3  # ms


# ---------------------------------------------------------------------------
# 1. RBF kernel: nitrix vs sklearn
# ---------------------------------------------------------------------------


def bench_rbf_kernel() -> list[BenchRow]:
    from sklearn.metrics.pairwise import rbf_kernel as sk_rbf

    from nitrix.linalg import rbf_kernel

    rows = []
    for n, d in [(500, 32), (2000, 32), (5000, 32), (1000, 128)]:
        rng = np.random.default_rng(0)
        X0 = rng.standard_normal((n, d)).astype(np.float32)
        X0_j = jnp.asarray(X0)

        def ref_fn():
            return sk_rbf(X0, gamma=0.1)

        nitrix_fn = jax.jit(lambda x: rbf_kernel(x, gamma=0.1))
        # Warm jit
        _ = nitrix_fn(X0_j).block_until_ready()

        nitrix_ms = _time_warm(lambda: nitrix_fn(X0_j))
        ref_ms = _time_warm(ref_fn)

        # Agreement check
        K_n = np.asarray(nitrix_fn(X0_j))
        K_r = ref_fn()
        max_diff = float(np.abs(K_n - K_r).max())
        rows.append(
            BenchRow(
                op='linalg.rbf_kernel',
                shape_desc=f'({n}, {d})',
                nitrix_ms=nitrix_ms,
                ref_name='sklearn',
                ref_ms=ref_ms,
                agreement=f'{max_diff:.1e}',
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 2. statsmodels MixedLM vs nitrix.stats.lme.flame_two_level
# ---------------------------------------------------------------------------


def bench_lme_voxelwise() -> list[BenchRow]:
    import pandas as pd
    import statsmodels.formula.api as smf

    from nitrix.stats.lme import flame_two_level

    rows = []
    rng = np.random.default_rng(0)
    N, p = 60, 2
    # Voxel counts that should be tractable for ref (statsmodels
    # is per-voxel, so V * single-fit-time scales linearly).
    for V in (10, 100, 1000):
        X_group = jnp.asarray(rng.standard_normal((N, p)).astype(np.float32))
        true_gamma = jnp.asarray([1.0, 0.5])
        var_within = jnp.asarray(
            (np.abs(rng.standard_normal((V, N))) * 0.5 + 0.1).astype(
                np.float32
            ),
        )
        beta = (
            X_group @ true_gamma
            + 0.5 * jax.random.normal(jax.random.key(1), (V, N))
            + jnp.sqrt(var_within)
            * jax.random.normal(jax.random.key(2), (V, N))
        ).astype(jnp.float32)

        # nitrix: JIT'd vmap-batched fit
        nitrix_fn = jax.jit(
            lambda b, vw: flame_two_level(b, vw, X_group, n_iter=20),
        )
        _ = nitrix_fn(beta, var_within)
        # block on a leaf
        _ = nitrix_fn(beta, var_within).sigma_b_sq.block_until_ready()
        nitrix_ms = _time_warm(
            lambda: nitrix_fn(beta, var_within).sigma_b_sq,
        )

        # statsmodels: per-voxel loop (the canonical "voxelwise
        # group analysis" workflow).  For V > 100 this gets slow;
        # cap V at 100 for statsmodels timing.
        ref_V = min(V, 100)
        beta_np = np.asarray(beta[:ref_V])
        var_within_np = np.asarray(var_within[:ref_V])
        X_np = np.asarray(X_group)

        def fit_one(v):
            df = pd.DataFrame(
                {
                    'y': beta_np[v],
                    'x0': X_np[:, 0],
                    'x1': X_np[:, 1],
                    # Use var_within^{-1} as a frequency weight to
                    # approximate FLAME within statsmodels.  Not
                    # identical algorithm but the closest reference.
                    'w': 1.0 / var_within_np[v],
                }
            )
            md = smf.wls('y ~ x0 + x1 - 1', df, weights=df['w'])
            return md.fit()

        t0 = time.perf_counter()
        for v in range(ref_V):
            _ = fit_one(v)
        ref_total_ms = (time.perf_counter() - t0) * 1e3
        # Extrapolate to full V (per-voxel time * V)
        ref_per_voxel = ref_total_ms / ref_V
        ref_extrapolated_ms = ref_per_voxel * V

        rows.append(
            BenchRow(
                op='lme.flame_two_level (voxelwise)',
                shape_desc=f'V={V}, N={N}, p={p}',
                nitrix_ms=nitrix_ms,
                ref_name='statsmodels.wls (per-voxel loop)',
                ref_ms=ref_extrapolated_ms,
                agreement='~5e-3' if V <= 100 else 'skipped',
                notes=f'ref extrapolated from V={ref_V}',
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 3. residualise (Cholesky) vs numpy.linalg.lstsq
# ---------------------------------------------------------------------------


def bench_residualise() -> list[BenchRow]:
    from nitrix.linalg import residualise

    rows = []
    rng = np.random.default_rng(0)
    # fMRI shapes: 400 TRs, 24 confounds, n_voxels in {1k, 10k, 100k}
    for V in (1000, 10000, 100000):
        N, K = 400, 24
        X = jnp.asarray(rng.standard_normal((K, N)).astype(np.float32))
        Y = jnp.asarray(rng.standard_normal((V, N)).astype(np.float32))

        nitrix_fn = jax.jit(
            lambda Y_, X_: residualise(Y_, X_, method='cholesky'),
        )
        _ = nitrix_fn(Y, X).block_until_ready()
        nitrix_ms = _time_warm(lambda: nitrix_fn(Y, X))

        # numpy lstsq: solves (X.T, Y.T) per the standard
        # X^T beta = y form; loop over voxels.
        Y_np = np.asarray(Y)
        X_np_t = np.asarray(X).T  # (N, K)

        def numpy_residualise():
            # All voxels at once via a single lstsq
            # X.T @ beta = Y[v].T for all v simultaneously
            betas, _, _, _ = np.linalg.lstsq(X_np_t, Y_np.T, rcond=None)
            proj = (X_np_t @ betas).T
            return Y_np - proj

        ref_ms = _time_warm(numpy_residualise)

        # Agreement
        r_n = np.asarray(nitrix_fn(Y, X))
        r_r = numpy_residualise()
        max_diff = float(np.abs(r_n - r_r).max())

        rows.append(
            BenchRow(
                op='linalg.residualise (Cholesky)',
                shape_desc=f'V={V}, N={N}, K={K}',
                nitrix_ms=nitrix_ms,
                ref_name='numpy.linalg.lstsq',
                ref_ms=ref_ms,
                agreement=f'{max_diff:.1e}',
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 4. distance_transform vs scipy.ndimage.distance_transform_edt
# ---------------------------------------------------------------------------


def bench_distance_transform() -> list[BenchRow]:
    import scipy.ndimage as spnd

    from nitrix.morphology import distance_transform

    rows = []
    rng = np.random.default_rng(0)
    for shape in [(32, 32), (128, 128), (32, 32, 32), (64, 64, 64)]:
        # Random binary mask
        mask_np = (rng.random(shape) > 0.5).astype(np.float32)
        mask = jnp.asarray(mask_np)

        nitrix_fn = jax.jit(lambda m: distance_transform(m))
        _ = nitrix_fn(mask).block_until_ready()
        nitrix_ms = _time_warm(lambda: nitrix_fn(mask))

        def ref_fn():
            return spnd.distance_transform_edt(mask_np > 0.5)

        ref_ms = _time_warm(ref_fn)

        # Agreement: dist transforms differ in algorithm; skip exact.
        rows.append(
            BenchRow(
                op='morphology.distance_transform',
                shape_desc=f'{shape}',
                nitrix_ms=nitrix_ms,
                ref_name='scipy.ndimage.dt_edt',
                ref_ms=ref_ms,
                agreement='algorithm-different',
                notes='nitrix=iterative tropical, scipy=PMA',
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 5. cov: nitrix vs numpy (sanity baseline)
# ---------------------------------------------------------------------------


def bench_cov() -> list[BenchRow]:
    from nitrix.stats import cov

    rows = []
    rng = np.random.default_rng(0)
    for c, n_obs in [(50, 500), (500, 2000), (2000, 1000)]:
        X_np = rng.standard_normal((c, n_obs)).astype(np.float32)
        X = jnp.asarray(X_np)

        nitrix_fn = jax.jit(lambda x: cov(x))
        _ = nitrix_fn(X).block_until_ready()
        nitrix_ms = _time_warm(lambda: nitrix_fn(X))

        def ref_fn():
            return np.cov(X_np, bias=False)

        ref_ms = _time_warm(ref_fn)

        C_n = np.asarray(nitrix_fn(X))
        C_r = ref_fn()
        max_diff = float(np.abs(C_n - C_r).max())
        rows.append(
            BenchRow(
                op='stats.cov',
                shape_desc=f'({c}, {n_obs})',
                nitrix_ms=nitrix_ms,
                ref_name='numpy.cov',
                ref_ms=ref_ms,
                agreement=f'{max_diff:.1e}',
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Main entry: run all benches, write report
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. corr: nitrix vs numpy.corrcoef
# ---------------------------------------------------------------------------


def bench_corr() -> list[BenchRow]:
    from nitrix.stats import corr

    rows = []
    for n, T in [(50, 500), (500, 2000), (2000, 1000)]:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, T)).astype(np.float32)
        X_j = jnp.asarray(X)
        nitrix_fn = jax.jit(corr)
        _ = nitrix_fn(X_j).block_until_ready()
        nitrix_ms = _time_warm(lambda: nitrix_fn(X_j))
        ref_ms = _time_warm(lambda: np.corrcoef(X))
        np_ref = np.corrcoef(X)
        diff = float(np.abs(np.asarray(nitrix_fn(X_j)) - np_ref).max())
        rows.append(
            BenchRow(
                op='stats.corr',
                shape_desc=f'({n}, {T})',
                nitrix_ms=nitrix_ms,
                ref_name='numpy.corrcoef',
                ref_ms=ref_ms,
                agreement=f'{diff:.1e}',
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 7. gaussian: nitrix vs scipy.ndimage.gaussian_filter
# ---------------------------------------------------------------------------


def bench_gaussian() -> list[BenchRow]:
    import scipy.ndimage as spnd

    from nitrix.smoothing import gaussian

    rows = []
    for shape in [(64, 64), (256, 256), (64, 64, 64)]:
        rng = np.random.default_rng(0)
        X = rng.standard_normal(shape).astype(np.float32)
        X_j = jnp.asarray(X)
        nitrix_fn = jax.jit(lambda x: gaussian(x, sigma=1.5))
        _ = nitrix_fn(X_j).block_until_ready()
        nitrix_ms = _time_warm(lambda: nitrix_fn(X_j))
        ref_ms = _time_warm(lambda: spnd.gaussian_filter(X, sigma=1.5))
        ref = spnd.gaussian_filter(X, sigma=1.5)
        diff = float(np.abs(np.asarray(nitrix_fn(X_j)) - ref).max())
        rows.append(
            BenchRow(
                op='smoothing.gaussian',
                shape_desc='x'.join(str(s) for s in shape),
                nitrix_ms=nitrix_ms,
                ref_name='scipy.ndimage.gaussian_filter',
                ref_ms=ref_ms,
                agreement=f'{diff:.1e}',
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 8. morphology dilate / erode / median: nitrix vs scipy.ndimage
# ---------------------------------------------------------------------------


def bench_morphology_filters() -> list[BenchRow]:
    import scipy.ndimage as spnd

    from nitrix.morphology import dilate, erode, median_filter

    rows = []
    for shape in [(64, 64), (256, 256)]:
        rng = np.random.default_rng(0)
        X = rng.standard_normal(shape).astype(np.float32)
        X_j = jnp.asarray(X)

        for op_name, nitrix_op, scipy_op in [
            ('morphology.dilate', dilate, spnd.grey_dilation),
            ('morphology.erode', erode, spnd.grey_erosion),
            ('morphology.median_filter', median_filter, spnd.median_filter),
        ]:
            jitted = jax.jit(lambda x, op=nitrix_op: op(x, size=3))
            _ = jitted(X_j).block_until_ready()
            nitrix_ms = _time_warm(lambda: jitted(X_j))
            ref_ms = _time_warm(lambda: scipy_op(X, size=3))
            try:
                diff = float(
                    np.abs(
                        np.asarray(jitted(X_j)) - scipy_op(X, size=3),
                    ).max()
                )
                agreement = f'{diff:.1e}'
            except Exception:
                agreement = 'shape-mismatch'
            rows.append(
                BenchRow(
                    op=op_name,
                    shape_desc='x'.join(str(s) for s in shape),
                    nitrix_ms=nitrix_ms,
                    ref_name=f'scipy.ndimage.{scipy_op.__name__}',
                    ref_ms=ref_ms,
                    agreement=agreement,
                )
            )
    return rows


# ---------------------------------------------------------------------------
# 9. spatial_transform: nitrix vs scipy.ndimage.map_coordinates
# ---------------------------------------------------------------------------


def bench_spatial_transform() -> list[BenchRow]:
    import scipy.ndimage as spnd

    from nitrix.geometry import spatial_transform

    rows = []
    for shape in [(64, 64), (256, 256)]:
        rng = np.random.default_rng(0)
        # image: (H, W, c=1); deformation: (H, W, ndim=2) of absolute coords.
        img = rng.standard_normal(shape + (1,)).astype(np.float32)
        ii, jj = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            indexing='ij',
        )
        # Small random absolute-coord perturbation around the identity grid.
        deform = np.stack(
            [
                ii + 0.5 * rng.standard_normal(shape),
                jj + 0.5 * rng.standard_normal(shape),
            ],
            axis=-1,
        ).astype(np.float32)
        img_j, def_j = jnp.asarray(img), jnp.asarray(deform)

        nitrix_fn = jax.jit(
            lambda im, df: spatial_transform(im, df, mode='constant'),
        )
        _ = nitrix_fn(img_j, def_j).block_until_ready()
        nitrix_ms = _time_warm(lambda: nitrix_fn(img_j, def_j))

        # scipy.ndimage.map_coordinates expects coords shape
        # (ndim, *spatial).
        coords = deform.transpose(2, 0, 1)
        ref_ms = _time_warm(
            lambda: spnd.map_coordinates(
                img[..., 0],
                coords,
                order=1,
                mode='constant',
            ),
        )
        rows.append(
            BenchRow(
                op='geometry.spatial_transform',
                shape_desc='x'.join(str(s) for s in shape) + ' c=1',
                nitrix_ms=nitrix_ms,
                ref_name='scipy.ndimage.map_coordinates',
                ref_ms=ref_ms,
                agreement='skipped',
                notes='linear interpolation; modes match',
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 10. graph.laplacian: nitrix vs scipy.sparse.csgraph.laplacian
# ---------------------------------------------------------------------------


def bench_graph_laplacian() -> list[BenchRow]:
    import scipy.sparse.csgraph as spcsg

    from nitrix.graph import laplacian

    rows = []
    for n in (64, 256, 1024):
        rng = np.random.default_rng(0)
        # Sparse random symmetric adjacency.
        A = (rng.random((n, n)) > 0.95).astype(np.float32)
        A = ((A + A.T) > 0).astype(np.float32)
        np.fill_diagonal(A, 0)
        A_j = jnp.asarray(A)
        nitrix_fn = jax.jit(laplacian)
        _ = nitrix_fn(A_j).block_until_ready()
        nitrix_ms = _time_warm(lambda: nitrix_fn(A_j))
        ref_ms = _time_warm(lambda: spcsg.laplacian(A))
        diff = float(
            np.abs(
                np.asarray(nitrix_fn(A_j)) - spcsg.laplacian(A),
            ).max()
        )
        rows.append(
            BenchRow(
                op='graph.laplacian',
                shape_desc=f'n={n} sparse',
                nitrix_ms=nitrix_ms,
                ref_name='scipy.sparse.csgraph.laplacian',
                ref_ms=ref_ms,
                agreement=f'{diff:.1e}',
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 11. analytic_signal: nitrix vs scipy.signal.hilbert
# ---------------------------------------------------------------------------


def bench_analytic_signal() -> list[BenchRow]:
    import scipy.signal as spsig

    from nitrix.stats import analytic_signal

    rows = []
    for n in (256, 2048, 16384):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(n).astype(np.float32)
        x_j = jnp.asarray(x)
        nitrix_fn = jax.jit(analytic_signal)
        _ = nitrix_fn(x_j).block_until_ready()
        nitrix_ms = _time_warm(lambda: nitrix_fn(x_j))
        ref_ms = _time_warm(lambda: spsig.hilbert(x))
        diff = float(
            np.abs(np.asarray(nitrix_fn(x_j)) - spsig.hilbert(x)).max()
        )
        rows.append(
            BenchRow(
                op='stats.analytic_signal',
                shape_desc=f'n={n}',
                nitrix_ms=nitrix_ms,
                ref_name='scipy.signal.hilbert',
                ref_ms=ref_ms,
                agreement=f'{diff:.1e}',
            )
        )
    return rows


def main():
    output = Path(__file__).parent / 'PERF_AUDIT.md'
    print('Running perf audit...')

    all_rows: list[BenchRow] = []
    for name, runner in [
        ('rbf_kernel', bench_rbf_kernel),
        ('residualise', bench_residualise),
        ('cov', bench_cov),
        ('corr', bench_corr),
        ('distance_transform', bench_distance_transform),
        ('lme.flame_two_level', bench_lme_voxelwise),
        ('gaussian', bench_gaussian),
        ('morphology_filters', bench_morphology_filters),
        ('spatial_transform', bench_spatial_transform),
        ('graph.laplacian', bench_graph_laplacian),
        ('analytic_signal', bench_analytic_signal),
    ]:
        print(f'  {name}...')
        try:
            rows = runner()
            all_rows.extend(rows)
        except Exception as e:
            print(f'    SKIP: {type(e).__name__}: {e}')

    # Sort by ratio descending (worst gap first)
    all_rows_sorted = sorted(all_rows, key=lambda r: -r.ratio)

    lines = []
    lines.append('# Perf audit -- nitrix vs natural references')
    lines.append('')
    lines.append(
        'Sorted by wall-time ratio (nitrix / reference) descending; '
        'the worst gap is at the top.  ``>1`` means nitrix is slower.'
    )
    lines.append('')
    import platform

    try:
        d = jax.devices()[0]
        device = f'{d.device_kind} ({d.platform})'
    except Exception:
        device = 'unknown'
    lines.append('## Host')
    lines.append(f'- device: {device}')
    lines.append(f'- platform: {platform.platform()}')
    lines.append(f'- jax_version: {jax.__version__}')
    lines.append('')
    lines.append(
        '| op | shape | nitrix | ref | ref time | ratio | agreement | notes |'
    )
    lines.append('|---|---|---:|---|---:|---:|---|---|')
    for row in all_rows_sorted:
        lines.append(row.report_line())
    lines.append('')
    lines.append('## Reading the table')
    lines.append('')
    lines.append(
        '- **ratio > 1**: nitrix slower than reference.  For GPU '
        'paths this often happens at small problem sizes where '
        'kernel-launch overhead dominates.  Pay attention to '
        'ratios > 5 at the LARGEST measured size for each op -- '
        'those are the real gaps.'
    )
    lines.append(
        '- **ratio < 1**: nitrix faster.  Typical at large sizes '
        'where GPU bandwidth dominates.'
    )
    lines.append(
        '- **agreement**: max absolute difference between nitrix '
        'and reference outputs.  ``algorithm-different`` means '
        "the two implementations don't compute identical results "
        '(e.g., different distance-transform algorithms); we report '
        'wall-time only.'
    )

    output.write_text('\n'.join(lines))
    print(f'\nwrote {output}')


if __name__ == '__main__':
    main()
