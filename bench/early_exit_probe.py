# -*- coding: utf-8 -*-
"""Gate probe for the single-pair while_loop early-exit (lever B).

The registration optimisers run a FIXED iteration count (`lax.scan`); a
convergence-gated `lax.while_loop` would skip the spin after the solver has
converged.  The acceptance gate (docs/feature-requests/registration-early-
stopping-while-loop.md) is a **clean win on a hard case** -- the easy warp
always favours early-exit and is unrepresentative.

This probe builds a while_loop early-exit IC driver beside the fixed-scan one
and compares, on a HARD and an EASY rigid pair: recovery (must match) and warm
wall-clock + the iteration count actually used.

Run::  ./.venv/bin/python bench/early_exit_probe.py
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from _util import bench_call, timed_jit  # type: ignore[import-not-found]

from nitrix.geometry import (
    affine_grid,
    gaussian_pyramid,
    rigid_exp,
    spatial_transform,
)
from nitrix.metrics import ncc
from nitrix.register import RegistrationSpec, rigid_register
from nitrix.register._inverse_compositional import (
    _rigid_compositional_update,
    ic_reference,
)


def _blobs(n=96, seed=0):
    grids = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    rng = np.random.RandomState(seed)
    img = np.zeros((n, n), dtype='float32')
    for _ in range(6):
        c = rng.uniform(0.3, 0.7, 2) * n
        s = rng.uniform(0.1, 0.16) * n
        img += rng.uniform(0.4, 1.0) * np.exp(
            -sum((g - ci) ** 2 for g, ci in zip(grids, c)) / (2 * s * s)
        )
    return jnp.asarray(img)


def _pair(n, params):
    fixed = _blobs(n)
    center = (jnp.asarray((n, n), dtype=fixed.dtype) - 1.0) / 2.0
    grid = affine_grid(
        rigid_exp(jnp.asarray(params), ndim=2), (n, n), center=center
    )
    moving = spatial_transform(fixed[..., None], grid, mode='nearest')[..., 0]
    return moving, fixed


def _ic_level_while(moving, ref, matrix0, *, ndim, spec, tol, max_iters):
    fixed, sd, h_inv, center = ref

    def step_delta(matrix):
        grid = affine_grid(matrix, fixed.shape, center=center)
        warped = spatial_transform(
            moving[..., None], grid, mode=spec.boundary_mode, cval=spec.cval
        )[..., 0]
        err = (warped - fixed).reshape(-1)
        return h_inv @ (sd.T @ err)

    def cond(carry):
        _, i, dnorm = carry
        # converge on the parameter update magnitude ||delta|| (radians+voxels)
        return (i < max_iters) & (dnorm > tol)

    def body(carry):
        matrix, i, _ = carry
        delta = step_delta(matrix)
        update = _rigid_compositional_update(delta, ndim)
        return matrix @ update, i + 1, jnp.sqrt(jnp.sum(delta * delta))

    init = (matrix0, jnp.asarray(0), jnp.asarray(jnp.inf, moving.dtype))
    matrix, iters, _ = jax.lax.while_loop(cond, body, init)
    return matrix, iters


def _ic_register_while(moving, fixed, *, spec, tol, max_iters):
    ndim = 2
    pyr_f = gaussian_pyramid(
        fixed[..., None], levels=spec.levels, factor=spec.pyramid_factor
    )
    ref_levels = ic_reference(pyr_f, ndim)
    pyr_m = gaussian_pyramid(
        moving[..., None], levels=spec.levels, factor=spec.pyramid_factor
    )
    matrix = jnp.eye(ndim + 1, dtype=moving.dtype)
    total = jnp.asarray(0)
    prev_shape = None
    for level in range(spec.levels - 1, -1, -1):
        ref = ref_levels[level]
        m_l = pyr_m[level][..., 0]
        f_shape = ref[0].shape
        if prev_shape is not None:
            ratio = jnp.asarray(f_shape, dtype=moving.dtype) / jnp.asarray(
                prev_shape, dtype=moving.dtype
            )
            matrix = matrix.at[:ndim, ndim].set(matrix[:ndim, ndim] * ratio)
        matrix, iters = _ic_level_while(
            m_l,
            ref,
            matrix,
            ndim=ndim,
            spec=spec,
            tol=tol,
            max_iters=max_iters,
        )
        total = total + iters
        prev_shape = f_shape
    full = ref_levels[0][0].shape
    center = (jnp.asarray(full, dtype=moving.dtype) - 1.0) / 2.0
    grid = affine_grid(matrix, full, center=center)
    warped = spatial_transform(
        moving[..., None], grid, mode=spec.boundary_mode
    )[..., 0]
    return warped, total


def _run(label, params, n=96):
    moving, fixed = _pair(n, params)
    spec = RegistrationSpec(levels=3, iterations=30)
    init = float(ncc(moving, fixed))

    scan_fn = timed_jit(lambda m, f: rigid_register(m, f, spec=spec).warped)
    s_scan = bench_call(scan_fn, moving, fixed, warmup=2, repeats=5)
    ncc_scan = float(ncc(jax.block_until_ready(scan_fn(moving, fixed)), fixed))

    while_fn = timed_jit(
        lambda m, f: _ic_register_while(
            m, f, spec=spec, tol=1e-4, max_iters=30
        )
    )
    s_while = bench_call(while_fn, moving, fixed, warmup=2, repeats=5)
    warped_w, iters_w = while_fn(moving, fixed)
    ncc_while = float(ncc(jax.block_until_ready(warped_w), fixed))

    print(
        f'  {label}: init {init:.3f} | scan(30x3) {s_scan.warm_s * 1e3:6.1f}ms '
        f'ncc {ncc_scan:.4f} | while(tol1e-4) {s_while.warm_s * 1e3:6.1f}ms '
        f'ncc {ncc_while:.4f} iters {int(iters_w)} | '
        f'speedup {s_scan.warm_s / s_while.warm_s:.2f}x'
    )


def main() -> Any:
    print('while_loop early-exit gate probe (single-pair rigid IC, 96^2):\n')
    _run('easy (5deg, 2vox)', [0.09, 2.0, -1.5])
    _run('hard (28deg, 14vox)', [0.49, 14.0, -11.0])


if __name__ == '__main__':
    main()
