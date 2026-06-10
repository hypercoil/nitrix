# -*- coding: utf-8 -*-
"""Single-pair rigid: forward Gauss-Newton/LM vs inverse-compositional.

Quantifies the top rigid/affine perf lever — wiring the existing
inverse-compositional kernel (constant-template Hessian, `_inverse_-
compositional.py`, today used only by `volreg`) into the *single-pair*
`rigid_register`. The forward path re-linearises the moving image every
iteration (`jacfwd` = P tangent warps + the residual warp + LM's trial-cost
warp); the IC path builds the steepest-descent + Hessian once per level and
each iteration is one warp + one (M,P) projection.

Includes the IC reference build in the timed region (the honest single-pair
cost — no amortisation over a series). Reports warm wall-clock and recovery
NCC (IC must recover too, not just be fast).

Run::  ./.venv/bin/python bench/ic_vs_forward.py
"""

from __future__ import annotations

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
    ic_reference,
    ic_register_core,
)


def _blobs(shape, seed=0):
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
    rng = np.random.RandomState(seed)
    img = np.zeros(shape, dtype='float32')
    for _ in range(6):
        c = [rng.uniform(0.25, 0.75) * s for s in shape]
        sig = rng.uniform(0.1, 0.18) * min(shape)
        amp = rng.uniform(0.4, 1.0)
        r2 = sum((g - ci) ** 2 for g, ci in zip(grids, c))
        img += amp * np.exp(-r2 / (2 * sig * sig))
    return jnp.asarray(img)


def _rigid_pair(shape, params):
    fixed = _blobs(shape)
    ndim = len(shape)
    center = (jnp.asarray(shape, dtype=fixed.dtype) - 1.0) / 2.0
    grid = affine_grid(
        rigid_exp(jnp.asarray(params), ndim=ndim), shape, center=center
    )
    moving = spatial_transform(fixed[..., None], grid, mode='nearest')[..., 0]
    return moving, fixed


def _ic_full(spec, ndim):
    def f(moving, fixed):
        pyr_f = gaussian_pyramid(
            fixed[..., None], levels=spec.levels, factor=spec.pyramid_factor
        )
        ref = ic_reference(pyr_f, ndim)
        init = jnp.eye(ndim + 1, dtype=moving.dtype)
        return ic_register_core(
            moving, ref, ndim=ndim, spec=spec, init_matrix=init
        )

    return f


def main():
    spec = RegistrationSpec(levels=3, iterations=20)
    params = [0.20, -0.15, 0.18, 8.0, -6.0, 7.0]  # a hard-ish 3-D rigid warp
    print('single-pair rigid, L3x20, f32 (forward GN/LM vs IC):\n')
    for n in (64, 96, 128):
        shape = (n, n, n)
        moving, fixed = _rigid_pair(shape, params)
        init = float(ncc(moving, fixed))

        fwd = timed_jit(lambda m, f: rigid_register(m, f, spec=spec).warped)
        s_fwd = bench_call(fwd, moving, fixed, warmup=2, repeats=5)
        ncc_fwd = float(ncc(jax.block_until_ready(fwd(moving, fixed)), fixed))

        ic = timed_jit(lambda m, f: _ic_full(spec, 3)(m, f).warped)
        s_ic = bench_call(ic, moving, fixed, warmup=2, repeats=5)
        ncc_ic = float(ncc(jax.block_until_ready(ic(moving, fixed)), fixed))

        print(
            f'  {n}^3: init ncc {init:.3f} | '
            f'forward warm {s_fwd.warm_s * 1e3:7.1f}ms (ncc {ncc_fwd:.3f}) | '
            f'IC warm {s_ic.warm_s * 1e3:7.1f}ms (ncc {ncc_ic:.3f}) | '
            f'speedup {s_fwd.warm_s / s_ic.warm_s:.1f}x'
        )


if __name__ == '__main__':
    main()
