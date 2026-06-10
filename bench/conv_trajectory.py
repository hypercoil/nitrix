# -*- coding: utf-8 -*-
"""Convergence-trajectory / iso-accuracy probe for the registration recipes.

The fair comparison against ANTs (which early-stops) is iso-accuracy:
wall-clock to reach a target recovery, which counts both per-iteration cost
*and* iteration count.  This probe measures the iteration-count axis directly
-- how much of nitrix's FIXED ``lax.scan`` budget is actually needed -- on an
**easy** vs a **hard** (large-deformation) case, for rigid and greedy SyN.

For each run we read the recipe's ``cost_history`` (concatenated per-iteration
``cost`` over levels), isolate the finest level, and find the iteration at
which the cost reaches 95 % / 99 % of that level's total improvement.  We also
report the actual recovery NCC.

Decision relevance:
- If the **hard** case still converges well before budget -> a ``while_loop``
  early-exit would help even on representative cases (but breaks vmap-cohort).
- If the **hard** case needs most of its budget -> the fixed scan is justified;
  the easy-case early-exit saving is unrepresentative (the gate's warning).

Run::  ./.venv/bin/python bench/conv_trajectory.py
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nitrix.geometry import (
    affine_grid,
    identity_grid,
    integrate_velocity_field,
    rigid_exp,
    spatial_transform,
)
from nitrix.metrics import ncc
from nitrix.register import (
    RegistrationSpec,
    SyNSpec,
    greedy_syn_register,
    rigid_register,
)
from nitrix.smoothing import gaussian


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


def _smooth_velocity(shape, sigma, target_disp, seed):
    """Smooth random velocity field renormalised so its max per-voxel
    displacement magnitude is ``target_disp`` voxels (heavy Gaussian
    smoothing otherwise crushes the amplitude to ~0)."""
    ndim = len(shape)
    rng = np.random.RandomState(seed)
    v = rng.standard_normal(shape + (ndim,)).astype('float32')
    v = np.moveaxis(v, -1, 0)
    v = np.asarray(gaussian(jnp.asarray(v), sigma=sigma, spatial_rank=ndim))
    v = np.moveaxis(v, 0, -1)
    norm = np.sqrt((v**2).sum(-1)).max()
    return jnp.asarray((target_disp / (norm + 1e-12)) * v)


def _rigid_pair(shape, params):
    fixed = _blobs(shape)
    ndim = len(shape)
    center = (jnp.asarray(shape, dtype=fixed.dtype) - 1.0) / 2.0
    grid = affine_grid(
        rigid_exp(jnp.asarray(params), ndim=ndim), shape, center=center
    )
    moving = spatial_transform(fixed[..., None], grid, mode='nearest')[..., 0]
    return moving, fixed


def _syn_pair(shape, target_disp):
    fixed = _blobs(shape)
    v = _smooth_velocity(
        shape, sigma=0.10 * min(shape), target_disp=target_disp, seed=1
    )
    grid = identity_grid(shape, dtype=fixed.dtype)
    s = integrate_velocity_field(v)
    moving = spatial_transform(fixed[..., None], grid + s, mode='nearest')[
        ..., 0
    ]
    return moving, fixed


def _iters_to_fraction(level_costs, fracs=(0.95, 0.99)):
    """Iteration (1-based) at which a level's cost first reaches each fraction
    of its total (first - last) improvement."""
    c = np.asarray(level_costs)
    total = c[0] - c[-1]
    out = {}
    for fr in fracs:
        if total <= 0:
            out[fr] = 0
            continue
        target = c[0] - fr * total
        hit = np.argmax(c <= target)
        out[fr] = int(hit) + 1 if c[hit] <= target else len(c)
    return out


def _analyse(label, cost_history, levels, iters, init_ncc, final_ncc):
    ch = np.asarray(cost_history)
    per = len(ch) // levels  # may be iters or iters+1 (initial cost recorded)
    finest = ch.reshape(levels, per)[-1]
    fr = _iters_to_fraction(finest)
    budget = per - 1
    print(
        f'  {label}: init ncc {init_ncc:.3f} -> final {final_ncc:.3f} | '
        f'finest level cost {finest[0]:.4g} -> {finest[-1]:.4g} | '
        f'steps to 95%={fr[0.95]}/{budget}, 99%={fr[0.99]}/{budget}'
    )
    return fr


def main():
    shape = (96, 96, 96)
    print(f'Convergence trajectory @ {shape}, f32 (finest level shown):\n')

    # --- rigid: easy vs hard ---
    rlev, rit = 3, 20
    rspec = RegistrationSpec(levels=rlev, iterations=rit)
    print('rigid (SSD/LM), levels=3 iters=20:')
    for label, params in [
        ('easy  (5deg, ~2vox)', [0.05, -0.04, 0.06, 1.5, -1.0, 1.2]),
        ('hard  (25deg, ~12vox)', [0.30, -0.25, 0.35, 12.0, -9.0, 10.0]),
    ]:
        m, f = _rigid_pair(shape, params)
        res = rigid_register(m, f, spec=rspec)
        _analyse(
            label,
            res.cost_history,
            rlev,
            rit,
            float(ncc(m, f)),
            float(ncc(res.warped, f)),
        )

    # --- greedy SyN: easy vs hard ---
    slev, sit = 3, 40
    print('\ngreedy SyN (LNCC), levels=3 iters=40:')
    for label, scale in [
        ('easy  (~3vox max disp)', 3.0),
        ('hard  (~9vox max disp)', 9.0),
    ]:
        m, f = _syn_pair(shape, scale)
        sspec = SyNSpec(levels=slev, iterations=sit, radius=2, step=0.5)
        res = greedy_syn_register(m, f, spec=sspec)
        _analyse(
            label,
            res.cost_history,
            slev,
            sit,
            float(ncc(m, f)),
            float(ncc(res.warped, f)),
        )


if __name__ == '__main__':
    main()
