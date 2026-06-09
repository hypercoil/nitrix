# -*- coding: utf-8 -*-
"""Performance benchmark for the registration recipes.

Times the GPU-native registration paths forward and
forward-plus-backward, on representative 2-D and 3-D volumes:

- ``rigid_register`` / ``affine_register`` (SSD, matrix-free
  Gauss-Newton/LM with the assembled small-``P`` normal equations);
- ``diffeomorphic_demons_register`` (log-Demons);
- a **non-SSD differentiable layer**: an LNCC rigid registration whose
  argmin is differentiated through the optimum by ``implicit_minimize``
  (the general scalar-objective IFT) -- the path the BFGS metrics
  (LNCC/MI/CR) take to become a differentiable layer.

The forward-plus-backward number is the **differentiable-layer** cost:
``jax.grad`` of a scalar loss of the warped image w.r.t. the moving image
(the unrolled ``lax.scan`` path for the recipes; the implicit-function
backward for the ``implicit_minimize`` row).  Everything is
``jax.jit``-compiled with the static config captured, so the timed loop
is steady-state (see ``bench/_util.py`` -- first calls don't count).

All paths are pure on-device composition (``matrix_exp`` is pure-matmul,
the SPD solves are matrix-free ``cg``, no factorisation), so the numbers
transfer to a healthy GPU even though the dev box has a wedged cuSolver
pool.

Run::

    python bench/perf_registration.py

Writes ``PERF_REGISTRATION.md`` alongside this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from _util import bench_call, timed_jit  # type: ignore[import-not-found]

from nitrix.geometry import affine_grid, rigid_exp, spatial_transform
from nitrix.linalg import implicit_minimize
from nitrix.register import (
    LNCC,
    DemonsSpec,
    RegistrationSpec,
    Rigid,
    affine_register,
    diffeomorphic_demons_register,
    rigid_register,
)


def _blobs(shape: tuple[int, ...], seed: int = 0) -> jax.Array:
    """A smooth multi-blob test image of the given spatial shape."""
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
    rng = np.random.RandomState(seed)
    img = np.zeros(shape, dtype='float64')
    for _ in range(6):
        center = [rng.uniform(0.25, 0.75) * s for s in shape]
        sigma = rng.uniform(0.1, 0.18) * min(shape)
        amp = rng.uniform(0.4, 1.0)
        r2 = sum((g - c) ** 2 for g, c in zip(grids, center))
        img += amp * np.exp(-r2 / (2 * sigma * sigma))
    return jnp.asarray(img)


def _make_pair(shape: tuple[int, ...]) -> tuple[jax.Array, jax.Array]:
    fixed = _blobs(shape)
    ndim = len(shape)
    center = (jnp.asarray(shape, dtype=fixed.dtype) - 1.0) / 2.0
    if ndim == 2:
        params = jnp.asarray([0.08, 2.0, -1.5])
    else:
        params = jnp.asarray([0.05, -0.04, 0.06, 1.5, -1.0, 1.2])
    grid = affine_grid(rigid_exp(params, ndim=ndim), shape, center=center)
    moving = spatial_transform(fixed[..., None], grid, mode='nearest')[..., 0]
    return moving, fixed


def _implicit_lncc_warp(
    moving: jax.Array,
    fixed: jax.Array,
    *,
    maxiter: int,
) -> jax.Array:
    """LNCC rigid registration whose argmin is differentiated through the
    optimum by ``implicit_minimize`` (the non-SSD differentiable layer).

    A single-resolution registration: minimise the LNCC cost over the
    rigid parameters with BFGS, then return ``moving`` warped by the
    recovered transform.  ``jax.grad`` w.r.t. ``moving`` exercises the
    exact-Hessian implicit-function backward.
    """
    model = Rigid()
    metric = LNCC()
    ndim = moving.ndim
    center = (jnp.asarray(moving.shape, dtype=moving.dtype) - 1.0) / 2.0

    def warp(img: jax.Array, theta: jax.Array) -> jax.Array:
        grid = affine_grid(
            model.exp(theta, ndim=ndim), img.shape, center=center
        )
        return spatial_transform(img[..., None], grid, mode='nearest')[..., 0]

    def objective(data: tuple[jax.Array, jax.Array], theta: jax.Array):
        mov, fix = data
        return metric.cost(warp(mov, theta), fix)

    theta = implicit_minimize(
        objective,
        (moving, fixed),
        jnp.zeros(model.n_params(ndim), dtype=moving.dtype),
        maxiter=maxiter,
        ridge=1e-6,
    )
    return warp(moving, theta)


def _bench_recipe(
    label: str,
    recipe: Callable[[jax.Array, jax.Array], jax.Array],
    moving: jax.Array,
    fixed: jax.Array,
) -> tuple[str, float, float, float, float]:
    fwd = timed_jit(recipe)

    def loss(m: jax.Array, f: jax.Array) -> jax.Array:
        return jnp.sum(recipe(m, f) ** 2)

    grad = timed_jit(jax.grad(loss))
    s_fwd = bench_call(fwd, moving, fixed, warmup=2, repeats=5)
    s_bwd = bench_call(grad, moving, fixed, warmup=2, repeats=5)
    return (
        label,
        s_fwd.compile_s,
        s_fwd.warm_s,
        s_bwd.compile_s,
        s_bwd.warm_s,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--levels', type=int, default=3)
    parser.add_argument('--rigid-iters', type=int, default=20)
    parser.add_argument('--demons-iters', type=int, default=20)
    parser.add_argument('--lncc-iters', type=int, default=30)
    args = parser.parse_args()

    rspec = RegistrationSpec(levels=args.levels, iterations=args.rigid_iters)
    dspec = DemonsSpec(levels=args.levels, iterations=args.demons_iters)

    cases = [
        ('2-D 128x128', (128, 128)),
        ('3-D 48x48x48', (48, 48, 48)),
    ]
    rows = []
    for name, shape in cases:
        moving, fixed = _make_pair(shape)
        rows.append(
            (name, 'rigid (SSD/LM)')
            + _bench_recipe(
                name,
                lambda m, f: rigid_register(m, f, spec=rspec).warped,
                moving,
                fixed,
            )[1:]
        )
        rows.append(
            (name, 'affine (SSD/LM)')
            + _bench_recipe(
                name,
                lambda m, f: affine_register(m, f, spec=rspec).warped,
                moving,
                fixed,
            )[1:]
        )
        rows.append(
            (name, 'demons')
            + _bench_recipe(
                name,
                lambda m, f: (
                    diffeomorphic_demons_register(m, f, spec=dspec).warped
                ),
                moving,
                fixed,
            )[1:]
        )
        rows.append(
            (name, 'rigid LNCC (implicit)')
            + _bench_recipe(
                name,
                lambda m, f: _implicit_lncc_warp(
                    m, f, maxiter=args.lncc_iters
                ),
                moving,
                fixed,
            )[1:]
        )

    backend = jax.default_backend()
    lines = [
        '# Registration recipe benchmark',
        '',
        f'- Backend: `{backend}` ({jax.devices()[0].device_kind})',
        f'- Spec: levels={args.levels}, rigid_iters={args.rigid_iters}, '
        f'demons_iters={args.demons_iters}, lncc_iters={args.lncc_iters}',
        '- Warm time is the median post-warmup wall-time; compile is the '
        'first (trace+compile) call.',
        "- Compile is **cold** (fresh XLA cache); JAX's persistent "
        'compilation cache amortises it across runs of the same shapes, so '
        'a deployment pays each shape once, not per process.',
        '',
        '| case | recipe | fwd compile (s) | fwd warm (s) | '
        'grad compile (s) | grad warm (s) |',
        '|---|---|---|---|---|---|',
    ]
    for name, recipe, fc, fw, gc, gw in rows:
        lines.append(
            f'| {name} | {recipe} | {fc:.2f} | {fw:.4f} | {gc:.2f} | {gw:.4f} |'
        )
    report = '\n'.join(lines) + '\n'
    out = Path(__file__).with_name('PERF_REGISTRATION.md')
    out.write_text(report)
    print(report)


if __name__ == '__main__':
    main()
