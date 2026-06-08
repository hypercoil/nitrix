# -*- coding: utf-8 -*-
"""Performance benchmark for the registration recipes.

Times the two GPU-native recipes -- ``rigid_register`` (SSD, matrix-free
Gauss-Newton/LM) and ``diffeomorphic_demons_register`` (log-Demons) --
forward and forward-plus-backward, on representative 2-D and 3-D volumes.

The forward-plus-backward number is the **differentiable-layer** cost:
``jax.grad`` of a scalar loss of the warped image w.r.t. the moving image
(the unrolled path).  The recipes are ``jax.jit``-compiled with the
``spec`` captured statically, so the timed loop is steady-state (see
``bench/_util.py`` -- first calls don't count).

``affine_register`` and the BFGS metric paths are intentionally excluded:
on a stack with the cuSolver wedge the affine ``matrix_exp`` falls back to
CPU (a dev-box artifact, not a production cost) and BFGS is not
differentiable through the solve.  Rigid-SSD and Demons are pure on-device
composition, so their numbers transfer to a healthy GPU.

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
from nitrix.register import (
    DemonsSpec,
    RegistrationSpec,
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
            (name, 'demons')
            + _bench_recipe(
                name,
                lambda m, f: diffeomorphic_demons_register(
                    m, f, spec=dspec
                ).warped,
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
        f'demons_iters={args.demons_iters}',
        '- Warm time is the median post-warmup wall-time; compile is the '
        'first (trace+compile) call.',
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
