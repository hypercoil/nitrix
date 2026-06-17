# -*- coding: utf-8 -*-
"""Fresh-process GPU validation for the LME size-dispatch / cuSOLVER bypass.

The companion FR
(``docs/feature-requests/gpu-cusolver-first-call-handle-failure.md``) is
explicit: a cuSOLVER handle-creation failure is intermittent and process-state
dependent, so a change to this path must be shown robust by *repeated
measurement* -- "treat one green run as no evidence."  This script is one such
fresh-process trial.  Run it many times (see the ``--repeat`` driver in the
module docstring of ``run``); each invocation is a brand-new interpreter, so
the op under test is the *first* cuSOLVER activity in the process -- exactly the
condition that used to skip ``flame_two_level``.

Each trial, on the default backend (GPU here):

1. runs ``flame_two_level`` (p=1 intercept -- the dominant FLAME design) and
   ``reml_fit`` (p=3 -- exercises the general hand-Cholesky path) and forces
   the results (so a late cuSOLVER-handle failure surfaces),
2. re-runs both with inputs pinned to CPU as the oracle,
3. asserts the GPU output is finite and matches the CPU output.

Exits 0 (PASS) / 1 (FAIL) and prints a one-line verdict, so a shell loop can
tally a pass rate across >=50 processes.

Usage::

    n=50; pass=0
    for i in $(seq $n); do
        ./.venv/bin/python bench/validate_lme_gpu.py >/dev/null 2>&1 && pass=$((pass+1))
    done
    echo "$pass / $n"
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.stats.lme import flame_two_level, reml_fit


def _flame_inputs(rng: np.random.Generator):
    V, N = 8192, 60
    beta = jnp.asarray(rng.standard_normal((V, N)) * 0.5 + 0.3)
    var_within = jnp.asarray(np.abs(rng.standard_normal((V, N))) * 0.4 + 0.1)
    X_group = jnp.ones((N, 1))  # p = 1
    return beta, var_within, X_group


def _reml_inputs(rng: np.random.Generator):
    g, n_per, V = 8, 25, 1024
    N = g * n_per
    group = np.repeat(np.arange(g), n_per)
    X = jnp.asarray(
        np.column_stack(
            [np.ones(N), rng.standard_normal(N), rng.standard_normal(N)]
        )
    )  # p = 3
    Z = np.zeros((N, g))
    for i in range(g):
        Z[group == i, i] = 1.0
    Y = jnp.asarray(rng.standard_normal((V, N)))
    return Y, X, jnp.asarray(Z)


def main() -> int:
    cpu = jax.devices('cpu')[0]
    rng = np.random.default_rng(0)

    # --- FLAME (the op that used to skip on GPU) ---
    fb = _flame_inputs(rng)
    fr_gpu = flame_two_level(*fb, n_iter=30)
    jax.block_until_ready((fr_gpu.sigma_b_sq, fr_gpu.gamma_hat))
    fb_cpu = jax.tree_util.tree_map(lambda a: jax.device_put(a, cpu), fb)
    fr_cpu = flame_two_level(*fb_cpu, n_iter=30)

    if not bool(jnp.all(jnp.isfinite(fr_gpu.sigma_b_sq))):
        print('FAIL flame: non-finite sigma_b_sq on GPU')
        return 1
    if 'cuda' not in str(fr_gpu.sigma_b_sq.devices()).lower():
        print(f'FAIL flame: did not run on GPU ({fr_gpu.sigma_b_sq.devices()})')
        return 1
    fb_err = float(
        np.max(np.abs(np.asarray(fr_gpu.sigma_b_sq) - np.asarray(fr_cpu.sigma_b_sq)))
    )
    if fb_err > 1e-6:
        print(f'FAIL flame: GPU vs CPU sigma_b_sq max|diff|={fb_err:.2e}')
        return 1

    # --- REML p=3 (general hand-Cholesky path) ---
    rb = _reml_inputs(rng)
    rr_gpu = reml_fit(*rb, n_iter=50)
    jax.block_until_ready((rr_gpu.theta_hat, rr_gpu.beta_hat))
    if not bool(jnp.all(jnp.isfinite(rr_gpu.beta_hat))):
        print('FAIL reml: non-finite beta_hat on GPU')
        return 1

    print(f'PASS  flame sigma_b_sq GPU/CPU max|diff|={fb_err:.2e}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
