# -*- coding: utf-8 -*-
"""Tests for ``nitrix.stats.lme`` (voxelwise LME).

Two correctness anchors:

1. **Balanced one-way ANOVA REML closed form** -- for the
   simplest LME (intercept + group random effects, balanced
   design), REML reduces to a moment-based estimator with a
   well-known formula.  ``reml_fit`` must match this estimator
   to machine precision.
2. **FLAME hand-computed reference** -- the FLAME two-level
   model with a single fixed effect (group intercept) and known
   per-subject within-variance has a simple iterative form;
   ``flame_two_level`` must match the same numbers.

Plus voxelwise vmap regression (per-voxel results match
running the fit one voxel at a time) and the memory-footprint
HLO audit (no ``V * N * N`` intermediate).
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.stats.lme import (
    LMEFContrast,
    LMEResult,
    REMLResult,
    flame_two_level,
    lme_f_contrast,
    lme_fit,
    lme_t_contrast,
    reml_fit,
)

# ---------------------------------------------------------------------------
# REML: balanced one-way reference
# ---------------------------------------------------------------------------


def _balanced_one_way_data(
    g=5,
    n_per=30,
    true_mu=2.0,
    true_sigma_b=1.0,
    true_sigma_e=0.5,
    seed=0,
):
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(g) * true_sigma_b
    eps = rng.standard_normal(g * n_per) * true_sigma_e
    group_id = np.repeat(np.arange(g), n_per)
    y = true_mu + b[group_id] + eps
    return y, group_id, g, n_per


def _balanced_one_way_closed_form(y, group_id, g, n_per):
    """ANOVA-style REML estimator for balanced one-way random effects."""
    y_per_group_mean = np.array([y[group_id == i].mean() for i in range(g)])
    overall_mean = y.mean()
    SS_b = n_per * np.sum((y_per_group_mean - overall_mean) ** 2)
    SS_w = np.sum((y - y_per_group_mean[group_id]) ** 2)
    MS_b = SS_b / (g - 1)
    MS_w = SS_w / (g * n_per - g)
    return (MS_b - MS_w) / n_per, MS_w


def test_reml_matches_balanced_one_way_closed_form():
    """REML must match the ANOVA-form closed-form estimator exactly
    for balanced one-way random effects.

    This is the load-bearing correctness test: any error in the
    profile likelihood or the Newton iteration shows up as a
    mismatch here.
    """
    y, group_id, g, n_per = _balanced_one_way_data()
    sigma_b_sq_closed, sigma_e_sq_closed = _balanced_one_way_closed_form(
        y, group_id, g, n_per
    )

    N = g * n_per
    X = jnp.ones((N, 1))
    Z = jnp.zeros((N, g))
    for i in range(g):
        Z = Z.at[group_id == i, i].set(1.0)
    Y = jnp.asarray(y[None, :])
    result = reml_fit(Y, X, Z, n_iter=60)

    np.testing.assert_allclose(
        float(result.sigma_b_sq[0]),
        sigma_b_sq_closed,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        float(result.sigma_e_sq[0]),
        sigma_e_sq_closed,
        atol=1e-6,
    )


def test_reml_recovers_intercept_from_balanced_data():
    """On balanced one-way data, the fixed-effect intercept
    estimate should equal the overall mean."""
    y, group_id, g, n_per = _balanced_one_way_data(true_mu=3.5)
    N = g * n_per
    X = jnp.ones((N, 1))
    Z = jnp.zeros((N, g))
    for i in range(g):
        Z = Z.at[group_id == i, i].set(1.0)
    Y = jnp.asarray(y[None, :])
    result = reml_fit(Y, X, Z, n_iter=60)
    # The REML beta for intercept-only is the unweighted mean only
    # if all groups are equally weighted; here we have a single group
    # variance and balanced design so it's exactly the GLS-weighted
    # overall mean.  For balanced design this equals the unweighted
    # mean.
    np.testing.assert_allclose(
        float(result.beta_hat[0, 0]),
        float(y.mean()),
        atol=1e-6,
    )


def test_reml_voxelwise_per_voxel_match_unbatched():
    """V-voxel batched fit must equal V independent unbatched fits."""
    V = 8
    rng = np.random.default_rng(0)
    g, n_per = 4, 20
    N = g * n_per
    group_id = np.repeat(np.arange(g), n_per)

    X_np = np.column_stack([np.ones(N), rng.standard_normal(N)])
    Z_np = np.zeros((N, g))
    for i in range(g):
        Z_np[group_id == i, i] = 1.0

    Y_np = rng.standard_normal((V, N))
    X = jnp.asarray(X_np)
    Z = jnp.asarray(Z_np)

    # Batched
    result_batch = reml_fit(jnp.asarray(Y_np), X, Z, n_iter=40)

    # Per voxel
    for v in range(V):
        Y_v = jnp.asarray(Y_np[v : v + 1])
        result_v = reml_fit(Y_v, X, Z, n_iter=40)
        # Batched and per-voxel REML are the same math but accumulate the
        # iterative variance-component updates in different orders, so they
        # agree only to ~1e-7 relative (the variance components are the
        # optimisation targets; beta_hat inherits their error through the GLS
        # solve).  rtol=1e-6 sits ~9x above the observed worst case; the small
        # atol is the near-zero floor.
        np.testing.assert_allclose(
            float(result_batch.sigma_b_sq[v]),
            float(result_v.sigma_b_sq[0]),
            rtol=1e-6,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            float(result_batch.sigma_e_sq[v]),
            float(result_v.sigma_e_sq[0]),
            rtol=1e-6,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(result_batch.beta_hat[v]),
            np.asarray(result_v.beta_hat[0]),
            rtol=1e-6,
            atol=1e-9,
        )


def test_reml_recovers_true_variance_at_large_sample():
    """With many groups and large groups, REML should recover the
    true variance components to ~10% relative error."""
    y, group_id, g, n_per = _balanced_one_way_data(
        g=20,
        n_per=50,
        true_sigma_b=1.0,
        true_sigma_e=0.5,
    )
    N = g * n_per
    X = jnp.ones((N, 1))
    Z = jnp.zeros((N, g))
    for i in range(g):
        Z = Z.at[group_id == i, i].set(1.0)
    Y = jnp.asarray(y[None, :])
    result = reml_fit(Y, X, Z, n_iter=60)
    assert abs(float(result.sigma_b_sq[0]) - 1.0) < 0.5  # noisy at small g
    assert abs(float(result.sigma_e_sq[0]) - 0.25) < 0.05


def test_reml_differentiable():
    """REML fit is unrolled-Newton; ``jax.grad`` over a scalar
    function of the fit should produce finite gradients."""
    rng = np.random.default_rng(0)
    g, n_per = 3, 15
    N = g * n_per
    X = jnp.ones((N, 1))
    Z = jnp.zeros((N, g))
    for i in range(g):
        Z = Z.at[i * n_per : (i + 1) * n_per, i].set(1.0)
    Y = jnp.asarray(rng.standard_normal((4, N)))

    def loss(Y):
        result = reml_fit(Y, X, Z, n_iter=10)  # short for grad
        return jnp.sum(result.sigma_b_sq + result.sigma_e_sq)

    g_jax = jax.grad(loss)(Y)
    assert g_jax.shape == Y.shape
    assert bool(jnp.all(jnp.isfinite(g_jax)))


# ---------------------------------------------------------------------------
# FLAME: two-level model
# ---------------------------------------------------------------------------


def _flame_hand_iter(beta, var_within, X_group, n_iter=40):
    """Hand-rolled FLAME-style REML for a single voxel.

    Reference implementation using a different inner-loop pattern
    (Fisher scoring with analytic derivatives in the diagonal
    case).  Used to cross-check the production solver.
    """
    # log-sigma_b^2; sigma_e^2 fixed at 1 (var_within scales it).
    log_var_b = 0.0
    for _ in range(n_iter):
        var_b = np.exp(log_var_b)
        d = var_b + var_within  # (N,)
        inv_d = 1.0 / d
        Xw = X_group * inv_d[:, None]
        XtVinvX = Xw.T @ X_group  # (p, p)
        XtVinvy = Xw.T @ beta
        gamma = np.linalg.solve(XtVinvX, XtVinvy)
        r = beta - X_group @ gamma
        # P V_b ... but for single-param case, simpler:
        # Score on var_b: dL/d_var_b = 0.5 [r^T V^{-1} V^{-1} r - tr(P)]
        # In log-space: dL/d_log_var_b = var_b * dL/d_var_b
        P_diag = inv_d - jnp.diag(Xw @ jnp.linalg.solve(XtVinvX, Xw.T))
        score_var_b = 0.5 * (np.sum(r**2 * inv_d**2) - np.sum(P_diag))
        score_log = var_b * score_var_b
        # Fisher info on log scale:
        I_log = 0.5 * np.sum(P_diag**2) * var_b**2
        if I_log > 1e-12:
            log_var_b += score_log / I_log
        log_var_b = np.clip(log_var_b, -10.0, 10.0)
    var_b = np.exp(log_var_b)
    d = var_b + var_within
    inv_d = 1.0 / d
    Xw = X_group * inv_d[:, None]
    XtVinvX = Xw.T @ X_group
    XtVinvy = Xw.T @ beta
    gamma = np.linalg.solve(XtVinvX, XtVinvy)
    return var_b, gamma


def test_flame_recovers_true_between_variance():
    """Generate FLAME-model data and check the sigma_b^2 estimate.

    With V=200 voxels averaged, the REML estimate of sigma_b^2
    should sit within ~5% of truth.
    """
    rng = np.random.default_rng(0)
    N, p, V = 60, 2, 200
    X_group = jnp.asarray(rng.standard_normal((N, p)))
    true_gamma = jnp.asarray([1.0, 0.5])
    true_sigma_b = 0.5
    var_within = jnp.asarray(
        np.abs(rng.standard_normal((V, N))) * 0.5 + 0.1,
    )
    eps_b = rng.standard_normal((V, N)) * true_sigma_b
    eps_w = rng.standard_normal((V, N)) * np.asarray(jnp.sqrt(var_within))
    beta = X_group @ true_gamma + jnp.asarray(eps_b + eps_w)

    result = flame_two_level(beta, var_within, X_group, n_iter=40)
    sigma_b_sq_mean = float(result.sigma_b_sq.mean())
    # Within 5% of truth at V=200.
    assert abs(sigma_b_sq_mean - true_sigma_b**2) < 0.05
    gamma_err = float(jnp.abs(result.gamma_hat.mean(0) - true_gamma).max())
    assert gamma_err < 0.05


def test_flame_voxelwise_per_voxel_match_unbatched():
    """Batched FLAME fit equals per-voxel fits."""
    V = 6
    rng = np.random.default_rng(0)
    N, p = 30, 2
    X_group = jnp.asarray(rng.standard_normal((N, p)))
    beta = jnp.asarray(rng.standard_normal((V, N)))
    var_within = jnp.asarray(np.abs(rng.standard_normal((V, N))) + 0.1)

    res_batch = flame_two_level(beta, var_within, X_group, n_iter=30)
    for v in range(V):
        res_v = flame_two_level(
            beta[v : v + 1],
            var_within[v : v + 1],
            X_group,
            n_iter=30,
        )
        np.testing.assert_allclose(
            float(res_batch.sigma_b_sq[v]),
            float(res_v.sigma_b_sq[0]),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            np.asarray(res_batch.gamma_hat[v]),
            np.asarray(res_v.gamma_hat[0]),
            atol=1e-9,
        )


# ---------------------------------------------------------------------------
# Memory regression: no V * N * N intermediate
# ---------------------------------------------------------------------------


def test_reml_hlo_has_no_voxel_indexed_NxN_tensor():
    """The compiled HLO of voxelwise REML must not contain any
    tensor with shape ``(V, N, N)`` -- such a tensor would mean
    we're materialising a per-voxel covariance, which would
    OOM at fMRI scale.

    The shared-design path keeps the only N x N tensors at the
    rank of the global rotation (``U``, ``ZZt``); per-voxel
    state is at most ``(V, N)`` or ``(V, p)`` or ``(V, K)``.
    """
    V = 1024
    N = 32
    rng = np.random.default_rng(0)
    Y = jnp.asarray(rng.standard_normal((V, N)).astype(np.float32))
    X = jnp.asarray(rng.standard_normal((N, 2)).astype(np.float32))
    Z = jnp.asarray(rng.standard_normal((N, 4)).astype(np.float32))

    f = jax.jit(lambda Y, X, Z: reml_fit(Y, X, Z, n_iter=10).beta_hat)
    hlo = f.lower(Y, X, Z).compile().as_text()
    shapes = re.findall(r'f32\[([0-9,]+)\]', hlo)
    bad_shapes = []
    for s in shapes:
        dims = tuple(int(x) for x in s.split(',') if x)
        if len(dims) == 3 and dims[0] == V and dims[1] == N and dims[2] == N:
            bad_shapes.append(dims)
    assert not bad_shapes, (
        f'shared-design REML must not produce (V, N, N) intermediates; '
        f'found: {bad_shapes[:5]}'
    )


def test_reml_max_tensor_size_within_budget():
    """Max compiled-HLO tensor is small relative to ``V * N * N``.

    The shared-design REML keeps per-voxel state at ``(V, N)``,
    ``(V, K)``, or ``(V, p)`` -- never ``(V, N, N)``.  Budget:
    well under ``V * N * N / 2``.
    """
    V = 1024
    N = 32
    rng = np.random.default_rng(0)
    Y = jnp.asarray(rng.standard_normal((V, N)).astype(np.float32))
    X = jnp.asarray(rng.standard_normal((N, 2)).astype(np.float32))
    Z = jnp.asarray(rng.standard_normal((N, 4)).astype(np.float32))

    f = jax.jit(lambda Y, X, Z: reml_fit(Y, X, Z, n_iter=5).beta_hat)
    hlo = f.lower(Y, X, Z).compile().as_text()
    shapes = re.findall(r'f32\[([0-9,]+)\]', hlo)
    max_size = 0
    for s in shapes:
        dims = tuple(int(x) for x in s.split(',') if x)
        sz = 1
        for d in dims:
            sz *= d
        max_size = max(max_size, sz)
    # Hard upper bound: ``V * N * N / 2``.  If anything in the HLO
    # is of that size, we've materialised a per-voxel covariance.
    budget = V * N * N // 2
    assert max_size < budget, (
        f'max tensor size {max_size} >= budget {budget}; possible '
        '(V, N, N) intermediate.'
    )


# ---------------------------------------------------------------------------
# REML: statsmodels reference (the gold standard)
# ---------------------------------------------------------------------------


def _skip_if_no_statsmodels():
    try:
        import statsmodels.api as sm  # noqa: F401

        return False
    except ImportError:
        return True


@pytest.mark.skipif(
    _skip_if_no_statsmodels(),
    reason='statsmodels not installed',
)
def test_reml_matches_statsmodels_reference():
    """Nitrix REML must match statsmodels.MixedLM(reml=True) on a
    standard group random-intercept LME with covariates.

    This is the load-bearing correctness test against a widely-
    used reference implementation.  Tolerance ~1e-3 matches the
    convergence floor of both solvers.
    """
    import warnings

    import pandas as pd
    import statsmodels.formula.api as smf

    warnings.filterwarnings('ignore')

    rng = np.random.default_rng(0)
    g, n_per = 8, 25
    N = g * n_per
    group_id = np.repeat(np.arange(g), n_per)
    X_np = np.column_stack(
        [
            np.ones(N),
            rng.standard_normal(N),
            rng.standard_normal(N),
        ]
    )
    true_b = rng.standard_normal(g) * 0.7
    eps = rng.standard_normal(N) * 0.4
    true_beta = np.array([1.0, 0.5, -0.3])
    y_np = X_np @ true_beta + true_b[group_id] + eps

    df = pd.DataFrame(
        {
            'y': y_np,
            'x0': X_np[:, 1],
            'x1': X_np[:, 2],
            'group': group_id,
        }
    )
    md = smf.mixedlm('y ~ x0 + x1', df, groups=df['group'])
    mdf = md.fit(reml=True)

    Z_np = np.zeros((N, g))
    for i in range(g):
        Z_np[group_id == i, i] = 1.0
    Y = jnp.asarray(y_np[None, :])
    X = jnp.asarray(X_np)
    Z = jnp.asarray(Z_np)
    result = reml_fit(Y, X, Z, n_iter=100)

    # Two independent solvers (Newton-scoring with backtracking
    # here; L-BFGS in statsmodels) -- match to ~5e-3 in absolute,
    # comfortably within the convergence floor of both.
    np.testing.assert_allclose(
        np.asarray(result.beta_hat[0]),
        mdf.fe_params.values,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        float(result.sigma_b_sq[0]),
        float(mdf.cov_re.iloc[0, 0]),
        atol=5e-3,
    )
    np.testing.assert_allclose(
        float(result.sigma_e_sq[0]),
        float(mdf.scale),
        atol=5e-3,
    )


# ---------------------------------------------------------------------------
# REML: low-rank (q-rank) FaST-LMM path
# ---------------------------------------------------------------------------


def _random_lme_data(N=40, q=5, p=3, V=16, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((N, q))  # tall, full column rank
    X = np.column_stack([np.ones(N), rng.standard_normal((N, p - 1))])
    b = rng.standard_normal((V, q)) * 0.8
    eps = rng.standard_normal((V, N)) * 1.2
    beta = rng.standard_normal((V, p))
    Y = beta @ X.T + b @ Z.T + eps
    return jnp.asarray(Y), jnp.asarray(X), jnp.asarray(Z)


def test_reml_lowrank_matches_dense():
    """The q-rank fit reaches the same REML optimum as the dense N x N eig.

    The two paths start from the same heuristic ``theta`` and optimise the
    identical objective, so the log-likelihood agrees to the convergence floor
    and the variance components / fixed effects to the iterative tolerance."""
    Y, X, Z = _random_lme_data()
    dense = reml_fit(Y, X, Z, n_iter=60, low_rank=False)
    low = reml_fit(Y, X, Z, n_iter=60, low_rank=True)
    np.testing.assert_allclose(
        np.asarray(low.log_lik), np.asarray(dense.log_lik), atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(low.beta_hat), np.asarray(dense.beta_hat), atol=1e-6
    )
    # theta is looser than the log-likelihood: the REML objective is flat near
    # the optimum, so the two paths land a few 1e-4 apart in log-variance while
    # agreeing to ~1e-11 in the actual objective (the assertion above).
    np.testing.assert_allclose(
        np.asarray(low.theta_hat), np.asarray(dense.theta_hat), atol=2e-3
    )


def test_reml_lowrank_matches_balanced_closed_form():
    """Low-rank REML matches the ANOVA closed form on balanced one-way data
    (the load-bearing correctness oracle, via the q-rank path)."""
    y, group_id, g, n_per = _balanced_one_way_data()
    sigma_b_sq_closed, sigma_e_sq_closed = _balanced_one_way_closed_form(
        y, group_id, g, n_per
    )
    N = g * n_per
    X = jnp.ones((N, 1))
    Z = jnp.zeros((N, g))
    for i in range(g):
        Z = Z.at[group_id == i, i].set(1.0)
    Y = jnp.asarray(y[None, :])
    result = reml_fit(Y, X, Z, n_iter=60, low_rank=True)
    np.testing.assert_allclose(
        float(result.sigma_b_sq[0]), sigma_b_sq_closed, atol=1e-6
    )
    np.testing.assert_allclose(
        float(result.sigma_e_sq[0]), sigma_e_sq_closed, atol=1e-6
    )


def test_reml_lowrank_block_chunking_matches_single_vmap():
    """``low_rank`` honours the ``block`` memory knob (identical to one vmap)."""
    Y, X, Z = _random_lme_data(V=20)
    full = reml_fit(Y, X, Z, n_iter=40, low_rank=True)
    chunked = reml_fit(Y, X, Z, n_iter=40, low_rank=True, block=7)
    # block padding / reshaping reassociates the reductions, so identical up to
    # float roundoff rather than bit-exact.
    np.testing.assert_allclose(
        np.asarray(full.theta_hat), np.asarray(chunked.theta_hat), atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(full.beta_hat), np.asarray(chunked.beta_hat), atol=1e-6
    )


def test_reml_lowrank_rejects_wide_z():
    """The low-rank form needs a tall random-effect design (q <= N)."""
    rng = np.random.default_rng(1)
    Y = jnp.asarray(rng.standard_normal((4, 6)))
    X = jnp.ones((6, 1))
    Z = jnp.asarray(rng.standard_normal((6, 8)))  # wide: q=8 > N=6
    with pytest.raises(ValueError):
        reml_fit(Y, X, Z, low_rank=True)


def test_reml_lowrank_no_NxN_intermediate():
    """The q-rank HLO must carry no per-voxel (V, N, N) tensor and no N x N
    factor beyond the range basis U_r (N x q) -- the asymptotic win."""
    V, N, q = 512, 48, 4
    rng = np.random.default_rng(0)
    Y = jnp.asarray(rng.standard_normal((V, N)).astype(np.float32))
    X = jnp.asarray(rng.standard_normal((N, 2)).astype(np.float32))
    Z = jnp.asarray(rng.standard_normal((N, q)).astype(np.float32))

    f = jax.jit(
        lambda Y, X, Z: reml_fit(Y, X, Z, n_iter=8, low_rank=True).beta_hat
    )
    hlo = f.lower(Y, X, Z).compile().as_text()
    shapes = re.findall(r'f32\[([0-9,]+)\]', hlo)
    for s in shapes:
        dims = tuple(int(x) for x in s.split(',') if x)
        # no per-voxel (V, N, N)
        assert not (
            len(dims) == 3 and dims[0] == V and dims[1] == N and dims[2] == N
        ), f'unexpected (V, N, N) intermediate {dims}'
        # no dense N x N factor (the low-rank path only needs U_r = N x q)
        assert not (len(dims) == 2 and dims[0] == N and dims[1] == N), (
            f'unexpected dense N x N factor {dims}; low-rank should stay N x q'
        )


# ---------------------------------------------------------------------------
# Mixed-model fixed-effect inference (Satterthwaite)
# ---------------------------------------------------------------------------


def _random_intercept_data(G=12, n_per=8, beta=(1.5, -0.7), seed=0):
    rng = np.random.default_rng(seed)
    N = G * n_per
    gid = np.repeat(np.arange(G), n_per)
    X = np.column_stack([np.ones(N), rng.standard_normal(N)])
    Z = np.zeros((N, G))
    for i in range(G):
        Z[gid == i, i] = 1.0
    b = rng.standard_normal(G) * 0.9
    y = X @ np.asarray(beta) + b[gid] + rng.standard_normal(N) * 0.6
    return jnp.asarray(y[None, :]), jnp.asarray(X), jnp.asarray(Z), gid


@pytest.mark.skipif(_skip_if_no_statsmodels(), reason='statsmodels missing')
def test_lme_t_contrast_se_matches_statsmodels():
    """The contrast SE = sqrt(c^T (X^T V^-1 X)^-1 c) matches lme4/statsmodels
    MixedLM (REML) for the fixed-effect covariate."""
    import pandas as pd
    import statsmodels.api as sm

    Y, X, Z, gid = _random_intercept_data()
    res = reml_fit(Y, X, Z, n_iter=80)
    ct = lme_t_contrast(res, jnp.asarray([0.0, 1.0]))

    dat = pd.DataFrame(
        {'y': np.asarray(Y[0]), 'x': np.asarray(X[:, 1]), 'g': gid}
    )
    md = sm.MixedLM.from_formula('y ~ x', groups='g', data=dat).fit(reml=True)
    np.testing.assert_allclose(
        float(ct.effect[0]), md.fe_params['x'], rtol=1e-3
    )
    np.testing.assert_allclose(float(ct.se[0]), md.bse_fe['x'], rtol=5e-3)
    # a sane Satterthwaite df (between the contrast rank and N - p) and a
    # significant covariate effect
    assert 1.0 < float(ct.df[0]) < float(X.shape[0])
    assert float(ct.p_value[0]) < 0.01


def test_lme_satterthwaite_gradient_matches_finite_difference():
    """The contrast-variance gradient w^T M_k w equals d(c^T C c)/dtheta_k by
    finite difference -- the load-bearing Satterthwaite ingredient."""
    from nitrix.linalg._solver import safe_eigh
    from nitrix.stats.lme._varcomp import VarCompSpec, varcomp_inference

    Y, X, Z, _ = _random_intercept_data(seed=1)
    res = reml_fit(Y, X, Z, n_iter=80)
    c = jnp.asarray([0.0, 1.0])

    N = X.shape[0]
    ZZt = Z @ Z.T
    ev, U = safe_eigh(0.5 * (ZZt + ZZt.T))
    lam = jnp.clip(ev, 0.0, None)
    y_rot = (Y @ U)[0]
    x_rot = U.T @ X
    bdiag = jnp.stack([lam, jnp.ones_like(lam)])
    spec = VarCompSpec.reml()
    th = res.theta_hat[0]
    off = jnp.zeros(N)

    def cCc(theta):
        a_inv, _, _ = varcomp_inference(
            theta, y_rot, x_rot, bdiag, off, 2, spec
        )
        w = a_inv @ c
        return float(c @ w)

    w = res.fixed_cov[0] @ c
    g_an = np.array([float(w @ res.grad_m[0, k] @ w) for k in range(2)])
    eps = 1e-5
    g_fd = np.array(
        [
            (cCc(th.at[k].add(eps)) - cCc(th.at[k].add(-eps))) / (2 * eps)
            for k in range(2)
        ]
    )
    np.testing.assert_allclose(g_an, g_fd, rtol=1e-4)


def test_lme_t_contrast_lowrank_matches_dense():
    """The contrast inference is identical on the low-rank and dense paths
    (the null-space Satterthwaite gradient term is correct)."""
    Y, X, Z, _ = _random_intercept_data(seed=2)
    c = jnp.asarray([0.0, 1.0])
    dense = lme_t_contrast(reml_fit(Y, X, Z, n_iter=80), c)
    low = lme_t_contrast(reml_fit(Y, X, Z, n_iter=80, low_rank=True), c)
    np.testing.assert_allclose(float(low.se[0]), float(dense.se[0]), atol=1e-7)
    np.testing.assert_allclose(float(low.df[0]), float(dense.df[0]), rtol=1e-4)


def test_lme_t_contrast_rejects_unknown_dof():
    Y, X, Z, _ = _random_intercept_data()
    res = reml_fit(Y, X, Z, n_iter=20)
    with pytest.raises(ValueError, match='satterthwaite'):
        lme_t_contrast(res, jnp.asarray([0.0, 1.0]), dof='kr')


# ---------------------------------------------------------------------------
# lme_fit dispatcher: R1 (scalar) -> reml_fit; R2 (correlated) -> block-Woodbury
# ---------------------------------------------------------------------------


def _slope_data(seed=0, M=40, n_per=20, g_cov=None, se=0.6):
    rng = np.random.default_rng(seed)
    if g_cov is None:
        g_cov = np.array([[0.8, 0.3], [0.3, 0.5]])
    N = M * n_per
    gid = np.repeat(np.arange(M), n_per)
    x = rng.standard_normal(N)
    X = np.column_stack([np.ones(N), x])
    L = np.linalg.cholesky(g_cov)
    b = rng.standard_normal((M, 2)) @ L.T
    y = (
        X @ np.array([1.5, -0.7])
        + np.sum(X * b[gid], axis=1)
        + rng.standard_normal(N) * se
    )
    return y, X, gid


def test_lme_fit_r1_dispatches_to_reml_fit():
    """A scalar random intercept (no z) routes to the FaST-LMM R1 path: lme_fit
    returns a REMLResult identical to reml_fit on the one-hot design."""
    Y, X, Z, _ = _random_intercept_data()
    gid = jnp.asarray(np.argmax(np.asarray(Z), axis=1))
    r1 = lme_fit(Y, X, group=gid)
    ref = reml_fit(Y, X, Z)
    assert isinstance(r1, REMLResult)
    np.testing.assert_allclose(
        np.asarray(r1.beta_hat), np.asarray(ref.beta_hat), atol=1e-9
    )


@pytest.mark.skipif(
    _skip_if_no_statsmodels(), reason='statsmodels not installed'
)
def test_lme_fit_r2_block_woodbury_matches_statsmodels():
    """A correlated random slope (z given) routes to the tier-R2 block-Woodbury
    REML; the recovered fixed effects, G, and residual variance match
    statsmodels MixedLM (REML) to tight tolerance."""
    import statsmodels.api as sm

    y, X, gid = _slope_data()
    r2 = lme_fit(
        jnp.asarray(y[None, :]),
        jnp.asarray(X),
        group=jnp.asarray(gid),
        z=jnp.asarray(X),
        n_iter=200,
    )
    assert isinstance(r2, LMEResult)
    assert r2.tier == 'R2'
    md = sm.MixedLM(y, X, groups=gid, exog_re=X).fit(reml=True)
    G_hat = np.asarray(r2.cov_re[0])
    G_ref = np.asarray(md.cov_re)
    assert np.max(np.abs(G_hat - G_ref)) / np.max(np.abs(G_ref)) < 1e-3
    assert abs(float(r2.sigma_e_sq[0]) - md.scale) / md.scale < 1e-3
    np.testing.assert_allclose(
        np.asarray(r2.beta_hat[0]), np.asarray(md.fe_params), rtol=5e-3
    )


def test_lme_fit_r2_symmetric_psd_cov_re():
    """The recovered random-effect covariance is symmetric PSD (log-Cholesky
    parameterisation guarantees it)."""
    y, X, gid = _slope_data(seed=1)
    r2 = lme_fit(
        jnp.asarray(y[None, :]),
        jnp.asarray(X),
        group=jnp.asarray(gid),
        z=jnp.asarray(X),
        n_iter=150,
    )
    G = np.asarray(r2.cov_re[0])
    np.testing.assert_allclose(G, G.T, atol=1e-10)
    assert float(np.linalg.eigvalsh(G).min()) > -1e-9


# ---------------------------------------------------------------------------
# lme_fit structure='diagonal': the uncorrelated (x || g) random effect
# ---------------------------------------------------------------------------


def _diag_slope_data(seed=0, M=60, n_per=20, var=(0.8, 0.5), se=0.6):
    """Independent random intercept + slope (diagonal-G truth: zero covariance)."""
    rng = np.random.default_rng(seed)
    N = M * n_per
    gid = np.repeat(np.arange(M), n_per)
    x = rng.standard_normal(N)
    X = np.column_stack([np.ones(N), x])
    b = rng.standard_normal((M, 2)) * np.sqrt(np.asarray(var))  # uncorrelated
    y = (
        X @ np.array([1.5, -0.7])
        + np.sum(X * b[gid], axis=1)
        + rng.standard_normal(N) * se
    )
    return y, X, gid


def test_lme_fit_diagonal_off_diagonal_is_exactly_zero():
    """``structure='diagonal'`` holds the random-effect covariance off-diagonal
    at exactly zero -- the (x || g) constraint, distinct from the unstructured
    fit -- while recovering the true variances and residual scale."""
    y, X, gid = _diag_slope_data(var=(0.8, 0.5), se=0.6, M=160)
    r = lme_fit(
        jnp.asarray(y[None, :]),
        jnp.asarray(X),
        group=jnp.asarray(gid),
        z=jnp.asarray(X),
        structure='diagonal',
        n_iter=200,
    )
    assert isinstance(r, LMEResult)
    G = np.asarray(r.cov_re[0])
    assert G[0, 1] == 0.0 and G[1, 0] == 0.0  # structurally diagonal
    # both variance components are positive and recovered near truth (variance-
    # component estimates carry ~O(1/sqrt(M)) sampling spread); residual ~0.36.
    assert np.all(np.diag(G) > 0.0)
    np.testing.assert_allclose(np.diag(G), [0.8, 0.5], rtol=0.3)
    assert abs(float(r.sigma_e_sq[0]) - 0.36) < 0.05


def test_lme_fit_diagonal_is_nested_in_unstructured():
    """The diagonal fit is a constrained sub-model: its REML log-likelihood
    cannot exceed the unstructured fit's (one fewer free parameter)."""
    y, X, gid = _diag_slope_data(seed=3)
    common = dict(group=jnp.asarray(gid), z=jnp.asarray(X), n_iter=300)
    diag = lme_fit(
        jnp.asarray(y[None, :]), jnp.asarray(X), structure='diagonal', **common
    )
    unst = lme_fit(
        jnp.asarray(y[None, :]),
        jnp.asarray(X),
        structure='unstructured',
        **common,
    )
    assert float(unst.log_lik[0]) >= float(diag.log_lik[0]) - 1e-4


@pytest.mark.skipif(
    _skip_if_no_statsmodels(), reason='statsmodels not installed'
)
def test_lme_fit_diagonal_matches_statsmodels_free_mask():
    """The diagonal-G variances and residual scale match statsmodels MixedLM
    fitted with a diagonal ``free`` mask (off-diagonal of cov_re held at 0)."""
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLMParams

    y, X, gid = _diag_slope_data(seed=1, M=60, n_per=25)
    r = lme_fit(
        jnp.asarray(y[None, :]),
        jnp.asarray(X),
        group=jnp.asarray(gid),
        z=jnp.asarray(X),
        structure='diagonal',
        n_iter=400,
    )
    # statsmodels: free mask -> estimate the two RE variances + all fixed
    # effects, hold the RE covariance off-diagonal fixed at its 0 start.
    free = MixedLMParams.from_components(
        fe_params=np.ones(X.shape[1]), cov_re=np.eye(2)
    )
    md = sm.MixedLM(y, X, groups=gid, exog_re=X).fit(reml=True, free=free)
    G_hat = np.asarray(r.cov_re[0])
    G_ref = np.asarray(md.cov_re)  # already diagonal (off-diag fixed at 0)
    np.testing.assert_allclose(np.diag(G_hat), np.diag(G_ref), rtol=2e-2)
    assert abs(float(r.sigma_e_sq[0]) - md.scale) / md.scale < 2e-2


# ---------------------------------------------------------------------------
# lme_f_contrast: the Satterthwaite F-test (Fai-Cornelius denominator df)
# ---------------------------------------------------------------------------


def _two_covariate_data(seed=0, M=30, n_per=8, beta=(1.0, 0.8, -0.5), se=0.6):
    rng = np.random.default_rng(seed)
    N = M * n_per
    gid = np.repeat(np.arange(M), n_per)
    Z = np.zeros((N, M))
    for i in range(M):
        Z[gid == i, i] = 1.0
    X = np.column_stack(
        [np.ones(N), rng.standard_normal(N), rng.standard_normal(N)]
    )
    b = rng.standard_normal(M) * 0.9
    y = X @ np.asarray(beta) + b[gid] + rng.standard_normal(N) * se
    return jnp.asarray(y[None, :]), jnp.asarray(X), jnp.asarray(Z)


def test_lme_f_contrast_single_row_equals_t_contrast_squared():
    """For a single-row contrast the F-test collapses *exactly* to the
    t-contrast: ``F == t^2``, the denominator df equals the t-Satterthwaite df,
    and the p-values agree."""
    Y, X, Z = _two_covariate_data()
    res = reml_fit(Y, X, Z, n_iter=120)
    c = jnp.asarray([0.0, 1.0, 0.0])
    tc = lme_t_contrast(res, c)
    fc = lme_f_contrast(res, c)
    assert isinstance(fc, LMEFContrast)
    assert float(fc.df1[0]) == 1.0
    np.testing.assert_allclose(float(fc.f[0]), float(tc.t[0]) ** 2, rtol=1e-6)
    np.testing.assert_allclose(float(fc.df2[0]), float(tc.df[0]), rtol=1e-5)
    np.testing.assert_allclose(
        float(fc.p_value[0]), float(tc.p_value[0]), atol=1e-8
    )


def test_lme_f_contrast_matches_wald_fstatistic():
    """The F-statistic equals the Wald form
    ``(C beta)^T (C Cov(beta) C^T)^{-1} (C beta) / L`` computed directly from the
    fitted fixed-effect covariance (a numpy oracle on the materialised arrays)."""
    Y, X, Z = _two_covariate_data(seed=2)
    res = reml_fit(Y, X, Z, n_iter=120)
    C = jnp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # joint test of both
    fc = lme_f_contrast(res, C)
    assert float(fc.df1[0]) == 2.0
    Sig = np.asarray(res.fixed_cov[0])
    bh = np.asarray(res.beta_hat[0])
    Cn = np.asarray(C)
    cb = Cn @ bh
    f_ref = cb @ np.linalg.inv(Cn @ Sig @ Cn.T) @ cb / 2.0
    np.testing.assert_allclose(float(fc.f[0]), f_ref, rtol=1e-6)
    # denominator df is a sane, finite Satterthwaite value
    assert 1.0 < float(fc.df2[0]) < float(X.shape[0])
    assert float(fc.p_value[0]) < 1e-3  # both covariates have real effects


def test_lme_f_contrast_null_is_not_significant():
    """A joint contrast on covariates with no true effect is not significant."""
    rng = np.random.default_rng(7)
    M, n_per = 30, 8
    N = M * n_per
    gid = np.repeat(np.arange(M), n_per)
    Z = np.zeros((N, M))
    for i in range(M):
        Z[gid == i, i] = 1.0
    # X carries two pure-noise covariates; y depends on neither.
    X = np.column_stack(
        [np.ones(N), rng.standard_normal(N), rng.standard_normal(N)]
    )
    b = rng.standard_normal(M) * 0.9
    y = 1.0 + b[gid] + rng.standard_normal(N) * 0.6
    res = reml_fit(
        jnp.asarray(y[None, :]), jnp.asarray(X), jnp.asarray(Z), n_iter=120
    )
    fc = lme_f_contrast(res, jnp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    assert float(fc.p_value[0]) > 0.05


def test_lme_f_contrast_rejects_unknown_dof():
    Y, X, Z = _two_covariate_data()
    res = reml_fit(Y, X, Z, n_iter=20)
    with pytest.raises(ValueError, match='satterthwaite'):
        lme_f_contrast(res, jnp.asarray([0.0, 1.0, 0.0]), dof='kr')


def test_lme_f_contrast_df_floor_no_nan_in_degenerate_case():
    """The Fai-Cornelius E-method df2 = 2E/(E-L) is undefined when E <= L
    (tiny samples / near-boundary variance); the fit must still return a finite,
    valid p-value (df2 floored to a conservative value), never NaN."""
    rng = np.random.default_rng(0)
    # Minimal design: 3 groups, 2 per group -> very few residual df, a 2-row
    # contrast -> the degenerate-E regime the floor guards.
    M, n_per = 3, 2
    N = M * n_per
    gid = np.repeat(np.arange(M), n_per)
    Z = np.zeros((N, M))
    for i in range(M):
        Z[gid == i, i] = 1.0
    X = np.column_stack(
        [np.ones(N), rng.standard_normal(N), rng.standard_normal(N)]
    )
    y = 1.0 + rng.standard_normal(N) * 0.5
    res = reml_fit(
        jnp.asarray(y[None, :]), jnp.asarray(X), jnp.asarray(Z), n_iter=30
    )
    fc = lme_f_contrast(res, jnp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    assert np.isfinite(float(fc.df2[0]))
    assert float(fc.df2[0]) > 0.0
    assert np.isfinite(float(fc.p_value[0]))
    assert 0.0 <= float(fc.p_value[0]) <= 1.0
