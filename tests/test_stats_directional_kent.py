# -*- coding: utf-8 -*-
"""Tests for the Kent (Fisher--Bingham FB5) distribution and the unnormalised
energy path in ``nitrix.stats.directional``.

Kent is the elliptical vMF on :math:`S^2`: density :math:`\\propto \\exp(\\kappa
\\gamma_1^\\top x + \\beta[(\\gamma_2^\\top x)^2 - (\\gamma_3^\\top x)^2])`.
Coverage: the series normaliser vs mpmath (reusing ``log_iv``), reduction to vMF
at :math:`\\beta = 0`, the surface-measure density, moment-estimator recovery,
and the ``normalize=False`` energy (the Gibbs/MRF potential).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.stats import (
    kent_fit,
    kent_log_prob,
    log_kent_normaliser,
    vmf_log_prob,
    watson_log_prob,
)

mpmath = pytest.importorskip('mpmath')
mpmath.mp.dps = 40

_E1 = jnp.asarray([0.0, 0.0, 1.0])
_E2 = jnp.asarray([1.0, 0.0, 0.0])
_E3 = jnp.asarray([0.0, 1.0, 0.0])


def _sphere_grid(n=600):
    th = np.linspace(0, np.pi, n)
    ph = np.linspace(0, 2 * np.pi, n)
    t, p = np.meshgrid(th, ph, indexing='ij')
    x = np.stack(
        [np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], axis=-1
    )
    d_area = np.sin(t) * (th[1] - th[0]) * (ph[1] - ph[0])
    return jnp.asarray(x), d_area


def _ref_log_c(kappa, beta):
    logterms = []
    for j in range(120):
        lb = (
            2 * j * mpmath.log(beta)
            if beta > 0
            else (mpmath.mpf('-inf') if j > 0 else mpmath.mpf(0))
        )
        logterms.append(
            mpmath.log(mpmath.gamma(j + mpmath.mpf(1) / 2))
            - mpmath.log(mpmath.gamma(j + 1))
            + lb
            + (2 * j + mpmath.mpf(1) / 2) * mpmath.log(2 / kappa)
            + mpmath.log(mpmath.besseli(2 * j + mpmath.mpf(1) / 2, kappa))
        )
    m = max(logterms)
    s = sum(mpmath.e ** (t - m) for t in logterms)
    return float(mpmath.log(2 * mpmath.pi) + m + mpmath.log(s))


def _kent_sample(kappa, beta, n, seed):
    """Test-only rejection sampler from the Kent density (uniform proposal)."""
    rng = np.random.default_rng(seed)
    g1, g2, g3 = np.eye(3)[[2, 0, 1]]
    out = []
    while len(out) < n:
        v = rng.standard_normal((8 * n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        energy = kappa * (v @ g1) + beta * ((v @ g2) ** 2 - (v @ g3) ** 2)
        accept = rng.uniform(size=len(v)) < np.exp(energy - kappa)
        out.extend(v[accept].tolist())
    return jnp.asarray(np.array(out[:n]))


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('kappa', [1.0, 5.0, 50.0, 200.0])
@pytest.mark.parametrize('frac', [0.0, 0.3, 0.49])
def test_kent_normaliser_matches_mpmath(kappa, frac):
    beta = frac * kappa / 2.0
    got = float(log_kent_normaliser(jnp.asarray(kappa), jnp.asarray(beta)))
    assert abs(got - _ref_log_c(kappa, beta)) < 1e-10


def test_kent_reduces_to_vmf_at_beta_zero():
    """At beta = 0 the Kent density equals the vMF density (and c = 1/C_3)."""
    x = jax.random.normal(jax.random.PRNGKey(0), (60, 3))
    x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
    for kappa in (2.0, 10.0):
        lk = kent_log_prob(
            x, _E1, _E2, _E3, jnp.asarray(kappa), jnp.asarray(0.0)
        )
        lv = vmf_log_prob(x, _E1, jnp.asarray(kappa))
        np.testing.assert_allclose(lk, lv, atol=1e-12)


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('kappa,beta', [(8.0, 3.0), (20.0, 8.0), (5.0, 0.0)])
def test_kent_density_integrates_to_one(kappa, beta):
    x, d_area = _sphere_grid()
    dens = np.exp(
        np.asarray(
            kent_log_prob(
                x, _E1, _E2, _E3, jnp.asarray(kappa), jnp.asarray(beta)
            )
        )
    )
    assert abs(float((dens * d_area).sum()) - 1.0) < 5e-3


def test_kent_elliptical_contours():
    """beta > 0 elongates along gamma2 vs gamma3: equal-colatitude points on the
    major axis are denser than on the minor axis."""
    on_major = jnp.asarray([np.sin(0.3), 0.0, np.cos(0.3)])  # tilted toward g2
    on_minor = jnp.asarray([0.0, np.sin(0.3), np.cos(0.3)])  # toward g3
    kappa, beta = jnp.asarray(10.0), jnp.asarray(4.0)
    assert float(kent_log_prob(on_major, _E1, _E2, _E3, kappa, beta)) > float(
        kent_log_prob(on_minor, _E1, _E2, _E3, kappa, beta)
    )


# ---------------------------------------------------------------------------
# Moment-estimator fit
# ---------------------------------------------------------------------------


def test_kent_fit_recovers_frame_and_kappa():
    """Frame recovered (essentially) exactly; kappa within a few percent."""
    data = _kent_sample(100.0, 20.0, 8000, 1)
    fit = kent_fit(data)
    assert abs(float(fit.gamma1[2])) > 0.99  # mean direction ~ z
    assert abs(float(fit.gamma2[0])) > 0.98  # major axis ~ x
    assert abs(float(fit.kappa) - 100.0) / 100.0 < 0.05
    assert float(fit.beta) > 0.0


def test_kent_fit_beta_accurate_at_high_kappa_moderate_eccentricity():
    """Where the moment estimator is valid (high kappa, 2 beta / kappa < 0.5),
    beta is recovered within ~10%."""
    data = _kent_sample(200.0, 40.0, 12000, 2)
    fit = kent_fit(data)
    assert abs(float(fit.kappa) - 200.0) / 200.0 < 0.05
    assert abs(float(fit.beta) - 40.0) / 40.0 < 0.1


def test_kent_fit_vmf_limit_recovers_zero_beta():
    """Isotropic (vMF) data gives beta ~ 0 and kappa ~ 1/(1 - Rbar)."""
    from nitrix.stats import vmf_sample

    data = vmf_sample(jax.random.PRNGKey(3), _E1, jnp.asarray(30.0), (6000,))
    fit = kent_fit(data)
    assert abs(float(fit.beta)) < 3.0  # small relative to kappa ~ 30
    assert abs(float(fit.kappa) - 30.0) / 30.0 < 0.15


# ---------------------------------------------------------------------------
# Unnormalised energy (the Gibbs / MRF potential)
# ---------------------------------------------------------------------------


def test_kent_energy_omits_normaliser():
    x = jnp.asarray([0.5, 0.3, np.sqrt(1 - 0.25 - 0.09)])
    kappa, beta = jnp.asarray(8.0), jnp.asarray(3.0)
    energy = kent_log_prob(x, _E1, _E2, _E3, kappa, beta, normalize=False)
    full = kent_log_prob(x, _E1, _E2, _E3, kappa, beta)
    expected = kappa * x[2] + beta * (x[0] ** 2 - x[1] ** 2)
    np.testing.assert_allclose(energy, expected, atol=1e-12)
    # full = energy - log c(kappa, beta)
    np.testing.assert_allclose(
        full, energy - log_kent_normaliser(kappa, beta), atol=1e-12
    )


def test_energy_paths_skip_normaliser_all_families():
    """normalize=False returns the bare exponent for vMF / Watson / Kent."""
    x = jnp.asarray([0.6, 0.0, 0.8])
    k = jnp.asarray(4.0)
    np.testing.assert_allclose(
        vmf_log_prob(x, _E1, k, normalize=False), k * x[2], atol=1e-12
    )
    np.testing.assert_allclose(
        watson_log_prob(x, _E1, k, normalize=False), k * x[2] ** 2, atol=1e-12
    )
    np.testing.assert_allclose(
        kent_log_prob(x, _E1, _E2, _E3, k, jnp.asarray(0.0), normalize=False),
        k * x[2],
        atol=1e-12,
    )


def test_kent_grad_and_jit():
    x, _ = _sphere_grid(50)
    lp = jax.jit(
        lambda xx: kent_log_prob(
            xx, _E1, _E2, _E3, jnp.asarray(6.0), jnp.asarray(2.0)
        )
    )(x)
    assert bool(jnp.all(jnp.isfinite(lp)))
    g = jax.grad(
        lambda b: kent_log_prob(x[0, 0], _E1, _E2, _E3, jnp.asarray(6.0), b)
    )(jnp.asarray(2.0))
    assert bool(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# Fisher--Bingham energy (the general quadratic-exponential family, any p)
# ---------------------------------------------------------------------------


def _orthonormal_frame(p, seed):
    a = np.random.default_rng(seed).standard_normal((p, p))
    q, _ = np.linalg.qr(a)
    return jnp.asarray(q)


def _unit(n, p, seed):
    x = np.random.default_rng(seed).standard_normal((n, p))
    return jnp.asarray(x / np.linalg.norm(x, axis=1, keepdims=True))


def test_fb_energy_matches_s2_kent_energy():
    """At p=3 with beta=(0, b, -b) it equals kent_log_prob(normalize=False)."""
    from nitrix.stats import fisher_bingham_energy

    frame = _orthonormal_frame(3, 0)
    g1, g2, g3 = frame[:, 0], frame[:, 1], frame[:, 2]
    x = _unit(20, 3, 1)
    b = jnp.asarray(3.0)
    fb = fisher_bingham_energy(
        x, frame, jnp.asarray(8.0), jnp.asarray([0.0, 3.0, -3.0])
    )
    s2 = kent_log_prob(x, g1, g2, g3, jnp.asarray(8.0), b, normalize=False)
    np.testing.assert_allclose(fb, s2, atol=1e-12)


def test_fb_energy_subsumes_vmf():
    """beta = 0 gives the vMF energy kappa * (mu^T x)."""
    from nitrix.stats import fisher_bingham_energy

    frame = _orthonormal_frame(4, 2)
    x = _unit(30, 4, 3)
    e = fisher_bingham_energy(x, frame, jnp.asarray(5.0), jnp.zeros(4))
    np.testing.assert_allclose(e, 5.0 * (x @ frame[:, 0]), atol=1e-12)


def test_fb_energy_subsumes_bingham():
    """kappa = 0 gives the Bingham energy x^T A x, A = frame diag(beta) frame^T."""
    from nitrix.stats import fisher_bingham_energy

    frame = _orthonormal_frame(5, 4)
    beta = jnp.asarray([0.5, -0.2, -0.3, 0.1, -0.1])
    a = np.asarray(frame) @ np.diag(np.asarray(beta)) @ np.asarray(frame).T
    x = _unit(24, 5, 5)
    e = fisher_bingham_energy(x, frame, jnp.asarray(0.0), beta)
    xtax = jnp.einsum('ni,ij,nj->n', x, jnp.asarray(a), x)
    np.testing.assert_allclose(e, xtax, atol=1e-12)


def test_fb_energy_subsumes_watson():
    """kappa = 0 with a single non-zero beta_j is the Watson energy."""
    from nitrix.stats import fisher_bingham_energy

    frame = _orthonormal_frame(6, 6)
    axis = frame[:, 2]  # the Watson axis
    beta = jnp.zeros(6).at[2].set(4.0)
    x = _unit(20, 6, 7)
    e = fisher_bingham_energy(x, frame, jnp.asarray(0.0), beta)
    np.testing.assert_allclose(
        e,
        watson_log_prob(x, axis, jnp.asarray(4.0), normalize=False),
        atol=1e-12,
    )


def test_fb_energy_high_dim_jit_vmap_grad():
    from nitrix.stats import fisher_bingham_energy

    p = 8
    frame = _orthonormal_frame(p, 8)
    x = _unit(32, p, 9)
    kappa = jnp.asarray(6.0)
    beta = jnp.asarray([0.0, 2.0, 1.0, 0.5, -0.5, -1.0, -1.0, -1.0])
    e = fisher_bingham_energy(x, frame, kappa, beta)
    assert e.shape == (32,)
    ej = jax.jit(lambda xx: fisher_bingham_energy(xx, frame, kappa, beta))(x)
    np.testing.assert_allclose(e, ej, atol=1e-12)
    ev = jax.vmap(lambda xi: fisher_bingham_energy(xi, frame, kappa, beta))(x)
    np.testing.assert_allclose(e, ev, atol=1e-12)
    g = jax.grad(lambda b: fisher_bingham_energy(x[0], frame, kappa, b).sum())(
        beta
    )
    assert bool(jnp.all(jnp.isfinite(g)))
