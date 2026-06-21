# -*- coding: utf-8 -*-
"""Tests for the nitrix.stats effect-size / interval helpers (audit N5).

``confidence_interval`` and ``standardized_effect`` are thin, fitter-agnostic
readouts on a contrast's ``(effect, se, dof)``.  The Student-t quantile behind
the CI is computed per element (Newton on ``betainc``) and pinned against
``scipy.stats.t.ppf``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from scipy.stats import t as scit

from nitrix.stats import (
    GAUSSIAN,
    confidence_interval,
    glm_fit,
    standardized_effect,
    t_contrast,
)


@pytest.mark.parametrize('level', [0.90, 0.95, 0.99])
@pytest.mark.parametrize('dof', [3.0, 5.0, 10.0, 30.0, 100.0])
def test_confidence_interval_matches_t_quantile(level, dof):
    effect, se = 0.7, 0.15
    lo, hi = confidence_interval(
        jnp.asarray(effect), jnp.asarray(se), jnp.asarray(dof), level=level
    )
    tc = scit.ppf(0.5 * (1.0 + level), dof)
    np.testing.assert_allclose(float(lo), effect - tc * se, rtol=1e-6)
    np.testing.assert_allclose(float(hi), effect + tc * se, rtol=1e-6)


def test_confidence_interval_per_element_dof():
    """``dof`` may vary per element (e.g. Satterthwaite) -- the t-quantile is
    evaluated per element, not once."""
    eff = jnp.array([0.5, -0.3, 1.0])
    se = jnp.array([0.1, 0.2, 0.05])
    dof = jnp.array([4.0, 25.0, 80.0])
    lo, hi = confidence_interval(eff, se, dof)
    tc = scit.ppf(0.975, np.asarray(dof))
    np.testing.assert_allclose(
        np.asarray(lo), np.asarray(eff) - tc * np.asarray(se), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(hi), np.asarray(eff) + tc * np.asarray(se), rtol=1e-6
    )


def test_confidence_interval_integrates_with_t_contrast():
    """A CI from a GLM contrast brackets the effect, and widens with level."""
    rng = np.random.default_rng(0)
    n = 200
    X = np.c_[np.ones(n), rng.standard_normal(n)]
    y = X @ np.array([1.0, 0.5]) + rng.standard_normal(n) * 0.5
    res = glm_fit(jnp.asarray(y[None, :]), jnp.asarray(X), family=GAUSSIAN)
    effect, se, _t, _p = t_contrast(res, jnp.asarray([0.0, 1.0]))
    lo95, hi95 = confidence_interval(effect, se, res.dof_resid, level=0.95)
    lo99, hi99 = confidence_interval(effect, se, res.dof_resid, level=0.99)
    assert float(lo95[0]) < float(effect[0]) < float(hi95[0])
    width95 = float(hi95[0]) - float(lo95[0])
    width99 = float(hi99[0]) - float(lo99[0])
    assert width99 > width95  # higher coverage -> wider interval


def test_standardized_effect():
    eff = jnp.array([0.5, -0.4])
    sd = jnp.array([0.25, 0.8])
    np.testing.assert_allclose(
        np.asarray(standardized_effect(eff, sd)), [2.0, -0.5]
    )


def test_confidence_interval_rejects_bad_level():
    with pytest.raises(ValueError, match='level'):
        confidence_interval(
            jnp.asarray(0.5), jnp.asarray(0.1), jnp.asarray(10.0), level=1.5
        )
