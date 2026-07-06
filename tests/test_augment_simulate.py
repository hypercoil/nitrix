# -*- coding: utf-8 -*-
"""Tests for ``nitrix.augment.simulate`` (synthetic connectivity / time series).

The correctness mandate for these generators is that the data **provably has
the structure it claims**: each test recovers the planted ground truth (spectral
support, covariance, mixing, community layout, transition matrix) from the
generator's output, rather than asserting parity with any legacy output.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.augment import (
    band_limited_signals,
    color_signals,
    lowrank_block_connectome,
    markov_state_sequence,
    mix_signals,
    sparse_mixture_matrix,
)
from nitrix.stats import whiten


def _key(i=0):
    return jax.random.PRNGKey(i)


# ---------------------------------------------------------------------------
# Band-limited sources: spectral support is the planted structure
# ---------------------------------------------------------------------------


def test_band_limited_spectral_support():
    """Power is concentrated inside the passband and (near) zero outside it,
    with no brick-wall ringing."""
    n_t = 4096
    s = band_limited_signals(
        _key(), 16, n_t, low=0.0, high=0.1, transition=0.03
    )
    power = jnp.abs(jnp.fft.rfft(s, axis=-1)) ** 2
    f = jnp.fft.rfftfreq(n_t) * 2.0
    in_band = power[:, f <= 0.13].sum()
    out_band = power[:, f > 0.16].sum()
    assert float(out_band / in_band) < 1e-6


def test_band_limited_bandpass_excludes_dc():
    """A band with ``low > 0`` suppresses the low-frequency / DC content."""
    n_t = 4096
    s = band_limited_signals(
        _key(1), 8, n_t, low=0.2, high=0.4, transition=0.03
    )
    power = jnp.abs(jnp.fft.rfft(s, axis=-1)) ** 2
    f = jnp.fft.rfftfreq(n_t) * 2.0
    below = power[:, f < 0.15].sum()
    inside = power[:, (f >= 0.2) & (f <= 0.4)].sum()
    assert float(below / inside) < 1e-6


def test_band_limited_unit_variance_and_determinism():
    s = band_limited_signals(_key(2), 5, 1024)
    np.testing.assert_allclose(jnp.std(s, axis=-1), jnp.ones(5), atol=1e-6)
    s2 = band_limited_signals(_key(2), 5, 1024)
    assert bool(jnp.array_equal(s, s2))


# ---------------------------------------------------------------------------
# Colouring: the whitening inverse -- plants an exact covariance
# ---------------------------------------------------------------------------


def _spd(d, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((d, d))
    return jnp.asarray(a @ a.T + np.eye(d))


def test_color_recovers_planted_covariance():
    """Empirical covariance of the coloured signals recovers ``target_cov``."""
    sigma = _spd(6, 0)
    y = color_signals(_key(3), sigma, 40000)
    rel = jnp.max(jnp.abs(jnp.cov(y) - sigma)) / jnp.max(jnp.abs(sigma))
    assert float(rel) < 0.05


def test_color_is_whitening_inverse():
    """Colouring to ``Sigma`` then ZCA-whitening returns unit covariance --
    colouring (``Sigma^{1/2}``) and whitening (``Sigma^{-1/2}``) are inverses."""
    sigma = _spd(5, 1)
    y = color_signals(_key(4), sigma, 30000)  # (d, T): features x obs
    whitened = whiten(y.T)  # (obs, features)
    np.testing.assert_allclose(
        jnp.cov(whitened, rowvar=False), jnp.eye(5), atol=1e-10
    )


def test_color_differentiable_and_jit():
    sigma = _spd(5, 2)
    g = jax.grad(lambda m: color_signals(_key(5), m, 200).sum())(sigma)
    assert bool(jnp.all(jnp.isfinite(g)))
    yj = jax.jit(lambda m: color_signals(_key(5), m, 200))(sigma)
    assert bool(jnp.all(jnp.isfinite(yj)))


# ---------------------------------------------------------------------------
# Sparse mixing + the mixture forward model
# ---------------------------------------------------------------------------


def test_mixture_rows_l1_normalised():
    m = sparse_mixture_matrix(_key(6), 200, 20, expected_nnz=5.0)
    row_sums = m.sum(-1)
    # Every (non-empty) row sums to 1; with p = 5/20 no row is empty here.
    np.testing.assert_allclose(row_sums, jnp.ones(200), atol=1e-12)
    assert bool(jnp.all(m >= 0))


def test_mixture_expected_cardinality():
    m = sparse_mixture_matrix(_key(7), 400, 40, expected_nnz=8.0)
    mean_nnz = jnp.mean((m > 0).sum(-1).astype(jnp.float64))
    assert abs(float(mean_nnz) - 8.0) < 1.0


def test_mixture_forward_model_covariance():
    """``cov(mix_signals(M, sources)) == M cov(sources) M^T``."""
    m = sparse_mixture_matrix(_key(8), 12, 6, expected_nnz=3.0)
    src = band_limited_signals(_key(9), 6, 30000, low=0.0, high=0.25)
    obs = mix_signals(m, src)
    np.testing.assert_allclose(
        jnp.cov(obs), m @ jnp.cov(src) @ m.T, atol=1e-10
    )


def test_mix_signals_local_component_added():
    m = sparse_mixture_matrix(_key(10), 8, 4, expected_nnz=2.0)
    src = jax.random.normal(_key(11), (4, 100))
    local = jax.random.normal(_key(12), (8, 100))
    np.testing.assert_allclose(
        mix_signals(m, src, local=local), m @ src + local, atol=1e-12
    )


# ---------------------------------------------------------------------------
# Low-rank-block connectome
# ---------------------------------------------------------------------------


def test_lowrank_block_symmetric_and_community_structure():
    comm = jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    c = lowrank_block_connectome(
        _key(13), comm, 3, rank=6, within_scale=0.1, noise_scale=0.1
    )
    np.testing.assert_allclose(c, c.T, atol=0.0)
    same = comm[:, None] == comm[None, :]
    off = ~jnp.eye(comm.shape[0], dtype=bool)
    within = c[same & off].mean()
    between = c[~same].mean()
    assert float(within) > float(between)


def test_lowrank_block_communities_recoverable():
    """The planted partition is recoverable: each node's strongest off-diagonal
    connections are dominated by its own community."""
    comm = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    c = lowrank_block_connectome(
        _key(14), comm, 3, rank=8, within_scale=0.05, noise_scale=0.05
    )
    n = comm.shape[0]
    c_off = c - jnp.diag(jnp.diag(c))
    # For each node, mean connection to its community exceeds mean to any other.
    for i in range(n):
        my = comm[i]
        by_comm = jnp.array(
            [
                c_off[i, (comm == g) & (jnp.arange(n) != i)].mean()
                for g in range(3)
            ]
        )
        assert int(jnp.argmax(by_comm)) == int(my)


def test_lowrank_block_parameterised_layout():
    """Block layout is not hard-coded: varying communities / rank works."""
    comm = jax.random.randint(_key(15), (30,), 0, 5)
    c = lowrank_block_connectome(_key(16), comm, 5, rank=3, noise_rank=2)
    assert c.shape == (30, 30)
    np.testing.assert_allclose(c, c.T, atol=0.0)


# ---------------------------------------------------------------------------
# Markov state sequences
# ---------------------------------------------------------------------------


def _transition(seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.1, 1.0, (4, 4))
    return jnp.asarray(t / t.sum(1, keepdims=True))


def test_markov_recovers_transition_matrix():
    t = _transition(0)
    seq = markov_state_sequence(_key(17), t, 200000)
    emp = jnp.stack(
        [
            jnp.array(
                [((seq[1:][seq[:-1] == i]) == j).mean() for j in range(4)]
            )
            for i in range(4)
        ]
    )
    assert float(jnp.max(jnp.abs(emp - t))) < 0.02


def test_markov_recovers_stationary_distribution():
    """Long-run state occupancy matches the transition's stationary law."""
    t = _transition(1)
    # Stationary distribution via the dominant left eigenvector (power iterate).
    pi = jnp.ones(4) / 4
    for _ in range(200):
        pi = pi @ t
    seq = markov_state_sequence(_key(18), t, 200000)
    occ = jnp.array([(seq == j).mean() for j in range(4)])
    assert float(jnp.max(jnp.abs(occ - pi))) < 0.02


def test_markov_deterministic_jit_and_dtype():
    t = _transition(2)
    seq = markov_state_sequence(_key(19), t, 500)
    assert seq.dtype == jnp.int32
    seq2 = jax.jit(lambda k: markov_state_sequence(k, t, 500))(_key(19))
    assert bool(jnp.array_equal(seq, seq2))
    assert bool(jnp.all((seq >= 0) & (seq < 4)))
