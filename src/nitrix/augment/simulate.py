# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Synthetic connectivity & time-series generators with known ground truth.

Keyed forward models that synthesise fMRI-like signals and connectomes whose
structure is *planted* -- a recoverable latent covariance, community layout, or
transition matrix -- so the whole ecosystem (and nitrix's own suites) can be
tested and benchmarked against a known answer.  Each is a pure function of an
explicit :class:`jax.Array` PRNG key, statically shaped, and jit/vmap-clean.

Two complementary ways plant a known cross-signal covariance :math:`\Sigma`:

- :func:`color_signals` -- **colouring**: impose an *exact* target covariance
  by mapping white noise through :math:`\Sigma^{1/2}`.  This is the direct
  inverse of ZCA whitening (:func:`nitrix.stats.whiten`, which applies
  :math:`\Sigma^{-1/2}`), and reuses the same cuSOLVER-free Newton-Schulz
  square root -- the reason this family sits beside the whitening work.
- :func:`sparse_mixture_matrix` + :func:`mix_signals` -- **mixing**: the
  ICA-style forward model ``mixture @ sources``, whose population covariance is
  ``mixture @ cov(sources) @ mixture.T`` (``= M Mᵀ`` for standardised
  independent sources) and which additionally plants a recoverable sparse
  *mixing* structure, not only a covariance.

The temporal structure of the latent sources comes from:

- :func:`band_limited_signals` -- latent sources as coloured (band-limited)
  noise, shaped in frequency by a *smooth* spectral window (not a brick-wall
  bin mask).

and two connectome/state generators:

- :func:`lowrank_block_connectome` -- a symmetric ``tanh(L Lᵀ) + N Nᵀ``
  connectivity matrix with planted community block structure plus low-rank
  noise.
- :func:`markov_state_sequence` -- a keyed discrete-state trajectory from a
  transition matrix (dynamic-FC / state-switching synthesis), via
  :func:`jax.lax.scan`.

These are leaf generators only; augmentation *policy* (compose / registries /
multi-crop) belongs to the consumer layers.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int

from ..linalg import symsqrt
from ._common import _default_float

__all__ = [
    'band_limited_signals',
    'color_signals',
    'sparse_mixture_matrix',
    'mix_signals',
    'lowrank_block_connectome',
    'markov_state_sequence',
]


def _cosine_ramp(t: Float[Array, '...']) -> Float[Array, '...']:
    """Raised-cosine s-curve: ``0`` for ``t <= 0``, ``1`` for ``t >= 1``.

    Smooth (``C^1``) monotone ramp used to taper spectral-window edges.
    """
    return 0.5 * (1.0 - jnp.cos(jnp.pi * jnp.clip(t, 0.0, 1.0)))


def band_limited_signals(
    key: Array,
    n_signals: int,
    n_timepoints: int,
    *,
    low: float = 0.0,
    high: float = 0.1,
    transition: float = 0.05,
    dtype: Optional[DTypeLike] = None,
) -> Float[Array, 'n_signals n_timepoints']:
    r"""Band-limited coloured-noise sources via smooth spectral shaping.

    White Gaussian noise is shaped in the frequency domain by a **smooth
    raised-cosine** band ``[low, high]`` (normalised frequency, ``1`` =
    Nyquist) with cosine transition bands of width ``transition`` on each edge,
    then transformed back and standardised to zero mean / unit variance per
    signal.  Unlike a brick-wall bin mask, the smooth window has no spectral
    ringing; the output's expected power spectral density is proportional to
    the window squared, so its spectral support is the planted ground truth.

    ``low = 0`` gives low-pass ("slow") sources -- the fMRI-BOLD regime.

    Parameters
    ----------
    key
        PRNG key for the white-noise draw.
    n_signals, n_timepoints
        Number of signals and samples per signal.
    low, high
        Passband edges as fractions of the Nyquist frequency, in ``[0, 1]``.
    transition
        Width of each cosine transition band (same units); ``0`` collapses to
        a (leaky) brick wall and is discouraged.
    dtype
        Floating dtype; defaults to the x64-aware default float.

    Returns
    -------
    Float[Array, 'n_signals n_timepoints']
        Band-limited signals, each with (near) zero mean and unit variance.
    """
    dt = _default_float() if dtype is None else dtype
    noise = jax.random.normal(key, (n_signals, n_timepoints), dtype=dt)
    # Normalised frequency in [0, 1] (1 = Nyquist).
    freq = jnp.fft.rfftfreq(n_timepoints).astype(dt) * 2.0
    rise = _cosine_ramp((freq - (low - transition)) / transition)
    fall = _cosine_ramp(((high + transition) - freq) / transition)
    window = rise * fall
    spectrum = jnp.fft.rfft(noise, axis=-1) * window
    shaped = jnp.fft.irfft(spectrum, n=n_timepoints, axis=-1).astype(dt)
    shaped = shaped - jnp.mean(shaped, axis=-1, keepdims=True)
    std = jnp.std(shaped, axis=-1, keepdims=True)
    return shaped / jnp.where(std > 0, std, 1.0)


def color_signals(
    key: Array,
    target_cov: Float[Array, 'n_nodes n_nodes'],
    n_timepoints: int,
    *,
    dtype: Optional[DTypeLike] = None,
) -> Float[Array, 'n_nodes n_timepoints']:
    r"""Colour white noise to an exact planted cross-signal covariance.

    Maps unit white noise ``W`` (``n_nodes`` by ``n_timepoints``) through the
    colouring matrix :math:`\Sigma^{1/2}` so the output ``Y = \Sigma^{1/2} W``
    has population covariance :math:`\Sigma` (``= target_cov``): each time
    sample is a draw from :math:`\mathcal{N}(0, \Sigma)`, so the empirical
    covariance of ``Y`` recovers the planted ``target_cov``.

    Colouring is the exact inverse of ZCA whitening -- whitening applies
    :math:`\Sigma^{-1/2}`, colouring :math:`\Sigma^{+1/2}` -- and uses the same
    **cuSOLVER-free Newton-Schulz** square root
    (:func:`nitrix.linalg.symsqrt`, ``driver='newton_schulz'``), so it never
    lowers to the fragile solver pool and is differentiable w.r.t. ``target_cov``
    even at a repeated spectrum.

    Parameters
    ----------
    key
        PRNG key for the white-noise draw.
    target_cov
        The planted covariance ``(n_nodes, n_nodes)``: symmetric positive
        (semi-)definite.  Ridge a rank-deficient target (``+ eps I``) before
        calling; the matmul-only root cannot truncate zero eigenvalues.
    n_timepoints
        Number of time samples to draw.
    dtype
        Floating dtype; defaults to the x64-aware default float.

    Returns
    -------
    Float[Array, 'n_nodes n_timepoints']
        Coloured signals whose empirical covariance approaches ``target_cov``.
    """
    dt = _default_float() if dtype is None else dtype
    n_nodes = target_cov.shape[-1]
    white = jax.random.normal(key, (n_nodes, n_timepoints), dtype=dt)
    colouring = symsqrt(target_cov, driver='newton_schulz')  # Sigma^{1/2}
    return colouring @ white


def sparse_mixture_matrix(
    key: Array,
    n_observed: int,
    n_latent: int,
    *,
    expected_nnz: float = 3.0,
    dtype: Optional[DTypeLike] = None,
) -> Float[Array, 'n_observed n_latent']:
    r"""A sparse, row-``L1``-normalised mixing matrix.

    Each row is a random sparse combination of the latent sources: entries are
    kept with probability ``expected_nnz / n_latent`` (a Bernoulli mask), given
    positive weights, and ``L1``-normalised so the row sums to one.  The
    per-row cardinality is therefore ``Binomial(n_latent, p)`` with mean
    ``expected_nnz`` -- the fixed-shape, jit-safe reformulation of a
    Poisson-cardinality row (they agree as ``n_latent`` grows).  The matrix is
    the planted ground-truth mixing structure.

    Parameters
    ----------
    key
        PRNG key (split internally for the mask and the weights).
    n_observed, n_latent
        Number of observed signals (rows) and latent sources (columns).
    expected_nnz
        Expected number of active sources per observed signal.
    dtype
        Floating dtype; defaults to the x64-aware default float.

    Returns
    -------
    Float[Array, 'n_observed n_latent']
        Row-``L1``-normalised mixing matrix (an all-zero row -- possible when a
        row's mask is empty -- is left at zero rather than divided by zero).
    """
    dt = _default_float() if dtype is None else dtype
    k_mask, k_weight = jax.random.split(key, 2)
    p = jnp.clip(expected_nnz / n_latent, 0.0, 1.0)
    mask = jax.random.bernoulli(k_mask, p, (n_observed, n_latent))
    weights = jax.random.uniform(
        k_weight, (n_observed, n_latent), dtype=dt, minval=0.0, maxval=1.0
    )
    matrix = jnp.where(mask, weights, 0.0)
    row_sum = jnp.sum(matrix, axis=-1, keepdims=True)
    return matrix / jnp.where(row_sum > 0, row_sum, 1.0)


def mix_signals(
    mixture: Float[Array, 'n_observed n_latent'],
    sources: Float[Array, 'n_latent n_timepoints'],
    *,
    local: Optional[Float[Array, 'n_observed n_timepoints']] = None,
) -> Float[Array, 'n_observed n_timepoints']:
    r"""The mixture forward model ``mixture @ sources`` (+ optional local term).

    Produces observed signals with a known population covariance:
    :math:`\operatorname{cov}(\text{observed}) = M\,\operatorname{cov}(\text{
    sources})\,M^{\top}` (plus :math:`\operatorname{cov}(\text{local})` when a
    local component is supplied).  With standardised independent sources
    (:func:`band_limited_signals`) this is :math:`M M^{\top}` -- a planted
    connectome recoverable from the empirical covariance of the output.

    Parameters
    ----------
    mixture
        The mixing matrix ``(n_observed, n_latent)``
        (:func:`sparse_mixture_matrix`).
    sources
        Latent source signals ``(n_latent, n_timepoints)``.
    local
        Optional additive per-observed-signal component ``(n_observed,
        n_timepoints)`` (e.g. observation noise or a node-local signal),
        already generated by the caller.

    Returns
    -------
    Float[Array, 'n_observed n_timepoints']
        The observed signals.
    """
    observed = mixture @ sources
    if local is not None:
        observed = observed + local
    return observed


def lowrank_block_connectome(
    key: Array,
    communities: Int[Array, 'n_nodes'],
    n_communities: int,
    *,
    rank: int = 8,
    within_scale: float = 0.15,
    noise_rank: int = 4,
    noise_scale: float = 0.3,
    dtype: Optional[DTypeLike] = None,
) -> Float[Array, 'n_nodes n_nodes']:
    r"""A symmetric low-rank-plus-noise connectome with planted communities.

    Builds ``C = tanh(L Lᵀ) + N Nᵀ`` where the low-rank factor ``L`` gives
    same-community nodes a shared latent loading (a community centroid plus
    small within-community jitter), so nodes in a community are strongly
    connected; ``tanh`` bounds the block term to a correlation-like ``(-1, 1)``;
    and ``N Nᵀ`` adds a symmetric low-rank noise floor.  The result is
    symmetric by construction, and ``communities`` is the planted ground truth
    (recoverable from ``C``).  The block layout is fully parameterised (no
    hard-coded column count).

    Parameters
    ----------
    key
        PRNG key (split for the community centroids, the within-community
        jitter, and the noise factor).
    communities
        Per-node community label in ``[0, n_communities)``, shape
        ``(n_nodes,)`` -- the planted partition.
    n_communities
        Number of communities (static: sizes the centroid table).
    rank
        Rank of the block (signal) factor ``L``.
    within_scale
        Standard deviation of the within-community loading jitter (smaller =
        cleaner blocks).
    noise_rank, noise_scale
        Rank and scale of the additive ``N Nᵀ`` noise-outer-product floor
        (``noise_scale = 0`` gives a noise-free connectome).
    dtype
        Floating dtype; defaults to the x64-aware default float.

    Returns
    -------
    Float[Array, 'n_nodes n_nodes']
        The symmetric synthetic connectome.
    """
    dt = _default_float() if dtype is None else dtype
    n_nodes = communities.shape[0]
    k_centroid, k_jitter, k_noise = jax.random.split(key, 3)
    centroids = jax.random.normal(k_centroid, (n_communities, rank), dtype=dt)
    jitter = within_scale * jax.random.normal(
        k_jitter, (n_nodes, rank), dtype=dt
    )
    loadings = centroids[communities] + jitter  # (n_nodes, rank)
    block = jnp.tanh(loadings @ loadings.T)
    noise_factor = noise_scale * jax.random.normal(
        k_noise, (n_nodes, noise_rank), dtype=dt
    )
    connectome = block + noise_factor @ noise_factor.T
    return 0.5 * (connectome + connectome.T)


def markov_state_sequence(
    key: Array,
    transition: Float[Array, 'n_states n_states'],
    n_steps: int,
    *,
    initial_state: int = 0,
) -> Int[Array, 'n_steps']:
    r"""A keyed discrete-state trajectory from a Markov transition matrix.

    Samples ``n_steps`` states by a :func:`jax.lax.scan` over per-step keys:
    from the current state ``s`` the next is drawn ``Categorical(transition[s])``
    (in log space, so exact-zero transition probabilities are handled).  The
    row-stochastic ``transition`` is the planted ground truth: the empirical
    transition frequencies of a long trajectory recover it, and its stationary
    distribution is the long-run state occupancy.

    Parameters
    ----------
    key
        PRNG key (split into one sub-key per step).
    transition
        Row-stochastic transition matrix ``(n_states, n_states)`` (row ``i`` is
        the distribution over the next state given state ``i``).
    n_steps
        Number of states to emit (the first is drawn from
        ``transition[initial_state]``).
    initial_state
        The state preceding the first emitted step.

    Returns
    -------
    Int[Array, 'n_steps']
        The sampled state sequence.
    """
    log_transition = jnp.log(transition)
    keys = jax.random.split(key, n_steps)

    def step(
        state: Int[Array, ''],
        step_key: Array,
    ) -> tuple[Int[Array, ''], Int[Array, '']]:
        nxt = jax.random.categorical(step_key, log_transition[state]).astype(
            jnp.int32
        )
        return nxt, nxt

    _, sequence = jax.lax.scan(
        step, jnp.asarray(initial_state, dtype=jnp.int32), keys
    )
    return sequence
