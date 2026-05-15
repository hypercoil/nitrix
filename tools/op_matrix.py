# -*- coding: utf-8 -*-
"""
Op-matrix generator: catalogue every public nitrix op + live probe
its transformation support, perf characteristics, and declared
invariants.  Output: ``docs/op_matrix.md`` with provenance metadata.

The "**turn the matrix green**" goal: every public op should pass
the standard probes (jit, grad, vmap, jit-of-grad) and have a
known performance characterisation vs natural references.

Hardware / CUDA / backend caveat
--------------------------------

Status cells are **observed on the generating host**.  cuSolver-
broken stacks, missing Pallas kernels, and driver-version
incompatibilities all surface as red cells here that would be
green on a different runner.  The rendered Markdown carries the
host snapshot (device, JAX pin, driver) at the top so readers can
calibrate.

A future iteration could merge matrices across multiple runners
(one CI job per hardware target) into a colour-coded "supported
on X | broken on Y" table.  For first cut: single-host.

Run::

    python tools/op_matrix.py

Writes ``docs/op_matrix.md`` and ``docs/op_matrix.json``
(machine-readable form for downstream tooling).
"""
from __future__ import annotations

import json
import platform
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Probe primitives
# ---------------------------------------------------------------------------


def _safe_call(fn, *args, **kwargs):
    '''Run ``fn(*args, **kwargs)`` and return (out, err).

    Errors are caught and reported as short strings so the matrix
    runner doesn't bail on a single failing op.
    '''
    try:
        out = fn(*args, **kwargs)
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
        return out, None
    except Exception as e:  # noqa: BLE001
        msg = f'{type(e).__name__}: {str(e)[:120]}'
        return None, msg


def probe_jit(fn, args, kwargs):
    '''Does this op survive ``jax.jit``?'''
    jitted = jax.jit(lambda *a: fn(*a, **kwargs))
    _, err = _safe_call(jitted, *args)
    return 'pass' if err is None else err


def probe_vmap(fn, args, kwargs, vmap_arg: int):
    '''Does this op survive ``jax.vmap`` over leading axis of arg
    ``vmap_arg``?'''
    if vmap_arg is None:
        return 'n/a'
    new_args = list(args)
    # Add a leading singleton axis to the target arg
    target = new_args[vmap_arg]
    new_args[vmap_arg] = target[None]
    in_axes = [None] * len(new_args)
    in_axes[vmap_arg] = 0
    vmapped = jax.vmap(
        lambda *a: fn(*a, **kwargs), in_axes=tuple(in_axes),
    )
    _, err = _safe_call(vmapped, *new_args)
    return 'pass' if err is None else err


def probe_grad(fn, args, kwargs, diff_arg: int, reducer=None):
    '''Does ``jax.grad`` over a scalar reduction of ``fn`` work?'''
    if diff_arg is None:
        return 'n/a'
    if reducer is None:
        reducer = lambda x: jnp.sum(x ** 2)
    arg_target = args[diff_arg]

    def loss(x):
        new_args = list(args)
        new_args[diff_arg] = x
        out = fn(*new_args, **kwargs)
        # If fn returns a pytree (e.g., LME result), take first leaf
        leaves = jax.tree_util.tree_leaves(out)
        return reducer(leaves[0])

    g, err = _safe_call(jax.grad(loss), arg_target)
    if err is not None:
        return err
    if not bool(jnp.all(jnp.isfinite(g))):
        return 'grad has nan/inf'
    return 'pass'


def probe_jit_of_grad(fn, args, kwargs, diff_arg: int, reducer=None):
    '''The double-transform case ``jit(grad(...))``.'''
    if diff_arg is None:
        return 'n/a'
    if reducer is None:
        reducer = lambda x: jnp.sum(x ** 2)
    arg_target = args[diff_arg]

    def loss(x):
        new_args = list(args)
        new_args[diff_arg] = x
        out = fn(*new_args, **kwargs)
        leaves = jax.tree_util.tree_leaves(out)
        return reducer(leaves[0])

    grad_fn = jax.jit(jax.grad(loss))
    _, err = _safe_call(grad_fn, arg_target)
    return 'pass' if err is None else err


# ---------------------------------------------------------------------------
# Op metadata: hand-curated registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpInfo:
    '''Hand-curated metadata for a single public op.

    Attributes
    ----------
    qualname
        Dotted fully-qualified name, e.g. ``nitrix.linalg.rbf_kernel``.
    fixture
        Zero-arg callable returning ``(args_tuple, kwargs_dict)`` --
        a representative input.  Used to drive all probes.
    diff_arg
        Index into ``args`` of the differentiation target.  ``None``
        if the op has no natural diff target (e.g., it consumes only
        integer indices).
    vmap_arg
        Index into ``args`` of the vmap-batch axis.  ``None`` if the
        op is fundamentally batched already (in which case vmap is
        redundant) or has no natural batch axis.
    invariants
        Free-text tags for the algorithmic invariants the op
        exploits.  Hand-curated; matched against a known vocabulary
        for the rendered legend.
    perf_cpu_baseline, perf_gpu_baseline
        Short identifier strings for the CPU and GPU reference
        implementations the op's wall-time is compared against
        (e.g. ``"scipy.ndimage.gaussian_filter"`` /
        ``"cuDNN fp32"``).  Both are independent: an op can declare
        one, both, or neither.  Ratios get populated from the bench
        reports via ``load_perf_data``; cells with a declared name
        but no measured ratio render ``?``, cells with no name
        render ``—``.
    notes
        Free-text annotation (rendered in the table's "notes"
        column).
    reducer
        Optional custom scalar reducer for the grad probe.  Default
        ``sum(x**2)``; override when the natural reduction differs.
    '''

    qualname: str
    fixture: Callable[[], tuple]
    diff_arg: Optional[int] = 0
    vmap_arg: Optional[int] = 0
    invariants: tuple[str, ...] = ()
    # Perf baselines.  Two reference points: a CPU implementation (e.g.,
    # scipy / numpy / sklearn) and a GPU implementation (e.g., cuDNN /
    # torch / jnp built-in).  In both cases the ratio rendered is
    # ``nitrix-on-GPU / baseline-wall-time``; ``< 1`` means nitrix wins.
    # The name strings show up verbatim in the rendered table; the
    # ratios are scraped from ``bench/PERF_*.md`` via ``load_perf_data``.
    perf_cpu_baseline: Optional[str] = None
    perf_gpu_baseline: Optional[str] = None
    notes: str = ''
    reducer: Optional[Callable] = None
    skip_jit: bool = False  # construction-time op; JIT doesn't apply
    # Optional override: when set, the probes invoke this callable
    # instead of resolving ``qualname``.  Use for ops whose natural
    # signature carries non-array positional args (callables,
    # dataclasses) that need to be baked in via a closure to be
    # tractable under ``jax.jit`` / ``jax.vmap``.
    fn_override: Optional[Callable] = None

    def resolve(self):
        '''Return the callable used by the probes.

        If ``fn_override`` is set, returns it directly.  Otherwise
        imports by walking ``qualname``: imports the longest *strict
        prefix* (excluding the symbol itself) via ``importlib``,
        then ``getattr``s the final part.  This avoids the
        pathological case where ``nitrix.signal.tsconv`` could
        resolve to either the ``tsconv`` module or the ``tsconv``
        function inside it -- we always want the latter (modules
        aren't callable).
        '''
        if self.fn_override is not None:
            return self.fn_override
        import importlib
        parts = self.qualname.split('.')
        # Try the longest STRICT prefix first; never the whole
        # qualname (modules aren't useful here).
        mod = None
        for i in range(len(parts) - 1, 0, -1):
            try:
                mod = importlib.import_module('.'.join(parts[:i]))
                rest = parts[i:]
                break
            except ImportError:
                continue
        if mod is None:
            raise ImportError(f'cannot import any prefix of {self.qualname}')
        for p in rest:
            mod = getattr(mod, p)
        return mod


# ---------------------------------------------------------------------------
# Catalogue: per-op fixtures.  Curated by hand.
# ---------------------------------------------------------------------------


def _key(seed=0):
    return jax.random.key(seed)


CATALOGUE: list[OpInfo] = []


def register(op: OpInfo):
    CATALOGUE.append(op)
    return op


# --- linalg -----------------------------------------------------------------

register(OpInfo(
    'nitrix.linalg.symmetric',
    fixture=lambda: ((jax.random.normal(_key(), (4, 4)),), {}),
    invariants=('idempotent on symmetric input',),
))
register(OpInfo(
    'nitrix.linalg.sym2vec',
    fixture=lambda: ((jax.random.normal(_key(), (5, 5)),), {}),
    invariants=('strict upper triangle only', 'custom_vjp'),
    notes='returns vec of strict upper triangle',
))
register(OpInfo(
    'nitrix.linalg.vec2sym',
    fixture=lambda: ((jax.random.normal(_key(), (10,)),), {}),
    invariants=('mirrors upper triangle', 'custom_vjp'),
))
register(OpInfo(
    'nitrix.linalg.toeplitz_2d',
    fixture=lambda: ((jnp.arange(5.0), jnp.arange(5.0)), {}),
    invariants=('vmap-over-roll recipe',),
))
register(OpInfo(
    'nitrix.linalg.recondition_eigenspaces',
    fixture=lambda: (
        (jax.random.normal(_key(), (5, 5)) @ jax.random.normal(_key(), (5, 5)).T,),
        {'psi': 0.1},
    ),
    invariants=('PSD-preserving',),
))
register(OpInfo(
    'nitrix.linalg.residualise',
    fixture=lambda: (
        (
            jax.random.normal(_key(0), (10, 200)),
            jax.random.normal(_key(1), (3, 200)),
        ),
        {'method': 'cholesky'},
    ),
    invariants=('Cholesky-normal-equations',),
    perf_cpu_baseline='numpy.linalg.lstsq',
    notes='Cholesky path; ~800x faster than numpy lstsq at V=100k',
))
register(OpInfo(
    'nitrix.linalg.linear_kernel',
    fixture=lambda: ((jax.random.normal(_key(), (50, 16)),), {}),
    invariants=('shared with linear_distance via identity formula',),
    perf_cpu_baseline='sklearn.metrics.pairwise.linear_kernel',
))
register(OpInfo(
    'nitrix.linalg.linear_distance',
    fixture=lambda: ((jax.random.normal(_key(), (50, 16)),), {}),
    invariants=('|x-y|^2 = |x|^2 + |y|^2 - 2 x.y identity (O(nm) memory)',),
    perf_cpu_baseline='sklearn.metrics.pairwise_distances',
    notes='1000x memory reduction vs naive at d=1000',
))
register(OpInfo(
    'nitrix.linalg.rbf_kernel',
    fixture=lambda: ((jax.random.normal(_key(), (50, 16)),), {'gamma': 0.5}),
    invariants=('exp(-gamma * |x-y|^2)',),
    perf_cpu_baseline='sklearn.metrics.pairwise.rbf_kernel',
    notes='~375x faster than sklearn at (5000, 32)',
))
register(OpInfo(
    'nitrix.linalg.symlog',
    fixture=lambda: (
        (
            jax.random.normal(_key(), (5, 5)) @ jax.random.normal(_key(), (5, 5)).T
            + 0.5 * jnp.eye(5),
        ),
        {},
    ),
    invariants=('SPEC 4.1 stability rewrite', 'eigvalue-clip threshold'),
    perf_cpu_baseline='scipy.linalg.logm',
    notes='eigh-based; routes through safe_eigh cuSolver fallback',
))
register(OpInfo(
    'nitrix.linalg.symsqrt',
    fixture=lambda: (
        (
            jax.random.normal(_key(), (5, 5)) @ jax.random.normal(_key(), (5, 5)).T
            + 0.5 * jnp.eye(5),
        ),
        {},
    ),
    invariants=('eigvalue-clip threshold',),
    perf_cpu_baseline='scipy.linalg.sqrtm',
))
register(OpInfo(
    'nitrix.linalg.sympower',
    fixture=lambda: (
        (
            jax.random.normal(_key(), (5, 5)) @ jax.random.normal(_key(), (5, 5)).T
            + 0.5 * jnp.eye(5),
        ),
        {'power': -0.5},
    ),
    invariants=('arbitrary real power via eigh',),
    perf_cpu_baseline='scipy.linalg.fractional_matrix_power',
))
register(OpInfo(
    'nitrix.linalg.mean_log_euclidean',
    fixture=lambda: (
        (
            jnp.stack([
                jax.random.normal(_key(i), (4, 4)) @ jax.random.normal(_key(i), (4, 4)).T
                + 0.5 * jnp.eye(4) for i in range(3)
            ]),
        ),
        {},
    ),
    invariants=('closed-form Frechet mean on log-Euclidean metric',),
    vmap_arg=None,  # already batched
))

# --- stats ------------------------------------------------------------------

register(OpInfo(
    'nitrix.stats.cov',
    fixture=lambda: ((jax.random.normal(_key(), (5, 100)),), {}),
    invariants=('complex-Hermitian preserved', 'np.cov parity at fp64'),
    perf_cpu_baseline='numpy.cov',
    perf_gpu_baseline='jnp.cov',
    notes='130x faster than numpy at (2000, 1000)',
))
register(OpInfo(
    'nitrix.stats.corr',
    fixture=lambda: ((jax.random.normal(_key(), (5, 100)),), {}),
    invariants=('diagonal=1', 'complex-Hermitian preserved'),
    perf_cpu_baseline='numpy.corrcoef',
))
register(OpInfo(
    'nitrix.stats.partialcov',
    fixture=lambda: ((jax.random.normal(_key(), (5, 100)),), {}),
    invariants=('precision-matrix-derived',),
))
register(OpInfo(
    'nitrix.stats.precision',
    fixture=lambda: ((jax.random.normal(_key(), (5, 100)),), {}),
    invariants=('inverse of cov',),
))
register(OpInfo(
    'nitrix.stats.analytic_signal',
    fixture=lambda: ((jax.random.normal(_key(), (200,)),), {}),
    invariants=('vectorised Hilbert mask', 'scipy.signal.hilbert parity'),
    perf_cpu_baseline='scipy.signal.hilbert',
    # Complex output; reduce via |.|^2 so grad sees a real scalar.
    reducer=lambda x: jnp.sum(jnp.abs(x) ** 2),
))
register(OpInfo(
    'nitrix.stats.hilbert_transform',
    fixture=lambda: ((jax.random.normal(_key(), (200,)),), {}),
    invariants=('imag part of analytic_signal',),
    perf_cpu_baseline='scipy.signal.hilbert (.imag)',
))
register(OpInfo(
    'nitrix.stats.envelope',
    fixture=lambda: ((jax.random.normal(_key(), (200,)),), {}),
    invariants=('|analytic_signal|',),
    perf_cpu_baseline='numpy.abs(scipy.signal.hilbert)',
))


def _lme_fixture():
    rng = np.random.default_rng(0)
    g, n_per = 4, 20
    N = g * n_per
    Y = jnp.asarray(rng.standard_normal((8, N)).astype(np.float32))
    X = jnp.asarray(rng.standard_normal((N, 2)).astype(np.float32))
    Z = jnp.zeros((N, g), dtype=jnp.float32)
    for i in range(g):
        Z = Z.at[i * n_per:(i + 1) * n_per, i].set(1.0)
    return (Y, X, Z), {'n_iter': 10}


register(OpInfo(
    'nitrix.stats.lme.reml_fit',
    fixture=_lme_fixture,
    diff_arg=0,
    vmap_arg=None,  # already voxelwise
    invariants=('FaST-LMM spectral rotation', 'no V*N^2 intermediate'),
    perf_cpu_baseline='statsmodels.MixedLM',
    notes='~5e-3 parity with statsmodels.MixedLM',
))


def _flame_fixture():
    rng = np.random.default_rng(0)
    N, p, V = 30, 2, 16
    beta = jnp.asarray(rng.standard_normal((V, N)).astype(np.float32))
    var_within = jnp.asarray(
        (np.abs(rng.standard_normal((V, N))) + 0.1).astype(np.float32),
    )
    X_group = jnp.asarray(rng.standard_normal((N, p)).astype(np.float32))
    return (beta, var_within, X_group), {'n_iter': 10}


register(OpInfo(
    'nitrix.stats.lme.flame_two_level',
    fixture=_flame_fixture,
    diff_arg=0,
    vmap_arg=None,
    invariants=('single-parameter REML (identifiability)', 'shared X_group'),
    perf_cpu_baseline='statsmodels.WLS (voxelwise loop)',
))

# --- signal -----------------------------------------------------------------


def _interp_fixture():
    rng = np.random.default_rng(0)
    data = jnp.asarray(rng.standard_normal(50).astype(np.float32))
    mask = jnp.asarray((rng.random(50) > 0.3))
    return (data, mask), {}


register(OpInfo(
    'nitrix.signal.linear_interpolate',
    fixture=_interp_fixture,
    diff_arg=0,
    vmap_arg=None,
    invariants=('associative_scan (O(log n) parallel)', 'no while_loop'),
    perf_cpu_baseline='scipy.interpolate.interp1d',
))


def _ls_fixture():
    rng = np.random.default_rng(0)
    data = jnp.asarray(rng.standard_normal(100).astype(np.float32))
    mask = jnp.asarray((rng.random(100) > 0.2))
    return (data, mask), {'dt': 1.0}


register(OpInfo(
    'nitrix.signal.lomb_scargle_interpolate',
    fixture=_ls_fixture,
    diff_arg=0,
    vmap_arg=None,
    invariants=(
        'joint-GLM (passes through observed)',
        'shared-Gram across voxels',
        'no boundary discontinuity',
    ),
))
register(OpInfo(
    'nitrix.signal.lomb_scargle_periodogram',
    fixture=_ls_fixture,
    diff_arg=0,
    vmap_arg=None,
    invariants=('Scargle 1982 normalisation', 'scipy.lombscargle parity'),
    perf_cpu_baseline='scipy.signal.lombscargle',
))
register(OpInfo(
    'nitrix.signal.polynomial_detrend',
    fixture=lambda: ((jax.random.normal(_key(), (5, 100)),), {'degree': 2}),
    invariants=('rescaled Vandermonde (stability)', 'routes through residualise'),
    perf_cpu_baseline='scipy.signal.detrend (poly)',
))


def _tsconv_fixture():
    X = jax.random.normal(_key(0), (4, 100))
    w = jax.random.normal(_key(1), (4, 4, 5))
    return (X, w), {}


register(OpInfo(
    'nitrix.signal.tsconv',
    fixture=_tsconv_fixture,
    diff_arg=0,
    vmap_arg=None,
    invariants=('thin lax.conv_general_dilated wrapper',),
    perf_cpu_baseline='scipy.signal.convolve',
    perf_gpu_baseline='lax.conv_general_dilated (raw)',
))

# --- numerics ---------------------------------------------------------------

register(OpInfo(
    'nitrix.numerics.zscore_normalize',
    fixture=lambda: ((jax.random.normal(_key(), (5, 100)),), {}),
    invariants=('zero mean, unit std per axis',),
    perf_cpu_baseline='scipy.stats.zscore',
))
register(OpInfo(
    'nitrix.numerics.intensity_normalize',
    fixture=lambda: ((jax.random.normal(_key(), (1000,)) * 10 + 50,), {}),
    invariants=('percentile-clip to [0, 1]',),
))
register(OpInfo(
    'nitrix.numerics.complex_decompose',
    fixture=lambda: (
        (jax.random.normal(_key(0), (20,)) + 1j * jax.random.normal(_key(1), (20,)),),
        {},
    ),
    invariants=('amplitude / phase split',),
))

# --- geometry ---------------------------------------------------------------

register(OpInfo(
    'nitrix.geometry.identity_grid',
    fixture=lambda: (((4, 4),), {}),
    diff_arg=None,
    vmap_arg=None,
    skip_jit=True,
    invariants=('shape-static constructor',),
    notes='shape-tuple input; not a runtime op',
))


def _spatial_transform_fixture():
    image = jax.random.normal(_key(0), (4, 4, 2))
    deform = jax.random.normal(_key(1), (4, 4, 2)) * 0.5
    return (image, deform), {'mode': 'nearest'}


register(OpInfo(
    'nitrix.geometry.spatial_transform',
    fixture=_spatial_transform_fixture,
    invariants=('mode pass-through to map_coordinates', 'accepts leading batch'),
    perf_cpu_baseline='scipy.ndimage.map_coordinates',
))


def _ivf_fixture():
    v = jax.random.normal(_key(), (8, 8, 2)) * 0.1
    return (v,), {'n_steps': 5}


register(OpInfo(
    'nitrix.geometry.integrate_velocity_field',
    fixture=_ivf_fixture,
    invariants=('scaling-and-squaring', "default mode='nearest' (voxelmorph)"),
))
register(OpInfo(
    'nitrix.geometry.jacobian_det_displacement',
    fixture=lambda: ((jax.random.normal(_key(), (8, 8, 2)) * 0.1,), {}),
    invariants=('explicit det for d<=3', 'no cuSolver call'),
))
register(OpInfo(
    'nitrix.geometry.sphere_grid_pad_2d',
    fixture=lambda: ((jax.random.normal(_key(), (4, 4)),), {'pad': 1}),
    invariants=('pole-flip + W/2 roll', 'longitudinal wrap'),
))

# --- graph ------------------------------------------------------------------


def _laplacian_fixture():
    rng = np.random.default_rng(0)
    n = 20
    A = (rng.random((n, n)) > 0.7).astype(np.float32)
    A = (A + A.T).clip(max=1.0) - jnp.diag(jnp.diag(jnp.asarray(A)))
    return (jnp.asarray(A) + jnp.asarray(A).T,), {}


register(OpInfo(
    'nitrix.graph.laplacian',
    fixture=_laplacian_fixture,
    invariants=('symmetric / random_walk / combinatorial variants',),
    perf_cpu_baseline='scipy.sparse.csgraph.laplacian',
))
register(OpInfo(
    'nitrix.graph.degree_vector',
    fixture=_laplacian_fixture,
    invariants=('dense / ELL / SectionedELL dispatch',),
))


def _laplacian_eigenmap_fixture():
    rng = np.random.default_rng(0)
    n = 32
    A = (rng.random((n, n)) > 0.5).astype(np.float32)
    A = ((A + A.T) > 0).astype(np.float32)
    return (jnp.asarray(A),), {'n_components': 3, 'solver': 'eigh'}


register(OpInfo(
    'nitrix.graph.laplacian_eigenmap',
    fixture=_laplacian_eigenmap_fixture,
    invariants=(
        'safe_eigh cuSolver fallback',
        'LOBPCG implicit-VJP for sparse paths',
    ),
    vmap_arg=None,
    perf_cpu_baseline='sklearn.manifold.SpectralEmbedding',
    notes='dense=eigh, sparse=lobpcg; differentiable end-to-end',
))

# --- morphology -------------------------------------------------------------

register(OpInfo(
    'nitrix.morphology.dilate',
    fixture=lambda: ((jax.random.normal(_key(), (10, 10)),), {'size': 3}),
    invariants=('TROPICAL_MAX_PLUS specialisation',),
    perf_cpu_baseline='scipy.ndimage.grey_dilation',
))
register(OpInfo(
    'nitrix.morphology.erode',
    fixture=lambda: ((jax.random.normal(_key(), (10, 10)),), {'size': 3}),
    invariants=('TROPICAL_MIN_PLUS specialisation',),
    perf_cpu_baseline='scipy.ndimage.grey_erosion',
))
register(OpInfo(
    'nitrix.morphology.distance_transform',
    fixture=lambda: (
        ((jax.random.normal(_key(), (12, 12)) > 0).astype(jnp.float32),),
        {},
    ),
    invariants=('iterative TROPICAL_MIN_PLUS', 'chamfer (not exact EDT)'),
    perf_cpu_baseline='scipy.ndimage.distance_transform_edt',
    notes='15x slower than scipy EDT at (64,64) -- algorithm mismatch',
))
register(OpInfo(
    'nitrix.morphology.median_filter',
    fixture=lambda: ((jax.random.normal(_key(), (16, 16)),), {'size': 3}),
    invariants=('gather + nanmedian (not a semiring op)',),
    perf_cpu_baseline='scipy.ndimage.median_filter',
))


def _max_pool_fixture():
    # (B=1, C=2, H=8, W=8); pool 2x2 -> (1, 2, 4, 4) with indices
    x = jax.random.normal(_key(), (1, 2, 8, 8))
    return (x,), {'pool_size': 2, 'spatial_rank': 2}


def _max_pool_reducer(out):
    # max_pool_with_indices_nd returns (pooled, indices).
    # grad probe takes the first leaf via tree_leaves -- which is
    # the pooled tensor.  Reduce that to a scalar.
    return jnp.sum(out ** 2)


register(OpInfo(
    'nitrix.morphology.max_pool_with_indices_nd',
    fixture=_max_pool_fixture,
    diff_arg=0,
    vmap_arg=None,  # batch axis already in fixture; vmap-over-extra-leading is redundant
    invariants=(
        'global flat C-order argmax indices',
        'window-unfold + argmax composition',
    ),
    reducer=_max_pool_reducer,
    perf_gpu_baseline='torch.nn.MaxPool3d (return_indices)',
    notes='returns (pooled, indices); paired with max_unpool_nd for encoder-decoder',
))


def _max_unpool_fixture():
    rng = np.random.default_rng(0)
    pooled = jnp.asarray(rng.standard_normal((1, 2, 4, 4)).astype(np.float32))
    indices = jnp.asarray(rng.integers(0, 64, (1, 2, 4, 4)).astype(np.int32))
    return (pooled, indices), {'output_shape': (8, 8), 'spatial_rank': 2}


register(OpInfo(
    'nitrix.morphology.max_unpool_nd',
    fixture=_max_unpool_fixture,
    diff_arg=0,
    vmap_arg=None,
    invariants=(
        'vmapped per-channel scatter',
        'argmax-agreement parity (not raw-logit allclose)',
    ),
    perf_gpu_baseline='torch.nn.MaxUnpool3d',
    notes='inverts max_pool_with_indices_nd; indices are int (non-diff)',
))

# --- smoothing --------------------------------------------------------------

register(OpInfo(
    'nitrix.smoothing.gaussian',
    fixture=lambda: ((jax.random.normal(_key(), (16, 16)),), {'sigma': 1.5}),
    invariants=('separable n-D', 'scipy.ndimage parity at fp64'),
    perf_cpu_baseline='scipy.ndimage.gaussian_filter',
))
register(OpInfo(
    'nitrix.smoothing.bilateral_gaussian',
    fixture=lambda: (
        (
            jax.random.normal(_key(0), (32, 1)),
            jax.random.normal(_key(1), (32, 2)),
        ),
        {
            'sigma_features': jnp.asarray([1.0, 1.0]),
            'neighbourhood': 8,
        },
    ),
    invariants=('semiring_ell_matmul over feature adjacency',),
))

# --- semiring ---------------------------------------------------------------

register(OpInfo(
    'nitrix.semiring.semiring_matmul',
    fixture=lambda: (
        (
            jax.random.normal(_key(0), (32, 16)),
            jax.random.normal(_key(1), (16, 32)),
        ),
        {},
    ),
    invariants=(
        'streaming kernel (no O(MKN) materialisation)',
        'Pallas/JAX fallback',
    ),
    perf_cpu_baseline='numpy.matmul',
    perf_gpu_baseline='lax.fori_loop (same algebra)',
    notes='6-16x faster than JAX fori_loop on REAL/TROPICAL/EUCLIDEAN',
))


def _ell_fixture():
    rng = np.random.default_rng(0)
    n = 20
    k_max = 5
    values = jnp.asarray(rng.standard_normal((n, k_max)).astype(np.float32))
    indices = jnp.asarray(rng.integers(0, n, (n, k_max)).astype(np.int32))
    B = jnp.asarray(rng.standard_normal((n, 4)).astype(np.float32))
    return (values, indices, B), {'n_cols': n, 'backend': 'jax'}


register(OpInfo(
    'nitrix.semiring.semiring_ell_matmul',
    fixture=_ell_fixture,
    diff_arg=0,
    vmap_arg=None,
    invariants=('sparse ELL matmul',),
    perf_cpu_baseline='scipy.sparse @ dense',
))


def _ell_edge_setup():
    '''Build a GCN-style closure that takes ``x`` as its only
    positional arg, baking in edge_fn / ell / semiring.  Returns
    the closure plus a fixture-of-the-closure.
    '''
    from nitrix.semiring import REAL, semiring_ell_edge_aggregate
    from nitrix.sparse import ELL
    rng = np.random.default_rng(0)
    n, k_max, d_in = 8, 4, 3
    values = jnp.asarray(rng.standard_normal((n, k_max)).astype(np.float32))
    indices = jnp.asarray(rng.integers(0, n, (n, k_max)).astype(np.int32))
    W = jnp.asarray(rng.standard_normal((4, d_in)).astype(np.float32))
    ell = ELL(values=values, indices=indices, n_cols=n, identity=0.0)

    def edge_fn(h_i, h_j, w, ij):
        return w * (W @ h_j)

    def op(x):
        return semiring_ell_edge_aggregate(edge_fn, ell, x, semiring=REAL)

    def fixture():
        x = jnp.asarray(rng.standard_normal((n, d_in)).astype(np.float32))
        return (x,), {}

    return op, fixture


_edge_agg_op, _edge_agg_fixture = _ell_edge_setup()

register(OpInfo(
    'nitrix.semiring.semiring_ell_edge_aggregate',
    fixture=_edge_agg_fixture,
    fn_override=_edge_agg_op,
    diff_arg=0,
    vmap_arg=0,
    invariants=(
        'gather + nested vmap + semiring reduction',
        'REAL / TROPICAL_MAX_PLUS / TROPICAL_MIN_PLUS supported',
        'edge_fn signature (h_i, h_j, w, ij)',
    ),
    perf_gpu_baseline='torch_geometric.nn.MessagePassing',
    notes='probed with GCN closure; covers GCN/GAT/EdgeConv/MoNet/ChebNet',
))


def _conv_fixture():
    X = jax.random.normal(_key(0), (1, 8, 8, 1))
    K = jax.random.normal(_key(1), (3, 3, 1, 1))
    return (X, K), {}


register(OpInfo(
    'nitrix.semiring.semiring_conv',
    fixture=_conv_fixture,
    invariants=(
        'NaN-safe patch extraction (jnp.take, not lax.conv_general_dilated_patches)',
        'explicit im2col + semiring_matmul',
    ),
    perf_cpu_baseline='scipy.ndimage.convolve',
    perf_gpu_baseline='cuDNN fp32 (precision=highest)',
    notes='1.7-1.9x slower than cuDNN fp32 (literature expected)',
))

# --- sparse -----------------------------------------------------------------

register(OpInfo(
    'nitrix.sparse.ell_from_dense',
    fixture=lambda: ((jax.random.normal(_key(), (10, 10)),), {}),
    diff_arg=None,
    vmap_arg=None,
    skip_jit=True,
    invariants=('host-side ELL construction',),
))
register(OpInfo(
    'nitrix.sparse.grid_laplacian',
    fixture=lambda: (((8, 8),), {}),
    diff_arg=None,
    vmap_arg=None,
    skip_jit=True,
    invariants=('regular-grid stencil', 'scipy.ndimage.laplace parity'),
))


def _mesh_fixture():
    from nitrix.sparse import icosphere
    m = icosphere(1)
    return (m,), {}


register(OpInfo(
    'nitrix.sparse.mesh_k_ring_adjacency',
    fixture=_mesh_fixture,
    diff_arg=None,
    vmap_arg=None,
    skip_jit=True,
    invariants=('BFS k-ring on triangle mesh', 'host-side construction'),
))


def _mesh_pool_setup():
    '''Build a closure-wrapped mesh_pool_max + a fixture that returns
    only the fine-features ``x`` (ELL is baked in).
    '''
    from nitrix.sparse import (
        icosphere_cross_level_adjacency, icosphere_hierarchy, mesh_pool_max,
    )
    rng = np.random.default_rng(0)
    h = icosphere_hierarchy(max_level=1)
    pool_ell = icosphere_cross_level_adjacency(h, 0, 1)
    n_fine = h.meshes[1].n_vertices

    def op(x):
        return mesh_pool_max(pool_ell, x)

    def fixture():
        x = jnp.asarray(rng.standard_normal((n_fine, 3)).astype(np.float32))
        return (x,), {}

    return op, fixture


_pool_op, _pool_fixture = _mesh_pool_setup()

register(OpInfo(
    'nitrix.sparse.mesh_pool_max',
    fixture=_pool_fixture,
    fn_override=_pool_op,
    diff_arg=0,
    vmap_arg=0,
    invariants=(
        'TROPICAL_MAX_PLUS with zero values',
        'composes semiring_ell_matmul',
    ),
    notes='probed against icosphere(0->1) cross-level adjacency',
))


def _mesh_unpool_setup():
    from nitrix.sparse import (
        icosphere_bary_upsampler, icosphere_hierarchy, mesh_unpool_max,
    )
    rng = np.random.default_rng(0)
    h = icosphere_hierarchy(max_level=1)
    bary = icosphere_bary_upsampler(h, 0, 1)
    n_coarse = h.meshes[0].n_vertices

    def op(x):
        return mesh_unpool_max(bary, x)

    def fixture():
        x = jnp.asarray(rng.standard_normal((n_coarse, 3)).astype(np.float32))
        return (x,), {}

    return op, fixture


_unpool_op, _unpool_fixture = _mesh_unpool_setup()

register(OpInfo(
    'nitrix.sparse.mesh_unpool_max',
    fixture=_unpool_fixture,
    fn_override=_unpool_op,
    diff_arg=0,
    vmap_arg=0,
    invariants=(
        'TROPICAL_MAX_PLUS symmetric of mesh_pool_max',
    ),
))


def _mesh_bary_setup():
    from nitrix.sparse import (
        icosphere_bary_upsampler, icosphere_hierarchy, mesh_bary_upsample,
    )
    rng = np.random.default_rng(0)
    h = icosphere_hierarchy(max_level=1)
    bary = icosphere_bary_upsampler(h, 0, 1)
    n_coarse = h.meshes[0].n_vertices

    def op(coarse):
        return mesh_bary_upsample(bary, coarse)

    def fixture():
        coarse = jnp.asarray(rng.standard_normal((n_coarse, 3)).astype(np.float32))
        return (coarse,), {}

    return op, fixture


_bary_op, _bary_fixture = _mesh_bary_setup()

register(OpInfo(
    'nitrix.sparse.mesh_bary_upsample',
    fixture=_bary_fixture,
    fn_override=_bary_op,
    diff_arg=0,
    vmap_arg=0,
    invariants=(
        'REAL semiring_ell_matmul (weighted sum)',
    ),
))


# Icosphere hierarchy construction (Sprint B) -- host-side, skip_jit.

register(OpInfo(
    'nitrix.sparse.icosphere_hierarchy',
    fixture=lambda: ((2,), {}),
    diff_arg=None,
    vmap_arg=None,
    skip_jit=True,
    invariants=(
        'recursive subdivision with midpoint cache',
        'parent-child bookkeeping for cross-level helpers',
    ),
    notes='returns IcosphereHierarchy(meshes, parents); host-side',
))


def _hier_pair_fixture():
    from nitrix.sparse import icosphere_hierarchy
    h = icosphere_hierarchy(max_level=1)
    return (h, 0, 1), {}


register(OpInfo(
    'nitrix.sparse.icosphere_cross_level_adjacency',
    fixture=_hier_pair_fixture,
    diff_arg=None,
    vmap_arg=None,
    skip_jit=True,
    invariants=(
        'coarse-to-fine ELL from subdivision parents',
        'k_max = 1 + max_coarse_degree (6 at L=0->1, 7 at L>=1)',
    ),
    notes='consecutive-level only; compose for multi-level',
))


register(OpInfo(
    'nitrix.sparse.icosphere_bary_upsampler',
    fixture=_hier_pair_fixture,
    diff_arg=None,
    vmap_arg=None,
    skip_jit=True,
    invariants=(
        'fine-from-coarse ELL with bary weights (1, 0) or (0.5, 0.5)',
        'k_max = 2',
    ),
    notes='consecutive-level only; feeds mesh_bary_upsample',
))


# ---------------------------------------------------------------------------
# Host snapshot
# ---------------------------------------------------------------------------


def host_snapshot() -> dict:
    '''Capture the host configuration for the rendered matrix.

    Every cell in the rendered table is *observed on this host*;
    a different runner may produce different statuses.
    '''
    try:
        d = jax.devices()[0]
        device = f'{d.device_kind} ({d.platform})'
    except Exception:  # noqa: BLE001
        device = 'unknown'
    return {
        'device': device,
        'platform': platform.platform(),
        'jax_version': jax.__version__,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }


# ---------------------------------------------------------------------------
# Perf-data loader: scrape PERF_*.md files
# ---------------------------------------------------------------------------


def _parse_ratio_token(tok: str) -> Optional[float]:
    '''Parse a ratio cell like ``0.06x`` / ``1.70×`` / ``16.50×``.

    Tolerates lowercase ``x``, unicode ``×``, and trailing whitespace.
    Returns ``None`` if the token doesn't look like a ratio.
    '''
    s = tok.strip().replace('×', '').replace('x', '')
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _parse_perf_audit(audit_path: Path) -> dict[str, float]:
    '''Read PERF_AUDIT.md and pull the **best** (smallest) ratio per
    op across all measured sizes.

    The audit's "op" column embeds the subpackage and function name
    (``stats.cov``, ``morphology.distance_transform``, etc.); strip
    any parenthetical suffix (``" (Cholesky)"``, ``" (voxelwise)"``)
    so the keys match the qualname short-name used elsewhere in the
    matrix.

    Returns a dict mapping the *normalised* op short-name
    (last dotted segment) to the best observed ratio (nitrix / CPU
    baseline; smaller = bigger nitrix win).
    '''
    out: dict[str, float] = {}
    if not audit_path.exists():
        return out
    for line in audit_path.read_text().splitlines():
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.split('|')[1:-1]]
        if len(cells) < 6:
            continue
        # Skip header / separator rows.
        if cells[0].lower() in ('op', '---', ''):
            continue
        op_raw = cells[0]
        ratio = _parse_ratio_token(cells[5])
        if ratio is None:
            continue
        # Strip parenthetical suffix; use the leaf (function name).
        op_clean = op_raw.split('(')[0].strip()
        short = op_clean.split('.')[-1]
        if short not in out or ratio < out[short]:
            out[short] = ratio
    return out


def _parse_perf_semiring_conv(report_path: Path) -> Optional[float]:
    '''Pull the ``REAL (Pallas mm) / fp32`` ratio for ``semiring_conv``.

    The PERF_SEMIRING_CONV.md report has 2D and 3D rows; take the
    smaller (best apples-to-apples) ratio.
    '''
    if not report_path.exists():
        return None
    best: Optional[float] = None
    for line in report_path.read_text().splitlines():
        if not line.startswith('|') or 'cuDNN' in line or 'shape' in line:
            continue
        cells = [c.strip() for c in line.split('|')[1:-1]]
        # 2D table: shape | cuDNN TC | cuDNN fp32 | REAL Pallas | / TC | / fp32 | ...
        if len(cells) < 6:
            continue
        ratio = _parse_ratio_token(cells[5])
        if ratio is None:
            continue
        best = ratio if best is None else min(best, ratio)
    return best


def _parse_perf_semiring_matmul(report_path: Path) -> Optional[float]:
    '''Pull the best Pallas-vs-JAX ratio for ``semiring_matmul``.

    The bench reports Pallas speed-up (e.g., 16.50×); convert to
    a nitrix / baseline ratio (1 / speed-up) so it's directionally
    consistent with the other cells (smaller = bigger nitrix win).
    Take the best (smallest) ratio across measured shapes.

    Caveat: the "JAX" baseline here is the same-algorithm
    ``lax.fori_loop`` fallback, not ``jnp.matmul`` with tensor cores.
    The ratio cell carries this baseline name verbatim from
    ``OpInfo.perf_gpu_baseline``.
    '''
    if not report_path.exists():
        return None
    best: Optional[float] = None
    for line in report_path.read_text().splitlines():
        if not line.startswith('|') or 'algebra' in line.lower() or 'Pallas speed-up' in line:
            continue
        cells = [c.strip() for c in line.split('|')[1:-1]]
        # Steady-state table: m | k | n | algebra | JAX | Pallas | Pallas speed-up
        if len(cells) < 7:
            continue
        algebra = cells[3].lower()
        # Only REAL is comparable to the GPU baseline we'd name
        # ("JAX fori_loop"); other algebras have no third-party GPU
        # ref so we'd render '?' anyway.
        if algebra != 'real':
            continue
        speedup = _parse_ratio_token(cells[6])
        if speedup is None or speedup == 0:
            continue
        ratio = 1.0 / speedup
        best = ratio if best is None else min(best, ratio)
    return best


def load_perf_data() -> dict[tuple[str, str], float]:
    '''Build the unified perf-ratio lookup.

    Returns a dict keyed by ``(op_short_name, 'cpu' | 'gpu')`` whose
    value is the best observed ratio against the named baseline.
    Cells with a declared baseline name but no scraped ratio render
    as ``?``; cells with no baseline name render as ``—``.
    '''
    bench_dir = Path(__file__).parent.parent / 'bench'
    out: dict[tuple[str, str], float] = {}

    # CPU baselines: PERF_AUDIT covers most external-library references.
    for short, ratio in _parse_perf_audit(bench_dir / 'PERF_AUDIT.md').items():
        out[(short, 'cpu')] = ratio

    # GPU baselines.
    conv_ratio = _parse_perf_semiring_conv(bench_dir / 'PERF_SEMIRING_CONV.md')
    if conv_ratio is not None:
        out[('semiring_conv', 'gpu')] = conv_ratio

    mm_ratio = _parse_perf_semiring_matmul(bench_dir / 'PERF_SEMIRING_MATMUL.md')
    if mm_ratio is not None:
        out[('semiring_matmul', 'gpu')] = mm_ratio

    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class CellResult:
    op: OpInfo
    jit: str
    grad: str
    vmap: str
    jit_of_grad: str


def run_probes(op: OpInfo) -> CellResult:
    fn = op.resolve()
    args, kwargs = op.fixture()
    if op.skip_jit:
        jit_status = 'n/a'
        vmap_status = 'n/a'
        grad_status = 'n/a'
        jit_grad_status = 'n/a'
    else:
        jit_status = probe_jit(fn, args, kwargs)
        vmap_status = probe_vmap(fn, args, kwargs, op.vmap_arg)
        grad_status = probe_grad(fn, args, kwargs, op.diff_arg, op.reducer)
        jit_grad_status = probe_jit_of_grad(fn, args, kwargs, op.diff_arg, op.reducer)
    return CellResult(
        op=op,
        jit=jit_status,
        grad=grad_status,
        vmap=vmap_status,
        jit_of_grad=jit_grad_status,
    )


def _status_to_glyph(status: str) -> str:
    if status == 'pass':
        return '✅'
    if status == 'n/a':
        return '—'
    # error message
    return '❌'


def _format_ratio(r: float) -> str:
    '''Render a ratio as ``0.06x`` / ``1.70x`` / ``<0.01x``.

    Numbers under 0.01 collapse to ``<0.01x`` to keep cells tight
    while flagging "much bigger than 100x win" clearly.
    '''
    if r < 0.01:
        return '<0.01x'
    if r < 1.0:
        return f'{r:.2f}x'
    return f'{r:.1f}x'


def render_markdown(
    cells: list[CellResult],
    host: dict,
    perf_data: dict[tuple[str, str], float],
) -> str:
    lines = []
    lines.append('# nitrix op matrix')
    lines.append('')
    lines.append(
        'Auto-generated by ``tools/op_matrix.py``.  Status cells are '
        '**observed on the generating host** -- a different hardware / '
        'CUDA / JAX pin may produce different statuses.  See '
        '``docs/design/op-matrix.md`` for the design rationale.'
    )
    lines.append('')
    lines.append('## Host snapshot')
    lines.append('')
    for k, v in host.items():
        lines.append(f'- **{k}**: {v}')
    lines.append('')
    lines.append('## Legend')
    lines.append('')
    lines.append(
        '- ✅ probe passes (transformation works on this host).'
    )
    lines.append('- ❌ probe fails (cell shows the truncated error message).')
    lines.append('- — not applicable for this op (no natural diff target, etc.).')
    lines.append('')
    lines.append('### Perf columns')
    lines.append('')
    lines.append(
        '- **CPU baseline / GPU baseline**: short identifier of the '
        'reference implementation each ratio is computed against.  '
        '``—`` means no natural baseline of that kind exists for the op.'
    )
    lines.append(
        '- **CPU ratio / GPU ratio**: ``nitrix-on-GPU / baseline-wall-time``, '
        'best observed across measured sizes.  ``< 1`` means nitrix is '
        'faster than the baseline (a "win"); ``> 1`` means slower.  ``?`` '
        'means a baseline is declared but no benchmark has been run yet.'
    )
    lines.append('')

    # Group by package
    by_package: dict[str, list[CellResult]] = {}
    for c in cells:
        pkg = '.'.join(c.op.qualname.split('.')[:2])
        by_package.setdefault(pkg, []).append(c)

    for pkg in sorted(by_package):
        lines.append(f'## {pkg}')
        lines.append('')
        lines.append(
            '| op | jit | grad | vmap | jit(grad) | CPU baseline | CPU ratio | '
            'GPU baseline | GPU ratio | invariants | notes |'
        )
        lines.append(
            '|---|:--:|:--:|:--:|:--:|---|---:|---|---:|---|---|'
        )
        for c in by_package[pkg]:
            short = c.op.qualname.split('.')[-1]
            jit_g = _status_to_glyph(c.jit)
            grad_g = _status_to_glyph(c.grad)
            vmap_g = _status_to_glyph(c.vmap)
            jgrad_g = _status_to_glyph(c.jit_of_grad)

            cpu_name = c.op.perf_cpu_baseline or '—'
            gpu_name = c.op.perf_gpu_baseline or '—'
            cpu_ratio_raw = perf_data.get((short, 'cpu'))
            gpu_ratio_raw = perf_data.get((short, 'gpu'))

            if c.op.perf_cpu_baseline is None:
                cpu_ratio = '—'
            elif cpu_ratio_raw is None:
                cpu_ratio = '?'
            else:
                cpu_ratio = _format_ratio(cpu_ratio_raw)

            if c.op.perf_gpu_baseline is None:
                gpu_ratio = '—'
            elif gpu_ratio_raw is None:
                gpu_ratio = '?'
            else:
                gpu_ratio = _format_ratio(gpu_ratio_raw)

            inv = '; '.join(c.op.invariants) if c.op.invariants else ''
            notes = c.op.notes
            # Embed error message if status != pass / n/a
            err_parts = []
            for label, status in [
                ('jit', c.jit), ('grad', c.grad),
                ('vmap', c.vmap), ('jit(grad)', c.jit_of_grad),
            ]:
                if status not in ('pass', 'n/a'):
                    err_parts.append(f'`{label}: {status}`')
            if err_parts:
                notes = (notes + ' — errors: ' + '; '.join(err_parts)).lstrip(' —')
            lines.append(
                f'| `{short}` | {jit_g} | {grad_g} | {vmap_g} | '
                f'{jgrad_g} | {cpu_name} | {cpu_ratio} | {gpu_name} | '
                f'{gpu_ratio} | {inv} | {notes} |'
            )
        lines.append('')

    lines.append('## Summary')
    lines.append('')
    n_total = len(cells)
    n_jit = sum(1 for c in cells if c.jit == 'pass')
    n_grad = sum(1 for c in cells if c.grad == 'pass')
    n_vmap = sum(1 for c in cells if c.vmap == 'pass')
    n_jgrad = sum(1 for c in cells if c.jit_of_grad == 'pass')
    n_grad_app = sum(1 for c in cells if c.op.diff_arg is not None)
    n_vmap_app = sum(1 for c in cells if c.op.vmap_arg is not None)
    n_cpu_named = sum(1 for c in cells if c.op.perf_cpu_baseline is not None)
    n_gpu_named = sum(1 for c in cells if c.op.perf_gpu_baseline is not None)
    n_cpu_measured = sum(
        1 for c in cells
        if (c.op.qualname.split('.')[-1], 'cpu') in perf_data
    )
    n_gpu_measured = sum(
        1 for c in cells
        if (c.op.qualname.split('.')[-1], 'gpu') in perf_data
    )
    n_cpu_wins = sum(
        1 for c in cells
        if perf_data.get((c.op.qualname.split('.')[-1], 'cpu'), 2.0) < 1.0
    )
    n_gpu_wins = sum(
        1 for c in cells
        if perf_data.get((c.op.qualname.split('.')[-1], 'gpu'), 2.0) < 1.0
    )
    lines.append(f'- **ops catalogued**: {n_total}')
    lines.append(f'- **jit pass**: {n_jit} / {n_total}')
    lines.append(f'- **grad pass**: {n_grad} / {n_grad_app} (applicable)')
    lines.append(f'- **vmap pass**: {n_vmap} / {n_vmap_app} (applicable)')
    lines.append(f'- **jit(grad) pass**: {n_jgrad} / {n_grad_app}')
    lines.append(
        f'- **CPU baseline declared / measured**: {n_cpu_named} / '
        f'{n_cpu_measured} (of {n_total})'
    )
    lines.append(
        f'- **GPU baseline declared / measured**: {n_gpu_named} / '
        f'{n_gpu_measured} (of {n_total})'
    )
    lines.append(
        f'- **wins (ratio < 1) against CPU / GPU baselines**: '
        f'{n_cpu_wins} / {n_gpu_wins}'
    )
    return '\n'.join(lines)


def render_json(
    cells: list[CellResult],
    host: dict,
    perf_data: dict[tuple[str, str], float],
) -> str:
    out = {
        'host': host,
        'ops': [],
    }
    for c in cells:
        short = c.op.qualname.split('.')[-1]
        out['ops'].append({
            'qualname': c.op.qualname,
            'jit': c.jit,
            'grad': c.grad,
            'vmap': c.vmap,
            'jit_of_grad': c.jit_of_grad,
            'perf_cpu_baseline': c.op.perf_cpu_baseline,
            'perf_cpu_ratio': perf_data.get((short, 'cpu')),
            'perf_gpu_baseline': c.op.perf_gpu_baseline,
            'perf_gpu_ratio': perf_data.get((short, 'gpu')),
            'invariants': list(c.op.invariants),
            'notes': c.op.notes,
        })
    return json.dumps(out, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    docs_dir = Path(__file__).parent.parent / 'docs'
    docs_dir.mkdir(exist_ok=True)

    print(f'Running op-matrix probes ({len(CATALOGUE)} ops)...')
    cells = []
    for op in CATALOGUE:
        try:
            cells.append(run_probes(op))
        except Exception as e:  # noqa: BLE001
            print(f'  catastrophic fail on {op.qualname}: {type(e).__name__}')
            cells.append(CellResult(
                op=op,
                jit=f'fixture failed: {type(e).__name__}',
                grad='not-run',
                vmap='not-run',
                jit_of_grad='not-run',
                perf_ratio=None,
            ))

    host = host_snapshot()
    perf_data = load_perf_data()

    md = render_markdown(cells, host, perf_data)
    js = render_json(cells, host, perf_data)

    (docs_dir / 'op_matrix.md').write_text(md)
    (docs_dir / 'op_matrix.json').write_text(js)
    print(f'wrote docs/op_matrix.md ({len(cells)} ops)')
    print(f'wrote docs/op_matrix.json')

    n_pass = sum(1 for c in cells if c.jit == 'pass')
    n_grad_app = sum(1 for c in cells if c.op.diff_arg is not None)
    n_grad_pass = sum(1 for c in cells if c.grad == 'pass')
    print(
        f'\nSummary: jit {n_pass}/{len(cells)}, '
        f'grad {n_grad_pass}/{n_grad_app}'
    )


if __name__ == '__main__':
    main()
