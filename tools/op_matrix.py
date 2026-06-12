# -*- coding: utf-8 -*-
"""
Op-matrix generator: catalogue every public nitrix op + live probe
its transformation support and declared invariants.  Output:
``docs/op_matrix.md`` with provenance metadata.

The "**turn the matrix green**" goal: every public op should pass
the standard probes (jit, grad, vmap, jit-of-grad).

This matrix is **capability-only**.  Performance lives in the sibling
**nitrix-perf-bench** suite (cross-framework, multi-platform, history,
fidelity-gated) and its hosted dashboard -- richer than any matrix cell, so
the perf columns + the ``bench/`` scrapers were retired from here.  Capability
(the jit/grad/vmap probes + invariants) is intrinsic to the op and lets this
matrix regenerate standalone.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Probe primitives
# ---------------------------------------------------------------------------


def _safe_call(fn, *args, **kwargs):
    """Run ``fn(*args, **kwargs)`` and return (out, err).

    Errors are caught and reported as short strings so the matrix
    runner doesn't bail on a single failing op.
    """
    try:
        out = fn(*args, **kwargs)
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
        return out, None
    except Exception as e:  # noqa: BLE001
        msg = f'{type(e).__name__}: {str(e)[:120]}'
        return None, msg


def probe_jit(fn, args, kwargs):
    """Does this op survive ``jax.jit``?"""
    jitted = jax.jit(lambda *a: fn(*a, **kwargs))
    _, err = _safe_call(jitted, *args)
    return 'pass' if err is None else err


def probe_vmap(fn, args, kwargs, vmap_arg: int):
    """Does this op survive ``jax.vmap`` over leading axis of arg
    ``vmap_arg``?"""
    if vmap_arg is None:
        return 'n/a'
    new_args = list(args)
    # Add a leading singleton axis to the target arg
    target = new_args[vmap_arg]
    new_args[vmap_arg] = target[None]
    in_axes = [None] * len(new_args)
    in_axes[vmap_arg] = 0
    vmapped = jax.vmap(
        lambda *a: fn(*a, **kwargs),
        in_axes=tuple(in_axes),
    )
    _, err = _safe_call(vmapped, *new_args)
    return 'pass' if err is None else err


def probe_grad(fn, args, kwargs, diff_arg: int, reducer=None):
    """Does ``jax.grad`` over a scalar reduction of ``fn`` work?"""
    if diff_arg is None:
        return 'n/a'
    if reducer is None:

        def reducer(x):
            return jnp.sum(x**2)

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
    """The double-transform case ``jit(grad(...))``."""
    if diff_arg is None:
        return 'n/a'
    if reducer is None:

        def reducer(x):
            return jnp.sum(x**2)

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
    """Hand-curated metadata for a single public op.

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
    notes
        Free-text annotation (rendered in the table's "notes"
        column).
    reducer
        Optional custom scalar reducer for the grad probe.  Default
        ``sum(x**2)``; override when the natural reduction differs.
    """

    qualname: str
    fixture: Callable[[], tuple]
    diff_arg: Optional[int] = 0
    vmap_arg: Optional[int] = 0
    invariants: tuple[str, ...] = ()
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
        """Return the callable used by the probes.

        If ``fn_override`` is set, returns it directly.  Otherwise
        imports by walking ``qualname``: imports the longest *strict
        prefix* (excluding the symbol itself) via ``importlib``,
        then ``getattr``s the final part.  This avoids the
        pathological case where ``nitrix.signal.tsconv`` could
        resolve to either the ``tsconv`` module or the ``tsconv``
        function inside it -- we always want the latter (modules
        aren't callable).
        """
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

register(
    OpInfo(
        'nitrix.linalg.symmetric',
        fixture=lambda: ((jax.random.normal(_key(), (4, 4)),), {}),
        invariants=('idempotent on symmetric input',),
    )
)
register(
    OpInfo(
        'nitrix.linalg.sym2vec',
        fixture=lambda: ((jax.random.normal(_key(), (5, 5)),), {}),
        invariants=('strict upper triangle only', 'custom_vjp'),
        notes='returns vec of strict upper triangle',
    )
)
register(
    OpInfo(
        'nitrix.linalg.vec2sym',
        fixture=lambda: ((jax.random.normal(_key(), (10,)),), {}),
        invariants=('mirrors upper triangle', 'custom_vjp'),
    )
)
register(
    OpInfo(
        'nitrix.linalg.toeplitz_2d',
        fixture=lambda: ((jnp.arange(5.0), jnp.arange(5.0)), {}),
        invariants=('vmap-over-roll recipe',),
    )
)
register(
    OpInfo(
        'nitrix.linalg.recondition_eigenspaces',
        fixture=lambda: (
            (
                jax.random.normal(_key(), (5, 5))
                @ jax.random.normal(_key(), (5, 5)).T,
            ),
            {'psi': 0.1},
        ),
        invariants=('PSD-preserving',),
    )
)
register(
    OpInfo(
        'nitrix.linalg.residualise',
        fixture=lambda: (
            (
                jax.random.normal(_key(0), (10, 200)),
                jax.random.normal(_key(1), (3, 200)),
            ),
            {'method': 'cholesky'},
        ),
        invariants=('Cholesky-normal-equations',),
        notes='Cholesky path; ~800x faster than numpy lstsq at V=100k',
    )
)
register(
    OpInfo(
        'nitrix.linalg.linear_kernel',
        fixture=lambda: ((jax.random.normal(_key(), (50, 16)),), {}),
        invariants=('shared with linear_distance via identity formula',),
    )
)
register(
    OpInfo(
        'nitrix.linalg.linear_distance',
        fixture=lambda: ((jax.random.normal(_key(), (50, 16)),), {}),
        invariants=(
            '|x-y|^2 = |x|^2 + |y|^2 - 2 x.y identity (O(nm) memory)',
        ),
        notes='1000x memory reduction vs naive at d=1000',
    )
)
register(
    OpInfo(
        'nitrix.linalg.rbf_kernel',
        fixture=lambda: (
            (jax.random.normal(_key(), (50, 16)),),
            {'gamma': 0.5},
        ),
        invariants=('exp(-gamma * |x-y|^2)',),
        notes='~375x faster than sklearn at (5000, 32)',
    )
)
register(
    OpInfo(
        'nitrix.linalg.symlog',
        fixture=lambda: (
            (
                jax.random.normal(_key(), (5, 5))
                @ jax.random.normal(_key(), (5, 5)).T
                + 0.5 * jnp.eye(5),
            ),
            {},
        ),
        invariants=('SPEC 4.1 stability rewrite', 'eigvalue-clip threshold'),
        notes='eigh-based; routes through safe_eigh cuSolver fallback',
    )
)
register(
    OpInfo(
        'nitrix.linalg.symsqrt',
        fixture=lambda: (
            (
                jax.random.normal(_key(), (5, 5))
                @ jax.random.normal(_key(), (5, 5)).T
                + 0.5 * jnp.eye(5),
            ),
            {},
        ),
        invariants=('eigvalue-clip threshold',),
    )
)
register(
    OpInfo(
        'nitrix.linalg.sympower',
        fixture=lambda: (
            (
                jax.random.normal(_key(), (5, 5))
                @ jax.random.normal(_key(), (5, 5)).T
                + 0.5 * jnp.eye(5),
            ),
            {'power': -0.5},
        ),
        invariants=('arbitrary real power via eigh',),
    )
)
register(
    OpInfo(
        'nitrix.linalg.mean_log_euclidean',
        fixture=lambda: (
            (
                jnp.stack(
                    [
                        jax.random.normal(_key(i), (4, 4))
                        @ jax.random.normal(_key(i), (4, 4)).T
                        + 0.5 * jnp.eye(4)
                        for i in range(3)
                    ]
                ),
            ),
            {},
        ),
        invariants=('closed-form Frechet mean on log-Euclidean metric',),
        vmap_arg=None,  # already batched
    )
)

# --- stats ------------------------------------------------------------------

register(
    OpInfo(
        'nitrix.stats.cov',
        fixture=lambda: ((jax.random.normal(_key(), (5, 100)),), {}),
        invariants=('complex-Hermitian preserved', 'np.cov parity at fp64'),
        notes='130x faster than numpy at (2000, 1000)',
    )
)
register(
    OpInfo(
        'nitrix.stats.corr',
        fixture=lambda: ((jax.random.normal(_key(), (5, 100)),), {}),
        invariants=('diagonal=1', 'complex-Hermitian preserved'),
    )
)
register(
    OpInfo(
        'nitrix.stats.partialcov',
        fixture=lambda: ((jax.random.normal(_key(), (5, 100)),), {}),
        invariants=('precision-matrix-derived',),
    )
)
register(
    OpInfo(
        'nitrix.stats.precision',
        fixture=lambda: ((jax.random.normal(_key(), (5, 100)),), {}),
        invariants=('inverse of cov',),
    )
)
register(
    OpInfo(
        'nitrix.stats.analytic_signal',
        fixture=lambda: ((jax.random.normal(_key(), (200,)),), {}),
        invariants=('vectorised Hilbert mask', 'scipy.signal.hilbert parity'),
        # Complex output; reduce via |.|^2 so grad sees a real scalar.
        reducer=lambda x: jnp.sum(jnp.abs(x) ** 2),
    )
)
register(
    OpInfo(
        'nitrix.stats.hilbert_transform',
        fixture=lambda: ((jax.random.normal(_key(), (200,)),), {}),
        invariants=('imag part of analytic_signal',),
    )
)
register(
    OpInfo(
        'nitrix.stats.envelope',
        fixture=lambda: ((jax.random.normal(_key(), (200,)),), {}),
        invariants=('|analytic_signal|',),
    )
)


def _lme_fixture():
    rng = np.random.default_rng(0)
    g, n_per = 4, 20
    N = g * n_per
    Y = jnp.asarray(rng.standard_normal((8, N)).astype(np.float32))
    X = jnp.asarray(rng.standard_normal((N, 2)).astype(np.float32))
    Z = jnp.zeros((N, g), dtype=jnp.float32)
    for i in range(g):
        Z = Z.at[i * n_per : (i + 1) * n_per, i].set(1.0)
    return (Y, X, Z), {'n_iter': 10}


register(
    OpInfo(
        'nitrix.stats.lme.reml_fit',
        fixture=_lme_fixture,
        diff_arg=0,
        vmap_arg=None,  # already voxelwise
        invariants=('FaST-LMM spectral rotation', 'no V*N^2 intermediate'),
        notes='~5e-3 parity with statsmodels.MixedLM',
    )
)


def _flame_fixture():
    rng = np.random.default_rng(0)
    N, p, V = 30, 2, 16
    beta = jnp.asarray(rng.standard_normal((V, N)).astype(np.float32))
    var_within = jnp.asarray(
        (np.abs(rng.standard_normal((V, N))) + 0.1).astype(np.float32),
    )
    X_group = jnp.asarray(rng.standard_normal((N, p)).astype(np.float32))
    return (beta, var_within, X_group), {'n_iter': 10}


register(
    OpInfo(
        'nitrix.stats.lme.flame_two_level',
        fixture=_flame_fixture,
        diff_arg=0,
        vmap_arg=None,
        invariants=(
            'single-parameter REML (identifiability)',
            'shared X_group',
        ),
    )
)

# --- signal -----------------------------------------------------------------


def _interp_fixture():
    rng = np.random.default_rng(0)
    data = jnp.asarray(rng.standard_normal(50).astype(np.float32))
    mask = jnp.asarray((rng.random(50) > 0.3))
    return (data, mask), {}


register(
    OpInfo(
        'nitrix.signal.linear_interpolate',
        fixture=_interp_fixture,
        diff_arg=0,
        vmap_arg=None,
        invariants=('associative_scan (O(log n) parallel)', 'no while_loop'),
    )
)


def _ls_fixture():
    rng = np.random.default_rng(0)
    data = jnp.asarray(rng.standard_normal(100).astype(np.float32))
    mask = jnp.asarray((rng.random(100) > 0.2))
    return (data, mask), {'dt': 1.0}


register(
    OpInfo(
        'nitrix.signal.lomb_scargle_interpolate',
        fixture=_ls_fixture,
        diff_arg=0,
        vmap_arg=None,
        invariants=(
            'joint-GLM (passes through observed)',
            'shared-Gram across voxels',
            'no boundary discontinuity',
        ),
    )
)
register(
    OpInfo(
        'nitrix.signal.lomb_scargle_periodogram',
        fixture=_ls_fixture,
        diff_arg=0,
        vmap_arg=None,
        invariants=('Scargle 1982 normalisation', 'scipy.lombscargle parity'),
    )
)
register(
    OpInfo(
        'nitrix.signal.polynomial_detrend',
        fixture=lambda: (
            (jax.random.normal(_key(), (5, 100)),),
            {'degree': 2},
        ),
        invariants=(
            'rescaled Vandermonde (stability)',
            'routes through residualise',
        ),
    )
)


def _tsconv_fixture():
    X = jax.random.normal(_key(0), (4, 100))
    w = jax.random.normal(_key(1), (4, 4, 5))
    return (X, w), {}


register(
    OpInfo(
        'nitrix.signal.tsconv',
        fixture=_tsconv_fixture,
        diff_arg=0,
        vmap_arg=None,
        invariants=('thin lax.conv_general_dilated wrapper',),
    )
)

# --- numerics ---------------------------------------------------------------

register(
    OpInfo(
        'nitrix.numerics.zscore_normalize',
        fixture=lambda: ((jax.random.normal(_key(), (5, 100)),), {}),
        invariants=('zero mean, unit std per axis',),
    )
)
register(
    OpInfo(
        'nitrix.numerics.intensity_normalize',
        fixture=lambda: ((jax.random.normal(_key(), (1000,)) * 10 + 50,), {}),
        invariants=('percentile-clip to [0, 1]',),
    )
)
register(
    OpInfo(
        'nitrix.numerics.complex_decompose',
        fixture=lambda: (
            (
                jax.random.normal(_key(0), (20,))
                + 1j * jax.random.normal(_key(1), (20,)),
            ),
            {},
        ),
        invariants=('amplitude / phase split',),
    )
)

# --- geometry ---------------------------------------------------------------

register(
    OpInfo(
        'nitrix.geometry.identity_grid',
        fixture=lambda: (((4, 4),), {}),
        diff_arg=None,
        vmap_arg=None,
        skip_jit=True,
        invariants=('shape-static constructor',),
        notes='shape-tuple input; not a runtime op',
    )
)


def _spatial_transform_fixture():
    image = jax.random.normal(_key(0), (4, 4, 2))
    deform = jax.random.normal(_key(1), (4, 4, 2)) * 0.5
    return (image, deform), {'mode': 'nearest'}


register(
    OpInfo(
        'nitrix.geometry.spatial_transform',
        fixture=_spatial_transform_fixture,
        invariants=(
            'dispatches over method= (Linear default); mode pass-through',
            'accepts leading batch',
        ),
    )
)


def _spatial_transform_lanczos():
    from nitrix.geometry import Lanczos, spatial_transform

    def fn(image, deformation):
        return spatial_transform(
            image,
            deformation,
            method=Lanczos(order=3),
            mode='nearest',
        )

    return fn


register(
    OpInfo(
        'nitrix.geometry.spatial_transform[lanczos]',
        fixture=lambda: (
            (
                jax.random.normal(_key(0), (6, 6, 2)),
                jax.random.normal(_key(1), (6, 6, 2)) * 0.5,
            ),
            {},
        ),
        fn_override=_spatial_transform_lanczos(),
        invariants=('scattered windowed-sinc gather; differentiable warp',),
        notes='the high-fidelity differentiable warp kernel',
    )
)


def _ivf_fixture():
    v = jax.random.normal(_key(), (8, 8, 2)) * 0.1
    return (v,), {'n_steps': 5}


register(
    OpInfo(
        'nitrix.geometry.integrate_velocity_field',
        fixture=_ivf_fixture,
        invariants=(
            'scaling-and-squaring',
            "default mode='nearest' (voxelmorph)",
        ),
    )
)
register(
    OpInfo(
        'nitrix.geometry.jacobian_det_displacement',
        fixture=lambda: ((jax.random.normal(_key(), (8, 8, 2)) * 0.1,), {}),
        invariants=('explicit det for d<=3', 'no cuSolver call'),
    )
)
register(
    OpInfo(
        'nitrix.geometry.sphere_grid_pad_2d',
        fixture=lambda: ((jax.random.normal(_key(), (4, 4)),), {'pad': 1}),
        invariants=('pole-flip + W/2 roll', 'longitudinal wrap'),
    )
)

# --- graph ------------------------------------------------------------------


def _laplacian_fixture():
    rng = np.random.default_rng(0)
    n = 20
    A = (rng.random((n, n)) > 0.7).astype(np.float32)
    A = (A + A.T).clip(max=1.0) - jnp.diag(jnp.diag(jnp.asarray(A)))
    return (jnp.asarray(A) + jnp.asarray(A).T,), {}


register(
    OpInfo(
        'nitrix.graph.laplacian',
        fixture=_laplacian_fixture,
        invariants=('symmetric / random_walk / combinatorial variants',),
    )
)
register(
    OpInfo(
        'nitrix.graph.degree_vector',
        fixture=_laplacian_fixture,
        invariants=('dense / ELL / SectionedELL dispatch',),
    )
)


def _laplacian_eigenmap_fixture():
    rng = np.random.default_rng(0)
    n = 32
    A = (rng.random((n, n)) > 0.5).astype(np.float32)
    A = ((A + A.T) > 0).astype(np.float32)
    return (jnp.asarray(A),), {'n_components': 3, 'solver': 'eigh'}


register(
    OpInfo(
        'nitrix.graph.laplacian_eigenmap',
        fixture=_laplacian_eigenmap_fixture,
        invariants=(
            'safe_eigh cuSolver fallback',
            'LOBPCG implicit-VJP for sparse paths',
        ),
        vmap_arg=None,
        notes='dense=eigh, sparse=lobpcg; differentiable end-to-end',
    )
)

# --- morphology -------------------------------------------------------------

register(
    OpInfo(
        'nitrix.morphology.dilate',
        fixture=lambda: ((jax.random.normal(_key(), (10, 10)),), {'size': 3}),
        invariants=('TROPICAL_MAX_PLUS specialisation',),
    )
)
register(
    OpInfo(
        'nitrix.morphology.erode',
        fixture=lambda: ((jax.random.normal(_key(), (10, 10)),), {'size': 3}),
        invariants=('TROPICAL_MIN_PLUS specialisation',),
    )
)
register(
    OpInfo(
        'nitrix.morphology.distance_transform',
        fixture=lambda: (
            ((jax.random.normal(_key(), (12, 12)) > 0).astype(jnp.float32),),
            {},
        ),
        invariants=(
            'exact EDT (default): separable per-axis TROPICAL_MIN_PLUS matmul',
            'opt-in chamfer via metric=/structuring_element (TROPICAL_MIN_PLUS)',
        ),
        notes='euclidean default matches cupy EDT @64^3, beats scipy on CPU',
    )
)
register(
    OpInfo(
        'nitrix.morphology.distance_transform_edt',
        fixture=lambda: (
            ((jax.random.normal(_key(), (12, 12)) > 0).astype(jnp.float32),),
            {},
        ),
        invariants=(
            'exact Euclidean DT (separable min-plus matmul on semiring kernel)',
            'alias of distance_transform(metric="euclidean")',
        ),
        notes='scipy distance_transform_edt parity; reverse-mode differentiable',
    )
)
register(
    OpInfo(
        'nitrix.morphology.median_filter',
        fixture=lambda: ((jax.random.normal(_key(), (16, 16)),), {'size': 3}),
        invariants=('gather + nanmedian (not a semiring op)',),
    )
)


def _max_pool_fixture():
    # (B=1, C=2, H=8, W=8); pool 2x2 -> (1, 2, 4, 4) with indices
    x = jax.random.normal(_key(), (1, 2, 8, 8))
    return (x,), {'pool_size': 2, 'spatial_rank': 2}


def _max_pool_reducer(out):
    # max_pool_with_indices_nd returns (pooled, indices).
    # grad probe takes the first leaf via tree_leaves -- which is
    # the pooled tensor.  Reduce that to a scalar.
    return jnp.sum(out**2)


register(
    OpInfo(
        'nitrix.morphology.max_pool_with_indices_nd',
        fixture=_max_pool_fixture,
        diff_arg=0,
        vmap_arg=None,  # batch axis already in fixture; vmap-over-extra-leading is redundant
        invariants=(
            'global flat C-order argmax indices',
            'window-unfold + argmax composition',
        ),
        reducer=_max_pool_reducer,
        notes='returns (pooled, indices); paired with max_unpool_nd for encoder-decoder',
    )
)


def _max_unpool_fixture():
    rng = np.random.default_rng(0)
    pooled = jnp.asarray(rng.standard_normal((1, 2, 4, 4)).astype(np.float32))
    indices = jnp.asarray(rng.integers(0, 64, (1, 2, 4, 4)).astype(np.int32))
    return (pooled, indices), {'output_shape': (8, 8), 'spatial_rank': 2}


register(
    OpInfo(
        'nitrix.morphology.max_unpool_nd',
        fixture=_max_unpool_fixture,
        diff_arg=0,
        vmap_arg=None,
        invariants=(
            'vmapped per-channel scatter',
            'argmax-agreement parity (not raw-logit allclose)',
        ),
        notes='inverts max_pool_with_indices_nd; indices are int (non-diff)',
    )
)

# --- smoothing --------------------------------------------------------------

register(
    OpInfo(
        'nitrix.smoothing.gaussian',
        fixture=lambda: (
            (jax.random.normal(_key(), (16, 16)),),
            {'sigma': 1.5},
        ),
        invariants=('separable n-D', 'scipy.ndimage parity at fp64'),
    )
)


def _bilateral_fixture():
    # v0.4 replaced the diagonal ``sigma_features`` kwarg with a factored
    # ``metric: FeatureMetric`` (DiagonalMetric == the old diagonal case).
    from nitrix.smoothing import DiagonalMetric

    return (
        (
            jax.random.normal(_key(0), (32, 1)),
            jax.random.normal(_key(1), (32, 2)),
        ),
        {
            'metric': DiagonalMetric(jnp.asarray([1.0, 1.0])),
            'neighbourhood': 8,
        },
    )


register(
    OpInfo(
        'nitrix.smoothing.bilateral_gaussian',
        fixture=_bilateral_fixture,
        invariants=(
            'semiring_ell_matmul over feature adjacency',
            'factored FeatureMetric (DiagonalMetric here)',
        ),
    )
)

# --- semiring ---------------------------------------------------------------

register(
    OpInfo(
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
        notes='6-16x faster than JAX fori_loop on REAL/TROPICAL/EUCLIDEAN',
    )
)


def _ell_fixture():
    rng = np.random.default_rng(0)
    n = 20
    k_max = 5
    values = jnp.asarray(rng.standard_normal((n, k_max)).astype(np.float32))
    indices = jnp.asarray(rng.integers(0, n, (n, k_max)).astype(np.int32))
    B = jnp.asarray(rng.standard_normal((n, 4)).astype(np.float32))
    return (values, indices, B), {'n_cols': n, 'backend': 'jax'}


register(
    OpInfo(
        'nitrix.semiring.semiring_ell_matmul',
        fixture=_ell_fixture,
        diff_arg=0,
        vmap_arg=None,
        invariants=('sparse ELL matmul',),
    )
)


def _ell_rmatvec_fixture():
    # Adjoint matvec ``Aᵀ X``: X is indexed by the ELL *row* axis (square
    # operator here, m == n_cols == 20); no ``backend`` arg (JAX-only).
    rng = np.random.default_rng(0)
    n = 20
    k_max = 5
    values = jnp.asarray(rng.standard_normal((n, k_max)).astype(np.float32))
    indices = jnp.asarray(rng.integers(0, n, (n, k_max)).astype(np.int32))
    X = jnp.asarray(rng.standard_normal((n, 4)).astype(np.float32))
    return (values, indices, X), {'n_cols': n}


register(
    OpInfo(
        'nitrix.semiring.semiring_ell_rmatvec',
        fixture=_ell_rmatvec_fixture,
        diff_arg=0,
        vmap_arg=None,
        invariants=('sparse ELL adjoint (transpose) matmul',),
    )
)


def _ell_edge_setup():
    """Build a GCN-style closure that takes ``x`` as its only
    positional arg, baking in edge_fn / ell / semiring.  Returns
    the closure plus a fixture-of-the-closure.
    """
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

register(
    OpInfo(
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
        notes='probed with GCN closure; covers GCN/GAT/EdgeConv/MoNet/ChebNet',
    )
)


def _conv_fixture():
    X = jax.random.normal(_key(0), (1, 8, 8, 1))
    K = jax.random.normal(_key(1), (3, 3, 1, 1))
    return (X, K), {}


register(
    OpInfo(
        'nitrix.semiring.semiring_conv',
        fixture=_conv_fixture,
        invariants=(
            'NaN-safe patch extraction (jnp.take, not lax.conv_general_dilated_patches)',
            'explicit im2col + semiring_matmul',
        ),
        notes='1.7-1.9x slower than cuDNN fp32 (literature expected)',
    )
)

# --- sparse -----------------------------------------------------------------

register(
    OpInfo(
        'nitrix.sparse.ell_from_dense',
        fixture=lambda: ((jax.random.normal(_key(), (10, 10)),), {}),
        diff_arg=None,
        vmap_arg=None,
        skip_jit=True,
        invariants=('host-side ELL construction',),
    )
)
register(
    OpInfo(
        'nitrix.sparse.grid_laplacian',
        fixture=lambda: (((8, 8),), {}),
        diff_arg=None,
        vmap_arg=None,
        skip_jit=True,
        invariants=('regular-grid stencil', 'scipy.ndimage.laplace parity'),
    )
)


def _mesh_fixture():
    from nitrix.sparse import icosphere

    m = icosphere(1)
    return (m,), {}


register(
    OpInfo(
        'nitrix.sparse.mesh_k_ring_adjacency',
        fixture=_mesh_fixture,
        diff_arg=None,
        vmap_arg=None,
        skip_jit=True,
        invariants=('BFS k-ring on triangle mesh', 'host-side construction'),
    )
)


def _mesh_pool_setup():
    """Build a closure-wrapped mesh_pool_max + a fixture that returns
    only the fine-features ``x`` (ELL is baked in).
    """
    from nitrix.sparse import (
        icosphere_cross_level_adjacency,
        icosphere_hierarchy,
        mesh_pool_max,
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

register(
    OpInfo(
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
    )
)


def _mesh_unpool_setup():
    from nitrix.sparse import (
        icosphere_bary_upsampler,
        icosphere_hierarchy,
        mesh_unpool_max,
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

register(
    OpInfo(
        'nitrix.sparse.mesh_unpool_max',
        fixture=_unpool_fixture,
        fn_override=_unpool_op,
        diff_arg=0,
        vmap_arg=0,
        invariants=('TROPICAL_MAX_PLUS symmetric of mesh_pool_max',),
    )
)


def _mesh_bary_setup():
    from nitrix.sparse import (
        icosphere_bary_upsampler,
        icosphere_hierarchy,
        mesh_bary_upsample,
    )

    rng = np.random.default_rng(0)
    h = icosphere_hierarchy(max_level=1)
    bary = icosphere_bary_upsampler(h, 0, 1)
    n_coarse = h.meshes[0].n_vertices

    def op(coarse):
        return mesh_bary_upsample(bary, coarse)

    def fixture():
        coarse = jnp.asarray(
            rng.standard_normal((n_coarse, 3)).astype(np.float32)
        )
        return (coarse,), {}

    return op, fixture


_bary_op, _bary_fixture = _mesh_bary_setup()

register(
    OpInfo(
        'nitrix.sparse.mesh_bary_upsample',
        fixture=_bary_fixture,
        fn_override=_bary_op,
        diff_arg=0,
        vmap_arg=0,
        invariants=('REAL semiring_ell_matmul (weighted sum)',),
    )
)


# Icosphere hierarchy construction (Sprint B) -- host-side, skip_jit.

register(
    OpInfo(
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
    )
)


def _hier_pair_fixture():
    from nitrix.sparse import icosphere_hierarchy

    h = icosphere_hierarchy(max_level=1)
    return (h, 0, 1), {}


register(
    OpInfo(
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
    )
)


register(
    OpInfo(
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
    )
)


# ===========================================================================
# Inventory-completeness pass (2026-06-02).  Catalogue the remaining public
# ops so the matrix matches the surface.  Guarded by
# tests/test_op_matrix_completeness.py (which also holds the EXCLUDE allowlist
# for the ops deliberately left out -- aliases, reference impls, matvec
# closures, shape helpers, metric constructors).
# ===========================================================================


def _spd(d: int = 5, seed: int = 0):
    """A small SPD matrix fixture input."""
    a = jax.random.normal(_key(seed), (d, d))
    return a @ a.T + 0.5 * jnp.eye(d)


def _sym_adj(n: int = 20, seed: int = 0):
    """A symmetric binary adjacency (zero diagonal)."""
    rng = np.random.default_rng(seed)
    a = (rng.random((n, n)) > 0.6).astype(np.float32)
    a = np.triu(a, 1)
    return jnp.asarray(a + a.T)


# --- linalg (kernels, sym* family, structural) ------------------------------

register(
    OpInfo(
        'nitrix.linalg.symexp',
        fixture=lambda: ((_spd(5),), {}),
        invariants=('matrix exponential via eigh',),
    )
)
register(
    OpInfo(
        'nitrix.linalg.symmap',
        fixture=lambda: ((_spd(5),), {'fn': jnp.tanh}),
        invariants=('apply a scalar fn to eigenvalues',),
        notes='probed with fn=tanh; eigh-based, routes through safe_eigh',
    )
)
register(
    OpInfo(
        'nitrix.linalg.tangent_project_spd',
        fixture=lambda: ((_spd(5, 0), _spd(5, 1)), {}),
        diff_arg=0,
        vmap_arg=0,
        invariants=(
            'affine-invariant log map at reference (not log-Euclidean)',
        ),
        notes='log(R^-1/2 X R^-1/2); composes sympower(-1/2)+symlog',
    )
)
register(
    OpInfo(
        'nitrix.linalg.cone_project_spd',
        fixture=lambda: ((_spd(5, 0), _spd(5, 1)), {}),
        diff_arg=0,
        vmap_arg=0,
        invariants=('projection onto the SPD cone at a reference',),
    )
)
register(
    OpInfo(
        'nitrix.linalg.mean_euclidean',
        fixture=lambda: ((jnp.stack([_spd(4, i) for i in range(3)]),), {}),
        vmap_arg=None,
        invariants=('Euclidean (arithmetic) mean over an SPD batch',),
    )
)
register(
    OpInfo(
        'nitrix.linalg.squareform',
        fixture=lambda: ((jnp.arange(1.0, 7.0),), {}),
        invariants=('square <-> condensed conversion',),
        notes='probed vec->square (jit-safe); the square->vec direction branches '
        'on a jnp.allclose symmetry check (not jit-safe -- use sym2vec)',
    )
)
register(
    OpInfo(
        'nitrix.linalg.toeplitz',
        fixture=lambda: ((jnp.arange(1.0, 6.0),), {}),
        invariants=('Toeplitz matrix from its first column',),
    )
)
register(
    OpInfo(
        'nitrix.linalg.delete_diagonal',
        fixture=lambda: ((jax.random.normal(_key(), (5, 5)),), {}),
        invariants=('zero the diagonal',),
    )
)
register(
    OpInfo(
        'nitrix.linalg.fill_diagonal',
        fixture=lambda: ((jax.random.normal(_key(), (5, 5)),), {'fill': 1.0}),
        invariants=('set the diagonal to a constant',),
    )
)
register(
    OpInfo(
        'nitrix.linalg.gaussian_kernel',
        fixture=lambda: ((jax.random.normal(_key(), (40, 12)),), {}),
        invariants=('exp(-|x-y|^2) Gaussian / RBF family',),
    )
)
register(
    OpInfo(
        'nitrix.linalg.polynomial_kernel',
        fixture=lambda: ((jax.random.normal(_key(), (40, 12)),), {'order': 3}),
        invariants=('(gamma <x,y> + r)^order',),
    )
)
register(
    OpInfo(
        'nitrix.linalg.sigmoid_kernel',
        fixture=lambda: ((jax.random.normal(_key(), (40, 12)),), {}),
        invariants=('tanh(gamma <x,y> + r)',),
    )
)
register(
    OpInfo(
        'nitrix.linalg.cosine_kernel',
        fixture=lambda: ((jax.random.normal(_key(), (40, 12)),), {}),
        invariants=('normalised inner-product kernel',),
    )
)
register(
    OpInfo(
        'nitrix.linalg.parameterised_norm',
        fixture=lambda: ((jax.random.normal(_key(), (40, 12)),), {}),
        invariants=('theta-weighted row norm',),
    )
)

# --- stats (coefficients + DSP) ---------------------------------------------


def _xy_fixture():
    return (
        (
            jax.random.normal(_key(0), (5, 100)),
            jax.random.normal(_key(1), (3, 100)),
        ),
        {},
    )


register(
    OpInfo(
        'nitrix.stats.partialcorr',
        fixture=lambda: ((jax.random.normal(_key(), (5, 100)),), {}),
        invariants=('partial correlation (precision-normalised)',),
    )
)
register(
    OpInfo(
        'nitrix.stats.conditionalcov',
        fixture=_xy_fixture,
        diff_arg=0,
        vmap_arg=0,
        invariants=('covariance of X conditioned on Y',),
    )
)
register(
    OpInfo(
        'nitrix.stats.conditionalcorr',
        fixture=_xy_fixture,
        diff_arg=0,
        vmap_arg=0,
        invariants=('correlation of X conditioned on Y',),
    )
)
register(
    OpInfo(
        'nitrix.stats.pairedcov',
        fixture=_xy_fixture,
        diff_arg=0,
        vmap_arg=0,
        invariants=('cross-covariance between X and Y',),
    )
)
register(
    OpInfo(
        'nitrix.stats.pairedcorr',
        fixture=_xy_fixture,
        diff_arg=0,
        vmap_arg=0,
        invariants=('cross-correlation between X and Y',),
    )
)
register(
    OpInfo(
        'nitrix.stats.product_filter',
        fixture=lambda: (
            (jax.random.normal(_key(0), (4, 100)), jnp.ones(51)),
            {},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('frequency-domain product filter (rfft multiply)',),
    )
)
register(
    OpInfo(
        'nitrix.stats.product_filtfilt',
        fixture=lambda: (
            (jax.random.normal(_key(0), (4, 100)), jnp.ones(51)),
            {},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('zero-phase product_filter (forward + reverse)',),
    )
)
register(
    OpInfo(
        'nitrix.stats.instantaneous_phase',
        fixture=lambda: ((jax.random.normal(_key(), (200,)),), {}),
        invariants=('phase of the analytic signal',),
    )
)
register(
    OpInfo(
        'nitrix.stats.instantaneous_frequency',
        fixture=lambda: ((jax.random.normal(_key(), (200,)),), {}),
        invariants=('time-derivative of instantaneous phase',),
    )
)
register(
    OpInfo(
        'nitrix.stats.env_inst',
        fixture=lambda: ((jax.random.normal(_key(), (200,)),), {}),
        invariants=('envelope + instantaneous phase/frequency (Hilbert)',),
        notes='returns a 3-tuple; grad probe reduces the first leaf',
    )
)

# --- signal (IIR family + windowing) ----------------------------------------


def _sig_x():
    return jax.random.normal(_key(), (4, 100))


_IDENTITY_SOS = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])

register(
    OpInfo(
        'nitrix.signal.lowpass',
        fixture=lambda: ((_sig_x(),), {'cutoff': 0.2}),
        diff_arg=0,
        vmap_arg=0,
        invariants=('maxflat IIR low-pass',),
    )
)
register(
    OpInfo(
        'nitrix.signal.highpass',
        fixture=lambda: ((_sig_x(),), {'cutoff': 0.2}),
        diff_arg=0,
        vmap_arg=0,
        invariants=('maxflat IIR high-pass',),
    )
)
register(
    OpInfo(
        'nitrix.signal.bandpass',
        fixture=lambda: ((_sig_x(),), {'lo': 0.1, 'hi': 0.3}),
        diff_arg=0,
        vmap_arg=0,
        invariants=('maxflat IIR band-pass',),
    )
)
register(
    OpInfo(
        'nitrix.signal.bandstop',
        fixture=lambda: ((_sig_x(),), {'lo': 0.1, 'hi': 0.3}),
        diff_arg=0,
        vmap_arg=0,
        invariants=('maxflat IIR band-stop',),
    )
)
register(
    OpInfo(
        'nitrix.signal.iir_filter',
        fixture=lambda: ((_sig_x(),), {'btype': 'lowpass', 'hi': 0.2}),
        diff_arg=0,
        vmap_arg=0,
        invariants=('generic IIR (design + apply)',),
    )
)
register(
    OpInfo(
        'nitrix.signal.sosfilt',
        fixture=lambda: ((_sig_x(),), {'sos': _IDENTITY_SOS}),
        diff_arg=0,
        vmap_arg=0,
        invariants=('second-order-sections IIR (causal)',),
    )
)
register(
    OpInfo(
        'nitrix.signal.sosfiltfilt',
        fixture=lambda: ((_sig_x(),), {'sos': _IDENTITY_SOS}),
        diff_arg=0,
        vmap_arg=0,
        invariants=('zero-phase second-order-sections IIR',),
    )
)
register(
    OpInfo(
        'nitrix.signal.butterworth_sos',
        fixture=lambda: ((), {'order': 2, 'btype': 'lowpass', 'hi': 0.2}),
        diff_arg=None,
        vmap_arg=None,
        skip_jit=True,
        invariants=('host-side Butterworth SOS design (returns np.ndarray)',),
    )
)
register(
    OpInfo(
        'nitrix.signal.sample_windows',
        fixture=lambda: (
            (jax.random.normal(_key(), (8, 64)),),
            {'window_size': 16, 'key': _key(0)},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('random fixed-size windowed sampling',),
    )
)

# --- numerics (normalize tail + complex recompose) --------------------------

register(
    OpInfo(
        'nitrix.numerics.complex_recompose',
        fixture=lambda: (
            (
                jax.random.normal(_key(0), (20,)),
                jax.random.normal(_key(1), (20,)),
            ),
            {},
        ),
        diff_arg=0,
        vmap_arg=0,
        invariants=('ampl, phase -> complex (inverse of complex_decompose)',),
        reducer=lambda x: jnp.sum(jnp.abs(x) ** 2),
    )
)
register(
    OpInfo(
        'nitrix.numerics.demean',
        fixture=lambda: ((jax.random.normal(_key(), (5, 100)),), {}),
        invariants=('subtract the mean along an axis',),
    )
)
register(
    OpInfo(
        'nitrix.numerics.percentile_rescale',
        fixture=lambda: ((jax.random.normal(_key(), (1000,)) * 10 + 50,), {}),
        invariants=('shift by p_lo, scale by p_hi, clip to [0, 1]',),
    )
)
register(
    OpInfo(
        'nitrix.numerics.psc_normalize',
        fixture=lambda: (
            (jnp.abs(jax.random.normal(_key(), (5, 100))) + 5,),
            {},
        ),
        invariants=('percent signal change vs the mean',),
    )
)
register(
    OpInfo(
        'nitrix.numerics.robust_zscore_normalize',
        fixture=lambda: ((jax.random.normal(_key(), (5, 100)),), {}),
        invariants=('median / MAD robust z-score',),
    )
)

# --- geometry (resample, sphere, coords) ------------------------------------

register(
    OpInfo(
        'nitrix.geometry.resample',
        fixture=lambda: (
            (jax.random.normal(_key(), (8, 8, 2)),),
            {'target_shape': (4, 4)},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=(
            'align-corners resize; dispatches over method= (Linear default)',
        ),
    )
)


# resample interpolation-method variants.  Each bakes a ``method=``
# record via ``fn_override`` so the probes see an array-only signature.
def _resampler(method):
    from nitrix.geometry import resample as _resample

    def fn(image, target_shape):
        return _resample(image, target_shape, method=method)

    return fn


_RESAMPLE_VARIANT_FIXTURE = lambda: (  # noqa: E731
    (jax.random.normal(_key(), (8, 8, 2)),),
    {'target_shape': (5, 5)},
)


def _resample_method_variants():
    from nitrix.geometry import (
        CubicBSpline,
        Lanczos,
        MultiLabel,
        NearestNeighbour,
    )

    register(
        OpInfo(
            'nitrix.geometry.resample[nearest]',
            fixture=_RESAMPLE_VARIANT_FIXTURE,
            fn_override=_resampler(NearestNeighbour()),
            diff_arg=0,
            vmap_arg=None,
            invariants=(
                'order-0 nearest; explicit gather (GPU) / map_coordinates (CPU)',
            ),
            notes='value-differentiable; coordinate-flat (grad wrt coords ~= 0)',
        )
    )
    register(
        OpInfo(
            'nitrix.geometry.resample[lanczos]',
            fixture=_RESAMPLE_VARIANT_FIXTURE,
            fn_override=_resampler(Lanczos(order=3)),
            diff_arg=0,
            vmap_arg=None,
            invariants=(
                'windowed-sinc order 3; separable 1-D passes; partition of unity',
            ),
            notes='differentiable in values and coordinates',
        )
    )
    register(
        OpInfo(
            'nitrix.geometry.resample[cubic]',
            fixture=_RESAMPLE_VARIANT_FIXTURE,
            fn_override=_resampler(CubicBSpline()),
            diff_arg=0,
            vmap_arg=None,
            invariants=(
                'order-3 B-spline; recursive prefilter (scan/assoc-scan) + 4-tap gather',
            ),
            notes='bit-exact scipy order=3 (mode=mirror); differentiable',
        )
    )
    register(
        OpInfo(
            'nitrix.geometry.resample[multilabel]',
            fixture=lambda: (
                (
                    jax.random.randint(_key(0), (8, 8, 1), 0, 4).astype(
                        jnp.float32
                    ),
                ),
                {'target_shape': (5, 5)},
            ),
            fn_override=_resampler(MultiLabel(labels=(0, 1, 2, 3))),
            diff_arg=0,
            vmap_arg=None,
            invariants=(
                'per-label arg-max over a static label set; output in label set',
            ),
            notes='non-differentiable (hard arg-max); grad runs but is zero',
        )
    )


_resample_method_variants()
register(
    OpInfo(
        'nitrix.geometry.spherical_geodesic_distance',
        fixture=lambda: ((jax.random.normal(_key(), (8, 3)),), {}),
        invariants=('great-circle (geodesic) distance on the sphere',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.jacobian_displacement',
        fixture=lambda: ((jax.random.normal(_key(), (8, 8, 2)) * 0.1,), {}),
        invariants=('central-difference Jacobian of a displacement field',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.center_of_mass_grid',
        fixture=lambda: ((jnp.abs(jax.random.normal(_key(), (8, 8))),), {}),
        invariants=('weighted centre of mass on a regular grid',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.cartesian_to_latlong',
        fixture=lambda: ((jax.random.normal(_key(), (10, 3)),), {}),
        invariants=('Cartesian -> (lat, long)',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.latlong_to_cartesian',
        fixture=lambda: ((jax.random.normal(_key(), (10, 2)) * 0.5,), {}),
        invariants=('(lat, long) -> Cartesian on the sphere',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.center_of_mass_points',
        fixture=lambda: (
            (
                jnp.abs(jax.random.normal(_key(0), (2, 10))),
                jax.random.normal(_key(1), (10, 3)),
            ),
            {},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('weighted centre of mass over a point cloud',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.compactness_penalty',
        fixture=lambda: (
            (
                jnp.abs(jax.random.normal(_key(0), (2, 10))),
                jax.random.normal(_key(1), (10, 3)),
            ),
            {},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('spatial dispersion penalty around the CoM',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.displacement_from_reference_grid',
        fixture=lambda: (
            (
                jnp.abs(jax.random.normal(_key(0), (8, 8))),
                jnp.asarray([3.5, 3.5]),
            ),
            {},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('CoM displacement from a reference (grid)',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.displacement_from_reference_points',
        fixture=lambda: (
            (
                jnp.abs(jax.random.normal(_key(0), (2, 10))),
                jax.random.normal(_key(1), (2, 3)),
                jax.random.normal(_key(2), (10, 3)),
            ),
            {},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('CoM displacement from a reference (points)',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.spherical_conv',
        fixture=lambda: (
            (
                jax.random.normal(_key(0), (12, 2)),
                jax.random.normal(_key(1), (12, 3)),
            ),
            {'sigma': 1.0, 'neighbourhood': 6},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('geodesic-neighbourhood conv via semiring_ell_matmul',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.sphere_grid_unpad_2d',
        fixture=lambda: ((jax.random.normal(_key(), (6, 6)),), {'pad': 1}),
        diff_arg=0,
        vmap_arg=None,
        invariants=('inverse of sphere_grid_pad_2d',),
    )
)

# --- graph (modularity family + diffusion) ----------------------------------

register(
    OpInfo(
        'nitrix.graph.girvan_newman_null',
        fixture=lambda: ((_sym_adj(20),), {}),
        invariants=('degree-product (configuration) null model',),
    )
)
register(
    OpInfo(
        'nitrix.graph.modularity_matrix',
        fixture=lambda: ((_sym_adj(20),), {}),
        invariants=('B = A - gamma * null',),
    )
)
register(
    OpInfo(
        'nitrix.graph.coaffiliation',
        fixture=lambda: ((jax.random.normal(_key(), (20, 4)),), {}),
        invariants=('community co-affiliation C C^T',),
    )
)
register(
    OpInfo(
        'nitrix.graph.relaxed_modularity',
        fixture=lambda: (
            (_sym_adj(20), jax.random.normal(_key(9), (20, 4))),
            {},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('relaxed (continuous) modularity objective',),
    )
)
register(
    OpInfo(
        'nitrix.graph.diffusion_embedding',
        fixture=lambda: (
            (_sym_adj(32),),
            {'n_components': 3, 'solver': 'eigh'},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('diffusion-map embedding', 'safe_eigh fallback'),
        notes='returns (vectors, values); grad probe reduces the first leaf',
    )
)

# --- morphology (opening / closing) -----------------------------------------

register(
    OpInfo(
        'nitrix.morphology.open',
        fixture=lambda: ((jax.random.normal(_key(), (10, 10)),), {'size': 3}),
        invariants=('erode then dilate (TROPICAL opening)',),
    )
)
register(
    OpInfo(
        'nitrix.morphology.close',
        fixture=lambda: ((jax.random.normal(_key(), (10, 10)),), {'size': 3}),
        invariants=('dilate then erode (TROPICAL closing)',),
    )
)

# --- smoothing (susan + knn + neighbourhood) --------------------------------

register(
    OpInfo(
        'nitrix.smoothing.susan_emulator',
        fixture=lambda: (
            (jax.random.normal(_key(), (16, 16)),),
            {'sigma_space': 1.5, 'sigma_intensity': 0.5},
        ),
        invariants=('bilateral_gaussian + median composition',),
    )
)
register(
    OpInfo(
        'nitrix.smoothing.brute_force_knn',
        fixture=lambda: ((jax.random.normal(_key(), (32, 3)),), {'k': 8}),
        diff_arg=None,
        vmap_arg=0,
        invariants=('brute-force k-NN indices (integer output)',),
        notes='returns Int indices -> non-differentiable',
    )
)
register(
    OpInfo(
        'nitrix.smoothing.spatial_cube_neighbourhood',
        fixture=lambda: (((8, 8),), {}),
        diff_arg=None,
        vmap_arg=None,
        skip_jit=True,
        invariants=('host-side grid-box neighbour index construction',),
    )
)

# --- semiring (ELL row softmax) ---------------------------------------------


def _ell_row_softmax_setup():
    from nitrix.semiring import ell_row_softmax
    from nitrix.sparse import ELL

    rng = np.random.default_rng(0)
    n, k_max = 8, 4
    idx = jnp.asarray(rng.integers(0, n, (n, k_max)).astype(np.int32))
    ell = ELL(
        values=jnp.asarray(rng.standard_normal((n, k_max)).astype(np.float32)),
        indices=idx,
        n_cols=n,
        identity=0.0,
    )

    def op(scores):
        return ell_row_softmax(scores, ell)

    def fixture():
        s = jnp.asarray(rng.standard_normal((n, k_max)).astype(np.float32))
        return (s,), {}

    return op, fixture


_rowsoftmax_op, _rowsoftmax_fixture = _ell_row_softmax_setup()

register(
    OpInfo(
        'nitrix.semiring.ell_row_softmax',
        fixture=_rowsoftmax_fixture,
        fn_override=_rowsoftmax_op,
        diff_arg=0,
        vmap_arg=0,
        invariants=('row-wise softmax over ELL neighbours; masks pads',),
    )
)

# --- sparse (ELL transforms via fn_override; host-side constructors) --------


def _sparse_array_setups():
    """ELL ops threaded through their differentiable array (values),
    returning arrays so the standard probes apply."""
    from nitrix.semiring import REAL
    from nitrix.sparse import (
        ELL,
        ell_add_self_loops,
        ell_mask,
        ell_pad,
        ell_to_dense,
    )

    rng = np.random.default_rng(0)
    n, k = 10, 4
    idx = jnp.asarray(rng.integers(0, n, (n, k)).astype(np.int32))
    valid = jnp.asarray(rng.random(n) > 0.3)

    def mk(values):
        return ELL(values=values, indices=idx, n_cols=n, identity=0.0)

    def fixture():
        v = jnp.asarray(rng.standard_normal((n, k)).astype(np.float32))
        return (v,), {}

    ops = {
        'ell_to_dense': (
            lambda v: ell_to_dense(mk(v)),
            ('ELL -> dense materialisation',),
        ),
        'ell_mask': (
            lambda v: ell_mask(mk(v), valid, semiring=REAL).values,
            ('annihilator-masked edges (jnp.where)',),
        ),
        'ell_pad': (
            lambda v: ell_pad(v, idx, k_max=k + 2, n_cols=n).values,
            ('pad ragged rows to k_max with identity',),
        ),
        'ell_add_self_loops': (
            lambda v: ell_add_self_loops(mk(v))[0].values,
            ('append (i, i) self-edge per row',),
        ),
    }
    return ops, fixture


_sparse_ops, _sparse_fixture = _sparse_array_setups()
for _sp_name, (_sp_op, _sp_inv) in _sparse_ops.items():
    register(
        OpInfo(
            f'nitrix.sparse.{_sp_name}',
            fixture=_sparse_fixture,
            fn_override=_sp_op,
            diff_arg=0,
            vmap_arg=0,
            invariants=_sp_inv,
        )
    )


def _meanpool_setup():
    from nitrix.sparse import (
        icosphere_cross_level_adjacency,
        icosphere_hierarchy,
        mesh_coarsen_meanpool,
    )

    rng = np.random.default_rng(0)
    h = icosphere_hierarchy(max_level=1)
    ell = icosphere_cross_level_adjacency(h, 0, 1)
    n_fine = h.meshes[1].n_vertices

    def op(x):
        return mesh_coarsen_meanpool(ell, x)

    def fixture():
        return (
            jnp.asarray(rng.standard_normal((n_fine, 3)).astype(np.float32)),
        ), {}

    return op, fixture


_meanpool_op, _meanpool_fixture = _meanpool_setup()

register(
    OpInfo(
        'nitrix.sparse.mesh_coarsen_meanpool',
        fixture=_meanpool_fixture,
        fn_override=_meanpool_op,
        diff_arg=0,
        vmap_arg=0,
        invariants=('mean-pool sibling of mesh_pool_max (validity-weighted)',),
    )
)


def _sectioned_setup():
    from nitrix.sparse import (
        sectioned_ell_from_ragged,
        sectioned_semiring_ell_matmul,
    )

    rng = np.random.default_rng(0)
    n = 12
    degs = [2, 3, 2, 4, 1, 3, 2, 2, 3, 1, 2, 3]
    vals = [
        jnp.asarray(rng.standard_normal(d).astype(np.float32)) for d in degs
    ]
    idxs = [jnp.asarray(rng.integers(0, n, d).astype(np.int32)) for d in degs]
    sec = sectioned_ell_from_ragged(vals, idxs, n_cols=n)

    def op(b):
        return sectioned_semiring_ell_matmul(sec, b)

    def fixture():
        return (
            jnp.asarray(rng.standard_normal((n, 4)).astype(np.float32)),
        ), {}

    return op, fixture


_sectioned_op, _sectioned_fixture = _sectioned_setup()

register(
    OpInfo(
        'nitrix.sparse.sectioned_semiring_ell_matmul',
        fixture=_sectioned_fixture,
        fn_override=_sectioned_op,
        diff_arg=0,
        vmap_arg=0,
        invariants=('bucketed (variable-degree) ELL matmul',),
    )
)


def _sectioned_rmatvec_setup():
    from nitrix.sparse import (
        sectioned_ell_from_ragged,
        sectioned_semiring_ell_rmatvec,
    )

    rng = np.random.default_rng(0)
    n = 12
    degs = [2, 3, 2, 4, 1, 3, 2, 2, 3, 1, 2, 3]
    vals = [
        jnp.asarray(rng.standard_normal(d).astype(np.float32)) for d in degs
    ]
    idxs = [jnp.asarray(rng.integers(0, n, d).astype(np.int32)) for d in degs]
    sec = sectioned_ell_from_ragged(vals, idxs, n_cols=n)

    def op(x):
        return sectioned_semiring_ell_rmatvec(sec, x)

    def fixture():
        # X indexed by the original row axis (n_rows == n_cols == 12).
        return (
            jnp.asarray(rng.standard_normal((n, 4)).astype(np.float32)),
        ), {}

    return op, fixture


_sectioned_rmatvec_op, _sectioned_rmatvec_fixture = _sectioned_rmatvec_setup()

register(
    OpInfo(
        'nitrix.sparse.sectioned_semiring_ell_rmatvec',
        fixture=_sectioned_rmatvec_fixture,
        fn_override=_sectioned_rmatvec_op,
        diff_arg=0,
        vmap_arg=0,
        invariants=('bucketed (variable-degree) ELL adjoint matmul',),
    )
)


def _ico1_mesh():
    from nitrix.sparse import icosphere

    return icosphere(1)


def _ragged_fixture():
    rng = np.random.default_rng(0)
    n = 12
    degs = [2, 3, 2, 4, 1, 3, 2, 2, 3, 1, 2, 3]
    vals = [
        jnp.asarray(rng.standard_normal(d).astype(np.float32)) for d in degs
    ]
    idxs = [jnp.asarray(rng.integers(0, n, d).astype(np.int32)) for d in degs]
    return (vals, idxs), {'n_cols': n}


def _hier_levels_fixture():
    from nitrix.sparse import icosphere_hierarchy

    h = icosphere_hierarchy(max_level=1)
    return (h.meshes, h.parents), {}


register(
    OpInfo(
        'nitrix.sparse.icosphere',
        fixture=lambda: ((1,), {}),
        diff_arg=None,
        vmap_arg=None,
        skip_jit=True,
        invariants=('host-side icosphere subdivision (returns Mesh)',),
    )
)
register(
    OpInfo(
        'nitrix.sparse.grid_identity',
        fixture=lambda: (((8, 8),), {}),
        diff_arg=None,
        vmap_arg=None,
        skip_jit=True,
        invariants=('identity ELL over a regular grid',),
    )
)
register(
    OpInfo(
        'nitrix.sparse.regular_grid_stencil',
        fixture=lambda: (
            (
                (8, 8),
                [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]],
                jnp.full(5, 0.2),
            ),
            {},
        ),
        diff_arg=None,
        vmap_arg=None,
        skip_jit=True,
        invariants=('regular-grid stencil -> ELL',),
    )
)
register(
    OpInfo(
        'nitrix.sparse.mesh_cotangent_laplacian',
        fixture=lambda: ((_ico1_mesh(),), {}),
        diff_arg=None,
        vmap_arg=None,
        skip_jit=True,
        invariants=('cotangent-weight Laplacian -> ELL',),
    )
)
register(
    OpInfo(
        'nitrix.sparse.sectioned_ell_from_ragged',
        fixture=_ragged_fixture,
        diff_arg=None,
        vmap_arg=None,
        skip_jit=True,
        invariants=('ragged rows -> bucketed SectionedELL',),
    )
)
register(
    OpInfo(
        'nitrix.sparse.icosphere_hierarchy_from_levels',
        fixture=_hier_levels_fixture,
        diff_arg=None,
        vmap_arg=None,
        skip_jit=True,
        invariants=(
            'package caller-supplied meshes + parents into a hierarchy',
        ),
    )
)

# --- bias (N4 + Nyul-Udupa + B-spline) --------------------------------------


def _bias_image():
    return jnp.abs(jax.random.normal(_key(), (16, 16))) + 1.0


register(
    OpInfo(
        'nitrix.bias.histogram_match',
        fixture=lambda: (
            (
                jax.random.normal(_key(0), (16, 16)),
                jax.random.normal(_key(1), (16, 16)),
            ),
            {},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('Nyul-Udupa landmark histogram match',),
    )
)
register(
    OpInfo(
        'nitrix.bias.sharpen_histogram',
        fixture=lambda: ((_bias_image(),), {}),
        invariants=('Wiener histogram deconvolution (N4 inner step)',),
    )
)
register(
    OpInfo(
        'nitrix.bias.bspline_approximate',
        fixture=lambda: (
            (jax.random.normal(_key(), (16, 16)),),
            {'control_points': 4},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('separable B-spline least-squares approximation',),
    )
)
register(
    OpInfo(
        'nitrix.bias.n4_bias_field_correction',
        fixture=lambda: (
            (_bias_image(),),
            {
                'max_iterations': 2,
                'n_fitting_levels': 1,
                'n_control_points': 4,
            },
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=(
            'iterative B-spline N4 (lax.while_loop; reverse-grad unsupported)',
        ),
        notes='grad expected to FAIL: while_loop has no reverse-mode rule',
    )
)
register(
    OpInfo(
        'nitrix.bias.bias_field_correction',
        fixture=lambda: (
            (_bias_image(),),
            {
                'method': 'n4',
                'max_iterations': 2,
                'n_fitting_levels': 1,
                'n_control_points': 4,
            },
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('dispatcher over correction methods (N4 default)',),
        notes='method=n4 -> grad FAIL (while_loop)',
    )
)

# ===========================================================================
# Registration suite + this round's backlog: the public ops graduated for the
# rigid/affine/diffeomorphic recipes (geometry Lie chart, matrix-free
# optimisers) plus the new score-kernels, normalisers, and augmentation
# primitives.  Internals are excluded; everything below is public API we want
# capability-probed and benchmark-flagged.
# ===========================================================================

# --- linalg (matrix-free optimisers + dense solves + matrix exp) ------------

register(
    OpInfo(
        'nitrix.linalg.matrix_exp',
        fixture=lambda: ((jax.random.normal(_key(), (4, 4)),), {}),
        invariants=(
            'scaling-and-squaring + Taylor (pure matmul, cuSolver-free)',
            'det(exp A) = exp(tr A)',
        ),
    )
)


def _spd_matrix():
    m = jax.random.normal(_key(1), (5, 5))
    return m @ m.T + 5.0 * jnp.eye(5)


register(
    OpInfo(
        'nitrix.linalg.solve',
        fixture=lambda: (
            (
                jnp.eye(5) + 0.1 * jax.random.normal(_key(1), (5, 5)),
                jax.random.normal(_key(2), (5,)),
            ),
            {},
        ),
        diff_arg=0,
        vmap_arg=0,
        invariants=('general A x = b via safe_solve (cuSolver-routed)',),
        notes='prefer cho_solve for SPD',
    )
)
register(
    OpInfo(
        'nitrix.linalg.cho_solve',
        fixture=lambda: (
            (_spd_matrix(), jax.random.normal(_key(3), (5,))),
            {},
        ),
        diff_arg=0,
        vmap_arg=0,
        invariants=('SPD (A + l2 I) x = b via Cholesky (safe_cho_solve)',),
    )
)
register(
    OpInfo(
        'nitrix.linalg.cg',
        fixture=lambda: (
            (_spd_matrix(), jax.random.normal(_key(2), (5,))),
            {},
        ),
        diff_arg=1,
        vmap_arg=1,
        invariants=('matrix-free CG for SPD systems (cuSolver-free)',),
    )
)


# Matrix-free nonlinear least squares share a fixed linear residual
# r(x) = A x - y, so a Gauss-Newton step converges in one iteration.
_LS_A = jax.random.normal(_key(1), (6, 4))
_LS_Y = jax.random.normal(_key(2), (6,))


def _ls_residual(x):
    return _LS_A @ x - _LS_Y


def _gn_op(x0):
    from nitrix.linalg import gauss_newton

    return gauss_newton(_ls_residual, x0, n_iters=5)


def _lm_op(x0):
    from nitrix.linalg import levenberg_marquardt

    return levenberg_marquardt(_ls_residual, x0, n_iters=8)


def _opt_fixture():
    return (jax.random.normal(_key(3), (4,)),), {}


register(
    OpInfo(
        'nitrix.linalg.gauss_newton',
        fixture=_opt_fixture,
        fn_override=_gn_op,
        diff_arg=0,
        vmap_arg=0,
        invariants=('matrix-free Gauss-Newton normal equations (cg inner)',),
        notes='unrolled loop -> jit/grad-clean; returns OptimizeResult',
    )
)
register(
    OpInfo(
        'nitrix.linalg.levenberg_marquardt',
        fixture=_opt_fixture,
        fn_override=_lm_op,
        diff_arg=0,
        vmap_arg=0,
        invariants=('damped normal equations; jnp.where accept/reject',),
        notes='monotone cost_history; returns OptimizeResult',
    )
)


def _ils_residual(data, x):
    return _LS_A @ x - data


def _ils_op(data, x0):
    from nitrix.linalg import implicit_least_squares

    return implicit_least_squares(_ils_residual, data, x0, n_iters=8)


register(
    OpInfo(
        'nitrix.linalg.implicit_least_squares',
        fixture=lambda: (
            (jax.random.normal(_key(2), (6,)), jnp.zeros((4,))),
            {},
        ),
        fn_override=_ils_op,
        diff_arg=0,
        vmap_arg=None,
        invariants=('argmin differentiable by the implicit-function theorem',),
        notes='custom_vjp: exact grad w.r.t data without unrolling',
    )
)

# --- geometry (Lie chart, affine algebra, multi-resolution, deformation) ----


def _rigid_matrix_3d():
    from nitrix.geometry import rigid_exp

    return rigid_exp(jax.random.normal(_key(0), (6,)) * 0.1, ndim=3)


def _affine_matrix_3d():
    from nitrix.geometry import affine_exp

    return affine_exp(jax.random.normal(_key(0), (12,)) * 0.1, ndim=3)


def _affine_par_3d():
    return jnp.asarray(
        [0.5, -0.3, 0.2, 10.0, 20.0, 30.0, 1.1, 0.9, 1.05, 0.01, 0.0, 0.0]
    )


def _affine_mat_34():
    from nitrix.geometry import params_to_affine_matrix

    return params_to_affine_matrix(_affine_par_3d(), ndim=3)


def _rotation_3d():
    from nitrix.geometry import angles_to_rotation_matrix

    return angles_to_rotation_matrix(jnp.asarray([10.0, 20.0, 30.0]))


register(
    OpInfo(
        'nitrix.geometry.rigid_exp',
        fixture=lambda: ((jax.random.normal(_key(), (6,)),), {'ndim': 3}),
        diff_arg=0,
        vmap_arg=0,
        invariants=(
            'SE(n) chart: closed-form Rodrigues + direct translation',
        ),
    )
)
register(
    OpInfo(
        'nitrix.geometry.rigid_log',
        fixture=lambda: ((_rigid_matrix_3d(),), {'ndim': 3}),
        diff_arg=0,
        vmap_arg=0,
        invariants=('inverse of rigid_exp (axis-angle recovery)',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.affine_exp',
        fixture=lambda: (
            (jax.random.normal(_key(), (12,)) * 0.1,),
            {'ndim': 3},
        ),
        diff_arg=0,
        vmap_arg=0,
        invariants=('GL(n) chart: linear block = matrix_exp(generator)',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.apply_affine',
        fixture=lambda: (
            (jax.random.normal(_key(0), (10, 3)), _affine_matrix_3d()),
            {},
        ),
        diff_arg=0,
        vmap_arg=0,
        invariants=('coordinate map M(p - c) + t + c',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.affine_grid',
        fixture=lambda: ((_affine_matrix_3d(),), {'spatial_shape': (6, 6, 6)}),
        diff_arg=0,
        vmap_arg=None,
        invariants=('absolute sample coords of a homogeneous transform',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.make_square_affine',
        fixture=lambda: ((_affine_mat_34(),), {}),
        diff_arg=0,
        vmap_arg=0,
        invariants=('(N,N+1) -> (N+1,N+1) homogenisation',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.invert_affine',
        fixture=lambda: ((_affine_mat_34(),), {}),
        diff_arg=0,
        vmap_arg=0,
        invariants=('affine inverse via safe_inv (cuSolver-routed)',),
    )
)


def _compose_affine_op(a, b):
    from nitrix.geometry import compose_affine

    return compose_affine([a, b])


register(
    OpInfo(
        'nitrix.geometry.compose_affine',
        fixture=lambda: ((_affine_mat_34(), _affine_mat_34()), {}),
        fn_override=_compose_affine_op,
        diff_arg=0,
        vmap_arg=0,
        invariants=('compose homogeneous affines',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.fit_affine',
        fixture=lambda: (
            (
                jax.random.normal(_key(0), (10, 3)),
                jax.random.normal(_key(1), (10, 3)),
            ),
            {},
        ),
        diff_arg=0,
        vmap_arg=0,
        invariants=('closed-form (weighted) LS affine via safe_cho_solve',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.angles_to_rotation_matrix',
        fixture=lambda: ((jnp.asarray([10.0, 20.0, 30.0]),), {}),
        diff_arg=0,
        vmap_arg=0,
        invariants=('Euler intrinsic X@Y@Z (2-D / 3-D)',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.rotation_matrix_to_angles',
        fixture=lambda: ((_rotation_3d(),), {}),
        diff_arg=0,
        vmap_arg=0,
        invariants=('rotation -> Euler angles (gimbal-lock branch)',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.params_to_affine_matrix',
        fixture=lambda: ((_affine_par_3d(),), {'ndim': 3}),
        diff_arg=0,
        vmap_arg=0,
        invariants=('compose affine T@R@S@E from geometric params',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.affine_matrix_to_params',
        fixture=lambda: ((_affine_mat_34(),), {}),
        diff_arg=0,
        vmap_arg=0,
        invariants=('decompose affine -> params (Cholesky of MᵀM)',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.sample_at_points',
        fixture=lambda: (
            (
                jax.random.normal(_key(0), (12, 12, 12)),
                jax.random.uniform(_key(1), (20, 3)) * 11.0,
            ),
            {},
        ),
        diff_arg=0,
        vmap_arg=1,
        invariants=('scattered-point sampling (complement of grid warp)',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.spatial_gradient',
        fixture=lambda: (
            (jax.random.normal(_key(), (16, 16)),),
            {'spatial_rank': 2},
        ),
        diff_arg=0,
        vmap_arg=0,
        invariants=('separable central/sobel/scharr gradient (correlate1d)',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.downsample',
        fixture=lambda: ((jax.random.normal(_key(), (16, 16, 1)),), {}),
        diff_arg=0,
        vmap_arg=None,
        invariants=('anti-alias gaussian + align-corners resample',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.upsample',
        fixture=lambda: (
            (jax.random.normal(_key(), (8, 8, 1)),),
            {'target_shape': (16, 16)},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('align-corners prolongation (resample up)',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.gaussian_pyramid',
        fixture=lambda: (
            (jax.random.normal(_key(), (16, 16, 1)),),
            {'levels': 3},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('levels-deep anti-aliased pyramid (finest first)',),
        notes='returns a tuple; grad reduces the finest leaf',
    )
)


def _disp_field():
    return jax.random.normal(_key(0), (8, 8, 2)) * 0.1


register(
    OpInfo(
        'nitrix.geometry.compose_displacement',
        fixture=lambda: ((_disp_field(), _disp_field()), {}),
        diff_arg=0,
        vmap_arg=None,
        invariants=('(id+outer)∘(id+inner) displacement composition',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.compose_velocity',
        fixture=lambda: ((_disp_field(), _disp_field()), {}),
        diff_arg=0,
        vmap_arg=None,
        invariants=('BCH velocity composition (order 1 add; 2 adds ½[v,u])',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.invert_displacement',
        fixture=lambda: ((_disp_field(),), {}),
        diff_arg=0,
        vmap_arg=None,
        invariants=('inverse displacement via differentiable fixed point',),
    )
)
register(
    OpInfo(
        'nitrix.geometry.field_log',
        fixture=lambda: ((_disp_field(),), {}),
        diff_arg=0,
        vmap_arg=None,
        invariants=(
            'stationary-velocity log via inverse scaling-and-squaring; '
            'exact round-trip exp(field_log(s))==s on the SVF submanifold',
        ),
    )
)

# --- numerics (normalisers, spatial windowing, ODE, fixed point) ------------

register(
    OpInfo(
        'nitrix.numerics.l2_normalize',
        fixture=lambda: ((jax.random.normal(_key(), (8, 16)),), {}),
        invariants=(
            'unit-L2 projection (clamp denominator, torch convention)',
        ),
    )
)
register(
    OpInfo(
        'nitrix.numerics.lp_normalize',
        fixture=lambda: ((jax.random.normal(_key(), (8, 16)),), {}),
        invariants=('unit-Lp projection x / max(||x||_p, eps)',),
    )
)
register(
    OpInfo(
        'nitrix.numerics.instance_norm',
        fixture=lambda: (
            (jax.random.normal(_key(), (2, 3, 8, 8)),),
            {'axes': (-2, -1)},
        ),
        invariants=('per-instance centre-and-scale (no trainable affine)',),
    )
)
register(
    OpInfo(
        'nitrix.numerics.pad_to_multiple',
        fixture=lambda: (
            (jax.random.normal(_key(), (15, 15)),),
            {'multiple': 8, 'spatial_rank': 2},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('pad spatial axes to a multiple (returns pad widths)',),
    )
)
register(
    OpInfo(
        'nitrix.numerics.crop_to_multiple',
        fixture=lambda: (
            (jax.random.normal(_key(), (20, 20)),),
            {'multiple': 8, 'spatial_rank': 2},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('crop spatial axes to a multiple (returns crop widths)',),
    )
)
register(
    OpInfo(
        'nitrix.numerics.nonzero_bounding_box',
        fixture=lambda: ((jax.random.normal(_key(), (16, 16)),), {}),
        diff_arg=None,
        vmap_arg=None,
        invariants=('bounding box of x>threshold as (lo,hi) indices',),
        notes='returns Int indices -> non-differentiable',
    )
)
register(
    OpInfo(
        'nitrix.numerics.gaussian_window',
        fixture=lambda: (((8, 8),), {}),
        diff_arg=None,
        vmap_arg=None,
        skip_jit=True,
        invariants=('host-side separable Gaussian patch window (peak 1)',),
    )
)
register(
    OpInfo(
        'nitrix.numerics.overlap_add',
        fixture=lambda: (
            (
                jax.random.normal(_key(0), (8, 8)),
                jnp.abs(jax.random.normal(_key(1), (8, 8))) + 0.1,
            ),
            {},
        ),
        diff_arg=0,
        vmap_arg=0,
        invariants=('weighted overlap-add finalisation Σ(w·p)/Σw',),
    )
)


def _decay_field(t, y):
    return -y


def _ode_fixture():
    return (
        jax.random.normal(_key(0), (4,)),
        jnp.linspace(0.0, 1.0, 8),
    ), {}


def _euler_op(y0, t):
    from nitrix.numerics import euler

    return euler(_decay_field, y0, t)


def _rk4_op(y0, t):
    from nitrix.numerics import rk4

    return rk4(_decay_field, y0, t)


def _odeint_op(y0, t):
    from nitrix.numerics import odeint

    return odeint(_decay_field, y0, t, method='rk4')


register(
    OpInfo(
        'nitrix.numerics.euler',
        fixture=_ode_fixture,
        fn_override=_euler_op,
        diff_arg=0,
        vmap_arg=0,
        invariants=('explicit Euler via lax.scan (grad through the solver)',),
    )
)
register(
    OpInfo(
        'nitrix.numerics.rk4',
        fixture=_ode_fixture,
        fn_override=_rk4_op,
        diff_arg=0,
        vmap_arg=0,
        invariants=('classic 4th-order Runge-Kutta via lax.scan',),
    )
)
register(
    OpInfo(
        'nitrix.numerics.odeint',
        fixture=_ode_fixture,
        fn_override=_odeint_op,
        diff_arg=0,
        vmap_arg=0,
        invariants=('fixed-step ODE dispatch (euler / rk4)',),
    )
)


def _fp_map(p, x):
    return 0.5 * x + 0.5 * p


def _fp_op(params, x0):
    from nitrix.numerics import fixed_point_solve

    return fixed_point_solve(_fp_map, params, x0)


register(
    OpInfo(
        'nitrix.numerics.fixed_point_solve',
        fixture=lambda: (
            (jax.random.normal(_key(0), (8,)), jnp.zeros((8,))),
            {},
        ),
        fn_override=_fp_op,
        diff_arg=0,
        vmap_arg=None,
        invariants=('implicit-VJP fixed-point iteration (grad by IFT)',),
    )
)

# --- morphology (connected components) --------------------------------------

register(
    OpInfo(
        'nitrix.morphology.connected_components',
        fixture=lambda: (
            (jax.random.uniform(_key(), (12, 12)) > 0.5,),
            {},
        ),
        diff_arg=None,
        vmap_arg=None,
        invariants=('pointer-jumping label propagation O(log diameter)',),
        notes='returns Int labels -> non-differentiable; lax.while_loop',
    )
)
register(
    OpInfo(
        'nitrix.morphology.largest_connected_component',
        fixture=lambda: (
            (jax.random.uniform(_key(), (12, 12)) > 0.5,),
            {},
        ),
        diff_arg=None,
        vmap_arg=None,
        invariants=('largest CC mask (bincount over components)',),
        notes='returns Bool mask -> non-differentiable; lax.while_loop',
    )
)

# --- sparse (triangle-mesh vertex ops) --------------------------------------


def _tetra_mesh():
    vertices = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = jnp.asarray(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=jnp.int32
    )
    return vertices, faces


register(
    OpInfo(
        'nitrix.sparse.compute_vertex_normals',
        fixture=lambda: (_tetra_mesh(), {}),
        diff_arg=0,
        vmap_arg=None,
        invariants=('area-weighted face normals scattered + L2-normed',),
    )
)
register(
    OpInfo(
        'nitrix.sparse.mesh_laplacian_smooth',
        fixture=lambda: (_tetra_mesh(), {'iterations': 2}),
        diff_arg=0,
        vmap_arg=None,
        invariants=('uniform (combinatorial) Laplacian vertex smoothing',),
    )
)

# --- stats (diagonal Gaussian + PCA) ----------------------------------------

register(
    OpInfo(
        'nitrix.stats.kl_diagonal_gaussian',
        fixture=lambda: (
            (
                jax.random.normal(_key(0), (8, 16)),
                jax.random.normal(_key(1), (8, 16)),
            ),
            {},
        ),
        diff_arg=0,
        vmap_arg=0,
        invariants=('closed-form KL(N(μ, diag e^logvar) || N(0, I))',),
    )
)
register(
    OpInfo(
        'nitrix.stats.gaussian_nll',
        fixture=lambda: (
            (
                jax.random.normal(_key(0), (8, 16)),
                jax.random.normal(_key(1), (8, 16)),
                jax.random.normal(_key(2), (8, 16)),
            ),
            {},
        ),
        diff_arg=1,
        vmap_arg=0,
        invariants=('diagonal-Gaussian NLL (log-variance parameterised)',),
    )
)
register(
    OpInfo(
        'nitrix.stats.pca_fit',
        fixture=lambda: (
            (jax.random.normal(_key(), (20, 8)),),
            {'n_components': 3},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=(
            'covariance-eigendecomposition PCA via safe_eigh (no svd)',
        ),
        notes='returns PCAResult; grad reduces the components leaf',
    )
)
register(
    OpInfo(
        'nitrix.stats.pca_transform',
        fixture=lambda: (
            (
                jax.random.normal(_key(0), (20, 8)),
                jax.random.normal(_key(1), (3, 8)),
                jax.random.normal(_key(2), (8,)),
            ),
            {},
        ),
        diff_arg=0,
        vmap_arg=0,
        invariants=('project onto a (pre-fitted) PCA basis',),
    )
)
register(
    OpInfo(
        'nitrix.stats.pca_inverse_transform',
        fixture=lambda: (
            (
                jax.random.normal(_key(0), (20, 3)),
                jax.random.normal(_key(1), (3, 8)),
                jax.random.normal(_key(2), (8,)),
            ),
            {},
        ),
        diff_arg=0,
        vmap_arg=0,
        invariants=('reconstruct from PCA coordinates',),
    )
)

# --- metrics (intensity / information / overlap / classification / SSL) -----


def _img_pair(shape=(16, 16)):
    return (
        jax.random.normal(_key(0), shape),
        jax.random.normal(_key(1), shape),
    )


register(
    OpInfo(
        'nitrix.metrics.ssd',
        fixture=lambda: (_img_pair(), {}),
        diff_arg=0,
        vmap_arg=0,
        invariants=('(weighted) mean/sum of squared differences',),
    )
)
register(
    OpInfo(
        'nitrix.metrics.ncc',
        fixture=lambda: (_img_pair(), {}),
        diff_arg=0,
        vmap_arg=0,
        invariants=('global Pearson cross-correlation (stats.corr shape)',),
    )
)
register(
    OpInfo(
        'nitrix.metrics.lncc',
        fixture=lambda: (_img_pair(), {'radius': 4, 'spatial_rank': 2}),
        diff_arg=0,
        vmap_arg=0,
        invariants=('windowed local NCC (ANTs form; separable box filter)',),
    )
)
register(
    OpInfo(
        'nitrix.metrics.joint_histogram',
        fixture=lambda: (_img_pair(), {}),
        diff_arg=0,
        vmap_arg=0,
        invariants=('differentiable Parzen soft-binned joint histogram',),
    )
)
register(
    OpInfo(
        'nitrix.metrics.mutual_information',
        fixture=lambda: (_img_pair(), {}),
        diff_arg=0,
        vmap_arg=0,
        invariants=('MI / NMI from the soft joint histogram',),
    )
)
register(
    OpInfo(
        'nitrix.metrics.correlation_ratio',
        fixture=lambda: (_img_pair(), {}),
        diff_arg=0,
        vmap_arg=0,
        invariants=("Roche's η² between-group variance ratio",),
    )
)


def _soft_mask_pair():
    return (
        jax.nn.sigmoid(jax.random.normal(_key(0), (4, 16, 16))),
        jax.nn.sigmoid(jax.random.normal(_key(1), (4, 16, 16))),
    )


register(
    OpInfo(
        'nitrix.metrics.dice',
        fixture=lambda: (_soft_mask_pair(), {}),
        diff_arg=0,
        vmap_arg=0,
        invariants=('soft Sørensen-Dice 2|A∩B|/(|A|+|B|)',),
    )
)
register(
    OpInfo(
        'nitrix.metrics.jaccard',
        fixture=lambda: (_soft_mask_pair(), {}),
        diff_arg=0,
        vmap_arg=0,
        invariants=('soft Jaccard / IoU |A∩B|/|A∪B|',),
    )
)
register(
    OpInfo(
        'nitrix.metrics.bce_with_logits',
        fixture=lambda: (
            (
                jax.random.normal(_key(0), (8, 16)),
                jax.nn.sigmoid(jax.random.normal(_key(1), (8, 16))),
            ),
            {},
        ),
        diff_arg=0,
        vmap_arg=0,
        invariants=('stable BCE-from-logits max(x,0)-xt+log1p(e^{-|x|})',),
    )
)
register(
    OpInfo(
        'nitrix.metrics.cross_entropy_with_logits',
        fixture=lambda: (
            (
                jax.random.normal(_key(0), (8, 5, 4)),
                jax.random.randint(_key(1), (8, 4), 0, 5),
            ),
            {'axis': 1},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('categorical CE from logits (log_softmax + gather)',),
    )
)
register(
    OpInfo(
        'nitrix.metrics.focal_loss',
        fixture=lambda: (
            (
                jax.random.normal(_key(0), (8, 16)),
                jax.nn.sigmoid(jax.random.normal(_key(1), (8, 16))),
            ),
            {},
        ),
        diff_arg=0,
        vmap_arg=0,
        invariants=('binary focal loss (1-p_t)^γ · BCE',),
    )
)
register(
    OpInfo(
        'nitrix.metrics.info_nce',
        fixture=lambda: (
            (
                jax.random.normal(_key(0), (8, 16)),
                jax.random.normal(_key(1), (8, 16)),
            ),
            {},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('InfoNCE / NT-Xent cross-view (no self-pair bias)',),
    )
)
register(
    OpInfo(
        'nitrix.metrics.dino_cross_entropy',
        fixture=lambda: (
            (
                jax.random.normal(_key(0), (8, 64)),
                jax.random.normal(_key(1), (8, 64)),
                jax.random.normal(_key(2), (64,)),
            ),
            {},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('centred / sharpened self-distillation CE (DINO)',),
    )
)
register(
    OpInfo(
        'nitrix.metrics.ibot_cross_entropy',
        fixture=lambda: (
            (
                jax.random.normal(_key(0), (4, 6, 32)),
                jax.random.normal(_key(1), (4, 6, 32)),
                jax.random.normal(_key(2), (32,)),
                jax.random.uniform(_key(3), (4, 6)) > 0.5,
            ),
            {},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('masked-token self-distillation CE (iBOT; mask mean)',),
    )
)
register(
    OpInfo(
        'nitrix.metrics.koleo',
        fixture=lambda: ((jax.random.normal(_key(), (8, 16)),), {}),
        diff_arg=0,
        vmap_arg=None,
        invariants=('Kozachenko-Leonenko feature-spread regulariser',),
    )
)

# --- register (end-to-end recipes + explicit field penalties) ---------------


def _reg_images():
    return (
        jax.random.normal(_key(0), (16, 16)),
        jax.random.normal(_key(1), (16, 16)),
    )


def _rigid_reg_fixture():
    from nitrix.register import RegistrationSpec

    return _reg_images(), {'spec': RegistrationSpec(levels=1, iterations=3)}


def _demons_fixture():
    from nitrix.register import DemonsSpec

    return _reg_images(), {
        'spec': DemonsSpec(levels=1, iterations=3, n_steps=2)
    }


register(
    OpInfo(
        'nitrix.register.rigid_register',
        fixture=_rigid_reg_fixture,
        diff_arg=None,
        vmap_arg=None,
        invariants=('coarse-to-fine GN/LM intensity registration (SE(n))',),
        notes='end-to-end optimiser; differentiate via implicit_least_squares',
    )
)
register(
    OpInfo(
        'nitrix.register.affine_register',
        fixture=_rigid_reg_fixture,
        diff_arg=None,
        vmap_arg=None,
        invariants=('coarse-to-fine GN/LM affine registration (exp block)',),
        notes='end-to-end optimiser; differentiate via implicit_least_squares',
    )
)
register(
    OpInfo(
        'nitrix.register.diffeomorphic_demons_register',
        fixture=_demons_fixture,
        diff_arg=None,
        vmap_arg=None,
        invariants=('log-domain diffeomorphic Demons (SVF; ESM force)',),
        notes='end-to-end optimiser (benchmark-worthy)',
    )
)
register(
    OpInfo(
        'nitrix.register.gradient_smoothness',
        fixture=lambda: ((_disp_field(),), {}),
        diff_arg=0,
        vmap_arg=None,
        invariants=('diffusion (first-order) smoothness ‖∇u‖²',),
    )
)
register(
    OpInfo(
        'nitrix.register.bending_energy',
        fixture=lambda: ((_disp_field(),), {}),
        diff_arg=0,
        vmap_arg=None,
        invariants=('thin-plate (second-order) bending ‖∇²u‖²',),
    )
)
register(
    OpInfo(
        'nitrix.register.jacobian_folding_penalty',
        fixture=lambda: ((_disp_field(),), {}),
        diff_arg=0,
        vmap_arg=None,
        invariants=('folding penalty relu(-det J), J = I + ∇u',),
    )
)

# --- augment (intensity / geometric / synthesis generators) -----------------

register(
    OpInfo(
        'nitrix.augment.gamma_contrast',
        fixture=lambda: (
            (jax.nn.sigmoid(jax.random.normal(_key(0), (8, 8))), 2.0),
            {},
        ),
        diff_arg=0,
        vmap_arg=0,
        invariants=('gamma tone curve within a normalised bracket',),
    )
)
register(
    OpInfo(
        'nitrix.augment.random_histogram_shift',
        fixture=lambda: (
            (jax.random.normal(_key(0), (8, 8)), _key(1)),
            {},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('random monotone piecewise-linear intensity remap',),
    )
)
register(
    OpInfo(
        'nitrix.augment.gibbs_ringing',
        fixture=lambda: ((jax.random.normal(_key(0), (8, 8)), 0.3), {}),
        diff_arg=0,
        vmap_arg=None,
        invariants=('Gibbs (truncation) ringing via hard k-space cutoff',),
    )
)
register(
    OpInfo(
        'nitrix.augment.gaussian_noise',
        fixture=lambda: (
            (jax.random.normal(_key(0), (8, 8)), _key(1)),
            {'sigma': 0.1},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('additive i.i.d. Gaussian noise',),
    )
)
register(
    OpInfo(
        'nitrix.augment.rician_noise',
        fixture=lambda: (
            (jnp.abs(jax.random.normal(_key(0), (8, 8))), _key(1)),
            {'sigma': 0.1},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('Rician magnitude-noise sqrt((x+n_r)²+n_i²)',),
    )
)
register(
    OpInfo(
        'nitrix.augment.random_flip',
        fixture=lambda: (
            (jax.random.normal(_key(0), (8, 8)), _key(1)),
            {},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('per-axis Bernoulli reflection',),
    )
)
register(
    OpInfo(
        'nitrix.augment.random_crop',
        fixture=lambda: (
            (jax.random.normal(_key(0), (8, 8)), _key(1)),
            {'size': (4, 4)},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('random-offset fixed-size crop (dynamic_slice)',),
    )
)
register(
    OpInfo(
        'nitrix.augment.random_resized_crop',
        fixture=lambda: (
            (jax.random.normal(_key(0), (8, 8, 1)), _key(1)),
            {'size': (8, 8)},
        ),
        diff_arg=0,
        vmap_arg=None,
        invariants=('zoom crop: random window resampled to a fixed shape',),
    )
)


def _random_svf_op(key):
    from nitrix.augment import random_svf_displacement

    return random_svf_displacement((8, 8), key)


def _simulate_bias_op(key):
    from nitrix.augment import simulate_bias_field

    return simulate_bias_field((8, 8), key)


register(
    OpInfo(
        'nitrix.augment.random_affine_matrix',
        fixture=lambda: ((_key(0),), {'ndim': 3}),
        diff_arg=None,
        vmap_arg=None,
        invariants=('random affine in geometric params (T@R@S@E)',),
    )
)
register(
    OpInfo(
        'nitrix.augment.random_svf_displacement',
        fixture=lambda: ((_key(0),), {}),
        fn_override=_random_svf_op,
        diff_arg=None,
        vmap_arg=None,
        invariants=('smooth diffeomorphic SVF displacement (integrate exp)',),
    )
)
register(
    OpInfo(
        'nitrix.augment.gmm_label_to_image',
        fixture=lambda: (
            (
                jax.random.randint(_key(0), (8, 8), 0, 4),
                jnp.asarray([0.0, 0.3, 0.6, 0.9]),
                jnp.asarray([0.05, 0.05, 0.05, 0.05]),
                _key(1),
            ),
            {},
        ),
        diff_arg=1,
        vmap_arg=None,
        invariants=(
            'per-label Gaussian-mixture render (domain randomisation)',
        ),
    )
)
register(
    OpInfo(
        'nitrix.augment.simulate_bias_field',
        fixture=lambda: ((_key(0),), {}),
        fn_override=_simulate_bias_op,
        diff_arg=None,
        vmap_arg=None,
        invariants=(
            'forward multiplicative INU/bias field (exp coarse field)',
        ),
    )
)


# ---------------------------------------------------------------------------
# Host snapshot
# ---------------------------------------------------------------------------


def host_snapshot() -> dict:
    """Capture the host configuration for the rendered matrix.

    Every cell in the rendered table is *observed on this host*;
    a different runner may produce different statuses.
    """
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
        jit_grad_status = probe_jit_of_grad(
            fn, args, kwargs, op.diff_arg, op.reducer
        )
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


def render_markdown(cells: list[CellResult], host: dict) -> str:
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
    lines.append(
        'This matrix is **capability-only** (jit / grad / vmap / jit-of-grad '
        '+ invariants).  **Performance lives in the sibling '
        '``nitrix-perf-bench`` suite** + its hosted dashboard -- richer '
        '(cross-framework, multi-platform, history, fidelity-gated) than a '
        'matrix cell could be.'
    )
    lines.append('')
    lines.append('## Host snapshot')
    lines.append('')
    for k, v in host.items():
        lines.append(f'- **{k}**: {v}')
    lines.append('')
    lines.append('## Legend')
    lines.append('')
    lines.append('- ✅ probe passes (transformation works on this host).')
    lines.append('- ❌ probe fails (cell shows the truncated error message).')
    lines.append(
        '- — not applicable for this op (no natural diff target, etc.).'
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
            '| op | jit | grad | vmap | jit(grad) | invariants | notes |'
        )
        lines.append('|---|:--:|:--:|:--:|:--:|---|---|')
        for c in by_package[pkg]:
            short = c.op.qualname.split('.')[-1]
            jit_g = _status_to_glyph(c.jit)
            grad_g = _status_to_glyph(c.grad)
            vmap_g = _status_to_glyph(c.vmap)
            jgrad_g = _status_to_glyph(c.jit_of_grad)
            inv = '; '.join(c.op.invariants) if c.op.invariants else ''
            notes = c.op.notes
            # Embed error message if status != pass / n/a
            err_parts = []
            for label, status in [
                ('jit', c.jit),
                ('grad', c.grad),
                ('vmap', c.vmap),
                ('jit(grad)', c.jit_of_grad),
            ]:
                if status not in ('pass', 'n/a'):
                    err_parts.append(f'`{label}: {status}`')
            if err_parts:
                notes = (notes + ' — errors: ' + '; '.join(err_parts)).lstrip(
                    ' —'
                )
            lines.append(
                f'| `{short}` | {jit_g} | {grad_g} | {vmap_g} | '
                f'{jgrad_g} | {inv} | {notes} |'
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
    lines.append(f'- **ops catalogued**: {n_total}')
    lines.append(f'- **jit pass**: {n_jit} / {n_total}')
    lines.append(f'- **grad pass**: {n_grad} / {n_grad_app} (applicable)')
    lines.append(f'- **vmap pass**: {n_vmap} / {n_vmap_app} (applicable)')
    lines.append(f'- **jit(grad) pass**: {n_jgrad} / {n_grad_app}')
    lines.append(
        '- **performance**: see the nitrix-perf-bench suite + dashboard '
        '(this matrix is capability-only).'
    )
    return '\n'.join(lines)


def render_json(cells: list[CellResult], host: dict) -> str:
    out = {
        'host': host,
        # Capability-only; performance lives in the nitrix-perf-bench suite.
        'perf_source': 'nitrix-perf-bench',
        'ops': [],
    }
    for c in cells:
        out['ops'].append(
            {
                'qualname': c.op.qualname,
                'jit': c.jit,
                'grad': c.grad,
                'vmap': c.vmap,
                'jit_of_grad': c.jit_of_grad,
                'invariants': list(c.op.invariants),
                'notes': c.op.notes,
            }
        )
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
            cells.append(
                CellResult(
                    op=op,
                    jit=f'fixture failed: {type(e).__name__}',
                    grad='not-run',
                    vmap='not-run',
                    jit_of_grad='not-run',
                )
            )

    host = host_snapshot()

    md = render_markdown(cells, host)
    js = render_json(cells, host)

    (docs_dir / 'op_matrix.md').write_text(md)
    (docs_dir / 'op_matrix.json').write_text(js)
    print(f'wrote docs/op_matrix.md ({len(cells)} ops)')
    print('wrote docs/op_matrix.json')

    n_pass = sum(1 for c in cells if c.jit == 'pass')
    n_grad_app = sum(1 for c in cells if c.op.diff_arg is not None)
    n_grad_pass = sum(1 for c in cells if c.grad == 'pass')
    print(
        f'\nSummary: jit {n_pass}/{len(cells)}, '
        f'grad {n_grad_pass}/{n_grad_app}'
    )


if __name__ == '__main__':
    main()
