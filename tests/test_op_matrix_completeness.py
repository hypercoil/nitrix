# -*- coding: utf-8 -*-
"""Completeness guard for the op-matrix catalogue.

The op matrix (``tools/op_matrix.py`` -> ``docs/op_matrix.{json,md}``) is a
hand-curated catalogue of ``OpInfo`` entries.  It is also the join key for the
``nitrix-perf-bench`` coverage report: an op absent from the catalogue gets no
perf-baseline credit.  Historically the catalogue silently fell ~2/3 behind the
public surface because nothing guarded against it (the inventory-gap request,
``docs/feature-requests/doc-op-matrix-inventory-gaps.md``).

This test is that guard: every public op must either be in the catalogue or in
the explicit ``EXCLUDE`` allowlist below.  A new public op with neither fails
CI, so the matrix cannot drift behind the surface again.

What counts as "a public op": a name in a subpackage ``__all__`` that is
callable and **not a class**.  This automatically drops the substrate
*vocabulary* (dataclasses / NamedTuples like ``ELL`` / ``LMEResult``; Protocols
like ``Monoid``; the pre-built algebra *instances* ``REAL`` / ``LOG`` / ... ,
which are non-callable) -- none of which are ops.  The ``EXCLUDE`` allowlist
then names the callable functions we deliberately keep out, with rationale.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

# The public subpackages whose ``__all__`` defines the op surface.
SUBPKGS = [
    'linalg',
    'stats',
    'stats.lme',
    'signal',
    'numerics',
    'geometry',
    'graph',
    'morphology',
    'smoothing',
    'semiring',
    'sparse',
    'bias',
    'metrics',
    'register',
    'augment',
]

# Callable public functions intentionally kept OUT of the op matrix.  Membership
# in the matrix signals perf-bench to add a baseline, so we exclude anything
# that does not warrant one.  Keyed by ``<subpkg>.<name>``.
EXCLUDE: frozenset[str] = frozenset(
    {
        # -- Legacy aliases (dedupe to canonical; slated for v0.1 removal) --------
        'geometry.cmass_regular_grid',  # -> center_of_mass_grid
        'geometry.rescale',  # -> resample
        'geometry.vec_int',  # -> integrate_velocity_field
        'geometry.cmass_coor',  # -> center_of_mass_points
        'geometry.cmass_reference_displacement_coor',  # -> displacement_from_reference_points
        'geometry.cmass_reference_displacement_grid',  # -> displacement_from_reference_grid
        'geometry.diffuse',  # -> compactness_penalty
        # -- Thin "Alias for X" wrappers in stats --------------------------------
        'stats.ccov',  # -> conditionalcov
        'stats.ccorr',  # -> conditionalcorr
        'stats.corrcoef',  # -> corr (numpy convention)
        'stats.pcorr',  # -> partialcorr
        # -- Reference (pure-JAX) impls: the correctness floor the real ops are ---
        #    benchmarked *against*; baselining them separately is circular.
        'semiring.reference_semiring_matmul',
        'semiring.reference_semiring_conv',
        'semiring.reference_semiring_ell_matmul',
        'semiring.reference_semiring_ell_rmatvec',
        # -- Implicit-operator matvec closures: cost is the underlying matmul -----
        #    (already cataloged); used inside solvers, not standalone.
        'graph.laplacian_matvec',
        'graph.modularity_matrix_matvec',
        # -- Config-bound inner cost: bbr_cost takes a TransformModel / -----------
        #    Interpolator and the inverse affine, evaluated *inside*
        #    register.bbr_register (cataloged) -- the user op is bbr_register.
        'register.bbr_cost',
        # -- Pure structural / shape-layout helpers: ~free, no cross-framework ----
        #    baseline.
        'numerics.apply_mask',
        'numerics.broadcast_ignoring',
        'numerics.conform_mask',
        'numerics.fold_axis',
        'numerics.orient_and_conform',
        'numerics.promote_to_rank',
        'numerics.unfold_axes',
        # -- Metric constructors: build a FeatureMetric pytree, not an array op ---
        'smoothing.block_diagonal_metric',
        'smoothing.metric_from_spd',
        # -- Thin vmap wrapper of an already-cataloged op -------------------------
        'geometry.spatial_transform_batched',  # vmap of spatial_transform
    }
)


def _load_catalogue() -> set[str]:
    """Import ``tools/op_matrix.py`` and return its cataloged qualnames
    (stripped of the leading ``nitrix.``)."""
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        '_op_matrix_gen',
        root / 'tools' / 'op_matrix.py',
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so the module's own ``from __future__ import
    # annotations`` + ``@dataclass`` resolves its string annotations (the
    # dataclass machinery looks the module up in ``sys.modules``).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return {op.qualname.split('nitrix.', 1)[1] for op in mod.CATALOGUE}


def _public_ops() -> dict[str, object]:
    """Every callable, non-class public symbol across the subpackages,
    keyed by ``<subpkg>.<name>``."""
    ops: dict[str, object] = {}
    for sp in SUBPKGS:
        mod = importlib.import_module('nitrix.' + sp)
        for name in getattr(mod, '__all__', []):
            obj = getattr(mod, name, None)
            if obj is None or not callable(obj) or inspect.isclass(obj):
                continue
            ops[f'{sp}.{name}'] = obj
    return ops


def test_op_matrix_covers_all_public_ops():
    """Every public op is cataloged or explicitly excluded."""
    cataloged = _load_catalogue()
    public = set(_public_ops())
    missing = sorted(public - cataloged - EXCLUDE)
    assert not missing, (
        f'{len(missing)} public op(s) absent from the op-matrix catalogue '
        f'(tools/op_matrix.py) and from the EXCLUDE allowlist '
        f'(tests/test_op_matrix_completeness.py): {missing}.  Add an OpInfo '
        f'entry, or add the symbol to EXCLUDE with a rationale.'
    )


def test_exclude_allowlist_has_no_stale_entries():
    """Every EXCLUDE entry is a real public symbol (guards against typos /
    renames silently widening the allowlist)."""
    public = set(_public_ops())
    stale = sorted(EXCLUDE - public)
    assert not stale, (
        f'EXCLUDE names that are no longer public functions: {stale}.  '
        f'Remove them from the allowlist.'
    )


def test_catalogue_entries_are_not_excluded():
    """An op cannot be both cataloged and excluded (catches contradictory
    bookkeeping)."""
    cataloged = _load_catalogue()
    overlap = sorted(cataloged & EXCLUDE)
    assert not overlap, (
        f'These ops are both cataloged and in EXCLUDE: {overlap}.'
    )
