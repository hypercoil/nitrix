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
    'nn',
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
        # -- A type alias and two compositions over already-cataloged ops: --------
        #    ConvergenceMode is ``Literal['fixed','early_exit']`` (a non-callable
        #    type alias, not an op); apply_transform dispatches to the cataloged
        #    spatial_transform / apply_affine (a warp wrapper, no new kernel);
        #    syn_pipeline chains the cataloged rigid_register / affine_register /
        #    greedy_syn_register (its cost is its stages', benchmarked there).
        'register.ConvergenceMode',
        'register.apply_transform',
        'register.syn_pipeline',
        # -- A type alias, not an op: ``Similarity`` is ``Literal['euclidean',
        #    'cosine','correlation']`` (a similarity-metric selector consumed by
        #    the cataloged kmeans / normalized_cut). --------------------------------
        'numerics.Similarity',
        # -- Coefficient-design helper, not an array op: chebyshev_coefficients
        #    takes a scalar function + static order and returns filter
        #    coefficients (no array input to benchmark); it feeds the cataloged
        #    chebyshev_apply / matrix_polynomial / graph_wavelet_transform. --------
        'linalg.chebyshev_coefficients',
        # -- Grid-construction helper: sht_grid takes a static band-limit and
        #    returns the sampling points (no array input to benchmark); the
        #    cataloged sht_forward / sht_inverse are the transforms. --------------
        'geometry.sht_grid',
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
        # -- Family constructor: builds a Family config record (link/variance/    -
        #    deviance), consumed by glm_fit / gam_fit -- not itself a score kernel.
        'stats.negbinomial',
        'stats.tweedie',
        # -- Correlation-structure constructors: build a CorrSpec config record    -
        #    (per-group whitening + log-det), consumed by stats.lme.gls_fit -- not
        #    themselves array ops.
        'stats.lme.ar1',
        'stats.lme.car1',
        'stats.lme.cs',
        'stats.lme.iid',
        # -- Variance-function constructors: build a VarFunc config record --------
        #    (per-observation residual-scale + Jacobian), consumed by gls_fit.
        'stats.lme.var_power',
        'stats.lme.var_ident',
        # -- Thin vmap wrapper of an already-cataloged op -------------------------
        'geometry.spatial_transform_batched',  # vmap of spatial_transform
        # -- Post-fit read-outs on a GLMResult / GAMResult: contrasts, goodness- --
        #    of-fit, model comparison, prediction.  They operate on a *fitted*
        #    result pytree (not raw arrays); the fit (glm_fit / gam_fit) is the
        #    cataloged op and the dominant cost -- these are cheap reductions /
        #    contrasts on its stored sufficient statistics, benchmarked with it.
        'stats.t_contrast',
        'stats.f_contrast',
        'stats.sandwich_cov',  # robust vcov from a fitted GLMResult + its data
        'stats.r_squared',
        'stats.adj_r_squared',
        'stats.deviance_explained',
        'stats.log_likelihood',
        'stats.aic',
        'stats.bic',
        'stats.compare_models',
        'stats.predict',
        'stats.smooth_partial_effect',  # renders a fitted GAMResult smooth
        'stats.lme.lme_t_contrast',  # consumes a fitted REMLResult (contrast test)
        'stats.lme.lme_f_contrast',  # consumes a fitted REMLResult (F-contrast test)
        # -- Construction-time / re-evaluation spline-basis helpers: build a ------
        #    SplineBasis / TensorBasis / REBasis pytree (or render its design on a
        #    grid) to set up a GAM -- one-off, not the mass-univariate hot path.
        #    gam_fit is the cataloged op.
        'stats.bspline_basis',
        'stats.cyclic_cubic_basis',
        'stats.thinplate_regression_basis',
        'stats.cr_basis',
        'stats.gp_basis',
        'stats.mrf_smooth',
        'stats.tensor_product_basis',
        'stats.re_smooth',  # builds a random-effect (design, penalty) block
        'stats.by_factor_smooth',  # builds per-level SplineBasis blocks (s(x,by=f))
        'stats.varying_coefficient_smooth',  # builds a SplineBasis (s(x,by=z))
        'stats.spline_design',
        'stats.tensor_product_design',
        # -- GP / HSGP construction-time basis builders: build a Smooth / --------
        #    TensorBasis design pytree (one-off, pre-fit), like the GAM basis
        #    helpers above; gp_fit / hgp_fit are the cataloged hot-path ops.
        'stats.gp_factor_smooth',
        'stats.hsgp_basis',
        'stats.hsgp_basis_nd',
        # -- Prior constructors: build a PriorFn lengthscale-penalty closure ------
        #    (config), like the negbinomial / tweedie / Family constructors.
        'stats.halfnormal_prior',
        'stats.invgamma_prior',
        'stats.lognormal_prior',
        'stats.PriorFn',  # a Callable type alias (non-callable typedef), not an op
        # -- Family / link resolvers: normalise a str|Family / str|Link spec to ---
        #    its canonical config record (constructor dispatch), not array kernels.
        'stats.resolve_family',
        'stats.resolve_link',
        # -- Post-fit readouts on a fitted GP / HGP / GLM / GAM result pytree: ----
        #    confidence intervals, GP information criteria, GP prediction, and the
        #    smooth-significance test -- cheap reductions on stored sufficient
        #    statistics (like t_contrast / aic / bic / predict / smooth_partial_effect).
        'stats.confidence_interval',
        'stats.standardized_effect',  # effect/scale Cohen's-d ratio (readout)
        'stats.gp_aic',
        'stats.gp_bic',
        'stats.gp_predict',
        'stats.hgp_predict',
        'stats.smooth_significance',
        # -- Thin dispatcher over the cataloged stationary-kernel spectral --------
        #    densities (se_spectral_density / matern_spectral_density).
        'linalg.spectral_density',
        # -- Sparse-operator type alias (ELL | SectionedELL): non-callable. -------
        'sparse.MeshOperator',
        # -- Host-side mesh topology / QA / cleanup (combinatorial, not jit/diff/ -
        #    vmap-able; run post-hoc): Euler characteristic & genus are scalar
        #    topology read-outs; the self-intersection detect/repair and the
        #    sphere-bijectivity gate are host-side QA, not benchmark kernels.
        'geometry.euler_characteristic',
        'geometry.genus',
        'geometry.find_self_intersections',
        'geometry.remove_self_intersections',
        'geometry.is_bijective_sphere_map',
        # -- Priority-flood watershed: inherently serial, host-orchestrated -------
        #    segmentation over mesh adjacency (host-side -> JAX array pattern).
        'graph.mesh_watershed',
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
    keyed by ``<subpkg>.<name>``.

    A *cross-level* re-export -- a parent package re-exporting a subpackage op --
    is **one** op for the matrix.  The mixed-effects ops live in ``stats.lme`` but
    are also hoisted into ``stats`` (parity with the sibling fitters ``glm`` /
    ``gam`` / ``gp`` / ``glmm``, which are all top-level), so the same function
    object is reachable as both ``stats.lme.lme_fit`` and ``stats.lme_fit``.  Key
    it once, under its most-specific (home submodule) path, so a single catalogue
    / EXCLUDE entry covers it -- never two (which would double-count it in the
    perf-bench join) and never a spurious "uncataloged alias" miss.

    *Same-level* aliases (two names in one subpackage for one object, e.g. the
    legacy ``geometry.rescale`` -> ``resample``) are **not** collapsed: the
    EXCLUDE allowlist tracks each individually, so both must stay visible.
    """
    ops: dict[str, object] = {}
    canonical: dict[int, str] = {}  # id(obj) -> its most-specific public key
    for sp in SUBPKGS:
        mod = importlib.import_module('nitrix.' + sp)
        for name in getattr(mod, '__all__', []):
            obj = getattr(mod, name, None)
            if obj is None or not callable(obj) or inspect.isclass(obj):
                continue
            key = f'{sp}.{name}'
            prev = canonical.get(id(obj))
            # Collapse only cross-level re-exports (differing path depth); leave
            # same-depth aliases (e.g. geometry.rescale / resample) both visible.
            if prev is not None and prev.count('.') != key.count('.'):
                if prev.count('.') > key.count('.'):
                    continue  # already keyed under a more specific path
                del ops[prev]  # this key is more specific; supersede the alias
            canonical[id(obj)] = key
            ops[key] = obj
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
