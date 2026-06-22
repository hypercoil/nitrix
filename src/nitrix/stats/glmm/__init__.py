# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Mass-univariate generalised **linear mixed** models (GLMM) via PQL.

``glmm_fit`` fits, per element (voxel / vertex / fixel), a random-intercept
GLMM under a non-Gaussian :class:`~nitrix.stats._family.Family`::

    g(E[y | b]) = X beta + b[group],   b_j ~ N(0, sigma_b^2),   j = 1 .. q

by **penalised quasi-likelihood** (PQL; Breslow & Clayton 1993): an outer IRLS
loop forms the working response ``z`` and weights ``W`` from the current fit,
then a single variance-component (Fellner-Schall) step updates the random-effect
precision ``lambda = phi / sigma_b^2`` -- exactly the v1 "penalised GLM ==
variance-components REML" identity, now under a GLM family.  This is the §1.2
estimator: ``mgcv::gam(family=, s(g, bs="re"))`` / ``glmmPQL`` (with the
documented PQL attenuation for binary / low-count responses; Laplace is the
Tier-2 follow-up).

Performance-preserving dispatch (v3 §0.1 / §2 routing)
----------------------------------------------------

A random effect widens the per-voxel system by ``q`` = #levels, so the dense
penalised solve is ``O((p + q)^3)`` per voxel.  That is **cheap for few-level
factors** (site / scanner / batch, ``q ~ 10-50``) but a latency cliff for
**many-level** factors (random intercept per *subject*, ``q ~ 100-1000``).
``glmm_fit`` therefore dispatches on the level count:

- **few-level** (``q <= few_level_max``): the dense GAMM path -- ``gam_fit`` with
  a :func:`~nitrix.stats.basis.re_smooth` block and the GLM family.  This *is*
  the optimal solver in this regime (no cheaper exact route), and reuses the
  shipped GAM machinery verbatim.
- **many-level** (``q > few_level_max``): a **structured** PQL that never forms
  the ``(p + q) x (p + q)`` system.  Because a single grouping factor makes the
  random-effect block of the normal equations **diagonal across groups**, the
  Schur complement onto the ``p``-dimensional fixed block costs ``O(N p^2 + q)``
  per voxel (the §1.1 block-Woodbury structure, weighted by the IRLS ``W`` and
  wrapped in the PQL loop) -- linear in the level count, cuSOLVER-free.

The two paths run the *same* PQL iteration (same working response, same
Fellner-Schall update, same iteration budget), so they agree to the iterative
tolerance; the dispatch only changes the linear-algebra cost, never the answer.

Scope.  Beyond the **scalar random intercept** ``(1 | g)`` (the FR §1.2 headline:
"binary outcomes / lesion counts per subject with random intercepts"), ``glmm_fit``
also fits non-Gaussian random **slopes** via the ``z`` / ``structure`` arguments:
diagonal ``(x || g)`` as independent ``re_smooth`` blocks through ``gam_fit``, and
correlated ``(1 + x | g)`` via the block-Woodbury REML wrapped in the PQL loop (the
penalised-IRLS = iteratively-reweighted-REML identity).  The Gaussian-family slope
fit is the same REML estimator as ``lme_fit`` (R2 block-Woodbury), to the
iterative (optimiser) tolerance.  Random slopes are also
served under the **Laplace** marginal likelihood (``method='laplace'``, the
``r``-dimensional conditional-mode integral + ``r x r`` determinant correction),
which corrects the PQL attenuation for binary / low-count slopes.

References
----------
- Breslow, N. E. & Clayton, D. G. (1993). Approximate inference in generalized
  linear mixed models.  JASA 88, 9-25.
- Wood, S. N. & Fasiolo, M. (2017). A generalized Fellner-Schall method for
  smoothing parameter optimization.  Biometrics 73, 1071-1081.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from .._family import GAUSSIAN, Family, resolve_family
from ._agq import _glmm_agq_slope
from ._base import _AGQ_MAX_NODES, GLMMResult
from ._laplace import _glmm_laplace, _glmm_laplace_slope
from ._pql import (
    _glmm_few_level,
    _glmm_many_level,
    _glmm_slope_diagonal,
    _glmm_slope_structured,
)

__all__ = ['GLMMResult', 'glmm_fit']

GLMMStructure = Literal['unstructured', 'diagonal']
GLMMMethod = Literal['pql', 'laplace', 'agq']


def glmm_fit(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    *,
    group: Int[Array, 'N'],
    n_groups: Optional[int] = None,
    z: Optional[Float[Array, 'N r']] = None,
    structure: GLMMStructure = 'unstructured',
    family: Union[str, Family] = GAUSSIAN,
    method: GLMMMethod = 'pql',
    few_level_max: int = 64,
    n_outer: int = 20,
    n_inner: int = 10,
    n_mode: int = 20,
    n_quad: int = 5,
    ridge: float = 1e-8,
    lam_floor: float = 1e-6,
    lam_ceil: float = 1e8,
    damping: float = 1e-6,
    block: Optional[int] = None,
) -> GLMMResult:
    """Mass-univariate random-intercept GLMM via PQL, dispatched on level count.

    Fits, per element, ``g(E[y | b]) = X beta + b[group]`` with ``b_j ~ N(0,
    sigma_b^2)`` under the GLM ``family``, by penalised quasi-likelihood (working
    response -> one variance-component step -> repeat).  The fixed design ``X`` is
    used as given (include your own intercept column); ``group`` is the
    ``(N,)`` integer grouping factor (levels ``0 .. q-1``).

    Dispatch (the v3 §0.1 performance-preserving invariant / §2 routing)
    -------------------------------------------------------------------

    - ``q <= few_level_max`` -- **few-level**: the dense GAMM path (``gam_fit`` +
      ``re_smooth``).  Optimal here; the per-voxel cost ``O((p + q)^3)`` is tiny.
    - ``q > few_level_max`` -- **many-level**: a structured Schur-complement PQL
      costing ``O(N p^2 + q)`` per voxel (the §1.1 block-Woodbury structure,
      weighted and wrapped in the IRLS loop), avoiding the dense ``(p + q)``-wide
      solve.

    Both paths run the identical PQL iteration, so they agree to the iterative
    tolerance -- the dispatch changes only the linear-algebra cost.

    Parameters
    ----------
    Y
        ``(V, N)`` responses.
    X
        ``(N, p)`` fixed-effect design (shared across elements; carries its own
        intercept).
    group
        ``(N,)`` integer grouping factor (random intercept per level).
    n_groups
        Optional **static** level count ``q`` (levels are ``0 .. q-1``).  When
        ``None`` (default) it is derived eagerly as ``int(max(group)) + 1`` --
        byte-identical to before, but that concretises ``group`` and so makes
        ``glmm_fit`` untraceable under ``jax.jit``.  Pass the count explicitly (a
        Python ``int``) to trace the whole fit under ``jit`` -- e.g. to fuse it
        into a larger program -- for every family / structure / method.
    z
        Optional ``(N, r)`` random-effect design for a random **slope** (e.g.
        ``[1, x]`` -> random intercept + random slope of ``x``).  ``None``
        (default) is the scalar random intercept ``(1 | g)``.  Mirrors
        ``lme_fit``'s ``z=`` argument; the Gaussian-family slope GLMM is the same
        REML fit as ``lme_fit(z=, structure=)`` (to optimiser tolerance).
    structure
        Random-effect covariance for a slope (``z`` given): ``'unstructured'``
        (full ``r x r`` ``G``, the correlated ``(1 + x | g)``) or ``'diagonal'``
        (independent variance components, the uncorrelated ``(x || g)``).  Ignored
        when ``z is None``.
    family
        GLM family (``'binomial'`` / ``'poisson'`` / ``'gamma'`` /
        ``'negbinomial'`` / ``'gaussian'`` or a :class:`Family`).  Gaussian is
        accepted (it reduces to the LME) but ``lme_fit`` / ``reml_fit`` are the
        direct route for a Gaussian mixed model.
    method
        ``'pql'`` (default) -- penalised quasi-likelihood (cheap; documented
        small-cluster attenuation for binary / low-count responses).
        ``'laplace'`` -- the Laplace-approximate marginal likelihood (per-group
        conditional-mode integral + curvature correction): more accurate for
        binary / low-count GLMMs (it corrects the PQL bias), at the cost of an
        inner mode-finding loop.  Fits a scalar random intercept, or a random
        **slope** when ``z`` is given (the ``r``-dimensional mode + ``r x r``
        determinant correction, ``structure`` selecting a full / diagonal ``G``);
        the ``tier`` is ``'laplace'``.
        ``'agq'`` -- adaptive Gauss-Hermite quadrature (**random slope only**, ``z``
        required): the marginal random-effect integral by ``n_quad``-point tensor
        GH, centred / scaled at each group's mode and curvature.  ``n_quad = 1`` is
        exactly Laplace; more nodes integrate the density directly, converging to
        the exact marginal (the ``lme4`` ``nAGQ`` accuracy tier, the gold standard
        for small / low-count clusters).  ``tier`` is ``'agq'``.
    few_level_max
        Dispatch threshold on the number of levels ``q`` (default ``64`` -- the
        regime where the dense solve stops being trivially cheap).  ``'pql'``
        only.
    n_outer, n_inner, n_mode, n_quad
        PQL outer (Fellner-Schall) and inner (IRLS) iteration budgets; ``n_mode``
        is the per-group conditional-mode Newton budget for ``method='laplace'`` /
        ``'agq'``; ``n_quad`` is the GH nodes **per dimension** for ``'agq'``
        (default ``5``; the integral uses ``n_quad ** r`` tensor nodes).
    ridge, lam_floor, lam_ceil, block
        Normal-equation stabiliser, smoothing-parameter clamps, and the optional
        element-block size bounding peak memory.

    Returns
    -------
    ``GLMMResult`` -- ``beta_hat``, per-level ``blups``, ``re_var``
    (``sigma_b^2``, or the ``r``-vector / ``r x r`` ``G`` for a slope),
    ``dispersion``, ``deviance``, ``edf_total``, and the ``tier`` that ran
    (``'few'`` / ``'many'`` / ``'slope'`` / ``'laplace'`` / ``'agq'``).

    Notes
    -----
    PQL carries the documented small-cluster bias for binary / low-count
    responses (it under-estimates the variance component; the bias vanishes as
    the per-group information grows).  Random *slopes* (``z`` + ``structure``) are
    fit by PQL -- diagonal via ``gam_fit`` blocks, correlated via the joint-Schur
    + REML-EM solver (``tier='slope'``) -- by Laplace (``method='laplace'``), or by
    adaptive Gauss-Hermite quadrature (``method='agq'``, ``tier='agq'``), the
    accuracy ladder for the slope-variance attenuation: AGQ with ``n_quad``
    nodes converges to the exact marginal (``n_quad = 1`` is Laplace).
    """
    family = resolve_family(family)
    n = X.shape[0]
    group = jnp.asarray(group)
    if Y.shape[-1] != n:
        raise ValueError(
            f'glmm_fit: Y.shape[-1]={Y.shape[-1]} must match N={n}.'
        )
    if group.shape[0] != n:
        raise ValueError(
            f'glmm_fit: group has {group.shape[0]} labels; expected N={n}.'
        )
    # The level count is a *static* shape driver (segment sums, output shapes,
    # the few/many dispatch).  Deriving it from the data with int(jnp.max(...))
    # concretises a tracer, so the eager-only default makes glmm_fit untraceable
    # under jax.jit (P7); pass n_groups explicitly -- the count the caller
    # already knows -- to fuse the whole fit (mirrors lme_fit / reml_fit, where
    # the level structure is likewise caller-supplied).
    # Resolve the static level count exactly as before: int(jnp.max(group))
    # concretises a tracer, so without an explicit n_groups this (correctly)
    # raises under jit -- the P7 contract (pass n_groups to trace).
    if n_groups is None:
        n_groups = int(jnp.max(group)) + 1
    # Validate the label range on the host when `group` is concrete (the eager
    # case, or a constant captured by an outer jit -- np.asarray reads the
    # constant without building a tracer). one_hot / segment_sum / b[group] all
    # silently zero out-of-range or negative indices, so a bad label would drop
    # that observation from the group structure (a wrong-but-finite fit). A
    # genuinely traced `group` is skipped (the caller owns the contract).
    if not isinstance(group, jax.core.Tracer):
        g_host = np.asarray(group)
        g_min, g_max = int(g_host.min()), int(g_host.max())
        if g_min < 0 or g_max >= n_groups:
            raise ValueError(
                f'glmm_fit: group labels must lie in [0, n_groups={n_groups}); '
                f'got min={g_min}, max={g_max}. Pass contiguous 0-based labels '
                'so no observation silently drops out of the group structure.'
            )

    if z is not None:
        z = jnp.asarray(z, dtype=X.dtype)
        if z.ndim != 2 or z.shape[0] != n:
            raise ValueError(
                f'glmm_fit: z must be (N, r) with N={n}; got shape '
                f'{tuple(z.shape)}.'
            )
        if structure not in ('unstructured', 'diagonal'):
            raise ValueError(
                f"glmm_fit: structure={structure!r}; expected 'unstructured' "
                "or 'diagonal'."
            )
        diagonal = structure == 'diagonal'
        if method == 'laplace':
            return _glmm_laplace_slope(
                Y,
                X,
                group,
                n_groups,
                z,
                family,
                n_outer,
                n_mode,
                damping,
                diagonal,
                block,
            )
        if method == 'agq':
            r = z.shape[-1]
            n_nodes = n_quad**r
            if n_nodes > _AGQ_MAX_NODES:
                raise ValueError(
                    f'glmm_fit: AGQ tensor grid has n_quad**r = {n_quad}**{r} '
                    f'= {n_nodes} nodes (> {_AGQ_MAX_NODES}); the per-element '
                    f'graph grows as n_quad**r and is differentiated through the '
                    f'mode scan, so a large r is a compile / memory cliff.  Use a '
                    f"smaller n_quad, method='laplace' (= AGQ with n_quad=1), or "
                    f'a lower-dimensional random effect.'
                )
            return _glmm_agq_slope(
                Y,
                X,
                group,
                n_groups,
                z,
                family,
                n_outer,
                n_mode,
                damping,
                diagonal,
                n_quad,
                block,
            )
        if method != 'pql':
            raise ValueError(
                f"glmm_fit: method={method!r}; expected 'pql', 'laplace' or "
                "'agq'."
            )
        if diagonal:
            return _glmm_slope_diagonal(
                Y,
                X,
                group,
                n_groups,
                z,
                family,
                n_outer,
                n_inner,
                ridge,
                lam_floor,
                lam_ceil,
                block,
            )
        return _glmm_slope_structured(
            Y,
            X,
            group,
            n_groups,
            z,
            family,
            n_outer,
            n_inner,
            ridge,
            block,
        )

    if method == 'laplace':
        return _glmm_laplace(
            Y, X, group, n_groups, family, n_outer, n_mode, damping, block
        )
    if method == 'agq':
        raise NotImplementedError(
            "glmm_fit: adaptive Gauss-Hermite (method='agq') currently requires "
            'a random slope -- pass z= (use z=ones((N, 1)) for a scalar '
            "intercept), or use method='laplace'."
        )
    if method != 'pql':
        raise ValueError(
            f"glmm_fit: method={method!r}; expected 'pql' or 'laplace'."
        )

    args = (
        Y,
        X,
        group,
        n_groups,
        family,
        n_outer,
        n_inner,
        ridge,
        lam_floor,
        lam_ceil,
        block,
    )
    if n_groups <= few_level_max:
        return _glmm_few_level(*args)
    return _glmm_many_level(*args)
